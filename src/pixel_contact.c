// SPDX-FileCopyrightText: 2023 Erin Catto
// SPDX-License-Identifier: MIT

#include "pixel_contact.h"

#include "core.h"
#include "pixel_shape.h"

#include "box2d/box2d.h"

#include <float.h>
#include <math.h>
#include <stdint.h>

enum
{
	b2_maxPixelSourceFeatures = 1024,
	b2_maxPixelCellVisits = 1024,
	b2_maxPixelDiskTests = 256,
	b2_maxPixelRawContacts = 96,
	b2_maxPixelEmbeddedExitSearch = 256,
};

typedef struct b2PixelRawContact
{
	b2Vec2 point;
	b2Vec2 normal;
	float penetration;
	uint16_t featureIdA;
	uint16_t featureIdB;
	uint8_t typeA;
	uint8_t typeB;
} b2PixelRawContact;

typedef struct b2PixelRawBuffer
{
	b2PixelRawContact contacts[b2_maxPixelRawContacts];
	int count;
	int sourceFeatures;
	int cellVisits;
	int diskTests;
	int rawContactAttempts;
	bool sourceFeaturesCapped;
	bool cellVisitsCapped;
	bool diskTestsCapped;
	bool rawContactsCapped;
} b2PixelRawBuffer;

typedef struct b2PixelManifoldSelection
{
	b2Vec2 normal;
	int indexA;
	int indexB;
} b2PixelManifoldSelection;

typedef struct b2PixelNormalContext
{
	const b2PixelShape* pixelA;
	const b2PixelShape* pixelB;
	b2Transform xfA;
	b2Transform xfB;
	const b2PixelRawBuffer* buffer;
	float pixelSize;
} b2PixelNormalContext;

static void b2PublishPixelStats( const b2PixelRawBuffer* buffer, const b2Manifold* manifold, bool rescueCandidate, bool rescueUsed,
								 bool invalidInput, b2PixelNarrowphaseStats* stats )
{
	if ( stats == NULL )
	{
		return;
	}

	stats->sourceFeatures = buffer == NULL ? 0 : buffer->sourceFeatures;
	stats->cellVisits = buffer == NULL ? 0 : buffer->cellVisits;
	stats->diskTests = buffer == NULL ? 0 : buffer->diskTests;
	stats->rawContacts = buffer == NULL ? 0 : buffer->count;
	stats->rawContactAttempts = buffer == NULL ? 0 : buffer->rawContactAttempts;
	stats->manifoldPoints = manifold == NULL ? 0 : manifold->pointCount;
	stats->sourceFeaturesCapped = buffer != NULL && buffer->sourceFeaturesCapped;
	stats->cellVisitsCapped = buffer != NULL && buffer->cellVisitsCapped;
	stats->diskTestsCapped = buffer != NULL && buffer->diskTestsCapped;
	stats->rawContactsCapped = buffer != NULL && buffer->rawContactsCapped;
	stats->rescueCandidate = rescueCandidate;
	stats->rescueUsed = rescueUsed;
	stats->invalidInput = invalidInput;
}

static b2Vec2 b2PixelShape_GetWorldCenter( const b2PixelShape* shape, b2Transform xf )
{
	return b2TransformPoint( xf, b2GetPixelShapeCentroid( shape ) );
}

static uint16_t b2PixelFeatureIdFromRef( const b2PixelAsset* asset, const b2PixelFeatureRef* ref )
{
	return ref->id != 0 ? ref->id : b2PixelAsset_GetFeatureId( asset, ref->x, ref->y );
}

static uint16_t b2MakePixelContactId( uint16_t featureIdA, uint16_t featureIdB )
{
	uint32_t x = (uint32_t)featureIdA * UINT32_C( 0x9e37 ) ^ ( (uint32_t)featureIdB << 7 ) ^ ( (uint32_t)featureIdB >> 3 );
	x ^= x >> 16;
	uint16_t id = (uint16_t)( x & 0xffffu );
	return id == 0 ? 1 : id;
}

static bool b2PixelFeatureIdToCell( const b2PixelAsset* asset, uint16_t featureId, int* x, int* y )
{
	if ( asset == NULL || featureId == 0 || asset->width <= 0 || asset->height <= 0 || asset->width > INT32_MAX / asset->height )
	{
		return false;
	}

	int index = (int)featureId - 1;
	if ( index < 0 || index >= asset->width * asset->height )
	{
		return false;
	}

	*x = index % asset->width;
	*y = index / asset->width;
	return true;
}

static b2Vec2 b2PixelFeatureOutwardLocal( const b2PixelAsset* asset, int x, int y )
{
	b2Vec2 normal = b2Vec2_zero;
	if ( b2PixelAsset_IsOccupied( asset, x, y - 1 ) == false )
	{
		normal.y -= 1.0f;
	}
	if ( b2PixelAsset_IsOccupied( asset, x + 1, y ) == false )
	{
		normal.x += 1.0f;
	}
	if ( b2PixelAsset_IsOccupied( asset, x, y + 1 ) == false )
	{
		normal.y += 1.0f;
	}
	if ( b2PixelAsset_IsOccupied( asset, x - 1, y ) == false )
	{
		normal.x -= 1.0f;
	}

	if ( b2LengthSquared( normal ) < 1.0e-10f )
	{
		return b2Vec2_zero;
	}

	return b2Normalize( normal );
}

static b2Vec2 b2ComputePixelPairNormalAB( const b2PixelShape* shapeA, b2Transform xfA, const b2PixelFeatureRef* featureA,
										  const b2PixelShape* shapeB, b2Transform xfB, int xB, int yB, uint8_t typeB,
										  b2Vec2 deltaNormal )
{
	const b2PixelAsset* assetA = shapeA->asset;
	const b2PixelAsset* assetB = shapeB->asset;
	if ( featureA->type == b2_pixelFeatureEdge && typeB != b2_pixelFeatureEdge )
	{
		b2Vec2 outward = b2PixelFeatureOutwardLocal( assetA, featureA->x, featureA->y );
		if ( b2LengthSquared( outward ) > 0.0f )
		{
			return b2RotateVector( xfA.q, outward );
		}
	}

	if ( typeB == b2_pixelFeatureEdge && featureA->type != b2_pixelFeatureEdge )
	{
		b2Vec2 outward = b2PixelFeatureOutwardLocal( assetB, xB, yB );
		if ( b2LengthSquared( outward ) > 0.0f )
		{
			return b2Neg( b2RotateVector( xfB.q, outward ) );
		}
	}

	return deltaNormal;
}

static void b2PushPixelRawContact( b2PixelRawBuffer* buffer, b2PixelRawContact contact )
{
	buffer->rawContactAttempts += 1;
	if ( buffer->rawContactAttempts > b2_maxPixelRawContacts )
	{
		buffer->rawContactsCapped = true;
	}

	for ( int i = 0; i < buffer->count; ++i )
	{
		b2PixelRawContact* existing = buffer->contacts + i;
		if ( existing->featureIdA == contact.featureIdA && existing->featureIdB == contact.featureIdB )
		{
			if ( contact.penetration > existing->penetration )
			{
				*existing = contact;
			}
			return;
		}
	}

	if ( buffer->count < b2_maxPixelRawContacts )
	{
		buffer->contacts[buffer->count++] = contact;
		return;
	}

	buffer->rawContactsCapped = true;
	int shallowest = 0;
	for ( int i = 1; i < buffer->count; ++i )
	{
		if ( buffer->contacts[i].penetration < buffer->contacts[shallowest].penetration )
		{
			shallowest = i;
		}
	}

	if ( contact.penetration > buffer->contacts[shallowest].penetration )
	{
		buffer->contacts[shallowest] = contact;
	}
}

static bool b2DiskTestPixelFeatures( const b2PixelShape* shapeA, b2Transform xfA, const b2PixelFeatureRef* featureA,
									 const b2PixelShape* shapeB, b2Transform xfB, int xB, int yB, uint8_t typeB,
									 bool reverseToOriginalOrder, b2Vec2 fallbackNormal, b2PixelRawContact* out )
{
	const b2PixelAsset* assetA = shapeA->asset;
	const b2PixelAsset* assetB = shapeB->asset;
	b2Vec2 localA = b2PixelShape_GetPixelCenter( shapeA, featureA->x, featureA->y );
	b2Vec2 localB = b2PixelShape_GetPixelCenter( shapeB, xB, yB );
	b2Vec2 worldA = b2TransformPoint( xfA, localA );
	b2Vec2 worldB = b2TransformPoint( xfB, localB );
	b2Vec2 d = b2Sub( worldB, worldA );
	float radiusA = b2GetPixelShapeDiskRadius( shapeA ) * assetA->pixelSize;
	float radiusB = b2GetPixelShapeDiskRadius( shapeB ) * assetB->pixelSize;
	float radius = radiusA + radiusB;
	float distanceSqr = b2LengthSquared( d );
	if ( distanceSqr > radius * radius )
	{
		return false;
	}

	float distance = 0.0f;
	b2Vec2 normal = fallbackNormal;
	if ( distanceSqr > 1.0e-10f )
	{
		distance = sqrtf( distanceSqr );
		normal = b2MulSV( 1.0f / distance, d );
	}
	normal = b2ComputePixelPairNormalAB( shapeA, xfA, featureA, shapeB, xfB, xB, yB, typeB, normal );

	b2PixelRawContact contact = { 0 };
	contact.point = b2MulSV( 0.5f, b2Add( worldA, worldB ) );
	contact.normal = normal;
	contact.penetration = radius - distance;
	contact.featureIdA = b2PixelFeatureIdFromRef( assetA, featureA );
	contact.featureIdB = b2PixelAsset_GetFeatureId( assetB, xB, yB );
	contact.typeA = featureA->type;
	contact.typeB = typeB;

	if ( reverseToOriginalOrder )
	{
		contact.normal = b2Neg( contact.normal );
		uint16_t id = contact.featureIdA;
		contact.featureIdA = contact.featureIdB;
		contact.featureIdB = id;
		uint8_t type = contact.typeA;
		contact.typeA = contact.typeB;
		contact.typeB = type;
	}

	*out = contact;
	return true;
}

static void b2GatherPixelCornerContacts( const b2PixelShape* source, b2Transform xfSource, const b2PixelShape* target,
										 b2Transform xfTarget, bool reverseToOriginalOrder, b2PixelRawBuffer* buffer )
{
	const b2PixelAsset* assetSource = source->asset;
	const b2PixelAsset* assetTarget = target->asset;
	float radiusSource = b2GetPixelShapeDiskRadius( source ) * assetSource->pixelSize;
	float radiusTarget = b2GetPixelShapeDiskRadius( target ) * assetTarget->pixelSize;
	float searchRadius = radiusSource + radiusTarget;
	int searchCells = (int)ceilf( searchRadius / assetTarget->pixelSize ) + 1;
	b2Vec2 fallbackNormal = b2Normalize( b2Sub( b2PixelShape_GetWorldCenter( target, xfTarget ),
												b2PixelShape_GetWorldCenter( source, xfSource ) ) );
	if ( b2LengthSquared( fallbackNormal ) == 0.0f )
	{
		fallbackNormal = (b2Vec2){ 1.0f, 0.0f };
	}

	for ( int cornerIndex = 0; cornerIndex < assetSource->cornerCount; ++cornerIndex )
	{
		const b2PixelFeatureRef* corner = assetSource->corners + cornerIndex;
		b2Vec2 worldCorner = b2TransformPoint( xfSource, b2PixelShape_GetPixelCenter( source, corner->x, corner->y ) );
		b2Vec2 targetLocal = b2Sub( b2InvTransformPoint( xfTarget, worldCorner ), target->localOrigin );
		int centerX = (int)floorf( targetLocal.x / assetTarget->pixelSize + 0.5f * (float)assetTarget->width );
		int centerY = (int)floorf( targetLocal.y / assetTarget->pixelSize + 0.5f * (float)assetTarget->height );
		if ( centerX < -searchCells || centerX >= assetTarget->width + searchCells || centerY < -searchCells ||
			 centerY >= assetTarget->height + searchCells )
		{
			continue;
		}

		if ( buffer->sourceFeatures >= b2_maxPixelSourceFeatures )
		{
			buffer->sourceFeaturesCapped = true;
			return;
		}

		buffer->sourceFeatures += 1;
		int minX = b2MaxInt( 0, centerX - searchCells );
		int maxX = b2MinInt( assetTarget->width - 1, centerX + searchCells );
		int minY = b2MaxInt( 0, centerY - searchCells );
		int maxY = b2MinInt( assetTarget->height - 1, centerY + searchCells );
		for ( int y = minY; y <= maxY; ++y )
		{
			for ( int x = minX; x <= maxX; ++x )
			{
				if ( buffer->cellVisits >= b2_maxPixelCellVisits )
				{
					buffer->cellVisitsCapped = true;
					return;
				}

				buffer->cellVisits += 1;
				if ( buffer->diskTests >= b2_maxPixelDiskTests )
				{
					buffer->diskTestsCapped = true;
					return;
				}

				uint8_t typeB = b2PixelAsset_GetFeatureType( assetTarget, x, y );
				if ( typeB == b2_pixelFeatureEmpty || typeB == b2_pixelFeatureInternal )
				{
					continue;
				}

				buffer->diskTests += 1;
				b2PixelRawContact contact;
				if ( b2DiskTestPixelFeatures( source, xfSource, corner, target, xfTarget, x, y, typeB,
											   reverseToOriginalOrder, fallbackNormal, &contact ) )
				{
					b2PushPixelRawContact( buffer, contact );
				}
			}
		}
	}
}

static bool b2FindPixelEmbeddedExit( const b2PixelAsset* asset, int startX, int startY, b2Vec2* exitLocal, float* distance )
{
	if ( b2PixelAsset_IsOccupied( asset, startX, startY ) == false )
	{
		return false;
	}

	static const int directions[8][2] = {
		{ 1, 0 },
		{ -1, 0 },
		{ 0, 1 },
		{ 0, -1 },
		{ 1, 1 },
		{ -1, 1 },
		{ 1, -1 },
		{ -1, -1 },
	};

	float bestDistance = FLT_MAX;
	b2Vec2 bestExit = b2Vec2_zero;
	int maxSteps = b2MinInt( b2MaxInt( asset->width, asset->height ) + 1, b2_maxPixelEmbeddedExitSearch );
	for ( int directionIndex = 0; directionIndex < 8; ++directionIndex )
	{
		int dx = directions[directionIndex][0];
		int dy = directions[directionIndex][1];
		float directionLength = dx != 0 && dy != 0 ? sqrtf( 2.0f ) : 1.0f;
		for ( int step = 1; step <= maxSteps; ++step )
		{
			int x = startX + dx * step;
			int y = startY + dy * step;
			if ( b2PixelAsset_IsOccupied( asset, x, y ) )
			{
				continue;
			}

			float candidateDistance = ( (float)step - 0.5f ) * directionLength * asset->pixelSize;
			if ( candidateDistance < bestDistance )
			{
				bestDistance = candidateDistance;
				bestExit = b2Normalize( (b2Vec2){ (float)dx, (float)dy } );
			}
			break;
		}
	}

	if ( bestDistance == FLT_MAX )
	{
		return false;
	}

	*exitLocal = bestExit;
	*distance = bestDistance;
	return true;
}

static void b2GatherPixelEmbeddedContacts( const b2PixelShape* source, b2Transform xfSource, const b2PixelShape* target,
										   b2Transform xfTarget, bool reverseToOriginalOrder, b2PixelRawBuffer* buffer )
{
	const b2PixelAsset* assetSource = source->asset;
	const b2PixelAsset* assetTarget = target->asset;
	float radiusSource = b2GetPixelShapeDiskRadius( source ) * assetSource->pixelSize;
	float radiusTarget = b2GetPixelShapeDiskRadius( target ) * assetTarget->pixelSize;

	for ( int cornerIndex = 0; cornerIndex < assetSource->cornerCount; ++cornerIndex )
	{
		const b2PixelFeatureRef* corner = assetSource->corners + cornerIndex;
		b2Vec2 worldCorner = b2TransformPoint( xfSource, b2PixelShape_GetPixelCenter( source, corner->x, corner->y ) );
		b2Vec2 targetLocal = b2Sub( b2InvTransformPoint( xfTarget, worldCorner ), target->localOrigin );
		int centerX = (int)floorf( targetLocal.x / assetTarget->pixelSize + 0.5f * (float)assetTarget->width );
		int centerY = (int)floorf( targetLocal.y / assetTarget->pixelSize + 0.5f * (float)assetTarget->height );
		if ( b2PixelAsset_IsOccupied( assetTarget, centerX, centerY ) == false )
		{
			continue;
		}

		if ( buffer->sourceFeatures >= b2_maxPixelSourceFeatures )
		{
			buffer->sourceFeaturesCapped = true;
			return;
		}

		b2Vec2 exitLocal;
		float exitDistance;
		if ( b2FindPixelEmbeddedExit( assetTarget, centerX, centerY, &exitLocal, &exitDistance ) == false )
		{
			continue;
		}

		buffer->sourceFeatures += 1;
		b2Vec2 exitWorld = b2RotateVector( xfTarget.q, exitLocal );
		b2PixelRawContact contact = { 0 };
		contact.point = worldCorner;
		contact.normal = b2Neg( exitWorld );
		contact.penetration = exitDistance + radiusSource + radiusTarget;
		contact.featureIdA = b2PixelFeatureIdFromRef( assetSource, corner );
		contact.featureIdB = b2PixelAsset_GetFeatureId( assetTarget, centerX, centerY );
		contact.typeA = corner->type;
		contact.typeB = b2PixelAsset_GetFeatureType( assetTarget, centerX, centerY );

		if ( reverseToOriginalOrder )
		{
			contact.normal = b2Neg( contact.normal );
			uint16_t id = contact.featureIdA;
			contact.featureIdA = contact.featureIdB;
			contact.featureIdB = id;
			uint8_t type = contact.typeA;
			contact.typeA = contact.typeB;
			contact.typeB = type;
		}

		b2PushPixelRawContact( buffer, contact );
	}
}

static bool b2HasPixelRescueCandidate( const b2AABB* aabbA, const b2AABB* aabbB, const b2PixelRawBuffer* buffer )
{
	return buffer != NULL && buffer->count == 0 && b2AABB_Overlaps( *aabbA, *aabbB ) &&
		   ( buffer->sourceFeaturesCapped || buffer->cellVisitsCapped || buffer->diskTestsCapped ||
			 buffer->rawContactsCapped );
}

static int b2PickPixelDeepest( const b2PixelRawBuffer* buffer )
{
	int best = 0;
	for ( int i = 1; i < buffer->count; ++i )
	{
		const b2PixelRawContact* a = buffer->contacts + i;
		const b2PixelRawContact* b = buffer->contacts + best;
		if ( a->penetration > b->penetration ||
			 ( a->penetration == b->penetration &&
			   b2MakePixelContactId( a->featureIdA, a->featureIdB ) < b2MakePixelContactId( b->featureIdA, b->featureIdB ) ) )
		{
			best = i;
		}
	}
	return best;
}

static bool b2PixelContactTieLess( const b2PixelRawContact* a, const b2PixelRawContact* b )
{
	uint16_t idA = b2MakePixelContactId( a->featureIdA, a->featureIdB );
	uint16_t idB = b2MakePixelContactId( b->featureIdA, b->featureIdB );
	return idA < idB;
}

static b2Vec2 b2ReducePixelNormal( const b2PixelRawBuffer* buffer )
{
	b2Vec2 normal = b2Vec2_zero;
	for ( int i = 0; i < buffer->count; ++i )
	{
		const b2PixelRawContact* contact = buffer->contacts + i;
		float weight = b2MaxFloat( contact->penetration, 0.001f );
		normal = b2MulAdd( normal, weight, contact->normal );
	}

	if ( b2LengthSquared( normal ) < 1.0e-10f )
	{
		return (b2Vec2){ 1.0f, 0.0f };
	}

	return b2Normalize( normal );
}

static int b2CountPixelRawFeatureOverlap( const b2PixelNormalContext* context, b2Vec2 shiftA, b2Vec2 shiftB )
{
	int count = 0;
	b2Transform shiftedA = context->xfA;
	shiftedA.p = b2Add( shiftedA.p, shiftA );
	b2Transform shiftedB = context->xfB;
	shiftedB.p = b2Add( shiftedB.p, shiftB );

	const b2PixelAsset* assetA = context->pixelA->asset;
	const b2PixelAsset* assetB = context->pixelB->asset;
	for ( int i = 0; i < context->buffer->count; ++i )
	{
		const b2PixelRawContact* contact = context->buffer->contacts + i;
		int xA, yA;
		if ( b2PixelFeatureIdToCell( assetA, contact->featureIdA, &xA, &yA ) )
		{
			b2Vec2 worldA = b2TransformPoint( shiftedA, b2PixelShape_GetPixelCenter( context->pixelA, xA, yA ) );
			b2Vec2 localInB = b2InvTransformPoint( shiftedB, worldA );
			count += b2PointInPixelShape( context->pixelB, localInB ) ? 1 : 0;
		}

		int xB, yB;
		if ( b2PixelFeatureIdToCell( assetB, contact->featureIdB, &xB, &yB ) )
		{
			b2Vec2 worldB = b2TransformPoint( shiftedB, b2PixelShape_GetPixelCenter( context->pixelB, xB, yB ) );
			b2Vec2 localInA = b2InvTransformPoint( shiftedA, worldB );
			count += b2PointInPixelShape( context->pixelA, localInA ) ? 1 : 0;
		}
	}

	return count;
}

static b2Vec2 b2ComputePixelOverlapGradientAB( const b2PixelNormalContext* context )
{
	float eps = 0.5f * context->pixelSize;
	int oxp = b2CountPixelRawFeatureOverlap( context, (b2Vec2){ eps, 0.0f }, b2Vec2_zero );
	int oxm = b2CountPixelRawFeatureOverlap( context, (b2Vec2){ -eps, 0.0f }, b2Vec2_zero );
	int oyp = b2CountPixelRawFeatureOverlap( context, (b2Vec2){ 0.0f, eps }, b2Vec2_zero );
	int oym = b2CountPixelRawFeatureOverlap( context, (b2Vec2){ 0.0f, -eps }, b2Vec2_zero );
	b2Vec2 gradient = { (float)( oxp - oxm ), (float)( oyp - oym ) };
	if ( b2LengthSquared( gradient ) < 1.0e-10f )
	{
		return b2Vec2_zero;
	}

	return b2Normalize( gradient );
}

static b2Vec2 b2CanonicalizePixelNormalAB( const b2PixelNormalContext* context, b2Vec2 normal )
{
	if ( b2LengthSquared( normal ) < 1.0e-10f )
	{
		return (b2Vec2){ 1.0f, 0.0f };
	}

	normal = b2Normalize( normal );
	float eps = 0.5f * context->pixelSize;
	b2Vec2 separationA = b2MulSV( -eps, normal );
	b2Vec2 separationB = b2MulSV( eps, normal );
	int separated = b2CountPixelRawFeatureOverlap( context, separationA, separationB );
	int compressed = b2CountPixelRawFeatureOverlap( context, b2Neg( separationA ), b2Neg( separationB ) );
	if ( separated > compressed )
	{
		normal = b2Neg( normal );
	}

	return normal;
}

static b2Vec2 b2StabilizePixelNormalAB( const b2PixelNormalContext* context, b2Vec2 normal )
{
	int overlap = b2CountPixelRawFeatureOverlap( context, b2Vec2_zero, b2Vec2_zero );
	if ( overlap > 0 )
	{
		b2Vec2 gradient = b2ComputePixelOverlapGradientAB( context );
		if ( b2LengthSquared( gradient ) > 0.0f )
		{
			if ( b2Dot( normal, gradient ) < -0.15f )
			{
				normal = gradient;
			}
			else
			{
				b2Vec2 blended = b2MulAdd( normal, 0.25f, gradient );
				if ( b2LengthSquared( blended ) > 1.0e-10f )
				{
					normal = b2Normalize( blended );
				}
			}
		}
	}

	return b2CanonicalizePixelNormalAB( context, normal );
}

static bool b2PixelDepthCandidateLess( const b2PixelRawBuffer* buffer, int a, int b )
{
	const b2PixelRawContact* contactA = buffer->contacts + a;
	const b2PixelRawContact* contactB = buffer->contacts + b;
	if ( contactA->penetration != contactB->penetration )
	{
		return contactA->penetration > contactB->penetration;
	}

	return b2PixelContactTieLess( contactA, contactB );
}

static b2PixelManifoldSelection b2SelectPixelManifoldContacts( const b2PixelNormalContext* context, b2Vec2 centerNormal )
{
	const b2PixelRawBuffer* buffer = context->buffer;
	int primary = b2PickPixelDeepest( buffer );
	b2Vec2 normal = b2ReducePixelNormal( buffer );
	if ( b2LengthSquared( centerNormal ) > 0.0f && b2Dot( normal, centerNormal ) < 0.0f )
	{
		normal = b2Neg( normal );
	}
	normal = b2StabilizePixelNormalAB( context, normal );

	b2Vec2 tangent = b2LeftPerp( normal );
	int candidates[b2_maxPixelRawContacts];
	int candidateCount = 0;
	for ( int i = 0; i < buffer->count; ++i )
	{
		int candidate = i;
		int insert = candidateCount;
		while ( insert > 0 && b2PixelDepthCandidateLess( buffer, candidate, candidates[insert - 1] ) )
		{
			candidates[insert] = candidates[insert - 1];
			insert -= 1;
		}
		candidates[insert] = candidate;
		candidateCount += 1;
	}

	int secondary = primary;
	float bestScore = 0.0f;
	int maxCandidates = b2MinInt( candidateCount, 64 );
	for ( int i = 0; i < maxCandidates; ++i )
	{
		int index = candidates[i];
		if ( index == primary )
		{
			continue;
		}
		const b2PixelRawContact* contact = buffer->contacts + index;
		const b2PixelRawContact* first = buffer->contacts + primary;
		float separation = b2AbsFloat( b2Dot( b2Sub( contact->point, first->point ), tangent ) );
		if ( separation <= 0.45f * context->pixelSize )
		{
			continue;
		}
		float alignment = b2Dot( contact->normal, normal );
		if ( alignment < -0.25f )
		{
			continue;
		}
		float score = separation * ( 0.35f + 0.65f * b2MaxFloat( 0.0f, alignment ) ) + 0.15f * contact->penetration;
		if ( score > bestScore || ( score == bestScore && b2PixelContactTieLess( contact, buffer->contacts + secondary ) ) )
		{
			bestScore = score;
			secondary = index;
		}
	}

	b2PixelManifoldSelection selection;
	selection.normal = normal;
	selection.indexA = primary;
	selection.indexB = secondary;
	return selection;
}

static void b2FillPixelManifoldPoint( b2ManifoldPoint* mp, const b2PixelRawContact* contact, b2Transform xfA, b2Transform xfB )
{
	mp->clipPoint = contact->point;
	mp->anchorA = b2Sub( contact->point, xfA.p );
	mp->anchorB = b2Sub( contact->point, xfB.p );
	mp->separation = -contact->penetration;
	mp->id = b2MakePixelContactId( contact->featureIdA, contact->featureIdB );
}

b2Manifold b2CollidePixelShapesWithStats( const b2PixelShape* pixelA, b2Transform xfA, const b2PixelShape* pixelB,
										  b2Transform xfB, b2PixelNarrowphaseStats* stats )
{
	b2Manifold manifold = { 0 };
	if ( b2IsPixelShapeUsable( pixelA ) == false || b2IsPixelShapeUsable( pixelB ) == false )
	{
		b2PublishPixelStats( NULL, &manifold, false, false, true, stats );
		return manifold;
	}

	b2AABB aabbA = b2ComputePixelShapeAABB( pixelA, xfA );
	b2AABB aabbB = b2ComputePixelShapeAABB( pixelB, xfB );
	if ( b2AABB_Overlaps( aabbA, aabbB ) == false )
	{
		b2PublishPixelStats( NULL, &manifold, false, false, false, stats );
		return manifold;
	}

	b2PixelRawBuffer buffer = { 0 };
	b2GatherPixelCornerContacts( pixelA, xfA, pixelB, xfB, false, &buffer );
	b2GatherPixelCornerContacts( pixelB, xfB, pixelA, xfA, true, &buffer );
	if ( buffer.count == 0 )
	{
		b2GatherPixelEmbeddedContacts( pixelA, xfA, pixelB, xfB, false, &buffer );
		b2GatherPixelEmbeddedContacts( pixelB, xfB, pixelA, xfA, true, &buffer );
	}
	if ( buffer.count == 0 )
	{
		bool rescueCandidate = b2HasPixelRescueCandidate( &aabbA, &aabbB, &buffer );
		b2PublishPixelStats( &buffer, &manifold, rescueCandidate, false, false, stats );
		return manifold;
	}

	b2Vec2 centerNormal = b2Normalize( b2Sub( b2PixelShape_GetWorldCenter( pixelB, xfB ), b2PixelShape_GetWorldCenter( pixelA, xfA ) ) );
	float pixelSize = b2MinFloat( pixelA->asset->pixelSize, pixelB->asset->pixelSize );
	b2PixelNormalContext normalContext = {
		.pixelA = pixelA,
		.pixelB = pixelB,
		.xfA = xfA,
		.xfB = xfB,
		.buffer = &buffer,
		.pixelSize = pixelSize,
	};
	b2PixelManifoldSelection selection = b2SelectPixelManifoldContacts( &normalContext, centerNormal );
	manifold.normal = selection.normal;
	b2Vec2 tangent = b2LeftPerp( manifold.normal );

	b2FillPixelManifoldPoint( manifold.points + 0, buffer.contacts + selection.indexA, xfA, xfB );
	manifold.pointCount = 1;

	if ( selection.indexB != selection.indexA )
	{
		float tangentSeparation =
			b2AbsFloat( b2Dot( b2Sub( buffer.contacts[selection.indexB].point, buffer.contacts[selection.indexA].point ), tangent ) );
		float minSeparation = 0.25f * pixelSize;
		if ( tangentSeparation >= minSeparation )
		{
			b2FillPixelManifoldPoint( manifold.points + 1, buffer.contacts + selection.indexB, xfA, xfB );
			manifold.pointCount = 2;
		}
	}

	b2PublishPixelStats( &buffer, &manifold, false, false, false, stats );
	return manifold;
}

b2Manifold b2CollidePixelShapes( const b2PixelShape* pixelA, b2Transform xfA, const b2PixelShape* pixelB, b2Transform xfB )
{
	return b2CollidePixelShapesWithStats( pixelA, xfA, pixelB, xfB, NULL );
}

b2Manifold b2PixelShapeManifold( const b2Shape* shapeA, b2Transform xfA, const b2Shape* shapeB, b2Transform xfB )
{
	return b2CollidePixelShapes( &shapeA->pixel, xfA, &shapeB->pixel, xfB );
}
