// SPDX-FileCopyrightText: 2023 Erin Catto
// SPDX-License-Identifier: MIT

#include "pixel_contact.h"

#include "core.h"
#include "pixel_shape.h"

#include "box2d/box2d.h"

#include <math.h>
#include <stdint.h>

enum
{
	b2_maxPixelSourceFeatures = 1024,
	b2_maxPixelCellVisits = 1024,
	b2_maxPixelDiskTests = 256,
	b2_maxPixelRawContacts = 96,
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
		if ( buffer->sourceFeatures >= b2_maxPixelSourceFeatures )
		{
			buffer->sourceFeaturesCapped = true;
			return;
		}

		buffer->sourceFeatures += 1;
		const b2PixelFeatureRef* corner = assetSource->corners + cornerIndex;
		b2Vec2 worldCorner = b2TransformPoint( xfSource, b2PixelShape_GetPixelCenter( source, corner->x, corner->y ) );
		b2Vec2 targetLocal = b2Sub( b2InvTransformPoint( xfTarget, worldCorner ), target->localOrigin );
		int centerX = (int)floorf( targetLocal.x / assetTarget->pixelSize + 0.5f * (float)assetTarget->width );
		int centerY = (int)floorf( targetLocal.y / assetTarget->pixelSize + 0.5f * (float)assetTarget->height );
		centerX = b2ClampInt( centerX, 0, assetTarget->width - 1 );
		centerY = b2ClampInt( centerY, 0, assetTarget->height - 1 );

		for ( int dy = -searchCells; dy <= searchCells; ++dy )
		{
			for ( int dx = -searchCells; dx <= searchCells; ++dx )
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

				int x = centerX + dx;
				int y = centerY + dy;
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

static bool b2HasPixelRescueCandidate( const b2AABB* aabbA, const b2AABB* aabbB, const b2PixelRawBuffer* buffer )
{
	return buffer != NULL && buffer->count == 0 && b2AABB_Overlaps( *aabbA, *aabbB ) &&
		   ( buffer->sourceFeaturesCapped || buffer->cellVisitsCapped || buffer->diskTestsCapped ||
			 buffer->rawContactsCapped );
}

static int b2PickPixelPrimary( const b2PixelRawBuffer* buffer )
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

static int b2PickPixelSecondary( const b2PixelRawBuffer* buffer, int primary, b2Vec2 tangent )
{
	int best = primary;
	float bestDistance = 0.0f;
	b2Vec2 p0 = buffer->contacts[primary].point;
	for ( int i = 0; i < buffer->count; ++i )
	{
		if ( i == primary )
		{
			continue;
		}

		float distance = b2AbsFloat( b2Dot( b2Sub( buffer->contacts[i].point, p0 ), tangent ) );
		if ( distance > bestDistance )
		{
			bestDistance = distance;
			best = i;
		}
		else if ( distance == bestDistance && best != primary )
		{
			uint16_t idA = b2MakePixelContactId( buffer->contacts[i].featureIdA, buffer->contacts[i].featureIdB );
			uint16_t idB = b2MakePixelContactId( buffer->contacts[best].featureIdA, buffer->contacts[best].featureIdB );
			if ( idA < idB )
			{
				best = i;
			}
		}
	}
	return best;
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
		bool rescueCandidate = b2HasPixelRescueCandidate( &aabbA, &aabbB, &buffer );
		b2PublishPixelStats( &buffer, &manifold, rescueCandidate, false, false, stats );
		return manifold;
	}

	manifold.normal = b2ReducePixelNormal( &buffer );
	b2Vec2 centerNormal = b2Normalize( b2Sub( b2PixelShape_GetWorldCenter( pixelB, xfB ), b2PixelShape_GetWorldCenter( pixelA, xfA ) ) );
	if ( b2LengthSquared( centerNormal ) > 0.0f && b2Dot( manifold.normal, centerNormal ) < 0.25f )
	{
		manifold.normal = centerNormal;
	}
	b2Vec2 tangent = b2LeftPerp( manifold.normal );
	int primary = b2PickPixelPrimary( &buffer );
	int secondary = b2PickPixelSecondary( &buffer, primary, tangent );

	b2FillPixelManifoldPoint( manifold.points + 0, buffer.contacts + primary, xfA, xfB );
	manifold.pointCount = 1;

	if ( secondary != primary )
	{
		float tangentSeparation = b2AbsFloat( b2Dot( b2Sub( buffer.contacts[secondary].point, buffer.contacts[primary].point ), tangent ) );
		float minSeparation = 0.25f * b2MinFloat( pixelA->asset->pixelSize, pixelB->asset->pixelSize );
		if ( tangentSeparation >= minSeparation )
		{
			b2FillPixelManifoldPoint( manifold.points + 1, buffer.contacts + secondary, xfA, xfB );
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
