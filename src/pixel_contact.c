// SPDX-FileCopyrightText: 2023 Erin Catto
// SPDX-License-Identifier: MIT

#include "pixel_contact.h"

#include "core.h"
#include "pixel_shape.h"

#include "box2d/box2d.h"
#include "box2d/constants.h"

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
	b2_maxPixelEdgeSuppressionSlots = 32,
	b2_maxPixelContinuousSamples = 32,
	b2_pixelContinuousRefineIterations = 4,
};

static const float b2_pixelContinuousTunnelThreshold = 0.4f;

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
	int sourceFeatureIterations;
	int sourceFeatures;
	int cellVisits;
	int diskTests;
	int rawContactAttempts;
	bool sourceFeaturesCapped;
	bool cellVisitsCapped;
	bool diskTestsCapped;
	bool rawContactsCapped;
} b2PixelRawBuffer;

typedef struct b2PixelEdgeSuppressionSlot
{
	b2PixelRawContact contact;
	int patchX;
	int patchY;
	int line;
	uint8_t axis;
	bool occupied;
} b2PixelEdgeSuppressionSlot;

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

	stats->sourceFeatureIterations = buffer == NULL ? 0 : buffer->sourceFeatureIterations;
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

static b2Manifold b2SamplePixelContinuousManifold( const b2PixelShape* pixelA, const b2Sweep* sweepA, const b2PixelShape* pixelB,
												   const b2Sweep* sweepB, float fraction, b2PixelShapeContinuousStats* stats )
{
	if ( stats != NULL )
	{
		stats->queryCount += 1;
	}

	b2Transform xfA = b2GetSweepTransform( sweepA, fraction );
	b2Transform xfB = b2GetSweepTransform( sweepB, fraction );
	return b2CollidePixelShapesWithStats( pixelA, xfA, pixelB, xfB, NULL );
}

static float b2ComputePixelSweepMotionBound( const b2PixelShape* pixelA, const b2Sweep* sweepA, const b2PixelShape* pixelB,
											 const b2Sweep* sweepB, float maxFraction )
{
	b2Transform xfA1 = b2GetSweepTransform( sweepA, 0.0f );
	b2Transform xfA2 = b2GetSweepTransform( sweepA, maxFraction );
	b2Transform xfB1 = b2GetSweepTransform( sweepB, 0.0f );
	b2Transform xfB2 = b2GetSweepTransform( sweepB, maxFraction );

	b2Vec2 centerA1 = b2PixelShape_GetWorldCenter( pixelA, xfA1 );
	b2Vec2 centerA2 = b2PixelShape_GetWorldCenter( pixelA, xfA2 );
	b2Vec2 centerB1 = b2PixelShape_GetWorldCenter( pixelB, xfB1 );
	b2Vec2 centerB2 = b2PixelShape_GetWorldCenter( pixelB, xfB2 );
	b2Vec2 relativeDelta = b2Sub( b2Sub( centerB2, centerB1 ), b2Sub( centerA2, centerA1 ) );

	float extentA = b2GetPixelShapeMaxExtent( pixelA, b2GetPixelShapeCentroid( pixelA ) );
	float extentB = b2GetPixelShapeMaxExtent( pixelB, b2GetPixelShapeCentroid( pixelB ) );
	float angularA = extentA * b2AbsFloat( b2RelativeAngle( xfA1.q, xfA2.q ) );
	float angularB = extentB * b2AbsFloat( b2RelativeAngle( xfB1.q, xfB2.q ) );
	return b2Length( relativeDelta ) + angularA + angularB;
}

b2PixelShapeContinuousResult b2ComputePixelShapeContinuousHit( const b2PixelShape* pixelA, b2Sweep sweepA,
															  const b2PixelShape* pixelB, b2Sweep sweepB,
															  float maxFraction, b2PixelShapeContinuousStats* stats )
{
	b2PixelShapeContinuousResult result = {
		.hit = false,
		.fraction = maxFraction,
		.point = b2Vec2_zero,
		.normal = b2Vec2_zero,
	};

	if ( stats != NULL )
	{
		*stats = (b2PixelShapeContinuousStats){ 0 };
	}

	if ( maxFraction <= 0.0f || b2IsPixelShapeUsable( pixelA ) == false || b2IsPixelShapeUsable( pixelB ) == false )
	{
		return result;
	}

	maxFraction = b2MinFloat( maxFraction, 1.0f );

	b2Manifold initial = b2SamplePixelContinuousManifold( pixelA, &sweepA, pixelB, &sweepB, 0.0f, stats );
	if ( initial.pointCount > 0 )
	{
		if ( stats != NULL )
		{
			stats->initialOverlap = true;
		}
		return result;
	}

	float pixelSize = b2MinFloat( pixelA->asset->pixelSize, pixelB->asset->pixelSize );
	float threshold = b2_pixelContinuousTunnelThreshold * pixelSize;
	if ( threshold <= 0.0f )
	{
		return result;
	}

	float motionBound = b2ComputePixelSweepMotionBound( pixelA, &sweepA, pixelB, &sweepB, maxFraction );
	int sampleCount = b2MaxInt( 1, (int)ceilf( motionBound / threshold ) );
	if ( sampleCount > b2_maxPixelContinuousSamples )
	{
		sampleCount = b2_maxPixelContinuousSamples;
		if ( stats != NULL )
		{
			stats->sampleLimitReached = true;
		}
	}

	float previousFraction = 0.0f;
	for ( int i = 1; i <= sampleCount; ++i )
	{
		float sampleFraction = maxFraction * (float)i / (float)sampleCount;
		if ( stats != NULL )
		{
			stats->sampleCount += 1;
		}

		b2Manifold manifold = b2SamplePixelContinuousManifold( pixelA, &sweepA, pixelB, &sweepB, sampleFraction, stats );
		if ( manifold.pointCount == 0 )
		{
			previousFraction = sampleFraction;
			continue;
		}

		float low = previousFraction;
		float high = sampleFraction;
		float bestFraction = sampleFraction;
		b2Manifold bestManifold = manifold;
		for ( int refineIndex = 0; refineIndex < b2_pixelContinuousRefineIterations; ++refineIndex )
		{
			float mid = 0.5f * ( low + high );
			if ( stats != NULL )
			{
				stats->refineCount += 1;
			}

			b2Manifold refined = b2SamplePixelContinuousManifold( pixelA, &sweepA, pixelB, &sweepB, mid, stats );
			if ( refined.pointCount > 0 )
			{
				high = mid;
				bestFraction = mid;
				bestManifold = refined;
			}
			else
			{
				low = mid;
			}
		}

		result.hit = true;
		result.fraction = bestFraction;
		result.normal = bestManifold.normal;
		result.point = bestManifold.points[0].clipPoint;
		return result;
	}

	return result;
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

static int b2LowerBoundPixelCornerId( const b2PixelAsset* asset, int first, int last, int featureId )
{
	int lo = first;
	int hi = last;
	while ( lo < hi )
	{
		int mid = lo + ( hi - lo ) / 2;
		if ( (int)asset->corners[mid].id < featureId )
		{
			lo = mid + 1;
		}
		else
		{
			hi = mid;
		}
	}

	return lo;
}

static bool b2ComputePixelSourceCandidateRange( const b2PixelShape* source, b2Transform xfSource,
												const b2PixelShape* target, b2Transform xfTarget, int searchCells,
												int* minX, int* maxX, int* minY, int* maxY )
{
	const b2PixelAsset* assetSource = source->asset;
	const b2PixelAsset* assetTarget = target->asset;
	float inflate = (float)searchCells * assetTarget->pixelSize;
	b2AABB targetLocal = assetTarget->occupiedAABB;
	targetLocal.lowerBound = b2Add( targetLocal.lowerBound, target->localOrigin );
	targetLocal.upperBound = b2Add( targetLocal.upperBound, target->localOrigin );
	targetLocal.lowerBound.x -= inflate;
	targetLocal.lowerBound.y -= inflate;
	targetLocal.upperBound.x += inflate;
	targetLocal.upperBound.y += inflate;

	b2Vec2 targetPoints[4] = {
		{ targetLocal.lowerBound.x, targetLocal.lowerBound.y },
		{ targetLocal.upperBound.x, targetLocal.lowerBound.y },
		{ targetLocal.upperBound.x, targetLocal.upperBound.y },
		{ targetLocal.lowerBound.x, targetLocal.upperBound.y },
	};

	float sourceLowerX = FLT_MAX;
	float sourceLowerY = FLT_MAX;
	float sourceUpperX = -FLT_MAX;
	float sourceUpperY = -FLT_MAX;
	for ( int i = 0; i < 4; ++i )
	{
		b2Vec2 sourceLocal = b2InvTransformPoint( xfSource, b2TransformPoint( xfTarget, targetPoints[i] ) );
		sourceLowerX = b2MinFloat( sourceLowerX, sourceLocal.x );
		sourceLowerY = b2MinFloat( sourceLowerY, sourceLocal.y );
		sourceUpperX = b2MaxFloat( sourceUpperX, sourceLocal.x );
		sourceUpperY = b2MaxFloat( sourceUpperY, sourceLocal.y );
	}

	float sourcePixelSize = assetSource->pixelSize;
	float sourceHalfWidth = 0.5f * (float)assetSource->width;
	float sourceHalfHeight = 0.5f * (float)assetSource->height;
	float slop = b2MaxFloat( B2_LINEAR_SLOP, 1.0e-5f * sourcePixelSize );
	int candidateMinX = (int)ceilf( ( sourceLowerX - source->localOrigin.x - slop ) / sourcePixelSize + sourceHalfWidth - 0.5f );
	int candidateMaxX = (int)floorf( ( sourceUpperX - source->localOrigin.x + slop ) / sourcePixelSize + sourceHalfWidth - 0.5f );
	int candidateMinY = (int)ceilf( ( sourceLowerY - source->localOrigin.y - slop ) / sourcePixelSize + sourceHalfHeight - 0.5f );
	int candidateMaxY = (int)floorf( ( sourceUpperY - source->localOrigin.y + slop ) / sourcePixelSize + sourceHalfHeight - 0.5f );

	candidateMinX = b2MaxInt( 0, candidateMinX );
	candidateMaxX = b2MinInt( assetSource->width - 1, candidateMaxX );
	candidateMinY = b2MaxInt( 0, candidateMinY );
	candidateMaxY = b2MinInt( assetSource->height - 1, candidateMaxY );
	if ( candidateMinX > candidateMaxX || candidateMinY > candidateMaxY )
	{
		return false;
	}

	*minX = candidateMinX;
	*maxX = candidateMaxX;
	*minY = candidateMinY;
	*maxY = candidateMaxY;
	return true;
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

static bool b2PixelFeatureTypeIsEdge( uint8_t type )
{
	return type == b2_pixelFeatureEdgeX || type == b2_pixelFeatureEdgeY;
}

static bool b2PixelFeatureTypeIsAxisEdge( uint8_t type )
{
	return type == b2_pixelFeatureEdgeX || type == b2_pixelFeatureEdgeY;
}

static uint8_t b2PixelFeatureEdgeAxis( uint8_t type )
{
	if ( type == b2_pixelFeatureEdgeX )
	{
		return 1;
	}
	if ( type == b2_pixelFeatureEdgeY )
	{
		return 2;
	}
	return 0;
}

static bool b2PixelPatchHasUniformAxisEdges( const b2PixelAsset* asset, int patchMinX, int patchMinY, uint8_t axis )
{
	if ( axis == 0 )
	{
		return false;
	}

	int edgeCount = 0;
	for ( int y = patchMinY; y <= patchMinY + 1; ++y )
	{
		for ( int x = patchMinX; x <= patchMinX + 1; ++x )
		{
			uint8_t type = b2PixelAsset_GetFeatureType( asset, x, y );
			if ( type == b2_pixelFeatureEmpty || type == b2_pixelFeatureInternal )
			{
				continue;
			}

			if ( b2PixelFeatureEdgeAxis( type ) != axis )
			{
				return false;
			}
			edgeCount += 1;
		}
	}

	return edgeCount > 0;
}

static bool b2PixelShouldUseAxisPatchHandling( const b2PixelAsset* asset, int patchMinX, int patchMinY, int x, int y,
											   uint8_t type )
{
	uint8_t axis = b2PixelFeatureEdgeAxis( type );
	return axis != 0 && patchMinX <= x && x <= patchMinX + 1 && patchMinY <= y && y <= patchMinY + 1 &&
		   b2PixelPatchHasUniformAxisEdges( asset, patchMinX, patchMinY, axis );
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

static bool b2PixelLocalNormalPointsToOccupiedNeighbor( const b2PixelAsset* asset, int x, int y, b2Vec2 normal )
{
	if ( normal.x > 1.0e-6f && b2PixelAsset_IsOccupied( asset, x + 1, y ) )
	{
		return true;
	}
	if ( normal.x < -1.0e-6f && b2PixelAsset_IsOccupied( asset, x - 1, y ) )
	{
		return true;
	}
	if ( normal.y > 1.0e-6f && b2PixelAsset_IsOccupied( asset, x, y + 1 ) )
	{
		return true;
	}
	if ( normal.y < -1.0e-6f && b2PixelAsset_IsOccupied( asset, x, y - 1 ) )
	{
		return true;
	}
	return false;
}

static float b2PixelSignOrOne( float value )
{
	return value < 0.0f ? -1.0f : 1.0f;
}

static bool b2TryPixelAxisNormal( const b2PixelAsset* asset, int x, int y, b2Vec2 normal, b2Vec2* out )
{
	if ( b2LengthSquared( normal ) == 0.0f || b2PixelLocalNormalPointsToOccupiedNeighbor( asset, x, y, normal ) )
	{
		return false;
	}

	*out = normal;
	return true;
}

static bool b2PixelFeatureHasSingleOpenAxisSide( const b2PixelAsset* asset, int x, int y, uint8_t type )
{
	if ( type == b2_pixelFeatureEdgeX )
	{
		bool leftOpen = b2PixelAsset_IsOccupied( asset, x - 1, y ) == false;
		bool rightOpen = b2PixelAsset_IsOccupied( asset, x + 1, y ) == false;
		return leftOpen != rightOpen;
	}
	if ( type == b2_pixelFeatureEdgeY )
	{
		bool upOpen = b2PixelAsset_IsOccupied( asset, x, y - 1 ) == false;
		bool downOpen = b2PixelAsset_IsOccupied( asset, x, y + 1 ) == false;
		return upOpen != downOpen;
	}
	return false;
}

static b2Vec2 b2PixelFeatureFallbackLocalNormal( const b2PixelAsset* asset, int x, int y, uint8_t type, b2Vec2 candidate )
{
	b2Vec2 normal = b2Vec2_zero;
	if ( type == b2_pixelFeatureEdgeX )
	{
		bool leftOpen = b2PixelAsset_IsOccupied( asset, x - 1, y ) == false;
		bool rightOpen = b2PixelAsset_IsOccupied( asset, x + 1, y ) == false;
		if ( leftOpen == rightOpen )
		{
			return b2Vec2_zero;
		}
		if ( b2TryPixelAxisNormal( asset, x, y, (b2Vec2){ rightOpen ? 1.0f : -1.0f, 0.0f }, &normal ) )
		{
			return normal;
		}
		return b2Vec2_zero;
	}
	if ( type == b2_pixelFeatureEdgeY )
	{
		bool upOpen = b2PixelAsset_IsOccupied( asset, x, y - 1 ) == false;
		bool downOpen = b2PixelAsset_IsOccupied( asset, x, y + 1 ) == false;
		if ( upOpen == downOpen )
		{
			return b2Vec2_zero;
		}
		if ( b2TryPixelAxisNormal( asset, x, y, (b2Vec2){ 0.0f, downOpen ? 1.0f : -1.0f }, &normal ) )
		{
			return normal;
		}
		return b2Vec2_zero;
	}
	float sx = b2PixelSignOrOne( candidate.x );
	float sy = b2PixelSignOrOne( candidate.y );
	bool preferX = b2AbsFloat( candidate.x ) >= b2AbsFloat( candidate.y );
	if ( preferX )
	{
		if ( b2TryPixelAxisNormal( asset, x, y, (b2Vec2){ -sx, 0.0f }, &normal ) ||
			 b2TryPixelAxisNormal( asset, x, y, (b2Vec2){ 0.0f, -sy }, &normal ) ||
			 b2TryPixelAxisNormal( asset, x, y, (b2Vec2){ sx, 0.0f }, &normal ) ||
			 b2TryPixelAxisNormal( asset, x, y, (b2Vec2){ 0.0f, sy }, &normal ) )
		{
			return normal;
		}
	}
	else
	{
		if ( b2TryPixelAxisNormal( asset, x, y, (b2Vec2){ 0.0f, -sy }, &normal ) ||
			 b2TryPixelAxisNormal( asset, x, y, (b2Vec2){ -sx, 0.0f }, &normal ) ||
			 b2TryPixelAxisNormal( asset, x, y, (b2Vec2){ 0.0f, sy }, &normal ) ||
			 b2TryPixelAxisNormal( asset, x, y, (b2Vec2){ sx, 0.0f }, &normal ) )
		{
			return normal;
		}
	}

	return b2Vec2_zero;
}

static b2Vec2 b2ClampPixelFeatureLocalNormal( const b2PixelAsset* asset, int x, int y, uint8_t type, b2Vec2 candidate )
{
	if ( b2LengthSquared( candidate ) < 1.0e-10f )
	{
		return b2PixelFeatureFallbackLocalNormal( asset, x, y, type, candidate );
	}

	b2Vec2 clamped = candidate;
	if ( type == b2_pixelFeatureEdgeX )
	{
		if ( b2PixelFeatureHasSingleOpenAxisSide( asset, x, y, type ) == false )
		{
			return b2Vec2_zero;
		}
		clamped.y = 0.0f;
	}
	else if ( type == b2_pixelFeatureEdgeY )
	{
		if ( b2PixelFeatureHasSingleOpenAxisSide( asset, x, y, type ) == false )
		{
			return b2Vec2_zero;
		}
		clamped.x = 0.0f;
	}

	if ( clamped.x > 1.0e-6f && b2PixelAsset_IsOccupied( asset, x + 1, y ) )
	{
		clamped.x = 0.0f;
	}
	else if ( clamped.x < -1.0e-6f && b2PixelAsset_IsOccupied( asset, x - 1, y ) )
	{
		clamped.x = 0.0f;
	}
	if ( clamped.y > 1.0e-6f && b2PixelAsset_IsOccupied( asset, x, y + 1 ) )
	{
		clamped.y = 0.0f;
	}
	else if ( clamped.y < -1.0e-6f && b2PixelAsset_IsOccupied( asset, x, y - 1 ) )
	{
		clamped.y = 0.0f;
	}

	if ( b2LengthSquared( clamped ) > 1.0e-10f )
	{
		return b2Normalize( clamped );
	}

	return b2PixelFeatureFallbackLocalNormal( asset, x, y, type, candidate );
}

static b2Vec2 b2ClampPixelCornerLocalNormal( const b2PixelAsset* asset, int x, int y, b2Vec2 candidate )
{
	if ( b2LengthSquared( candidate ) < 1.0e-10f )
	{
		return b2Vec2_zero;
	}

	b2Vec2 clamped = candidate;
	if ( clamped.x > 1.0e-6f && b2PixelAsset_IsOccupied( asset, x + 1, y ) )
	{
		clamped.x = 0.0f;
	}
	else if ( clamped.x < -1.0e-6f && b2PixelAsset_IsOccupied( asset, x - 1, y ) )
	{
		clamped.x = 0.0f;
	}
	if ( clamped.y > 1.0e-6f && b2PixelAsset_IsOccupied( asset, x, y + 1 ) )
	{
		clamped.y = 0.0f;
	}
	else if ( clamped.y < -1.0e-6f && b2PixelAsset_IsOccupied( asset, x, y - 1 ) )
	{
		clamped.y = 0.0f;
	}

	if ( b2LengthSquared( clamped ) < 1.0e-10f )
	{
		return b2Vec2_zero;
	}

	return b2Normalize( clamped );
}

static b2Vec2 b2ComputePixelPairNormalAB( const b2PixelShape* shapeA, b2Transform xfA, const b2PixelFeatureRef* featureA,
										  const b2PixelShape* shapeB, b2Transform xfB, int xB, int yB, uint8_t typeB,
										  b2Vec2 deltaNormal, bool axisClampEnabled )
{
	const b2PixelAsset* assetA = shapeA->asset;
	const b2PixelAsset* assetB = shapeB->asset;
	b2Vec2 localA = b2PixelShape_GetPixelCenter( shapeA, featureA->x, featureA->y );
	b2Vec2 localB = b2PixelShape_GetPixelCenter( shapeB, xB, yB );
	b2Vec2 worldA = b2TransformPoint( xfA, localA );
	b2Vec2 worldB = b2TransformPoint( xfB, localB );
	b2Vec2 d = b2Sub( worldB, worldA );

	if ( axisClampEnabled && b2PixelFeatureTypeIsAxisEdge( featureA->type ) &&
		 b2PixelFeatureTypeIsEdge( typeB ) == false )
	{
		b2Vec2 localDelta = b2InvRotateVector( xfA.q, d );
		b2Vec2 outward = b2ClampPixelFeatureLocalNormal( assetA, featureA->x, featureA->y, featureA->type, localDelta );
		if ( b2LengthSquared( outward ) > 0.0f )
		{
			return b2RotateVector( xfA.q, outward );
		}
	}

	if ( axisClampEnabled && b2PixelFeatureTypeIsAxisEdge( typeB ) &&
		 b2PixelFeatureTypeIsEdge( featureA->type ) == false )
	{
		b2Vec2 localDelta = b2InvRotateVector( xfB.q, b2Neg( d ) );
		b2Vec2 outward = b2ClampPixelFeatureLocalNormal( assetB, xB, yB, typeB, localDelta );
		if ( b2LengthSquared( outward ) > 0.0f )
		{
			return b2Neg( b2RotateVector( xfB.q, outward ) );
		}
	}

	if ( b2PixelFeatureTypeIsEdge( featureA->type ) && b2PixelFeatureTypeIsEdge( typeB ) == false )
	{
		b2Vec2 outward = b2PixelFeatureOutwardLocal( assetA, featureA->x, featureA->y );
		if ( b2LengthSquared( outward ) > 0.0f )
		{
			return b2RotateVector( xfA.q, outward );
		}
	}

	if ( b2PixelFeatureTypeIsEdge( typeB ) && b2PixelFeatureTypeIsEdge( featureA->type ) == false )
	{
		b2Vec2 outward = b2PixelFeatureOutwardLocal( assetB, xB, yB );
		if ( b2LengthSquared( outward ) > 0.0f )
		{
			return b2Neg( b2RotateVector( xfB.q, outward ) );
		}
	}

	if ( featureA->type == b2_pixelFeatureCorner )
	{
		b2Vec2 localAFromA = b2InvRotateVector( xfA.q, deltaNormal );
		b2Vec2 clampedA = b2ClampPixelCornerLocalNormal( assetA, featureA->x, featureA->y, localAFromA );
		if ( b2LengthSquared( clampedA ) > 0.0f )
		{
			b2Vec2 worldAClamped = b2RotateVector( xfA.q, clampedA );
			if ( b2Dot( worldAClamped, deltaNormal ) > 0.0f )
			{
				deltaNormal = worldAClamped;
			}
		}
	}

	if ( typeB == b2_pixelFeatureCorner )
	{
		b2Vec2 localBFromB = b2InvRotateVector( xfB.q, b2Neg( deltaNormal ) );
		b2Vec2 clampedB = b2ClampPixelCornerLocalNormal( assetB, xB, yB, localBFromB );
		if ( b2LengthSquared( clampedB ) > 0.0f )
		{
			b2Vec2 worldBClamped = b2Neg( b2RotateVector( xfB.q, clampedB ) );
			if ( b2Dot( worldBClamped, deltaNormal ) > 0.0f )
			{
				deltaNormal = worldBClamped;
			}
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

static bool b2PixelSuppressedContactLess( const b2PixelRawContact* a, const b2PixelRawContact* b )
{
	if ( a->penetration != b->penetration )
	{
		return a->penetration > b->penetration;
	}

	return b2MakePixelContactId( a->featureIdA, a->featureIdB ) < b2MakePixelContactId( b->featureIdA, b->featureIdB );
}

static void b2QueuePixelSuppressedEdgeContact( b2PixelEdgeSuppressionSlot* slots, int* slotCount, int patchX, int patchY,
											   int x, int y, uint8_t type, b2PixelRawContact contact,
											   b2PixelRawBuffer* buffer )
{
	uint8_t axis = b2PixelFeatureEdgeAxis( type );
	if ( axis == 0 )
	{
		b2PushPixelRawContact( buffer, contact );
		return;
	}

	int line = axis == 1 ? x : y;
	for ( int i = 0; i < *slotCount; ++i )
	{
		b2PixelEdgeSuppressionSlot* slot = slots + i;
		if ( slot->occupied && slot->patchX == patchX && slot->patchY == patchY && slot->axis == axis &&
			 slot->line == line )
		{
			if ( b2PixelSuppressedContactLess( &contact, &slot->contact ) )
			{
				slot->contact = contact;
			}
			return;
		}
	}

	if ( *slotCount < b2_maxPixelEdgeSuppressionSlots )
	{
		b2PixelEdgeSuppressionSlot* slot = slots + *slotCount;
		slot->contact = contact;
		slot->patchX = patchX;
		slot->patchY = patchY;
		slot->line = line;
		slot->axis = axis;
		slot->occupied = true;
		*slotCount += 1;
		return;
	}

	b2PushPixelRawContact( buffer, contact );
}

static void b2FlushPixelSuppressedEdgeContacts( b2PixelEdgeSuppressionSlot* slots, int slotCount, b2PixelRawBuffer* buffer )
{
	for ( int i = 0; i < slotCount; ++i )
	{
		if ( slots[i].occupied )
		{
			b2PushPixelRawContact( buffer, slots[i].contact );
		}
	}
}

static bool b2DiskTestPixelFeatures( const b2PixelShape* shapeA, b2Transform xfA, const b2PixelFeatureRef* featureA,
									 const b2PixelShape* shapeB, b2Transform xfB, int xB, int yB, uint8_t typeB,
									 bool reverseToOriginalOrder, b2Vec2 fallbackNormal, bool axisClampEnabled,
									 b2PixelRawContact* out )
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
	normal = b2ComputePixelPairNormalAB( shapeA, xfA, featureA, shapeB, xfB, xB, yB, typeB, normal, axisClampEnabled );

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

	int sourceMinX = 0;
	int sourceMaxX = 0;
	int sourceMinY = 0;
	int sourceMaxY = 0;
	if ( b2ComputePixelSourceCandidateRange( source, xfSource, target, xfTarget, searchCells,
											 &sourceMinX, &sourceMaxX, &sourceMinY, &sourceMaxY ) == false )
	{
		return;
	}

	int cornerSearchStart = 0;
	for ( int sourceY = sourceMinY; sourceY <= sourceMaxY; ++sourceY )
	{
		int firstFeatureId = sourceY * assetSource->width + sourceMinX + 1;
		if ( firstFeatureId > UINT16_MAX )
		{
			break;
		}

		int lastFeatureId = sourceY * assetSource->width + sourceMaxX + 1;
		lastFeatureId = b2MinInt( lastFeatureId, UINT16_MAX );
		int cornerIndex =
			b2LowerBoundPixelCornerId( assetSource, cornerSearchStart, assetSource->cornerCount, firstFeatureId );
		cornerSearchStart = cornerIndex;
		for ( ; cornerIndex < assetSource->cornerCount && (int)assetSource->corners[cornerIndex].id <= lastFeatureId;
			  ++cornerIndex )
		{
			const b2PixelFeatureRef* corner = assetSource->corners + cornerIndex;
			buffer->sourceFeatureIterations += 1;
			b2Vec2 worldCorner = b2TransformPoint( xfSource, b2PixelShape_GetPixelCenter( source, corner->x, corner->y ) );
			b2Vec2 targetLocal = b2Sub( b2InvTransformPoint( xfTarget, worldCorner ), target->localOrigin );
			int centerX = (int)floorf( targetLocal.x / assetTarget->pixelSize + 0.5f * (float)assetTarget->width );
			int centerY = (int)floorf( targetLocal.y / assetTarget->pixelSize + 0.5f * (float)assetTarget->height );
			int patchMinX = (int)floorf( targetLocal.x / assetTarget->pixelSize + 0.5f * (float)assetTarget->width - 0.5f );
			int patchMinY = (int)floorf( targetLocal.y / assetTarget->pixelSize + 0.5f * (float)assetTarget->height - 0.5f );
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
			b2PixelEdgeSuppressionSlot suppressedEdges[b2_maxPixelEdgeSuppressionSlots] = { 0 };
			int suppressedEdgeCount = 0;
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
					bool useAxisPatchHandling =
						b2PixelShouldUseAxisPatchHandling( assetTarget, patchMinX, patchMinY, x, y, typeB );
					b2PixelRawContact contact;
					if ( b2DiskTestPixelFeatures( source, xfSource, corner, target, xfTarget, x, y, typeB,
												   reverseToOriginalOrder, fallbackNormal, useAxisPatchHandling, &contact ) )
					{
						if ( useAxisPatchHandling )
						{
							b2QueuePixelSuppressedEdgeContact( suppressedEdges, &suppressedEdgeCount, patchMinX, patchMinY, x, y,
															   typeB, contact, buffer );
						}
						else
						{
							b2PushPixelRawContact( buffer, contact );
						}
					}
				}
			}
			b2FlushPixelSuppressedEdgeContacts( suppressedEdges, suppressedEdgeCount, buffer );
		}

		cornerSearchStart = cornerIndex;
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
		buffer->sourceFeatureIterations += 1;
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
	if ( buffer->count >= 8 && b2LengthSquared( centerNormal ) > 0.0f )
	{
		float centerAlignment = b2Dot( normal, centerNormal );
		if ( centerAlignment > 0.0f && centerAlignment < 0.5f )
		{
			normal = centerNormal;
		}
	}

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
