// SPDX-FileCopyrightText: 2023 Erin Catto
// SPDX-License-Identifier: MIT

#include "pixel_shape.h"

#include "core.h"

#include <math.h>
#include <stdint.h>

static bool b2IsPixelAssetUsable( const b2PixelAsset* asset )
{
	return asset != NULL && asset->width > 0 && asset->height > 0 && asset->pixelSize > 0.0f;
}

static bool b2FeatureRefsAreCanonical( const b2PixelFeatureRef* features, int count )
{
	uint16_t previousId = 0;
	for ( int i = 0; i < count; ++i )
	{
		const b2PixelFeatureRef* feature = features + i;
		if ( feature->id == 0 || feature->id <= previousId )
		{
			return false;
		}

		if ( feature->type != b2_pixelFeatureCorner && feature->type != b2_pixelFeatureEdge )
		{
			return false;
		}

		previousId = feature->id;
	}

	return true;
}

bool b2IsPixelShapeUsable( const b2PixelShape* shape )
{
	const b2PixelAsset* asset = shape == NULL ? NULL : shape->asset;
	return b2IsPixelAssetUsable( asset ) && asset->occupancyBits != NULL && asset->featureTypes != NULL && asset->chunks != NULL &&
		   asset->chunkCount > 0 && asset->corners != NULL && asset->cornerCount > 0 && asset->solidCount > 0 &&
		   asset->occupancyWordCount >= ( asset->width * asset->height + 63 ) / 64 && b2IsValidVec2( asset->centroid ) &&
		   b2IsValidFloat( asset->rotationalInertia ) && asset->rotationalInertia >= 0.0f &&
		   b2IsValidVec2( shape->localOrigin ) && b2IsValidFloat( shape->diskRadius ) && shape->diskRadius >= 0.0f;
}

bool b2IsPixelAssetValid( const b2PixelAsset* asset )
{
	if ( b2IsPixelAssetUsable( asset ) == false || asset->occupancyBits == NULL || asset->featureTypes == NULL ||
		 asset->chunks == NULL || asset->chunkCount <= 0 || asset->corners == NULL || asset->cornerCount <= 0 ||
		 asset->solidCount <= 0 || b2IsValidAABB( asset->occupiedAABB ) == false || b2IsValidVec2( asset->centroid ) == false ||
		 b2IsValidFloat( asset->rotationalInertia ) == false || asset->rotationalInertia < 0.0f )
	{
		return false;
	}

	if ( asset->occupancyWordCount < ( asset->width * asset->height + 63 ) / 64 )
	{
		return false;
	}

	if ( b2FeatureRefsAreCanonical( asset->corners, asset->cornerCount ) == false )
	{
		return false;
	}

	if ( asset->edgeCount > 0 )
	{
		if ( asset->edges == NULL || b2FeatureRefsAreCanonical( asset->edges, asset->edgeCount ) == false )
		{
			return false;
		}
	}

	int16_t previousY = INT16_MIN;
	int16_t previousX = INT16_MIN;
	for ( int i = 0; i < asset->chunkCount; ++i )
	{
		const b2PixelChunk* chunk = asset->chunks + i;
		if ( chunk->width == 0 || chunk->height == 0 || chunk->solidCount == 0 || b2IsValidAABB( chunk->localAABB ) == false )
		{
			return false;
		}

		if ( i > 0 && ( chunk->y < previousY || ( chunk->y == previousY && chunk->x <= previousX ) ) )
		{
			return false;
		}

		if ( chunk->firstCorner + chunk->cornerCount > (uint32_t)asset->cornerCount ||
			 chunk->firstEdge + chunk->edgeCount > (uint32_t)asset->edgeCount )
		{
			return false;
		}

		previousY = chunk->y;
		previousX = chunk->x;
	}

	return true;
}

static bool b2PixelIndexInRange( const b2PixelAsset* asset, int x, int y )
{
	return 0 <= x && x < asset->width && 0 <= y && y < asset->height;
}

static bool b2PixelAsset_GetBit( const b2PixelAsset* asset, int index )
{
	if ( asset->occupancyBits == NULL || index < 0 )
	{
		return false;
	}

	int wordIndex = index >> 6;
	if ( wordIndex < 0 || wordIndex >= asset->occupancyWordCount )
	{
		return false;
	}

	uint64_t mask = UINT64_C( 1 ) << ( index & 63 );
	return ( asset->occupancyBits[wordIndex] & mask ) != 0;
}

bool b2PixelAsset_IsOccupied( const b2PixelAsset* asset, int x, int y )
{
	if ( b2IsPixelAssetUsable( asset ) == false || b2PixelIndexInRange( asset, x, y ) == false )
	{
		return false;
	}

	return b2PixelAsset_GetBit( asset, y * asset->width + x );
}

uint8_t b2PixelAsset_GetFeatureType( const b2PixelAsset* asset, int x, int y )
{
	if ( b2IsPixelAssetUsable( asset ) == false || b2PixelIndexInRange( asset, x, y ) == false )
	{
		return b2_pixelFeatureEmpty;
	}

	int index = y * asset->width + x;
	if ( asset->featureTypes != NULL )
	{
		return asset->featureTypes[index];
	}

	return b2PixelAsset_GetBit( asset, index ) ? b2_pixelFeatureEdge : b2_pixelFeatureEmpty;
}

uint16_t b2PixelAsset_GetFeatureId( const b2PixelAsset* asset, int x, int y )
{
	if ( b2IsPixelAssetUsable( asset ) == false || b2PixelIndexInRange( asset, x, y ) == false )
	{
		return 0;
	}

	int index = y * asset->width + x;
	if ( index >= UINT16_MAX )
	{
		return UINT16_MAX;
	}

	return (uint16_t)( index + 1 );
}

b2Vec2 b2PixelShape_GetPixelCenter( const b2PixelShape* shape, int x, int y )
{
	const b2PixelAsset* asset = shape->asset;
	float pixelSize = asset->pixelSize;
	b2Vec2 p = {
		( (float)x + 0.5f - 0.5f * (float)asset->width ) * pixelSize + shape->localOrigin.x,
		( (float)y + 0.5f - 0.5f * (float)asset->height ) * pixelSize + shape->localOrigin.y,
	};
	return p;
}

bool b2PixelShape_GetLocalInfo( const b2PixelShape* shape, b2Vec2 localPoint, b2PixelLocalInfo* info )
{
	const b2PixelAsset* asset = shape == NULL ? NULL : shape->asset;
	if ( b2IsPixelAssetUsable( asset ) == false )
	{
		return false;
	}

	b2Vec2 p = b2Sub( localPoint, shape->localOrigin );
	int x = (int)floorf( p.x / asset->pixelSize + 0.5f * (float)asset->width );
	int y = (int)floorf( p.y / asset->pixelSize + 0.5f * (float)asset->height );
	if ( b2PixelIndexInRange( asset, x, y ) == false )
	{
		return false;
	}

	if ( info != NULL )
	{
		info->x = x;
		info->y = y;
		info->index = y * asset->width + x;
	}

	return true;
}

bool b2PointInPixelShape( const b2PixelShape* shape, b2Vec2 localPoint )
{
	b2PixelLocalInfo info;
	if ( b2PixelShape_GetLocalInfo( shape, localPoint, &info ) == false )
	{
		return false;
	}

	return b2PixelAsset_GetBit( shape->asset, info.index );
}

static b2AABB b2GetPixelShapeLocalAABB( const b2PixelShape* shape )
{
	const b2PixelAsset* asset = shape == NULL ? NULL : shape->asset;
	if ( b2IsPixelAssetUsable( asset ) == false || asset->solidCount <= 0 )
	{
		return (b2AABB){ shape == NULL ? b2Vec2_zero : shape->localOrigin, shape == NULL ? b2Vec2_zero : shape->localOrigin };
	}

	b2AABB aabb = asset->occupiedAABB;
	aabb.lowerBound = b2Add( aabb.lowerBound, shape->localOrigin );
	aabb.upperBound = b2Add( aabb.upperBound, shape->localOrigin );
	return aabb;
}

b2AABB b2ComputePixelShapeAABB( const b2PixelShape* shape, b2Transform transform )
{
	b2AABB local = b2GetPixelShapeLocalAABB( shape );
	b2Vec2 points[4] = {
		{ local.lowerBound.x, local.lowerBound.y },
		{ local.upperBound.x, local.lowerBound.y },
		{ local.upperBound.x, local.upperBound.y },
		{ local.lowerBound.x, local.upperBound.y },
	};

	for ( int i = 0; i < 4; ++i )
	{
		points[i] = b2TransformPoint( transform, points[i] );
	}

	float radius = shape != NULL && shape->asset != NULL ? shape->diskRadius * shape->asset->pixelSize : 0.0f;
	return b2MakeAABB( points, 4, radius );
}

b2Vec2 b2GetPixelShapeCentroid( const b2PixelShape* shape )
{
	const b2PixelAsset* asset = shape == NULL ? NULL : shape->asset;
	if ( b2IsPixelAssetUsable( asset ) == false || asset->solidCount <= 0 )
	{
		return shape == NULL ? b2Vec2_zero : shape->localOrigin;
	}

	return b2Add( asset->centroid, shape->localOrigin );
}

b2MassData b2ComputePixelShapeMass( const b2PixelShape* shape, float density )
{
	b2MassData massData = { 0 };
	const b2PixelAsset* asset = shape == NULL ? NULL : shape->asset;
	if ( b2IsPixelAssetUsable( asset ) == false || asset->solidCount <= 0 || density == 0.0f )
	{
		massData.center = shape == NULL ? b2Vec2_zero : shape->localOrigin;
		return massData;
	}

	float pixelArea = asset->pixelSize * asset->pixelSize;
	massData.mass = density * pixelArea * (float)asset->solidCount;
	massData.center = b2GetPixelShapeCentroid( shape );
	massData.rotationalInertia = density * asset->rotationalInertia;
	return massData;
}

float b2GetPixelShapeMaxExtent( const b2PixelShape* shape, b2Vec2 localCenter )
{
	b2AABB aabb = b2GetPixelShapeLocalAABB( shape );
	b2Vec2 points[4] = {
		{ aabb.lowerBound.x, aabb.lowerBound.y },
		{ aabb.upperBound.x, aabb.lowerBound.y },
		{ aabb.upperBound.x, aabb.upperBound.y },
		{ aabb.lowerBound.x, aabb.upperBound.y },
	};

	float maxDistanceSqr = 0.0f;
	for ( int i = 0; i < 4; ++i )
	{
		maxDistanceSqr = b2MaxFloat( maxDistanceSqr, b2LengthSquared( b2Sub( points[i], localCenter ) ) );
	}

	float radius = shape != NULL && shape->asset != NULL ? shape->diskRadius * shape->asset->pixelSize : 0.0f;
	return sqrtf( maxDistanceSqr ) + radius;
}
