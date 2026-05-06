// SPDX-FileCopyrightText: 2023 Erin Catto
// SPDX-License-Identifier: MIT

#include "pixel_shape.h"

#include "core.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined( _MSC_VER )
#include <intrin.h>
#endif

static bool b2PixelAsset_GetBit( const b2PixelAsset* asset, int index );

static int32_t b2PopCount64( uint64_t value )
{
#if defined( _MSC_VER ) && defined( _M_X64 )
	return (int32_t)__popcnt64( value );
#elif defined( __GNUC__ ) || defined( __clang__ )
	return (int32_t)__builtin_popcountll( value );
#else
	int32_t count = 0;
	while ( value != 0 )
	{
		value &= value - UINT64_C( 1 );
		++count;
	}
	return count;
#endif
}

static int32_t b2CountTrailingZeros64( uint64_t value )
{
	B2_ASSERT( value != 0 );
#if defined( _MSC_VER ) && defined( _M_X64 )
	unsigned long index = 0;
	_BitScanForward64( &index, value );
	return (int32_t)index;
#elif defined( __GNUC__ ) || defined( __clang__ )
	return (int32_t)__builtin_ctzll( value );
#else
	int32_t count = 0;
	while ( ( value & UINT64_C( 1 ) ) == 0 )
	{
		value >>= 1;
		++count;
	}
	return count;
#endif
}

static uint64_t b2BitMaskRange64( int32_t beginBit, int32_t endBit )
{
	B2_ASSERT( 0 <= beginBit && beginBit < endBit && endBit <= 64 );
	uint64_t lowerMask = beginBit == 0 ? UINT64_MAX : UINT64_MAX << beginBit;
	uint64_t upperMask = endBit == 64 ? UINT64_MAX : ( UINT64_C( 1 ) << endBit ) - UINT64_C( 1 );
	return lowerMask & upperMask;
}

static uint64_t b2PixelAsset_GetMaskedWord( const uint64_t* occupancyBits, int32_t occupancyWordCount, int32_t wordIndex,
											 uint64_t mask )
{
	if ( occupancyBits == NULL || wordIndex < 0 || wordIndex >= occupancyWordCount )
	{
		return UINT64_C( 0 );
	}

	return occupancyBits[wordIndex] & mask;
}

static bool b2IsPixelAssetUsable( const b2PixelAsset* asset )
{
	return asset != NULL && asset->width > 0 && asset->height > 0 && asset->pixelSize > 0.0f &&
		   b2IsValidFloat( asset->pixelSize );
}

static bool b2PixelAsset_GetCellCount( const b2PixelAsset* asset, int32_t* cellCount )
{
	if ( b2IsPixelAssetUsable( asset ) == false || asset->width > INT32_MAX / asset->height )
	{
		return false;
	}

	int32_t count = asset->width * asset->height;
	if ( count <= 0 || count >= UINT16_MAX || asset->width > INT16_MAX || asset->height > INT16_MAX )
	{
		return false;
	}

	if ( cellCount != NULL )
	{
		*cellCount = count;
	}

	return true;
}

static bool b2PixelFeatureTypeIsEdge( uint8_t type )
{
	return type == b2_pixelFeatureEdgeX || type == b2_pixelFeatureEdgeY;
}

static bool b2FeatureRefIsValid( const b2PixelAsset* asset, const b2PixelFeatureRef* feature, uint8_t expectedType )
{
	if ( feature->id == 0 || feature->type != expectedType )
	{
		return false;
	}

	if ( feature->x < 0 || feature->x >= asset->width || feature->y < 0 || feature->y >= asset->height )
	{
		return false;
	}

	uint16_t expectedId = b2PixelAsset_GetFeatureId( asset, feature->x, feature->y );
	if ( feature->id != expectedId )
	{
		return false;
	}

	uint8_t type = b2PixelAsset_GetFeatureType( asset, feature->x, feature->y );
	return type == expectedType && b2PixelAsset_IsOccupied( asset, feature->x, feature->y );
}

static bool b2EdgeFeatureRefIsValid( const b2PixelAsset* asset, const b2PixelFeatureRef* feature )
{
	if ( feature->id == 0 || b2PixelFeatureTypeIsEdge( feature->type ) == false )
	{
		return false;
	}

	if ( feature->x < 0 || feature->x >= asset->width || feature->y < 0 || feature->y >= asset->height )
	{
		return false;
	}

	uint16_t expectedId = b2PixelAsset_GetFeatureId( asset, feature->x, feature->y );
	if ( feature->id != expectedId )
	{
		return false;
	}

	uint8_t type = b2PixelAsset_GetFeatureType( asset, feature->x, feature->y );
	return type == feature->type && b2PixelFeatureTypeIsEdge( type ) && b2PixelAsset_IsOccupied( asset, feature->x, feature->y );
}

static float b2NormalizePixelDiskRadiusValue( float diskRadius )
{
	return diskRadius == 0.0f ? b2_defaultPixelDiskRadius : diskRadius;
}

static bool b2AlmostEqualFloat( float a, float b )
{
	float scale = b2MaxFloat( 1.0f, b2MaxFloat( b2AbsFloat( a ), b2AbsFloat( b ) ) );
	return b2AbsFloat( a - b ) <= 1.0e-4f * scale;
}

static bool b2AlmostEqualVec2( b2Vec2 a, b2Vec2 b )
{
	return b2AlmostEqualFloat( a.x, b.x ) && b2AlmostEqualFloat( a.y, b.y );
}

static bool b2AlmostEqualAABB( b2AABB a, b2AABB b )
{
	return b2AlmostEqualVec2( a.lowerBound, b.lowerBound ) && b2AlmostEqualVec2( a.upperBound, b.upperBound );
}

bool b2IsPixelShapeUsable( const b2PixelShape* shape )
{
	const b2PixelAsset* asset = shape == NULL ? NULL : shape->asset;
	int32_t cellCount = 0;
	if ( b2IsPixelAssetUsable( asset ) == false || b2PixelAsset_GetCellCount( asset, &cellCount ) == false ||
		 asset->occupancyBits == NULL || asset->featureTypes == NULL || asset->corners == NULL ||
		 asset->cornerCount <= 0 || ( asset->edgeCount > 0 && asset->edges == NULL ) ||
		 asset->solidCount <= 0 || asset->occupancyWordCount < ( cellCount + 63 ) / 64 ||
		 b2IsValidAABB( asset->occupiedAABB ) == false || b2IsValidVec2( asset->centroid ) == false ||
		 b2IsValidFloat( asset->rotationalInertia ) == false || asset->rotationalInertia < 0.0f ||
		 b2IsValidVec2( shape->localOrigin ) == false )
	{
		return false;
	}

	return b2GetPixelShapeDiskRadius( shape ) > 0.0f;
}

bool b2IsPixelAssetValid( const b2PixelAsset* asset )
{
	return b2ValidatePixelAsset( asset );
}

bool b2ValidatePixelAsset( const b2PixelAsset* asset )
{
	int32_t cellCount = 0;
	if ( b2IsPixelAssetUsable( asset ) == false || b2PixelAsset_GetCellCount( asset, &cellCount ) == false ||
		 asset->occupancyBits == NULL || asset->featureTypes == NULL || asset->corners == NULL ||
		 asset->cornerCount <= 0 || asset->solidCount <= 0 ||
		 b2IsValidAABB( asset->occupiedAABB ) == false || b2IsValidVec2( asset->centroid ) == false ||
		 b2IsValidFloat( asset->rotationalInertia ) == false || asset->rotationalInertia < 0.0f )
	{
		return false;
	}

	if ( asset->occupancyWordCount < ( cellCount + 63 ) / 64 )
	{
		return false;
	}

	if ( asset->edgeCount > 0 && asset->edges == NULL )
	{
		return false;
	}

	int32_t countedSolid = 0;
	int32_t countedCorners = 0;
	int32_t countedEdges = 0;
	int32_t minX = asset->width;
	int32_t minY = asset->height;
	int32_t maxX = -1;
	int32_t maxY = -1;
	b2Vec2 centroidSum = b2Vec2_zero;
	float halfWidth = 0.5f * (float)asset->width * asset->pixelSize;
	float halfHeight = 0.5f * (float)asset->height * asset->pixelSize;
	for ( int32_t index = 0; index < cellCount; ++index )
	{
		bool occupied = b2PixelAsset_GetBit( asset, index );
		uint8_t type = asset->featureTypes[index];
		if ( occupied )
		{
			int x = index % asset->width;
			int y = index / asset->width;
			countedSolid += 1;
			if ( type != b2_pixelFeatureInternal && b2PixelFeatureTypeIsEdge( type ) == false &&
				 type != b2_pixelFeatureCorner )
			{
				return false;
			}

			countedCorners += type == b2_pixelFeatureCorner ? 1 : 0;
			countedEdges += b2PixelFeatureTypeIsEdge( type ) ? 1 : 0;
			minX = b2MinInt( minX, x );
			minY = b2MinInt( minY, y );
			maxX = b2MaxInt( maxX, x );
			maxY = b2MaxInt( maxY, y );
			centroidSum.x += ( (float)x + 0.5f ) * asset->pixelSize - halfWidth;
			centroidSum.y += ( (float)y + 0.5f ) * asset->pixelSize - halfHeight;
		}
		else if ( type != b2_pixelFeatureEmpty )
		{
			return false;
		}

	}

	if ( countedSolid != asset->solidCount || countedCorners != asset->cornerCount || countedEdges != asset->edgeCount )
	{
		return false;
	}

	b2AABB computedAABB = { 0 };
	computedAABB.lowerBound = (b2Vec2){ (float)minX * asset->pixelSize - halfWidth, (float)minY * asset->pixelSize - halfHeight };
	computedAABB.upperBound =
		(b2Vec2){ ( (float)maxX + 1.0f ) * asset->pixelSize - halfWidth,
				  ( (float)maxY + 1.0f ) * asset->pixelSize - halfHeight };
	b2Vec2 computedCentroid = { centroidSum.x / (float)countedSolid, centroidSum.y / (float)countedSolid };
	float pixelArea = asset->pixelSize * asset->pixelSize;
	float cellInertia = pixelArea * asset->pixelSize * asset->pixelSize / 6.0f;
	float computedInertia = 0.0f;
	for ( int32_t index = 0; index < cellCount; ++index )
	{
		if ( b2PixelAsset_GetBit( asset, index ) == false )
		{
			continue;
		}

		int x = index % asset->width;
		int y = index / asset->width;
		b2Vec2 center = { ( (float)x + 0.5f ) * asset->pixelSize - halfWidth,
						  ( (float)y + 0.5f ) * asset->pixelSize - halfHeight };
		b2Vec2 d = b2Sub( center, computedCentroid );
		computedInertia += pixelArea * b2LengthSquared( d ) + cellInertia;
	}

	if ( b2AlmostEqualAABB( computedAABB, asset->occupiedAABB ) == false ||
		 b2AlmostEqualVec2( computedCentroid, asset->centroid ) == false ||
		 b2AlmostEqualFloat( computedInertia, asset->rotationalInertia ) == false )
	{
		return false;
	}

	uint16_t previousCornerId = 0;
	for ( int i = 0; i < asset->cornerCount; ++i )
	{
		const b2PixelFeatureRef* feature = asset->corners + i;
		if ( b2FeatureRefIsValid( asset, feature, b2_pixelFeatureCorner ) == false || feature->id <= previousCornerId )
		{
			return false;
		}
		previousCornerId = feature->id;
	}

	uint16_t previousEdgeId = 0;
	for ( int i = 0; i < asset->edgeCount; ++i )
	{
		const b2PixelFeatureRef* feature = asset->edges + i;
		if ( b2EdgeFeatureRefIsValid( asset, feature ) == false || feature->id <= previousEdgeId )
		{
			return false;
		}
		previousEdgeId = feature->id;
	}

	return true;
}

bool b2IsPixelShapeValid( const b2PixelShape* shape )
{
	if ( shape == NULL || b2ValidatePixelAsset( shape->asset ) == false || b2IsValidVec2( shape->localOrigin ) == false ||
		 b2IsValidFloat( shape->diskRadius ) == false )
	{
		return false;
	}

	float radius = b2NormalizePixelDiskRadiusValue( shape->diskRadius );
	return radius > 0.0f && b2IsValidFloat( radius );
}

float b2GetPixelShapeDiskRadius( const b2PixelShape* shape )
{
	if ( shape == NULL || b2IsValidFloat( shape->diskRadius ) == false || shape->diskRadius < 0.0f )
	{
		return -1.0f;
	}

	return b2NormalizePixelDiskRadiusValue( shape->diskRadius );
}

static bool b2SourceOccupancyBit( const uint64_t* occupancyBits, int32_t occupancyWordCount, int32_t index )
{
	int32_t wordIndex = index >> 6;
	if ( occupancyBits == NULL || wordIndex < 0 || wordIndex >= occupancyWordCount )
	{
		return false;
	}

	return ( occupancyBits[wordIndex] & ( UINT64_C( 1 ) << ( index & 63 ) ) ) != 0;
}

static bool b2SourceOccupancyCell( const uint64_t* occupancyBits, int32_t occupancyWordCount, int32_t width, int32_t height,
								   int32_t x, int32_t y )
{
	if ( x < 0 || x >= width || y < 0 || y >= height )
	{
		return false;
	}

	return b2SourceOccupancyBit( occupancyBits, occupancyWordCount, y * width + x );
}

static bool b2ClipPixelRayAxis( float origin, float translation, float lower, float upper, float* lowerFraction,
								float* upperFraction, b2Vec2* lowerNormal, b2Vec2 axisNormal )
{
	if ( b2AbsFloat( translation ) < 1.0e-12f )
	{
		return lower <= origin && origin <= upper;
	}

	float invTranslation = 1.0f / translation;
	float t1 = ( lower - origin ) * invTranslation;
	float t2 = ( upper - origin ) * invTranslation;
	b2Vec2 n1 = axisNormal;
	if ( t1 > t2 )
	{
		float t = t1;
		t1 = t2;
		t2 = t;
		n1 = b2Neg( axisNormal );
	}

	if ( t1 > *lowerFraction )
	{
		*lowerFraction = t1;
		*lowerNormal = n1;
	}
	if ( t2 < *upperFraction )
	{
		*upperFraction = t2;
	}

	return *lowerFraction <= *upperFraction;
}

b2CastOutput b2RayCastPixelShape( const b2PixelShape* shape, const b2RayCastInput* input )
{
	b2CastOutput output = { 0 };
	const b2PixelAsset* asset = shape == NULL ? NULL : shape->asset;
	if ( b2IsPixelShapeUsable( shape ) == false || input == NULL || b2IsValidRay( input ) == false )
	{
		return output;
	}

	if ( input->maxFraction <= 0.0f )
	{
		if ( b2PointInPixelShape( shape, input->origin ) )
		{
			output.point = input->origin;
			output.hit = true;
		}
		return output;
	}

	const float pixelSize = asset->pixelSize;
	const float halfWidth = 0.5f * (float)asset->width;
	const float halfHeight = 0.5f * (float)asset->height;
	b2Vec2 gridOrigin = {
		( input->origin.x - shape->localOrigin.x ) / pixelSize + halfWidth,
		( input->origin.y - shape->localOrigin.y ) / pixelSize + halfHeight,
	};
	b2Vec2 gridTranslation = {
		input->translation.x / pixelSize,
		input->translation.y / pixelSize,
	};

	float entryFraction = 0.0f;
	float exitFraction = input->maxFraction;
	b2Vec2 entryNormal = b2Vec2_zero;
	if ( b2ClipPixelRayAxis( gridOrigin.x, gridTranslation.x, 0.0f, (float)asset->width, &entryFraction, &exitFraction,
							  &entryNormal, (b2Vec2){ -1.0f, 0.0f } ) == false ||
		 b2ClipPixelRayAxis( gridOrigin.y, gridTranslation.y, 0.0f, (float)asset->height, &entryFraction, &exitFraction,
							  &entryNormal, (b2Vec2){ 0.0f, -1.0f } ) == false )
	{
		return output;
	}

	if ( exitFraction < 0.0f || entryFraction > input->maxFraction )
	{
		return output;
	}

	float currentFraction = b2MaxFloat( entryFraction, 0.0f );
	b2Vec2 p = {
		gridOrigin.x + currentFraction * gridTranslation.x,
		gridOrigin.y + currentFraction * gridTranslation.y,
	};
	int x = (int)floorf( p.x );
	int y = (int)floorf( p.y );
	if ( x == asset->width && gridTranslation.x <= 0.0f )
	{
		x = asset->width - 1;
	}
	if ( y == asset->height && gridTranslation.y <= 0.0f )
	{
		y = asset->height - 1;
	}
	x = b2ClampInt( x, 0, asset->width - 1 );
	y = b2ClampInt( y, 0, asset->height - 1 );

	if ( b2PixelAsset_IsOccupied( asset, x, y ) )
	{
		output.fraction = currentFraction;
		output.point = b2MulAdd( input->origin, currentFraction, input->translation );
		output.normal = currentFraction == 0.0f ? b2Vec2_zero : entryNormal;
		output.hit = true;
		return output;
	}

	const int stepX = gridTranslation.x > 0.0f ? 1 : -1;
	const int stepY = gridTranslation.y > 0.0f ? 1 : -1;
	float nextX = FLT_MAX;
	float nextY = FLT_MAX;
	float deltaX = FLT_MAX;
	float deltaY = FLT_MAX;
	if ( b2AbsFloat( gridTranslation.x ) >= 1.0e-12f )
	{
		float boundaryX = gridTranslation.x > 0.0f ? (float)( x + 1 ) : (float)x;
		nextX = ( boundaryX - gridOrigin.x ) / gridTranslation.x;
		deltaX = 1.0f / b2AbsFloat( gridTranslation.x );
		if ( nextX < currentFraction )
		{
			nextX = currentFraction;
		}
	}
	if ( b2AbsFloat( gridTranslation.y ) >= 1.0e-12f )
	{
		float boundaryY = gridTranslation.y > 0.0f ? (float)( y + 1 ) : (float)y;
		nextY = ( boundaryY - gridOrigin.y ) / gridTranslation.y;
		deltaY = 1.0f / b2AbsFloat( gridTranslation.y );
		if ( nextY < currentFraction )
		{
			nextY = currentFraction;
		}
	}

	const float maxFraction = b2MinFloat( exitFraction, input->maxFraction );
	while ( true )
	{
		b2Vec2 normal = b2Vec2_zero;
		if ( nextX < nextY )
		{
			currentFraction = nextX;
			nextX += deltaX;
			x += stepX;
			normal = (b2Vec2){ (float)-stepX, 0.0f };
		}
		else if ( nextY < nextX )
		{
			currentFraction = nextY;
			nextY += deltaY;
			y += stepY;
			normal = (b2Vec2){ 0.0f, (float)-stepY };
		}
		else
		{
			currentFraction = nextX;
			nextX += deltaX;
			nextY += deltaY;
			x += stepX;
			y += stepY;
			normal = b2Normalize( (b2Vec2){ (float)-stepX, (float)-stepY } );
		}

		if ( currentFraction > maxFraction )
		{
			break;
		}
		if ( x < 0 || x >= asset->width || y < 0 || y >= asset->height )
		{
			break;
		}
		if ( b2PixelAsset_IsOccupied( asset, x, y ) )
		{
			output.fraction = currentFraction;
			output.point = b2MulAdd( input->origin, currentFraction, input->translation );
			output.normal = normal;
			output.hit = true;
			return output;
		}
	}

	return output;
}

static uint8_t b2ClassifyPixelFeature( const uint64_t* occupancyBits, int32_t occupancyWordCount, int32_t width, int32_t height,
									   int32_t x, int32_t y, int32_t minX, int32_t minY, int32_t maxX, int32_t maxY,
									   int32_t supportCornerInterval )
{
	bool n = b2SourceOccupancyCell( occupancyBits, occupancyWordCount, width, height, x, y - 1 );
	bool e = b2SourceOccupancyCell( occupancyBits, occupancyWordCount, width, height, x + 1, y );
	bool s = b2SourceOccupancyCell( occupancyBits, occupancyWordCount, width, height, x, y + 1 );
	bool w = b2SourceOccupancyCell( occupancyBits, occupancyWordCount, width, height, x - 1, y );
	bool ne = b2SourceOccupancyCell( occupancyBits, occupancyWordCount, width, height, x + 1, y - 1 );
	bool se = b2SourceOccupancyCell( occupancyBits, occupancyWordCount, width, height, x + 1, y + 1 );
	bool sw = b2SourceOccupancyCell( occupancyBits, occupancyWordCount, width, height, x - 1, y + 1 );
	bool nw = b2SourceOccupancyCell( occupancyBits, occupancyWordCount, width, height, x - 1, y - 1 );

	if ( n && e && s && w && ne && se && sw && nw )
	{
		return b2_pixelFeatureInternal;
	}

	bool convexCorner = ( !n && !w ) || ( !n && !e ) || ( !s && !w ) || ( !s && !e );
	bool concaveCorner = ( n && e && !ne ) || ( e && s && !se ) || ( s && w && !sw ) || ( w && n && !nw );
	bool supportCorner = false;
	if ( supportCornerInterval > 0 )
	{
		if ( maxY == minY )
		{
			supportCorner = ( x - minX ) % supportCornerInterval == 0;
		}
		else if ( maxX == minX )
		{
			supportCorner = ( y - minY ) % supportCornerInterval == 0;
		}
		else
		{
			supportCorner = ( x - minX ) % supportCornerInterval == 0 || ( y - minY ) % supportCornerInterval == 0;
		}
	}

	if ( convexCorner || concaveCorner || supportCorner )
	{
		return b2_pixelFeatureCorner;
	}

	return ( n && s ) ? b2_pixelFeatureEdgeX : b2_pixelFeatureEdgeY;
}

static int b2ComparePixelFeatureRefById( const void* a, const void* b )
{
	const b2PixelFeatureRef* left = (const b2PixelFeatureRef*)a;
	const b2PixelFeatureRef* right = (const b2PixelFeatureRef*)b;
	if ( left->id < right->id )
	{
		return -1;
	}
	if ( left->id > right->id )
	{
		return 1;
	}
	return 0;
}

static void b2PixelAsset_SetBit( uint64_t* occupancyBits, int32_t occupancyWordCount, int32_t index, bool occupied )
{
	int32_t wordIndex = index >> 6;
	if ( occupancyBits == NULL || wordIndex < 0 || wordIndex >= occupancyWordCount )
	{
		return;
	}

	uint64_t mask = UINT64_C( 1 ) << ( index & 63 );
	if ( occupied )
	{
		occupancyBits[wordIndex] |= mask;
	}
	else
	{
		occupancyBits[wordIndex] &= ~mask;
	}
}

static b2Vec2 b2PixelAsset_ComputeCentroidFromMoments( int32_t width, int32_t height, float pixelSize, int32_t solidCount,
													   int64_t momentSumX, int64_t momentSumY )
{
	if ( solidCount <= 0 )
	{
		return b2Vec2_zero;
	}

	float halfWidth = 0.5f * (float)width * pixelSize;
	float halfHeight = 0.5f * (float)height * pixelSize;
	b2Vec2 centroid = {
		( (float)momentSumX + 0.5f * (float)solidCount ) * pixelSize / (float)solidCount - halfWidth,
		( (float)momentSumY + 0.5f * (float)solidCount ) * pixelSize / (float)solidCount - halfHeight,
	};
	return centroid;
}

static float b2PixelAsset_ComputeInertiaFromMoments( int32_t width, int32_t height, float pixelSize, int32_t solidCount,
													 int64_t momentSumX, int64_t momentSumY, int64_t momentSumX2,
													 int64_t momentSumY2, b2Vec2 centroid )
{
	if ( solidCount <= 0 )
	{
		return 0.0f;
	}

	float halfWidth = 0.5f * (float)width * pixelSize;
	float halfHeight = 0.5f * (float)height * pixelSize;
	float solid = (float)solidCount;
	float pixelSize2 = pixelSize * pixelSize;
	float sumCellX2 = pixelSize2 * ( (float)momentSumX2 + (float)momentSumX + 0.25f * solid ) -
					  2.0f * halfWidth * pixelSize * ( (float)momentSumX + 0.5f * solid ) + halfWidth * halfWidth * solid;
	float sumCellY2 = pixelSize2 * ( (float)momentSumY2 + (float)momentSumY + 0.25f * solid ) -
					  2.0f * halfHeight * pixelSize * ( (float)momentSumY + 0.5f * solid ) + halfHeight * halfHeight * solid;
	float pixelArea = pixelSize2;
	float cellInertia = pixelArea * pixelSize2 / 6.0f;
	return pixelArea * ( sumCellX2 + sumCellY2 - solid * ( centroid.x * centroid.x + centroid.y * centroid.y ) ) +
		   solid * cellInertia;
}

static bool b2PixelAsset_ComputeBoundsFromCounts( const int32_t* rowSolidCounts, int32_t rowSolidCount,
												  const int32_t* colSolidCounts, int32_t colSolidCount, int32_t* minX,
												  int32_t* minY, int32_t* maxX, int32_t* maxY )
{
	if ( rowSolidCounts == NULL || colSolidCounts == NULL || rowSolidCount <= 0 || colSolidCount <= 0 )
	{
		return false;
	}

	int32_t left = 0;
	while ( left < colSolidCount && colSolidCounts[left] <= 0 )
	{
		++left;
	}
	if ( left >= colSolidCount )
	{
		return false;
	}

	int32_t right = colSolidCount - 1;
	while ( right >= left && colSolidCounts[right] <= 0 )
	{
		--right;
	}

	int32_t top = 0;
	while ( top < rowSolidCount && rowSolidCounts[top] <= 0 )
	{
		++top;
	}

	int32_t bottom = rowSolidCount - 1;
	while ( bottom >= top && rowSolidCounts[bottom] <= 0 )
	{
		--bottom;
	}

	*minX = left;
	*maxX = right;
	*minY = top;
	*maxY = bottom;
	return true;
}

static void b2PixelAsset_MarkFeatureCell( uint8_t* markers, int32_t index, int32_t* featureCellsReclassified )
{
	if ( markers[index] == 0 )
	{
		markers[index] = 1;
		*featureCellsReclassified += 1;
	}
}

static bool b2PixelAsset_ReclassifyMarkedCell( const b2PixelAssetDirtyUpdateConfig* config, uint64_t* occupancyBits,
											   int32_t occupancyWordCount, b2PixelAssetBuildBuffers const* buffers,
											   int32_t minX, int32_t minY, int32_t maxX, int32_t maxY, int32_t index,
											   int32_t* cornerCount, int32_t* edgeCount, int32_t* featureRefsAdded )
{
	int32_t x = index % config->width;
	int32_t y = index / config->width;
	bool occupied = b2SourceOccupancyBit( occupancyBits, occupancyWordCount, index );
	if ( occupied == false )
	{
		buffers->featureTypes[index] = b2_pixelFeatureEmpty;
		return true;
	}

	uint8_t type = b2ClassifyPixelFeature( occupancyBits, occupancyWordCount, config->width, config->height, x, y, minX, minY,
										   maxX, maxY, config->supportCornerInterval );
	buffers->featureTypes[index] = type;
	if ( type != b2_pixelFeatureCorner && b2PixelFeatureTypeIsEdge( type ) == false )
	{
		return true;
	}

	b2PixelFeatureRef feature = { 0 };
	feature.x = (int16_t)x;
	feature.y = (int16_t)y;
	feature.id = (uint16_t)( index + 1 );
	feature.type = type;
	if ( type == b2_pixelFeatureCorner )
	{
		if ( *cornerCount >= buffers->cornerCapacity )
		{
			return false;
		}
		buffers->corners[( *cornerCount )++] = feature;
	}
	else
	{
		if ( buffers->edges == NULL || *edgeCount >= buffers->edgeCapacity )
		{
			return false;
		}
		buffers->edges[( *edgeCount )++] = feature;
	}
	*featureRefsAdded += 1;
	return true;
}

b2PixelAssetBuildConfig b2DefaultPixelAssetBuildConfig( void )
{
	b2PixelAssetBuildConfig config = { 0 };
	config.pixelSize = 1.0f;
	config.supportCornerInterval = 4;
	config.topologyVersion = 1;
	return config;
}

b2PixelAssetBuildResult b2BuildPixelAssetFromOccupancy( const b2PixelAssetBuildConfig* config,
														const uint64_t* sourceOccupancyBits,
														int32_t sourceOccupancyWordCount,
														const b2PixelAssetBuildBuffers* buffers )
{
	b2PixelAssetBuildResult result = { 0 };
	if ( config == NULL || sourceOccupancyBits == NULL || config->width <= 0 || config->height <= 0 ||
		 config->width > INT16_MAX || config->height > INT16_MAX || config->width > INT32_MAX / config->height ||
		 b2IsValidFloat( config->pixelSize ) == false || config->pixelSize <= 0.0f )
	{
		result.invalidInput = true;
		return result;
	}

	int32_t cellCount = config->width * config->height;
	if ( cellCount <= 0 || cellCount >= UINT16_MAX )
	{
		result.invalidInput = true;
		return result;
	}

	result.requiredOccupancyWords = ( cellCount + 63 ) / 64;
	result.requiredFeatureTypes = cellCount;
	result.requiredRowSolidCounts = config->height;
	result.requiredColSolidCounts = config->width;
	if ( sourceOccupancyWordCount < result.requiredOccupancyWords )
	{
		result.invalidInput = true;
		return result;
	}

	int32_t minX = config->width;
	int32_t minY = config->height;
	int32_t maxX = -1;
	int32_t maxY = -1;
	int32_t solidCount = 0;
	b2Vec2 centroidSum = b2Vec2_zero;
	int64_t momentSumX = 0;
	int64_t momentSumY = 0;
	int64_t momentSumX2 = 0;
	int64_t momentSumY2 = 0;
	float halfWidth = 0.5f * (float)config->width * config->pixelSize;
	float halfHeight = 0.5f * (float)config->height * config->pixelSize;
	if ( buffers != NULL && buffers->rowSolidCounts != NULL && buffers->rowSolidCountCapacity >= config->height )
	{
		memset( buffers->rowSolidCounts, 0, sizeof( int32_t ) * (size_t)config->height );
	}
	if ( buffers != NULL && buffers->colSolidCounts != NULL && buffers->colSolidCountCapacity >= config->width )
	{
		memset( buffers->colSolidCounts, 0, sizeof( int32_t ) * (size_t)config->width );
	}
	for ( int32_t y = 0; y < config->height; ++y )
	{
		int32_t rowStart = y * config->width;
		int32_t rowEnd = rowStart + config->width;
		int32_t firstWord = rowStart >> 6;
		int32_t lastWord = ( rowEnd - 1 ) >> 6;
		for ( int32_t wordIndex = firstWord; wordIndex <= lastWord; ++wordIndex )
		{
			int32_t beginBit = wordIndex == firstWord ? rowStart & 63 : 0;
			int32_t endBit = wordIndex == lastWord ? ( ( rowEnd - 1 ) & 63 ) + 1 : 64;
			uint64_t word = b2PixelAsset_GetMaskedWord( sourceOccupancyBits, sourceOccupancyWordCount, wordIndex,
														b2BitMaskRange64( beginBit, endBit ) );
			result.solidGatherWordsVisited += 1;
			if ( word == 0 )
			{
				continue;
			}

			int32_t rowSetBits = b2PopCount64( word );
			result.solidGatherSetBitsVisited += rowSetBits;
			if ( buffers != NULL && buffers->rowSolidCounts != NULL && buffers->rowSolidCountCapacity >= config->height )
			{
				buffers->rowSolidCounts[y] += rowSetBits;
			}
			solidCount += rowSetBits;
			while ( word != 0 )
			{
				int32_t bit = b2CountTrailingZeros64( word );
				int32_t index = ( wordIndex << 6 ) + bit;
				int32_t x = index - rowStart;
				minX = b2MinInt( minX, x );
				minY = b2MinInt( minY, y );
				maxX = b2MaxInt( maxX, x );
				maxY = b2MaxInt( maxY, y );
				centroidSum.x += ( (float)x + 0.5f ) * config->pixelSize - halfWidth;
				centroidSum.y += ( (float)y + 0.5f ) * config->pixelSize - halfHeight;
				momentSumX += x;
				momentSumY += y;
				momentSumX2 += (int64_t)x * (int64_t)x;
				momentSumY2 += (int64_t)y * (int64_t)y;
				if ( buffers != NULL && buffers->colSolidCounts != NULL && buffers->colSolidCountCapacity >= config->width )
				{
					buffers->colSolidCounts[x] += 1;
				}
				word &= word - UINT64_C( 1 );
			}
		}
	}

	if ( solidCount == 0 )
	{
		result.invalidInput = true;
		return result;
	}

	b2Vec2 centroid = { centroidSum.x / (float)solidCount, centroidSum.y / (float)solidCount };
	float pixelArea = config->pixelSize * config->pixelSize;
	float cellInertia = pixelArea * config->pixelSize * config->pixelSize / 6.0f;
	float rotationalInertia = 0.0f;
	for ( int32_t y = 0; y < config->height; ++y )
	{
		int32_t rowStart = y * config->width;
		int32_t rowEnd = rowStart + config->width;
		int32_t firstWord = rowStart >> 6;
		int32_t lastWord = ( rowEnd - 1 ) >> 6;
		for ( int32_t wordIndex = firstWord; wordIndex <= lastWord; ++wordIndex )
		{
			int32_t beginBit = wordIndex == firstWord ? rowStart & 63 : 0;
			int32_t endBit = wordIndex == lastWord ? ( ( rowEnd - 1 ) & 63 ) + 1 : 64;
			uint64_t word = b2PixelAsset_GetMaskedWord( sourceOccupancyBits, sourceOccupancyWordCount, wordIndex,
														b2BitMaskRange64( beginBit, endBit ) );
			while ( word != 0 )
			{
				int32_t bit = b2CountTrailingZeros64( word );
				int32_t index = ( wordIndex << 6 ) + bit;
				int32_t x = index - rowStart;
				b2Vec2 center = { ( (float)x + 0.5f ) * config->pixelSize - halfWidth,
								  ( (float)y + 0.5f ) * config->pixelSize - halfHeight };
				b2Vec2 d = b2Sub( center, centroid );
				rotationalInertia += pixelArea * b2LengthSquared( d ) + cellInertia;
				word &= word - UINT64_C( 1 );
			}
		}
	}

	for ( int32_t y = 0; y < config->height; ++y )
	{
		int32_t rowStart = y * config->width;
		int32_t rowEnd = rowStart + config->width;
		int32_t firstWord = rowStart >> 6;
		int32_t lastWord = ( rowEnd - 1 ) >> 6;
		for ( int32_t wordIndex = firstWord; wordIndex <= lastWord; ++wordIndex )
		{
			int32_t beginBit = wordIndex == firstWord ? rowStart & 63 : 0;
			int32_t endBit = wordIndex == lastWord ? ( ( rowEnd - 1 ) & 63 ) + 1 : 64;
			uint64_t word = b2PixelAsset_GetMaskedWord( sourceOccupancyBits, sourceOccupancyWordCount, wordIndex,
														b2BitMaskRange64( beginBit, endBit ) );
			while ( word != 0 )
			{
				int32_t bit = b2CountTrailingZeros64( word );
				int32_t index = ( wordIndex << 6 ) + bit;
				int32_t x = index - rowStart;
				uint8_t type = b2ClassifyPixelFeature( sourceOccupancyBits, sourceOccupancyWordCount, config->width,
														config->height, x, y, minX, minY, maxX, maxY,
														config->supportCornerInterval );
				result.requiredCorners += type == b2_pixelFeatureCorner ? 1 : 0;
				result.requiredEdges += b2PixelFeatureTypeIsEdge( type ) ? 1 : 0;
				word &= word - UINT64_C( 1 );
			}
		}
	}

	if ( result.requiredCorners <= 0 )
	{
		result.invalidInput = true;
		return result;
	}

	bool hasCapacity = buffers != NULL && buffers->occupancyBits != NULL && buffers->featureTypes != NULL &&
					   buffers->corners != NULL &&
					   buffers->occupancyWordCapacity >= result.requiredOccupancyWords &&
					   buffers->featureTypeCapacity >= result.requiredFeatureTypes &&
					   buffers->cornerCapacity >= result.requiredCorners &&
					   ( buffers->rowSolidCounts == NULL || buffers->rowSolidCountCapacity >= result.requiredRowSolidCounts ) &&
					   ( buffers->colSolidCounts == NULL || buffers->colSolidCountCapacity >= result.requiredColSolidCounts ) &&
					   ( result.requiredEdges == 0 ||
						 ( buffers->edges != NULL && buffers->edgeCapacity >= result.requiredEdges ) );
	if ( hasCapacity == false )
	{
		result.overflow = true;
		return result;
	}

	for ( int32_t i = 0; i < result.requiredOccupancyWords; ++i )
	{
		buffers->occupancyBits[i] = sourceOccupancyBits[i];
	}

	for ( int32_t i = 0; i < cellCount; ++i )
	{
		buffers->featureTypes[i] = b2_pixelFeatureEmpty;
	}

	int32_t cornerCount = 0;
	int32_t edgeCount = 0;
	for ( int32_t y = 0; y < config->height; ++y )
	{
		int32_t rowStart = y * config->width;
		int32_t rowEnd = rowStart + config->width;
		int32_t firstWord = rowStart >> 6;
		int32_t lastWord = ( rowEnd - 1 ) >> 6;
		for ( int32_t wordIndex = firstWord; wordIndex <= lastWord; ++wordIndex )
		{
			int32_t beginBit = wordIndex == firstWord ? rowStart & 63 : 0;
			int32_t endBit = wordIndex == lastWord ? ( ( rowEnd - 1 ) & 63 ) + 1 : 64;
			uint64_t word = b2PixelAsset_GetMaskedWord( sourceOccupancyBits, sourceOccupancyWordCount, wordIndex,
														b2BitMaskRange64( beginBit, endBit ) );
			while ( word != 0 )
			{
				int32_t bit = b2CountTrailingZeros64( word );
				int32_t index = ( wordIndex << 6 ) + bit;
				int32_t x = index - rowStart;

				uint8_t type = b2ClassifyPixelFeature( sourceOccupancyBits, sourceOccupancyWordCount, config->width,
														config->height, x, y, minX, minY, maxX, maxY,
														config->supportCornerInterval );
				buffers->featureTypes[index] = type;

				b2PixelFeatureRef feature = { 0 };
				feature.x = (int16_t)x;
				feature.y = (int16_t)y;
				feature.id = (uint16_t)( index + 1 );
				feature.type = type;

				if ( type == b2_pixelFeatureCorner )
				{
					buffers->corners[cornerCount++] = feature;
				}
				else if ( b2PixelFeatureTypeIsEdge( type ) )
				{
					buffers->edges[edgeCount++] = feature;
				}
				word &= word - UINT64_C( 1 );
			}
		}
	}

	result.asset.width = config->width;
	result.asset.height = config->height;
	result.asset.pixelSize = config->pixelSize;
	result.asset.occupancyBits = buffers->occupancyBits;
	result.asset.occupancyWordCount = result.requiredOccupancyWords;
	result.asset.featureTypes = buffers->featureTypes;
	result.asset.corners = buffers->corners;
	result.asset.cornerCount = cornerCount;
	result.asset.edges = result.requiredEdges > 0 ? buffers->edges : NULL;
	result.asset.edgeCount = edgeCount;
	result.asset.rowSolidCounts = buffers->rowSolidCounts;
	result.asset.rowSolidCount = buffers->rowSolidCounts != NULL ? config->height : 0;
	result.asset.colSolidCounts = buffers->colSolidCounts;
	result.asset.colSolidCount = buffers->colSolidCounts != NULL ? config->width : 0;
	result.asset.momentSumX = momentSumX;
	result.asset.momentSumY = momentSumY;
	result.asset.momentSumX2 = momentSumX2;
	result.asset.momentSumY2 = momentSumY2;
	result.asset.occupiedAABB.lowerBound =
		(b2Vec2){ (float)minX * config->pixelSize - halfWidth, (float)minY * config->pixelSize - halfHeight };
	result.asset.occupiedAABB.upperBound =
		(b2Vec2){ ( (float)maxX + 1.0f ) * config->pixelSize - halfWidth,
				  ( (float)maxY + 1.0f ) * config->pixelSize - halfHeight };
	result.asset.centroid = centroid;
	result.asset.rotationalInertia = rotationalInertia;
	result.asset.solidCount = solidCount;
	result.asset.topologyVersion = config->topologyVersion;
	result.success = b2ValidatePixelAsset( &result.asset );
	result.invalidInput = result.success == false;
	return result;
}

b2PixelAssetDirtyUpdateResult b2UpdatePixelAssetFromDirtyOccupancy( const b2PixelAssetDirtyUpdateConfig* config,
																	const b2PixelAsset* previousAsset,
																	const uint64_t* updatedOccupancyBits,
																	int32_t updatedOccupancyWordCount,
																	const b2PixelAssetBuildBuffers* buffers )
{
	b2PixelAssetDirtyUpdateResult result = { 0 };
	if ( config == NULL || previousAsset == NULL || updatedOccupancyBits == NULL || buffers == NULL || config->width <= 0 ||
		 config->height <= 0 || config->width > INT16_MAX || config->height > INT16_MAX ||
		 config->width > INT32_MAX / config->height || b2IsValidFloat( config->pixelSize ) == false ||
		 config->pixelSize <= 0.0f || previousAsset->width != config->width || previousAsset->height != config->height ||
		 previousAsset->occupancyBits == NULL || previousAsset->featureTypes == NULL ||
		 previousAsset->rowSolidCounts == NULL || previousAsset->colSolidCounts == NULL ||
		 previousAsset->rowSolidCount < config->height || previousAsset->colSolidCount < config->width )
	{
		result.invalidInput = true;
		return result;
	}

	int32_t cellCount = config->width * config->height;
	if ( cellCount <= 0 || cellCount >= UINT16_MAX )
	{
		result.invalidInput = true;
		return result;
	}

	result.requiredOccupancyWords = ( cellCount + 63 ) / 64;
	result.requiredFeatureTypes = cellCount;
	result.requiredRowSolidCounts = config->height;
	result.requiredColSolidCounts = config->width;
	if ( updatedOccupancyWordCount < result.requiredOccupancyWords )
	{
		result.invalidInput = true;
		return result;
	}
	if ( previousAsset->occupancyWordCount < result.requiredOccupancyWords )
	{
		result.invalidInput = true;
		return result;
	}

	int32_t dirtyMinX = b2ClampInt( config->dirtyX, 0, config->width );
	int32_t dirtyMinY = b2ClampInt( config->dirtyY, 0, config->height );
	int32_t dirtyMaxX = b2ClampInt( config->dirtyX + b2MaxInt( config->dirtyWidth, 0 ), 0, config->width );
	int32_t dirtyMaxY = b2ClampInt( config->dirtyY + b2MaxInt( config->dirtyHeight, 0 ), 0, config->height );
	if ( dirtyMaxX <= dirtyMinX || dirtyMaxY <= dirtyMinY )
	{
		result.invalidInput = true;
		return result;
	}

	bool hasCapacity = buffers->occupancyBits != NULL && buffers->featureTypes != NULL &&
					   buffers->corners != NULL && buffers->rowSolidCounts != NULL && buffers->colSolidCounts != NULL &&
					   buffers->scratchCells != NULL &&
					   buffers->occupancyWordCapacity >= result.requiredOccupancyWords &&
					   buffers->featureTypeCapacity >= result.requiredFeatureTypes &&
					   buffers->cornerCapacity > 0 && buffers->rowSolidCountCapacity >= result.requiredRowSolidCounts &&
					   buffers->colSolidCountCapacity >= result.requiredColSolidCounts &&
					   buffers->scratchCellCapacity >= cellCount;
	if ( hasCapacity == false )
	{
		result.overflow = true;
		return result;
	}

	memcpy( buffers->occupancyBits, previousAsset->occupancyBits, sizeof( uint64_t ) * (size_t)result.requiredOccupancyWords );
	memcpy( buffers->featureTypes, previousAsset->featureTypes, sizeof( uint8_t ) * (size_t)cellCount );
	memcpy( buffers->rowSolidCounts, previousAsset->rowSolidCounts, sizeof( int32_t ) * (size_t)config->height );
	memcpy( buffers->colSolidCounts, previousAsset->colSolidCounts, sizeof( int32_t ) * (size_t)config->width );
	memset( buffers->scratchCells, 0, sizeof( uint8_t ) * (size_t)cellCount );
	result.dirtyOccupancyWordsCopied = result.requiredOccupancyWords;
	result.dirtyFeatureCellsCopied = cellCount;
	result.dirtyRowCountsCopied = config->height;
	result.dirtyColCountsCopied = config->width;
	result.dirtyScratchCellsCleared = cellCount;

	int32_t solidCount = previousAsset->solidCount;
	int64_t momentSumX = previousAsset->momentSumX;
	int64_t momentSumY = previousAsset->momentSumY;
	int64_t momentSumX2 = previousAsset->momentSumX2;
	int64_t momentSumY2 = previousAsset->momentSumY2;

	int32_t dirtyWidth = dirtyMaxX - dirtyMinX;
	bool useWordDeltaPath = dirtyWidth >= 64;
	if ( useWordDeltaPath )
	{
		for ( int32_t y = dirtyMinY; y < dirtyMaxY; ++y )
		{
			int32_t rowStart = y * config->width;
			int32_t spanStart = rowStart + dirtyMinX;
			int32_t spanEnd = rowStart + dirtyMaxX;
			int32_t firstWord = spanStart >> 6;
			int32_t lastWord = ( spanEnd - 1 ) >> 6;
			result.dirtyWordDeltaRowsVisited += 1;
			result.dirtyCellsScanned += dirtyWidth;
			for ( int32_t wordIndex = firstWord; wordIndex <= lastWord; ++wordIndex )
			{
				int32_t beginBit = wordIndex == firstWord ? spanStart & 63 : 0;
				int32_t endBit = wordIndex == lastWord ? ( ( spanEnd - 1 ) & 63 ) + 1 : 64;
				uint64_t mask = b2BitMaskRange64( beginBit, endBit );
				uint64_t oldWord =
					b2PixelAsset_GetMaskedWord( previousAsset->occupancyBits, previousAsset->occupancyWordCount, wordIndex, mask );
				uint64_t newWord = b2PixelAsset_GetMaskedWord( updatedOccupancyBits, updatedOccupancyWordCount, wordIndex, mask );
				buffers->occupancyBits[wordIndex] = ( buffers->occupancyBits[wordIndex] & ~mask ) | newWord;
				result.dirtyWordDeltaWordsVisited += 1;

				int32_t rowDelta = b2PopCount64( newWord ) - b2PopCount64( oldWord );
				buffers->rowSolidCounts[y] += rowDelta;
				solidCount += rowDelta;

				uint64_t changedWord = oldWord ^ newWord;
				while ( changedWord != 0 )
				{
					int32_t bit = b2CountTrailingZeros64( changedWord );
					int32_t index = ( wordIndex << 6 ) + bit;
					int32_t x = index - rowStart;
					int32_t delta = ( newWord & ( UINT64_C( 1 ) << bit ) ) != 0 ? 1 : -1;
					buffers->colSolidCounts[x] += delta;
					momentSumX += (int64_t)delta * (int64_t)x;
					momentSumY += (int64_t)delta * (int64_t)y;
					momentSumX2 += (int64_t)delta * (int64_t)x * (int64_t)x;
					momentSumY2 += (int64_t)delta * (int64_t)y * (int64_t)y;
					result.dirtyWordDeltaBitsVisited += 1;
					changedWord &= changedWord - UINT64_C( 1 );
				}
			}
		}
	}
	else
	{
		for ( int32_t y = dirtyMinY; y < dirtyMaxY; ++y )
		{
			for ( int32_t x = dirtyMinX; x < dirtyMaxX; ++x )
			{
				int32_t index = y * config->width + x;
				bool wasOccupied = b2SourceOccupancyBit( previousAsset->occupancyBits, previousAsset->occupancyWordCount, index );
				bool isOccupied = b2SourceOccupancyBit( updatedOccupancyBits, updatedOccupancyWordCount, index );
				b2PixelAsset_SetBit( buffers->occupancyBits, result.requiredOccupancyWords, index, isOccupied );
				result.dirtyCellsScanned += 1;
				if ( wasOccupied == isOccupied )
				{
					continue;
				}

				int32_t delta = isOccupied ? 1 : -1;
				buffers->rowSolidCounts[y] += delta;
				buffers->colSolidCounts[x] += delta;
				solidCount += delta;
				momentSumX += (int64_t)delta * (int64_t)x;
				momentSumY += (int64_t)delta * (int64_t)y;
				momentSumX2 += (int64_t)delta * (int64_t)x * (int64_t)x;
				momentSumY2 += (int64_t)delta * (int64_t)y * (int64_t)y;
			}
		}
	}

	if ( solidCount <= 0 )
	{
		result.invalidInput = true;
		return result;
	}

	int32_t oldMinX = 0;
	int32_t oldMinY = 0;
	int32_t oldMaxX = 0;
	int32_t oldMaxY = 0;
	int32_t minX = 0;
	int32_t minY = 0;
	int32_t maxX = 0;
	int32_t maxY = 0;
	if ( b2PixelAsset_ComputeBoundsFromCounts( previousAsset->rowSolidCounts, previousAsset->rowSolidCount,
											   previousAsset->colSolidCounts, previousAsset->colSolidCount, &oldMinX, &oldMinY,
											   &oldMaxX, &oldMaxY ) == false ||
		 b2PixelAsset_ComputeBoundsFromCounts( buffers->rowSolidCounts, config->height, buffers->colSolidCounts, config->width,
											   &minX, &minY, &maxX, &maxY ) == false )
	{
		result.invalidInput = true;
		return result;
	}

	int32_t featureMinX = b2MaxInt( 0, dirtyMinX - 1 );
	int32_t featureMinY = b2MaxInt( 0, dirtyMinY - 1 );
	int32_t featureMaxX = b2MinInt( config->width, dirtyMaxX + 1 );
	int32_t featureMaxY = b2MinInt( config->height, dirtyMaxY + 1 );
	for ( int32_t y = featureMinY; y < featureMaxY; ++y )
	{
		for ( int32_t x = featureMinX; x < featureMaxX; ++x )
		{
			b2PixelAsset_MarkFeatureCell( buffers->scratchCells, y * config->width + x, &result.featureCellsReclassified );
		}
	}

	bool supportAnchorChanged = config->supportCornerInterval > 0 &&
								( oldMinX != minX || oldMinY != minY || oldMaxX != maxX || oldMaxY != maxY );
	if ( supportAnchorChanged )
	{
		for ( int32_t i = 0; i < previousAsset->cornerCount; ++i )
		{
			int32_t index = (int32_t)previousAsset->corners[i].id - 1;
			if ( 0 <= index && index < cellCount )
			{
				b2PixelAsset_MarkFeatureCell( buffers->scratchCells, index, &result.featureCellsReclassified );
			}
		}
		for ( int32_t i = 0; i < previousAsset->edgeCount; ++i )
		{
			int32_t index = (int32_t)previousAsset->edges[i].id - 1;
			if ( 0 <= index && index < cellCount )
			{
				b2PixelAsset_MarkFeatureCell( buffers->scratchCells, index, &result.featureCellsReclassified );
			}
		}
	}

	int32_t cornerCount = 0;
	int32_t edgeCount = 0;
	for ( int32_t i = 0; i < previousAsset->cornerCount; ++i )
	{
		int32_t index = (int32_t)previousAsset->corners[i].id - 1;
		if ( 0 <= index && index < cellCount && buffers->scratchCells[index] != 0 )
		{
			result.featureRefsRemoved += 1;
			continue;
		}
		if ( cornerCount >= buffers->cornerCapacity )
		{
			result.overflow = true;
			return result;
		}
		buffers->corners[cornerCount++] = previousAsset->corners[i];
	}

	for ( int32_t i = 0; i < previousAsset->edgeCount; ++i )
	{
		int32_t index = (int32_t)previousAsset->edges[i].id - 1;
		if ( 0 <= index && index < cellCount && buffers->scratchCells[index] != 0 )
		{
			result.featureRefsRemoved += 1;
			continue;
		}
		if ( buffers->edges == NULL || edgeCount >= buffers->edgeCapacity )
		{
			result.overflow = true;
			return result;
		}
		buffers->edges[edgeCount++] = previousAsset->edges[i];
	}

	for ( int32_t y = featureMinY; y < featureMaxY; ++y )
	{
		for ( int32_t x = featureMinX; x < featureMaxX; ++x )
		{
			int32_t index = y * config->width + x;
			if ( buffers->scratchCells[index] == 0 )
			{
				continue;
			}
			if ( b2PixelAsset_ReclassifyMarkedCell( config, buffers->occupancyBits, result.requiredOccupancyWords, buffers,
													minX, minY, maxX, maxY, index, &cornerCount, &edgeCount,
													&result.featureRefsAdded ) == false )
			{
				result.overflow = true;
				return result;
			}
			buffers->scratchCells[index] = 0;
		}
	}

	if ( supportAnchorChanged )
	{
		for ( int32_t i = 0; i < previousAsset->cornerCount; ++i )
		{
			int32_t index = (int32_t)previousAsset->corners[i].id - 1;
			if ( 0 <= index && index < cellCount && buffers->scratchCells[index] != 0 )
			{
				if ( b2PixelAsset_ReclassifyMarkedCell( config, buffers->occupancyBits, result.requiredOccupancyWords, buffers,
														minX, minY, maxX, maxY, index, &cornerCount, &edgeCount,
														&result.featureRefsAdded ) == false )
				{
					result.overflow = true;
					return result;
				}
				buffers->scratchCells[index] = 0;
			}
		}
		for ( int32_t i = 0; i < previousAsset->edgeCount; ++i )
		{
			int32_t index = (int32_t)previousAsset->edges[i].id - 1;
			if ( 0 <= index && index < cellCount && buffers->scratchCells[index] != 0 )
			{
				if ( b2PixelAsset_ReclassifyMarkedCell( config, buffers->occupancyBits, result.requiredOccupancyWords, buffers,
														minX, minY, maxX, maxY, index, &cornerCount, &edgeCount,
														&result.featureRefsAdded ) == false )
				{
					result.overflow = true;
					return result;
				}
				buffers->scratchCells[index] = 0;
			}
		}
	}

	if ( cornerCount <= 0 )
	{
		result.invalidInput = true;
		return result;
	}

	qsort( buffers->corners, (size_t)cornerCount, sizeof( b2PixelFeatureRef ), b2ComparePixelFeatureRefById );
	if ( edgeCount > 0 )
	{
		qsort( buffers->edges, (size_t)edgeCount, sizeof( b2PixelFeatureRef ), b2ComparePixelFeatureRefById );
	}

	b2Vec2 centroid = b2PixelAsset_ComputeCentroidFromMoments( config->width, config->height, config->pixelSize, solidCount,
															   momentSumX, momentSumY );
	float rotationalInertia =
		b2PixelAsset_ComputeInertiaFromMoments( config->width, config->height, config->pixelSize, solidCount, momentSumX,
												momentSumY, momentSumX2, momentSumY2, centroid );
	float halfWidth = 0.5f * (float)config->width * config->pixelSize;
	float halfHeight = 0.5f * (float)config->height * config->pixelSize;

	result.requiredCorners = cornerCount;
	result.requiredEdges = edgeCount;
	result.asset.width = config->width;
	result.asset.height = config->height;
	result.asset.pixelSize = config->pixelSize;
	result.asset.occupancyBits = buffers->occupancyBits;
	result.asset.occupancyWordCount = result.requiredOccupancyWords;
	result.asset.featureTypes = buffers->featureTypes;
	result.asset.corners = buffers->corners;
	result.asset.cornerCount = cornerCount;
	result.asset.edges = edgeCount > 0 ? buffers->edges : NULL;
	result.asset.edgeCount = edgeCount;
	result.asset.rowSolidCounts = buffers->rowSolidCounts;
	result.asset.rowSolidCount = config->height;
	result.asset.colSolidCounts = buffers->colSolidCounts;
	result.asset.colSolidCount = config->width;
	result.asset.momentSumX = momentSumX;
	result.asset.momentSumY = momentSumY;
	result.asset.momentSumX2 = momentSumX2;
	result.asset.momentSumY2 = momentSumY2;
	result.asset.occupiedAABB.lowerBound =
		(b2Vec2){ (float)minX * config->pixelSize - halfWidth, (float)minY * config->pixelSize - halfHeight };
	result.asset.occupiedAABB.upperBound =
		(b2Vec2){ ( (float)maxX + 1.0f ) * config->pixelSize - halfWidth,
				  ( (float)maxY + 1.0f ) * config->pixelSize - halfHeight };
	result.asset.centroid = centroid;
	result.asset.rotationalInertia = rotationalInertia;
	result.asset.solidCount = solidCount;
	result.asset.topologyVersion = config->topologyVersion;
	result.success = b2IsValidAABB( result.asset.occupiedAABB ) && b2IsValidVec2( result.asset.centroid ) &&
					 b2IsValidFloat( result.asset.rotationalInertia ) && result.asset.rotationalInertia >= 0.0f;
	result.invalidInput = result.success == false;
	return result;
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

	return b2_pixelFeatureEmpty;
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

	float radius = shape != NULL && shape->asset != NULL ? b2GetPixelShapeDiskRadius( shape ) * shape->asset->pixelSize : 0.0f;
	radius = b2MaxFloat( radius, 0.0f );
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

	float radius = shape != NULL && shape->asset != NULL ? b2GetPixelShapeDiskRadius( shape ) * shape->asset->pixelSize : 0.0f;
	radius = b2MaxFloat( radius, 0.0f );
	return sqrtf( maxDistanceSqr ) + radius;
}

static int b2PixelShape_CellFromAABBBound( float bound, float pixelSize, float halfCellCount, float cellOffset )
{
	float cell = bound / pixelSize + halfCellCount + cellOffset;
	return (int)floorf( cell + 1.0e-4f );
}

static void b2PixelShape_GetOccupiedCellBounds( const b2PixelAsset* asset, int* minX, int* maxX, int* minY, int* maxY )
{
	float halfWidth = 0.5f * (float)asset->width;
	float halfHeight = 0.5f * (float)asset->height;
	*minX = b2ClampInt( b2PixelShape_CellFromAABBBound( asset->occupiedAABB.lowerBound.x, asset->pixelSize, halfWidth, 0.0f ), 0,
						asset->width - 1 );
	*maxX = b2ClampInt( b2PixelShape_CellFromAABBBound( asset->occupiedAABB.upperBound.x, asset->pixelSize, halfWidth, -1.0f ), 0,
						asset->width - 1 );
	*minY = b2ClampInt( b2PixelShape_CellFromAABBBound( asset->occupiedAABB.lowerBound.y, asset->pixelSize, halfHeight, 0.0f ), 0,
						asset->height - 1 );
	*maxY = b2ClampInt( b2PixelShape_CellFromAABBBound( asset->occupiedAABB.upperBound.y, asset->pixelSize, halfHeight, -1.0f ), 0,
						asset->height - 1 );

	if ( *minX > *maxX || *minY > *maxY )
	{
		*minX = 0;
		*maxX = asset->width - 1;
		*minY = 0;
		*maxY = asset->height - 1;
	}
}

static float b2PixelShape_DistanceToCellIntervalSqr( float coordinate, int lower, int upper )
{
	if ( lower > upper )
	{
		return FLT_MAX;
	}

	if ( coordinate < (float)lower )
	{
		float d = (float)lower - coordinate;
		return d * d;
	}

	if ( coordinate > (float)upper )
	{
		float d = coordinate - (float)upper;
		return d * d;
	}

	return 0.0f;
}

static float b2PixelShape_DistanceToCellRectSqr( float gx, float gy, int minX, int maxX, int minY, int maxY )
{
	float dxSqr = b2PixelShape_DistanceToCellIntervalSqr( gx, minX, maxX );
	float dySqr = b2PixelShape_DistanceToCellIntervalSqr( gy, minY, maxY );
	if ( dxSqr == FLT_MAX || dySqr == FLT_MAX )
	{
		return FLT_MAX;
	}

	return dxSqr + dySqr;
}

static float b2PixelShape_OutsideRingLowerBoundSqr( float gx, float gy, int baseX, int baseY, int ring, int minX, int maxX,
													int minY, int maxY, float pixelSize )
{
	int left = baseX - ring;
	int right = baseX + ring;
	int top = baseY - ring;
	int bottom = baseY + ring;

	float lowerBound = FLT_MAX;
	if ( minX < left )
	{
		lowerBound = b2MinFloat( lowerBound, b2PixelShape_DistanceToCellRectSqr( gx, gy, minX, left - 1, minY, maxY ) );
	}
	if ( right < maxX )
	{
		lowerBound = b2MinFloat( lowerBound, b2PixelShape_DistanceToCellRectSqr( gx, gy, right + 1, maxX, minY, maxY ) );
	}
	if ( minY < top )
	{
		lowerBound = b2MinFloat( lowerBound, b2PixelShape_DistanceToCellRectSqr( gx, gy, minX, maxX, minY, top - 1 ) );
	}
	if ( bottom < maxY )
	{
		lowerBound = b2MinFloat( lowerBound, b2PixelShape_DistanceToCellRectSqr( gx, gy, minX, maxX, bottom + 1, maxY ) );
	}

	return lowerBound == FLT_MAX ? FLT_MAX : lowerBound * pixelSize * pixelSize;
}

static void b2PixelShape_TryClosestCell( const b2PixelAsset* asset, int x, int y, float gx, float gy, float* bestDistanceSqr,
										 int* bestIndex )
{
	int index = y * asset->width + x;
	if ( b2PixelAsset_GetBit( asset, index ) == false )
	{
		return;
	}

	float dx = ( (float)x - gx ) * asset->pixelSize;
	float dy = ( (float)y - gy ) * asset->pixelSize;
	float distanceSqr = dx * dx + dy * dy;
	if ( distanceSqr < *bestDistanceSqr || ( distanceSqr == *bestDistanceSqr && index < *bestIndex ) )
	{
		*bestDistanceSqr = distanceSqr;
		*bestIndex = index;
	}
}

b2Vec2 b2GetPixelShapeClosestPoint( const b2PixelShape* shape, b2Transform transform, b2Vec2 target )
{
	const b2PixelAsset* asset = shape == NULL ? NULL : shape->asset;
	if ( b2IsPixelAssetUsable( asset ) == false || asset->occupancyBits == NULL || asset->solidCount <= 0 )
	{
		return target;
	}

	if ( b2PointInPixelShape( shape, b2InvTransformPoint( transform, target ) ) )
	{
		return target;
	}

	b2Vec2 localTarget = b2InvTransformPoint( transform, target );
	float halfWidth = 0.5f * (float)asset->width;
	float halfHeight = 0.5f * (float)asset->height;
	float gx = ( localTarget.x - shape->localOrigin.x ) / asset->pixelSize + halfWidth - 0.5f;
	float gy = ( localTarget.y - shape->localOrigin.y ) / asset->pixelSize + halfHeight - 0.5f;

	int minX = 0;
	int maxX = asset->width - 1;
	int minY = 0;
	int maxY = asset->height - 1;
	b2PixelShape_GetOccupiedCellBounds( asset, &minX, &maxX, &minY, &maxY );

	int baseX = b2ClampInt( (int)floorf( gx + 0.5f ), minX, maxX );
	int baseY = b2ClampInt( (int)floorf( gy + 0.5f ), minY, maxY );
	int maxRing = b2MaxInt( b2MaxInt( baseX - minX, maxX - baseX ), b2MaxInt( baseY - minY, maxY - baseY ) );

	float bestDistanceSqr = FLT_MAX;
	int bestIndex = INT32_MAX;
	for ( int ring = 0; ring <= maxRing; ++ring )
	{
		int y0 = b2MaxInt( minY, baseY - ring );
		int y1 = b2MinInt( maxY, baseY + ring );
		int x0 = b2MaxInt( minX, baseX - ring );
		int x1 = b2MinInt( maxX, baseX + ring );

		for ( int y = y0; y <= y1; ++y )
		{
			if ( y == baseY - ring || y == baseY + ring )
			{
				for ( int x = x0; x <= x1; ++x )
				{
					b2PixelShape_TryClosestCell( asset, x, y, gx, gy, &bestDistanceSqr, &bestIndex );
				}
			}
			else
			{
				if ( baseX - ring >= minX )
				{
					b2PixelShape_TryClosestCell( asset, baseX - ring, y, gx, gy, &bestDistanceSqr, &bestIndex );
				}
				if ( ring > 0 && baseX + ring <= maxX )
				{
					b2PixelShape_TryClosestCell( asset, baseX + ring, y, gx, gy, &bestDistanceSqr, &bestIndex );
				}
			}
		}

		if ( bestIndex != INT32_MAX )
		{
			float outsideLowerBoundSqr =
				b2PixelShape_OutsideRingLowerBoundSqr( gx, gy, baseX, baseY, ring, minX, maxX, minY, maxY, asset->pixelSize );
			if ( outsideLowerBoundSqr > bestDistanceSqr )
			{
				break;
			}
		}
	}

	if ( bestIndex == INT32_MAX )
	{
		return target;
	}

	return b2TransformPoint( transform, b2PixelShape_GetPixelCenter( shape, bestIndex % asset->width, bestIndex / asset->width ) );
}
