// SPDX-FileCopyrightText: 2023 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "box2d/collision.h"
#include "box2d/types.h"

typedef struct b2PixelLocalInfo
{
	int x;
	int y;
	int index;
	b2BlastMaterialId materialId;
	float density;
} b2PixelLocalInfo;

b2AABB b2ComputePixelShapeAABB( const b2PixelShape* shape, b2Transform transform );
b2MassData b2ComputePixelShapeMass( const b2PixelShape* shape, float density );
b2Vec2 b2GetPixelShapeCentroid( const b2PixelShape* shape );
float b2GetPixelShapeMaxExtent( const b2PixelShape* shape, b2Vec2 localCenter );
b2CastOutput b2RayCastPixelShape( const b2PixelShape* shape, const b2RayCastInput* input );
bool b2PointInPixelShape( const b2PixelShape* shape, b2Vec2 localPoint );
bool b2IsPixelShapeUsable( const b2PixelShape* shape );
bool b2IsPixelAssetValid( const b2PixelAsset* asset );
bool b2IsPixelShapeValid( const b2PixelShape* shape );
float b2GetPixelShapeDiskRadius( const b2PixelShape* shape );
b2Vec2 b2GetPixelShapeClosestPoint( const b2PixelShape* shape, b2Transform transform, b2Vec2 target );

bool b2PixelShape_GetLocalInfo( const b2PixelShape* shape, b2Vec2 localPoint, b2PixelLocalInfo* info );
bool b2PixelAsset_IsOccupied( const b2PixelAsset* asset, int x, int y );
uint8_t b2PixelAsset_GetFeatureType( const b2PixelAsset* asset, int x, int y );
b2BlastMaterialId b2PixelAsset_GetMaterialId( const b2PixelAsset* asset, int x, int y );
float b2PixelAsset_GetMaterialDensity( const b2PixelAsset* asset, int x, int y, float fallbackDensity );
b2Vec2 b2PixelShape_GetPixelCenter( const b2PixelShape* shape, int x, int y );
uint16_t b2PixelAsset_GetFeatureId( const b2PixelAsset* asset, int x, int y );
