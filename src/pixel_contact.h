// SPDX-FileCopyrightText: 2023 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "shape.h"

typedef struct b2PixelShapeContinuousStats
{
	int queryCount;
	int sampleCount;
	int refineCount;
	bool initialOverlap;
	bool sampleLimitReached;
} b2PixelShapeContinuousStats;

typedef struct b2PixelShapeContinuousResult
{
	bool hit;
	float fraction;
	b2Vec2 point;
	b2Vec2 normal;
} b2PixelShapeContinuousResult;

b2Manifold b2PixelShapeManifold( const b2Shape* shapeA, b2Transform xfA, const b2Shape* shapeB, b2Transform xfB );

b2PixelShapeContinuousResult b2ComputePixelShapeContinuousHit( const b2PixelShape* pixelA, b2Sweep sweepA,
															  const b2PixelShape* pixelB, b2Sweep sweepB,
															  float maxFraction, b2PixelShapeContinuousStats* stats );
