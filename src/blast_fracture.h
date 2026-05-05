// SPDX-FileCopyrightText: 2026 Alchemy
// SPDX-License-Identifier: MIT

#pragma once

#include "box2d/blast2d.h"

typedef struct b2Body b2Body;
typedef struct b2Shape b2Shape;
typedef struct b2World b2World;

typedef struct b2BlastFractureWorld
{
	struct b2BlastActor* actors;
	int actorCount;
	int actorCapacity;
	struct b2BlastActor* freeActors;

	b2BlastFractureCommand* commands;
	int commandCount;
	int commandCapacity;

	uint32_t nextConstraintId;
	uint32_t constraintRowCount;
	uint32_t actorTransitionCount;
	uint32_t reauthoredFallbackCount;
	uint32_t legacyHostFracturePathCount;
	uint32_t stepAllocationFallbackCount;
	uint32_t scratchCapacityHighWater;
	float maxImpactDemand;
	float maxLoadDemand;
	float maxDamage;
} b2BlastFractureWorld;

void b2BlastFractureWorld_Create( b2BlastFractureWorld* fractureWorld );
void b2BlastFractureWorld_Destroy( b2BlastFractureWorld* fractureWorld );
void b2BlastFractureWorld_ClearStep( b2BlastFractureWorld* fractureWorld );

b2BlastFractureActorId b2BlastFractureWorld_UpsertPixelShapeActor(
	b2World* world, b2Body* body, b2Shape* shape, b2BlastActorMobility mobility );
void b2BlastFractureWorld_UnbindShape( b2World* world, b2Shape* shape );

void b2BlastFractureWorld_CollectAndStep( b2World* world, float timeStep );
b2BlastFractureDebugSnapshot b2BlastFractureWorld_GetDebugSnapshot( const b2BlastFractureWorld* fractureWorld );
