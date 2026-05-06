// SPDX-FileCopyrightText: 2026 Alchemy
// SPDX-License-Identifier: MIT

#pragma once

#include "box2d/blast2d.h"

typedef struct b2Body b2Body;
typedef struct b2ContactSim b2ContactSim;
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

	b2BlastActorTransition* transitions;
	int transitionCount;
	int transitionCapacity;
	int32_t* transitionCells;
	int transitionCellCount;
	int transitionCellCapacity;
	uint32_t nextTransitionId;

	b2BlastFractureActorId pendingShapeBindActorId;
	bool hasPendingShapeBindActor;

	b2BlastOverlayActorView* overlayActorViews;
	int overlayActorViewCount;
	int overlayActorViewCapacity;
	uint32_t overlayReadRevision;
	uint32_t overlayDirectReadCount;

	uint32_t nextConstraintId;
	uint32_t constraintRowCount;
	uint32_t actorTransitionCount;
	uint32_t reauthoredFallbackCount;
	uint32_t legacyHostFracturePathCount;
	uint32_t stepAllocationFallbackCount;
	uint32_t ignoredOffTargetEventCount;
	uint32_t unboundLoadPortDropCount;
	uint32_t refinedThisStep;
	uint32_t brokenThisStep;
	uint32_t demandRecomputeCount;
	uint32_t substepSolveCount;
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
bool b2BlastFractureWorld_TryConsumePendingShapeActorBinding(
	b2World* world, b2Body* body, b2Shape* shape, b2BlastActorMobility mobility );

void b2BlastFractureWorld_CollectAndStep( b2World* world, float timeStep );
void b2BlastFractureWorld_BeginStep( b2World* world );
void b2BlastFractureWorld_BeginSubstep( b2World* world );
void b2BlastFractureWorld_EndSubstep( b2World* world );
void b2BlastFractureWorld_EndStep( b2World* world );
void b2BlastFractureWorld_ConsumeTouchingContactsIfNoRows( b2World* world, float timeStep );
void b2BlastFractureWorld_ConsumeContactConstraintRow( b2World* world, const b2ContactSim* contactSim, int pointIndex,
														b2Vec2 normal, b2Vec2 anchorA, b2Vec2 anchorB, float normalImpulse,
														float tangentImpulse, float unresolvedNormalImpulse, bool yielded, float timeStep );
b2BlastFractureDebugSnapshot b2BlastFractureWorld_GetDebugSnapshot( const b2BlastFractureWorld* fractureWorld );
