// SPDX-FileCopyrightText: 2026 Alchemy
// SPDX-License-Identifier: MIT

#pragma once

#include "base.h"
#include "collision.h"
#include "types.h"

#include <stdbool.h>
#include <stdint.h>

/// @defgroup blast2d Blast2D
/// Alchemy-specific pixel physics and fracture extension API.
/// @{

typedef struct b2BlastFractureActorId
{
	uint32_t index;
	uint16_t revision;
	uint16_t world;
} b2BlastFractureActorId;

typedef enum b2BlastActorMobility
{
	b2_blastActorMobilityAnchored = 0,
	b2_blastActorMobilityDynamic = 1,
	b2_blastActorMobilityKinematic = 2,
	b2_blastActorMobilitySleepingDynamic = 3,
} b2BlastActorMobility;

typedef enum b2BlastExternalConstraintKind
{
	b2_blastConstraintWorldAnchor = 0,
	b2_blastConstraintParentAnchor = 1,
	b2_blastConstraintActorAnchor = 2,
	b2_blastConstraintBox2DJointBridge = 3,
	b2_blastConstraintContactPatch = 4,
	b2_blastConstraintMotorDrive = 5,
	b2_blastConstraintRopeOrCable = 6,
	b2_blastConstraintUserPinned = 7,
} b2BlastExternalConstraintKind;

typedef enum b2BlastConstraintSource
{
	b2_blastConstraintSourceUnknown = 0,
	b2_blastConstraintSourceContact = 1,
	b2_blastConstraintSourceJoint = 2,
	b2_blastConstraintSourceMotor = 3,
	b2_blastConstraintSourceAnchor = 4,
	b2_blastConstraintSourceRope = 5,
	b2_blastConstraintSourceUserPin = 6,
} b2BlastConstraintSource;

typedef enum b2BlastFractureCommandKind
{
	b2_blastFractureCommandNone = 0,
	b2_blastFractureCommandRefine = 1,
	b2_blastFractureCommandBreak = 2,
	b2_blastFractureCommandDetachActor = 3,
	b2_blastFractureCommandIntegrityDamage = 4,
	b2_blastFractureCommandResidualImpulse = 5,
} b2BlastFractureCommandKind;

typedef struct b2BlastExternalConstraintEndpoint
{
	b2BlastFractureActorId actorId;
	uint32_t clusterIndex;
	uint32_t leafIndex;
	b2Vec2 localPoint;
} b2BlastExternalConstraintEndpoint;

typedef struct b2BlastExternalConstraint
{
	uint32_t id;
	b2BlastExternalConstraintKind kind;
	b2BlastExternalConstraintEndpoint a;
	b2BlastExternalConstraintEndpoint b;
	uint32_t dofMask;
	float stiffness;
	float damping;
	float maxNormalImpulse;
	float maxTangentImpulse;
	float maxTension;
	float maxShear;
	float maxTorque;
	float accumulatedDamage;
	float sustainedLoadTime;
	bool hasEndpointB;
	bool contributesToFracture;
	bool broken;
} b2BlastExternalConstraint;

typedef struct b2BlastImpactEvent
{
	b2BlastFractureActorId actorId;
	uint32_t source;
	uint32_t clusterIndex;
	uint32_t leafIndex;
	b2Vec2 localPoint;
	b2Vec2 direction;
	float impulse;
	float radius;
	float damageHint;
} b2BlastImpactEvent;

typedef struct b2BlastConstraintRow
{
	b2BlastFractureActorId actorIdA;
	b2BlastFractureActorId actorIdB;
	uint32_t constraintId;
	b2BlastConstraintSource source;
	b2BlastExternalConstraintEndpoint endpointA;
	b2BlastExternalConstraintEndpoint endpointB;
	b2Vec2 point;
	b2Vec2 normal;
	b2Vec2 tangent;
	float normalImpulse;
	float tangentImpulse;
	float requiredNormalImpulse;
	float appliedNormalImpulse;
	float unresolvedNormalImpulse;
	float force;
	float torque;
	bool hasEndpointB;
	bool yielded;
} b2BlastConstraintRow;

typedef struct b2BlastFractureCommand
{
	b2BlastFractureCommandKind kind;
	b2BlastFractureActorId actorId;
	b2BlastFractureActorId childActorId;
	uint32_t clusterIndex;
	uint32_t leafIndex;
	uint32_t bondIndex;
	b2BlastActorMobility targetMobility;
	b2Vec2 point;
	b2Vec2 impulse;
	float value;
} b2BlastFractureCommand;

typedef struct b2BlastMaterialHotData
{
	b2BlastMaterialId materialId;
	uint16_t authoringShaderIndex;
	uint16_t damageShaderIndex;
	uint16_t flags;
	float density;
	float friction;
	float restitution;
	float hardness;
	float contactCapacity;
	float toughness;
	float brittleness;
	float compressStrength;
	float shearStrength;
	float tensileStrength;
	float onset;
	float breakRatio;
} b2BlastMaterialHotData;

typedef struct b2BlastLeaf
{
	uint32_t firstCell;
	uint32_t cellCount;
	uint32_t firstBond;
	uint32_t bondCount;
	b2Vec2 centroid;
	float mass;
	b2BlastMaterialId dominantMaterialId;
	uint16_t flags;
	uint32_t supportConstraintMask;
} b2BlastLeaf;

typedef struct b2BlastActiveBond
{
	uint32_t leafA;
	uint32_t leafB;
	uint32_t clusterA;
	uint32_t clusterB;
	float area;
	float capacity;
	float toughness;
	float damage;
	float impactDemand;
	float loadDemand;
	float propagationWeight;
	uint16_t flags;
	uint16_t materialMix;
} b2BlastActiveBond;

typedef struct b2BlastFractureDebugSnapshot
{
	uint32_t actorCount;
	uint32_t anchoredActorCount;
	uint32_t dynamicActorCount;
	uint32_t leafCount;
	uint32_t activeBondCount;
	uint32_t brokenBondCount;
	uint32_t commandCount;
	uint32_t constraintRowCount;
	uint32_t actorTransitionCount;
	uint32_t reauthoredFallbackCount;
	uint32_t legacyHostFracturePathCount;
	uint32_t stepAllocationFallbackCount;
	uint32_t scratchCapacityHighWater;
	uint32_t hotMaterialIdBytesPerCell;
	uint32_t materialHotDataSize;
	uint32_t leafSize;
	uint32_t activeBondSize;
	float maxImpactDemand;
	float maxLoadDemand;
	float maxDamage;
} b2BlastFractureDebugSnapshot;

/// Return immutable Blast2D fracture layout diagnostics for tests/debug.
B2_API b2BlastFractureDebugSnapshot b2BlastFracture_GetDebugSnapshot( void );

/// Return Blast2D fracture world diagnostics. This is the runtime-owned actor pool state.
B2_API b2BlastFractureDebugSnapshot b2World_GetBlastFractureDebugSnapshot( b2WorldId worldId );

/// Submit a host-authored Blast2D impact into the world's fracture actor pool.
B2_API bool b2World_SubmitBlastImpactAtPoint(
	b2WorldId worldId, b2BodyId bodyId, b2Vec2 worldPoint, b2Vec2 direction, float impulse, float radius, float damageHint );

/// Submit a bound external-constraint load into the world's fracture actor pool.
B2_API bool b2World_SubmitBlastLoadAtPoint(
	b2WorldId worldId, b2BodyId bodyId, b2Vec2 worldPoint, b2Vec2 force, b2BlastExternalConstraintKind kind, uint32_t constraintId );

/// @}
