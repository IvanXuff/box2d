// SPDX-FileCopyrightText: 2026 Alchemy
// SPDX-License-Identifier: MIT

#pragma once

#include "base.h"
#include "collision.h"
#include "id.h"
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

typedef enum b2BlastActorTransitionAction
{
	b2_blastActorTransitionNone = 0,
	b2_blastActorTransitionCreateDynamicBody = 1,
} b2BlastActorTransitionAction;

typedef struct b2BlastActorTransition
{
	uint32_t transitionId;
	b2BlastActorTransitionAction action;
	b2BlastFractureActorId sourceActorId;
	b2BlastFractureActorId childActorId;
	b2BodyId sourceBodyId;
	b2ShapeId sourceShapeId;
	b2BodyId createdBodyId;
	b2ShapeId createdShapeId;
	b2BlastActorMobility sourceMobility;
	b2BlastActorMobility targetMobility;
	int32_t sourceMinX;
	int32_t sourceMinY;
	int32_t sourceMaxX;
	int32_t sourceMaxY;
	int32_t sourceWidth;
	int32_t sourceHeight;
	int32_t cellOffset;
	int32_t cellCount;
	uint32_t sourceTopologyRevision;
	uint32_t childTopologyRevision;
	uint64_t materialHash;
	b2Vec2 worldPoint;
	b2Vec2 linearVelocity;
	float angularVelocity;
	bool committed;
	bool sourcePruned;
} b2BlastActorTransition;

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
	uint16_t minX;
	uint16_t minY;
	uint16_t maxX;
	uint16_t maxY;
	uint32_t supportConstraintMask;
} b2BlastLeaf;

typedef enum b2BlastLeafDebugFlags
{
	b2_blastLeafDebugWorldAnchor = 0x0001,
	b2_blastLeafDebugDetached = 0x0002,
} b2BlastLeafDebugFlags;

typedef enum b2BlastBondDebugFlags
{
	b2_blastBondDebugBroken = 0x0001,
	b2_blastBondDebugBreakCandidate = 0x0002,
} b2BlastBondDebugFlags;

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

typedef struct b2BlastOverlayReadQuery
{
	b2AABB worldAABB;
	uint32_t flags;
} b2BlastOverlayReadQuery;

typedef enum b2BlastOverlayReadFlags
{
	b2_blastOverlayReadLeafGraph = 0x0001,
} b2BlastOverlayReadFlags;

typedef struct b2BlastOverlayCluster
{
	uint32_t clusterIndex;
	uint32_t firstLeaf;
	uint32_t leafCount;
	b2Vec2 centroid;
	float mass;
	uint16_t level;
	uint16_t flags;
	uint16_t minX;
	uint16_t minY;
	uint16_t maxX;
	uint16_t maxY;
} b2BlastOverlayCluster;

typedef enum b2BlastClusterDebugFlags
{
	b2_blastClusterDebugWorldAnchor = 0x0001,
} b2BlastClusterDebugFlags;

typedef struct b2BlastOverlayActorView
{
	b2BlastFractureActorId actorId;
	b2BodyId bodyId;
	b2ShapeId shapeId;
	b2BlastActorMobility mobility;
	b2Transform transform;
	b2Vec2 localOrigin;
	int32_t width;
	int32_t height;
	float pixelSize;
	uint32_t revision;
	uint32_t topologyVersion;
	uint64_t materialHash;
	const b2BlastLeaf* leaves;
	int32_t leafCount;
	const b2BlastActiveBond* bonds;
	int32_t bondCount;
	const uint32_t* cellToLeaf;
	int32_t cellToLeafCount;
	const b2BlastOverlayCluster* activeClusters;
	int32_t activeClusterCount;
	const b2BlastActiveBond* activeBonds;
	int32_t activeBondCount;
	const uint32_t* cellToActiveCluster;
	int32_t cellToActiveClusterCount;
	uint16_t activeLevel;
} b2BlastOverlayActorView;

typedef struct b2BlastOverlayReadView
{
	const b2BlastOverlayActorView* actors;
	int32_t actorCount;
	uint32_t readRevision;
	bool skipped;
} b2BlastOverlayReadView;

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
	uint32_t jointConstraintRowCount;
	uint32_t actorTransitionCount;
	uint32_t appliedForceLoadRowCount;
	uint32_t appliedImpulseImpactRowCount;
	uint32_t torqueInputIgnoredCount;
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
	uint32_t hotMaterialIdBytesPerCell;
	uint32_t materialHotDataSize;
	uint32_t leafSize;
	uint32_t activeBondSize;
	uint32_t clusterCount;
	uint32_t rootClusterCount;
	uint32_t worldAnchorCount;
	uint32_t authoringHashLow;
	uint32_t authoringHashHigh;
	uint32_t impactDemandHash;
	uint32_t loadDemandHash;
	float maxImpactDemand;
	float maxLoadDemand;
	float maxDamage;
} b2BlastFractureDebugSnapshot;

/// Return immutable Blast2D fracture layout diagnostics for tests/debug.
B2_API b2BlastFractureDebugSnapshot b2BlastFracture_GetDebugSnapshot( void );

/// Return Blast2D fracture world diagnostics. This is the runtime-owned actor pool state.
B2_API b2BlastFractureDebugSnapshot b2World_GetBlastFractureDebugSnapshot( b2WorldId worldId );

/// Begin a zero-copy read of Blast2D actor graph data for overlay rendering.
/// The returned view is valid until EndBlastOverlayRead or the next world mutation.
B2_API bool b2World_BeginBlastOverlayRead(
	b2WorldId worldId, const b2BlastOverlayReadQuery* query, b2BlastOverlayReadView* outView );

/// End a Blast2D overlay read view.
B2_API void b2World_EndBlastOverlayRead( b2WorldId worldId, const b2BlastOverlayReadView* view );

/// Compatibility shim for legacy callers. Prefer b2Body_ApplyLinearImpulse* for Blast2D impact input.
B2_API bool b2World_SubmitBlastImpactAtPoint(
	b2WorldId worldId, b2BodyId bodyId, b2Vec2 worldPoint, b2Vec2 direction, float impulse, float radius, float damageHint );

/// Compatibility shim for legacy callers. Prefer b2Body_ApplyForce* for Blast2D load input.
B2_API bool b2World_SubmitBlastLoadAtPoint(
	b2WorldId worldId, b2BodyId bodyId, b2Vec2 worldPoint, b2Vec2 force, b2BlastExternalConstraintKind kind, uint32_t constraintId );

/// Commit pending Blast2D actor transitions after host-side shape metadata has been attached.
B2_API void b2World_CommitBlastFractureTransitions( b2WorldId worldId );

/// Return the number of committed Blast2D actor transition rows available until the next world step.
B2_API int32_t b2World_GetBlastFractureTransitionCount( b2WorldId worldId );

/// Copy committed Blast2D actor transition rows. Rows are owned by Blast2D; this copies POD data only.
B2_API int32_t b2World_CopyBlastFractureTransitions(
	b2WorldId worldId, b2BlastActorTransition* transitions, int32_t transitionCapacity );

/// Copy source actor cell indices for a committed transition row.
B2_API int32_t b2World_CopyBlastFractureTransitionCells(
	b2WorldId worldId, int32_t transitionIndex, int32_t* cells, int32_t cellCapacity );

/// Acknowledge committed Blast2D actor transition rows after host-side adoption has recorded them.
B2_API void b2World_AcknowledgeBlastFractureTransitions( b2WorldId worldId );

/// @}
