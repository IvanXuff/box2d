// SPDX-FileCopyrightText: 2026 Alchemy
// SPDX-License-Identifier: MIT

#include "blast_fracture.h"

#include "body.h"
#include "contact.h"
#include "core.h"
#include "joint.h"
#include "physics_world.h"
#include "pixel_shape.h"
#include "shape.h"
#include "solver_set.h"

#include "box2d/box2d.h"

#include <float.h>
#include <math.h>
#include <string.h>

_Static_assert( sizeof( b2BlastMaterialId ) == 2, "Blast2D PixelShape material ids must stay 16-bit" );
_Static_assert( sizeof( b2BlastMaterialHotData ) <= 64, "Blast2D material hot data must fit in one cache line" );
_Static_assert( sizeof( b2BlastLeaf ) <= 64, "Blast2D leaf hot data must stay compact" );
_Static_assert( sizeof( b2BlastActiveBond ) <= 64, "Blast2D active bond hot data must stay compact" );

enum
{
	b2_blastActorFlagInUse = 0x0001,
	b2_blastActorFlagOwnsWorldAnchor = 0x0002,
	b2_blastActorFlagDirtyGraph = 0x0004,
	b2_blastActorFlagOwnsPixelAsset = 0x0008,
	b2_blastLeafFlagWorldAnchor = 0x0001,
	b2_blastLeafFlagDetached = 0x0002,
	b2_blastBondFlagBroken = 0x0001,
	b2_blastBondFlagBreakCandidate = 0x0002,
	b2_blastInitialActorCapacity = 8,
	b2_blastInitialCommandCapacity = 64,
	b2_blastInitialScratchCapacity = 64,
	b2_blastClusterFlagWorldAnchor = 0x0001,
};

static void b2BlastClassifyAndCommandSplits( b2BlastFractureWorld* fractureWorld, struct b2BlastActor* actor );

enum
{
	b2_blastShaderChaos = 1303382207,
	b2_blastShaderNoisyConcrete = 1656274117,
	b2_blastShaderWoodGrain = 824297244,
	b2_blastShaderGlassRadial = 1466905759,
	b2_blastShaderMasonryMortar = 2120062292,
};

static const float b2_blastDefaultImpactRange = 42.0f;
static const float b2_blastDefaultImpactDecay = 26.0f;
static const float b2_blastDefaultContactShield = 0.55f;
static const float b2_blastReferenceImpactImpulse = 120.0f;
static const float b2_blastDefaultContactDamageRadius = 3.0f;
static const float b2_blastConstraintImpulseEpsilon = 0.0001f;
static const int b2_blastDefaultMaxBreaksPerStep = 8;

typedef struct b2BlastCluster
{
	uint32_t id;
	uint32_t parent;
	uint32_t firstChild;
	uint32_t childCount;
	uint32_t firstLeaf;
	uint32_t leafCount;
	uint16_t level;
	uint16_t flags;
	uint16_t minX;
	uint16_t minY;
	uint16_t maxX;
	uint16_t maxY;
	b2Vec2 centroid;
	float mass;
} b2BlastCluster;

typedef struct b2BlastSeed
{
	float x;
	float y;
	float weight;
	uint32_t id;
} b2BlastSeed;

typedef struct b2BlastActor
{
	b2BlastFractureActorId id;
	int bodyId;
	int shapeId;
	b2BlastActorMobility mobility;
	uint32_t flags;
	uint32_t revision;
	uint32_t topologyVersion;
	uint64_t materialHash;

	b2BlastLeaf* leaves;
	int leafCount;
	int leafCapacity;

	b2BlastActiveBond* bonds;
	int bondCount;
	int bondCapacity;

	b2BlastCluster* clusters;
	int clusterCount;
	int clusterCapacity;

	uint32_t* clusterChildren;
	int clusterChildCount;
	int clusterChildCapacity;

	uint32_t* clusterLeaves;
	int clusterLeafRefCount;
	int clusterLeafRefCapacity;

	uint32_t* activeClusters;
	int activeClusterCount;
	int activeClusterCapacity;

	uint32_t* cellToLeaf;
	int cellToLeafCount;
	int cellToLeafCapacity;

	b2BlastSeed* seedScratch;
	int seedScratchCapacity;
	int32_t* assignScratch;
	int assignScratchCapacity;
	int32_t* cellScratch;
	int cellScratchCapacity;
	uint8_t* cellVisitScratch;
	int cellVisitScratchCapacity;
	uint32_t* leafRemapScratch;
	int leafRemapScratchCapacity;
	int* componentScratch;
	int componentScratchCapacity;
	int* queueScratch;
	int queueScratchCapacity;
	uint8_t* visitScratch;
	int visitScratchCapacity;
	float* graphDistanceScratch;
	int graphDistanceScratchCapacity;
	int* graphParentBondScratch;
	int graphParentBondScratchCapacity;

	b2PixelAsset ownedAsset;
	uint64_t* ownedOccupancyBits;
	int ownedOccupancyWordCapacity;
	b2BlastMaterialId* ownedMaterialIds;
	int ownedMaterialIdCapacity;
	uint8_t* ownedFeatureTypes;
	int ownedFeatureTypeCapacity;
	uint8_t* ownedNormalIndices;
	int ownedNormalIndexCapacity;
	b2PixelFeatureRef* ownedCorners;
	int ownedCornerCapacity;
	b2PixelFeatureRef* ownedEdges;
	int ownedEdgeCapacity;
	int32_t* ownedRowSolidCounts;
	int ownedRowSolidCountCapacity;
	int32_t* ownedColSolidCounts;
	int ownedColSolidCountCapacity;
	uint8_t* ownedPixelScratch;
	int ownedPixelScratchCapacity;

	uint32_t rootCluster;
	uint16_t initialActiveLevel;
	uint32_t worldAnchorCount;
	uint64_t authoringHash;
} b2BlastActor;

static b2BlastFractureActorId b2BlastNullActorId( uint16_t worldId )
{
	return (b2BlastFractureActorId){ UINT32_MAX, 0, worldId };
}

static bool b2BlastActorIdValid( b2BlastFractureActorId id )
{
	return id.index != UINT32_MAX && id.revision != 0;
}

static b2BlastActorMobility b2BlastMobilityFromBodyTypeLocal( b2BodyType type )
{
	switch ( type )
	{
		case b2_staticBody:
			return b2_blastActorMobilityAnchored;
		case b2_kinematicBody:
			return b2_blastActorMobilityKinematic;
		case b2_dynamicBody:
		default:
			return b2_blastActorMobilityDynamic;
	}
}

static void* b2BlastResize( void* oldMem, int oldCount, int newCount, int elementSize )
{
	if ( newCount <= 0 )
	{
		if ( oldMem != NULL && oldCount > 0 )
		{
			b2Free( oldMem, oldCount * elementSize );
		}
		return NULL;
	}

	return b2GrowAllocZeroInit( oldMem, oldCount * elementSize, newCount * elementSize );
}

static void b2BlastActor_FreeArrays( b2BlastActor* actor )
{
	if ( actor == NULL )
	{
		return;
	}
	b2Free( actor->leaves, actor->leafCapacity * (int)sizeof( b2BlastLeaf ) );
	b2Free( actor->bonds, actor->bondCapacity * (int)sizeof( b2BlastActiveBond ) );
	b2Free( actor->clusters, actor->clusterCapacity * (int)sizeof( b2BlastCluster ) );
	b2Free( actor->clusterChildren, actor->clusterChildCapacity * (int)sizeof( uint32_t ) );
	b2Free( actor->clusterLeaves, actor->clusterLeafRefCapacity * (int)sizeof( uint32_t ) );
	b2Free( actor->activeClusters, actor->activeClusterCapacity * (int)sizeof( uint32_t ) );
	b2Free( actor->cellToLeaf, actor->cellToLeafCapacity * (int)sizeof( uint32_t ) );
	b2Free( actor->seedScratch, actor->seedScratchCapacity * (int)sizeof( b2BlastSeed ) );
	b2Free( actor->assignScratch, actor->assignScratchCapacity * (int)sizeof( int32_t ) );
	b2Free( actor->cellScratch, actor->cellScratchCapacity * (int)sizeof( int32_t ) );
	b2Free( actor->cellVisitScratch, actor->cellVisitScratchCapacity * (int)sizeof( uint8_t ) );
	b2Free( actor->leafRemapScratch, actor->leafRemapScratchCapacity * (int)sizeof( uint32_t ) );
	b2Free( actor->componentScratch, actor->componentScratchCapacity * (int)sizeof( int ) );
	b2Free( actor->queueScratch, actor->queueScratchCapacity * (int)sizeof( int ) );
	b2Free( actor->visitScratch, actor->visitScratchCapacity * (int)sizeof( uint8_t ) );
	b2Free( actor->graphDistanceScratch, actor->graphDistanceScratchCapacity * (int)sizeof( float ) );
	b2Free( actor->graphParentBondScratch, actor->graphParentBondScratchCapacity * (int)sizeof( int ) );
	b2Free( actor->ownedOccupancyBits, actor->ownedOccupancyWordCapacity * (int)sizeof( uint64_t ) );
	b2Free( actor->ownedMaterialIds, actor->ownedMaterialIdCapacity * (int)sizeof( b2BlastMaterialId ) );
	b2Free( actor->ownedFeatureTypes, actor->ownedFeatureTypeCapacity * (int)sizeof( uint8_t ) );
	b2Free( actor->ownedNormalIndices, actor->ownedNormalIndexCapacity * (int)sizeof( uint8_t ) );
	b2Free( actor->ownedCorners, actor->ownedCornerCapacity * (int)sizeof( b2PixelFeatureRef ) );
	b2Free( actor->ownedEdges, actor->ownedEdgeCapacity * (int)sizeof( b2PixelFeatureRef ) );
	b2Free( actor->ownedRowSolidCounts, actor->ownedRowSolidCountCapacity * (int)sizeof( int32_t ) );
	b2Free( actor->ownedColSolidCounts, actor->ownedColSolidCountCapacity * (int)sizeof( int32_t ) );
	b2Free( actor->ownedPixelScratch, actor->ownedPixelScratchCapacity * (int)sizeof( uint8_t ) );
	actor->leaves = NULL;
	actor->bonds = NULL;
	actor->clusters = NULL;
	actor->clusterChildren = NULL;
	actor->clusterLeaves = NULL;
	actor->activeClusters = NULL;
	actor->cellToLeaf = NULL;
	actor->seedScratch = NULL;
	actor->assignScratch = NULL;
	actor->cellScratch = NULL;
	actor->cellVisitScratch = NULL;
	actor->leafRemapScratch = NULL;
	actor->componentScratch = NULL;
	actor->queueScratch = NULL;
	actor->visitScratch = NULL;
	actor->graphDistanceScratch = NULL;
	actor->graphParentBondScratch = NULL;
	actor->ownedOccupancyBits = NULL;
	actor->ownedMaterialIds = NULL;
	actor->ownedFeatureTypes = NULL;
	actor->ownedNormalIndices = NULL;
	actor->ownedCorners = NULL;
	actor->ownedEdges = NULL;
	actor->ownedRowSolidCounts = NULL;
	actor->ownedColSolidCounts = NULL;
	actor->ownedPixelScratch = NULL;
	actor->ownedAsset = (b2PixelAsset){ 0 };
	actor->leafCapacity = 0;
	actor->bondCapacity = 0;
	actor->clusterCapacity = 0;
	actor->clusterChildCapacity = 0;
	actor->clusterLeafRefCapacity = 0;
	actor->activeClusterCapacity = 0;
	actor->cellToLeafCapacity = 0;
	actor->seedScratchCapacity = 0;
	actor->assignScratchCapacity = 0;
	actor->cellScratchCapacity = 0;
	actor->cellVisitScratchCapacity = 0;
	actor->leafRemapScratchCapacity = 0;
	actor->componentScratchCapacity = 0;
	actor->queueScratchCapacity = 0;
	actor->visitScratchCapacity = 0;
	actor->graphDistanceScratchCapacity = 0;
	actor->graphParentBondScratchCapacity = 0;
	actor->ownedOccupancyWordCapacity = 0;
	actor->ownedMaterialIdCapacity = 0;
	actor->ownedFeatureTypeCapacity = 0;
	actor->ownedNormalIndexCapacity = 0;
	actor->ownedCornerCapacity = 0;
	actor->ownedEdgeCapacity = 0;
	actor->ownedRowSolidCountCapacity = 0;
	actor->ownedColSolidCountCapacity = 0;
	actor->ownedPixelScratchCapacity = 0;
	actor->leafCount = 0;
	actor->bondCount = 0;
	actor->clusterCount = 0;
	actor->clusterChildCount = 0;
	actor->clusterLeafRefCount = 0;
	actor->activeClusterCount = 0;
	actor->cellToLeafCount = 0;
}

static b2BlastActor* b2BlastGetActor( b2BlastFractureWorld* fractureWorld, b2BlastFractureActorId id )
{
	if ( fractureWorld == NULL || b2BlastActorIdValid( id ) == false || id.index >= (uint32_t)fractureWorld->actorCount )
	{
		return NULL;
	}

	b2BlastActor* actor = fractureWorld->actors + id.index;
	if ( ( actor->flags & b2_blastActorFlagInUse ) == 0 || actor->id.revision != id.revision )
	{
		return NULL;
	}

	return actor;
}

static void b2BlastRepairOwnedPixelAssetShapePointers( b2World* world )
{
	if ( world == NULL )
	{
		return;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	for ( int i = 0; i < fractureWorld->actorCount; ++i )
	{
		b2BlastActor* actor = fractureWorld->actors + i;
		if ( ( actor->flags & b2_blastActorFlagOwnsPixelAsset ) == 0 || actor->shapeId == B2_NULL_INDEX )
		{
			continue;
		}
		b2Shape* shape = b2ShapeArray_Get( &world->shapes, actor->shapeId );
		if ( shape == NULL || shape->type != b2_pixelShape )
		{
			continue;
		}
		shape->pixel.asset = &actor->ownedAsset;
	}
}

static bool b2BlastActorIdEqual( b2BlastFractureActorId a, b2BlastFractureActorId b )
{
	return a.index == b.index && a.revision == b.revision && a.world == b.world;
}

static void b2BlastEnsureTransitionCapacity( b2BlastFractureWorld* fractureWorld, int transitionCapacity, int cellCapacity )
{
	if ( transitionCapacity > fractureWorld->transitionCapacity )
	{
		int oldCapacity = fractureWorld->transitionCapacity;
		int newCapacity = oldCapacity < 16 ? 16 : oldCapacity;
		while ( newCapacity < transitionCapacity )
		{
			newCapacity += newCapacity >> 1;
		}
		fractureWorld->transitions =
			b2BlastResize( fractureWorld->transitions, oldCapacity, newCapacity, (int)sizeof( b2BlastActorTransition ) );
		fractureWorld->transitionCapacity = newCapacity;
	}
	if ( cellCapacity > fractureWorld->transitionCellCapacity )
	{
		int oldCapacity = fractureWorld->transitionCellCapacity;
		int newCapacity = oldCapacity < 128 ? 128 : oldCapacity;
		while ( newCapacity < cellCapacity )
		{
			newCapacity += newCapacity >> 1;
		}
		fractureWorld->transitionCells =
			b2BlastResize( fractureWorld->transitionCells, oldCapacity, newCapacity, (int)sizeof( int32_t ) );
		fractureWorld->transitionCellCapacity = newCapacity;
	}
}

static void b2BlastEnsureOverlayActorViewCapacity( b2BlastFractureWorld* fractureWorld, int capacity )
{
	if ( fractureWorld == NULL || capacity <= fractureWorld->overlayActorViewCapacity )
	{
		return;
	}
	int oldCapacity = fractureWorld->overlayActorViewCapacity;
	int newCapacity = oldCapacity < 16 ? 16 : oldCapacity;
	while ( newCapacity < capacity )
	{
		newCapacity += newCapacity >> 1;
	}
	fractureWorld->overlayActorViews =
		b2BlastResize( fractureWorld->overlayActorViews, oldCapacity, newCapacity, (int)sizeof( b2BlastOverlayActorView ) );
	fractureWorld->overlayActorViewCapacity = newCapacity;
}

static void b2BlastEnsureOverlayClusterCapacity( b2BlastFractureWorld* fractureWorld, int capacity )
{
	if ( fractureWorld == NULL || capacity <= fractureWorld->overlayClusterCapacity )
	{
		return;
	}
	int oldCapacity = fractureWorld->overlayClusterCapacity;
	int newCapacity = oldCapacity < 32 ? 32 : oldCapacity;
	while ( newCapacity < capacity )
	{
		newCapacity += newCapacity >> 1;
	}
	fractureWorld->overlayClusters =
		b2BlastResize( fractureWorld->overlayClusters, oldCapacity, newCapacity, (int)sizeof( b2BlastOverlayCluster ) );
	fractureWorld->overlayClusterCapacity = newCapacity;
}

static void b2BlastEnsureOverlayBondCapacity( b2BlastFractureWorld* fractureWorld, int capacity )
{
	if ( fractureWorld == NULL || capacity <= fractureWorld->overlayBondCapacity )
	{
		return;
	}
	int oldCapacity = fractureWorld->overlayBondCapacity;
	int newCapacity = oldCapacity < 64 ? 64 : oldCapacity;
	while ( newCapacity < capacity )
	{
		newCapacity += newCapacity >> 1;
	}
	fractureWorld->overlayBonds =
		b2BlastResize( fractureWorld->overlayBonds, oldCapacity, newCapacity, (int)sizeof( b2BlastActiveBond ) );
	fractureWorld->overlayBondCapacity = newCapacity;
}

static void b2BlastEnsureBodyInputCapacity( b2BlastFractureWorld* fractureWorld, int capacity )
{
	if ( fractureWorld == NULL || capacity <= fractureWorld->bodyInputCapacity )
	{
		return;
	}
	int newCapacity = b2MaxInt( capacity, fractureWorld->bodyInputCapacity * 2 );
	fractureWorld->bodyInputs = b2BlastResize(
		fractureWorld->bodyInputs, fractureWorld->bodyInputCapacity, newCapacity, (int)sizeof( b2BlastBodyInputRecord ) );
	fractureWorld->bodyInputCapacity = newCapacity;
}

static void b2BlastEnsureOverlayCellToActiveClusterCapacity( b2BlastFractureWorld* fractureWorld, int capacity )
{
	if ( fractureWorld == NULL || capacity <= fractureWorld->overlayCellToActiveClusterCapacity )
	{
		return;
	}
	int oldCapacity = fractureWorld->overlayCellToActiveClusterCapacity;
	int newCapacity = oldCapacity < 256 ? 256 : oldCapacity;
	while ( newCapacity < capacity )
	{
		newCapacity += newCapacity >> 1;
	}
	fractureWorld->overlayCellToActiveCluster = b2BlastResize(
		fractureWorld->overlayCellToActiveCluster, oldCapacity, newCapacity, (int)sizeof( uint32_t ) );
	fractureWorld->overlayCellToActiveClusterCapacity = newCapacity;
}

static void b2BlastEnsureOverlayLeafRemapScratchCapacity( b2BlastFractureWorld* fractureWorld, int capacity )
{
	if ( fractureWorld == NULL || capacity <= fractureWorld->overlayLeafRemapScratchCapacity )
	{
		return;
	}
	int oldCapacity = fractureWorld->overlayLeafRemapScratchCapacity;
	int newCapacity = oldCapacity < 256 ? 256 : oldCapacity;
	while ( newCapacity < capacity )
	{
		newCapacity += newCapacity >> 1;
	}
	fractureWorld->overlayLeafRemapScratch =
		b2BlastResize( fractureWorld->overlayLeafRemapScratch, oldCapacity, newCapacity, (int)sizeof( uint32_t ) );
	fractureWorld->overlayLeafRemapScratchCapacity = newCapacity;
}

static bool b2BlastAABBOverlaps( b2AABB a, b2AABB b )
{
	return a.lowerBound.x <= b.upperBound.x && a.upperBound.x >= b.lowerBound.x && a.lowerBound.y <= b.upperBound.y &&
		   a.upperBound.y >= b.lowerBound.y;
}

static b2BlastActor* b2BlastAllocActor( b2World* world, uint16_t worldId )
{
	if ( world == NULL )
	{
		return NULL;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	if ( fractureWorld->actorCount == fractureWorld->actorCapacity )
	{
		int oldCapacity = fractureWorld->actorCapacity;
		int newCapacity = oldCapacity < b2_blastInitialActorCapacity ? b2_blastInitialActorCapacity : oldCapacity + ( oldCapacity >> 1 );
		fractureWorld->actors = b2BlastResize( fractureWorld->actors, oldCapacity, newCapacity, (int)sizeof( b2BlastActor ) );
		fractureWorld->actorCapacity = newCapacity;
		b2BlastRepairOwnedPixelAssetShapePointers( world );
	}

	int index = fractureWorld->actorCount++;
	b2BlastActor* actor = fractureWorld->actors + index;
	uint16_t revision = actor->id.revision;
	*actor = (b2BlastActor){ 0 };
	actor->id.index = (uint32_t)index;
	actor->id.revision = revision == 0 ? 1 : (uint16_t)( revision + 1 );
	actor->id.world = worldId;
	actor->flags = b2_blastActorFlagInUse;
	actor->bodyId = B2_NULL_INDEX;
	actor->shapeId = B2_NULL_INDEX;
	return actor;
}

static void b2BlastRollbackAllocatedActor( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor )
{
	if ( fractureWorld == NULL || actor == NULL || fractureWorld->actors == NULL )
	{
		return;
	}

	int index = (int)( actor - fractureWorld->actors );
	if ( index < 0 || index >= fractureWorld->actorCount )
	{
		return;
	}

	b2BlastActor_FreeArrays( actor );
	*actor = (b2BlastActor){ 0 };
	actor->id.index = UINT32_MAX;
	actor->bodyId = B2_NULL_INDEX;
	actor->shapeId = B2_NULL_INDEX;
	if ( index == fractureWorld->actorCount - 1 )
	{
		fractureWorld->actorCount -= 1;
	}
}

static void b2BlastEnsureActorCapacity(
	b2BlastActor* actor, int leafCapacity, int bondCapacity, int clusterCapacity, int clusterRefCapacity, int cellCount, int seedCapacity )
{
	if ( leafCapacity > actor->leafCapacity )
	{
		actor->leaves = b2BlastResize( actor->leaves, actor->leafCapacity, leafCapacity, (int)sizeof( b2BlastLeaf ) );
		actor->componentScratch =
			b2BlastResize( actor->componentScratch, actor->componentScratchCapacity, leafCapacity, (int)sizeof( int ) );
		actor->queueScratch = b2BlastResize( actor->queueScratch, actor->queueScratchCapacity, leafCapacity, (int)sizeof( int ) );
		actor->visitScratch = b2BlastResize( actor->visitScratch, actor->visitScratchCapacity, leafCapacity, (int)sizeof( uint8_t ) );
		actor->graphDistanceScratch =
			b2BlastResize( actor->graphDistanceScratch, actor->graphDistanceScratchCapacity, leafCapacity, (int)sizeof( float ) );
		actor->graphParentBondScratch =
			b2BlastResize( actor->graphParentBondScratch, actor->graphParentBondScratchCapacity, leafCapacity, (int)sizeof( int ) );
		actor->leafCapacity = leafCapacity;
		actor->componentScratchCapacity = leafCapacity;
		actor->queueScratchCapacity = leafCapacity;
		actor->visitScratchCapacity = leafCapacity;
		actor->graphDistanceScratchCapacity = leafCapacity;
		actor->graphParentBondScratchCapacity = leafCapacity;
	}

	if ( seedCapacity > actor->seedScratchCapacity )
	{
		actor->seedScratch =
			b2BlastResize( actor->seedScratch, actor->seedScratchCapacity, seedCapacity, (int)sizeof( b2BlastSeed ) );
		actor->seedScratchCapacity = seedCapacity;
	}

	if ( cellCount > actor->assignScratchCapacity )
	{
		actor->assignScratch =
			b2BlastResize( actor->assignScratch, actor->assignScratchCapacity, cellCount, (int)sizeof( int32_t ) );
		actor->assignScratchCapacity = cellCount;
	}

	if ( cellCount > actor->cellScratchCapacity )
	{
		actor->cellScratch =
			b2BlastResize( actor->cellScratch, actor->cellScratchCapacity, cellCount, (int)sizeof( int32_t ) );
		actor->cellScratchCapacity = cellCount;
	}

	if ( cellCount > actor->cellVisitScratchCapacity )
	{
		actor->cellVisitScratch =
			b2BlastResize( actor->cellVisitScratch, actor->cellVisitScratchCapacity, cellCount, (int)sizeof( uint8_t ) );
		actor->cellVisitScratchCapacity = cellCount;
	}

	if ( seedCapacity > actor->leafRemapScratchCapacity )
	{
		actor->leafRemapScratch =
			b2BlastResize( actor->leafRemapScratch, actor->leafRemapScratchCapacity, seedCapacity, (int)sizeof( uint32_t ) );
		actor->leafRemapScratchCapacity = seedCapacity;
	}

	if ( bondCapacity > actor->bondCapacity )
	{
		actor->bonds = b2BlastResize( actor->bonds, actor->bondCapacity, bondCapacity, (int)sizeof( b2BlastActiveBond ) );
		actor->bondCapacity = bondCapacity;
	}

	if ( clusterCapacity > actor->clusterCapacity )
	{
		actor->clusters =
			b2BlastResize( actor->clusters, actor->clusterCapacity, clusterCapacity, (int)sizeof( b2BlastCluster ) );
		actor->clusterCapacity = clusterCapacity;
	}

	if ( clusterCapacity > actor->activeClusterCapacity )
	{
		actor->activeClusters =
			b2BlastResize( actor->activeClusters, actor->activeClusterCapacity, clusterCapacity, (int)sizeof( uint32_t ) );
		actor->activeClusterCapacity = clusterCapacity;
	}

	if ( clusterRefCapacity > actor->clusterChildCapacity )
	{
		actor->clusterChildren =
			b2BlastResize( actor->clusterChildren, actor->clusterChildCapacity, clusterRefCapacity, (int)sizeof( uint32_t ) );
		actor->clusterChildCapacity = clusterRefCapacity;
	}

	if ( clusterRefCapacity > actor->clusterLeafRefCapacity )
	{
		actor->clusterLeaves =
			b2BlastResize( actor->clusterLeaves, actor->clusterLeafRefCapacity, clusterRefCapacity, (int)sizeof( uint32_t ) );
		actor->clusterLeafRefCapacity = clusterRefCapacity;
	}

	if ( cellCount > actor->cellToLeafCapacity )
	{
		actor->cellToLeaf = b2BlastResize( actor->cellToLeaf, actor->cellToLeafCapacity, cellCount, (int)sizeof( uint32_t ) );
		actor->cellToLeafCapacity = cellCount;
	}
}

static bool b2BlastEnsureOwnedPixelAssetCapacity( b2BlastActor* actor, int width, int height )
{
	if ( actor == NULL || width <= 0 || height <= 0 || width > INT16_MAX || height > INT16_MAX || width > INT32_MAX / height )
	{
		return false;
	}
	int cellCount = width * height;
	int wordCount = ( cellCount + 63 ) / 64;
	int featureRefCapacity = b2MaxInt( 4, cellCount * 4 );
	if ( wordCount > actor->ownedOccupancyWordCapacity )
	{
		actor->ownedOccupancyBits =
			b2BlastResize( actor->ownedOccupancyBits, actor->ownedOccupancyWordCapacity, wordCount, (int)sizeof( uint64_t ) );
		actor->ownedOccupancyWordCapacity = wordCount;
	}
	if ( cellCount > actor->ownedMaterialIdCapacity )
	{
		actor->ownedMaterialIds =
			b2BlastResize( actor->ownedMaterialIds, actor->ownedMaterialIdCapacity, cellCount, (int)sizeof( b2BlastMaterialId ) );
		actor->ownedMaterialIdCapacity = cellCount;
	}
	if ( cellCount > actor->ownedFeatureTypeCapacity )
	{
		actor->ownedFeatureTypes =
			b2BlastResize( actor->ownedFeatureTypes, actor->ownedFeatureTypeCapacity, cellCount, (int)sizeof( uint8_t ) );
		actor->ownedFeatureTypeCapacity = cellCount;
	}
	if ( cellCount > actor->ownedNormalIndexCapacity )
	{
		actor->ownedNormalIndices =
			b2BlastResize( actor->ownedNormalIndices, actor->ownedNormalIndexCapacity, cellCount, (int)sizeof( uint8_t ) );
		actor->ownedNormalIndexCapacity = cellCount;
	}
	if ( featureRefCapacity > actor->ownedCornerCapacity )
	{
		actor->ownedCorners =
			b2BlastResize( actor->ownedCorners, actor->ownedCornerCapacity, featureRefCapacity, (int)sizeof( b2PixelFeatureRef ) );
		actor->ownedCornerCapacity = featureRefCapacity;
	}
	if ( featureRefCapacity > actor->ownedEdgeCapacity )
	{
		actor->ownedEdges =
			b2BlastResize( actor->ownedEdges, actor->ownedEdgeCapacity, featureRefCapacity, (int)sizeof( b2PixelFeatureRef ) );
		actor->ownedEdgeCapacity = featureRefCapacity;
	}
	if ( height > actor->ownedRowSolidCountCapacity )
	{
		actor->ownedRowSolidCounts =
			b2BlastResize( actor->ownedRowSolidCounts, actor->ownedRowSolidCountCapacity, height, (int)sizeof( int32_t ) );
		actor->ownedRowSolidCountCapacity = height;
	}
	if ( width > actor->ownedColSolidCountCapacity )
	{
		actor->ownedColSolidCounts =
			b2BlastResize( actor->ownedColSolidCounts, actor->ownedColSolidCountCapacity, width, (int)sizeof( int32_t ) );
		actor->ownedColSolidCountCapacity = width;
	}
	if ( cellCount > actor->ownedPixelScratchCapacity )
	{
		actor->ownedPixelScratch =
			b2BlastResize( actor->ownedPixelScratch, actor->ownedPixelScratchCapacity, cellCount, (int)sizeof( uint8_t ) );
		actor->ownedPixelScratchCapacity = cellCount;
	}
	return actor->ownedOccupancyBits != NULL && actor->ownedMaterialIds != NULL && actor->ownedFeatureTypes != NULL &&
		   actor->ownedNormalIndices != NULL && actor->ownedCorners != NULL && actor->ownedEdges != NULL &&
		   actor->ownedRowSolidCounts != NULL && actor->ownedColSolidCounts != NULL && actor->ownedPixelScratch != NULL;
}

static uint32_t b2BlastHash32( uint32_t value )
{
	value ^= value >> 16;
	value *= 0x7feb352du;
	value ^= value >> 15;
	value *= 0x846ca68bu;
	value ^= value >> 16;
	return value;
}

static float b2BlastRand01( int32_t i, int32_t j, int32_t k )
{
	uint32_t h = b2BlastHash32( (uint32_t)i * 374761393u ^ (uint32_t)j * 668265263u ^ (uint32_t)k * 2246822519u );
	return (float)( h & 0x00ffffffu ) / 16777216.0f;
}

static float b2BlastRandRange( int32_t seed, float a, float b )
{
	return a + ( b - a ) * b2BlastRand01( seed, seed * 17 + 31, seed * 131 + 7 );
}

static uint64_t b2BlastMixHash( uint64_t hash, uint64_t value )
{
	hash ^= value;
	hash *= 1099511628211ULL;
	return hash;
}

static uint32_t b2BlastFloatBits( float value )
{
	uint32_t bits = 0;
	memcpy( &bits, &value, sizeof( bits ) );
	return bits;
}

static uint64_t b2BlastComputeActorHash( const b2BlastActor* actor )
{
	if ( actor == NULL )
	{
		return 0;
	}

	const b2PixelAsset* asset = &actor->ownedAsset;
	uint64_t hash = 14695981039346656037ULL;
	hash = b2BlastMixHash( hash, (uint64_t)b2MaxInt( 0, asset->width ) );
	hash = b2BlastMixHash( hash, (uint64_t)b2MaxInt( 0, asset->height ) );
	hash = b2BlastMixHash( hash, (uint64_t)actor->leafCount );
	hash = b2BlastMixHash( hash, (uint64_t)actor->bondCount );
	hash = b2BlastMixHash( hash, (uint64_t)actor->clusterCount );
	hash = b2BlastMixHash( hash, actor->materialHash != 0 ? actor->materialHash : asset->materialHash );
	for ( int i = 0; i < actor->leafCount; ++i )
	{
		const b2BlastLeaf* leaf = actor->leaves + i;
		hash = b2BlastMixHash( hash, leaf->cellCount );
		hash = b2BlastMixHash( hash, leaf->dominantMaterialId );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( leaf->centroid.x ) );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( leaf->centroid.y ) );
	}
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		const b2BlastActiveBond* bond = actor->bonds + i;
		hash = b2BlastMixHash( hash, ( (uint64_t)bond->leafA << 32 ) | bond->leafB );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( bond->area ) );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( bond->capacity ) );
	}
	return hash;
}

static b2Vec2 b2BlastCellCenter( const b2PixelAsset* asset, const b2Shape* shape, int x, int y )
{
	float halfWidth = 0.5f * (float)asset->width * asset->pixelSize;
	float halfHeight = 0.5f * (float)asset->height * asset->pixelSize;
	return (b2Vec2){ ( (float)x + 0.5f ) * asset->pixelSize - halfWidth + shape->pixel.localOrigin.x,
					 ( (float)y + 0.5f ) * asset->pixelSize - halfHeight + shape->pixel.localOrigin.y };
}

static float b2BlastMaterialDensity( const b2PixelAsset* asset, b2BlastMaterialId materialId, float fallbackDensity )
{
	if ( asset != NULL && asset->materialTable != NULL && asset->materialTable->denseMaterials != NULL &&
		 (int)materialId < asset->materialTable->denseMaterialCount )
	{
		float density = asset->materialTable->denseMaterials[materialId].density;
		if ( density > 0.0f && b2IsValidFloat( density ) )
		{
			return density;
		}
	}
	return fallbackDensity > 0.0f ? fallbackDensity : 1.0f;
}

static const b2BlastMaterialPhysics* b2BlastFindMaterial( const b2PixelAsset* asset, b2BlastMaterialId materialId )
{
	if ( asset == NULL || asset->materialTable == NULL )
	{
		return NULL;
	}

	const b2BlastMaterialTable* table = asset->materialTable;
	if ( table->denseMaterials != NULL && (int)materialId < table->denseMaterialCount )
	{
		const b2BlastMaterialPhysics* material = table->denseMaterials + materialId;
		return material->materialId == materialId ? material : NULL;
	}

	for ( int i = 0; i < table->materialCount; ++i )
	{
		if ( table->materials[i].materialId == materialId )
		{
			return table->materials + i;
		}
	}

	return NULL;
}

typedef struct b2BlastAuthoringParams
{
	int leafTarget;
	int maxClusterLevel;
	int startLevel;
	uint32_t seed;
	int shaderId;
	float seedSpacing;
	float grainNoise;
	float anisotropy;
	float fiberAngleRadians;
	float grainParallelScale;
	float grainPerpScale;
	float seamStrengthScale;
	bool anisotropicGrain;
	bool radialSeeds;
	bool masonryMortar;
} b2BlastAuthoringParams;

enum
{
	b2_blastTargetLeafEdgeCells = 8,
	b2_blastTargetActiveEdgeCells = 64,
};

static b2BlastAuthoringParams b2BlastDefaultAuthoringParams( void )
{
	b2BlastAuthoringParams params = { 0 };
	params.leafTarget = 150;
	params.maxClusterLevel = 3;
	params.startLevel = 3;
	params.seed = 13;
	params.shaderId = b2_blastShaderChaos;
	params.seedSpacing = 3.3f;
	params.grainNoise = 0.45f;
	params.anisotropy = 0.15f;
	params.fiberAngleRadians = 0.0f;
	params.grainParallelScale = 1.0f;
	params.grainPerpScale = 1.0f;
	params.seamStrengthScale = 1.0f;
	return params;
}

static b2BlastAuthoringParams b2BlastAuthoringParamsFromAsset( const b2PixelAsset* asset, b2BlastMaterialId dominantMaterial )
{
	b2BlastAuthoringParams params = b2BlastDefaultAuthoringParams();
	const b2BlastMaterialPhysics* material = b2BlastFindMaterial( asset, dominantMaterial );
	if ( material != NULL )
	{
		params.leafTarget = b2ClampInt( material->leafTarget > 0 ? material->leafTarget : params.leafTarget, 16, 500 );
		params.maxClusterLevel = b2ClampInt( material->maxClusterLevel > 0 ? material->maxClusterLevel : params.maxClusterLevel, 1, 8 );
		params.startLevel = b2ClampInt( material->startLevel >= 0 ? material->startLevel : params.startLevel, 0, 8 );
		params.seed = material->seed != 0 ? material->seed : params.seed;
		params.shaderId = material->authoringShaderId != 0 ? material->authoringShaderId : params.shaderId;
		params.grainNoise = material->grainNoise > 0.0f ? material->grainNoise : params.grainNoise;
		params.anisotropy = b2ClampFloat( material->anisotropy, 0.0f, 1.0f );
		params.fiberAngleRadians = material->fiberAngleDeg * 0.017453292519943295f;
	}

	if ( params.shaderId == b2_blastShaderWoodGrain )
	{
		params.seedSpacing = 4.5f;
		params.anisotropicGrain = true;
		params.grainParallelScale = 0.16f;
		params.grainPerpScale = 2.15f;
	}
	else if ( params.shaderId == b2_blastShaderGlassRadial )
	{
		params.seedSpacing = 3.0f;
		params.radialSeeds = true;
	}
	else if ( params.shaderId == b2_blastShaderMasonryMortar )
	{
		params.seedSpacing = 3.3f;
		params.masonryMortar = true;
		params.seamStrengthScale = 0.55f;
	}
	else if ( params.shaderId == b2_blastShaderNoisyConcrete )
	{
		params.seedSpacing = 3.2f;
		params.seamStrengthScale = 0.88f;
	}
	else
	{
		params.shaderId = b2_blastShaderChaos;
		params.seedSpacing = 3.3f;
	}
	return params;
}

static float b2BlastAuthoringTargetLeafCellArea( const b2BlastAuthoringParams* params )
{
	(void)params;
	return (float)( b2_blastTargetLeafEdgeCells * b2_blastTargetLeafEdgeCells );
}

static float b2BlastAuthoringTargetLevelCellEdge( int level )
{
	float targetEdge = (float)b2_blastTargetLeafEdgeCells;
	for ( int i = 0; i < level; ++i )
	{
		targetEdge *= 2.0f;
	}
	return b2MaxFloat( 1.0f, targetEdge );
}

static float b2BlastAuthoringTargetLevelCellArea( int level )
{
	const float targetEdge = b2BlastAuthoringTargetLevelCellEdge( level );
	return targetEdge * targetEdge;
}

static float b2BlastAuthoringTargetActiveCellArea( const b2BlastAuthoringParams* params )
{
	const int startLevel = params != NULL ? b2MaxInt( 0, params->startLevel ) : 3;
	return b2MaxFloat( 1.0f, b2BlastAuthoringTargetLevelCellArea( startLevel ) );
}

static b2BlastAuthoringParams b2BlastAuthoringParamsFromActor( const b2BlastActor* actor )
{
	b2BlastMaterialId dominantMaterial = 0;
	if ( actor != NULL )
	{
		for ( int leafIndex = 0; leafIndex < actor->leafCount; ++leafIndex )
		{
			if ( actor->leaves[leafIndex].cellCount > 0 )
			{
				dominantMaterial = actor->leaves[leafIndex].dominantMaterialId;
				break;
			}
		}
	}
	return b2BlastAuthoringParamsFromAsset( actor != NULL ? &actor->ownedAsset : NULL, dominantMaterial );
}

static float b2BlastBondCapacity( const b2PixelAsset* asset, b2BlastMaterialId materialA, b2BlastMaterialId materialB, float area )
{
	const b2BlastMaterialPhysics* a = b2BlastFindMaterial( asset, materialA );
	const b2BlastMaterialPhysics* b = b2BlastFindMaterial( asset, materialB );
	float strengthA = a != NULL ? b2MaxFloat( a->tensileStrength, b2MaxFloat( a->shearStrength, a->contactCapacity ) ) : 64.0f;
	float strengthB = b != NULL ? b2MaxFloat( b->tensileStrength, b2MaxFloat( b->shearStrength, b->contactCapacity ) ) : strengthA;
	float brittlenessA = a != NULL ? a->brittleness : 0.5f;
	float brittlenessB = b != NULL ? b->brittleness : brittlenessA;
	float weakSide = b2MinFloat( strengthA, strengthB );
	float brittleScale = 1.0f - 0.35f * b2ClampFloat( 0.5f * ( brittlenessA + brittlenessB ), 0.0f, 1.0f );
	return b2MaxFloat( 1.0f, weakSide * b2MaxFloat( area, 1.0f ) * brittleScale );
}

static bool b2BlastPixelAsset_CellHasSupport( const b2PixelAsset* asset, int cell, int cellCount )
{
	return asset != NULL && asset->supportMask != NULL && asset->supportMaskCount >= cellCount && cell >= 0 && cell < cellCount &&
		   asset->supportMask[cell] != 0;
}

static bool b2BlastClusterIsAdjacentToGroup( const b2BlastActor* actor, uint32_t candidateLocalCluster, int groupLabel )
{
	if ( actor == NULL || actor->leafRemapScratch == NULL || actor->componentScratch == NULL )
	{
		return false;
	}
	for ( int bondIndex = 0; bondIndex < actor->bondCount; ++bondIndex )
	{
		const b2BlastActiveBond* bond = actor->bonds + bondIndex;
		if ( bond->leafA >= (uint32_t)actor->leafCount || bond->leafB >= (uint32_t)actor->leafCount )
		{
			continue;
		}
		const uint32_t clusterA = actor->leafRemapScratch[bond->leafA];
		const uint32_t clusterB = actor->leafRemapScratch[bond->leafB];
		if ( clusterA == UINT32_MAX || clusterB == UINT32_MAX )
		{
			continue;
		}
		if ( clusterA == candidateLocalCluster && actor->componentScratch[clusterB] == groupLabel )
		{
			return true;
		}
		if ( clusterB == candidateLocalCluster && actor->componentScratch[clusterA] == groupLabel )
		{
			return true;
		}
	}
	return false;
}

static bool b2BlastAppendClusterChildToParent( b2BlastActor* actor, int currentStart, int childLocalIndex, b2BlastCluster* parent )
{
	if ( actor == NULL || parent == NULL || childLocalIndex < 0 )
	{
		return false;
	}
	b2BlastCluster* child = actor->clusters + currentStart + childLocalIndex;
	child->parent = parent->id;
	if ( actor->clusterChildCount >= actor->clusterChildCapacity || actor->clusterLeafRefCount + (int)child->leafCount > actor->clusterLeafRefCapacity )
	{
		return false;
	}
	actor->clusterChildren[actor->clusterChildCount++] = child->id;
	parent->mass += child->mass;
	parent->centroid.x += child->centroid.x * child->mass;
	parent->centroid.y += child->centroid.y * child->mass;
	parent->minX = (uint16_t)b2MinInt( parent->minX, child->minX );
	parent->minY = (uint16_t)b2MinInt( parent->minY, child->minY );
	parent->maxX = (uint16_t)b2MaxInt( parent->maxX, child->maxX );
	parent->maxY = (uint16_t)b2MaxInt( parent->maxY, child->maxY );
	parent->flags |= child->flags;
	for ( uint32_t li = 0; li < child->leafCount; ++li )
	{
		actor->clusterLeaves[actor->clusterLeafRefCount++] = actor->clusterLeaves[child->firstLeaf + li];
		parent->leafCount += 1;
	}
	parent->childCount += 1;
	return true;
}

static void b2BlastActor_ResetActiveClustersAtLevel( b2BlastActor* actor, uint16_t activeLevel )
{
	if ( actor == NULL )
	{
		return;
	}
	actor->activeClusterCount = 0;
	if ( actor->activeClusterCapacity < actor->clusterCount )
	{
		actor->activeClusters =
			b2BlastResize( actor->activeClusters, actor->activeClusterCapacity, actor->clusterCount, (int)sizeof( uint32_t ) );
		actor->activeClusterCapacity = actor->clusterCount;
	}
	for ( int clusterIndex = 0; clusterIndex < actor->clusterCount; ++clusterIndex )
	{
		const b2BlastCluster* cluster = actor->clusters + clusterIndex;
		if ( cluster->level == activeLevel && cluster->leafCount > 0 && actor->activeClusterCount < actor->activeClusterCapacity )
		{
			actor->activeClusters[actor->activeClusterCount++] = cluster->id;
		}
	}
	if ( actor->activeClusterCount == 0 && actor->leafCount > 0 )
	{
		for ( int clusterIndex = 0; clusterIndex < actor->clusterCount; ++clusterIndex )
		{
			const b2BlastCluster* cluster = actor->clusters + clusterIndex;
			if ( cluster->level == 0 && cluster->leafCount > 0 && actor->activeClusterCount < actor->activeClusterCapacity )
			{
				actor->activeClusters[actor->activeClusterCount++] = cluster->id;
			}
		}
		actor->initialActiveLevel = 0;
	}
}

static bool b2BlastClusterContainsLeaf( const b2BlastActor* actor, const b2BlastCluster* cluster, uint32_t leafIndex )
{
	if ( actor == NULL || cluster == NULL || cluster->firstLeaf == UINT32_MAX ||
		 cluster->firstLeaf + cluster->leafCount > (uint32_t)actor->clusterLeafRefCount )
	{
		return false;
	}
	for ( uint32_t li = 0; li < cluster->leafCount; ++li )
	{
		if ( actor->clusterLeaves[cluster->firstLeaf + li] == leafIndex )
		{
			return true;
		}
	}
	return false;
}

static bool b2BlastActor_ActiveContains( const b2BlastActor* actor, uint32_t clusterId )
{
	if ( actor == NULL || actor->activeClusters == NULL )
	{
		return false;
	}
	for ( int i = 0; i < actor->activeClusterCount; ++i )
	{
		if ( actor->activeClusters[i] == clusterId )
		{
			return true;
		}
	}
	return false;
}

static bool b2BlastActor_AddActiveCluster( b2BlastActor* actor, uint32_t clusterId )
{
	if ( actor == NULL || b2BlastActor_ActiveContains( actor, clusterId ) )
	{
		return false;
	}
	if ( actor->activeClusterCount >= actor->activeClusterCapacity )
	{
		const int oldCapacity = actor->activeClusterCapacity;
		const int newCapacity = oldCapacity + b2MaxInt( 16, oldCapacity >> 1 );
		actor->activeClusters = b2BlastResize( actor->activeClusters, oldCapacity, newCapacity, (int)sizeof( uint32_t ) );
		actor->activeClusterCapacity = newCapacity;
	}
	if ( actor->activeClusterCount >= actor->activeClusterCapacity )
	{
		return false;
	}
	actor->activeClusters[actor->activeClusterCount++] = clusterId;
	return true;
}

static void b2BlastActor_SortUniqueActiveClusters( b2BlastActor* actor )
{
	if ( actor == NULL || actor->activeClusterCount <= 1 )
	{
		return;
	}
	for ( int i = 1; i < actor->activeClusterCount; ++i )
	{
		uint32_t value = actor->activeClusters[i];
		int j = i - 1;
		while ( j >= 0 && actor->activeClusters[j] > value )
		{
			actor->activeClusters[j + 1] = actor->activeClusters[j];
			--j;
		}
		actor->activeClusters[j + 1] = value;
	}
	int write = 0;
	for ( int read = 0; read < actor->activeClusterCount; ++read )
	{
		if ( write == 0 || actor->activeClusters[read] != actor->activeClusters[write - 1] )
		{
			actor->activeClusters[write++] = actor->activeClusters[read];
		}
	}
	actor->activeClusterCount = write;
}

static int b2BlastCluster_CellCount( const b2BlastActor* actor, const b2BlastCluster* cluster )
{
	if ( actor == NULL || cluster == NULL || cluster->firstLeaf == UINT32_MAX ||
		 cluster->firstLeaf + cluster->leafCount > (uint32_t)actor->clusterLeafRefCount )
	{
		return 0;
	}

	int cellCount = 0;
	for ( uint32_t ref = 0; ref < cluster->leafCount; ++ref )
	{
		const uint32_t leafIndex = actor->clusterLeaves[cluster->firstLeaf + ref];
		if ( leafIndex < (uint32_t)actor->leafCount )
		{
			cellCount += actor->leaves[leafIndex].cellCount;
		}
	}
	return cellCount;
}

static int b2BlastCluster_CellWidth( const b2BlastCluster* cluster )
{
	if ( cluster == NULL || cluster->minX == UINT16_MAX )
	{
		return 0;
	}
	return b2MaxInt( 0, (int)cluster->maxX - (int)cluster->minX + 1 );
}

static int b2BlastCluster_CellHeight( const b2BlastCluster* cluster )
{
	if ( cluster == NULL || cluster->minY == UINT16_MAX )
	{
		return 0;
	}
	return b2MaxInt( 0, (int)cluster->maxY - (int)cluster->minY + 1 );
}

static bool b2BlastCluster_WithinTargetCellScale(
	const b2BlastActor* actor, const b2BlastCluster* cluster, float targetCellArea, float targetCellEdge )
{
	return (float)b2BlastCluster_CellCount( actor, cluster ) <= targetCellArea &&
		   (float)b2BlastCluster_CellWidth( cluster ) <= targetCellEdge &&
		   (float)b2BlastCluster_CellHeight( cluster ) <= targetCellEdge;
}

static bool b2BlastCluster_ShouldRefineForShape(
	const b2BlastActor* actor, const b2BlastCluster* cluster, float targetCellEdge )
{
	if ( actor == NULL || cluster == NULL || cluster->level == 0 || cluster->childCount == 0 )
	{
		return false;
	}

	const int width = b2BlastCluster_CellWidth( cluster );
	const int height = b2BlastCluster_CellHeight( cluster );
	if ( width <= 0 || height <= 0 )
	{
		return false;
	}

	const int minDim = b2MinInt( width, height );
	const int maxDim = b2MaxInt( width, height );
	const int cellCount = b2BlastCluster_CellCount( actor, cluster );
	const float fill = (float)cellCount / b2MaxFloat( 1.0f, (float)( width * height ) );
	const float leafEdge = b2BlastAuthoringTargetLevelCellEdge( 0 );

	if ( (float)minDim <= leafEdge && (float)maxDim > leafEdge * 1.5f )
	{
		return true;
	}
	if ( (float)minDim <= targetCellEdge * 0.5f && (float)maxDim >= targetCellEdge * 0.75f )
	{
		return true;
	}
	if ( fill < 0.65f && (float)maxDim >= targetCellEdge * 0.75f )
	{
		return true;
	}
	return false;
}

static bool b2BlastPendingClusterReachedTargetScale(
	const b2BlastActor* actor, const b2BlastCluster* cluster, float targetCellArea, float targetCellEdge )
{
	if ( cluster == NULL || cluster->childCount == 0 )
	{
		return false;
	}
	return (float)b2BlastCluster_CellCount( actor, cluster ) >= targetCellArea ||
		   (float)b2BlastCluster_CellWidth( cluster ) >= targetCellEdge ||
		   (float)b2BlastCluster_CellHeight( cluster ) >= targetCellEdge;
}

static bool b2BlastActor_AddActiveClustersByArea(
	b2BlastActor* actor, uint32_t clusterId, float targetCellArea, float targetCellEdge )
{
	if ( actor == NULL || clusterId >= (uint32_t)actor->clusterCount )
	{
		return false;
	}

	const b2BlastCluster* cluster = actor->clusters + clusterId;
	const bool withinTarget = b2BlastCluster_WithinTargetCellScale( actor, cluster, targetCellArea, targetCellEdge );
	const bool shapeNeedsRefine = b2BlastCluster_ShouldRefineForShape( actor, cluster, targetCellEdge );
	if ( cluster->level == 0 || cluster->childCount == 0 || ( withinTarget && shapeNeedsRefine == false ) )
	{
		return b2BlastActor_AddActiveCluster( actor, clusterId );
	}

	bool addedChild = false;
	for ( uint32_t childRef = 0; childRef < cluster->childCount; ++childRef )
	{
		if ( cluster->firstChild + childRef >= (uint32_t)actor->clusterChildCount )
		{
			continue;
		}
		const uint32_t childId = actor->clusterChildren[cluster->firstChild + childRef];
		addedChild = b2BlastActor_AddActiveClustersByArea( actor, childId, targetCellArea, targetCellEdge ) || addedChild;
	}
	if ( addedChild == false )
	{
		return b2BlastActor_AddActiveCluster( actor, clusterId );
	}
	return true;
}

static void b2BlastActor_ResetActiveClustersByArea( b2BlastActor* actor, const b2BlastAuthoringParams* params )
{
	if ( actor == NULL )
	{
		return;
	}

	actor->activeClusterCount = 0;
	if ( actor->activeClusterCapacity < actor->clusterCount )
	{
		actor->activeClusters =
			b2BlastResize( actor->activeClusters, actor->activeClusterCapacity, actor->clusterCount, (int)sizeof( uint32_t ) );
		actor->activeClusterCapacity = actor->clusterCount;
	}

	if ( actor->rootCluster != UINT32_MAX && actor->rootCluster < (uint32_t)actor->clusterCount )
	{
		const int startLevel = params != NULL ? b2MaxInt( 0, params->startLevel ) : 3;
		(void)b2BlastActor_AddActiveClustersByArea(
			actor, actor->rootCluster, b2BlastAuthoringTargetActiveCellArea( params ),
			b2BlastAuthoringTargetLevelCellEdge( startLevel ) );
	}
	if ( actor->activeClusterCount == 0 )
	{
		b2BlastActor_ResetActiveClustersAtLevel( actor, actor->initialActiveLevel );
	}
	b2BlastActor_SortUniqueActiveClusters( actor );
}

static bool b2BlastActor_RefineActiveCluster( b2BlastActor* actor, uint32_t clusterId )
{
	if ( actor == NULL || actor->activeClusterCount <= 0 || actor->activeClusters == NULL )
	{
		return false;
	}
	for ( int activeIndex = 0; activeIndex < actor->activeClusterCount; ++activeIndex )
	{
		if ( actor->activeClusters[activeIndex] != clusterId || clusterId >= (uint32_t)actor->clusterCount )
		{
			continue;
		}
		const b2BlastCluster* cluster = actor->clusters + clusterId;
		if ( cluster->level == 0 || cluster->childCount == 0 )
		{
			continue;
		}
		actor->activeClusters[activeIndex] = actor->activeClusters[actor->activeClusterCount - 1];
		actor->activeClusterCount -= 1;
		for ( uint32_t childRef = 0; childRef < cluster->childCount; ++childRef )
		{
			if ( cluster->firstChild + childRef >= (uint32_t)actor->clusterChildCount )
			{
				continue;
			}
			const uint32_t childId = actor->clusterChildren[cluster->firstChild + childRef];
			(void)b2BlastActor_AddActiveCluster( actor, childId );
		}
		b2BlastActor_SortUniqueActiveClusters( actor );
		return true;
	}
	return false;
}

static bool b2BlastActor_RefineActiveClusterContainingLeaf( b2BlastActor* actor, uint32_t leafIndex )
{
	if ( actor == NULL || actor->activeClusterCount <= 0 || actor->activeClusters == NULL )
	{
		return false;
	}
	for ( int activeIndex = 0; activeIndex < actor->activeClusterCount; ++activeIndex )
	{
		const uint32_t clusterId = actor->activeClusters[activeIndex];
		if ( clusterId >= (uint32_t)actor->clusterCount )
		{
			continue;
		}
		const b2BlastCluster* cluster = actor->clusters + clusterId;
		if ( cluster->level == 0 || cluster->childCount == 0 || b2BlastClusterContainsLeaf( actor, cluster, leafIndex ) == false )
		{
			continue;
		}
		return b2BlastActor_RefineActiveCluster( actor, clusterId );
	}
	return false;
}

static uint32_t b2BlastActor_FindActiveClusterContainingLeaf( const b2BlastActor* actor, uint32_t leafIndex )
{
	if ( actor == NULL || actor->activeClusterCount <= 0 || actor->activeClusters == NULL )
	{
		return UINT32_MAX;
	}
	for ( int activeIndex = 0; activeIndex < actor->activeClusterCount; ++activeIndex )
	{
		const uint32_t clusterId = actor->activeClusters[activeIndex];
		if ( clusterId >= (uint32_t)actor->clusterCount )
		{
			continue;
		}
		const b2BlastCluster* cluster = actor->clusters + clusterId;
		if ( b2BlastClusterContainsLeaf( actor, cluster, leafIndex ) )
		{
			return clusterId;
		}
	}
	return UINT32_MAX;
}

static void b2BlastActor_RefineActivePathToLeaf( b2BlastActor* actor, uint32_t leafIndex )
{
	if ( actor == NULL || leafIndex >= (uint32_t)actor->leafCount )
	{
		return;
	}
	bool changed = true;
	while ( changed && b2BlastActor_ActiveContains( actor, leafIndex ) == false )
	{
		changed = b2BlastActor_RefineActiveClusterContainingLeaf( actor, leafIndex );
	}
	if ( b2BlastActor_ActiveContains( actor, leafIndex ) == false )
	{
		(void)b2BlastActor_AddActiveCluster( actor, leafIndex );
	}
	b2BlastActor_SortUniqueActiveClusters( actor );
}

static void b2BlastActor_PruneActiveClusters( b2BlastActor* actor )
{
	if ( actor == NULL || actor->activeClusterCount <= 0 )
	{
		return;
	}
	int write = 0;
	for ( int read = 0; read < actor->activeClusterCount; ++read )
	{
		const uint32_t clusterId = actor->activeClusters[read];
		if ( clusterId >= (uint32_t)actor->clusterCount )
		{
			continue;
		}
		const b2BlastCluster* cluster = actor->clusters + clusterId;
		if ( cluster->leafCount == 0 || cluster->firstLeaf == UINT32_MAX ||
			 cluster->firstLeaf + cluster->leafCount > (uint32_t)actor->clusterLeafRefCount )
		{
			continue;
		}
		bool hasLiveLeaf = false;
		for ( uint32_t li = 0; li < cluster->leafCount; ++li )
		{
			const uint32_t leafIndex = actor->clusterLeaves[cluster->firstLeaf + li];
			if ( leafIndex < (uint32_t)actor->leafCount &&
				 ( actor->leaves[leafIndex].flags & b2_blastLeafFlagDetached ) == 0 &&
				 actor->leaves[leafIndex].cellCount > 0 )
			{
				hasLiveLeaf = true;
				break;
			}
		}
		if ( hasLiveLeaf )
		{
			actor->activeClusters[write++] = clusterId;
		}
	}
	actor->activeClusterCount = write;
	b2BlastActor_SortUniqueActiveClusters( actor );
}

static void b2BlastActor_ClearRuntimeDemand( b2BlastActor* actor )
{
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		actor->bonds[i].impactDemand = 0.0f;
		actor->bonds[i].loadDemand = 0.0f;
		actor->bonds[i].flags &= (uint16_t)~b2_blastBondFlagBreakCandidate;
	}
}

static bool b2BlastActor_AuthorFromPixelShape( b2BlastActor* actor, b2Shape* shape )
{
	if ( actor == NULL || shape == NULL || shape->type != b2_pixelShape || b2IsPixelShapeUsable( &shape->pixel ) == false )
	{
		return false;
	}

	const b2PixelAsset* asset = shape->pixel.asset;
	actor->ownedAsset = *asset;
	const int width = asset->width;
	const int height = asset->height;
	const int cellCount = width * height;
	const int maxLeaves = b2MaxInt( 1, asset->solidCount );
	const int maxBonds = b2MaxInt( 1, asset->solidCount * 2 );
	const int maxClusters = b2MaxInt( maxLeaves + 1, maxLeaves * ( b2BlastDefaultAuthoringParams().maxClusterLevel + 2 ) + 8 );
	const int maxClusterRefs = b2MaxInt( maxClusters * 4, maxLeaves * 8 + 8 );

	b2BlastEnsureActorCapacity( actor, maxLeaves, maxBonds, maxClusters, maxClusterRefs, cellCount, maxLeaves );
	for ( int i = 0; i < cellCount; ++i )
	{
		actor->cellToLeaf[i] = UINT32_MAX;
		actor->assignScratch[i] = -1;
		actor->cellVisitScratch[i] = 0;
	}

	actor->leafCount = 0;
	actor->bondCount = 0;
	actor->clusterCount = 0;
	actor->clusterChildCount = 0;
	actor->clusterLeafRefCount = 0;
	actor->cellToLeafCount = cellCount;
	actor->worldAnchorCount = 0;
	actor->rootCluster = UINT32_MAX;
	actor->initialActiveLevel = 0;

	int occupiedCount = 0;
	int occupiedMinX = width;
	int occupiedMinY = height;
	int occupiedMaxX = -1;
	int occupiedMaxY = -1;
	uint32_t materialVotes[8] = { 0 };
	b2BlastMaterialId materialVoteIds[8] = { 0 };
	for ( int y = 0; y < height; ++y )
	{
		for ( int x = 0; x < width; ++x )
		{
			if ( b2PixelAsset_IsOccupied( asset, x, y ) == false )
			{
				continue;
			}
			int cell = y * width + x;
			actor->cellScratch[occupiedCount++] = cell;
			occupiedMinX = b2MinInt( occupiedMinX, x );
			occupiedMinY = b2MinInt( occupiedMinY, y );
			occupiedMaxX = b2MaxInt( occupiedMaxX, x );
			occupiedMaxY = b2MaxInt( occupiedMaxY, y );
			b2BlastMaterialId materialId = b2PixelAsset_GetMaterialId( asset, x, y );
			bool voted = false;
			for ( int vote = 0; vote < 8; ++vote )
			{
				if ( materialVotes[vote] > 0 && materialVoteIds[vote] == materialId )
				{
					materialVotes[vote] += 1;
					voted = true;
					break;
				}
			}
			if ( voted == false )
			{
				for ( int vote = 0; vote < 8; ++vote )
				{
					if ( materialVotes[vote] == 0 )
					{
						materialVoteIds[vote] = materialId;
						materialVotes[vote] = 1;
						break;
					}
				}
			}
		}
	}

	if ( occupiedCount == 0 )
	{
		return true;
	}

	b2BlastMaterialId dominantMaterial = 0;
	uint32_t bestMaterialVotes = 0;
	for ( int vote = 0; vote < 8; ++vote )
	{
		if ( materialVotes[vote] > bestMaterialVotes )
		{
			bestMaterialVotes = materialVotes[vote];
			dominantMaterial = materialVoteIds[vote];
		}
	}

	const b2BlastAuthoringParams params = b2BlastAuthoringParamsFromAsset( asset, dominantMaterial );
	const float targetLeafCellArea = b2BlastAuthoringTargetLeafCellArea( &params );
	const int areaTargetSeeds = b2MaxInt( 1, (int)ceilf( (float)occupiedCount / targetLeafCellArea ) );
	const float targetLeafCellEdge = b2BlastAuthoringTargetLevelCellEdge( 0 );
	const int occupiedSpanX = b2MaxInt( 1, occupiedMaxX - occupiedMinX + 1 );
	const int occupiedSpanY = b2MaxInt( 1, occupiedMaxY - occupiedMinY + 1 );
	const int spanTargetSeeds = b2MaxInt( 1, (int)ceilf( (float)occupiedSpanX / targetLeafCellEdge ) ) *
								b2MaxInt( 1, (int)ceilf( (float)occupiedSpanY / targetLeafCellEdge ) );
	const int targetSeeds = b2ClampInt( b2MaxInt( areaTargetSeeds, spanTargetSeeds ), 1, maxLeaves );
	const float cellArea = asset->pixelSize * asset->pixelSize;

	int seedCount = 0;
	int attempts = 0;
	const int seedBase = (int)params.seed;
	while ( seedCount < targetSeeds && attempts < targetSeeds * 60 )
	{
		++attempts;
		int pick = (int)floorf( b2BlastRand01( seedBase + attempts, 7, 11 ) * (float)occupiedCount ) % occupiedCount;
		int cell = actor->cellScratch[pick];
		float sx = (float)( cell % width ) + 0.5f;
		float sy = (float)( cell / width ) + 0.5f;
		float minD = FLT_MAX;
		for ( int i = 0; i < seedCount; ++i )
		{
			float dx = sx - actor->seedScratch[i].x;
			float dy = sy - actor->seedScratch[i].y;
			minD = b2MinFloat( minD, dx * dx + dy * dy );
		}
		float desired = sqrtf( targetLeafCellArea );
		if ( seedCount < 8 || minD > desired * desired * b2BlastRandRange( attempts, 0.55f, 1.2f ) )
		{
			actor->seedScratch[seedCount] = (b2BlastSeed){ sx, sy, b2BlastRandRange( attempts, 0.8f, 1.25f ), (uint32_t)seedCount };
			++seedCount;
		}
	}
	while ( seedCount < targetSeeds )
	{
		int cell = actor->cellScratch[seedCount % occupiedCount];
		actor->seedScratch[seedCount] =
			(b2BlastSeed){ (float)( cell % width ) + 0.5f, (float)( cell / width ) + 0.5f, 1.0f, (uint32_t)seedCount };
		++seedCount;
	}

	const float ca = cosf( params.fiberAngleRadians );
	const float sa = sinf( params.fiberAngleRadians );
	for ( int ci = 0; ci < occupiedCount; ++ci )
	{
		int cell = actor->cellScratch[ci];
		int x = cell % width;
		int y = cell / width;
		int best = -1;
		float bestD = FLT_MAX;
		for ( int si = 0; si < seedCount; ++si )
		{
			const b2BlastSeed* seed = actor->seedScratch + si;
			float dx = (float)x + 0.5f - seed->x;
			float dy = (float)y + 0.5f - seed->y;
			float d = 0.0f;
			if ( params.anisotropicGrain || params.anisotropy >= 0.75f )
			{
				float par = dx * ca + dy * sa;
				float perp = -dx * sa + dy * ca;
				d = par * par * params.grainParallelScale + perp * perp * params.grainPerpScale;
			}
			else if ( params.radialSeeds )
			{
				float cx = (float)width * 0.5f;
				float cy = (float)height * 0.5f;
				float r0x = (float)x + 0.5f - cx;
				float r0y = (float)y + 0.5f - cy;
				float rsx = seed->x - cx;
				float rsy = seed->y - cy;
				float dr = sqrtf( r0x * r0x + r0y * r0y ) - sqrtf( rsx * rsx + rsy * rsy );
				d = dx * dx + dy * dy + 0.2f * dr * dr;
			}
			else
			{
				d = dx * dx + dy * dy;
			}
			d *= seed->weight;
			d += ( b2BlastRand01( x + seedBase, y, (int)seed->id ) - 0.5f ) * params.grainNoise * 1.6f;
			if ( d < bestD )
			{
				bestD = d;
				best = (int)seed->id;
			}
		}
		actor->assignScratch[cell] = best;
	}

	memset( actor->cellVisitScratch, 0, (size_t)cellCount * sizeof( uint8_t ) );
	for ( int ci = 0; ci < occupiedCount; ++ci )
	{
		const int start = actor->cellScratch[ci];
		if ( actor->cellVisitScratch[start] != 0 )
		{
			continue;
		}

		const int assignment = actor->assignScratch[start];
		b2BlastLeaf leaf = { 0 };
		leaf.firstCell = (uint32_t)start;
		leaf.dominantMaterialId = 0;
		leaf.minX = UINT16_MAX;
		leaf.minY = UINT16_MAX;
		leaf.maxX = 0;
		leaf.maxY = 0;
		uint32_t leafMaterialVotes[8] = { 0 };
		b2BlastMaterialId leafMaterialVoteIds[8] = { 0 };
		float cx = 0.0f;
		float cy = 0.0f;
		float mass = 0.0f;
		bool leafHasExplicitSupport = false;
		int head = 0;
		int tail = 0;
		actor->queueScratch[tail++] = start;
		actor->cellVisitScratch[start] = 1;
		while ( head < tail )
		{
			int cell = actor->queueScratch[head++];
			int x = cell % width;
			int y = cell / width;
			b2BlastMaterialId materialId = b2PixelAsset_GetMaterialId( asset, x, y );
			float cellMass = b2BlastMaterialDensity( asset, materialId, shape->density ) * cellArea;
			b2Vec2 center = b2BlastCellCenter( asset, shape, x, y );
			cx += center.x * cellMass;
			cy += center.y * cellMass;
			mass += cellMass;
			leaf.cellCount += 1;
			leaf.minX = (uint16_t)b2MinInt( leaf.minX, x );
			leaf.minY = (uint16_t)b2MinInt( leaf.minY, y );
			leaf.maxX = (uint16_t)b2MaxInt( leaf.maxX, x );
			leaf.maxY = (uint16_t)b2MaxInt( leaf.maxY, y );
			if ( asset->supportMask != NULL && asset->supportMaskCount >= width * height && asset->supportMask[cell] != 0 )
			{
				leafHasExplicitSupport = true;
			}
			actor->cellToLeaf[cell] = (uint32_t)actor->leafCount;

			bool voted = false;
			for ( int vote = 0; vote < 8; ++vote )
			{
				if ( leafMaterialVotes[vote] > 0 && leafMaterialVoteIds[vote] == materialId )
				{
					leafMaterialVotes[vote] += 1;
					voted = true;
					break;
				}
			}
			if ( voted == false )
			{
				for ( int vote = 0; vote < 8; ++vote )
				{
					if ( leafMaterialVotes[vote] == 0 )
					{
						leafMaterialVoteIds[vote] = materialId;
						leafMaterialVotes[vote] = 1;
						break;
					}
				}
			}

			const int neighbors[4] = { cell + 1, cell - 1, cell + width, cell - width };
			for ( int ni = 0; ni < 4; ++ni )
			{
				int next = neighbors[ni];
				if ( next < 0 || next >= cellCount )
				{
					continue;
				}
				int nx = next % width;
				int ny = next / width;
				if ( b2AbsInt( nx - x ) + b2AbsInt( ny - y ) != 1 )
				{
					continue;
				}
				if ( actor->cellVisitScratch[next] != 0 || actor->assignScratch[next] != assignment ||
					 b2PixelAsset_IsOccupied( asset, nx, ny ) == false )
				{
					continue;
				}
				actor->cellVisitScratch[next] = 1;
				actor->queueScratch[tail++] = next;
			}
		}

		uint32_t bestVotes = 0;
		for ( int vote = 0; vote < 8; ++vote )
		{
			if ( leafMaterialVotes[vote] > bestVotes )
			{
				bestVotes = leafMaterialVotes[vote];
				leaf.dominantMaterialId = leafMaterialVoteIds[vote];
			}
		}
		leaf.mass = mass;
		leaf.centroid = mass > 0.0f ? (b2Vec2){ cx / mass, cy / mass } : b2Vec2_zero;
		bool hasWorldAnchor = false;
		if ( actor->mobility == b2_blastActorMobilityAnchored )
		{
			if ( asset->supportMask != NULL && asset->supportMaskCount >= width * height )
			{
				hasWorldAnchor = leafHasExplicitSupport;
			}
		}
		if ( hasWorldAnchor )
		{
			leaf.flags |= b2_blastLeafFlagWorldAnchor;
			leaf.supportConstraintMask = 1u;
			actor->flags |= b2_blastActorFlagOwnsWorldAnchor;
			actor->worldAnchorCount += 1;
		}
		actor->leaves[actor->leafCount++] = leaf;
	}

	for ( int y = 0; y < height; ++y )
	{
		for ( int x = 0; x < width; ++x )
		{
			const int indexA = y * width + x;
			if ( actor->cellToLeaf[indexA] == UINT32_MAX )
			{
				continue;
			}

			const int nx[2] = { x + 1, x };
			const int ny[2] = { y, y + 1 };
			for ( int dir = 0; dir < 2; ++dir )
			{
				if ( nx[dir] >= width || ny[dir] >= height )
				{
					continue;
				}
				const int indexB = ny[dir] * width + nx[dir];
				uint32_t leafA = actor->cellToLeaf[indexA];
				uint32_t leafB = actor->cellToLeaf[indexB];
				if ( leafB == UINT32_MAX || leafA == leafB )
				{
					continue;
				}
				if ( leafB < leafA )
				{
					uint32_t tmp = leafA;
					leafA = leafB;
					leafB = tmp;
				}
				b2BlastActiveBond* bond = NULL;
				for ( int i = 0; i < actor->bondCount; ++i )
				{
					if ( actor->bonds[i].leafA == leafA && actor->bonds[i].leafB == leafB )
					{
						bond = actor->bonds + i;
						break;
					}
				}
				if ( bond == NULL )
				{
					if ( actor->bondCount == actor->bondCapacity )
					{
						const int oldCapacity = actor->bondCapacity;
						actor->bondCapacity = actor->bondCapacity + b2MaxInt( 16, actor->bondCapacity >> 1 );
						actor->bonds =
							b2BlastResize( actor->bonds, oldCapacity, actor->bondCapacity, (int)sizeof( b2BlastActiveBond ) );
						if ( actor->bonds == NULL )
						{
							return false;
						}
					}
					if ( actor->bondCount == actor->bondCapacity )
					{
						continue;
					}
					bond = actor->bonds + actor->bondCount++;
					*bond = (b2BlastActiveBond){ 0 };
					bond->leafA = leafA;
					bond->leafB = leafB;
					bond->clusterA = leafA;
					bond->clusterB = leafB;
					bond->materialMix = actor->leaves[leafA].dominantMaterialId == actor->leaves[leafB].dominantMaterialId ? 0 : 1;
				}
				bond->area += asset->pixelSize;
			}
		}
	}

	for ( int i = 0; i < actor->bondCount; ++i )
	{
		b2BlastActiveBond* bond = actor->bonds + i;
		b2BlastMaterialId materialA = actor->leaves[bond->leafA].dominantMaterialId;
		b2BlastMaterialId materialB = actor->leaves[bond->leafB].dominantMaterialId;
		bond->capacity = b2BlastBondCapacity( asset, materialA, materialB, bond->area );
		bond->toughness = b2MaxFloat( 1.0f, 0.45f * bond->capacity );
		float shaderWeak = 1.0f;
		if ( params.anisotropicGrain || params.anisotropy >= 0.75f )
		{
			const b2Vec2 a = actor->leaves[bond->leafA].centroid;
			const b2Vec2 b = actor->leaves[bond->leafB].centroid;
			b2Vec2 axis = b2Sub( b, a );
			float len = b2Length( axis );
			float align = len > 0.0001f ? b2AbsFloat( axis.x / len ) : 0.0f;
			shaderWeak *= 1.25f + ( 0.75f - 1.25f ) * align;
		}
		if ( params.masonryMortar )
		{
			const b2Vec2 a = actor->leaves[bond->leafA].centroid;
			const b2Vec2 b = actor->leaves[bond->leafB].centroid;
			b2Vec2 mid = b2MulSV( 0.5f, b2Add( a, b ) );
			float row = floorf( mid.y / 6.0f );
			if ( b2AbsFloat( fmodf( mid.y - 28.0f, 6.0f ) ) < 0.8f ||
				 b2AbsFloat( fmodf( mid.x - 14.0f - fmodf( row, 2.0f ) * 6.0f, 12.0f ) ) < 0.8f )
			{
				shaderWeak *= params.seamStrengthScale;
			}
		}
		const float noise = 1.0f + ( b2BlastRand01( (int)bond->leafA, (int)bond->leafB, 37 ) - 0.5f ) * params.grainNoise * 0.8f;
		bond->capacity = b2MaxFloat( 0.05f, bond->capacity * noise * shaderWeak );
		bond->toughness *= 0.85f + 0.3f * b2BlastRand01( (int)bond->leafA, (int)bond->leafB, 59 );
		bond->propagationWeight = 1.0f / b2MaxFloat( 1.0f, bond->area );
		bond->damage = b2ClampFloat( bond->damage, 0.0f, 1.0f );
	}

	for ( int leafIndex = 0; leafIndex < actor->leafCount; ++leafIndex )
	{
		b2BlastLeaf* leaf = actor->leaves + leafIndex;
		leaf->firstBond = UINT32_MAX;
		leaf->bondCount = 0;
	}
	for ( int bondIndex = 0; bondIndex < actor->bondCount; ++bondIndex )
	{
		b2BlastActiveBond* bond = actor->bonds + bondIndex;
		b2BlastLeaf* a = actor->leaves + bond->leafA;
		b2BlastLeaf* b = actor->leaves + bond->leafB;
		a->firstBond = a->firstBond == UINT32_MAX ? (uint32_t)bondIndex : b2MinInt( (int)a->firstBond, bondIndex );
		b->firstBond = b->firstBond == UINT32_MAX ? (uint32_t)bondIndex : b2MinInt( (int)b->firstBond, bondIndex );
		a->bondCount += 1;
		b->bondCount += 1;
	}

	for ( int leafIndex = 0; leafIndex < actor->leafCount; ++leafIndex )
	{
		b2BlastLeaf* leaf = actor->leaves + leafIndex;
		b2BlastCluster cluster = { 0 };
		cluster.id = (uint32_t)actor->clusterCount;
		cluster.parent = UINT32_MAX;
		cluster.firstChild = UINT32_MAX;
		cluster.firstLeaf = (uint32_t)actor->clusterLeafRefCount;
		cluster.leafCount = 1;
		cluster.level = 0;
		cluster.flags = ( leaf->flags & b2_blastLeafFlagWorldAnchor ) != 0 ? b2_blastClusterFlagWorldAnchor : 0;
		cluster.minX = leaf->minX;
		cluster.minY = leaf->minY;
		cluster.maxX = leaf->maxX;
		cluster.maxY = leaf->maxY;
		cluster.centroid = leaf->centroid;
		cluster.mass = leaf->mass;
		actor->clusterLeaves[actor->clusterLeafRefCount++] = (uint32_t)leafIndex;
		actor->clusters[actor->clusterCount++] = cluster;
	}

	int currentStart = 0;
	int currentCount = actor->leafCount;
	int level = 1;
	while ( currentCount > 1 && level <= params.maxClusterLevel && actor->clusterCount < actor->clusterCapacity )
	{
		const float targetCellArea = b2BlastAuthoringTargetLevelCellArea( level );
		const float targetCellEdge = b2BlastAuthoringTargetLevelCellEdge( level );
		int newStart = actor->clusterCount;
		int newCount = 0;
		int assignedCount = 0;
		for ( int i = 0; i < actor->leafCount; ++i )
		{
			actor->leafRemapScratch[i] = UINT32_MAX;
		}
		for ( int ci = 0; ci < currentCount; ++ci )
		{
			const b2BlastCluster* cluster = actor->clusters + currentStart + ci;
			actor->componentScratch[ci] = -1;
			actor->visitScratch[ci] = 0;
			for ( uint32_t li = 0; li < cluster->leafCount; ++li )
			{
				const uint32_t leafIndex = actor->clusterLeaves[cluster->firstLeaf + li];
				if ( leafIndex < (uint32_t)actor->leafCount )
				{
					actor->leafRemapScratch[leafIndex] = (uint32_t)ci;
				}
			}
		}
		while ( assignedCount < currentCount && actor->clusterCount < actor->clusterCapacity )
		{
			b2BlastCluster parent = { 0 };
			parent.id = (uint32_t)actor->clusterCount;
			parent.parent = UINT32_MAX;
			parent.firstChild = (uint32_t)actor->clusterChildCount;
			parent.childCount = 0;
			parent.firstLeaf = (uint32_t)actor->clusterLeafRefCount;
			parent.level = (uint16_t)level;
			parent.minX = UINT16_MAX;
			parent.minY = UINT16_MAX;
			int seedCluster = B2_NULL_INDEX;
			for ( int ci = 0; ci < currentCount; ++ci )
			{
				if ( actor->visitScratch[ci] == 0 )
				{
					seedCluster = ci;
					break;
				}
			}
			if ( seedCluster == B2_NULL_INDEX )
			{
				break;
			}
			if ( b2BlastAppendClusterChildToParent( actor, currentStart, seedCluster, &parent ) == false )
			{
				break;
			}
			actor->visitScratch[seedCluster] = 1;
			actor->componentScratch[seedCluster] = newCount;
			actor->queueScratch[0] = seedCluster;
			assignedCount += 1;
			while ( b2BlastPendingClusterReachedTargetScale( actor, &parent, targetCellArea, targetCellEdge ) == false &&
					assignedCount < currentCount )
			{
				int bestCluster = B2_NULL_INDEX;
				float bestScore = FLT_MAX;
				const b2Vec2 groupCentroid = parent.mass > 0.0f
					? (b2Vec2){ parent.centroid.x / parent.mass, parent.centroid.y / parent.mass }
					: actor->clusters[currentStart + seedCluster].centroid;
				for ( int ci = 0; ci < currentCount; ++ci )
				{
					if ( actor->visitScratch[ci] != 0 )
					{
						continue;
					}
					const bool adjacent = b2BlastClusterIsAdjacentToGroup( actor, (uint32_t)ci, newCount );
					if ( adjacent == false )
					{
						continue;
					}
					const b2BlastCluster* candidate = actor->clusters + currentStart + ci;
					const b2Vec2 delta = b2Sub( candidate->centroid, groupCentroid );
					const float score = b2Dot( delta, delta );
					if ( score < bestScore )
					{
						bestScore = score;
						bestCluster = ci;
					}
				}
				if ( bestCluster == B2_NULL_INDEX )
				{
					break;
				}
				if ( b2BlastAppendClusterChildToParent( actor, currentStart, bestCluster, &parent ) == false )
				{
					break;
				}
				actor->visitScratch[bestCluster] = 1;
				actor->componentScratch[bestCluster] = newCount;
				actor->queueScratch[parent.childCount - 1] = bestCluster;
				assignedCount += 1;
			}
			if ( parent.mass > 0.0f )
			{
				parent.centroid.x /= parent.mass;
				parent.centroid.y /= parent.mass;
			}
			actor->clusters[actor->clusterCount++] = parent;
			newCount += 1;
		}
		currentStart = newStart;
		currentCount = newCount;
		level += 1;
	}

	if ( currentCount > 1 && actor->clusterCount < actor->clusterCapacity )
	{
		b2BlastCluster root = { 0 };
		root.id = (uint32_t)actor->clusterCount;
		root.parent = UINT32_MAX;
		root.firstChild = (uint32_t)actor->clusterChildCount;
		root.childCount = (uint32_t)currentCount;
		root.firstLeaf = (uint32_t)actor->clusterLeafRefCount;
		root.level = (uint16_t)level;
		root.minX = UINT16_MAX;
		root.minY = UINT16_MAX;
		for ( int i = 0; i < currentCount; ++i )
		{
			b2BlastCluster* child = actor->clusters + currentStart + i;
			child->parent = root.id;
			actor->clusterChildren[actor->clusterChildCount++] = child->id;
			root.mass += child->mass;
			root.centroid.x += child->centroid.x * child->mass;
			root.centroid.y += child->centroid.y * child->mass;
			root.minX = (uint16_t)b2MinInt( root.minX, child->minX );
			root.minY = (uint16_t)b2MinInt( root.minY, child->minY );
			root.maxX = (uint16_t)b2MaxInt( root.maxX, child->maxX );
			root.maxY = (uint16_t)b2MaxInt( root.maxY, child->maxY );
			root.flags |= child->flags;
			for ( uint32_t li = 0; li < child->leafCount; ++li )
			{
				actor->clusterLeaves[actor->clusterLeafRefCount++] = actor->clusterLeaves[child->firstLeaf + li];
				root.leafCount += 1;
			}
		}
		if ( root.mass > 0.0f )
		{
			root.centroid.x /= root.mass;
			root.centroid.y /= root.mass;
		}
		actor->rootCluster = root.id;
		actor->clusters[actor->clusterCount++] = root;
	}
	else if ( currentCount == 1 )
	{
		actor->rootCluster = actor->clusters[currentStart].id;
	}

	uint16_t maxLevel = 0;
	uint16_t maxNonRootLevel = 0;
	for ( int i = 0; i < actor->clusterCount; ++i )
	{
		const b2BlastCluster* cluster = actor->clusters + i;
		maxLevel = (uint16_t)b2MaxInt( maxLevel, cluster->level );
		if ( cluster->id != actor->rootCluster )
		{
			maxNonRootLevel = (uint16_t)b2MaxInt( maxNonRootLevel, cluster->level );
		}
	}
	const uint16_t highestRuntimeLevel = maxNonRootLevel > 0 ? maxNonRootLevel : maxLevel;
	actor->initialActiveLevel = (uint16_t)b2ClampInt( params.startLevel, 0, highestRuntimeLevel );
	b2BlastActor_ResetActiveClustersByArea( actor, &params );

	actor->topologyVersion = asset->topologyVersion;
	actor->materialHash = asset->materialHash;
	uint64_t hash = 14695981039346656037ULL;
	hash = b2BlastMixHash( hash, (uint64_t)width );
	hash = b2BlastMixHash( hash, (uint64_t)height );
	hash = b2BlastMixHash( hash, (uint64_t)actor->leafCount );
	hash = b2BlastMixHash( hash, (uint64_t)actor->bondCount );
	hash = b2BlastMixHash( hash, (uint64_t)actor->clusterCount );
	hash = b2BlastMixHash( hash, asset->materialHash );
	for ( int i = 0; i < actor->leafCount; ++i )
	{
		const b2BlastLeaf* leaf = actor->leaves + i;
		hash = b2BlastMixHash( hash, leaf->cellCount );
		hash = b2BlastMixHash( hash, leaf->dominantMaterialId );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( leaf->centroid.x ) );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( leaf->centroid.y ) );
	}
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		const b2BlastActiveBond* bond = actor->bonds + i;
		hash = b2BlastMixHash( hash, ( (uint64_t)bond->leafA << 32 ) | bond->leafB );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( bond->area ) );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( bond->capacity ) );
	}
	actor->authoringHash = hash;
	actor->revision += 1;
	actor->flags &= (uint32_t)~b2_blastActorFlagDirtyGraph;
	return true;
}

static bool b2BlastActor_RebuildClustersFromCurrentLeaves( b2BlastActor* actor )
{
	if ( actor == NULL || actor->ownedAsset.width <= 0 || actor->ownedAsset.height <= 0 )
	{
		return false;
	}

	const b2PixelAsset* asset = &actor->ownedAsset;
	b2BlastMaterialId dominantMaterial = 0;
	for ( int leafIndex = 0; leafIndex < actor->leafCount; ++leafIndex )
	{
		if ( actor->leaves[leafIndex].cellCount > 0 )
		{
			dominantMaterial = actor->leaves[leafIndex].dominantMaterialId;
			break;
		}
	}
	const b2BlastAuthoringParams params = b2BlastAuthoringParamsFromAsset( asset, dominantMaterial );
	const uint16_t preservedInitialActiveLevel = actor->initialActiveLevel;
	actor->clusterCount = 0;
	actor->clusterChildCount = 0;
	actor->clusterLeafRefCount = 0;
	actor->rootCluster = UINT32_MAX;

	for ( int leafIndex = 0; leafIndex < actor->leafCount; ++leafIndex )
	{
		const b2BlastLeaf* leaf = actor->leaves + leafIndex;
		if ( leaf->cellCount == 0 )
		{
			continue;
		}
		if ( actor->clusterCount >= actor->clusterCapacity || actor->clusterLeafRefCount >= actor->clusterLeafRefCapacity )
		{
			return false;
		}

		b2BlastCluster cluster = { 0 };
		cluster.id = (uint32_t)actor->clusterCount;
		cluster.parent = UINT32_MAX;
		cluster.firstChild = UINT32_MAX;
		cluster.firstLeaf = (uint32_t)actor->clusterLeafRefCount;
		cluster.leafCount = 1;
		cluster.level = 0;
		cluster.flags = ( leaf->flags & b2_blastLeafFlagWorldAnchor ) != 0 ? b2_blastClusterFlagWorldAnchor : 0;
		cluster.minX = leaf->minX;
		cluster.minY = leaf->minY;
		cluster.maxX = leaf->maxX;
		cluster.maxY = leaf->maxY;
		cluster.centroid = leaf->centroid;
		cluster.mass = leaf->mass;
		actor->clusterLeaves[actor->clusterLeafRefCount++] = (uint32_t)leafIndex;
		actor->clusters[actor->clusterCount++] = cluster;
	}

	if ( actor->clusterCount == 0 )
	{
		return true;
	}

	int currentStart = 0;
	int currentCount = actor->clusterCount;
	int level = 1;
	while ( currentCount > 1 && level <= params.maxClusterLevel && actor->clusterCount < actor->clusterCapacity )
	{
		const float targetCellArea = b2BlastAuthoringTargetLevelCellArea( level );
		const float targetCellEdge = b2BlastAuthoringTargetLevelCellEdge( level );
		int newStart = actor->clusterCount;
		int newCount = 0;
		int assignedCount = 0;
		for ( int i = 0; i < actor->leafCount; ++i )
		{
			actor->leafRemapScratch[i] = UINT32_MAX;
		}
		for ( int ci = 0; ci < currentCount; ++ci )
		{
			const b2BlastCluster* cluster = actor->clusters + currentStart + ci;
			actor->componentScratch[ci] = -1;
			actor->visitScratch[ci] = 0;
			for ( uint32_t li = 0; li < cluster->leafCount; ++li )
			{
				const uint32_t leafIndex = actor->clusterLeaves[cluster->firstLeaf + li];
				if ( leafIndex < (uint32_t)actor->leafCount )
				{
					actor->leafRemapScratch[leafIndex] = (uint32_t)ci;
				}
			}
		}
		while ( assignedCount < currentCount && actor->clusterCount < actor->clusterCapacity )
		{
			b2BlastCluster parent = { 0 };
			parent.id = (uint32_t)actor->clusterCount;
			parent.parent = UINT32_MAX;
			parent.firstChild = (uint32_t)actor->clusterChildCount;
			parent.childCount = 0;
			parent.firstLeaf = (uint32_t)actor->clusterLeafRefCount;
			parent.level = (uint16_t)level;
			parent.minX = UINT16_MAX;
			parent.minY = UINT16_MAX;
			int seedCluster = B2_NULL_INDEX;
			for ( int ci = 0; ci < currentCount; ++ci )
			{
				if ( actor->visitScratch[ci] == 0 )
				{
					seedCluster = ci;
					break;
				}
			}
			if ( seedCluster == B2_NULL_INDEX )
			{
				break;
			}
			if ( b2BlastAppendClusterChildToParent( actor, currentStart, seedCluster, &parent ) == false )
			{
				break;
			}
			actor->visitScratch[seedCluster] = 1;
			actor->componentScratch[seedCluster] = newCount;
			actor->queueScratch[0] = seedCluster;
			assignedCount += 1;
			while ( b2BlastPendingClusterReachedTargetScale( actor, &parent, targetCellArea, targetCellEdge ) == false &&
					assignedCount < currentCount )
			{
				int bestCluster = B2_NULL_INDEX;
				float bestScore = FLT_MAX;
				const b2Vec2 groupCentroid = parent.mass > 0.0f
					? (b2Vec2){ parent.centroid.x / parent.mass, parent.centroid.y / parent.mass }
					: actor->clusters[currentStart + seedCluster].centroid;
				for ( int ci = 0; ci < currentCount; ++ci )
				{
					if ( actor->visitScratch[ci] != 0 )
					{
						continue;
					}
					const bool adjacent = b2BlastClusterIsAdjacentToGroup( actor, (uint32_t)ci, newCount );
					if ( adjacent == false )
					{
						continue;
					}
					const b2BlastCluster* candidate = actor->clusters + currentStart + ci;
					const b2Vec2 delta = b2Sub( candidate->centroid, groupCentroid );
					const float score = b2Dot( delta, delta );
					if ( score < bestScore )
					{
						bestScore = score;
						bestCluster = ci;
					}
				}
				if ( bestCluster == B2_NULL_INDEX )
				{
					break;
				}
				if ( b2BlastAppendClusterChildToParent( actor, currentStart, bestCluster, &parent ) == false )
				{
					break;
				}
				actor->visitScratch[bestCluster] = 1;
				actor->componentScratch[bestCluster] = newCount;
				actor->queueScratch[parent.childCount - 1] = bestCluster;
				assignedCount += 1;
			}
			if ( parent.mass > 0.0f )
			{
				parent.centroid.x /= parent.mass;
				parent.centroid.y /= parent.mass;
			}
			actor->clusters[actor->clusterCount++] = parent;
			newCount += 1;
		}
		currentStart = newStart;
		currentCount = newCount;
		level += 1;
	}

	if ( currentCount > 1 && actor->clusterCount < actor->clusterCapacity )
	{
		b2BlastCluster root = { 0 };
		root.id = (uint32_t)actor->clusterCount;
		root.parent = UINT32_MAX;
		root.firstChild = (uint32_t)actor->clusterChildCount;
		root.childCount = (uint32_t)currentCount;
		root.firstLeaf = (uint32_t)actor->clusterLeafRefCount;
		root.level = (uint16_t)level;
		root.minX = UINT16_MAX;
		root.minY = UINT16_MAX;
		for ( int i = 0; i < currentCount; ++i )
		{
			b2BlastCluster* child = actor->clusters + currentStart + i;
			child->parent = root.id;
			if ( actor->clusterChildCount >= actor->clusterChildCapacity ||
				 actor->clusterLeafRefCount + (int)child->leafCount > actor->clusterLeafRefCapacity )
			{
				return false;
			}
			actor->clusterChildren[actor->clusterChildCount++] = child->id;
			root.mass += child->mass;
			root.centroid.x += child->centroid.x * child->mass;
			root.centroid.y += child->centroid.y * child->mass;
			root.minX = (uint16_t)b2MinInt( root.minX, child->minX );
			root.minY = (uint16_t)b2MinInt( root.minY, child->minY );
			root.maxX = (uint16_t)b2MaxInt( root.maxX, child->maxX );
			root.maxY = (uint16_t)b2MaxInt( root.maxY, child->maxY );
			root.flags |= child->flags;
			for ( uint32_t li = 0; li < child->leafCount; ++li )
			{
				actor->clusterLeaves[actor->clusterLeafRefCount++] = actor->clusterLeaves[child->firstLeaf + li];
				root.leafCount += 1;
			}
		}
		if ( root.mass > 0.0f )
		{
			root.centroid.x /= root.mass;
			root.centroid.y /= root.mass;
		}
		actor->rootCluster = root.id;
		actor->clusters[actor->clusterCount++] = root;
	}
	else if ( currentCount == 1 )
	{
		actor->rootCluster = actor->clusters[currentStart].id;
	}
	actor->initialActiveLevel = preservedInitialActiveLevel;
	return true;
}

static bool b2BlastActor_RecomputeLeavesAndBondsAfterErase( b2BlastActor* actor, b2Shape* shape )
{
	if ( actor == NULL || shape == NULL || shape->type != b2_pixelShape || b2IsPixelShapeUsable( &shape->pixel ) == false )
	{
		return false;
	}

	const b2PixelAsset* asset = shape->pixel.asset;
	const int width = asset->width;
	const int height = asset->height;
	const int cellCount = width * height;
	const float cellArea = asset->pixelSize * asset->pixelSize;
	actor->worldAnchorCount = 0;
	actor->flags &= (uint32_t)~b2_blastActorFlagOwnsWorldAnchor;
	memset( actor->visitScratch, 0, (size_t)actor->leafCapacity * sizeof( uint8_t ) );
	for ( int activeIndex = 0; activeIndex < actor->activeClusterCount; ++activeIndex )
	{
		const uint32_t activeId = actor->activeClusters[activeIndex];
		if ( activeId < (uint32_t)actor->clusterCount )
		{
			const b2BlastCluster* cluster = actor->clusters + activeId;
			if ( cluster->firstLeaf != UINT32_MAX && cluster->firstLeaf + cluster->leafCount <= (uint32_t)actor->clusterLeafRefCount )
			{
				for ( uint32_t li = 0; li < cluster->leafCount; ++li )
				{
					const uint32_t leafIndex = actor->clusterLeaves[cluster->firstLeaf + li];
					if ( leafIndex < (uint32_t)actor->leafCapacity )
					{
						actor->visitScratch[leafIndex] = 1;
					}
				}
				continue;
			}
		}
		if ( activeId < (uint32_t)actor->leafCapacity )
		{
			actor->visitScratch[activeId] = 1;
		}
	}

	const int oldLeafCount = actor->leafCount;
	for ( int leafIndex = 0; leafIndex < actor->leafCount; ++leafIndex )
	{
		b2BlastLeaf* leaf = actor->leaves + leafIndex;
		const uint16_t preservedFlags = leaf->flags & b2_blastLeafFlagDetached;
		*leaf = (b2BlastLeaf){ 0 };
		leaf->firstCell = UINT32_MAX;
		leaf->dominantMaterialId = 0;
		leaf->minX = UINT16_MAX;
		leaf->minY = UINT16_MAX;
		leaf->maxX = 0;
		leaf->maxY = 0;
		leaf->firstBond = UINT32_MAX;
		leaf->flags = preservedFlags;

		uint32_t materialVotes[8] = { 0 };
		b2BlastMaterialId materialVoteIds[8] = { 0 };
		float cx = 0.0f;
		float cy = 0.0f;
		float mass = 0.0f;
		bool leafHasExplicitSupport = false;
		for ( int cell = 0; cell < cellCount; ++cell )
		{
			if ( actor->cellToLeaf[cell] != (uint32_t)leafIndex )
			{
				continue;
			}
			const int x = cell % width;
			const int y = cell / width;
			if ( b2PixelAsset_IsOccupied( asset, x, y ) == false )
			{
				actor->cellToLeaf[cell] = UINT32_MAX;
				continue;
			}
			b2BlastMaterialId materialId = b2PixelAsset_GetMaterialId( asset, x, y );
			float cellMass = b2BlastMaterialDensity( asset, materialId, shape->density ) * cellArea;
			b2Vec2 center = b2BlastCellCenter( asset, shape, x, y );
			cx += center.x * cellMass;
			cy += center.y * cellMass;
			mass += cellMass;
			leaf->cellCount += 1;
			leaf->firstCell = leaf->firstCell == UINT32_MAX ? (uint32_t)cell : leaf->firstCell;
			leaf->minX = (uint16_t)b2MinInt( leaf->minX, x );
			leaf->minY = (uint16_t)b2MinInt( leaf->minY, y );
			leaf->maxX = (uint16_t)b2MaxInt( leaf->maxX, x );
			leaf->maxY = (uint16_t)b2MaxInt( leaf->maxY, y );
			if ( asset->supportMask != NULL && asset->supportMaskCount >= cellCount && asset->supportMask[cell] != 0 )
			{
				leafHasExplicitSupport = true;
			}

			bool voted = false;
			for ( int vote = 0; vote < 8; ++vote )
			{
				if ( materialVotes[vote] > 0 && materialVoteIds[vote] == materialId )
				{
					materialVotes[vote] += 1;
					voted = true;
					break;
				}
			}
			if ( voted == false )
			{
				for ( int vote = 0; vote < 8; ++vote )
				{
					if ( materialVotes[vote] == 0 )
					{
						materialVoteIds[vote] = materialId;
						materialVotes[vote] = 1;
						break;
					}
				}
			}
		}

		if ( leaf->cellCount == 0 )
		{
			leaf->flags |= b2_blastLeafFlagDetached;
			continue;
		}
		uint32_t bestVotes = 0;
		for ( int vote = 0; vote < 8; ++vote )
		{
			if ( materialVotes[vote] > bestVotes )
			{
				bestVotes = materialVotes[vote];
				leaf->dominantMaterialId = materialVoteIds[vote];
			}
		}
		leaf->mass = mass;
		leaf->centroid = mass > 0.0f ? (b2Vec2){ cx / mass, cy / mass } : b2Vec2_zero;
		bool hasWorldAnchor = false;
		if ( actor->mobility == b2_blastActorMobilityAnchored )
		{
			if ( asset->supportMask != NULL && asset->supportMaskCount >= cellCount )
			{
				hasWorldAnchor = leafHasExplicitSupport;
			}
		}
		if ( hasWorldAnchor )
		{
			leaf->flags |= b2_blastLeafFlagWorldAnchor;
			leaf->supportConstraintMask = 1u;
			actor->flags |= b2_blastActorFlagOwnsWorldAnchor;
			actor->worldAnchorCount += 1;
		}
	}

	for ( int leafIndex = 0; leafIndex < oldLeafCount; ++leafIndex )
	{
		actor->leafRemapScratch[leafIndex] = UINT32_MAX;
	}
	int compactLeafCount = 0;
	for ( int leafIndex = 0; leafIndex < oldLeafCount; ++leafIndex )
	{
		if ( actor->leaves[leafIndex].cellCount == 0 || ( actor->leaves[leafIndex].flags & b2_blastLeafFlagDetached ) != 0 )
		{
			continue;
		}
		actor->leafRemapScratch[leafIndex] = (uint32_t)compactLeafCount;
		if ( compactLeafCount != leafIndex )
		{
			actor->leaves[compactLeafCount] = actor->leaves[leafIndex];
		}
		compactLeafCount += 1;
	}
	for ( int cell = 0; cell < cellCount; ++cell )
	{
		const uint32_t oldLeaf = actor->cellToLeaf[cell];
		if ( oldLeaf < (uint32_t)oldLeafCount && actor->leafRemapScratch[oldLeaf] != UINT32_MAX )
		{
			actor->cellToLeaf[cell] = actor->leafRemapScratch[oldLeaf];
		}
		else
		{
			actor->cellToLeaf[cell] = UINT32_MAX;
		}
	}
	actor->activeClusterCount = 0;
	for ( int leafIndex = 0; leafIndex < oldLeafCount; ++leafIndex )
	{
		if ( actor->visitScratch[leafIndex] != 0 && actor->leafRemapScratch[leafIndex] != UINT32_MAX )
		{
			(void)b2BlastActor_AddActiveCluster( actor, actor->leafRemapScratch[leafIndex] );
		}
	}
	actor->leafCount = compactLeafCount;

	for ( int bondIndex = 0; bondIndex < actor->bondCount; ++bondIndex )
	{
		b2BlastActiveBond* bond = actor->bonds + bondIndex;
		if ( bond->leafA >= (uint32_t)oldLeafCount || bond->leafB >= (uint32_t)oldLeafCount ||
			 actor->leafRemapScratch[bond->leafA] == UINT32_MAX || actor->leafRemapScratch[bond->leafB] == UINT32_MAX )
		{
			bond->leafA = UINT32_MAX;
			bond->leafB = UINT32_MAX;
			bond->area = 0.0f;
			bond->flags |= b2_blastBondFlagBroken;
			continue;
		}
		uint32_t leafA = actor->leafRemapScratch[bond->leafA];
		uint32_t leafB = actor->leafRemapScratch[bond->leafB];
		if ( leafA == leafB )
		{
			bond->leafA = UINT32_MAX;
			bond->leafB = UINT32_MAX;
			bond->area = 0.0f;
			bond->flags |= b2_blastBondFlagBroken;
			continue;
		}
		if ( leafB < leafA )
		{
			uint32_t temp = leafA;
			leafA = leafB;
			leafB = temp;
		}
		bond->leafA = leafA;
		bond->leafB = leafB;
		bond->clusterA = leafA;
		bond->clusterB = leafB;
		bond->area = 0.0f;
	}
	int compactBondCount = 0;
	for ( int bondIndex = 0; bondIndex < actor->bondCount; ++bondIndex )
	{
		b2BlastActiveBond bond = actor->bonds[bondIndex];
		if ( bond.leafA >= (uint32_t)actor->leafCount || bond.leafB >= (uint32_t)actor->leafCount || bond.leafA == bond.leafB )
		{
			continue;
		}
		b2BlastActiveBond* existing = NULL;
		for ( int i = 0; i < compactBondCount; ++i )
		{
			if ( actor->bonds[i].leafA == bond.leafA && actor->bonds[i].leafB == bond.leafB )
			{
				existing = actor->bonds + i;
				break;
			}
		}
		if ( existing != NULL )
		{
			existing->damage = b2MaxFloat( existing->damage, bond.damage );
			existing->flags |= bond.flags;
			existing->impactDemand += bond.impactDemand;
			existing->loadDemand += bond.loadDemand;
			continue;
		}
		bond.area = 0.0f;
		bond.clusterA = bond.leafA;
		bond.clusterB = bond.leafB;
		actor->bonds[compactBondCount++] = bond;
	}
	actor->bondCount = compactBondCount;
	const b2BlastAuthoringParams params = b2BlastAuthoringParamsFromAsset( asset, actor->leafCount > 0 ? actor->leaves[0].dominantMaterialId : 0 );
	for ( int y = 0; y < height; ++y )
	{
		for ( int x = 0; x < width; ++x )
		{
			const int indexA = y * width + x;
			if ( actor->cellToLeaf[indexA] == UINT32_MAX )
			{
				continue;
			}
			const int nx[2] = { x + 1, x };
			const int ny[2] = { y, y + 1 };
			for ( int dir = 0; dir < 2; ++dir )
			{
				if ( nx[dir] >= width || ny[dir] >= height )
				{
					continue;
				}
				const int indexB = ny[dir] * width + nx[dir];
				uint32_t leafA = actor->cellToLeaf[indexA];
				uint32_t leafB = actor->cellToLeaf[indexB];
				if ( leafB == UINT32_MAX || leafA == leafB )
				{
					continue;
				}
				if ( leafB < leafA )
				{
					uint32_t tmp = leafA;
					leafA = leafB;
					leafB = tmp;
				}
				b2BlastActiveBond* bond = NULL;
				for ( int i = 0; i < actor->bondCount; ++i )
				{
					if ( actor->bonds[i].leafA == leafA && actor->bonds[i].leafB == leafB )
					{
						bond = actor->bonds + i;
						break;
					}
				}
				if ( bond == NULL )
				{
					if ( actor->bondCount >= actor->bondCapacity )
					{
						return false;
					}
					bond = actor->bonds + actor->bondCount++;
					*bond = (b2BlastActiveBond){ 0 };
					bond->leafA = leafA;
					bond->leafB = leafB;
					bond->clusterA = leafA;
					bond->clusterB = leafB;
					bond->materialMix = actor->leaves[leafA].dominantMaterialId == actor->leaves[leafB].dominantMaterialId ? 0 : 1;
				}
				bond->area += asset->pixelSize;
			}
		}
	}

	for ( int bondIndex = 0; bondIndex < actor->bondCount; ++bondIndex )
	{
		b2BlastActiveBond* bond = actor->bonds + bondIndex;
		if ( bond->leafA >= (uint32_t)actor->leafCount || bond->leafB >= (uint32_t)actor->leafCount ||
			 actor->leaves[bond->leafA].cellCount == 0 || actor->leaves[bond->leafB].cellCount == 0 || bond->area <= 0.0f )
		{
			bond->flags |= b2_blastBondFlagBroken;
			continue;
		}
		b2BlastMaterialId materialA = actor->leaves[bond->leafA].dominantMaterialId;
		b2BlastMaterialId materialB = actor->leaves[bond->leafB].dominantMaterialId;
		bond->capacity = b2BlastBondCapacity( asset, materialA, materialB, bond->area );
		bond->toughness = b2MaxFloat( 1.0f, 0.45f * bond->capacity );
		float shaderWeak = 1.0f;
		if ( params.anisotropicGrain || params.anisotropy >= 0.75f )
		{
			const b2Vec2 a = actor->leaves[bond->leafA].centroid;
			const b2Vec2 b = actor->leaves[bond->leafB].centroid;
			b2Vec2 axis = b2Sub( b, a );
			float len = b2Length( axis );
			float align = len > 0.0001f ? b2AbsFloat( axis.x / len ) : 0.0f;
			shaderWeak *= 1.25f + ( 0.75f - 1.25f ) * align;
		}
		if ( params.masonryMortar )
		{
			const b2Vec2 a = actor->leaves[bond->leafA].centroid;
			const b2Vec2 b = actor->leaves[bond->leafB].centroid;
			b2Vec2 mid = b2MulSV( 0.5f, b2Add( a, b ) );
			float row = floorf( mid.y / 6.0f );
			if ( b2AbsFloat( fmodf( mid.y - 28.0f, 6.0f ) ) < 0.8f ||
				 b2AbsFloat( fmodf( mid.x - 14.0f - fmodf( row, 2.0f ) * 6.0f, 12.0f ) ) < 0.8f )
			{
				shaderWeak *= params.seamStrengthScale;
			}
		}
		const float noise = 1.0f + ( b2BlastRand01( (int)bond->leafA, (int)bond->leafB, 37 ) - 0.5f ) * params.grainNoise * 0.8f;
		bond->capacity = b2MaxFloat( 0.05f, bond->capacity * noise * shaderWeak );
		bond->toughness *= 0.85f + 0.3f * b2BlastRand01( (int)bond->leafA, (int)bond->leafB, 59 );
		bond->propagationWeight = 1.0f / b2MaxFloat( 1.0f, bond->area );
		bond->damage = b2ClampFloat( bond->damage, 0.0f, 1.0f );
	}

	for ( int leafIndex = 0; leafIndex < actor->leafCount; ++leafIndex )
	{
		actor->leaves[leafIndex].firstBond = UINT32_MAX;
		actor->leaves[leafIndex].bondCount = 0;
	}
	for ( int bondIndex = 0; bondIndex < actor->bondCount; ++bondIndex )
	{
		b2BlastActiveBond* bond = actor->bonds + bondIndex;
		if ( bond->area <= 0.0f || bond->leafA >= (uint32_t)actor->leafCount || bond->leafB >= (uint32_t)actor->leafCount )
		{
			continue;
		}
		b2BlastLeaf* a = actor->leaves + bond->leafA;
		b2BlastLeaf* b = actor->leaves + bond->leafB;
		a->firstBond = a->firstBond == UINT32_MAX ? (uint32_t)bondIndex : b2MinInt( (int)a->firstBond, bondIndex );
		b->firstBond = b->firstBond == UINT32_MAX ? (uint32_t)bondIndex : b2MinInt( (int)b->firstBond, bondIndex );
		a->bondCount += 1;
		b->bondCount += 1;
	}

	if ( b2BlastActor_RebuildClustersFromCurrentLeaves( actor ) == false )
	{
		return false;
	}
	b2BlastActor_PruneActiveClusters( actor );
	if ( actor->activeClusterCount == 0 && actor->leafCount > 0 )
	{
		const b2BlastAuthoringParams params = b2BlastAuthoringParamsFromActor( actor );
		b2BlastActor_ResetActiveClustersByArea( actor, &params );
	}
	return true;
}

static bool b2BlastActor_RepairFromPixelShapeDirtyUpdate( b2BlastActor* actor, b2Shape* shape )
{
	if ( actor == NULL || shape == NULL || shape->type != b2_pixelShape || b2IsPixelShapeUsable( &shape->pixel ) == false ||
		 actor->cellToLeafCount <= 0 || actor->leafCount <= 0 )
	{
		return false;
	}

	const b2PixelAsset* asset = shape->pixel.asset;
	const int oldWidth = actor->ownedAsset.width;
	const int oldHeight = actor->ownedAsset.height;
	const int oldCellCount = actor->cellToLeafCount;
	const int newCellCount = asset->width * asset->height;
	if ( oldWidth != asset->width || oldHeight != asset->height || oldCellCount != newCellCount ||
		 shape->pixel.dirtyWidth <= 0 || shape->pixel.dirtyHeight <= 0 )
	{
		return false;
	}

	const int maxLeaves = b2MaxInt( 1, asset->solidCount );
	const int maxBonds = b2MaxInt( 1, asset->solidCount * 2 );
	const int maxClusters = b2MaxInt( maxLeaves + 1, maxLeaves * ( b2BlastDefaultAuthoringParams().maxClusterLevel + 2 ) + 8 );
	const int maxClusterRefs = b2MaxInt( maxClusters * 4, maxLeaves * 8 + 8 );
	b2BlastEnsureActorCapacity( actor, maxLeaves, maxBonds, maxClusters, maxClusterRefs, newCellCount, maxLeaves );
	if ( actor->leafCapacity < maxLeaves || actor->bondCapacity < maxBonds || actor->cellVisitScratchCapacity < newCellCount )
	{
		return false;
	}

	const int dirtyX = b2ClampInt( shape->pixel.dirtyX, 0, oldWidth );
	const int dirtyY = b2ClampInt( shape->pixel.dirtyY, 0, oldHeight );
	const int dirtyWidth = b2ClampInt( shape->pixel.dirtyWidth, 0, oldWidth - dirtyX );
	const int dirtyHeight = b2ClampInt( shape->pixel.dirtyHeight, 0, oldHeight - dirtyY );
	if ( dirtyWidth <= 0 || dirtyHeight <= 0 )
	{
		return false;
	}

	const int oldLeafCount = actor->leafCount;
	memset( actor->visitScratch, 0, (size_t)actor->leafCapacity * sizeof( uint8_t ) );
	int addedCount = 0;
	bool changedAny = false;
	for ( int y = dirtyY; y < dirtyY + dirtyHeight; ++y )
	{
		for ( int x = dirtyX; x < dirtyX + dirtyWidth; ++x )
		{
			const int cell = y * oldWidth + x;
			const uint32_t oldLeaf = actor->cellToLeaf[cell];
			const bool oldOccupied = oldLeaf != UINT32_MAX && oldLeaf < (uint32_t)oldLeafCount;
			const bool newOccupied = b2PixelAsset_IsOccupied( asset, x, y );
			bool cellChanged = false;
			if ( newOccupied && oldOccupied == false )
			{
				if ( addedCount >= actor->cellScratchCapacity )
				{
					return false;
				}
				actor->cellScratch[addedCount++] = cell;
				changedAny = true;
				cellChanged = true;
			}
			if ( newOccupied && oldOccupied )
			{
				const b2BlastMaterialId oldMaterial = b2PixelAsset_GetMaterialId( &actor->ownedAsset, x, y );
				const b2BlastMaterialId newMaterial = b2PixelAsset_GetMaterialId( asset, x, y );
				if ( oldMaterial != newMaterial )
				{
					actor->visitScratch[oldLeaf] = 1;
					changedAny = true;
					cellChanged = true;
				}
				const bool oldSupport = b2BlastPixelAsset_CellHasSupport( &actor->ownedAsset, cell, oldCellCount );
				const bool newSupport = b2BlastPixelAsset_CellHasSupport( asset, cell, newCellCount );
				if ( oldSupport != newSupport )
				{
					actor->visitScratch[oldLeaf] = 1;
					changedAny = true;
					cellChanged = true;
				}
			}
			if ( oldOccupied && newOccupied == false )
			{
				actor->cellToLeaf[cell] = UINT32_MAX;
				actor->visitScratch[oldLeaf] = 1;
				changedAny = true;
				cellChanged = true;
			}
			if ( cellChanged == false )
			{
				continue;
			}
			if ( oldOccupied == false && newOccupied == false )
			{
				continue;
			}
			const int neighborCells[4] = { cell + 1, cell - 1, cell + oldWidth, cell - oldWidth };
			for ( int ni = 0; ni < 4; ++ni )
			{
				const int next = neighborCells[ni];
				if ( next < 0 || next >= newCellCount )
				{
					continue;
				}
				const int nx = next % oldWidth;
				const int ny = next / oldWidth;
				if ( b2AbsInt( nx - x ) + b2AbsInt( ny - y ) != 1 )
				{
					continue;
				}
				const uint32_t neighborLeaf = actor->cellToLeaf[next];
				if ( neighborLeaf < (uint32_t)oldLeafCount && actor->visitScratch[neighborLeaf] == 0 )
				{
					actor->visitScratch[neighborLeaf] = 2;
				}
			}
		}
	}
	if ( changedAny == false )
	{
		actor->ownedAsset = *asset;
		actor->topologyVersion = asset->topologyVersion;
		actor->materialHash = asset->materialHash;
		actor->revision += 1;
		actor->flags &= (uint32_t)~b2_blastActorFlagDirtyGraph;
		return true;
	}

	for ( int leafIndex = 0; leafIndex < oldLeafCount; ++leafIndex )
	{
		if ( actor->visitScratch[leafIndex] != 0 )
		{
			b2BlastActor_RefineActivePathToLeaf( actor, (uint32_t)leafIndex );
		}
	}

	memset( actor->cellVisitScratch, 0, (size_t)newCellCount * sizeof( uint8_t ) );
	for ( int leafIndex = 0; leafIndex < oldLeafCount; ++leafIndex )
	{
		if ( actor->visitScratch[leafIndex] != 1 )
		{
			continue;
		}

		bool keptFirstComponent = false;
		for ( int start = 0; start < newCellCount; ++start )
		{
			if ( actor->cellToLeaf[start] != (uint32_t)leafIndex || actor->cellVisitScratch[start] != 0 )
			{
				continue;
			}
			const uint32_t targetLeaf = keptFirstComponent ? (uint32_t)actor->leafCount++ : (uint32_t)leafIndex;
			if ( targetLeaf >= (uint32_t)actor->leafCapacity )
			{
				return false;
			}
			if ( keptFirstComponent )
			{
				actor->leaves[targetLeaf] = actor->leaves[leafIndex];
				actor->leaves[targetLeaf].flags &= (uint16_t)~b2_blastLeafFlagDetached;
				(void)b2BlastActor_AddActiveCluster( actor, targetLeaf );
			}
			keptFirstComponent = true;

			int head = 0;
			int tail = 0;
			actor->queueScratch[tail++] = start;
			actor->cellVisitScratch[start] = 1;
			actor->cellToLeaf[start] = targetLeaf;
			while ( head < tail )
			{
				const int cell = actor->queueScratch[head++];
				const int x = cell % oldWidth;
				const int y = cell / oldWidth;
				const int neighbors[4] = { cell + 1, cell - 1, cell + oldWidth, cell - oldWidth };
				for ( int ni = 0; ni < 4; ++ni )
				{
					const int next = neighbors[ni];
					if ( next < 0 || next >= newCellCount )
					{
						continue;
					}
					const int nx = next % oldWidth;
					const int ny = next / oldWidth;
					if ( b2AbsInt( nx - x ) + b2AbsInt( ny - y ) != 1 )
					{
						continue;
					}
					if ( actor->cellToLeaf[next] != (uint32_t)leafIndex || actor->cellVisitScratch[next] != 0 ||
						 b2PixelAsset_IsOccupied( asset, nx, ny ) == false )
					{
						continue;
					}
					actor->cellVisitScratch[next] = 1;
					actor->cellToLeaf[next] = targetLeaf;
					actor->queueScratch[tail++] = next;
				}
			}
		}
	}

	for ( int addedIndex = 0; addedIndex < addedCount; ++addedIndex )
	{
		const int start = actor->cellScratch[addedIndex];
		if ( start < 0 || start >= newCellCount || actor->cellVisitScratch[start] != 0 )
		{
			continue;
		}
		const int sx = start % oldWidth;
		const int sy = start / oldWidth;
		if ( b2PixelAsset_IsOccupied( asset, sx, sy ) == false || actor->cellToLeaf[start] != UINT32_MAX )
		{
			continue;
		}
		const uint32_t targetLeaf = (uint32_t)actor->leafCount++;
		if ( targetLeaf >= (uint32_t)actor->leafCapacity )
		{
			return false;
		}
		actor->leaves[targetLeaf] = (b2BlastLeaf){ 0 };
		actor->leaves[targetLeaf].firstCell = UINT32_MAX;
		actor->leaves[targetLeaf].minX = UINT16_MAX;
		actor->leaves[targetLeaf].minY = UINT16_MAX;
		(void)b2BlastActor_AddActiveCluster( actor, targetLeaf );

		int head = 0;
		int tail = 0;
		actor->queueScratch[tail++] = start;
		actor->cellVisitScratch[start] = 1;
		actor->cellToLeaf[start] = targetLeaf;
		while ( head < tail )
		{
			const int cell = actor->queueScratch[head++];
			const int x = cell % oldWidth;
			const int y = cell / oldWidth;
			const int neighbors[4] = { cell + 1, cell - 1, cell + oldWidth, cell - oldWidth };
			for ( int ni = 0; ni < 4; ++ni )
			{
				const int next = neighbors[ni];
				if ( next < 0 || next >= newCellCount )
				{
					continue;
				}
				const int nx = next % oldWidth;
				const int ny = next / oldWidth;
				if ( b2AbsInt( nx - x ) + b2AbsInt( ny - y ) != 1 )
				{
					continue;
				}
				if ( actor->cellVisitScratch[next] != 0 || actor->cellToLeaf[next] != UINT32_MAX ||
					 b2PixelAsset_IsOccupied( asset, nx, ny ) == false )
				{
					continue;
				}
				actor->cellVisitScratch[next] = 1;
				actor->cellToLeaf[next] = targetLeaf;
				actor->queueScratch[tail++] = next;
			}
		}
	}

	if ( b2BlastActor_RecomputeLeavesAndBondsAfterErase( actor, shape ) == false )
	{
		return false;
	}

	actor->ownedAsset = *asset;
	actor->topologyVersion = asset->topologyVersion;
	actor->materialHash = asset->materialHash;
	uint64_t hash = 14695981039346656037ULL;
	hash = b2BlastMixHash( hash, (uint64_t)oldWidth );
	hash = b2BlastMixHash( hash, (uint64_t)oldHeight );
	hash = b2BlastMixHash( hash, (uint64_t)actor->leafCount );
	hash = b2BlastMixHash( hash, (uint64_t)actor->bondCount );
	hash = b2BlastMixHash( hash, (uint64_t)actor->clusterCount );
	hash = b2BlastMixHash( hash, asset->materialHash );
	for ( int i = 0; i < actor->leafCount; ++i )
	{
		const b2BlastLeaf* leaf = actor->leaves + i;
		hash = b2BlastMixHash( hash, leaf->cellCount );
		hash = b2BlastMixHash( hash, leaf->dominantMaterialId );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( leaf->centroid.x ) );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( leaf->centroid.y ) );
	}
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		const b2BlastActiveBond* bond = actor->bonds + i;
		hash = b2BlastMixHash( hash, ( (uint64_t)bond->leafA << 32 ) | bond->leafB );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( bond->area ) );
		hash = b2BlastMixHash( hash, b2BlastFloatBits( bond->capacity ) );
	}
	actor->authoringHash = hash;
	actor->revision += 1;
	actor->flags &= (uint32_t)~b2_blastActorFlagDirtyGraph;
	return true;
}

void b2BlastFractureWorld_Create( b2BlastFractureWorld* fractureWorld )
{
	*fractureWorld = (b2BlastFractureWorld){ 0 };
	fractureWorld->nextConstraintId = 1;
	fractureWorld->nextTransitionId = 1;
	fractureWorld->commandCapacity = b2_blastInitialCommandCapacity;
	fractureWorld->commands = b2AllocZeroInit( fractureWorld->commandCapacity * (int)sizeof( b2BlastFractureCommand ) );
	fractureWorld->transitionCapacity = 16;
	fractureWorld->transitions =
		b2AllocZeroInit( fractureWorld->transitionCapacity * (int)sizeof( b2BlastActorTransition ) );
	fractureWorld->overlayActorViewCapacity = 16;
	fractureWorld->overlayActorViews =
		b2AllocZeroInit( fractureWorld->overlayActorViewCapacity * (int)sizeof( b2BlastOverlayActorView ) );
	fractureWorld->overlayClusterCapacity = 64;
	fractureWorld->overlayClusters =
		b2AllocZeroInit( fractureWorld->overlayClusterCapacity * (int)sizeof( b2BlastOverlayCluster ) );
	fractureWorld->overlayBondCapacity = 128;
	fractureWorld->overlayBonds =
		b2AllocZeroInit( fractureWorld->overlayBondCapacity * (int)sizeof( b2BlastActiveBond ) );
	fractureWorld->bodyInputCapacity = 64;
	fractureWorld->bodyInputs =
		b2AllocZeroInit( fractureWorld->bodyInputCapacity * (int)sizeof( b2BlastBodyInputRecord ) );
	fractureWorld->overlayCellToActiveClusterCapacity = 512;
	fractureWorld->overlayCellToActiveCluster =
		b2AllocZeroInit( fractureWorld->overlayCellToActiveClusterCapacity * (int)sizeof( uint32_t ) );
	fractureWorld->overlayLeafRemapScratchCapacity = 512;
	fractureWorld->overlayLeafRemapScratch =
		b2AllocZeroInit( fractureWorld->overlayLeafRemapScratchCapacity * (int)sizeof( uint32_t ) );
}

void b2BlastFractureWorld_Destroy( b2BlastFractureWorld* fractureWorld )
{
	if ( fractureWorld == NULL )
	{
		return;
	}
	for ( int i = 0; i < fractureWorld->actorCount; ++i )
	{
		b2BlastActor_FreeArrays( fractureWorld->actors + i );
	}
	b2Free( fractureWorld->actors, fractureWorld->actorCapacity * (int)sizeof( b2BlastActor ) );
	b2Free( fractureWorld->commands, fractureWorld->commandCapacity * (int)sizeof( b2BlastFractureCommand ) );
	b2Free( fractureWorld->transitions, fractureWorld->transitionCapacity * (int)sizeof( b2BlastActorTransition ) );
	b2Free( fractureWorld->transitionCells, fractureWorld->transitionCellCapacity * (int)sizeof( int32_t ) );
	b2Free( fractureWorld->overlayActorViews, fractureWorld->overlayActorViewCapacity * (int)sizeof( b2BlastOverlayActorView ) );
	b2Free( fractureWorld->overlayClusters, fractureWorld->overlayClusterCapacity * (int)sizeof( b2BlastOverlayCluster ) );
	b2Free( fractureWorld->overlayBonds, fractureWorld->overlayBondCapacity * (int)sizeof( b2BlastActiveBond ) );
	b2Free( fractureWorld->bodyInputs, fractureWorld->bodyInputCapacity * (int)sizeof( b2BlastBodyInputRecord ) );
	b2Free( fractureWorld->overlayCellToActiveCluster,
			fractureWorld->overlayCellToActiveClusterCapacity * (int)sizeof( uint32_t ) );
	b2Free( fractureWorld->overlayLeafRemapScratch,
			fractureWorld->overlayLeafRemapScratchCapacity * (int)sizeof( uint32_t ) );
	*fractureWorld = (b2BlastFractureWorld){ 0 };
}

static void b2BlastFractureWorld_KeepCommittedTransitionJournal( b2BlastFractureWorld* fractureWorld )
{
	if ( fractureWorld == NULL || fractureWorld->transitionCount <= 0 )
	{
		return;
	}

	int writeTransition = 0;
	int writeCell = 0;
	for ( int readTransition = 0; readTransition < fractureWorld->transitionCount; ++readTransition )
	{
		b2BlastActorTransition transition = fractureWorld->transitions[readTransition];
		if ( transition.committed == false )
		{
			continue;
		}

		if ( transition.cellCount > 0 )
		{
			if ( transition.cellOffset < 0 ||
				 transition.cellOffset + transition.cellCount > fractureWorld->transitionCellCount )
			{
				transition.cellOffset = writeCell;
				transition.cellCount = 0;
			}
			else
			{
				if ( transition.cellOffset != writeCell )
				{
					memmove(
						fractureWorld->transitionCells + writeCell,
						fractureWorld->transitionCells + transition.cellOffset,
						(size_t)transition.cellCount * sizeof( int32_t ) );
				}
				transition.cellOffset = writeCell;
				writeCell += transition.cellCount;
			}
		}
		else
		{
			transition.cellOffset = writeCell;
		}
		fractureWorld->transitions[writeTransition++] = transition;
	}
	fractureWorld->transitionCount = writeTransition;
	fractureWorld->transitionCellCount = writeCell;
}

void b2BlastFractureWorld_ClearStep( b2BlastFractureWorld* fractureWorld )
{
	if ( fractureWorld == NULL )
	{
		return;
	}
	b2BlastFractureWorld_KeepCommittedTransitionJournal( fractureWorld );
	fractureWorld->commandCount = 0;
	fractureWorld->constraintRowCount = 0;
	fractureWorld->contactPairRowCount = 0;
	fractureWorld->contactImpactBudgetRowCount = 0;
	fractureWorld->contactLoadBudgetRowCount = 0;
	fractureWorld->contactYieldQueryCount = 0;
	fractureWorld->jointConstraintRowCount = 0;
	fractureWorld->actorTransitionCount = 0;
	fractureWorld->appliedForceLoadRowCount = 0;
	fractureWorld->appliedImpulseImpactRowCount = 0;
	fractureWorld->torqueInputIgnoredCount = fractureWorld->pendingTorqueInputIgnoredCount;
	fractureWorld->pendingTorqueInputIgnoredCount = 0;
	fractureWorld->ignoredOffTargetEventCount = 0;
	fractureWorld->unboundLoadPortDropCount = 0;
	fractureWorld->refinedThisStep = 0;
	fractureWorld->brokenThisStep = 0;
	fractureWorld->demandRecomputeCount = 0;
	fractureWorld->substepSolveCount = 0;
	fractureWorld->maxImpactDemand = 0.0f;
	fractureWorld->maxLoadDemand = 0.0f;
	fractureWorld->maxDamage = 0.0f;
	for ( int i = 0; i < fractureWorld->actorCount; ++i )
	{
		if ( ( fractureWorld->actors[i].flags & b2_blastActorFlagInUse ) != 0 )
		{
			b2BlastActor_ClearRuntimeDemand( fractureWorld->actors + i );
		}
	}
}

b2BlastFractureActorId b2BlastFractureWorld_UpsertPixelShapeActor(
	b2World* world, b2Body* body, b2Shape* shape, b2BlastActorMobility mobility )
{
	if ( world == NULL || body == NULL || shape == NULL || shape->type != b2_pixelShape )
	{
		return b2BlastNullActorId( world == NULL ? 0 : world->worldId );
	}

	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	b2BlastActor* actor = b2BlastGetActor( fractureWorld, shape->blastActorId );
	bool existingActor = actor != NULL;
	if ( actor == NULL )
	{
		actor = b2BlastAllocActor( world, world->worldId );
	}

	actor->bodyId = body->id;
	actor->shapeId = shape->id;
	actor->mobility = mobility;
	actor->flags |= b2_blastActorFlagInUse | b2_blastActorFlagDirtyGraph;
	const b2PixelShapeUpdateKind updateKind = shape->pixel.updateKind;
	const bool localOriginOnly =
		existingActor && updateKind == b2_pixelShapeUpdateLocalOriginOnly &&
		shape->pixel.asset != NULL && actor->topologyVersion == shape->pixel.asset->topologyVersion;
	if ( localOriginOnly == false )
	{
		actor->flags &= (uint32_t)~b2_blastActorFlagOwnsWorldAnchor;
		const bool dirtyRepair = existingActor && updateKind == b2_pixelShapeUpdateDirtyUpdate;
		const bool authored = dirtyRepair ? b2BlastActor_RepairFromPixelShapeDirtyUpdate( actor, shape )
										  : b2BlastActor_AuthorFromPixelShape( actor, shape );
		if ( authored == false )
		{
			fractureWorld->reauthoredFallbackCount += 1;
		}
	}

	body->blastActorId = actor->id;
	body->blastRevision = actor->revision;
	body->blastFlags |= 1u;
	shape->blastActorId = actor->id;
	shape->blastRevision = actor->revision;
	shape->blastFlags |= 1u;
	shape->pixelAssetRevision = shape->pixel.asset == NULL ? 0 : shape->pixel.asset->topologyVersion;
	shape->surfaceLookupKey = (uint32_t)shape->id;
	if ( localOriginOnly == false )
	{
		b2BlastClassifyAndCommandSplits( fractureWorld, actor );
	}
	return actor->id;
}

void b2BlastFractureWorld_UnbindShape( b2World* world, b2Shape* shape )
{
	if ( world == NULL || shape == NULL )
	{
		return;
	}
	b2BlastActor* actor = b2BlastGetActor( &world->blastFractureWorld, shape->blastActorId );
	if ( actor != NULL && actor->shapeId == shape->id )
	{
		actor->shapeId = B2_NULL_INDEX;
		actor->bodyId = B2_NULL_INDEX;
		actor->flags &= (uint32_t)~b2_blastActorFlagInUse;
	}
	shape->blastActorId = b2BlastNullActorId( world->worldId );
	shape->blastFlags = 0;
	shape->blastRevision = 0;
	shape->pixelAssetRevision = 0;
	shape->surfaceLookupKey = 0;
}

bool b2BlastFractureWorld_BindPixelShapeActor(
	b2World* world, b2Body* body, b2Shape* shape, b2BlastFractureActorId actorId, b2BlastActorMobility mobility )
{
	if ( world == NULL || body == NULL || shape == NULL || shape->type != b2_pixelShape )
	{
		return false;
	}

	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	b2BlastActor* actor = b2BlastGetActor( fractureWorld, actorId );
	if ( actor == NULL )
	{
		return false;
	}

	actor->bodyId = body->id;
	actor->shapeId = shape->id;
	actor->mobility = mobility;
	actor->flags |= b2_blastActorFlagInUse | b2_blastActorFlagDirtyGraph;
	actor->flags &= (uint32_t)~b2_blastActorFlagOwnsWorldAnchor;
	if ( b2BlastActor_AuthorFromPixelShape( actor, shape ) == false )
	{
		fractureWorld->reauthoredFallbackCount += 1;
		return false;
	}

	body->blastActorId = actor->id;
	body->blastRevision = actor->revision;
	body->blastFlags |= 1u;
	shape->blastActorId = actor->id;
	shape->blastRevision = actor->revision;
	shape->blastFlags |= 1u;
	shape->pixelAssetRevision = shape->pixel.asset == NULL ? 0 : shape->pixel.asset->topologyVersion;
	shape->surfaceLookupKey = (uint32_t)shape->id;
	return true;
}

static float b2BlastDistanceSquared( b2Vec2 a, b2Vec2 b )
{
	float dx = a.x - b.x;
	float dy = a.y - b.y;
	return dx * dx + dy * dy;
}

static int b2BlastFindNearestLeaf( const b2BlastActor* actor, b2Vec2 localPoint )
{
	int best = B2_NULL_INDEX;
	float bestDistance = FLT_MAX;
	for ( int i = 0; i < actor->leafCount; ++i )
	{
		float distance = b2BlastDistanceSquared( actor->leaves[i].centroid, localPoint );
		if ( distance < bestDistance )
		{
			bestDistance = distance;
			best = i;
		}
	}
	return best;
}

static int b2BlastFindEndpointLeaf( b2BlastFractureWorld* fractureWorld, const b2BlastActor* actor, const b2Shape* shape,
									b2Vec2 localPoint, bool allowNearestFallback )
{
	if ( actor == NULL || shape == NULL || shape->type != b2_pixelShape )
	{
		if ( fractureWorld != NULL )
		{
			fractureWorld->ignoredOffTargetEventCount += 1;
		}
		return B2_NULL_INDEX;
	}

	b2PixelLocalInfo info;
	if ( b2PixelShape_GetLocalInfo( &shape->pixel, localPoint, &info ) &&
		 b2PixelAsset_IsOccupied( shape->pixel.asset, info.x, info.y ) && info.index >= 0 && info.index < actor->cellToLeafCount )
	{
		uint32_t leaf = actor->cellToLeaf[info.index];
		if ( leaf != UINT32_MAX && leaf < (uint32_t)actor->leafCount )
		{
			if ( ( actor->leaves[leaf].flags & b2_blastLeafFlagDetached ) != 0 )
			{
				if ( fractureWorld != NULL )
				{
					fractureWorld->ignoredOffTargetEventCount += 1;
				}
				return B2_NULL_INDEX;
			}
			return (int)leaf;
		}
	}

	if ( allowNearestFallback )
	{
		return b2BlastFindNearestLeaf( actor, localPoint );
	}

	if ( fractureWorld != NULL )
	{
		fractureWorld->ignoredOffTargetEventCount += 1;
	}
	return B2_NULL_INDEX;
}

typedef struct b2BlastFractureContactSide
{
	b2BlastActor* actor;
	const b2Shape* shape;
	b2Vec2 localPoint;
	int leafIndex;
	b2BlastMaterialId materialId;
	float capacity;
	float hardness;
	float damageSoftening;
	bool destructible;
} b2BlastFractureContactSide;

typedef struct b2BlastFractureContactPair
{
	b2BlastFractureContactSide sides[2];
	float yieldImpulse;
	float damageShare[2];
	bool hasDestructibleSide;
} b2BlastFractureContactPair;

static float b2BlastContactPairEffectiveSupportCap( const b2BlastFractureContactPair* pair )
{
	if ( pair == NULL || pair->hasDestructibleSide == false )
	{
		return FLT_MAX;
	}

	// Local material/bond capacity is a damage-solve threshold, not a contact solver hard cap.
	// An intact PixelShape must support the regular rigid contact impulse until Blast2D has
	// actually broken the local active graph and committed topology at a solver safe point.
	float cap = FLT_MAX;
	for ( int sideIndex = 0; sideIndex < 2; ++sideIndex )
	{
		const b2BlastFractureContactSide* side = pair->sides + sideIndex;
		if ( side->destructible == false || side->actor == NULL || side->leafIndex == B2_NULL_INDEX )
		{
			continue;
		}

		if ( ( side->actor->leaves[side->leafIndex].flags & b2_blastLeafFlagDetached ) != 0 )
		{
			cap = b2MinFloat( cap, side->capacity );
			continue;
		}

		const uint32_t contactCluster = b2BlastActor_FindActiveClusterContainingLeaf( side->actor, (uint32_t)side->leafIndex );
		if ( contactCluster == UINT32_MAX )
		{
			continue;
		}

		bool activeSupportOpened = false;
		for ( int bondIndex = 0; bondIndex < side->actor->bondCount; ++bondIndex )
		{
			const b2BlastActiveBond* bond = side->actor->bonds + bondIndex;
			if ( ( bond->flags & b2_blastBondFlagBroken ) == 0 )
			{
				continue;
			}

			const uint32_t activeA = b2BlastActor_FindActiveClusterContainingLeaf( side->actor, bond->leafA );
			const uint32_t activeB = b2BlastActor_FindActiveClusterContainingLeaf( side->actor, bond->leafB );
			if ( activeA == UINT32_MAX || activeB == UINT32_MAX || activeA == activeB )
			{
				continue;
			}
			if ( activeA == contactCluster || activeB == contactCluster )
			{
				activeSupportOpened = true;
				break;
			}
		}

		if ( activeSupportOpened )
		{
			cap = b2MinFloat( cap, side->capacity );
		}
	}

	return cap;
}

static float b2BlastContactSideCapacity(
	const b2BlastActor* actor, int leafIndex, float normalVelocity, float* outHardness, float* outDamageSoftening )
{
	if ( outHardness != NULL )
	{
		*outHardness = 1.0f;
	}
	if ( outDamageSoftening != NULL )
	{
		*outDamageSoftening = 0.0f;
	}
	if ( actor == NULL || leafIndex == B2_NULL_INDEX || leafIndex < 0 || leafIndex >= actor->leafCount )
	{
		return FLT_MAX;
	}

	const b2BlastLeaf* leaf = actor->leaves + leafIndex;
	const b2BlastMaterialPhysics* material = b2BlastFindMaterial( &actor->ownedAsset, leaf->dominantMaterialId );
	const float compressStrength = material != NULL ? b2MaxFloat( material->compressStrength, material->contactCapacity ) : 64.0f;
	const float shearStrength = material != NULL ? b2MaxFloat( material->shearStrength, 0.0f ) : compressStrength;
	const float tensileStrength = material != NULL ? b2MaxFloat( material->tensileStrength, 0.0f ) : shearStrength;
	const float hardness = material != NULL ? b2MaxFloat( material->hardness, 0.05f ) : 1.0f;
	const float toughness = material != NULL ? b2MaxFloat( material->toughness, 0.05f ) : 1.0f;
	const float strength = b2MaxFloat( 0.05f, b2MinFloat( compressStrength, b2MaxFloat( shearStrength, tensileStrength ) ) );
	const float leafArea = b2MaxFloat( 1.0f, (float)leaf->cellCount );
	const float patchArea = b2MaxFloat( 1.0f, 0.5f * sqrtf( leafArea ) );
	float materialCapacity = strength * ( 2.0f + 0.75f * patchArea ) * ( 0.75f + 0.25f * toughness );

	float weakestBond = FLT_MAX;
	float bondSum = 0.0f;
	float damageSum = 0.0f;
	int bondCount = 0;
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		const b2BlastActiveBond* bond = actor->bonds + i;
		if ( ( bond->flags & b2_blastBondFlagBroken ) != 0 )
		{
			continue;
		}
		if ( (int)bond->leafA != leafIndex && (int)bond->leafB != leafIndex )
		{
			continue;
		}
		const float remaining = b2MaxFloat( 0.0f, bond->capacity * ( 1.0f - b2ClampFloat( bond->damage, 0.0f, 0.95f ) ) );
		weakestBond = b2MinFloat( weakestBond, remaining );
		bondSum += remaining;
		damageSum += b2ClampFloat( bond->damage, 0.0f, 1.0f );
		bondCount += 1;
	}

	float capacity = materialCapacity;
	float averageDamage = 0.0f;
	if ( bondCount > 0 )
	{
		const float localBondCapacity = 0.65f * weakestBond + 0.35f * ( bondSum / (float)bondCount );
		capacity = b2MinFloat( materialCapacity, b2MaxFloat( 0.25f, localBondCapacity ) );
		averageDamage = damageSum / (float)bondCount;
	}

	const float damageSoftening = 1.0f - 0.70f * b2ClampFloat( averageDamage, 0.0f, 0.95f );
	const float approachSpeed = b2MaxFloat( 0.0f, -normalVelocity );
	const float rateHardening = 1.0f + 0.08f * log1pf( approachSpeed );
	capacity = b2MaxFloat( 0.05f, capacity * damageSoftening * rateHardening );

	if ( outHardness != NULL )
	{
		*outHardness = hardness;
	}
	if ( outDamageSoftening != NULL )
	{
		*outDamageSoftening = 1.0f - damageSoftening;
	}
	return capacity;
}

static void b2BlastBuildContactSide( b2World* world, b2BlastFractureWorld* fractureWorld, const b2Shape* shape, b2Body* body,
									 b2Vec2 worldPoint, float normalVelocity, b2BlastFractureContactSide* side )
{
	*side = (b2BlastFractureContactSide){ 0 };
	side->shape = shape;
	side->leafIndex = B2_NULL_INDEX;
	side->capacity = FLT_MAX;
	side->hardness = 1.0f;

	if ( fractureWorld == NULL || shape == NULL || body == NULL )
	{
		return;
	}

	b2BlastActor* actor = b2BlastGetActor( fractureWorld, shape->blastActorId );
	if ( actor == NULL || shape->type != b2_pixelShape )
	{
		return;
	}

	b2Transform transform = b2GetBodyTransformQuick( world, body );
	b2Vec2 localPoint = b2InvTransformPoint( transform, worldPoint );
	int leaf = b2BlastFindEndpointLeaf( fractureWorld, actor, shape, localPoint, true );
	if ( leaf == B2_NULL_INDEX )
	{
		return;
	}

	side->actor = actor;
	side->localPoint = localPoint;
	side->leafIndex = leaf;
	side->materialId = actor->leaves[leaf].dominantMaterialId;
	side->capacity = b2BlastContactSideCapacity( actor, leaf, normalVelocity, &side->hardness, &side->damageSoftening );
	side->destructible = true;
}

static bool b2BlastBuildContactPair( b2World* world, const b2ContactSim* contactSim, b2Vec2 anchorA, b2Vec2 anchorB,
									 float normalVelocity, b2BlastFractureContactPair* pair )
{
	if ( pair == NULL )
	{
		return false;
	}
	*pair = (b2BlastFractureContactPair){ 0 };
	pair->yieldImpulse = FLT_MAX;
	pair->damageShare[0] = 0.0f;
	pair->damageShare[1] = 0.0f;
	if ( world == NULL || contactSim == NULL )
	{
		return false;
	}

	b2Shape* shapeA = b2ShapeArray_Get( &world->shapes, contactSim->shapeIdA );
	b2Shape* shapeB = b2ShapeArray_Get( &world->shapes, contactSim->shapeIdB );
	b2Body* bodyA = b2BodyArray_Get( &world->bodies, shapeA->bodyId );
	b2Body* bodyB = b2BodyArray_Get( &world->bodies, shapeB->bodyId );
	if ( shapeA == NULL || shapeB == NULL || bodyA == NULL || bodyB == NULL )
	{
		return false;
	}

	b2Transform transformA = b2GetBodyTransformQuick( world, bodyA );
	b2Transform transformB = b2GetBodyTransformQuick( world, bodyB );
	b2Vec2 worldPointA = b2Add( transformA.p, anchorA );
	b2Vec2 worldPointB = b2Add( transformB.p, anchorB );
	b2Vec2 worldPoint = b2MulSV( 0.5f, b2Add( worldPointA, worldPointB ) );

	b2BlastBuildContactSide( world, &world->blastFractureWorld, shapeA, bodyA, worldPoint, normalVelocity, pair->sides + 0 );
	b2BlastBuildContactSide( world, &world->blastFractureWorld, shapeB, bodyB, worldPoint, normalVelocity, pair->sides + 1 );
	pair->hasDestructibleSide = pair->sides[0].destructible || pair->sides[1].destructible;
	if ( pair->hasDestructibleSide == false )
	{
		return false;
	}

	if ( pair->sides[0].destructible )
	{
		pair->yieldImpulse = b2MinFloat( pair->yieldImpulse, pair->sides[0].capacity );
	}
	if ( pair->sides[1].destructible )
	{
		pair->yieldImpulse = b2MinFloat( pair->yieldImpulse, pair->sides[1].capacity );
	}
	if ( pair->yieldImpulse <= 0.0f || pair->yieldImpulse == FLT_MAX )
	{
		pair->yieldImpulse = FLT_MAX;
	}

	float vulnerabilityA = pair->sides[0].destructible ? 1.0f / b2MaxFloat( pair->sides[0].capacity, 0.05f ) : 0.0f;
	float vulnerabilityB = pair->sides[1].destructible ? 1.0f / b2MaxFloat( pair->sides[1].capacity, 0.05f ) : 0.0f;
	float vulnerabilitySum = vulnerabilityA + vulnerabilityB;
	if ( vulnerabilitySum > 0.0f )
	{
		pair->damageShare[0] = vulnerabilityA / vulnerabilitySum;
		pair->damageShare[1] = vulnerabilityB / vulnerabilitySum;
	}
	return true;
}

float b2BlastFractureWorld_ComputeContactYieldImpulse( b2World* world, const b2ContactSim* contactSim, int pointIndex,
														b2Vec2 normal, b2Vec2 anchorA, b2Vec2 anchorB, float normalVelocity )
{
	B2_UNUSED( pointIndex );
	B2_UNUSED( normal );
	b2BlastFractureContactPair pair;
	if ( b2BlastBuildContactPair( world, contactSim, anchorA, anchorB, normalVelocity, &pair ) == false )
	{
		return FLT_MAX;
	}
	world->blastFractureWorld.contactYieldQueryCount += 1;
	return b2BlastContactPairEffectiveSupportCap( &pair );
}

static float b2BlastBondGraphCost( const b2BlastActor* actor, const b2BlastActiveBond* bond )
{
	b2Vec2 a = actor->leaves[bond->leafA].centroid;
	b2Vec2 b = actor->leaves[bond->leafB].centroid;
	float length = b2Length( b2Sub( b, a ) );
	float areaScale = 1.0f / b2MaxFloat( 1.0f, bond->area );
	float toughnessScale = sqrtf( b2MaxFloat( 1.0f, bond->toughness ) ) / b2MaxFloat( 1.0f, sqrtf( bond->capacity ) );
	return b2MaxFloat( 0.01f, length * ( 0.65f + 0.35f * bond->propagationWeight ) + areaScale * toughnessScale );
}

static bool b2BlastComputeGraphDistances( b2BlastActor* actor, int sourceLeaf )
{
	if ( actor == NULL || sourceLeaf == B2_NULL_INDEX || sourceLeaf >= actor->leafCount ||
		 actor->graphDistanceScratchCapacity < actor->leafCount || actor->graphParentBondScratchCapacity < actor->leafCount ||
		 actor->visitScratchCapacity < actor->leafCount )
	{
		return false;
	}

	for ( int i = 0; i < actor->leafCount; ++i )
	{
		actor->graphDistanceScratch[i] = FLT_MAX;
		actor->graphParentBondScratch[i] = B2_NULL_INDEX;
		actor->visitScratch[i] = 0;
	}
	actor->graphDistanceScratch[sourceLeaf] = 0.0f;

	for ( int iter = 0; iter < actor->leafCount; ++iter )
	{
		int leaf = B2_NULL_INDEX;
		float best = FLT_MAX;
		for ( int i = 0; i < actor->leafCount; ++i )
		{
			if ( actor->visitScratch[i] == 0 && actor->graphDistanceScratch[i] < best )
			{
				best = actor->graphDistanceScratch[i];
				leaf = i;
			}
		}

		if ( leaf == B2_NULL_INDEX )
		{
			break;
		}
		actor->visitScratch[leaf] = 1;

		for ( int bondIndex = 0; bondIndex < actor->bondCount; ++bondIndex )
		{
			const b2BlastActiveBond* bond = actor->bonds + bondIndex;
			if ( ( bond->flags & b2_blastBondFlagBroken ) != 0 )
			{
				continue;
			}
			if ( ( actor->leaves[bond->leafA].flags & b2_blastLeafFlagDetached ) != 0 ||
				 ( actor->leaves[bond->leafB].flags & b2_blastLeafFlagDetached ) != 0 )
			{
				continue;
			}
			int other = B2_NULL_INDEX;
			if ( (int)bond->leafA == leaf )
			{
				other = (int)bond->leafB;
			}
			else if ( (int)bond->leafB == leaf )
			{
				other = (int)bond->leafA;
			}

			if ( other == B2_NULL_INDEX || actor->visitScratch[other] != 0 )
			{
				continue;
			}

			float candidate = best + b2BlastBondGraphCost( actor, bond );
			if ( candidate < actor->graphDistanceScratch[other] )
			{
				actor->graphDistanceScratch[other] = candidate;
				actor->graphParentBondScratch[other] = bondIndex;
			}
		}
	}

	return true;
}

static float b2BlastDirectionAlignment( b2Vec2 direction, b2Vec2 source, b2Vec2 a, b2Vec2 b, bool impact )
{
	if ( b2LengthSquared( direction ) <= 0.000001f )
	{
		return 1.0f;
	}

	b2Vec2 dir = b2Normalize( direction );
	b2Vec2 mid = b2MulSV( 0.5f, b2Add( a, b ) );
	b2Vec2 toMid = b2Sub( mid, source );
	float directional = 1.0f;
	if ( impact && b2LengthSquared( toMid ) > 0.000001f )
	{
		directional = 0.18f + 0.82f * b2MaxFloat( 0.0f, b2Dot( dir, b2Normalize( toMid ) ) );
	}

	b2Vec2 axis = b2Sub( b, a );
	float axisLen = b2Length( axis );
	float axisAlign = 1.0f;
	if ( axisLen > 0.0001f )
	{
		axisAlign = 0.35f + 0.65f * fabsf( b2Dot( b2MulSV( 1.0f / axisLen, axis ), dir ) );
	}
	return directional * axisAlign;
}

static float b2BlastBondNotchFactor( const b2BlastActor* actor, const b2BlastActiveBond* bond )
{
	if ( actor == NULL || bond == NULL || actor->ownedAsset.width <= 0 || actor->ownedAsset.height <= 0 )
	{
		return 1.0f;
	}

	b2Vec2 a = actor->leaves[bond->leafA].centroid;
	b2Vec2 b = actor->leaves[bond->leafB].centroid;
	b2Vec2 mid = b2MulSV( 0.5f, b2Add( a, b ) );
	int cx = (int)floorf( mid.x );
	int cy = (int)floorf( mid.y );
	int occupied = 0;
	int empty = 0;
	for ( int dy = -2; dy <= 2; ++dy )
	{
		for ( int dx = -2; dx <= 2; ++dx )
		{
			int x = cx + dx;
			int y = cy + dy;
			if ( x < 0 || y < 0 || x >= actor->ownedAsset.width || y >= actor->ownedAsset.height )
			{
				continue;
			}
			if ( b2PixelAsset_IsOccupied( &actor->ownedAsset, x, y ) )
			{
				occupied += 1;
			}
			else
			{
				empty += 1;
			}
		}
	}

	const float total = (float)( occupied + empty );
	if ( total <= 0.0f )
	{
		return 1.0f;
	}
	float concavity = (float)empty / total;
	return 1.0f + b2ClampFloat( ( concavity - 0.25f ) * 1.4f, 0.0f, 0.75f );
}

static bool b2BlastBondCrackTipNear( const b2BlastActor* actor, const b2BlastActiveBond* bond )
{
	if ( actor == NULL || bond == NULL )
	{
		return false;
	}
	if ( bond->damage > 0.25f )
	{
		return true;
	}
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		const b2BlastActiveBond* other = actor->bonds + i;
		if ( other == bond || ( other->flags & b2_blastBondFlagBroken ) == 0 )
		{
			continue;
		}
		if ( other->leafA == bond->leafA || other->leafA == bond->leafB || other->leafB == bond->leafA ||
			 other->leafB == bond->leafB )
		{
			return true;
		}
	}
	return false;
}

static float b2BlastBondSupportNearFactor( const b2BlastActor* actor, const b2BlastActiveBond* bond )
{
	if ( actor == NULL || bond == NULL )
	{
		return 1.0f;
	}
	if ( ( actor->leaves[bond->leafA].flags & b2_blastLeafFlagWorldAnchor ) != 0 ||
		 ( actor->leaves[bond->leafB].flags & b2_blastLeafFlagWorldAnchor ) != 0 )
	{
		b2BlastMaterialId materialId = actor->leaves[bond->leafA].dominantMaterialId;
		const b2BlastMaterialPhysics* material = b2BlastFindMaterial( &actor->ownedAsset, materialId );
		float supportReflection = material != NULL ? b2MaxFloat( material->supportReflection, 0.0f ) : 1.0f;
		return 1.0f + 0.25f * supportReflection;
	}
	return 1.0f;
}

static void b2BlastApplyImpactWaveFromLeafAtPoint( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor, int leafIndex,
												   b2Vec2 source, float impulse, b2Vec2 direction, float contactRadius )
{
	if ( actor == NULL || leafIndex == B2_NULL_INDEX || impulse <= 0.0f )
	{
		return;
	}

	if ( b2BlastComputeGraphDistances( actor, leafIndex ) == false )
	{
		fractureWorld->stepAllocationFallbackCount += 1;
		return;
	}

	const b2BlastMaterialPhysics* sourceMaterial =
		b2BlastFindMaterial( &actor->ownedAsset, actor->leaves[leafIndex].dominantMaterialId );
	const float brittleness = sourceMaterial != NULL ? b2ClampFloat( sourceMaterial->brittleness, 0.0f, 2.0f ) : 1.0f;
	const float impactRange =
		sourceMaterial != NULL && sourceMaterial->impactRange > 0.0f ? sourceMaterial->impactRange : b2_blastDefaultImpactRange;
	const float impactDecay = b2MaxFloat( 0.1f,
		sourceMaterial != NULL && sourceMaterial->impactDecay > 0.0f ? sourceMaterial->impactDecay : b2_blastDefaultImpactDecay );
	const float impactGain = 0.85f + 0.35f * brittleness;
	const float shield = b2ClampFloat(
		sourceMaterial != NULL && sourceMaterial->contactShield > 0.0f ? sourceMaterial->contactShield : b2_blastDefaultContactShield,
		0.0f, 8.0f );
	const float radius = contactRadius > 0.0f ? contactRadius : b2_blastDefaultContactDamageRadius;

	for ( int i = 0; i < actor->bondCount; ++i )
	{
		b2BlastActiveBond* bond = actor->bonds + i;
		if ( ( bond->flags & b2_blastBondFlagBroken ) != 0 )
		{
			continue;
		}
		if ( ( actor->leaves[bond->leafA].flags & b2_blastLeafFlagDetached ) != 0 ||
			 ( actor->leaves[bond->leafB].flags & b2_blastLeafFlagDetached ) != 0 )
		{
			continue;
		}
		b2Vec2 a = actor->leaves[bond->leafA].centroid;
		b2Vec2 b = actor->leaves[bond->leafB].centroid;
		float da = actor->graphDistanceScratch[bond->leafA];
		float db = actor->graphDistanceScratch[bond->leafB];
		if ( da == FLT_MAX && db == FLT_MAX )
		{
			continue;
		}
		float distance = da == FLT_MAX ? db : ( db == FLT_MAX ? da : 0.5f * ( da + db ) );
		if ( distance > impactRange )
		{
			continue;
		}
		float align = b2BlastDirectionAlignment( direction, source, a, b, true );
		float areaAttenuation = 1.0f / powf( b2MaxFloat( 1.0f, bond->area ), 0.42f );
		float demand = ( impulse / b2_blastReferenceImpactImpulse ) * expf( -distance / impactDecay ) * ( 0.45f + 0.75f * align ) *
					   areaAttenuation * impactGain;
		demand *= b2BlastBondNotchFactor( actor, bond ) * b2BlastBondSupportNearFactor( actor, bond );
		if ( b2BlastBondCrackTipNear( actor, bond ) )
		{
			demand *= 1.4f;
		}
		if ( distance <= radius * 1.5f )
		{
			demand *= shield;
		}
		bond->impactDemand += demand;
		fractureWorld->maxImpactDemand = b2MaxFloat( fractureWorld->maxImpactDemand, bond->impactDemand );
	}
}

static void b2BlastApplyImpactWaveFromLeaf( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor, int leafIndex,
											float impulse, b2Vec2 direction )
{
	if ( actor == NULL || leafIndex == B2_NULL_INDEX || leafIndex >= actor->leafCount )
	{
		return;
	}
	b2BlastApplyImpactWaveFromLeafAtPoint(
		fractureWorld, actor, leafIndex, actor->leaves[leafIndex].centroid, impulse, direction, b2_blastDefaultContactDamageRadius );
}

static int b2BlastFindNearestSupportLeaf( const b2BlastActor* actor )
{
	int bestLeaf = B2_NULL_INDEX;
	float bestDistance = FLT_MAX;
	for ( int i = 0; i < actor->leafCount; ++i )
	{
		if ( ( actor->leaves[i].flags & b2_blastLeafFlagDetached ) != 0 )
		{
			continue;
		}
		if ( ( actor->leaves[i].flags & b2_blastLeafFlagWorldAnchor ) == 0 )
		{
			continue;
		}
		float distance = actor->graphDistanceScratch[i];
		if ( distance < bestDistance )
		{
			bestDistance = distance;
			bestLeaf = i;
		}
	}
	return bestLeaf;
}

static void b2BlastApplyLoadPathFromLeaf( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor, int leafIndex, float force,
										  b2Vec2 direction )
{
	if ( actor == NULL || leafIndex == B2_NULL_INDEX || force <= 0.0f )
	{
		return;
	}
	if ( ( actor->flags & b2_blastActorFlagOwnsWorldAnchor ) == 0 )
	{
		return;
	}
	if ( b2BlastComputeGraphDistances( actor, leafIndex ) == false )
	{
		fractureWorld->stepAllocationFallbackCount += 1;
		return;
	}

	int supportLeaf = b2BlastFindNearestSupportLeaf( actor );
	if ( supportLeaf == B2_NULL_INDEX || actor->graphDistanceScratch[supportLeaf] == FLT_MAX )
	{
		return;
	}

	b2Vec2 source = actor->leaves[leafIndex].centroid;
	int current = supportLeaf;
	float pathLength = b2MaxFloat( actor->graphDistanceScratch[supportLeaf], 1.0f );
	int guard = 0;
	while ( current != leafIndex && guard++ < actor->leafCount )
	{
		int bondIndex = actor->graphParentBondScratch[current];
		if ( bondIndex == B2_NULL_INDEX )
		{
			break;
		}
		b2BlastActiveBond* bond = actor->bonds + bondIndex;
		if ( ( bond->flags & b2_blastBondFlagBroken ) != 0 )
		{
			break;
		}
		if ( ( actor->leaves[bond->leafA].flags & b2_blastLeafFlagDetached ) != 0 ||
			 ( actor->leaves[bond->leafB].flags & b2_blastLeafFlagDetached ) != 0 )
		{
			break;
		}

		b2Vec2 a = actor->leaves[bond->leafA].centroid;
		b2Vec2 b = actor->leaves[bond->leafB].centroid;
		float align = b2BlastDirectionAlignment( direction, source, a, b, false );
		float pathFactor = 1.0f + 0.35f * ( actor->graphDistanceScratch[current] / pathLength );
		float section = 1.0f + 2.6f / b2MaxFloat( 1.0f, bond->area );
		float capacityWeight = 1.0f / ( 1.0f + 0.025f * sqrtf( b2MaxFloat( 1.0f, bond->capacity ) ) );
		float demand = force * align * pathFactor * section * capacityWeight;
		bond->loadDemand += demand;
		fractureWorld->maxLoadDemand = b2MaxFloat( fractureWorld->maxLoadDemand, bond->loadDemand );

		int next = (int)bond->leafA == current ? (int)bond->leafB : (int)bond->leafA;
		current = next;
	}
}

static void b2BlastApplyLocalContactLoadFromLeaf(
	b2BlastFractureWorld* fractureWorld, b2BlastActor* actor, int leafIndex, float force, b2Vec2 direction )
{
	if ( actor == NULL || leafIndex == B2_NULL_INDEX || force <= 0.0f )
	{
		return;
	}
	if ( b2BlastComputeGraphDistances( actor, leafIndex ) == false )
	{
		fractureWorld->stepAllocationFallbackCount += 1;
		return;
	}

	b2Vec2 source = actor->leaves[leafIndex].centroid;
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		b2BlastActiveBond* bond = actor->bonds + i;
		if ( ( bond->flags & b2_blastBondFlagBroken ) != 0 )
		{
			continue;
		}
		if ( ( actor->leaves[bond->leafA].flags & b2_blastLeafFlagDetached ) != 0 ||
			 ( actor->leaves[bond->leafB].flags & b2_blastLeafFlagDetached ) != 0 )
		{
			continue;
		}
		float da = actor->graphDistanceScratch[bond->leafA];
		float db = actor->graphDistanceScratch[bond->leafB];
		if ( da == FLT_MAX && db == FLT_MAX )
		{
			continue;
		}
		float distance = da == FLT_MAX ? db : ( db == FLT_MAX ? da : 0.5f * ( da + db ) );
		b2Vec2 a = actor->leaves[bond->leafA].centroid;
		b2Vec2 b = actor->leaves[bond->leafB].centroid;
		float align = b2BlastDirectionAlignment( direction, source, a, b, false );
		float section = 1.0f + 1.2f / b2MaxFloat( 1.0f, bond->area );
		float capacityWeight = 1.0f / ( 1.0f + 0.035f * sqrtf( b2MaxFloat( 1.0f, bond->capacity ) ) );
		float decay = expf( -0.72f * distance );
		float demand = force * align * section * capacityWeight * decay;
		bond->loadDemand += demand;
		fractureWorld->maxLoadDemand = b2MaxFloat( fractureWorld->maxLoadDemand, bond->loadDemand );
	}
}

static void b2BlastApplyLoadBudgetFromLeaf(
	b2BlastFractureWorld* fractureWorld, b2BlastActor* actor, int leafIndex, float force, b2Vec2 direction )
{
	if ( fractureWorld == NULL || actor == NULL || leafIndex == B2_NULL_INDEX || force <= 0.0f )
	{
		return;
	}
	const float beforeMaxLoadDemand = fractureWorld->maxLoadDemand;
	if ( ( actor->flags & b2_blastActorFlagOwnsWorldAnchor ) != 0 )
	{
		b2BlastApplyLoadPathFromLeaf( fractureWorld, actor, leafIndex, force, direction );
	}
	if ( fractureWorld->maxLoadDemand <= beforeMaxLoadDemand + 0.000001f )
	{
		b2BlastApplyLocalContactLoadFromLeaf( fractureWorld, actor, leafIndex, force, direction );
	}
}

static bool b2BlastAppendTransitionForComponent( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor, int label )
{
	if ( fractureWorld == NULL || actor == NULL || label < 0 || actor->shapeId == B2_NULL_INDEX || actor->cellToLeafCount <= 0 )
	{
		return false;
	}

	for ( int transitionIndex = 0; transitionIndex < fractureWorld->transitionCount; ++transitionIndex )
	{
		const b2BlastActorTransition* existing = fractureWorld->transitions + transitionIndex;
		if ( existing->committed || b2BlastActorIdEqual( existing->sourceActorId, actor->id ) == false )
		{
			continue;
		}
		for ( int i = 0; i < existing->cellCount; ++i )
		{
			int cell = fractureWorld->transitionCells[existing->cellOffset + i];
			if ( cell >= 0 && cell < actor->cellToLeafCount )
			{
				uint32_t leaf = actor->cellToLeaf[cell];
				if ( leaf != UINT32_MAX && leaf < (uint32_t)actor->leafCount && actor->componentScratch[leaf] == label )
				{
					return true;
				}
			}
		}
	}

	int sourceWidth = actor->ownedAsset.width;
	if ( sourceWidth <= 0 )
	{
		return false;
	}

	int cellCount = 0;
	int minX = INT32_MAX;
	int minY = INT32_MAX;
	int maxX = INT32_MIN;
	int maxY = INT32_MIN;
	for ( int cell = 0; cell < actor->cellToLeafCount; ++cell )
	{
		uint32_t leaf = actor->cellToLeaf[cell];
		if ( leaf == UINT32_MAX || leaf >= (uint32_t)actor->leafCount || actor->componentScratch[leaf] != label )
		{
			continue;
		}
		int x = cell % sourceWidth;
		int y = cell / sourceWidth;
		minX = b2MinInt( minX, x );
		minY = b2MinInt( minY, y );
		maxX = b2MaxInt( maxX, x );
		maxY = b2MaxInt( maxY, y );
		++cellCount;
	}
	if ( cellCount <= 0 )
	{
		return false;
	}

	b2BlastEnsureTransitionCapacity(
		fractureWorld, fractureWorld->transitionCount + 1, fractureWorld->transitionCellCount + cellCount );
	int cellOffset = fractureWorld->transitionCellCount;
	for ( int cell = 0; cell < actor->cellToLeafCount; ++cell )
	{
		uint32_t leaf = actor->cellToLeaf[cell];
		if ( leaf != UINT32_MAX && leaf < (uint32_t)actor->leafCount && actor->componentScratch[leaf] == label )
		{
			fractureWorld->transitionCells[fractureWorld->transitionCellCount++] = cell;
		}
	}
	for ( int leaf = 0; leaf < actor->leafCount; ++leaf )
	{
		if ( actor->componentScratch[leaf] == label )
		{
			actor->leaves[leaf].flags |= b2_blastLeafFlagDetached;
		}
	}

	b2BlastActorTransition* transition = fractureWorld->transitions + fractureWorld->transitionCount++;
	*transition = (b2BlastActorTransition){ 0 };
	transition->transitionId = fractureWorld->nextTransitionId++;
	transition->action = b2_blastActorTransitionCreateDynamicBody;
	transition->sourceActorId = actor->id;
	transition->childActorId = b2BlastNullActorId( actor->id.world );
	transition->createdBodyId = b2_nullBodyId;
	transition->createdShapeId = b2_nullShapeId;
	transition->sourceMobility = actor->mobility;
	transition->targetMobility = b2_blastActorMobilityDynamic;
	transition->sourceMinX = minX;
	transition->sourceMinY = minY;
	transition->sourceMaxX = maxX;
	transition->sourceMaxY = maxY;
	transition->sourceWidth = actor->ownedAsset.width;
	transition->sourceHeight = actor->ownedAsset.height;
	transition->cellOffset = cellOffset;
	transition->cellCount = cellCount;
	transition->sourceTopologyRevision = actor->topologyVersion;
	transition->materialHash = actor->materialHash;
	fractureWorld->actorTransitionCount += 1;
	return true;
}

static void b2BlastClassifyAndCommandSplits( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor )
{
	if ( actor == NULL || actor->leafCount <= 1 || actor->componentScratchCapacity < actor->leafCount ||
		 actor->queueScratchCapacity < actor->leafCount || actor->visitScratchCapacity < actor->leafCount ||
		 actor->cellScratchCapacity < actor->leafCount || actor->graphParentBondScratchCapacity < actor->leafCount )
	{
		return;
	}

	memset( actor->visitScratch, 0, (size_t)actor->leafCount * sizeof( uint8_t ) );
	for ( int i = 0; i < actor->leafCount; ++i )
	{
		actor->componentScratch[i] = -1;
	}
	int componentCount = 0;
	for ( int start = 0; start < actor->leafCount; ++start )
	{
		if ( ( actor->leaves[start].flags & b2_blastLeafFlagDetached ) != 0 )
		{
			continue;
		}
		if ( actor->visitScratch[start] != 0 )
		{
			continue;
		}

		bool supported = ( actor->leaves[start].flags & b2_blastLeafFlagWorldAnchor ) != 0;
		int head = 0;
		int tail = 0;
		actor->queueScratch[tail++] = start;
		actor->visitScratch[start] = 1;
		while ( head < tail )
		{
			int leaf = actor->queueScratch[head++];
			actor->componentScratch[leaf] = componentCount;
			if ( ( actor->leaves[leaf].flags & b2_blastLeafFlagWorldAnchor ) != 0 )
			{
				supported = true;
			}
			for ( int i = 0; i < actor->bondCount; ++i )
			{
				const b2BlastActiveBond* bond = actor->bonds + i;
				if ( ( bond->flags & b2_blastBondFlagBroken ) != 0 )
				{
					continue;
				}
				int other = B2_NULL_INDEX;
				if ( (int)bond->leafA == leaf )
				{
					other = (int)bond->leafB;
				}
				else if ( (int)bond->leafB == leaf )
				{
					other = (int)bond->leafA;
				}
				if ( other != B2_NULL_INDEX && actor->visitScratch[other] == 0 )
				{
					actor->visitScratch[other] = 1;
					actor->queueScratch[tail++] = other;
				}
			}
		}

		actor->cellScratch[componentCount] = supported ? 1 : 0;
		actor->graphParentBondScratch[componentCount] = tail;
		componentCount += 1;
	}
	if ( componentCount <= 1 )
	{
		return;
	}
	int keepUnsupportedLabel = B2_NULL_INDEX;
	int keepUnsupportedSize = -1;
	bool hasSupportedComponent = false;
	for ( int label = 0; label < componentCount; ++label )
	{
		if ( actor->cellScratch[label] != 0 )
		{
			hasSupportedComponent = true;
		}
		else if ( actor->graphParentBondScratch[label] > keepUnsupportedSize )
		{
			keepUnsupportedSize = actor->graphParentBondScratch[label];
			keepUnsupportedLabel = label;
		}
	}
	for ( int label = 0; label < componentCount; ++label )
	{
		if ( actor->cellScratch[label] == 0 && ( hasSupportedComponent || label != keepUnsupportedLabel ) )
		{
			(void)b2BlastAppendTransitionForComponent( fractureWorld, actor, label );
		}
	}
}

static void b2BlastGetBondMaterialResponse( const b2BlastActor* actor, const b2BlastActiveBond* bond, float* outOnset,
											float* outRefineOnset, float* outBreakRatio, float* outDamageRate, float* outRecovery,
											int* outMaxBreaks )
{
	b2BlastMaterialId materialIdA = actor->leaves[bond->leafA].dominantMaterialId;
	b2BlastMaterialId materialIdB = actor->leaves[bond->leafB].dominantMaterialId;
	const b2BlastMaterialPhysics* materialA = b2BlastFindMaterial( &actor->ownedAsset, materialIdA );
	const b2BlastMaterialPhysics* materialB = b2BlastFindMaterial( &actor->ownedAsset, materialIdB );
	float onsetA = materialA != NULL ? materialA->onset : 0.35f;
	float onsetB = materialB != NULL ? materialB->onset : onsetA;
	float refineOnsetA = materialA != NULL ? materialA->refineOnset : 0.62f;
	float refineOnsetB = materialB != NULL ? materialB->refineOnset : refineOnsetA;
	float breakRatioA = materialA != NULL ? materialA->breakRatio : 1.0f;
	float breakRatioB = materialB != NULL ? materialB->breakRatio : breakRatioA;
	float damageRateA = materialA != NULL ? materialA->damageRate : 0.55f;
	float damageRateB = materialB != NULL ? materialB->damageRate : damageRateA;
	float recoveryA = materialA != NULL ? materialA->recovery : 0.0f;
	float recoveryB = materialB != NULL ? materialB->recovery : recoveryA;
	int maxBreaksA = materialA != NULL && materialA->maxBreaksPerStep > 0 ? materialA->maxBreaksPerStep : b2_blastDefaultMaxBreaksPerStep;
	int maxBreaksB = materialB != NULL && materialB->maxBreaksPerStep > 0 ? materialB->maxBreaksPerStep : maxBreaksA;

	*outOnset = b2ClampFloat( 0.5f * ( onsetA + onsetB ), 0.01f, 16.0f );
	*outRefineOnset = b2ClampFloat( 0.5f * ( refineOnsetA + refineOnsetB ), 0.01f, 16.0f );
	*outBreakRatio = b2ClampFloat( 0.5f * ( breakRatioA + breakRatioB ), 0.01f, 16.0f );
	*outDamageRate = b2ClampFloat( 0.5f * ( damageRateA + damageRateB ), 0.0f, 8.0f );
	*outRecovery = b2ClampFloat( 0.5f * ( recoveryA + recoveryB ), 0.0f, 1.0f );
	*outMaxBreaks = b2ClampInt( b2MaxInt( maxBreaksA, maxBreaksB ), 0, 512 );
}

static bool b2BlastRunDamageShaderRefinePass( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor )
{
	bool anyRefined = false;
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		b2BlastActiveBond* bond = actor->bonds + i;
		if ( ( bond->flags & b2_blastBondFlagBroken ) != 0 )
		{
			continue;
		}
		if ( ( actor->leaves[bond->leafA].flags & b2_blastLeafFlagDetached ) != 0 ||
			 ( actor->leaves[bond->leafB].flags & b2_blastLeafFlagDetached ) != 0 )
		{
			continue;
		}
		const float demand = bond->impactDemand + bond->loadDemand;
		if ( demand <= 0.0f )
		{
			continue;
		}

		float onset, refineOnset, breakRatio, damageRate, recovery;
		int maxBreaks;
		b2BlastGetBondMaterialResponse( actor, bond, &onset, &refineOnset, &breakRatio, &damageRate, &recovery, &maxBreaks );
		B2_UNUSED( onset );
		B2_UNUSED( breakRatio );
		B2_UNUSED( damageRate );
		B2_UNUSED( recovery );
		B2_UNUSED( maxBreaks );

		const float ratio = demand / b2MaxFloat( bond->capacity, 1.0f );
		if ( ratio <= refineOnset )
		{
			continue;
		}

		const bool refinedA = b2BlastActor_RefineActiveClusterContainingLeaf( actor, bond->leafA );
		const bool refinedB = b2BlastActor_RefineActiveClusterContainingLeaf( actor, bond->leafB );
		if ( refinedA || refinedB )
		{
			bond->flags |= b2_blastBondFlagBreakCandidate;
			anyRefined = true;
			fractureWorld->refinedThisStep += 1;
			fractureWorld->demandRecomputeCount += 1;
			if ( fractureWorld->commandCount < fractureWorld->commandCapacity )
			{
				b2BlastFractureCommand* command = fractureWorld->commands + fractureWorld->commandCount++;
				*command = (b2BlastFractureCommand){ 0 };
				command->kind = b2_blastFractureCommandRefine;
				command->actorId = actor->id;
				command->bondIndex = (uint32_t)i;
				command->value = ratio;
			}
		}
	}
	return anyRefined;
}

static void b2BlastRunDamageShaderBreakPass( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor )
{
	bool anyBreak = false;
	int broken = 0;
	int breakLimit = 0;
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		b2BlastActiveBond* bond = actor->bonds + i;
		if ( ( bond->flags & b2_blastBondFlagBroken ) != 0 )
		{
			continue;
		}
		if ( ( actor->leaves[bond->leafA].flags & b2_blastLeafFlagDetached ) != 0 ||
			 ( actor->leaves[bond->leafB].flags & b2_blastLeafFlagDetached ) != 0 )
		{
			continue;
		}
		float demand = bond->impactDemand + bond->loadDemand;
		if ( demand <= 0.0f )
		{
			float onset, refineOnset, breakRatio, damageRate, recovery;
			int maxBreaks;
			b2BlastGetBondMaterialResponse( actor, bond, &onset, &refineOnset, &breakRatio, &damageRate, &recovery, &maxBreaks );
			B2_UNUSED( onset );
			B2_UNUSED( refineOnset );
			B2_UNUSED( breakRatio );
			B2_UNUSED( damageRate );
			B2_UNUSED( maxBreaks );
			if ( bond->damage > 0.0f && recovery > 0.0f )
			{
				bond->damage = b2MaxFloat( 0.0f, bond->damage - recovery );
			}
			continue;
		}
		float ratio = demand / b2MaxFloat( bond->capacity, 1.0f );
		float onset, refineOnset, breakRatio, damageRate, recovery;
		int maxBreaks;
		b2BlastGetBondMaterialResponse( actor, bond, &onset, &refineOnset, &breakRatio, &damageRate, &recovery, &maxBreaks );
		B2_UNUSED( refineOnset );
		B2_UNUSED( recovery );
		breakLimit = b2MaxInt( breakLimit, maxBreaks );
		const uint32_t activeA = b2BlastActor_FindActiveClusterContainingLeaf( actor, bond->leafA );
		const uint32_t activeB = b2BlastActor_FindActiveClusterContainingLeaf( actor, bond->leafB );
		const bool currentActiveConstraint =
			activeA != UINT32_MAX && activeB != UINT32_MAX && activeA != activeB;
		if ( ratio > onset )
		{
			float added = ( ratio - onset ) * damageRate / b2MaxFloat( 0.45f, bond->toughness );
			if ( bond->impactDemand > 0.0f )
			{
				added *= 1.45f;
			}
			bond->damage = b2ClampFloat( bond->damage + added, 0.0f, 1.5f );
			bond->flags |= b2_blastBondFlagBreakCandidate;
			fractureWorld->maxDamage = b2MaxFloat( fractureWorld->maxDamage, bond->damage );
		}
		if ( currentActiveConstraint && broken < breakLimit && ( ratio > breakRatio || bond->damage >= 1.0f ) )
		{
			bond->flags |= b2_blastBondFlagBroken;
			anyBreak = true;
			broken += 1;
			fractureWorld->brokenThisStep += 1;
			if ( fractureWorld->commandCount < fractureWorld->commandCapacity )
			{
				b2BlastFractureCommand* command = fractureWorld->commands + fractureWorld->commandCount++;
				*command = (b2BlastFractureCommand){ 0 };
				command->kind = b2_blastFractureCommandBreak;
				command->actorId = actor->id;
				command->bondIndex = (uint32_t)i;
				command->value = ratio;
			}
		}
	}

	if ( anyBreak )
	{
		b2BlastClassifyAndCommandSplits( fractureWorld, actor );
	}
}

static void b2BlastRunDamageShader( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor )
{
	if ( fractureWorld == NULL || actor == NULL )
	{
		return;
	}
	for ( int pass = 0; pass < 4; ++pass )
	{
		if ( b2BlastRunDamageShaderRefinePass( fractureWorld, actor ) == false )
		{
			break;
		}
	}
	b2BlastRunDamageShaderBreakPass( fractureWorld, actor );
}

static void b2BlastApplyYieldedContactImpact( b2BlastFractureWorld* fractureWorld, const b2BlastFractureContactSide* side,
											  float impactImpulse, b2Vec2 direction )
{
	if ( fractureWorld == NULL || side == NULL || side->destructible == false || side->actor == NULL || side->leafIndex == B2_NULL_INDEX )
	{
		return;
	}
	if ( impactImpulse > b2_blastConstraintImpulseEpsilon )
	{
		b2BlastApplyImpactWaveFromLeafAtPoint(
			fractureWorld, side->actor, side->leafIndex, side->localPoint, impactImpulse, direction,
			b2_blastDefaultContactDamageRadius );
	}
}

static void b2BlastConsumeContact( b2World* world, b2Contact* contact, float timeStep )
{
	if ( world == NULL || contact == NULL )
	{
		return;
	}

	b2ContactSim* sim = b2GetContactSim( world, contact );
	if ( sim == NULL || sim->manifold.pointCount <= 0 )
	{
		return;
	}

	for ( int pointIndex = 0; pointIndex < sim->manifold.pointCount; ++pointIndex )
	{
		const b2ManifoldPoint* point = sim->manifold.points + pointIndex;
		float normalImpulse = b2MaxFloat( point->totalNormalImpulse, point->normalImpulse );
		float tangentImpulse = point->totalTangentImpulse;
		if ( normalImpulse <= b2_blastConstraintImpulseEpsilon && fabsf( tangentImpulse ) <= b2_blastConstraintImpulseEpsilon &&
			 ( point->yielded == false || point->unresolvedNormalImpulse <= b2_blastConstraintImpulseEpsilon ) )
		{
			continue;
		}

		b2BlastFractureWorld_ConsumeContactConstraintRow( world, sim, pointIndex, sim->manifold.normal, point->anchorA, point->anchorB,
			normalImpulse, tangentImpulse, point->requiredNormalImpulse, point->yieldImpulse, point->normalVelocity, point->normalMass,
			point->unresolvedNormalImpulse, point->yielded, point->persisted, timeStep );
	}
}

void b2BlastFractureWorld_ConsumeContactConstraintRow( b2World* world, const b2ContactSim* contactSim, int pointIndex,
														b2Vec2 normal, b2Vec2 anchorA, b2Vec2 anchorB, float normalImpulse,
														float tangentImpulse, float requiredNormalImpulse, float yieldImpulse,
														float normalVelocity, float normalMass, float unresolvedNormalImpulse,
														bool yielded, bool persisted, float timeStep )
{
	if ( world == NULL || contactSim == NULL || pointIndex < 0 || pointIndex >= contactSim->manifold.pointCount )
	{
		return;
	}

	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	b2BlastFractureContactPair pair;
	if ( b2BlastBuildContactPair( world, contactSim, anchorA, anchorB, normalVelocity, &pair ) == false )
	{
		return;
	}

	float invDt = timeStep > 0.0f ? 1.0f / timeStep : 0.0f;
	if ( normalImpulse <= b2_blastConstraintImpulseEpsilon && fabsf( tangentImpulse ) <= b2_blastConstraintImpulseEpsilon &&
		 ( yielded == false || unresolvedNormalImpulse <= b2_blastConstraintImpulseEpsilon ) )
	{
		return;
	}

	fractureWorld->constraintRowCount += 1;
	fractureWorld->contactPairRowCount += 1;
	b2Vec2 tangent = b2RightPerp( normal );
	B2_UNUSED( yieldImpulse );
	B2_UNUSED( normalMass );
	B2_UNUSED( persisted );
	B2_UNUSED( yielded );
	B2_UNUSED( unresolvedNormalImpulse );

	const float contactImpulse = b2MaxFloat( normalImpulse, requiredNormalImpulse );
	const float damageImpulse = b2MaxFloat( 0.0f, contactImpulse - pair.yieldImpulse );
	const float approachSpeed = b2MaxFloat( 0.0f, -normalVelocity );
	const float impactSpeedThreshold = b2MaxFloat( 0.0f, world->restitutionThreshold );
	if ( damageImpulse > b2_blastConstraintImpulseEpsilon && approachSpeed > impactSpeedThreshold )
	{
		fractureWorld->contactImpactBudgetRowCount += 1;
		b2Vec2 contactVector = b2Add( b2MulSV( normalImpulse, normal ), b2MulSV( tangentImpulse, tangent ) );
		if ( b2LengthSquared( contactVector ) <= 0.000001f )
		{
			contactVector = normal;
		}
		b2BlastApplyYieldedContactImpact(
			fractureWorld, pair.sides + 0, damageImpulse * pair.damageShare[0], b2Neg( contactVector ) );
		b2BlastApplyYieldedContactImpact(
			fractureWorld, pair.sides + 1, damageImpulse * pair.damageShare[1], contactVector );
	}
	else if ( normalImpulse > b2_blastConstraintImpulseEpsilon || fabsf( tangentImpulse ) > b2_blastConstraintImpulseEpsilon )
	{
		fractureWorld->contactLoadBudgetRowCount += 1;
	}
	B2_UNUSED( invDt );
}

static b2BlastActor* b2BlastFindFirstBodyActorForConstraintRow( b2World* world, b2Body* body, b2Shape** outShape )
{
	if ( outShape != NULL )
	{
		*outShape = NULL;
	}
	if ( world == NULL || body == NULL )
	{
		return NULL;
	}
	b2BlastActor* bodyActor = b2BlastGetActor( &world->blastFractureWorld, body->blastActorId );
	if ( bodyActor != NULL )
	{
		if ( outShape == NULL )
		{
			return bodyActor;
		}
		if ( bodyActor->shapeId != B2_NULL_INDEX )
		{
			b2Shape* shape = b2ShapeArray_Get( &world->shapes, bodyActor->shapeId );
			if ( shape != NULL && b2BlastActorIdEqual( shape->blastActorId, bodyActor->id ) )
			{
				*outShape = shape;
				return bodyActor;
			}
		}
	}
	for ( int shapeId = body->headShapeId; shapeId != B2_NULL_INDEX; )
	{
		b2Shape* shape = b2ShapeArray_Get( &world->shapes, shapeId );
		b2BlastActor* actor = b2BlastGetActor( &world->blastFractureWorld, shape->blastActorId );
		if ( actor != NULL )
		{
			if ( outShape != NULL )
			{
				*outShape = shape;
			}
			return actor;
		}
		shapeId = shape->nextShapeId;
	}
	return NULL;
}

static void b2BlastConsumeJointEndpointLoad( b2World* world, b2Body* body, b2Shape* shape, b2BlastActor* actor, b2Vec2 localPoint,
											 b2Vec2 force )
{
	if ( world == NULL || body == NULL || actor == NULL || shape == NULL || b2LengthSquared( force ) <= 0.000001f )
	{
		return;
	}
	if ( ( actor->flags & b2_blastActorFlagOwnsWorldAnchor ) == 0 )
	{
		return;
	}
	b2Transform transform = b2GetBodyTransformQuick( world, body );
	int leaf = b2BlastFindEndpointLeaf( &world->blastFractureWorld, actor, shape, localPoint, true );
	if ( leaf == B2_NULL_INDEX )
	{
		return;
	}
	b2BlastApplyLoadBudgetFromLeaf( &world->blastFractureWorld, actor, leaf, b2Length( force ), force );
}

static bool b2BlastRecordBodyInput(
	b2World* world, b2Body* body, b2BlastBodyInputKind kind, b2Vec2 worldPoint, b2Vec2 vector, bool useBodyCenter )
{
	if ( world == NULL || body == NULL || b2LengthSquared( vector ) <= 0.000001f )
	{
		return false;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	b2BlastEnsureBodyInputCapacity( fractureWorld, fractureWorld->bodyInputCount + 1 );
	b2BlastBodyInputRecord* record = fractureWorld->bodyInputs + fractureWorld->bodyInputCount++;
	*record = (b2BlastBodyInputRecord){ 0 };
	record->kind = kind;
	record->bodyId = (b2BodyId){ body->id + 1, world->worldId, body->generation };
	record->worldPoint = worldPoint;
	record->vector = vector;
	record->useBodyCenter = useBodyCenter;
	return true;
}

bool b2BlastFractureWorld_RecordBodyForce(
	b2World* world, b2Body* body, b2Vec2 worldPoint, b2Vec2 force, bool useBodyCenter )
{
	return b2BlastRecordBodyInput( world, body, b2_blastBodyInputForce, worldPoint, force, useBodyCenter );
}

bool b2BlastFractureWorld_RecordBodyLinearImpulse(
	b2World* world, b2Body* body, b2Vec2 worldPoint, b2Vec2 impulse, bool useBodyCenter )
{
	return b2BlastRecordBodyInput( world, body, b2_blastBodyInputImpulse, worldPoint, impulse, useBodyCenter );
}

bool b2BlastFractureWorld_RecordIgnoredBodyTorque( b2World* world, b2Body* body )
{
	if ( world == NULL || body == NULL || body->setIndex == b2_disabledSet )
	{
		return false;
	}
	b2Shape* shape = NULL;
	b2BlastActor* actor = b2BlastFindFirstBodyActorForConstraintRow( world, body, &shape );
	if ( actor == NULL )
	{
		return false;
	}
	world->blastFractureWorld.pendingTorqueInputIgnoredCount += 1;
	return true;
}

static bool b2BlastConsumeBodyInputRecord( b2World* world, const b2BlastBodyInputRecord* record )
{
	if ( world == NULL || record == NULL || record->bodyId.world0 != world->worldId || b2Body_IsValid( record->bodyId ) == false ||
		 b2LengthSquared( record->vector ) <= 0.000001f )
	{
		return false;
	}

	b2Body* body = b2GetBodyFullId( world, record->bodyId );
	if ( body == NULL || body->setIndex == b2_disabledSet )
	{
		return false;
	}

	b2Shape* shape = NULL;
	b2BlastActor* actor = b2BlastFindFirstBodyActorForConstraintRow( world, body, &shape );
	if ( actor == NULL || shape == NULL )
	{
		return false;
	}

	b2Transform transform = b2GetBodyTransformQuick( world, body );
	b2Vec2 worldPoint = record->worldPoint;
	if ( record->useBodyCenter )
	{
		b2BodySim* bodySim = b2GetBodySim( world, body );
		worldPoint = bodySim->center;
	}
	b2Vec2 localPoint = b2InvTransformPoint( transform, worldPoint );
	int leaf = b2BlastFindEndpointLeaf( &world->blastFractureWorld, actor, shape, localPoint, false );
	if ( leaf == B2_NULL_INDEX )
	{
		world->blastFractureWorld.ignoredOffTargetEventCount += 1;
		return false;
	}

	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	fractureWorld->constraintRowCount += 1;
	if ( record->kind == b2_blastBodyInputImpulse )
	{
		fractureWorld->appliedImpulseImpactRowCount += 1;
		b2BlastApplyImpactWaveFromLeafAtPoint(
			fractureWorld, actor, leaf, localPoint, b2Length( record->vector ), record->vector,
			b2_blastDefaultContactDamageRadius );
	}
	else
	{
		fractureWorld->appliedForceLoadRowCount += 1;
		b2BlastApplyLoadBudgetFromLeaf( fractureWorld, actor, leaf, b2Length( record->vector ), record->vector );
	}
	return true;
}

static void b2BlastFractureWorld_ConsumePendingBodyInputs( b2World* world )
{
	if ( world == NULL )
	{
		return;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	const int inputCount = fractureWorld->bodyInputCount;
	fractureWorld->bodyInputCount = 0;
	for ( int i = 0; i < inputCount; ++i )
	{
		(void)b2BlastConsumeBodyInputRecord( world, fractureWorld->bodyInputs + i );
	}
}

void b2BlastFractureWorld_ConsumePendingBodyInputsForCompatibility( b2World* world )
{
	if ( world == NULL )
	{
		return;
	}
	b2BlastFractureWorld_BeginStep( world );
	b2BlastFractureWorld_BeginSubstep( world );
	b2BlastFractureWorld_EndSubstep( world );
	if ( world->locked == false )
	{
		b2BlastFractureWorld_EndStep( world );
	}
}

void b2BlastFractureWorld_ConsumeJointConstraintRows( b2World* world, float timeStep )
{
	if ( world == NULL || timeStep <= 0.0f )
	{
		return;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	for ( int i = 0; i < world->joints.count; ++i )
	{
		b2Joint* joint = b2JointArray_Get( &world->joints, i );
		if ( joint == NULL || joint->setIndex == B2_NULL_INDEX || joint->type != b2_targetPointJoint )
		{
			continue;
		}
		b2JointSim* sim = b2GetJointSim( world, joint );
		if ( sim == NULL )
		{
			continue;
		}
		b2Vec2 forceOnB = b2GetTargetPointJointForce( world, sim );
		if ( b2LengthSquared( forceOnB ) <= 0.000001f )
		{
			continue;
		}
		b2Body* bodyA = b2BodyArray_Get( &world->bodies, sim->bodyIdA );
		b2Body* bodyB = b2BodyArray_Get( &world->bodies, sim->bodyIdB );
		b2Shape* shapeA = NULL;
		b2Shape* shapeB = NULL;
		b2BlastActor* actorA = b2BlastFindFirstBodyActorForConstraintRow( world, bodyA, &shapeA );
		b2BlastActor* actorB = b2BlastFindFirstBodyActorForConstraintRow( world, bodyB, &shapeB );
		if ( actorA == NULL && actorB == NULL )
		{
			continue;
		}
		fractureWorld->constraintRowCount += 1;
		fractureWorld->jointConstraintRowCount += 1;
		b2BlastConsumeJointEndpointLoad( world, bodyA, shapeA, actorA, sim->localFrameA.p, b2Neg( forceOnB ) );
		b2BlastConsumeJointEndpointLoad( world, bodyB, shapeB, actorB, sim->localFrameB.p, forceOnB );
	}
}

void b2BlastFractureWorld_BeginStep( b2World* world )
{
	if ( world == NULL )
	{
		return;
	}
	b2BlastFractureWorld_ClearStep( &world->blastFractureWorld );
}

void b2BlastFractureWorld_BeginSubstep( b2World* world )
{
	if ( world == NULL )
	{
		return;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	for ( int i = 0; i < fractureWorld->actorCount; ++i )
	{
		if ( ( fractureWorld->actors[i].flags & b2_blastActorFlagInUse ) != 0 )
		{
			b2BlastActor_ClearRuntimeDemand( fractureWorld->actors + i );
		}
	}
	b2BlastFractureWorld_ConsumePendingBodyInputs( world );
}

void b2BlastFractureWorld_EndSubstep( b2World* world )
{
	if ( world == NULL )
	{
		return;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	fractureWorld->substepSolveCount += 1;
	for ( int i = 0; i < fractureWorld->actorCount; ++i )
	{
		b2BlastActor* actor = fractureWorld->actors + i;
		if ( ( actor->flags & b2_blastActorFlagInUse ) == 0 )
		{
			continue;
		}
		b2BlastRunDamageShader( fractureWorld, actor );
		fractureWorld->scratchCapacityHighWater =
			b2MaxInt( (int)fractureWorld->scratchCapacityHighWater,
				actor->componentScratchCapacity + actor->queueScratchCapacity + actor->assignScratchCapacity +
					actor->cellScratchCapacity + actor->graphDistanceScratchCapacity + actor->graphParentBondScratchCapacity );
	}
}

static bool b2BlastBuildTransitionPixelAsset( b2BlastActor* child, const b2BlastActor* source, const b2Shape* sourceShape,
	const b2BlastActorTransition* transition, const int32_t* transitionCells )
{
	if ( child == NULL || source == NULL || sourceShape == NULL || sourceShape->type != b2_pixelShape ||
		 b2IsPixelShapeUsable( &sourceShape->pixel ) == false || transition == NULL || transitionCells == NULL ||
		 transition->cellCount <= 0 )
	{
		return false;
	}

	const b2PixelAsset* sourceAsset = sourceShape->pixel.asset;
	const int childWidth = transition->sourceMaxX - transition->sourceMinX + 1;
	const int childHeight = transition->sourceMaxY - transition->sourceMinY + 1;
	if ( childWidth <= 0 || childHeight <= 0 ||
		 b2BlastEnsureOwnedPixelAssetCapacity( child, childWidth, childHeight ) == false )
	{
		return false;
	}

	const int childCellCount = childWidth * childHeight;
	const int childWordCount = ( childCellCount + 63 ) / 64;
	memset( child->ownedOccupancyBits, 0, (size_t)childWordCount * sizeof( uint64_t ) );
	memset( child->ownedMaterialIds, 0, (size_t)childCellCount * sizeof( b2BlastMaterialId ) );

	for ( int i = 0; i < transition->cellCount; ++i )
	{
		int sourceCell = transitionCells[transition->cellOffset + i];
		if ( sourceCell < 0 || sourceCell >= source->cellToLeafCount )
		{
			continue;
		}
		int sx = sourceCell % sourceAsset->width;
		int sy = sourceCell / sourceAsset->width;
		int cx = sx - transition->sourceMinX;
		int cy = sy - transition->sourceMinY;
		if ( cx < 0 || cy < 0 || cx >= childWidth || cy >= childHeight )
		{
			continue;
		}
		int childCell = cy * childWidth + cx;
		child->ownedOccupancyBits[childCell >> 6] |= UINT64_C( 1 ) << ( childCell & 63 );
		child->ownedMaterialIds[childCell] = b2PixelAsset_GetMaterialId( sourceAsset, sx, sy );
	}

	b2PixelAssetBuildConfig config = b2DefaultPixelAssetBuildConfig();
	config.width = childWidth;
	config.height = childHeight;
	config.pixelSize = sourceAsset->pixelSize;
	config.supportCornerInterval = 0;
	config.topologyVersion = sourceAsset->topologyVersion + transition->transitionId;
	config.materialIds = child->ownedMaterialIds;
	config.materialIdCount = childCellCount;
	config.materialTable = sourceAsset->materialTable;
	config.materialHash = sourceAsset->materialHash;
	b2PixelAssetBuildBuffers buffers = { 0 };
	buffers.occupancyBits = child->ownedOccupancyBits;
	buffers.occupancyWordCapacity = child->ownedOccupancyWordCapacity;
	buffers.materialIds = child->ownedMaterialIds;
	buffers.materialIdCapacity = child->ownedMaterialIdCapacity;
	buffers.featureTypes = child->ownedFeatureTypes;
	buffers.featureTypeCapacity = child->ownedFeatureTypeCapacity;
	buffers.normalIndices = child->ownedNormalIndices;
	buffers.normalIndexCapacity = child->ownedNormalIndexCapacity;
	buffers.corners = child->ownedCorners;
	buffers.cornerCapacity = child->ownedCornerCapacity;
	buffers.edges = child->ownedEdges;
	buffers.edgeCapacity = child->ownedEdgeCapacity;
	buffers.rowSolidCounts = child->ownedRowSolidCounts;
	buffers.rowSolidCountCapacity = child->ownedRowSolidCountCapacity;
	buffers.colSolidCounts = child->ownedColSolidCounts;
	buffers.colSolidCountCapacity = child->ownedColSolidCountCapacity;
	buffers.scratchCells = child->ownedPixelScratch;
	buffers.scratchCellCapacity = child->ownedPixelScratchCapacity;
	b2PixelAssetBuildResult result =
		b2BuildPixelAssetFromOccupancy( &config, child->ownedOccupancyBits, childWordCount, &buffers );
	if ( result.success == false )
	{
		return false;
	}

	child->ownedAsset = result.asset;
	child->flags |= b2_blastActorFlagOwnsPixelAsset;
	return true;
}

static void b2BlastActor_ActivateLeafGraphForTransitionChild( b2BlastActor* child )
{
	if ( child == NULL || child->leafCount <= 0 )
	{
		return;
	}

	child->activeClusterCount = 0;
	child->initialActiveLevel = 0;
	for ( int leafIndex = 0; leafIndex < child->leafCount; ++leafIndex )
	{
		const b2BlastLeaf* leaf = child->leaves + leafIndex;
		if ( leaf->cellCount <= 0 )
		{
			continue;
		}
		(void)b2BlastActor_AddActiveCluster( child, (uint32_t)leafIndex );
	}
	b2BlastActor_SortUniqueActiveClusters( child );
}

static void b2BlastCluster_RecomputeFromLeaves( b2BlastActor* actor, b2BlastCluster* cluster )
{
	if ( actor == NULL || cluster == NULL || cluster->firstLeaf == UINT32_MAX ||
		 cluster->firstLeaf + cluster->leafCount > (uint32_t)actor->clusterLeafRefCount )
	{
		return;
	}

	cluster->minX = UINT16_MAX;
	cluster->minY = UINT16_MAX;
	cluster->maxX = 0;
	cluster->maxY = 0;
	cluster->centroid = b2Vec2_zero;
	cluster->mass = 0.0f;
	cluster->flags = 0;
	for ( uint32_t ref = 0; ref < cluster->leafCount; ++ref )
	{
		const uint32_t leafIndex = actor->clusterLeaves[cluster->firstLeaf + ref];
		if ( leafIndex >= (uint32_t)actor->leafCount )
		{
			continue;
		}
		const b2BlastLeaf* leaf = actor->leaves + leafIndex;
		if ( leaf->cellCount == 0 )
		{
			continue;
		}
		cluster->minX = (uint16_t)b2MinInt( cluster->minX, leaf->minX );
		cluster->minY = (uint16_t)b2MinInt( cluster->minY, leaf->minY );
		cluster->maxX = (uint16_t)b2MaxInt( cluster->maxX, leaf->maxX );
		cluster->maxY = (uint16_t)b2MaxInt( cluster->maxY, leaf->maxY );
		cluster->mass += leaf->mass;
		cluster->centroid.x += leaf->centroid.x * leaf->mass;
		cluster->centroid.y += leaf->centroid.y * leaf->mass;
	}
	if ( cluster->mass > 0.0f )
	{
		cluster->centroid.x /= cluster->mass;
		cluster->centroid.y /= cluster->mass;
	}
}

static b2BlastActiveBond* b2BlastActor_FindBondByLeaves( b2BlastActor* actor, uint32_t leafA, uint32_t leafB )
{
	if ( actor == NULL || leafA == leafB )
	{
		return NULL;
	}
	if ( leafB < leafA )
	{
		uint32_t temp = leafA;
		leafA = leafB;
		leafB = temp;
	}
	for ( int bondIndex = 0; bondIndex < actor->bondCount; ++bondIndex )
	{
		b2BlastActiveBond* bond = actor->bonds + bondIndex;
		if ( bond->leafA == leafA && bond->leafB == leafB )
		{
			return bond;
		}
	}
	return NULL;
}

static bool b2BlastActor_MigrateTransitionChildGraphFromSource( b2BlastActor* child, const b2BlastActor* source,
															   const b2Shape* childShape,
															   const b2BlastActorTransition* transition,
															   const int32_t* transitionCells )
{
	if ( child == NULL || source == NULL || childShape == NULL || childShape->type != b2_pixelShape ||
		 b2IsPixelShapeUsable( &childShape->pixel ) == false || transition == NULL || transitionCells == NULL ||
		 transition->cellCount <= 0 || source->cellToLeaf == NULL || source->leafCount <= 0 )
	{
		return false;
	}

	const b2PixelAsset* childAsset = childShape->pixel.asset;
	const int childWidth = childAsset->width;
	const int childHeight = childAsset->height;
	const int childCellCount = childWidth * childHeight;
	if ( childWidth <= 0 || childHeight <= 0 || childCellCount <= 0 )
	{
		return false;
	}

	const int maxLeaves = b2MaxInt( 1, childAsset->solidCount );
	const int maxBonds = b2MaxInt( 1, source->bondCount + childAsset->solidCount * 2 );
	const int maxClusters = b2MaxInt( maxLeaves + source->clusterCount + 1, maxLeaves * 2 + 8 );
	const int maxClusterRefs = b2MaxInt( maxClusters * 4, source->clusterLeafRefCount + source->clusterChildCount + maxLeaves );
	const int remapCapacity = b2MaxInt( source->leafCount, maxLeaves );
	b2BlastEnsureActorCapacity( child, maxLeaves, maxBonds, maxClusters, maxClusterRefs, childCellCount, remapCapacity );
	if ( child->leafRemapScratchCapacity < source->leafCount || child->cellToLeafCapacity < childCellCount )
	{
		return false;
	}

	for ( int i = 0; i < source->leafCount; ++i )
	{
		child->leafRemapScratch[i] = UINT32_MAX;
	}
	for ( int cell = 0; cell < childCellCount; ++cell )
	{
		child->cellToLeaf[cell] = UINT32_MAX;
	}

	child->leafCount = 0;
	child->bondCount = 0;
	child->clusterCount = 0;
	child->clusterChildCount = 0;
	child->clusterLeafRefCount = 0;
	child->activeClusterCount = 0;
	child->worldAnchorCount = 0;
	child->flags &= (uint32_t)~b2_blastActorFlagOwnsWorldAnchor;
	child->rootCluster = UINT32_MAX;
	child->cellToLeafCount = childCellCount;

	for ( int i = 0; i < transition->cellCount; ++i )
	{
		const int sourceCell = transitionCells[transition->cellOffset + i];
		if ( sourceCell < 0 || sourceCell >= source->cellToLeafCount )
		{
			continue;
		}
		const uint32_t sourceLeaf = source->cellToLeaf[sourceCell];
		if ( sourceLeaf == UINT32_MAX || sourceLeaf >= (uint32_t)source->leafCount )
		{
			continue;
		}
		if ( child->leafRemapScratch[sourceLeaf] == UINT32_MAX )
		{
			if ( child->leafCount >= child->leafCapacity )
			{
				return false;
			}
			const uint32_t childLeaf = (uint32_t)child->leafCount++;
			child->leafRemapScratch[sourceLeaf] = childLeaf;
			child->leaves[childLeaf] = (b2BlastLeaf){ 0 };
			child->leaves[childLeaf].dominantMaterialId = source->leaves[sourceLeaf].dominantMaterialId;
			child->leaves[childLeaf].firstCell = UINT32_MAX;
			child->leaves[childLeaf].firstBond = UINT32_MAX;
			child->leaves[childLeaf].minX = UINT16_MAX;
			child->leaves[childLeaf].minY = UINT16_MAX;
		}

		const int sx = sourceCell % transition->sourceWidth;
		const int sy = sourceCell / transition->sourceWidth;
		const int cx = sx - transition->sourceMinX;
		const int cy = sy - transition->sourceMinY;
		if ( cx < 0 || cy < 0 || cx >= childWidth || cy >= childHeight )
		{
			continue;
		}
		child->cellToLeaf[cy * childWidth + cx] = child->leafRemapScratch[sourceLeaf];
	}
	if ( child->leafCount <= 0 )
	{
		return false;
	}

	const float cellArea = childAsset->pixelSize * childAsset->pixelSize;
	for ( int leafIndex = 0; leafIndex < child->leafCount; ++leafIndex )
	{
		b2BlastLeaf* leaf = child->leaves + leafIndex;
		const b2BlastMaterialId preservedMaterial = leaf->dominantMaterialId;
		*leaf = (b2BlastLeaf){ 0 };
		leaf->dominantMaterialId = preservedMaterial;
		leaf->firstCell = UINT32_MAX;
		leaf->firstBond = UINT32_MAX;
		leaf->minX = UINT16_MAX;
		leaf->minY = UINT16_MAX;
		uint32_t materialVotes[8] = { 0 };
		b2BlastMaterialId materialVoteIds[8] = { 0 };
		float cx = 0.0f;
		float cy = 0.0f;
		float mass = 0.0f;
		for ( int cell = 0; cell < childCellCount; ++cell )
		{
			if ( child->cellToLeaf[cell] != (uint32_t)leafIndex )
			{
				continue;
			}
			const int x = cell % childWidth;
			const int y = cell / childWidth;
			if ( b2PixelAsset_IsOccupied( childAsset, x, y ) == false )
			{
				child->cellToLeaf[cell] = UINT32_MAX;
				continue;
			}
			b2BlastMaterialId materialId = b2PixelAsset_GetMaterialId( childAsset, x, y );
			float cellMass = b2BlastMaterialDensity( childAsset, materialId, childShape->density ) * cellArea;
			b2Vec2 center = b2BlastCellCenter( childAsset, childShape, x, y );
			cx += center.x * cellMass;
			cy += center.y * cellMass;
			mass += cellMass;
			leaf->cellCount += 1;
			leaf->firstCell = leaf->firstCell == UINT32_MAX ? (uint32_t)cell : leaf->firstCell;
			leaf->minX = (uint16_t)b2MinInt( leaf->minX, x );
			leaf->minY = (uint16_t)b2MinInt( leaf->minY, y );
			leaf->maxX = (uint16_t)b2MaxInt( leaf->maxX, x );
			leaf->maxY = (uint16_t)b2MaxInt( leaf->maxY, y );
			bool voted = false;
			for ( int vote = 0; vote < 8; ++vote )
			{
				if ( materialVotes[vote] > 0 && materialVoteIds[vote] == materialId )
				{
					materialVotes[vote] += 1;
					voted = true;
					break;
				}
			}
			if ( voted == false )
			{
				for ( int vote = 0; vote < 8; ++vote )
				{
					if ( materialVotes[vote] == 0 )
					{
						materialVoteIds[vote] = materialId;
						materialVotes[vote] = 1;
						break;
					}
				}
			}
		}
		if ( leaf->cellCount == 0 )
		{
			return false;
		}
		uint32_t bestVotes = 0;
		for ( int vote = 0; vote < 8; ++vote )
		{
			if ( materialVotes[vote] > bestVotes )
			{
				bestVotes = materialVotes[vote];
				leaf->dominantMaterialId = materialVoteIds[vote];
			}
		}
		leaf->mass = mass;
		leaf->centroid = mass > 0.0f ? (b2Vec2){ cx / mass, cy / mass } : b2Vec2_zero;
	}

	for ( int sourceBondIndex = 0; sourceBondIndex < source->bondCount; ++sourceBondIndex )
	{
		const b2BlastActiveBond* sourceBond = source->bonds + sourceBondIndex;
		if ( sourceBond->leafA >= (uint32_t)source->leafCount || sourceBond->leafB >= (uint32_t)source->leafCount )
		{
			continue;
		}
		uint32_t leafA = child->leafRemapScratch[sourceBond->leafA];
		uint32_t leafB = child->leafRemapScratch[sourceBond->leafB];
		if ( leafA == UINT32_MAX || leafB == UINT32_MAX || leafA == leafB )
		{
			continue;
		}
		if ( leafB < leafA )
		{
			uint32_t temp = leafA;
			leafA = leafB;
			leafB = temp;
		}
		if ( child->bondCount >= child->bondCapacity )
		{
			return false;
		}
		b2BlastActiveBond bond = *sourceBond;
		bond.leafA = leafA;
		bond.leafB = leafB;
		bond.clusterA = leafA;
		bond.clusterB = leafB;
		bond.area = 0.0f;
		child->bonds[child->bondCount++] = bond;
	}

	for ( int y = 0; y < childHeight; ++y )
	{
		for ( int x = 0; x < childWidth; ++x )
		{
			const int indexA = y * childWidth + x;
			if ( child->cellToLeaf[indexA] == UINT32_MAX )
			{
				continue;
			}
			const int nx[2] = { x + 1, x };
			const int ny[2] = { y, y + 1 };
			for ( int dir = 0; dir < 2; ++dir )
			{
				if ( nx[dir] >= childWidth || ny[dir] >= childHeight )
				{
					continue;
				}
				const int indexB = ny[dir] * childWidth + nx[dir];
				uint32_t leafA = child->cellToLeaf[indexA];
				uint32_t leafB = child->cellToLeaf[indexB];
				if ( leafB == UINT32_MAX || leafA == leafB )
				{
					continue;
				}
				if ( leafB < leafA )
				{
					uint32_t temp = leafA;
					leafA = leafB;
					leafB = temp;
				}
				b2BlastActiveBond* bond = b2BlastActor_FindBondByLeaves( child, leafA, leafB );
				if ( bond == NULL )
				{
					if ( child->bondCount >= child->bondCapacity )
					{
						return false;
					}
					bond = child->bonds + child->bondCount++;
					*bond = (b2BlastActiveBond){ 0 };
					bond->leafA = leafA;
					bond->leafB = leafB;
					bond->clusterA = leafA;
					bond->clusterB = leafB;
					bond->materialMix =
						child->leaves[leafA].dominantMaterialId == child->leaves[leafB].dominantMaterialId ? 0 : 1;
				}
				bond->area += childAsset->pixelSize;
			}
		}
	}

	for ( int bondIndex = 0; bondIndex < child->bondCount; ++bondIndex )
	{
		b2BlastActiveBond* bond = child->bonds + bondIndex;
		if ( bond->leafA >= (uint32_t)child->leafCount || bond->leafB >= (uint32_t)child->leafCount )
		{
			continue;
		}
		if ( bond->area > 0.0f )
		{
			b2BlastMaterialId materialA = child->leaves[bond->leafA].dominantMaterialId;
			b2BlastMaterialId materialB = child->leaves[bond->leafB].dominantMaterialId;
			bond->capacity = b2BlastBondCapacity( childAsset, materialA, materialB, bond->area );
			bond->toughness = b2MaxFloat( bond->toughness, 0.45f * bond->capacity );
			bond->propagationWeight = 1.0f / b2MaxFloat( 1.0f, bond->area );
		}
		bond->damage = b2ClampFloat( bond->damage, 0.0f, 1.5f );
		bond->clusterA = bond->leafA;
		bond->clusterB = bond->leafB;
	}

	for ( int leafIndex = 0; leafIndex < child->leafCount; ++leafIndex )
	{
		child->leaves[leafIndex].firstBond = UINT32_MAX;
		child->leaves[leafIndex].bondCount = 0;
	}
	for ( int bondIndex = 0; bondIndex < child->bondCount; ++bondIndex )
	{
		b2BlastActiveBond* bond = child->bonds + bondIndex;
		if ( bond->leafA >= (uint32_t)child->leafCount || bond->leafB >= (uint32_t)child->leafCount )
		{
			continue;
		}
		b2BlastLeaf* a = child->leaves + bond->leafA;
		b2BlastLeaf* b = child->leaves + bond->leafB;
		a->firstBond = a->firstBond == UINT32_MAX ? (uint32_t)bondIndex : (uint32_t)b2MinInt( (int)a->firstBond, bondIndex );
		b->firstBond = b->firstBond == UINT32_MAX ? (uint32_t)bondIndex : (uint32_t)b2MinInt( (int)b->firstBond, bondIndex );
		a->bondCount += 1;
		b->bondCount += 1;
	}

	uint32_t* clusterRemap = NULL;
	if ( source->clusterCount > 0 )
	{
		clusterRemap = b2AllocZeroInit( source->clusterCount * (int)sizeof( uint32_t ) );
	}
	if ( clusterRemap == NULL )
	{
		return false;
	}
	for ( int i = 0; i < source->clusterCount; ++i )
	{
		clusterRemap[i] = UINT32_MAX;
	}

	child->clusterCount = 0;
	child->clusterChildCount = 0;
	child->clusterLeafRefCount = 0;
	for ( int leafIndex = 0; leafIndex < child->leafCount; ++leafIndex )
	{
		if ( child->clusterCount >= child->clusterCapacity || child->clusterLeafRefCount >= child->clusterLeafRefCapacity )
		{
			b2Free( clusterRemap, source->clusterCount * (int)sizeof( uint32_t ) );
			return false;
		}
		b2BlastCluster cluster = { 0 };
		cluster.id = (uint32_t)child->clusterCount;
		cluster.parent = UINT32_MAX;
		cluster.firstChild = UINT32_MAX;
		cluster.firstLeaf = (uint32_t)child->clusterLeafRefCount;
		cluster.leafCount = 1;
		cluster.level = 0;
		child->clusterLeaves[child->clusterLeafRefCount++] = (uint32_t)leafIndex;
		b2BlastCluster_RecomputeFromLeaves( child, &cluster );
		child->clusters[child->clusterCount++] = cluster;
	}
	for ( int sourceLeaf = 0; sourceLeaf < source->leafCount && sourceLeaf < source->clusterCount; ++sourceLeaf )
	{
		if ( child->leafRemapScratch[sourceLeaf] != UINT32_MAX )
		{
			clusterRemap[sourceLeaf] = child->leafRemapScratch[sourceLeaf];
		}
	}

	for ( int sourceClusterIndex = 0; sourceClusterIndex < source->clusterCount; ++sourceClusterIndex )
	{
		const b2BlastCluster* sourceCluster = source->clusters + sourceClusterIndex;
		if ( sourceCluster->level == 0 || sourceCluster->firstLeaf == UINT32_MAX ||
			 sourceCluster->firstLeaf + sourceCluster->leafCount > (uint32_t)source->clusterLeafRefCount )
		{
			continue;
		}
		const int leafStart = child->clusterLeafRefCount;
		for ( uint32_t ref = 0; ref < sourceCluster->leafCount; ++ref )
		{
			const uint32_t sourceLeaf = source->clusterLeaves[sourceCluster->firstLeaf + ref];
			if ( sourceLeaf < (uint32_t)source->leafCount && child->leafRemapScratch[sourceLeaf] != UINT32_MAX )
			{
				if ( child->clusterLeafRefCount >= child->clusterLeafRefCapacity )
				{
					b2Free( clusterRemap, source->clusterCount * (int)sizeof( uint32_t ) );
					return false;
				}
				child->clusterLeaves[child->clusterLeafRefCount++] = child->leafRemapScratch[sourceLeaf];
			}
		}
		const int leafCount = child->clusterLeafRefCount - leafStart;
		if ( leafCount <= 0 )
		{
			child->clusterLeafRefCount = leafStart;
			continue;
		}
		if ( child->clusterCount >= child->clusterCapacity )
		{
			b2Free( clusterRemap, source->clusterCount * (int)sizeof( uint32_t ) );
			return false;
		}

		b2BlastCluster cluster = { 0 };
		cluster.id = (uint32_t)child->clusterCount;
		cluster.parent = UINT32_MAX;
		cluster.firstChild = UINT32_MAX;
		cluster.firstLeaf = (uint32_t)leafStart;
		cluster.leafCount = (uint32_t)leafCount;
		cluster.level = sourceCluster->level;
		if ( sourceCluster->childCount > 0 && sourceCluster->firstChild != UINT32_MAX )
		{
			const int childStart = child->clusterChildCount;
			for ( uint32_t childRef = 0; childRef < sourceCluster->childCount; ++childRef )
			{
				if ( sourceCluster->firstChild + childRef >= (uint32_t)source->clusterChildCount )
				{
					continue;
				}
				const uint32_t sourceChildCluster = source->clusterChildren[sourceCluster->firstChild + childRef];
				if ( sourceChildCluster < (uint32_t)source->clusterCount && clusterRemap[sourceChildCluster] != UINT32_MAX )
				{
					if ( child->clusterChildCount >= child->clusterChildCapacity )
					{
						b2Free( clusterRemap, source->clusterCount * (int)sizeof( uint32_t ) );
						return false;
					}
					const uint32_t childCluster = clusterRemap[sourceChildCluster];
					child->clusterChildren[child->clusterChildCount++] = childCluster;
					if ( childCluster < (uint32_t)child->clusterCount )
					{
						child->clusters[childCluster].parent = cluster.id;
					}
				}
			}
			if ( child->clusterChildCount > childStart )
			{
				cluster.firstChild = (uint32_t)childStart;
				cluster.childCount = (uint32_t)( child->clusterChildCount - childStart );
			}
		}
		b2BlastCluster_RecomputeFromLeaves( child, &cluster );
		child->clusters[child->clusterCount++] = cluster;
		if ( sourceCluster->id < (uint32_t)source->clusterCount )
		{
			clusterRemap[sourceCluster->id] = cluster.id;
		}
	}

	child->rootCluster = UINT32_MAX;
	if ( source->rootCluster < (uint32_t)source->clusterCount && clusterRemap[source->rootCluster] != UINT32_MAX )
	{
		child->rootCluster = clusterRemap[source->rootCluster];
	}
	else if ( child->clusterCount > 0 )
	{
		child->rootCluster = child->clusters[child->clusterCount - 1].id;
	}

	child->activeClusterCount = 0;
	child->initialActiveLevel = source->initialActiveLevel;
	for ( int activeIndex = 0; activeIndex < source->activeClusterCount; ++activeIndex )
	{
		const uint32_t sourceActive = source->activeClusters[activeIndex];
		if ( sourceActive < (uint32_t)source->clusterCount && clusterRemap[sourceActive] != UINT32_MAX )
		{
			(void)b2BlastActor_AddActiveCluster( child, clusterRemap[sourceActive] );
		}
	}
	if ( child->activeClusterCount == 0 || ( child->activeClusterCount <= 1 && child->leafCount > 1 ) )
	{
		const b2BlastAuthoringParams params = b2BlastAuthoringParamsFromActor( child );
		b2BlastActor_ResetActiveClustersByArea( child, &params );
		if ( child->activeClusterCount == 0 )
		{
			b2BlastActor_ActivateLeafGraphForTransitionChild( child );
		}
	}
	else
	{
		b2BlastActor_SortUniqueActiveClusters( child );
	}
	b2Free( clusterRemap, source->clusterCount * (int)sizeof( uint32_t ) );
	child->authoringHash = b2BlastComputeActorHash( child );
	child->revision += 1;
	child->flags &= (uint32_t)~b2_blastActorFlagDirtyGraph;
	return true;
}

static bool b2BlastTransitionCellBelongsToSourcePrune(
	const b2BlastFractureWorld* fractureWorld, const b2BlastActorTransition* transition, b2BlastFractureActorId sourceActorId )
{
	return fractureWorld != NULL && transition != NULL && transition->committed && transition->sourcePruned == false &&
		   b2BlastActorIdEqual( transition->sourceActorId, sourceActorId ) && transition->cellCount > 0 &&
		   transition->cellOffset >= 0 && transition->cellOffset + transition->cellCount <= fractureWorld->transitionCellCount;
}

static bool b2BlastPruneSourceActorForCommittedTransitions( b2World* world, b2BlastActor* source )
{
	if ( world == NULL || source == NULL || source->shapeId == B2_NULL_INDEX || source->bodyId == B2_NULL_INDEX )
	{
		return false;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	b2Shape* sourceShape = b2ShapeArray_Get( &world->shapes, source->shapeId );
	b2Body* sourceBody = b2BodyArray_Get( &world->bodies, source->bodyId );
	if ( sourceShape == NULL || sourceBody == NULL || sourceShape->type != b2_pixelShape ||
		 b2IsPixelShapeUsable( &sourceShape->pixel ) == false )
	{
		return false;
	}

	const b2PixelAsset* sourceAsset = sourceShape->pixel.asset;
	const int width = sourceAsset->width;
	const int height = sourceAsset->height;
	const int cellCount = width * height;
	const int wordCount = ( cellCount + 63 ) / 64;
	if ( b2BlastEnsureOwnedPixelAssetCapacity( source, width, height ) == false )
	{
		fractureWorld->stepAllocationFallbackCount += 1;
		return false;
	}

	if ( source->ownedOccupancyBits != sourceAsset->occupancyBits )
	{
		memset( source->ownedOccupancyBits, 0, (size_t)wordCount * sizeof( uint64_t ) );
		if ( sourceAsset->occupancyBits != NULL && sourceAsset->occupancyWordCount >= wordCount )
		{
			memcpy( source->ownedOccupancyBits, sourceAsset->occupancyBits, (size_t)wordCount * sizeof( uint64_t ) );
		}
	}
	if ( sourceAsset->materialIds != NULL && sourceAsset->materialIdCount >= cellCount )
	{
		if ( source->ownedMaterialIds != sourceAsset->materialIds )
		{
			memcpy( source->ownedMaterialIds, sourceAsset->materialIds, (size_t)cellCount * sizeof( b2BlastMaterialId ) );
		}
	}
	else
	{
		memset( source->ownedMaterialIds, 0, (size_t)cellCount * sizeof( b2BlastMaterialId ) );
	}

	uint32_t topologyVersion = sourceAsset->topologyVersion + 1;
	bool prunedAny = false;
	for ( int transitionIndex = 0; transitionIndex < fractureWorld->transitionCount; ++transitionIndex )
	{
		b2BlastActorTransition* transition = fractureWorld->transitions + transitionIndex;
		if ( b2BlastTransitionCellBelongsToSourcePrune( fractureWorld, transition, source->id ) == false )
		{
			continue;
		}
		const uint32_t transitionTopologyVersion = sourceAsset->topologyVersion + transition->transitionId + 1;
		topologyVersion = topologyVersion > transitionTopologyVersion ? topologyVersion : transitionTopologyVersion;
		for ( int i = 0; i < transition->cellCount; ++i )
		{
			const int cell = fractureWorld->transitionCells[transition->cellOffset + i];
			if ( cell < 0 || cell >= cellCount )
			{
				continue;
			}
			source->ownedOccupancyBits[cell >> 6] &= ~( UINT64_C( 1 ) << ( cell & 63 ) );
			source->ownedMaterialIds[cell] = 0;
			prunedAny = true;
		}
	}
	if ( prunedAny == false )
	{
		return false;
	}

	b2PixelAssetBuildConfig config = b2DefaultPixelAssetBuildConfig();
	config.width = width;
	config.height = height;
	config.pixelSize = sourceAsset->pixelSize;
	config.supportCornerInterval = 0;
	config.topologyVersion = topologyVersion;
	config.materialIds = source->ownedMaterialIds;
	config.materialIdCount = cellCount;
	config.materialTable = sourceAsset->materialTable;
	config.materialHash = sourceAsset->materialHash;
	b2PixelAssetBuildBuffers buffers = { 0 };
	buffers.occupancyBits = source->ownedOccupancyBits;
	buffers.occupancyWordCapacity = source->ownedOccupancyWordCapacity;
	buffers.materialIds = source->ownedMaterialIds;
	buffers.materialIdCapacity = source->ownedMaterialIdCapacity;
	buffers.featureTypes = source->ownedFeatureTypes;
	buffers.featureTypeCapacity = source->ownedFeatureTypeCapacity;
	buffers.normalIndices = source->ownedNormalIndices;
	buffers.normalIndexCapacity = source->ownedNormalIndexCapacity;
	buffers.corners = source->ownedCorners;
	buffers.cornerCapacity = source->ownedCornerCapacity;
	buffers.edges = source->ownedEdges;
	buffers.edgeCapacity = source->ownedEdgeCapacity;
	buffers.rowSolidCounts = source->ownedRowSolidCounts;
	buffers.rowSolidCountCapacity = source->ownedRowSolidCountCapacity;
	buffers.colSolidCounts = source->ownedColSolidCounts;
	buffers.colSolidCountCapacity = source->ownedColSolidCountCapacity;
	buffers.scratchCells = source->ownedPixelScratch;
	buffers.scratchCellCapacity = source->ownedPixelScratchCapacity;
	b2PixelAssetBuildResult result =
		b2BuildPixelAssetFromOccupancy( &config, source->ownedOccupancyBits, wordCount, &buffers );
	if ( result.success == false )
	{
		fractureWorld->stepAllocationFallbackCount += 1;
		return false;
	}

	sourceShape->pixel.asset = &result.asset;
	sourceShape->pixel.topologyVersion = result.asset.topologyVersion;
	sourceShape->pixelAssetRevision = result.asset.topologyVersion;
	sourceShape->localCentroid = b2GetShapeCentroid( sourceShape );
	sourceShape->aabbMargin = b2ComputeShapeMargin( sourceShape );
	sourceShape->pixel.dirtyX = 0;
	sourceShape->pixel.dirtyY = 0;
	sourceShape->pixel.dirtyWidth = width;
	sourceShape->pixel.dirtyHeight = height;
	sourceShape->pixel.updateKind = b2_pixelShapeUpdateDirtyUpdate;
	if ( b2BlastActor_RepairFromPixelShapeDirtyUpdate( source, sourceShape ) == false )
	{
		fractureWorld->stepAllocationFallbackCount += 1;
		return false;
	}
	source->flags |= b2_blastActorFlagOwnsPixelAsset;
	sourceShape->pixel.asset = &source->ownedAsset;
	sourceShape->pixel.topologyVersion = source->ownedAsset.topologyVersion;
	sourceShape->pixelAssetRevision = source->ownedAsset.topologyVersion;
	sourceShape->localCentroid = b2GetShapeCentroid( sourceShape );
	sourceShape->aabbMargin = b2ComputeShapeMargin( sourceShape );
	source->bodyId = sourceBody->id;
	source->shapeId = sourceShape->id;
	source->mobility = b2BlastMobilityFromBodyTypeLocal( sourceBody->type );
	source->flags |= b2_blastActorFlagInUse | b2_blastActorFlagOwnsPixelAsset;
	sourceBody->blastRevision = source->revision;
	sourceShape->blastRevision = source->revision;

	bool wakeBodies = true;
	bool destroyProxy = true;
	b2ResetProxy( world, sourceShape, wakeBodies, destroyProxy );
	b2UpdateBodyMassData( world, sourceBody );
	b2WakeBody( world, sourceBody );

	for ( int transitionIndex = 0; transitionIndex < fractureWorld->transitionCount; ++transitionIndex )
	{
		b2BlastActorTransition* transition = fractureWorld->transitions + transitionIndex;
		if ( b2BlastTransitionCellBelongsToSourcePrune( fractureWorld, transition, source->id ) )
		{
			transition->sourcePruned = true;
		}
	}
	return true;
}

static void b2BlastPruneCommittedTransitionSources( b2World* world )
{
	if ( world == NULL )
	{
		return;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	for ( int transitionIndex = 0; transitionIndex < fractureWorld->transitionCount; ++transitionIndex )
	{
		const b2BlastActorTransition* transition = fractureWorld->transitions + transitionIndex;
		if ( b2BlastTransitionCellBelongsToSourcePrune( fractureWorld, transition, transition->sourceActorId ) == false )
		{
			continue;
		}
		b2BlastActor* source = b2BlastGetActor( fractureWorld, transition->sourceActorId );
		(void)b2BlastPruneSourceActorForCommittedTransitions( world, source );
	}
}

static bool b2BlastCommitTransition( b2World* world, b2BlastActorTransition* transition )
{
	if ( world == NULL || transition == NULL || transition->committed ||
		 transition->action != b2_blastActorTransitionCreateDynamicBody )
	{
		return false;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	b2BlastActor* source = b2BlastGetActor( fractureWorld, transition->sourceActorId );
	if ( source == NULL || source->shapeId == B2_NULL_INDEX || source->bodyId == B2_NULL_INDEX )
	{
		fractureWorld->stepAllocationFallbackCount += 1;
		return false;
	}
	b2Shape* sourceShape = b2ShapeArray_Get( &world->shapes, source->shapeId );
	b2Body* sourceBody = b2BodyArray_Get( &world->bodies, source->bodyId );
	if ( sourceShape == NULL || sourceBody == NULL || sourceShape->type != b2_pixelShape )
	{
		fractureWorld->stepAllocationFallbackCount += 1;
		return false;
	}

	b2BlastActor* child = b2BlastAllocActor( world, world->worldId );
	if ( child == NULL )
	{
		fractureWorld->stepAllocationFallbackCount += 1;
		return false;
	}
	child->mobility = b2_blastActorMobilityDynamic;
	source = b2BlastGetActor( fractureWorld, transition->sourceActorId );
	if ( source == NULL )
	{
		b2BlastRollbackAllocatedActor( fractureWorld, child );
		fractureWorld->stepAllocationFallbackCount += 1;
		return false;
	}
	child->materialHash = source->materialHash;
	if ( b2BlastBuildTransitionPixelAsset( child, source, sourceShape, transition, fractureWorld->transitionCells ) == false )
	{
		b2BlastRollbackAllocatedActor( fractureWorld, child );
		fractureWorld->stepAllocationFallbackCount += 1;
		return false;
	}

	const b2PixelAsset* sourceAsset = sourceShape->pixel.asset;
	const float pixelSize = sourceAsset->pixelSize;
	const float sourceHalfWidth = 0.5f * (float)sourceAsset->width * pixelSize;
	const float sourceHalfHeight = 0.5f * (float)sourceAsset->height * pixelSize;
	b2Vec2 componentLocalCenter = {
		( 0.5f * (float)( transition->sourceMinX + transition->sourceMaxX + 1 ) ) * pixelSize - sourceHalfWidth +
			sourceShape->pixel.localOrigin.x,
		( 0.5f * (float)( transition->sourceMinY + transition->sourceMaxY + 1 ) ) * pixelSize - sourceHalfHeight +
			sourceShape->pixel.localOrigin.y,
	};
	b2PixelShape childPixelShape = { 0 };
	childPixelShape.asset = &child->ownedAsset;
	childPixelShape.localOrigin = b2Vec2_zero;
	childPixelShape.diskRadius = sourceShape->pixel.diskRadius;
	childPixelShape.topologyVersion = child->ownedAsset.topologyVersion;
	const b2MassData childMassAtBoundsCenter = b2ComputePixelShapeMass( &childPixelShape, sourceShape->density );
	const b2Vec2 childLocalOrigin = b2Neg( childMassAtBoundsCenter.center );
	b2Transform sourceTransform = b2GetBodyTransformQuick( world, sourceBody );
	b2Vec2 childPosition = b2TransformPoint( sourceTransform, b2Add( componentLocalCenter, childMassAtBoundsCenter.center ) );
	b2WorldId worldId = { (uint16_t)( world->worldId + 1 ), world->generation };
	b2BodyId sourceBodyId = { sourceBody->id + 1, world->worldId, sourceBody->generation };
	b2ShapeId sourceShapeId = { sourceShape->id + 1, world->worldId, sourceShape->generation };
	transition->sourceBodyId = sourceBodyId;
	transition->sourceShapeId = sourceShapeId;

	b2BodyDef bodyDef = b2DefaultBodyDef();
	bodyDef.type = b2_dynamicBody;
	bodyDef.position = childPosition;
	bodyDef.rotation = sourceTransform.q;
	if ( sourceBody->type == b2_dynamicBody )
	{
		bodyDef.linearVelocity = b2Body_GetLinearVelocity( sourceBodyId );
		bodyDef.angularVelocity = b2Body_GetAngularVelocity( sourceBodyId );
		bodyDef.linearDamping = b2Body_GetLinearDamping( sourceBodyId );
		bodyDef.angularDamping = b2Body_GetAngularDamping( sourceBodyId );
		bodyDef.gravityScale = b2Body_GetGravityScale( sourceBodyId );
		bodyDef.enableSleep = sourceBody->enableSleep;
		bodyDef.sleepThreshold = b2Body_GetSleepThreshold( sourceBodyId );
		bodyDef.motionLocks = b2Body_GetMotionLocks( sourceBodyId );
		bodyDef.isAwake = b2Body_IsAwake( sourceBodyId );
	}
	else
	{
		bodyDef.linearVelocity = b2Vec2_zero;
		bodyDef.angularVelocity = 0.0f;
		bodyDef.isAwake = true;
	}
	b2BodyId childBodyId = b2CreateBody( worldId, &bodyDef );
	if ( childBodyId.index1 == 0 )
	{
		b2BlastRollbackAllocatedActor( fractureWorld, child );
		fractureWorld->stepAllocationFallbackCount += 1;
		return false;
	}

	b2ShapeDef shapeDef = b2DefaultShapeDef();
	shapeDef.density = sourceShape->density;
	shapeDef.material = sourceShape->material;
	shapeDef.filter = sourceShape->filter;
	shapeDef.enableContactEvents = sourceShape->enableContactEvents;
	shapeDef.enableHitEvents = sourceShape->enableHitEvents;
	shapeDef.updateBodyMass = true;
	b2PixelShape pixelShape = { 0 };
	pixelShape.asset = &child->ownedAsset;
	pixelShape.localOrigin = childLocalOrigin;
	pixelShape.diskRadius = sourceShape->pixel.diskRadius;
	pixelShape.topologyVersion = child->ownedAsset.topologyVersion;
	b2ShapeId childShapeId = b2CreatePixelShapeBoundToBlastActor( childBodyId, &shapeDef, &pixelShape, child->id );
	if ( childShapeId.index1 == 0 )
	{
		b2DestroyBody( childBodyId );
		b2BlastRollbackAllocatedActor( fractureWorld, child );
		fractureWorld->stepAllocationFallbackCount += 1;
		return false;
	}
	const b2Shape* childShape = b2ShapeArray_Get( &world->shapes, childShapeId.index1 - 1 );
	if ( b2BlastActor_MigrateTransitionChildGraphFromSource( child, source, childShape, transition,
															 fractureWorld->transitionCells ) == false )
	{
		b2BlastActor_ActivateLeafGraphForTransitionChild( child );
		child->authoringHash = b2BlastComputeActorHash( child );
	}

	transition->childActorId = child->id;
	transition->createdBodyId = childBodyId;
	transition->createdShapeId = childShapeId;
	transition->childTopologyRevision = child->topologyVersion;
	transition->worldPoint = childPosition;
	transition->linearVelocity = bodyDef.linearVelocity;
	transition->angularVelocity = bodyDef.angularVelocity;
	transition->committed = true;
	return true;
}

void b2BlastFractureWorld_EndStep( b2World* world )
{
	if ( world == NULL )
	{
		return;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	for ( int i = 0; i < fractureWorld->transitionCount; ++i )
	{
		b2BlastActorTransition* transition = fractureWorld->transitions + i;
		if ( transition->committed == false )
		{
			(void)b2BlastCommitTransition( world, transition );
		}
	}
	b2BlastPruneCommittedTransitionSources( world );
}

void b2BlastFractureWorld_ConsumeTouchingContactsIfNoRows( b2World* world, float timeStep )
{
	if ( world == NULL )
	{
		return;
	}

	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	if ( fractureWorld->bodyInputCount > 0 )
	{
		b2BlastFractureWorld_BeginSubstep( world );
		b2BlastFractureWorld_EndSubstep( world );
	}

	B2_UNUSED( timeStep );
}

void b2BlastFractureWorld_CollectAndStep( b2World* world, float timeStep )
{
	if ( world == NULL )
	{
		return;
	}

	b2BlastFractureWorld_BeginStep( world );

	for ( int i = 0; i < world->contacts.count; ++i )
	{
		b2Contact* contact = world->contacts.data + i;
		if ( contact->setIndex == B2_NULL_INDEX || ( contact->flags & b2_contactTouchingFlag ) == 0 )
		{
			continue;
		}
		b2BlastConsumeContact( world, contact, timeStep );
	}

	b2BlastFractureWorld_EndSubstep( world );
}

b2BlastFractureDebugSnapshot b2BlastFractureWorld_GetDebugSnapshot( const b2BlastFractureWorld* fractureWorld )
{
	b2BlastFractureDebugSnapshot snapshot = b2BlastFracture_GetDebugSnapshot();
	if ( fractureWorld == NULL )
	{
		return snapshot;
	}

	for ( int i = 0; i < fractureWorld->actorCount; ++i )
	{
		const b2BlastActor* actor = fractureWorld->actors + i;
		if ( ( actor->flags & b2_blastActorFlagInUse ) == 0 )
		{
			continue;
		}
		snapshot.actorCount += 1;
		snapshot.leafCount += (uint32_t)actor->leafCount;
		snapshot.activeBondCount += (uint32_t)actor->bondCount;
		snapshot.clusterCount += (uint32_t)actor->clusterCount;
		snapshot.worldAnchorCount += actor->worldAnchorCount;
		if ( actor->rootCluster != UINT32_MAX )
		{
			snapshot.rootClusterCount += 1;
		}
		if ( actor->authoringHash != 0 )
		{
			snapshot.authoringHashLow ^= (uint32_t)( actor->authoringHash & 0xffffffffu );
			snapshot.authoringHashHigh ^= (uint32_t)( actor->authoringHash >> 32 );
		}
		if ( actor->mobility == b2_blastActorMobilityAnchored )
		{
			snapshot.anchoredActorCount += 1;
		}
		else if ( actor->mobility == b2_blastActorMobilityDynamic || actor->mobility == b2_blastActorMobilitySleepingDynamic )
		{
			snapshot.dynamicActorCount += 1;
		}
		for ( int bondIndex = 0; bondIndex < actor->bondCount; ++bondIndex )
		{
			if ( ( actor->bonds[bondIndex].flags & b2_blastBondFlagBroken ) != 0 )
			{
				snapshot.brokenBondCount += 1;
			}
			snapshot.impactDemandHash ^= b2BlastHash32(
				( uint32_t )( bondIndex + 1 ) * 2654435761u ^ b2BlastFloatBits( actor->bonds[bondIndex].impactDemand ) );
			snapshot.loadDemandHash ^= b2BlastHash32(
				( uint32_t )( bondIndex + 1 ) * 2246822519u ^ b2BlastFloatBits( actor->bonds[bondIndex].loadDemand ) );
		}
	}

	snapshot.commandCount = (uint32_t)fractureWorld->commandCount;
	snapshot.constraintRowCount = fractureWorld->constraintRowCount;
	snapshot.contactPairRowCount = fractureWorld->contactPairRowCount;
	snapshot.contactImpactBudgetRowCount = fractureWorld->contactImpactBudgetRowCount;
	snapshot.contactLoadBudgetRowCount = fractureWorld->contactLoadBudgetRowCount;
	snapshot.contactYieldQueryCount = fractureWorld->contactYieldQueryCount;
	snapshot.jointConstraintRowCount = fractureWorld->jointConstraintRowCount;
	snapshot.actorTransitionCount = fractureWorld->actorTransitionCount;
	snapshot.appliedForceLoadRowCount = fractureWorld->appliedForceLoadRowCount;
	snapshot.appliedImpulseImpactRowCount = fractureWorld->appliedImpulseImpactRowCount;
	snapshot.torqueInputIgnoredCount = fractureWorld->torqueInputIgnoredCount;
	snapshot.reauthoredFallbackCount = fractureWorld->reauthoredFallbackCount;
	snapshot.legacyHostFracturePathCount = fractureWorld->legacyHostFracturePathCount;
	snapshot.stepAllocationFallbackCount = fractureWorld->stepAllocationFallbackCount;
	snapshot.ignoredOffTargetEventCount = fractureWorld->ignoredOffTargetEventCount;
	snapshot.unboundLoadPortDropCount = fractureWorld->unboundLoadPortDropCount;
	snapshot.refinedThisStep = fractureWorld->refinedThisStep;
	snapshot.brokenThisStep = fractureWorld->brokenThisStep;
	snapshot.demandRecomputeCount = fractureWorld->demandRecomputeCount;
	snapshot.substepSolveCount = fractureWorld->substepSolveCount;
	snapshot.scratchCapacityHighWater = fractureWorld->scratchCapacityHighWater;
	snapshot.maxImpactDemand = fractureWorld->maxImpactDemand;
	snapshot.maxLoadDemand = fractureWorld->maxLoadDemand;
	snapshot.maxDamage = fractureWorld->maxDamage;
	return snapshot;
}

b2BlastFractureDebugSnapshot b2BlastFracture_GetDebugSnapshot( void )
{
	b2BlastFractureDebugSnapshot snapshot = { 0 };
	snapshot.hotMaterialIdBytesPerCell = (uint32_t)sizeof( b2BlastMaterialId );
	snapshot.materialHotDataSize = (uint32_t)sizeof( b2BlastMaterialHotData );
	snapshot.leafSize = (uint32_t)sizeof( b2BlastLeaf );
	snapshot.activeBondSize = (uint32_t)sizeof( b2BlastActiveBond );
	return snapshot;
}

static void b2BlastAppendOverlayActiveGraph( b2BlastFractureWorld* fractureWorld, const b2BlastActor* actor,
											 b2BlastOverlayActorView* view, bool leafGraph )
{
	if ( fractureWorld == NULL || actor == NULL || view == NULL || actor->leafCount <= 0 || actor->clusterCount <= 0 )
	{
		return;
	}
	if ( actor->clusters == NULL || actor->clusterLeaves == NULL || actor->clusterLeafRefCount <= 0 )
	{
		return;
	}

	b2BlastEnsureOverlayLeafRemapScratchCapacity( fractureWorld, actor->leafCount );
	uint32_t* leafRemapScratch = fractureWorld->overlayLeafRemapScratch;
	if ( leafRemapScratch == NULL )
	{
		return;
	}
	for ( int i = 0; i < actor->leafCount; ++i )
	{
		leafRemapScratch[i] = UINT32_MAX;
	}

	b2BlastEnsureOverlayClusterCapacity( fractureWorld, fractureWorld->overlayClusterCount + actor->clusterCount );
	b2BlastEnsureOverlayBondCapacity( fractureWorld, fractureWorld->overlayBondCount + actor->bondCount );
	b2BlastEnsureOverlayCellToActiveClusterCapacity(
		fractureWorld, fractureWorld->overlayCellToActiveClusterCount + actor->cellToLeafCount );

	const int clusterStart = fractureWorld->overlayClusterCount;
	uint16_t activeLevel = 0;
	for ( int clusterIndex = 0; clusterIndex < actor->clusterCount; ++clusterIndex )
	{
		const b2BlastCluster* cluster = actor->clusters + clusterIndex;
		if ( leafGraph )
		{
			if ( cluster->level != 0 )
			{
				continue;
			}
		}
		else if ( b2BlastActor_ActiveContains( actor, cluster->id ) == false )
		{
			continue;
		}
		if ( cluster->firstLeaf == UINT32_MAX || cluster->firstLeaf + cluster->leafCount > (uint32_t)actor->clusterLeafRefCount )
		{
			continue;
		}
		const uint32_t localClusterIndex = (uint32_t)( fractureWorld->overlayClusterCount - clusterStart );
		b2BlastOverlayCluster* overlayCluster = fractureWorld->overlayClusters + fractureWorld->overlayClusterCount++;
		*overlayCluster = (b2BlastOverlayCluster){ 0 };
		overlayCluster->clusterIndex = cluster->id;
		overlayCluster->firstLeaf = cluster->firstLeaf;
		overlayCluster->leafCount = cluster->leafCount;
		overlayCluster->centroid = cluster->centroid;
		overlayCluster->mass = cluster->mass;
		overlayCluster->level = cluster->level;
		overlayCluster->flags = cluster->flags;
		overlayCluster->minX = cluster->minX;
		overlayCluster->minY = cluster->minY;
		overlayCluster->maxX = cluster->maxX;
		overlayCluster->maxY = cluster->maxY;
		activeLevel = (uint16_t)b2MaxInt( activeLevel, cluster->level );

		for ( uint32_t leafRef = 0; leafRef < cluster->leafCount; ++leafRef )
		{
			const uint32_t leafIndex = actor->clusterLeaves[cluster->firstLeaf + leafRef];
			if ( leafIndex < (uint32_t)actor->leafCount )
			{
				leafRemapScratch[leafIndex] = localClusterIndex;
			}
		}
	}

	view->activeClusters = fractureWorld->overlayClusters + clusterStart;
	view->activeClusterCount = fractureWorld->overlayClusterCount - clusterStart;
	view->activeLevel = activeLevel;
	if ( view->activeClusterCount <= 0 )
	{
		return;
	}

	const int cellStart = fractureWorld->overlayCellToActiveClusterCount;
	for ( int cell = 0; cell < actor->cellToLeafCount; ++cell )
	{
		uint32_t activeCluster = UINT32_MAX;
		const uint32_t leafIndex = actor->cellToLeaf[cell];
		if ( leafIndex < (uint32_t)actor->leafCount )
		{
			activeCluster = leafRemapScratch[leafIndex];
		}
		fractureWorld->overlayCellToActiveCluster[fractureWorld->overlayCellToActiveClusterCount++] = activeCluster;
	}
	view->cellToActiveCluster = fractureWorld->overlayCellToActiveCluster + cellStart;
	view->cellToActiveClusterCount = fractureWorld->overlayCellToActiveClusterCount - cellStart;

	const int bondStart = fractureWorld->overlayBondCount;
	for ( int bondIndex = 0; bondIndex < actor->bondCount; ++bondIndex )
	{
		const b2BlastActiveBond* source = actor->bonds + bondIndex;
		if ( source->leafA >= (uint32_t)actor->leafCount || source->leafB >= (uint32_t)actor->leafCount )
		{
			continue;
		}
		if ( ( actor->leaves[source->leafA].flags & b2_blastLeafFlagDetached ) != 0 ||
			 ( actor->leaves[source->leafB].flags & b2_blastLeafFlagDetached ) != 0 )
		{
			continue;
		}
		uint32_t clusterA = leafRemapScratch[source->leafA];
		uint32_t clusterB = leafRemapScratch[source->leafB];
		if ( clusterA == UINT32_MAX || clusterB == UINT32_MAX || clusterA == clusterB )
		{
			continue;
		}
		if ( clusterB < clusterA )
		{
			uint32_t temp = clusterA;
			clusterA = clusterB;
			clusterB = temp;
		}

		b2BlastActiveBond* aggregate = NULL;
		for ( int i = bondStart; i < fractureWorld->overlayBondCount; ++i )
		{
			b2BlastActiveBond* candidate = fractureWorld->overlayBonds + i;
			if ( candidate->clusterA == clusterA && candidate->clusterB == clusterB )
			{
				aggregate = candidate;
				break;
			}
		}
		if ( aggregate == NULL )
		{
			aggregate = fractureWorld->overlayBonds + fractureWorld->overlayBondCount++;
			*aggregate = (b2BlastActiveBond){ 0 };
			aggregate->leafA = source->leafA;
			aggregate->leafB = source->leafB;
			aggregate->clusterA = clusterA;
			aggregate->clusterB = clusterB;
			aggregate->damage = source->damage;
			aggregate->flags = source->flags;
			aggregate->materialMix = source->materialMix;
		}
		aggregate->area += source->area;
		aggregate->capacity += source->capacity;
		aggregate->toughness += source->toughness;
		aggregate->impactDemand += source->impactDemand;
		aggregate->loadDemand += source->loadDemand;
		aggregate->propagationWeight = b2MaxFloat( aggregate->propagationWeight, source->propagationWeight );
		aggregate->damage = b2MaxFloat( aggregate->damage, source->damage );
		aggregate->flags |= source->flags;
		aggregate->materialMix |= source->materialMix;
	}
	view->activeBonds = fractureWorld->overlayBonds + bondStart;
	view->activeBondCount = fractureWorld->overlayBondCount - bondStart;
}

b2BlastFractureDebugSnapshot b2World_GetBlastFractureDebugSnapshot( b2WorldId worldId )
{
	b2World* world = b2GetWorldFromId( worldId );
	if ( world == NULL )
	{
		return b2BlastFracture_GetDebugSnapshot();
	}
	if ( world != NULL && world->locked == false )
	{
		b2BlastFractureWorld_EndStep( world );
	}
	return b2BlastFractureWorld_GetDebugSnapshot( &world->blastFractureWorld );
}

bool b2World_BeginBlastOverlayRead( b2WorldId worldId, const b2BlastOverlayReadQuery* query, b2BlastOverlayReadView* outView )
{
	if ( outView == NULL )
	{
		return false;
	}
	*outView = (b2BlastOverlayReadView){ 0 };
	b2World* world = b2GetWorldFromId( worldId );
	if ( world == NULL || world->locked )
	{
		outView->skipped = true;
		return false;
	}

	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	fractureWorld->overlayActorViewCount = 0;
	fractureWorld->overlayClusterCount = 0;
	fractureWorld->overlayBondCount = 0;
	fractureWorld->overlayCellToActiveClusterCount = 0;
	b2BlastEnsureOverlayActorViewCapacity( fractureWorld, fractureWorld->actorCount );
	int overlayClusterCapacity = 0;
	int overlayBondCapacity = 0;
	int overlayCellCapacity = 0;
	for ( int i = 0; i < fractureWorld->actorCount; ++i )
	{
		const b2BlastActor* actor = fractureWorld->actors + i;
		if ( ( actor->flags & b2_blastActorFlagInUse ) == 0 )
		{
			continue;
		}
		overlayClusterCapacity += actor->clusterCount;
		overlayBondCapacity += actor->bondCount;
		overlayCellCapacity += actor->cellToLeafCount;
	}
	b2BlastEnsureOverlayClusterCapacity( fractureWorld, overlayClusterCapacity );
	b2BlastEnsureOverlayBondCapacity( fractureWorld, overlayBondCapacity );
	b2BlastEnsureOverlayCellToActiveClusterCapacity( fractureWorld, overlayCellCapacity );
	const bool hasQuery = query != NULL && query->worldAABB.upperBound.x >= query->worldAABB.lowerBound.x &&
						  query->worldAABB.upperBound.y >= query->worldAABB.lowerBound.y;
	const bool leafGraph = query != NULL && ( query->flags & b2_blastOverlayReadLeafGraph ) != 0;
	for ( int i = 0; i < fractureWorld->actorCount; ++i )
	{
		const b2BlastActor* actor = fractureWorld->actors + i;
		if ( ( actor->flags & b2_blastActorFlagInUse ) == 0 || actor->leafCount <= 0 ||
			 actor->cellToLeaf == NULL || actor->cellToLeafCount <= 0 || actor->shapeId == B2_NULL_INDEX ||
			 actor->bodyId == B2_NULL_INDEX )
		{
			continue;
		}
		if ( actor->shapeId < 0 || actor->shapeId >= world->shapes.count || actor->bodyId < 0 || actor->bodyId >= world->bodies.count )
		{
			continue;
		}
		const b2Shape* shape = b2ShapeArray_Get( &world->shapes, actor->shapeId );
		const b2Body* body = b2BodyArray_Get( &world->bodies, actor->bodyId );
		if ( shape == NULL || body == NULL || shape->type != b2_pixelShape || shape->pixel.asset == NULL )
		{
			continue;
		}
		if ( shape->id != actor->shapeId || body->id != actor->bodyId || shape->bodyId != actor->bodyId ||
			 body->setIndex == B2_NULL_INDEX || b2BlastActorIdEqual( shape->blastActorId, actor->id ) == false )
		{
			continue;
		}
		if ( hasQuery && b2BlastAABBOverlaps( shape->fatAABB, query->worldAABB ) == false )
		{
			continue;
		}

		b2BlastOverlayActorView* view = fractureWorld->overlayActorViews + fractureWorld->overlayActorViewCount++;
		*view = (b2BlastOverlayActorView){ 0 };
		view->actorId = actor->id;
		view->bodyId = (b2BodyId){ body->id + 1, world->worldId, body->generation };
		view->shapeId = (b2ShapeId){ shape->id + 1, world->worldId, shape->generation };
		view->mobility = actor->mobility;
		view->transform = b2GetBodyTransformQuick( world, (b2Body*)body );
		view->localOrigin = shape->pixel.localOrigin;
		view->width = actor->ownedAsset.width;
		view->height = actor->ownedAsset.height;
		view->pixelSize = actor->ownedAsset.pixelSize;
		view->revision = actor->revision;
		view->topologyVersion = actor->topologyVersion;
		view->materialHash = actor->materialHash;
		view->leaves = actor->leaves;
		view->leafCount = actor->leafCount;
		view->bonds = actor->bonds;
		view->bondCount = actor->bondCount;
		view->cellToLeaf = actor->cellToLeaf;
		view->cellToLeafCount = actor->cellToLeafCount;
		b2BlastAppendOverlayActiveGraph( fractureWorld, actor, view, leafGraph );
	}

	fractureWorld->overlayDirectReadCount += 1;
	fractureWorld->overlayReadRevision += 1;
	outView->actors = fractureWorld->overlayActorViews;
	outView->actorCount = fractureWorld->overlayActorViewCount;
	outView->readRevision = fractureWorld->overlayReadRevision;
	return true;
}

void b2World_EndBlastOverlayRead( b2WorldId worldId, const b2BlastOverlayReadView* view )
{
	B2_UNUSED( view );
	b2World* world = b2GetWorldFromId( worldId );
	if ( world == NULL )
	{
		return;
	}
	world->blastFractureWorld.overlayActorViewCount = 0;
	world->blastFractureWorld.overlayClusterCount = 0;
	world->blastFractureWorld.overlayBondCount = 0;
	world->blastFractureWorld.overlayCellToActiveClusterCount = 0;
}

int32_t b2World_GetBlastFractureTransitionCount( b2WorldId worldId )
{
	b2World* world = b2GetWorldFromId( worldId );
	if ( world == NULL )
	{
		return 0;
	}
	return world->blastFractureWorld.transitionCount;
}

int32_t b2World_CopyBlastFractureTransitions(
	b2WorldId worldId, b2BlastActorTransition* transitions, int32_t transitionCapacity )
{
	b2World* world = b2GetWorldFromId( worldId );
	if ( world == NULL || transitions == NULL || transitionCapacity <= 0 )
	{
		return 0;
	}
	if ( world->locked == false )
	{
		b2BlastFractureWorld_EndStep( world );
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	int count = b2MinInt( transitionCapacity, fractureWorld->transitionCount );
	memcpy( transitions, fractureWorld->transitions, (size_t)count * sizeof( b2BlastActorTransition ) );
	return count;
}

int32_t b2World_CopyBlastFractureTransitionCells(
	b2WorldId worldId, int32_t transitionIndex, int32_t* cells, int32_t cellCapacity )
{
	b2World* world = b2GetWorldFromId( worldId );
	if ( world == NULL || cells == NULL || cellCapacity <= 0 )
	{
		return 0;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	if ( transitionIndex < 0 || transitionIndex >= fractureWorld->transitionCount )
	{
		return 0;
	}
	const b2BlastActorTransition* transition = fractureWorld->transitions + transitionIndex;
	int count = b2MinInt( cellCapacity, transition->cellCount );
	memcpy( cells, fractureWorld->transitionCells + transition->cellOffset, (size_t)count * sizeof( int32_t ) );
	return count;
}

void b2World_AcknowledgeBlastFractureTransitions( b2WorldId worldId )
{
	b2World* world = b2GetWorldFromId( worldId );
	if ( world == NULL )
	{
		return;
	}
	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	fractureWorld->transitionCount = 0;
	fractureWorld->transitionCellCount = 0;
}

void b2World_CommitBlastFractureTransitions( b2WorldId worldId )
{
	b2World* world = b2GetWorldFromId( worldId );
	if ( world == NULL )
	{
		return;
	}
	b2BlastFractureWorld_EndStep( world );
}
