// SPDX-FileCopyrightText: 2026 Alchemy
// SPDX-License-Identifier: MIT

#include "blast_fracture.h"

#include "body.h"
#include "contact.h"
#include "core.h"
#include "physics_world.h"
#include "pixel_shape.h"
#include "shape.h"

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
	b2_blastLeafFlagWorldAnchor = 0x0001,
	b2_blastBondFlagBroken = 0x0001,
	b2_blastBondFlagBreakCandidate = 0x0002,
	b2_blastDefaultLeafCellSpan = 4,
	b2_blastInitialActorCapacity = 8,
	b2_blastInitialCommandCapacity = 64,
};

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

	uint32_t* cellToLeaf;
	int cellToLeafCount;
	int cellToLeafCapacity;

	int* componentScratch;
	int componentScratchCapacity;
	int* queueScratch;
	int queueScratchCapacity;
	uint8_t* visitScratch;
	int visitScratchCapacity;
} b2BlastActor;

static b2BlastFractureActorId b2BlastNullActorId( uint16_t worldId )
{
	return (b2BlastFractureActorId){ UINT32_MAX, 0, worldId };
}

static bool b2BlastActorIdValid( b2BlastFractureActorId id )
{
	return id.index != UINT32_MAX && id.revision != 0;
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
	b2Free( actor->cellToLeaf, actor->cellToLeafCapacity * (int)sizeof( uint32_t ) );
	b2Free( actor->componentScratch, actor->componentScratchCapacity * (int)sizeof( int ) );
	b2Free( actor->queueScratch, actor->queueScratchCapacity * (int)sizeof( int ) );
	b2Free( actor->visitScratch, actor->visitScratchCapacity * (int)sizeof( uint8_t ) );
	actor->leaves = NULL;
	actor->bonds = NULL;
	actor->cellToLeaf = NULL;
	actor->componentScratch = NULL;
	actor->queueScratch = NULL;
	actor->visitScratch = NULL;
	actor->leafCapacity = 0;
	actor->bondCapacity = 0;
	actor->cellToLeafCapacity = 0;
	actor->componentScratchCapacity = 0;
	actor->queueScratchCapacity = 0;
	actor->visitScratchCapacity = 0;
	actor->leafCount = 0;
	actor->bondCount = 0;
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

static b2BlastActor* b2BlastAllocActor( b2BlastFractureWorld* fractureWorld, uint16_t worldId )
{
	if ( fractureWorld->actorCount == fractureWorld->actorCapacity )
	{
		int oldCapacity = fractureWorld->actorCapacity;
		int newCapacity = oldCapacity < b2_blastInitialActorCapacity ? b2_blastInitialActorCapacity : oldCapacity + ( oldCapacity >> 1 );
		fractureWorld->actors = b2BlastResize( fractureWorld->actors, oldCapacity, newCapacity, (int)sizeof( b2BlastActor ) );
		fractureWorld->actorCapacity = newCapacity;
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

static void b2BlastEnsureActorCapacity( b2BlastActor* actor, int leafCapacity, int bondCapacity, int cellCount )
{
	if ( leafCapacity > actor->leafCapacity )
	{
		actor->leaves = b2BlastResize( actor->leaves, actor->leafCapacity, leafCapacity, (int)sizeof( b2BlastLeaf ) );
		actor->componentScratch =
			b2BlastResize( actor->componentScratch, actor->componentScratchCapacity, leafCapacity, (int)sizeof( int ) );
		actor->queueScratch = b2BlastResize( actor->queueScratch, actor->queueScratchCapacity, leafCapacity, (int)sizeof( int ) );
		actor->visitScratch = b2BlastResize( actor->visitScratch, actor->visitScratchCapacity, leafCapacity, (int)sizeof( uint8_t ) );
		actor->leafCapacity = leafCapacity;
		actor->componentScratchCapacity = leafCapacity;
		actor->queueScratchCapacity = leafCapacity;
		actor->visitScratchCapacity = leafCapacity;
	}

	if ( bondCapacity > actor->bondCapacity )
	{
		actor->bonds = b2BlastResize( actor->bonds, actor->bondCapacity, bondCapacity, (int)sizeof( b2BlastActiveBond ) );
		actor->bondCapacity = bondCapacity;
	}

	if ( cellCount > actor->cellToLeafCapacity )
	{
		actor->cellToLeaf = b2BlastResize( actor->cellToLeaf, actor->cellToLeafCapacity, cellCount, (int)sizeof( uint32_t ) );
		actor->cellToLeafCapacity = cellCount;
	}
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
	const int width = asset->width;
	const int height = asset->height;
	const int cellCount = width * height;
	const int block = b2_blastDefaultLeafCellSpan;
	const int blockWidth = ( width + block - 1 ) / block;
	const int blockHeight = ( height + block - 1 ) / block;
	const int maxLeaves = blockWidth * blockHeight;
	const int maxBonds = b2MaxInt( 1, asset->solidCount * 2 );

	b2BlastEnsureActorCapacity( actor, maxLeaves, maxBonds, cellCount );
	for ( int i = 0; i < cellCount; ++i )
	{
		actor->cellToLeaf[i] = UINT32_MAX;
	}

	actor->leafCount = 0;
	actor->bondCount = 0;
	actor->cellToLeafCount = cellCount;

	const float halfWidth = 0.5f * (float)width * asset->pixelSize;
	const float halfHeight = 0.5f * (float)height * asset->pixelSize;
	const float cellArea = asset->pixelSize * asset->pixelSize;

	for ( int by = 0; by < blockHeight; ++by )
	{
		for ( int bx = 0; bx < blockWidth; ++bx )
		{
			b2BlastLeaf leaf = { 0 };
			leaf.firstCell = UINT32_MAX;
			leaf.dominantMaterialId = 0;
			float centroidX = 0.0f;
			float centroidY = 0.0f;
			float mass = 0.0f;
			uint32_t materialVotes[4] = { 0, 0, 0, 0 };
			b2BlastMaterialId materialVoteIds[4] = { 0, 0, 0, 0 };

			for ( int y = by * block; y < b2MinInt( height, ( by + 1 ) * block ); ++y )
			{
				for ( int x = bx * block; x < b2MinInt( width, ( bx + 1 ) * block ); ++x )
				{
					if ( b2PixelAsset_IsOccupied( asset, x, y ) == false )
					{
						continue;
					}

					const int cellIndex = y * width + x;
					if ( leaf.firstCell == UINT32_MAX )
					{
						leaf.firstCell = (uint32_t)cellIndex;
					}
					const b2BlastMaterialId materialId = b2PixelAsset_GetMaterialId( asset, x, y );
					float density = b2BlastMaterialDensity( asset, materialId, shape->density );
					float cellMass = density * cellArea;
					b2Vec2 center = { ( (float)x + 0.5f ) * asset->pixelSize - halfWidth + shape->pixel.localOrigin.x,
									  ( (float)y + 0.5f ) * asset->pixelSize - halfHeight + shape->pixel.localOrigin.y };
					centroidX += center.x * cellMass;
					centroidY += center.y * cellMass;
					mass += cellMass;
					leaf.cellCount += 1;
					actor->cellToLeaf[cellIndex] = (uint32_t)actor->leafCount;

					bool voted = false;
					for ( int vote = 0; vote < 4; ++vote )
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
						for ( int vote = 0; vote < 4; ++vote )
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

			if ( leaf.cellCount == 0 )
			{
				continue;
			}

			leaf.mass = mass;
			leaf.centroid = mass > 0.0f ? (b2Vec2){ centroidX / mass, centroidY / mass } : b2Vec2_zero;
			uint32_t bestVotes = 0;
			for ( int vote = 0; vote < 4; ++vote )
			{
				if ( materialVotes[vote] > bestVotes )
				{
					bestVotes = materialVotes[vote];
					leaf.dominantMaterialId = materialVoteIds[vote];
				}
			}
			if ( actor->mobility == b2_blastActorMobilityAnchored && by == blockHeight - 1 )
			{
				leaf.flags |= b2_blastLeafFlagWorldAnchor;
				actor->flags |= b2_blastActorFlagOwnsWorldAnchor;
			}
			actor->leaves[actor->leafCount++] = leaf;
		}
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
		bond->propagationWeight = 1.0f / b2MaxFloat( 1.0f, bond->area );
		bond->damage = b2ClampFloat( bond->damage, 0.0f, 1.0f );
	}

	actor->topologyVersion = asset->topologyVersion;
	actor->materialHash = asset->materialHash;
	actor->revision += 1;
	actor->flags &= (uint32_t)~b2_blastActorFlagDirtyGraph;
	return true;
}

void b2BlastFractureWorld_Create( b2BlastFractureWorld* fractureWorld )
{
	*fractureWorld = (b2BlastFractureWorld){ 0 };
	fractureWorld->nextConstraintId = 1;
	fractureWorld->commandCapacity = b2_blastInitialCommandCapacity;
	fractureWorld->commands = b2AllocZeroInit( fractureWorld->commandCapacity * (int)sizeof( b2BlastFractureCommand ) );
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
	*fractureWorld = (b2BlastFractureWorld){ 0 };
}

void b2BlastFractureWorld_ClearStep( b2BlastFractureWorld* fractureWorld )
{
	if ( fractureWorld == NULL )
	{
		return;
	}
	fractureWorld->commandCount = 0;
	fractureWorld->constraintRowCount = 0;
	fractureWorld->actorTransitionCount = 0;
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
	if ( actor == NULL )
	{
		actor = b2BlastAllocActor( fractureWorld, world->worldId );
	}

	actor->bodyId = body->id;
	actor->shapeId = shape->id;
	actor->mobility = mobility;
	actor->flags |= b2_blastActorFlagInUse | b2_blastActorFlagDirtyGraph;
	actor->flags &= (uint32_t)~b2_blastActorFlagOwnsWorldAnchor;
	if ( b2BlastActor_AuthorFromPixelShape( actor, shape ) == false )
	{
		fractureWorld->reauthoredFallbackCount += 1;
	}

	body->blastActorId = actor->id;
	body->blastRevision = actor->revision;
	body->blastFlags |= 1u;
	shape->blastActorId = actor->id;
	shape->blastRevision = actor->revision;
	shape->blastFlags |= 1u;
	shape->pixelAssetRevision = shape->pixel.asset == NULL ? 0 : shape->pixel.asset->topologyVersion;
	shape->surfaceLookupKey = (uint32_t)shape->id;
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

static void b2BlastApplyDemandFromLeaf( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor, int leafIndex, float impulse,
										b2Vec2 direction, bool impact )
{
	if ( actor == NULL || leafIndex == B2_NULL_INDEX || impulse <= 0.0f )
	{
		return;
	}

	const b2Vec2 source = actor->leaves[leafIndex].centroid;
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		b2BlastActiveBond* bond = actor->bonds + i;
		if ( ( bond->flags & b2_blastBondFlagBroken ) != 0 )
		{
			continue;
		}
		b2Vec2 a = actor->leaves[bond->leafA].centroid;
		b2Vec2 b = actor->leaves[bond->leafB].centroid;
		b2Vec2 mid = b2MulSV( 0.5f, b2Add( a, b ) );
		float distance = sqrtf( b2BlastDistanceSquared( source, mid ) );
		b2Vec2 axis = b2Sub( b, a );
		float axisLen = b2Length( axis );
		float align = 1.0f;
		if ( axisLen > 0.0001f && b2LengthSquared( direction ) > 0.0001f )
		{
			axis = b2MulSV( 1.0f / axisLen, axis );
			b2Vec2 dir = b2Normalize( direction );
			align = 0.35f + 0.65f * fabsf( b2Dot( axis, dir ) );
		}
		float bottleneck = 1.0f / b2MaxFloat( 1.0f, bond->area );
		float decay = impact ? expf( -0.34f * distance ) : 1.0f / ( 1.0f + 0.35f * distance );
		float demand = impulse * align * bottleneck * decay;
		if ( impact )
		{
			bond->impactDemand += demand;
			fractureWorld->maxImpactDemand = b2MaxFloat( fractureWorld->maxImpactDemand, bond->impactDemand );
		}
		else
		{
			bond->loadDemand += demand;
			fractureWorld->maxLoadDemand = b2MaxFloat( fractureWorld->maxLoadDemand, bond->loadDemand );
		}
	}
}

static void b2BlastClassifyAndCommandSplits( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor )
{
	if ( actor == NULL || actor->leafCount <= 1 || actor->componentScratchCapacity < actor->leafCount ||
		 actor->queueScratchCapacity < actor->leafCount || actor->visitScratchCapacity < actor->leafCount )
	{
		return;
	}

	memset( actor->visitScratch, 0, (size_t)actor->leafCount * sizeof( uint8_t ) );
	int componentCount = 0;
	for ( int start = 0; start < actor->leafCount; ++start )
	{
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

		if ( componentCount > 0 && supported == false && fractureWorld->commandCount < fractureWorld->commandCapacity )
		{
			b2BlastFractureCommand* command = fractureWorld->commands + fractureWorld->commandCount++;
			*command = (b2BlastFractureCommand){ 0 };
			command->kind = b2_blastFractureCommandDetachActor;
			command->actorId = actor->id;
			command->leafIndex = (uint32_t)start;
			command->targetMobility = b2_blastActorMobilityDynamic;
			fractureWorld->actorTransitionCount += 1;
		}
		componentCount += 1;
	}
}

static void b2BlastRunDamageShader( b2BlastFractureWorld* fractureWorld, b2BlastActor* actor )
{
	bool anyBreak = false;
	for ( int i = 0; i < actor->bondCount; ++i )
	{
		b2BlastActiveBond* bond = actor->bonds + i;
		if ( ( bond->flags & b2_blastBondFlagBroken ) != 0 )
		{
			continue;
		}
		float demand = bond->impactDemand + bond->loadDemand;
		if ( demand <= 0.0f )
		{
			continue;
		}
		float ratio = demand / b2MaxFloat( bond->capacity, 1.0f );
		if ( ratio > 0.35f )
		{
			float added = ( ratio - 0.35f ) * 0.55f;
			bond->damage = b2ClampFloat( bond->damage + added, 0.0f, 1.5f );
			bond->flags |= b2_blastBondFlagBreakCandidate;
			fractureWorld->maxDamage = b2MaxFloat( fractureWorld->maxDamage, bond->damage );
		}
		if ( ratio > 1.0f || bond->damage >= 1.0f )
		{
			bond->flags |= b2_blastBondFlagBroken;
			anyBreak = true;
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

static void b2BlastConsumeContact( b2World* world, b2Contact* contact, float timeStep )
{
	if ( contact == NULL )
	{
		return;
	}

	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	b2Shape* shapeA = b2ShapeArray_Get( &world->shapes, contact->shapeIdA );
	b2Shape* shapeB = b2ShapeArray_Get( &world->shapes, contact->shapeIdB );
	b2BlastActor* actorA = b2BlastGetActor( fractureWorld, shapeA->blastActorId );
	b2BlastActor* actorB = b2BlastGetActor( fractureWorld, shapeB->blastActorId );
	if ( actorA == NULL && actorB == NULL )
	{
		return;
	}

	b2ContactSim* sim = b2GetContactSim( world, contact );
	if ( sim == NULL || sim->manifold.pointCount <= 0 )
	{
		return;
	}

	b2Body* bodyA = b2BodyArray_Get( &world->bodies, contact->edges[0].bodyId );
	b2Body* bodyB = b2BodyArray_Get( &world->bodies, contact->edges[1].bodyId );
	b2Transform transformA = b2GetBodyTransformQuick( world, bodyA );
	b2Transform transformB = b2GetBodyTransformQuick( world, bodyB );
	float invDt = timeStep > 0.0f ? 1.0f / timeStep : 0.0f;

	for ( int pointIndex = 0; pointIndex < sim->manifold.pointCount; ++pointIndex )
	{
		const b2ManifoldPoint* point = sim->manifold.points + pointIndex;
		float normalImpulse = b2MaxFloat( point->totalNormalImpulse, point->normalImpulse );
		float tangentImpulse = point->totalTangentImpulse;
		if ( normalImpulse <= 0.0001f && fabsf( tangentImpulse ) <= 0.0001f && point->yielded == false )
		{
			continue;
		}

		fractureWorld->constraintRowCount += 1;
		b2Vec2 worldPointA = b2Add( transformA.p, point->anchorA );
		b2Vec2 worldPointB = b2Add( transformB.p, point->anchorB );
		b2Vec2 worldPoint = b2MulSV( 0.5f, b2Add( worldPointA, worldPointB ) );
		b2Vec2 tangent = b2RightPerp( sim->manifold.normal );

		if ( actorA != NULL )
		{
			b2Vec2 local = b2InvTransformPoint( transformA, worldPoint );
			int leaf = b2BlastFindNearestLeaf( actorA, local );
			if ( point->yielded && point->unresolvedNormalImpulse > 0.0001f )
			{
				b2BlastApplyDemandFromLeaf( fractureWorld, actorA, leaf, point->unresolvedNormalImpulse, sim->manifold.normal, true );
			}
			if ( ( actorA->flags & b2_blastActorFlagOwnsWorldAnchor ) != 0 )
			{
				float force = ( normalImpulse + fabsf( tangentImpulse ) ) * invDt;
				b2BlastApplyDemandFromLeaf( fractureWorld, actorA, leaf, force, b2Add( sim->manifold.normal, tangent ), false );
			}
		}

		if ( actorB != NULL )
		{
			b2Vec2 local = b2InvTransformPoint( transformB, worldPoint );
			int leaf = b2BlastFindNearestLeaf( actorB, local );
			b2Vec2 reverseNormal = b2Neg( sim->manifold.normal );
			if ( point->yielded && point->unresolvedNormalImpulse > 0.0001f )
			{
				b2BlastApplyDemandFromLeaf( fractureWorld, actorB, leaf, point->unresolvedNormalImpulse, reverseNormal, true );
			}
			if ( ( actorB->flags & b2_blastActorFlagOwnsWorldAnchor ) != 0 )
			{
				float force = ( normalImpulse + fabsf( tangentImpulse ) ) * invDt;
				b2BlastApplyDemandFromLeaf( fractureWorld, actorB, leaf, force, b2Sub( tangent, sim->manifold.normal ), false );
			}
		}
	}
}

void b2BlastFractureWorld_CollectAndStep( b2World* world, float timeStep )
{
	if ( world == NULL )
	{
		return;
	}

	b2BlastFractureWorld* fractureWorld = &world->blastFractureWorld;
	b2BlastFractureWorld_ClearStep( fractureWorld );

	for ( int i = 0; i < world->contacts.count; ++i )
	{
		b2Contact* contact = world->contacts.data + i;
		if ( contact->setIndex == B2_NULL_INDEX || ( contact->flags & b2_contactTouchingFlag ) == 0 )
		{
			continue;
		}
		b2BlastConsumeContact( world, contact, timeStep );
	}

	for ( int i = 0; i < fractureWorld->actorCount; ++i )
	{
		b2BlastActor* actor = fractureWorld->actors + i;
		if ( ( actor->flags & b2_blastActorFlagInUse ) == 0 )
		{
			continue;
		}
		b2BlastRunDamageShader( fractureWorld, actor );
		fractureWorld->scratchCapacityHighWater =
			b2MaxInt( (int)fractureWorld->scratchCapacityHighWater, actor->componentScratchCapacity + actor->queueScratchCapacity );
	}
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
		}
	}

	snapshot.commandCount = (uint32_t)fractureWorld->commandCount;
	snapshot.constraintRowCount = fractureWorld->constraintRowCount;
	snapshot.actorTransitionCount = fractureWorld->actorTransitionCount;
	snapshot.reauthoredFallbackCount = fractureWorld->reauthoredFallbackCount;
	snapshot.legacyHostFracturePathCount = fractureWorld->legacyHostFracturePathCount;
	snapshot.stepAllocationFallbackCount = fractureWorld->stepAllocationFallbackCount;
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

b2BlastFractureDebugSnapshot b2World_GetBlastFractureDebugSnapshot( b2WorldId worldId )
{
	b2World* world = b2GetWorldFromId( worldId );
	return b2BlastFractureWorld_GetDebugSnapshot( &world->blastFractureWorld );
}

static b2BlastActor* b2BlastFindBodyActor( b2World* world, b2Body* body )
{
	if ( world == NULL || body == NULL )
	{
		return NULL;
	}
	b2BlastActor* actor = b2BlastGetActor( &world->blastFractureWorld, body->blastActorId );
	if ( actor != NULL )
	{
		return actor;
	}
	for ( int shapeId = body->headShapeId; shapeId != B2_NULL_INDEX; )
	{
		b2Shape* shape = b2ShapeArray_Get( &world->shapes, shapeId );
		actor = b2BlastGetActor( &world->blastFractureWorld, shape->blastActorId );
		if ( actor != NULL )
		{
			return actor;
		}
		shapeId = shape->nextShapeId;
	}
	return NULL;
}

bool b2World_SubmitBlastImpactAtPoint(
	b2WorldId worldId, b2BodyId bodyId, b2Vec2 worldPoint, b2Vec2 direction, float impulse, float radius, float damageHint )
{
	B2_UNUSED( radius, damageHint );
	b2World* world = b2GetWorldFromId( worldId );
	b2Body* body = b2GetBodyFullId( world, bodyId );
	b2BlastActor* actor = b2BlastFindBodyActor( world, body );
	if ( actor == NULL || impulse <= 0.0f )
	{
		return false;
	}
	b2Transform transform = b2GetBodyTransformQuick( world, body );
	b2Vec2 localPoint = b2InvTransformPoint( transform, worldPoint );
	int leaf = b2BlastFindNearestLeaf( actor, localPoint );
	b2BlastApplyDemandFromLeaf( &world->blastFractureWorld, actor, leaf, impulse, direction, true );
	b2BlastRunDamageShader( &world->blastFractureWorld, actor );
	return true;
}

bool b2World_SubmitBlastLoadAtPoint(
	b2WorldId worldId, b2BodyId bodyId, b2Vec2 worldPoint, b2Vec2 force, b2BlastExternalConstraintKind kind, uint32_t constraintId )
{
	B2_UNUSED( kind, constraintId );
	b2World* world = b2GetWorldFromId( worldId );
	b2Body* body = b2GetBodyFullId( world, bodyId );
	b2BlastActor* actor = b2BlastFindBodyActor( world, body );
	if ( actor == NULL || b2LengthSquared( force ) <= 0.000001f )
	{
		return false;
	}
	if ( ( actor->flags & b2_blastActorFlagOwnsWorldAnchor ) == 0 )
	{
		return true;
	}
	b2Transform transform = b2GetBodyTransformQuick( world, body );
	b2Vec2 localPoint = b2InvTransformPoint( transform, worldPoint );
	int leaf = b2BlastFindNearestLeaf( actor, localPoint );
	b2BlastApplyDemandFromLeaf( &world->blastFractureWorld, actor, leaf, b2Length( force ), force, false );
	b2BlastRunDamageShader( &world->blastFractureWorld, actor );
	return true;
}
