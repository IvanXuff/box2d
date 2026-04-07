// SPDX-FileCopyrightText: 2023 Erin Catto
// SPDX-License-Identifier: MIT

#include "body.h"
#include "core.h"
#include "joint.h"
#include "physics_world.h"
#include "solver.h"
#include "solver_set.h"

// needed for dll export
#include "box2d/box2d.h"

#include <stdio.h>

b2Vec2 b2GetCharacterGroundJointForce( b2World* world, b2JointSim* base )
{
	b2Transform transformA = b2GetBodyTransform( world, base->bodyIdA );
	b2Vec2 tangent = b2RotateVector( transformA.q, b2RotateVector( base->localFrameA.q, (b2Vec2){ 1.0f, 0.0f } ) );
	b2Vec2 normal = b2RightPerp( tangent );

	b2CharacterGroundJoint* joint = &base->characterGroundJoint;
	float inv_h = world->inv_h;
	return b2Add(
		b2MulSV( inv_h * ( joint->motorImpulse + joint->tangentHoldImpulse ), tangent ),
		b2MulSV( inv_h * joint->normalImpulse, normal ) );
}

float b2GetCharacterGroundJointTorque( b2World* world, b2JointSim* base )
{
	B2_UNUSED( world, base );
	return 0.0f;
}

void b2PrepareCharacterGroundJoint( b2JointSim* base, b2StepContext* context )
{
	B2_ASSERT( base->type == b2_characterGroundJoint );

	int idA = base->bodyIdA;
	int idB = base->bodyIdB;

	b2World* world = context->world;
	b2Body* bodyA = b2BodyArray_Get( &world->bodies, idA );
	b2Body* bodyB = b2BodyArray_Get( &world->bodies, idB );

	B2_ASSERT( bodyA->setIndex == b2_awakeSet || bodyB->setIndex == b2_awakeSet );
	b2SolverSet* setA = b2SolverSetArray_Get( &world->solverSets, bodyA->setIndex );
	b2SolverSet* setB = b2SolverSetArray_Get( &world->solverSets, bodyB->setIndex );

	int localIndexA = bodyA->localIndex;
	int localIndexB = bodyB->localIndex;

	b2BodySim* bodySimA = b2BodySimArray_Get( &setA->bodySims, localIndexA );
	b2BodySim* bodySimB = b2BodySimArray_Get( &setB->bodySims, localIndexB );

	base->invMassA = bodySimA->invMass;
	base->invMassB = bodySimB->invMass;
	base->invIA = bodySimA->invInertia;
	base->invIB = bodySimB->invInertia;

	b2CharacterGroundJoint* joint = &base->characterGroundJoint;
	joint->indexA = bodyA->setIndex == b2_awakeSet ? localIndexA : B2_NULL_INDEX;
	joint->indexB = bodyB->setIndex == b2_awakeSet ? localIndexB : B2_NULL_INDEX;
	joint->frameA.q = b2MulRot( bodySimA->transform.q, base->localFrameA.q );
	joint->frameA.p = b2RotateVector( bodySimA->transform.q, b2Sub( base->localFrameA.p, bodySimA->localCenter ) );
	joint->anchorB = b2RotateVector( bodySimB->transform.q, b2Sub( base->localFrameB.p, bodySimB->localCenter ) );
	joint->deltaCenter = b2Sub( bodySimB->center, bodySimA->center );
	joint->tangentHoldSoftness = b2MakeSoft( joint->tangentHoldHertz, joint->tangentHoldDampingRatio, context->h );
	if ( joint->enableTangentHold == false || joint->maxTangentHoldForce <= 0.0f || joint->tangentHoldHertz <= 0.0f )
	{
		joint->tangentHoldImpulse = 0.0f;
	}

	if ( context->enableWarmStarting == false )
	{
		joint->normalImpulse = 0.0f;
		joint->motorImpulse = 0.0f;
		joint->tangentHoldImpulse = 0.0f;
	}
}

void b2WarmStartCharacterGroundJoint( b2JointSim* base, b2StepContext* context )
{
	B2_ASSERT( base->type == b2_characterGroundJoint );

	float mA = base->invMassA;
	float mB = base->invMassB;
	float iA = base->invIA;
	float iB = base->invIB;

	b2BodyState dummyState = b2_identityBodyState;
	b2CharacterGroundJoint* joint = &base->characterGroundJoint;

	b2BodyState* stateA = joint->indexA == B2_NULL_INDEX ? &dummyState : context->states + joint->indexA;
	b2BodyState* stateB = joint->indexB == B2_NULL_INDEX ? &dummyState : context->states + joint->indexB;

	b2Vec2 rA = b2RotateVector( stateA->deltaRotation, joint->frameA.p );
	b2Vec2 rB = b2RotateVector( stateB->deltaRotation, joint->anchorB );
	b2Vec2 tangent = b2RotateVector( stateA->deltaRotation, b2RotateVector( joint->frameA.q, (b2Vec2){ 1.0f, 0.0f } ) );
	b2Vec2 normal = b2RightPerp( tangent );

	b2Vec2 P = b2Add(
		b2MulSV( joint->motorImpulse + joint->tangentHoldImpulse, tangent ),
		b2MulSV( joint->normalImpulse, normal ) );
	float LA = b2Cross( rA, P );
	float LB = b2Cross( rB, P );

	if ( stateA->flags & b2_dynamicFlag )
	{
		stateA->linearVelocity = b2MulSub( stateA->linearVelocity, mA, P );
		stateA->angularVelocity -= iA * LA;
	}

	if ( stateB->flags & b2_dynamicFlag )
	{
		stateB->linearVelocity = b2MulAdd( stateB->linearVelocity, mB, P );
		stateB->angularVelocity += iB * LB;
	}
}

void b2SolveCharacterGroundJoint( b2JointSim* base, b2StepContext* context, bool useBias )
{
	B2_ASSERT( base->type == b2_characterGroundJoint );

	float mA = base->invMassA;
	float mB = base->invMassB;
	float iA = base->invIA;
	float iB = base->invIB;

	b2BodyState dummyState = b2_identityBodyState;
	b2CharacterGroundJoint* joint = &base->characterGroundJoint;

	b2BodyState* stateA = joint->indexA == B2_NULL_INDEX ? &dummyState : context->states + joint->indexA;
	b2BodyState* stateB = joint->indexB == B2_NULL_INDEX ? &dummyState : context->states + joint->indexB;

	b2Vec2 vA = stateA->linearVelocity;
	float wA = stateA->angularVelocity;
	b2Vec2 vB = stateB->linearVelocity;
	float wB = stateB->angularVelocity;

	b2Vec2 rA = b2RotateVector( stateA->deltaRotation, joint->frameA.p );
	b2Vec2 rB = b2RotateVector( stateB->deltaRotation, joint->anchorB );
	b2Vec2 d = b2Add( b2Add( b2Sub( stateB->deltaPosition, stateA->deltaPosition ), joint->deltaCenter ), b2Sub( rB, rA ) );

	b2Vec2 tangent = b2RotateVector( stateA->deltaRotation, b2RotateVector( joint->frameA.q, (b2Vec2){ 1.0f, 0.0f } ) );
	b2Vec2 normal = b2RightPerp( tangent );
	b2Softness softness = base->constraintSoftness;
	bool supportEngaged = false;

	{
		float sA = b2Cross( rA, normal );
		float sB = b2Cross( rB, normal );
		float kNormal = mA + mB + iA * sA * sA + iB * sB * sB;
		float normalMass = kNormal > 0.0f ? 1.0f / kNormal : 0.0f;
		float height = b2Dot( normal, d );
		float C = height - joint->targetHeight;
		float activationDistance = b2MaxFloat( joint->breakDistance, 0.0f );
		supportEngaged = C < activationDistance;

		if ( supportEngaged )
		{
			float bias = 0.0f;
			float massScale = 1.0f;
			float impulseScale = 0.0f;

			if ( C > 0.0f )
			{
				float safe = b2GetLengthUnitsPerMeter();
				bias = b2MinFloat( C, safe ) * context->inv_h;
			}
			else if ( useBias )
			{
				bias = softness.biasRate * C;
				massScale = softness.massScale;
				impulseScale = softness.impulseScale;
			}

			float Cdot = b2Dot( normal, b2Sub( vB, vA ) ) + sB * wB - sA * wA;
			float oldImpulse = joint->normalImpulse;
			float impulse = -normalMass * massScale * ( Cdot + bias ) - impulseScale * oldImpulse;
			joint->normalImpulse = b2MaxFloat( oldImpulse + impulse, 0.0f );
			impulse = joint->normalImpulse - oldImpulse;

			b2Vec2 P = b2MulSV( impulse, normal );
			float LA = impulse * sA;
			float LB = impulse * sB;

			vA = b2MulSub( vA, mA, P );
			wA -= iA * LA;
			vB = b2MulAdd( vB, mB, P );
			wB += iB * LB;
		}
		else
		{
			joint->normalImpulse = 0.0f;
			joint->tangentHoldImpulse = 0.0f;
		}
	}

	{
		float sA = b2Cross( rA, tangent );
		float sB = b2Cross( rB, tangent );
		float kTangent = mA + mB + iA * sA * sA + iB * sB * sB;
		float tangentMass = kTangent > 0.0f ? 1.0f / kTangent : 0.0f;
		float Cdot = b2Dot( tangent, b2Sub( vB, vA ) ) + sB * wB - sA * wA;

		if ( supportEngaged && joint->enableTangentHold && joint->maxTangentHoldForce > 0.0f && joint->tangentHoldHertz > 0.0f )
		{
			float tangentOffset = b2Dot( tangent, d );
			float C = tangentOffset - joint->targetTangentOffset;
			b2Softness holdSoftness = joint->tangentHoldSoftness;
			float bias = useBias ? holdSoftness.biasRate * C : 0.0f;
			float massScale = holdSoftness.massScale;
			float impulseScale = holdSoftness.impulseScale;
			float oldImpulse = joint->tangentHoldImpulse;
			float impulse = -massScale * tangentMass * ( Cdot + bias ) - impulseScale * oldImpulse;
			float maxImpulse = context->h * joint->maxTangentHoldForce;
			joint->tangentHoldImpulse = b2ClampFloat( oldImpulse + impulse, -maxImpulse, maxImpulse );
			impulse = joint->tangentHoldImpulse - oldImpulse;

			b2Vec2 P = b2MulSV( impulse, tangent );
			float LA = impulse * sA;
			float LB = impulse * sB;

			vA = b2MulSub( vA, mA, P );
			wA -= iA * LA;
			vB = b2MulAdd( vB, mB, P );
			wB += iB * LB;

			Cdot = b2Dot( tangent, b2Sub( vB, vA ) ) + sB * wB - sA * wA;
		}
		else
		{
			joint->tangentHoldImpulse = 0.0f;
		}

		float impulse = tangentMass * ( joint->motorSpeed - Cdot );
		float oldImpulse = joint->motorImpulse;
		float maxImpulse = context->h * joint->maxMotorForce;
		joint->motorImpulse = b2ClampFloat( oldImpulse + impulse, -maxImpulse, maxImpulse );
		impulse = joint->motorImpulse - oldImpulse;

		b2Vec2 P = b2MulSV( impulse, tangent );
		float LA = impulse * sA;
		float LB = impulse * sB;

		vA = b2MulSub( vA, mA, P );
		wA -= iA * LA;
		vB = b2MulAdd( vB, mB, P );
		wB += iB * LB;
	}

	stateA->linearVelocity = vA;
	stateA->angularVelocity = wA;
	stateB->linearVelocity = vB;
	stateB->angularVelocity = wB;
}

void b2DrawCharacterGroundJoint( b2DebugDraw* draw, b2JointSim* base, b2Transform transformA, b2Transform transformB, float drawScale )
{
	B2_UNUSED( drawScale );

	b2Vec2 pA = b2TransformPoint( transformA, base->localFrameA.p );
	b2Vec2 pB = b2TransformPoint( transformB, base->localFrameB.p );
	b2Vec2 tangent = b2RotateVector( transformA.q, b2RotateVector( base->localFrameA.q, (b2Vec2){ 1.0f, 0.0f } ) );
	b2Vec2 normal = b2RightPerp( tangent );
	b2Vec2 targetPoint = b2MulAdd( pA, base->characterGroundJoint.targetHeight, normal );
	b2Vec2 targetHoldPoint = b2MulAdd( targetPoint, base->characterGroundJoint.targetTangentOffset, tangent );

	draw->DrawLineFcn( pA, pB, b2_colorDarkSeaGreen, draw->context );
	draw->DrawLineFcn( pA, targetPoint, b2_colorGold, draw->context );
	draw->DrawLineFcn( targetPoint, b2MulAdd( targetPoint, 0.5f * b2GetLengthUnitsPerMeter(), tangent ), b2_colorGold, draw->context );
	if ( base->characterGroundJoint.enableTangentHold )
	{
		draw->DrawLineFcn( targetPoint, targetHoldPoint, b2_colorOrange, draw->context );
		draw->DrawPointFcn( targetHoldPoint, 4.0f, b2_colorOrange, draw->context );
	}
	draw->DrawPointFcn( pA, 5.0f, b2_colorGold, draw->context );
	draw->DrawPointFcn( pB, 5.0f, b2_colorYellowGreen, draw->context );
}
