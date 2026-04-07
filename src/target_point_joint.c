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

void b2TargetPointJoint_SetTargetPoint( b2JointId jointId, b2Vec2 targetPoint )
{
	b2JointSim* base = b2GetJointSimCheckType( jointId, b2_targetPointJoint );
	base->targetPointJoint.targetPoint = targetPoint;
}

b2Vec2 b2TargetPointJoint_GetTargetPoint( b2JointId jointId )
{
	b2JointSim* base = b2GetJointSimCheckType( jointId, b2_targetPointJoint );
	return base->targetPointJoint.targetPoint;
}

void b2TargetPointJoint_SetSpringHertz( b2JointId jointId, float hertz )
{
	b2JointSim* base = b2GetJointSimCheckType( jointId, b2_targetPointJoint );
	base->targetPointJoint.hertz = hertz;
}

float b2TargetPointJoint_GetSpringHertz( b2JointId jointId )
{
	b2JointSim* base = b2GetJointSimCheckType( jointId, b2_targetPointJoint );
	return base->targetPointJoint.hertz;
}

void b2TargetPointJoint_SetSpringDampingRatio( b2JointId jointId, float dampingRatio )
{
	b2JointSim* base = b2GetJointSimCheckType( jointId, b2_targetPointJoint );
	base->targetPointJoint.dampingRatio = dampingRatio;
}

float b2TargetPointJoint_GetSpringDampingRatio( b2JointId jointId )
{
	b2JointSim* base = b2GetJointSimCheckType( jointId, b2_targetPointJoint );
	return base->targetPointJoint.dampingRatio;
}

void b2TargetPointJoint_SetMaxForce( b2JointId jointId, float maxForce )
{
	b2JointSim* base = b2GetJointSimCheckType( jointId, b2_targetPointJoint );
	base->targetPointJoint.maxForce = b2MaxFloat( 0.0f, maxForce );
}

float b2TargetPointJoint_GetMaxForce( b2JointId jointId )
{
	b2JointSim* base = b2GetJointSimCheckType( jointId, b2_targetPointJoint );
	return base->targetPointJoint.maxForce;
}

void b2TargetPointJoint_SetBreakDistance( b2JointId jointId, float breakDistance )
{
	b2JointSim* base = b2GetJointSimCheckType( jointId, b2_targetPointJoint );
	base->targetPointJoint.breakDistance = b2MaxFloat( 0.0f, breakDistance );
}

float b2TargetPointJoint_GetBreakDistance( b2JointId jointId )
{
	b2JointSim* base = b2GetJointSimCheckType( jointId, b2_targetPointJoint );
	return base->targetPointJoint.breakDistance;
}

b2Vec2 b2GetTargetPointJointForce( b2World* world, b2JointSim* base )
{
	return b2MulSV( world->inv_h, base->targetPointJoint.impulse );
}

float b2GetTargetPointJointTorque( b2World* world, b2JointSim* base )
{
	B2_UNUSED( world, base );
	return 0.0f;
}

void b2PrepareTargetPointJoint( b2JointSim* base, b2StepContext* context )
{
	B2_ASSERT( base->type == b2_targetPointJoint );

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

	float mA = bodySimA->invMass;
	float iA = bodySimA->invInertia;
	float mB = bodySimB->invMass;
	float iB = bodySimB->invInertia;

	base->invMassA = mA;
	base->invMassB = mB;
	base->invIA = iA;
	base->invIB = iB;

	b2TargetPointJoint* joint = &base->targetPointJoint;
	joint->indexA = bodyA->setIndex == b2_awakeSet ? localIndexA : B2_NULL_INDEX;
	joint->indexB = bodyB->setIndex == b2_awakeSet ? localIndexB : B2_NULL_INDEX;
	joint->anchorA = b2RotateVector( bodySimA->transform.q, b2Sub( base->localFrameA.p, bodySimA->localCenter ) );
	joint->anchorB = b2RotateVector( bodySimB->transform.q, b2Sub( base->localFrameB.p, bodySimB->localCenter ) );
	joint->deltaCenter = b2Sub( bodySimB->center, bodySimA->center );
	joint->targetDelta = b2Sub( joint->targetPoint, b2Add( bodySimA->center, joint->anchorA ) );
	joint->springSoftness = b2MakeSoft( joint->hertz, joint->dampingRatio, context->h );

	b2Vec2 rA = joint->anchorA;
	b2Vec2 rB = joint->anchorB;
	b2Mat22 k;
	k.cx.x = mA + mB + rA.y * rA.y * iA + rB.y * rB.y * iB;
	k.cx.y = -rA.y * rA.x * iA - rB.y * rB.x * iB;
	k.cy.x = k.cx.y;
	k.cy.y = mA + mB + rA.x * rA.x * iA + rB.x * rB.x * iB;
	joint->linearMass = b2GetInverse22( k );

	if ( context->enableWarmStarting == false )
	{
		joint->impulse = b2Vec2_zero;
	}
}

void b2WarmStartTargetPointJoint( b2JointSim* base, b2StepContext* context )
{
	B2_ASSERT( base->type == b2_targetPointJoint );

	float mA = base->invMassA;
	float mB = base->invMassB;
	float iA = base->invIA;
	float iB = base->invIB;

	b2TargetPointJoint* joint = &base->targetPointJoint;
	b2BodyState dummyState = b2_identityBodyState;

	b2BodyState* stateA = joint->indexA == B2_NULL_INDEX ? &dummyState : context->states + joint->indexA;
	b2BodyState* stateB = joint->indexB == B2_NULL_INDEX ? &dummyState : context->states + joint->indexB;

	b2Vec2 rA = b2RotateVector( stateA->deltaRotation, joint->anchorA );
	b2Vec2 rB = b2RotateVector( stateB->deltaRotation, joint->anchorB );
	b2Vec2 impulse = joint->impulse;

	if ( stateA->flags & b2_dynamicFlag )
	{
		stateA->linearVelocity = b2MulSub( stateA->linearVelocity, mA, impulse );
		stateA->angularVelocity -= iA * b2Cross( rA, impulse );
	}

	if ( stateB->flags & b2_dynamicFlag )
	{
		stateB->linearVelocity = b2MulAdd( stateB->linearVelocity, mB, impulse );
		stateB->angularVelocity += iB * b2Cross( rB, impulse );
	}
}

void b2SolveTargetPointJoint( b2JointSim* base, b2StepContext* context, bool useBias )
{
	B2_UNUSED( useBias );
	B2_ASSERT( base->type == b2_targetPointJoint );

	float mA = base->invMassA;
	float mB = base->invMassB;
	float iA = base->invIA;
	float iB = base->invIB;

	b2TargetPointJoint* joint = &base->targetPointJoint;
	b2BodyState dummyState = b2_identityBodyState;

	b2BodyState* stateA = joint->indexA == B2_NULL_INDEX ? &dummyState : context->states + joint->indexA;
	b2BodyState* stateB = joint->indexB == B2_NULL_INDEX ? &dummyState : context->states + joint->indexB;

	b2Vec2 vA = stateA->linearVelocity;
	float wA = stateA->angularVelocity;
	b2Vec2 vB = stateB->linearVelocity;
	float wB = stateB->angularVelocity;
	b2Vec2 dcA = stateA->deltaPosition;
	b2Vec2 dcB = stateB->deltaPosition;

	b2Vec2 rA = b2RotateVector( stateA->deltaRotation, joint->anchorA );
	b2Vec2 rB = b2RotateVector( stateB->deltaRotation, joint->anchorB );
	b2Vec2 c = b2Add( b2Add( b2Sub( dcB, dcA ), b2Sub( rB, rA ) ), joint->deltaCenter );
	c = b2Sub( c, joint->targetDelta );

	if ( joint->breakDistance > 0.0f && b2LengthSquared( c ) > joint->breakDistance * joint->breakDistance )
	{
		joint->impulse = b2Vec2_zero;
		stateA->linearVelocity = vA;
		stateA->angularVelocity = wA;
		stateB->linearVelocity = vB;
		stateB->angularVelocity = wB;
		return;
	}

	b2Mat22 k;
	k.cx.x = mA + mB + rA.y * rA.y * iA + rB.y * rB.y * iB;
	k.cx.y = -rA.y * rA.x * iA - rB.y * rB.x * iB;
	k.cy.x = k.cx.y;
	k.cy.y = mA + mB + rA.x * rA.x * iA + rB.x * rB.x * iB;
	joint->linearMass = b2GetInverse22( k );

	b2Vec2 bias = b2MulSV( joint->springSoftness.biasRate, c );
	float massScale = joint->springSoftness.massScale;
	float impulseScale = joint->springSoftness.impulseScale;
	b2Vec2 cdot = b2Sub( b2Add( vB, b2CrossSV( wB, rB ) ), b2Add( vA, b2CrossSV( wA, rA ) ) );
	cdot = b2Add( cdot, bias );

	b2Vec2 b = b2MulMV( joint->linearMass, cdot );
	b2Vec2 oldImpulse = joint->impulse;
	b2Vec2 impulse = {
		-massScale * b.x - impulseScale * oldImpulse.x,
		-massScale * b.y - impulseScale * oldImpulse.y,
	};

	float maxImpulse = context->h * joint->maxForce;
	joint->impulse = b2Add( oldImpulse, impulse );
	float impulseLength = b2Length( joint->impulse );
	if ( impulseLength > maxImpulse && impulseLength > 0.0f )
	{
		joint->impulse = b2MulSV( maxImpulse / impulseLength, joint->impulse );
	}
	impulse = b2Sub( joint->impulse, oldImpulse );

	vA = b2MulSub( vA, mA, impulse );
	wA -= iA * b2Cross( rA, impulse );
	vB = b2MulAdd( vB, mB, impulse );
	wB += iB * b2Cross( rB, impulse );

	stateA->linearVelocity = vA;
	stateA->angularVelocity = wA;
	stateB->linearVelocity = vB;
	stateB->angularVelocity = wB;
}

void b2DrawTargetPointJoint( b2DebugDraw* draw, b2JointSim* base, b2Transform transformA, b2Transform transformB, float drawScale )
{
	B2_UNUSED( drawScale );

	b2Vec2 pA = b2TransformPoint( transformA, base->localFrameA.p );
	b2Vec2 pB = b2TransformPoint( transformB, base->localFrameB.p );
	b2Vec2 targetPoint = base->targetPointJoint.targetPoint;

	draw->DrawLineFcn( pA, targetPoint, b2_colorGold, draw->context );
	draw->DrawLineFcn( targetPoint, pB, b2_colorLightSkyBlue, draw->context );
	draw->DrawPointFcn( pA, 5.0f, b2_colorGold, draw->context );
	draw->DrawPointFcn( targetPoint, 5.0f, b2_colorOrange, draw->context );
	draw->DrawPointFcn( pB, 5.0f, b2_colorYellowGreen, draw->context );
}
