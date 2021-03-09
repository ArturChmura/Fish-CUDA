
#pragma once
#include "cuda_runtime.h"
#include "vectors.h"
#include <math.h>
#include <cuda_runtime.h>

# define M_PI           3.14159265358979323846

/// <summary>
/// Changes vector length, and keeps its direction
/// </summary>
/// <param name="v">Direction Vector</param>
/// <param name="newLength">Length of new calculated vector</param>
/// <returns>New vector of given length and direction</returns>
__host__ __device__ float3 ChangeLength(float3 v, float newLength)
{
	float length = v.x * v.x + v.y * v.y + v.z * v.z;

	length = sqrtf(length);

	float3 res{};
	res.x = newLength * v.x / length;
	res.y = newLength * v.y / length;
	res.z = newLength * v.z / length;

	return  res;
}

/// <summary>
/// Calculate squared distance between 2 vectors
/// </summary>
/// <param name="v1">First vector</param>
/// <param name="v2">Second vector</param>
/// <returns>Squared distance between given vectors</returns>
__host__ __device__ float DistanceSquared(float3 v1, float3 v2)
{
	float distanceSquared = 0;

	distanceSquared += (v1.x - v2.x) * (v1.x - v2.x);
	distanceSquared += (v1.y - v2.y) * (v1.y - v2.y);
	distanceSquared += (v1.z - v2.z) * (v1.z - v2.z);

	return distanceSquared;
}

/// <summary>
/// Add 2 vectors
/// </summary>
/// <param name="v1">First vector</param>
/// <param name="v2">Second vector</param>
/// <returns>New vector of summed coordinates</returns>
__host__ __device__ float3 AddVectors(float3 v1, float3 v2)
{
	return { v1.x + v2.x ,v1.y + v2.y ,v1.z + v2.z };
}
/// <summary>
/// Subtract vectors
/// </summary>
/// <param name="v1">First vector</param>
/// <param name="v2">Second vector</param>
/// <returns>New vector of subtracted coordinates</returns>
__host__ __device__ float3 SubtractVectors(float3 v1, float3 v2)
{
	return { v1.x - v2.x ,v1.y - v2.y ,v1.z - v2.z };
}

/// <summary>
/// Divide vector by const value
/// </summary>
/// <param name="v">In vector</param>
/// <param name="value">Const value</param>
/// <returns>New vector </returns>
__host__ __device__ float3 DivideVector(float3 v, float value)
{
	return { v.x / value ,v.y / value ,v.z / value };
}

__host__ __device__ float3 NegateVector(float3 v)
{
	return { -v.x ,-v.y ,-v.z };
}

/// <summary>
/// Calculates dot product of vectors a and b
/// </summary>
/// <param name="a">Firtst vector</param>
/// <param name="b">Second vector</param>
/// <returns>Dot product of given vectors</returns>
__host__ __device__ float Dot(float3 a, float3 b)  
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// <summary>
/// Calculates length of a vector
/// </summary>
/// <param name="a">Vector</param>
/// <returns>Length of a vector</returns>
__device__ float VectorLength(float3 a)  //calculates magnitude of a
{
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

/// <summary>
/// Calculates squared length of a vector
/// </summary>
/// <param name="a">Vector</param>
/// <returns>Squared length of a vector</returns>
__device__ float VectorLengthSquared(float3 a)  //calculates magnitude of a
{
	return a.x * a.x + a.y * a.y + a.z * a.z;
}

/// <summary>
/// Calculates angle between 2 vectors
/// </summary>
/// <param name="v1">First vector</param>
/// <param name="v2">Second vector</param>
/// <returns>Angle between given vectors</returns>
__device__ float Angle(float3 v1, float3 v2)
{
	float r = Dot(v1, v2) / (VectorLength(v1) * VectorLength(v2));
	float angle = acos(r);
	return angle;
}

/// <summary>
/// Calculates angle between current fish direction and position of its neighbor
/// </summary>
/// <param name="fishPosition">Current fish position</param>
/// <param name="fishVelocity">Vector of central fish velocity</param>
/// <param name="neighborPosition">Neighbor fish position</param>
/// <returns>Angle between  between current fish direction and position of its neighbor</returns>
__device__ float CalculateFishesAngle(float3 fishPosition, float3 fishVelocity, float3 neighborPosition)
{
	float3 vectorToNeighbor = make_float3(neighborPosition.x - fishPosition.x, neighborPosition.y - fishPosition.y, neighborPosition.z - fishPosition.z);
	return Angle(fishVelocity, vectorToNeighbor);
}

/// <summary>
/// Calcualtes roll, pitch and yaw of a vector. Initial vector is parallel to X asix.
/// </summary>
/// <param name="d">Rotation vector</param>
/// <returns>Roll, pitch, yaw</returns>
__device__ float3 CalculateRPY(float3 d)
{
	d = ChangeLength(d, 1);
	float roll = 0;
	float pitch = asin(-d.z);
	float yaw = atan2(d.y, d.x);

	return { roll,pitch, yaw };
}

/// <summary>
/// Calcualtes roll, pitch and yaw to rotate vector to other vector direction
/// </summary>
/// <param name="v1">Direction vector</param>
/// <param name="v2">Vector to rotate</param>
/// <returns>Roll, pitch, yaw</returns>
__device__ float3 CalculateRPY(float3 v1, float3 v2)
{
	float3 RPY1 = CalculateRPY(v1);
	float3 RPY2 = CalculateRPY(v2);

	float dY = RPY1.z - RPY2.z;
	if (dY < -M_PI) dY += 2 * M_PI;
	if (dY > M_PI) dY -= 2 * M_PI;

	float dP = RPY1.y - RPY2.y;

	return { 0,dP ,dY };
}

/// <summary>
/// Rotates triangle by giver roll, pitch and yaw
/// </summary>
/// <param name="roll">Roll</param>
/// <param name="pitch">Pitch</param>
/// <param name="yaw">Yaw</param>
/// <param name="t">Triangle to ratate</param>
/// <param name="mid">The middle of ratation</param>
__device__ void RotateTriangle(float roll, float pitch, float yaw, triangle* t, float3 mid) {

	float cosa = cos(yaw);
	float sina = sin(yaw);

	float cosb = cos(pitch);
	float sinb = sin(pitch);

	float cosc = cos(roll);
	float sinc = sin(roll);

	float Axx = cosa * cosb;
	float Axy = cosa * sinb * sinc - sina * cosc;
	float Axz = cosa * sinb * cosc + sina * sinc;

	float Ayx = sina * cosb;
	float Ayy = sina * sinb * sinc + cosa * cosc;
	float Ayz = sina * sinb * cosc - cosa * sinc;

	float Azx = -sinb;
	float Azy = cosb * sinc;
	float Azz = cosb * cosc;

	t->a.x -= mid.x;
	t->a.y -= mid.y;
	t->a.z -= mid.z;

	t->b.x -= mid.x;
	t->b.y -= mid.y;
	t->b.z -= mid.z;

	t->c.x -= mid.x;
	t->c.y -= mid.y;
	t->c.z -= mid.z;


	float px = (*t).a.x;
	float py = (*t).a.y;
	float pz = (*t).a.z;

	(*t).a.x = Axx * px + Axy * py + Axz * pz;
	(*t).a.y = Ayx * px + Ayy * py + Ayz * pz;
	(*t).a.z = Azx * px + Azy * py + Azz * pz;

	px = (*t).b.x;
	py = (*t).b.y;
	pz = (*t).b.z;

	(*t).b.x = Axx * px + Axy * py + Axz * pz;
	(*t).b.y = Ayx * px + Ayy * py + Ayz * pz;
	(*t).b.z = Azx * px + Azy * py + Azz * pz;

	px = (*t).c.x;
	py = (*t).c.y;
	pz = (*t).c.z;

	(*t).c.x = Axx * px + Axy * py + Axz * pz;
	(*t).c.y = Ayx * px + Ayy * py + Ayz * pz;
	(*t).c.z = Azx * px + Azy * py + Azz * pz;

	t->a.x += mid.x;
	t->a.y += mid.y;
	t->a.z += mid.z;

	t->b.x += mid.x;
	t->b.y += mid.y;
	t->b.z += mid.z;

	t->c.x += mid.x;
	t->c.y += mid.y;
	t->c.z += mid.z;
}

/// <summary>
/// Rotates vector by giver roll, pitch and yaw
/// </summary>
/// <param name="roll">Roll</param>
/// <param name="pitch">Pitch</param>
/// <param name="yaw">Yaw</param>
/// <param name="v">Vecotor to ratate</param>
__device__ void RotateVector(float roll, float pitch, float yaw, float3* v) {

	float cosa = cos(yaw);
	float sina = sin(yaw);

	float cosb = cos(pitch);
	float sinb = sin(pitch);

	float cosc = cos(roll);
	float sinc = sin(roll);

	float Axx = cosa * cosb;
	float Axy = cosa * sinb * sinc - sina * cosc;
	float Axz = cosa * sinb * cosc + sina * sinc;

	float Ayx = sina * cosb;
	float Ayy = sina * sinb * sinc + cosa * cosc;
	float Ayz = sina * sinb * cosc - cosa * sinc;

	float Azx = -sinb;
	float Azy = cosb * sinc;
	float Azz = cosb * cosc;

	float px = (*v).x;
	float py = (*v).y;
	float pz = (*v).z;

	(*v).x = Axx * px + Axy * py + Axz * pz;
	(*v).y = Ayx * px + Ayy * py + Ayz * pz;
	(*v).z = Azx * px + Azy * py + Azz * pz;
}

/// <summary>
/// Converts given coordinates to values between [-1,1]
/// </summary>
/// <param name="v">Coordinates</param>
/// <param name="cubeSize">Size of cube (max coorinates value)</param>
/// <returns>Converted values between [-1,1]</returns>
__device__ float5 ConvertToOGL(float5 v, int cubeSize)
{
	return { v.x * 2 / cubeSize - 1.0f, v.y * 2 / cubeSize - 1.0f, v.z * 2 / cubeSize - 1.0f, v.d4,v.color };
}

