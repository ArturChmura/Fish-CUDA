#pragma once
#include <stdlib.h>
#include <time.h>
#include <vector>
#include "dataModel.h"
#include "fishesTypes.h"
#include "cudaVectorHelpers.h"

/// <summary>
/// Initialize arrays with random values 
/// </summary>
/// <param name="n">Number of elements</param>
/// <param name="sociableProportion">Portion of sociable fishes</param>
/// <param name="asocialProportion">Portion of asocial fishes</param>
/// <param name="crowdFollowingProportion">Portion of crowd following fishes</param>
/// <param name="maxSpeed">Maximum speed of a fish</param>
/// <param name="cubeSize">Cube size</param>
/// <param name="outPositions">Position array to be filled with random values</param>
/// <param name="outVelocities">Velocities array to be filled with random values</param>
/// <param name="outFishesTypes">Types array to be filled with values</param>
void radnomInitialize(
	int n,
	float sociableProportion,
	float asocialProportion,
	float crowdFollowingProportion,
	float maxSpeed,
	int cubeSize,
	float3* outPositions,
	float3* outVelocities,
	unsigned char* outFishesTypes
)
{
	float sum = sociableProportion + asocialProportion + crowdFollowingProportion;
	int socialCount = int(n * sociableProportion / sum);
	int asocialCount = int(n * asocialProportion / sum);
	for (int i = 0; i < socialCount; i++) outFishesTypes[i] = sociable;
	for (int i = socialCount; i < (socialCount + asocialCount); i++) outFishesTypes[i] = asocial;
	for (int i = (socialCount + asocialCount); i < n; i++) outFishesTypes[i] = crowdFollowing;


	srand((int)time(NULL));

	for (size_t i = 0; i < n; i++)
	{
		float xSpeed = ((float)rand()) / (float)RAND_MAX - 0.5f;
		float ySpeed = ((float)rand()) / (float)RAND_MAX - 0.5f;
		float zSpeed = ((float)rand()) / (float)RAND_MAX - 0.5f;

		outVelocities[i] = { xSpeed,ySpeed,zSpeed };
		float speed = (((float)rand()) / ((float)RAND_MAX) / 2 + 0.5) * maxSpeed;
		outVelocities[i] = ChangeLength(outVelocities[i], speed);


		float xFloat = ((float)rand()) / ((float)RAND_MAX);
		float yFloat = ((float)rand()) / ((float)RAND_MAX);
		float zFloat = ((float)rand()) / ((float)RAND_MAX);
		float xPosition = cubeSize * xFloat;
		float yPosition = cubeSize * yFloat;
		float zPosition = cubeSize * zFloat;

		outPositions[i] = { xPosition ,yPosition,zPosition };
	}
}

