#pragma once

#include "vectors.h"

// struct of data required for cuda calculation
struct Data
{
	int n;

	triangle* openGLTriangles;
	float3* dev_positions;
	float3* dev_sortedPositions;

	float3* dev_velocitiesToRead;
	float3* dev_sortedVelocities;
	float3* dev_velocitiesToWrite;

	unsigned char* dev_fishesTypes;
	unsigned char* dev_sortedNewFishesType;

	int* dev_indexesInGrid;
	int* dev_indexes;
	vectorI* dev_startIndexesForGridCells;
	
	
	

	float radious;
	int gridSize;
	int length;
	float maxSpeed;
	float maxAcceleration;
};

