#include "dataModel.h"
#include <vector_types.h>
#include <cmath>
#include "config.h"
#include <cuda_runtime.h>


void Allocate(Data* data,
	Config config,
	float3* startPositions,
	float3* startVelocities,
	unsigned char* startFishesType)
{
	int gridSize;
	gridSize = (int)ceil(float(config.cubeSideLength) / float(config.radious));

	int gridCellCount = gridSize * gridSize * gridSize;

	data->dev_positions = nullptr;
	data->dev_velocitiesToRead = nullptr;
	data->dev_velocitiesToWrite = nullptr;
	data->dev_fishesTypes = nullptr;
	data->dev_indexesInGrid = nullptr;
	data->dev_indexes = nullptr;
	data->dev_sortedPositions = nullptr;
	data->dev_sortedVelocities = nullptr;
	data->dev_sortedNewFishesType = nullptr;
	data->dev_startIndexesForGridCells = nullptr;

	data->radious = config.radious;
	data->gridSize = gridSize;
	data->n = config.fishesCount;
	data->length = config.cubeSideLength;
	data->maxSpeed = config.maxSpeed;
	data->maxAcceleration = config.maxAcceleration;

	cudaMalloc(&data->dev_positions, config.fishesCount * sizeof(float3));
	cudaMalloc(&data->dev_velocitiesToRead, config.fishesCount * sizeof(float3));
	cudaMalloc(&data->dev_velocitiesToWrite, config.fishesCount * sizeof(float3));
	cudaMalloc(&data->dev_fishesTypes, config.fishesCount * sizeof(unsigned char));
	cudaMalloc(&data->dev_indexesInGrid, config.fishesCount * sizeof(int));
	cudaMalloc(&data->dev_startIndexesForGridCells, gridCellCount * sizeof(vectorI));
	cudaMalloc(&data->dev_indexes, config.fishesCount * sizeof(int));
	cudaMalloc(&data->dev_sortedPositions, config.fishesCount * sizeof(float3));
	cudaMalloc(&data->dev_sortedVelocities, config.fishesCount * sizeof(float3));
	cudaMalloc(&data->dev_sortedNewFishesType, config.fishesCount * sizeof(unsigned char));

	cudaMemcpy(data->dev_positions, startPositions, config.fishesCount * sizeof(float3), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(data->dev_velocitiesToRead, startVelocities, config.fishesCount * sizeof(float3), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(data->dev_fishesTypes, startFishesType, config.fishesCount * sizeof(unsigned char), cudaMemcpyKind::cudaMemcpyHostToDevice);
}