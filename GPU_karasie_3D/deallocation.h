#pragma once
#include "dataModel.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

int Deallocate(Data gpuData)
{
	cudaFree(gpuData.dev_positions);
	cudaFree(gpuData.dev_velocitiesToRead);
	cudaFree(gpuData.dev_velocitiesToWrite);
	cudaFree(gpuData.dev_fishesTypes);
	cudaFree(gpuData.dev_indexesInGrid);
	cudaFree(gpuData.dev_startIndexesForGridCells);
	cudaFree(gpuData.dev_indexes);
	cudaFree(gpuData.dev_sortedPositions);
	cudaFree(gpuData.dev_sortedVelocities);
	cudaFree(gpuData.dev_sortedNewFishesType);

	if (cudaDeviceReset() != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}