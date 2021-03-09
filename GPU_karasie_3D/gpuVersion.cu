
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>  
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm.hpp>
#include "config.h"
#include "initalization.h"
#include <cuda_gl_interop.h>
#include <cudagl.h>
#include <math.h>
#include <helper_cuda.h>  
#include <helper_functions.h> 
#include <helper_gl.h>
#include <thrust\device_ptr.h>
#include <thrust\sequence.h>
#include <thrust\sort.h>
#include "cudaVectorHelpers.h"
#include "cuda_runtime.h"
#include "dataModel.h"
#include "vectors.h"
#include "fishesTypes.h"
#include "device_launch_parameters.h"
#include "allocation.h"
#include "deallocation.h"
#include "helper_math.h"
#include "VBOs.h"
#include "helpers.h"

#define FISHSIZE 20 // height of fish triangle, also size for collisions
# define M_PI           3.14159265358979323846
# define M_PIper2           1.57079632679
#define FOV 2 // fishes field of view on one side [0, PI] 
#define MAXTURN (M_PI/50) // maximal turn that fish can make in one iteration > 0


Config config;
Data gpuData;
int REFRESH_DELAY = 0; //delay between iterations 
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;

//start camera position 
float rotate_x = 25, rotate_y = 30.0;
float translate_z = -4.0;

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;

// for FPS calculations
int fpsCount = 0;
int fpsLimit = 1;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
StopWatchInterface* timer = NULL;

//function definitions
void timerEvent(int value);
bool initGL(int* argc, char** argv);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void computeFPS();
void reshape(GLsizei width, GLsizei height);

//device pointer for thrust sorting
thrust::device_ptr<int> t_iig;
thrust::device_ptr<int> t_i;

/// <summary>
/// For each fish, sets its grid cell number
/// </summary>
/// <param name="n">Number of fish</param>
/// <param name="positions">Fish position</param>
/// <param name="radious">Fish radious of view</param>
/// <param name="gridSize">Number of cells in one direction</param>
/// <param name="outIndexesInGrid">Array to be filled with indexes</param>
/// <param name="indexes">Array to be filled with [1...n] values</param>
/// <returns></returns>
__global__ void setIndexesInGrid(int n, float3* positions, float radious, int gridSize, int* outIndexesInGrid, int* indexes)
{
	int i = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
	if (i >= n) return;
	int xIndex = floorf(positions[i].x / (radious));
	int yIndex = floorf(positions[i].y / (radious));
	int zIndex = floorf(positions[i].z / (radious));
	outIndexesInGrid[i] = gridSize * gridSize * zIndex + gridSize * yIndex + xIndex;
	indexes[i] = i;
}

/// <summary>
/// Sorts data arrays by index array
/// </summary>
/// <param name="n">Number of fish</param>
/// <param name="positions">Fish position array</param>
/// <param name="velocities">Fish velocities array</param>
/// <param name="fishesTypes">Fish types array</param>
/// <param name="indexes">Indexes for sotring</param>
/// <param name="outPositions">Out fish position array</param>
/// <param name="outVelocities">Out fish velocities array</param>
/// <param name="outFishesTypes">Out fish types array</param>
/// <param name="sortedIndexesInGrid">Array of sorted index in grid </param>
/// <param name="outStartEndIndexsesForGridCells">Out array with start and end indexes for each grid cell</param>
__global__ void sortValuesInGrid(
	int n,
	float3* positions,
	float3* velocities,
	unsigned char* fishesTypes,
	int* indexes,
	float3* outPositions,
	float3* outVelocities,
	unsigned char* outFishesTypes,
	int* sortedIndexesInGrid,
	vectorI* outStartEndIndexsesForGridCells
)
{
	int i = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
	if (i >= n) return;
	int fishIndex = indexes[i];

	outVelocities[i] = velocities[fishIndex];
	outPositions[i] = positions[fishIndex];
	outFishesTypes[i] = fishesTypes[fishIndex];

	if (i == 0)
	{
		outStartEndIndexsesForGridCells[sortedIndexesInGrid[i]].x = 0;
	}

	else if (i == n - 1)
	{
		outStartEndIndexsesForGridCells[sortedIndexesInGrid[n - 1]].y = n - 1;
	}


	if (sortedIndexesInGrid[i] != sortedIndexesInGrid[i - 1] && i > 0)
	{
		outStartEndIndexsesForGridCells[sortedIndexesInGrid[i]].x = i;
		outStartEndIndexsesForGridCells[sortedIndexesInGrid[i - 1]].y = i - 1;
	}

}

/// <summary>
/// Coputes new velocities and position for each fish, based on its type and its neighbors
/// </summary>
/// <param name="n">Number of fish</param>
/// <param name="positions">Fish position array</param>
/// <param name="velocities">Fish velocities array</param>
/// <param name="maxSpeed">Fish max speed</param>
/// <param name="maxAcceleraion">Fish max acceleration in one iteration</param>
/// <param name="indexesInGrid">Fish grid cell indexes</param>
/// <param name="radious">Fish neighbor visibility distance</param>
/// <param name="gridSize">Number of cells in one direction</param>
/// <param name="cubeLength">Size of cude side</param>
/// <param name="startIndexesForGridCells">Array with start and end indexes for each grid cell</param>
/// <param name="fishesTypes">Fish types array</param>
/// <param name="outVelocities">Calculated velocities array</param>
/// <param name="outPositions">Calculated positions array</param>
/// <param name="openGLTriangles">Array of triangles for OpenGL to draw</param>
__global__ void computeNewVelocities(
	int n,
	float3* positions,
	float3* velocities,
	float maxSpeed,
	float maxAcceleraion,
	int* indexesInGrid,
	float radious,
	int gridSize,
	int cubeLength,
	vectorI* startIndexesForGridCells,
	unsigned char* fishesTypes,
	float3* outVelocities,
	float3* outPositions,
	triangle* openGLTriangles
)
{
	int i = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
	if (i >= n) return;

	int gridIndex = indexesInGrid[i];
	int gridCellZ = gridIndex / (gridSize * gridSize);
	int gridCellY = (gridIndex - (gridSize * gridSize * gridCellZ)) / gridSize;
	int gridCellX = gridIndex - (gridSize * gridSize * gridCellZ) - (gridSize * gridCellY);

	int neighborCount = 0;
	float3 mid = { 0,0,0 }; // œrodek ciê¿koœci 
	float3 averageVelocity = { 0,0,0 };// œrednia prêdkoœæ
	unsigned char fishType = fishesTypes[i];

	float3 velFix = { 0,0,0 }; // prêdkoœæ poprawiaj¹ca pozycjê, jeœli wyst¹pi³a kolizja miêdzy rybami 
	bool isVelFix = false; // czy ryby skolidowa³y
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int z = -1; z <= 1; z++)
			{
				// nighbor cell in grid
				int neighborX = gridCellX + x;
				int neighborY = gridCellY + y;
				int neighborZ = gridCellZ + z;

				// w przypadkach brzegowych przesuwamy s¹siadów z drugiego koñca
				float dx = 0;
				float dy = 0;
				float dz = 0;
				if (neighborX < 0) {
					neighborX = gridSize - 1;
					dx = -(float)cubeLength;
				}
				if (neighborX >= gridSize) {
					neighborX = 0;
					dx = (float)cubeLength;
				}
				if (neighborY < 0) {
					neighborY = gridSize - 1;
					dy = -(float)cubeLength;
				}
				if (neighborY >= gridSize) {
					neighborY = 0;
					dy = (float)cubeLength;
				}
				if (neighborZ < 0) {
					neighborZ = gridSize - 1;
					dz = -(float)cubeLength;
				}
				if (neighborZ >= gridSize) {
					neighborZ = 0;
					dz = (float)cubeLength;
				}

				int neighborIndex = neighborZ * gridSize * gridSize + neighborY * gridSize + neighborX;

				int startIndex = startIndexesForGridCells[neighborIndex].x;
				int endIndex = startIndexesForGridCells[neighborIndex].y;

				if (startIndex < 0) continue; // w komurce nie ma ¿adnych ryb

				for (int index = startIndex; index <= endIndex; index++)
				{
					if (index != i)
					{
						float3 relativePosition = positions[index] + make_float3(dx, dy, dz); //przesuwamy rybê jeœli wychodzi za krawêdz 
						float distanceSquared = DistanceSquared(positions[i], relativePosition);
						if (distanceSquared <= radious * radious && CalculateFishesAngle(positions[i], velocities[i], relativePosition) < FOV) //sprawdzamy czy ryba mo¿e widzieæ s¹siada 
						{
							if (distanceSquared < FISHSIZE / 2 * FISHSIZE / 2) //wyst¹pi³a kolizja 
							{
								float3 fix = relativePosition - positions[i];
								fix = ChangeLength(fix, FISHSIZE / 3 * FISHSIZE / 3 - distanceSquared);
								velFix = fix + velFix;
								isVelFix = true;
							}
							neighborCount++;
							mid = AddVectors(mid, relativePosition);
							averageVelocity = AddVectors(averageVelocity, velocities[index]);

						}
					}
				}
			}
		}
	}


	float3 newVel; //nowa obliczana prêdkoœæ
	float3 oldVel = velocities[i]; //stara prêdkœæ

	if (neighborCount > 0)
	{
		if (isVelFix) //jeœli wyst¹pi³a kolizja, ignorujemy resztê 
		{
			newVel = velFix;
		}
		else
		{
			mid = mid / (float)neighborCount;
			averageVelocity = averageVelocity / (float)neighborCount;
			// obliczanie kierunku w jakim chce p³yn¹æ ryba
			if (fishType == sociable)
			{
				newVel = SubtractVectors(mid, positions[i]);
			}
			else if (fishType == asocial)
			{
				newVel = SubtractVectors(positions[i], mid);
			}
			else if (fishType == crowdFollowing) {
				newVel = averageVelocity;
			}

			// uwzglêdniamy maksymalne przyspieszenie   
			if (VectorLengthSquared(newVel) > maxAcceleraion * maxAcceleraion) {
				newVel = ChangeLength(newVel, maxAcceleraion);
			}
		}

		//zmiana perspektywy do osi OX
		float3 RPYold = CalculateRPY(oldVel);
		RotateVector(-RPYold.x, -RPYold.y, -RPYold.z, &newVel);
		float3 RPYnew = CalculateRPY(newVel);

		// uwzglêdniamy maksymalny obrót jaki mo¿e wykonaæ ryba
		float angle = Angle(newVel, { 1,0,0 });
		if (angle > MAXTURN)
		{
			RPYnew = RPYnew / (angle / MAXTURN);
		}

		//zmiana perspektywy do pierwotnej
		RotateVector(RPYnew.x, RPYnew.y, RPYnew.z, &oldVel);

		newVel = oldVel;
	}
	else // jeœli ryba nie ma s¹siadów to p³ynie dalej w tym samym kierunku 
	{
		newVel = velocities[i];
	}


	outVelocities[i] = newVel;

	outPositions[i] = AddVectors(positions[i], outVelocities[i]);

	// przeniesienie ryby, jeœli wyp³ynê³a poza kostkê 
	while (outPositions[i].x < 0) outPositions[i].x += cubeLength;
	while (outPositions[i].y < 0) outPositions[i].y += cubeLength;
	while (outPositions[i].z < 0) outPositions[i].z += cubeLength;
	while (outPositions[i].x >= cubeLength) outPositions[i].x -= cubeLength;
	while (outPositions[i].y >= cubeLength) outPositions[i].y -= cubeLength;
	while (outPositions[i].z >= cubeLength) outPositions[i].z -= cubeLength;




	// obliczanie trójk¹tów dla OpenGL
	triangle t;
	float3 pos = outPositions[i];
	color color;
	if (fishType == sociable) {
		color = sociableColor;
	}
	else if (fishType == asocial) {
		color = asocialColor;
	}
	else {
		color = crowdFollowingColor;
	}

	t.c = { pos.x + FISHSIZE * 2 / 3, pos.y, pos.z  , 1.0f, frontColor };
	t.a = { pos.x - FISHSIZE / 3 , pos.y, pos.z - FISHSIZE / 4 , 1.0f,color };
	t.b = { pos.x - FISHSIZE / 3 , pos.y, pos.z + FISHSIZE / 4  , 1.0f,color };

	float3 RPY = CalculateRPY(outVelocities[i]);
	RotateTriangle(RPY.x, RPY.y, RPY.z, &t, pos);

	openGLTriangles[i].a = ConvertToOGL(t.a, cubeLength);
	openGLTriangles[i].b = ConvertToOGL(t.b, cubeLength);
	openGLTriangles[i].c = ConvertToOGL(t.c, cubeLength);
}


/// <summary>
/// Function to run in every interation
/// </summary>
void gpuAnimationIteration()
{
	dim3 threadsPerBlock(16, 16, 2);
	int threadsInOneBlock = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;
	int blockCount = (int)ceil(float(gpuData.n) / float(threadsInOneBlock));
	dim3 numBlocks(blockCount);

	setIndexesInGrid << <numBlocks, threadsPerBlock >> > (gpuData.n, gpuData.dev_positions, gpuData.radious, gpuData.gridSize, gpuData.dev_indexesInGrid, gpuData.dev_indexes);

	cudaDeviceSynchronize();

	thrust::sort_by_key(t_iig, t_iig + gpuData.n, t_i);

	cudaMemset(gpuData.dev_startIndexesForGridCells, -1, gpuData.gridSize * gpuData.gridSize * gpuData.gridSize * sizeof(vectorI));

	sortValuesInGrid << <numBlocks, threadsPerBlock >> > (
		gpuData.n,
		gpuData.dev_positions,
		gpuData.dev_velocitiesToRead,
		gpuData.dev_fishesTypes,
		gpuData.dev_indexes,
		gpuData.dev_sortedPositions,
		gpuData.dev_sortedVelocities,
		gpuData.dev_sortedNewFishesType,
		gpuData.dev_indexesInGrid,
		gpuData.dev_startIndexesForGridCells);

	cudaDeviceSynchronize();

	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&gpuData.openGLTriangles, &num_bytes, cuda_vbo_resource));

	computeNewVelocities << <numBlocks, threadsPerBlock >> > (
		gpuData.n,
		gpuData.dev_sortedPositions,
		gpuData.dev_sortedVelocities,
		gpuData.maxSpeed,
		gpuData.maxAcceleration,
		gpuData.dev_indexesInGrid,
		gpuData.radious,
		gpuData.gridSize,
		gpuData.length,
		gpuData.dev_startIndexesForGridCells,
		gpuData.dev_sortedNewFishesType,
		gpuData.dev_velocitiesToWrite,
		gpuData.dev_positions,
		gpuData.openGLTriangles
		);

	cudaDeviceSynchronize();

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 20, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 20, (GLvoid*)16);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawArrays(GL_TRIANGLES, 0, 3 * gpuData.n);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	drawEdges();

	glutSwapBuffers();

	swap(gpuData.dev_velocitiesToRead, gpuData.dev_velocitiesToWrite);
	swap(gpuData.dev_fishesTypes, gpuData.dev_sortedNewFishesType);

	
	computeFPS();

}


int main(int argc, char** argv)
{

#pragma region INIT
	config = configure();

	cudaError cudaStatus;
	cudaStatus = setupCuda();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "setup failed!");
		return 1;
	}

	if (config.maxFPS <= 0)
	{
		REFRESH_DELAY = 0;
	}
	else
	{
		REFRESH_DELAY = 1000 / config.maxFPS;
	}

	float3* startPositions = new float3[config.fishesCount];
	float3* startVelocities = new float3[config.fishesCount];
	unsigned char* startFishesType = new unsigned char[config.fishesCount];

	radnomInitialize(
		config.fishesCount,
		config.sociableProportion,
		config.asocialProportion,
		config.crowdFollowingProportion,
		config.maxSpeed,
		config.cubeSideLength,
		startPositions,
		startVelocities,
		startFishesType);

	Allocate(&gpuData, config, startPositions, startVelocities, startFishesType);

	t_iig = thrust::device_ptr<int>(gpuData.dev_indexesInGrid);
	t_i = thrust::device_ptr<int>(gpuData.dev_indexes);
#pragma endregion

	initGL(&argc, argv);
	glutDisplayFunc(gpuAnimationIteration);

	createVBO(config.fishesCount, &vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);



	glutMainLoop();

	sdkDeleteTimer(&timer);
	return Deallocate(gpuData);
}

/// <summary>
/// Redrows window for each timer tick
/// </summary>
void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}


/// <summary>
/// Initializes OpenGL window and functions
/// </summary>
/// <returns>Success</returns>
bool initGL(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(config.windowWidth, config.windowHeight);
	glutCreateWindow("Karasie w pelnej krasie");

	sdkCreateTimer(&timer);
	glutDisplayFunc(gpuAnimationIteration);
	glutMotionFunc(motion);
	glutMouseFunc(mouse);
	glutReshapeFunc(reshape);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	glClearColor(43.0f / 255, 173.0f / 255, 169.0f / 255, 1.0);
	glDisable(GL_DEPTH_TEST);


	SDK_CHECK_ERROR_GL();

	return true;
}




/// <summary>
/// Mouse event handler
/// </summary>
/// <param name="button">Button clicked</param>
/// <param name="state">Mouse state</param>
/// <param name="x">Mouse X position</param>
/// <param name="y">Mouse Y position</param>
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

/// <summary>
/// Rotates the view
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

/// <summary>
/// Cumputes FPS
/// </summary>
void computeFPS()
{
	sdkStopTimer(&timer);
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1000.0f / sdkGetAverageTimerValue(&timer);
		sprintf(fps, "FPS: %3.f fps ", ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;
		fpsLimit = (int)MAX(ifps, 1.0f);
		sdkResetTimer(&timer);
	}
	sdkStartTimer(&timer);
}


/// <summary>
/// Reshapes the view when the window is resized
/// </summary>
/// <param name="width">Window width</param>
/// <param name="height">Window height</param>
void reshape(GLsizei width, GLsizei height) {

	if (height == 0) height = 1;
	GLfloat aspect = (GLfloat)width / (GLfloat)height;
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, aspect, 0.1f, gpuData.length);
}