#pragma once
#include <helper_gl.h>
#include "vectors.h"
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

/// <summary>
/// Creates buffer for OpenGL arrays
/// </summary>
void createVBO(int n, GLuint* vbo, struct cudaGraphicsResource** vbo_res,
	unsigned int vbo_res_flags)
{
	assert(vbo);
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	unsigned int size = n * sizeof(triangle);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
	SDK_CHECK_ERROR_GL();
}

/// <summary>
/// Deletes buffer of OpenGL arrays
/// </summary>
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
{
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}