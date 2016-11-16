#include <fstream>
#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <random>

#define show(x) std::cout << #x ": " << x << std::endl;

#define BLOCKSIZE 128
 
__global__ void pi(float *blockSums, int stepsPerThread, float dx) {

	__shared__ float threadSums[BLOCKSIZE];

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int istart = id * stepsPerThread;
	int istop = istart + stepsPerThread;

	float accum = 0.0f;
	for (int i = istart; i < istop; i++) {
		float x = (i + 0.5f) * dx;
		accum += 4.0f / (1.0f + x*x);
	}
	threadSums[threadIdx.x] = accum;

	__syncthreads();
	if (threadIdx.x == 0) {
		float blockSum = 0.0f;
		for (int j = 0; j < blockDim.x; j++) {
			blockSum += threadSums[j];
		}
		blockSums[blockIdx.x] = blockSum;
	}
}


int main()
{
	cudaError_t err;

	const int stepsPerThread = 512 * 2 * 2;
	const int blockSize = BLOCKSIZE;
	const int numBlocks = 256;

	const int numSteps = blockSize * numBlocks * stepsPerThread;
	const float dx = 1.0f / numSteps;

	float *h_blockSums = (float *)malloc(sizeof(float) * numBlocks);
	float *d_blockSums; err = cudaMalloc((void**)&d_blockSums, sizeof(float) * numBlocks); assert(err == cudaSuccess);

	err = cudaMemcpy(d_blockSums, h_blockSums, sizeof(float) * numBlocks, cudaMemcpyHostToDevice); assert(err == cudaSuccess);
	pi<<<numBlocks, blockSize>>> (d_blockSums, stepsPerThread, dx);
	err = cudaMemcpy(h_blockSums, d_blockSums, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost); assert(err == cudaSuccess);

	float pi = 0.0f;
	for (int i = 0; i < numBlocks; i++)
		pi += h_blockSums[i];
	pi *= dx;

	printf("pi approximately equals: %f\n", pi);

	cudaFree(d_blockSums);
	free(h_blockSums);

	return 0;
}