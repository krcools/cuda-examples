#include <stdio.h>
 

__global__ void add(int a, int b, int *c) 
{
	//a[threadIdx.x] += b[threadIdx.x];
	*c = a + b;
}
 
int main()
{
	int c;
	int *dev_c;
	cudaMalloc( (void**)&dev_c, sizeof(int) );
	add << <1, 1 >> > (2, 7, dev_c);
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("2+7 = %d\n", c);
	cudaFree(dev_c);

	int count;
	cudaGetDeviceCount(&count);
	printf("Number of devices: %d\n", count);

	cudaDeviceProp prop;
	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("Device %d:\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Global memory: %ld\n", prop.totalGlobalMem);
		printf("MP count: %d\n", prop.multiProcessorCount);
		printf("Threads in warp: %d", prop.warpSize);
	}
}