#include <stdio.h>
 
#define N 10

__global__ void add(int *a, int *b, int *c)
{
	int tid = blockIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}
 
int main()
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// allocate memory on the device
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	// Initialise data in the host's memory
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i*i;
	}

	// copy over the data to the device
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	// launch the kernel
	add << <N, 1 >> > (dev_a, dev_b, dev_c);

	// copy the results back over to the host
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	return 0;
}