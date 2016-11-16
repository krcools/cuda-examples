#include <fstream>
#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <random>

#define show(x) std::cout << #x ": " << x << std::endl;
 
// M: the number of rows of A
// N: the number of cols of B
// K: the number of cols of A (= the number of rows in B)
__global__ void matmat(float *A, float *B, float *C, int M, int N, int K) {

	int m = threadIdx.x + blockDim.x * blockIdx.x;
	int n = threadIdx.y + blockDim.y * blockIdx.y;

	int mn = m + n * M;

	if (m < M && n < N) {
		float c = 0.0f;
		for (int k = 0; k < K; k++) {
			int mk = m + k * M;
			int kn = k + n * K;
			c += A[mk] * B[kn];
		}
		C[mn] = c;
	}
}

int main()
{
	cudaError_t err;

	const int M = 200;
	const int N = 200;
	const int K = 200;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0.0, 1.0);

	float *h_A = (float *)malloc(sizeof(float) * M * K);
	for (int i = 0; i < M*K; i++) h_A[i] = dis(gen);
	float *h_B = (float *)malloc(sizeof(float) * K * N);
	for (int i = 0; i < K*N; i++) h_B[i] = dis(gen);
	float *h_C = (float *)malloc(sizeof(float) * M * N);

	float *d_A; err = cudaMalloc((void**)&d_A, M * K * sizeof(float)); assert(err == cudaSuccess);
	float *d_B; err = cudaMalloc((void**)&d_B, K * N * sizeof(float)); assert(err == cudaSuccess);
	float *d_C; err = cudaMalloc((void**)&d_C, M * N * sizeof(float)); assert(err == cudaSuccess);

	err = cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice); assert(err == cudaSuccess);
	err = cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice); assert(err == cudaSuccess);

	dim3 numBlocks(M, N);
	dim3 threadsPerBlock(1, 1);
	matmat << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, M, N, K);
	
	err = cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost); assert(err == cudaSuccess);

	float totalError = 0.0f;
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			int mn = m + n * M;
			float c = 0.0f;
			for (int k = 0; k < K; k++) {
				int mk = m + k * M;
				int kn = k + n * K;
				c += h_A[mk] * h_B[kn];
			}
			float error = std::fabs(c - h_C[mn]);
			if (error > 1.0e-4) {
				printf("Error level of %f in entry [%d,%d]", error, m, n);
				printf("CPU result: %f\n", c);
				printf("GPU result: %f\n", h_C[mn]);
				return -1;
			}
			totalError += std::fabs(c - h_C[mn]);
		}
	}

	printf("Total error per element in caluclation: %f\n", totalError / (M*N));

	cudaFree(d_C);
	cudaFree(d_B);
	cudaFree(d_A);

	free(h_C);
	free(h_B);
	free(h_A);

	return 0;
}