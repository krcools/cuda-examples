#include <fstream>

const int width = 1920;
const int height = 1080;

struct cuComplex {
	float r;
	float i;
	__device__ cuComplex(float a, float b) : r(a), i(b) {}
	__device__ float magnitude2() { return r*r + i*i;  }
	__device__ cuComplex operator*(const cuComplex &a) { return cuComplex(r*a.r-i*a.i, r*a.i + i*a.r); }
	__device__ cuComplex operator+(const cuComplex &a) { return cuComplex(r+a.r, i+a.i); }
};

__device__ int julia(int x, int y) {
	const float scale = 1.5;
	float jx = scale * (float)(width  / 2 - x) / (width  / 2);
	float jy = scale * (float)(height / 2 - y) / (height / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

__global__ void kernel(unsigned char *ptr) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	int juliaValue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}



int main()
{
	const int bitmapSize = 4 * width * height;

	unsigned char *hst_bitmap = (unsigned char *)malloc(bitmapSize);
	unsigned char *dev_bitmap; cudaMalloc((void**)&dev_bitmap, bitmapSize);
	dim3 grid(width, height);
	kernel << <grid, 1 >> > (dev_bitmap);
	cudaMemcpy(hst_bitmap, dev_bitmap, bitmapSize, cudaMemcpyDeviceToHost);

	auto ofs = std::ofstream("bm.txt");
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++)
			ofs << (int)hst_bitmap[4*(j*width + i)] << "\t";
		ofs << std::endl;
	}
	ofs.close();

	return 0;
}