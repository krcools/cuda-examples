#include <fstream>
#include <iostream>
#include <stdlib.h>

#define show(x) std::cout << #x ": " << x << std::endl;
 
int main()
{
	const float T = 5.0f;
	const float X = 1.0f;
	const float c = 1.0f;

	float dt = 0.010;
	float dx = 0.025;

	int nt = (int)(T/dt) + 1;
	int nx = (int)(X/dx) + 1;

	dt = T / (nt-1);
	dx = X / (nx-1);
	float r = c * dt / dx;

	float *u = (float *)calloc(nx * nt, sizeof(float));

	// fill in the two first rows
	for (int j = 1; j < nx - 1; j++) {
		float x = j*dx;
		float y = x - 0.5;
		u[j + nx] = u[j] = exp(-40 * y*y);
	}

	for (int i = 1; i < nt-1; i++) {
		for (int j = 1; j < nx - 1; j++) {
			int id = j + nx * (i + 1);
			float uijp = u[j + 1 + nx * i];
			float uij0 = u[j + 0 + nx * i];
			float uijn = u[j - 1 + nx * i];
			float uinj = u[j + nx * (i - 1)];
			u[id] = r*r*uijp + 2 * (1 - r*r)*uij0 + r*r*uijn - uinj;
		}
	}

	std::ofstream ofs("output.txt");
	for (int i = 0; i < nt; i++) {
		for (int j = 0; j < nx; j++)
			ofs << u[i*nx + j] << "\t";
		ofs << std::endl;
	}
	ofs.close();

	free(u);

	return 0;
}
