#include <curand.h>
#include <stdio.h>

__global__
void add(int *a, int *b, int *c)
{
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

#define N 512
int main(void)
{
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int size = N * sizeof(int);

  // Allocate space on the device
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Inputs
  a = (int *)malloc(size);
  b = (int *)malloc(size);
  c = (int *)malloc(size);

  for (int i = 0; i < N; i++)
  {
    *(a + i) = 2;
    *(b + i) = 4;
  }

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Perform computation on device
  add<<<N,1>>>(d_a, d_b, d_c);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++)
  {
    printf("result: %d\n", *c);
  }

  // Clean up
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}
