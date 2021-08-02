#include "kernel.h"
#include "stdio.h"
#define TX 32
#define TY 32



__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__global__
void distanceKernel(int* distanceData, int w, int h, int2 pos) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= w) || (y >= h)) return; // Check if within image bounds
    const int i = x + y * w; // 1D indexing
    const int dist = sqrtf((x - pos.x) * (x - pos.x) +
                           (y - pos.y) * (y - pos.y));
    distanceData[i]=dist;
}

__global__
void imageKernel(uchar4 *d_out,const int* distanceData, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= w) || (y >= h)) return; // Check if within image bounds
    const int i = x + y * w; // 1D indexing
    const int dist = distanceData[i];
    const unsigned char intensity = clip(255 - dist);
    d_out[i].x = intensity;
    d_out[i].y = intensity;
    d_out[i].z = 0;
    d_out[i].w = 255;
}
int* distanceCUDA;

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos) {
    // 3d version
    const dim3 blockSize(TX, TY, 1);
    const dim3 gridSize = dim3((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, 1); // + TX - 1 for w size that is not divisible by TX

    distanceKernel<<<gridSize, blockSize>>>(distanceCUDA, w, h, pos);
    gpuErrchk( cudaPeekAtLastError() );
    imageKernel<<<gridSize, blockSize>>>(d_out,distanceCUDA, w, h);
    gpuErrchk( cudaPeekAtLastError() );
}

void init(int w, int h){
    cudaMalloc((void**)&distanceCUDA,w*h*sizeof(*distanceCUDA));
}

void destroy(){
    cudaFree(distanceCUDA);
}