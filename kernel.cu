#include "kernel.h"
#include "stdio.h"
#define TX 32
#define TY 32


#define DECIMAL double

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

template <typename T>
__global__
void calcMandel(size_t xPixel, size_t yPixel, size_t maxIter, T CyMin, T CxMin, T CyMax, T CxMax, T* dataCUDA) {


    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= xPixel) || (y >= yPixel)) return; // Check if within image bounds
    const int i = x + y * xPixel; // 1D indexing



    T PixelHeight = fabs(CyMax - CyMin) / (T)yPixel;
    T PixelWidth = fabs(CxMax - CxMin) / (T)xPixel;

    //for(size_t y = 0; y<yPixel;y++){
    T Cy = CyMin + y * PixelHeight;
    //for(size_t x = 0; x<xPixel;x++){
    T Cx = CxMin + x * PixelWidth;
    T ix = 0;
    T iy = 0;
    int iter =0;
    for (int k = 0; k < maxIter; k++) {
        T ixtemp = ix;
        // iteration: Z_{n+1} = Z_n^2 + C where C = x + yi
        // => Z = a^2 - b^2 + 2abi + x + yi = a^2 - b^2 + x + (2ab + y)i
        ix = ix * ix - iy * iy + Cx;
        iy = 2 * ixtemp * iy + Cy;
        if(ix*ix+iy*iy>40 && iter == 0) {
            iter = k;
        }
    }

    dataCUDA[i*3+0] = ix;
    dataCUDA[i*3+1] = iy;
    dataCUDA[i*3+2] = iter;


}

template <typename T>
__global__
void imageKernel(size_t w, size_t h, T* dataCUDA, uchar4* img, bool itermode) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= w) || (y >= h)) return; // Check if within image bounds
    const int i = x + y * w; // 1D indexing

    int r = MIN(255,MAX(0, (-dataCUDA[i*3+0] * 255)));
    int g = MIN(255,MAX(0, (-dataCUDA[i*3+1] * 255)));
    int iter =MIN(255,MAX(0, (dataCUDA[i*3+2]*1)));
    if(itermode == true){
        iter =MIN(255,MAX(0, (log(dataCUDA[i*3+2])*20)));
    }

    //MIN(255,MAX(0, (dataCUDA[i*3+2]*1)));// MIN(255,MAX(0, (log(dataCUDA[i*3+2])*20)));
    if(dataCUDA[i*3+0]!=0||dataCUDA[i*3+1]!=0){
        //iter = 0;
    }
    img[i].x = r;//r;
    img[i].y = g;
    img[i].z = iter;
    img[i].w = 255;


}

DECIMAL* dataCUDA;


//double timer = 100;
double2 midPoint = { -0.5, 0 };
int2 oldPos = { 0,0 };
void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos, int scroll,bool itermode) {
    double2 calcWindow = {1.0,h*1.0/w};
    const dim3 blockSize(TX, TY, 1);
    const dim3 gridSize = dim3((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, 1); // + TX - 1 for w size that is not divisible by TX
    double zoom = pow(1.2,scroll);
    double2 curWindow = {calcWindow.x/zoom,calcWindow.y/zoom};
    const double stepsize = 0.01;
    int2 curPos = {pos.x - oldPos.x,pos.y-oldPos.y};
    midPoint = { midPoint.x+ (double)curPos.x*stepsize*curWindow.x,midPoint.y+ (double)curPos.y*stepsize*curWindow.y};
    oldPos = {pos.x,pos.y};
    calcMandel<DECIMAL><<<gridSize, blockSize >>>(w, h, 500, midPoint.y - curWindow.y , midPoint.x - curWindow.x, midPoint.y+ curWindow.y, midPoint.x + curWindow.x,dataCUDA);
    gpuErrchk( cudaPeekAtLastError() );
    imageKernel<DECIMAL><<<gridSize, blockSize >>>(w, h,dataCUDA, d_out,itermode);
    gpuErrchk( cudaPeekAtLastError() );
}

void init(int w, int h){
    cudaMalloc((void**)&dataCUDA,3*w*h*sizeof(*dataCUDA));
}

void destroy(){
    cudaFree(dataCUDA);
}