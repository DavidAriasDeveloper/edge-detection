#include "utils.h"
#include "stdio.h"
#include "math.h"

//Maximo de hilos por bloque en GTX 970
#define TxB 1024

__global__ void rgba_to_gray_kernel(const uchar4* const rgbaImage,
                                    unsigned char* const grayImage,
                                    int rows,
                                    int cols){
    /*El mapeo de uchar4 a RGBA es:
      .x -> R
      .y -> G
      .z -> B
      .w -> A

    La salida debe ser resultado de aplicar la siguiente formula
    Pixel = .299f * R + .587f * G + .114f * B
    */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    uchar4 px = rgbaImage[i];//Pixel procesado por el hilo
    grayImage[i] =  .299f * px.x +
                    .587f * px.y +
                    .114f * px.z;
}

void rgba_to_gray(uchar4 * const d_rgbaImage,
                  unsigned char* const d_grayImage,
                  size_t rows,
                  size_t cols){
    long long int total_px = rows * cols;
    long int grids_n = ceil(total_px/TxB);
    const dim3 blockSize(TxB,1,1);
    const dim3 gridSize(grids_n,1,1);
    rgba_to_gray_kernel<<<gridSize,blockSize>>>(d_rgbaImage,d_grayImage,rows,cols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
