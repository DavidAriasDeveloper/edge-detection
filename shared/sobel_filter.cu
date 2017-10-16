#include "utils.h"
#include "stdio.h"
#include "math.h"

#define TILE_SIZE 32
#define MASK_WIDTH 5

//Maximo de hilos por bloque en GTX 970
#define TxB 1024

__device__ unsigned char clamp(int value){
  if(value < 0) value = 0;
  if(value > 255) value = 255;
  return (unsigned char) value;
}

__global__ void sobel_filter_kernel(unsigned char* const inputImage,
                              unsigned char* const outputImage,
                              unsigned int maskWidth,
                              char *M,
                              int rows,
                              int cols){

  __shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE+ MASK_WIDTH - 1];

  int n = maskWidth/2;
  int dest = threadIdx.y*TILE_SIZE+threadIdx.x,
      destY=dest/(TILE_SIZE+maskWidth-1),
      destX = dest % (TILE_SIZE+maskWidth-1),
      srcY = blockIdx.y * TILE_SIZE + destY - n,
      srcX = blockIdx.x * TILE_SIZE + destX - n,
      src = (srcY * cols + srcX);

  if (srcY >= 0 && srcY < rows && srcX >= 0 && srcX < cols)
        N_ds[destY][destX] = inputImage[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + MASK_WIDTH - 1), destX = dest % (TILE_SIZE + MASK_WIDTH - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * cols + srcX);
    if (destY < TILE_SIZE + MASK_WIDTH - 1) {
        if (srcY >= 0 && srcY < rows && srcX >= 0 && srcX < cols)
            N_ds[destY][destX] = inputImage[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int accum = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * MASK_WIDTH + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < rows && x < cols)
        outputImage[(y * cols + x)] = clamp(accum);
    __syncthreads();
}

void sobel_filter(unsigned char* const d_inputImage,
                  unsigned char* const d_outputImage,
                  unsigned int maskWidth,
                  char *M,
                  size_t rows,
                  size_t cols){
    int blockSize = TILE_SIZE;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(cols/float(blockSize)),ceil(rows/float(blockSize)),1);
    sobel_filter_kernel<<<dimGrid,dimBlock>>>(d_inputImage,d_outputImage,maskWidth,M,rows,cols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
