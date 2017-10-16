#include "utils.h"
#include "stdio.h"
#include "math.h"

#define TILE_SIZE 32
#define MASK_WIDTH 5

//Maximo de hilos por bloque en GTX 970
#define TxB 1024

__constant__ char M[MASK_WIDTH*MASK_WIDTH];

__device__ unsigned char clamp(int value){
  if(value < 0) value = 0;
  if(value > 255) value = 255;
  return (unsigned char) value;
}

__global__ void sobel_filter_kernel(unsigned char* const inputImage,
                              unsigned char* const outputImage,
                              unsigned int maskWidth,
                              int rows,
                              int cols){
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  int Pvalue = 0;

  int N_start_point_row = row - (maskWidth/2);
  int N_start_point_col = col - (maskWidth/2);

  for(int i=0; i<maskWidth; i++){//Filas
    for(int j=0; j<maskWidth; j++){//Columnas
      if((N_start_point_col + j >=0 && N_start_point_col + j < cols) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < rows)){
        //Pvalue = 158;
        Pvalue += inputImage[(N_start_point_row+i)*cols + (N_start_point_col+j)]* M[i*maskWidth+j];
      }
    }
  }

  outputImage[row*cols+col] = clamp(Pvalue);

  //outputImage[row*cols+col] = inputImage[row*cols+col];
}

void sobel_filter(unsigned char* const d_inputImage,
                  unsigned char* const d_outputImage,
                  unsigned int maskWidth,
                  size_t rows,
                  size_t cols){
    int blockSize = TILE_SIZE;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(cols/float(blockSize)),ceil(rows/float(blockSize)),1);
    sobel_filter_kernel<<<dimGrid,dimBlock>>>(d_inputImage,d_outputImage,maskWidth,rows,cols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void cpyConvolutionKernel(char **h_convolutionKernel){
  checkCudaErrors(
    cudaMemcpyToSymbol(M,
              *h_convolutionKernel,
              sizeof(char)*MASK_WIDTH*MASK_WIDTH)
  );
}

void freeConvolutionKernel(){
  cudaFree(M);
}
