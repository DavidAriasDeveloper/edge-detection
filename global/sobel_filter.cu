#include "utils.h"
#include "stdio.h"
#include "math.h"

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
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  int Pvalue = 0;

  int N_start_point_row = row - (maskWidth/2);
  int N_start_point_col = col - (maskWidth/2);

  if(row == 120 && col == 120){
    printf("Pixel: %d,%d - Punto de inicio %d,%d\n", row,col, N_start_point_row, N_start_point_col);
  }

  for(int i=0; i<maskWidth; i++){//Filas
    for(int j=0; j<maskWidth; j++){//Columnas
      if((N_start_point_col + j >=0 && N_start_point_col + j < cols) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < rows)){
        //Pvalue = 158;
        if(row == 120 && col == 120){
          printf("Pixel %d,%d = %d\n", N_start_point_row+i, N_start_point_col+j, inputImage[(N_start_point_row+i)*cols + (N_start_point_col+j)]);
        }
        Pvalue += inputImage[(N_start_point_row+i)*cols + (N_start_point_col+j)]* M[i*maskWidth+j];
      }
    }
  }
  if(row == 120 && col == 120){
    printf("Valor de P %d\n", Pvalue);
  }

  outputImage[row*cols+col] = clamp(Pvalue);

  //outputImage[row*cols+col] = inputImage[row*cols+col];
}

void sobel_filter(unsigned char* const d_inputImage,
                  unsigned char* const d_outputImage,
                  unsigned int maskWidth,
                  char *M,
                  size_t rows,
                  size_t cols){
    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(cols/float(blockSize)),ceil(rows/float(blockSize)),1);
    sobel_filter_kernel<<<dimGrid,dimBlock>>>(d_inputImage,d_outputImage,maskWidth,M,rows,cols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
