#include "utils.h"
#include "stdio.h"
#include "math.h"

#define TILE_SIZE 32
#define MASK_WIDTH 3

//Maximo de hilos por bloque en GTX 970
#define TxB 1024

//Funcion para evitar errores en los limites
__device__ unsigned char clamp(int value){
  if(value < 0) value = 0;
  if(value > 255) value = 255;
  return (unsigned char) value;
}

__global__ void sobel_filter_kernel(unsigned char* const inputImage,
                              unsigned char* const outputImage,
                              char *M,
                              int rows,
                              int cols){
  //La variable compartida por bloque va a ser el trozo de la imagen a multiplicar
  __shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE+ MASK_WIDTH - 1];

  int n = MASK_WIDTH/2;//Mitad de la mascara para referencias
  int dest = threadIdx.y*TILE_SIZE+threadIdx.x;//Pixel unidimiensional de destino del trozo
  int destY=dest/(TILE_SIZE+MASK_WIDTH-1);//Coordenada y del pixel unidimiensional de destino del trozo
  int destX = dest % (TILE_SIZE+MASK_WIDTH-1);//Coordenada x del pixel unidimiensional de destino del trozo
  int srcY = blockIdx.y * TILE_SIZE + destY - n;//Coordenada del pixel unidimiensional de la imagen fuente (Iniciando desde la mascara)
  int srcX = blockIdx.x * TILE_SIZE + destX - n;//Coordenada del pixel unidimiensional de la imagen fuente (Iniciando desde la mascara)
  int src = (srcY * cols + srcX);//Coordenada del pixel unidimiensional de la imagen fuente

  /*** Carga de la variable compartida ***/
  if (srcY >= 0 && srcY < rows && srcX >= 0 && srcX < cols)//Si la Coordenada de la imagen se encuentra entre los limites
        N_ds[destY][destX] = inputImage[src];//Nuestro trozo de imagen se actualizara
    else
        N_ds[destY][destX] = 0;//Sino, se le da un valor de negro


    dest = threadIdx.y * TILE_SIZE + threadIdx.x + (TILE_SIZE * TILE_SIZE);
    destY = dest /(TILE_SIZE + MASK_WIDTH - 1);
    destX = dest % (TILE_SIZE + MASK_WIDTH - 1);
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

    /** Multiplicacion de trozo de imagen por kernel de convolucion **/
    int accum = 0;
    int y, x;
    for (y = 0; y < MASK_WIDTH; y++)
        for (x = 0; x < MASK_WIDTH; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * MASK_WIDTH + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < rows && x < cols)
        outputImage[(y * cols + x)] = clamp(accum);
    __syncthreads();
}

void sobel_filter(unsigned char* const d_inputImage,
                  unsigned char* const d_outputImage,
                  char *M,
                  size_t rows,
                  size_t cols){
    int blockSize = TILE_SIZE;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(cols/float(blockSize)),ceil(rows/float(blockSize)),1);
    sobel_filter_kernel<<<dimGrid,dimBlock>>>(d_inputImage,d_outputImage,M,rows,cols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
