#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

using namespace cv;
using namespace std;

Mat imageRGBA;
Mat imageGray;
Mat imageEdge;

uchar4 *d_rgbaImage__;
unsigned char *d_grayImage__;
unsigned char *d_edgeImage__;

//Retorna un puntero de la version RGBA de la imagen de entrada y un puntero a la imagen de salida para host y gpu

void grayscale_preProcess(uchar4 **inputImage,unsigned char **grayImage,
                uchar4 **d_rgbaImage,unsigned char **d_grayImage,
                const string &filename){

  //Comprueba que el contexto se inicializa bien
  checkCudaErrors(cudaFree(0));

  Mat image;
  image = imread(filename.c_str(),CV_LOAD_IMAGE_COLOR);//Leemos la imagen

  if(!image.data){//Verificamos que se haya cargado la imagen correctamente
    cerr << "No se puede abrir la imagen" << endl;
    exit(1);
  }

  cvtColor(image,imageRGBA,CV_BGR2RGBA);

  //Reservamos memoria para output
  imageGray.create(image.rows,image.cols,CV_8UC1);

  //Establecemos punteros (ptr limpia el objeto una vez destruidas todas las instancias)
  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *grayImage = imageGray.ptr<unsigned char>(0);

  const size_t numPixels = imageRGBA.rows * imageRGBA.cols;

  //Reservamos memoria en el dispositivo
  checkCudaErrors(
    cudaMalloc(d_rgbaImage,sizeof(uchar4)*numPixels)
  );
  checkCudaErrors(
    cudaMalloc(d_grayImage,sizeof(uchar4)*numPixels)
  );

  cudaMemset(*d_grayImage,0,numPixels*sizeof(unsigned char));

  //Copiamos el input en la gpu
  checkCudaErrors(
    cudaMemcpy(*d_rgbaImage,
              *inputImage,
              sizeof(uchar4)*numPixels,
              cudaMemcpyHostToDevice)
  );

  d_rgbaImage__ = *d_rgbaImage;
  d_grayImage__ = *d_grayImage;
}

void sobel_preProcess(unsigned char **edgeImage, unsigned char **d_edgeImage,
                  char **h_convolutionKernel, char **d_convolutionKernel){
  const size_t numPixels = imageRGBA.rows * imageRGBA.cols;
  imageEdge.create(imageRGBA.rows,imageRGBA.cols,CV_8UC1);

  *edgeImage = imageEdge.ptr<unsigned char>(0);

  checkCudaErrors(
    cudaMalloc(d_edgeImage,sizeof(uchar4)*numPixels)
  );
  checkCudaErrors(
    cudaMalloc(d_convolutionKernel,sizeof(char)*CONV_KERNEL_SIZE)
  );

  checkCudaErrors(
    cudaMemcpy(*d_convolutionKernel,
              *h_convolutionKernel,
              sizeof(char)*CONV_KERNEL_SIZE,
              cudaMemcpyHostToDevice)
  );

  cudaMemset(*d_edgeImage,0,numPixels*sizeof(unsigned char));

  d_edgeImage__ = *d_edgeImage;
}
