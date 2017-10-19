#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>

#define CONV_KERNEL_SIZE 9

//Declaramos las funciones de kernel
void rgba_to_gray(uchar4 * const d_rgbaImage,
                  unsigned char* const d_grayImage,
                  size_t rows,
                  size_t cols);

void sobel_filter(unsigned char* const d_inputImage,
                  unsigned char* const d_outputImage,
                  unsigned int maskWidth,
                  size_t rows,
                  size_t cols);

void cpyConvolutionKernel(char **h_convolutionKernel);
void freeConvolutionKernel();

//Se incluyen las definiciones del fichero
#include "preprocess.cpp"

//Declaramos las funciones que cargan el kernel de convolucion
void loadConvolutionKernel(int option,char *conv_kernel){
  switch(option){
      case 3://3X3
      conv_kernel[0] = -1;
      conv_kernel[1] = 0;
      conv_kernel[2] = 1;
      conv_kernel[3] = -2;
      conv_kernel[4] = 0;
      conv_kernel[5] = 2;
      conv_kernel[6] = -1;
      conv_kernel[7] = 0;
      conv_kernel[8] = 1;
      break;
    case 5://5x5
      conv_kernel[0] = 2;
      conv_kernel[1] = 1;
      conv_kernel[2] = 0;
      conv_kernel[3] = -1;
      conv_kernel[4] = -2;

      conv_kernel[5] = 3;
      conv_kernel[6] = 2;
      conv_kernel[7] = 0;
      conv_kernel[8] = -2;
      conv_kernel[9] = -3;

      conv_kernel[10] = 4;
      conv_kernel[11] = 3;
      conv_kernel[12] = 0;
      conv_kernel[13] = -3;
      conv_kernel[14] = -4;

      conv_kernel[15] = 3;
      conv_kernel[16] = 2;
      conv_kernel[17] = 0;
      conv_kernel[18] = -2;
      conv_kernel[19] = -3;

      conv_kernel[20] = 2;
      conv_kernel[21] = 1;
      conv_kernel[22] = 0;
      conv_kernel[23] = -1;
      conv_kernel[24] = -2;
      break;
    default:
      cerr << "Kernel de convolucion indefinido" <<endl;
      return;
  }
  return;
}

int main(int argc,char** argv){
  //Nombres de ficheros
  string input_file;
  string output_file;

  //Variables de tiempo
  clock_t startGPU, endGPU;
  double gpu_time_used;

  //Imagenes
  uchar4 *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_grayImage, *d_grayImage;
  unsigned char *h_edgeImage, *d_edgeImage;

  //Matriz de convolucion
  char* h_convolutionKernel = (char*)malloc(CONV_KERNEL_SIZE*sizeof(char));

  switch (argc) {
    case 1:
      cerr << "No se especifico ningun nombre de fichero" <<endl;
      exit(1);
      break;
    case 2:
      input_file = string(argv[1]);
      output_file = "output.png";
      break;
    default:
      cerr << "Demasiados parametros" <<endl;
      exit(1);
  }

  loadImage(input_file);

  loadConvolutionKernel(sqrt(CONV_KERNEL_SIZE),h_convolutionKernel);

  startGPU = clock();//Iniciamos el cronometro

  //Cargamos la imagen y preparamos los punteros de entrada y salida
  grayscale_preProcess( &h_rgbaImage,
              &h_grayImage,
              &d_rgbaImage,
              &d_grayImage,
              &h_edgeImage);

  size_t numPixels = imageRGBA.rows * imageRGBA.cols;

  //Invocamos al kernel
  rgba_to_gray( d_rgbaImage,
                d_grayImage,
                imageRGBA.rows,
                imageRGBA.cols);


  checkCudaErrors(
    cudaMemcpy(h_grayImage,d_grayImage,sizeof(unsigned char)*numPixels,cudaMemcpyDeviceToHost)
  );

  //Liberamos memoria
  cudaFree(d_rgbaImage__);

  //Aqui va la convolucion
  sobel_preProcess( &h_edgeImage,
                    &d_edgeImage,
                    &h_convolutionKernel);

  sobel_filter(     d_grayImage,
                    d_edgeImage,
                    sqrt(CONV_KERNEL_SIZE),
                    imageRGBA.rows,
                    imageRGBA.cols);

  checkCudaErrors(
    cudaMemcpy(h_edgeImage,d_edgeImage,sizeof(unsigned char)*numPixels,cudaMemcpyDeviceToHost)
  );

  endGPU = clock();//Finalizamos el cronometro

  gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
  printf("Tiempo Algoritmo Paralelo (constante): %.10f\n",gpu_time_used);

  //Imagen de salida
  Mat output( imageRGBA.rows,
              imageRGBA.cols,
              CV_8UC1,
              (void*)h_edgeImage);

  //namedWindow("Display Window", WINDOW_AUTOSIZE);//Creamos una ventana para mostrar la imagen

  //Mostramos los resultados obtenidos
  //imshow("Display Window",output);//Mostramos la imagen
  //cvWaitKey(0);
  //cvDestroyWindow("Display Window");

  imwrite(output_file.c_str(),output);

  //Liberamos memoria
  freeConvolutionKernel();
  cudaFree(d_grayImage__);
  cudaFree(d_edgeImage__);

  return 0;
}
