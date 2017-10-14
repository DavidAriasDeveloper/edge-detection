#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>

//Declaramos la funcion que invoca al kernel
void rgba_to_gray(uchar4 * const d_rgbaImage,
                  unsigned char* const d_grayImage,
                  size_t rows,
                  size_t cols);

//Se incluyen las definiciones del fichero
#include "preprocess.cpp"

int main(int argc,char** argv){
  uchar4 *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_grayImage, *d_grayImage;
  string input_file;
  string output_file;

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

  //Cargamos la imagen y preparamos los punteros de entrada y salida
  preProcess( &h_rgbaImage,
              &h_grayImage,
              &d_rgbaImage,
              &d_grayImage,
              input_file);

  //Invocamos al kernel

  rgba_to_gray( d_rgbaImage,
                d_grayImage,
                imageRGBA.rows,
                imageRGBA.cols);

  size_t numPixels = imageRGBA.rows * imageRGBA.cols;

  checkCudaErrors(
    cudaMemcpy(h_grayImage,d_grayImage,sizeof(unsigned char)*numPixels,cudaMemcpyDeviceToHost)
  );

  //Imagen de salida
  Mat output( imageRGBA.rows,
              imageRGBA.cols,
              CV_8UC1,
              (void*)h_grayImage);

  namedWindow("Display Window", WINDOW_AUTOSIZE);//Creamos una ventana para mostrar la imagen

  //Mostramos los resultados obtenidos
  imshow("Display Window",output);//Mostramos la imagen
  cvWaitKey(0);
  cvDestroyWindow("Display Window");

  imwrite(output_file.c_str(),output);

  //Liberamos memoria
  cudaFree(d_rgbaImage__);
  cudaFree(d_grayImage__);

  return 0;
}
