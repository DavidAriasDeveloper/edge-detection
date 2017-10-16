#include <iostream>
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc,char** argv){
  string input_file;
  string output_file;

  //Variables de tiempo
  clock_t startCPU, endCPU;
  double cpu_time_used;

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

  Mat image;
  image = imread(input_file, 1);

  startCPU = clock();//Iniciamos el cronometro

  Mat gray_image, grad_x, output;
  cvtColor(image, gray_image, CV_BGR2GRAY);
  Sobel(gray_image,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
  convertScaleAbs(grad_x, output);

  endCPU = clock();//Finalizamos el cronometro

  cpu_time_used = ((double) (endCPU - startCPU)) / CLOCKS_PER_SEC;
  printf("Tiempo Algoritmo Secuencial: %.10f\n",cpu_time_used);

  //namedWindow("Display Window", WINDOW_AUTOSIZE);//Creamos una ventana para mostrar la imagen

  //Mostramos los resultados obtenidos
  //imshow("Display Window",output);//Mostramos la imagen
  //cvWaitKey(0);
  //cvDestroyWindow("Display Window");

  imwrite(output_file.c_str(),output);

  return 0;
}
