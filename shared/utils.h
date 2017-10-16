#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#define checkCudaErrors(val) check((val),#val,__FILE__,__LINE__)

using namespace std;

template<typename T>
void check( T err,
            const char* const func,
            const char* const file,
            const int line){
  if(err != cudaSuccess){
    cerr << "CUDA error en: " << file << ":" << line << endl;
    cerr << cudaGetErrorString(err)<< " " << func << endl;
    exit(1);
  }
}
#endif
