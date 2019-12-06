#include <stdio.h>
#include <omp.h>

//nvcc   -Xcompiler -fopenmp 06hello_openmp.cu   -o exe
//https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options-compiler-options
//-Xcompiler: passing specific options directly to compiler/preprocessor(gcc or g++ on Linux)
//-fopenmp: one of gcc's C Language Options

int main() {

    int i;
    printf("Hello World\n");
    #pragma omp parallel
    {
		#pragma omp for
		for(i=0;i<6;i++) {
            printf("Iter:%d\n",i);
		}
    }
    printf("GoodBye World\n");

}

/*
result:

Hello World
Iter:0
Iter:4
Iter:3
Iter:2
Iter:5
Iter:1
GoodBye World

note that the order will change every time it's executed
*/
