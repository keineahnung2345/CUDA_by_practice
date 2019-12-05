#include <stdio.h>
#include <assert.h>
#include "common/Error.h"
#include "common/Matrix.h"
#include "common/GpuTimer.h"
#include "common/CpuTimer.h"

#define N 1000
#define K 16

void compareResults(Matrix<int> h_a, Matrix<int> h_b){

	int i,j;

	for(i=0; i < h_a.width; i++){
		for(j = 0; j < h_a.height; j++){
			assert(h_a.elements[ j * h_a.width + i ]  == h_b.elements[ i * h_b.width + j ]  );
		}
	}
}


__global__ void transposedMatrixKernel_pitch(Matrix<int> d_a, Matrix<int> d_b){

	int i = 0;
	int j = 0;

	while( i < d_a.width){
		j = 0;
		while( j < d_a.height){
			d_b.setElement( i, j, d_a.getElement(j, i) );
			j++;
		}
		i++;
	}
}


__global__ void transposedMatrixKernel(Matrix<int> d_a, Matrix<int> d_b){

	int i = 0;
	int j = 0;

	while( i < d_a.width){
		j = 0;
		while( j < d_a.height){
			d_b.setElement( i, j, d_a.getElement(j, i) );
			j++;
		}
		i++;
	}
	   
}


void onDevice(Matrix<int> h_a, Matrix<int> h_b){

    // declare GPU data
	Matrix<int> d_a, d_b;
	d_a.width = h_a.width;
	d_a.height = h_a.height;

	d_b.width = h_b.width;
	d_b.height = h_b.height;	


	GpuTimer timer;
	size_t pitchA;
	size_t pitchB;


	const int ARRAY_BYTES = d_a.width * d_a.height * sizeof(int);


    // -*- [Pitch Allocation] -*-
    // allocate  memory on the GPU

        //https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c
        /*
        __host__​cudaError_t cudaMallocPitch ( void** devPtr, size_t* pitch, size_t width, size_t height )
        Allocates pitched memory on the device.
        Allocates at least width (in bytes) * height bytes of linear memory on the device and returns in *devPtr a pointer to the allocated memory. The function may pad the allocation to ensure that corresponding pointers in any given row will continue to meet the alignment requirements for coalescing as the address is updated from row to row. The pitch returned in *pitch by cudaMallocPitch() is the width in bytes of the allocation. The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array. Given the row and column of an array element of type T, the address is computed as:

‎    T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
        */
        //d_a.width * sizeof(int): the width needed in bytes
        //pitchA: the allocated width in bytes
	HANDLER_ERROR_ERR(cudaMallocPitch((void **)(&d_a.elements), &pitchA, d_a.width * sizeof(int), d_a.height));
	HANDLER_ERROR_ERR(cudaMallocPitch((void **)(&d_b.elements), &pitchB, d_b.width * sizeof(int), d_b.height));
	printf("pitch = %li\n", pitchA/sizeof(int) );

	//copy from host memory to device
        //http://yzhu84.blogspot.com/2012/05/cudamallocpitch-and-cudamemcpy2d.html
        //here should use cudaMemcpy2D() instead?
	HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));
	HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));

  	timer.Start();	

	transposedMatrixKernel_pitch<<<1,1>>>( d_a, d_b );
	HANDLER_ERROR_MSG("kernel panic!!!");
   	timer.Stop(); 
    printf( "Time Device pitch:  %f ms\n", timer.Elapsed() );
    // copy data back from the GPU to the CPU
	HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost));	
	
	compareResults(h_a, h_b);

    // free GPU memory
	HANDLER_ERROR_ERR(cudaFree(d_a.elements));
	HANDLER_ERROR_ERR(cudaFree(d_b.elements));


    // -*- [CUDA Malloc Allocation] -*-

    // allocate  memory on the GPU
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements,ARRAY_BYTES));
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements,ARRAY_BYTES));

    // copy data from CPU the GPU
	HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));
	HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));

  	timer.Start();
	transposedMatrixKernel<<<1,1>>>( d_a, d_b );
    HANDLER_ERROR_MSG("kernel panic!!!");
   	timer.Stop(); 
    printf( "Time Device threads and blocks:  %f ms\n", timer.Elapsed() );
    // copy data back from the GPU to the CPU
	HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost));	
	compareResults(h_a, h_b);

    // free GPU memory
	HANDLER_ERROR_ERR(cudaFree(d_a.elements));
	HANDLER_ERROR_ERR(cudaFree(d_b.elements));

}


void onHost(){


	Matrix<int> h_a, h_b;
	h_a.width = N;
	h_a.height = N;

	h_b.width = N;
	h_b.height = N;

	h_a.elements = (int*)malloc(h_a.width  * h_b.height  * sizeof(int));
	h_b.elements = (int*)malloc(h_b.width  * h_b.height  * sizeof(int));


	int i,j,k=0;

	for( i = 0; i < h_a.width; i++){
		for( j = 0; j < h_a.height; j++){
			h_a.elements[j*h_a.width + i] = k;	
			h_b.elements[j*h_b.width + i] = 0;	
			k++;			
		}
	}

    // call device configuration
	onDevice(h_a,h_b);


    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);

}


int main(){

	onHost();
}

/*
pitch = 1024
Time Device pitch:  128.752060 ms
Time Device threads and blocks:  128.749695 ms
-: successful execution :-
*/
//Ref to:
//https://devtalk.nvidia.com/default/topic/1010074/cuda-programming-and-performance/bad-performance-using-mallocpitch-and-memcpy2d/post/5153032/#5153032
//:pitched allocation is useful on early GPU, but are of less significance on modern GPUs
