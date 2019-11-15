#include <assert.h>
#include "Error.h"
#include "Vector.h"
#include "GpuTimer.h"

const int BLOCKSIZE	= 128;
const int NUMBLOCKS = 1000;					
const int N 		= BLOCKSIZE*NUMBLOCKS;
const int ARRAY_BYTES = N * sizeof(int);


__global__ void tileKernelv2(Vector<int> d_in, Vector<int> d_out){
    
/*
    // Change next operation in order to use the tiling technique 
    int i = threadIdx.x + blockIdx.x*blockDim.x; 
    d_out.elements[i] = (d_in.getElement(i-2) + d_in.getElement(i-1) + d_in.getElement(i) + d_in.getElement(i+1) + d_in.getElement(i+2)) / 5.0f;
*/
    // -:YOUR CODE HERE:- 
    //used to index grid
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    //used to index a block
    int tix = threadIdx.x;
    __shared__ int s[5 * BLOCKSIZE];
    s[tix * 5 + 0] = d_in.getElement(i-2);
    s[tix * 5 + 1] = d_in.getElement(i-1);
    s[tix * 5 + 2] = d_in.getElement(i);
    s[tix * 5 + 3] = d_in.getElement(i+1);
    s[tix * 5 + 4] = d_in.getElement(i+2);

    __syncthreads();

    int tmp = 0;
    tmp = (s[tix * 5 + 0] + s[tix * 5 + 1] + s[tix * 5 + 2] + s[tix * 5 + 3] + s[tix * 5 + 4])/5.0f;
    d_out.elements[i] = tmp;
} 

void onDevice(Vector<int> h_in, Vector<int> h_out){

    Vector<int> d_in, d_out;
    d_in.length = N; 
    d_out.length = N;  


    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_in.elements,ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_out.elements,ARRAY_BYTES));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_in.elements, h_in.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));

    //launch kernel
    tileKernelv2<<<N/BLOCKSIZE, BLOCKSIZE>>>(d_in, d_out);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_out.elements, d_out.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost)); 

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_in.elements));
    HANDLER_ERROR_ERR(cudaFree(d_out.elements));        

}


void test(Vector<int> h_in,  Vector<int> h_out){


	int aux = 0;
	for (int i=2; i<N-2; i++)
	{
 		aux = (h_in.getElement(i-2) + h_in.getElement(i-1) + h_in.getElement(i) + h_in.getElement(i+1)  + h_in.getElement(i+2)) / 5.0f;
                //printf("%d %d\n", aux, h_out.getElement(i));
		assert(aux == h_out.getElement(i));

	}

}


void onHost(){


    Vector<int> h_in, h_out; 
    h_in.length = N; 
    h_out.length = N;                 

    h_in.elements = (int*)malloc( ARRAY_BYTES );
    h_out.elements = (int*)malloc( ARRAY_BYTES );  

	for (int i=0; i<N; i++) 
	{
		h_in.setElement(i,2*i); 
	}                  

    // call device configuration
    onDevice(h_in, h_out);

    //testing
    test(h_in, h_out);

    printf("-: successful execution :-\n");

    free(h_in.elements); 
    free(h_out.elements); 
           
} 


int main( void ) {
    	
        onHost();
    	return 0;
}
