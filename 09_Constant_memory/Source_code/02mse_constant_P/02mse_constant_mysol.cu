#include "Error.h"
#include "Vector.h"
#include "GpuTimer.h"
#include <assert.h>

#define N   16 //the area of the matrix
#define BLOCK_SIZE 2
#define STRIDE 4 //grid size
#define POW(x) (x) * (x)

// constant Vectors on the GPU
    // -:YOUR CODE HERE:-  
__constant__ float d_a[N];
__constant__ float d_b[N];

__global__ void mseKernel( Vector<float> d_c ){

    // -:YOUR CODE HERE:-   
    __shared__ float cache[BLOCK_SIZE * BLOCK_SIZE];

    //used to index a thread in a grid
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    //used to index an element in cache(it's shape is just like a flattened block)
    int cacheIndex = threadIdx.y * blockDim.x + threadIdx.x;

    //move grid through the whole STRIDE * STRIDE (=N)  matrix
    float temp = 0;
    for (int gx = tx; gx < STRIDE; gx += blockDim.x * gridDim.x){
        for (int gy = ty; gy < STRIDE; gy += blockDim.y * gridDim.y){
            temp += POW(d_a[gy * STRIDE + gx] - d_b[gy * STRIDE + gx]);
        }
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    //reduce all threads in one block into one
    int i =  BLOCK_SIZE * BLOCK_SIZE / 2; 
    while (i != 0){
        __syncthreads();
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex +i];
        }
        i /= 2;
    }

    //one element of d_c is corresponding to one block
    if (cacheIndex == 0){
        d_c.setElement(blockIdx.y * blockDim.x + blockIdx.x, cache[0]);
    }
}


void onDevice( Vector<float> h_a, Vector<float> h_b, Vector<float> h_c ){

    // declare GPU vectors
    // -:YOUR CODE HERE:-
    Vector<float> d_c;
    //block count
    d_c.length = (STRIDE / BLOCK_SIZE) * (STRIDE / BLOCK_SIZE);

    // start timer
    GpuTimer timer;
    timer.Start();

    const int ARRAY_BYTES = N * sizeof(float);
    const int P_ARRAY_BYTES = d_c.length * sizeof(float);

    // allocate  memory on the GPU
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMemcpyToSymbol(d_a, h_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMemcpyToSymbol(d_b, h_b.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements, P_ARRAY_BYTES));


    //execution configuration
    dim3 GridBlocks( 2,2 );
    dim3 ThreadsBlocks( 2,2 );

    mseKernel<<<GridBlocks,ThreadsBlocks>>>( d_c );
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMemcpy(h_c.elements, d_c.elements, P_ARRAY_BYTES, cudaMemcpyDeviceToHost));

    // stop timer
    timer.Stop(); 
    
    // print time
    printf( "Time :  %f ms\n", timer.Elapsed() );

    // free GPU memory
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaFree(d_c.elements));
}


void test(){

    Vector<float> h_a, h_b, h_c;
    h_a.length = N;
    h_b.length = N;
    h_c.length = 4; //a grid contains 4 blocks = (STRIDE/BLOCK_SIZE)*(STRIDE/BLOCK_SIZE)

    //both int and float takes 4 bytes
    h_a.elements = (float*)malloc(  h_a.length * sizeof(int) );
    h_b.elements = (float*)malloc(  h_a.length * sizeof(int) );
    h_c.elements = (float*)malloc(  4 * sizeof(int) );

    int i,j = 16, k=1;

    for( i = 0; i < h_a.length; i++){
            h_a.setElement( i, k );  
            h_b.setElement( i, j );                 
            k++;
            j--;            
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);

     // verify that the GPU did the work we requested
    float d_mse = 0;
    for (int i=0; i<4; i++) {
            printf( " [%i] = %f \n", i, h_c.getElement(i) );
            d_mse += h_c.getElement(i);
    }
    printf("MSE from device: %f\n", d_mse/(N*N));

    float h_mse = 0;
    for (int i=0; i<N; i++) {
            h_mse += POW(h_a.getElement(i) - h_b.getElement(i));
    }
    printf("MSE from host: %f\n", h_mse/(N*N));

    assert(d_mse == h_mse);

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements); 

}

int main( void ) {
    	
        test();
    	return 0;
}
