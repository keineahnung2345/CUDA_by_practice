#include "Error.h"
#include "Vector.h"
#include "GpuTimer.h"

#define N   16
#define BLOCK_SIZE 2
#define STRIDE 4
#define POW(x) (x) * (x)

__global__ void mseKernel( Vector<float> d_a, Vector<float> d_b, Vector<float> d_c ){

    // -:YOUR CODE HERE:-   
    __shared__ float cache[BLOCK_SIZE * BLOCK_SIZE];
    //tid ranges from 0 to N-1, used to index threads in the whole grid
    int tid = (threadIdx.y + blockIdx.y * blockDim.y) * STRIDE + threadIdx.x + blockIdx.x * blockDim.x;
    //used to index threads in the cache
    int cacheIndex = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    //tried 1
/*
    float temp = 0;
    while(tid < N){
        temp += POW(d_a.getElement(tid) - d_b.getElement(tid));
        tid += blockDim.x * blockDim.y;
    }

    cache[cacheIndex] = temp;
*/
    //tid ranges from 0 to N-1, so don't need to use while loop like above
    cache[cacheIndex] = POW(d_a.getElement(tid) - d_b.getElement(tid));

    //tried 1
    //int i = blockDim.x/2;
    //this should be half of number of threads in one block
    int i = blockDim.x * blockDim.y /2;
    while(i != 0){
       __syncthreads();
       if(cacheIndex < i){
           cache[cacheIndex] += cache[cacheIndex +i];
       } 
       i /= 2;
    }

    if(cacheIndex == 0){
        //d_c's size is blockDim.x * blockDim.y
        d_c.setElement(blockIdx.y * BLOCK_SIZE + blockIdx.x, cache[0]);
    }
}


void onDevice( Vector<float> h_a, Vector<float> h_b, Vector<float> h_c ){

    // declare GPU vectors
    // -:YOUR CODE HERE:-
    Vector<float> d_a, d_b, d_c;
    d_a.length = N;
    d_b.length = N;
    d_c.length = N/BLOCK_SIZE/BLOCK_SIZE;

    // start timer
    GpuTimer timer;
    timer.Start();

    const int ARRAY_BYTES = N * sizeof(float);
    const int P_ARRAY_BYTES = d_c.length * sizeof(float);

    // allocate  memory on the GPU
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements, P_ARRAY_BYTES));

    // copy data from CPU the GPU
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));

    //execution configuration
    dim3 GridBlocks( 2,2 );
    dim3 ThreadsBlocks( 2,2 );

    mseKernel<<<GridBlocks,ThreadsBlocks>>>( d_a, d_b, d_c );
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
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
    HANDLER_ERROR_ERR(cudaFree(d_c.elements));
}


void test(){

    Vector<float> h_a, h_b, h_c;
    h_a.length = N;
    h_b.length = N;
    //h_c.length = 4;    
    h_c.length = N / STRIDE;    

    h_a.elements = (float*)malloc(  h_a.length * sizeof(float) );
    h_b.elements = (float*)malloc(  h_a.length * sizeof(float) );
    h_c.elements = (float*)malloc(  h_c.length * sizeof(float) );


    int i,j = N, k=1;

    for( i = 0; i < h_a.length; i++){
            h_a.setElement( i, k );  
            h_b.setElement( i, j );    
            //h_c.setElement( i, 0.0 );                
            k++;
            j--;            
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);

     // verify that the GPU did the work we requested
    int d_se = 0, h_se = 0;
    for (int i=0; i<4; i++) {
            d_se += h_c.getElement(i);
            printf( " [%i] = %f \n", i, h_c.getElement(i) );
    }
    printf("%d\n", d_se);

    for (int i=0; i<N; i++) {
            h_se += POW(h_a.getElement(i) - h_b.getElement(i));
    }
    printf("%d\n", h_se);



    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements); 

}

int main( void ) {
    	
        test();
    	return 0;
}
