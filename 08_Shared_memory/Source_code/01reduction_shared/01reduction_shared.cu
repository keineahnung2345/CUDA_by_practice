#include "Error.h"
#include "Vector.h"
#include "GpuTimer.h"

#define N   (32 * 1024)
#define THREADS 256
#define BLOCKS 32

const int ARRAY_BYTES = N * sizeof(float);
const int P_ARRAY_BYTES = BLOCKS * sizeof(float);

__global__ void dotKernel( Vector<float> d_a, Vector<float> d_b, Vector<float> d_c ){
    //shared memory's size is the same as the block's size
    __shared__ float cache[THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    // a kernel function is executed is each block 
    //the block is responsible for range(threadIdx.x, N, blockDim.x * gridDim.x)
    float   temp = 0;
    while (tid < N) {
        temp += d_a.getElement(tid) * d_b.getElement(tid);
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    //__syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    // a kernel function is executed is each block, 
    // and here i is the half width of a block
    int i = blockDim.x/2;
    while (i != 0) {
        //without __syncthreads(), the program can still run, but the result will change
        __syncthreads();
        if (cacheIndex < i){
            //cacheIndex ranges from 0 to i-1
            //add cache[i:2*i] to cache[0:i]
            //we can image it as split the vector cache[:2*i] into half,
            //and add the later half to former half
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        i /= 2;
    }
    //after the loop, all the value the block has calculated has be summed to cache[0]

    if (cacheIndex == 0)
        d_c.setElement(blockIdx.x, cache[0]);

}


void onDevice( Vector<float> h_a, Vector<float> h_b, Vector<float> h_c ){

    Vector<float> d_a, d_b, d_c;
    d_a.length = N;
    d_b.length = N;
    d_c.length = BLOCKS; 

    // start timer
    GpuTimer timer;
    timer.Start();


    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements,ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements,ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements, P_ARRAY_BYTES));

    // copy data from CPU the GPU
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));


    dotKernel<<<BLOCKS,THREADS>>>( d_a, d_b, d_c );
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_c.elements, d_c.elements, P_ARRAY_BYTES, cudaMemcpyDeviceToHost)); 

    // stop timer
    timer.Stop(); 
    
    // print time
    printf( "Time :  %f ms\n", timer.Elapsed() );

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
    HANDLER_ERROR_ERR(cudaFree(d_c.elements));
}


void onHost(){

    Vector<float> h_a, h_b, h_c;
    h_a.length = N;
    h_b.length = N;
    h_c.length = BLOCKS;    

    h_a.elements = (float*)malloc(  ARRAY_BYTES );
    h_b.elements = (float*)malloc(  ARRAY_BYTES );
    h_c.elements = (float*)malloc(  P_ARRAY_BYTES );


    int i;

    for( i = 0; i < h_a.length; i++){
            h_a.setElement( i, 1.0 );  
            h_b.setElement( i, 1.0 );                
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);

    float finalValue=0.0;

     // verify that the GPU did the work we requested
    for (int i=0; i<BLOCKS; i++) {
          finalValue += h_c.getElement(i);
    }

    printf( "Dot result = %f \n", finalValue);

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements); 

}

int main( void ) {
    	
        onHost();
    	return 0;
}
