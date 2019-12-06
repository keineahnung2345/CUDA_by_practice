#include <curand.h>
#include "common/Error.h"
#include "common/Vector.h"
#include "common/GpuTimer.h"
#include "common/CUTThread.h"

#define N   (32 * 1024)  // (32768/2)=16384
#define THREADS 256
#define BLOCKS 32

const int ARRAY_BYTES = N * sizeof(float);
const int P_ARRAY_BYTES = BLOCKS * sizeof(float);

struct DataStruct {
    int     deviceID;
    int     size;
    float   *a;
    float   *b;
    float   returnValue;
};

__global__ void dotKernel( Vector<float> d_a, Vector<float> d_b, Vector<float> d_c, int size ){

    __shared__ float cache[THREADS];
    //used to index the whole grid
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //used to index "cache", same size as a block
    int cacheIndex = threadIdx.x;

    //move the grid of size THREADS*BLOCKS through the whole array of length size(=PARTIAL_ARRAY_SIZE)
    float   temp = 0;
    while (tid < size) {
        temp += d_a.getElement(tid) * d_b.getElement(tid);
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();
    // now all THREADS elements of cache are set
    
    // reduce THREADS elements of cache to cache[0]
    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        __syncthreads();
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        i /= 2;
    }

    //"cache" is belong to a block
    //every block calculates its dot value
    //d_c's size is BLOCKS
    if (cacheIndex == 0)
        d_c.setElement(blockIdx.x, cache[0]);

}


void* onDevice( void *pvoidData ){
    
    DataStruct  *data = (DataStruct*)pvoidData;
    //declare the device to use for this thread
    HANDLER_ERROR_ERR( cudaSetDevice( data->deviceID ) );

    const int PARTIAL_ARRAY_SIZE = data->size;
    const int PARTIAL_ARRAY_BYTES = PARTIAL_ARRAY_SIZE * sizeof(float);

    Vector<float> h_a, h_b, h_c;
    Vector<float> d_a, d_b, d_c;
    d_a.length = PARTIAL_ARRAY_SIZE;
    d_b.length = PARTIAL_ARRAY_SIZE;
    d_c.length = BLOCKS; 

    // allocate memory on the CPU side
    h_a.elements = data->a;
    h_b.elements = data->b;
    h_c.elements = (float*)malloc( P_ARRAY_BYTES );

    // start timer
    GpuTimer timer;
    timer.Start();

    // allocate  memory on the GPU
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements,PARTIAL_ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements,PARTIAL_ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements,P_ARRAY_BYTES));

    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, PARTIAL_ARRAY_BYTES, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, PARTIAL_ARRAY_BYTES, cudaMemcpyHostToDevice)); 
  

    dotKernel<<<BLOCKS,THREADS>>>( d_a, d_b, d_c, PARTIAL_ARRAY_SIZE );
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    HANDLER_ERROR_ERR(cudaMemcpy(h_c.elements, d_c.elements, P_ARRAY_BYTES, cudaMemcpyDeviceToHost)); 

    // finish up on the CPU side
    float partial = 0;
    for (int i=0; i<BLOCKS; i++) {
        partial += h_c.getElement(i);
    }

    // stop timer
    timer.Stop(); 
    
    // print time
    printf( "Time :  %f ms\n", timer.Elapsed() );

    // free GPU memory
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
    HANDLER_ERROR_ERR(cudaFree(d_c.elements));

    // free CPU memory
    free(h_c.elements);
    data->returnValue = partial;
    return 0;

}


void test(){

    Vector<float> h_a, h_b;
    h_a.length = N;
    h_b.length = N;

    // allocate memory
    h_a.elements = (float*)malloc( ARRAY_BYTES );
    h_b.elements = (float*)malloc( ARRAY_BYTES );

    int i;
    for(i=0; i < N; i++){

        h_a.setElement(i,1.0);
        h_b.setElement(i,1.0);
    }

    // prepare for multithread
    DataStruct  data[2];
    data[0].deviceID = 0;
    data[0].size = N/2;
    data[0].a = h_a.elements;
    data[0].b = h_b.elements;

    data[1].deviceID = 1;
    data[1].size = N/2;
    data[1].a = h_a.elements + N/2;
    data[1].b = h_b.elements + N/2;

    //"thread": responsible for data[0]
    CUTThread   thread = start_thread( onDevice, &(data[0]) );
    //data[1]: not use a separate thread, but complete it in main thread
    onDevice( &(data[1]) );
    //"thread": responsible for data[0]
    end_thread( thread );


    float finalValue=  data[0].returnValue + data[1].returnValue;

    printf( "Dot result = %f \n", finalValue);

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);

}

void checkDeviceProps(){
    //properties validation
    int deviceCount;
    HANDLER_ERROR_ERR( cudaGetDeviceCount( &deviceCount ) );
    if (deviceCount < 2) {
        printf( "We need at least two compute 1.0 or greater "
                "devices, but only found %d\n", deviceCount );
        //return 0;
    }
 }


int main( void ) {
    	checkDeviceProps();
        test();
    	return 0;
}

/*
Time :  1.943776 ms
Time :  2.574656 ms
Dot result = 32768.000000 
-: successful execution :-
*/
