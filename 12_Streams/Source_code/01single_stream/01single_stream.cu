#include <math.h>
#include <stdio.h>
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Vector.h"
#include <assert.h>
#include <limits> //std::numeric_limits<float>::epsilon()

#define N 10000
#define DIMGRID 10
#define DIMBLOCK 10
#define XMIN -10.0f
#define XMAX 10.0f

const int ARRAY_BYTES = N * sizeof(float);

__host__ __device__ float function1(float x)
{
	return x*x;
}

__device__ float function1_d(float x){
        return __fmul_rn(x, x);
}

__host__ __device__ float function2(float x)
{
	return sinf(x);
}

__device__ float function2_d(float x)
{
	return __sinf(x);
}

__global__ void functionKernel1( Vector<float> d_a, int n)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	float x, dx;
	
	dx = (XMAX - (XMIN)) / ((float) N-1);
	while (i < n)
	{
		//x = XMIN + i*dx;
		x = __fadd_rn(XMIN, __fmul_rn(i, dx));
		//d_a.setElement( i, function1(x) );
		d_a.setElement( i, function1_d(x) );
		i += blockDim.x*gridDim.x;
	}
	
}

__global__ void functionKernel2( Vector<float> d_a, int n)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	float x, dx;
	
	dx = (XMAX-(XMIN))/((float)N -1);
	
	while (i < n)
	{
		//x = XMIN + i*dx;
		x = __fadd_rn(XMIN, __fmul_rn(i, dx));
		//d_a.setElement( i, function2(x) );
		d_a.setElement( i, function2_d(x) );
		i += blockDim.x*gridDim.x;
	}
	
}


void onDevice(Vector<float> h_a, Vector<float> h_b)
{
	   
	Vector<float> d_a, d_b;
	
	// create the stream
	cudaStream_t stream1;	
	HANDLER_ERROR_ERR(cudaStreamCreate (&stream1));	
	
	GpuTimer timer;
    timer.Start(stream1);

    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));

	functionKernel1<<< DIMGRID, DIMBLOCK , 0, stream1 >>>(d_a, N);

	HANDLER_ERROR_MSG("kernel panic!!!"); 	
	
	//cudaThreadSynchronize(); 
        //https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__THREAD__DEPRECATED.html#group__CUDART__THREAD__DEPRECATED_1g86ba4d3d51221f18a9a13663e6105fc1
        //cudaThreadSynchronize is deprecated
        cudaDeviceSynchronize();
	

	functionKernel2<<< DIMGRID, DIMBLOCK, 0, stream1  >>>(d_b, N);
	HANDLER_ERROR_MSG("kernel panic!!!");

	HANDLER_ERROR_ERR(cudaMemcpy(h_a.elements, d_a.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost));
	HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements, d_b.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost));	

    // stop timer
    timer.Stop(stream1); 	

    // print time
    printf( "Time :  %f ms\n", timer.Elapsed() );
	

	//destroy stream
	HANDLER_ERROR_ERR(cudaStreamDestroy(stream1));	
	
	//free device memory
	HANDLER_ERROR_ERR(cudaFree(d_a.elements));
	HANDLER_ERROR_ERR(cudaFree(d_b.elements));
	
}

void checkDeviceProps(){
    cudaDeviceProp  prop;
    int whichDevice;
    HANDLER_ERROR_ERR( cudaGetDevice( &whichDevice ) );
    printf("We are on device %d\n", whichDevice);
    HANDLER_ERROR_ERR( cudaGetDeviceProperties( &prop, whichDevice ) );
    printf("prop.deviceOverlap: %d\n", prop.deviceOverlap);
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
    }
}


void test(){

    Vector<float> h_a, h_b;
    h_a.length = N;
    h_b.length = N;

    h_a.elements = (float*)malloc( ARRAY_BYTES );
    h_b.elements = (float*)malloc( ARRAY_BYTES );
	
    // call device configuration
    onDevice(h_a, h_b);

/*
    printf("square:\n");
    h_a.print();
    printf("sin:\n");
    h_b.print();
*/         

    for(int i = 0; i < N; i++){
        float x, dx;
        dx = (XMAX - (XMIN)) / ((float) N-1);
        x = XMIN + i*dx;

        assert(h_a.getElement(i) == function1(x));
        assert(std::fabs(h_b.getElement(i) - function2(x)) <= std::numeric_limits<float>::epsilon() * 10);
    }

    // free host memory
    free(h_a.elements);
    free(h_b.elements);

}

int main(){

	checkDeviceProps();
	test();

}
