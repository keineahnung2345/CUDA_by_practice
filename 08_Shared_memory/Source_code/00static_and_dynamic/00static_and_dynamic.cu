#include <stdio.h>
#include <assert.h>
#include "Vector.h"
#include "Error.h"
#include "GpuTimer.h"

#define N 4
const int ARRAY_BYTES = N * sizeof(int);

__global__ void globalReverseKernel(Vector<int> d_a, Vector<int> d_b)
{
  //from d_a to d_b
  int t = threadIdx.x;
  //d_b.setElement(t, d_a.getElement(N-t-1));
  d_b.setElement(N-t-1, d_a.getElement(t));
}

__global__ void staticReverseKernel(Vector<int> d_a)
{
  __shared__ int s[N];
  int t = threadIdx.x;
  int tr = N-t-1;
  s[t] = d_a.getElement(t);
  __syncthreads();
  d_a.setElement(t, s[tr]);
}

__global__ void dynamicReverseKernel(Vector<int> d_a)
{
  // -* note the empty brackets and the use of the extern specifier *-
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = N-t-1;
  s[t] = d_a.getElement(t);
  __syncthreads();
  d_a.setElement(t, s[tr]);
}

void test(Vector<int> d_b, Vector<int> d_c){
    for (int i = 0; i < N; i++) {
      assert(d_b.getElement(i) == d_c.getElement(i)); 
    }

}

void onDevice(Vector<int> d_a, Vector<int> d_b, Vector<int> d_c){

  Vector<int> d_d;
  d_d.length = N;    


  // allocate  memory on the GPU
  HANDLER_ERROR_ERR(cudaMalloc((void**)&d_d.elements,ARRAY_BYTES));

  // copy data from CPU the GPU
  HANDLER_ERROR_ERR(cudaMemcpy(d_d.elements, d_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));
    
  GpuTimer timer;
  timer.Start();

  //run with STATIC shared memory  
  staticReverseKernel<<<1,N>>>(d_d);
  HANDLER_ERROR_MSG("kernel panic!!!");
    
  // stop timer
  timer.Stop();
 
  // print time
  printf( "Time :  %f ms\n", timer.Elapsed() );

  // copy data back from the GPU to the CPU
  HANDLER_ERROR_ERR(cudaMemcpy(d_c.elements, d_d.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost)); 

  //checing results
  test(d_b, d_c);



  // copy data from CPU the GPU
  HANDLER_ERROR_ERR(cudaMemcpy(d_d.elements, d_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));

  timer.Start();
  
  //run with DYNAMIC shared memory 
  dynamicReverseKernel<<<1,N,N*sizeof(int)>>>(d_d);

  // stop timer
  timer.Stop();

  // print time
  printf( "Time :  %f ms\n", timer.Elapsed() );

  HANDLER_ERROR_MSG("kernel panic!!!");

  // copy data back from the GPU to the CPU
  HANDLER_ERROR_ERR(cudaMemcpy(d_c.elements, d_d.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost)); 

  //checing results
  test(d_b, d_c);


  //global kernel
  Vector<int> d_e;
  d_e.length = N;


  // allocate  memory on the GPU
  HANDLER_ERROR_ERR(cudaMalloc((void**)&d_e.elements,ARRAY_BYTES));

  // copy data from CPU the GPU
  HANDLER_ERROR_ERR(cudaMemcpy(d_e.elements, d_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));

  timer.Start();

  //run with global memory  
  globalReverseKernel<<<1,N>>>(d_e, d_d);
  HANDLER_ERROR_MSG("kernel panic!!!");

  // stop timer
  timer.Stop();

  // print time
  printf( "Time :  %f ms\n", timer.Elapsed() );

  // copy data back from the GPU to the CPU
  HANDLER_ERROR_ERR(cudaMemcpy(d_c.elements, d_d.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost));

  //checing results
  test(d_b, d_c);



  HANDLER_ERROR_ERR(cudaFree(d_d.elements));
}




void onHost(){

    Vector<int> h_a, h_b, h_c;
    h_a.length = N;
    h_b.length = N;
    h_c.length = N;    

    h_a.elements = (int*)malloc(  ARRAY_BYTES );
    h_b.elements = (int*)malloc(  ARRAY_BYTES );
    h_c.elements = (int*)malloc(  ARRAY_BYTES );
    

    for (int i = 0; i < N; i++) {
      h_a.setElement(i,i);
      h_b.setElement(i,N-i-1);
      h_c.setElement(i,0);
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements); 

}


int main(void)
{
  onHost();
}

/*
Time :  0.010240 ms
Time :  0.005120 ms
Time :  0.004096 ms
*/
