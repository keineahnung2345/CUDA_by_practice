#include <stdio.h>
#include "Error.h"
#include "GpuTimer.h"
#include "Matrix.h"

#define N 40
#define K 2

__global__ void matrixMultiplicationKernel(Matrix<float> d_a, Matrix<float> d_b, Matrix<float> d_c){

        // -:YOUR CODE HERE:-   
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        float sum = 0;
        for(int k = 0; k < N; k++){
           sum += d_a.getElement(i, k) * d_b.getElement(k, j);
        }
        d_c.setElement(i, j, sum);
}


void onDevice(Matrix<float> h_a, Matrix<float> h_b, Matrix<float> h_c  ){


    // declare GPU matrices
        // -:YOUR CODE HERE:-
    Matrix<float> d_a, d_b, d_c;
    d_a.width = h_a.width;
    d_a.height = h_a.height;

    d_b.width = h_b.width;
    d_b.height = h_b.height;

    d_c.width = h_c.width;
    d_c.height = h_c.height;

        // start timer
    GpuTimer timer;
        timer.Start();

        const int ARRAY_BYTES = N * N * sizeof(float);

    // allocate  memory on the GPU
        // -:YOUR CODE HERE:-
    cudaMalloc((void**)&d_a.elements, ARRAY_BYTES);
    cudaMalloc((void**)&d_b.elements, ARRAY_BYTES);
    cudaMalloc((void**)&d_c.elements, ARRAY_BYTES);

    // copy data from CPU the GPU
        // -:YOUR CODE HERE:-
    cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c.elements, h_c.elements, ARRAY_BYTES, cudaMemcpyHostToDevice);

    //execution configuration
        // -:YOUR CODE HERE:-
    dim3 GridBlocks(N/K, N/K);
    dim3 ThreadsBlocks(K, K);

         //run the kernel
        matrixMultiplicationKernel<<<GridBlocks,ThreadsBlocks>>>( d_a, d_b, d_c );
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
        // -:YOUR CODE HERE:-
    cudaMemcpy(h_c.elements, d_c.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost);


        // stop timer
        timer.Stop();

    // print time
        printf( "Time :  %f ms\n", timer.Elapsed() );

    // free GPU memory
        // -:YOUR CODE HERE:-
    cudaFree(d_a.elements);
    cudaFree(d_b.elements);
    cudaFree(d_c.elements);
}


void test(){

        Matrix<float> h_a, h_b, h_c;

        h_a.width = N;
        h_a.height = N;

        h_b.width = N;
        h_b.height = N;

        h_c.width = N;
        h_c.height = N;

        h_a.elements = (float*)malloc(h_a.width  * h_b.height  * sizeof(int));
        h_b.elements = (float*)malloc(h_b.width  * h_b.height  * sizeof(int));
        h_c.elements = (float*)malloc(h_c.width  * h_c.height  * sizeof(int));

        int i,j, k=1;

        for( i = 0; i < h_a.height; i++){
                for( j = 0; j < h_a.height; j++){
                        h_a.setElement( i, j, k );
                        h_b.setElement( i, j, 1.0 );
                        h_c.setElement( i, j, 0.0 );
                        k++;
                }
        }

    // call device configuration
        onDevice(h_a, h_b, h_c);


        // print  result
        for(i=0; i < h_c.width; i++){
                for(j = 0; j < h_c.height; j++){
                        printf("%.2f ", h_c.elements[ i * h_c.width + j ] );
                }
                printf("\n");
        }

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements);
}


int main(){

        test();
}
