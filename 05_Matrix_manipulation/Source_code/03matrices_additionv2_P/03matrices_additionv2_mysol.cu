#include <stdio.h>
#include <assert.h>
#include "Error.h"



#define N 500

__global__ void additionMatricesKernel(int *d_a, int *d_b, int *d_c){


        // -:YOUR CODE HERE:-   
        int tx = threadIdx.x, ty = threadIdx.y;
        int bx = blockIdx.x, by = blockIdx.y;

        //the index in a grid
        //note it's blockDim, not gridDim!!
        int y = (by * blockDim.y + ty);
        int x = (bx * blockDim.x + tx);

        for(int runy = y; runy < N; runy += blockDim.y * gridDim.y){
            for(int runx = x; runx < N; runx += blockDim.x * gridDim.x){
/*
for loop tried 1:
//        int runy = y, runx = x;
//        for(runy = y; runy * blockDim.x * gridDim.x + runx < N * N; runy += blockDim.y * gridDim.y){
//            for(runx = x; runy * blockDim.x * gridDim.x + runx < N * N; runx += blockDim.x * gridDim.x){
*/
/*
for loop tried 2:
//        if(true){
//            if(true){
*/
                //index tried 1:
                //d_c[runy * blockDim.x * gridDim.x + runx] = d_a[runy * blockDim.x * gridDim.x + runx] + d_b[runy * blockDim.x * gridDim.x + runx];

                //use N instead of blockDim.x * gridDim.x because runy is the index on the whole matrix!!
                d_c[runy * N + runx] = d_a[runy * N + runx] + d_b[runy * N + runx];
            }
        }

}


void onDevice(int h_a[][N], int h_b[][N], int h_c[][N] ){

    // declare GPU memory pointers
        int *d_a, *d_b, *d_c;

        const int ARRAY_BYTES = N * N * sizeof(int);

    // allocate  memory on the GPU
        // -:YOUR CODE HERE:-           
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c, ARRAY_BYTES));


    // copy data from CPU the GPU
        // -:YOUR CODE HERE:-           
    HANDLER_ERROR_ERR(cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_c, h_c, ARRAY_BYTES, cudaMemcpyHostToDevice));


    //execution configuration
        dim3 GridBlocks( 4,4 );
        dim3 ThreadsBlocks( 8,8 );

         //run the kernel
        additionMatricesKernel<<<GridBlocks,ThreadsBlocks>>>( d_a, d_b, d_c );
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
        // -:YOUR CODE HERE:-           
    HANDLER_ERROR_ERR(cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    HANDLER_ERROR_ERR(cudaMemcpy(h_b, d_b, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    HANDLER_ERROR_ERR(cudaMemcpy(h_c, d_c, ARRAY_BYTES, cudaMemcpyDeviceToHost));


    // free GPU memory
        // -:YOUR CODE HERE:-           
    HANDLER_ERROR_ERR(cudaFree(d_a));
    HANDLER_ERROR_ERR(cudaFree(d_b));
    HANDLER_ERROR_ERR(cudaFree(d_c));

}
void test(int h_a[][N], int h_b[][N], int h_c[][N] ){

        for(int i=0; i < N; i++){
                for(int j = 0; j < N; j++){
                        printf("%d ", h_c[i][j]);
                        if(j == N-1) printf("\n");
                        assert(h_a[i][j] + h_b[i][j] == h_c[i][j]);
                }
        }

    printf("-: successful execution :-\n");
}

void onHost(){


        int i,j;
        int h_a[N][N], h_b[N][N], h_c[N][N];

        for(i = 0; i < N; i++){
                for(j = 0; j < N; j++){
                        h_a[i][j] = h_b[i][j] = i+j;
                        h_c[i][j] = 0;
                }
        }

    // call device configuration
        onDevice(h_a,h_b,h_c);
        test(h_a,h_b,h_c);

}


int main(){

        onHost();
}
