# Notes
## #pragma unroll
[What does #pragma unroll do exactly? Does it affect the number of threads?](https://stackoverflow.com/questions/22278631/what-does-pragma-unroll-do-exactly-does-it-affect-the-number-of-threads)

## theoretical bandwidth & effective bandwidth
Chapter 8.2 of [CUDA_C_Best_Practices_Guide](https://docs.nvidia.com/pdf/CUDA_C_Best_Practices_Guide.pdf)

Theoretical bandwidth:

![image](https://github.com/keineahnung2345/CUDA_by_practice/blob/note/11_Throughput/theoretical_bandwidth.png)

Effective bandwidth:

![alt text](https://github.com/keineahnung2345/CUDA_by_practice/blob/note/11_Throughput/effective_bandwidth.png)

## transposedMatrixKernelFinal
Still not understand.
```cpp
__global__ void transposedMatrixKernelFinal(Matrix<int> d_a, Matrix<int> d_b){

        // (i,j) locations of the tile corners for input & output matrices:
        int in_corner_i  = blockIdx.x * blockDim.x, in_corner_j  = blockIdx.y * blockDim.y;
        int out_corner_i = blockIdx.y * blockDim.y, out_corner_j = blockIdx.x * blockDim.x;

        int x = threadIdx.x, y = threadIdx.y;

        __shared__ float tile[K][K+1]; //in 01matrix_transpose_events.cu - 
        //__shared__ float tile[K][K]; //in 03matrix_transpose_S.cu

        while(in_corner_j + x < N){
                in_corner_i = blockIdx.x * blockDim.x;
                out_corner_j  = blockIdx.x * blockDim.x;
                while( in_corner_i + y < N) {

                        tile[y][x] = d_a.elements[(in_corner_i + x) + (in_corner_j + y)*N];

                        __syncthreads();


                        d_b.elements[(out_corner_i + x) + (out_corner_j + y)*N] = tile[x][y];

                        in_corner_i += blockDim.x * gridDim.x;
                        out_corner_j += blockDim.x * gridDim.x;
                }
                in_corner_j +=  gridDim.y * blockDim.y;
                out_corner_i += gridDim.y * blockDim.y;

        }

}
```
