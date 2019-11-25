# Notes
## transposedMatrixKernel_tile_padded
The reason for the `K+1` in `__shared__ float tile[K][K+1];`: 

[CUDA Toolkit Documentation - 9.2.2.3. Shared Memory in Matrix Multiplication (C=AAT)](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-aa)
