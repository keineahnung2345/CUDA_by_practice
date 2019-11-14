# Notes

## calculate index
Defining kernel of shape:
```cpp
dim3 ThreadsBlocks( 3, 3 );
dim3 GridBlocks( 2 , 2);
matrixKernel<<< GridBlocks, ThreadsBlocks >>>( d_in, d_out );
```

Calculate index:
```cpp
BLOCK_SIZE = blockDim.x;
STRIDES = gridDim.x * BLOCK_SIZE;

// Block index
int bx = blockIdx.x;
int by = blockIdx.y;

// Thread index (current coefficient)
int tx= threadIdx.x;
int ty= threadIdx.y;

//indices in y direction are multiplied with STRIDE
//indices of block are multiplied with BLOCK_SIZE
d_a[ (by * BLOCK_SIZE + ty) * STRIDE + (bx* BLOCK_SIZE + tx) ];
```
