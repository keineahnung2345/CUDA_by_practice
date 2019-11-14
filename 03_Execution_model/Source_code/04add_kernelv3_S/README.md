# Notes

```cpp
#define ARRAY_SIZE 1000
#define THREADS 3
#define BLOCKS 7

__global__ void addKernel3(  int  *d_a, int *d_b, int *d_result){

  int SIZE = 143;

  int idx = threadIdx.x + blockIdx.x * SIZE;
    while(idx < (SIZE*(blockIdx.x+1)) && idx < ARRAY_SIZE ){
      d_result[idx] = d_a[idx] + d_b[idx];
      //d_result[idx] = idx;
      idx += blockDim.x;
    }
}
```

| blockIdx.x    | threadIdx.x   | idx                 | block range  |
| :------------ |:--------------| :-------------------| :----------- |
| 0             | 0             | 0,7,14,...,140      | 0~142        |
| 0             | 1             | 1,8,15,...,141      |
| 0             | 2             | 2,9,16,...,142      |
| 1             | 0             | 143,150,157,...,283 | 143~285      |
| 1             | 1             | 144,151,158,...,284 |
| 1             | 2             | 145,152,159,...,285 |
| 2             | 0             | 286,293,300,...,426 | 286~429      |
| 2             | 1             | 287,294,301,...,427 |
| 2             | 2             | 288,295,302,...,428 |
| 3             | -             | -                   | 430~571      |
| 4             | -             | -                   | 572~715      |
| 5             | -             | -                   | 716~858      |
| 6             | -             | -                   | 859~999(1001)|

The number `143` is calculated as ⌈ARRAY_SIZE/BLOCKS⌉, meaning each block need to do about 143 calculations.
