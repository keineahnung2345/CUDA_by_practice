# Notes

## extern
Ref: [C/C++ 中的 static, extern 的變數](https://medium.com/@alan81920/c-c-%E4%B8%AD%E7%9A%84-static-extern-%E7%9A%84%E8%AE%8A%E6%95%B8-9b42d000688f).

If we want to notify the compiler that a variable exists, but don't want to decalre it in current file, we can use `extern` keyword.

## dynamic shared memory
```cpp
__global__ void dynamicReverseKernel(Vector<int> d_a)
{ 
  // -* note the empty brackets and the use of the extern specifier *-
  extern __shared__ int s[];
  //...
}

void onDevice(Vector<int> d_a, Vector<int> d_b, Vector<int> d_c){
  //...
  dynamicReverseKernel<<<1,N,N*sizeof(int)>>>(d_d);
  //...
}
```

The size of dynamic shared memory `s` is specified using optional third execution configuration parameter.

## shared memory not faster than global memory!?

Ref: [CUDA: Is coalesced global memory access faster than shared memory? Also, does allocating a large shared memory array slow down the program?](https://stackoverflow.com/questions/9196134/cuda-is-coalesced-global-memory-access-faster-than-shared-memory-also-does-al/9196695)
