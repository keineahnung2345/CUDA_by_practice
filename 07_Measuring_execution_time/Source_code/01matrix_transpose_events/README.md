# Notes
## \_\_device\_\_ and \_\_global\_\_ function
From [Difference between global and device functions](https://stackoverflow.com/questions/12373940/difference-between-global-and-device-functions): host code cannot call device function, but can call global function. 

## cudaStream_t
`cudaStream_t` is used in the funciton `cudaEventRecord`.
```cpp
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
```

About `cudaStream_t`, please check [CUDA随笔之Stream的使用](https://zhuanlan.zhihu.com/p/51402722)
