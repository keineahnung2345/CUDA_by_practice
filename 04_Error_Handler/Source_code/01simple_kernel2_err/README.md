# Notes
## device variable
```cpp
__device__ int two = 0;
```
From [Scope of \_\_device\_\_ variable](https://devtalk.nvidia.com/default/topic/932699/scope-of-__device__-variable/), after adding `__device__` before the declaration of a variable, that variable can be used in any kernel function.

