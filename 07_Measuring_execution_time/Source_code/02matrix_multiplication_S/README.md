# Notes
## \_\_device\_\_ \_\_host\_\_
From [What is: \_\_host_\_ \_\_device\_\_](https://devtalk.nvidia.com/default/topic/519806/what-is-__host__-__device__/), we can use \_\_device\_\_ and \_\_host\_\_ together.

```cpp
template <typename T>
struct Matrix{
        int width;
        int height;
        T* elements;

        __device__ __host__ T getElement( int row, int col)
        {
        return elements[row * width + col];
        }

        __device__ __host__ void setElement(int row, int col, T value)
        {
        elements[row * width + col] = value;
        }


} ;
```
