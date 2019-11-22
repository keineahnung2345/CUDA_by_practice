# Notes
## atomicCAS & atomicExch
[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
```cuda
__device__ void lock( void ) {
    //every thread is waiting to get the mutex, once a thread won the mutex, it changes 
    //the value to 1    
    //int atomicCAS(int* address, int compare, int val);
    //compare "address"'s value and "compare", if true set "address"'s value to "val"
    //the function returns "address"'s old value
    //it's effect: when *mutex equals to 0, the function set *mutex as 1
    //once *mutex is set as 1, the function continues to return 1 and we are trapped in the while loop
    while( atomicCAS( mutex, 0, 1 ) != 0 );
}

//When the thread end its work, it release the mutex. Now it's ready to be taken.
__device__ void unlock( void ) {
    //int atomicExch(int* address, int val);
    //set *mutex as 0 and return *mutex's old value
    //once *mutex set as 0, we can jump out the while loop in "lock"
    atomicExch( mutex, 0 );
}
```
