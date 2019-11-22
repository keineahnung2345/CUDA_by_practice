
#ifndef __LOCK_H__
#define __LOCK_H__

struct Lock {
    int *mutex;

    Lock( void ) {
        //0 means green light, 1 means red light
        int state = 0;
        HANDLER_ERROR_ERR( cudaMalloc( (void**)&mutex, sizeof(int) ) );
        HANDLER_ERROR_ERR( cudaMemcpy( mutex, &state, sizeof(int), cudaMemcpyHostToDevice ) );
    }

    ~Lock( void ) {
        cudaFree( mutex );
    }

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
};

#endif
