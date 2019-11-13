#include <stdio.h>

__global__ void kernel( void ) {
}

//make exe(make), make clean, make run

int main( void ) {
    kernel<<<1,1>>>();
    printf( "Hello, World!\n" );
    return 0;
}
