#include <stdio.h>
#include <omp.h>

//gcc -fopenmp 06hello_openmp.c

//note that "gcc 06hello_openmp.c" will also generate a "./a.out", and it can run successully, 
//but the result are deterministic(not execute parallelly)

int main() {

    int i;
    printf("Hello World\n");
    #pragma omp parallel
    {
		#pragma omp for
		for(i=0;i<6;i++) {
            printf("Iter:%d\n",i);
		}
    }
    printf("GoodBye World\n");

}

/*
result:
Hello World
Iter:0
Iter:1
Iter:5
Iter:3
Iter:2
Iter:4
GoodBye World

note the order will change every time it is executed
*/
