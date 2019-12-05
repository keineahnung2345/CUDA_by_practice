#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include "common/CUTThread.h"

#define THREADS 4

int intervalsT=100000000;
double partialStore[]={0.0, 0.0, 0.0, 0.0};
    // -:YOUR CODE HERE:-  


void *threadRoutine(void *param) {
    // -:YOUR CODE HERE:-  
        int tid = *(int*)(param);
        printf("tid: %d\n", tid);
        int i;
        double height,x;
        double base;

        int intervals=100000000;
        base=(double)(1.0/intervals);

        for (i=intervals/THREADS*tid, x=0.0; i<intervals/THREADS*(tid+1); i++) {
                x = i * base;
                height = 4/(1 + x * x);
                partialStore[tid] += base * height;
        }
	return 0;
}

void calculatePIHostMultiple(){

    // -:YOUR CODE HERE:-  
    pthread_t threads[THREADS];
    double store;
    int tids[THREADS];
    for(int i = 0; i < THREADS; i++){
        tids[i] = i;
    }
    for(int i = 0; i < THREADS; i++){
        //if start_thread(threadRoutine, &i), what the function receives will be undeterministic
        threads[i] = start_thread(threadRoutine, &tids[i]);
    }
    wait_for_threads(threads, THREADS); 
    for(int i = 0; i < THREADS; i++){
        store += partialStore[i];
    }
    
    printf("PI (multiple th) =%f \n",store);
}

void calculatePIHostSingle(){

	int i;
	double height,x;
	double store,base;

	int intervals=100000000;
	base=(double)(1.0/intervals);

	//for (i=0, store=0.0, x=0.0; i<intervals; i++, x+=base) {
	for (i=0, store=0.0, x=0.0; i<intervals; i++) {
		x = i * base;
		height = 4/(1 + x * x);
		store += base * height;
	}


	printf("PI (single th) =%f \n",store);

}

int main(){

	calculatePIHostSingle();

	calculatePIHostMultiple();

}
