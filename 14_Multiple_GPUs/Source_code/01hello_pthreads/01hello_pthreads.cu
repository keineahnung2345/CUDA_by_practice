#include <stdio.h>
#include <pthread.h>
#include <unistd.h> //sleep

void *hello (void *arg) {
	int *ptrToThreadNumber = (int *)arg;
	sleep(5);
	printf("HelloThread %d\n",*ptrToThreadNumber);
	return 0;
}

int main(void) {
	pthread_t tid;
	int threadNum=1;

	//pthread_create creates a new thread and makes it executable. 
	//This routine can be called any number of times from anywhere within your code.
	/*
        int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                          void *(*start_routine) (void *), void *arg);
        The pthread_create() function starts a new thread in the calling
        process.  The new thread starts execution by invoking
        start_routine(); arg is passed as the sole argument of
        start_routine().
        If attr is NULL, then the thread is created with
        default attributes.
        */
	pthread_create(&tid,NULL,hello,&threadNum);

	//"Joining" is one way to accomplish synchronization between threads
	//The pthread_join() subroutine blocks the calling thread until the 
	//specified threadid thread terminates.
        /*
        int pthread_join(pthread_t thread, void **retval);
        If retval is not NULL, then pthread_join() copies the exit status of
        the target thread into the location pointed to by retval. 
        */
	pthread_join(tid,NULL);
	printf("------------------------\n");
	return(0);
}

/*
result:
HelloThread 1
------------------------
*/

/*
result without pthread_join:
------------------------
*/
