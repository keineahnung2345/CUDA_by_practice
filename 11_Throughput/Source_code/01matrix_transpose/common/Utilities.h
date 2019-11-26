#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <stdio.h>     


void bandwidth(int n, float time){
        //time: in ms

	cudaDeviceProp  prop;

	float efbw,tbw, ram;

	cudaGetDeviceProperties( &prop, 0 );

        //int cudaDeviceProp::memoryClockRate [inherited]
        //Peak memory clock frequency in kilohertz
        //should be prop.memoryClockRate * 10e3f? (from KHz to Hz)
        
        //int cudaDeviceProp::memoryBusWidth [inherited]
        //Global memory bus width in bits
        //should be time * 10e-3f?(from ms to s)

	//tbw = (((prop.memoryClockRate * 10e2f) * (prop.memoryBusWidth/8))*2)/10e9f;
	//efbw = ((n*n*4*2)/10e9)/(time*10e-4f);

	tbw = (((prop.memoryClockRate * 10e3f) * (prop.memoryBusWidth/8))*2)/10e9f;
	efbw = ((n*n*4*2)/10e9)/(time*10e-3f);
	ram = (efbw / tbw)*100;

	printf("TBw = %.3f GB/s, EBw = %.3f GB/s, RAM Utilization = %.2f %%  \n", tbw, efbw, ram);
    // 40 - 60 % OK
    // 60 - 75 % Good
    // > 75 % excellent
}

#endif
