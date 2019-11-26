#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__



struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float Elapsed()
  {
    float elapsed;
    //__host__​cudaError_t cudaEventSynchronize ( cudaEvent_t event )
    //Waits for an event to complete.
    cudaEventSynchronize(stop);
    int counter = 0;
    while( cudaEventQuery(stop) ==  cudaErrorNotReady ){ 
      counter++;
      printf("Waiting for the Device to finish: %d\n",counter); 
    }
    //__host__​cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end )
    //Computes the elapsed time between events.
    //Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds).
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

#endif  /* GPU_TIMER_H__ */
