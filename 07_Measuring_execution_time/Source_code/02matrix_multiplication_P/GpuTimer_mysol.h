#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__


struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    // create events 
    // -:YOUR CODE HERE:- 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    // delete events 
    // -:YOUR CODE HERE:- 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    // start event
    // -:YOUR CODE HERE:- 
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    // stop event
    // -:YOUR CODE HERE:- 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
  }

  float Elapsed()
  {
    // elapsed time 
    // -:YOUR CODE HERE:- 
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

#endif  /* GPU_TIMER_H__ */
