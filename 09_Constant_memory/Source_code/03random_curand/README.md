# Notes
## Compilation
From [CURAND Library - Compiling Error - Undefined reference to functions](https://stackoverflow.com/questions/11734578/curand-library-compiling-error-undefined-reference-to-functions).

This should be compiled with:
```sh
nvcc 03random_curand.cu -lcurand
```

Otherwise it gives the following error:
```
/tmp/tmpxft_00007317_00000000-10_03random_curand.o: In function `randomNumbersGenerator(int)':
tmpxft_00007317_00000000-5_03random_curand.cudafe1.cpp:(.text+0x140): undefined reference to `curandCreateGenerator'
tmpxft_00007317_00000000-5_03random_curand.cudafe1.cpp:(.text+0x195): undefined reference to `curandSetPseudoRandomGeneratorSeed'
tmpxft_00007317_00000000-5_03random_curand.cudafe1.cpp:(.text+0x1da): undefined reference to `curandGenerateUniform'
tmpxft_00007317_00000000-5_03random_curand.cudafe1.cpp:(.text+0x28c): undefined reference to `curandDestroyGenerator'
collect2: error: ld returned 1 exit status
```
