# Notes
## nvcc

Compilation flow: CUDA C/C++ device code source --> PTX --> SASS(Streaming ASSembler)

For `nvcc -gencode arch=compute_60,code=sm_60`, `-arch` specifies what type of PTX code to be generated, `-code` specifies what type of SASS code to be generated.

Ref: 
[What is the purpose of using multiple “arch” flags in Nvidia's NVCC compiler?](https://stackoverflow.com/questions/17599189/what-is-the-purpose-of-using-multiple-arch-flags-in-nvidias-nvcc-compiler/17599585#17599585)

[CUDA: How to use -arch and -code and SM vs COMPUTE](https://stackoverflow.com/questions/35656294/cuda-how-to-use-arch-and-code-and-sm-vs-compute)

[what is “SASS” short for? [closed]](https://stackoverflow.com/questions/9798258/what-is-sass-short-for)
