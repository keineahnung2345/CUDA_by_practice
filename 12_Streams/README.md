# Notes

## pageable host memory
Ref: [Paging in Operating System](https://www.geeksforgeeks.org/paging-in-operating-system/)

## page-locked memory
Ref: [CUDA Pinned memory](http://www.orangeowlsolutions.com/archives/443)
```
Page-locked (pinned) memory enables a DMA on the GPU to request transfers to and from the host memory 
without the involvement of the CPU. 

Locked memory is stored in the physical memory (RAM), so the GPU (or device, in the language of GPGPU)
can fetch it without the help of the host (synchronous copy).
```

## directory memory access(DMA)
Ref: [Direct Memory Access (DMA)](https://www.techopedia.com/definition/2767/direct-memory-access-dma)

```
Direct memory access (DMA) is a method that allows an input/output (I/O) device to send or receive 
data directly to or from the main memory, bypassing the CPU to speed up memory operations.
```

## FSB
Ref: [FSB](https://www.computerhope.com/jargon/f/fsb.htm)

```
Short for front-side bus, FSB is also known as the processor bus, memory bus, or system bus and 
connects the CPU (chipset) with the main memory and L2 cache.
```

## memory mapping
Ref: [What is Memory mapping?](http://ecomputernotes.com/fundamental/input-output-and-memory/memory-mapping)

```
Memory mapping is the translation between the logical address space and the physical memory.
```

## PCI express bus
Ref: [Peripheral Component Interconnect Express (PCIe, PCI-E)](https://searchdatacenter.techtarget.com/definition/PCI-Express)

```
Peripheral Component Interconnect Express (PCIe or PCI-E) is a serial expansion bus standard for 
connecting a computer to one or more peripheral devices.
```

## GPU engine
Ref: [GPUs in the Task Manager](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)

```
A GPU engine represents an independent unit of silicon on the GPU that can be scheduled and 
can operate in parallel with one another. GPU engines are made up of GPU cores.
```

![image](https://github.com/keineahnung2345/CUDA_by_practice/blob/note/12_Streams/GPU_engine_cores.png)
