# CMU 15618 Final Project: Wavefront Path Tracing

[Project site](https://zymk9.github.io/cmu15618_project/).

This repo contains the course project for CMU 15-618: Parallel Computer Architecture and Programming Fall 2023. We adatped a CPU path tracer from [Yocto/GL](https://github.com/xelatihy/yocto-gl) to CUDA and extended it with the wavefront design and the wide BVH from the following papers:

- [Megakernels Considered Harmful: Wavefront Path Tracing on GPUs](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- [Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs](https://research.nvidia.com/publication/2017-07_efficient-incoherent-ray-traversal-gpus-through-compressed-wide-bvhs)

We compare the performance of our wavefront path tracer, the original megakernel version in CUDA (without using OptiX), and the multithreaded CPU version. Our implementation achieves a speedup of 1.02x - 1.79x comparing to the megakernel CUDA version, and 1.49x - 8.20x comparing to running on a 16-core CPU.


## Compilation
First, check the original [repo](https://github.com/xelatihy/yocto-gl) for requirements. By default, `cmake ..` will generate configuration for the wavefront version. To generate the megakernel version without OptiX, run
```bash
cmake -DWAVEFRONT=OFF ..
```

To generate the original CUDA version using OptiX, run
```bash
cmake -DWAVEFRONT=OFF -DCUSTOM_CUDA=OFF ..
```

The target names are `wavefront_trace`, `cuda_trace`, and `ycutrace`, respectively. 

To use the wide BVH, specify `--wbvh` in the command line. To use the three-stage (logic, material, ray cast) wavefront pipeline instead of the two-stage one (logic, ray cast), add `--matstage` to the command for `wavefront_trace`.
