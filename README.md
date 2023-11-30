# CMU 15618 Final Project

Adatped from Yocto/GL.

Our files to work on:
- `yocto/cuda_trace.{h,cpp,cu}`

Main source of adaptation:
- `yocto/yocto_trace.{h,cpp}`: path tracing of surfaces and hairs supporting
  area and environment illumination, microfacet GGX and subsurface scattering,
  multiple importance sampling
- `yocto/yocto_cutrace.{h,cpp,cu}`: CUDA/OptiX version.


Main demo:
- `apps/ytrace.cpp`: offline and interactive scene rendering
- `apps/ycutrace.cpp`: offline and interactive scene rendering with CUDA


## Compilation

This library requires a C++17 compiler and is know to compiled on
OsX (Xcode >= 11), Windows (MSVC >= 2019) and Linux (gcc >= 9, clang >= 9).

You can build the example applications using CMake with
`mkdir build; cd build; cmake ..; cmake --build .`

Yocto/GL required dependencies are included in the distribution and do not
need to be installed separately.

Yocto/GL optionally supports building OpenGL demos. OpenGL support is enabled
by defining the cmake option `YOCTO_OPENGL`. 
OpenGL dependencies are included in this repo.

Yocto/GL optionally supports the use of Intel's Embree for ray casting.
See the main CMake file for how to link to it. Embree support is enabled by
defining the cmake option `YOCTO_EMBREE`. Embree needs to be installed separately.

Yocto/GL optionally supports the use of Intel's Open Image Denoise for denoising.
See the main CMake file for how to link to it. Open Image Denoise support
is enabled by defining the cmake option `YOCTO_DENOISE`.
OIDN needs to be installed separately.
