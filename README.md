![Demo gif.](demo.gif)

# NS2D (Navier--Stokes in two dimensions)

NS2D is a python package for simulating two-dimensional turbulence on the torus. 

## Why use this package?
- *NS2D is extremely transparent*. The core modules are written in high-level python, meaning it's easy to see how the model works. Users specify explicitly which mechanisms they want in their model, such as types of dissipation and forcing, allowing for a range of model variations. In geophysical terms we solve the barotropic vorticity equation with the option of a beta plane configuration.

- *GPU compatibility*. NS2D runs on GPU. Pseudo-spectral solvers like this one rely on fast FFT implementations to be fast. Through [CuPy](https://cupy.dev/) we exploit NVIDIA's highly optimised [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) library without any complicated bookkeeping.


NS2D borrows heavily from existing pseudo-spectral turbulence models, especially the [pyqg](https://github.com/pyqg) project. While it is currently a single layer model with a square domain, its extension to general rectangular n-layer configurations would likely be straightforward. 

Note that an earlier (much slower) CPU version of NS2D, which relies on [pyfftw](https://pyfftw.readthedocs.io/en/latest/), is available as [v1.0.0](https://github.com/mtbrolly/NS2D/releases/tag/v1.0.0).