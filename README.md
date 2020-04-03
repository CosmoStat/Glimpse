# Glimpse
[![Build Status](https://travis-ci.org/CosmoStat/Glimpse.svg?branch=master)](https://travis-ci.org/CosmoStat/Glimpse)

Glimpse is a sparsity based mass-mapping algorithm. See the
[Glimpse page](http://www.cosmostat.org/software/glimpse) on the
CosmoStat website for more information.

Note, the Glimpse version corresponding to [(Lanusse et al. 2016)](https://arxiv.org/abs/1603.01599) is [v1.0](https://github.com/CosmoStat/Glimpse/tree/v1.0)

## Installation

### Requirements

Glimpse depends on the following software:

* **Armadillo**
* **Boost**
* **FFTW** version 3.3 or later
* **NFFT** version 3.2 or later
* **cfitsio** and **CCFits**
* **GSL**
* **CMake** version 2.8 or later

These dependencies can easily be installed using a package manager:

* Setting up requirements on **Linux**:
  Simply use the package manager provided by your Linux distribution, for instance on Ubuntu Linux:
  ```
    $ sudo apt-get install cmake libgsl0-dev libfftw3-3  libccfits-dev libnfft3-dev libarmadillo-dev libboost-all-dev
  ```
* Setting up requirements on **MacOS X**:
  The preferred installation method for the dependencies is through [Homebrew](https://brew.sh/):
  ```  
    $ brew tap brewsci/science
    $ brew install libomp cmake nfft gsl fftw ccfits armadillo boost
  ```
  Note that when using `brew` to install dependencies, compiling with `gcc` is **not supported**, please use the default AppleClang instead.  

### Compilation

Once the requirements are installed, Glimpse can be compiled by running the  following commands:
  ```
    $ cd Glimpse
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
  ```
This will create a Glimpse executable in the build directory.

## 2D Usage

Glimpse expects input shear and/or flexion data as columns in a FITS file. A simple python script for converting .txt files to FITS is provided in the *utils* folder.

All the options of the reconstruction algorithm can be specified in a *config.ini* file such as the one provided in the *example* directory.

Glimpse can be run with the following command line:
  ```
    $ glimpse config.ini cat_3_0.fits kappa.fits
  ```
Where *kappa.fits* is the reconstructed convergence map (scaled for sources at infinite redshift) and *cat_3_0.fits* is the input data file.

## 3D Usage

Glimpse can be used to recontruct a 3D field using the same command line:
  ```
    $ glimpse config3d.ini cat_3_0.fits delta.fits
  ```
An example of *config3d.ini* can be found in the *example* directory.

## GP-GPU

Reconstructing a 3D field is very computationally demanding, and using a GPU is higly recommended to speed up the reconstruction. If an installation of CUDA can be
detected on your system, CMake will automatically compile GPU specific code to
replace the CPU implementation.

Glimpse uses peer-to-peer memory transfer between GPUs on a multi-gpu system. You
may choose at runtime which GPUs to use in your system by providing the -g option:
  ```
    $ glimpse -g 1,2 config3d.ini cat_3_0.fits delta.fits
  ```
This option will use only GPUs 1 and 2 even if more are installed on the system. To
get a list of CUDA capable devices on your system, use the deviceQuery command
provided with CUDA.
