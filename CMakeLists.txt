cmake_minimum_required(VERSION 3.0)

# Paths to external libraries
set(NICAEA_URL https://github.com/EiffL/nicaea)
set(SPARSE2D_URL https://github.com/CosmoStat/Sparse2D.git)

# Adding customized cmake module
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

include(FindPkgConfig)
include(ExternalProject)

project(glimpse)

# Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type "Debug")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# Include external libraries
find_package(GSL       REQUIRED)
find_package(CCFits    REQUIRED)
find_package(FFTW      REQUIRED)
find_package(NFFT      REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Boost     COMPONENTS program_options REQUIRED)

# Include external projects
## NICAEA
ExternalProject_Add(NICAEA
        PREFIX NICAEA
        GIT_REPOSITORY ${NICAEA_URL}
	GIT_TAG f7850e5009d0341226d4c1f38b116f316d8a0516)
        set(NICAEA_LIBRARY_DIR ${CMAKE_BINARY_DIR}/NICAEA/src/NICAEA/lib)
        set(NICAEA_INCLUDE_DIR ${CMAKE_BINARY_DIR}/NICAEA/src/NICAEA/include)
        set(NICAEA_LIBRARIES -lnicaea)

## Sparse2D
ExternalProject_Add(SPARSE2D
         PREFIX SPARSE2D
         GIT_REPOSITORY ${SPARSE2D_URL}
         GIT_TAG v2.1.4
         CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/extern)
        set(SPARSE2D_LIBRARY_DIR ${CMAKE_BINARY_DIR}/extern/lib)
        set(SPARSE2D_INCLUDE_DIR ${CMAKE_BINARY_DIR}/extern/include)
        set(SPARSE2D_LIBRARIES -lsparse2d -ltools -lsparse1d)

include_directories(${SPARSE2D_INCLUDE_DIR}
		    ${GSL_INCLUDE_DIR}
		    ${CCFITS_INCLUDE_DIR}
		    ${CFITSIO_INCLUDE_DIR}
		    ${Boost_INCLUDE_DIR}
		    ${NICAEA_INCLUDE_DIR}
		    ${NFFT_INCLUDE_DIR}
		    ${FFTW_INCLUDE_DIR}
		    ${ARMADILLO_INCLUDE_DIRS})

link_directories(${SPARSE2D_LIBRARY_DIR}
		    ${GSL_LIBRARY_DIR}
		    ${CCFITS_LIBRARY_DIR}
		    ${CFITSIO_LIBRARY_DIR}
		    ${Boost_LIBRARY_DIR}
		    ${NICAEA_LIBRARY_DIR}
		    ${NFFT_LIBRARY_DIR}
		    ${FFTW_LIBRARY_DIR})

set(GLIMPSE_SRC src/survey.cpp
		src/redshift_distribution.cpp
		src/field.cpp
		src/surface_reconstruction.cpp
		src/density_reconstruction.cpp
		src/starlet_2d.cpp
		src/wavelet_transform.cpp
		src/gpu_utils.c)

# VERSIONING from git
execute_process(
  COMMAND git describe --tags
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
configure_file(
  ${CMAKE_SOURCE_DIR}/src/version.h.in
  ${CMAKE_SOURCE_DIR}/src/version.h
)

find_package(CUDA)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}  -O3")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  find_package(OpenMP REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fpermissive -DMACOS -Xpreprocessor -fopenmp -lomp")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -fpermissive -DMACOS -DDEBUG_FITS -Xpreprocessor -fopenmp -lomp")
  get_filename_component(OpenMP_CXX_LIBRARY_DIR ${OpenMP_CXX_LIBRARIES} DIRECTORY)
  link_directories(${OpenMP_CXX_LIBRARY_DIR})
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fpermissive -fopenmp -std=c++11")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -fpermissive -fopenmp -std=c++11 -DDEBUG_FITS")
endif()

if(${CUDA_FOUND})
    message("Compiling CUDA accelerated reconstruction code, with 3D support")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_ACC")
    set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_20,code=sm_20
    )

    cuda_add_executable(glimpse src/glimpse.cpp ${GLIMPSE_SRC} src/spg.cu src/spg.cpp)
    cuda_add_cufft_to_target(glimpse)
else(${CUDA_FOUND})
    message("Compiling without CUDA acceleration")
    add_executable(glimpse src/glimpse.cpp ${GLIMPSE_SRC})
endif(${CUDA_FOUND})

add_dependencies(glimpse NICAEA SPARSE2D)

target_link_libraries(glimpse ${NICAEA_LIBRARIES}
                              ${SPARSE2D_LIBRARIES}
                              ${Boost_LIBRARIES}
                              ${GSL_LIBRARIES}
                              ${NFFT_LIBRARIES}
                              ${FFTW_LIBRARIES}
                              ${CCFITS_LIBRARY}
                              ${CFITSIO_LIBRARY}
                              ${ARMADILLO_LIBRARIES})
