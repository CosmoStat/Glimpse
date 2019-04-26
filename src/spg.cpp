/*! Copyright CEA, 2015-2016
 * author : Francois Lanusse < francois.lanusse@gmail.com >
 *
 * This software is a computer program whose purpose is to reconstruct mass maps
 * from weak gravitational lensing.
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 */

#include <sparse2d/IM_IO.h>
#include <iostream>
#include "spg.h"
#include "spg.cuh"

spg::spg ( int npix, int nz, int nframes, const double *P, const float *l1_weights ) :
    npix ( npix ), nz ( nz ), nframes ( nframes )
{
    // Look for the number of available GPUs
    getDeviceCount ( &nGPU );
    getGPUs ( whichGPUs );

    std::cout << "Running SPG algorithm on " << nGPU << " GPUs" << std::endl;

    // Create device pointer arrays to hold the memory space for each GPUs
    d_x     = ( float ** ) malloc ( sizeof ( float * ) * nGPU );
    d_u     = ( float ** ) malloc ( sizeof ( float * ) * nGPU );
    d_u_pos = ( float ** ) malloc ( sizeof ( float * ) * nGPU );
    d_w     = ( float ** ) malloc ( sizeof ( float * ) * nGPU );

    // Stride between coefficients processed by different GPUs
    coeff_stride = ( unsigned long * ) malloc ( sizeof ( unsigned long ) * nGPU );
    coeff_stride_pos = ( unsigned long * ) malloc ( sizeof ( unsigned long ) * nGPU );

    // Memory allocation for all GPUs
    for ( int i = 0; i < nGPU; i++ ) {

        // Set strided data for each GPU, leaving the last GPU to handle any extra coefficients
        coeff_stride[i] = npix * npix * nframes / nGPU;
        coeff_stride_pos[i] = npix * npix / nGPU;
        if ( i == ( nGPU - 1 ) ) {
            coeff_stride[i] += npix * npix * nframes % nGPU;
            coeff_stride_pos[i] += npix * npix % nGPU;
        }

        // Select GPU
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );


        // Allocate arrays
        checkCudaErrors ( cudaMalloc ( ( void ** ) &d_x[i],     sizeof ( float ) * coeff_stride[i] * nz ) );
        checkCudaErrors ( cudaMalloc ( ( void ** ) &d_u[i],     sizeof ( float ) * coeff_stride[i] * nz ) );
        checkCudaErrors ( cudaMalloc ( ( void ** ) &d_u_pos[i],     sizeof ( float ) * coeff_stride_pos[i] * nz ) );
        checkCudaErrors ( cudaMalloc ( ( void ** ) &d_w[i],     sizeof ( float ) * coeff_stride[i] * nz ) );

        // Initialise these arrays to zero
        checkCudaErrors ( cudaMemset ( d_x[i],    0, sizeof ( float ) * coeff_stride[i] * nz ) );
        checkCudaErrors ( cudaMemset ( d_u[i],    0, sizeof ( float ) * coeff_stride[i] * nz ) );
        checkCudaErrors ( cudaMemset ( d_u_pos[i],    0, sizeof ( float ) * coeff_stride_pos[i] * nz ) );
        checkCudaErrors ( cudaMemset ( d_w[i],    0, sizeof ( float ) * coeff_stride[i] * nz ) );

        // Store the l1_weights for each GPU    
        checkCudaErrors (cudaMemcpy2DAsync(d_w[i], coeff_stride[i]*sizeof ( float ), &l1_weights[i * coeff_stride[0]], npix * npix * nframes * sizeof ( float ), coeff_stride[i]*sizeof ( float ), nz, cudaMemcpyHostToDevice ) );


    }

    // Compute and store the preconditioning matrix
    pp = ( float * ) malloc ( sizeof ( float ) * nz * nz );
    p = ( float * ) malloc ( sizeof ( float ) * nz * nz );
    for ( int z1 = 0; z1 < nz; z1++ )
        for ( int z2 = 0; z2 < nz; z2++ ) {
            p[z1 * nz + z2] = P[z1 * nz + z2];
        }
    for ( int z1 = 0; z1 < nz; z1++ ) {
        for ( int z2 = 0; z2 < nz; z2++ ) {
            double toto = 0;
            for ( int z3 = 0; z3 < nz; z3++ ) {
                toto += P[z1 * nz + z3] * P[z3 * nz + z2];
            }
            pp[z1 * nz + z2] = toto;
        }
    }

    // Memory allocation for all GPUs
    for ( int i = 0; i < nGPU; i++ ) {
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
        spg_store_matrix ( nz, p, pp );
    }

    timer = NULL;
    sdkCreateTimer ( &timer );
    sdkResetTimer ( &timer );
}

spg::~spg()
{
    // Free allocated resources
    for ( int i = 0; i < nGPU; i++ ) {
        checkCudaErrors ( cudaFree ( d_x[i] ) );
        checkCudaErrors ( cudaFree ( d_u[i] ) );
        checkCudaErrors ( cudaFree ( d_u_pos[i] ) );
        checkCudaErrors ( cudaFree ( d_w[i] ) );

    }
    free ( d_x );
    free ( d_u_pos );
    free ( d_u );
    free ( d_w );
    free ( coeff_stride );
    free ( coeff_stride_pos );
    free ( p );
    free ( pp );
}

void spg::prox_pos ( float *delta, int niter )
{
    sdkResetTimer ( &timer );
    sdkStartTimer ( &timer );
    // Copy data
    for ( int i = 0; i < nGPU; i++ ) {
        // Select GPU
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );

        // Copy wavelet coefficients to device
        checkCudaErrors ( cudaMemcpy2DAsync ( d_x[i], coeff_stride_pos[i]*sizeof ( float ), &delta[i * coeff_stride_pos[0]], npix * npix * sizeof ( float ), coeff_stride_pos[i]*sizeof ( float ), nz, cudaMemcpyHostToDevice ) );
    }

    for ( int i = 0; i < nGPU; i++ ) {
        // Select GPU
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );

        // Run the spg algorithm for a given number of iterations
        spg_pos ( coeff_stride_pos[i], nz, d_u_pos[i], d_x[i], niter );
    }

    // Wait for all GPUs to be done
    for ( int i = 0; i < nGPU; i++ ) {
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );

        // Recover wavelet coefficients from device
        checkCudaErrors ( cudaMemcpy2DAsync ( &delta[i * coeff_stride_pos[0]], npix * npix * sizeof ( float ), d_x[i], coeff_stride_pos[i]*sizeof ( float ), coeff_stride_pos[i]*sizeof ( float ), nz, cudaMemcpyDeviceToHost ) );
    }

    for ( int i = 0; i < nGPU; i++ ) {
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );

        checkCudaErrors ( cudaDeviceSynchronize() );
        checkCudaErrors ( cudaPeekAtLastError() );
    }
    sdkStopTimer ( &timer );
    std::cout << "Time spent for solving positivity spg " <<  sdkGetTimerValue ( &timer ) << std::endl;
}

void spg::prox_l1 ( float *alpha, int niter )
{
    sdkResetTimer ( &timer );
    sdkStartTimer ( &timer );
    // Copy data
    for ( int i = 0; i < nGPU; i++ ) {
        // Select GPU
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );

        // Copy wavelet coefficients to device
        checkCudaErrors ( cudaMemcpy2DAsync ( d_x[i], coeff_stride[i]*sizeof ( float ), &alpha[i * coeff_stride[0]], npix * npix * nframes * sizeof ( float ), coeff_stride[i]*sizeof ( float ), nz, cudaMemcpyHostToDevice ) );
        
    }

    for ( int i = 0; i < nGPU; i++ ) {
        // Select GPU
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );

        // Run the spg algorithm for a given number of iterations
        spg_l1(coeff_stride[i], nz, d_u[i], d_x[i], d_w[i], niter);
    }

    // Wait for all GPUs to be done
    for ( int i = 0; i < nGPU; i++ ) {
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );

        // Recover wavelet coefficients from device
        checkCudaErrors ( cudaMemcpy2DAsync ( &alpha[i * coeff_stride[0]], npix * npix * nframes * sizeof ( float ), d_x[i], coeff_stride[i]*sizeof ( float ), coeff_stride[i] * sizeof ( float ), nz, cudaMemcpyDeviceToHost ) );
    }

    for ( int i = 0; i < nGPU; i++ ) {
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );

        checkCudaErrors ( cudaDeviceSynchronize() );
        checkCudaErrors ( cudaPeekAtLastError() );
    }
    sdkStopTimer ( &timer );
    std::cout << "Time spent for solving l1 spg " <<  sdkGetTimerValue ( &timer ) << std::endl;
}

void spg::update_weights ( float *l1_weights )
{
    // Memory allocation for all GPUs
    for ( int i = 0; i < nGPU; i++ ) {
        // Store the l1_weights for each GPU       
        checkCudaErrors (cudaMemcpy2DAsync(d_w[i], coeff_stride[i]*sizeof ( float ), &l1_weights[i * coeff_stride[0]], npix * npix * nframes * sizeof ( float ), coeff_stride[i]*sizeof ( float ), nz, cudaMemcpyHostToDevice ) );
    }

    // Wait for all GPUs to be done
    for ( int i = 0; i < nGPU; i++ ) {
        checkCudaErrors ( cudaSetDevice ( whichGPUs[i] ) );
        checkCudaErrors ( cudaDeviceSynchronize() );
    }
}
