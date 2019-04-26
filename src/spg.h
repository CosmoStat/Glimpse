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

#ifndef SPG_H
#define SPG_H
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_timer.h"
#include "gpu_utils.h"

class spg
{
    int npix;
    int nz;
    int nframes;
    int nGPU;
    int whichGPUs[MAX_GPUS];
    
    // Stride between coefficients proccessed by different GPUs
    unsigned long * coeff_stride;
    unsigned long * coeff_stride_pos;
        
    // Device pointer arrays, storing  pointers for each device
    float ** d_x;
    float ** d_u;
    float ** d_u_pos;
    float ** d_w;
    
    float *p;
    float *pp;
    
    StopWatchInterface *timer;
    
public:
    
    /* Initialise cuda accelerated SPG algorithm for evaluating simple
     * proximal operators.
     * 
     */
    spg( int npix, int nz, int nframes, const double *P, const float *l1_weights );
    
    /* Destructor.
     * 
     */
    ~spg();
    
    /* Compute the proximity operator of the sparsity constraint.
     * 
     */
    void prox_l1(float *alpha, int niter=10000);


    /* Compute the proximity operator of the positivity constraint.
     *
     */
    void prox_pos(float *delta, int niter=10000);
    
    /* Updates the l1 thresholds
     * 
     */
    void update_weights(float *l1_weights);
    
};


#endif
