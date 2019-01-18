/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2008-2017 Imperial College London
 * Copyright 2018-2019 King's College London
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0  
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mirtk/ReconstructionCardiacVelocity4D.h"
#include "mirtk/Resampling.h"
#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageTransformation.h"
#include "mirtk/LinearInterpolateImageFunction.hxx"
#include <math.h>




using namespace std;

namespace mirtk {


// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
ReconstructionCardiacVelocity4D::ReconstructionCardiacVelocity4D():ReconstructionCardiac4D()
{
    _recon_type = _3D;
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
ReconstructionCardiacVelocity4D::~ReconstructionCardiacVelocity4D() { }



// -----------------------------------------------------------------------------
// Parallel Simulate Slices - specific for velocity
// -----------------------------------------------------------------------------
class ParallelSimulateSlicesCardiac4DVelocity {
    irtkReconstructionCardiac4DVelocity *reconstructor;

public:
    ParallelSimulateSlicesCardiac4DVelocity( irtkReconstructionCardiac4DVelocity *_reconstructor ) :
    reconstructor(_reconstructor) { }

    void operator() (const blocked_range<size_t> &r) const {

        for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
            //Calculate simulated slice
            reconstructor->_simulated_slices[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
            reconstructor->_simulated_slices[inputIndex] = 0;

            reconstructor->_simulated_weights[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
            reconstructor->_simulated_weights[inputIndex] = 0;

            reconstructor->_simulated_inside[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
            reconstructor->_simulated_inside[inputIndex] = 0;

            reconstructor->_slice_inside[inputIndex] = false;

            POINT3D p;
            for ( int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++ )
                for ( int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++ )
                    if ( reconstructor->_slices[inputIndex](i, j, 0) != -1 ) {
                        double weight = 0;
                        int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                        for ( int k = 0; k < n; k++ ) {
                            p = reconstructor->_volcoeffs[inputIndex][i][j][k]; // PSF


                            // TAR --- grad_moments for current slice
                            int dirIndex = reconstructor->_stack_index[inputIndex]+1;
                            vector<vector<double> > _grad_moment_dirs;
                            // cout<<"Debug: size of _grad_moment_dirs ... "<<_grad_moment_dirs.size()<<endl;
                            // exit(1);

                            // double grad_moments_x=reconstructor->_directions[0][dirIndex];
                            // double grad_moments_y=reconstructor->_directions[1][dirIndex];
                            // double grad_moments_z=reconstructor->_directions[2][dirIndex];
                            // reconstructor->RotateDirections(gx,gy,gz,inputIndex);
                            // double bval=reconstructor->_bvalues[dirIndex];
                            //
                      	    // irtkMatrix grad_moments_matrix(1,3);
                      	    // grad_moments_matrix(0,0)=grad_moments_x;
                      	    // grad_moments_matrix(0,1)=grad_moments_y;
                      	    // grad_moments_matrix(0,2)=grad_moments_z;
                      	    // double slice_grad_moments;
                            // slice_grad_moments = 0;

                            // THINK I NEED TO BRING grad_moments IN HERE --- see irtkReconstructionDTI.cc for equivalent

                            // double sim_signal;


                            // for(int g = 0; i < 3; i++) {
                                for ( int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++ ) {

                                  // _reconstructed4DVelocity --- needs to be <vector> of length 3, for loop over g
                                    // reconstructor->_simulated_slices[inputIndex](i, j, 0) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_reconstructed4DVelocity(p.x, p.y, p.z, outputIndex);

                                    // sim_signal = 0;

                                    // for(unsigned int l = 0; l < basis.Cols(); l++ )
			                        //     sim_signal += reconstructor->_G_coeffs(p.x, p.y, p.z,l)*basis(0,l);

                                    // reconstructor->_simulated_slices[inputIndex](i, j, 0) += p.value * sim_signal;


                                    reconstructor->_simulated_slices[inputIndex](i, j, 0) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_reconstructed4D(p.x, p.y, p.z, outputIndex);
                                    weight += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                }
                                if (reconstructor->_mask(p.x, p.y, p.z) == 1) {
                                    reconstructor->_simulated_inside[inputIndex](i, j, 0) = 1;
                                    reconstructor->_slice_inside[inputIndex] = true;
                                }
                            // }
                        }
                        if( weight > 0 ) {
                            reconstructor->_simulated_slices[inputIndex](i,j,0) /= weight;
                            reconstructor->_simulated_weights[inputIndex](i,j,0) = weight;
                        }
                    }

        }
    }

    // execute
    void operator() () const {

        task_scheduler_init init(tbb_no_threads);
        parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                      *this );
        init.terminate();
    }

};



// -----------------------------------------------------------------------------
// TAR --- thick I need to change this to be gradient moments equivalent
// -----------------------------------------------------------------------------
void InitGradientMoments( Array< Array<double> > gm_dirs, Array<double> gm_vals )
{

    Array<Array<double> > _grad_moment_dirs;
    Array<double> _grad_moment_vals;

    _grad_moment_dirs = gm_dirs;
    _grad_moment_vals = gm_vals;

    // cout<<"Debug: size of _grad_moment_dirs ... "<<_grad_moment_dirs.size()<<endl;

}


// // Taken from irtkReconstructionDTI.cc
// // Possibly need to adapt this for gradient_moments:
// void irtkReconstructionDTI::CreateSliceGradientMoments(vector< vector<double> >& directions, vector<double>& bvalues)
// {
//   _directions = directions;
//   _bvalues = bvalues;
//
//   cout<<"B-values: ";
//   for(uint i=0;i<_bvalues.size(); i++)
//     cout<<_bvalues[i]<<" ";
//   cout<<endl;
//
//   cout<<"B-vectors: ";
//   for(uint j=0; j<_directions.size(); j++)
//   {
//     for(uint i=0;i<_directions[j].size(); i++)
//       cout<<_directions[j][i]<<" ";
//     cout<<endl;
//   }
//   cout<<endl;
// }







// -----------------------------------------------------------------------------
// ...
// -----------------------------------------------------------------------------
void SimulateSlicesCardiacVelocity4D()
{
    if (_debug)
      cout<<"Simulating Slices..."<<endl;


    // NOT SURE ABOUT BELOW --- MIGHT NEED TO ADJUST?

    ParallelSimulateSlicesCardiac4DVelocity parallelSimulateSlices( this );
    parallelSimulateSlices();

     if (_debug)
      cout<<"\t...Simulating Slices done."<<endl;

}


//------------------------------------------------------------------- 


    
} // namespace mirtk

