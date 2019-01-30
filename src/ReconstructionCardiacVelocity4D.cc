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


// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
ReconstructionCardiacVelocity4D::ReconstructionCardiacVelocity4D():ReconstructionCardiac4D()
{
    _recon_type = _3D;

    // initialise velocity direction array
    Array<double> tmp;
    double t;  

    _velocity_scale = 1000;

    _v_directions.clear();
    for (int i=0; i<3; i++) {
        tmp.clear();
        for (int j=0; j<3; j++) {
            if (i==j)
                t = 1;
            else 
                t = 0;
            tmp.push_back(t);
        }
        _v_directions.push_back(tmp);
    }

    
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
ReconstructionCardiacVelocity4D::~ReconstructionCardiacVelocity4D() { }



//------------------------------------------------------------------- 

void ReconstructionCardiacVelocity4D::InitializeVelocityVolumes()
{

    _reconstructed4D = GetReconstructedCardiac4D();

    _reconstructed5DVelocity.clear();
    for ( int i=0; i<_v_directions.size(); i++ )
        _reconstructed5DVelocity.push_back(_reconstructed4D);

    _confidence_maps_velocity.clear();
    for ( int i=0; i<_v_directions.size(); i++ )
        _confidence_maps_velocity.push_back(_reconstructed4D);

}


//------------------------------------------------------------------- 

void ReconstructionCardiacVelocity4D::GaussianReconstructionCardiacVelocity4D()
{
    if(_debug)
        {
            cout << "Gaussian reconstruction ... " << endl;
            cout << "\tinput slice:  ";
            cout.flush();
        }
        unsigned int inputIndex, outputIndex;
        int k, n;
        RealImage slice;
        double scale;
        POINT3D p;
        Array<int> voxel_num;
        int slice_vox_num;
        
        int gradientIndex, velocityIndex;
        double gval, gx, gy, gz, dx, dy, dz, dotp, v_component; 
        double sigma=0.2, w, tw;

        // clear _reconstructed image
        _reconstructed4D = 0;
        
 

        Array<RealImage> recon5D;

        for ( int i=0; i<_v_directions.size(); i++ )
            recon5D.push_back(_reconstructed4D);

        Array<RealImage> weights = recon5D;


        for ( inputIndex = 0; inputIndex < _slices.size(); ++inputIndex ) {
            
            if (_slice_excluded[inputIndex]==0) {
                
                if(_debug)
                {
                    cout << inputIndex << " - ";
                    cout.flush();
                }
                // copy the current slice
                slice = _slices[inputIndex];
                // alias the current bias image
                RealImage& b = _bias[inputIndex];
                //read current scale factor
                scale = _scale[inputIndex];
                
                slice_vox_num = 0;

                // gradient direction for current slice
                gradientIndex = _stack_index[inputIndex];
                gx = _g_directions[gradientIndex][0];
                gy = _g_directions[gradientIndex][1];
                gz = _g_directions[gradientIndex][2];

                RotateDirections(gx, gy, gz, inputIndex);

                gval = _g_values[gradientIndex];


                for ( velocityIndex=0; velocityIndex<_v_directions.size(); velocityIndex++ ) {

                    // velocity direction vector
                    dx = _v_directions[velocityIndex][0];
                    dy = _v_directions[velocityIndex][1];
                    dz = _v_directions[velocityIndex][2];

                                        
                    // ???? double check - whether it is correct for V = PHASE / (GAMMA*G)
                    dotp = (dx*gx+dy*gy+dz*gz)/sqrt((dx*dx+dy*dy+dz*dz)*(gx*gx+gy*gy+gz*gz));

                    // tw = (fabs(dotp)-1)/sigma;
	                // w=exp(-tw*tw)/(6.28*sigma);

                    v_component = dotp /( gval * gamma);

                    cout << " g: [" << gx << " " << gy << " " << gz << "] / ";
                    cout << " v: [" << dx << " " << dy << " " << dz << "] / "; 
                    cout << " dotp: " << dotp << " / " << " v_comp: " << v_component << endl;

                    // distribute slice intensities to the volume
                    for ( int i = 0; i < slice.GetX(); i++ ) {
                        for ( int j = 0; j < slice.GetY(); j++ ) {

                            if (slice(i, j, 0) != -1) {
                                // biascorrect and scale the slice
                                slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                                
                                // number of volume voxels with non-zero coefficients for current slice voxel
                                n = _volcoeffs[inputIndex][i][j].size();
                                
                                // if given voxel is not present in reconstructed volume at all, pad it
                                
                                // if (n == 0)
                                // _slices[inputIndex].PutAsDouble(i, j, 0, -1);
                                // calculate num of vox in a slice that have overlap with roi
                                if (n>0)
                                    slice_vox_num++;
                                
                                // add contribution of current slice voxel to all voxel volumes to which it contributes
                                for ( k = 0; k < n; k++ ) {

                                    p = _volcoeffs[inputIndex][i][j][k];

                                    for ( outputIndex=0; outputIndex<_reconstructed_cardiac_phases.size(); outputIndex++ )  {

                                        // _reconstructed4D(p.x, p.y, p.z, outputIndex) += _slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0);
                                        recon5D[velocityIndex](p.x, p.y, p.z, outputIndex) += v_component * slice(i, j, 0) * _slice_temporal_weight[outputIndex][inputIndex] * p.value;

                                        // do we need to multiply by v_component? or dotp?
                                        weights[velocityIndex](p.x, p.y, p.z, outputIndex) += _slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                            
                                    } // end of loop for cardiac phases 
                                }
                            }
                        }
                    } 

                } // end of loop for velocity directions
                
                voxel_num.push_back(slice_vox_num);
            
            } // end of if (_slice_excluded[inputIndex]==0)
            
        } // end of loop for a slice inputIndex
        
        // normalize the volume by proportion of contributing slice voxels
        // for each volume voxe

        for (int i=0; i<_v_directions.size(); i++) {
            recon5D[i] /= weights[i];
        }

        _reconstructed5DVelocity = recon5D; 

        char buffer[256];

        if (_debug) {

            for (int i=0; i<_v_directions.size(); i++) {
        
                recon5D[i] *= 1000;
                sprintf(buffer,"recon4Dgaussian-velocity-%i.nii.gz", i);
                recon5D[i].Write(buffer);

                sprintf(buffer,"weights-velocity-%i.nii.gz", i);
                weights[i].Write(buffer);

            }

        }

        // exit(1);

        if(_debug) {
            cout << inputIndex << "\b\b." << endl;
            cout << "... Gaussian reconstruction done." << endl << endl;
            cout.flush();
        }
        
        // if (_debug)
        //     _reconstructed4D.Write("init.nii.gz");
        
        // now find slices with small overlap with ROI and exclude them.
        
        Array<int> voxel_num_tmp;
        for (unsigned int i=0; i<voxel_num.size(); i++)
            voxel_num_tmp.push_back(voxel_num[i]);
        
        // find median
        sort(voxel_num_tmp.begin(),voxel_num_tmp.end());
        int median = voxel_num_tmp[round(voxel_num_tmp.size()*0.5)];
        
        // remember slices with small overlap with ROI
        _small_slices.clear();
        for (unsigned int i=0; i<voxel_num.size(); i++)
            if (voxel_num[i]<0.1*median)
                _small_slices.push_back(i);
        
        if (_debug) {
            cout<<"Small slices:";
            for (unsigned int i=0; i<_small_slices.size(); i++)
            cout<<" "<<_small_slices[i];
            cout<<endl;
        }


}

//------------------------------------------------------------------- 


class ParallelGaussianReconstructionCardiacVelocity4DxT {
public:
        ReconstructionCardiacVelocity4D *reconstructor;


        ParallelGaussianReconstructionCardiacVelocity4DxT(ReconstructionCardiacVelocity4D *_reconstructor) :
        reconstructor(_reconstructor) { }

        
        void operator() (const blocked_range<size_t> &r) const {
            

            for ( size_t outputIndex = r.begin(); outputIndex != r.end(); ++outputIndex ) {

                char buffer[256];

                unsigned int inputIndex;
                int k, n;
                RealImage slice;
                double scale;
                POINT3D p;
                
                int gradientIndex, velocityIndex;
                double gval, gx, gy, gz, dx, dy, dz, dotp, v_component; 

                // volume for reconstruction
                Array<RealImage> reconstructed3DVelocityArray;
                reconstructed3DVelocityArray = reconstructor->_globalReconstructed4DVelocityArray[outputIndex];

                
                RealImage weights = reconstructed3DVelocityArray[0];

                
                for ( inputIndex = 0; inputIndex < reconstructor->_slices.size(); ++inputIndex ) {
                    
                    if (reconstructor->_slice_excluded[inputIndex]==0) {
                        
                        if(reconstructor->_debug)
                        {
                            cout << inputIndex << " , ";
                            cout.flush();
                        }

                        // copy the current slice
                        slice = reconstructor->_slices[inputIndex];
                        // alias the current bias image
                        RealImage& b = reconstructor->_bias[inputIndex];
                        //read current scale factor
                        scale = reconstructor->_scale[inputIndex];

                        // gradient direction for current slice
                        gradientIndex = reconstructor->_stack_index[inputIndex];
                        gx = reconstructor->_g_directions[gradientIndex][0];
                        gy = reconstructor->_g_directions[gradientIndex][1];
                        gz = reconstructor->_g_directions[gradientIndex][2];

                        reconstructor->RotateDirections(gx, gy, gz, inputIndex);

                        gval = reconstructor->_g_values[gradientIndex];

                        for ( velocityIndex = 0; velocityIndex < reconstructor->_v_directions.size(); velocityIndex++ ) {

                            // velocity direction vector
                            dx = reconstructor->_v_directions[velocityIndex][0];
                            dy = reconstructor->_v_directions[velocityIndex][1];
                            dz = reconstructor->_v_directions[velocityIndex][2];
                                                
                            // ???? double check - whether it is correct for V = PHASE / (GAMMA*G)
                            dotp = (dx*gx+dy*gy+dz*gz)/sqrt((dx*dx+dy*dy+dz*dz)*(gx*gx+gy*gy+gz*gz));

                            v_component = dotp /( gval * reconstructor->gamma);


                            // distribute slice intensities to the volume
                            for ( int i = 0; i < slice.GetX(); i++ ) {
                                for ( int j = 0; j < slice.GetY(); j++ ) {

                                    if (slice(i, j, 0) != -1) {
                                        // biascorrect and scale the slice
                                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                                        
                                        // number of volume voxels with non-zero coefficients for current slice voxel
                                        n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                                                                                
                                        // add contribution of current slice voxel to all voxel volumes to which it contributes
                                        for ( k = 0; k < n; k++ ) {

                                            p = reconstructor->_volcoeffs[inputIndex][i][j][k];

                                            // for ( outputIndex=0; outputIndex<reconstructor->_reconstructed_cardiac_phases.size(); outputIndex++ )  {

                                                // _reconstructed4D(p.x, p.y, p.z, outputIndex) += _slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0);
                                                // reconstructed4DVelocity(p.x, p.y, p.z, outputIndex) += v_component * slice(i, j, 0) * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;

                                                reconstructed3DVelocityArray[velocityIndex](p.x, p.y, p.z) += v_component * slice(i, j, 0) * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;

                                                // do we need to multiply by v_component? or dotp?
                                                // weights(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                                weights(p.x, p.y, p.z) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                                    
                                            // } // end of loop for cardiac phases 
                                        }
                                    }
                                }
                            } 
                        } // end of velocity vector loop
                    
                    } // end of if (_slice_excluded[inputIndex]==0)
                    
                } // end of loop for a slice inputIndex
                
                // normalize the volume by proportion of contributing slice voxels
                // for each volume voxel

                for ( velocityIndex = 0; velocityIndex < reconstructor->_v_directions.size(); velocityIndex++ ) {
                    reconstructed3DVelocityArray[velocityIndex] /= weights;

                    reconstructor->_globalReconstructed4DVelocityArray[outputIndex][velocityIndex] = reconstructed3DVelocityArray[velocityIndex];
                }


            } // parallel cardiac phase loop

        }


        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed_cardiac_phases.size()), *this );
            //init.terminate();
        }
        

};



void ReconstructionCardiacVelocity4D::GaussianReconstructionCardiacVelocity4DxT()
{
 
    RealImage reconstructed3DVelocity = _reconstructed4D.GetRegion(0,0,0,0,_reconstructed4D.GetX(),_reconstructed4D.GetY(),_reconstructed4D.GetZ(),1);
    reconstructed3DVelocity = 0;

    Array<RealImage> reconstructed3DVelocityArray;
    for (int i=0; i<_v_directions.size(); i++)
        reconstructed3DVelocityArray.push_back(reconstructed3DVelocity);

    _globalReconstructed4DVelocityArray.clear();
    for (int i=0; i<_reconstructed_cardiac_phases.size(); i++)
        _globalReconstructed4DVelocityArray.push_back(reconstructed3DVelocityArray);


    if(_debug) {
        cout << "- Gaussian reconstruction : " << endl;
    }

    ParallelGaussianReconstructionCardiacVelocity4DxT *gr = new ParallelGaussianReconstructionCardiacVelocity4DxT(this);
    (*gr)();

    int X, Y, Z, T, V;
    X = _reconstructed4D.GetX();
    Y = _reconstructed4D.GetY();
    Z = _reconstructed4D.GetZ();
    T = _reconstructed4D.GetT();
    V = _reconstructed5DVelocity.size();

    RealImage tmp3D, tmp4D;
    
    tmp4D = _reconstructed4D; 
    tmp4D = 0;
    
    for (int v=0; v<V ; v++) {

        tmp4D = 0;
        for (int t=0; t<T ; t++) {

            tmp3D = _globalReconstructed4DVelocityArray[t][v];
            for (int z=0; z<Z ; z++) {
                for (int y=0; y<Y ; y++) {
                    for (int x=0; x<X ; x++) {
                        tmp4D(x,y,z,t) = tmp3D(x,y,z);
                    }
                }
            }
        }

        _reconstructed5DVelocity[v] = tmp4D;
    }

    delete gr;
    _globalReconstructed4DVelocityArray.clear();

    char buffer[256];

    if (_debug) {

        for (int i=0; i<_v_directions.size(); i++) {
        
            RealImage scaled = _reconstructed5DVelocity[i];
            scaled *= _velocity_scale;
            sprintf(buffer,"recon4D-gaussian-velocity-%i.nii.gz", i);
            scaled.Write(buffer);

            // sprintf(buffer,"weights-velocity-%i.nii.gz", i);
            // weights[i].Write(buffer);

        }

    }

}


//------------------------------------------------------------------- 


class ParallelGaussianReconstructionCardiac4DxT {
public:
        ReconstructionCardiacVelocity4D *reconstructor;


        ParallelGaussianReconstructionCardiac4DxT(ReconstructionCardiacVelocity4D *_reconstructor) :
        reconstructor(_reconstructor) { }

        
        void operator() (const blocked_range<size_t> &r) const {
            

            for ( size_t outputIndex = r.begin(); outputIndex != r.end(); ++outputIndex ) {

                char buffer[256];
                
                unsigned int inputIndex;
                int k, n;
                RealImage slice;
                double scale;
                POINT3D p;

                // volume for reconstruction
                RealImage reconstructed3D;
                reconstructed3D = reconstructor->_globalReconstructed4DArray[outputIndex];
                
                RealImage weights = reconstructed3D;


                for ( inputIndex = 0; inputIndex < reconstructor->_slices.size(); ++inputIndex ) {
                    
                    if (reconstructor->_slice_excluded[inputIndex]==0) {
                        
                        if(reconstructor->_debug)
                        {
                            cout << inputIndex << " , ";
                            cout.flush();
                        }

                        // copy the current slice
                        slice = reconstructor->_slices[inputIndex];
                        // alias the current bias image
                        RealImage& b = reconstructor->_bias[inputIndex];
                        //read current scale factor
                        scale = reconstructor->_scale[inputIndex];

                            // distribute slice intensities to the volume
                            for ( int i = 0; i < slice.GetX(); i++ ) {
                                for ( int j = 0; j < slice.GetY(); j++ ) {

                                    if (slice(i, j, 0) != -1) {
                                        // biascorrect and scale the slice
                                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                                        
                                        // number of volume voxels with non-zero coefficients for current slice voxel
                                        n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                                                                                
                                        // add contribution of current slice voxel to all voxel volumes to which it contributes
                                        for ( k = 0; k < n; k++ ) {

                                            p = reconstructor->_volcoeffs[inputIndex][i][j][k];

                                            // for ( outputIndex=0; outputIndex<reconstructor->_reconstructed_cardiac_phases.size(); outputIndex++ )  {

                                                reconstructed3D(p.x, p.y, p.z) += slice(i, j, 0) * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;

                                                weights(p.x, p.y, p.z) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                                    
                                            // } // end of loop for cardiac phases 
                                        }
                                    }
                                }
                            } 
                        // } // end of velocity vector loop
                    
                    } // end of if (_slice_excluded[inputIndex]==0)
                    
                } // end of loop for a slice inputIndex
                
                // normalize the volume by proportion of contributing slice voxels
                // for each volume voxel

                reconstructed3D /= weights;

                reconstructor->_globalReconstructed4DArray[outputIndex] = reconstructed3D;

            } // parallel cardiac phase loop

        }


        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed_cardiac_phases.size()), *this );
            //init.terminate();
        }
        

};



void ReconstructionCardiacVelocity4D::GaussianReconstructionCardiac4DxT()
{
 
    RealImage reconstructed3D= _reconstructed4D.GetRegion(0,0,0,0,_reconstructed4D.GetX(),_reconstructed4D.GetY(),_reconstructed4D.GetZ(),1);
    reconstructed3D = 0;

    _globalReconstructed4DArray.clear();
    for (int i=0; i<_reconstructed_cardiac_phases.size(); i++)
        _globalReconstructed4DArray.push_back(reconstructed3D);


    if(_debug) {
        cout << "- Gaussian reconstruction : " << endl;
    }

    ParallelGaussianReconstructionCardiac4DxT *gr = new ParallelGaussianReconstructionCardiac4DxT(this);
    (*gr)();

    int X, Y, Z, T;
    X = _reconstructed4D.GetX();
    Y = _reconstructed4D.GetY();
    Z = _reconstructed4D.GetZ();
    T = _reconstructed4D.GetT();

    RealImage tmp3D, tmp4D;
    
    tmp4D = _reconstructed4D; 
    tmp4D = 0;
    
    for (int t=0; t<T ; t++) {

        tmp3D = _globalReconstructed4DArray[t];
        for (int z=0; z<Z ; z++) {
            for (int y=0; y<Y ; y++) {
                for (int x=0; x<X ; x++) {
                    tmp4D(x,y,z,t) = tmp3D(x,y,z);
                }
            }
        }
    }

    _reconstructed4D = tmp4D;

    delete gr;
    _globalReconstructed4DArray.clear();

    char buffer[256];

    if (_debug) {

        sprintf(buffer,"recon4D-gaussian-phase.nii.gz");
        _reconstructed4D.Write(buffer);

    }

}


//------------------------------------------------------------------- 


class ParallelGaussianReconstructionCardiacVelocity4D {
public:
        ReconstructionCardiacVelocity4D *reconstructor;
        
        ParallelGaussianReconstructionCardiacVelocity4D(ReconstructionCardiacVelocity4D *_reconstructor) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            

            for ( size_t velocityIndex = r.begin(); velocityIndex != r.end(); ++velocityIndex ) {

                char buffer[256];

                if(reconstructor->_debug) {
                    cout << "- Gaussian reconstruction : " << velocityIndex << endl;
                    cout.flush();
                }

                unsigned int inputIndex, outputIndex;
                int k, n;
                RealImage slice;
                double scale;
                POINT3D p;
                
                int gradientIndex;
                double gval, gx, gy, gz, dx, dy, dz, dotp, v_component; 

                // clear _reconstructed image
                RealImage reconstructed4DVelocity = reconstructor->_reconstructed4D;
                reconstructed4DVelocity = 0;

                RealImage weights = reconstructed4DVelocity;

                for ( inputIndex = 0; inputIndex < reconstructor->_slices.size(); ++inputIndex ) {
                    
                    if (reconstructor->_slice_excluded[inputIndex]==0) {
                        
                        // if(_debug)
                        // {
                        //     cout << inputIndex << " - ";
                        //     cout.flush();
                        // }

                        // copy the current slice
                        slice = reconstructor->_slices[inputIndex];
                        // alias the current bias image
                        RealImage& b = reconstructor->_bias[inputIndex];
                        //read current scale factor
                        scale = reconstructor->_scale[inputIndex];

                        // gradient direction for current slice
                        gradientIndex = reconstructor->_stack_index[inputIndex];
                        gx = reconstructor->_g_directions[gradientIndex][0];
                        gy = reconstructor->_g_directions[gradientIndex][1];
                        gz = reconstructor->_g_directions[gradientIndex][2];

                        reconstructor->RotateDirections(gx, gy, gz, inputIndex);

                        gval = reconstructor->_g_values[gradientIndex];


                        // for ( velocityIndex=0; velocityIndex<_v_directions.size(); velocityIndex++ ) {

                            // velocity direction vector
                            dx = reconstructor->_v_directions[velocityIndex][0];
                            dy = reconstructor->_v_directions[velocityIndex][1];
                            dz = reconstructor->_v_directions[velocityIndex][2];
                                                
                            // ???? double check - whether it is correct for V = PHASE / (GAMMA*G)
                            dotp = (dx*gx+dy*gy+dz*gz)/sqrt((dx*dx+dy*dy+dz*dz)*(gx*gx+gy*gy+gz*gz));

                            v_component = dotp /( gval * reconstructor->gamma);

                            // cout << " g: [" << gx << " " << gy << " " << gz << "] / ";
                            // cout << " v: [" << dx << " " << dy << " " << dz << "] / "; 
                            // cout << " dotp: " << dotp << " / " << " v_comp: " << v_component << endl;

                            // distribute slice intensities to the volume
                            for ( int i = 0; i < slice.GetX(); i++ ) {
                                for ( int j = 0; j < slice.GetY(); j++ ) {

                                    if (slice(i, j, 0) != -1) {
                                        // biascorrect and scale the slice
                                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                                        
                                        // number of volume voxels with non-zero coefficients for current slice voxel
                                        n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                                                                                
                                        // add contribution of current slice voxel to all voxel volumes to which it contributes
                                        for ( k = 0; k < n; k++ ) {

                                            p = reconstructor->_volcoeffs[inputIndex][i][j][k];

                                            for ( outputIndex=0; outputIndex<reconstructor->_reconstructed_cardiac_phases.size(); outputIndex++ )  {

                                                // _reconstructed4D(p.x, p.y, p.z, outputIndex) += _slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0);
                                                reconstructed4DVelocity(p.x, p.y, p.z, outputIndex) += v_component * slice(i, j, 0) * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;

                                                // do we need to multiply by v_component? or dotp?
                                                weights(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                                    
                                            } // end of loop for cardiac phases 
                                        }
                                    }
                                }
                            } 
                    
                    } // end of if (_slice_excluded[inputIndex]==0)
                    
                } // end of loop for a slice inputIndex
                
                // normalize the volume by proportion of contributing slice voxels
                // for each volume voxel
                reconstructed4DVelocity /= weights;

                reconstructor->_reconstructed5DVelocity[velocityIndex] = reconstructed4DVelocity; 

            }

        }

        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_v_directions.size() ), *this );
            //init.terminate();
        }
        

};

void ReconstructionCardiacVelocity4D::GaussianReconstructionCardiacVelocity4Dx3()
{

    ParallelGaussianReconstructionCardiacVelocity4D *gr = new ParallelGaussianReconstructionCardiacVelocity4D(this);
    (*gr)();
        
    delete gr;

    char buffer[256];

    if (_debug) {

        for (int i=0; i<_v_directions.size(); i++) {
        
            RealImage scaled = _reconstructed5DVelocity[i];
            scaled *= _velocity_scale;
            sprintf(buffer,"recon4D-gaussian-velocity-%i.nii.gz", i);
            scaled.Write(buffer);

            // sprintf(buffer,"weights-velocity-%i.nii.gz", i);
            // weights[i].Write(buffer);

        }

    }


}



//------------------------------------------------------------------- 

class ParallelSimulateSlicesCardiacVelocity4D {
        ReconstructionCardiacVelocity4D *reconstructor;
        
    public:
        ParallelSimulateSlicesCardiacVelocity4D( ReconstructionCardiacVelocity4D *_reconstructor ) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {

                // calculate simulated slice
                reconstructor->_simulated_slices[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
                reconstructor->_simulated_slices[inputIndex] = 0;
                
                reconstructor->_simulated_weights[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
                reconstructor->_simulated_weights[inputIndex] = 0;
                
                reconstructor->_simulated_inside[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
                reconstructor->_simulated_inside[inputIndex] = 0;
                
                reconstructor->_slice_inside[inputIndex] = false;

                // gradient direction for current slice
                int gradientIndex = reconstructor->_stack_index[inputIndex];
                double gx = reconstructor->_g_directions[gradientIndex][0];
                double gy = reconstructor->_g_directions[gradientIndex][1];
                double gz = reconstructor->_g_directions[gradientIndex][2];
                reconstructor->RotateDirections(gx, gy, gz, inputIndex);
                double gval = reconstructor->_g_values[gradientIndex];

                Array<double> g_direction; 
                g_direction.push_back(gx);
                g_direction.push_back(gy);
                g_direction.push_back(gz);

                
                POINT3D p;
                for ( int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++ ) {
                    for ( int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++ ) {
                        if ( reconstructor->_slices[inputIndex](i, j, 0) != -1 ) {
                            double weight = 0;
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();

                            for ( int k = 0; k < n; k++ ) {

                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                for ( int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++ ) {

                                    // simulation of phase volume from velocity volumes
                                    double sim_signal = 0;

                                    for( int velocityIndex = 0; velocityIndex < reconstructor->_reconstructed5DVelocity.size(); velocityIndex++ ) {
                                        sim_signal += reconstructor->_reconstructed5DVelocity[velocityIndex](p.x, p.y, p.z, outputIndex)*g_direction[velocityIndex]*gval;
                                    }

                                    sim_signal = sim_signal * reconstructor->gamma; 

                                    // to be updated !!! (reconstract phase back from velocity - does not work now)
                                    // reconstructor->_simulated_slices[inputIndex](i, j, 0) += sim_signal * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value; // * reconstructor->_reconstructed4D(p.x, p.y, p.z, outputIndex);


                                    reconstructor->_simulated_slices[inputIndex](i, j, 0) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_reconstructed4D(p.x, p.y, p.z, outputIndex);


                                    // reconstructor->_simulated_slices[inputIndex](i, j, 0) += sim_signal[outputIndex] * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_reconstructed4D(p.x, p.y, p.z, outputIndex);
                                    weight += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;

                                }
                                if ( reconstructor->_mask(p.x, p.y, p.z) == 1 ) {
                                    reconstructor->_simulated_inside[inputIndex](i, j, 0) = 1;
                                    reconstructor->_slice_inside[inputIndex] = true;
                                }
                            }
                            if( weight > 0 ) {
                                reconstructor->_simulated_slices[inputIndex](i,j,0) /= weight;
                                reconstructor->_simulated_weights[inputIndex](i,j,0) = weight;
                            }
                        }
                    }
                }
                
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                         *this );
            //init.terminate();
        }
        
    };
    
//------------------------------------------------------------------- 

    void ReconstructionCardiacVelocity4D::SimulateSlicesCardiacVelocity4D()
    {
        if (_debug)
        cout<<"Simulating Slices..."<<endl;
        
        ParallelSimulateSlicesCardiacVelocity4D parallelSimulateSlices( this );
        parallelSimulateSlices();
        
        if (_debug)
        cout<<"\t...Simulating Slices done."<<endl;
    }

//------------------------------------------------------------------- 

class ParallelSuperresolutionCardiacVelocity4D {
        ReconstructionCardiacVelocity4D* reconstructor;
    public:
        Array<RealImage> confidence_maps;
        Array<RealImage> addons;

        RealImage addon_main;
        RealImage confidence_map_main;
        
        void operator()( const blocked_range<size_t>& r ) {

            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                //read the current weight image
                RealImage& w = reconstructor->_weights[inputIndex];
                
                //read the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];
                
                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];


                //direction for current slice
                int gradientIndex = reconstructor->_stack_index[inputIndex];
                
                double gx = reconstructor->_g_directions[gradientIndex][0];
                double gy = reconstructor->_g_directions[gradientIndex][1];
                double gz = reconstructor->_g_directions[gradientIndex][2];
                reconstructor->RotateDirections(gx,gy,gz,inputIndex);

                double gval = reconstructor->_g_values[gradientIndex];

                Array<double> g_direction; 
                g_direction.push_back(gx);
                g_direction.push_back(gy);
                g_direction.push_back(gz);
                
                //Update reconstructed volume using current slice

                for ( int velocityIndex = 0; velocityIndex < reconstructor->_v_directions.size(); velocityIndex++ ) {

                    double dx = reconstructor->_v_directions[velocityIndex][0];
                    double dy = reconstructor->_v_directions[velocityIndex][1];
                    double dz = reconstructor->_v_directions[velocityIndex][2];

                    // check whether it is a correct expression and gamma values
                    double dotp = (dx*gx+dy*gy+dz*gz)/sqrt((dx*dx+dy*dy+dz*dz)*(gx*gx+gy*gy+gz*gz));
                    double v_component = dotp / (reconstructor->gamma*gval);
                
                    //Distribute error to the volume
                    POINT3D p;
                    for ( int i = 0; i < slice.GetX(); i++ ) {
                        for ( int j = 0; j < slice.GetY(); j++ ) {

                            if (slice(i, j, 0) != -1) {
                                //bias correct and scale the slice
                                slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                                
                                if ( reconstructor->_simulated_slices[inputIndex](i,j,0) > 0 )
                                    slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                                else
                                    slice(i,j,0) = 0;
                                
                                int n = reconstructor->_volcoeffs[inputIndex][i][j].size();

                                for ( int k = 0; k < n; k++ ) {

                                    p = reconstructor->_volcoeffs[inputIndex][i][j][k];

                                    for ( int outputIndex=0; outputIndex<reconstructor->_reconstructed4D.GetT(); outputIndex++ ) {

                                        if(reconstructor->_robust_slices_only) {

                                            addons[velocityIndex](p.x, p.y, p.z, outputIndex) += v_component * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                            confidence_maps[velocityIndex](p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_slice_weight[inputIndex];

                                            addon_main(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                            confidence_map_main(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_slice_weight[inputIndex];

                                        }
                                        else {

                                            addons[velocityIndex](p.x, p.y, p.z, outputIndex) += v_component * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                            confidence_maps[velocityIndex](p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];

                                            addon_main(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                            confidence_map_main(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        }
                                    }
                                }
                            }
                        }
                    }
                } // end of loop for velocity directions 

            } //end of loop for a slice inputIndex
        
        }

        ParallelSuperresolutionCardiacVelocity4D( ParallelSuperresolutionCardiacVelocity4D& x, split ) :
        reconstructor(x.reconstructor)
        {
            // Clear addon
            RealImage addon = reconstructor->_reconstructed4D;
            addon = 0;

            addons.clear();
            for (int i=0; i<reconstructor->_reconstructed5DVelocity.size(); i++) {
                addons.push_back(addon);
            }
            
            // Clear confidence map
            RealImage confidence_map = reconstructor->_reconstructed4D;
            confidence_map = 0;
            
            confidence_maps.clear();
            for (int i=0; i<reconstructor->_reconstructed5DVelocity.size(); i++) {
                confidence_maps.push_back(confidence_map);
            }

            //Clear addon
            addon_main.Initialize( reconstructor->_reconstructed4D.GetImageAttributes() );
            addon_main = 0;
            
            //Clear confidence map
            confidence_map_main.Initialize( reconstructor->_reconstructed4D.GetImageAttributes() );
            confidence_map_main = 0;

        }
        
        void join( const ParallelSuperresolutionCardiacVelocity4D& y ) {

            for (int i=0; i<addons.size(); i++) {
                addons[i] += y.addons[i];
                confidence_maps[i] += y.confidence_maps[i];
            }

            addon_main += y.addon_main;
            confidence_map_main += y.confidence_map_main;

        }
        
        ParallelSuperresolutionCardiacVelocity4D( ReconstructionCardiacVelocity4D *reconstructor ) :
        reconstructor(reconstructor)
        {
            // Clear addon
            RealImage addon = reconstructor->_reconstructed4D;
            addon = 0;

            addons.clear();
            for (int i=0; i<reconstructor->_reconstructed5DVelocity.size(); i++) {
                addons.push_back(addon);
            }
            
            // Clear confidence map
            RealImage confidence_map = reconstructor->_reconstructed4D;
            confidence_map = 0;
            
            confidence_maps.clear();
            for (int i=0; i<reconstructor->_reconstructed5DVelocity.size(); i++) {
                confidence_maps.push_back(confidence_map);
            }

            //Clear addon
            addon_main.Initialize( reconstructor->_reconstructed4D.GetImageAttributes() );
            addon_main = 0;
            
            //Clear confidence map
            confidence_map_main.Initialize( reconstructor->_reconstructed4D.GetImageAttributes() );
            confidence_map_main = 0;

        }
        
        // execute
        void operator() () {
            //task_scheduler_init init(tbb_no_threads);
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()),
                            *this );
            //init.terminate();
        }
    };
    

//------------------------------------------------------------------- 


    void ReconstructionCardiacVelocity4D::SuperresolutionCardiacVelocity4D( int iter )
    {
        if (_debug)
        cout << "Superresolution " << iter << endl;
        
        char buffer[256];

        Array<RealImage> addons, originals;
        

        // Remember current reconstruction for edge-preserving smoothing
        originals = _reconstructed5DVelocity;
        
        ParallelSuperresolutionCardiacVelocity4D parallelSuperresolution(this);
        parallelSuperresolution();
        
        addons = parallelSuperresolution.addons;
        _confidence_maps_velocity = parallelSuperresolution.confidence_maps;


        RealImage original_main = _reconstructed4D;

        RealImage addon_main = parallelSuperresolution.addon_main;
        _confidence_map = parallelSuperresolution.confidence_map_main;
        
        if(_debug) {
            // char buffer[256];
            //sprintf(buffer,"confidence-map%i.nii.gz",iter);
            //_confidence_map.Write(buffer);

            for ( int i=0; i<_confidence_maps_velocity.size(); i++ ) {

                sprintf(buffer,"confidence-map-velocity-%i-%i.nii.gz",i,iter);
                _confidence_maps_velocity[i].Write(buffer);
            }
            //sprintf(buffer,"addon%i.nii.gz",iter);
            //addon.Write(buffer);
        }
        
        if (!_adaptive)
        for ( int v = 0; v < _v_directions.size(); v++ )
        for ( int x = 0; x < addons[v].GetX(); x++ )
        for ( int y = 0; y < addons[v].GetY(); y++ )
        for ( int z = 0; z < addons[v].GetZ(); z++ )
        for ( int t = 0; t < addons[v].GetT(); t++ )
        if (_confidence_map(x, y, z, t) > 0) {
            // ISSUES if _confidence_map(i, j, k, t) is too small leading to bright pixels
            addons[v](x, y, z, t) /= _confidence_maps_velocity[v](x, y, z, t);
            //this is to revert to normal (non-adaptive) regularisation
            _confidence_maps_velocity[v](x, y, z, t) = 1;

            addon_main(x, y, z, t) /= _confidence_map(x, y, z, t);
            //this is to revert to normal (non-adaptive) regularisation
            _confidence_map(x, y, z, t) = 1;
        }
        
        for ( int v = 0; v < _v_directions.size(); v++ ) 
            _reconstructed5DVelocity[v] += addons[v] * _alpha; //_average_volume_weight;
        

        _reconstructed4D += addon_main * _alpha; 


        // check for velocities 

        // 
        // //bound the intensities
        // for (i = 0; i < _reconstructed4D.GetX(); i++) {
        //     for (j = 0; j < _reconstructed4D.GetY(); j++) {
        //         for (k = 0; k < _reconstructed4D.GetZ(); k++) {
        //             for (t = 0; t < _reconstructed4D.GetT(); t++) {
        //                 if (_reconstructed4D(i, j, k, t) < _min_intensity * 0.9)
        //                 _reconstructed4D(i, j, k, t) = _min_intensity * 0.9;
        //                 if (_reconstructed4D(i, j, k, t) > _max_intensity * 1.1)
        //                 _reconstructed4D(i, j, k, t) = _max_intensity * 1.1;
        //             }
        //         }
        //     }
        // }
        // 
        
        //Smooth the reconstructed image
        AdaptiveRegularizationCardiacVelocity4D(iter, originals, original_main);
        
        //Remove the bias in the reconstructed volume compared to previous iteration
        //  TODO: update adaptive regularisation for 4d
        //  if (_global_bias_correction)
        //  BiasCorrectVolume(original);
        //  
    }
    


// -----------------------------------------------------------------------------
    // Parallel Adaptive Regularization Class 1: calculate smoothing factor, b
    // -----------------------------------------------------------------------------
    class ParallelAdaptiveRegularization1CardiacVelocity4D {
        ReconstructionCardiacVelocity4D *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;
        
    public:
        ParallelAdaptiveRegularization1CardiacVelocity4D( ReconstructionCardiacVelocity4D *_reconstructor,
                                                 Array<RealImage> &_b,
                                                 Array<double> &_factor,
                                                 RealImage &_original) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            int dx = reconstructor->_reconstructed4D.GetX();
            int dy = reconstructor->_reconstructed4D.GetY();
            int dz = reconstructor->_reconstructed4D.GetZ();
            int dt = reconstructor->_reconstructed4D.GetT();
            for ( size_t i = r.begin(); i != r.end(); ++i ) {
                //b[i] = reconstructor->_reconstructed;
                // b[i].Initialize( reconstructor->_reconstructed.GetImageAttributes() );
                
                int x, y, z, xx, yy, zz, t;
                double diff;
                for (x = 0; x < dx; x++)
                for (y = 0; y < dy; y++)
                for (z = 0; z < dz; z++) {
                    xx = x + reconstructor->_directions[i][0];
                    yy = y + reconstructor->_directions[i][1];
                    zz = z + reconstructor->_directions[i][2];
                    for (t = 0; t < dt; t++) {
                        if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                            && (reconstructor->_confidence_map(x, y, z, t) > 0) && (reconstructor->_confidence_map(xx, yy, zz, t) > 0)) {
                            diff = (original(xx, yy, zz, t) - original(x, y, z, t)) * sqrt(factor[i]) / reconstructor->_delta;
                            b[i](x, y, z, t) = factor[i] / sqrt(1 + diff * diff);
                            
                        }
                        else
                        b[i](x, y, z, t) = 0;
                    }
                }
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, 13),
                         *this );
            //init.terminate();
        }
        
    };
    
    
    // -----------------------------------------------------------------------------
    // Parallel Adaptive Regularization Class 2: compute regularisation update
    // -----------------------------------------------------------------------------
    class ParallelAdaptiveRegularization2CardiacVelocity4D {
        ReconstructionCardiacVelocity4D *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;
        
    public:
        ParallelAdaptiveRegularization2CardiacVelocity4D( ReconstructionCardiacVelocity4D *_reconstructor,
                                                 Array<RealImage> &_b,
                                                 Array<double> &_factor,
                                                 RealImage &_original) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            int dx = reconstructor->_reconstructed4D.GetX();
            int dy = reconstructor->_reconstructed4D.GetY();
            int dz = reconstructor->_reconstructed4D.GetZ();
            int dt = reconstructor->_reconstructed4D.GetT();
            for ( size_t x = r.begin(); x != r.end(); ++x ) {
                int xx, yy, zz;
                for (int y = 0; y < dy; y++)
                for (int z = 0; z < dz; z++)
                for (int t = 0; t < dt; t++) {
                    if(reconstructor->_confidence_map(x,y,z,t)>0)
                    {
                        double val = 0;
                        double sum = 0;
                        for (int i = 0; i < 13; i++)
                        {
                            xx = x + reconstructor->_directions[i][0];
                            yy = y + reconstructor->_directions[i][1];
                            zz = z + reconstructor->_directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                            if(reconstructor->_confidence_map(xx,yy,zz,t)>0)
                            {
                                val += b[i](x, y, z, t) * original(xx, yy, zz, t);
                                sum += b[i](x, y, z, t);
                            }
                        }
                        
                        for (int i = 0; i < 13; i++) {
                            xx = x - reconstructor->_directions[i][0];
                            yy = y - reconstructor->_directions[i][1];
                            zz = z - reconstructor->_directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                            if(reconstructor->_confidence_map(xx,yy,zz,t)>0)
                            {
                                val += b[i](x, y, z, t) * original(xx, yy, zz, t);
                                sum += b[i](x, y, z, t);
                            }
                        }
                        
                        val -= sum * original(x, y, z, t);
                        val = original(x, y, z, t)
                        + reconstructor->_alpha * reconstructor->_lambda / (reconstructor->_delta * reconstructor->_delta) * val;
                        reconstructor->_reconstructed4D(x, y, z, t) = val;
                    }
                }
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed4D.GetX()),
                         *this );
            //init.terminate();
        }
        
    };
    


//------------------------------------------------------------------- 
// Adaptive Regularization
//------------------------------------------------------------------- 

    void ReconstructionCardiacVelocity4D::AdaptiveRegularizationCardiacVelocity4D(int iter, Array<RealImage>& originals, RealImage& original_main)
    {
        if (_debug)
        cout << "AdaptiveRegularizationCardiacVelocity4D."<< endl;
        //cout << "AdaptiveRegularizationCardiac4D: _delta = "<<_delta<<" _lambda = "<<_lambda <<" _alpha = "<<_alpha<< endl;
        
        Array<double> factor(13,0);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
            factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }

        RealImage original1, original2;

        RealImage reconstructed4D_main, confidence_map_main;
        

        Array<RealImage> b;//(13);
        for (int i = 0; i < 13; i++)
            b.push_back( _reconstructed4D );
            

        original1 = original_main;
        ParallelAdaptiveRegularization1CardiacVelocity4D parallelAdaptiveRegularization1( this,
                                                                                    b,
                                                                                    factor,
                                                                                    original1 );
        parallelAdaptiveRegularization1();
            
        original2 = _reconstructed4D;
        ParallelAdaptiveRegularization2CardiacVelocity4D parallelAdaptiveRegularization2( this,
                                                                                    b,
                                                                                    factor,
                                                                                    original2 );
        parallelAdaptiveRegularization2();

        reconstructed4D_main = _reconstructed4D;
        confidence_map_main = _confidence_map;



        if (_alpha * _lambda / (_delta * _delta) > 0.068) {
                cerr
                << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068."
                << endl;
        }
            
        

        for (int i=0; i<_reconstructed5DVelocity.size(); i++) {

            _reconstructed4D = _reconstructed5DVelocity[i];
            original1 = originals[i];

            _confidence_map = _confidence_maps_velocity[i];
        
            b.clear();
            for (int i = 0; i < 13; i++)
                b.push_back( _reconstructed4D );
            
            ParallelAdaptiveRegularization1CardiacVelocity4D parallelAdaptiveRegularization1( this,
                                                                                    b,
                                                                                    factor,
                                                                                    original1 );
            parallelAdaptiveRegularization1();
            
            original2 = _reconstructed4D;
            ParallelAdaptiveRegularization2CardiacVelocity4D parallelAdaptiveRegularization2( this,
                                                                                    b,
                                                                                    factor,
                                                                                    original2 );
            parallelAdaptiveRegularization2();


            _reconstructed5DVelocity[i] = _reconstructed4D;

        }


        _reconstructed4D = reconstructed4D_main;
        _confidence_map = confidence_map_main;


    }
    


//------------------------------------------------------------------- 

void ReconstructionCardiacVelocity4D::InitializeGradientMoments(Array<Array<double>> g_directions, Array<double> g_values)
{
  _g_directions = g_directions;
  _g_values = g_values;

}

//------------------------------------------------------------------- 

void ReconstructionCardiacVelocity4D::RotateDirections(double &dx, double &dy, double &dz, int i)
{

  //vector end-point
  double x,y,z;
  //origin
  double ox,oy,oz;
  
  if (_debug)
  {
    //cout<<"Original direction "<<i<<"(dir"<<_stack_index[i]+1<<"): ";
    //cout<<dx<<", "<<dy<<", "<<dz<<". ";
    //cout<<endl;
  }

  //origin
  ox=0;oy=0;oz=0;
  _transformations[i].Transform(ox,oy,oz);
    
  //end-point
  x=dx;
  y=dy;
  z=dz;
  _transformations[i].Transform(x,y,z);
    
  dx=x-ox;
  dy=y-oy;
  dz=z-oz;
    
  if (_debug)
  {
    //cout<<"Rotated direction "<<i<<"(dir"<<_stack_index[i]+1<<"): ";
    //cout<<dx<<", "<<dy<<", "<<dz<<". ";
    //cout<<endl;
  }

}



//------------------------------------------------------------------- 



    void ReconstructionCardiacVelocity4D::MaskSlicesPhase()
    {
        cout << "Masking slices ... ";
        
        double x, y, z;
        int i, j;
        
        //Check whether we have a mask
        if (!_have_mask) {
            cout << "Could not mask slices because no mask has been set." << endl;
            return;
        }
        
        //mask slices
        for (int unsigned inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            RealImage& slice = _slices[inputIndex];
            for (i = 0; i < slice.GetX(); i++)
            for (j = 0; j < slice.GetY(); j++) {
                //if the value is smaller than 1 assume it is padding
                // if (slice(i,j,0) < 0.01)
                //     slice(i,j,0) = -1;
                //image coordinates of a slice voxel
                x = i;
                y = j;
                z = 0;
                //change to world coordinates in slice space
                slice.ImageToWorld(x, y, z);
                //world coordinates in volume space
                _transformations[inputIndex].Transform(x, y, z);
                //image coordinates in volume space
                _mask.WorldToImage(x, y, z);
                x = round(x);
                y = round(y);
                z = round(z);
                //if the voxel is outside mask ROI set it to -1 (padding value)
                if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0)
                    && (z < _mask.GetZ())) {
                    if (_mask(x, y, z) == 0)
                    slice(i, j, 0) = -1;
                }
                else
                slice(i, j, 0) = -1;
            }
            //remember masked slice
            //_slices[inputIndex] = slice;
        }
        cout << "done." << endl;
    }
    




//------------------------------------------------------------------- 


//------------------------------------------------------------------- 


//------------------------------------------------------------------- 


//------------------------------------------------------------------- 


//------------------------------------------------------------------- 


//------------------------------------------------------------------- 


    
} // namespace mirtk

