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
        
        _no_sr = true;
        
        
        // initialise velocity direction array
        Array<double> tmp;
        double t;
        
        _velocity_scale = 1;
        
        _adaptive_regularisation = true;
        _limit_intensities = false;
        
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
        
        _reconstructed4D = 0;
        
        _reconstructed5DVelocity.clear();
        for ( int i=0; i<_v_directions.size(); i++ )
            _reconstructed5DVelocity.push_back(_reconstructed4D);
        
        _confidence_maps_velocity.clear();
        for ( int i=0; i<_v_directions.size(); i++ )
            _confidence_maps_velocity.push_back(_reconstructed4D);
        
    }
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionCardiacVelocity4D::SimulateSignal(int iter)
    {
        _simulated_signal = _reconstructed4D;
        _simulated_signal = 0;
        
        double tmp_signal;
        
        int gradientIndex = 1;
        
        double gval = _g_values[gradientIndex];
        Array<double> g_direction = _g_directions[gradientIndex];
        
        for (int t=0; t<_simulated_signal.GetT(); t++) {
            for (int z=0; z<_simulated_signal.GetZ(); z++) {
                for (int y=0; y<_simulated_signal.GetY(); y++) {
                    for (int x=0; x<_simulated_signal.GetX(); x++) {
                        
                        tmp_signal = 0;
                        for (int v=0; v<_reconstructed5DVelocity.size(); v++) {
                            tmp_signal += _reconstructed5DVelocity[v](x, y, z, t)*g_direction[v]*gval;
                        }
                        
                        _simulated_signal(x,y,z,t) = tmp_signal;
                        
                    }
                }
            }
        }
        
        
        char buffer[256];
        
        sprintf(buffer,"simulated-signal-%i.nii.gz", iter);
        _simulated_signal.Write(buffer);
        
    }
    
    
    //-------------------------------------------------------------------
    
    
    double ReconstructionCardiacVelocity4D::Consistency()
    {
        double ssd=0;
        int num = 0;
        double diff;
        
        for(int index = 0; index< _slices.size(); index++) {
            for(int i=0;i<_slices[index].GetX();i++)
                for(int j=0;j<_slices[index].GetY();j++)
                    if((_slices[index](i,j,0)>=0)&&(_simulated_inside[index](i,j,0)==1)) {
                        diff = _slices[index](i,j,0)-_simulated_slices[index](i,j,0);
                        ssd+=diff*diff;
                        num++;
                    }
        }
        
        cout<<" - Consistency : " << ssd << " " << sqrt(ssd/num) << endl;
        
        return ssd;
    }
    
    
    
    //-------------------------------------------------------------------
    
    
    
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
                
                
                
                double weight;
                POINT3D p;
                for ( int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++ ) {
                    for ( int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++ ) {
                        if ( reconstructor->_slices[inputIndex](i, j, 0) > -100 ) {
                            weight = 0;
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            
                            for ( int k = 0; k < n; k++ ) {
                                
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                for ( int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++ ) {
                                    
                                    if (reconstructor->_reconstructed4D.GetT() == 1) {
                                        reconstructor->_slice_temporal_weight[outputIndex][inputIndex] = 1;
                                    }
   
                                    // simulation of phase volume from velocity volumes
                                    double sim_signal = 0;
                                    
                                    for( int velocityIndex = 0; velocityIndex < reconstructor->_reconstructed5DVelocity.size(); velocityIndex++ ) {
                                        sim_signal += reconstructor->_reconstructed5DVelocity[velocityIndex](p.x, p.y, p.z, outputIndex)*g_direction[velocityIndex]*gval;
                                    }
                                    
                                    sim_signal = sim_signal * reconstructor->gamma;
                                    
                                    // reconstruction from velocity
                                    reconstructor->_simulated_slices[inputIndex](i, j, 0) += sim_signal * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value; // * reconstructor->_reconstructed4D(p.x, p.y, p.z, outputIndex);

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
            cout<<"Simulating Slices ...";
        
        ParallelSimulateSlicesCardiacVelocity4D parallelSimulateSlices( this );
        parallelSimulateSlices();
        
        if (_debug)
        cout<<" done."<<endl;
    }
    
    
    //-------------------------------------------------------------------
    

    
    //-------------------------------------------------------------------
    
    
    void ReconstructionCardiacVelocity4D::InitialisationCardiacVelocity4D(Array<int> stack_numbers)
    {
        
        
        for (int x=0; x<_reconstructed4D.GetX(); x++) {
            for (int y=0; y<_reconstructed4D.GetY(); y++) {
                for (int z=0; z<_reconstructed4D.GetZ(); z++) {
                    for (int t=0; t<_reconstructed4D.GetT(); t++) {
                        
                        int c_index = _slice_contributions_volume(x,y,z,t);
                        
                        Array<double> p_values;
                        Array<double> v_values;
                        Array<Array<double>> g_values;
                        
//                        cout << " - [" << x << " " << y << " " << z << "] : " << _slice_contributions_array[c_index].size() << endl;
                      
                        for (int s=1; s<_slice_contributions_array[c_index].size(); s++) {
                            
                            POINT3DS ps;
                            ps = _slice_contributions_array[c_index][s];
                            
                            bool s_add = false;
                            for (int a=0; a<stack_numbers.size(); a++) {
                                if(_stack_index[ps.i] == stack_numbers[a])
                                    s_add = true;
                            }
                                

                            int g_index = _stack_index[ps.i];
                            double gx, gy, gz;
                            gx = _g_directions[g_index][0];
                            gy = _g_directions[g_index][1];
                            gz = _g_directions[g_index][2];
                            RotateDirections(gx, gy, gz, ps.i);
                            
                            Array<double> g_value;
                            g_value.push_back(gx);
                            g_value.push_back(gy);
                            g_value.push_back(gz);
                            
                            double dg_limint = 0.01;
                            
                            for (int a=0; a<p_values.size(); a++) {
                                
                                if( abs(gx-g_values[a][0])<dg_limint && abs(gy-g_values[a][1])<dg_limint && abs(gz-g_values[a][2])<dg_limint )
                                    s_add = false;
                            }

                            if (s_add) {
                                g_values.push_back(g_value);
                                p_values.push_back(ps.value);

                            }

                        }
                        if (g_values.size()>2) {

                            v_values = InverseVelocitySolution(p_values, g_values);
                        
                            for (int v=0; v<_reconstructed5DVelocity.size(); v++)
                                _reconstructed5DVelocity[v](x,y,z,t) = v_values[v];
                            
                        }
                        
                        
                    }
                }
            }
        }

        char buffer[256];
        
        for (int v=0; v<_reconstructed5DVelocity.size(); v++) {
            
            GaussianBlurring<RealPixel> gb(_reconstructed5DVelocity[0].GetXSize()*0.25);
            gb.Input(&_reconstructed5DVelocity[v]);
            gb.Output(&_reconstructed5DVelocity[v]);
            gb.Run();
            
            sprintf(buffer,"init-velocity-%i.nii.gz", v);
            _reconstructed5DVelocity[v].Write(buffer);
        }
        

    }
    
    
    //-------------------------------------------------------------------
    
    
    Array<double> ReconstructionCardiacVelocity4D::InverseVelocitySolution(Array<double> p_values, Array<Array<double>> g_values)
    {
        
        int N = p_values.size();
        Matrix p_vector(N, 1);
        Matrix v_vector(N, 1);
        Matrix g_matrix(N, N);

        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                if(i==j)
                    g_matrix(i,j) = 1;
                if (j<3)
                    g_matrix(i,j) = g_values[i][j];
            }
            p_vector(i,0) = p_values[i];
        }
        
        g_matrix.Invert();

        v_vector = g_matrix * p_vector;
        
        v_vector /= _g_values[0]*gamma;
        
        Array<double> v_values;
        
        for (int v=0; v<_reconstructed5DVelocity.size(); v++)
            v_values.push_back(v_vector(v));
        
        
        return v_values;
        
    }
    
    
    //-------------------------------------------------------------------
    
    
    //-------------------------------------------------------------------
    
    
    
    //-------------------------------------------------------------------
    
    class ParallelSuperresolutionCardiacVelocity4D {
        ReconstructionCardiacVelocity4D* reconstructor;
    public:
        Array<RealImage> confidence_maps;
        Array<RealImage> addons;
        
        
        void operator()( const blocked_range<size_t>& r ) {
            
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                // read the current simulated slice
                RealImage sim = reconstructor->_simulated_slices[inputIndex];
                
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
                
 
                
                for ( int velocityIndex = 0; velocityIndex < reconstructor->_v_directions.size(); velocityIndex++ ) {
                    
                    
                    double v_component;
                    
                    v_component = g_direction[velocityIndex] / (3*reconstructor->gamma*gval);
                    
                    //Update reconstructed velocity volumes using current slice
                
                    //Distribute error to the volume
                    POINT3D p;
                    for ( int i = 0; i < slice.GetX(); i++ ) {
                        for ( int j = 0; j < slice.GetY(); j++ ) {
                            
                            if (slice(i, j, 0) > -100) {
                                //bias correct and scale the slice
                                
//                                slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                                
                                
                                if ( reconstructor->_simulated_slices[inputIndex](i,j,0) > -100 ) {
                                    slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                                }
                                else
                                    slice(i,j,0) = 0;
                                
                                
//                                // .............................................
//                                if ( reconstructor->_simulated_slices[inputIndex](i,j,0) != 0 ) {
//                                    if ( reconstructor->_simulated_slices[inputIndex](i,j,0) > reconstructor->_max_phase*1.1 || reconstructor->_simulated_slices[inputIndex](i,j,0) < reconstructor->_min_phase*0.9 )
//                                        phase_weight(i,j,0) = 0.0;
//                                    else
//                                        phase_weight(i,j,0) = 1.0;
//                                }
//                                // .............................................
                                
                                
                                int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                                
                                for ( int k = 0; k < n; k++ ) {
                                    
                                    p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                    
                                    for ( int outputIndex=0; outputIndex<reconstructor->_reconstructed4D.GetT(); outputIndex++ ) {
                                        
                                        
                                        if (reconstructor->_robust_slices_only) {
                                            
                                            addons[velocityIndex](p.x, p.y, p.z, outputIndex) += v_component * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                            confidence_maps[velocityIndex](p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_slice_weight[inputIndex];
                                            
                                        }
                                        else {
                                            
                                            addons[velocityIndex](p.x, p.y, p.z, outputIndex) += v_component * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                            
                                            confidence_maps[velocityIndex](p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                            
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
            
        }
        
        void join( const ParallelSuperresolutionCardiacVelocity4D& y ) {
            
            for (int i=0; i<addons.size(); i++) {
                addons[i] += y.addons[i];
                confidence_maps[i] += y.confidence_maps[i];
            }

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
            
        }
        
        // execute
        void operator() () {
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()),
                            *this );
        }
    };
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionCardiacVelocity4D::SuperresolutionCardiacVelocity4D( int iter )
    {
        if (_debug)
            cout << "Superresolution " << iter << " ( " << _alpha << " )" << endl;
        
        char buffer[256];
        
        Array<RealImage> addons, originals;
        
        
        // Remember current reconstruction for edge-preserving smoothing
        originals = _reconstructed5DVelocity;
        
        ParallelSuperresolutionCardiacVelocity4D parallelSuperresolution(this);
        parallelSuperresolution();
        
        addons = parallelSuperresolution.addons;
        _confidence_maps_velocity = parallelSuperresolution.confidence_maps;
        

        if(_debug) {
            
            if (_reconstructed5DVelocity[0].GetT() == 1) {
                
                ImageAttributes attr = _reconstructed5DVelocity[0].GetImageAttributes();
                attr._t = 3;
                
                RealImage output_4D(attr);
                
                for (int t=0; t<output_4D.GetT(); t++)
                    for (int z=0; z<output_4D.GetZ(); z++)
                        for (int y=0; y<output_4D.GetY(); y++)
                            for (int x=0; x<output_4D.GetX(); x++)
                                output_4D(x,y,z,t) = addons[t](x,y,z,0);

                sprintf(buffer,"addon-velocity-%i.nii.gz", iter);
                output_4D.Write(buffer);
                
                for (int t=0; t<output_4D.GetT(); t++)
                    for (int z=0; z<output_4D.GetZ(); z++)
                        for (int y=0; y<output_4D.GetY(); y++)
                            for (int x=0; x<output_4D.GetX(); x++)
                                output_4D(x,y,z,t) = _confidence_maps_velocity[t](x,y,z,0);
                
                sprintf(buffer,"confidence-map-velocity-%i.nii.gz", iter);
                output_4D.Write(buffer);
                
            }
            else {
                for ( int i=0; i<_v_directions.size(); i++ ) {

                    sprintf(buffer,"addon-velocity-%i-%i.nii.gz",i,iter);
                    addons[i].Write(buffer);

                    sprintf(buffer,"confidence-map-velocity-%i-%i.nii.gz",i,iter);
                    _confidence_maps_velocity[i].Write(buffer);
                }

            }
        }

        _adaptive = false;
        
        
        if (!_adaptive)
            for ( int v = 0; v < _v_directions.size(); v++ )
                for ( int x = 0; x < addons[v].GetX(); x++ )
                    for ( int y = 0; y < addons[v].GetY(); y++ )
                        for ( int z = 0; z < addons[v].GetZ(); z++ )
                            for ( int t = 0; t < addons[v].GetT(); t++ )
                                if (_confidence_maps_velocity[v](x, y, z, t) > 0) {
                                    // ISSUES if _confidence_map(i, j, k, t) is too small leading to bright pixels
                                    addons[v](x, y, z, t) /= _confidence_maps_velocity[v](x, y, z, t);
                                    //this is to revert to normal (non-adaptive) regularisation
                                    _confidence_maps_velocity[v](x, y, z, t) = 1;
                                    
                                }
        
        for ( int v = 0; v < _v_directions.size(); v++ )
            _reconstructed5DVelocity[v] += addons[v] * _alpha; //_average_volume_weight;
        
        
        if(_debug) {

            for ( int i=0; i<_v_directions.size(); i++ ) {

                sprintf(buffer,"new-velocity-%i-%i.nii.gz",i,iter);
                _reconstructed5DVelocity[i].Write(buffer);
            }

        }

        
        if(_limit_intensities) {
         
             //bound the intensities (test whether we need it)
             for (int x = 0; x < _reconstructed4D.GetX(); x++) {
                 for (int y = 0; y < _reconstructed4D.GetY(); y++) {
                     for (int z = 0; z < _reconstructed4D.GetZ(); z++) {
                         for (int t = 0; t < _reconstructed4D.GetT(); t++) {
                             
                             double v_sum = _reconstructed5DVelocity[0](x,y,z,t)+_reconstructed5DVelocity[1](x,y,z,t)+_reconstructed5DVelocity[2](x,y,z,t);
                             
                             for ( int v = 0; v < _v_directions.size(); v++ ) {
                                 
                                 if (_reconstructed5DVelocity[v](x, y, z, t) < _min_velocity*0.9)
                                     _reconstructed5DVelocity[v](x, y, z, t) = _min_velocity*0.9;
                                 
                                 if (_reconstructed5DVelocity[v](x, y, z, t) > _max_velocity*1.1)
                                     _reconstructed5DVelocity[v](x, y, z, t) = _max_velocity*1.1;
                                 
                             }
                             
                         }
                     }
                 }
             }
        
        }

        
        _adaptive_regularisation = false;

        if(_adaptive_regularisation) {

            // Smooth the reconstructed image
            AdaptiveRegularizationCardiacVelocity4D(iter, originals);
            
        }

        
        //Remove the bias in the reconstructed volume compared to previous iteration
        //  TODO: update adaptive regularisation for 4d
        //  if (_global_bias_correction)
        //  BiasCorrectVolume(original);
        
        
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
                for (int y = 0; y < dy; y++) {
                    for (int z = 0; z < dz; z++) {
                            for (int t = 0; t < dt; t++) {
                                if(reconstructor->_confidence_map(x,y,z,t)>0) {
                                    double val = 0;
                                    double sum = 0;
                                    for (int i = 0; i < 13; i++) {
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
    
    void ReconstructionCardiacVelocity4D::AdaptiveRegularizationCardiacVelocity4D(int iter, Array<RealImage>& originals) //, RealImage& original_main)
    {
        if (_debug)
            cout << "AdaptiveRegularizationCardiacVelocity4D."<< endl;

        Array<double> factor(13,0);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }
        
        RealImage original1, original2;
        
        Array<RealImage> b;//(13);
        for (int i = 0; i < 13; i++)
            b.push_back( _reconstructed4D );
        
        
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
        
        
    }
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionCardiacVelocity4D::StaticMaskReconstructedVolume5D()
    {
        for ( int v = 0; v < _reconstructed5DVelocity.size(); v++ ) {
            for ( int x = 0; x < _mask.GetX(); x++ ) {
                for ( int y = 0; y < _mask.GetY(); y++ ) {
                    for ( int z = 0; z < _mask.GetZ(); z++ ) {
                        
                        if ( _mask(x,y,z) == 0 ) {
                            
                            for ( int t = 0; t < _reconstructed4D.GetT(); t++ ) {
                                
                                _reconstructed5DVelocity[v](x,y,z,t) = -1;
                                
                            }
                        }
                    }
                }
            }
        }
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
        
        RigidTransformation tmp = _transformations[i];
        
//        tmp.PutTranslationX(0);
//        tmp.PutTranslationY(0);
//        tmp.PutTranslationZ(0);
        
//        _transformations[i].Invert();
        
        //origin
        ox=0;oy=0;oz=0;
        tmp.Transform(ox,oy,oz);
        
        //end-point
        x=dx;
        y=dy;
        z=dz;
        tmp.Transform(x,y,z);
        
        dx=x-ox;
        dy=y-oy;
        dz=z-oz;
        
        
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
                            slice(i, j, 0) = -150;
                    }
                    else
                        slice(i, j, 0) = -150;
                }
            //remember masked slice
            //_slices[inputIndex] = slice;
        }
        cout << "done." << endl;
    }
    
    
    
    
    
    //-------------------------------------------------------------------
    
    
    //-------------------------------------------------------------------
    

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
                    
                    
                    if (reconstructor->_reconstructed4D.GetT() == 1) {
                        reconstructor->_slice_temporal_weight[outputIndex][inputIndex] = 1;
                    }
                    
                    
                    if (reconstructor->_slice_excluded[inputIndex]==0) {
                        
                        if(reconstructor->_debug)
                        {
                            cout << inputIndex << " ";
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
                        
                        Array<double> g_direction;
                        g_direction.push_back(gx);
                        g_direction.push_back(gy);
                        g_direction.push_back(gz);
                        
                        
                        for ( velocityIndex = 0; velocityIndex < reconstructor->_v_directions.size(); velocityIndex++ ) {
                            
                            Array<double> v_componets;
                            
                            if ( g_direction[velocityIndex]>0.01 || g_direction[velocityIndex]<-0.01 )
                                v_component = 1/(g_direction[velocityIndex]*reconstructor->gamma*gval);
                            else
                                v_component = 0;
                            
                            
                            // distribute slice intensities to the volume
                            for ( int i = 0; i < slice.GetX(); i++ ) {
                                for ( int j = 0; j < slice.GetY(); j++ ) {
                                    
                                    if (slice(i, j, 0) > -100) {
                                        // biascorrect and scale the slice
                                        //                                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                                        
                                        // number of volume voxels with non-zero coefficients for current slice voxel
                                        n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                                        
                                        double max_p = 0;
                                        
                                        // add contribution of current slice voxel to all voxel volumes to which it contributes
                                        for ( k = 0; k < n; k++ ) {
                                            
                                            p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                            
                                            if (p.value>max_p)
                                                max_p = p.value;
                                            
                                            reconstructed3DVelocityArray[velocityIndex](p.x, p.y, p.z) += v_component * slice(i, j, 0) * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                            weights(p.x, p.y, p.z) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                            
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
        
        
        if(_limit_intensities) {
            
            //bound the intensities (test whether we need it)
            for (int x = 0; x < _reconstructed4D.GetX(); x++) {
                for (int y = 0; y < _reconstructed4D.GetY(); y++) {
                    for (int z = 0; z < _reconstructed4D.GetZ(); z++) {
                        for (int t = 0; t < _reconstructed4D.GetT(); t++) {
                            for ( int v = 0; v < _v_directions.size(); v++ ) {
                                
                                
                                if (_reconstructed5DVelocity[v](x, y, z, t) < _min_velocity*0.9)
                                    _reconstructed5DVelocity[v](x, y, z, t) = _min_velocity*0.9;
                                
                                if (_reconstructed5DVelocity[v](x, y, z, t) > _max_velocity*1.1)
                                    _reconstructed5DVelocity[v](x, y, z, t) = _max_velocity*1.1;
                                
                            }
                        }
                    }
                }
            }
            
        }
        
        
        char buffer[256];
        
        if (_debug) {
            
            
            if (_reconstructed5DVelocity[0].GetT() == 1) {
                
                ImageAttributes attr = _reconstructed5DVelocity[0].GetImageAttributes();
                attr._t = 3;
                
                RealImage output_4D(attr);
                
                for (int t=0; t<output_4D.GetT(); t++)
                    for (int z=0; z<output_4D.GetZ(); z++)
                        for (int y=0; y<output_4D.GetY(); y++)
                            for (int x=0; x<output_4D.GetX(); x++)
                                output_4D(x,y,z,t) = _reconstructed5DVelocity[t](x,y,z,0);
                
                sprintf(buffer,"recon4D-gaussian-velocity-vector.nii.gz");
                output_4D.Write(buffer);
                
            }
            
        }
        
        
    }
    
    
    
    //-------------------------------------------------------------------
    
    
    /*
     
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
     
     
     // gradient direction for current slice
     double gradientIndex = reconstructor->_stack_index[inputIndex];
     double g = reconstructor->_g_directions[gradientIndex][0];
     //                        gy = reconstructor->_g_directions[gradientIndex][1];
     //                        gz = reconstructor->_g_directions[gradientIndex][2];
     
     
     // distribute slice intensities to the volume
     for ( int i = 0; i < slice.GetX(); i++ ) {
     for ( int j = 0; j < slice.GetY(); j++ ) {
     
     if (slice(i, j, 0) > -100) {
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
     
     
     
     //bound the intensities
     for (int x = 0; x < _reconstructed4D.GetX(); x++) {
     for (int y = 0; y < _reconstructed4D.GetY(); y++) {
     for (int z = 0; z < _reconstructed4D.GetZ(); z++) {
     for (int t = 0; t < _reconstructed4D.GetT(); t++) {
     
     if (_reconstructed4D(x, y, z, t) < _min_phase*0.9)
     _reconstructed4D(x, y, z, t) = _min_phase*0.9;
     
     if (_reconstructed4D(x, y, z, t) > _max_phase*1.1)
     _reconstructed4D(x, y, z, t) = _max_phase*1.1;
     }
     }
     }
     
     }
     
     
     
     char buffer[256];
     
     if (_debug) {
     
     sprintf(buffer,"recon4D-gaussian-phase.nii.gz");
     _reconstructed4D.Write(buffer);
     
     }
     
     }
     
     */
    
    
    //-------------------------------------------------------------------
    
    
    
    /*
     void ReconstructionCardiacVelocity4D::InitialiseInverse(Array<RealImage> stacks)
     {
     
     char buffer[256];
     
     int N = stacks.size();
     
     Matrix p_vector(N, 1);
     Matrix v_vector(N, 1);
     Matrix g_matrix(N, N);
     
     for (int i=0; i<N; i++) {
     for (int j=0; j<N; j++) {
     if(i==j)
     g_matrix(i,j) = 1;
     if (j<3)
     g_matrix(i,j) = _g_directions[i][j];
     }
     }
     
     
     g_matrix.Print();
     g_matrix.Invert();
     g_matrix.Print();
     
     
     RealImage tmp_volume = _reconstructed4D;
     tmp_volume = 0;
     
     Array<RealImage> v_volumes;
     for (int v=0; v<N; v++) {
     v_volumes.push_back(tmp_volume);
     }
     
     ImageAttributes attr = _reconstructed4D.GetImageAttributes();
     attr._t = N;
     
     RealImage v_4D(attr);
     v_4D = 0;
     
     
     
     for (int z=0; z<tmp_volume.GetZ(); z++) {
     for (int y=0; y<tmp_volume.GetY(); y++) {
     for (int x=0; x<tmp_volume.GetX(); x++) {
     for (int t=0; t<tmp_volume.GetT(); t++) {
     
     double main_wx, main_wy, main_wz, wx, wy, wz;
     int ix, iy, iz;
     
     main_wx = x;
     main_wy = y;
     main_wz = z;
     
     _reconstructed4D.ImageToWorld(main_wx, main_wy, main_wz);
     
     for (int n=0; n<N; n++) {
     
     wx = main_wx;
     wy = main_wy;
     wz = main_wz;
     
     stacks[n].WorldToImage(wx, wy, wz);
     
     ix = round(wx);
     iy = round(wy);
     iz = round(wz);
     
     if (ix>-1 && ix<stacks[n].GetX() && iy>-1 && iy<stacks[n].GetY() && iz>-1 && iz<stacks[n].GetZ() )
     p_vector(n,0) = stacks[n](ix, iy, iz)/(gamma*_g_values[n]);
     else
     p_vector(n,0) = 0;
     
     
     }
     
     v_vector = g_matrix * p_vector;
     
     for (int v=0; v<N; v++) {
     v_volumes[v](x,y,z) = v_vector(v,0);
     v_4D(x,y,z,v) = v_vector(v,0);
     }
     
     for (int v=0; v<3; v++) {
     _reconstructed5DVelocity[v](x,y,z) = v_vector(v,0);
     }
     
     }
     }
     }
     }
     
     
     
     //        for (int v=0; v<v_volumes.size(); v++) {
     //
     //            sprintf(buffer,"inverted-velocity-%i.nii.gz", v);
     //            v_volumes[v].Write(buffer);
     //
     //        }
     
     v_4D.Write("inverted-velocity-vector.nii.gz");
     
     }
     
     */
    
    
    
    //-------------------------------------------------------------------
    
    
    //-------------------------------------------------------------------
    
    
    //-------------------------------------------------------------------
    
    
    
} // namespace mirtk



