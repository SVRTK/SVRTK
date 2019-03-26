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
        
        
        current_stack_for_processing = 0;
        
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
                        if ( reconstructor->_slices[inputIndex](i, j, 0) > -10 ) {
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
            cout<<"Simulating slices ...";
        
        ParallelSimulateSlicesCardiacVelocity4D parallelSimulateSlices( this );
        parallelSimulateSlices();
        
        if (_debug)
        cout<<" done."<<endl;
    }
    
    
    //-------------------------------------------------------------------
    

    
    //-------------------------------------------------------------------
    
    
    void ReconstructionCardiacVelocity4D::InitialisationCardiacVelocity4D(Array<int> stack_numbers)
    {
        
        cout << "Analytical initialisation." << endl;
        
        for (int x=0; x<_reconstructed4D.GetX(); x++) {
            for (int y=0; y<_reconstructed4D.GetY(); y++) {
                for (int z=0; z<_reconstructed4D.GetZ(); z++) {
                    for (int t=0; t<_reconstructed4D.GetT(); t++) {
                        
                        int c_index = _slice_contributions_volume(x,y,z,t);
                        
                        Array<double> p_values;
                        Array<double> v_values;
                        Array<double> p_weights;
                        Array<Array<double>> g_values;
                    
                        double w = 0;
                      
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
                            
                            
                            /*
                            
                            double dgx, dgy, dgz;
                            
                            dgx = gx;
                            dgy = gy;
                            dgz = gz;
                            
                            RigidTransformation tmp = _transformations[ps.i];
                            
                            double rx, ry, rz;
                            
                            rx = tmp.GetRotationX();
                            ry = tmp.GetRotationY();
                            rz = tmp.GetRotationZ();
                            
                            if ( abs(rx) > 0 || abs(ry) > 0 || abs(rz) > 0 ) {

//                                cout << gx << " " << gy << " " << gz << endl;
                                
    //                            tmp.Rotate(gx, gy, gz);
                                
                                RotateDirections(gx, gy, gz, ps.i);
                                
                                
//                                dgx = abs(dgx-gx);
//                                dgy = abs(dgy-gy);
//                                dgz = abs(dgz-gz);
//
//                                if ( dgx > 0.0001 || dgy > 0.0001 || dgz > 0.0001 )
//                                    cout << dgx << " " << dgy << " " << dgz << " | " << rx << " " << ry << " " << rz << endl;
//                                cout << gx << " " << gy << " " << gz << endl;
//
//                                exit(1);
                            
                            }
                            
                            */
                            
                            
                            Array<double> g_value;
                            g_value.push_back(gx);
                            g_value.push_back(gy);
                            g_value.push_back(gz);
                            
                            double dg_limint = 0.0001;
                            
                            
                            for (int a=0; a<p_values.size(); a++) {
                                
                                if( abs( gx-g_values[a][0])<dg_limint && abs(gy-g_values[a][1])<dg_limint && abs(gz-g_values[a][2])<dg_limint ) {
                                    
                                    s_add = false;
                                    
                                    if ( ps.w > p_weights[a] ) {
                                        
                                        p_weights[a] = ps.w;
                                        g_values[a][0] = gx;
                                        g_values[a][1] = gy;
                                        g_values[a][2] = gz;
                                        p_values[a] = ps.value;
                                        
                                    }
                                    
                                }
                            }
                            
                            if (ps.value>3.14 || ps.value<-3.14)
                                s_add = false;

                            if (s_add) {
                                g_values.push_back(g_value);
                                p_values.push_back(ps.value);
                                p_weights.push_back(ps.w);
                            }

                        }
                        
//                        cout << " - [" << x << " " << y << " " << z << "] : " << _slice_contributions_array[c_index].size() << " / " << g_values.size() << endl;
                            
                        
                        if (g_values.size()>2) {

                            v_values = InverseVelocitySolution(p_values, g_values);
                        
                            for (int v=0; v<_reconstructed5DVelocity.size(); v++) {
                                
                                if (v_values[v] < _min_velocity)
                                    v_values[v] = _min_velocity*2;
                                
                                if (v_values[v] > _max_velocity)
                                    v_values[v] = _max_velocity*2;
                                
                                _reconstructed5DVelocity[v](x,y,z,t) = v_values[v];
                            }
                            
                        }

                        
//                        else {
//                            for (int v=0; v<_reconstructed5DVelocity.size(); v++)
//                                _reconstructed5DVelocity[v](x,y,z,t) = -50;
//                        }
                        
                        
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
    
    
    //-------------------------------------------------------------------
    
    
    //-------------------------------------------------------------------
    
    class ParallelSuperresolutionCardiacVelocity4D {
        ReconstructionCardiacVelocity4D* reconstructor;
    public:
        Array<RealImage> confidence_maps;
        Array<RealImage> addons_p;
        Array<RealImage> addons_n;
        Array<RealImage> addons;
        
        
        void operator()( const blocked_range<size_t>& r ) {
            
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                // read the current simulated slice
                RealImage sim = reconstructor->_simulated_slices[inputIndex];
                
                RealImage sss = slice - sim;
                
                
                for ( int i = 0; i < slice.GetX(); i++ ) {
                    for ( int j = 0; j < slice.GetY(); j++ ) {
                        
                        if (slice(i,j,0)>-10 && sim(i,j,0)>-10)
                            reconstructor->_dif_stacks[reconstructor->_stack_index[inputIndex]](i,j,reconstructor->_stack_loc_index[inputIndex],0) = sss(i,j,0);
                        else {
                            sss(i,j,0) = 0;
                            reconstructor->_dif_stacks[reconstructor->_stack_index[inputIndex]](i,j,reconstructor->_stack_loc_index[inputIndex],0) = 0;
                        }
                        
                    }
                }
                
//                 for ( int i = 0; i < slice.GetX(); i++ ) {
//                     for ( int j = 0; j < slice.GetY(); j++ ) {
                        
//                         if (slice(i,j,0)>-10)
//                             reconstructor->_dif_stacks[reconstructor->_stack_index[inputIndex]](i,j,reconstructor->_stack_loc_index[inputIndex],0) = sss(i,j,0);
//                         else {
//                             sss(i,j,0) = -15;
//                             reconstructor->_dif_stacks[reconstructor->_stack_index[inputIndex]](i,j,reconstructor->_stack_loc_index[inputIndex],0) = 0;
//                         }
                        
//                     }
//                 }
                
                
                slice = sss;
                

                //read the current weight image
                RealImage& w = reconstructor->_weights[inputIndex];
                
//                 //read the current bias image
//                 RealImage& b = reconstructor->_bias[inputIndex];
                
//                 //identify scale factor
//                 double scale = reconstructor->_scale[inputIndex];
                
                
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
                
                
//                char buffer[256];
//                sprintf(buffer," - %i (%i) : %d %d %d ", inputIndex, gradientIndex, gx, gy, gz);
//
//                cout << buffer << endl;
                
//                cout << inputIndex << " - " << gradientIndex << " : " << gx << " " << gy << " " << gz << endl;
                
                
                
                for ( int velocityIndex = 0; velocityIndex < reconstructor->_v_directions.size(); velocityIndex++ ) {
                    
                    
                    double v_component;
                    
                    v_component = g_direction[velocityIndex] / (3*reconstructor->gamma*gval);
                    

                    //Update reconstructed velocity volumes using current slice
                    
                    //Distribute error to the volume
                    POINT3D p;
                    for ( int i = 0; i < slice.GetX(); i++ ) {
                        for ( int j = 0; j < slice.GetY(); j++ ) {
                            
                            if (slice(i, j, 0) > -10) {
                                //bias correct and scale the slice
                                
                                //                                slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                                
                                if (sim(i,j,0)<-10) 
                                    slice(i,j,0) = 0;
                                    
//                                if ( reconstructor->_simulated_slices[inputIndex](i,j,0) > -10 ) {
//                                    slice(i,j,0) = slice(i,j,0) - sim(i,j,0); //reconstructor->_simulated_slices[inputIndex](i,j,0);
//                                }
//                                else
//                                    slice(i,j,0) = 0;
                                
                                
                                
                                int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                                
                                
                                for ( int k = 0; k < n; k++ ) {
                                    
                                    p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                    
                                    if (p.value>0.0) {
                                        
                                        for ( int outputIndex=0; outputIndex<reconstructor->_reconstructed4D.GetT(); outputIndex++ ) {
                                            
                                            if (reconstructor->_robust_slices_only) {
                                                
                                                
                                                
//                                                addons[velocityIndex](p.x, p.y, p.z, outputIndex) += v_component * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * reconstructor->_slice_weight[inputIndex];
//                                                confidence_maps[velocityIndex](p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_slice_weight[inputIndex];
                                                
                                                addons[velocityIndex](p.x, p.y, p.z, outputIndex) += v_component * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * w(i, j, 0);
                                                
                                                confidence_maps[velocityIndex](p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * w(i, j, 0);
                                                
                                            }
                                            else {
                                                
//                                                double a_tmp = slice(i, j, 0);
//
//
                                                
                                                addons[velocityIndex](p.x, p.y, p.z, outputIndex) += v_component * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                                
                                                confidence_maps[velocityIndex](p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
//
//
//                                                if(a_tmp>0)
//                                                    addons_p[velocityIndex](p.x, p.y, p.z, outputIndex) += a_tmp;
//                                                else
//                                                    addons_n[velocityIndex](p.x, p.y, p.z, outputIndex) += a_tmp;
                                                
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                } // end of loop for velocity directions
                
                
                //                }
                
                
            } //end of loop for a slice inputIndex
            
        }
        
        ParallelSuperresolutionCardiacVelocity4D( ParallelSuperresolutionCardiacVelocity4D& x, split ) :
        reconstructor(x.reconstructor)
        {
            // Clear addon
            RealImage addon = reconstructor->_reconstructed4D;
            addon = 0;
            
            addons.clear();
//            addons_n.clear();
//            addons_p.clear();
            for (int i=0; i<reconstructor->_reconstructed5DVelocity.size(); i++) {
                addons.push_back(addon);
//                addons_p.push_back(addon);
//                addons_n.push_back(addon);
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
//                addons_n[i] += y.addons_n[i];
//                addons_p[i] += y.addons_p[i];
                
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
//            addons_n.clear();
//            addons_p.clear();
            for (int i=0; i<reconstructor->_reconstructed5DVelocity.size(); i++) {
                addons.push_back(addon);
//                addons_p.push_back(addon);
//                addons_n.push_back(addon);
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
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()), *this );
        }
    };
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionCardiacVelocity4D::SuperresolutionCardiacVelocity4D( int iter )
    {
        if (_debug)
            cout << "Superresolution ... ";
        
        char buffer[256];
        
        Array<RealImage> addons, originals;
        
        
        for (int i=0; i<_dif_stacks.size(); i++) {
            _dif_stacks[i] = 0;
        }
        
        
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
       

//        if (!_adaptive_regularisation)
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
        
        

        if(_adaptive_regularisation) {

            // Smooth the reconstructed image
            AdaptiveRegularizationCardiacVelocity4D(iter, originals);
            
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

        
        //Remove the bias in the reconstructed volume compared to previous iteration
        //  TODO: update adaptive regularisation for 4d
        //  if (_global_bias_correction)
        //  BiasCorrectVolume(original);
        
        
        if (_debug)
            cout << "done." << endl;
        
        
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
//        if (_debug)
//            cout << "AdaptiveRegularizationCardiacVelocity4D."<< endl;

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
                                
                                _reconstructed5DVelocity[v](x,y,z,t) = -15;
                                
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
        
//        for (int i=0; i<_transformations.size(); i++) {
//            
//            _transformations[i].PutRotationX(UnitRandom());
//            _transformations[i].PutRotationY(UnitRandom());
//            _transformations[i].PutRotationZ(UnitRandom());
//            
//        }
        
        
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionCardiacVelocity4D::RotateDirections(double &dx, double &dy, double &dz, int i)
    {
        
       //vector end-point
        double x,y,z;
        //origin
        double ox,oy,oz;
        
        RigidTransformation tmp = _transformations[i];
//
//
//        if (_random_transformations.size() > 0 )
//            tmp = _random_transformations[i];
        
        
//        RigidTransformation tmp = _random_transformations[i];
        
        
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
    
    
    
    void ReconstructionCardiacVelocity4D::RandomRotations(Array<RealImage> stacks)
    {
        
        
        /*
        _random_transformations.clear();
    
        for (int i=0; i<_transformations.size(); i++) {
            
            RigidTransformation *tmp = new RigidTransformation;
            
            RigidTransformation random_transf = *tmp; //_transformations[i];
            
            int r1, r2;
            double rx, ry, rz;
            double range;
            int f;
            
//            f = 10;
//            range = 5;
//
//            r1 = rand() % f + 0;
//            r2 = rand() % f + 0;
//            rx = (double)r1/range - (double)r2/range;
//
//            r1 = rand() % f + 0;
//            r2 = rand() % f + 0;
//            ry = (double)r1/range - (double)r2/range;
//
//            r1 = rand() % f + 0;
//            r2 = rand() % f + 0;
//            rz = (double)r1/range - (double)r2/range;
//
//            cout << " - " << rx << " " << ry << " " << rz << endl;
//
//            random_transf.PutRotationX(rx);
//            random_transf.PutRotationY(ry);
//            random_transf.PutRotationZ(rz);
//
//
//
//            rx = 0;
//            ry = 0;
//            rz = 0;
            
            
            f = 8;
            range = 7;

            r1 = rand() % f + 0;
            r2 = rand() % f + 0;
            rx = (double)r1/range - (double)r2/range;

            r1 = rand() % f + 0;
            r2 = rand() % f + 0;
            ry = (double)r1/range - (double)r2/range;

            r1 = rand() % f + 0;
            r2 = rand() % f + 0;
            rz = (double)r1/range - (double)r2/range;
            
            
            random_transf.PutTranslationX(rx);
            random_transf.PutTranslationY(ry);
            random_transf.PutTranslationZ(rz);

            _random_transformations.push_back(random_transf);
            
        }
        
         */
        
        
        _random_transformations.clear();
        _random_transformations = _transformations;

        for (int i=0; i<_transformations.size(); i++) {

            if (_stack_index[i] == 2) {

//                _random_transformations[i].PutTranslationX(-20);
//                _random_transformations[i].PutTranslationY(-20);
//                _random_transformations[i].PutTranslationZ(-20);
                
                _random_transformations[i].PutRotationX(5);
                _random_transformations[i].PutRotationY(0);
                _random_transformations[i].PutRotationZ(0);

                
            }


        }
        
        
        double source_padding = 0;
        double target_padding = -inf;
        bool dofin_invert = false;
        bool twod = false;
        
        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        
        
        InterpolationMode interpolation_nn = Interpolation_NN;
        UniquePtr<InterpolateImageFunction> interpolator_nn;
        interpolator_nn.reset(InterpolateImageFunction::New(interpolation_nn));
        
        
        
        _transformations.clear();
        _transformations = _random_transformations;
        
        
        
        for (int i=0; i<_random_transformations.size(); i++) {
        
            ImageTransformation *imagetransformation = new ImageTransformation;
            
            double ox, oy, oz;
            stacks[_stack_index[i]].GetOrigin(ox,oy,oz);
            stacks[_stack_index[i]].PutOrigin(0,0,0);
            
            
            RealImage output_volume = stacks[_stack_index[i]];
            output_volume = 0;
            
            imagetransformation->Input(&stacks[_stack_index[i]]);
            imagetransformation->Transformation(&_random_transformations[i]);
            imagetransformation->Output(&output_volume);
            imagetransformation->TargetPaddingValue(target_padding);
            imagetransformation->SourcePaddingValue(source_padding);
            imagetransformation->Interpolator(interpolator_nn.get());  // (&interpolator); //
            imagetransformation->TwoD(twod);
            imagetransformation->Invert(dofin_invert);
            imagetransformation->Run();
            
            output_volume.PutOrigin(ox,oy,oz);
            stacks[_stack_index[i]].PutOrigin(ox,oy,oz);
            
            RealImage output_slice = _slices[i];
            
            int z = 0;
            for (int x=0; x<stacks[_stack_index[i]].GetX(); x++) {
                for (int y=0; y<stacks[_stack_index[i]].GetY(); y++) {
                    
                    int ix, iy, iz;
                    double dx, dy, dz;
                    dx = x;
                    dy = y;
                    dz = z;
                    
                    _slices[i].ImageToWorld(dx,dy,dz);
                    output_volume.WorldToImage(dx,dy,dz);
                    
                    ix = round(dx);
                    iy = round(dy);
                    iz = round(dz);
                    
                    _slices[i](x,y,0) = output_volume(ix,iy,iz);

                }
            }
            
//            _random_transformations[i].Invert();
//            _transformations.push_back(_random_transformations[i]);
        }
        

        
    }
    
    
    
    //-------------------------------------------------------------------
    
    
    
    void ReconstructionCardiacVelocity4D::SaveOriginal( Array<RealImage> stacks )
    {
        
        if (_debug)
            cout << "Saving original slices as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> simstacks;
        
        for (int i = 0; i < stacks.size(); i++) {
            stacks[i] = 0;
        }
        
        for (int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                    stacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _slices[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "original%03i.nii.gz", i);
            stacks[i].Write(buffer);
        }
        
        
        
        if (_debug)
            cout << " done." << endl;
        
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
                    if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0) && (z < _mask.GetZ())) {
                        if (_mask(x, y, z) == 0)
                            slice(i, j, 0) = -15;
                    }
                    else
                        slice(i, j, 0) = -15;
                }
            //remember masked slice
            //_slices[inputIndex] = slice;
        }
        cout << "done." << endl;
    }
    
    

    //-------------------------------------------------------------------
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionCardiacVelocity4D::InitializeEMVelocity4D()
    {
        if (_debug)
            cout << "InitializeEM" << endl;
        
        _weights.clear();
        _bias.clear();
        _scale.clear();
        _slice_weight.clear();
        
        for (unsigned int i = 0; i < _slices.size(); i++) {
            //Create images for voxel weights and bias fields
            RealImage tmp = _slices[i];
            
            tmp = 1;
            _weights.push_back(tmp);
            
            tmp = 0;
            _bias.push_back(tmp);
            
            //Create and initialize scales
            _scale.push_back(1);
            
            //Create and initialize slice weights
            _slice_weight.push_back(1);
        }
        
        //Find the range of intensities
        _max_intensity = voxel_limits<RealPixel>::min();
        _min_intensity = voxel_limits<RealPixel>::max();
        
//        _max_intensity = -3.14;
//        _min_intensity = 3.14;
        
        
        for (unsigned int i = 0; i < _slices.size(); i++) {
            //to update minimum we need to exclude padding value
            RealPixel *ptr = _slices[i].GetPointerToVoxels();
            
            for (int ind = 0; ind < _slices[i].GetNumberOfVoxels(); ind++) {
                if (*ptr > -10) {
                    
                    double tmp = abs(*ptr);
                    
                    if (tmp > _max_intensity)
                        _max_intensity = tmp;
                    if (tmp < _min_intensity)
                        _min_intensity = tmp;
                    
//                    if (*ptr > _max_intensity)
//                        _max_intensity = *ptr;
//                    if (*ptr < _min_intensity)
//                        _min_intensity = *ptr;
                    
                    
                }
                ptr++;
            }
        }
        
//        _max_intensity = _max_intensity*2;
        
        cout << " - min : " << _min_intensity << " | max : " << _max_intensity << endl;
    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionCardiacVelocity4D::InitializeEMValuesVelocity4D()
    {
        if (_debug)
            cout << "InitializeEMValues" << endl;
        
        for (unsigned int i = 0; i < _slices.size(); i++) {
            //Initialise voxel weights and bias values
            RealPixel *pw = _weights[i].GetPointerToVoxels();
            RealPixel *pb = _bias[i].GetPointerToVoxels();
            RealPixel *pi = _slices[i].GetPointerToVoxels();
            
            for (int j = 0; j < _weights[i].GetNumberOfVoxels(); j++) {
                if (*pi > -10) {
                    *pw = 1;
                    *pb = 0;
                }
                else {
                    *pw = 0;
                    *pb = 0;
                }
                pi++;
                pw++;
                pb++;
            }
            
            //Initialise slice weights
            _slice_weight[i] = 1;
            
            //Initialise scaling factors for intensity matching
            _scale[i] = 1;
        }
        
        //Force exclusion of slices predefined by user
        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            _slice_weight[_force_excluded[i]] = 0;
        
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionCardiacVelocity4D::InitializeRobustStatisticsVelocity4D()
    {
        if (_debug)
            cout << "InitializeRobustStatistics" << endl;
        
        //Initialise parameter of EM robust statistics
        int i, j;
        RealImage slice, sim;
        double sigma = 0;
        int num = 0;
        
        //for each slice
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            slice = _slices[inputIndex];
            
            // read the current simulated slice
            sim = _simulated_slices[inputIndex];
            RealImage sss = slice - sim;

            for ( int i = 0; i < slice.GetX(); i++ ) {
                for ( int j = 0; j < slice.GetY(); j++ ) {
                    if (slice(i,j,0)<-10 )
                        sss(i,j,0) = -15;
                    else
                        sss(i,j,0) = (sss(i,j,0));
                }
            }
            slice = sss;
            
            
            //Voxel-wise sigma will be set to stdev of volumetric errors
            //For each slice voxel
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) > -10) {
                        //calculate stev of the errors
                        if ( (_simulated_inside[inputIndex](i, j, 0)==1) &&(_simulated_weights[inputIndex](i,j,0)>0.99) ) {
//                            slice(i,j,0) -= _simulated_slices[inputIndex](i,j,0);
                            sigma += slice(i, j, 0) * slice(i, j, 0);
                            num++;
                        }
                    }
            
            //if slice does not have an overlap with ROI, set its weight to zero
            if (!_slice_inside[inputIndex])
                _slice_weight[inputIndex] = 0;
        }
        
        //Force exclusion of slices predefined by user
        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            _slice_weight[_force_excluded[i]] = 0;
        
        //initialize sigma for voxelwise robust statistics
        _sigma = sigma / num;
        //initialize sigma for slice-wise robust statistics
        _sigma_s = 0.025;
        //initialize mixing proportion for inlier class in voxel-wise robust statistics
        _mix = 0.9;
        //initialize mixing proportion for outlier class in slice-wise robust statistics
        _mix_s = 0.9;
        //Initialise value for uniform distribution according to the range of intensities
        _m = 1 / (2.1 * _max_intensity - 1.9 * _min_intensity);
        
        
//        _mix = 0.8;
        
        if (_debug)
            cout << "Initializing robust statistics: " << "sigma=" << sqrt(_sigma) << " " << "m=" << _m
            << " " << "mix=" << _mix << " " << "mix_s=" << _mix_s << endl;
        
    }
    
    
    //-------------------------------------------------------------------
    
    class ParallelEStepardiacVelocity4D {
        ReconstructionCardiacVelocity4D* reconstructor;
        Array<double> &slice_potential;
        
    public:
        
        void operator()( const blocked_range<size_t>& r ) const {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                //read current weight image
                reconstructor->_weights[inputIndex] = 0;
                
                //alias the current bias image
//                RealImage& b = reconstructor->_bias[inputIndex];
                
                //identify scale factor
//                double scale = reconstructor->_scale[inputIndex];
                
                
                // read the current simulated slice
                RealImage sim = reconstructor->_simulated_slices[inputIndex];
                RealImage sss = slice - sim;
 
                for ( int i = 0; i < slice.GetX(); i++ ) {
                    for ( int j = 0; j < slice.GetY(); j++ ) {
                        if (slice(i,j,0)<-10 )
                            sss(i,j,0) = -15;
                        else
                            sss(i,j,0) = (sss(i,j,0));
                    }
                }
                slice = sss;
                
                double num = 0;
                //Calculate error, voxel weights, and slice potential
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -10) {
                            
                            //bias correct and scale the slice
//                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                            
                            //number of volumetric voxels to which
                            // current slice voxel contributes
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            
                            // if n == 0, slice voxel has no overlap with volumetric ROI,
                            // do not process it
                            
                            if ( (n>0) && (reconstructor->_simulated_weights[inputIndex](i,j,0) > 0) ) {
                                
//                                slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                                
                                //calculate norm and voxel-wise weights
                                
                                //Gaussian distribution for inliers (likelihood)
                                double g = reconstructor->G(slice(i, j, 0), reconstructor->_sigma);
                                //Uniform distribution for outliers (likelihood)
                                double m = reconstructor->M(reconstructor->_m);
                                
                                //voxel_wise posterior
                                double weight = g * reconstructor->_mix / (g *reconstructor->_mix + m * (1 - reconstructor->_mix));
                                
//                                weight = slice(i, j, 0);
                                
                                reconstructor->_weights[inputIndex].PutAsDouble(i, j, 0, weight);
                                
                                //calculate slice potentials
                                if(reconstructor->_simulated_weights[inputIndex](i,j,0)>0.99) {
                                    slice_potential[inputIndex] += (1 - weight) * (1 - weight);
                                    num++;
                                }
                            }
                            else
                                reconstructor->_weights[inputIndex].PutAsDouble(i, j, 0, 0);
                        }
                
                //evaluate slice potential
                if (num > 0)
                    slice_potential[inputIndex] = sqrt(slice_potential[inputIndex] / num);
                else
                    slice_potential[inputIndex] = -1; // slice has no unpadded voxels
            }
        }
        
        ParallelEStepardiacVelocity4D( ReconstructionCardiacVelocity4D *reconstructor, Array<double> &slice_potential ) :
        reconstructor(reconstructor), slice_potential(slice_potential)
        { }
        
        // execute
        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionCardiacVelocity4D::EStepVelocity4D()
    {
        //EStep performs calculation of voxel-wise and slice-wise posteriors (weights)
        if (_debug)
            cout << "EStep: " << endl;
        
        unsigned int inputIndex;
        RealImage slice, w, b, sim;
        int num = 0;
        Array<double> slice_potential(_slices.size(), 0);
        
        ParallelEStepardiacVelocity4D parallelEStep( this, slice_potential );
        parallelEStep();
        
        //To force-exclude slices predefined by a user, set their potentials to -1
        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            slice_potential[_force_excluded[i]] = -1;
        
        //exclude slices identified as having small overlap with ROI, set their potentials to -1
        for (unsigned int i = 0; i < _small_slices.size(); i++)
            slice_potential[_small_slices[i]] = -1;
        
        //these are unrealistic scales pointing at misregistration - exclude the corresponding slices
        for (inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
            if ((_scale[inputIndex]<0.2)||(_scale[inputIndex]>5)) {
                slice_potential[inputIndex] = -1;
            }
        
        // exclude unrealistic transformations
        if(_debug) {
            cout<<endl<<"Slice potentials: ";
            for (inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
                cout<<slice_potential[inputIndex]<<" ";
            cout<<endl;
        }
        
        
        //Calulation of slice-wise robust statistics parameters.
        //This is theoretically M-step,
        //but we want to use latest estimate of slice potentials
        //to update the parameters
        
        //Calculate means of the inlier and outlier potentials
        double sum = 0, den = 0, sum2 = 0, den2 = 0, maxs = 0, mins = 1;
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            if (slice_potential[inputIndex] >= 0) {
                //calculate means
                sum += slice_potential[inputIndex] * _slice_weight[inputIndex];
                den += _slice_weight[inputIndex];
                sum2 += slice_potential[inputIndex] * (1 - _slice_weight[inputIndex]);
                den2 += (1 - _slice_weight[inputIndex]);
                
                //calculate min and max of potentials in case means need to be initalized
                if (slice_potential[inputIndex] > maxs)
                    maxs = slice_potential[inputIndex];
                if (slice_potential[inputIndex] < mins)
                    mins = slice_potential[inputIndex];
            }
        
        if (den > 0)
            _mean_s = sum / den;
        else
            _mean_s = mins;
        
        if (den2 > 0)
            _mean_s2 = sum2 / den2;
        else
            _mean_s2 = (maxs + _mean_s) / 2;
        
        //Calculate the variances of the potentials
        sum = 0;
        den = 0;
        sum2 = 0;
        den2 = 0;
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            if (slice_potential[inputIndex] >= 0) {
                sum += (slice_potential[inputIndex] - _mean_s) * (slice_potential[inputIndex] - _mean_s) * _slice_weight[inputIndex];
                den += _slice_weight[inputIndex];
                
                sum2 += (slice_potential[inputIndex] - _mean_s2) * (slice_potential[inputIndex] - _mean_s2) * (1 - _slice_weight[inputIndex]);
                den2 += (1 - _slice_weight[inputIndex]);
                
            }
        
        //_sigma_s
        if ((sum > 0) && (den > 0)) {
            _sigma_s = sum / den;
            //do not allow too small sigma
            if (_sigma_s < _step * _step / 6.28)
                _sigma_s = _step * _step / 6.28;
        }
        else {
            _sigma_s = 0.025;
            if (_debug) {
                if (sum <= 0)
                    cout << "All slices are equal. ";
                if (den < 0) //this should not happen
                    cout << "All slices are outliers. ";
                cout << "Setting sigma to " << sqrt(_sigma_s) << endl;
            }
        }
        
        //sigma_s2
        if ((sum2 > 0) && (den2 > 0)) {
            _sigma_s2 = sum2 / den2;
            //do not allow too small sigma
            if (_sigma_s2 < _step * _step / 6.28)
                _sigma_s2 = _step * _step / 6.28;
        }
        else {
            _sigma_s2 = (_mean_s2 - _mean_s) * (_mean_s2 - _mean_s) / 4;
            //do not allow too small sigma
            if (_sigma_s2 < _step * _step / 6.28)
                _sigma_s2 = _step * _step / 6.28;
            
            if (_debug) {
                if (sum2 <= 0)
                    cout << "All slices are equal. ";
                if (den2 <= 0)
                    cout << "All slices inliers. ";
                cout << "Setting sigma_s2 to " << sqrt(_sigma_s2) << endl;
            }
        }
        
        //Calculate slice weights
        double gs1, gs2;
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            //Slice does not have any voxels in volumetric ROI
            if (slice_potential[inputIndex] == -1) {
                _slice_weight[inputIndex] = 0;
                continue;
            }
            
            //All slices are outliers or the means are not valid
            if ((den <= 0) || (_mean_s2 <= _mean_s)) {
                _slice_weight[inputIndex] = 1;
                continue;
            }
            
            //likelihood for inliers
            if (slice_potential[inputIndex] < _mean_s2)
                gs1 = G(slice_potential[inputIndex] - _mean_s, _sigma_s);
            else
                gs1 = 0;
            
            //likelihood for outliers
            if (slice_potential[inputIndex] > _mean_s)
                gs2 = G(slice_potential[inputIndex] - _mean_s2, _sigma_s2);
            else
                gs2 = 0;
            
            //calculate slice weight
            double likelihood = gs1 * _mix_s + gs2 * (1 - _mix_s);
            if (likelihood > 0)
                _slice_weight[inputIndex] = gs1 * _mix_s / likelihood;
            else {
                if (slice_potential[inputIndex] <= _mean_s)
                    _slice_weight[inputIndex] = 1;
                if (slice_potential[inputIndex] >= _mean_s2)
                    _slice_weight[inputIndex] = 0;
                if ((slice_potential[inputIndex] < _mean_s2) && (slice_potential[inputIndex] > _mean_s)) //should not happen
                    _slice_weight[inputIndex] = 1;
            }
        }
        
        //Update _mix_s this should also be part of MStep
        sum = 0;
        num = 0;
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            if (slice_potential[inputIndex] >= 0) {
                sum += _slice_weight[inputIndex];
                num++;
            }
        
        if (num > 0)
            _mix_s = sum / num;
        else {
            cout << "All slices are outliers. Setting _mix_s to 0.9." << endl;
            _mix_s = 0.9;
        }
        
        if (_debug) {
            cout << setprecision(3);
            cout << "Slice robust statistics parameters: ";
            cout << "means: " << _mean_s << " " << _mean_s2 << "  ";
            cout << "sigmas: " << sqrt(_sigma_s) << " " << sqrt(_sigma_s2) << "  ";
            cout << "proportions: " << _mix_s << " " << 1 - _mix_s << endl;
            cout << "Slice weights: ";
            for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                cout << _slice_weight[inputIndex] << " ";
            cout << endl;
        }
        
    }
    
    //-------------------------------------------------------------------
    
    
    class ParallelMStepCardiacVelocity4D{
        ReconstructionCardiacVelocity4D* reconstructor;
    public:
        double sigma;
        double mix;
        double num;
        double min;
        double max;
        
        void operator()( const blocked_range<size_t>& r ) {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                //alias the current weight image
                RealImage& w = reconstructor->_weights[inputIndex];
                
                //alias the current bias image
//                RealImage& b = reconstructor->_bias[inputIndex];
                
                //identify scale factor
//                double scale = reconstructor->_scale[inputIndex];
                
                
                // read the current simulated slice
                RealImage sim = reconstructor->_simulated_slices[inputIndex];
                RealImage sss = slice - sim;
                
                
                for ( int i = 0; i < slice.GetX(); i++ ) {
                    for ( int j = 0; j < slice.GetY(); j++ ) {
                        if (slice(i,j,0)<-10 )
                            sss(i,j,0) = -15;
                        else
                            sss(i,j,0) = (sss(i,j,0));
                    }
                }
                slice = sss;
                
                
                //calculate error
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -10) {
                            //bias correct and scale the slice
//                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                            
                            //otherwise the error has no meaning - it is equal to slice intensity
                            if ( reconstructor->_simulated_weights[inputIndex](i,j,0) > 0.99 ) {
                                
//                                slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                                
                                //sigma and mix
                                double e = slice(i, j, 0);
                                sigma += e * e * w(i, j, 0);
                                mix += w(i, j, 0);
                                
                                //_m
                                if (e < min)
                                    min = e;
                                if (e > max)
                                    max = e;
                                
                                num++;
                            }
                        }
            } //end of loop for a slice inputIndex
        }
        
        ParallelMStepCardiacVelocity4D( ParallelMStepCardiacVelocity4D& x, split ) :
        reconstructor(x.reconstructor)
        {
            sigma = 0;
            mix = 0;
            num = 0;
            min = 0;
            max = 0;
        }
        
        void join( const ParallelMStepCardiacVelocity4D& y ) {
            if (y.min < min)
                min = y.min;
            if (y.max > max)
                max = y.max;
            
            sigma += y.sigma;
            mix += y.mix;
            num += y.num;
        }
        
        ParallelMStepCardiacVelocity4D( ReconstructionCardiacVelocity4D *reconstructor ) :
        reconstructor(reconstructor)
        {
            sigma = 0;
            mix = 0;
            num = 0;
            min = voxel_limits<RealPixel>::max();
            max = voxel_limits<RealPixel>::min();
        }
        
        // execute
        void operator() () {
            parallel_reduce( blocked_range<size_t>(0, reconstructor->_slices.size()), *this );
        }
    };
    
    //-------------------------------------------------------------------
    
    void ReconstructionCardiacVelocity4D::MStepVelocity4D(int iter)
    {
        if (_debug)
            cout << "MStep" << endl;
        
        ParallelMStepCardiacVelocity4D parallelMStep(this);
        parallelMStep();
        double sigma = parallelMStep.sigma;
        double mix = parallelMStep.mix;
        double num = parallelMStep.num;
        double min = parallelMStep.min;
        double max = parallelMStep.max;
        
        //Calculate sigma and mix
        if (mix > 0) {
            _sigma = sigma / mix;
        }
        else {
            cerr << "Something went wrong: sigma=" << sigma << " mix=" << mix << endl;
            exit(1);
        }
        if (_sigma < _step * _step / 6.28)
            _sigma = _step * _step / 6.28;
        if (iter > 1)
            _mix = mix / num;
        
        //Calculate m
        _m = 1 / (max - min);
        
        if (_debug) {
            cout << "Voxel-wise robust statistics parameters: ";
            cout << "sigma = " << sqrt(_sigma) << " mix = " << _mix << " ";
            cout << " m = " << _m; //<< endl;
            cout << " max = " << max << " min = " << min << endl;
        }
        
    }
    
    //-------------------------------------------------------------------
    
    
    //-------------------------------------------------------------------
    
    
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
                    
                    
//                    if (reconstructor->_reconstructed4D.GetT() == 1) {
//                        reconstructor->_slice_temporal_weight[outputIndex][inputIndex] = 1;
//                    }
                    
                    
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
                                    
                                    if (slice(i, j, 0) > -10) {
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
     
     if (slice(i, j, 0) > -10) {
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
    
    
    
    
    
    //-------------------------------------------------------------------
    
    
    //-------------------------------------------------------------------
    
    
    //-------------------------------------------------------------------
    
    
    
} // namespace mirtk



