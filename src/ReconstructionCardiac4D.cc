/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2018-2020 King's College London
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

// MIRTK
#include "mirtk/Resampling.h"
#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageTransformation.h"
#include "mirtk/LinearInterpolateImageFunction.hxx"

// SVRTK
#include "svrtk/ReconstructionCardiac4D.h"
#include "svrtk/Profiling.h"
#include "svrtk/Parallel.h"

 // Boost
#include <boost/format.hpp>

// C++ Standard
#include <math.h>

using namespace std;
using namespace mirtk;

namespace svrtk {    

    // -----------------------------------------------------------------------------
    // Determine Reconstructed Spatial Resolution
    // -----------------------------------------------------------------------------
    //determine resolution of volume to reconstruct
    double ReconstructionCardiac4D::GetReconstructedResolutionFromTemplateStack( RealImage stack )
    {
        double dx, dy, dz, d;
        stack.GetPixelSize(&dx, &dy, &dz);
        if ((dx <= dy) && (dx <= dz))
            d = dx;
        else if (dy <= dz)
            d = dy;
        else
            d = dz;
        return d;
    }
    
    
    // -----------------------------------------------------------------------------
    // Get Slice-Location Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::ReadSliceTransformation(const char *folder) {
        cout << "Reading slice transformations from: " << folder << endl;
        Array<RigidTransformation> loc_transformations;
        ReadTransformations(folder, _loc_index.back() + 1, loc_transformations);

        // Assign transformations to single-frame images
        _transformations.clear();
        for (size_t i = 0; i < _slices.size(); i++)
            _transformations.push_back(loc_transformations[_loc_index[i]]);
    }

    // -----------------------------------------------------------------------------
    // Initialise Reconstructed Volume from Static Mask
    // -----------------------------------------------------------------------------
    // Create zero image as a template for reconstructed volume
    void ReconstructionCardiac4D::CreateTemplateCardiac4DFromStaticMask( RealImage mask, double resolution )
    {
        // Get mask attributes - image size and voxel size
        ImageAttributes attr = mask.Attributes();
        
        // Set temporal dimension
        attr._t = _reconstructed_cardiac_phases.size();
        attr._dt = _reconstructed_temporal_resolution;
        
        // Create volume
        RealImage volume4D(attr);
        
        // Initialise volume values
        volume4D = 0;
        
        // Resample to specified resolution
        InterpolationMode interpolation = Interpolation_NN;
        UniquePtr<InterpolateImageFunction> interpolator;
        interpolator.reset(InterpolateImageFunction::New(interpolation));
        
        Resampling<RealPixel> resampling(resolution,resolution,resolution);
        resampling.Input(&volume4D);
        resampling.Output(&volume4D);
        resampling.Interpolator(interpolator.get());
        resampling.Run();
        
        // Set recontructed 4D volume
        _reconstructed4D = volume4D;
        _template_created = true;

        
        // Set reconstructed 3D volume for reference by existing functions of irtkReconstruction class
        ImageAttributes attr4d = volume4D.Attributes();
        ImageAttributes attr3d = attr4d;
        attr3d._t = 1;
        RealImage volume3D(attr3d);
        _reconstructed = volume3D;
        
        
        // Debug
        if (_debug)
        {
            cout << "CreateTemplateCardiac4DFromStaticMask: created template 4D volume with "<<attr4d._t<<" time points and "<<attr4d._dt<<" s temporal resolution."<<endl;
        }
    }
    
    
    // -----------------------------------------------------------------------------
    // Match Stack Intensities With Masking
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::MatchStackIntensitiesWithMasking(Array<RealImage>& stacks,
                                                                   Array<RigidTransformation>& stack_transformations, double averageValue, bool together)
    {
        if (_debug)
            cout << "Matching intensities of stacks. ";
        
        cout<<setprecision(6);
        
        //Calculate the averages of intensities for all stacks
        double sum, num;
        char buffer[256];
        unsigned int ind;
        int i, j, k;
        double x, y, z;
        Array<double> stack_average;
        RealImage m;
        
        //remember the set average value
        _average_value = averageValue;
        
        //averages need to be calculated only in ROI
        for (ind = 0; ind < stacks.size(); ind++) {
            ImageAttributes attr = stacks[ind].Attributes();
            attr._t = 1;
            m.Initialize(attr);
            m = 0;
            sum = 0;
            num = 0;
            for (i = 0; i < stacks[ind].GetX(); i++)
                for (j = 0; j < stacks[ind].GetY(); j++)
                    for (k = 0; k < stacks[ind].GetZ(); k++) {
                        //image coordinates of the stack voxel
                        x = i;
                        y = j;
                        z = k;
                        //change to world coordinates
                        stacks[ind].ImageToWorld(x, y, z);
                        //transform to template (and also _mask) space
                        stack_transformations[ind].Transform(x, y, z);
                        //change to mask image coordinates - mask is aligned with template
                        _mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);
                        //if the voxel is inside mask ROI include it
                        if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0)
                            && (z < _mask.GetZ()))
                        {
                            if (_mask(x, y, z) == 1)
                            {
                                m(i,j,k)=1;
                                for ( int f = 0; f < stacks[ind].GetT(); f++)
                                {
                                    sum += stacks[ind](i, j, k, f);
                                    num++;
                                }
                            }
                        }
                    }
            if(_debug)
            {
                sprintf(buffer,"maskformatching%03i.nii.gz",ind);
                m.Write(buffer);
            }
            //calculate average for the stack
            if (num > 0)
                stack_average.push_back(sum / num);
            else {
                cerr << "Stack " << ind << " has no overlap with ROI" << endl;
                exit(1);
            }
        }
        
        double global_average;
        if (together) {
            global_average = 0;
            for(ind=0;ind<stack_average.size();ind++)
                global_average += stack_average[ind];
            global_average/=stack_average.size();
        }
        
        if (_debug) {
            cout << "Stack average intensities are ";
            for (ind = 0; ind < stack_average.size(); ind++)
                cout << stack_average[ind] << " ";
            cout << endl;
            cout << "The new average value is " << averageValue << endl;
        }
        
        //Rescale stacks
        RealPixel *ptr;
        double factor;
        for (ind = 0; ind < stacks.size(); ind++) {
            if (together) {
                factor = averageValue / global_average;
                _stack_factor.push_back(factor);
            }
            else {
                factor = averageValue / stack_average[ind];
                _stack_factor.push_back(factor);
                
            }
            
            ptr = stacks[ind].Data();
            for (i = 0; i < stacks[ind].NumberOfVoxels(); i++) {
                if (*ptr > 0)
                    *ptr *= factor;
                ptr++;
            }
        }
        
        if (_debug) {
            for (ind = 0; ind < stacks.size(); ind++) {
                sprintf(buffer, "rescaledstack%03i.nii.gz", ind);
                stacks[ind].Write(buffer);
            }
            
            cout << "Slice intensity factors are ";
            for (ind = 0; ind < stack_average.size(); ind++)
                cout << _stack_factor[ind] << " ";
            cout << endl;
            cout << "The new average value is " << averageValue << endl;
        }

        cout << setprecision(3);
    }

    // -----------------------------------------------------------------------------
    // Create Slices and Associated Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CreateSlicesAndTransformationsCardiac4D( Array<RealImage> &stacks,
                                                                          Array<RigidTransformation> &stack_transformations,
                                                                          Array<double> &thickness,
                                                                          const Array<RealImage> &probability_maps )
    {
        
        double sliceAcqTime;
        int loc_index = 0;
        
        if (_debug)
            cout << "CreateSlicesAndTransformations" << endl;
        
        //for each stack
        for (unsigned int i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            ImageAttributes attr = stacks[i].Attributes();
            
            //attr._z is number of slices in the stack
            for (int j = 0; j < attr._z; j++) {
                
                //attr._t is number of frames in the stack
                for (int k = 0; k < attr._t; k++) {
                    
                    //create slice by selecting the appropreate region of the stack
                    RealImage slice = stacks[i].GetRegion(0, 0, j, k, attr._x, attr._y, j + 1, k + 1);
                    //set correct voxel size in the stack. Z size is equal to slice thickness.
                    slice.PutPixelSize(attr._dx, attr._dy, thickness[i], attr._dt);
                    //set slice acquisition time
                    sliceAcqTime = attr._torigin + k * attr._dt;
                    _slice_time.push_back(sliceAcqTime);
                    //set slice temporal resolution
                    _slice_dt.push_back(attr._dt);
                    //remember the slice
                    _slices.push_back(slice);
                    _simulated_slices.push_back(slice);
                    _simulated_weights.push_back(slice);
                    _simulated_inside.push_back(slice);
                    _zero_slices.push_back(1);
                    //remeber stack indices for this slice
                    _stack_index.push_back(i);
                    _loc_index.push_back(loc_index);
                    _stack_loc_index.push_back(j);
                    _stack_dyn_index.push_back(k);
                    //initialize slice transformation with the stack transformation
                    _transformations.push_back(stack_transformations[i]);
                    //initialise slice exclusion flags
                    _slice_excluded.push_back(0);
                    if ( probability_maps.size() > 0 ) {
                        RealImage proba = probability_maps[i].GetRegion(0, 0, j, k, attr._x, attr._y, j + 1, k + 1);
                        proba.PutPixelSize(attr._dx, attr._dy, thickness[i], attr._dt);
                        _probability_maps.push_back(proba);
                    }
                    //initialise cardiac phase to use for 2D-3D registration
                    _slice_svr_card_index.push_back(0);
                }
                loc_index++;
            }
        }
        cout << "Number of images: " << _slices.size() << endl;
        //set excluded slices
        for (size_t i = 0; i < _force_excluded.size(); i++)
            _slice_excluded[_force_excluded[i]] = true;
        for (size_t i = 0; i < _force_excluded_stacks.size(); i++)
            for (size_t j = 0; j < _slices.size(); j++)
                if (_force_excluded_stacks[i] == _stack_index[j])
                    _slice_excluded[j] = true;
        for (size_t i = 0; i < _force_excluded_locs.size(); i++)
            for (size_t j = 0; j < _slices.size(); j++)
                if (_force_excluded_locs[i] == _loc_index[j])
                    _slice_excluded[j] = true;
    }

    // -----------------------------------------------------------------------------
    // Calculate Slice Temporal Weights
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateSliceTemporalWeights()
    {
        if (_debug)
            cout << "CalculateSliceTemporalWeights" << endl;
        InitSliceTemporalWeights();
        
        
        for (unsigned int outputIndex = 0; outputIndex < _reconstructed_cardiac_phases.size(); outputIndex++)
        {
            for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            {
                _slice_temporal_weight[outputIndex][inputIndex] = CalculateTemporalWeight( _reconstructed_cardiac_phases[outputIndex], _slice_cardphase[inputIndex], _slice_dt[inputIndex], _slice_rr[inputIndex], _wintukeypct );
                
               // option for time window thresholding for velocity reconstruction
               if (_no_ts) {
                   if (_slice_temporal_weight[outputIndex][inputIndex] < 0.9)
                       _slice_temporal_weight[outputIndex][inputIndex] = 0;
               }

            }
        }

        
    }

    // -----------------------------------------------------------------------------
    // Calculate Temporal Weight
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateTemporalWeight( double cardphase0, double cardphase, double dt, double rr, double alpha )
    {
        double angdiff, dtrad, sigma, temporalweight;
        
        // Angular Difference
        angdiff = CalculateAngularDifference( cardphase0, cardphase );
        
        // Temporal Resolution in Radians
        dtrad = 2 * PI * dt / rr;
        
        // Temporal Weight
        if (_is_temporalpsf_gauss) {
            // Gaussian
            sigma = dtrad / 2.355;  // sigma = ~FWHM/2.355
            temporalweight = exp( -( angdiff * angdiff ) / (2 * sigma * sigma ) );
        }
        else {
            // Sinc
            temporalweight = sinc( PI * angdiff / dtrad ) * wintukey( angdiff, alpha );
        }
        
        return temporalweight;
    }

    // -----------------------------------------------------------------------------
    // Calculate Transformation Matrix Between Slices and Voxels
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CoeffInitCardiac4D()
    {
        if (_debug)
            cout << "CoeffInit" << endl;
        
        
        //clear slice-volume matrix from previous iteration
        _volcoeffs.clear();
        _volcoeffs.resize(_slices.size());
        
        //clear indicator of slice having and overlap with volumetric mask
        _slice_inside.clear();
        _slice_inside.resize(_slices.size());
        
        cout << "Initialising matrix coefficients...";
        cout.flush();
        Parallel::CoeffInitCardiac4D coeffinit(this);
        coeffinit();
        cout << " ... done." << endl;
        
        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        cout << "Computing 4D volume weights..." << endl;
        ImageAttributes volAttr = _reconstructed4D.Attributes();
        _volume_weights.Initialize( volAttr );
        _volume_weights = 0;
        
        
        
        // TODO: investigate if this loop is taking a long time to compute, and consider parallelisation
        int i, j, n, k, outputIndex;
        unsigned int inputIndex;
        POINT3D p;
        cout << "    ... for input slice: ";
        

        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            cout << inputIndex << ", ";
            cout.flush();
            for ( i = 0; i < _slices[inputIndex].GetX(); i++)
                for ( j = 0; j < _slices[inputIndex].GetY(); j++) {
                    n = _volcoeffs[inputIndex][i][j].size();
                    for (k = 0; k < n; k++) {
                        p = _volcoeffs[inputIndex][i][j][k];
                        
                        for (outputIndex=0; outputIndex<_reconstructed4D.GetT(); outputIndex++)
                        {
                            _volume_weights(p.x, p.y, p.z, outputIndex) += _slice_temporal_weight[outputIndex][inputIndex] * p.value;
                            
                        }
                    }
                }
        }
        cout << "\b\b." << endl;
        // if (_debug)
        //     _volume_weights.Write("volume_weights.nii.gz");
        

        //find average volume weight to modify alpha parameters accordingly
        double sum = 0;
        int num=0;
        for (i=0; i<_volume_weights.GetX(); i++)
            for (j=0; j<_volume_weights.GetY(); j++)
                for (k=0; k<_volume_weights.GetZ(); k++)
                    if (_mask(i,j,k)==1)
                        for (int f=0; f<_volume_weights.GetT(); f++) {
                            sum += _volume_weights(i,j,k,f);
                            num++;
                        }
        
        _average_volume_weight = sum/num;
        
        if(_debug) {
            cout<<"Average volume weight is "<<_average_volume_weight<<endl;
        }
        
    }
    
    
    // -----------------------------------------------------------------------------
    // PSF-Weighted Reconstruction
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::GaussianReconstructionCardiac4D()
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
        
        //clear _reconstructed image
        _reconstructed4D = 0;
        
        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            
            if (_slice_excluded[inputIndex]==0) {
                
                if(_debug)
                {
                    cout << inputIndex << ", ";
                    cout.flush();
                }
                //copy the current slice
                slice = _slices[inputIndex];
                //alias the current bias image
                RealImage& b = _bias[inputIndex];
                //read current scale factor
                scale = _scale[inputIndex];
                
                slice_vox_num=0;
                
                //Distribute slice intensities to the volume
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //biascorrect and scale the slice
                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                            
                            //number of volume voxels with non-zero coefficients
                            //for current slice voxel
                            n = _volcoeffs[inputIndex][i][j].size();
                            
                            //if given voxel is not present in reconstructed volume at all,
                            //pad it
                            
                            //if (n == 0)
                            //_slices[inputIndex].PutAsDouble(i, j, 0, -1);
                            //calculate num of vox in a slice that have overlap with roi
                            if (n>0)
                                slice_vox_num++;
                            
                            //add contribution of current slice voxel to all voxel volumes
                            //to which it contributes
                            for (k = 0; k < n; k++) {
                                p = _volcoeffs[inputIndex][i][j][k];
                                for (outputIndex=0; outputIndex<_reconstructed_cardiac_phases.size(); outputIndex++)
                                {
                                    _reconstructed4D(p.x, p.y, p.z, outputIndex) += _slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0);
                                }
                            }
                        }
                voxel_num.push_back(slice_vox_num);
            } //end of if (_slice_excluded[inputIndex]==0)
        } //end of loop for a slice inputIndex
        
        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxe
        _reconstructed4D /= _volume_weights;
        
        if(_debug)
        {
            cout << inputIndex << "\b\b." << endl;
            cout << "... Gaussian reconstruction done." << endl << endl;
            cout.flush();
        }
        
        
         if (_debug)
             _reconstructed4D.Write("init.nii.gz");
        
        //now find slices with small overlap with ROI and exclude them.
        
        Array<int> voxel_num_tmp;
        for (unsigned int i=0;i<voxel_num.size();i++)
            voxel_num_tmp.push_back(voxel_num[i]);
        
        //find median
        sort(voxel_num_tmp.begin(),voxel_num_tmp.end());
        int median = voxel_num_tmp[round(voxel_num_tmp.size()*0.5)];
        
        //remember slices with small overlap with ROI
        _small_slices.clear();
        for (unsigned int i=0;i<voxel_num.size();i++)
            if (voxel_num[i]<0.1*median)
                _small_slices.push_back(i);
        
        if (_debug) {
            cout<<"Small slices:";
            for (unsigned int i=0;i<_small_slices.size();i++)
                cout<<" "<<_small_slices[i];
            cout<<endl;
        }

    }

    // -----------------------------------------------------------------------------
    // Calculate Corrected Slices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateCorrectedSlices()
    {
        if (_debug)
            cout<<"Calculating corrected slices..."<<endl;
        
        Parallel::CalculateCorrectedSlices parallelCalculateCorrectedSlices(this);
        parallelCalculateCorrectedSlices();
        
        if (_debug)
            cout<<"\t...calculating corrected slices done."<<endl;
    }
    
    
    // -----------------------------------------------------------------------------
    // Simulate Slices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SimulateSlicesCardiac4D()
    {
        if (_debug)
            cout<<"Simulating Slices..."<<endl;
        
        Parallel::SimulateSlicesCardiac4D parallelSimulateSlices(this);
        parallelSimulateSlices();
        
        if (_debug)
            cout<<"\t...Simulating Slices done."<<endl;
    }
    
    
    // -----------------------------------------------------------------------------
    // Simulate Stacks
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SimulateStacksCardiac4D(Array<bool> stack_excluded)
    {
        
        cout << "SimulateStacksCardiac4D" << endl;
        
        //Initialise simlated images
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex)    _simulated_slices[inputIndex] = 0;
        
        //Initialise indicator of images having overlap with volume
        _slice_inside.clear();
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex)
            _slice_inside.push_back(true);
        
        //Simulate images
        cout << "Simulating...";
        cout.flush();
        Parallel::SimulateStacksCardiac4D simulatestacks(this);
        simulatestacks();
        cout << " ... done." << endl;
        
    }
    
    
    
    // -----------------------------------------------------------------------------
    // Calculate Error
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateError()
    {
        if (_debug)
            cout<<"Calculating error..."<<endl;
        
        Parallel::CalculateError parallelCalculateError(this);
        parallelCalculateError();
        
        if (_debug)
            cout<<"\t...calculating error done."<<endl;
    }
    
    
    // -----------------------------------------------------------------------------
    // Normalise Bias
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::NormaliseBiasCardiac4D(int iter, int rec_iter)
    {
        if(_debug)
            cout << "Normalise Bias ... ";
        
        Parallel::NormaliseBiasCardiac4D parallelNormaliseBias(this);
        parallelNormaliseBias();
        RealImage bias = parallelNormaliseBias.bias;
        RealImage volweight3d = parallelNormaliseBias.volweight3d;
        
        // normalize the volume by proportion of contributing slice voxels for each volume voxel
        bias /= volweight3d;
        
        if(_debug)
            cout << "done." << endl;
        
        bias = StaticMaskVolume4D(bias,0);
        RealImage m = _mask;
        GaussianBlurring<RealPixel> gb(_sigma_bias);
        gb.Input(&bias);
        gb.Output(&bias);
        gb.Run();
        gb.Input(&m);
        gb.Output(&m);
        gb.Run();
        
        bias/=m;
        
        if (_debug) {
            char buffer[256];
            sprintf(buffer,"averagebias_mc%02isr%02i.nii.gz",iter,rec_iter);
            bias.Write(buffer);
        }
        
        for ( int i = 0; i < _reconstructed4D.GetX(); i++)
            for ( int j = 0; j < _reconstructed4D.GetY(); j++)
                for ( int k = 0; k < _reconstructed4D.GetZ(); k++)
                    for ( int f = 0; f < _reconstructed4D.GetT(); f++)
                        if(_reconstructed4D(i,j,k,f)!=-1)
                            _reconstructed4D(i,j,k,f) /=exp(-bias(i,j,k));
        
    }
    
    
    // -----------------------------------------------------------------------------
    // Calculate Target Cardiac Phase in Reconstructed Volume for Slice-To-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateSliceToVolumeTargetCardiacPhase()
    {
        int card_index;
        double angdiff;
        if (_debug)
            cout << "CalculateSliceToVolumeTargetCardiacPhase" << endl;
        _slice_svr_card_index.clear();
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            angdiff = PI + 0.001;  // NOTE: init angdiff larger than any possible calculated angular difference
            card_index = -1;
            for (unsigned int outputIndex = 0; outputIndex < _reconstructed_cardiac_phases.size(); outputIndex++)
            {
                if ( fabs( CalculateAngularDifference( _reconstructed_cardiac_phases[outputIndex], _slice_cardphase[inputIndex] ) ) < angdiff)
                {
                    angdiff = fabs( CalculateAngularDifference( _reconstructed_cardiac_phases[outputIndex], _slice_cardphase[inputIndex] ) );
                    card_index = outputIndex;
                }
            }
            _slice_svr_card_index.push_back(card_index);
            // if (_debug)
            //   cout << inputIndex << ":" << _slice_svr_card_index[inputIndex] << ", ";
        }
        // if (_debug)
        //   cout << "\b\b." << endl;
    }

    // -----------------------------------------------------------------------------
    // Slice-to-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SliceToVolumeRegistrationCardiac4D()
    {
        
        if (_debug)
            cout << "SliceToVolumeRegistrationCardiac4D" << endl;
        Parallel::SliceToVolumeRegistrationCardiac4D registration(this);
        registration();
    }

    // -----------------------------------------------------------------------------
    // Remote Slice-to-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::RemoteSliceToVolumeRegistrationCardiac4D(int iter, string str_mirtk_path, string str_current_main_file_path, string str_current_exchange_file_path)
    {
        
        ImageAttributes attr_recon = _reconstructed4D.Attributes();
        
        if (_debug)
            cout << "RemoteSliceToVolumeRegistrationCardiac4D" << endl;
        
        for (int t=0; t<_reconstructed4D.GetT(); t++) {
        
            string str_source = str_current_exchange_file_path + "/current-source-" + to_string(t) + ".nii.gz";
            RealImage source = _reconstructed4D.GetRegion( 0, 0, 0, t, attr_recon._x, attr_recon._y, attr_recon._z, (t+1));
            source.Write(str_source.c_str());
            
        }
        
        if (iter == 1) {
            
            _offset_matrices.clear();
            
            for (int inputIndex=0; inputIndex<_slices.size(); inputIndex++) {
                
                ResamplingWithPadding<RealPixel> resampling(attr_recon._dx, attr_recon._dx, attr_recon._dx, -1);
                GenericLinearInterpolateImageFunction<RealImage> interpolator;
                
                RealImage target = _slices[inputIndex];
                resampling.Input(&_slices[inputIndex]);
                resampling.Output(&target);
                resampling.Interpolator(&interpolator);
                resampling.Run();
                
                // put origin to zero
                RigidTransformation offset;
                ResetOrigin(target, offset);
                
                
                RealPixel tmin, tmax;
                target.GetMinMax(&tmin, &tmax);
                if (tmax > 1 && ((tmax-tmin) > 1) ) {
                    _zero_slices[inputIndex] = 1;
                } else {
                    _zero_slices[inputIndex] = -1;
                }
                
                
                string str_target = str_current_exchange_file_path + "/res-slice-" + to_string(inputIndex) + ".nii.gz";
                target.Write(str_target.c_str());
                
                Matrix mo = offset.GetMatrix();
                _offset_matrices.push_back(mo);

            }
        }
        
        
        for (int inputIndex=0; inputIndex<_slices.size(); inputIndex++) {
            
            RigidTransformation r_transform = _transformations[inputIndex];
            Matrix m = r_transform.GetMatrix();
            m=m*_offset_matrices[inputIndex];
            r_transform.PutMatrix(m);
            
            string str_dofin = str_current_exchange_file_path + "/res-transformation-" + to_string(inputIndex) + ".dof";
            r_transform.Write(str_dofin.c_str());

        }
        
        
        int stride = 32;
        int svr_range_start = 0;
        int svr_range_stop = svr_range_start + stride;

        while (svr_range_start < _slices.size()) {
            Parallel::RemoteSliceToVolumeRegistration registration(this, svr_range_start, svr_range_stop, str_mirtk_path, str_current_exchange_file_path, true, _slice_svr_card_index);
            registration();
            
            svr_range_start = svr_range_stop;
            svr_range_stop = svr_range_start + stride;
            
            if (svr_range_stop > _slices.size()) {
                svr_range_stop = _slices.size();
            }
        
        }
        
        for (int inputIndex=0; inputIndex<_slices.size(); inputIndex++) {

            string str_dofout = str_current_exchange_file_path + "/res-transformation-" + to_string(inputIndex) + ".dof";

            Transformation *tmp_transf = Transformation::New(str_dofout.c_str());
            RigidTransformation *tmp_r_transf = dynamic_cast<RigidTransformation*> (tmp_transf);
            _transformations[inputIndex] = *tmp_r_transf;

            //undo the offset
            Matrix mo = _offset_matrices[inputIndex];
            mo.Invert();
            Matrix m = _transformations[inputIndex].GetMatrix();
            m=m*mo;
            _transformations[inputIndex].PutMatrix(m);

        }
        
        
        
    }
    
    
    // -----------------------------------------------------------------------------
    // Volume-to-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::VolumeToVolumeRegistration(GreyImage target, GreyImage source, RigidTransformation& rigidTransf)
    {
        
        
        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        Insert(params, "Image (dis-)similarity measure", "NMI");
        if (_nmi_bins>0)
            Insert(params, "No. of bins", _nmi_bins);
        Insert(params, "Background value for image 1", -1);
        Insert(params, "Background value for image 2", -1);
        
        GenericRegistrationFilter registration;
        registration.Parameter(params);
        registration.Input(&target, &source);
        
        
        Transformation *dofout = nullptr;
        registration.Output(&dofout);
        
        RigidTransformation dofin = rigidTransf;
        registration.InitialGuess(&dofin);
        
        registration.GuessParameter();
        registration.Run();
        
        RigidTransformation *rigidTransf_dofout = dynamic_cast<RigidTransformation*> (dofout);
        rigidTransf = *rigidTransf_dofout;

    // -----------------------------------------------------------------------------
    // Common Calculate Function
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::Calculate(const RigidTransformation& drift, double (ReconstructionCardiac4D::*Calculate)()) {
        //Initialise
        Matrix d = drift.GetMatrix();

        //Remove drift
        for (size_t i = 0; i < _transformations.size(); i++)
            _transformations[i].PutMatrix(d * _transformations[i].GetMatrix());

        //Calculate
        const double mean = (this->*Calculate)();

        //Return drift
        d.Invert();
        for (size_t i = 0; i < _transformations.size(); i++)
            _transformations[i].PutMatrix(d * _transformations[i].GetMatrix());

        //Output
        return mean;
    }

    // -----------------------------------------------------------------------------
    // Calculate Displacement
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateDisplacement()
    {
        
        if (_debug)
            cout << "CalculateDisplacment" << endl;
        
        _slice_displacement.clear();
        _slice_tx.clear();
        _slice_ty.clear();
        _slice_tz.clear();
        double x,y,z,tx,ty,tz;
        double disp_sum_slice, disp_sum_total = 0;
        double tx_sum_slice, ty_sum_slice, tz_sum_slice = 0;
        int num_voxel_slice, num_voxel_total = 0;
        int k = 0;
        double slice_disp, mean_disp;
        double tx_slice, ty_slice, tz_slice;
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            disp_sum_slice = 0;
            num_voxel_slice = 0;
            slice_disp = -1;
            tx_slice = 0;
            ty_slice = 0;
            tz_slice = 0;
            tx_sum_slice = 0;
            ty_sum_slice = 0;
            tz_sum_slice = 0;
            
            if (_slice_excluded[inputIndex]==0) {
                for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                        if (_slices[inputIndex](i,j,k)!=-1) {
                            x = i;
                            y = j;
                            z = k;
                            _slices[inputIndex].ImageToWorld(x,y,z);
                            tx = x; ty = y; tz = z;
                            _transformations[inputIndex].Transform(tx,ty,tz);
                            disp_sum_slice += sqrt((tx-x)*(tx-x)+(ty-y)*(ty-y)+(tz-z)*(tz-z));
                            tx_sum_slice += tx-x;
                            ty_sum_slice += ty-y;
                            tz_sum_slice += tz-z;
                            num_voxel_slice += 1;
                        }
                    }
                }
                if ( num_voxel_slice>0 ) {
                    slice_disp = disp_sum_slice / num_voxel_slice;
                    tx_slice = tx_sum_slice / num_voxel_slice;
                    ty_slice = ty_sum_slice / num_voxel_slice;
                    tz_slice = tz_sum_slice / num_voxel_slice;
                    disp_sum_total += disp_sum_slice;
                    num_voxel_total += num_voxel_slice;
                }
            }
            
            _slice_displacement.push_back(slice_disp);
            _slice_tx.push_back(tx_slice);
            _slice_ty.push_back(ty_slice);
            _slice_tz.push_back(tz_slice);
        }

        return num_voxel_total > 0 ? disp_sum_total / num_voxel_total : -1;
    }

    // -----------------------------------------------------------------------------
    // Calculate Weighted Displacement
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateWeightedDisplacement()
    {
        
        if (_debug)
            cout << "CalculateWeightedDisplacement" << endl;
        
        _slice_weighted_displacement.clear();
        
        double x,y,z,tx,ty,tz;
        double disp_sum_slice, disp_sum_total = 0;
        double weight_slice, weight_total = 0;
        int k = 0;
        double slice_disp, mean_disp = 0;
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            disp_sum_slice = 0;
            weight_slice = 0;
            slice_disp = -1;
            
            if (_slice_excluded[inputIndex]==0) {
                for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                        if (_slices[inputIndex](i,j,k)!=-1) {
                            x = i;
                            y = j;
                            z = k;
                            _slices[inputIndex].ImageToWorld(x,y,z);
                            tx = x; ty = y; tz = z;
                            _transformations[inputIndex].Transform(tx,ty,tz);
                            disp_sum_slice += _slice_weight[inputIndex]*_weights[inputIndex](i,j,k)*sqrt((tx-x)*(tx-x)+(ty-y)*(ty-y)+(tz-z)*(tz-z));
                            weight_slice += _slice_weight[inputIndex]*_weights[inputIndex](i,j,k);
                        }
                    }
                }
                if (weight_slice>0) {
                    slice_disp = disp_sum_slice / weight_slice;
                    disp_sum_total += disp_sum_slice;
                    weight_total += weight_slice;
                }
            }
            
            _slice_weighted_displacement.push_back(slice_disp);
            
        }

        return weight_total > 0 ? disp_sum_total / weight_total : -1;
    }

    // -----------------------------------------------------------------------------
    // Calculate Target Registration Error
    // -----------------------------------------------------------------------------

    double ReconstructionCardiac4D::CalculateTRE() {
        if (_debug)
            cout << "CalculateTRE" << endl;
        
        _slice_tre.clear();
        
        double x,y,z,cx,cy,cz,px,py,pz;
        double tre_sum_slice, tre_sum_total = 0;
        int num_voxel_slice, num_voxel_total = 0;
        int k = 0;
        double slice_tre, mean_tre;
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            tre_sum_slice = 0;
            num_voxel_slice = 0;
            slice_tre = -1;
            
            if (_slice_excluded[inputIndex]==0) {
                for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                        if (_slices[inputIndex](i,j,k)!=-1) {
                            x = i; y = j; z = k;
                            _slices[inputIndex].ImageToWorld(x,y,z);
                            cx = x; cy = y; cz = z;
                            px = x; py = y; pz = z;
                            _transformations[inputIndex].Transform(cx,cy,cz);
                            _ref_transformations[inputIndex].Transform(px,py,pz);
                            tre_sum_slice += sqrt((cx-px)*(cx-px)+(cy-py)*(cy-py)+(cz-pz)*(cz-pz));
                            num_voxel_slice += 1;
                        }
                    }
                }
                if ( num_voxel_slice>0 ) {
                    slice_tre = tre_sum_slice / num_voxel_slice;
                    tre_sum_total += tre_sum_slice;
                    num_voxel_total += num_voxel_slice;
                }
            }
            
            _slice_tre.push_back(slice_tre);
        }

        return num_voxel_total > 0 ? tre_sum_total / num_voxel_total : -1;
    }

    // -----------------------------------------------------------------------------
    // Smooth Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SmoothTransformations(double sigma_seconds, int niter, bool use_slice_inside)
    {
        
        if (_debug)
            cout<<"SmoothTransformations"<<endl<<"\tsigma = "<<sigma_seconds<<" s"<<endl;
        
        int i,j,iter,par;
        
        //Reset origin for transformations
        GreyImage t = _reconstructed4D;
        RigidTransformation offset;
        ResetOrigin(t,offset);
        Matrix m;
        Matrix mo = offset.GetMatrix();
        Matrix imo = mo;
        imo.Invert();
        if (_debug)
            offset.Write("reset_origin.dof");
        for(i=0;i<int(_transformations.size());i++)
        {
            m = _transformations[i].GetMatrix();
            m=imo*m*mo;
            _transformations[i].PutMatrix(m);
        }
        
        //initial weights
        Matrix parameters(6,_transformations.size());
        Matrix weights(6,_transformations.size());
        ofstream fileOut("motion.txt", ofstream::out | ofstream::app);
        
        for(i=0;i<int(_transformations.size());i++)
        {
            parameters(0,i)=_transformations[i].GetTranslationX();
            parameters(1,i)=_transformations[i].GetTranslationY();
            parameters(2,i)=_transformations[i].GetTranslationZ();
            parameters(3,i)=_transformations[i].GetRotationX();
            parameters(4,i)=_transformations[i].GetRotationY();
            parameters(5,i)=_transformations[i].GetRotationZ();
            
            //write unprocessed parameters to file
            for(j=0;j<6;j++)
            {
                fileOut<<parameters(j,i);
                if(j<5)
                    fileOut<<",";
                else
                    fileOut<<endl;
            }
            
            for(j=0;j<6;j++)
                weights(j,i)=0;             //initialise as zero
            if (!_slice_excluded[i])
            {
                if (!use_slice_inside)      //set weights based on image intensities
                {
                    RealPixel smin, smax;
                    _slices[i].GetMinMax(&smin,&smax);
                    if(smax>-1)
                        for(j=0;j<6;j++)
                            weights(j,i)=1;
                }
                else                        //set weights based on _slice_inside
                {
                    if(_slice_inside.size()>0)
                    {
                        if(_slice_inside[i])
                            for(j=0;j<6;j++)
                                weights(j,i)=1;
                    }
                }
            }
        }
        
        //initialise
        Matrix den(6,_transformations.size());
        Matrix num(6,_transformations.size());
        Matrix kr(6,_transformations.size());
        Array<double> error,tmp,kernel;
        double median;
        double sigma;
        int nloc = 0;
        for(i=0;i<int(_transformations.size());i++)
            if ((_loc_index[i]+1)>nloc)
                nloc = _loc_index[i] + 1;
        if (_debug)
            cout << "\tnumber of slice-locations = " << nloc << endl;
        int dim = _transformations.size()/nloc; // assuming equal number of dynamic images for every slice-location
        int loc;
        
        //step size for sampling volume in error calculation in kernel regression
        int step;
        double nstep=15;
        step = ceil(_reconstructed4D.GetX()/nstep);
        if(step>ceil(_reconstructed4D.GetY()/nstep))
            step = ceil(_reconstructed4D.GetY()/nstep);
        if(step>ceil(_reconstructed4D.GetZ()/nstep))
            step = ceil(_reconstructed4D.GetZ()/nstep);
        
        //kernel regression
        for(iter = 0; iter<niter;iter++)
        {
            
            for(loc=0;loc<nloc;loc++)
            {
                
                //gaussian kernel for current slice-location
                kernel.clear();
                sigma = ceil( sigma_seconds / _slice_dt[loc*dim] );
                for(j=-3*sigma; j<=3*sigma; j++)
                {
                    kernel.push_back(exp(-(j*_slice_dt[loc*dim]/sigma_seconds)*(j*_slice_dt[loc*dim]/sigma_seconds)));
                }
                
                //kernel-weighted summation
                for(par=0;par<6;par++)
                {
                    for(i=loc*dim;i<(loc+1)*dim;i++)
                    {
                        if(!_slice_excluded[i])
                        {
                            for(j=-3*sigma;j<=3*sigma;j++)
                                if(((i+j)>=loc*dim) && ((i+j)<(loc+1)*dim))
                                {
                                    num(par,i)+=parameters(par,i+j)*kernel[j+3*sigma]*weights(par,i+j);
                                    den(par,i)+=kernel[j+3*sigma]*weights(par,i+j);
                                }
                        }
                        else
                        {
                            num(par,i)=parameters(par,i);
                            den(par,i)=1;
                        }
                    }
                }
            }
            
            //kernel-weighted normalisation
            for(par=0;par<6;par++)
                for(i=0;i<int(_transformations.size());i++)
                    kr(par,i)=num(par,i)/den(par,i);
            
            //recalculate weights using target registration error with original transformations as targets
            error.clear();
            tmp.clear();
            for(i=0;i<int(_transformations.size());i++)
            {
                
                RigidTransformation processed;
                processed.PutTranslationX(kr(0,i));
                processed.PutTranslationY(kr(1,i));
                processed.PutTranslationZ(kr(2,i));
                processed.PutRotationX(kr(3,i));
                processed.PutRotationY(kr(4,i));
                processed.PutRotationZ(kr(5,i));
                
                RigidTransformation orig = _transformations[i];
                
                //need to convert the transformations back to the original coordinate system
                m = orig.GetMatrix();
                m=mo*m*imo;
                orig.PutMatrix(m);
                
                m = processed.GetMatrix();
                m=mo*m*imo;
                processed.PutMatrix(m);
                
                RealImage slice = _slices[i];
                
                int n=0;
                double e = 0;
                
                if(!_slice_excluded[i]) {
                    for(int ii=0;ii<_reconstructed4D.GetX();ii=ii+step)
                        for(int jj=0;jj<_reconstructed4D.GetY();jj=jj+step)
                            for(int kk=0;kk<_reconstructed4D.GetZ();kk=kk+step)
                                if(_reconstructed4D(ii,jj,kk,0)>-1)
                                {
                                    double x = ii, y = jj, z = kk;
                                    _reconstructed4D.ImageToWorld(x,y,z);
                                    double xx = x, yy = y, zz = z;
                                    orig.Transform(x,y,z);
                                    processed.Transform(xx,yy,zz);
                                    x-=xx;
                                    y-=yy;
                                    z-=zz;
                                    e += sqrt(x*x+y*y+z*z);
                                    n++;
                                }
                }
                
                if(n>0)
                {
                    e/=n;
                    error.push_back(e);
                    tmp.push_back(e);
                }
                else
                    error.push_back(-1);
            }
            
            sort(tmp.begin(),tmp.end());
            median = tmp[round(tmp.size()*0.5)];
            
            if ((_debug)&(iter==0))
                cout<<"\titeration:median_error(mm)...";
            if (_debug) {
                cout<<iter<<":"<<median<<", ";
                cout.flush();
            }
            
            for(i=0;i<int(_transformations.size());i++)
            {
                if((error[i]>=0)&(!_slice_excluded[i]))
                {
                    if(error[i]<=median*1.35)
                        for(par=0;par<6;par++)
                            weights(par,i)=1;
                    else
                        for(par=0;par<6;par++)
                            weights(par,i)=median*1.35/error[i];
                }
                else
                    for(par=0;par<6;par++)
                        weights(par,i)=0;
            }
            
        }
        
        if (_debug)
            cout<<"\b\b."<<endl;
        
        ofstream fileOut2("motion-processed.txt", ofstream::out | ofstream::app);
        ofstream fileOut3("weights.txt", ofstream::out | ofstream::app);
        ofstream fileOut4("outliers.txt", ofstream::out | ofstream::app);
        ofstream fileOut5("empty.txt", ofstream::out | ofstream::app);
        
        for(i=0;i<int(_transformations.size());i++)
        {
            fileOut3<<weights(j,i)<<" ";
            
            if(weights(0,i)<=0)
                fileOut5<<i<<" ";
            
            _transformations[i].PutTranslationX(kr(0,i));
            _transformations[i].PutTranslationY(kr(1,i));
            _transformations[i].PutTranslationZ(kr(2,i));
            _transformations[i].PutRotationX(kr(3,i));
            _transformations[i].PutRotationY(kr(4,i));
            _transformations[i].PutRotationZ(kr(5,i));
            
            for(j=0;j<6;j++)
            {
                fileOut2<<kr(j,i);
                if(j<5)
                    fileOut2<<",";
                else
                    fileOut2<<endl;
            }
        }
        
        //Put origin back
        for(i=0;i<int(_transformations.size());i++)
        {
            m = _transformations[i].GetMatrix();
            m=mo*m*imo;
            _transformations[i].PutMatrix(m);
        }
        
    }
    
    
    // -----------------------------------------------------------------------------
    // Scale Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::ScaleTransformations(double scale)
    {
        
        if (_debug)
            cout<<"Scaling transformations."<<endl<<"scale = "<<scale<<"."<<endl;
        
        if (scale==1)
            return;
        
        if (scale<0) {
            cerr<<"Scaling of transformations undefined for scale < 0.";
            exit(1);
        }
        
        unsigned int i;
        
        //Reset origin for transformations
        GreyImage t = _reconstructed4D;
        RigidTransformation offset;
        ResetOrigin(t,offset);
        Matrix m;
        Matrix mo = offset.GetMatrix();
        Matrix imo = mo;
        imo.Invert();
        if (_debug)
            offset.Write("reset_origin.dof");
        for(i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m = imo * m * mo;
            _transformations[i].PutMatrix(m);
        }
        
        //Scale transformations
        Matrix orig, scaled;
        int row, col;
        for(i=0;i<_transformations.size();i++)
        {
            orig = logm( _transformations[i].GetMatrix() );
            scaled = orig;
            for(row=0;row<4;row++)
                for(col=0;col<4;col++)
                    scaled(row,col) = scale * orig(row,col);
            _transformations[i].PutMatrix( expm( scaled ) );
        }
        
        //Put origin back
        for (size_t i = 0; i < _transformations.size(); i++)
            _transformations[i].PutMatrix(mo * _transformations[i].GetMatrix() * imo);
    }

    // -----------------------------------------------------------------------------
    // Apply Static Mask to 4D Volume
    // -----------------------------------------------------------------------------
    RealImage ReconstructionCardiac4D::StaticMaskVolume4D(RealImage volume, double padding)
    {
        for ( int i = 0; i < volume.GetX(); i++) {
            for ( int j = 0; j < volume.GetY(); j++) {
                for ( int k = 0; k < volume.GetZ(); k++) {
                    if ( _mask(i,j,k) == 0 ) {
                        for ( int t = 0; t < volume.GetT(); t++) {
                            volume(i,j,k,t) = padding;
                        }
                    }
                }
            }
        }
        return volume;
    }

    // -----------------------------------------------------------------------------
    // Super-Resolution of 4D Volume
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SuperresolutionCardiac4D( int iter )
    {
        if (_debug)
            cout << "Superresolution " << iter << endl;
        
        int i, j, k, t;
        RealImage addon, original;
        
        
        //Remember current reconstruction for edge-preserving smoothing
        original = _reconstructed4D;

        Parallel::SuperresolutionCardiac4D parallelSuperresolution(this);
        parallelSuperresolution();
        
        addon = parallelSuperresolution.addon;
        _confidence_map = parallelSuperresolution.confidence_map;
        
        if(_debug) {
            char buffer[256];
            sprintf(buffer,"confidence-map%i.nii.gz",iter);
            _confidence_map.Write(buffer);
            
            sprintf(buffer,"addon%i.nii.gz",iter);
            addon.Write(buffer);
        }
        
        if (!_adaptive)
            for (i = 0; i < addon.GetX(); i++)
                for (j = 0; j < addon.GetY(); j++)
                    for (k = 0; k < addon.GetZ(); k++)
                        for (t = 0; t < addon.GetT(); t++)
                            if (_confidence_map(i, j, k, t) > 0) {
                                // ISSUES if _confidence_map(i, j, k, t) is too small leading
                                // to bright pixels
                                addon(i, j, k, t) /= _confidence_map(i, j, k, t);
                                //this is to revert to normal (non-adaptive) regularisation
                                _confidence_map(i,j,k,t) = 1;
                            }
        
        _reconstructed4D += addon * _alpha; //_average_volume_weight;
        
        //bound the intensities
        for (i = 0; i < _reconstructed4D.GetX(); i++)
            for (j = 0; j < _reconstructed4D.GetY(); j++)
                for (k = 0; k < _reconstructed4D.GetZ(); k++)
                    for (t = 0; t < _reconstructed4D.GetT(); t++)
                    {
                        if (_reconstructed4D(i, j, k, t) < _min_intensity * 0.9)
                            _reconstructed4D(i, j, k, t) = _min_intensity * 0.9;
                        if (_reconstructed4D(i, j, k, t) > _max_intensity * 1.1)
                            _reconstructed4D(i, j, k, t) = _max_intensity * 1.1;
                    }
        
        //Smooth the reconstructed image
        AdaptiveRegularizationCardiac4D(iter, original);
        
        //Remove the bias in the reconstructed volume compared to previous iteration
        /* TODO: update adaptive regularisation for 4d
         if (_global_bias_correction)
         BiasCorrectVolume(original);
         */
    }

    // -----------------------------------------------------------------------------
    // Adaptive Regularization
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::AdaptiveRegularizationCardiac4D(int iter, RealImage& original)
    {
        if (_debug)
            cout << "AdaptiveRegularizationCardiac4D."<< endl;
        //cout << "AdaptiveRegularizationCardiac4D: _delta = "<<_delta<<" _lambda = "<<_lambda <<" _alpha = "<<_alpha<< endl;
        
        Array<double> factor(13,0);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }

        Array<RealImage> b(13, _reconstructed4D);

        Parallel::AdaptiveRegularization1Cardiac4D parallelAdaptiveRegularization1(this, b, factor, original);
        parallelAdaptiveRegularization1();
        
        RealImage original2 = _reconstructed4D;
        Parallel::AdaptiveRegularization2Cardiac4D parallelAdaptiveRegularization2(this, b, factor, original2);
        parallelAdaptiveRegularization2();

        if (_alpha * _lambda / (_delta * _delta) > 0.068)
            cerr << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068." << endl;
    }

    // -----------------------------------------------------------------------------
    // ReadRefTransformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::ReadRefTransformations(const char *folder) {
        cout << "Reading reference transformations from: " << folder << endl;
        ReadTransformations(folder, _slices.size(), _ref_transformations);
    }

    // -----------------------------------------------------------------------------
    // Calculate Entropy
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateEntropy()
    {
        
        double x;
        double sum_x_sq = 0;
        double x_max = 0;
        double entropy = 0;
        
        for ( int i = 0; i < _reconstructed4D.GetX(); i++)
            for ( int j = 0; j < _reconstructed4D.GetY(); j++)
                for ( int k = 0; k < _reconstructed4D.GetZ(); k++)
                    if ( _mask(i,j,k) == 1 )
                        for ( int f = 0; f < _reconstructed4D.GetT(); f++)
                        {
                            x = _reconstructed4D(i,j,k,f);
                            sum_x_sq += x*x;
                        }
        
        x_max = sqrt( sum_x_sq );
        
        for ( int i = 0; i < _reconstructed4D.GetX(); i++)
            for ( int j = 0; j < _reconstructed4D.GetY(); j++)
                for ( int k = 0; k < _reconstructed4D.GetZ(); k++)
                    if ( _mask(i,j,k) == 1 )
                        for ( int f = 0; f < _reconstructed4D.GetT(); f++)
                        {
                            x = _reconstructed4D(i,j,k,f);
                            if (x>0)
                                entropy += x/x_max * log( x/x_max );
                        }

        return -entropy;
    }

    // -----------------------------------------------------------------------------
    // Common save functions
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::Save(const Array<RealImage>& save_stacks, const char *message, const char *filename_prefix) {
        if (_verbose)
            _verbose_log << message;
        for (size_t i = 0; i < save_stacks.size(); i++)
            save_stacks[i].Write((boost::format("%s%05i.nii.gz") % filename_prefix % i).str().c_str());
        if (_verbose)
            _verbose_log << " done." << endl;
    }

    void ReconstructionCardiac4D::Save(const Array<RealImage>& source, const Array<RealImage>& stacks, int iter, int rec_iter, const char *message, const char *filename_prefix) {
        if (_verbose)
            _verbose_log << message;

        Array<RealImage> save_stacks;
        for (size_t i = 0; i < stacks.size(); i++)
            save_stacks.push_back(RealImage(stacks[i].Attributes()));

        for (size_t inputIndex = 0; inputIndex < source.size(); inputIndex++)
            for (int i = 0; i < source[inputIndex].GetX(); i++)
                for (int j = 0; j < source[inputIndex].GetY(); j++)
                    save_stacks[_stack_index[inputIndex]](i, j, _stack_loc_index[inputIndex], _stack_dyn_index[inputIndex]) = source[inputIndex](i, j, 0);

        for (size_t i = 0; i < stacks.size(); i++) {
            string filename;
            if (iter < 0 || rec_iter < 0)
                filename = (boost::format("%s%03i.nii.gz") % filename_prefix % i).str();
            else
                filename = (boost::format("%s%03i_mc%02isr%02i.nii.gz") % filename_prefix % i % iter % rec_iter).str();
            save_stacks[i].Write(filename.c_str());
        }

        if (_verbose)
            _verbose_log << " done." << endl;
    }

    // -----------------------------------------------------------------------------
    // SaveSlices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SaveSlices(const Array<RealImage>& stacks) {
        if (_verbose)
            _verbose_log << "Saving slices as stacks ... ";

        Array<RealImage> imagestacks;
        Array<RealImage> wstacks;

        for (size_t i = 0; i < stacks.size(); i++) {
            RealImage stack(stacks[i].Attributes());
            imagestacks.push_back(stack);
            wstacks.push_back(move(stack));
        }

        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            for (int i = 0; i < _slices[inputIndex].GetX(); i++)
                for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                    imagestacks[_stack_index[inputIndex]](i, j, _stack_loc_index[inputIndex], _stack_dyn_index[inputIndex]) = _slices[inputIndex](i, j, 0);
                    wstacks[_stack_index[inputIndex]](i, j, _stack_loc_index[inputIndex], _stack_dyn_index[inputIndex]) = 10 * _weights[inputIndex](i, j, 0);
                }

        for (size_t i = 0; i < stacks.size(); i++) {
            imagestacks[i].Write((boost::format("stack%03i.nii.gz") % i).str().c_str());
            wstacks[i].Write((boost::format("w%03i.nii.gz") % i).str().c_str());
        }

        if (_verbose)
            _verbose_log << "done." << endl;
    }

    // -----------------------------------------------------------------------------
    // SlicesInfo
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SlicesInfoCardiac4D( const char* filename,
                                                      Array<string> &stack_files )
    {
        ofstream info;
        info.open( filename );
        
        info<<setprecision(3);
        
        // header
        info << "StackIndex" << "\t"
        << "StackLocIndex" << "\t"
        << "StackDynIndex" << "\t"
        << "LocIndex" << "\t"
        << "InputIndex" << "\t"
        << "File" << "\t"
        << "Scale" << "\t"
        << "StackFactor" << "\t"
        << "Time" << "\t"
        << "TemporalResolution" << "\t"
        << "CardiacPhase" << "\t"
        << "ReconCardPhaseIndex" << "\t"
        << "Included" << "\t" // Included slices
        << "Excluded" << "\t"  // Excluded slices
        << "Outside" << "\t"  // Outside slices
        << "Weight" << "\t"
        << "MeanDisplacement" << "\t"
        << "MeanDisplacementX" << "\t"
        << "MeanDisplacementY" << "\t"
        << "MeanDisplacementZ" << "\t"
        << "WeightedMeanDisplacement" << "\t"
        << "TRE" << "\t"
        << "TranslationX" << "\t"
        << "TranslationY" << "\t"
        << "TranslationZ" << "\t"
        << "RotationX" << "\t"
        << "RotationY" << "\t"
        << "RotationZ" << "\t";
        info << "\b" << endl;
        
        for (unsigned int i = 0; i < _slices.size(); i++) {
            RigidTransformation& t = _transformations[i];
            info << _stack_index[i] << "\t"
            << _stack_loc_index[i] << "\t"
            << _stack_dyn_index[i] << "\t"
            << _loc_index[i] << "\t"
            << i << "\t"
            << stack_files[_stack_index[i]] << "\t"
            << _scale[i] << "\t"
            << _stack_factor[_stack_index[i]] << "\t"
            << _slice_time[i] << "\t"
            << _slice_dt[i] << "\t"
            << _slice_cardphase[i] << "\t"
            << _slice_svr_card_index[i] << "\t"
            << (((_slice_weight[i] >= 0.5) && (_slice_inside[i]))?1:0) << "\t" // Included slices
            << (((_slice_weight[i] < 0.5) && (_slice_inside[i]))?1:0) << "\t"  // Excluded slices
            << ((!(_slice_inside[i]))?1:0) << "\t"  // Outside slices
            << _slice_weight[i] << "\t"
            << _slice_displacement[i] << "\t"
            << _slice_tx[i] << "\t"
            << _slice_ty[i] << "\t"
            << _slice_tz[i] << "\t"
            << _slice_weighted_displacement[i] << "\t"
            << _slice_tre[i] << "\t"
            << t.GetTranslationX() << "\t"
            << t.GetTranslationY() << "\t"
            << t.GetTranslationZ() << "\t"
            << t.GetRotationX() << "\t"
            << t.GetRotationY() << "\t"
            << t.GetRotationZ() << "\t";
            info << "\b" << endl;
        }
        
        info.close();
    }

    // -----------------------------------------------------------------------------

} // namespace svrtk
