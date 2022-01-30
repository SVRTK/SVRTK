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

// SVRTK
#include "svrtk/ReconstructionCardiac4D.h"
#include "svrtk/Profiling.h"
#include "svrtk/Parallel.h"

using namespace std;
using namespace mirtk;
using namespace svrtk::Utility;

namespace svrtk {

    // -----------------------------------------------------------------------------
    // Determine Reconstructed Spatial Resolution
    // -----------------------------------------------------------------------------
    //determine resolution of volume to reconstruct
    double ReconstructionCardiac4D::GetReconstructedResolutionFromTemplateStack(const RealImage& stack) {
        double dx, dy, dz, d;
        stack.GetPixelSize(&dx, &dy, &dz);
        if (dx <= dy && dx <= dz)
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
    void ReconstructionCardiac4D::ReadSliceTransformations(const char *folder) {
        cout << "Reading slice transformations from: " << folder << endl;
        Array<RigidTransformation> loc_transformations;
        ReadTransformations(folder, _loc_index.back() + 1, loc_transformations);

        // Assign transformations to single-frame images
        ClearAndReserve(_transformations, _slices.size());
        for (size_t i = 0; i < _slices.size(); i++)
            _transformations.push_back(loc_transformations[_loc_index[i]]);
    }

    // -----------------------------------------------------------------------------
    // Initialise Reconstructed Volume from Static Mask
    // -----------------------------------------------------------------------------
    // Create zero image as a template for reconstructed volume
    void ReconstructionCardiac4D::CreateTemplateCardiac4DFromStaticMask(const RealImage& mask, double resolution) {
        // Get mask attributes - image size and voxel size
        ImageAttributes attr = mask.Attributes();

        // Set temporal dimension
        attr._t = _reconstructed_cardiac_phases.size();
        attr._dt = _reconstructed_temporal_resolution;

        // Create volume and initialise volume values
        RealImage volume4D(attr);

        // Resample to specified resolution
        unique_ptr<InterpolateImageFunction> interpolator(InterpolateImageFunction::New(Interpolation_NN));

        Resampling<RealPixel> resampling(resolution, resolution, resolution);
        resampling.Input(&volume4D);
        resampling.Output(&volume4D);
        resampling.Interpolator(interpolator.get());
        resampling.Run();

        // Set reconstructed 4D volume
        _reconstructed4D = move(volume4D);
        _template_created = true;

        // Set reconstructed 3D volume for reference by existing functions of irtkReconstruction class
        const ImageAttributes& attr4d = _reconstructed4D.Attributes();
        ImageAttributes attr3d = attr4d;
        attr3d._t = 1;
        RealImage volume3D(attr3d);
        _reconstructed = move(volume3D);

        if (_debug)
            cout << "CreateTemplateCardiac4DFromStaticMask created template 4D volume with " << attr4d._t << " time points and " << attr4d._dt << " s temporal resolution." << endl;
    }

    // -----------------------------------------------------------------------------
    // Match Stack Intensities With Masking
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::MatchStackIntensitiesWithMasking(Array<RealImage>& stacks, const Array<RigidTransformation>& stack_transformations, double averageValue, bool together) {
        if (_debug)
            cout << "Matching intensities of stacks. ";

        cout << setprecision(6);

        //Calculate the averages of intensities for all stacks
        RealImage m;
        Array<double> stack_average;
        stack_average.reserve(stacks.size());

        //remember the set average value
        _average_value = averageValue;

        //averages need to be calculated only in ROI
        for (size_t ind = 0; ind < stacks.size(); ind++) {
            ImageAttributes attr = stacks[ind].Attributes();
            attr._t = 1;
            m.Initialize(attr);
            double sum = 0, num = 0;

            #pragma omp parallel for reduction(+: sum, num)
            for (int i = 0; i < stacks[ind].GetX(); i++)
                for (int j = 0; j < stacks[ind].GetY(); j++)
                    for (int k = 0; k < stacks[ind].GetZ(); k++) {
                        //image coordinates of the stack voxel
                        double x = i;
                        double y = j;
                        double z = k;
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
                        if (_mask.IsInside(x, y, z)) {
                            if (_mask(x, y, z) == 1) {
                                m(i, j, k) = 1;
                                for (int f = 0; f < stacks[ind].GetT(); f++) {
                                    sum += stacks[ind](i, j, k, f);
                                    num++;
                                }
                            }
                        }
                    }
            if (_debug)
                m.Write((boost::format("maskformatching%03i.nii.gz") % ind).str().c_str());

            //calculate average for the stack
            if (num > 0) {
                stack_average.push_back(sum / num);
            } else {
                throw runtime_error("Stack " + to_string(ind) + " has no overlap with ROI");
            }
        }

        double global_average = 0;
        if (together) {
            for (size_t i = 0; i < stack_average.size(); i++)
                global_average += stack_average[i];
            global_average /= stack_average.size();
        }

        if (_debug) {
            cout << "Stack average intensities are ";
            for (size_t ind = 0; ind < stack_average.size(); ind++)
                cout << stack_average[ind] << " ";
            cout << "\nThe new average value is " << averageValue << endl;
        }

        //Rescale stacks
        ClearAndReserve(_stack_factor, stacks.size());
        for (size_t ind = 0; ind < stacks.size(); ind++) {
            double factor = averageValue / (together ? global_average : stack_average[ind]);
            _stack_factor.push_back(factor);

            RealPixel *ptr = stacks[ind].Data();
            #pragma omp parallel for
            for (int i = 0; i < stacks[ind].NumberOfVoxels(); i++) {
                if (ptr[i] > 0)
                    ptr[i] *= factor;
            }
        }

        if (_debug) {
            #pragma omp parallel for
            for (size_t ind = 0; ind < stacks.size(); ind++)
                stacks[ind].Write((boost::format("rescaledstack%03i.nii.gz") % ind).str().c_str());

            cout << "Slice intensity factors are ";
            for (size_t ind = 0; ind < _stack_factor.size(); ind++)
                cout << _stack_factor[ind] << " ";
            cout << "\nThe new average value is " << averageValue << endl;
        }

        cout << setprecision(3);
    }

    // -----------------------------------------------------------------------------
    // Create Slices and Associated Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CreateSlicesAndTransformationsCardiac4D(const Array<RealImage>& stacks, const Array<RigidTransformation>& stack_transformations, const Array<double>& thickness, const Array<RealImage>& probability_maps) {
        if (_debug)
            cout << "CreateSlicesAndTransformations" << endl;

        // Reset and allocate memory
        const size_t reserve_size = stacks.size() * stacks[0].Attributes()._z * stacks[0].Attributes()._t;
        ClearAndReserve(_slice_time, reserve_size);
        ClearAndReserve(_slice_dt, reserve_size);
        ClearAndReserve(_slices, reserve_size);
        ClearAndReserve(_simulated_slices, reserve_size);
        ClearAndReserve(_simulated_weights, reserve_size);
        ClearAndReserve(_simulated_inside, reserve_size);
        ClearAndReserve(_zero_slices, reserve_size);
        ClearAndReserve(_stack_index, reserve_size);
        ClearAndReserve(_loc_index, reserve_size);
        ClearAndReserve(_stack_loc_index, reserve_size);
        ClearAndReserve(_stack_dyn_index, reserve_size);
        ClearAndReserve(_transformations, reserve_size);
        ClearAndReserve(_slice_excluded, reserve_size);
        ClearAndReserve(_slice_svr_card_index, reserve_size);
        if (!probability_maps.empty())
            ClearAndReserve(_probability_maps, reserve_size);

        for (size_t i = 0, loc_index = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            const ImageAttributes& attr = stacks[i].Attributes();

            //attr._z is number of slices in the stack
            for (int j = 0; j < attr._z; j++, loc_index++) {
                //attr._t is number of frames in the stack
                for (int k = 0; k < attr._t; k++) {
                    //create slice by selecting the appropriate region of the stack
                    RealImage slice = stacks[i].GetRegion(0, 0, j, k, attr._x, attr._y, j + 1, k + 1);
                    //set correct voxel size in the stack. Z size is equal to slice thickness.
                    slice.PutPixelSize(attr._dx, attr._dy, thickness[i], attr._dt);
                    //set slice acquisition time
                    const double sliceAcqTime = attr._torigin + k * attr._dt;
                    _slice_time.push_back(sliceAcqTime);
                    //set slice temporal resolution
                    _slice_dt.push_back(attr._dt);
                    //remember the slice
                    _slices.push_back(slice);
                    _simulated_slices.push_back(slice);
                    _simulated_weights.push_back(slice);
                    _simulated_inside.push_back(move(slice));
                    _zero_slices.push_back(1);
                    //remember stack indices for this slice
                    _stack_index.push_back(i);
                    _loc_index.push_back(loc_index);
                    _stack_loc_index.push_back(j);
                    _stack_dyn_index.push_back(k);
                    //initialize slice transformation with the stack transformation
                    _transformations.push_back(stack_transformations[i]);
                    //initialise slice exclusion flags
                    _slice_excluded.push_back(0);
                    if (!probability_maps.empty()) {
                        RealImage proba = probability_maps[i].GetRegion(0, 0, j, k, attr._x, attr._y, j + 1, k + 1);
                        proba.PutPixelSize(attr._dx, attr._dy, thickness[i], attr._dt);
                        _probability_maps.push_back(move(proba));
                    }
                    //initialise cardiac phase to use for 2D-3D registration
                    _slice_svr_card_index.push_back(0);
                }
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
    void ReconstructionCardiac4D::CalculateSliceTemporalWeights() {
        if (_debug)
            cout << "CalculateSliceTemporalWeights" << endl;

        InitSliceTemporalWeights();

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < _reconstructed_cardiac_phases.size(); i++) {
            for (size_t j = 0; j < _slices.size(); j++) {
                _slice_temporal_weight[i][j] = CalculateTemporalWeight(_reconstructed_cardiac_phases[i], _slice_cardphase[j], _slice_dt[j], _slice_rr[j], _wintukeypct);

                // option for time window thresholding for velocity reconstruction
                if (_no_ts && _slice_temporal_weight[i][j] < 0.9)
                    _slice_temporal_weight[i][j] = 0;
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Calculate Temporal Weight
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateTemporalWeight(double cardphase0, double cardphase, double dt, double rr, double alpha) {
        // Angular Difference
        const double angdiff = CalculateAngularDifference(cardphase0, cardphase);

        // Temporal Resolution in Radians
        const double dtrad = 2 * PI * dt / rr;

        // Temporal Weight
        double temporalweight;
        if (_is_temporalpsf_gauss) {
            // Gaussian
            const double sigma = dtrad / 2.355;  // sigma = ~FWHM/2.355
            temporalweight = exp(-(angdiff * angdiff) / (2 * sigma * sigma));
        } else {
            // Sinc
            temporalweight = sinc(PI * angdiff / dtrad) * wintukey(angdiff, alpha);
        }

        return temporalweight;
    }

    // -----------------------------------------------------------------------------
    // Calculate Transformation Matrix Between Slices and Voxels
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CoeffInitCardiac4D() {
        SVRTK_START_TIMING();

        //resize slice-volume matrix from previous iteration
        ClearAndResize(_volcoeffs, _slices.size());

        //resize indicator of slice having and overlap with volumetric mask
        ClearAndResize(_slice_inside, _slices.size());

        if (_verbose)
            _verbose_log << "Initialising matrix coefficients... ";
        Parallel::CoeffInitCardiac4D coeffinit(this);
        coeffinit();
        if (_verbose)
            _verbose_log << "done." << endl;

        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        if (_verbose)
            _verbose_log << "Computing 4D volume weights..." << endl;
        _volume_weights.Initialize(_reconstructed4D.Attributes());

        if (_verbose)
            _verbose_log << "    ... for input slice: ";
        // Do not parallelise: It would cause data inconsistencies
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            if (_verbose)
                _verbose_log << inputIndex << ", ";
            // Do not parallelise: It would cause data inconsistencies
            for (size_t i = 0; i < _volcoeffs[inputIndex].size(); i++)
                for (size_t j = 0; j < _volcoeffs[inputIndex][i].size(); j++)
                    for (size_t k = 0; k < _volcoeffs[inputIndex][i][j].size(); k++) {
                        const POINT3D& p = _volcoeffs[inputIndex][i][j][k];
                        for (int outputIndex = 0; outputIndex < _reconstructed4D.GetT(); outputIndex++)
                            _volume_weights(p.x, p.y, p.z, outputIndex) += _slice_temporal_weight[outputIndex][inputIndex] * p.value;
                    }
        }
        if (_verbose)
            _verbose_log << "\b\b" << endl;
        // if (_debug)
        //     _volume_weights.Write("volume_weights.nii.gz");

        //find average volume weight to modify alpha parameters accordingly
        double sum = 0;
        int num = 0;
        #pragma omp parallel for reduction(+: sum, num)
        for (int i = 0; i < _volume_weights.GetX(); i++)
            for (int j = 0; j < _volume_weights.GetY(); j++)
                for (int k = 0; k < _volume_weights.GetZ(); k++)
                    if (_mask(i, j, k) == 1)
                        for (int f = 0; f < _volume_weights.GetT(); f++) {
                            sum += _volume_weights(i, j, k, f);
                            num++;
                        }

        _average_volume_weight = sum / num;

        if (_verbose)
            _verbose_log << "Average volume weight is " << _average_volume_weight << endl;

        SVRTK_END_TIMING("CoeffInitCardiac4D");
    }

    // -----------------------------------------------------------------------------
    // PSF-Weighted Reconstruction
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::GaussianReconstructionCardiac4D() {
        SVRTK_START_TIMING();

        if (_verbose) {
            _verbose_log << "Gaussian reconstruction ... " << endl;
            _verbose_log << "\tinput slice:  ";
        }

        RealImage slice;
        Array<int> voxel_num;
        voxel_num.reserve(_slices.size() * _slices[0].GetX() * _slices[0].GetY());

        //clear _reconstructed image
        memset(_reconstructed4D.Data(), 0, sizeof(RealPixel) * _reconstructed4D.NumberOfVoxels());

        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            if (_slice_excluded[inputIndex])
                continue;

            if (_verbose)
                _verbose_log << inputIndex << ", ";

            //copy the current slice
            slice = _slices[inputIndex];
            //alias the current bias image
            const RealImage& b = _bias[inputIndex];
            //read current scale factor
            const double scale = _scale[inputIndex];

            int slice_vox_num = 0;

            //Distribute slice intensities to the volume
            for (size_t i = 0; i < _volcoeffs[inputIndex].size(); i++)
                for (size_t j = 0; j < _volcoeffs[inputIndex][i].size(); j++)
                    if (slice(i, j, 0) != -1) {
                        //biascorrect and scale the slice
                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                        //number of volume voxels with non-zero coefficients
                        //for current slice voxel
                        const size_t n = _volcoeffs[inputIndex][i][j].size();

                        //if given voxel is not present in reconstructed volume at all,
                        //pad it

                        //if (n == 0)
                        //_slices[inputIndex].PutAsDouble(i, j, 0, -1);
                        //calculate num of vox in a slice that have overlap with roi
                        if (n > 0)
                            slice_vox_num++;

                        //add contribution of current slice voxel to all voxel volumes
                        //to which it contributes
                        for (size_t k = 0; k < n; k++) {
                            const POINT3D& p = _volcoeffs[inputIndex][i][j][k];
                            for (size_t outputIndex = 0; outputIndex < _reconstructed_cardiac_phases.size(); outputIndex++)
                                _reconstructed4D(p.x, p.y, p.z, outputIndex) += _slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0);
                        }
                    }
            voxel_num.push_back(slice_vox_num);
        } //end of loop for a slice inputIndex

        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxel
        _reconstructed4D /= _volume_weights;

        if (_verbose)
            _verbose_log << "\b\b\n... Gaussian reconstruction done.\n" << endl;

        if (_debug)
            _reconstructed4D.Write("init.nii.gz");

        // find slices with small overlap with ROI and exclude them.
        //find median
        Array<int> voxel_num_tmp = voxel_num;
        int median = round(voxel_num_tmp.size() * 0.5) - 1;
        nth_element(voxel_num_tmp.begin(), voxel_num_tmp.begin() + median, voxel_num_tmp.end());
        median = voxel_num_tmp[median];

        //remember slices with small overlap with ROI
        ClearAndReserve(_small_slices, voxel_num.size());
        for (size_t i = 0; i < voxel_num.size(); i++)
            if (voxel_num[i] < 0.1 * median)
                _small_slices.push_back(i);

        if (_verbose) {
            _verbose_log << "Small slices:";
            for (size_t i = 0; i < _small_slices.size(); i++)
                _verbose_log << " " << _small_slices[i];
            _verbose_log << endl;
        }

        SVRTK_END_TIMING("GaussianReconstructionCardiac4D");
    }

    // -----------------------------------------------------------------------------
    // Calculate Corrected Slices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateCorrectedSlices() {
        if (_verbose)
            _verbose_log << "Calculating corrected slices ... ";

        Parallel::CalculateCorrectedSlices parallelCalculateCorrectedSlices(this);
        parallelCalculateCorrectedSlices();

        if (_verbose)
            _verbose_log << "done." << endl;
    }

    // -----------------------------------------------------------------------------
    // Simulate Slices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SimulateSlicesCardiac4D() {
        SVRTK_START_TIMING();

        if (_verbose)
            _verbose_log << "Simulating slices ... ";

        Parallel::SimulateSlicesCardiac4D parallelSimulateSlices(this);
        parallelSimulateSlices();

        if (_verbose)
            _verbose_log << "done." << endl;

        SVRTK_END_TIMING("SimulateSlicesCardiac4D");
    }

    // -----------------------------------------------------------------------------
    // Simulate Stacks
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SimulateStacksCardiac4D() {
        if (_verbose)
            _verbose_log << "Simulating stacks ... ";

        //Initialise simulated images
        #pragma omp parallel for
        for (size_t i = 0; i < _slices.size(); i++)
            memset(_simulated_slices[i].Data(), 0, sizeof(RealPixel) * _simulated_slices[i].NumberOfVoxels());

        //Initialise indicator of images having overlap with volume
        _slice_inside = Array<bool>(_slices.size(), true);

        //Simulate images
        Parallel::SimulateStacksCardiac4D simulatestacks(this);
        simulatestacks();

        if (_verbose)
            _verbose_log << "done." << endl;
    }

    // -----------------------------------------------------------------------------
    // Calculate Error
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateError() {
        if (_verbose)
            _verbose_log << "Calculating error ... ";

        Parallel::CalculateError parallelCalculateError(this);
        parallelCalculateError();

        if (_verbose)
            _verbose_log << "done." << endl;
    }

    // -----------------------------------------------------------------------------
    // Normalise Bias
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::NormaliseBiasCardiac4D(int iter, int rec_iter) {
        SVRTK_START_TIMING();

        if (_verbose)
            _verbose_log << "Normalise Bias ... ";

        Parallel::NormaliseBiasCardiac4D parallelNormaliseBias(this);
        parallelNormaliseBias();
        RealImage& bias = parallelNormaliseBias.bias;
        RealImage& volweight3d = parallelNormaliseBias.volweight3d;

        // normalize the volume by proportion of contributing slice voxels for each volume voxel
        bias /= volweight3d;

        if (_verbose)
            _verbose_log << "done." << endl;

        StaticMaskVolume4D(bias, _mask, 0);
        RealImage m = _mask;
        GaussianBlurring<RealPixel> gb(_sigma_bias);
        gb.Input(&bias);
        gb.Output(&bias);
        gb.Run();
        gb.Input(&m);
        gb.Output(&m);
        gb.Run();

        bias /= m;

        if (_debug)
            bias.Write((boost::format("averagebias_mc%02isr%02i.nii.gz") % iter % rec_iter).str().c_str());

        #pragma omp parallel for
        for (int i = 0; i < _reconstructed4D.GetX(); i++)
            for (int j = 0; j < _reconstructed4D.GetY(); j++)
                for (int k = 0; k < _reconstructed4D.GetZ(); k++)
                    for (int f = 0; f < _reconstructed4D.GetT(); f++)
                        if (_reconstructed4D(i, j, k, f) != -1)
                            _reconstructed4D(i, j, k, f) /= exp(-bias(i, j, k));

        SVRTK_END_TIMING("NormaliseBiasCardiac4D");
    }

    // -----------------------------------------------------------------------------
    // Calculate Target Cardiac Phase in Reconstructed Volume for Slice-to-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateSliceToVolumeTargetCardiacPhase() {
        if (_debug)
            cout << "CalculateSliceToVolumeTargetCardiacPhase" << endl;

        ClearAndResize(_slice_svr_card_index, _slices.size());
        #pragma omp parallel for
        for (size_t i = 0; i < _slices.size(); i++) {
            double angdiff = PI + 0.001;  // NOTE: init angdiff larger than any possible calculated angular difference
            int card_index = -1;
            for (size_t j = 0; j < _reconstructed_cardiac_phases.size(); j++) {
                const double nangdiff = fabs(CalculateAngularDifference(_reconstructed_cardiac_phases[j], _slice_cardphase[i]));
                if (nangdiff < angdiff) {
                    angdiff = nangdiff;
                    card_index = j;
                }
            }
            _slice_svr_card_index[i] = card_index;
            // if (_debug)
            //   cout << i << ":" << _slice_svr_card_index[i] << ", ";
        }
        // if (_debug)
        //   cout << "\b\b." << endl;
    }

    // -----------------------------------------------------------------------------
    // Slice-to-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SliceToVolumeRegistrationCardiac4D() {
        SVRTK_START_TIMING();

        if (_verbose)
            _verbose_log << "SliceToVolumeRegistrationCardiac4D" << endl;

        Parallel::SliceToVolumeRegistrationCardiac4D registration(this);
        registration();

        SVRTK_END_TIMING("SliceToVolumeRegistrationCardiac4D");
    }

    // -----------------------------------------------------------------------------
    // Remote Slice-to-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::RemoteSliceToVolumeRegistrationCardiac4D(int iter, const string& str_mirtk_path, const string& str_current_exchange_file_path) {
        const ImageAttributes& attr_recon = _reconstructed4D.Attributes();
        RealImage source, target;

        if (_verbose)
            _verbose_log << "RemoteSliceToVolumeRegistrationCardiac4D" << endl;

        #pragma omp parallel for
        for (int t = 0; t < _reconstructed4D.GetT(); t++) {
            string str_source = str_current_exchange_file_path + "/current-source-" + to_string(t) + ".nii.gz";
            source = _reconstructed4D.GetRegion(0, 0, 0, t, attr_recon._x, attr_recon._y, attr_recon._z, t + 1);
            source.Write(str_source.c_str());
        }

        if (iter == 1) {
            ClearAndReserve(_offset_matrices, _slices.size());

            GenericLinearInterpolateImageFunction<RealImage> interpolator;
            ResamplingWithPadding<RealPixel> resampling(attr_recon._dx, attr_recon._dx, attr_recon._dx, -1);
            resampling.Output(&target);
            resampling.Interpolator(&interpolator);

            // Do not parallelise: ResamplingWithPadding has already been parallelised!
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                target.Initialize(_slices[inputIndex].Attributes());
                resampling.Input(&_slices[inputIndex]);
                resampling.Run();

                // put origin to zero
                RigidTransformation offset;
                ResetOrigin(target, offset);

                RealPixel tmin, tmax;
                target.GetMinMax(&tmin, &tmax);
                _zero_slices[inputIndex] = tmax > 1 && (tmax - tmin) > 1 ? 1 : -1;

                const string str_target = str_current_exchange_file_path + "/res-slice-" + to_string(inputIndex) + ".nii.gz";
                target.Write(str_target.c_str());

                _offset_matrices.push_back(offset.GetMatrix());
            }
        }

        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            RigidTransformation r_transform = _transformations[inputIndex];
            r_transform.PutMatrix(r_transform.GetMatrix() * _offset_matrices[inputIndex]);

            const string str_dofin = str_current_exchange_file_path + "/res-transformation-" + to_string(inputIndex) + ".dof";
            r_transform.Write(str_dofin.c_str());
        }

        int stride = 32;
        int svr_range_start = 0;
        int svr_range_stop = svr_range_start + stride;

        while (svr_range_start < _slices.size()) {
            Parallel::RemoteSliceToVolumeRegistration registration(this, svr_range_start, svr_range_stop, str_mirtk_path, str_current_exchange_file_path, true, _slice_svr_card_index);
            registration();

            svr_range_start = svr_range_stop;
            svr_range_stop = min(svr_range_start + stride, (int)_slices.size());
        }

        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            const string str_dofout = str_current_exchange_file_path + "/res-transformation-" + to_string(inputIndex) + ".dof";
            _transformations[inputIndex].Read(str_dofout.c_str());

            //undo the offset
            _transformations[inputIndex].PutMatrix(_transformations[inputIndex].GetMatrix() * _offset_matrices[inputIndex].Inverse());
        }
    }

    // -----------------------------------------------------------------------------
    // Volume-to-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::VolumeToVolumeRegistration(const GreyImage& target, const GreyImage& source, RigidTransformation& rigidTransf) {
        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        Insert(params, "Image (dis-)similarity measure", "NMI");
        if (_nmi_bins > 0)
            Insert(params, "No. of bins", _nmi_bins);
        Insert(params, "Background value for image 1", -1);
        Insert(params, "Background value for image 2", -1);

        GenericRegistrationFilter registration;
        registration.Parameter(params);
        registration.Input(&target, &source);
        Transformation *dofout = nullptr;
        registration.Output(&dofout);
        registration.InitialGuess(&rigidTransf);
        registration.GuessParameter();
        registration.Run();

        unique_ptr<RigidTransformation> rigidTransf_dofout(dynamic_cast<RigidTransformation*>(dofout));
        rigidTransf = *rigidTransf_dofout;
    }

    // -----------------------------------------------------------------------------
    // Common Calculate Function
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::Calculate(const RigidTransformation& drift, double (ReconstructionCardiac4D::*Calculate)()) {
        //Initialise
        Matrix d = drift.GetMatrix();

        //Remove drift
        #pragma omp parallel for
        for (size_t i = 0; i < _transformations.size(); i++)
            _transformations[i].PutMatrix(d * _transformations[i].GetMatrix());

        //Calculate
        const double mean = (this->*Calculate)();

        //Return drift
        d.Invert();
        #pragma omp parallel for
        for (size_t i = 0; i < _transformations.size(); i++)
            _transformations[i].PutMatrix(d * _transformations[i].GetMatrix());

        //Output
        return mean;
    }

    // -----------------------------------------------------------------------------
    // Calculate Displacement
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateDisplacement() {
        if (_debug)
            cout << "CalculateDisplacement" << endl;

        ClearAndResize(_slice_displacement, _slices.size());
        ClearAndResize(_slice_tx, _slices.size());
        ClearAndResize(_slice_ty, _slices.size());
        ClearAndResize(_slice_tz, _slices.size());

        double disp_sum_total = 0;
        int num_voxel_total = 0;

        #pragma omp parallel for reduction(+: disp_sum_total, num_voxel_total)
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            int num_voxel_slice = 0;
            double slice_disp = -1, disp_sum_slice = 0;
            double tx_slice = 0, ty_slice = 0, tz_slice = 0;
            double tx_sum_slice = 0, ty_sum_slice = 0, tz_sum_slice = 0;

            if (!_slice_excluded[inputIndex]) {
                for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                        if (_slices[inputIndex](i, j, 0) != -1) {
                            double x = i, y = j, z = 0;
                            _slices[inputIndex].ImageToWorld(x, y, z);
                            double tx = x, ty = y, tz = z;
                            _transformations[inputIndex].Transform(tx, ty, tz);
                            disp_sum_slice += sqrt((tx - x) * (tx - x) + (ty - y) * (ty - y) + (tz - z) * (tz - z));
                            tx_sum_slice += tx - x;
                            ty_sum_slice += ty - y;
                            tz_sum_slice += tz - z;
                            num_voxel_slice++;
                        }
                    }
                }
                if (num_voxel_slice > 0) {
                    slice_disp = disp_sum_slice / num_voxel_slice;
                    tx_slice = tx_sum_slice / num_voxel_slice;
                    ty_slice = ty_sum_slice / num_voxel_slice;
                    tz_slice = tz_sum_slice / num_voxel_slice;
                    disp_sum_total += disp_sum_slice;
                    num_voxel_total += num_voxel_slice;
                }
            }

            _slice_displacement[inputIndex] = slice_disp;
            _slice_tx[inputIndex] = tx_slice;
            _slice_ty[inputIndex] = ty_slice;
            _slice_tz[inputIndex] = tz_slice;
        }

        return num_voxel_total > 0 ? disp_sum_total / num_voxel_total : -1;
    }

    // -----------------------------------------------------------------------------
    // Calculate Weighted Displacement
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateWeightedDisplacement() {
        if (_debug)
            cout << "CalculateWeightedDisplacement" << endl;

        ClearAndResize(_slice_weighted_displacement, _slices.size());

        double disp_sum_total = 0, weight_total = 0;

        #pragma omp parallel for reduction(+: disp_sum_total, weight_total)
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            double disp_sum_slice = 0, weight_slice = 0, slice_disp = -1;

            if (!_slice_excluded[inputIndex]) {
                for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                        if (_slices[inputIndex](i, j, 0) != -1) {
                            double x = i, y = j, z = 0;
                            _slices[inputIndex].ImageToWorld(x, y, z);
                            double tx = x, ty = y, tz = z;
                            _transformations[inputIndex].Transform(tx, ty, tz);
                            disp_sum_slice += _slice_weight[inputIndex] * _weights[inputIndex](i, j, 0) * sqrt((tx - x) * (tx - x) + (ty - y) * (ty - y) + (tz - z) * (tz - z));
                            weight_slice += _slice_weight[inputIndex] * _weights[inputIndex](i, j, 0);
                        }
                    }
                }
                if (weight_slice > 0) {
                    slice_disp = disp_sum_slice / weight_slice;
                    disp_sum_total += disp_sum_slice;
                    weight_total += weight_slice;
                }
            }

            _slice_weighted_displacement[inputIndex] = slice_disp;
        }

        return weight_total > 0 ? disp_sum_total / weight_total : -1;
    }

    // -----------------------------------------------------------------------------
    // Calculate Target Registration Error
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateTRE() {
        if (_debug)
            cout << "CalculateTRE" << endl;

        ClearAndReserve(_slice_tre, _slices.size());

        double tre_sum_total = 0;
        int num_voxel_total = 0;

        #pragma omp parallel for reduction(+: tre_sum_total, num_voxel_total)
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            double tre_sum_slice = 0, slice_tre = -1;
            int num_voxel_slice = 0;

            if (!_slice_excluded[inputIndex]) {
                for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                        if (_slices[inputIndex](i, j, 0) != -1) {
                            double x = i, y = j, z = 0;
                            _slices[inputIndex].ImageToWorld(x, y, z);
                            double cx = x, cy = y, cz = z;
                            double px = x, py = y, pz = z;
                            _transformations[inputIndex].Transform(cx, cy, cz);
                            _ref_transformations[inputIndex].Transform(px, py, pz);
                            tre_sum_slice += sqrt((cx - px) * (cx - px) + (cy - py) * (cy - py) + (cz - pz) * (cz - pz));
                            num_voxel_slice++;
                        }
                    }
                }
                if (num_voxel_slice > 0) {
                    slice_tre = tre_sum_slice / num_voxel_slice;
                    tre_sum_total += tre_sum_slice;
                    num_voxel_total += num_voxel_slice;
                }
            }

            _slice_tre[inputIndex] = slice_tre;
        }

        return num_voxel_total > 0 ? tre_sum_total / num_voxel_total : -1;
    }

    // -----------------------------------------------------------------------------
    // Smooth Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SmoothTransformations(double sigma_seconds, int niter, bool use_slice_inside) {
        if (_debug)
            cout << "SmoothTransformations\n\tsigma = " << sigma_seconds << " s" << endl;

        //Reset origin for transformations
        GreyImage t = _reconstructed4D;
        RigidTransformation offset;
        ResetOrigin(t, offset);
        if (_debug)
            offset.Write("reset_origin.dof");

        const Matrix& mo = offset.GetMatrix();
        const Matrix imo = mo.Inverse();
        for (size_t i = 0; i < _transformations.size(); i++)
            _transformations[i].PutMatrix(imo * _transformations[i].GetMatrix() * mo);

        //initial weights
        Matrix parameters(6, _transformations.size());
        Matrix weights(6, _transformations.size());
        ofstream fileOut("motion.txt", ofstream::out | ofstream::app);

        for (size_t i = 0; i < _transformations.size(); i++) {
            parameters(0, i) = _transformations[i].GetTranslationX();
            parameters(1, i) = _transformations[i].GetTranslationY();
            parameters(2, i) = _transformations[i].GetTranslationZ();
            parameters(3, i) = _transformations[i].GetRotationX();
            parameters(4, i) = _transformations[i].GetRotationY();
            parameters(5, i) = _transformations[i].GetRotationZ();

            //write unprocessed parameters to file
            for (int j = 0; j < 6; j++) {
                fileOut << parameters(j, i);
                if (j < 5)
                    fileOut << ",";
                else
                    fileOut << endl;
            }

            for (int j = 0; j < 6; j++)
                weights(j, i) = 0;             //initialise as zero

            if (!_slice_excluded[i]) {
                if (!use_slice_inside) {    //set weights based on image intensities
                    RealPixel smin, smax;
                    _slices[i].GetMinMax(&smin, &smax);
                    if (smax > -1)
                        for (int j = 0; j < 6; j++)
                            weights(j, i) = 1;
                } else {    //set weights based on _slice_inside
                    if (!_slice_inside.empty() && _slice_inside[i])
                        for (int j = 0; j < 6; j++)
                            weights(j, i) = 1;
                }
            }
        }

        //initialise
        Matrix den(6, _transformations.size());
        Matrix num(6, _transformations.size());
        Matrix kr(6, _transformations.size());

        int nloc = 0;
        for (size_t i = 0; i < _transformations.size(); i++)
            nloc = max(nloc, _loc_index[i] + 1);
        if (_debug)
            cout << "\tnumber of slice-locations = " << nloc << endl;
        const int dim = _transformations.size() / nloc; // assuming equal number of dynamic images for every slice-location

        //step size for sampling volume in error calculation in kernel regression
        constexpr double nstep = 15;
        int step = ceil(_reconstructed4D.GetX() / nstep);
        if (step > ceil(_reconstructed4D.GetY() / nstep))
            step = ceil(_reconstructed4D.GetY() / nstep);
        if (step > ceil(_reconstructed4D.GetZ() / nstep))
            step = ceil(_reconstructed4D.GetZ() / nstep);

        //kernel regression
        Array<double> kernel, error, tmp;
        error.reserve(_transformations.size());
        tmp.reserve(_transformations.size());
        for (int iter = 0; iter < niter; iter++) {
            for (int loc = 0; loc < nloc; loc++) {
                //gaussian kernel for current slice-location
                const double sigma = ceil(sigma_seconds / _slice_dt[loc * dim]);
                for (int j = -3 * sigma; j <= 3 * sigma; j++)
                    kernel.push_back(exp(-(j * _slice_dt[loc * dim] / sigma_seconds) * (j * _slice_dt[loc * dim] / sigma_seconds)));

                //kernel-weighted summation
                #pragma omp parallel for collapse(2)
                for (int par = 0; par < 6; par++) {
                    for (int i = loc * dim; i < (loc + 1) * dim; i++) {
                        if (!_slice_excluded[i]) {
                            for (int j = -3 * sigma; j <= 3 * sigma; j++)
                                if (((i + j) >= loc * dim) && ((i + j) < (loc + 1) * dim)) {
                                    num(par, i) += parameters(par, i + j) * kernel[j + 3 * sigma] * weights(par, i + j);
                                    den(par, i) += kernel[j + 3 * sigma] * weights(par, i + j);
                                }
                        } else {
                            num(par, i) = parameters(par, i);
                            den(par, i) = 1;
                        }
                    }
                }

                kernel.clear();
            }

            //kernel-weighted normalisation
            #pragma omp parallel for collapse(2)
            for (int par = 0; par < 6; par++)
                for (size_t i = 0; i < _transformations.size(); i++)
                    kr(par, i) = num(par, i) / den(par, i);

            //recalculate weights using target registration error with original transformations as targets
            for (size_t i = 0; i < _transformations.size(); i++) {
                RigidTransformation processed;
                processed.PutTranslationX(kr(0, i));
                processed.PutTranslationY(kr(1, i));
                processed.PutTranslationZ(kr(2, i));
                processed.PutRotationX(kr(3, i));
                processed.PutRotationY(kr(4, i));
                processed.PutRotationZ(kr(5, i));

                RigidTransformation orig = _transformations[i];

                //need to convert the transformations back to the original coordinate system
                orig.PutMatrix(mo * orig.GetMatrix() * imo);
                processed.PutMatrix(mo * processed.GetMatrix() * imo);

                int n = 0;
                double e = 0;

                if (!_slice_excluded[i]) {
                    #pragma omp parallel for collapse(3) reduction(+: n, e)
                    for (int ii = 0; ii < _reconstructed4D.GetX(); ii += step)
                        for (int jj = 0; jj < _reconstructed4D.GetY(); jj += step)
                            for (int kk = 0; kk < _reconstructed4D.GetZ(); kk += step)
                                if (_reconstructed4D(ii, jj, kk, 0) > -1) {
                                    double x = ii, y = jj, z = kk;
                                    _reconstructed4D.ImageToWorld(x, y, z);
                                    double xx = x, yy = y, zz = z;
                                    orig.Transform(x, y, z);
                                    processed.Transform(xx, yy, zz);
                                    x -= xx;
                                    y -= yy;
                                    z -= zz;
                                    e += sqrt(x * x + y * y + z * z);
                                    n++;
                                }
                }

                if (n > 0) {
                    e /= n;
                    error.push_back(e);
                    tmp.push_back(e);
                } else
                    error.push_back(-1);
            }

            sort(tmp.begin(), tmp.end());
            const double median = tmp[round(tmp.size() * 0.5) - 1];

            if (_debug && iter == 0)
                cout << "\titeration:median_error(mm)...";
            if (_debug) {
                cout << iter << ":" << median << ", ";
                cout.flush();
            }

            #pragma omp parallel for
            for (size_t i = 0; i < _transformations.size(); i++) {
                double value = 0;
                if (error[i] >= 0 && !_slice_excluded[i]) {
                    if (error[i] <= median * 1.35)
                        value = 1;
                    else
                        value = median * 1.35 / error[i];
                }
                for (int par = 0; par < 6; par++)
                    weights(par, i) = value;
            }

            error.clear();
            tmp.clear();
        }

        if (_debug)
            cout << "\b\b" << endl;

        ofstream fileOut2("motion-processed.txt", ofstream::out | ofstream::app);
        ofstream fileOut3("weights.txt", ofstream::out | ofstream::app);
        ofstream fileOut4("outliers.txt", ofstream::out | ofstream::app);
        ofstream fileOut5("empty.txt", ofstream::out | ofstream::app);

        for (size_t i = 0; i < _transformations.size(); i++) {
            fileOut3 << weights(0, i) << " ";

            if (weights(0, i) <= 0)
                fileOut5 << i << " ";

            _transformations[i].PutTranslationX(kr(0, i));
            _transformations[i].PutTranslationY(kr(1, i));
            _transformations[i].PutTranslationZ(kr(2, i));
            _transformations[i].PutRotationX(kr(3, i));
            _transformations[i].PutRotationY(kr(4, i));
            _transformations[i].PutRotationZ(kr(5, i));

            for (int j = 0; j < 6; j++) {
                fileOut2 << kr(j, i);
                if (j < 5)
                    fileOut2 << ",";
                else
                    fileOut2 << endl;
            }
        }

        //Put origin back
        #pragma omp parallel for
        for (size_t i = 0; i < _transformations.size(); i++)
            _transformations[i].PutMatrix(mo * _transformations[i].GetMatrix() * imo);
    }

    // -----------------------------------------------------------------------------
    // Scale Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::ScaleTransformations(double scale) {
        if (_verbose)
            _verbose_log << "Scaling transformations." << endl << "scale = " << scale << "." << endl;

        if (scale == 1)
            return;

        if (scale < 0)
            throw runtime_error("Scaling of transformations undefined for scale < 0.");

        //Reset origin for transformations
        GreyImage t = _reconstructed4D;
        RigidTransformation offset;
        ResetOrigin(t, offset);
        if (_debug)
            offset.Write("reset_origin.dof");

        const Matrix& mo = offset.GetMatrix();
        const Matrix imo = mo.Inverse();

        #pragma omp parallel for
        for (size_t i = 0; i < _transformations.size(); i++) {
            _transformations[i].PutMatrix(imo * _transformations[i].GetMatrix() * mo);

            //Scale transformations
            const Matrix orig = logm(_transformations[i].GetMatrix());
            Matrix scaled = orig;
            for (int row = 0; row < 4; row++)
                for (int col = 0; col < 4; col++)
                    scaled(row, col) = scale * orig(row, col);
            _transformations[i].PutMatrix(expm(scaled));

            //Put origin back
            _transformations[i].PutMatrix(mo * _transformations[i].GetMatrix() * imo);
        }
    }

    // -----------------------------------------------------------------------------
    // Super-Resolution of 4D Volume
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SuperresolutionCardiac4D(int iter) {
        SVRTK_START_TIMING();

        if (_verbose)
            _verbose_log << "Superresolution " << iter << endl;

        //Remember current reconstruction for edge-preserving smoothing
        RealImage original = _reconstructed4D;

        Parallel::SuperresolutionCardiac4D parallelSuperresolution(this);
        parallelSuperresolution();

        RealImage& addon = parallelSuperresolution.addon;
        _confidence_map = move(parallelSuperresolution.confidence_map);

        if (_debug) {
            _confidence_map.Write((boost::format("confidence-map%i.nii.gz") % iter).str().c_str());
            addon.Write((boost::format("addon%i.nii.gz") % iter).str().c_str());
        }

        if (!_adaptive) {
            RealPixel *pa = addon.Data();
            RealPixel *pcm = _confidence_map.Data();
            #pragma omp parallel for
            for (int i = 0; i < addon.NumberOfVoxels(); i++) {
                if (pcm[i] > 0) {
                    // ISSUES if pcm[i] is too small leading to bright pixels
                    pa[i] /= pcm[i];
                    //this is to revert to normal (non-adaptive) regularisation
                    pcm[i] = 1;
                }
            }
        }

        _reconstructed4D += addon * _alpha; //_average_volume_weight;

        //bound the intensities
        RealPixel *pr = _reconstructed4D.Data();
        #pragma omp parallel for
        for (int i = 0; i < _reconstructed4D.NumberOfVoxels(); i++) {
            if (pr[i] < _min_intensity * 0.9)
                pr[i] = _min_intensity * 0.9;
            if (pr[i] > _max_intensity * 1.1)
                pr[i] = _max_intensity * 1.1;
        }

        //Smooth the reconstructed image
        AdaptiveRegularizationCardiac4D(iter, original);

        //Remove the bias in the reconstructed volume compared to previous iteration
        /* TODO: update adaptive regularisation for 4d
         if (_global_bias_correction)
         BiasCorrectVolume(original);
         */

        SVRTK_END_TIMING("SuperresolutionCardiac4D");
    }

    // -----------------------------------------------------------------------------
    // Adaptive Regularization
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::AdaptiveRegularizationCardiac4D(int iter, const RealImage& original) {
        if (_verbose)
            _verbose_log << "AdaptiveRegularizationCardiac4D." << endl;
        //_verbose_log << "AdaptiveRegularizationCardiac4D: _delta = "<<_delta<<" _lambda = "<<_lambda <<" _alpha = "<<_alpha<< endl;

        Array<double> factor(13);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }

        Array<RealImage> b(13, _reconstructed4D);

        Parallel::AdaptiveRegularization1Cardiac4D parallelAdaptiveRegularization1(this, b, factor, original);
        parallelAdaptiveRegularization1();

        RealImage original2 = _reconstructed4D;
        Parallel::AdaptiveRegularization2Cardiac4D parallelAdaptiveRegularization2(this, b, original2);
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
    double ReconstructionCardiac4D::CalculateEntropy() {
        double sum_x_sq = 0;

        #pragma omp parallel for reduction(+: sum_x_sq)
        for (int i = 0; i < _reconstructed4D.GetX(); i++)
            for (int j = 0; j < _reconstructed4D.GetY(); j++)
                for (int k = 0; k < _reconstructed4D.GetZ(); k++)
                    if (_mask(i, j, k) == 1)
                        for (int f = 0; f < _reconstructed4D.GetT(); f++) {
                            const double x = _reconstructed4D(i, j, k, f);
                            sum_x_sq += x * x;
                        }

        const double x_max = sqrt(sum_x_sq);
        double entropy = 0;

        #pragma omp parallel for reduction(+: entropy)
        for (int i = 0; i < _reconstructed4D.GetX(); i++)
            for (int j = 0; j < _reconstructed4D.GetY(); j++)
                for (int k = 0; k < _reconstructed4D.GetZ(); k++)
                    if (_mask(i, j, k) == 1)
                        for (int f = 0; f < _reconstructed4D.GetT(); f++) {
                            const double x = _reconstructed4D(i, j, k, f);
                            if (x > 0)
                                entropy += x / x_max * log(x / x_max);
                        }

        return -entropy;
    }

    // -----------------------------------------------------------------------------
    // Common save functions
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::Save(const Array<RealImage>& save_stacks, const char *message, const char *filename_prefix) {
        if (_verbose)
            _verbose_log << message;

        #pragma omp parallel for
        for (size_t i = 0; i < save_stacks.size(); i++)
            save_stacks[i].Write((boost::format("%s%05i.nii.gz") % filename_prefix % i).str().c_str());

        if (_verbose)
            _verbose_log << " done." << endl;
    }

    void ReconstructionCardiac4D::Save(const Array<RealImage>& source, const Array<RealImage>& stacks, int iter, int rec_iter, const char *message, const char *filename_prefix) {
        if (_verbose)
            _verbose_log << message;

        Array<RealImage> save_stacks;
        save_stacks.reserve(stacks.size());

        for (size_t i = 0; i < stacks.size(); i++)
            save_stacks.push_back(RealImage(stacks[i].Attributes()));

        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < source.size(); inputIndex++)
            for (int i = 0; i < source[inputIndex].GetX(); i++)
                for (int j = 0; j < source[inputIndex].GetY(); j++)
                    save_stacks[_stack_index[inputIndex]](i, j, _stack_loc_index[inputIndex], _stack_dyn_index[inputIndex]) = source[inputIndex](i, j, 0);

        #pragma omp parallel for
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

        Array<RealImage> imagestacks, wstacks;
        imagestacks.reserve(stacks.size());
        wstacks.reserve(stacks.size());

        for (size_t i = 0; i < stacks.size(); i++) {
            RealImage stack(stacks[i].Attributes());
            imagestacks.push_back(stack);
            wstacks.push_back(move(stack));
        }

        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            for (int i = 0; i < _slices[inputIndex].GetX(); i++)
                for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                    imagestacks[_stack_index[inputIndex]](i, j, _stack_loc_index[inputIndex], _stack_dyn_index[inputIndex]) = _slices[inputIndex](i, j, 0);
                    wstacks[_stack_index[inputIndex]](i, j, _stack_loc_index[inputIndex], _stack_dyn_index[inputIndex]) = 10 * _weights[inputIndex](i, j, 0);
                }

        #pragma omp parallel for 
        for (size_t i = 0; i < stacks.size(); i++) {
            imagestacks[i].Write((boost::format("stack%03i.nii.gz") % i).str().c_str());
            wstacks[i].Write((boost::format("w%03i.nii.gz") % i).str().c_str());
        }

        if (_verbose)
            _verbose_log << "done." << endl;
    }

    // -----------------------------------------------------------------------------
    // Save slice info
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SaveSliceInfoCardiac4D(const char *filename, const Array<string>& stack_files) {
        ofstream info(filename);

        info << setprecision(3);

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

        for (size_t i = 0; i < _slices.size(); i++) {
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
                << ((_slice_weight[i] >= 0.5 && _slice_inside[i]) ? 1 : 0) << "\t" // Included slices
                << ((_slice_weight[i] < 0.5 && _slice_inside[i]) ? 1 : 0) << "\t"  // Excluded slices
                << ((!(_slice_inside[i])) ? 1 : 0) << "\t"  // Outside slices
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
    }

    // -----------------------------------------------------------------------------

} // namespace svrtk
