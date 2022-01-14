/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2018-2021 King's College London
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
#include "svrtk/ReconstructionCardiacVelocity4D.h"
#include "svrtk/Profiling.h"
#include "svrtk/Parallel.h"

using namespace std;
using namespace mirtk;
using namespace svrtk::Utility;

namespace svrtk {

    // -----------------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------------
    ReconstructionCardiacVelocity4D::ReconstructionCardiacVelocity4D() : ReconstructionCardiac4D() {
        _recon_type = _3D;

        _no_sr = true;
        _adaptive_regularisation = true;
        _limit_intensities = false;

        _v_directions.clear();
        for (int i = 0; i < 3; i++) {
            Array<double> tmp;
            for (int j = 0; j < 3; j++)
                tmp.push_back(i == j ? 1 : 0);
            _v_directions.push_back(move(tmp));
        }
    }

    // -----------------------------------------------------------------------------
    // Check reconstruction quality
    // -----------------------------------------------------------------------------
    double ReconstructionCardiacVelocity4D::Consistency() {
        double ssd = 0;
        int num = 0;

        #pragma omp parallel for reduction(+: ssd, num)
        for (size_t index = 0; index < _slices.size(); index++)
            for (int i = 0; i < _slices[index].GetX(); i++)
                for (int j = 0; j < _slices[index].GetY(); j++)
                    if (_slices[index](i, j, 0) >= 0 && _simulated_inside[index](i, j, 0) == 1) {
                        const double diff = _slices[index](i, j, 0) - _simulated_slices[index](i, j, 0);
                        ssd += diff * diff;
                        num++;
                    }

        cout << " - Consistency : " << ssd << " " << sqrt(ssd / num) << endl;

        return ssd;
    }

    // -----------------------------------------------------------------------------
    // Simulation of phase slices from velocity volumes
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::SimulateSlicesCardiacVelocity4D() {
        if (_debug)
            cout << "Simulating slices ... ";

        Parallel::SimulateSlicesCardiacVelocity4D parallelSimulateSlices(this);
        parallelSimulateSlices();

        if (_debug)
            cout << "done." << endl;
    }

    // -----------------------------------------------------------------------------
    // Initialisation of slice gradients with respect to slice transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::InitializeSliceGradients4D() {
        ClearAndResize(_slice_g_directions, _slices.size());
        ClearAndResize(_simulated_velocities, _slices.size());

        #pragma omp parallel for
        for (size_t i = 0; i < _slices.size(); i++) {
            double gx = _g_directions[_stack_index[i]][0];
            double gy = _g_directions[_stack_index[i]][1];
            double gz = _g_directions[_stack_index[i]][2];

            _transformations[i].Rotate(gx, gy, gz);
            _slice_g_directions[i] = {gx, gy, gz};

            RealImage slice(_slices[i].Attributes());
            Array<RealImage> array_slices(3, slice);
            _simulated_velocities[i] = move(array_slices);
        }

        if (_debug) {
            ofstream GD_csv_file("input-gradient-directions.csv");
            GD_csv_file << "Stack,Slice,Gx - org,Gy - org,Gz - org,Gx - new,Gy - new,Gz - new,Rx,Ry,Rz\n";
            for (size_t i = 0; i < _slice_g_directions.size(); i++) {
                double rx = _transformations[i].GetRotationX();
                double ry = _transformations[i].GetRotationY();
                double rz = _transformations[i].GetRotationZ();
                GD_csv_file << _stack_index[i] << "," << i << "," << _g_directions[_stack_index[i]][0] << "," << _g_directions[_stack_index[i]][1] << "," << _g_directions[_stack_index[i]][2] << "," << _slice_g_directions[i][0] << "," << _slice_g_directions[i][1] << "," << _slice_g_directions[i][2] << "," << rx << "," << ry << "," << rz << "\n";
            }
        }

        if (!_slice_temporal_weight.empty() && _reconstructed4D.GetT() == 1) {
            #pragma omp simd
            for (int t = 0; t < _reconstructed4D.GetT(); t++)
                for (size_t i = 0; i < _slices.size(); i++)
                    _slice_temporal_weight[t][i] = 1;
        }
    }

    // -----------------------------------------------------------------------------
    // Saving slice info int .csv
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::SaveSliceInfo() {
        ofstream GD_csv_file("summary-slice-info.csv");
        GD_csv_file << "Stack,Slice,G value,Gx - org,Gy - org,Gz - org,Gx - new,Gy - new,Gz - new,Rx,Ry,Rz,Tx,Ty,Tz,Weight,Inside\n";

        for (size_t i = 0; i < _slice_g_directions.size(); i++) {
            double rx = _transformations[i].GetRotationX();
            double ry = _transformations[i].GetRotationY();
            double rz = _transformations[i].GetRotationZ();
            double tx = _transformations[i].GetTranslationX();
            double ty = _transformations[i].GetTranslationY();
            double tz = _transformations[i].GetTranslationZ();

            int inside = 0;
            if (_slice_inside[i])
                inside = 1;

            GD_csv_file << _stack_index[i] << "," << i << "," << _g_values[_stack_index[i]] << "," << _g_directions[_stack_index[i]][0] << "," << _g_directions[_stack_index[i]][1] << "," << _g_directions[_stack_index[i]][2] << "," << _slice_g_directions[i][0] << "," << _slice_g_directions[i][1] << "," << _slice_g_directions[i][2] << "," << rx << "," << ry << "," << rz << "," << tx << "," << ty << "," << tz << "," << _slice_weight[i] << "," << inside << endl;
        }

        ofstream W_csv_file("temporal-weights.csv");
        for (size_t i = 0; i < _slices.size(); i++) {
            W_csv_file << i << ",";
            for (int t = 0; t < _reconstructed4D.GetT(); t++)
                W_csv_file << _slice_temporal_weight[t][i] << ",";
            W_csv_file << "\n";
        }
    }

    // -----------------------------------------------------------------------------
    // Gradient descend step of velocity estimation
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::SuperresolutionCardiacVelocity4D(int iter) {
        SVRTK_START_TIMING();

        if (_debug)
            cout << "Superresolution" << endl;

        // Remember current reconstruction for edge-preserving smoothing
        Array<RealImage> originals = _reconstructed5DVelocity;

        Parallel::SuperresolutionCardiacVelocity4D parallelSuperresolution(this);
        parallelSuperresolution();

        Array<RealImage>& addons = parallelSuperresolution.addons;
        _confidence_maps_velocity = move(parallelSuperresolution.confidence_maps);

        if (_debug) {
            if (_reconstructed5DVelocity[0].GetT() == 1) {
                ImageAttributes attr = _reconstructed5DVelocity[0].Attributes();
                attr._t = 3;
                RealImage output_4D(attr);

                #pragma omp parallel for
                for (int x = 0; x < output_4D.GetX(); x++)
                    for (int y = 0; y < output_4D.GetY(); y++)
                        for (int z = 0; z < output_4D.GetZ(); z++)
                            for (int t = 0; t < output_4D.GetT(); t++)
                                output_4D(x, y, z, t) = addons[t](x, y, z, 0);

                output_4D.Write((boost::format("addon-velocity-%i.nii.gz") % iter).str().c_str());
            } else {
                #pragma omp parallel for num_threads(_io_thread_count)
                for (size_t i = 0; i < _v_directions.size(); i++) {
                    addons[i].Write((boost::format("addon-velocity-%i-%i.nii.gz") % i % iter).str().c_str());
                    _confidence_maps_velocity[i].Write((boost::format("confidence-map-velocity-%i-%i.nii.gz") % i % iter).str().c_str());
                }
            }
        }

        for (size_t v = 0; v < _v_directions.size(); v++) {
            RealPixel *pa = addons[v].Data();
            RealPixel *pcm = _confidence_maps_velocity[v].Data();
            #pragma omp parallel for
            for (int i = 0; i < addons[v].NumberOfVoxels(); i++) {
                if (pcm[i] > 0) {
                    // ISSUES if pcm[i] is too small leading to bright pixels
                    pa[i] /= pcm[i];
                    //this is to revert to normal (non-adaptive) regularisation
                    pcm[i] = 1;
                }
            }
        }

        #pragma omp parallel for
        for (size_t v = 0; v < _v_directions.size(); v++)
            _reconstructed5DVelocity[v] += addons[v] * _alpha; //_average_volume_weight;

        // Smooth the reconstructed image
        if (_adaptive_regularisation)
            AdaptiveRegularizationCardiacVelocity4D(iter, originals);

        // Limit velocity values with respect to maximum / minimum values (in order to avoid discontinuities)
        if (_limit_intensities) {
            for (size_t v = 0; v < _v_directions.size(); v++) {
                RealPixel *pr = _reconstructed5DVelocity[v].Data();
                #pragma omp parallel for
                for (int i = 0; i < _reconstructed5DVelocity[v].NumberOfVoxels(); i++) {
                    if (pr[i] < _min_velocity * 0.9)
                        pr[i] = _min_velocity * 0.9;
                    if (pr[i] > _max_velocity * 1.1)
                        pr[i] = _max_velocity * 1.1;
                }
            }
        }

        //Remove the bias in the reconstructed volume compared to previous iteration
        //  TODO: update adaptive regularisation for 4d
        //  if (_global_bias_correction)
        //  BiasCorrectVolume(original);

        SVRTK_END_TIMING("Superresolution");
    }

    // -----------------------------------------------------------------------------
    // Adaptive Regularization
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::AdaptiveRegularizationCardiacVelocity4D(int iter, Array<RealImage>& originals) {
        if (_debug)
            cout << "AdaptiveRegularizationCardiacVelocity4D" << endl;

        Array<double> factor(13, 0);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }

        RealImage original;
        Array<RealImage> b(13);

        if (_alpha * _lambda / (_delta * _delta) > 0.068)
            cerr << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068." << endl;

        for (size_t i = 0; i < _reconstructed5DVelocity.size(); i++) {
            _reconstructed4D = move(_reconstructed5DVelocity[i]);
            _confidence_map = _confidence_maps_velocity[i];

            #pragma omp parallel for
            for (size_t j = 0; j < b.size(); j++)
                b[j] = _reconstructed4D;

            Parallel::AdaptiveRegularization1Cardiac4D parallelAdaptiveRegularization1(this, b, factor, originals[i]);
            parallelAdaptiveRegularization1();

            original = _reconstructed4D;
            Parallel::AdaptiveRegularization2Cardiac4D parallelAdaptiveRegularization2(this, b, original);
            parallelAdaptiveRegularization2();

            _reconstructed5DVelocity[i] = move(_reconstructed4D);
        }

        // Restore to its original value
        _reconstructed4D = move(b[0]);
    }

    // -----------------------------------------------------------------------------
    // Save output files (simulated velocity and phase)
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::SaveOuput(Array<RealImage> stacks) {
        if (_debug)
            cout << "Saving simulated velocity slices as stacks ...";

        #pragma omp parallel for
        for (size_t i = 0; i < stacks.size(); i++)
            memset(stacks[i].Data(), 0, sizeof(RealPixel) * stacks[i].NumberOfVoxels());

        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            for (int i = 0; i < _slices[inputIndex].GetX(); i++)
                for (int j = 0; j < _slices[inputIndex].GetY(); j++)
                    stacks[_stack_index[inputIndex]](i, j, _stack_loc_index[inputIndex], _stack_dyn_index[inputIndex]) = _simulated_slices[inputIndex](i, j, 0);

        #pragma omp parallel for num_threads(_io_thread_count)
        for (size_t i = 0; i < stacks.size(); i++)
            stacks[i].Write((boost::format("simulated-phase-%i.nii.gz") % i).str().c_str());

        if (!_simulated_velocities.empty()) {
            Array<RealImage> velocity_volumes;

            if (_reconstructed4D.GetT() == 1) {
                velocity_volumes.resize(stacks.size());
                #pragma omp parallel for
                for (size_t i = 0; i < stacks.size(); i++) {
                    ImageAttributes attr = stacks[i].Attributes();
                    attr._t = 3;
                    velocity_volumes[i] = RealImage(attr);
                }
            }

            for (int v = 0; v < 3; v++) {
                #pragma omp parallel for
                for (size_t i = 0; i < stacks.size(); i++)
                    memset(stacks[i].Data(), 0, sizeof(RealPixel) * stacks[i].NumberOfVoxels());

                #pragma omp parallel for
                for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                    for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                        for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                            double vel_val = _simulated_velocities[inputIndex][v](i, j, 0);

                            if (vel_val < _min_velocity || vel_val > _max_velocity)
                                vel_val = -1;

                            if (_reconstructed4D.GetT() > 1)
                                stacks[_stack_index[inputIndex]](i, j, _stack_loc_index[inputIndex], _stack_dyn_index[inputIndex]) = vel_val;
                            else
                                velocity_volumes[_stack_index[inputIndex]](i, j, _stack_loc_index[inputIndex], v) = vel_val;
                        }
                    }
                }

                if (_reconstructed4D.GetT() > 1) {
                    #pragma omp parallel for num_threads(_io_thread_count)
                    for (size_t i = 0; i < stacks.size(); i++)
                        stacks[i].Write((boost::format("simulated-velocity-%i-%i.nii.gz") % i % v).str().c_str());
                }
            }

            if (_reconstructed4D.GetT() == 1) {
                #pragma omp parallel for num_threads(_io_thread_count)
                for (size_t i = 0; i < stacks.size(); i++)
                    velocity_volumes[i].Write((boost::format("simulated-velocity-vector-%i.nii.gz") % i).str().c_str());
            }
        }

        for (int v = 0; v < 3; v++) {
            #pragma omp parallel for
            for (size_t i = 0; i < stacks.size(); i++)
                memset(stacks[i].Data(), 0, sizeof(RealPixel) * stacks[i].NumberOfVoxels());

            #pragma omp parallel for
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                for (int i = 0; i < _slices[inputIndex].GetX(); i++)
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++)
                        stacks[_stack_index[inputIndex]](i, j, _stack_loc_index[inputIndex], _stack_dyn_index[inputIndex]) = _weights[inputIndex](i, j, 0);

            #pragma omp parallel for num_threads(_io_thread_count)
            for (size_t i = 0; i < stacks.size(); i++)
                stacks[i].Write((boost::format("weight-%i-%i.nii.gz") % i % v).str().c_str());
        }

        if (_debug)
            cout << " done." << endl;
    }

    // -----------------------------------------------------------------------------
    // Mask phase slices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::MaskSlicesPhase() {
        cout << "Masking slices ... ";

        //Check whether we have a mask
        if (!_have_mask) {
            cerr << "Could not mask slices because no mask has been set." << endl;
            return;
        }

        //mask slices
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            RealImage& slice = _slices[inputIndex];
            for (int i = 0; i < slice.GetX(); i++)
                for (int j = 0; j < slice.GetY(); j++) {
                    //image coordinates of a slice voxel
                    double x = i;
                    double y = j;
                    double z = 0;
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
                    if (_mask.IsInside(x, y, z)) {
                        if (_mask(x, y, z) == 0)
                            slice(i, j, 0) = -15;
                    } else
                        slice(i, j, 0) = -15;
                }
        }

        cout << "done." << endl;
    }

    // -----------------------------------------------------------------------------
    // Initialisation of EM step
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::InitializeEMVelocity4D() {
        if (_debug)
            cout << "InitializeEM" << endl;

        ClearAndResize(_weights, _slices.size());
        ClearAndResize(_bias, _slices.size());
        ClearAndResize(_scale, _slices.size(), 1.0);
        ClearAndResize(_slice_weight, _slices.size(), 1.0);

        #pragma omp parallel for
        for (size_t i = 0; i < _slices.size(); i++) {
            //Create images for voxel weights and bias fields
            RealImage tmp(_slices[i].Attributes());

            _bias[i] = tmp;

            tmp = 1;
            _weights[i] = move(tmp);
        }

        //Find the range of intensities
        _max_intensity = voxel_limits<RealPixel>::min();
        _min_intensity = voxel_limits<RealPixel>::max();

        #pragma omp parallel for reduction(min: _min_intensity) reduction(max: _max_intensity)
        for (size_t i = 0; i < _slices.size(); i++) {
            //to update minimum we need to exclude padding value
            RealPixel *ptr = _slices[i].Data();
            for (int ind = 0; ind < _slices[i].NumberOfVoxels(); ind++) {
                if (ptr[ind] > -10) {
                    const double tmp = abs(ptr[ind]);
                    if (tmp > _max_intensity)
                        _max_intensity = tmp;
                    if (tmp < _min_intensity)
                        _min_intensity = tmp;
                }
            }
        }

        if (_debug)
            cout << " - min : " << _min_intensity << " | max : " << _max_intensity << endl;
    }

    // -----------------------------------------------------------------------------
    // Initialisation of EM values
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::InitializeEMValuesVelocity4D() {
        SVRTK_START_TIMING();

        #pragma omp parallel for
        for (size_t i = 0; i < _slices.size(); i++) {
            //Initialise voxel weights and bias values
            const RealPixel *pi = _slices[i].Data();
            RealPixel *pw = _weights[i].Data();
            RealPixel *pb = _bias[i].Data();
            for (int j = 0; j < _weights[i].NumberOfVoxels(); j++) {
                pw[j] = pi[j] > -10 ? 1 : 0;
                pb[j] = 0;
            }

            //Initialise slice weights
            _slice_weight[i] = 1;

            //Initialise scaling factors for intensity matching
            _scale[i] = 1;
        }

        //Force exclusion of slices predefined by user
        for (size_t i = 0; i < _force_excluded.size(); i++)
            _slice_weight[_force_excluded[i]] = 0;

        SVRTK_END_TIMING("InitializeEMValues");
    }

    // -----------------------------------------------------------------------------
    // Initialisation of robust statistics
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::InitializeRobustStatisticsVelocity4D() {
        if (_debug) {
            cout << "Initialising robust statistics: ";
            cout.flush();
        }

        //Initialise parameter of EM robust statistics
        double sigma = 0;
        int num = 0;

        //for each slice
        #pragma omp parallel for reduction(+: sigma, num)
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            const RealImage& current_slice = _slices[inputIndex];
            RealImage slice = current_slice - _simulated_slices[inputIndex];

            for (int i = 0; i < current_slice.GetX(); i++)
                for (int j = 0; j < current_slice.GetY(); j++)
                    if (current_slice(i, j, 0) < -10)
                        slice(i, j, 0) = -15;

            //Voxel-wise sigma will be set to stdev of volumetric errors
            //For each slice voxel
            for (int i = 0; i < slice.GetX(); i++)
                for (int j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) > -10) {
                        //calculate stev of the errors
                        if ((_simulated_inside[inputIndex](i, j, 0) == 1) && (_simulated_weights[inputIndex](i, j, 0) > 0.99)) {
                            sigma += slice(i, j, 0) * slice(i, j, 0);
                            num++;
                        }
                    }

            //if slice does not have an overlap with ROI, set its weight to zero
            if (!_slice_inside[inputIndex])
                _slice_weight[inputIndex] = 0;
        }

        //Force exclusion of slices predefined by user
        for (size_t i = 0; i < _force_excluded.size(); i++)
            _slice_weight[_force_excluded[i]] = 0;

        //initialize sigma for voxel-wise robust statistics
        _sigma = sigma / num;
        //initialize sigma for slice-wise robust statistics
        _sigma_s = 0.025;
        //initialize mixing proportion for inlier class in voxel-wise robust statistics
        _mix = 0.9;
        //initialize mixing proportion for outlier class in slice-wise robust statistics
        _mix_s = 0.9;
        //Initialise value for uniform distribution according to the range of intensities
        _m = 1 / (2.1 * _max_intensity - 1.9 * _min_intensity);

        if (_debug)
            cout << "sigma=" << sqrt(_sigma) << " m=" << _m << " mix=" << _mix << " mix_s=" << _mix_s << endl;
    }

    // -----------------------------------------------------------------------------
    // E-step (todo: optimise for velocity)
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::EStepVelocity4D() {
        //EStep performs calculation of voxel-wise and slice-wise posteriors (weights)
        if (_debug)
            cout << "EStep: " << endl;

        Array<double> slice_potential(_slices.size());

        Parallel::EStepCardiacVelocity4D parallelEStep(this, slice_potential);
        parallelEStep();

        //To force-exclude slices predefined by a user, set their potentials to -1
        for (size_t i = 0; i < _force_excluded.size(); i++)
            slice_potential[_force_excluded[i]] = -1;

        //exclude slices identified as having small overlap with ROI, set their potentials to -1
        for (size_t i = 0; i < _small_slices.size(); i++)
            slice_potential[_small_slices[i]] = -1;

        //these are unrealistic scales pointing at misregistration - exclude the corresponding slices
        for (size_t inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
            if (_scale[inputIndex] < 0.2 || _scale[inputIndex] > 5)
                slice_potential[inputIndex] = -1;

        // exclude unrealistic transformations
        if (_debug) {
            cout << endl << "Slice potentials: ";
            for (size_t inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
                cout << slice_potential[inputIndex] << " ";
            cout << endl;
        }

        //Calculation of slice-wise robust statistics parameters.
        //This is theoretically M-step,
        //but we want to use latest estimate of slice potentials
        //to update the parameters

        //Calculate means of the inlier and outlier potentials
        double sum = 0, den = 0, sum2 = 0, den2 = 0, maxs = 0, mins = 1;
        #pragma omp parallel for reduction(+: sum, sum2, den, den2) reduction(max: maxs) reduction(min: mins)
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
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

        _mean_s = den > 0 ? sum / den : mins;
        _mean_s2 = den2 > 0 ? sum2 / den2 : (maxs + _mean_s) / 2;

        //Calculate the variances of the potentials
        sum = 0; den = 0; sum2 = 0; den2 = 0;
        #pragma omp parallel for reduction(+: sum, sum2, den, den2)
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            if (slice_potential[inputIndex] >= 0) {
                sum += (slice_potential[inputIndex] - _mean_s) * (slice_potential[inputIndex] - _mean_s) * _slice_weight[inputIndex];
                den += _slice_weight[inputIndex];
                sum2 += (slice_potential[inputIndex] - _mean_s2) * (slice_potential[inputIndex] - _mean_s2) * (1 - _slice_weight[inputIndex]);
                den2 += 1 - _slice_weight[inputIndex];
            }

        //_sigma_s
        if ((sum > 0) && (den > 0)) {
            //do not allow too small sigma
            _sigma_s = max(sum / den, _step * _step / 6.28);
        } else {
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
            //do not allow too small sigma
            _sigma_s2 = max(sum2 / den2, _step * _step / 6.28);
        } else {
            //do not allow too small sigma
            _sigma_s2 = max((_mean_s2 - _mean_s) * (_mean_s2 - _mean_s) / 4, _step * _step / 6.28);

            if (_debug) {
                if (sum2 <= 0)
                    cout << "All slices are equal. ";
                if (den2 <= 0)
                    cout << "All slices inliers. ";
                cout << "Setting sigma_s2 to " << sqrt(_sigma_s2) << endl;
            }
        }

        //Calculate slice weights
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            //Slice does not have any voxels in volumetric ROI
            if (slice_potential[inputIndex] == -1) {
                _slice_weight[inputIndex] = 0;
                continue;
            }

            //All slices are outliers or the means are not valid
            if (den <= 0 || _mean_s2 <= _mean_s) {
                _slice_weight[inputIndex] = 1;
                continue;
            }

            //likelihood for inliers
            const double gs1 = slice_potential[inputIndex] < _mean_s2 ? G(slice_potential[inputIndex] - _mean_s, _sigma_s) : 0;

            //likelihood for outliers
            const double gs2 = slice_potential[inputIndex] > _mean_s ? G(slice_potential[inputIndex] - _mean_s2, _sigma_s2) : 0;

            //calculate slice weight
            const double likelihood = gs1 * _mix_s + gs2 * (1 - _mix_s);
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
        int num = 0; sum = 0;
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
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
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                cout << _slice_weight[inputIndex] << " ";
            cout << endl;
        }
    }

    // -----------------------------------------------------------------------------
    // M-step (todo: optimise for velocity)
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::MStepVelocity4D(int iter) {
        if (_debug)
            cout << "MStep" << endl;

        Parallel::MStepCardiacVelocity4D parallelMStep(this);
        parallelMStep();
        const double sigma = parallelMStep.sigma;
        const double mix = parallelMStep.mix;
        const double num = parallelMStep.num;
        const double min = parallelMStep.min;
        const double max = parallelMStep.max;

        //Calculate sigma and mix
        if (mix > 0) {
            _sigma = sigma / mix;
        } else {
            throw runtime_error("Something went wrong: sigma=" + to_string(sigma) + " mix=" + to_string(mix));
        }
        if (_sigma < _step * _step / 6.28)
            _sigma = _step * _step / 6.28;
        if (iter > 1)
            _mix = mix / num;

        //Calculate m
        _m = 1 / (max - min);

        if (_debug)
            cout << "Voxel-wise robust statistics parameters: sigma = " << sqrt(_sigma) << " mix = " << _mix << " m = " << _m << " max = " << max << " min = " << min << endl;
    }

    // -----------------------------------------------------------------------------
    // Save reconstructed velocity volumes
    // -----------------------------------------------------------------------------
    void ReconstructionCardiacVelocity4D::SaveReconstructedVelocity4D(int iter) {
        string filename;

        cout << " - Reconstructed velocity : " << endl;

        if (_reconstructed5DVelocity[0].GetT() == 1) {
            ImageAttributes attr = _reconstructed5DVelocity[0].Attributes();
            attr._t = 3;
            RealImage output_4D(attr);

            #pragma omp parallel for
            for (int x = 0; x < output_4D.GetX(); x++)
                for (int y = 0; y < output_4D.GetY(); y++)
                    for (int z = 0; z < output_4D.GetZ(); z++)
                        for (int t = 0; t < output_4D.GetT(); t++)
                            output_4D(x, y, z, t) = _reconstructed5DVelocity[t](x, y, z, 0);

            if (iter < 0)
                filename = "velocity-vector-init.nii.gz";
            else if (iter < 50)
                filename = (boost::format("velocity-vector-%i.nii.gz") % iter).str();
            else
                filename = "../velocity-vector-final.nii.gz";

            output_4D.Write(filename.c_str());
            cout << "        " << filename << endl;

            attr._t = 1;
            RealImage output_sum(attr);

            for (int v = 0; v < output_4D.GetT(); v++)
                output_sum += output_4D.GetRegion(0, 0, 0, v, attr._x, attr._y, attr._z, v + 1);

            output_sum.Write("sum-velocity.nii.gz");
        } else {
            RealImage output_sum(_reconstructed5DVelocity[0].Attributes());

            for (int i = 0; i < _reconstructed5DVelocity.size(); i++) {
                output_sum += _reconstructed5DVelocity[i];

                if (iter < 50)
                    filename = (boost::format("velocity-%i-%i.nii.gz") % i % iter).str();
                else
                    filename = (boost::format("velocity-final-%i.nii.gz") % i).str();

                _reconstructed5DVelocity[i].Write(filename.c_str());
                cout << "     " << filename << endl;
            }

            if (iter < 50)
                filename = (boost::format("sum-velocity-%i.nii.gz") % iter).str();
            else
                filename = "sum-velocity-final.nii.gz";

            output_sum.Write(filename.c_str());
        }

        cout << "..............................................\n";
        cout << ".............................................." << endl;
    }

} // namespace svrtk
