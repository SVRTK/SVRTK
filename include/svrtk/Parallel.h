/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2021- King's College London
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

#pragma once

// SVRTK
#include "svrtk/ReconstructionCardiacVelocity4D.h"

using namespace mirtk;
using namespace svrtk;
using namespace svrtk::Utility;

namespace svrtk::Parallel {

    /// Class for computing average of all stacks
    class Average {
        const Reconstruction *reconstructor;
        const Array<RealImage>& stacks;
        const Array<RigidTransformation>& stack_transformations;

        /// Padding value in target (voxels in the target image with this value will be ignored)
        double targetPadding;

        /// Padding value in source (voxels outside the source image will be set to this value)
        double sourcePadding;

        double background;

        // Volumetric registrations are stack-to-template while slice-to-volume
        // registrations are actually performed as volume-to-slice
        // (reasons: technicalities of implementation) so transformations need to be inverted beforehand.

        bool linear;

    public:
        RealImage average;
        RealImage weights;

        Average(
            const Reconstruction *reconstructor,
            const Array<RealImage>& stacks,
            const Array<RigidTransformation>& stack_transformations,
            double targetPadding,
            double sourcePadding,
            double background,
            bool linear = false) :
            reconstructor(reconstructor),
            stacks(stacks),
            stack_transformations(stack_transformations),
            targetPadding(targetPadding),
            sourcePadding(sourcePadding),
            background(background),
            linear(linear) {
            average.Initialize(reconstructor->_reconstructed.Attributes());
            weights.Initialize(reconstructor->_reconstructed.Attributes());
        }

        Average(Average& x, split) : Average(x.reconstructor, x.stacks, x.stack_transformations, x.targetPadding, x.sourcePadding, x.background, x.linear) {}

        void operator()(const blocked_range<size_t>& r) {
            GenericLinearInterpolateImageFunction<RealImage> interpolator;
            ImageTransformation imagetransformation;
            RealImage image;

            for (size_t i = r.begin(); i < r.end(); i++) {
                image.Initialize(reconstructor->_reconstructed.Attributes());

                imagetransformation.Input(&stacks[i]);
                imagetransformation.Transformation(&stack_transformations[i]);
                imagetransformation.Output(&image);
                imagetransformation.TargetPaddingValue(targetPadding);
                imagetransformation.SourcePaddingValue(sourcePadding);
                imagetransformation.Interpolator(&interpolator);
                imagetransformation.Run();

                const RealPixel *pi = image.Data();
                RealPixel *pa = average.Data();
                RealPixel *pw = weights.Data();
                for (int p = 0; p < average.NumberOfVoxels(); p++) {
                    if (pi[p] != background) {
                        pa[p] += pi[p];
                        pw[p]++;
                    }
                }
            }
        }

        void join(const Average& y) {
            average += y.average;
            weights += y.weights;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, stacks.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class QualityReport {
        Reconstruction *reconstructor;

    public:
        double out_global_ncc;
        double out_global_nrmse;

        QualityReport(Reconstruction *reconstructor) : reconstructor(reconstructor) {
            out_global_ncc = 0;
            out_global_nrmse = 0;
        }

        QualityReport(QualityReport& x, split) : QualityReport(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                const double out_ncc = ComputeNCC(reconstructor->_slices[inputIndex], reconstructor->_simulated_slices[inputIndex], 0.1);

                if (out_ncc > 0)
                    out_global_ncc += out_ncc;

                double s_diff = 0;
                double s_t = 0;
                int s_n = 0;
                double point_ns = 0;
                double point_nt = 0;

                for (int x = 0; x < reconstructor->_slices[inputIndex].GetX(); x++) {
                    for (int y = 0; y < reconstructor->_slices[inputIndex].GetY(); y++) {
                        for (int z = 0; z < reconstructor->_slices[inputIndex].GetZ(); z++) {
                            double test_val = reconstructor->_simulated_slices[inputIndex](x, y, z);
                            if (reconstructor->_no_masking_background)
                                test_val = reconstructor->_simulated_slices[inputIndex](x, y, z) * reconstructor->_slice_masks[inputIndex](x, y, z);

                            if (reconstructor->_slices[inputIndex](x, y, z) > 0 && test_val > 0) {
                                point_nt = reconstructor->_slices[inputIndex](x, y, z) * exp(-(reconstructor->_bias[inputIndex])(x, y, 0)) * reconstructor->_scale[inputIndex];
                                point_ns = reconstructor->_simulated_slices[inputIndex](x, y, z);

                                s_t += point_nt;
                                s_diff += (point_nt - point_ns) * (point_nt - point_ns);
                                s_n++;
                            }
                        }
                    }
                }

                if (s_n > 0 && s_t > 0) {
                    const double out_nrmse = sqrt(s_diff / s_n) / (s_t / s_n);
                    out_global_nrmse += out_nrmse;
                }

                if (!isfinite(out_global_nrmse))
                    out_global_nrmse = 0;

            } //end of loop for a slice inputIndex
        }

        void join(const QualityReport& y) {
            out_global_ncc += y.out_global_ncc;
            out_global_nrmse += y.out_global_nrmse;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    // class for global 3D stack registration
    class StackRegistrations {
        const Reconstruction *reconstructor;
        const Array<RealImage>& stacks;
        Array<RigidTransformation>& stack_transformations;
        int templateNumber;
        const RealImage& target;
        const RigidTransformation& offset;

    public:
        StackRegistrations(
            const Reconstruction *reconstructor,
            const Array<RealImage>& stacks,
            Array<RigidTransformation>& stack_transformations,
            int templateNumber,
            const RealImage& target,
            const RigidTransformation& offset) :
            reconstructor(reconstructor),
            stacks(stacks),
            stack_transformations(stack_transformations),
            templateNumber(templateNumber),
            target(target),
            offset(offset) {}

        void operator()(const blocked_range<size_t>& r) const {
            RealImage r_source, r_target;
            GenericLinearInterpolateImageFunction<RealImage> interpolator;
            ResamplingWithPadding<RealPixel> resampling(1.2, 1.2, 1.2, 0);
            resampling.Interpolator(&interpolator);

            ParameterList params;
            Insert(params, "Transformation model", "Rigid");
            if (reconstructor->_nmi_bins > 0)
                Insert(params, "No. of bins", reconstructor->_nmi_bins);

            if (reconstructor->_masked_stacks)
                Insert(params, "Background value", 0);
            else
                Insert(params, "Background value for image 1", 0);

            // Insert(params, "Background value", 0);
            // Insert(params, "Image (dis-)similarity measure", "NCC");
            // Insert(params, "Image interpolation mode", "Linear");
            // string type = "box";
            // string units = "mm";
            // double width = 0;
            // Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);

            Transformation *dofout;
            GenericRegistrationFilter registration;
            registration.Parameter(params);
            registration.Output(&dofout);

            for (size_t i = r.begin(); i != r.end(); i++) {

                r_source.Initialize(stacks[i].Attributes());
                resampling.Input(&stacks[i]);
                resampling.Output(&r_source);
                resampling.Run();

                r_target.Initialize(target.Attributes());
                resampling.Input(&target);
                resampling.Output(&r_target);
                resampling.Run();

                const Matrix& mo = offset.GetMatrix();
                stack_transformations[i].PutMatrix(stack_transformations[i].GetMatrix() * mo);

                registration.Input(&r_target, &r_source);
                registration.InitialGuess(&stack_transformations[i]);
                registration.GuessParameter();
                registration.Run();

                unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(dofout));
                stack_transformations[i] = *rigidTransf;

                stack_transformations[i].PutMatrix(stack_transformations[i].GetMatrix() * mo.Inverse());

                //save volumetric registrations
                if (reconstructor->_debug) {
                    stack_transformations[i].Write((boost::format("global-transformation%1%.dof") % i).str().c_str());
                    stacks[i].Write((boost::format("global-stack%1%.nii.gz") % i).str().c_str());
                }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, stacks.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    // class for simulation of slice masks
    class SimulateMasks {
        Reconstruction *reconstructor;

    public:
        SimulateMasks(SimulateMasks& x, split) : SimulateMasks(x.reconstructor) {}

        SimulateMasks(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                //Calculate simulated slice
                RealImage& sim_mask = reconstructor->_slice_masks[inputIndex];
                memset(sim_mask.Data(), 0, sizeof(RealPixel) * sim_mask.NumberOfVoxels());
                bool slice_inside = false;

                for (int i = 0; i < reconstructor->_slice_attributes[inputIndex]._x; i++) {
                    for (int j = 0; j < reconstructor->_slice_attributes[inputIndex]._y; j++) {
                        if (reconstructor->_slices[inputIndex](i, j, 0) > -0.01) {
                            double weight = 0;
                            const size_t n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for (size_t k = 0; k < n; k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                sim_mask(i, j, 0) += p.value * reconstructor->_evaluation_mask(p.x, p.y, p.z);
                                weight += p.value;
                            }
                            if (weight > 0)
                                sim_mask(i, j, 0) /= weight;
                            if (sim_mask(i, j, 0) > 0.1)
                                slice_inside = true;
                        } else {
                            sim_mask(i, j, 0) = 0;
                        }
                    }
                }

                if (slice_inside) {
                    GaussianBlurring<RealPixel> gb(2);
                    gb.Input(&sim_mask);
                    gb.Output(&sim_mask);
                    gb.Run();

                    RealPixel *pm = sim_mask.Data();
                    for (int i = 0; i < sim_mask.NumberOfVoxels(); i++)
                        if (pm[i] > 0.3)
                            pm[i] = 1;
                }

            } //end of loop for a slice inputIndex
        }

        void join(const SimulateMasks& y) {}

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for simulation of slices from the current reconstructed 3D volume - v2 (use this one)
    class SimulateSlices {
        Reconstruction *reconstructor;

    public:
        SimulateSlices(SimulateSlices& x, split) : SimulateSlices(x.reconstructor) {}

        SimulateSlices(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                //Calculate simulated slice
                RealImage& sim_slice = reconstructor->_simulated_slices[inputIndex];
                RealImage& sim_weight = reconstructor->_simulated_weights[inputIndex];
                RealImage& sim_inside = reconstructor->_simulated_inside[inputIndex];

                memset(sim_slice.Data(), 0, sizeof(RealPixel) * sim_slice.NumberOfVoxels());
                memset(sim_weight.Data(), 0, sizeof(RealPixel) * sim_weight.NumberOfVoxels());
                memset(sim_inside.Data(), 0, sizeof(RealPixel) * sim_inside.NumberOfVoxels());
                reconstructor->_slice_inside[inputIndex] = false;

                if (reconstructor->_multiple_channels_flag) {
                    for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                        for ( int i = 0; i < reconstructor->_simulated_slices[inputIndex].GetX(); i++ ) {
                            for ( int j = 0; j < reconstructor->_simulated_slices[inputIndex].GetY(); j++ ) {
                                reconstructor->_mc_simulated_slices[inputIndex][nc]->PutAsDouble(i, j, 0, 0);
                            }
                        }
                    }
                }

                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++) {
                            double test_val = reconstructor->_slices[inputIndex](i, j, 0);
                            if (reconstructor->_no_masking_background)
                                test_val = reconstructor->_not_masked_slices[inputIndex](i, j, 0);
                            if (test_val > -0.01) {
                                double weight = 0;
                                for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                    const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                    sim_slice(i, j, 0) += p.value * reconstructor->_reconstructed(p.x, p.y, p.z);
                                    weight += p.value;

                                    for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                        double tmp_val = reconstructor->_mc_simulated_slices[inputIndex][nc]->GetAsDouble(i, j, 0) + p.value * reconstructor->_mc_reconstructed[nc](p.x, p.y, p.z);
                                        reconstructor->_mc_simulated_slices[inputIndex][nc]->PutAsDouble(i, j, 0, tmp_val);
                                    }

                                    if (reconstructor->_no_masking_background || reconstructor->_mask(p.x, p.y, p.z) > 0.1) {
                                        sim_inside(i, j, 0) = 1;
                                        reconstructor->_slice_inside[inputIndex] = true;
                                    }
                                }
                                if (weight > 0) {
                                    sim_slice(i, j, 0) /= weight;
                                    sim_weight(i, j, 0) = weight;

                                    for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                        double tmp_val = reconstructor->_mc_simulated_slices[inputIndex][nc]->GetAsDouble(i, j, 0) / weight;
                                        reconstructor->_mc_simulated_slices[inputIndex][nc]->PutAsDouble(i, j, 0, tmp_val);
                                    }

                                }
                            }
                        };
            } //end of loop for a slice inputIndex
        }

        void join(const SimulateSlices& y) {}

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class SimulateSlicesCardiac4D {
        ReconstructionCardiac4D *reconstructor;

    public:
        SimulateSlicesCardiac4D(ReconstructionCardiac4D *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                const RealImage& slice = reconstructor->_slices[inputIndex];
                //Calculate simulated slice
                reconstructor->_simulated_slices[inputIndex].Initialize(slice.Attributes());
                reconstructor->_simulated_weights[inputIndex].Initialize(slice.Attributes());
                reconstructor->_simulated_inside[inputIndex].Initialize(slice.Attributes());
                reconstructor->_slice_inside[inputIndex] = false;

                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (slice(i, j, 0) != -1) {
                            double weight = 0;
                            for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                for (int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++) {
                                    reconstructor->_simulated_slices[inputIndex](i, j, 0) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_reconstructed4D(p.x, p.y, p.z, outputIndex);
                                    weight += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                }
                                if (reconstructor->_mask(p.x, p.y, p.z) == 1) {
                                    reconstructor->_simulated_inside[inputIndex](i, j, 0) = 1;
                                    reconstructor->_slice_inside[inputIndex] = true;
                                }
                            }
                            if (weight > 0) {
                                reconstructor->_simulated_slices[inputIndex](i, j, 0) /= weight;
                                reconstructor->_simulated_weights[inputIndex](i, j, 0) = weight;
                            }
                        }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Simulation of phase slices from velocity volumes
    class SimulateSlicesCardiacVelocity4D {
        ReconstructionCardiacVelocity4D *reconstructor;

    public:
        SimulateSlicesCardiacVelocity4D(ReconstructionCardiacVelocity4D *reconstructor) : reconstructor(reconstructor) {}

        void operator() (const blocked_range<size_t> &r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                // calculate simulated slice
                reconstructor->_simulated_slices[inputIndex].Initialize(reconstructor->_slices[inputIndex].Attributes());
                reconstructor->_simulated_weights[inputIndex].Initialize(reconstructor->_slices[inputIndex].Attributes());
                reconstructor->_simulated_inside[inputIndex].Initialize(reconstructor->_slices[inputIndex].Attributes());
                reconstructor->_slice_inside[inputIndex] = false;

                memset(reconstructor->_simulated_velocities[inputIndex][0].Data(), 0, sizeof(RealPixel) * reconstructor->_simulated_velocities[inputIndex][0].NumberOfVoxels());
                memset(reconstructor->_simulated_velocities[inputIndex][1].Data(), 0, sizeof(RealPixel) * reconstructor->_simulated_velocities[inputIndex][1].NumberOfVoxels());
                memset(reconstructor->_simulated_velocities[inputIndex][2].Data(), 0, sizeof(RealPixel) * reconstructor->_simulated_velocities[inputIndex][2].NumberOfVoxels());

                // Gradient magnitude for the current slice
                const int gradientIndex = reconstructor->_stack_index[inputIndex];
                const double gval = reconstructor->_g_values[gradientIndex];

                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (reconstructor->_slices[inputIndex](i, j, 0) > -10) {
                            double weight = 0;
                            for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                for (int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++) {
                                    if (reconstructor->_reconstructed4D.GetT() == 1)
                                        reconstructor->_slice_temporal_weight[outputIndex][inputIndex] = 1;

                                    // Simulation of phase signal from velocity volumes
                                    double sim_signal = 0;

                                    for (size_t velocityIndex = 0; velocityIndex < reconstructor->_reconstructed5DVelocity.size(); velocityIndex++) {
                                        sim_signal += reconstructor->_reconstructed5DVelocity[velocityIndex](p.x, p.y, p.z, outputIndex) * gval * reconstructor->_slice_g_directions[inputIndex][velocityIndex];
                                        reconstructor->_simulated_velocities[inputIndex][velocityIndex](i, j, 0) += reconstructor->_reconstructed5DVelocity[velocityIndex](p.x, p.y, p.z, outputIndex) * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                    }

                                    reconstructor->_simulated_slices[inputIndex](i, j, 0) += sim_signal * reconstructor->gamma * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                    weight += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                }
                                if (reconstructor->_mask(p.x, p.y, p.z) == 1) {
                                    reconstructor->_simulated_inside[inputIndex](i, j, 0) = 1;
                                    reconstructor->_slice_inside[inputIndex] = true;
                                }
                            }

                            if (weight > 0) {
                                reconstructor->_simulated_slices[inputIndex](i, j, 0) /= weight;
                                reconstructor->_simulated_weights[inputIndex](i, j, 0) = weight;

                                for (size_t velocityIndex = 0; velocityIndex < reconstructor->_reconstructed5DVelocity.size(); velocityIndex++)
                                    reconstructor->_simulated_velocities[inputIndex][velocityIndex](i, j, 0) /= weight;
                            }
                        }
            }
        }

        // execute
        void operator() () const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for parallel SVR
    class SliceToVolumeRegistration {
        Reconstruction *reconstructor;

    public:
        SliceToVolumeRegistration(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            GreyPixel smin, smax, tmin, tmax;
            GreyImage target;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                target = reconstructor->_grey_slices[inputIndex];
                target.GetMinMax(&smin, &smax);

                if (smax > 1 && (smax - smin) > 1) {
                    ParameterList params;
                    Insert(params, "Transformation model", "Rigid");
                    reconstructor->_grey_reconstructed.GetMinMax(&tmin, &tmax);

                    if (!reconstructor->_no_offset_registration) {
                        
                        if (smin < 1)
                            Insert(params, "Background value for image 1", -1);
                        if (tmin < 0)
                            Insert(params, "Background value for image 2", -1);
                        else if (tmin < 1)
                            Insert(params, "Background value for image 2", 0);
                        
                    }

                    if (!reconstructor->_ncc_reg) {
                        Insert(params, "Image (dis-)similarity measure", "NMI");
                        if (reconstructor->_nmi_bins > 0)
                            Insert(params, "No. of bins", reconstructor->_nmi_bins);
                    } else {
                        Insert(params, "Image (dis-)similarity measure", "NCC");
                        const string type = "sigma";
                        const string units = "mm";
                        constexpr double width = 0;
                        Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);
                    }

                    GenericRegistrationFilter registration;
                    registration.Parameter(params);

                    //put origin to zero
                    RigidTransformation offset;
                    auto& transformation = reconstructor->_transformations[inputIndex];
                    
                    if (!reconstructor->_no_offset_registration) {
                        ResetOrigin(target, offset);
                        const Matrix& mo = offset.GetMatrix();
                        transformation.PutMatrix(transformation.GetMatrix() * mo);
                    }

                    // run registration
                    registration.Input(&target, &reconstructor->_grey_reconstructed);
                    Transformation *dofout;
                    registration.Output(&dofout);
                    registration.InitialGuess(&transformation);
                    registration.GuessParameter();
                    registration.Run();

                    // output transformation
                    unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(dofout));
                    transformation = *rigidTransf;

                    if (!reconstructor->_no_offset_registration) {
                        //undo the offset
                        const Matrix& mo = offset.GetMatrix();
                        transformation.PutMatrix(transformation.GetMatrix() * mo.Inverse());
                    }

                    // save log outputs
                    if (reconstructor->_reg_log) {
                        const double tx = transformation.GetTranslationX();
                        const double ty = transformation.GetTranslationY();
                        const double tz = transformation.GetTranslationZ();

                        const double rx = transformation.GetRotationX();
                        const double ry = transformation.GetRotationY();
                        const double rz = transformation.GetRotationZ();

                        cout << boost::format("%1% %1% %1% %1% %1% %1% %1%") % inputIndex % tx % ty % tz % rx % ry % rz << endl;
                    }
                }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class SliceToVolumeRegistrationCardiac4D {
        ReconstructionCardiac4D *reconstructor;

    public:
        SliceToVolumeRegistrationCardiac4D(ReconstructionCardiac4D *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            const ImageAttributes& attr = reconstructor->_reconstructed4D.Attributes();
            GreyImage source, target;

            ParameterList params;
            Insert(params, "Transformation model", "Rigid");
            Insert(params, "Image (dis-)similarity measure", "NMI");
            if (reconstructor->_nmi_bins > 0)
                Insert(params, "No. of bins", reconstructor->_nmi_bins);

            Insert(params, "Background value for image 1", 0); // -1
            Insert(params, "Background value for image 2", -1);

            // Insert(params, "Image (dis-)similarity measure", "NCC");
            // Insert(params, "Image interpolation mode", "Linear");
            // string type = "sigma";
            // string units = "mm";
            // double width = 0;
            // Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);

            Transformation *dofout;
            GenericRegistrationFilter registration;
            registration.Parameter(params);
            registration.Output(&dofout);

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                if (reconstructor->_slice_excluded[inputIndex])
                    continue;

                // ResamplingWithPadding<RealPixel> resampling(attr._dx, attr._dx, attr._dx, -1);
                // GenericLinearInterpolateImageFunction<RealImage> interpolator;
                // // TARGET
                // // get current slice
                // RealImage t(reconstructor->_slices[inputIndex].Attributes());
                // // resample to spatial resolution of reconstructed volume
                // resampling.Input(&reconstructor->_slices[inputIndex]);
                // resampling.Output(&t);
                // resampling.Interpolator(&interpolator);
                // resampling.Run();
                // target = t;

                // get pixel value min and max
                GreyPixel smin, smax;
                target = reconstructor->_slices[inputIndex];
                target.GetMinMax(&smin, &smax);

                // SOURCE
                if (smax > 0 && (smax - smin) > 1) {
                    // put origin to zero
                    RigidTransformation offset;
                    ResetOrigin(target, offset);
                    const Matrix& mo = offset.GetMatrix();
                    auto& transformation = reconstructor->_transformations[inputIndex];
                    transformation.PutMatrix(transformation.GetMatrix() * mo);

                    // TODO: extract the nearest cardiac phase from reconstructed 4D to use as source
                    source = reconstructor->_reconstructed4D.GetRegion(0, 0, 0, reconstructor->_slice_svr_card_index[inputIndex], attr._x, attr._y, attr._z, reconstructor->_slice_svr_card_index[inputIndex] + 1);
                    registration.Input(&target, &source);
                    registration.InitialGuess(&transformation);
                    registration.GuessParameter();
                    registration.Run();
                    unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(dofout));
                    transformation = *rigidTransf;

                    //undo the offset
                    transformation.PutMatrix(transformation.GetMatrix() * mo.Inverse());
                }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for FFD SVR
    class SliceToVolumeRegistrationFFD {
        Reconstruction *reconstructor;

    public:
        SliceToVolumeRegistrationFFD(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {


            // define registration model
            ParameterList params_init;
            
            if (reconstructor->_combined_rigid_ffd) {
                Insert(params_init, "Transformation model", "Rigid+FFD");
            } else {
                Insert(params_init, "Transformation model", "FFD");
            }

            if (reconstructor->_ncc_reg) {
                Insert(params_init, "Image (dis-)similarity measure", "NCC");
                const string type = "box";
                const string units = "vox";
                constexpr double width = 5;
                Insert(params_init, string("Local window size [") + type + string("]"), ToString(width) + units);
            }

            if (reconstructor->_cp_spacing.size() > 0) {
                const int& cp_spacing = reconstructor->_cp_spacing[reconstructor->_current_iteration];
                Insert(params_init, "Control point spacing in X", cp_spacing);
                Insert(params_init, "Control point spacing in Y", cp_spacing);
                Insert(params_init, "Control point spacing in Z", cp_spacing);
            }

            RealPixel smin, smax;
            reconstructor->_reconstructed.GetMinMax(&smin, &smax);

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {

                GenericRegistrationFilter registration;

                RealPixel tmin, tmax;
                const RealImage& target = reconstructor->_slices[inputIndex];
                target.GetMinMax(&smin, &smax);

                if (smax > 1 && (smax - smin) > 1) {

                    if (reconstructor->_ffd_global_only) {

                        MultiLevelFreeFormTransformation *mffd_init;
                        int current_stack=reconstructor->_stack_index[inputIndex];
                        Transformation *tt = Transformation::New((boost::format("ms-%1%.dof") % current_stack).str().c_str());
                        mffd_init = dynamic_cast<MultiLevelFreeFormTransformation*> (tt);
                        reconstructor->_mffd_transformations[inputIndex] = mffd_init;

                    } else {

                        ParameterList params = params_init;
                        // run registration
                        registration.Parameter(params);
                        registration.Input(&target, &reconstructor->_reconstructed);

                        if (reconstructor->_current_iteration == 0) {
                            MultiLevelFreeFormTransformation *mffd_init;
                            int current_stack=reconstructor->_stack_index[inputIndex];
                            Transformation *tt = Transformation::New((boost::format("ms-%1%.dof") % current_stack).str().c_str());
                            mffd_init = dynamic_cast<MultiLevelFreeFormTransformation*> (tt);
                            registration.InitialGuess(mffd_init);

                        } else {
                            registration.InitialGuess(reconstructor->_mffd_transformations[inputIndex]);
                        }

                        Transformation *dofout;
                        registration.Output(&dofout);
                        registration.GuessParameter();
                        registration.Run();

                        MultiLevelFreeFormTransformation *mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*> (dofout);
                        reconstructor->_mffd_transformations[inputIndex] = mffd_dofout;
                    }

                }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class RemoteSliceToVolumeRegistration {
        Reconstruction *reconstructor;
        int svr_range_start;
        int svr_range_stop;
        const string& str_mirtk_path;
        const string& str_current_exchange_file_path;
        bool rigid, is4d;
        const Array<int>& source_indexes;
        string register_cmd;

    public:
        RemoteSliceToVolumeRegistration(
            Reconstruction *reconstructor,
            int svr_range_start,
            int svr_range_stop,
            const string& str_mirtk_path,
            const string& str_current_exchange_file_path,
            bool rigid = true,
            const Array<int>& source_indexes = {}) :
            reconstructor(reconstructor),
            svr_range_start(svr_range_start),
            svr_range_stop(svr_range_stop),
            str_mirtk_path(str_mirtk_path),
            str_current_exchange_file_path(str_current_exchange_file_path),
            rigid(rigid),
            source_indexes(source_indexes),
            is4d(!source_indexes.empty()) {
            // Preallocate memory area
            register_cmd.reserve(1500);
            // Define registration parameters
            const string file_prefix = str_current_exchange_file_path + (rigid ? "/res-" : "/");
            register_cmd = str_mirtk_path + "/register ";
            register_cmd += file_prefix + "slice-%1%.nii.gz ";  // Target
            register_cmd += str_current_exchange_file_path + "/current-source" + (is4d ? "-%2%" : "") + ".nii.gz";  // Source
            register_cmd += string(" -model ") + (rigid ? "Rigid" : "FFD");
            register_cmd += " -bg1 0 -bg2 -1";
            register_cmd += " -dofin " + file_prefix + "transformation-%1%.dof";
            register_cmd += " -dofout " + file_prefix + "transformation-%1%.dof";
            if (reconstructor->_ncc_reg)
                register_cmd += string(" -sim NCC -window ") + (rigid ? "0" : "5");

            if (!rigid && reconstructor->_cp_spacing.size() > 0)
                register_cmd += " -ds " + to_string(reconstructor->_cp_spacing[reconstructor->_current_iteration]);
            register_cmd += " -threads 1";  // Remove parallelisation of the register command
            register_cmd += " > " + str_current_exchange_file_path + "/log%1%.txt"; // Log
        }

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                if (reconstructor->_zero_slices[inputIndex] > 0) {
                    auto register_cmd_formatter = boost::format(register_cmd) % inputIndex;
                    if (is4d)
                        register_cmd_formatter % source_indexes[inputIndex];
                    if (system(register_cmd_formatter.str().c_str()) == -1)
                        cerr << "The register command couldn't be executed!" << endl;
                }
            } // end of inputIndex
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(svr_range_start, svr_range_stop), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for calculation of transformation matrices
    class CoeffInit {
        Reconstruction *reconstructor;

    public:
        CoeffInit(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            const RealImage global_reconstructed(reconstructor->_grey_reconstructed.Attributes());
            RealImage global_slice, PSF, tPSF;
            //get resolution of the volume
            double vx, vy, vz;
            global_reconstructed.GetPixelSize(&vx, &vy, &vz);
            //volume is always isotropic
            const double& res = vx;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                global_slice.Initialize(reconstructor->_slices[inputIndex].Attributes());

                //prepare storage variable
                VOXELCOEFFS empty;
                SLICECOEFFS slicecoeffs(global_slice.GetX(), Array<VOXELCOEFFS>(global_slice.GetY(), empty));

                // To check whether the slice has an overlap with mask ROI
                bool slice_inside = false;

                //PSF will be calculated in slice space in higher resolution

                //get slice voxel size to define PSF
                double dx, dy, dz;
                global_slice.GetPixelSize(&dx, &dy, &dz);

                //sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM = dz through-plane)
                double sigmax, sigmay, sigmaz;
                switch (reconstructor->_recon_type) {
                case _3D:
                    sigmax = 1.2 * dx / 2.3548;
                    sigmay = 1.2 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                    break;

                case _1D:
                    sigmax = 0.5 * dx / 2.3548;
                    sigmay = 0.5 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                    break;

                case _interpolate:
                    sigmax = sigmay = sigmaz = 0.5 * dx / 2.3548;
                    break;
                }

                if (reconstructor->_no_sr) {
                    sigmax = 0.6 * dx / 2.3548;
                    sigmay = 0.6 * dy / 2.3548;
                    sigmaz = 0.6 * dz / 2.3548;
                }

                //calculate discretized PSF

                //isotropic voxel size of PSF - derived from resolution of reconstructed volume
                const double size = res / reconstructor->_quality_factor;

                //number of voxels in each direction
                //the ROI is 2*voxel dimension
                int xDim = round(2 * dx / size);
                int yDim = round(2 * dy / size);
                int zDim = round(2 * dz / size);

                ///test to make dimension always odd
                xDim = xDim / 2 * 2 + 1;
                yDim = yDim / 2 * 2 + 1;
                zDim = zDim / 2 * 2 + 1;

                //image corresponding to PSF
                ImageAttributes attr;
                attr._x = xDim;
                attr._y = yDim;
                attr._z = zDim;
                attr._dx = size;
                attr._dy = size;
                attr._dz = size;
                PSF.Initialize(attr);

                //centre of PSF
                double cx = 0.5 * (xDim - 1);
                double cy = 0.5 * (yDim - 1);
                double cz = 0.5 * (zDim - 1);
                PSF.ImageToWorld(cx, cy, cz);

                double sum = 0;
                for (int i = 0; i < xDim; i++)
                    for (int j = 0; j < yDim; j++)
                        for (int k = 0; k < zDim; k++) {
                            double x = i;
                            double y = j;
                            double z = k;
                            PSF.ImageToWorld(x, y, z);
                            x -= cx;
                            y -= cy;
                            z -= cz;
                            //continuous PSF does not need to be normalized as discrete will be
                            PSF(i, j, k) = exp(-x * x / (2 * sigmax * sigmax) - y * y / (2 * sigmay * sigmay) - z * z / (2 * sigmaz * sigmaz));
                            sum += PSF(i, j, k);
                        }
                PSF /= sum;

                if (reconstructor->_debug && inputIndex == 0)
                    PSF.Write("PSF.nii.gz");

                //prepare storage for PSF transformed and resampled to the space of reconstructed volume
                //maximum dim of rotated kernel - the next higher odd integer plus two to accound for rounding error of tx,ty,tz.
                //Note conversion from PSF image coordinates to tPSF image coordinates *size/res
                int dim = floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2) * 2 + 1 + 2;
                //prepare image attributes. Voxel dimension will be taken from the reconstructed volume
                attr._x = dim;
                attr._y = dim;
                attr._z = dim;
                attr._dx = res;
                attr._dy = res;
                attr._dz = res;
                //create matrix from transformed PSF
                tPSF.Initialize(attr);
                //calculate centre of tPSF in image coordinates
                const int centre = (dim - 1) / 2;

                //for each voxel in current slice calculate matrix coefficients
                bool excluded_slice = false;
                for (size_t ff = 0; ff < reconstructor->_force_excluded.size(); ff++) {
                    if (inputIndex == reconstructor->_force_excluded[ff]) {
                        excluded_slice = true;
                        break;
                    }
                }

                for (int i = 0; i < global_slice.GetX(); i++)
                    for (int j = 0; j < global_slice.GetY(); j++) {
                        double test_val = reconstructor->_slices[inputIndex](i, j, 0);
                        if (reconstructor->_no_masking_background)
                            test_val = reconstructor->_not_masked_slices[inputIndex](i, j, 0);

                        if (test_val > -0.01 && reconstructor->_structural_slice_weight[inputIndex] > 0 && !excluded_slice) {
                            //calculate centrepoint of slice voxel in volume space (tx,ty,tz)
                            double x = i;
                            double y = j;
                            double z = 0;
                            global_slice.ImageToWorld(x, y, z);

                            if (!reconstructor->_ffd)
                                reconstructor->_transformations[inputIndex].Transform(x, y, z);
                            else
                                reconstructor->_mffd_transformations[inputIndex]->Transform(-1, 1, x, y, z);

                            global_reconstructed.WorldToImage(x, y, z);
                            int tx = round(x);
                            int ty = round(y);
                            int tz = round(z);

                            // Clear the transformed PSF
                            memset(tPSF.Data(), 0, sizeof(RealPixel) * tPSF.NumberOfVoxels());

                            //for each POINT3D of the PSF
                            for (int ii = 0; ii < xDim; ii++)
                                for (int jj = 0; jj < yDim; jj++)
                                    for (int kk = 0; kk < zDim; kk++) {
                                        //Calculate the position of the POINT3D of PSF centred over current slice voxel
                                        //This is a bit complicated because slices can be oriented in any direction

                                        //PSF image coordinates
                                        x = ii;
                                        y = jj;
                                        z = kk;
                                        //change to PSF world coordinates - now real sizes in mm
                                        PSF.ImageToWorld(x, y, z);
                                        //centre around the centrepoint of the PSF
                                        x -= cx;
                                        y -= cy;
                                        z -= cz;

                                        //Need to convert (x,y,z) to slice image coordinates because slices can have
                                        //transformations included in them (they are nifti) and those are not reflected in
                                        //PSF. In slice image coordinates we are sure that z is through-plane

                                        //adjust according to voxel size
                                        x /= dx;
                                        y /= dy;
                                        z /= dz;
                                        //centre over current voxel
                                        x += i;
                                        y += j;

                                        //convert from slice image coordinates to world coordinates
                                        global_slice.ImageToWorld(x, y, z);

                                        //Transform to space of reconstructed volume
                                        if (!reconstructor->_ffd)
                                            reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                        else
                                            reconstructor->_mffd_transformations[inputIndex]->Transform(-1, 1, x, y, z);

                                        //Change to image coordinates
                                        global_reconstructed.WorldToImage(x, y, z);

                                        //determine coefficients of volume voxels for position x,y,z
                                        //using linear interpolation

                                        //Find the 8 closest volume voxels

                                        //lowest corner of the cube
                                        int nx = (int)floor(x);
                                        int ny = (int)floor(y);
                                        int nz = (int)floor(z);

                                        //not all neighbours might be in ROI, thus we need to normalize
                                        //(l,m,n) are image coordinates of 8 neighbours in volume space
                                        //for each we check whether it is in volume
                                        sum = 0;
                                        //to find wether the current slice voxel has overlap with ROI
                                        bool inside = false;
                                        for (int l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < global_reconstructed.GetX()))
                                                for (int m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < global_reconstructed.GetY()))
                                                        for (int n = nz; n <= nz + 1; n++)
                                                            if ((n >= 0) && (n < global_reconstructed.GetZ())) {
                                                                const double weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                                                sum += weight;

                                                                if (reconstructor->_mask(l, m, n) == 1 || reconstructor->_no_masking_background) {
                                                                    inside = true;
                                                                    slice_inside = true;
                                                                }
                                                            }

                                        //if there were no voxels do nothing
                                        if (sum <= 0 || !inside)
                                            continue;

                                        //now calculate the transformed PSF
                                        for (int l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < global_reconstructed.GetX()))
                                                for (int m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < global_reconstructed.GetY()))
                                                        for (int n = nz; n <= nz + 1; n++)
                                                            if (n >= 0 && n < global_reconstructed.GetZ()) {
                                                                const double weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));

                                                                //image coordinates in tPSF
                                                                //(centre,centre,centre) in tPSF is aligned with (tx,ty,tz)
                                                                int aa = l - tx + centre;
                                                                int bb = m - ty + centre;
                                                                int cc = n - tz + centre;

                                                                //resulting value
                                                                const double value = PSF(ii, jj, kk) * weight / sum;

                                                                //Check that we are in tPSF
                                                                if (!(aa < 0 || aa >= dim || bb < 0 || bb >= dim || cc < 0 || cc >= dim))
                                                                    //update transformed PSF
                                                                    tPSF(aa, bb, cc) += value;
                                                            }

                                    } //end of the loop for PSF points

                            //store tPSF values
                            for (int ii = 0; ii < dim; ii++)
                                for (int jj = 0; jj < dim; jj++)
                                    for (int kk = 0; kk < dim; kk++)
                                        if (tPSF(ii, jj, kk) > 0) {
                                            POINT3D p;
                                            p.x = ii + tx - centre;
                                            p.y = jj + ty - centre;
                                            p.z = kk + tz - centre;
                                            p.value = tPSF(ii, jj, kk);
                                            slicecoeffs[i][j].push_back(p);
                                        }
                        } //end of loop for slice voxels
                        }

                reconstructor->_volcoeffs[inputIndex] = slicecoeffs; //move(slicecoeffs);
                reconstructor->_slice_inside[inputIndex] = slice_inside;

            }  //end of loop through the slices
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Another version of CoeffInit
    class CoeffInitSF {
        Reconstruction *reconstructor;
        size_t begin;
        size_t end;

    public:
        CoeffInitSF(
            Reconstruction *reconstructor,
            size_t begin,
            size_t end) :
            reconstructor(reconstructor),
            begin(begin),
            end(end) {}

        void operator()(const blocked_range<size_t>& r) const {
            RealImage PSF, tPSF;
            //get resolution of the volume
            double vx, vy, vz;
            reconstructor->_reconstructed.GetPixelSize(&vx, &vy, &vz);
            //volume is always isotropic
            const double& res = vx;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                const RealImage& slice = reconstructor->_withMB ? reconstructor->_slicesRwithMB[inputIndex] : reconstructor->_slices[inputIndex];

                //prepare structures for storage
                SLICECOEFFS slicecoeffs(slice.GetX(), Array<VOXELCOEFFS>(slice.GetY()));

                //to check whether the slice has an overlap with mask ROI
                bool slice_inside = false;

                //PSF will be calculated in slice space in higher resolution

                //get slice voxel size to define PSF
                double dx, dy, dz;
                slice.GetPixelSize(&dx, &dy, &dz);

                //sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM = dz through-plane)
                double sigmax, sigmay, sigmaz;
                switch (reconstructor->_recon_type) {
                case _3D:
                    sigmax = 1.2 * dx / 2.3548;
                    sigmay = 1.2 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                    break;

                case _1D:
                    sigmax = 0.5 * dx / 2.3548;
                    sigmay = 0.5 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                    break;

                case _interpolate:
                    sigmax = sigmay = sigmaz = 0.5 * dx / 2.3548;
                    break;
                }

                //calculate discretized PSF

                //isotropic voxel size of PSF - derived from resolution of reconstructed volume
                double size = res / reconstructor->_quality_factor;

                //number of voxels in each direction
                //the ROI is 2*voxel dimension
                int xDim = round(2 * dx / size);
                int yDim = round(2 * dy / size);
                int zDim = round(2 * dz / size);

                ///test to make dimension always odd
                xDim = xDim / 2 * 2 + 1;
                yDim = yDim / 2 * 2 + 1;
                zDim = zDim / 2 * 2 + 1;

                //image corresponding to PSF
                ImageAttributes attr;
                attr._x = xDim;
                attr._y = yDim;
                attr._z = zDim;
                attr._dx = size;
                attr._dy = size;
                attr._dz = size;
                PSF.Initialize(attr);

                //centre of PSF
                double cx = 0.5 * (xDim - 1);
                double cy = 0.5 * (yDim - 1);
                double cz = 0.5 * (zDim - 1);
                PSF.ImageToWorld(cx, cy, cz);

                double sum = 0;
                for (int i = 0; i < xDim; i++)
                    for (int j = 0; j < yDim; j++)
                        for (int k = 0; k < zDim; k++) {
                            double x = i;
                            double y = j;
                            double z = k;
                            PSF.ImageToWorld(x, y, z);
                            x -= cx;
                            y -= cy;
                            z -= cz;
                            //continuous PSF does not need to be normalized as discrete will be
                            PSF(i, j, k) = exp(-x * x / (2 * sigmax * sigmax) - y * y / (2 * sigmay * sigmay) - z * z / (2 * sigmaz * sigmaz));
                            sum += PSF(i, j, k);
                        }
                PSF /= sum;

                if (reconstructor->_debug && inputIndex == 0)
                    PSF.Write("PSF.nii.gz");

                //prepare storage for PSF transformed and resampled to the space of reconstructed volume
                //maximum dim of rotated kernel - the next higher odd integer plus two to accound for rounding error of tx,ty,tz.
                //Note conversion from PSF image coordinates to tPSF image coordinates *size/res
                int dim = floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2) * 2 + 1 + 2;
                //prepare image attributes. Voxel dimension will be taken from the reconstructed volume
                attr._x = dim;
                attr._y = dim;
                attr._z = dim;
                attr._dx = res;
                attr._dy = res;
                attr._dz = res;
                //create matrix from transformed PSF
                tPSF.Initialize(attr);
                //calculate centre of tPSF in image coordinates
                int centre = (dim - 1) / 2;

                //for each voxel in current slice calculate matrix coefficients
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            //calculate centrepoint of slice voxel in volume space (tx,ty,tz)
                            double x = i;
                            double y = j;
                            double z = 0;
                            slice.ImageToWorld(x, y, z);

                            if (reconstructor->_withMB)
                                reconstructor->_transformationsRwithMB[inputIndex].Transform(x, y, z);
                            else
                                reconstructor->_transformations[inputIndex].Transform(x, y, z);

                            reconstructor->_reconstructed.WorldToImage(x, y, z);
                            int tx = round(x);
                            int ty = round(y);
                            int tz = round(z);

                            //Clear the transformed PSF
                            memset(tPSF.Data(), 0, sizeof(RealPixel) * tPSF.NumberOfVoxels());

                            //for each POINT3D of the PSF
                            for (int ii = 0; ii < xDim; ii++)
                                for (int jj = 0; jj < yDim; jj++)
                                    for (int kk = 0; kk < zDim; kk++) {
                                        //Calculate the position of the POINT3D of
                                        //PSF centred over current slice voxel
                                        //This is a bit complicated because slices
                                        //can be oriented in any direction

                                        //PSF image coordinates
                                        x = ii;
                                        y = jj;
                                        z = kk;
                                        //change to PSF world coordinates - now real sizes in mm
                                        PSF.ImageToWorld(x, y, z);
                                        //centre around the centrepoint of the PSF
                                        x -= cx;
                                        y -= cy;
                                        z -= cz;

                                        //Need to convert (x,y,z) to slice image
                                        //coordinates because slices can have
                                        //transformations included in them (they are
                                        //nifti) and those are not reflected in
                                        //PSF. In slice image coordinates we are
                                        //sure that z is through-plane

                                        //adjust according to voxel size
                                        x /= dx;
                                        y /= dy;
                                        z /= dz;
                                        //center over current voxel
                                        x += i;
                                        y += j;

                                        //convert from slice image coordinates to world coordinates
                                        slice.ImageToWorld(x, y, z);

                                        //x+=(vx-cx); y+=(vy-cy); z+=(vz-cz);
                                        //Transform to space of reconstructed volume
                                        if (reconstructor->_withMB)
                                            reconstructor->_transformationsRwithMB[inputIndex].Transform(x, y, z);
                                        else
                                            reconstructor->_transformations[inputIndex].Transform(x, y, z);

                                        //Change to image coordinates
                                        reconstructor->_reconstructed.WorldToImage(x, y, z);

                                        //determine coefficients of volume voxels for position x,y,z
                                        //using linear interpolation

                                        //Find the 8 closest volume voxels

                                        //lowest corner of the cube
                                        int nx = (int)floor(x);
                                        int ny = (int)floor(y);
                                        int nz = (int)floor(z);

                                        //not all neighbours might be in ROI, thus we need to normalize
                                        //(l,m,n) are image coordinates of 8 neighbours in volume space
                                        //for each we check whether it is in volume
                                        sum = 0;
                                        //to find wether the current slice voxel has overlap with ROI
                                        bool inside = false;
                                        for (int l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < reconstructor->_reconstructed.GetX()))
                                                for (int m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < reconstructor->_reconstructed.GetY()))
                                                        for (int n = nz; n <= nz + 1; n++)
                                                            if ((n >= 0) && (n < reconstructor->_reconstructed.GetZ())) {
                                                                const double weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                                                sum += weight;
                                                                if (reconstructor->_mask(l, m, n) == 1) {
                                                                    inside = true;
                                                                    slice_inside = true;
                                                                }
                                                            }

                                        //if there were no voxels do nothing
                                        if (sum <= 0 || !inside)
                                            continue;

                                        //now calculate the transformed PSF
                                        for (int l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < reconstructor->_reconstructed.GetX()))
                                                for (int m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < reconstructor->_reconstructed.GetY()))
                                                        for (int n = nz; n <= nz + 1; n++)
                                                            if ((n >= 0) && (n < reconstructor->_reconstructed.GetZ())) {
                                                                const double weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));

                                                                //image coordinates in tPSF
                                                                //(centre,centre,centre) in tPSF is aligned with (tx,ty,tz)
                                                                int aa = l - tx + centre;
                                                                int bb = m - ty + centre;
                                                                int cc = n - tz + centre;

                                                                //resulting value
                                                                const double value = PSF(ii, jj, kk) * weight / sum;

                                                                //Check that we are in tPSF
                                                                if (aa < 0 || aa >= dim || bb < 0 || bb >= dim || cc < 0 || cc >= dim) {
                                                                    stringstream err;
                                                                    err << "Error while trying to populate tPSF. " << aa << " " << bb << " " << cc << endl;
                                                                    err << l << " " << m << " " << n << endl;
                                                                    err << tx << " " << ty << " " << tz << endl;
                                                                    err << centre << endl;
                                                                    tPSF.Write("tPSF.nii.gz");
                                                                    throw runtime_error(err.str());
                                                                } else
                                                                    //update transformed PSF
                                                                    tPSF(aa, bb, cc) += value;
                                                            }

                                    } //end of the loop for PSF points

                            //store tPSF values
                            for (int ii = 0; ii < dim; ii++)
                                for (int jj = 0; jj < dim; jj++)
                                    for (int kk = 0; kk < dim; kk++)
                                        if (tPSF(ii, jj, kk) > 0) {
                                            POINT3D p;
                                            p.x = ii + tx - centre;
                                            p.y = jj + ty - centre;
                                            p.z = kk + tz - centre;
                                            p.value = tPSF(ii, jj, kk);
                                            slicecoeffs[i][j].push_back(p);
                                        }
                        } //end of loop for slice voxels

                // move assignment operation for slicecoeffs should have been more performant and memory efficient
                // but it turned out that it consumes more memory and it's less performant (GCC 9.3.0)
                reconstructor->_volcoeffsSF[inputIndex % reconstructor->_slicePerDyn] = slicecoeffs;
                reconstructor->_slice_insideSF[inputIndex % reconstructor->_slicePerDyn] = slice_inside;
                //cerr<<" Done "<<inputIndex % (reconstructor->_slicePerDyn)<<endl;
            }  //end of loop through the slices
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(begin, end), *this);
        }
    };

    //-------------------------------------------------------------------

    class CoeffInitCardiac4D {
        ReconstructionCardiac4D *reconstructor;

    public:
        CoeffInitCardiac4D(ReconstructionCardiac4D *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            RealImage PSF, tPSF;
            //get resolution of the volume
            double vx, vy, vz;
            reconstructor->_reconstructed4D.GetPixelSize(&vx, &vy, &vz);
            //volume is always isotropic
            const double& res = vx;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                if (reconstructor->_slice_excluded[inputIndex])
                    continue;

                //read the slice
                RealImage& slice = reconstructor->_slices[inputIndex];

                //prepare structures for storage
                SLICECOEFFS slicecoeffs(slice.GetX(), Array<VOXELCOEFFS>(slice.GetY()));

                //to check whether the slice has an overlap with mask ROI
                bool slice_inside = false;

                //start of a loop for a slice inputIndex
                reconstructor->GetVerboseLog() << inputIndex << " ";

                //PSF will be calculated in slice space in higher resolution

                //get slice voxel size to define PSF
                double dx, dy, dz;
                slice.GetPixelSize(&dx, &dy, &dz);

                //sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM = dz through-plane)

                double sigmax, sigmay, sigmaz;
                switch (reconstructor->_recon_type) {
                case _3D:
                    sigmax = 1.2 * dx / 2.3548;
                    sigmay = 1.2 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                    break;

                case _1D:
                    sigmax = 0.5 * dx / 2.3548;
                    sigmay = 0.5 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                    break;

                case _interpolate:
                    sigmax = sigmay = sigmaz = 0.5 * dx / 2.3548;
                    break;
                }

                if (reconstructor->_no_sr) {
                    sigmax = 0.5 * dx / 2.3548; // 1.2
                    sigmay = 0.5 * dy / 2.3548; // 1.2
                    sigmaz = 0.5 * dz / 2.3548;   // 1
                }

                //calculate discretized PSF

                //isotropic voxel size of PSF - derived from resolution of reconstructed volume
                const double size = res / reconstructor->_quality_factor;

                //number of voxels in each direction
                //the ROI is 2*voxel dimension

                int xDim = round(2 * dx / size);
                int yDim = round(2 * dy / size);
                int zDim = round(2 * dz / size);
                ///test to make dimension always odd
                xDim = xDim / 2 * 2 + 1;
                yDim = yDim / 2 * 2 + 1;
                zDim = zDim / 2 * 2 + 1;
                ///end test

                //image corresponding to PSF
                ImageAttributes attr;
                attr._x = xDim;
                attr._y = yDim;
                attr._z = zDim;
                attr._dx = size;
                attr._dy = size;
                attr._dz = size;
                PSF.Initialize(attr);

                //centre of PSF
                double cx, cy, cz;
                cx = 0.5 * (xDim - 1);
                cy = 0.5 * (yDim - 1);
                cz = 0.5 * (zDim - 1);
                PSF.ImageToWorld(cx, cy, cz);

                double sum = 0;
                for (int i = 0; i < xDim; i++)
                    for (int j = 0; j < yDim; j++)
                        for (int k = 0; k < zDim; k++) {
                            double x = i;
                            double y = j;
                            double z = k;
                            PSF.ImageToWorld(x, y, z);
                            x -= cx;
                            y -= cy;
                            z -= cz;
                            //continuous PSF does not need to be normalized as discrete will be
                            PSF(i, j, k) = exp(-x * x / (2 * sigmax * sigmax) - y * y / (2 * sigmay * sigmay) - z * z / (2 * sigmaz * sigmaz));
                            sum += PSF(i, j, k);
                        }
                PSF /= sum;

                if (reconstructor->_debug && inputIndex == 0)
                    PSF.Write("PSF.nii.gz");

                //prepare storage for PSF transformed and resampled to the space of reconstructed volume
                //maximum dim of rotated kernel - the next higher odd integer plus two to accound for rounding error of tx,ty,tz.
                //Note conversion from PSF image coordinates to tPSF image coordinates *size/res
                int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2)) * 2 + 1 + 2;
                //prepare image attributes. Voxel dimension will be taken from the reconstructed volume
                attr._x = dim;
                attr._y = dim;
                attr._z = dim;
                attr._dx = res;
                attr._dy = res;
                attr._dz = res;
                //create matrix from transformed PSF
                tPSF.Initialize(attr);
                //calculate centre of tPSF in image coordinates
                int centre = (dim - 1) / 2;

                //for each voxel in current slice calculate matrix coefficients
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //calculate centrepoint of slice voxel in volume space (tx,ty,tz)
                            double x = i;
                            double y = j;
                            double z = 0;
                            slice.ImageToWorld(x, y, z);
                            reconstructor->_transformations[inputIndex].Transform(x, y, z);
                            reconstructor->_reconstructed4D.WorldToImage(x, y, z);
                            int tx = round(x);
                            int ty = round(y);
                            int tz = round(z);

                            //Clear the transformed PSF
                            memset(tPSF.Data(), 0, sizeof(RealPixel) * tPSF.NumberOfVoxels());

                            //for each POINT3D of the PSF
                            for (int ii = 0; ii < xDim; ii++)
                                for (int jj = 0; jj < yDim; jj++)
                                    for (int kk = 0; kk < zDim; kk++) {
                                        //Calculate the position of the POINT3D of
                                        //PSF centred over current slice voxel
                                        //This is a bit complicated because slices
                                        //can be oriented in any direction

                                        //PSF image coordinates
                                        x = ii;
                                        y = jj;
                                        z = kk;
                                        //change to PSF world coordinates - now real sizes in mm
                                        PSF.ImageToWorld(x, y, z);
                                        //centre around the centrepoint of the PSF
                                        x -= cx;
                                        y -= cy;
                                        z -= cz;

                                        //Need to convert (x,y,z) to slice image
                                        //coordinates because slices can have
                                        //transformations included in them (they are
                                        //nifti)  and those are not reflected in
                                        //PSF. In slice image coordinates we are
                                        //sure that z is through-plane

                                        //adjust according to voxel size
                                        x /= dx;
                                        y /= dy;
                                        z /= dz;
                                        //center over current voxel
                                        x += i;
                                        y += j;

                                        //convert from slice image coordinates to world coordinates
                                        slice.ImageToWorld(x, y, z);

                                        //x+=(vx-cx); y+=(vy-cy); z+=(vz-cz);
                                        //Transform to space of reconstructed volume
                                        reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                        //Change to image coordinates
                                        reconstructor->_reconstructed4D.WorldToImage(x, y, z);

                                        //determine coefficients of volume voxels for position x,y,z
                                        //using linear interpolation

                                        //Find the 8 closest volume voxels

                                        //lowest corner of the cube
                                        int nx = (int)floor(x);
                                        int ny = (int)floor(y);
                                        int nz = (int)floor(z);

                                        //not all neighbours might be in ROI, thus we need to normalize
                                        //(l,m,n) are image coordinates of 8 neighbours in volume space
                                        //for each we check whether it is in volume
                                        sum = 0;
                                        //to find wether the current slice voxel has overlap with ROI
                                        bool inside = false;
                                        for (int l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < reconstructor->_reconstructed4D.GetX()))
                                                for (int m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < reconstructor->_reconstructed4D.GetY()))
                                                        for (int n = nz; n <= nz + 1; n++)
                                                            if ((n >= 0) && (n < reconstructor->_reconstructed4D.GetZ())) {
                                                                const double weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                                                sum += weight;
                                                                if (reconstructor->_mask(l, m, n) == 1) {
                                                                    inside = true;
                                                                    slice_inside = true;
                                                                }
                                                            }

                                        //if there were no voxels do nothing
                                        if (sum <= 0 || !inside)
                                            continue;

                                        //now calculate the transformed PSF
                                        for (int l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < reconstructor->_reconstructed4D.GetX()))
                                                for (int m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < reconstructor->_reconstructed4D.GetY()))
                                                        for (int n = nz; n <= nz + 1; n++)
                                                            if (n >= 0 && n < reconstructor->_reconstructed4D.GetZ()) {
                                                                const double weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));

                                                                //image coordinates in tPSF
                                                                //(centre,centre,centre) in tPSF is aligned with (tx,ty,tz)
                                                                int aa = l - tx + centre;
                                                                int bb = m - ty + centre;
                                                                int cc = n - tz + centre;

                                                                //resulting value
                                                                const double value = PSF(ii, jj, kk) * weight / sum;

                                                                //Check that we are in tPSF
                                                                if (aa < 0 || aa >= dim || bb < 0 || bb >= dim || cc < 0 || cc >= dim) {
                                                                    stringstream err;
                                                                    err << "Error while trying to populate tPSF. " << aa << " " << bb << " " << cc << endl;
                                                                    err << l << " " << m << " " << n << endl;
                                                                    err << tx << " " << ty << " " << tz << endl;
                                                                    err << centre << endl;
                                                                    tPSF.Write("tPSF.nii.gz");
                                                                    throw runtime_error(err.str());
                                                                } else
                                                                    //update transformed PSF
                                                                    tPSF(aa, bb, cc) += value;
                                                            }

                                    } //end of the loop for PSF points

                            //store tPSF values
                            for (int ii = 0; ii < dim; ii++)
                                for (int jj = 0; jj < dim; jj++)
                                    for (int kk = 0; kk < dim; kk++)
                                        if (tPSF(ii, jj, kk) > 0) {
                                            POINT3D p;
                                            p.x = ii + tx - centre;
                                            p.y = jj + ty - centre;
                                            p.z = kk + tz - centre;
                                            p.value = tPSF(ii, jj, kk);
                                            slicecoeffs[i][j].push_back(p);
                                        }

                        } //end of loop for slice voxels

                // move assignment operation for slicecoeffs should have been more performant and memory efficient
                // but it turned out that it consumes more memory and it's less performant (GCC 9.3.0)
                reconstructor->_volcoeffs[inputIndex] = slicecoeffs;
                reconstructor->_slice_inside[inputIndex] = slice_inside;

            }  //end of loop through the slices
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class SimulateStacksCardiac4D {
        ReconstructionCardiac4D *reconstructor;

    public:
        SimulateStacksCardiac4D(ReconstructionCardiac4D *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            RealImage PSF, tPSF;
            //get resolution of the volume
            double vx, vy, vz;
            reconstructor->_reconstructed4D.GetPixelSize(&vx, &vy, &vz);
            //volume is always isotropic
            const double& res = vx;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                if (reconstructor->_slice_excluded[inputIndex])
                    continue;

                reconstructor->GetVerboseLog() << inputIndex << " ";

                //read the slice
                RealImage& slice = reconstructor->_slices[inputIndex];

                //prepare structures for storage
                SLICECOEFFS slicecoeffs(slice.GetX(), Array<VOXELCOEFFS>(slice.GetY()));

                //PSF will be calculated in slice space in higher resolution

                //get slice voxel size to define PSF
                double dx, dy, dz;
                slice.GetPixelSize(&dx, &dy, &dz);

                //sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM = dz through-plane)

                double sigmax, sigmay, sigmaz;
                if (reconstructor->_recon_type == _3D) {
                    sigmax = 1.2 * dx / 2.3548;
                    sigmay = 1.2 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                }

                if (reconstructor->_recon_type == _1D) {
                    sigmax = 0.5 * dx / 2.3548;
                    sigmay = 0.5 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                }

                if (reconstructor->_recon_type == _interpolate) {
                    sigmax = 0.5 * dx / 2.3548;
                    sigmay = 0.5 * dx / 2.3548;
                    sigmaz = 0.5 * dx / 2.3548;
                }

                //calculate discretized PSF

                //isotropic voxel size of PSF - derived from resolution of reconstructed volume
                double size = res / reconstructor->_quality_factor;

                //number of voxels in each direction
                //the ROI is 2*voxel dimension

                int xDim = round(2 * dx / size);
                int yDim = round(2 * dy / size);
                int zDim = round(2 * dz / size);
                ///test to make dimension always odd
                xDim = xDim / 2 * 2 + 1;
                yDim = yDim / 2 * 2 + 1;
                zDim = zDim / 2 * 2 + 1;
                ///end test

                //image corresponding to PSF
                ImageAttributes attr;
                attr._x = xDim;
                attr._y = yDim;
                attr._z = zDim;
                attr._dx = size;
                attr._dy = size;
                attr._dz = size;
                PSF.Initialize(attr);

                //centre of PSF
                double cx, cy, cz;
                cx = 0.5 * (xDim - 1);
                cy = 0.5 * (yDim - 1);
                cz = 0.5 * (zDim - 1);
                PSF.ImageToWorld(cx, cy, cz);

                double x, y, z;
                double sum = 0;
                int i, j, k;
                for (i = 0; i < xDim; i++)
                    for (j = 0; j < yDim; j++)
                        for (k = 0; k < zDim; k++) {
                            x = i;
                            y = j;
                            z = k;
                            PSF.ImageToWorld(x, y, z);
                            x -= cx;
                            y -= cy;
                            z -= cz;
                            //continuous PSF does not need to be normalized as discrete will be
                            PSF(i, j, k) = exp(-x * x / (2 * sigmax * sigmax) - y * y / (2 * sigmay * sigmay) - z * z / (2 * sigmaz * sigmaz));
                            sum += PSF(i, j, k);
                        }
                PSF /= sum;

                if (reconstructor->_debug)
                    if (inputIndex == 0)
                        PSF.Write("PSF.nii.gz");

                //prepare storage for PSF transformed and resampled to the space of reconstructed volume
                //maximum dim of rotated kernel - the next higher odd integer plus two to accound for rounding error of tx,ty,tz.
                //Note conversion from PSF image coordinates to tPSF image coordinates *size/res
                int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2)) * 2 + 1 + 2;
                //prepare image attributes. Voxel dimension will be taken from the reconstructed volume
                attr._x = dim;
                attr._y = dim;
                attr._z = dim;
                attr._dx = res;
                attr._dy = res;
                attr._dz = res;
                //create matrix from transformed PSF
                tPSF.Initialize(attr);
                //calculate centre of tPSF in image coordinates
                int centre = (dim - 1) / 2;

                //for each voxel in current slice calculate matrix coefficients
                int ii, jj, kk;
                int tx, ty, tz;
                int nx, ny, nz;
                int l, m, n;
                for (i = 0; i < slice.GetX(); i++)
                    for (j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //calculate centrepoint of slice voxel in volume space (tx,ty,tz)
                            x = i;
                            y = j;
                            z = 0;
                            slice.ImageToWorld(x, y, z);
                            reconstructor->_transformations[inputIndex].Transform(x, y, z);
                            reconstructor->_reconstructed4D.WorldToImage(x, y, z);
                            tx = round(x);
                            ty = round(y);
                            tz = round(z);

                            //Clear the transformed PSF
                            memset(tPSF.Data(), 0, sizeof(RealPixel) * tPSF.NumberOfVoxels());

                            //for each POINT3D of the PSF
                            for (ii = 0; ii < xDim; ii++)
                                for (jj = 0; jj < yDim; jj++)
                                    for (kk = 0; kk < zDim; kk++) {
                                        //Calculate the position of the POINT3D of
                                        //PSF centred over current slice voxel
                                        //This is a bit complicated because slices
                                        //can be oriented in any direction

                                        //PSF image coordinates
                                        x = ii;
                                        y = jj;
                                        z = kk;
                                        //change to PSF world coordinates - now real sizes in mm
                                        PSF.ImageToWorld(x, y, z);
                                        //centre around the centrepoint of the PSF
                                        x -= cx;
                                        y -= cy;
                                        z -= cz;

                                        //Need to convert (x,y,z) to slice image
                                        //coordinates because slices can have
                                        //transformations included in them (they are
                                        //nifti)  and those are not reflected in
                                        //PSF. In slice image coordinates we are
                                        //sure that z is through-plane

                                        //adjust according to voxel size
                                        x /= dx;
                                        y /= dy;
                                        z /= dz;
                                        //center over current voxel
                                        x += i;
                                        y += j;

                                        //convert from slice image coordinates to world coordinates
                                        slice.ImageToWorld(x, y, z);

                                        //x+=(vx-cx); y+=(vy-cy); z+=(vz-cz);
                                        //Transform to space of reconstructed volume
                                        reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                        //Change to image coordinates
                                        reconstructor->_reconstructed4D.WorldToImage(x, y, z);

                                        //determine coefficients of volume voxels for position x,y,z
                                        //using linear interpolation

                                        //Find the 8 closest volume voxels

                                        //lowest corner of the cube
                                        nx = (int)floor(x);
                                        ny = (int)floor(y);
                                        nz = (int)floor(z);

                                        //not all neighbours might be in ROI, thus we need to normalize
                                        //(l,m,n) are image coordinates of 8 neighbours in volume space
                                        //for each we check whether it is in volume
                                        sum = 0;
                                        //to find wether the current slice voxel has overlap with ROI
                                        bool inside = false;
                                        for (l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < reconstructor->_reconstructed4D.GetX()))
                                                for (m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < reconstructor->_reconstructed4D.GetY()))
                                                        for (n = nz; n <= nz + 1; n++)
                                                            if ((n >= 0) && (n < reconstructor->_reconstructed4D.GetZ())) {
                                                                const double weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                                                sum += weight;
                                                                if (reconstructor->_mask(l, m, n) == 1)
                                                                    inside = true;
                                                            }
                                        //if there were no voxels do nothing
                                        if (sum <= 0 || !inside)
                                            continue;
                                        //now calculate the transformed PSF
                                        for (l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < reconstructor->_reconstructed4D.GetX()))
                                                for (m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < reconstructor->_reconstructed4D.GetY()))
                                                        for (n = nz; n <= nz + 1; n++)
                                                            if ((n >= 0) && (n < reconstructor->_reconstructed4D.GetZ())) {
                                                                const double weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));

                                                                //image coordinates in tPSF
                                                                //(centre,centre,centre) in tPSF is aligned with (tx,ty,tz)
                                                                int aa = l - tx + centre;
                                                                int bb = m - ty + centre;
                                                                int cc = n - tz + centre;

                                                                //resulting value
                                                                double value = PSF(ii, jj, kk) * weight / sum;

                                                                //Check that we are in tPSF
                                                                if ((aa < 0) || (aa >= dim) || (bb < 0) || (bb >= dim) || (cc < 0) || (cc >= dim)) {
                                                                    stringstream err;
                                                                    err << "Error while trying to populate tPSF. " << aa << " " << bb << " " << cc << endl;
                                                                    err << l << " " << m << " " << n << endl;
                                                                    err << tx << " " << ty << " " << tz << endl;
                                                                    err << centre << endl;
                                                                    tPSF.Write("tPSF.nii.gz");
                                                                    throw runtime_error(err.str());
                                                                } else
                                                                    //update transformed PSF
                                                                    tPSF(aa, bb, cc) += value;
                                                            }

                                    } //end of the loop for PSF points

                            //store tPSF values
                            for (ii = 0; ii < dim; ii++)
                                for (jj = 0; jj < dim; jj++)
                                    for (kk = 0; kk < dim; kk++)
                                        if (tPSF(ii, jj, kk) > 0) {
                                            POINT3D p;
                                            p.x = ii + tx - centre;
                                            p.y = jj + ty - centre;
                                            p.z = kk + tz - centre;
                                            p.value = tPSF(ii, jj, kk);
                                            slicecoeffs[i][j].push_back(p);
                                        }

                        } //end of loop for slice voxels

                //Calculate simulated slice
                reconstructor->_simulated_slices[inputIndex].Initialize(reconstructor->_slices[inputIndex].Attributes());
                reconstructor->_simulated_weights[inputIndex].Initialize(reconstructor->_slices[inputIndex].Attributes());

                for (int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++)
                    for (int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++)
                        if (reconstructor->_slices[inputIndex](i, j, 0) != -1) {
                            double weight = 0;
                            const size_t n = slicecoeffs[i][j].size();
                            for (size_t k = 0; k < n; k++) {
                                const POINT3D& p = slicecoeffs[i][j][k];
                                for (int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++) {
                                    reconstructor->_simulated_slices[inputIndex](i, j, 0) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_reconstructed4D(p.x, p.y, p.z, outputIndex);
                                    weight += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                }
                            }
                            if (weight > 0) {
                                reconstructor->_simulated_slices[inputIndex](i, j, 0) /= weight;
                                reconstructor->_simulated_weights[inputIndex](i, j, 0) = weight;
                            }
                        }
            }  //end of loop through the slices
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for EStep (RS)
    class EStep {
        Reconstruction *reconstructor;
        Array<double>& slice_potential;

    public:
        EStep(Reconstruction *reconstructor,
            Array<double>& slice_potential) :
            reconstructor(reconstructor),
            slice_potential(slice_potential) {}

        void operator()(const blocked_range<size_t>& r) const {
            RealImage slice;

            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice

                if (reconstructor->_no_masking_background)
                    slice = reconstructor->_not_masked_slices[inputIndex];
                else
                    slice = reconstructor->_slices[inputIndex];

                //read current weight image
                RealImage& weight = reconstructor->_weights[inputIndex];
                memset(weight.Data(), 0, sizeof(RealPixel) * weight.NumberOfVoxels());

                double num = 0;
                //Calculate error, voxel weights, and slice potential
                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-reconstructor->_bias[inputIndex](i, j, 0)) * reconstructor->_scale[inputIndex];

                            //number of volumetric voxels to which
                            // current slice voxel contributes
                            const size_t n = reconstructor->_volcoeffs[inputIndex][i][j].size();

                            // if n == 0, slice voxel has no overlap with volumetric ROI, do not process it

                            if (n > 0 && reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0) {
                                slice(i, j, 0) -= reconstructor->_simulated_slices[inputIndex](i, j, 0);

                                //calculate norm and voxel-wise weights

                                //Gaussian distribution for inliers (likelihood)
                                const double g = reconstructor->G(slice(i, j, 0), reconstructor->_sigma);
                                //Uniform distribution for outliers (likelihood)
                                const double m = reconstructor->M(reconstructor->_m);

                                //voxel_wise posterior
                                const double w = g * reconstructor->_mix / (g * reconstructor->_mix + m * (1 - reconstructor->_mix));
                                weight.PutAsDouble(i, j, 0, w);

                                //calculate slice potentials
                                if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
                                    slice_potential[inputIndex] += (1 - w) * (1 - w);
                                    num++;
                                }
                            } else
                                weight.PutAsDouble(i, j, 0, 0);
                        }

                //evaluate slice potential
                if (num > 0)
                    slice_potential[inputIndex] = sqrt(slice_potential[inputIndex] / num);
                else
                    slice_potential[inputIndex] = -1; // slice has no unpadded voxels
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// E-step (todo: optimise for velocity)
    class EStepCardiacVelocity4D {
        ReconstructionCardiacVelocity4D *reconstructor;
        Array<double>& slice_potential;

    public:
        EStepCardiacVelocity4D(
            ReconstructionCardiacVelocity4D *reconstructor,
            Array<double>& slice_potential) :
            reconstructor(reconstructor),
            slice_potential(slice_potential) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                const RealImage& current_slice = reconstructor->_slices[inputIndex];
                RealImage slice = current_slice - reconstructor->_simulated_slices[inputIndex];
                memset(reconstructor->_weights[inputIndex].Data(), 0, sizeof(RealPixel) * reconstructor->_weights[inputIndex].NumberOfVoxels());

                for (int i = 0; i < current_slice.GetX(); i++)
                    for (int j = 0; j < current_slice.GetY(); j++)
                        if (current_slice(i, j, 0) < -10)
                            slice(i, j, 0) = -15;

                double num = 0;
                //Calculate error, voxel weights, and slice potential
                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (slice(i, j, 0) > -10) {
                            //number of volumetric voxels to which
                            // current slice voxel contributes
                            const int n = reconstructor->_volcoeffs[inputIndex][i][j].size();

                            // if n == 0, slice voxel has no overlap with volumetric ROI, do not process it
                            if (n > 0 && reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0) {
                                //calculate norm and voxel-wise weights
                                //Gaussian distribution for inliers (likelihood)
                                const double g = reconstructor->G(slice(i, j, 0), reconstructor->_sigma);
                                //Uniform distribution for outliers (likelihood)
                                const double m = reconstructor->M(reconstructor->_m);

                                //voxel_wise posterior
                                const double weight = g * reconstructor->_mix / (g * reconstructor->_mix + m * (1 - reconstructor->_mix));

                                reconstructor->_weights[inputIndex].PutAsDouble(i, j, 0, weight);

                                //calculate slice potentials
                                if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
                                    slice_potential[inputIndex] += (1 - weight) * (1 - weight);
                                    num++;
                                }
                            } else
                                reconstructor->_weights[inputIndex].PutAsDouble(i, j, 0, 0);
                        }

                //evaluate slice potential
                if (num > 0)
                    slice_potential[inputIndex] = sqrt(slice_potential[inputIndex] / num);
                else
                    slice_potential[inputIndex] = -1; // slice has no unpadded voxels
            }
        }

        // execute
        void operator() () const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for slice scale calculation
    class Scale {
        Reconstruction *reconstructor;

    public:
        Scale(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                //alias the current bias image
                const RealImage& b = reconstructor->_bias[inputIndex];

                //initialise calculation of scale
                double scalenum = 0;
                double scaleden = 0;

                for (int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++)
                    for (int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++)
                        if (reconstructor->_slices[inputIndex](i, j, 0) > -0.01) {
                            if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
                                //scale - intensity matching
                                const double eb = exp(-b(i, j, 0));
                                scalenum += reconstructor->_weights[inputIndex](i, j, 0) * reconstructor->_slices[inputIndex](i, j, 0) * eb * reconstructor->_simulated_slices[inputIndex](i, j, 0);
                                scaleden += reconstructor->_weights[inputIndex](i, j, 0) * reconstructor->_slices[inputIndex](i, j, 0) * eb * reconstructor->_slices[inputIndex](i, j, 0) * eb;
                            }
                        }

                //calculate scale for this slice
                reconstructor->_scale[inputIndex] = scaleden > 0 ? scalenum / scaleden : 1;
            } //end of loop for a slice inputIndex
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for slice bias calculation
    class Bias {
        Reconstruction *reconstructor;

    public:
        Bias(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            RealImage slice, wb, wresidual;
            GaussianBlurring<RealPixel> gb(reconstructor->_sigma_bias);

            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                slice = reconstructor->_slices[inputIndex];

                //alias the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];

                //prepare weight image for bias field
                wb = reconstructor->_weights[inputIndex];

                wresidual.Initialize(slice.Attributes());

                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
                                //bias-correct and scale current slice
                                const double eb = exp(-b(i, j, 0));
                                slice(i, j, 0) *= eb * reconstructor->_scale[inputIndex];

                                //calculate weight image
                                wb(i, j, 0) *= slice(i, j, 0);

                                //calculate weighted residual image make sure it is far from zero to avoid numerical instability
                                if ((reconstructor->_simulated_slices[inputIndex](i, j, 0) > 1) && (slice(i, j, 0) > 1))
                                    wresidual(i, j, 0) = log(slice(i, j, 0) / reconstructor->_simulated_slices[inputIndex](i, j, 0)) * wb(i, j, 0);
                            } else {
                                //do not take into account this voxel when calculating bias field
                                wresidual(i, j, 0) = 0;
                                wb(i, j, 0) = 0;
                            }
                        }

                //calculate bias field for this slice
                //smooth weighted residual
                gb.Input(&wresidual);
                gb.Output(&wresidual);
                gb.Run();

                //smooth weight image
                gb.Input(&wb);
                gb.Output(&wb);
                gb.Run();

                //update bias field
                double sum = 0;
                double num = 0;
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            if (wb(i, j, 0) > 0)
                                b(i, j, 0) += wresidual(i, j, 0) / wb(i, j, 0);
                            sum += b(i, j, 0);
                            num++;
                        }

                //normalize bias field to have zero mean
                if (!reconstructor->_global_bias_correction && num > 0) {
                    const double mean = sum / num;
                    for (int i = 0; i < slice.GetX(); i++)
                        for (int j = 0; j < slice.GetY(); j++)
                            if (slice(i, j, 0) > -0.01)
                                b(i, j, 0) -= mean;
                }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for SR reconstruction
    class Superresolution {
        Reconstruction *reconstructor;

    public:
        RealImage confidence_map;
        RealImage addon;
        Array<RealImage> mc_addons;

        Superresolution(Reconstruction *reconstructor) : reconstructor(reconstructor) {
            //Clear addon
            addon.Initialize(reconstructor->_reconstructed.Attributes());

            //Clear confidence map
            confidence_map.Initialize(reconstructor->_reconstructed.Attributes());

            if (reconstructor->_multiple_channels_flag) {
                mc_addons.clear();
                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                    mc_addons.push_back(addon);
                }
            }

        }

        Superresolution(Superresolution& x, split) : Superresolution(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            //Update reconstructed volume using current slice
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                //Distribute error to the volume
                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++) {

                        double test_val;
                        if (reconstructor->_no_masking_background)
                            test_val = reconstructor->_not_masked_slices[inputIndex](i, j, 0);
                        else
                            test_val = reconstructor->_slices[inputIndex](i, j, 0);

                        if (test_val > -0.01) {
                            if (reconstructor->_simulated_slices[inputIndex](i, j, 0) < 0.01) {
                                reconstructor->_slice_dif[inputIndex](i, j, 0) = 0;

                                if (reconstructor->_multiple_channels_flag) {
                                    for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                        reconstructor->_mc_slice_dif[inputIndex][nc]->PutAsDouble(i, j, 0, 0);
                                    }
                                }
                            }


                            #pragma omp simd
                            for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                const auto multiplier = reconstructor->_robust_slices_only ? 1 : reconstructor->_weights[inputIndex](i, j, 0);
                                double ssim_weight = 1;
                                if (reconstructor->_structural)
                                    ssim_weight = reconstructor->_slice_ssim_maps[inputIndex](i, j, 0);

                                bool include_flag = true;
                                if (reconstructor->_ffd) {
                                    double jac = reconstructor->_mffd_transformations[inputIndex]->Jacobian(p.x, p.y, p.z, 0, 0);
                                    if ((100*jac) < reconstructor->_global_JAC_threshold)
                                        include_flag = false;
                                }

                                if (include_flag) {
                                    addon(p.x, p.y, p.z) += ssim_weight * multiplier * p.value * reconstructor->_slice_weight[inputIndex] * reconstructor->_slice_dif[inputIndex](i, j, 0);

                                    confidence_map(p.x, p.y, p.z) += ssim_weight * multiplier * p.value * reconstructor->_slice_weight[inputIndex];

                                    if (reconstructor->_multiple_channels_flag) {
                                        for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                            mc_addons[nc](p.x, p.y, p.z) += ssim_weight * multiplier * p.value * reconstructor->_slice_weight[inputIndex] * reconstructor->_mc_slice_dif[inputIndex][nc]->GetAsDouble(i, j, 0);
                                        }
                                    }

                                }

                            }
                        }
                    }
            } //end of loop for a slice inputIndex
        }

        void join(const Superresolution& y) {
            addon += y.addon;
            confidence_map += y.confidence_map;

            if (reconstructor->_multiple_channels_flag) {
                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                    mc_addons[nc] += y.mc_addons[nc];
                }
            }

        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class SuperresolutionCardiac4D {
        ReconstructionCardiac4D *reconstructor;

    public:
        RealImage confidence_map;
        RealImage addon;

        SuperresolutionCardiac4D(ReconstructionCardiac4D *reconstructor) : reconstructor(reconstructor) {
            //Clear addon
            addon.Initialize(reconstructor->_reconstructed4D.Attributes());

            //Clear confidence map
            confidence_map.Initialize(reconstructor->_reconstructed4D.Attributes());
        }

        SuperresolutionCardiac4D(SuperresolutionCardiac4D& x, split) : SuperresolutionCardiac4D(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            RealImage slice;

            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                if (reconstructor->_volcoeffs[inputIndex].empty())
                    continue;

                // read the current slice
                slice = reconstructor->_slices[inputIndex];

                //Update reconstructed volume using current slice
                //Distribute error to the volume
                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (slice(i, j, 0) != -1) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-reconstructor->_bias[inputIndex](i, j, 0)) * reconstructor->_scale[inputIndex];

                            if (reconstructor->_simulated_slices[inputIndex](i, j, 0) > 0)
                                slice(i, j, 0) -= reconstructor->_simulated_slices[inputIndex](i, j, 0);
                            else
                                slice(i, j, 0) = 0;

                            #pragma omp simd
                            for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                for (int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++) {
                                    const auto multiplier = reconstructor->_robust_slices_only ? 1 : reconstructor->_weights[inputIndex](i, j, 0);
                                    addon(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * multiplier * reconstructor->_slice_weight[inputIndex];
                                    confidence_map(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * multiplier * reconstructor->_slice_weight[inputIndex];
                                }
                            }
                        }
            } //end of loop for a slice inputIndex
        }

        void join(const SuperresolutionCardiac4D& y) {
            addon += y.addon;
            confidence_map += y.confidence_map;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Gradient descend step of velocity estimation
    class SuperresolutionCardiacVelocity4D {
        ReconstructionCardiacVelocity4D *reconstructor;

    public:
        Array<RealImage> confidence_maps;
        Array<RealImage> addons_p;
        Array<RealImage> addons_n;
        Array<RealImage> addons;

        SuperresolutionCardiacVelocity4D(ReconstructionCardiacVelocity4D *reconstructor) : reconstructor(reconstructor) {
            addons = confidence_maps = Array<RealImage>(reconstructor->_reconstructed5DVelocity.size(), RealImage(reconstructor->_reconstructed4D.Attributes()));
        }

        SuperresolutionCardiacVelocity4D(SuperresolutionCardiacVelocity4D& x, split) : SuperresolutionCardiacVelocity4D(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                if (reconstructor->_volcoeffs[inputIndex].empty())
                    continue;

                // Read the current slice
                const RealImage& current_slice = reconstructor->_slices[inputIndex];

                // Read the current simulated slice
                const RealImage& sim = reconstructor->_simulated_slices[inputIndex];

                // Compute error between simulated and original slices
                RealImage slice = current_slice - sim;
                for (int i = 0; i < current_slice.GetX(); i++)
                    for (int j = 0; j < current_slice.GetY(); j++)
                        if (current_slice(i, j, 0) > -10 && sim(i, j, 0) < -10)
                            slice(i, j, 0) = 0;

                // Read the current weight image
                const RealImage& w = reconstructor->_weights[inputIndex];

                // Gradient moment magnitude for the current slice
                const int gradientIndex = reconstructor->_stack_index[inputIndex];
                const double gval = reconstructor->_g_values[gradientIndex];

                // Update reconstructed velocity volumes using current slice
                for (size_t velocityIndex = 0; velocityIndex < reconstructor->_v_directions.size(); velocityIndex++) {
                    // Compute current velocity component factor
                    const double v_component = reconstructor->_slice_g_directions[inputIndex][velocityIndex] / (3 * reconstructor->gamma * gval);

                    // Distribute error to the volume
                    for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                        for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                            if (slice(i, j, 0) > -10) {
                                if (sim(i, j, 0) < -10)
                                    slice(i, j, 0) = 0;

                                for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                    const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                    if (p.value > 0.0) {
                                        for (int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++) {
                                            const auto multiplier = reconstructor->_robust_slices_only ? 1 : reconstructor->_slice_weight[inputIndex];
                                            addons[velocityIndex](p.x, p.y, p.z, outputIndex) += v_component * reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * w(i, j, 0) * multiplier;
                                            confidence_maps[velocityIndex](p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * w(i, j, 0) * multiplier;
                                        }
                                    }
                                }
                            }
                } // end of loop for velocity directions
            } //end of loop for a slice inputIndex
        }

        void join(const SuperresolutionCardiacVelocity4D& y) {
            for (size_t i = 0; i < addons.size(); i++) {
                addons[i] += y.addons[i];
                confidence_maps[i] += y.confidence_maps[i];
            }
        }

        // execute
        void operator() () {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };


    //-------------------------------------------------------------------


    // class for SStep (local structure-based outlier rejection)

    class SStep {
        Reconstruction *reconstructor;
        size_t N;

    public:
        SStep(Reconstruction *reconstructor, size_t N) : reconstructor(reconstructor),
            N(N) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {

                reconstructor->SliceSSIMMap(inputIndex);

            } //end of loop for a slice inputIndex
        }

        void operator()() {
            parallel_for(blocked_range<size_t>(0, N), *this);
        }
    };


    //-------------------------------------------------------------------


    /// Class for MStep (RS)
    class MStep {
        Reconstruction *reconstructor;

    public:
        double sigma;
        double mix;
        double num;
        double min;
        double max;

        MStep(Reconstruction *reconstructor) : reconstructor(reconstructor) {
            sigma = 0;
            mix = 0;
            num = 0;
            min = voxel_limits<RealPixel>::max();
            max = voxel_limits<RealPixel>::min();
        }

        MStep(MStep& x, split) : MStep(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            RealImage slice;

            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                slice = reconstructor->_slices[inputIndex];

                //calculate error
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-reconstructor->_bias[inputIndex](i, j, 0)) * reconstructor->_scale[inputIndex];

                            //otherwise the error has no meaning - it is equal to slice intensity
                            if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
                                slice(i, j, 0) -= reconstructor->_simulated_slices[inputIndex](i, j, 0);

                                //sigma and mix
                                const double e = slice(i, j, 0);
                                sigma += e * e * reconstructor->_weights[inputIndex](i, j, 0);
                                mix += reconstructor->_weights[inputIndex](i, j, 0);

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

        void join(const MStep& y) {
            if (y.min < min)
                min = y.min;
            if (y.max > max)
                max = y.max;

            sigma += y.sigma;
            mix += y.mix;
            num += y.num;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// M-step (todo: optimise for velocity)
    class MStepCardiacVelocity4D {
        ReconstructionCardiacVelocity4D *reconstructor;

    public:
        double sigma;
        double mix;
        double num;
        double min;
        double max;

        MStepCardiacVelocity4D(ReconstructionCardiacVelocity4D *reconstructor) : reconstructor(reconstructor) {
            sigma = 0;
            mix = 0;
            num = 0;
            min = voxel_limits<RealPixel>::max();
            max = voxel_limits<RealPixel>::min();
        }

        MStepCardiacVelocity4D(MStepCardiacVelocity4D& x, split) : MStepCardiacVelocity4D(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                const RealImage& current_slice = reconstructor->_slices[inputIndex];

                //alias the current weight image
                const RealImage& w = reconstructor->_weights[inputIndex];

                RealImage slice = current_slice - reconstructor->_simulated_slices[inputIndex];

                for (int i = 0; i < current_slice.GetX(); i++)
                    for (int j = 0; j < current_slice.GetY(); j++)
                        if (current_slice(i, j, 0) < -10)
                            slice(i, j, 0) = -15;

                //calculate error
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -10) {
                            //otherwise the error has no meaning - it is equal to slice intensity
                            if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
                                //sigma and mix
                                const double e = slice(i, j, 0);
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

        void join(const MStepCardiacVelocity4D& y) {
            if (y.min < min)
                min = y.min;
            if (y.max > max)
                max = y.max;

            sigma += y.sigma;
            mix += y.mix;
            num += y.num;
        }

        // execute
        void operator() () {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for adaptive regularisation (Part I)
    class AdaptiveRegularization1 {
        const Reconstruction *reconstructor;
        Array<RealImage>& b;
        const Array<double>& factor;
        const RealImage& original;

    public:
        AdaptiveRegularization1(
            const Reconstruction *reconstructor,
            Array<RealImage>& b,
            const Array<double>& factor,
            const RealImage& original) :
            reconstructor(reconstructor),
            b(b),
            factor(factor),
            original(original) {}

        void operator()(const blocked_range<size_t>& r) const {
            const int dx = reconstructor->_reconstructed.GetX();
            const int dy = reconstructor->_reconstructed.GetY();
            const int dz = reconstructor->_reconstructed.GetZ();
            for (size_t i = r.begin(); i != r.end(); i++) {
                for (int x = 0; x < dx; x++)
                    for (int y = 0; y < dy; y++)
                        for (int z = 0; z < dz; z++) {
                            const int xx = x + reconstructor->_directions[i][0];
                            const int yy = y + reconstructor->_directions[i][1];
                            const int zz = z + reconstructor->_directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                && (reconstructor->_confidence_map(x, y, z) > 0) && (reconstructor->_confidence_map(xx, yy, zz) > 0)) {
                                const double diff = (original(xx, yy, zz) - original(x, y, z)) * sqrt(factor[i]) / reconstructor->_delta;
                                b[i](x, y, z) = factor[i] / sqrt(1 + diff * diff);
                            } else
                                b[i](x, y, z) = 0;
                        }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, 13), *this);
        }
    };


    //-------------------------------------------------------------------


    class AdaptiveRegularization1MC {
        const Reconstruction *reconstructor;
        Array<Array<RealImage>> &b;
        const Array<double> &factor;
        const Array<RealImage> &original;

    public:
        AdaptiveRegularization1MC(const Reconstruction *_reconstructor,
                                  Array<Array<RealImage>> &_b,
                                  const Array<double> &_factor,
                                  const Array<RealImage> &_original) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }

        void operator() (const blocked_range<size_t> &r) const {
            const int dx = reconstructor->_reconstructed.GetX();
            const int dy = reconstructor->_reconstructed.GetY();
            const int dz = reconstructor->_reconstructed.GetZ();
            for ( size_t i = r.begin(); i != r.end(); ++i ) {

                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {

                    int x, y, z, xx, yy, zz;
//                    double diff;
                    for (x = 0; x < dx; x++)
                    for (y = 0; y < dy; y++)
                    for (z = 0; z < dz; z++) {
                        xx = x + reconstructor->_directions[i][0];
                        yy = y + reconstructor->_directions[i][1];
                        zz = z + reconstructor->_directions[i][2];
                        if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                            && (reconstructor->_confidence_map(x, y, z) > 0) && (reconstructor->_confidence_map(xx, yy, zz) > 0)) {
                            const double diff = (original[nc](xx, yy, zz) - original[nc](x, y, z)) * sqrt(factor[i]) / reconstructor->_delta;
                            b[i][nc](x, y, z) = factor[i] / sqrt(1 + diff * diff);
                        }
                        else {
                            b[i][nc](x, y, z) = 0;
                        }
                    }

                }
            }
        }

        void operator() () const {
            parallel_for( blocked_range<size_t>(0, 13), *this );
        }

    };


    //-------------------------------------------------------------------

    /// Class for adaptive regularisation (Part II)
    class AdaptiveRegularization2 {
        Reconstruction *reconstructor;
        const Array<RealImage>& b;
        const RealImage& original;

    public:
        AdaptiveRegularization2(
            Reconstruction *reconstructor,
            const Array<RealImage>& b,
            const RealImage& original) :
            reconstructor(reconstructor),
            b(b),
            original(original) {}

        void operator()(const blocked_range<size_t>& r) const {
            const int dx = reconstructor->_reconstructed.GetX();
            const int dy = reconstructor->_reconstructed.GetY();
            const int dz = reconstructor->_reconstructed.GetZ();
            for (size_t x = r.begin(); x != r.end(); ++x) {
                for (int y = 0; y < dy; y++)
                    for (int z = 0; z < dz; z++)
                        if (reconstructor->_confidence_map(x, y, z) > 0) {
                            double val = 0;
                            double sum = 0;

                            // Don't combine the loops, it breaks the compiler optimisation (GCC 9)
                            for (int i = 0; i < 13; i++) {
                                const int xx = x + reconstructor->_directions[i][0];
                                const int yy = y + reconstructor->_directions[i][1];
                                const int zz = z + reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                    && reconstructor->_confidence_map(xx, yy, zz) > 0) {
                                    val += b[i](x, y, z) * original(xx, yy, zz);
                                    sum += b[i](x, y, z);
                                }
                            }

                            for (int i = 0; i < 13; i++) {
                                const int xx = x - reconstructor->_directions[i][0];
                                const int yy = y - reconstructor->_directions[i][1];
                                const int zz = z - reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                    && reconstructor->_confidence_map(xx, yy, zz) > 0) {
                                    val += b[i](x, y, z) * original(xx, yy, zz);
                                    sum += b[i](x, y, z);
                                }
                            }

                            val -= sum * original(x, y, z);
                            val = original(x, y, z) + reconstructor->_alpha * reconstructor->_lambda / (reconstructor->_delta * reconstructor->_delta) * val;
                            reconstructor->_reconstructed(x, y, z) = val;
                        }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_reconstructed.GetX()), *this);
        }
    };

    //-------------------------------------------------------------------


    class AdaptiveRegularization2MC {
        Reconstruction *reconstructor;
        Array<Array<RealImage>> &b;
        Array<double> &factor;
        Array<RealImage> &original;

        public:
        AdaptiveRegularization2MC( Reconstruction *_reconstructor,
                                           Array<Array<RealImage>> &_b,
                                           Array<double> &_factor,
                                           Array<RealImage> &_original) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }

        void operator() (const blocked_range<size_t> &r) const {
            int dx = reconstructor->_reconstructed.GetX();
            int dy = reconstructor->_reconstructed.GetY();
            int dz = reconstructor->_reconstructed.GetZ();
            for ( size_t x = r.begin(); x != r.end(); ++x ) {
                int xx, yy, zz;
                for (int y = 0; y < dy; y++)
                for (int z = 0; z < dz; z++)
                if(reconstructor->_confidence_map(x,y,z)>0)
                {
                    for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                        double val = 0;
                        double sum = 0;

                        for (int i = 0; i < 13; i++) {
                            xx = x + reconstructor->_directions[i][0];
                            yy = y + reconstructor->_directions[i][1];
                            zz = z + reconstructor->_directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                            if(reconstructor->_confidence_map(xx,yy,zz)>0)
                            {
                                val += b[i][nc](x, y, z) * original[nc](xx, yy, zz);
                                sum += b[i][nc](x, y, z);
                            }
                        }

                        for (int i = 0; i < 13; i++) {
                            xx = x - reconstructor->_directions[i][0];
                            yy = y - reconstructor->_directions[i][1];
                            zz = z - reconstructor->_directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                            if(reconstructor->_confidence_map(xx,yy,zz)>0)
                            {
                                val += b[i][nc](x, y, z) * original[nc](xx, yy, zz);
                                sum += b[i][nc](x, y, z);
                            }
                        }
                        val -= sum * original[nc](x, y, z);
                        val = original[nc](x, y, z) + reconstructor->_alpha * reconstructor->_lambda / (reconstructor->_delta * reconstructor->_delta) * val;
                        reconstructor->_mc_reconstructed[nc](x, y, z) = val;
                    }
                }

            }
        }

        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed.GetX()), *this );
        }

    };


    //-------------------------------------------------------------------

    class AdaptiveRegularization1Cardiac4D {
        const ReconstructionCardiac4D *reconstructor;
        Array<RealImage>& b;
        const Array<double>& factor;
        const RealImage& original;

    public:
        AdaptiveRegularization1Cardiac4D(
            const ReconstructionCardiac4D *reconstructor,
            Array<RealImage>& b,
            const Array<double>& factor,
            const RealImage& original) :
            reconstructor(reconstructor),
            b(b),
            factor(factor),
            original(original) {}

        void operator()(const blocked_range<size_t>& r) const {
            const int dx = reconstructor->_reconstructed4D.GetX();
            const int dy = reconstructor->_reconstructed4D.GetY();
            const int dz = reconstructor->_reconstructed4D.GetZ();
            const int dt = reconstructor->_reconstructed4D.GetT();
            for (size_t i = r.begin(); i != r.end(); i++) {
                for (int x = 0; x < dx; x++)
                    for (int y = 0; y < dy; y++)
                        for (int z = 0; z < dz; z++) {
                            const int xx = x + reconstructor->_directions[i][0];
                            const int yy = y + reconstructor->_directions[i][1];
                            const int zz = z + reconstructor->_directions[i][2];
                            for (int t = 0; t < dt; t++) {
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                    && (reconstructor->_confidence_map(x, y, z, t) > 0) && (reconstructor->_confidence_map(xx, yy, zz, t) > 0)) {
                                    const double diff = (original(xx, yy, zz, t) - original(x, y, z, t)) * sqrt(factor[i]) / reconstructor->_delta;
                                    b[i](x, y, z, t) = factor[i] / sqrt(1 + diff * diff);
                                } else
                                    b[i](x, y, z, t) = 0;
                            }
                        }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, 13), *this);
        }
    };

    //-------------------------------------------------------------------

    class AdaptiveRegularization2Cardiac4D {
        ReconstructionCardiac4D *reconstructor;
        const Array<RealImage>& b;
        const RealImage& original;

    public:
        AdaptiveRegularization2Cardiac4D(
            ReconstructionCardiac4D *reconstructor,
            const Array<RealImage>& b,
            const RealImage& original) :
            reconstructor(reconstructor),
            b(b),
            original(original) {}

        void operator()(const blocked_range<size_t>& r) const {
            const int dx = reconstructor->_reconstructed4D.GetX();
            const int dy = reconstructor->_reconstructed4D.GetY();
            const int dz = reconstructor->_reconstructed4D.GetZ();
            const int dt = reconstructor->_reconstructed4D.GetT();
            for (size_t x = r.begin(); x != r.end(); ++x) {
                for (int y = 0; y < dy; y++)
                    for (int z = 0; z < dz; z++)
                        for (int t = 0; t < dt; t++) {
                            if (reconstructor->_confidence_map(x, y, z, t) > 0) {
                                double val = 0;
                                double sum = 0;
                                for (int i = 0; i < 13; i++) {
                                    const int xx = x + reconstructor->_directions[i][0];
                                    const int yy = y + reconstructor->_directions[i][1];
                                    const int zz = z + reconstructor->_directions[i][2];
                                    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                        && (reconstructor->_confidence_map(xx, yy, zz, t) > 0)) {
                                        val += b[i](x, y, z, t) * original(xx, yy, zz, t);
                                        sum += b[i](x, y, z, t);
                                    }
                                }

                                for (int i = 0; i < 13; i++) {
                                    const int xx = x - reconstructor->_directions[i][0];
                                    const int yy = y - reconstructor->_directions[i][1];
                                    const int zz = z - reconstructor->_directions[i][2];
                                    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                        && (reconstructor->_confidence_map(xx, yy, zz, t) > 0)) {
                                            val += b[i](x, y, z, t) * original(xx, yy, zz, t);
                                            sum += b[i](x, y, z, t);
                                        }
                                }

                                val -= sum * original(x, y, z, t);
                                val = original(x, y, z, t) + reconstructor->_alpha * reconstructor->_lambda / (reconstructor->_delta * reconstructor->_delta) * val;
                                reconstructor->_reconstructed4D(x, y, z, t) = val;
                            }
                        }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_reconstructed4D.GetX()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for bias normalisation
    class NormaliseBias {
        Reconstruction *reconstructor;

    public:
        RealImage bias;

        NormaliseBias(Reconstruction *reconstructor) : reconstructor(reconstructor) {
            bias.Initialize(reconstructor->_reconstructed.Attributes());
        }

        NormaliseBias(NormaliseBias& x, split) : NormaliseBias(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            RealImage b;

            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                if (reconstructor->_volcoeffs[inputIndex].empty())
                    continue;

                if (reconstructor->_verbose)
                    reconstructor->_verbose_log << inputIndex << " ";

                // alias to the current slice
                const RealImage& slice = reconstructor->_no_masking_background ? reconstructor->_not_masked_slices[inputIndex] : reconstructor->_slices[inputIndex];

                //read the current bias image
                b = reconstructor->_bias[inputIndex];

                //read current scale factor
                const double scale = reconstructor->_scale[inputIndex];

                const RealPixel *pi = slice.Data();
                RealPixel *pb = b.Data();
                for (int i = 0; i < slice.NumberOfVoxels(); i++)
                    if (pi[i] > -1 && scale > 0)
                        pb[i] -= log(scale);

                //Distribute slice intensities to the volume
                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            //add contribution of current slice voxel to all voxel volumes to which it contributes
                            for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                bias(p.x, p.y, p.z) += p.value * b(i, j, 0);
                            }
                        }
                //end of loop for a slice inputIndex
            }
        }

        void join(const NormaliseBias& y) {
            bias += y.bias;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class NormaliseBiasCardiac4D {
        ReconstructionCardiac4D *reconstructor;

    public:
        RealImage bias;
        RealImage volweight3d;

        NormaliseBiasCardiac4D(ReconstructionCardiac4D *reconstructor) : reconstructor(reconstructor) {
            ImageAttributes attr = reconstructor->_reconstructed4D.Attributes();
            attr._t = 1;
            bias.Initialize(attr);
            volweight3d.Initialize(attr);
        }

        NormaliseBiasCardiac4D(NormaliseBiasCardiac4D& x, split) : NormaliseBiasCardiac4D(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            RealImage b;
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                if (reconstructor->_volcoeffs[inputIndex].empty())
                    continue;

                if (reconstructor->_verbose)
                    reconstructor->_verbose_log << inputIndex << " ";

                // alias to the current slice
                const RealImage& slice = reconstructor->_slices[inputIndex];

                //read the current bias image
                b = reconstructor->_bias[inputIndex];

                //read current scale factor
                const double scale = reconstructor->_scale[inputIndex];

                const RealPixel *pi = slice.Data();
                RealPixel *pb = b.Data();
                for (int i = 0; i < slice.NumberOfVoxels(); i++)
                    if (pi[i] > -1 && scale > 0)
                        pb[i] -= log(scale);

                //Distribute slice intensities to the volume
                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (slice(i, j, 0) != -1) {
                            //add contribution of current slice voxel to all voxel volumes
                            //to which it contributes
                            for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                bias(p.x, p.y, p.z) += p.value * b(i, j, 0);
                                volweight3d(p.x, p.y, p.z) += p.value;
                            }
                        }
                //end of loop for a slice inputIndex
            }
        }

        void join(const NormaliseBiasCardiac4D& y) {
            bias += y.bias;
            volweight3d += y.volweight3d;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for global stack similarity statistics (for the stack selection function)
    class GlobalSimilarityStats {
        const int nStacks;
        const Array<RealImage>& stacks;
        const Array<RealImage>& masks;
        Array<double>& all_global_ncc_array;
        Array<double>& all_global_volume_array;

    public:
        GlobalSimilarityStats(
            const int nStacks,
            const Array<RealImage>& stacks,
            const Array<RealImage>& masks,
            Array<double>& all_global_ncc_array,
            Array<double>& all_global_volume_array) :
            nStacks(nStacks),
            stacks(stacks),
            masks(masks),
            all_global_ncc_array(all_global_ncc_array),
            all_global_volume_array(all_global_volume_array) {}

        void operator()(const blocked_range<size_t>& r) const {
            Array<RigidTransformation> current_stack_transformations;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                double average_ncc = 0;
                double average_volume = 0;

                cout << " - " << inputIndex << endl;
                GlobalStackStats(stacks[inputIndex], masks[inputIndex], stacks, masks, average_ncc, average_volume, current_stack_transformations);
                current_stack_transformations.clear();
                cout << " + " << inputIndex << endl;

                all_global_ncc_array[inputIndex] = average_ncc;
                all_global_volume_array[inputIndex] = average_volume;
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, nStacks), *this);
        }
    };

    //-------------------------------------------------------------------

    class CalculateCorrectedSlices {
        ReconstructionCardiac4D *reconstructor;

    public:
        CalculateCorrectedSlices(ReconstructionCardiac4D *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                //read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];

                //calculate corrected voxels
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1)
                            //bias correct and scale the voxel
                            slice(i, j, 0) *= exp(-reconstructor->_bias[inputIndex](i, j, 0)) * reconstructor->_scale[inputIndex];

                reconstructor->_corrected_slices[inputIndex] = move(slice);
            } //end of loop for a slice inputIndex
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class CalculateError {
        ReconstructionCardiac4D *reconstructor;

    public:
        CalculateError(ReconstructionCardiac4D *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            RealImage slice;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                //read the current slice
                slice = reconstructor->_slices[inputIndex];

                //initialise
                reconstructor->_error[inputIndex].Initialize(slice.Attributes());

                //calculate error
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1)
                            if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0) {
                                //bias correct and scale the voxel
                                slice(i, j, 0) *= exp(-reconstructor->_bias[inputIndex](i, j, 0)) * reconstructor->_scale[inputIndex];
                                //subtract simulated voxel
                                slice(i, j, 0) -= reconstructor->_simulated_slices[inputIndex](i, j, 0);
                                //assign as error
                                reconstructor->_error[inputIndex](i, j, 0) = slice(i, j, 0);
                            }
            } //end of loop for a slice inputIndex
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------
}
