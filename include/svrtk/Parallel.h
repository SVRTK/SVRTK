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
#include "svrtk/Reconstruction.h"
#include "svrtk/ReconstructionCardiac4D.h"

// Boost
#include <boost/format.hpp>

using namespace mirtk;
using namespace svrtk;

namespace svrtk::Parallel {

    // Class for computing average of all stacks
    class Average {
        Reconstruction *reconstructor;
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
            Reconstruction *reconstructor,
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
            for (size_t i = r.begin(); i < r.end(); ++i) {
                GenericLinearInterpolateImageFunction<RealImage> interpolator;
                RealImage image(reconstructor->_reconstructed.Attributes());

                ImageTransformation imagetransformation;
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
                const double out_ncc = reconstructor->ComputeNCC(reconstructor->_slices[inputIndex], reconstructor->_simulated_slices[inputIndex], 0.1);
                out_global_ncc += out_ncc;

                double s_diff = 0;
                double s_t = 0;
                int s_n = 0;
                double point_ns = 0;
                double point_nt = 0;

                for (int x = 0; x < reconstructor->_slices[inputIndex].GetX(); x++) {
                    for (int y = 0; y < reconstructor->_slices[inputIndex].GetY(); y++) {
                        for (int z = 0; z < reconstructor->_slices[inputIndex].GetZ(); z++) {
                            if (reconstructor->_slices[inputIndex](x, y, z) > 0 && reconstructor->_simulated_slices[inputIndex](x, y, z) > 0) {
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
        Reconstruction *reconstructor;
        const Array<RealImage>& stacks;
        Array<RigidTransformation>& stack_transformations;
        int templateNumber;
        RealImage& target;
        RigidTransformation& offset;

    public:
        StackRegistrations(
            Reconstruction *reconstructor,
            const Array<RealImage>& stacks,
            Array<RigidTransformation>& stack_transformations,
            int templateNumber,
            RealImage& target,
            RigidTransformation& offset) :
            reconstructor(reconstructor),
            stacks(stacks),
            stack_transformations(stack_transformations),
            templateNumber(templateNumber),
            target(target),
            offset(offset) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                if (!reconstructor->_template_flag) {
                    //do not perform registration for template
                    if (i == templateNumber)
                        continue;
                }

                ResamplingWithPadding<RealPixel> resampling2(1.5, 1.5, 1.5, 0);
                GenericLinearInterpolateImageFunction<RealImage> interpolator2;
                resampling2.Interpolator(&interpolator2);

                RealImage r_source(stacks[i].Attributes());
                resampling2.Input(&stacks[i]);
                resampling2.Output(&r_source);
                resampling2.Run();

                RealImage r_target(target.Attributes());
                resampling2.Input(&target);
                resampling2.Output(&r_target);
                resampling2.Run();

                Matrix mo = offset.GetMatrix();
                Matrix m = stack_transformations[i].GetMatrix() * mo;
                stack_transformations[i].PutMatrix(m);

                ParameterList params;
                Insert(params, "Transformation model", "Rigid");
                if (reconstructor->_nmi_bins > 0)
                    Insert(params, "No. of bins", reconstructor->_nmi_bins);

                if (reconstructor->_masked_stacks) {
                    Insert(params, "Background value", 0);
                } else {
                    Insert(params, "Background value for image 1", 0);
                }

                // Insert(params, "Background value", 0);
                // Insert(params, "Image (dis-)similarity measure", "NCC");
                //// Insert(params, "Image interpolation mode", "Linear");
                // string type = "box";
                // string units = "mm";
                // double width = 0;
                // Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);

                GenericRegistrationFilter registration;
                registration.Parameter(params);
                registration.Input(&r_target, &r_source);
                Transformation *dofout = nullptr;
                registration.Output(&dofout);
                registration.InitialGuess(&stack_transformations[i]);
                registration.GuessParameter();
                registration.Run();

                RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*>(dofout);
                stack_transformations[i] = *rigidTransf;

                m = stack_transformations[i].GetMatrix() * mo.Invert();
                stack_transformations[i].PutMatrix(m);

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

    // class for simulation of slices from the current reconstructed 3D volume - v2 (use this one)
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

                for (int i = 0; i < reconstructor->_slice_attributes[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attributes[inputIndex]._y; j++)
                        if (reconstructor->_slices[inputIndex](i, j, 0) > -0.01) {
                            double weight = 0;
                            const size_t n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for (size_t k = 0; k < n; k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                sim_slice(i, j, 0) += p.value * reconstructor->_reconstructed(p.x, p.y, p.z);
                                weight += p.value;
                                if (reconstructor->_mask(p.x, p.y, p.z) > 0.1) {
                                    sim_inside(i, j, 0) = 1;
                                    reconstructor->_slice_inside[inputIndex] = true;
                                }
                            }
                            if (weight > 0) {
                                sim_slice(i, j, 0) /= weight;
                                sim_weight(i, j, 0) = weight;
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

        void operator() (const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                //Calculate simulated slice
                reconstructor->_simulated_slices[inputIndex].Initialize(reconstructor->_slices[inputIndex].Attributes());
                reconstructor->_simulated_weights[inputIndex].Initialize(reconstructor->_slices[inputIndex].Attributes());
                reconstructor->_simulated_inside[inputIndex].Initialize(reconstructor->_slices[inputIndex].Attributes());
                reconstructor->_slice_inside[inputIndex] = false;

                for (int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++)
                    for (int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++)
                        if (reconstructor->_slices[inputIndex](i, j, 0) != -1) {
                            double weight = 0;
                            const size_t n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for (size_t k = 0; k < n; k++) {
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

        void operator() () const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    // class for parallel SVR
    class SliceToVolumeRegistration {
        Reconstruction *reconstructor;

    public:
        SliceToVolumeRegistration(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                GreyPixel smin, smax;
                GreyImage target;
                target = reconstructor->_grey_slices[inputIndex];
                target.GetMinMax(&smin, &smax);

                if (smax > 1 && (smax - smin) > 1) {
                    ParameterList params;
                    Insert(params, "Transformation model", "Rigid");
                    GreyPixel tmin, tmax;
                    reconstructor->_grey_reconstructed.GetMinMax(&tmin, &tmax);

                    if (smin < 1) {
                        Insert(params, "Background value for image 1", -1);
                    }

                    if (tmin < 0) {
                        Insert(params, "Background value for image 2", -1);
                    } else if (tmin < 1) {
                        Insert(params, "Background value for image 2", 0);
                    }

                    if (!reconstructor->_ncc_reg) {
                        Insert(params, "Image (dis-)similarity measure", "NMI");
                        if (reconstructor->_nmi_bins > 0)
                            Insert(params, "No. of bins", reconstructor->_nmi_bins);
                    } else {
                        Insert(params, "Image (dis-)similarity measure", "NCC");
                        string type = "sigma";
                        string units = "mm";
                        double width = 0;
                        Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);
                    }

                    unique_ptr<GenericRegistrationFilter> registration(new GenericRegistrationFilter());
                    registration->Parameter(params);

                    //put origin to zero
                    RigidTransformation offset;
                    reconstructor->ResetOrigin(target, offset);
                    Matrix mo = offset.GetMatrix();
                    Matrix m = reconstructor->_transformations[inputIndex].GetMatrix() * mo;
                    reconstructor->_transformations[inputIndex].PutMatrix(m);

                    // run registration
                    registration->Input(&target, &reconstructor->_grey_reconstructed);
                    Transformation *dofout = nullptr;
                    registration->Output(&dofout);
                    registration->InitialGuess(&reconstructor->_transformations[inputIndex]);
                    registration->GuessParameter();
                    registration->Run();

                    // output transformation
                    RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*>(dofout);
                    reconstructor->_transformations[inputIndex] = *rigidTransf;

                    //undo the offset
                    m = reconstructor->_transformations[inputIndex].GetMatrix() * mo.Invert();
                    reconstructor->_transformations[inputIndex].PutMatrix(m);

                    // save log outputs
                    if (reconstructor->_reg_log) {
                        double tx = reconstructor->_transformations[inputIndex].GetTranslationX();
                        double ty = reconstructor->_transformations[inputIndex].GetTranslationY();
                        double tz = reconstructor->_transformations[inputIndex].GetTranslationZ();

                        double rx = reconstructor->_transformations[inputIndex].GetRotationX();
                        double ry = reconstructor->_transformations[inputIndex].GetRotationY();
                        double rz = reconstructor->_transformations[inputIndex].GetRotationZ();

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
    public:
        ReconstructionCardiac4D *reconstructor;

        SliceToVolumeRegistrationCardiac4D(ReconstructionCardiac4D *_reconstructor) :
            reconstructor(_reconstructor) {}

        void operator() (const blocked_range<size_t>& r) const {

            ImageAttributes attr = reconstructor->_reconstructed4D.Attributes();

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {

                if (!reconstructor->_slice_excluded[inputIndex]) {

                    GreyPixel smin, smax;
                    GreyImage target;
                    RealImage slice, w, b, t;


                    ResamplingWithPadding<RealPixel> resampling(attr._dx, attr._dx, attr._dx, -1);
                    Reconstruction dummy_reconstruction;

                    GenericLinearInterpolateImageFunction<RealImage> interpolator;

                    // TARGET
                    // get current slice
                    t = reconstructor->_slices[inputIndex];
                    // resample to spatial resolution of reconstructed volume
                    resampling.Input(&reconstructor->_slices[inputIndex]);
                    resampling.Output(&t);
                    resampling.Interpolator(&interpolator);
                    resampling.Run();
                    target = t;

                    target = reconstructor->_slices[inputIndex];

                    // get pixel value min and max
                    target.GetMinMax(&smin, &smax);

                    // SOURCE
                    if (smax > 0 && (smax - smin) > 1) {

                        ParameterList params;
                        Insert(params, "Transformation model", "Rigid");
                        Insert(params, "Image (dis-)similarity measure", "NMI");
                        if (reconstructor->_nmi_bins > 0)
                            Insert(params, "No. of bins", reconstructor->_nmi_bins);

                        // Insert(params, "Image interpolation mode", "Linear");

                        Insert(params, "Background value for image 1", 0); // -1
                        Insert(params, "Background value for image 2", -1);

                        // Insert(params, "Image (dis-)similarity measure", "NCC");
                        // Insert(params, "Image interpolation mode", "Linear");
                        // string type = "sigma";
                        // string units = "mm";
                        // double width = 0;
                        // Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);

                        GenericRegistrationFilter *registration = new GenericRegistrationFilter();
                        registration->Parameter(params);

                        // put origin to zero
                        RigidTransformation offset;
                        dummy_reconstruction.ResetOrigin(target, offset);
                        Matrix mo = offset.GetMatrix();
                        Matrix m = reconstructor->_transformations[inputIndex].GetMatrix();
                        m = m * mo;
                        reconstructor->_transformations[inputIndex].PutMatrix(m);

                        // TODO: extract nearest cardiac phase from reconstructed 4D to use as source
                        GreyImage source = reconstructor->_reconstructed4D.GetRegion(0, 0, 0, reconstructor->_slice_svr_card_index[inputIndex], attr._x, attr._y, attr._z, reconstructor->_slice_svr_card_index[inputIndex] + 1);

                        registration->Input(&target, &source);
                        Transformation *dofout = nullptr;
                        registration->Output(&dofout);

                        RigidTransformation tmp_dofin = reconstructor->_transformations[inputIndex];
                        registration->InitialGuess(&tmp_dofin);

                        registration->GuessParameter();
                        registration->Run();

                        RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*> (dofout);
                        reconstructor->_transformations[inputIndex] = *rigidTransf;

                        //undo the offset
                        mo.Invert();
                        m = reconstructor->_transformations[inputIndex].GetMatrix();
                        m = m * mo;
                        reconstructor->_transformations[inputIndex].PutMatrix(m);
                    }
                }
            }
        }

        void operator() () const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    // class for FFD SVR
    class SliceToVolumeRegistrationFFD {
        Reconstruction *reconstructor;

    public:
        SliceToVolumeRegistrationFFD(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                GreyPixel smin, smax;
                const GreyImage& target = reconstructor->_grey_slices[inputIndex];
                target.GetMinMax(&smin, &smax);

                if (smax > 1 && (smax - smin) > 1) {
                    // define registration model
                    ParameterList params;
                    Insert(params, "Transformation model", "FFD");

                    if (!reconstructor->_ncc_reg) {
                        Insert(params, "Image (dis-)similarity measure", "NMI");
                        if (reconstructor->_nmi_bins > 0)
                            Insert(params, "No. of bins", reconstructor->_nmi_bins);
                    } else {
                        Insert(params, "Image (dis-)similarity measure", "NCC");
                        string type = "sigma";
                        string units = "mm";
                        double width = 0;
                        Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);
                    }

                    GreyPixel tmin, tmax;
                    reconstructor->_grey_reconstructed.GetMinMax(&tmin, &tmax);
                    if (smin < 1)
                        Insert(params, "Background value for image 1", -1);

                    if (tmin < 0)
                        Insert(params, "Background value for image 2", -1);
                    else if (tmin < 1)
                        Insert(params, "Background value for image 2", 0);

                    if (reconstructor->_cp_spacing > 0) {
                        int cp_spacing = reconstructor->_cp_spacing;
                        Insert(params, "Control point spacing in X", cp_spacing);
                        Insert(params, "Control point spacing in Y", cp_spacing);
                        Insert(params, "Control point spacing in Z", cp_spacing);
                    }

                    // run registration
                    unique_ptr<GenericRegistrationFilter> registration(new GenericRegistrationFilter());
                    registration->Parameter(params);
                    registration->Input(&target, &reconstructor->_reconstructed);
                    Transformation *dofout = nullptr;
                    registration->Output(&dofout);
                    registration->InitialGuess(&(reconstructor->_mffd_transformations[inputIndex]));
                    registration->GuessParameter();
                    registration->Run();

                    // read output transformation
                    MultiLevelFreeFormTransformation *mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*>(dofout);
                    reconstructor->_mffd_transformations[inputIndex] = *mffd_dofout;
                }
            }
        }

        void operator()() const {
            task_scheduler_init init(8);
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
            init.terminate();
        }
    };

    //-------------------------------------------------------------------

    // class for remote rigid SVR
    class RemoteSliceToVolumeRegistration {
        Reconstruction *reconstructor;
        int svr_range_start;
        int svr_range_stop;
        string str_mirtk_path;
        string str_current_main_file_path;
        string str_current_exchange_file_path;
        bool rigid;
        string register_cmd;

    public:
        RemoteSliceToVolumeRegistration(
            Reconstruction *reconstructor,
            int svr_range_start,
            int svr_range_stop,
            string str_mirtk_path,
            string str_current_main_file_path,
            string str_current_exchange_file_path,
            bool rigid = true) :
            reconstructor(reconstructor),
            svr_range_start(svr_range_start),
            svr_range_stop(svr_range_stop),
            str_mirtk_path(str_mirtk_path),
            str_current_main_file_path(str_current_main_file_path),
            str_current_exchange_file_path(str_current_exchange_file_path),
            rigid(rigid) {
            // Preallocate memory area
            register_cmd.reserve(1500);
            // Define registration parameters
            const string file_prefix = str_current_exchange_file_path + (rigid ? "/res-" : "/");
            register_cmd = str_mirtk_path + "/register ";
            register_cmd += file_prefix + "slice-%1%.nii.gz ";  // Target
            register_cmd += str_current_exchange_file_path + "/current-source.nii.gz";  // Source
            register_cmd += string(" -model ") + (rigid ? "Rigid" : "FFD");
            register_cmd += " -bg1 0 -bg2 -1";
            register_cmd += " -dofin " + file_prefix + "transformation-%1%.dof";
            register_cmd += " -dofout " + file_prefix + "transformation-%1%.dof";
            if (reconstructor->_ncc_reg)
                register_cmd += string(" -sim NCC -window ") + (rigid ? "0" : "5");
            if (!rigid && reconstructor->_cp_spacing > 0)
                register_cmd += " -ds " + to_string(reconstructor->_cp_spacing);
            register_cmd += " -threads 1";  // Remove parallelisation of the register command
            register_cmd += " > " + str_current_exchange_file_path + "/log%1%.txt"; // Log
        }

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                if (reconstructor->_zero_slices[inputIndex] > 0) {
                    if (system((boost::format(register_cmd) % inputIndex).str().c_str()) == -1)
                        cerr << "The register command couldn't be executed!" << endl;
                }
            } // end of inputIndex
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(svr_range_start, svr_range_stop), *this);
        }
    };

    //-------------------------------------------------------------------

    class RemoteSliceToVolumeRegistrationCardiac4D {
    public:
        ReconstructionCardiac4D *reconstructor;
        int svr_range_start;
        int svr_range_stop;
        string str_mirtk_path;
        string str_current_main_file_path;
        string str_current_exchange_file_path;

        RemoteSliceToVolumeRegistrationCardiac4D(ReconstructionCardiac4D *in_reconstructor, int in_svr_range_start, int in_svr_range_stop, const string& in_str_mirtk_path, const string& in_str_current_main_file_path, const string& in_str_current_exchange_file_path) :
            reconstructor(in_reconstructor),
            svr_range_start(in_svr_range_start),
            svr_range_stop(in_svr_range_stop),
            str_mirtk_path(in_str_mirtk_path),
            str_current_main_file_path(in_str_current_main_file_path),
            str_current_exchange_file_path(in_str_current_exchange_file_path) {}

        void operator() (const blocked_range<size_t>& r) const {

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {

                if (reconstructor->_zero_slices[inputIndex] > 0) {

                    std::thread::id this_id = std::this_thread::get_id();
                    //                cout << inputIndex  << " - " << this_id << " - " << endl;     //


                    string str_target, str_source, str_reg_model, str_ds_setting, str_dofin, str_dofout, str_bg1, str_bg2, str_log;


                    str_target = str_current_exchange_file_path + "/res-slice-" + to_string(inputIndex) + ".nii.gz";
                    str_source = str_current_exchange_file_path + "/current-source-" + to_string(reconstructor->_slice_svr_card_index[inputIndex]) + ".nii.gz";
                    str_log = " > " + str_current_exchange_file_path + "/log" + to_string(inputIndex) + ".txt";


                    str_reg_model = "-model Rigid";

                    str_dofout = "-dofout " + str_current_exchange_file_path + "/res-transformation-" + to_string(inputIndex) + ".dof";

                    str_bg1 = "-bg1 " + to_string(0);

                    str_bg2 = "-bg2 " + to_string(-1);

                    str_dofin = "-dofin " + str_current_exchange_file_path + "/res-transformation-" + to_string(inputIndex) + ".dof";

                    string register_cmd = str_mirtk_path + "/register " + str_target + " " + str_source + " " + str_reg_model + " " + str_bg1 + " " + str_bg2 + " " + str_dofin + " " + str_dofout + " " + str_log;

                    int tmp_log = system(register_cmd.c_str());

                }
            } // end of inputIndex
        }

        // execute
        void operator() () const {
            parallel_for(blocked_range<size_t>(svr_range_start, svr_range_stop), *this);
        }
    };

    //-------------------------------------------------------------------

    // class for calculation of transformation matrices
    class CoeffInit {
        Reconstruction *reconstructor;

    public:
        CoeffInit(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            const RealImage global_reconstructed(reconstructor->_grey_reconstructed.Attributes());
            //get resolution of the volume
            double vx, vy, vz;
            global_reconstructed.GetPixelSize(&vx, &vy, &vz);
            //volume is always isotropic
            const double res = vx;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                reconstructor->_volcoeffs[inputIndex] = {};
                reconstructor->_slice_inside[inputIndex] = false;

                const RealImage global_slice(reconstructor->_grey_slices[inputIndex].Attributes());

                //prepare storage variable
                SLICECOEFFS slicecoeffs(global_slice.GetX(), Array<VOXELCOEFFS>(global_slice.GetY()));

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
                RealImage PSF(attr);

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
                RealImage tPSF(attr);
                //calculate centre of tPSF in image coordinates
                const int centre = (dim - 1) / 2;

                //for each voxel in current slice calculate matrix coefficients
                bool excluded_slice = false;
                for (int ff = 0; ff < reconstructor->_force_excluded.size(); ff++) {
                    if (inputIndex == reconstructor->_force_excluded[ff]) {
                        excluded_slice = true;
                        break;
                    }
                }

                // Check if the slice is excluded
                if (excluded_slice || reconstructor->_reg_slice_weight[inputIndex] <= 0)
                    continue;

                for (int i = 0; i < global_slice.GetX(); i++)
                    for (int j = 0; j < global_slice.GetY(); j++)
                        if (reconstructor->_slices[inputIndex](i, j, 0) > -0.01) {
                            //calculate centrepoint of slice voxel in volume space (tx,ty,tz)
                            double x = i;
                            double y = j;
                            double z = 0;
                            global_slice.ImageToWorld(x, y, z);

                            if (!reconstructor->_ffd)
                                reconstructor->_transformations[inputIndex].Transform(x, y, z);
                            else
                                reconstructor->_mffd_transformations[inputIndex].Transform(-1, 1, x, y, z);

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
                                        //Calculate the position of the POINT3D of PSF centered over current slice voxel
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
                                        //transformations included in them (they are nifti)  and those are not reflected in
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
                                            reconstructor->_mffd_transformations[inputIndex].Transform(-1, 1, x, y, z);

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

                reconstructor->_volcoeffs[inputIndex] = move(slicecoeffs);
                reconstructor->_slice_inside[inputIndex] = slice_inside;

            }  //end of loop through the slices
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    // another version of CoeffInit
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
            //get resolution of the volume
            double vx, vy, vz;
            reconstructor->_reconstructed.GetPixelSize(&vx, &vy, &vz);
            //volume is always isotropic
            const double res = vx;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                reconstructor->_volcoeffsSF[inputIndex % reconstructor->_slicePerDyn] = {};
                reconstructor->_slice_insideSF[inputIndex % reconstructor->_slicePerDyn] = false;

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
                RealImage PSF(attr);

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
                RealImage tPSF(attr);
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
                                                                    cerr << "Error while trying to populate tPSF. " << aa << " " << bb << " " << cc << endl;
                                                                    cerr << l << " " << m << " " << n << endl;
                                                                    cerr << tx << " " << ty << " " << tz << endl;
                                                                    cerr << centre << endl;
                                                                    tPSF.Write("tPSF.nii.gz");
                                                                    exit(1);
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

                reconstructor->_volcoeffsSF[inputIndex % reconstructor->_slicePerDyn] = move(slicecoeffs);
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

        void operator() (const blocked_range<size_t>& r) const {
            //get resolution of the volume
            double vx, vy, vz;
            reconstructor->_reconstructed4D.GetPixelSize(&vx, &vy, &vz);
            //volume is always isotropic
            const double res = vx;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                reconstructor->_volcoeffs[inputIndex] = {};
                reconstructor->_slice_inside[inputIndex] = false;

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
                RealImage PSF(attr);

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
                RealImage tPSF(attr);
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
                                                                    cerr << "Error while trying to populate tPSF. " << aa << " " << bb << " " << cc << endl;
                                                                    cerr << l << " " << m << " " << n << endl;
                                                                    cerr << tx << " " << ty << " " << tz << endl;
                                                                    cerr << centre << endl;
                                                                    tPSF.Write("tPSF.nii.gz");
                                                                    exit(1);
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

                reconstructor->_volcoeffs[inputIndex] = move(slicecoeffs);
                reconstructor->_slice_inside[inputIndex] = slice_inside;

            }  //end of loop through the slices
        }

        void operator() () const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class SimulateStacksCardiac4D {
        ReconstructionCardiac4D *reconstructor;

    public:
        SimulateStacksCardiac4D(ReconstructionCardiac4D *reconstructor) : reconstructor(reconstructor) {}

        void operator() (const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                if (!reconstructor->_slice_excluded[inputIndex]) {
                    reconstructor->GetVerboseLog() << inputIndex << " ";

                    //get resolution of the volume
                    double vx, vy, vz;
                    reconstructor->_reconstructed4D.GetPixelSize(&vx, &vy, &vz);
                    //volume is always isotropic
                    double res = vx;
                    //read the slice
                    RealImage& slice = reconstructor->_slices[inputIndex];

                    //prepare structures for storage
                    POINT3D p;
                    VOXELCOEFFS empty;
                    SLICECOEFFS slicecoeffs(slice.GetX(), Array < VOXELCOEFFS >(slice.GetY(), empty));

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
                    RealImage PSF(attr);

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
                                PSF(i, j, k) = exp(
                                    -x * x / (2 * sigmax * sigmax) - y * y / (2 * sigmay * sigmay)
                                    - z * z / (2 * sigmaz * sigmaz));
                                sum += PSF(i, j, k);
                            }
                    PSF /= sum;

                    if (reconstructor->_debug)
                        if (inputIndex == 0)
                            PSF.Write("PSF.nii.gz");

                    //prepare storage for PSF transformed and resampled to the space of reconstructed volume
                    //maximum dim of rotated kernel - the next higher odd integer plus two to accound for rounding error of tx,ty,tz.
                    //Note conversion from PSF image coordinates to tPSF image coordinates *size/res
                    int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2))
                        * 2 + 1 + 2;
                    //prepare image attributes. Voxel dimension will be taken from the reconstructed volume
                    attr._x = dim;
                    attr._y = dim;
                    attr._z = dim;
                    attr._dx = res;
                    attr._dy = res;
                    attr._dz = res;
                    //create matrix from transformed PSF
                    RealImage tPSF(attr);
                    //calculate centre of tPSF in image coordinates
                    int centre = (dim - 1) / 2;

                    //for each voxel in current slice calculate matrix coefficients
                    int ii, jj, kk;
                    int tx, ty, tz;
                    int nx, ny, nz;
                    int l, m, n;
                    double weight;
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
                                for (ii = 0; ii < dim; ii++)
                                    for (jj = 0; jj < dim; jj++)
                                        for (kk = 0; kk < dim; kk++)
                                            tPSF(ii, jj, kk) = 0;

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
                                                                    weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                                                    sum += weight;
                                                                    if (reconstructor->_mask(l, m, n) == 1) {
                                                                        inside = true;
                                                                    }
                                                                }
                                            //if there were no voxels do nothing
                                            if ((sum <= 0) || (!inside))
                                                continue;
                                            //now calculate the transformed PSF
                                            for (l = nx; l <= nx + 1; l++)
                                                if ((l >= 0) && (l < reconstructor->_reconstructed4D.GetX()))
                                                    for (m = ny; m <= ny + 1; m++)
                                                        if ((m >= 0) && (m < reconstructor->_reconstructed4D.GetY()))
                                                            for (n = nz; n <= nz + 1; n++)
                                                                if ((n >= 0) && (n < reconstructor->_reconstructed4D.GetZ())) {
                                                                    weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));

                                                                    //image coordinates in tPSF
                                                                    //(centre,centre,centre) in tPSF is aligned with (tx,ty,tz)
                                                                    int aa, bb, cc;
                                                                    aa = l - tx + centre;
                                                                    bb = m - ty + centre;
                                                                    cc = n - tz + centre;

                                                                    //resulting value
                                                                    double value = PSF(ii, jj, kk) * weight / sum;

                                                                    //Check that we are in tPSF
                                                                    if ((aa < 0) || (aa >= dim) || (bb < 0) || (bb >= dim) || (cc < 0)
                                                                        || (cc >= dim)) {
                                                                        cerr << "Error while trying to populate tPSF. " << aa << " " << bb
                                                                            << " " << cc << endl;
                                                                        cerr << l << " " << m << " " << n << endl;
                                                                        cerr << tx << " " << ty << " " << tz << endl;
                                                                        cerr << centre << endl;
                                                                        tPSF.Write("tPSF.nii.gz");
                                                                        exit(1);
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
                }  // if(!_slice_excluded[inputIndex])
            }  //end of loop through the slices
        }

        void operator() () const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    // class for EStep (RS)
    class EStep {
        Reconstruction *reconstructor;
        Array<double>& slice_potential;

    public:
        EStep(Reconstruction *reconstructor,
            Array<double>& slice_potential) :
            reconstructor(reconstructor),
            slice_potential(slice_potential) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];

                //read current weight image
                memset(reconstructor->_weights[inputIndex].Data(), 0, sizeof(RealPixel) * reconstructor->_weights[inputIndex].NumberOfVoxels());

                //alias the current bias image
                const RealImage& b = reconstructor->_bias[inputIndex];

                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];

                double num = 0;
                //Calculate error, voxel weights, and slice potential
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

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

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    // class for slice scale calculation
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
                                double eb = exp(-b(i, j, 0));
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

    // class for slice bias calculation
    class Bias {
        Reconstruction *reconstructor;

    public:
        Bias(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];

                //alias the current weight image
                const RealImage& w = reconstructor->_weights[inputIndex];

                //alias the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];

                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];

                //prepare weight image for bias field
                RealImage wb = w;

                RealImage wresidual(slice.Attributes());

                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
                                //bias-correct and scale current slice
                                const double eb = exp(-b(i, j, 0));
                                slice(i, j, 0) *= eb * scale;

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
                GaussianBlurring<RealPixel> gb(reconstructor->_sigma_bias);
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

    // class for SR reconstruction
    class Superresolution {
        Reconstruction *reconstructor;

    public:
        RealImage confidence_map;
        RealImage addon;

        Superresolution(Reconstruction *reconstructor) : reconstructor(reconstructor) {
            //Clear addon
            addon.Initialize(reconstructor->_reconstructed.Attributes());

            //Clear confidence map
            confidence_map.Initialize(reconstructor->_reconstructed.Attributes());
        }

        Superresolution(Superresolution& x, split) : Superresolution(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            //Update reconstructed volume using current slice
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                //Distribute error to the volume
                for (int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++)
                    for (int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++)
                        if (reconstructor->_slices[inputIndex](i, j, 0) > -0.01) {
                            if (reconstructor->_simulated_slices[inputIndex](i, j, 0) < 0.01)
                                reconstructor->_slice_dif[inputIndex](i, j, 0) = 0;

                            #pragma omp simd
                            for (int k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                const auto multiplier = reconstructor->_robust_slices_only ? 1 : reconstructor->_weights[inputIndex](i, j, 0);
                                addon(p.x, p.y, p.z) += multiplier * p.value * reconstructor->_slice_weight[inputIndex] * reconstructor->_slice_dif[inputIndex](i, j, 0);
                                confidence_map(p.x, p.y, p.z) += multiplier * p.value * reconstructor->_slice_weight[inputIndex];
                            }
                        }
            } //end of loop for a slice inputIndex
        }

        void join(const Superresolution& y) {
            addon += y.addon;
            confidence_map += y.confidence_map;
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
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];

                //read the current weight image
                RealImage& w = reconstructor->_weights[inputIndex];

                //read the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];

                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];

                //Update reconstructed volume using current slice

                //Distribute error to the volume
                POINT3D p;
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                            if (reconstructor->_simulated_slices[inputIndex](i, j, 0) > 0)
                                slice(i, j, 0) -= reconstructor->_simulated_slices[inputIndex](i, j, 0);
                            else
                                slice(i, j, 0) = 0;

                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for (int k = 0; k < n; k++) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                for (int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++) {
                                    if (reconstructor->_robust_slices_only) {
                                        addon(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        confidence_map(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_slice_weight[inputIndex];

                                    } else {
                                        addon(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        confidence_map(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                    }
                                }
                            }
                        }
            } //end of loop for a slice inputIndex
        }

        void join(const SuperresolutionCardiac4D& y) {
            addon += y.addon;
            confidence_map += y.confidence_map;
        }

        void operator() () {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    // class for MStep (RS)
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
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];

                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];

                //calculate error
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-reconstructor->_bias[inputIndex](i, j, 0)) * scale;

                            //otherwise the error has no meaning - it is equal to slice intensity
                            if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0.99) {
                                slice(i, j, 0) -= reconstructor->_simulated_slices[inputIndex](i, j, 0);

                                //sigma and mix
                                double e = slice(i, j, 0);
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

    // class for adaptive regularisation (Part I)
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
            for (size_t i = r.begin(); i != r.end(); ++i) {
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

    // class for adaptive regularisation (Part II)
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

    class AdaptiveRegularization1Cardiac4D {
        const ReconstructionCardiac4D *reconstructor;
        Array<RealImage>& b;
        const Array<double>& factor;
        const RealImage& original;

    public:
        AdaptiveRegularization1Cardiac4D(
            const ReconstructionCardiac4D *_reconstructor,
            Array<RealImage>& _b,
            const Array<double>& _factor,
            const RealImage& _original) :
            reconstructor(_reconstructor),
            b(_b),
            factor(_factor),
            original(_original) {}

        void operator() (const blocked_range<size_t>& r) const {
            int dx = reconstructor->_reconstructed4D.GetX();
            int dy = reconstructor->_reconstructed4D.GetY();
            int dz = reconstructor->_reconstructed4D.GetZ();
            int dt = reconstructor->_reconstructed4D.GetT();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                //b[i] = reconstructor->_reconstructed;
                // b[i].Initialize( reconstructor->_reconstructed.Attributes() );

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

                                } else
                                    b[i](x, y, z, t) = 0;
                            }
                        }
            }
        }

        void operator() () const {
            parallel_for(blocked_range<size_t>(0, 13), *this);
        }
    };

    //-------------------------------------------------------------------

    class AdaptiveRegularization2Cardiac4D {
        ReconstructionCardiac4D *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;

    public:
        AdaptiveRegularization2Cardiac4D(ReconstructionCardiac4D *_reconstructor,
            Array<RealImage> &_b,
            Array<double> &_factor,
            RealImage &_original) :
            reconstructor(_reconstructor),
            b(_b),
            factor(_factor),
            original(_original) {}

        void operator() (const blocked_range<size_t>& r) const {
            int dx = reconstructor->_reconstructed4D.GetX();
            int dy = reconstructor->_reconstructed4D.GetY();
            int dz = reconstructor->_reconstructed4D.GetZ();
            int dt = reconstructor->_reconstructed4D.GetT();
            for (size_t x = r.begin(); x != r.end(); ++x) {
                int xx, yy, zz;
                for (int y = 0; y < dy; y++)
                    for (int z = 0; z < dz; z++)
                        for (int t = 0; t < dt; t++) {
                            if (reconstructor->_confidence_map(x, y, z, t) > 0) {
                                double val = 0;
                                double sum = 0;
                                for (int i = 0; i < 13; i++) {
                                    xx = x + reconstructor->_directions[i][0];
                                    yy = y + reconstructor->_directions[i][1];
                                    zz = z + reconstructor->_directions[i][2];
                                    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                                        if (reconstructor->_confidence_map(xx, yy, zz, t) > 0) {
                                            val += b[i](x, y, z, t) * original(xx, yy, zz, t);
                                            sum += b[i](x, y, z, t);
                                        }
                                }

                                for (int i = 0; i < 13; i++) {
                                    xx = x - reconstructor->_directions[i][0];
                                    yy = y - reconstructor->_directions[i][1];
                                    zz = z - reconstructor->_directions[i][2];
                                    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                                        if (reconstructor->_confidence_map(xx, yy, zz, t) > 0) {
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

        void operator() () const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_reconstructed4D.GetX()), *this);
        }

    };

    //-------------------------------------------------------------------

    // class for bias normalisation
    class NormaliseBias {
        Reconstruction *reconstructor;
    public:
        RealImage bias;

        NormaliseBias(Reconstruction *reconstructor) : reconstructor(reconstructor) {
            bias.Initialize(reconstructor->_reconstructed.Attributes());
        }

        NormaliseBias(NormaliseBias& x, split) : NormaliseBias(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                if (reconstructor->_verbose)
                    reconstructor->_verbose_log << inputIndex << " ";

                //read the current bias image
                RealImage b = reconstructor->_bias[inputIndex];

                //read current scale factor
                const double scale = reconstructor->_scale[inputIndex];

                const RealPixel *pi = reconstructor->_slices[inputIndex].Data();
                RealPixel *pb = b.Data();
                for (int i = 0; i < reconstructor->_slices[inputIndex].NumberOfVoxels(); i++) {
                    if (pi[i] > -1 && scale > 0)
                        pb[i] -= log(scale);
                }

                //Distribute slice intensities to the volume
                for (int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++)
                    for (int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++)
                        if (reconstructor->_slices[inputIndex](i, j, 0) > -0.01) {
                            //number of volume voxels with non-zero coefficients for current slice voxel
                            const size_t n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            //add contribution of current slice voxel to all voxel volumes to which it contributes
                            for (size_t k = 0; k < n; k++) {
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
        ReconstructionCardiac4D* reconstructor;

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
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {

                if (reconstructor->_verbose)
                    reconstructor->_verbose_log << inputIndex << " ";

                // alias the current slice
                RealImage& slice = reconstructor->_slices[inputIndex];

                //read the current bias image
                RealImage b = reconstructor->_bias[inputIndex];

                //read current scale factor
                double scale = reconstructor->_scale[inputIndex];

                RealPixel *pi = slice.Data();
                RealPixel *pb = b.Data();
                for (int i = 0; i < slice.NumberOfVoxels(); i++) {
                    if ((*pi > -1) && (scale > 0))
                        *pb -= log(scale);
                    pb++;
                    pi++;
                }

                //Distribute slice intensities to the volume
                POINT3D p;
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //number of volume voxels with non-zero coefficients for current slice voxel
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            //add contribution of current slice voxel to all voxel volumes
                            //to which it contributes
                            for (int k = 0; k < n; k++) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
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

        void operator() () {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    // class for global stack similarity statists (for the stack selection function)
    class GlobalSimilarityStats {
        Reconstruction *reconstructor;
        int nStacks;
        const Array<RealImage>& stacks;
        const Array<RealImage>& masks;
        Array<double>& all_global_ncc_array;
        Array<double>& all_global_volume_array;

    public:
        GlobalSimilarityStats(
            Reconstruction *reconstructor,
            int nStacks,
            const Array<RealImage>& stacks,
            const Array<RealImage>& masks,
            Array<double>& all_global_ncc_array,
            Array<double>& all_global_volume_array) :
            reconstructor(reconstructor),
            nStacks(nStacks),
            stacks(stacks),
            masks(masks),
            all_global_ncc_array(all_global_ncc_array),
            all_global_volume_array(all_global_volume_array) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                double average_ncc = 0;
                double average_volume = 0;
                Array<RigidTransformation> current_stack_tranformations;

                cout << " - " << inputIndex << endl;
                reconstructor->GlobalStackStats(stacks[inputIndex], masks[inputIndex], stacks, masks, average_ncc, average_volume, current_stack_tranformations);
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
        CalculateCorrectedSlices(ReconstructionCardiac4D *_reconstructor) : reconstructor(_reconstructor) {}

        void operator() (const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                //read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];

                //alias the current bias image
                const RealImage& b = reconstructor->_bias[inputIndex];

                //identify scale factor
                const double scale = reconstructor->_scale[inputIndex];

                //calculate corrected voxels
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1)
                            //bias correct and scale the voxel
                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                reconstructor->_corrected_slices[inputIndex] = move(slice);
            } //end of loop for a slice inputIndex
        }

        void operator() () const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class CalculateError {
        ReconstructionCardiac4D *reconstructor;

    public:
        CalculateError(ReconstructionCardiac4D *reconstructor) : reconstructor(reconstructor) {}

        void operator() (const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                //initialise
                reconstructor->_error[inputIndex].Initialize(reconstructor->_slices[inputIndex].Attributes());

                //read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];

                //alias the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];

                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];

                //calculate error
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1)
                            if (reconstructor->_simulated_weights[inputIndex](i, j, 0) > 0) {
                                //bias correct and scale the voxel
                                slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                                //subtract simulated voxel
                                slice(i, j, 0) -= reconstructor->_simulated_slices[inputIndex](i, j, 0);
                                //assign as error
                                reconstructor->_error[inputIndex](i, j, 0) = slice(i, j, 0);
                            }
            } //end of loop for a slice inputIndex
        }

        void operator() () const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------
}
