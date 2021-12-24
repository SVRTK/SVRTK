/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2008-2017 Imperial College London
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
#include "svrtk/Reconstruction.h"
#include "svrtk/Profiling.h"
#include "svrtk/Parallel.h"

namespace svrtk {

    Reconstruction::Reconstruction() {
        _number_of_slices_org = 0;
        _average_thickness_org = 0;
        _cp_spacing = -1;

        _step = 0.0001;
        _debug = false;
        _verbose = false;
        _quality_factor = 2;
        _sigma_bias = 12;
        _sigma_s = 0.025;
        _sigma_s2 = 0.025;
        _mix_s = 0.9;
        _mix = 0.9;
        _delta = 1;
        _lambda = 0.1;
        _alpha = (0.05 / _lambda) * _delta * _delta;
        _low_intensity_cutoff = 0.01;
        _nmi_bins = -1;
        _global_NCC_threshold = 0.65;

        _template_created = false;
        _have_mask = false;
        _global_bias_correction = false;
        _adaptive = false;
        _robust_slices_only = false;
        _recon_type = _3D;
        _ffd = false;
        _blurring = false;
        _structural = false;
        _ncc_reg = false;
        _template_flag = false;
        _no_sr = false;
        _reg_log = false;
        _masked_stacks = false;
        _filtered_cmp_flag = false;
        _bg_flag = false;
    }

    //-------------------------------------------------------------------

    // Create average of all stacks based on the input transformations
    RealImage Reconstruction::CreateAverage(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations) {
        if (!_template_created)
            throw runtime_error("Please create the template before calculating the average of the stacks.");

        SVRTK_START_TIMING();
        const RealImage average = Utility::CreateAverage(this, stacks, stack_transformations);
        SVRTK_END_TIMING("CreateAverage");

        return average;
    }

    //-------------------------------------------------------------------

    // Set given template with specific resolution
    double Reconstruction::CreateTemplate(const RealImage& stack, double resolution) {
        double dx, dy, dz, d;

        // //Get image attributes - image size and voxel size
        // ImageAttributes attr = stack.Attributes();

        // //enlarge stack in z-direction in case top of the head is cut off
        // attr._z += 2;

        // //create enlarged image
        // RealImage enlarged(attr);

        //determine resolution of volume to reconstruct
        if (resolution <= 0) {
            //resolution was not given by user set it to min of res in x or y direction
            stack.GetPixelSize(&dx, &dy, &dz);
            if ((dx <= dy) && (dx <= dz))
                d = dx;
            else if (dy <= dz)
                d = dy;
            else
                d = dz;
        } else
            d = resolution;

        cout << "Reconstructed volume voxel size : " << d << " mm" << endl;

        RealImage temp(stack.Attributes());
        RealPixel smin, smax;
        stack.GetMinMax(&smin, &smax);

        // interpolate the input stack to the given resolution
        if (smin < -0.1) {
            GenericLinearInterpolateImageFunction<RealImage> interpolator;
            ResamplingWithPadding<RealPixel> resampler(d, d, d, -1);
            resampler.Input(&stack);
            resampler.Output(&temp);
            resampler.Interpolator(&interpolator);
            resampler.Run();
        } else if (smin < 0.1) {
            GenericLinearInterpolateImageFunction<RealImage> interpolator;
            ResamplingWithPadding<RealPixel> resampler(d, d, d, 0);
            resampler.Input(&stack);
            resampler.Output(&temp);
            resampler.Interpolator(&interpolator);
            resampler.Run();
        } else {
            //resample "temp" to resolution "d"
            unique_ptr<InterpolateImageFunction> interpolator(InterpolateImageFunction::New(Interpolation_Linear));
            Resampling<RealPixel> resampler(d, d, d);
            resampler.Input(&stack);
            resampler.Output(&temp);
            resampler.Interpolator(interpolator.get());
            resampler.Run();
        }

        //initialise reconstructed volume
        _reconstructed = move(temp);
        _template_created = true;
        _grey_reconstructed = _reconstructed;
        _attr_reconstructed = _reconstructed.Attributes();

        if (_debug)
            _reconstructed.Write("template.nii.gz");

        //return resulting resolution of the template image
        return d;
    }

    //-------------------------------------------------------------------

    // Create anisotropic template
    double Reconstruction::CreateTemplateAniso(const RealImage& stack) {
        ImageAttributes attr = stack.Attributes();
        attr._t = 1;
        RealImage temp(attr);

        cout << "Constructing volume with anisotropic voxel size " << attr._x << " " << attr._y << " " << attr._z << endl;

        //initialize reconstructed volume
        _reconstructed = move(temp);
        _template_created = true;

        //return resulting resolution of the template image
        return attr._x;
    }

    //-------------------------------------------------------------------

    // Set template
    void Reconstruction::SetTemplate(RealImage templateImage) {
        RealImage t2template(_reconstructed.Attributes());
        RigidTransformation tr;
        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        ImageTransformation imagetransformation;

        imagetransformation.Input(&templateImage);
        imagetransformation.Transformation(&tr);
        imagetransformation.Output(&t2template);
        //target contains zeros, need padding -1
        imagetransformation.TargetPaddingValue(-1);
        //need to fill voxels in target where there is no info from source with zeroes
        imagetransformation.SourcePaddingValue(0);
        imagetransformation.Interpolator(&interpolator);
        imagetransformation.Run();

        _reconstructed = move(t2template);
    }

    //-------------------------------------------------------------------

    // Set given mask to the reconstruction object
    void Reconstruction::SetMask(RealImage *mask, double sigma, double threshold) {
        if (!_template_created)
            throw runtime_error("Please create the template before setting the mask, so that the mask can be resampled to the correct dimensions.");

        _mask.Initialize(_reconstructed.Attributes());

        if (mask != NULL) {
            //if sigma is nonzero first smooth the mask
            if (sigma > 0) {
                //blur mask
                GaussianBlurring<RealPixel> gb(sigma);

                gb.Input(mask);
                gb.Output(mask);
                gb.Run();

                //binarise mask
                *mask = CreateMask(*mask, threshold);
            }

            //resample the mask according to the template volume using identity transformation
            RigidTransformation transformation;
            ImageTransformation imagetransformation;

            //GenericNearestNeighborExtrapolateImageFunction<RealImage> interpolator;
            InterpolationMode interpolation = Interpolation_NN;
            unique_ptr<InterpolateImageFunction> interpolator(InterpolateImageFunction::New(interpolation));

            imagetransformation.Input(mask);
            imagetransformation.Transformation(&transformation);
            imagetransformation.Output(&_mask);
            //target is zero image, need padding -1
            imagetransformation.TargetPaddingValue(-1);
            //need to fill voxels in target where there is no info from source with zeroes
            imagetransformation.SourcePaddingValue(0);
            imagetransformation.Interpolator(interpolator.get());
            imagetransformation.Run();
        } else {
            //fill the mask with ones
            _mask = 1;
        }
        //set flag that mask was created
        _have_mask = true;

        // compute mask volume
        double vol = 0;
        RealPixel *pm = _mask.Data();
        #pragma omp parallel for reduction(+: vol)
        for (int i = 0; i < _mask.NumberOfVoxels(); i++)
            if (pm[i] > 0.1)
                vol++;

        vol *= _reconstructed.GetXSize() * _reconstructed.GetYSize() * _reconstructed.GetZSize() / 1000;

        cout << "ROI volume : " << vol << " cc " << endl;

        if (_debug)
            _mask.Write("mask.nii.gz");
    }

    //-------------------------------------------------------------------

    // generate reconstruction quality report / metrics
    void Reconstruction::ReconQualityReport(double& out_ncc, double& out_nrmse, double& average_weight, double& ratio_excluded) {
        Parallel::QualityReport parallelQualityReport(this);
        parallelQualityReport();

        average_weight = _average_volume_weight;
        out_ncc = parallelQualityReport.out_global_ncc / _slices.size();
        out_nrmse = parallelQualityReport.out_global_nrmse / _slices.size();

        if (!isfinite(out_nrmse))
            out_nrmse = 0;

        if (!isfinite(out_ncc))
            out_ncc = 0;

        size_t count_excluded = 0;
        for (size_t i = 0; i < _slices.size(); i++)
            if (_slice_weight[i] < 0.5)
                count_excluded++;

        ratio_excluded = (double)count_excluded / _slices.size();
    }

    //-------------------------------------------------------------------

    // run global stack registration to the template
    void Reconstruction::StackRegistrations(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber) {
        SVRTK_START_TIMING();

        InvertStackTransformations(stack_transformations);

        // check whether to use the global template or the selected stack
        RealImage target = _template_flag ? _reconstructed : stacks[templateNumber];

        RealImage m_tmp = _mask;
        TransformMask(target, m_tmp, RigidTransformation());
        target *= m_tmp;
        target.Write("masked.nii.gz");

        if (_debug) {
            target.Write("target.nii.gz");
            stacks[0].Write("stack0.nii.gz");
        }

        RigidTransformation offset;
        ResetOrigin(target, offset);

        //register all stacks to the target
        Parallel::StackRegistrations p_reg(this, stacks, stack_transformations, templateNumber, target, offset);
        p_reg();

        InvertStackTransformations(stack_transformations);

        SVRTK_END_TIMING("StackRegistrations");
    }

    //-------------------------------------------------------------------

    // restore the original slice intensities
    void Reconstruction::RestoreSliceIntensities() {
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            //calculate scaling factor
            const double& factor = _stack_factor[_stack_index[inputIndex]];//_average_value;

            // read the pointer to current slice
            RealPixel *p = _slices[inputIndex].Data();
            for (int i = 0; i < _slices[inputIndex].NumberOfVoxels(); i++)
                if (p[i] > 0)
                    p[i] /= factor;
        }
    }

    //-------------------------------------------------------------------

    // scale the reconstructed volume
    void Reconstruction::ScaleVolume(RealImage& reconstructed) {
        double scalenum = 0, scaleden = 0;

        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            // alias for the current slice
            const RealImage& slice = _slices[inputIndex];

            //alias for the current weight image
            const RealImage& w = _weights[inputIndex];

            // alias for the current simulated slice
            const RealImage& sim = _simulated_slices[inputIndex];

            for (int i = 0; i < slice.GetX(); i++)
                for (int j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {
                        //scale - intensity matching
                        if (_simulated_weights[inputIndex](i, j, 0) > 0.99) {
                            scalenum += w(i, j, 0) * _slice_weight[inputIndex] * slice(i, j, 0) * sim(i, j, 0);
                            scaleden += w(i, j, 0) * _slice_weight[inputIndex] * sim(i, j, 0) * sim(i, j, 0);
                        }
                    }

        } //end of loop for a slice inputIndex

        //calculate scale for the volume
        double scale = 1;

        if (scaleden > 0)
            scale = scalenum / scaleden;

        if (_verbose)
            _verbose_log << "scale : " << scale << endl;

        RealPixel *ptr = reconstructed.Data();
        #pragma omp parallel for
        for (int i = 0; i < reconstructed.NumberOfVoxels(); i++)
            if (ptr[i] > 0)
                ptr[i] *= scale;
    }

    //-------------------------------------------------------------------

    // run simulation of slices from the reconstruction volume
    void Reconstruction::SimulateSlices() {
        SVRTK_START_TIMING();
        Parallel::SimulateSlices p_sim(this);
        p_sim();
        SVRTK_END_TIMING("SimulateSlices");
    }

    //-------------------------------------------------------------------

    // simulate stacks from the reconstructed volume
    void Reconstruction::SimulateStacks(Array<RealImage>& stacks) {
        RealImage sim;
        int z = -1;//this is the z coordinate of the stack
        int current_stack = -1; //we need to know when to start a new stack

        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            // read the current slice
            const RealImage& slice = _slices[inputIndex];

            //Calculate simulated slice
            sim.Initialize(slice.Attributes());

            //do not simulate excluded slice
            if (_slice_weight[inputIndex] > 0.5) {
                #pragma omp parallel for
                for (size_t i = 0; i < _volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < _volcoeffs[inputIndex][i].size(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            double weight = 0;
                            for (size_t k = 0; k < _volcoeffs[inputIndex][i][j].size(); k++) {
                                const POINT3D& p = _volcoeffs[inputIndex][i][j][k];
                                sim(i, j, 0) += p.value * _reconstructed(p.x, p.y, p.z);
                                weight += p.value;
                            }
                            if (weight > 0.98)
                                sim(i, j, 0) /= weight;
                            else
                                sim(i, j, 0) = 0;
                        }
            }

            if (_stack_index[inputIndex] == current_stack)
                z++;
            else {
                current_stack = _stack_index[inputIndex];
                z = 0;
            }

            #pragma omp parallel for
            for (int i = 0; i < sim.GetX(); i++)
                for (int j = 0; j < sim.GetY(); j++)
                    stacks[_stack_index[inputIndex]](i, j, z) = sim(i, j, 0);
        }
    }

    //-------------------------------------------------------------------

    // match stack intensities with respect to the average
    void Reconstruction::MatchStackIntensities(Array<RealImage>& stacks, const Array<RigidTransformation>& stack_transformations, double averageValue, bool together) {
        //Calculate the averages of intensities for all stacks
        Array<double> stack_average;
        stack_average.reserve(stacks.size());

        //remember the set average value
        _average_value = averageValue;

        //averages need to be calculated only in ROI
        for (size_t ind = 0; ind < stacks.size(); ind++) {
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
                        if (stacks[ind](i, j, k) > 0) {
                            sum += stacks[ind](i, j, k);
                            num++;
                        }
                    }
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

        if (_verbose) {
            _verbose_log << "Stack average intensities are ";
            for (size_t ind = 0; ind < stack_average.size(); ind++)
                _verbose_log << stack_average[ind] << " ";
            _verbose_log << endl;
            _verbose_log << "The new average value is " << averageValue << endl;
        }

        //Rescale stacks
        ClearAndReserve(_stack_factor, stacks.size());
        for (size_t ind = 0; ind < stacks.size(); ind++) {
            const double factor = averageValue / (together ? global_average : stack_average[ind]);
            _stack_factor.push_back(factor);

            RealPixel *ptr = stacks[ind].Data();
            #pragma omp parallel for
            for (int i = 0; i < stacks[ind].NumberOfVoxels(); i++)
                if (ptr[i] > 0)
                    ptr[i] *= factor;
        }

        if (_debug) {
            #pragma omp parallel for
            for (size_t ind = 0; ind < stacks.size(); ind++)
                stacks[ind].Write((boost::format("rescaled-stack%1%.nii.gz") % ind).str().c_str());
        }

        if (_verbose) {
            _verbose_log << "Slice intensity factors are ";
            for (size_t ind = 0; ind < _stack_factor.size(); ind++)
                _verbose_log << _stack_factor[ind] << " ";
            _verbose_log << endl;
            _verbose_log << "The new average value is " << averageValue << endl;
        }
    }

    //-------------------------------------------------------------------

    // evaluate reconstruction quality (NRMSE)
    double Reconstruction::EvaluateReconQuality(int stackIndex) {
        Array<double> rmse_values(_slices.size());
        Array<int> rmse_numbers(_slices.size());
        RealImage nt;

        // compute NRMSE between the simulated and original slice for non-zero voxels
        #pragma omp parallel for private(nt)
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            nt = _slices[inputIndex];
            const RealImage& ns = _simulated_slices[inputIndex];
            double s_diff = 0;
            double s_t = 0;
            int s_n = 0;
            for (int x = 0; x < nt.GetX(); x++) {
                for (int y = 0; y < nt.GetY(); y++) {
                    if (nt(x, y, 0) > 0 && ns(x, y, 0) > 0) {
                        nt(x, y, 0) *= exp(-(_bias[inputIndex])(x, y, 0)) * _scale[inputIndex];
                        s_t += nt(x, y, 0);
                        s_diff += (nt(x, y, 0) - ns(x, y, 0)) * (nt(x, y, 0) - ns(x, y, 0));
                        s_n++;
                    }
                }
            }
            const double nrmse = s_n > 0 ? sqrt(s_diff / s_n) / (s_t / s_n) : 0;
            rmse_values[inputIndex] = nrmse;
            if (nrmse > 0)
                rmse_numbers[inputIndex] = 1;
        }

        double rmse_total = 0;
        int slice_n = 0;
        #pragma omp simd
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            rmse_total += rmse_values[inputIndex];
            slice_n += rmse_numbers[inputIndex];
        }

        if (slice_n > 0)
            rmse_total /= slice_n;
        else
            rmse_total = 0;

        return rmse_total;
    }

    //-------------------------------------------------------------------

    // match stack intensities with respect to the masked ROI
    void Reconstruction::MatchStackIntensitiesWithMasking(Array<RealImage>& stacks, const Array<RigidTransformation>& stack_transformations, double averageValue, bool together) {
        SVRTK_START_TIMING();

        RealImage m;
        Array<double> stack_average;
        stack_average.reserve(stacks.size());

        //remember the set average value
        _average_value = averageValue;

        //Calculate the averages of intensities for all stacks in the mask ROI
        for (size_t ind = 0; ind < stacks.size(); ind++) {
            double sum = 0, num = 0;

            if (_debug)
                m = stacks[ind];

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
                                if (_debug)
                                    m(i, j, k) = 1;
                                sum += stacks[ind](i, j, k);
                                num++;
                            } else if (_debug)
                                m(i, j, k) = 0;
                        }
                    }
            if (_debug)
                m.Write((boost::format("mask-for-matching%1%.nii.gz") % ind).str().c_str());

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

        if (_verbose) {
            _verbose_log << "Stack average intensities are ";
            for (size_t ind = 0; ind < stack_average.size(); ind++)
                _verbose_log << stack_average[ind] << " ";
            _verbose_log << endl;
            _verbose_log << "The new average value is " << averageValue << endl;
        }

        //Rescale stacks
        ClearAndReserve(_stack_factor, stacks.size());
        for (size_t ind = 0; ind < stacks.size(); ind++) {
            double factor = averageValue / (together ? global_average : stack_average[ind]);
            _stack_factor.push_back(factor);

            RealPixel *ptr = stacks[ind].Data();
            #pragma omp parallel for
            for (int i = 0; i < stacks[ind].NumberOfVoxels(); i++)
                if (ptr[i] > 0)
                    ptr[i] *= factor;
        }

        if (_debug) {
            #pragma omp parallel for
            for (size_t ind = 0; ind < stacks.size(); ind++)
                stacks[ind].Write((boost::format("rescaled-stack%1%.nii.gz") % ind).str().c_str());
        }

        if (_verbose) {
            _verbose_log << "Slice intensity factors are ";
            for (size_t ind = 0; ind < stack_average.size(); ind++)
                _verbose_log << _stack_factor[ind] << " ";
            _verbose_log << endl;
            _verbose_log << "The new average value is " << averageValue << endl;
        }

        SVRTK_END_TIMING("MatchStackIntensitiesWithMasking");
    }

    //-------------------------------------------------------------------

    // create slices from the input stacks
    void Reconstruction::CreateSlicesAndTransformations(const Array<RealImage>& stacks, const Array<RigidTransformation>& stack_transformations, const Array<double>& thickness, const Array<RealImage>& probability_maps) {
        double average_thickness = 0;

        // Reset and allocate memory
        const size_t reserve_size = stacks.size() * stacks[0].Attributes()._z;
        ClearAndReserve(_zero_slices, reserve_size);
        ClearAndReserve(_slices, reserve_size);
        ClearAndReserve(_package_index, reserve_size);
        ClearAndReserve(_slice_attributes, reserve_size);
        ClearAndReserve(_grey_slices, reserve_size);
        ClearAndReserve(_slice_dif, reserve_size);
        ClearAndReserve(_simulated_slices, reserve_size);
        ClearAndReserve(_reg_slice_weight, reserve_size);
        ClearAndReserve(_slice_pos, reserve_size);
        ClearAndReserve(_simulated_weights, reserve_size);
        ClearAndReserve(_simulated_inside, reserve_size);
        ClearAndReserve(_stack_index, reserve_size);
        ClearAndReserve(_transformations, reserve_size);
        if (_ffd)
            ClearAndReserve(_mffd_transformations, reserve_size);
        if (!probability_maps.empty())
            ClearAndReserve(_probability_maps, reserve_size);

        //for each stack
        for (size_t i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            const ImageAttributes& attr = stacks[i].Attributes();

            int package_index = 0;
            int current_package = -1;

            //attr._z is number of slices in the stack
            for (int j = 0; j < attr._z; j++) {
                if (!_n_packages.empty()) {
                    current_package++;
                    if (current_package > _n_packages[i] - 1)
                        current_package = 0;
                } else {
                    current_package = 0;
                }

                bool excluded = false;
                for (size_t fe = 0; fe < _excluded_entirely.size(); fe++) {
                    if (j == _excluded_entirely[fe]) {
                        excluded = true;
                        break;
                    }
                }

                if (excluded)
                    continue;

                //create slice by selecting the appropriate region of the stack
                RealImage slice = stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                //set correct voxel size in the stack. Z size is equal to slice thickness.
                slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                //remember the slice
                RealPixel tmin, tmax;
                slice.GetMinMax(&tmin, &tmax);
                _zero_slices.push_back(tmax > 1 && (tmax - tmin) > 1 ? 1 : -1);

                // if 2D gaussian filtering is required
                if (_blurring) {
                    GaussianBlurring<RealPixel> gbt(0.6 * slice.GetXSize());
                    gbt.Input(&slice);
                    gbt.Output(&slice);
                    gbt.Run();
                }

                _slices.push_back(slice);
                _package_index.push_back(current_package);
                _slice_attributes.push_back(slice.Attributes());

                _grey_slices.push_back(slice);
                memset(slice.Data(), 0, sizeof(RealPixel) * slice.NumberOfVoxels());
                _slice_dif.push_back(slice);
                _simulated_slices.push_back(slice);
                _reg_slice_weight.push_back(1);
                _slice_pos.push_back(j);
                slice = 1;
                _simulated_weights.push_back(slice);
                _simulated_inside.push_back(move(slice));
                //remember stack index for this slice
                _stack_index.push_back(i);
                //initialize slice transformation with the stack transformation
                _transformations.push_back(stack_transformations[i]);

                // if non-rigid FFD registartion option was selected
                if (_ffd)
                    _mffd_transformations.push_back(MultiLevelFreeFormTransformation());

                if (!probability_maps.empty()) {
                    RealImage proba = probability_maps[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                    proba.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                    _probability_maps.push_back(move(proba));
                }

                average_thickness += thickness[i];
            }
        }
        cout << "Number of slices: " << _slices.size() << endl;
        _number_of_slices_org = _slices.size();
        _average_thickness_org = average_thickness / _number_of_slices_org;
    }

    //-------------------------------------------------------------------

    // reset slices array and read them from the input stacks
    void Reconstruction::ResetSlices(Array<RealImage>& stacks, Array<double>& thickness) {
        if (_verbose)
            _verbose_log << "ResetSlices" << endl;

        UpdateSlices(stacks, thickness);

        #pragma omp parallel for
        for (size_t i = 0; i < _slices.size(); i++) {
            _bias[i].Initialize(_slices[i].Attributes());
            _weights[i].Initialize(_slices[i].Attributes());
        }
    }

    //-------------------------------------------------------------------

    // set slices and transformation from the given array
    void Reconstruction::SetSlicesAndTransformations(const Array<RealImage>& slices, const Array<RigidTransformation>& slice_transformations, const Array<int>& stack_ids, const Array<double>& thickness) {
        ClearAndReserve(_slices, slices.size());
        ClearAndReserve(_stack_index, slices.size());
        ClearAndReserve(_transformations, slices.size());

        //for each slice
        for (size_t i = 0; i < slices.size(); i++) {
            //get slice
            RealImage slice = slices[i];
            cout << "setting slice " << i << "\n";
            slice.Print();
            //set correct voxel size in the stack. Z size is equal to slice thickness.
            slice.PutPixelSize(slice.GetXSize(), slice.GetYSize(), thickness[i]);
            //remember the slice
            _slices.push_back(move(slice));
            //remember stack index for this slice
            _stack_index.push_back(stack_ids[i]);
            //get slice transformation
            _transformations.push_back(slice_transformations[i]);
        }
    }

    //-------------------------------------------------------------------

    // update slices array based on the given stacks
    void Reconstruction::UpdateSlices(Array<RealImage>& stacks, Array<double>& thickness) {
        ClearAndReserve(_slices, stacks.size() * stacks[0].Attributes()._z);

        //for each stack
        for (size_t i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            const ImageAttributes& attr = stacks[i].Attributes();

            //attr._z is number of slices in the stack
            for (int j = 0; j < attr._z; j++) {
                //create slice by selecting the appropriate region of the stack
                RealImage slice = stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                //set correct voxel size in the stack. Z size is equal to slice thickness.
                slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                //remember the slice
                _slices.push_back(move(slice));
            }
        }
        cout << "Number of slices: " << _slices.size() << endl;
    }

    //-------------------------------------------------------------------

    // mask slices based on the reconstruction mask
    void Reconstruction::MaskSlices() {
        //Check whether we have a mask
        if (!_have_mask) {
            cerr << "Could not mask slices because no mask has been set." << endl;
            return;
        }

        Utility::MaskSlices(_slices, _mask, [&](size_t index, double& x, double& y, double& z) {
            if (!_ffd)
                _transformations[index].Transform(x, y, z);
            else
                _mffd_transformations[index].Transform(-1, 1, x, y, z);
        });
    }

    //-------------------------------------------------------------------

    // transform slice coordinates to the reconstructed space
    void Reconstruction::Transform2Reconstructed(const int inputIndex, int& i, int& j, int& k, const int mode) {
        double x = i;
        double y = j;
        double z = k;

        _slices[inputIndex].ImageToWorld(x, y, z);

        if (!_ffd)
            _transformations[inputIndex].Transform(x, y, z);
        else
            _mffd_transformations[inputIndex].Transform(-1, 1, x, y, z);

        _reconstructed.WorldToImage(x, y, z);

        if (mode == 0) {
            i = (int)round(x);
            j = (int)round(y);
            k = (int)round(z);
        } else {
            i = (int)floor(x);
            j = (int)floor(y);
            k = (int)floor(z);
        }
    }

    //-------------------------------------------------------------------

    // initialise slice transformations with stack transformations
    void Reconstruction::InitialiseWithStackTransformations(const Array<RigidTransformation>& stack_transformations) {
        #pragma omp parallel for
        for (size_t sliceIndex = 0; sliceIndex < _slices.size(); sliceIndex++) {
            const auto& stack_transformation = stack_transformations[_stack_index[sliceIndex]];
            auto& transformation = _transformations[sliceIndex];
            transformation.PutTranslationX(stack_transformation.GetTranslationX());
            transformation.PutTranslationY(stack_transformation.GetTranslationY());
            transformation.PutTranslationZ(stack_transformation.GetTranslationZ());
            transformation.PutRotationX(stack_transformation.GetRotationX());
            transformation.PutRotationY(stack_transformation.GetRotationY());
            transformation.PutRotationZ(stack_transformation.GetRotationZ());
            transformation.UpdateMatrix();
        }
    }

    //-------------------------------------------------------------------

    // structure-based outlier rejection step (NCC-based)
    void Reconstruction::StructuralExclusion() {
        SVRTK_START_TIMING();

        double source_padding = -1;
        const double target_padding = -inf;
        constexpr bool dofin_invert = false;
        constexpr bool twod = false;

        const RealImage& source = _reconstructed;
        RealPixel smin, smax;
        source.GetMinMax(&smin, &smax);

        if (smin < -0.1)
            source_padding = -1;
        else if (smin < 0.1)
            source_padding = 0;

        Array<double> reg_ncc(_slices.size());
        double mean_ncc = 0;

        cout << " - excluded : ";

        RealImage output, target, slice_mask;
        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        ImageTransformation imagetransformation;
        imagetransformation.Input(&source);
        imagetransformation.Output(&output);
        imagetransformation.TargetPaddingValue(target_padding);
        imagetransformation.SourcePaddingValue(source_padding);
        imagetransformation.Interpolator(&interpolator);
        imagetransformation.TwoD(twod);
        imagetransformation.Invert(dofin_invert);

        #pragma omp parallel for private(output, target, slice_mask) firstprivate(imagetransformation) reduction(+: mean_ncc)
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            // transfrom reconstructed volume to the slice space
            imagetransformation.Transformation(&_transformations[inputIndex]);
            imagetransformation.Run();

            // blur the original slice
            target.Initialize(_slices[inputIndex].Attributes());
            GaussianBlurringWithPadding<RealPixel> gb(target.GetXSize() * 0.6, source_padding);
            gb.Input(&_slices[inputIndex]);
            gb.Output(&target);
            gb.Run();

            // mask slices
            slice_mask = _mask;
            TransformMask(target, slice_mask, _transformations[inputIndex]);
            target *= slice_mask;
            output.Initialize(_slices[inputIndex].Attributes());
            output *= slice_mask;

            // compute NCC
            double output_ncc = ComputeNCC(target, output, 0);
            if (output_ncc == -1)
                output_ncc = 1;
            reg_ncc[inputIndex] = output_ncc;
            mean_ncc += output_ncc;

            // set slice weight
            if (output_ncc > _global_NCC_threshold)
                _reg_slice_weight[inputIndex] = 1;
            else {
                _reg_slice_weight[inputIndex] = -1;
                cout << inputIndex << " ";
            }
        }
        cout << endl;
        mean_ncc /= _slices.size();

        cout << " - mean registration ncc: " << mean_ncc << endl;

        SVRTK_END_TIMING("StructuralExclusion");
    }

    //-------------------------------------------------------------------

    // run SVR
    void Reconstruction::SliceToVolumeRegistration() {
        SVRTK_START_TIMING();

        if (_debug)
            _reconstructed.Write("target.nii.gz");

        _grey_reconstructed = _reconstructed;

        if (!_ffd) {
            Parallel::SliceToVolumeRegistration p_reg(this);
            p_reg();
        } else {
            Parallel::SliceToVolumeRegistrationFFD p_reg(this);
            p_reg();
        }

        SVRTK_END_TIMING("SliceToVolumeRegistration");
    }

    //-------------------------------------------------------------------

    // run remote SVR
    void Reconstruction::RemoteSliceToVolumeRegistration(int iter, const string& str_mirtk_path, const string& str_current_exchange_file_path) {
        SVRTK_START_TIMING();

        const ImageAttributes& attr_recon = _reconstructed.Attributes();
        const string str_source = str_current_exchange_file_path + "/current-source.nii.gz";
        _reconstructed.Write(str_source.c_str());

        RealImage target;
        ResamplingWithPadding<RealPixel> resampling(attr_recon._dx, attr_recon._dx, attr_recon._dx, -1);
        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        resampling.Interpolator(&interpolator);
        resampling.Output(&target);

        constexpr int stride = 32;
        int svr_range_start = 0;
        int svr_range_stop = svr_range_start + stride;

        if (!_ffd) {
            // rigid SVR
            if (iter < 3) {
                _offset_matrices.clear();

                // save slice .nii.gz files
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

            // save slice transformations
            #pragma omp parallel for
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                RigidTransformation r_transform = _transformations[inputIndex];
                r_transform.PutMatrix(r_transform.GetMatrix() * _offset_matrices[inputIndex]);

                const string str_dofin = str_current_exchange_file_path + "/res-transformation-" + to_string(inputIndex) + ".dof";
                r_transform.Write(str_dofin.c_str());
            }

            // run remote SVR in strides
            while (svr_range_start < _slices.size()) {
                Parallel::RemoteSliceToVolumeRegistration registration(this, svr_range_start, svr_range_stop, str_mirtk_path, str_current_exchange_file_path);
                registration();

                svr_range_start = svr_range_stop;
                svr_range_stop = min(svr_range_start + stride, (int)_slices.size());
            }

            // read output transformations
            #pragma omp parallel for
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                const string str_dofout = str_current_exchange_file_path + "/res-transformation-" + to_string(inputIndex) + ".dof";
                _transformations[inputIndex].Read(str_dofout.c_str());

                //undo the offset
                _transformations[inputIndex].PutMatrix(_transformations[inputIndex].GetMatrix() * _offset_matrices[inputIndex].Inverse());
            }
        } else {
            // FFD SVR
            if (iter < 3) {
                // save slice .nii.gz files and transformations
                // Do not parallelise: ResamplingWithPadding has already been parallelised!
                for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                    target.Initialize(_slices[inputIndex].Attributes());
                    resampling.Input(&_slices[inputIndex]);
                    resampling.Run();

                    RealPixel tmin, tmax;
                    target.GetMinMax(&tmin, &tmax);
                    _zero_slices[inputIndex] = tmax > 1 && (tmax - tmin) > 1 ? 1 : -1;

                    string str_target = str_current_exchange_file_path + "/slice-" + to_string(inputIndex) + ".nii.gz";
                    target.Write(str_target.c_str());

                    string str_dofin = str_current_exchange_file_path + "/transformation-" + to_string(inputIndex) + ".dof";
                    _mffd_transformations[inputIndex].Write(str_dofin.c_str());
                }
            }

            // run parallel remote FFD SVR in strides
            while (svr_range_start < _slices.size()) {
                Parallel::RemoteSliceToVolumeRegistration registration(this, svr_range_start, svr_range_stop, str_mirtk_path, str_current_exchange_file_path, false);
                registration();

                svr_range_start = svr_range_stop;
                svr_range_stop = min(svr_range_start + stride, (int)_slices.size());
            }

            // read output transformations
            #pragma omp parallel for
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                const string str_dofout = str_current_exchange_file_path + "/transformation-" + to_string(inputIndex) + ".dof";
                _mffd_transformations[inputIndex].Read(str_dofout.c_str());
            }
        }

        SVRTK_END_TIMING("RemoteSliceToVolumeRegistration");
    }

    //-------------------------------------------------------------------

    // save the current recon model (for remote reconstruction option) - can be deleted
    void Reconstruction::SaveModelRemote(const string& str_current_exchange_file_path, int status_flag, int current_iteration) {
        if (_verbose)
            _verbose_log << "SaveModelRemote : " << current_iteration << endl;

        // save slices
        if (status_flag > 0) {
            #pragma omp parallel for
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                const string str_slice = str_current_exchange_file_path + "/org-slice-" + to_string(inputIndex) + ".nii.gz";
                _slices[inputIndex].Write(str_slice.c_str());
            }
            const string str_mask = str_current_exchange_file_path + "/current-mask.nii.gz";
            _mask.Write(str_mask.c_str());
        }

        // save transformations
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            const string str_dofin = str_current_exchange_file_path + "/org-transformation-" + to_string(current_iteration) + "-" + to_string(inputIndex) + ".dof";
            _transformations[inputIndex].Write(str_dofin.c_str());
        }

        // save recon volume
        const string str_recon = str_current_exchange_file_path + "/latest-out-recon.nii.gz";
        _reconstructed.Write(str_recon.c_str());
    }

    //-------------------------------------------------------------------

    // load remotely reconstructed volume (for remote reconstruction option) - can be deleted
    void Reconstruction::LoadResultsRemote(const string& str_current_exchange_file_path, int current_number_of_slices, int current_iteration) {
        if (_verbose)
            _verbose_log << "LoadResultsRemote : " << current_iteration << endl;

        const string str_recon = str_current_exchange_file_path + "/latest-out-recon.nii.gz";
        _reconstructed.Read(str_recon.c_str());
    }

    // load the current recon model (for remote reconstruction option) - can be deleted
    void Reconstruction::LoadModelRemote(const string& str_current_exchange_file_path, int current_number_of_slices, double average_thickness, int current_iteration) {
        if (_verbose)
            _verbose_log << "LoadModelRemote : " << current_iteration << endl;

        const string str_recon = str_current_exchange_file_path + "/latest-out-recon.nii.gz";
        const string str_mask = str_current_exchange_file_path + "/current-mask.nii.gz";

        _reconstructed.Read(str_recon.c_str());
        _mask.Read(str_mask.c_str());

        _template_created = true;
        _grey_reconstructed = _reconstructed;
        _attr_reconstructed = _reconstructed.Attributes();

        _have_mask = true;

        for (int inputIndex = 0; inputIndex < current_number_of_slices; inputIndex++) {
            // load slices
            RealImage slice;
            const string str_slice = str_current_exchange_file_path + "/org-slice-" + to_string(inputIndex) + ".nii.gz";
            slice.Read(str_slice.c_str());
            slice.PutPixelSize(slice.GetXSize(), slice.GetYSize(), average_thickness);
            _slices.push_back(slice);

            // load transformations
            const string str_dofin = str_current_exchange_file_path + "/org-transformation-" + to_string(current_iteration) + "-" + to_string(inputIndex) + ".dof";
            Transformation *t = Transformation::New(str_dofin.c_str());
            unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(t));
            _transformations.push_back(*rigidTransf);

            RealPixel tmin, tmax;
            slice.GetMinMax(&tmin, &tmax);
            _zero_slices.push_back(tmax > 1 && (tmax - tmin) > 1 ? 1 : -1);

            _package_index.push_back(0);

            _slice_attributes.push_back(slice.Attributes());

            _grey_slices.push_back(GreyImage(slice));

            memset(slice.Data(), 0, sizeof(RealPixel) * slice.NumberOfVoxels());
            _slice_dif.push_back(slice);
            _simulated_slices.push_back(slice);

            _reg_slice_weight.push_back(1);
            _slice_pos.push_back(inputIndex);

            slice = 1;
            _simulated_weights.push_back(slice);
            _simulated_inside.push_back(move(slice));

            _stack_index.push_back(0);

            if (_ffd)
                _mffd_transformations.push_back(MultiLevelFreeFormTransformation());
        }
    }

    //-------------------------------------------------------------------

    // save slice info in .csv
    void Reconstruction::SaveSliceInfo(int current_iteration) {
        string file_name;
        ofstream GD_csv_file;

        if (current_iteration > 0)
            file_name = (boost::format("summary-slice-info-%1%.csv") % current_iteration).str();
        else
            file_name = "summary-slice-info.csv";

        GD_csv_file.open(file_name);
        GD_csv_file << "Stack" << "," << "Slice" << "," << "Rx" << "," << "Ry" << "," << "Rz" << "," << "Tx" << "," << "Ty" << "," << "Tz" << "," << "Weight" << "," << "Inside" << "," << "Scale" << endl;

        for (size_t i = 0; i < _slices.size(); i++) {
            const double rx = _transformations[i].GetRotationX();
            const double ry = _transformations[i].GetRotationY();
            const double rz = _transformations[i].GetRotationZ();

            const double tx = _transformations[i].GetTranslationX();
            const double ty = _transformations[i].GetTranslationY();
            const double tz = _transformations[i].GetTranslationZ();

            const int inside = _slice_inside[i] ? 1 : 0;

            GD_csv_file << _stack_index[i] << "," << i << "," << rx << "," << ry << "," << rz << "," << tx << "," << ty << "," << tz << "," << _slice_weight[i] << "," << inside << "," << _scale[i] << endl;
        }
    }

    //-------------------------------------------------------------------

    // run calculation of transformation matrices
    void Reconstruction::CoeffInit() {
        SVRTK_START_TIMING();

        //resize slice-volume matrix from previous iteration
        ClearAndResize(_volcoeffs, _slices.size());

        //resize indicator of slice having and overlap with volumetric mask
        ClearAndResize(_slice_inside, _slices.size());
        _attr_reconstructed = _reconstructed.Attributes();

        Parallel::CoeffInit coeffinit(this);
        coeffinit();

        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        _volume_weights.Initialize(_reconstructed.Attributes());

        // Do not parallelise: It would cause data inconsistencies
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            for (size_t i = 0; i < _volcoeffs[inputIndex].size(); i++)
                for (size_t j = 0; j < _volcoeffs[inputIndex][i].size(); j++)
                    for (size_t k = 0; k < _volcoeffs[inputIndex][i][j].size(); k++) {
                        const POINT3D& p = _volcoeffs[inputIndex][i][j][k];
                        _volume_weights(p.x, p.y, p.z) += p.value;
                    }
        }

        if (_debug)
            _volume_weights.Write("volume_weights.nii.gz");

        //find average volume weight to modify alpha parameters accordingly
        const RealPixel *ptr = _volume_weights.Data();
        const RealPixel *pm = _mask.Data();
        double sum = 0;
        int num = 0;
        #pragma omp parallel for reduction(+: sum, num)
        for (int i = 0; i < _volume_weights.NumberOfVoxels(); i++) {
            if (pm[i] == 1) {
                sum += ptr[i];
                num++;
            }
        }
        _average_volume_weight = sum / num;

        if (_verbose)
            _verbose_log << "Average volume weight is " << _average_volume_weight << endl;

        SVRTK_END_TIMING("CoeffInit");
    }

    //-------------------------------------------------------------------

    // run gaussian reconstruction based on SVR & coeffinit outputs
    void Reconstruction::GaussianReconstruction() {
        SVRTK_START_TIMING();

        RealImage slice;
        Array<int> voxel_num;
        voxel_num.reserve(_slices.size());

        //clear _reconstructed image
        memset(_reconstructed.Data(), 0, sizeof(RealPixel) * _reconstructed.NumberOfVoxels());

        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            if (_volcoeffs[inputIndex].empty())
                continue;

            int slice_vox_num = 0;
            //copy the current slice
            slice = _slices[inputIndex];
            //alias the current bias image
            const RealImage& b = _bias[inputIndex];
            //read current scale factor
            const double scale = _scale[inputIndex];

            //Distribute slice intensities to the volume
            for (size_t i = 0; i < _volcoeffs[inputIndex].size(); i++)
                for (size_t j = 0; j < _volcoeffs[inputIndex][i].size(); j++)
                    if (slice(i, j, 0) > -0.01) {
                        //biascorrect and scale the slice
                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                        //number of volume voxels with non-zero coefficients
                        //for current slice voxel
                        const size_t n = _volcoeffs[inputIndex][i][j].size();

                        //if given voxel is not present in reconstructed volume at all, pad it

                        //calculate num of vox in a slice that have overlap with roi
                        if (n > 0)
                            slice_vox_num++;

                        //add contribution of current slice voxel to all voxel volumes
                        //to which it contributes
                        for (size_t k = 0; k < n; k++) {
                            const POINT3D& p = _volcoeffs[inputIndex][i][j][k];
                            _reconstructed(p.x, p.y, p.z) += p.value * slice(i, j, 0);
                        }
                    }
            voxel_num.push_back(slice_vox_num);
            //end of loop for a slice inputIndex
        }

        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxel
        _reconstructed /= _volume_weights;

        _reconstructed.Write("init.nii.gz");

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

        SVRTK_END_TIMING("GaussianReconstruction");
    }

    //-------------------------------------------------------------------

    // another version of CoeffInit
    void Reconstruction::CoeffInitSF(int begin, int end) {
        //resize slice-volume matrix from previous iteration
        ClearAndResize(_volcoeffsSF, _slicePerDyn);

        //resize indicator of slice having and overlap with volumetric mask
        ClearAndResize(_slice_insideSF, _slicePerDyn);

        Parallel::CoeffInitSF coeffinit(this, begin, end);
        coeffinit();

        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        _volume_weightsSF.Initialize(_reconstructed.Attributes());

        // Do not parallelise: It would cause data inconsistencies
        for (int inputIndex = begin; inputIndex < end; inputIndex++)
            for (size_t i = 0; i < _volcoeffsSF[inputIndex].size(); i++)
                for (size_t j = 0; j < _volcoeffsSF[inputIndex][i].size(); j++)
                    for (size_t k = 0; k < _volcoeffsSF[inputIndex % _slicePerDyn][i][j].size(); k++) {
                        const POINT3D& p = _volcoeffsSF[inputIndex % _slicePerDyn][i][j][k];
                        _volume_weightsSF(p.x, p.y, p.z) += p.value;
                    }

        if (_debug)
            _volume_weightsSF.Write("volume_weights.nii.gz");

        //find average volume weight to modify alpha parameters accordingly
        const RealPixel *ptr = _volume_weightsSF.Data();
        const RealPixel *pm = _mask.Data();
        double sum = 0;
        int num = 0;
        #pragma omp parallel for reduction(+: sum, num)
        for (int i = 0; i < _volume_weightsSF.NumberOfVoxels(); i++) {
            if (pm[i] == 1) {
                sum += ptr[i];
                num++;
            }
        }
        _average_volume_weightSF = sum / num;

        if (_verbose)
            _verbose_log << "Average volume weight is " << _average_volume_weightSF << endl;

    }  //end of CoeffInitSF()

    //-------------------------------------------------------------------

    // another version of gaussian reconstruction
    void Reconstruction::GaussianReconstructionSF(const Array<RealImage>& stacks) {
        Array<int> voxel_num;
        Array<RigidTransformation> currentTransformations;
        Array<RealImage> currentSlices, currentBiases;
        Array<double> currentScales;
        RealImage interpolated, slice;

        // Preallocate memory
        const size_t reserve_size = stacks[0].Attributes()._z;
        voxel_num.reserve(reserve_size);
        currentTransformations.reserve(reserve_size);
        currentSlices.reserve(reserve_size);
        currentScales.reserve(reserve_size);
        currentBiases.reserve(reserve_size);

        // clean _reconstructed
        memset(_reconstructed.Data(), 0, sizeof(RealPixel) * _reconstructed.NumberOfVoxels());

        for (size_t dyn = 0, counter = 0; dyn < stacks.size(); dyn++) {
            const ImageAttributes& attr = stacks[dyn].Attributes();

            CoeffInitSF(counter, counter + attr._z);

            for (int s = 0; s < attr._z; s++) {
                currentTransformations.push_back(_transformations[counter + s]);
                currentSlices.push_back(_slices[counter + s]);
                currentScales.push_back(_scale[counter + s]);
                currentBiases.push_back(_bias[counter + s]);
            }

            interpolated.Initialize(_reconstructed.Attributes());

            for (size_t s = 0; s < currentSlices.size(); s++) {
                if (_volcoeffsSF[s].empty())
                    continue;

                //copy the current slice
                slice = currentSlices[s];
                //alias the current bias image
                const RealImage& b = currentBiases[s];
                //read current scale factor
                const double scale = currentScales[s];

                int slice_vox_num = 0;
                for (size_t i = 0; i < _volcoeffsSF[s].size(); i++)
                    for (size_t j = 0; j < _volcoeffsSF[s][i].size(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            //biascorrect and scale the slice
                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                            //number of volume voxels with non-zero coefficients for current slice voxel
                            const size_t n = _volcoeffsSF[s][i][j].size();

                            //if given voxel is not present in reconstructed volume at all, pad it

                            //calculate num of vox in a slice that have overlap with roi
                            if (n > 0)
                                slice_vox_num++;

                            //add contribution of current slice voxel to all voxel volumes
                            //to which it contributes
                            for (size_t k = 0; k < n; k++) {
                                const POINT3D& p = _volcoeffsSF[s][i][j][k];
                                interpolated(p.x, p.y, p.z) += p.value * slice(i, j, 0);
                            }
                        }
                voxel_num.push_back(slice_vox_num);
            }
            counter += attr._z;
            _reconstructed += interpolated / _volume_weightsSF;

            currentTransformations.clear();
            currentSlices.clear();
            currentScales.clear();
            currentBiases.clear();
        }
        _reconstructed /= stacks.size();

        cout << "done." << endl;
        if (_debug)
            _reconstructed.Write("init.nii.gz");

        //now find slices with small overlap with ROI and exclude them.
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
    }

    //-------------------------------------------------------------------

    // initialise slice EM step
    void Reconstruction::InitializeEM() {
        ClearAndReserve(_weights, _slices.size());
        ClearAndReserve(_bias, _slices.size());
        ClearAndReserve(_scale, _slices.size());
        ClearAndReserve(_slice_weight, _slices.size());

        for (size_t i = 0; i < _slices.size(); i++) {
            //Create images for voxel weights and bias fields
            _weights.push_back(_slices[i]);
            _bias.push_back(_slices[i]);

            //Create and initialize scales
            _scale.push_back(1);

            //Create and initialize slice weights
            _slice_weight.push_back(1);
        }

        //Find the range of intensities
        _max_intensity = voxel_limits<RealPixel>::min();
        _min_intensity = voxel_limits<RealPixel>::max();

        #pragma omp parallel for reduction(min: _min_intensity) reduction(max: _max_intensity)
        for (size_t i = 0; i < _slices.size(); i++) {
            //to update minimum we need to exclude padding value
            const RealPixel *ptr = _slices[i].Data();
            for (int ind = 0; ind < _slices[i].NumberOfVoxels(); ind++) {
                if (ptr[ind] > 0) {
                    if (ptr[ind] > _max_intensity)
                        _max_intensity = ptr[ind];
                    if (ptr[ind] < _min_intensity)
                        _min_intensity = ptr[ind];
                }
            }
        }
    }

    //-------------------------------------------------------------------

    // initialise / reset EM values
    void Reconstruction::InitializeEMValues() {
        SVRTK_START_TIMING();

        #pragma omp parallel for
        for (size_t i = 0; i < _slices.size(); i++) {
            //Initialise voxel weights and bias values
            const RealPixel *pi = _slices[i].Data();
            RealPixel *pw = _weights[i].Data();
            RealPixel *pb = _bias[i].Data();
            for (int j = 0; j < _weights[i].NumberOfVoxels(); j++) {
                pw[j] = pi[j] > -0.01 ? 1 : 0;
                pb[j] = 0;
            }

            //Initialise slice weights
            _slice_weight[i] = 1;

            //Initialise scaling factors for intensity matching
            _scale[i] = 1;
        }

        //Force exclusion of slices predefined by user
        for (size_t i = 0; i < _force_excluded.size(); i++)
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slices.size())
                _slice_weight[_force_excluded[i]] = 0;

        SVRTK_END_TIMING("InitializeEMValues");
    }

    //-------------------------------------------------------------------

    // initialise parameters of EM robust statistics
    void Reconstruction::InitializeRobustStatistics() {
        Array<int> sigma_numbers(_slices.size());
        Array<double> sigma_values(_slices.size());
        RealImage slice;

        #pragma omp parallel for private(slice)
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            slice = _slices[inputIndex];
            //Voxel-wise sigma will be set to stdev of volumetric errors
            //For each slice voxel
            for (int i = 0; i < slice.GetX(); i++)
                for (int j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) > -0.01) {
                        //calculate stev of the errors
                        if (_simulated_inside[inputIndex](i, j, 0) == 1 && _simulated_weights[inputIndex](i, j, 0) > 0.99) {
                            slice(i, j, 0) -= _simulated_slices[inputIndex](i, j, 0);
                            sigma_values[inputIndex] += slice(i, j, 0) * slice(i, j, 0);
                            sigma_numbers[inputIndex]++;
                        }
                    }

            //if slice does not have an overlap with ROI, set its weight to zero
            if (!_slice_inside[inputIndex])
                _slice_weight[inputIndex] = 0;
        }

        double sigma = 0;
        int num = 0;
        #pragma omp simd
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sigma += sigma_values[inputIndex];
            num += sigma_numbers[inputIndex];
        }

        //Force exclusion of slices predefined by user
        for (size_t i = 0; i < _force_excluded.size(); i++)
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slices.size())
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

        if (_verbose)
            _verbose_log << "Initializing robust statistics: sigma=" << sqrt(_sigma) << " m=" << _m << " mix=" << _mix << " mix_s=" << _mix_s << endl;
    }

    //-------------------------------------------------------------------

    // run EStep for calculation of voxel-wise and slice-wise posteriors (weights)
    void Reconstruction::EStep() {
        Array<double> slice_potential(_slices.size());

        Parallel::EStep parallelEStep(this, slice_potential);
        parallelEStep();

        //To force-exclude slices predefined by a user, set their potentials to -1
        for (size_t i = 0; i < _force_excluded.size(); i++)
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slices.size())
                slice_potential[_force_excluded[i]] = -1;

        //exclude slices identified as having small overlap with ROI, set their potentials to -1
        for (size_t i = 0; i < _small_slices.size(); i++)
            slice_potential[_small_slices[i]] = -1;

        //these are unrealistic scales pointing at misregistration - exclude the corresponding slices
        for (size_t inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
            if (_scale[inputIndex] < 0.2 || _scale[inputIndex] > 5)
                slice_potential[inputIndex] = -1;

        //Calculation of slice-wise robust statistics parameters.
        //This is theoretically M-step,
        //but we want to use latest estimate of slice potentials
        //to update the parameters

        if (_verbose) {
            _verbose_log << endl << "Slice potentials:";
            for (size_t inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
                _verbose_log << " " << slice_potential[inputIndex];
            _verbose_log << endl;
        }

        //Calculate means of the inlier and outlier potentials
        double sum = 0, den = 0, sum2 = 0, den2 = 0, maxs = 0, mins = 1;
        #pragma omp parallel for reduction(+: sum, sum2, den, den2) reduction(max: maxs) reduction(min: mins)
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            if (slice_potential[inputIndex] >= 0) {
                //calculate means
                sum += slice_potential[inputIndex] * _slice_weight[inputIndex];
                den += _slice_weight[inputIndex];
                sum2 += slice_potential[inputIndex] * (1 - _slice_weight[inputIndex]);
                den2 += 1 - _slice_weight[inputIndex];

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
        if (sum > 0 && den > 0) {
            //do not allow too small sigma
            _sigma_s = max(sum / den, _step * _step / 6.28);
        } else {
            _sigma_s = 0.025;
            if (_verbose) {
                if (sum <= 0)
                    _verbose_log << "All slices are equal. ";
                if (den < 0) //this should not happen
                    _verbose_log << "All slices are outliers. ";
                _verbose_log << "Setting sigma to " << sqrt(_sigma_s) << endl;
            }
        }

        //sigma_s2
        if (sum2 > 0 && den2 > 0) {
            //do not allow too small sigma
            _sigma_s2 = max(sum2 / den2, _step * _step / 6.28);
        } else {
            //do not allow too small sigma
            _sigma_s2 = max((_mean_s2 - _mean_s) * (_mean_s2 - _mean_s) / 4, _step * _step / 6.28);

            if (_verbose) {
                if (sum2 <= 0)
                    _verbose_log << "All slices are equal. ";
                if (den2 <= 0)
                    _verbose_log << "All slices inliers. ";
                _verbose_log << "Setting sigma_s2 to " << sqrt(_sigma_s2) << endl;
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
                if (slice_potential[inputIndex] < _mean_s2 && slice_potential[inputIndex] > _mean_s) //should not happen
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

        if (_verbose) {
            _verbose_log << setprecision(3);
            _verbose_log << "Slice robust statistics parameters: ";
            _verbose_log << "means: " << _mean_s << " " << _mean_s2 << "  ";
            _verbose_log << "sigmas: " << sqrt(_sigma_s) << " " << sqrt(_sigma_s2) << "  ";
            _verbose_log << "proportions: " << _mix_s << " " << 1 - _mix_s << endl;
            _verbose_log << "Slice weights:";
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                _verbose_log << " " << _slice_weight[inputIndex];
            _verbose_log << endl;
        }
    }

    //-------------------------------------------------------------------

    // run slice scaling
    void Reconstruction::Scale() {
        SVRTK_START_TIMING();

        Parallel::Scale parallelScale(this);
        parallelScale();

        if (_verbose) {
            _verbose_log << setprecision(3);
            _verbose_log << "Slice scale =";
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                _verbose_log << " " << _scale[inputIndex];
            _verbose_log << endl;
        }

        SVRTK_END_TIMING("Scale");
    }

    //-------------------------------------------------------------------

    // run slice bias correction
    void Reconstruction::Bias() {
        SVRTK_START_TIMING();
        Parallel::Bias parallelBias(this);
        parallelBias();
        SVRTK_END_TIMING("Bias");
    }

    //-------------------------------------------------------------------

    // compute difference between simulated and original slices
    void Reconstruction::SliceDifference() {
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            _slice_dif[inputIndex] = _slices[inputIndex];

            for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                    if (_slices[inputIndex](i, j, 0) > -0.01) {
                        _slice_dif[inputIndex](i, j, 0) *= exp(-(_bias[inputIndex])(i, j, 0)) * _scale[inputIndex];
                        _slice_dif[inputIndex](i, j, 0) -= _simulated_slices[inputIndex](i, j, 0);
                    } else
                        _slice_dif[inputIndex](i, j, 0) = 0;
                }
            }
        }
    }

    //-------------------------------------------------------------------

    // run SR reconstruction step
    void Reconstruction::Superresolution(int iter) {
        SVRTK_START_TIMING();

        // save current reconstruction for edge-preserving smoothing
        RealImage original = _reconstructed;

        SliceDifference();

        Parallel::Superresolution parallelSuperresolution(this);
        parallelSuperresolution();

        RealImage& addon = parallelSuperresolution.addon;
        _confidence_map = move(parallelSuperresolution.confidence_map);
        //_confidence4mask = _confidence_map;

        if (_debug) {
            _confidence_map.Write((boost::format("confidence-map%1%.nii.gz") % iter).str().c_str());
            addon.Write((boost::format("addon%1%.nii.gz") % iter).str().c_str());
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

        // update the volume with computed addon
        _reconstructed += addon * _alpha; //_average_volume_weight;

        //bound the intensities
        RealPixel *pr = _reconstructed.Data();
        #pragma omp parallel for
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {
            if (pr[i] < _min_intensity * 0.9)
                pr[i] = _min_intensity * 0.9;
            if (pr[i] > _max_intensity * 1.1)
                pr[i] = _max_intensity * 1.1;
        }

        //Smooth the reconstructed image with regularisation
        AdaptiveRegularization(iter, original);

        //Remove the bias in the reconstructed volume compared to previous iteration
        if (_global_bias_correction)
            BiasCorrectVolume(original);

        SVRTK_END_TIMING("Superresolution");
    }

    //-------------------------------------------------------------------

    // run MStep (RS)
    void Reconstruction::MStep(int iter) {
        Parallel::MStep parallelMStep(this);
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

        if (_verbose)
            _verbose_log << "Voxel-wise robust statistics parameters: sigma=" << sqrt(_sigma) << " mix=" << _mix << " m=" << _m << endl;
    }

    //-------------------------------------------------------------------

    // run adaptive regularisation of the SR reconstructed volume
    void Reconstruction::AdaptiveRegularization(int iter, const RealImage& original) {
        Array<double> factor(13);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }

        Array<RealImage> b(13, _reconstructed);

        Parallel::AdaptiveRegularization1 parallelAdaptiveRegularization1(this, b, factor, original);
        parallelAdaptiveRegularization1();

        const RealImage original2 = _reconstructed;
        Parallel::AdaptiveRegularization2 parallelAdaptiveRegularization2(this, b, original2);
        parallelAdaptiveRegularization2();

        if (_alpha * _lambda / (_delta * _delta) > 0.068)
            cerr << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068." << endl;
    }

    //-------------------------------------------------------------------

    // correct bias
    void Reconstruction::BiasCorrectVolume(const RealImage& original) {
        //remove low-frequency component in the reconstructed image which might have occurred due to overfitting of the biasfield
        RealImage residual = _reconstructed;
        RealImage weights = _mask;

        //calculate weighted residual
        const RealPixel *po = original.Data();
        RealPixel *pr = residual.Data();
        RealPixel *pw = weights.Data();
        #pragma omp parallel for
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {
            //second and term to avoid numerical problems
            if ((pw[i] == 1) && (po[i] > _low_intensity_cutoff * _max_intensity) && (pr[i] > _low_intensity_cutoff * _max_intensity)) {
                pr[i] = log(pr[i] / po[i]);
            } else {
                pw[i] = pr[i] = 0;
            }
        }

        //blurring needs to be same as for slices
        GaussianBlurring<RealPixel> gb(_sigma_bias);
        //blur weighted residual
        gb.Input(&residual);
        gb.Output(&residual);
        gb.Run();
        //blur weight image
        gb.Input(&weights);
        gb.Output(&weights);
        gb.Run();

        //calculate the bias field
        pr = residual.Data();
        pw = weights.Data();
        const RealPixel *pm = _mask.Data();
        RealPixel *pi = _reconstructed.Data();
        #pragma omp parallel for
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {
            if (pm[i] == 1) {
                //weighted gaussian smoothing
                //exponential to recover multiplicative bias field
                pr[i] = exp(pr[i] / pw[i]);
                //bias correct reconstructed
                pi[i] /= pr[i];
                //clamp intensities to allowed range
                if (pi[i] < _min_intensity * 0.9)
                    pi[i] = _min_intensity * 0.9;
                if (pi[i] > _max_intensity * 1.1)
                    pi[i] = _max_intensity * 1.1;
            } else {
                pr[i] = 0;
            }
        }
    }

    //-------------------------------------------------------------------

    // evaluation based on the number of excluded slices
    void Reconstruction::Evaluate(int iter, ostream& outstr) {
        outstr << "Iteration " << iter << ": " << endl;

        size_t included_count = 0, excluded_count = 0, outside_count = 0;
        string included, excluded, outside;

        for (size_t i = 0; i < _slices.size(); i++) {
            if (_slice_inside[i]) {
                if (_slice_weight[i] >= 0.5) {
                    included += " " + to_string(i);
                    included_count++;
                } else {
                    excluded += " " + to_string(i);
                    excluded_count++;
                }
            } else {
                outside += " " + to_string(i);
                outside_count++;
            }
        }

        outstr << "Included slices:" << included << "\n";
        outstr << "Total: " << included_count << "\n";
        outstr << "Excluded slices:" << excluded << "\n";
        outstr << "Total: " << excluded_count << "\n";
        outstr << "Outside slices:" << outside << "\n";
        outstr << "Total: " << outside_count << endl;
    }

    //-------------------------------------------------------------------

    // Normalise Bias
    void Reconstruction::NormaliseBias(int iter) {
        SVRTK_START_TIMING();

        Parallel::NormaliseBias parallelNormaliseBias(this);
        parallelNormaliseBias();
        RealImage& bias = parallelNormaliseBias.bias;

        // normalize the volume by proportion of contributing slice voxels for each volume voxel
        bias /= _volume_weights;

        MaskImage(bias, _mask, 0);
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
            bias.Write((boost::format("averagebias%1%.nii.gz") % iter).str().c_str());

        RealPixel *pi = _reconstructed.Data();
        const RealPixel *pb = bias.Data();
        #pragma omp parallel for
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++)
            if (pi[i] != -1)
                pi[i] /= exp(-(pb[i]));

        SVRTK_END_TIMING("NormaliseBias");
    }

    //-------------------------------------------------------------------

    void Reconstruction::ReadTransformations(const char *folder, size_t file_count, Array<RigidTransformation>& transformations) {
        if (_slices.empty())
            throw runtime_error("Please create slices before reading transformations!");

        ClearAndReserve(transformations, file_count);
        for (size_t i = 0; i < file_count; i++) {
            const string path = (boost::format("%1%/transformation%2%.dof") % (folder ? folder : ".") % i).str();
            Transformation *transformation = Transformation::New(path.c_str());
            unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(transformation));
            transformations.push_back(*rigidTransf);
            cout << path << endl;
        }
    }

    //-------------------------------------------------------------------

    void Reconstruction::ReadTransformations(const char *folder) {
        cout << "Reading transformations:" << endl;
        ReadTransformations(folder, _slices.size(), _transformations);
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveBiasFields() {
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _bias[inputIndex].Write((boost::format("bias%1%.nii.gz") % inputIndex).str().c_str());
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveConfidenceMap() {
        _confidence_map.Write("confidence-map.nii.gz");
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveSlices() {
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _slices[inputIndex].Write((boost::format("slice%1%.nii.gz") % inputIndex).str().c_str());
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveSlicesWithTiming() {
        cout << "Saving slices with timing: ";
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _slices[inputIndex].Write((boost::format("sliceTime%1%.nii.gz") % _slice_timing[inputIndex]).str().c_str());
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveSimulatedSlices() {
        cout << "Saving simulated slices ... ";
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _simulated_slices[inputIndex].Write((boost::format("simslice%1%.nii.gz") % inputIndex).str().c_str());
        cout << "done." << endl;
    }


    //-------------------------------------------------------------------

    void Reconstruction::SaveWeights() {
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _weights[inputIndex].Write((boost::format("weights%1%.nii.gz") % inputIndex).str().c_str());
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveRegistrationStep(const Array<RealImage>& stacks, const int step) {
        ImageAttributes attr = stacks[0].Attributes();
        int threshold = attr._z;
        int counter = 0;
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            if (inputIndex >= threshold) {
                counter++;
                attr = stacks[counter].Attributes();
                threshold += attr._z;
            }
            const int stack = counter;
            const int slice = inputIndex - (threshold - attr._z);
            _transformations[inputIndex].Write((boost::format("step%04i_travol%04islice%04i.dof") % step % stack % slice).str().c_str());
        }
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveTransformationsWithTiming(const int iter) {
        cout << "Saving transformations with timing: ";
        #pragma omp parallel for
        for (size_t i = 0; i < _transformations.size(); i++) {
            cout << i << " ";
            if (iter < 0)
                _transformations[i].Write((boost::format("transformationTime%1%.dof") % _slice_timing[i]).str().c_str());
            else
                _transformations[i].Write((boost::format("transformationTime%1%-%2%.dof") % iter % _slice_timing[i]).str().c_str());
        }
        cout << " done." << endl;
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveTransformations() {
        #pragma omp parallel for
        for (size_t i = 0; i < _transformations.size(); i++)
            _transformations[i].Write((boost::format("transformation%1%.dof") % i).str().c_str());
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveProbabilityMap(int i) {
        _brain_probability.Write((boost::format("probability_map%1%.nii") % i).str().c_str());
    }

    //-------------------------------------------------------------------

    // save slice info
    void Reconstruction::SaveSliceInfo(const char *filename, const Array<string>& stack_files) {
        ofstream info(filename);

        // header
        info << "stack_index\t"
            << "stack_name\t"
            << "included\t" // Included slices
            << "excluded\t"  // Excluded slices
            << "outside\t"  // Outside slices
            << "weight\t"
            << "scale\t"
            // << "_stack_factor\t"
            << "TranslationX\t"
            << "TranslationY\t"
            << "TranslationZ\t"
            << "RotationX\t"
            << "RotationY\t"
            << "RotationZ" << endl;

        for (size_t i = 0; i < _slices.size(); i++) {
            const RigidTransformation& t = _transformations[i];
            info << _stack_index[i] << "\t"
                << stack_files[_stack_index[i]] << "\t"
                << ((_slice_weight[i] >= 0.5 && _slice_inside[i]) ? 1 : 0) << "\t" // Included slices
                << ((_slice_weight[i] < 0.5 && _slice_inside[i]) ? 1 : 0) << "\t"  // Excluded slices
                << (!_slice_inside[i] ? 1 : 0) << "\t"  // Outside slices
                << _slice_weight[i] << "\t"
                << _scale[i] << "\t"
                // << _stack_factor[i] << "\t"
                << t.GetTranslationX() << "\t"
                << t.GetTranslationY() << "\t"
                << t.GetTranslationZ() << "\t"
                << t.GetRotationX() << "\t"
                << t.GetRotationY() << "\t"
                << t.GetRotationZ() << endl;
        }
    }

    //-------------------------------------------------------------------

    // updated package-to-volume registration
    void Reconstruction::newPackageToVolume(const Array<RealImage>& stacks, const Array<int>& pack_num, const Array<int>& multiband_vector, const Array<int>& order, const int step, const int rewinder, const int iter, const int steps) {
        // copying transformations from previous iterations
        _previous_transformations = _transformations;

        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        // Insert(params, "Image (dis-)similarity measure", "NMI");
        if (_nmi_bins > 0)
            Insert(params, "No. of bins", _nmi_bins);
        // Insert(params, "Image interpolation mode", "Linear");
        Insert(params, "Background value", -1);
        // Insert(params, "Background value for image 1", -1);
        // Insert(params, "Background value for image 2", -1);

        GenericRegistrationFilter rigidregistration;
        rigidregistration.Parameter(params);

        int wrapper = stacks.size() / steps;
        if (stacks.size() % steps > 0)
            wrapper++;

        GreyImage t;
        Array<RealImage> sstacks, packages;
        Array<int> spack_num, smultiband_vector, sorder, z_slice_order, t_slice_order, t_internal_slice_order;
        Array<RigidTransformation> internal_transformations;

        for (int w = 0; w < wrapper; w++) {
            const int doffset = w * steps;
            // preparing input for this iterations
            for (int s = 0; s < steps; s++) {
                if (s + doffset < stacks.size()) {
                    sstacks.push_back(stacks[s + doffset]);
                    spack_num.push_back(pack_num[s + doffset]);
                    smultiband_vector.push_back(multiband_vector[s + doffset]);
                    sorder.push_back(order[s + doffset]);
                }
            }

            SplitPackageswithMB(sstacks, spack_num, packages, smultiband_vector, sorder, step, rewinder, z_slice_order, t_slice_order);

            // other variables
            int counter1 = 0, counter2 = 0, counter3 = 0;

            for (size_t i = 0; i < sstacks.size(); i++) {
                const RealImage& firstPackage = packages[counter1];
                const int& multiband = smultiband_vector[i];
                int extra = (firstPackage.GetZ() / multiband) % spack_num[i];
                int startIterations = 0, endIterations = 0;

                // slice loop
                for (int sl = 0; sl < firstPackage.GetZ(); sl++) {
                    t_internal_slice_order.push_back(t_slice_order[counter3 + sl]);
                    internal_transformations.push_back(_transformations[counter2 + sl]);
                }

                // package look
                for (int j = 0; j < spack_num[i]; j++) {
                    // performing registration
                    const RealImage& target = packages[counter1];
                    const GreyImage& s = _reconstructed;
                    t = target;

                    if (_debug) {
                        t.Write((boost::format("target%1%-%2%-%3%.nii.gz") % iter % (i + doffset) % j).str().c_str());
                        s.Write((boost::format("source%1%-%2%-%3%.nii.gz") % iter % (i + doffset) % j).str().c_str());
                    }

                    //check whether package is empty (all zeros)
                    RealPixel tmin, tmax;
                    target.GetMinMax(&tmin, &tmax);

                    if (tmax > 0) {
                        RigidTransformation offset;
                        ResetOrigin(t, offset);
                        const Matrix& mo = offset.GetMatrix();
                        internal_transformations[j].PutMatrix(internal_transformations[j].GetMatrix() * mo);

                        rigidregistration.Input(&t, &s);
                        Transformation *dofout = nullptr;
                        rigidregistration.Output(&dofout);
                        rigidregistration.InitialGuess(&internal_transformations[j]);
                        rigidregistration.GuessParameter();
                        rigidregistration.Run();

                        unique_ptr<RigidTransformation> rigid_dofout(dynamic_cast<RigidTransformation*>(dofout));
                        internal_transformations[j] = *rigid_dofout;

                        internal_transformations[j].PutMatrix(internal_transformations[j].GetMatrix() * mo.Inverse());
                    }

                    if (_debug)
                        internal_transformations[j].Write((boost::format("transformation%1%-%2%-%3%.dof") % iter % (i + doffset) % j).str().c_str());

                    // saving transformations
                    int iterations = (firstPackage.GetZ() / multiband) / spack_num[i];
                    if (extra > 0) {
                        iterations++;
                        extra--;
                    }
                    endIterations += iterations;

                    for (int k = startIterations; k < endIterations; k++) {
                        for (size_t l = 0; l < t_internal_slice_order.size(); l++) {
                            if (k == t_internal_slice_order[l]) {
                                _transformations[counter2 + l].PutTranslationX(internal_transformations[j].GetTranslationX());
                                _transformations[counter2 + l].PutTranslationY(internal_transformations[j].GetTranslationY());
                                _transformations[counter2 + l].PutTranslationZ(internal_transformations[j].GetTranslationZ());
                                _transformations[counter2 + l].PutRotationX(internal_transformations[j].GetRotationX());
                                _transformations[counter2 + l].PutRotationY(internal_transformations[j].GetRotationY());
                                _transformations[counter2 + l].PutRotationZ(internal_transformations[j].GetRotationZ());
                                _transformations[counter2 + l].UpdateMatrix();
                            }
                        }
                    }
                    startIterations = endIterations;
                    counter1++;
                }
                // resetting variables for next dynamic
                startIterations = 0;
                endIterations = 0;
                counter2 += firstPackage.GetZ();
                counter3 += firstPackage.GetZ();

                t_internal_slice_order.clear();
                internal_transformations.clear();
            }

            //save overall slice order
            const ImageAttributes& attr = stacks[0].Attributes();
            const int slices_per_dyn = attr._z / multiband_vector[0];

            //slice order should repeat for each dynamic - only take first dynamic
            _slice_timing.clear();
            for (size_t dyn = 0; dyn < stacks.size(); dyn++)
                for (int i = 0; i < attr._z; i++) {
                    _slice_timing.push_back(dyn * slices_per_dyn + t_slice_order[i]);
                    cout << "slice timing = " << _slice_timing[i] << endl;
                }

            for (size_t i = 0; i < z_slice_order.size(); i++)
                cout << "z(" << i << ")=" << z_slice_order[i] << endl;

            for (size_t i = 0; i < t_slice_order.size(); i++)
                cout << "t(" << i << ")=" << t_slice_order[i] << endl;

            // save transformations and clear
            z_slice_order.clear();
            t_slice_order.clear();

            sstacks.clear();
            spack_num.clear();
            smultiband_vector.clear();
            sorder.clear();
            packages.clear();
        }
    }

    //-------------------------------------------------------------------

    // run package-to-volume registration
    void Reconstruction::PackageToVolume(const Array<RealImage>& stacks, const Array<int>& pack_num, const Array<RigidTransformation>& stack_transformations) {
        SVRTK_START_TIMING();

        int firstSlice = 0;
        Array<int> firstSlice_array;
        firstSlice_array.reserve(stacks.size());

        for (size_t i = 0; i < stacks.size(); i++) {
            firstSlice_array.push_back(firstSlice);
            firstSlice += stacks[i].GetZ();
        }

        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        // Insert(params, "Image interpolation mode", "Linear");
        Insert(params, "Background value for image 1", 0);
        Insert(params, "Background value for image 2", -1);

        if (_nmi_bins > 0)
            Insert(params, "No. of bins", _nmi_bins);

        GenericRegistrationFilter rigidregistration;
        RealImage mask, target;
        Array<RealImage> packages;

        #pragma omp parallel for private(mask, target, packages, rigidregistration)
        for (size_t i = 0; i < stacks.size(); i++) {
            SplitImage(stacks[i], pack_num[i], packages);
            for (size_t j = 0; j < packages.size(); j++) {
                if (_debug)
                    packages[j].Write((boost::format("package-%1%-%2%.nii.gz") % i % j).str().c_str());

                //packages are not masked at present
                mask = _mask;
                const RigidTransformation& mask_transform = stack_transformations[i]; //s[i];
                TransformMask(packages[j], mask, mask_transform);

                target = packages[j] * mask;
                const RealImage& source = _reconstructed;

                //find existing transformation
                double x = 0, y = 0, z = 0;
                packages[j].ImageToWorld(x, y, z);
                stacks[i].WorldToImage(x, y, z);

                const int firstSliceIndex = round(z) + firstSlice_array[i];
                // cout<<"First slice index for package "<<j<<" of stack "<<i<<" is "<<firstSliceIndex<<endl;

                //put origin in target to zero
                RigidTransformation offset;
                ResetOrigin(target, offset);
                const Matrix& mo = offset.GetMatrix();
                _transformations[firstSliceIndex].PutMatrix(_transformations[firstSliceIndex].GetMatrix() * mo);

                rigidregistration.Parameter(params);
                rigidregistration.Input(&target, &source);
                Transformation *dofout;
                rigidregistration.Output(&dofout);
                rigidregistration.InitialGuess(&_transformations[firstSliceIndex]);
                rigidregistration.GuessParameter();
                rigidregistration.Run();

                unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(dofout));
                _transformations[firstSliceIndex] = *rigidTransf;

                //undo the offset
                _transformations[firstSliceIndex].PutMatrix(_transformations[firstSliceIndex].GetMatrix() * mo.Inverse());

                if (_debug)
                    _transformations[firstSliceIndex].Write((boost::format("transformation-%1%-%2%.dof") % i % j).str().c_str());

                //set the transformation to all slices of the package
                // cout<<"Slices of the package "<<j<<" of the stack "<<i<<" are: ";
                for (int k = 0; k < packages[j].GetZ(); k++) {
                    x = 0; y = 0; z = k;
                    packages[j].ImageToWorld(x, y, z);
                    stacks[i].WorldToImage(x, y, z);
                    int sliceIndex = round(z) + firstSlice_array[i];
                    // cout<<sliceIndex<<" "<<endl;

                    if (sliceIndex >= _transformations.size())
                        throw runtime_error("Reconstruction::PackageToVolume: sliceIndex out of range.\n" + to_string(sliceIndex) + " " + to_string(_transformations.size()));

                    if (sliceIndex != firstSliceIndex) {
                        _transformations[sliceIndex].PutTranslationX(_transformations[firstSliceIndex].GetTranslationX());
                        _transformations[sliceIndex].PutTranslationY(_transformations[firstSliceIndex].GetTranslationY());
                        _transformations[sliceIndex].PutTranslationZ(_transformations[firstSliceIndex].GetTranslationZ());
                        _transformations[sliceIndex].PutRotationX(_transformations[firstSliceIndex].GetRotationX());
                        _transformations[sliceIndex].PutRotationY(_transformations[firstSliceIndex].GetRotationY());
                        _transformations[sliceIndex].PutRotationZ(_transformations[firstSliceIndex].GetRotationZ());
                        _transformations[sliceIndex].UpdateMatrix();
                    }
                }
            }
            // cout<<"End of stack "<<i<<endl<<endl;
        }

        SVRTK_END_TIMING("PackageToVolume");
    }

} // namespace svrtk
