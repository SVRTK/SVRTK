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

// Boost
#include <boost/format.hpp>


using namespace std;
using namespace mirtk;

namespace svrtk {

    // Extract specific image ROI
    void bbox(RealImage& stack, RigidTransformation& transformation, double& min_x, double& min_y, double& min_z, double& max_x, double& max_y, double& max_z) {
        min_x = DBL_MAX;
        min_y = DBL_MAX;
        min_z = DBL_MAX;
        max_x = -DBL_MAX;
        max_y = -DBL_MAX;
        max_z = -DBL_MAX;
        double x, y, z;
        // WARNING: do not search to increment by stack.GetZ()-1,
        // otherwise you would end up with a 0 increment for slices...
        for (int i = 0; i <= stack.GetX(); i += stack.GetX())
            for (int j = 0; j <= stack.GetY(); j += stack.GetY())
                for (int k = 0; k <= stack.GetZ(); k += stack.GetZ()) {
                    x = i;
                    y = j;
                    z = k;

                    stack.ImageToWorld(x, y, z);
                    transformation.Transform(x, y, z);

                    if (x < min_x)
                        min_x = x;
                    if (y < min_y)
                        min_y = y;
                    if (z < min_z)
                        min_z = z;
                    if (x > max_x)
                        max_x = x;
                    if (y > max_y)
                        max_y = y;
                    if (z > max_z)
                        max_z = z;
                }

    }

    //-------------------------------------------------------------------

    // Crop to non zero ROI
    void bboxCrop(RealImage& image) {
        int min_x, min_y, min_z, max_x, max_y, max_z;
        min_x = image.GetX() - 1;
        min_y = image.GetY() - 1;
        min_z = image.GetZ() - 1;
        max_x = 0;
        max_y = 0;
        max_z = 0;
        for (int i = 0; i < image.GetX(); i++)
            for (int j = 0; j < image.GetY(); j++)
                for (int k = 0; k < image.GetZ(); k++) {
                    if (image.Get(i, j, k) > 0) {
                        if (i < min_x)
                            min_x = i;
                        if (j < min_y)
                            min_y = j;
                        if (k < min_z)
                            min_z = k;
                        if (i > max_x)
                            max_x = i;
                        if (j > max_y)
                            max_y = j;
                        if (k > max_z)
                            max_z = k;
                    }
                }

        //Cut region of interest
        image = image.GetRegion(min_x, min_y, min_z, max_x, max_y, max_z);
    }

    //-------------------------------------------------------------------

    // Find centroid
    void centroid(RealImage& image, double& x, double& y, double& z) {
        double sum_x = 0;
        double sum_y = 0;
        double sum_z = 0;
        double norm = 0;
        double v;
        for (int i = 0; i < image.GetX(); i++)
            for (int j = 0; j < image.GetY(); j++)
                for (int k = 0; k < image.GetZ(); k++) {
                    v = image.Get(i, j, k);
                    if (v <= 0)
                        continue;
                    sum_x += v * i;
                    sum_y += v * j;
                    sum_z += v * k;
                    norm += v;
                }

        x = sum_x / norm;
        y = sum_y / norm;
        z = sum_z / norm;

        image.ImageToWorld(x, y, z);
    }

    //-------------------------------------------------------------------

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

    Reconstruction::~Reconstruction() {}

    //-------------------------------------------------------------------

    // Center stacks with respect to the centre of the mask
    void Reconstruction::CenterStacks(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber) {
        double x0, y0, z0;
        double x, y, z;
        int tx, ty, tz;
        RealImage mask = stacks[templateNumber];

        for (tx = 0; tx < mask.GetX(); tx++) {
            for (ty = 0; ty < mask.GetY(); ty++) {
                for (tz = 0; tz < mask.GetZ(); tz++) {
                    if (mask(tx, ty, tz) < 0)
                        mask(tx, ty, tz) = 0;
                }
            }
        }

        centroid(mask, x0, y0, z0);

        for (int i = 0; i < stacks.size(); i++) {
            if (i == templateNumber)
                continue;

            mask = stacks[i];
            for (tx = 0; tx < mask.GetX(); tx++) {
                for (ty = 0; ty < mask.GetY(); ty++) {
                    for (tz = 0; tz < mask.GetZ(); tz++) {
                        if (mask(tx, ty, tz) < 0)
                            mask(tx, ty, tz) = 0;
                    }
                }
            }

            centroid(mask, x, y, z);

            RigidTransformation translation;
            translation.PutTranslationX(x0 - x);
            translation.PutTranslationY(y0 - y);
            translation.PutTranslationZ(z0 - z);

            stack_transformations[i].PutMatrix(translation.GetMatrix() * stack_transformations[i].GetMatrix());
        }
    }

    //-------------------------------------------------------------------

    // Class for computing average of all stacks
    class ParallelAverage {
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

        ParallelAverage(
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

        ParallelAverage(ParallelAverage& x, split) : ParallelAverage(x.reconstructor, x.stacks, x.stack_transformations, x.targetPadding, x.sourcePadding, x.background, x.linear) {}

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

        void join(const ParallelAverage& y) {
            average += y.average;
            weights += y.weights;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, stacks.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    class ParallelSliceAverage {
        Reconstruction *reconstructor;
        Array<RealImage>& slices;
        Array<RigidTransformation>& slice_transformations;
        RealImage& average;
        RealImage& weights;

    public:
        void operator()(const blocked_range<size_t>& r) const {
            for (int k0 = r.begin(); k0 < r.end(); k0++) {
                for (int j0 = 0; j0 < average.GetY(); j0++) {
                    for (int i0 = 0; i0 < average.GetX(); i0++) {
                        double x0 = i0;
                        double y0 = j0;
                        double z0 = k0;
                        // Transform point into world coordinates
                        average.ImageToWorld(x0, y0, z0);
                        for (int inputIndex = 0; inputIndex < slices.size(); inputIndex++) {
                            double x = x0;
                            double y = y0;
                            double z = z0;
                            // Transform point
                            slice_transformations[inputIndex].Transform(x, y, z);
                            // Transform point into image coordinates
                            slices[inputIndex].WorldToImage(x, y, z);
                            int i = round(x);
                            int j = round(y);
                            int k = round(z);
                            // Check whether transformed point is in FOV of input
                            if ((i >= 0) && (i < slices[inputIndex].GetX()) &&
                                (j >= 0) && (j < slices[inputIndex].GetY()) &&
                                (k >= 0) && (k < slices[inputIndex].GetZ())) {
                                if (slices[inputIndex](i, j, k) > 0) {
                                    average.PutAsDouble(i0, j0, k0, 0, average(i0, j0, k0) + slices[inputIndex](i, j, k));
                                    weights.PutAsDouble(i0, j0, k0, 0, weights(i0, j0, k0) + 1);
                                }
                            }
                        }
                    }
                }
            }
        }

        ParallelSliceAverage(
            Reconstruction *reconstructor,
            Array<RealImage>& slices,
            Array<RigidTransformation>& slice_transformations,
            RealImage& average,
            RealImage& weights) :
            reconstructor(reconstructor),
            slices(slices),
            slice_transformations(slice_transformations),
            average(average),
            weights(weights) {
            average.Initialize(reconstructor->_reconstructed.Attributes());
            weights.Initialize(reconstructor->_reconstructed.Attributes());
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, average.GetZ()), *this);
        }
    };

    //-------------------------------------------------------------------

    // Create average of all stacks based on the input transformations
    RealImage Reconstruction::CreateAverage(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations) {
        SVRTK_START_TIMING();

        if (!_template_created) {
            cerr << "Please create the template before calculating the average of the stacks." << endl;
            exit(1);
        }

        InvertStackTransformations(stack_transformations);
        unique_ptr<ParallelAverage> p_average(new ParallelAverage(this,
            stacks,
            stack_transformations,
            -1, 0, 0, // target/source/background
            true));
        (*p_average)();
        const RealImage average = p_average->average / p_average->weights;

        InvertStackTransformations(stack_transformations);

        SVRTK_END_TIMING("CreateAverage");
        return average;
    }

    //-------------------------------------------------------------------

    // Set given template with specific resolution
    double Reconstruction::CreateTemplate(const RealImage& stack, double resolution) {
        double dx, dy, dz, d;

        //Get image attributes - image size and voxel size
        ImageAttributes attr = stack.Attributes();

        //enlarge stack in z-direction in case top of the head is cut off
        attr._z += 2;

        //create enlarged image
        RealImage enlarged(attr);

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

        cout << "Reconstructed volume voxel size : " << d << " mm " << endl;

        //resample "enlarged" to resolution "d"
        InterpolationMode interpolation = Interpolation_Linear;

        unique_ptr<InterpolateImageFunction> interpolator(InterpolateImageFunction::New(interpolation));

        RealPixel smin, smax;
        stack.GetMinMax(&smin, &smax);
        enlarged = stack;

        // interpolate the input stack to the given resolution
        if (smin < -0.1) {
            ResamplingWithPadding<RealPixel> resampler(d, d, d, -1);
            GenericLinearInterpolateImageFunction<RealImage> interpolator;
            resampler.Input(&stack);
            resampler.Output(&enlarged);
            resampler.Interpolator(&interpolator);
            resampler.Run();
        } else {
            if (smin < 0.1) {
                ResamplingWithPadding<RealPixel> resampler(d, d, d, 0);
                GenericLinearInterpolateImageFunction<RealImage> interpolator;
                resampler.Input(&stack);
                resampler.Output(&enlarged);
                resampler.Interpolator(&interpolator);
                resampler.Run();
            } else {
                Resampling<RealPixel> resampler(d, d, d);
                resampler.Input(&enlarged);
                resampler.Output(&enlarged);
                resampler.Interpolator(interpolator.get());
                resampler.Run();
            }
        }

        //initialize recontructed volume
        _reconstructed = move(enlarged);
        _template_created = true;

        if (_debug)
            _reconstructed.Write("template.nii.gz");
        _grey_reconstructed = _reconstructed;
        _attr_reconstructed = _reconstructed.Attributes();

        //return resulting resolution of the template image
        return d;
    }

    //-------------------------------------------------------------------

    // Create anisotropic template
    double Reconstruction::CreateTemplateAniso(const RealImage& stack) {
        double dx, dy, dz, d;

        //Get image attributes - image size and voxel size
        ImageAttributes attr = stack.Attributes();

        //enlarge stack in z-direction in case top of the head is cut off
        attr._t = 1;

        //create enlarged image
        RealImage enlarged(attr);

        cout << "Constructing volume with anisotropic voxel size " << attr._x << " " << attr._y << " " << attr._z << endl;

        //initialize recontructed volume
        _reconstructed = move(enlarged);
        _template_created = true;

        //return resulting resolution of the template image
        return attr._x;
    }

    //-------------------------------------------------------------------

    // Set template
    void Reconstruction::SetTemplate(RealImage templateImage) {
        RealImage t2template = _reconstructed;
        memset(t2template.Data(), 0, sizeof(RealPixel) * t2template.NumberOfVoxels());

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

    // binarise mask
    RealImage Reconstruction::CreateMask(RealImage image) {
        RealPixel *ptr = image.Data();
        for (int i = 0; i < image.NumberOfVoxels(); ++i)
            ptr[i] = ptr[i] > 0.5 ? 1 : 0;

        return image;
    }

    //-------------------------------------------------------------------

    // normalise and threshold mask
    RealImage Reconstruction::ThresholdNormalisedMask(RealImage image, double threshold) {
        RealPixel smin, smax;
        image.GetMinMax(&smin, &smax);

        if (smax > 0)
            image /= smax;

        RealPixel *ptr = image.Data();
        for (int i = 0; i < image.NumberOfVoxels(); ++i)
            ptr[i] = ptr[i] > threshold ? 1 : 0;

        return image;
    }

    //-------------------------------------------------------------------

    // create mask from dark / black background
    void Reconstruction::CreateMaskFromBlackBackground(const Array<RealImage>& stacks, Array<RigidTransformation> stack_transformations, double smooth_mask) {
        //Create average of the stack using currect stack transformations
        GreyImage average = CreateAverage(stacks, stack_transformations);

        GreyPixel *ptr = average.Data();
        for (int i = 0; i < average.NumberOfVoxels(); i++) {
            if (ptr[i] < 0)
                ptr[i] = 0;
        }

        //Create mask of the average from the black background
        MeanShift msh(average, 0, 256);
        msh.GenerateDensity();
        msh.SetTreshold();
        msh.RemoveBackground();
        GreyImage mask = msh.ReturnMask();

        //Calculate LCC of the mask to remove disconnected structures
        MeanShift msh2(mask, 0, 256);
        msh2.SetOutput(&mask);
        msh2.Lcc(1);
    }

    //-------------------------------------------------------------------

    // Set given mask to the reconstruction object
    void Reconstruction::SetMask(RealImage *mask, double sigma, double threshold) {
        if (!_template_created) {
            cerr << "Please create the template before setting the mask, so that the mask can be resampled to the correct dimensions." << endl;
            exit(1);
        }

        _mask = _reconstructed;

        if (mask != NULL) {
            //if sigma is nonzero first smooth the mask
            if (sigma > 0) {
                //blur mask
                GaussianBlurring<RealPixel> gb(sigma);

                gb.Input(mask);
                gb.Output(mask);
                gb.Run();

                //binarize mask
                RealPixel *ptr = mask->Data();
                for (int i = 0; i < mask->NumberOfVoxels(); i++)
                    ptr[i] = ptr[i] > threshold ? 1 : 0;
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
        for (int i = 0; i < _mask.GetX(); i++) {
            for (int j = 0; j < _mask.GetY(); j++) {
                for (int k = 0; k < _mask.GetZ(); k++) {
                    if (_mask(i, j, k) > 0.1)
                        vol++;
                }
            }
        }

        vol *= _reconstructed.GetXSize() * _reconstructed.GetYSize() * _reconstructed.GetZSize() / 1000;

        cout << "ROI volume : " << vol << " cc " << endl;

        if (_debug)
            _mask.Write("mask.nii.gz");
    }

    //-------------------------------------------------------------------

    // transform mask to the specified stack space
    void Reconstruction::TransformMask(const RealImage& image, RealImage& mask, const RigidTransformation& transformation) {
        //transform mask to the space of image
        ImageTransformation imagetransformation;

        InterpolationMode interpolation = Interpolation_NN;
        unique_ptr<InterpolateImageFunction> interpolator(InterpolateImageFunction::New(interpolation));

        imagetransformation.Input(&mask);
        imagetransformation.Transformation(&transformation);

        RealImage m = image;
        imagetransformation.Output(&m);
        //target contains zeros and ones image, need padding -1
        imagetransformation.TargetPaddingValue(-1);
        //need to fill voxels in target where there is no info from source with zeroes
        imagetransformation.SourcePaddingValue(0);
        imagetransformation.Interpolator(interpolator.get());
        imagetransformation.Run();
        mask = move(m);
    }

    //-------------------------------------------------------------------

    // reset image origin and save it into the output transforamtion (GreyImage)
    void Reconstruction::ResetOrigin(GreyImage& image, RigidTransformation& transformation) {
        double ox, oy, oz;
        image.GetOrigin(ox, oy, oz);
        image.PutOrigin(0, 0, 0);
        transformation.PutTranslationX(ox);
        transformation.PutTranslationY(oy);
        transformation.PutTranslationZ(oz);
        transformation.PutRotationX(0);
        transformation.PutRotationY(0);
        transformation.PutRotationZ(0);
    }

    //-------------------------------------------------------------------

    // reset image origin and save it into the output transforamtion (RealImage)
    void Reconstruction::ResetOrigin(RealImage& image, RigidTransformation& transformation) {
        double ox, oy, oz;
        image.GetOrigin(ox, oy, oz);
        image.PutOrigin(0, 0, 0);
        transformation.PutTranslationX(ox);
        transformation.PutTranslationY(oy);
        transformation.PutTranslationZ(oz);
        transformation.PutRotationX(0);
        transformation.PutRotationY(0);
        transformation.PutRotationZ(0);
    }

    //-------------------------------------------------------------------

    class ParallelQualityReport {
        Reconstruction *reconstructor;

    public:
        double out_global_ncc;
        double out_global_nrmse;

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

        void join(const ParallelQualityReport& y) {
            out_global_ncc += y.out_global_ncc;
            out_global_nrmse += y.out_global_nrmse;
        }

        ParallelQualityReport(ParallelQualityReport& x, split) : ParallelQualityReport(x.reconstructor) {}

        ParallelQualityReport(Reconstruction *reconstructor) : reconstructor(reconstructor) {
            out_global_ncc = 0;
            out_global_nrmse = 0;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    // generate reconstruction quality report / metrics
    void Reconstruction::ReconQualityReport(double& out_ncc, double& out_nrmse, double& average_weight, double& ratio_excluded) {
        ParallelQualityReport parallelQualityReport(this);
        parallelQualityReport();

        average_weight = _average_volume_weight;
        out_ncc = parallelQualityReport.out_global_ncc / _slices.size();
        out_nrmse = parallelQualityReport.out_global_nrmse / _slices.size();

        if (!isfinite(out_nrmse))
            out_nrmse = 0;

        if (!isfinite(out_ncc))
            out_ncc = 0;

        size_t count_excluded = 0;
        for (int i = 0; i < _slices.size(); i++) {
            if (_slice_weight[i] < 0.5)
                count_excluded++;
        }

        ratio_excluded = (double)count_excluded / _slices.size();
    }

    //-------------------------------------------------------------------

    // compute inter-slice volume NCC (motion metric)
    double Reconstruction::VolumeNCC(RealImage& input_stack, RealImage template_stack, const RealImage& mask) {
        template_stack *= mask;

        RigidTransformation r_init;
        r_init.PutTranslationX(0.0001);
        r_init.PutTranslationY(0.0001);
        r_init.PutTranslationZ(-0.0001);

        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        Insert(params, "Background value for image 1", 0);

        GenericRegistrationFilter registration;
        registration.Parameter(params);
        registration.Input(&template_stack, &input_stack);
        Transformation *dofout = nullptr;
        registration.Output(&dofout);
        registration.InitialGuess(&r_init);
        registration.GuessParameter();
        registration.Run();
        unique_ptr<RigidTransformation> r_dofout(dynamic_cast<RigidTransformation*>(dofout));

        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        double source_padding = 0;
        double target_padding = -inf;
        bool dofin_invert = false;
        bool twod = false;

        RealImage& output = template_stack;
        memset(output.Data(), 0, sizeof(RealPixel) * output.NumberOfVoxels());

        unique_ptr<ImageTransformation> imagetransformation(new ImageTransformation);
        imagetransformation->Input(&input_stack);
        imagetransformation->Transformation(r_dofout.get());
        imagetransformation->Output(&output);
        imagetransformation->TargetPaddingValue(target_padding);
        imagetransformation->SourcePaddingValue(source_padding);
        imagetransformation->Interpolator(&interpolator);
        imagetransformation->TwoD(twod);
        imagetransformation->Invert(dofin_invert);
        imagetransformation->Run();

        input_stack = output * mask;

        double ncc = 0;
        int count = 0;
        for (int z = 0; z < input_stack.GetZ() - 1; z++) {
            constexpr int sh = 5;
            const RealImage slice_1 = input_stack.GetRegion(sh, sh, z, input_stack.GetX() - sh, input_stack.GetY() - sh, z + 1);
            const RealImage slice_2 = input_stack.GetRegion(sh, sh, z + 1, input_stack.GetX() - sh, input_stack.GetY() - sh, z + 2);

            double slice_count = -1;
            const double slice_ncc = ComputeNCC(slice_1, slice_2, 0.1, &slice_count);
            if (slice_ncc > 0) {
                ncc += slice_ncc;
                count++;
            }
        }

        return ncc / count;
    }

    //-------------------------------------------------------------------

    // class for global 3D stack registration
    class ParallelStackRegistrations {
        Reconstruction *reconstructor;
        const Array<RealImage>& stacks;
        Array<RigidTransformation>& stack_transformations;
        int templateNumber;
        RealImage& target;
        RigidTransformation& offset;

    public:
        ParallelStackRegistrations(
            Reconstruction *reconstructor,
            const Array<RealImage>& stacks,
            Array<RigidTransformation>& stack_transformations,
            int _templateNumber,
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

                //                Insert(params, "Background value", 0);
                //                Insert(params, "Image (dis-)similarity measure", "NCC");
                ////                Insert(params, "Image interpolation mode", "Linear");
                //                string type = "box";
                //                string units = "mm";
                //                double width = 0;
                //                Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);

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

    // run global stack registration to the template
    void Reconstruction::StackRegistrations(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber) {
        SVRTK_START_TIMING();

        InvertStackTransformations(stack_transformations);

        RealImage target;
        // check whether to use the global template or the selected stack
        if (!_template_flag) {
            target = stacks[templateNumber];
        } else {
            target = _reconstructed;
        }

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
        unique_ptr<ParallelStackRegistrations> p_reg(new ParallelStackRegistrations(this, stacks, stack_transformations, templateNumber, target, offset));
        (*p_reg)();

        InvertStackTransformations(stack_transformations);

        SVRTK_END_TIMING("StackRegistrations");
    }

    //-------------------------------------------------------------------

    // restore the original slice intensities
    void Reconstruction::RestoreSliceIntensities() {
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            //calculate scaling factor
            const double factor = _stack_factor[_stack_index[inputIndex]];//_average_value;

            // read the pointer to current slice
            RealPixel *p = _slices[inputIndex].Data();
            for (int i = 0; i < _slices[inputIndex].NumberOfVoxels(); i++) {
                if (p[i] > 0)
                    p[i] /= factor;
            }
        }
    }

    //-------------------------------------------------------------------

    // scale the output reconstructed volume
    void Reconstruction::ScaleVolume() {
        double scalenum = 0, scaleden = 0;

        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
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
            scalenum /= scaleden;

        if (_verbose)
            _verbose_log << "scale : " << scale << endl;

        RealPixel *ptr = _reconstructed.Data();
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {
            if (ptr[i] > 0)
                ptr[i] *= scale;
        }
    }

    //-------------------------------------------------------------------

    // class for simulation of slices from the current reconstructed 3D volume - v1
    class ParallelSimulateSlices {
        Reconstruction *reconstructor;

    public:
        ParallelSimulateSlices(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                //Calculate simulated slice
                reconstructor->_simulated_slices[inputIndex] = 0;
                reconstructor->_simulated_weights[inputIndex] = 0;
                reconstructor->_simulated_inside[inputIndex] = 0;
                reconstructor->_slice_inside[inputIndex] = false;

                for (int i = 0; i < reconstructor->_slice_attributes[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attributes[inputIndex]._y; j++)
                        if (reconstructor->_slices[inputIndex](i, j, 0) > -0.01) {
                            double weight = 0;
                            const size_t n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for (size_t k = 0; k < n; k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                reconstructor->_simulated_slices[inputIndex](i, j, 0) += p.value * reconstructor->_reconstructed(p.x, p.y, p.z);
                                weight += p.value;
                                if (reconstructor->_mask(p.x, p.y, p.z) > 0.1) {
                                    reconstructor->_simulated_inside[inputIndex](i, j, 0) = 1;
                                    reconstructor->_slice_inside[inputIndex] = true;
                                }
                            }
                            if (weight > 0) {
                                reconstructor->_simulated_slices[inputIndex](i, j, 0) /= weight;
                                reconstructor->_simulated_weights[inputIndex](i, j, 0) = weight;
                            }
                        };
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    // class for simulation of slices from the current reconstructed 3D volume - v2 (use this one)
    class ParallelSimulateSlices2 {
        Reconstruction *reconstructor;

    public:
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
                            const int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
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

        void join(const ParallelSimulateSlices2& y) {}

        ParallelSimulateSlices2(ParallelSimulateSlices2& x, split) : ParallelSimulateSlices2(x.reconstructor) {}

        ParallelSimulateSlices2(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };


    //-------------------------------------------------------------------

    // run simulation of slices from the reconstruction volume
    void Reconstruction::SimulateSlices() {
        SVRTK_START_TIMING();
        unique_ptr<ParallelSimulateSlices2> p_sim(new ParallelSimulateSlices2(this));
        (*p_sim)();
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
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            double weight = 0;
                            int n = _volcoeffs[inputIndex][i][j].size();
                            for (int k = 0; k < n; k++) {
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

            for (int i = 0; i < sim.GetX(); i++)
                for (int j = 0; j < sim.GetY(); j++)
                    stacks[_stack_index[inputIndex]](i, j, z) = sim(i, j, 0);
        }
    }

    //-------------------------------------------------------------------

    // match stack intensities with respect to the average
    void Reconstruction::MatchStackIntensities(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, double averageValue, bool together) {
        //Calculate the averages of intensities for all stacks
        Array<double> stack_average(stacks.size());

        //remember the set average value
        _average_value = averageValue;

        //averages need to be calculated only in ROI
        for (int ind = 0; ind < stacks.size(); ind++) {
            double sum = 0, num = 0;
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
            if (num > 0)
                stack_average[ind] = sum / num;
            else {
                cerr << "Stack " << ind << " has no overlap with ROI" << endl;
                exit(1);
            }
        }

        double global_average = 0;
        if (together) {
            for (int i = 0; i < stack_average.size(); i++)
                global_average += stack_average[i];
            global_average /= stack_average.size();
        }

        if (_verbose) {
            _verbose_log << "Stack average intensities are ";
            for (int ind = 0; ind < stack_average.size(); ind++)
                _verbose_log << stack_average[ind] << " ";
            _verbose_log << endl;
            _verbose_log << "The new average value is " << averageValue << endl;
        }

        _stack_factor = Array<double>(stacks.size(), 1);

        //Rescale stacks
        for (int ind = 0; ind < stacks.size(); ++ind) {
            double factor = averageValue / (together ? global_average : stack_average[ind]);
            _stack_factor[ind] = factor;

            RealPixel *ptr = stacks[ind].Data();
            for (int i = 0; i < stacks[ind].NumberOfVoxels(); i++) {
                if (ptr[i] > 0)
                    ptr[i] *= factor;
            }
        }

        if (_debug) {
            for (int ind = 0; ind < stacks.size(); ind++)
                stacks[ind].Write((boost::format("rescaled-stack%1%.nii.gz") % ind).str().c_str());
        }

        if (_verbose) {
            _verbose_log << "Slice intensity factors are ";
            for (int ind = 0; ind < stack_average.size(); ind++)
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

        // compute NRMSE between the simulated and original slice for non-zero voxels
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            RealImage nt = _slices[inputIndex];
            const RealImage& ns = _simulated_slices[inputIndex];
            double nrmse = 0;
            double s_diff = 0;
            double s_t = 0;
            int s_n = 0;
            int z = 0;
            for (int x = 0; x < nt.GetX(); x++) {
                for (int y = 0; y < nt.GetY(); y++) {
                    if (nt(x, y, z) > 0 && ns(x, y, z) > 0) {
                        nt(x, y, z) *= exp(-(_bias[inputIndex])(x, y, z)) * _scale[inputIndex];
                        s_t += nt(x, y, z);
                        s_diff += (nt(x, y, z) - ns(x, y, z)) * (nt(x, y, z) - ns(x, y, z));
                        s_n++;
                    }
                }
            }
            if (s_n > 0)
                nrmse = sqrt(s_diff / s_n) / (s_t / s_n);
            else
                nrmse = 0;
            rmse_values[inputIndex] = nrmse;
            if (nrmse > 0)
                rmse_numbers[inputIndex] = 1;
        }

        double rmse_total = 0;
        int slice_n = 0;
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
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

    // mask stacks with respect to the reconstruction mask and given transformations
    void Reconstruction::MaskStacks(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations) {
        //Check whether we have a mask
        if (!_have_mask) {
            cout << "Could not mask slices because no mask has been set." << endl;
            return;
        }

        for (int inputIndex = 0; inputIndex < stacks.size(); inputIndex++) {
            RealImage& stack = stacks[inputIndex];
            for (int i = 0; i < stack.GetX(); i++)
                for (int j = 0; j < stack.GetY(); j++) {
                    for (int k = 0; k < stack.GetZ(); k++) {
                        //if the value is smaller than 1 assume it is padding
                        if (stack(i, j, k) < 0.01)
                            stack(i, j, k) = -1;
                        //image coordinates of a slice voxel
                        double x = i;
                        double y = j;
                        double z = k;
                        //change to world coordinates in slice space
                        stack.ImageToWorld(x, y, z);
                        //world coordinates in volume space
                        stack_transformations[inputIndex].Transform(x, y, z);
                        //image coordinates in volume space
                        _mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);
                        //if the voxel is outside mask ROI set it to -1 (padding value)
                        if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0) && (z < _mask.GetZ())) {
                            if (_mask(x, y, z) == 0)
                                stack(i, j, k) = -1;
                        } else
                            stack(i, j, k) = -1;
                    }
                }
        }
    }

    //-------------------------------------------------------------------

    // match stack intensities with respect to the masked ROI
    void Reconstruction::MatchStackIntensitiesWithMasking(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, double averageValue, bool together) {
        SVRTK_START_TIMING();

        Array<double> stack_average(stacks.size());

        //remember the set average value
        _average_value = averageValue;

        //Calculate the averages of intensities for all stacks in the mask ROI

        for (int ind = 0; ind < stacks.size(); ++ind) {
            double sum = 0, num = 0;
            RealImage m;

            if (_debug)
                m = stacks[ind];

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
                        z = round(z); _stack_factor.clear();
                        //if the voxel is inside mask ROI include it
                        if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0) && (z < _mask.GetZ())) {
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
                stack_average[ind] = sum / num;
            } else {
                cerr << "Stack " << ind << " has no overlap with ROI" << endl;
                exit(1);
            }
        }

        _stack_factor = Array<double>(stacks.size(), 1);

        double global_average = 0;
        if (together) {
            for (int i = 0; i < stack_average.size(); i++)
                global_average += stack_average[i];
            global_average /= stack_average.size();
        }

        if (_verbose) {
            _verbose_log << "Stack average intensities are ";
            for (int ind = 0; ind < stack_average.size(); ind++)
                _verbose_log << stack_average[ind] << " ";
            _verbose_log << endl;
            _verbose_log << "The new average value is " << averageValue << endl;
        }

        //Rescale stacks
        for (int ind = 0; ind < stacks.size(); ++ind) {
            double factor = averageValue / (together ? global_average : stack_average[ind]);
            _stack_factor[ind] = factor;

            RealPixel *ptr = stacks[ind].Data();
            for (int i = 0; i < stacks[ind].NumberOfVoxels(); i++) {
                if (ptr[i] > 0)
                    ptr[i] *= factor;
            }
        }

        if (_debug) {
            for (int ind = 0; ind < stacks.size(); ind++)
                stacks[ind].Write((boost::format("rescaled-stack%1%.nii.gz") % ind).str().c_str());
        }

        if (_verbose) {
            _verbose_log << "Slice intensity factors are ";
            for (int ind = 0; ind < stack_average.size(); ind++)
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

        //for each stack
        for (size_t i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            const ImageAttributes& attr = stacks[i].Attributes();

            int package_index = 0;
            int current_package = -1;

            //attr._z is number of slices in the stack
            for (int j = 0; j < attr._z; j++) {
                if (_n_packages.size() > 0) {
                    current_package++;
                    if (current_package > (_n_packages[i] - 1))
                        current_package = 0;
                } else {
                    current_package = 0;
                }

                bool excluded = false;
                for (int fe = 0; fe < _excluded_entirely.size(); fe++) {
                    if (j == _excluded_entirely[fe]) {
                        excluded = true;
                        break;
                    }
                }

                if (!excluded) {
                    //create slice by selecting the appropreate region of the stack
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
                    _simulated_inside.push_back(slice);
                    //remeber stack index for this slice
                    _stack_index.push_back(i);
                    //initialize slice transformation with the stack transformation
                    _transformations.push_back(stack_transformations[i]);

                    average_thickness = average_thickness + thickness[i];

                    // if non-rigid FFD registartion option was selected
                    if (_ffd)
                        _mffd_transformations.push_back(MultiLevelFreeFormTransformation());

                    if (probability_maps.size() > 0) {
                        RealImage proba = probability_maps[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                        proba.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                        _probability_maps.push_back(move(proba));
                    }
                }
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

        for (int i = 0; i < _slices.size(); i++) {
            _bias[i].Initialize(_slices[i].Attributes());
            _weights[i].Initialize(_slices[i].Attributes());
        }
    }

    //-------------------------------------------------------------------

    // set slices and transformation from the given array
    void Reconstruction::SetSlicesAndTransformations(const Array<RealImage>& slices, const Array<RigidTransformation>& slice_transformations, const Array<int>& stack_ids, const Array<double>& thickness) {
        _slices.clear();
        _stack_index.clear();
        _transformations.clear();

        //for each slice
        for (size_t i = 0; i < slices.size(); i++) {
            //get slice
            RealImage slice = slices[i];
            std::cout << "setting slice " << i << "\n";
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
        _slices.clear();
        //for each stack
        for (size_t i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            const ImageAttributes& attr = stacks[i].Attributes();

            //attr._z is number of slices in the stack
            for (int j = 0; j < attr._z; j++) {
                //create slice by selecting the appropreate region of the stack
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
            cout << "Could not mask slices because no mask has been set." << endl;
            return;
        }

        //mask slices
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            RealImage& slice = _slices[inputIndex];
            for (int i = 0; i < slice.GetX(); i++)
                for (int j = 0; j < slice.GetY(); j++) {
                    //if the value is smaller than 1 assume it is padding
                    if (slice(i, j, 0) < 0.01)
                        slice(i, j, 0) = -1;
                    //image coordinates of a slice voxel
                    double x = i;
                    double y = j;
                    double z = 0;
                    //change to world coordinates in slice space
                    slice.ImageToWorld(x, y, z);

                    // use either rigid or FFD transformation models
                    if (!_ffd)
                        _transformations[inputIndex].Transform(x, y, z);
                    else
                        _mffd_transformations[inputIndex].Transform(-1, 1, x, y, z);

                    //image coordinates in volume space
                    _mask.WorldToImage(x, y, z);
                    x = round(x);
                    y = round(y);
                    z = round(z);
                    //if the voxel is outside mask ROI set it to -1 (padding value)
                    if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0) && (z < _mask.GetZ())) {
                        if (_mask(x, y, z) == 0)
                            slice(i, j, 0) = -1;
                    } else
                        slice(i, j, 0) = -1;
                }
        }
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

    // class for parallel SVR
    class ParallelSliceToVolumeRegistration {
        Reconstruction *reconstructor;

    public:
        ParallelSliceToVolumeRegistration(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

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

    // initialise slice transfromations with stack transformations
    void Reconstruction::InitialiseWithStackTransformations(const Array<RigidTransformation>& stack_transformations) {
        for (int sliceIndex = 0; sliceIndex < _slices.size(); sliceIndex++) {
            int stackIndex = _stack_index[sliceIndex];
            _transformations[sliceIndex].PutTranslationX(stack_transformations[stackIndex].GetTranslationX());
            _transformations[sliceIndex].PutTranslationY(stack_transformations[stackIndex].GetTranslationY());
            _transformations[sliceIndex].PutTranslationZ(stack_transformations[stackIndex].GetTranslationZ());
            _transformations[sliceIndex].PutRotationX(stack_transformations[stackIndex].GetRotationX());
            _transformations[sliceIndex].PutRotationY(stack_transformations[stackIndex].GetRotationY());
            _transformations[sliceIndex].PutRotationZ(stack_transformations[stackIndex].GetRotationZ());
            _transformations[sliceIndex].UpdateMatrix();
        }
    }

    //-------------------------------------------------------------------

    // structure-based outlier rejection step (NCC-based)
    void Reconstruction::StructuralExclusion() {
        SVRTK_START_TIMING();

        const RealImage& source = _reconstructed;

        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        double source_padding = -1;
        double target_padding = -inf;
        bool dofin_invert = false;
        bool twod = false;

        RealPixel smin, smax;
        source.GetMinMax(&smin, &smax);

        if (smin < -0.1)
            source_padding = -1;
        else if (smin < 0.1)
            source_padding = 0;

        Array<double> reg_ncc;
        double mean_ncc = 0;

        cout << " - excluded : ";
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            RealImage output(_slices[inputIndex].Attributes());

            // transfrom reconstructed volume to the slice space
            unique_ptr<ImageTransformation> imagetransformation(new ImageTransformation);
            imagetransformation->Input(&source);
            imagetransformation->Transformation(&_transformations[inputIndex]);
            imagetransformation->Output(&output);
            imagetransformation->TargetPaddingValue(target_padding);
            imagetransformation->SourcePaddingValue(source_padding);
            imagetransformation->Interpolator(&interpolator);
            imagetransformation->TwoD(twod);
            imagetransformation->Invert(dofin_invert);
            imagetransformation->Run();

            // blur the original slice
            RealImage target(_slices[inputIndex].Attributes());
            GaussianBlurringWithPadding<RealPixel> gb(target.GetXSize() * 0.6, source_padding);
            gb.Input(&_slices[inputIndex]);
            gb.Output(&target);
            gb.Run();

            // mask slices
            RealImage slice_mask = _mask;
            TransformMask(target, slice_mask, _transformations[inputIndex]);
            target *= slice_mask;
            output *= slice_mask;

            // compute NCC
            double output_ncc = ComputeNCC(target, output, 0);
            if (output_ncc == -1)
                output_ncc = 1;
            reg_ncc.push_back(output_ncc);
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

    // class for FFD SVR
    class ParallelSliceToVolumeRegistrationFFD {
        Reconstruction *reconstructor;

    public:
        ParallelSliceToVolumeRegistrationFFD(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

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

    // run SVR 
    void Reconstruction::SliceToVolumeRegistration() {
        SVRTK_START_TIMING();

        if (_debug)
            _reconstructed.Write("target.nii.gz");

        _grey_reconstructed = _reconstructed;

        if (!_ffd) {
            unique_ptr<ParallelSliceToVolumeRegistration> p_reg(new ParallelSliceToVolumeRegistration(this));
            (*p_reg)();
        } else {
            unique_ptr<ParallelSliceToVolumeRegistrationFFD> p_reg(new ParallelSliceToVolumeRegistrationFFD(this));
            (*p_reg)();
        }

        SVRTK_END_TIMING("SliceToVolumeRegistration");
    }

    //-------------------------------------------------------------------

    // class for remote rigid SVR
    class ParallelRemoteSliceToVolumeRegistration {
        Reconstruction *reconstructor;
        int svr_range_start;
        int svr_range_stop;
        string str_mirtk_path;
        string str_current_main_file_path;
        string str_current_exchange_file_path;
        bool rigid;
        string register_cmd;

    public:
        ParallelRemoteSliceToVolumeRegistration(
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

    // run remote SVR
    void Reconstruction::RemoteSliceToVolumeRegistration(int iter, string str_mirtk_path, string str_current_main_file_path, string str_current_exchange_file_path) {
        SVRTK_START_TIMING();

        const ImageAttributes& attr_recon = _reconstructed.Attributes();
        const string str_source = str_current_exchange_file_path + "/current-source.nii.gz";
        _reconstructed.Write(str_source.c_str());

        if (!_ffd) {
            // rigid SVR
            if (iter < 3) {
                _offset_matrices.clear();

                // save slice .nii.gz files
                for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                    ResamplingWithPadding<RealPixel> resampling(attr_recon._dx, attr_recon._dx, attr_recon._dx, -1);
                    GenericLinearInterpolateImageFunction<RealImage> interpolator;

                    RealImage target(_slices[inputIndex].Attributes());
                    resampling.Input(&_slices[inputIndex]);
                    resampling.Output(&target);
                    resampling.Interpolator(&interpolator);
                    resampling.Run();

                    // put origin to zero
                    RigidTransformation offset;
                    ResetOrigin(target, offset);

                    RealPixel tmin, tmax;
                    target.GetMinMax(&tmin, &tmax);
                    _zero_slices[inputIndex] = tmax > 1 && (tmax - tmin) > 1 ? 1 : -1;

                    string str_target = str_current_exchange_file_path + "/res-slice-" + to_string(inputIndex) + ".nii.gz";
                    target.Write(str_target.c_str());

                    _offset_matrices.push_back(offset.GetMatrix());
                }
            }

            // save slice transformations
            for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                RigidTransformation r_transform = _transformations[inputIndex];
                Matrix m = r_transform.GetMatrix() * _offset_matrices[inputIndex];
                r_transform.PutMatrix(m);

                string str_dofin = str_current_exchange_file_path + "/res-transformation-" + to_string(inputIndex) + ".dof";
                r_transform.Write(str_dofin.c_str());
            }

            int stride = 32;
            int svr_range_start = 0;
            int svr_range_stop = svr_range_start + stride;

            // run remote SVR in strides
            while (svr_range_start < _slices.size()) {
                ParallelRemoteSliceToVolumeRegistration registration(this, svr_range_start, svr_range_stop, str_mirtk_path, str_current_main_file_path, str_current_exchange_file_path);
                registration();

                svr_range_start = svr_range_stop;
                svr_range_stop = min(svr_range_start + stride, (int)_slices.size());
            }

            // read output transformations
            for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                const string str_dofout = str_current_exchange_file_path + "/res-transformation-" + to_string(inputIndex) + ".dof";
                _transformations[inputIndex].Read(str_dofout.c_str());

                //undo the offset
                Matrix m = _transformations[inputIndex].GetMatrix() * _offset_matrices[inputIndex].Invert();
                _transformations[inputIndex].PutMatrix(m);
            }
        } else {
            // FFD SVR
            if (iter < 3) {
                // save slice .nii.gz files and transformations
                for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                    ResamplingWithPadding<RealPixel> resampling(attr_recon._dx, attr_recon._dx, attr_recon._dx, -1);
                    GenericLinearInterpolateImageFunction<RealImage> interpolator;

                    RealImage target(_slices[inputIndex].Attributes());
                    resampling.Input(&_slices[inputIndex]);
                    resampling.Output(&target);
                    resampling.Interpolator(&interpolator);
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

            int stride = 32;
            int svr_range_start = 0;
            int svr_range_stop = svr_range_start + stride;

            // run parallel remote FFD SVR in strides
            while (svr_range_start < _slices.size()) {
                ParallelRemoteSliceToVolumeRegistration registration(this, svr_range_start, svr_range_stop, str_mirtk_path, str_current_main_file_path, str_current_exchange_file_path, false);
                registration();

                svr_range_start = svr_range_stop;
                svr_range_stop = min(svr_range_start + stride, (int)_slices.size());
            }

            // read output transformations
            for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                const string str_dofout = str_current_exchange_file_path + "/transformation-" + to_string(inputIndex) + ".dof";
                _mffd_transformations[inputIndex].Read(str_dofout.c_str());
            }
        }

        SVRTK_END_TIMING("RemoteSliceToVolumeRegistration");
    }

    //-------------------------------------------------------------------

    // save the current recon model (for remote reconstruction option) - can be deleted
    void Reconstruction::SaveModelRemote(string str_current_exchange_file_path, int status_flag, int current_iteration) {
        if (_verbose)
            _verbose_log << "SaveModelRemote : " << current_iteration << endl;

        // save slices
        if (status_flag > 0) {
            for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                string str_slice = str_current_exchange_file_path + "/org-slice-" + to_string(inputIndex) + ".nii.gz";
                _slices[inputIndex].Write(str_slice.c_str());
            }
            string str_mask = str_current_exchange_file_path + "/current-mask.nii.gz";
            _mask.Write(str_mask.c_str());
        }

        // save transformations
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            string str_dofin = str_current_exchange_file_path + "/org-transformation-" + to_string(current_iteration) + "-" + to_string(inputIndex) + ".dof";
            _transformations[inputIndex].Write(str_dofin.c_str());
        }

        // save recon volume
        string str_recon = str_current_exchange_file_path + "/latest-out-recon.nii.gz";
        _reconstructed.Write(str_recon.c_str());
    }


    // load remotely reconstructed volume (for remote reconstruction option) - can be deleted
    void Reconstruction::LoadResultsRemote(string str_current_exchange_file_path, int current_number_of_slices, int current_iteration) {
        if (_verbose)
            _verbose_log << "LoadResultsRemote : " << current_iteration << endl;

        string str_recon = str_current_exchange_file_path + "/latest-out-recon.nii.gz";
        _reconstructed.Read(str_recon.c_str());

    }

    // load the current recon model (for remote reconstruction option) - can be deleted
    void Reconstruction::LoadModelRemote(string str_current_exchange_file_path, int current_number_of_slices, double average_thickness, int current_iteration) {
        if (_verbose)
            _verbose_log << "LoadModelRemote : " << current_iteration << endl;

        string str_recon = str_current_exchange_file_path + "/latest-out-recon.nii.gz";
        string str_mask = str_current_exchange_file_path + "/current-mask.nii.gz";

        _reconstructed.Read(str_recon.c_str());
        _mask.Read(str_mask.c_str());

        _template_created = true;
        _grey_reconstructed = _reconstructed;
        _attr_reconstructed = _reconstructed.Attributes();

        _have_mask = true;

        for (int inputIndex = 0; inputIndex < current_number_of_slices; inputIndex++) {
            // load slices
            RealImage slice;
            string str_slice = str_current_exchange_file_path + "/org-slice-" + to_string(inputIndex) + ".nii.gz";
            slice.Read(str_slice.c_str());
            slice.PutPixelSize(slice.GetXSize(), slice.GetYSize(), average_thickness);
            _slices.push_back(slice);

            // load transformations
            const string str_dofin = str_current_exchange_file_path + "/org-transformation-" + to_string(current_iteration) + "-" + to_string(inputIndex) + ".dof";
            unique_ptr<Transformation> t(Transformation::New(str_dofin.c_str()));
            RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*>(t.get());
            _transformations.push_back(*rigidTransf);

            RealPixel tmin, tmax;
            slice.GetMinMax(&tmin, &tmax);
            _zero_slices.push_back(tmax > 1 && (tmax - tmin) > 1 ? 1 : -1);

            _package_index.push_back(0);

            const ImageAttributes& attr_s = slice.Attributes();
            _slice_attributes.push_back(attr_s);

            GreyImage grey = slice;
            _grey_slices.push_back(grey);

            memset(slice.Data(), 0, sizeof(RealPixel) * slice.NumberOfVoxels());
            _slice_dif.push_back(slice);
            _simulated_slices.push_back(slice);

            _reg_slice_weight.push_back(1);
            _slice_pos.push_back(inputIndex);

            slice = 1;
            _simulated_weights.push_back(slice);
            _simulated_inside.push_back(slice);

            _stack_index.push_back(0);

            if (_ffd)
                _mffd_transformations.push_back(MultiLevelFreeFormTransformation());
        }
    }


    //-------------------------------------------------------------------

    // save slice info in .csv
    void Reconstruction::SaveSliceInfo(int current_iteration = -1) {
        string file_name;
        ofstream GD_csv_file;

        if (current_iteration > 0) {
            file_name = (boost::format("summary-slice-info-%1%.csv") % current_iteration).str();
        } else {
            file_name = "summary-slice-info.csv";
        }

        GD_csv_file.open(file_name);
        GD_csv_file << "Stack" << "," << "Slice" << "," << "Rx" << "," << "Ry" << "," << "Rz" << "," << "Tx" << "," << "Ty" << "," << "Tz" << "," << "Weight" << "," << "Inside" << "," << "Scale" << endl;

        for (int i = 0; i < _slices.size(); i++) {
            double rx = _transformations[i].GetRotationX();
            double ry = _transformations[i].GetRotationY();
            double rz = _transformations[i].GetRotationZ();

            double tx = _transformations[i].GetTranslationX();
            double ty = _transformations[i].GetTranslationY();
            double tz = _transformations[i].GetTranslationZ();

            int inside = 0;
            if (_slice_inside[i])
                inside = 1;

            GD_csv_file << _stack_index[i] << "," << i << "," << rx << "," << ry << "," << rz << "," << tx << "," << ty << "," << tz << "," << _slice_weight[i] << "," << inside << "," << _scale[i] << endl;
        }

        GD_csv_file.close();
    }

    //-------------------------------------------------------------------

    // perform nonlocal means filtering
    void Reconstruction::NLMFiltering(Array<RealImage>& stacks) {
        NLDenoising denoising;
        for (int i = 0; i < stacks.size(); i++) {
            stacks[i] = denoising.Run(stacks[i], 3, 1);
            stacks[i].Write((boost::format("denoised-%1%.nii.gz") % i).str().c_str());
        }
    }


    //-------------------------------------------------------------------

    // class for calculation of transformation matrices
    class ParallelCoeffInit {
        Reconstruction *reconstructor;

    public:
        ParallelCoeffInit(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            const RealImage global_reconstructed(reconstructor->_grey_reconstructed.Attributes());
            //get resolution of the volume
            double vx, vy, vz;
            global_reconstructed.GetPixelSize(&vx, &vy, &vz);
            //volume is always isotropic
            const double res = vx;

            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
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

                // Check if the slice is not excluded
                if (reconstructor->_reg_slice_weight[inputIndex] > 0 && !excluded_slice) {
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
                                                                if ((n >= 0) && (n < global_reconstructed.GetZ())) {
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

                } // end of condition for preliminary exclusion of slices

                reconstructor->_volcoeffs[inputIndex] = slicecoeffs;
                reconstructor->_slice_inside[inputIndex] = slice_inside;

            }  //end of loop through the slices
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };


    //-------------------------------------------------------------------

    // run calculation of transformation matrices
    void Reconstruction::CoeffInit() {
        SVRTK_START_TIMING();

        //clear slice-volume matrix from previous iteration
        _volcoeffs.clear();
        _volcoeffs.resize(_slices.size());

        //clear indicator of slice having and overlap with volumetric mask
        _slice_inside.clear();
        _slice_inside.resize(_slices.size());
        _attr_reconstructed = _reconstructed.Attributes();

        ParallelCoeffInit coeffinit(this);
        coeffinit();

        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        _volume_weights.Initialize(_reconstructed.Attributes());

        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            bool excluded = false;

            for (int fe = 0; fe < _force_excluded.size(); fe++) {
                if (inputIndex == _force_excluded[fe]) {
                    excluded = true;
                    break;
                }
            }

            if (!excluded) {
                for (int i = 0; i < _slices[inputIndex].GetX(); i++)
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++)
                        for (int k = 0; k < _volcoeffs[inputIndex][i][j].size(); k++) {
                            const POINT3D& p = _volcoeffs[inputIndex][i][j][k];
                            _volume_weights(p.x, p.y, p.z) += p.value;
                        }
            }
        }

        if (_debug)
            _volume_weights.Write("volume_weights.nii.gz");

        //find average volume weight to modify alpha parameters accordingly
        const RealPixel *ptr = _volume_weights.Data();
        const RealPixel *pm = _mask.Data();
        double sum = 0;
        int num = 0;
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

        Array<int> voxel_num;

        //clear _reconstructed image
        memset(_reconstructed.Data(), 0, sizeof(RealPixel) * _reconstructed.NumberOfVoxels());

        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            //copy the current slice
            RealImage slice = _slices[inputIndex];
            //alias the current bias image
            const RealImage& b = _bias[inputIndex];
            //read current scale factor
            double scale = _scale[inputIndex];

            int slice_vox_num = 0;

            bool excluded = false;

            for (int fe = 0; fe < _force_excluded.size(); fe++) {
                if (inputIndex == _force_excluded[fe]) {
                    excluded = true;
                    break;
                }
            }

            if (!excluded) {
                //Distribute slice intensities to the volume
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
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
        }

        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxe
        _reconstructed /= _volume_weights;

        _reconstructed.Write("init.nii.gz");

        // find slices with small overlap with ROI and exclude them.
        //find median
        Array<int> voxel_num_tmp = voxel_num;
        int median = round(voxel_num_tmp.size() * 0.5) - 1;
        nth_element(voxel_num_tmp.begin(), voxel_num_tmp.begin() + median, voxel_num_tmp.end());
        median = voxel_num_tmp[median];

        //remember slices with small overlap with ROI
        _small_slices.clear();
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
    class ParallelCoeffInitSF {
        Reconstruction *reconstructor;
        size_t begin;
        size_t end;

    public:
        ParallelCoeffInitSF(
            Reconstruction *reconstructor, 
            size_t begin, 
            size_t end) :
            reconstructor(reconstructor),
            begin(begin),
            end(end) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                //get resolution of the volume
                double vx, vy, vz;
                reconstructor->_reconstructed.GetPixelSize(&vx, &vy, &vz);
                //volume is always isotropic
                double res = vx;

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
                                        //PSF centered over current slice voxel
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

                reconstructor->_volcoeffsSF[inputIndex % (reconstructor->_slicePerDyn)] = slicecoeffs;
                reconstructor->_slice_insideSF[inputIndex % (reconstructor->_slicePerDyn)] = slice_inside;
                //cerr<<" Done "<<inputIndex % (reconstructor->_slicePerDyn)<<endl;
            }  //end of loop through the slices
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(begin, end), *this);
        }
    };

    //-------------------------------------------------------------------

    // another version of CoeffInit
    void Reconstruction::CoeffInitSF(int begin, int end) {
        //clear slice-volume matrix from previous iteration
        _volcoeffsSF.clear();
        _volcoeffsSF.resize(_slicePerDyn);

        //clear indicator of slice having and overlap with volumetric mask
        _slice_insideSF.clear();
        _slice_insideSF.resize(_slicePerDyn);

        ParallelCoeffInitSF coeffinit(this, begin, end);
        coeffinit();

        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        _volume_weightsSF.Initialize(_reconstructed.Attributes());

        const Array<RealImage>& slices = _withMB ? _slicesRwithMB : _slices;

        for (int inputIndex = begin; inputIndex < end; inputIndex++)
            for (int i = 0; i < slices[inputIndex].GetX(); i++)
                for (int j = 0; j < slices[inputIndex].GetY(); j++)
                    for (int k = 0; k < _volcoeffsSF[inputIndex % _slicePerDyn][i][j].size(); k++) {
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

        // clean _reconstructed
        memset(_reconstructed.Data(), 0, sizeof(RealPixel) * _reconstructed.NumberOfVoxels());

        for (int dyn = 0, counter = 0; dyn < stacks.size(); dyn++) {
            Array<RigidTransformation> currentTransformations;
            Array<RealImage> currentSlices;
            Array<double> currentScales;
            Array<RealImage> currentBiases;
            const ImageAttributes& attr = stacks[dyn].Attributes();

            CoeffInitSF(counter, counter + attr._z);

            for (int s = 0; s < attr._z; s++) {
                currentTransformations.push_back(_transformations[counter + s]);
                currentSlices.push_back(_slices[counter + s]);
                currentScales.push_back(_scale[counter + s]);
                currentBiases.push_back(_bias[counter + s]);
            }

            RealImage interpolated(_reconstructed.Attributes());

            for (int s = 0; s < currentSlices.size(); s++) {
                //copy the current slice
                RealImage slice = currentSlices[s];
                //alias the current bias image
                const RealImage& b = currentBiases[s];
                //read current scale factor
                double scale = currentScales[s];

                int slice_vox_num = 0;
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
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
        _small_slices.clear();
        for (int i = 0; i < voxel_num.size(); i++)
            if (voxel_num[i] < 0.1 * median)
                _small_slices.push_back(i);

        if (_verbose) {
            _verbose_log << "Small slices:";
            for (int i = 0; i < _small_slices.size(); i++)
                _verbose_log << " " << _small_slices[i];
            _verbose_log << endl;
        }
    }

    //-------------------------------------------------------------------

    // initialise slice EM step
    void Reconstruction::InitializeEM() {
        _weights.clear();
        _bias.clear();
        _scale.clear();
        _slice_weight.clear();

        for (int i = 0; i < _slices.size(); i++) {
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

        for (int i = 0; i < _slices.size(); i++) {
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

    // initiliase / reset EM values
    void Reconstruction::InitializeEMValues() {
        SVRTK_START_TIMING();

        for (int i = 0; i < _slices.size(); i++) {
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
        for (size_t i = 0; i < _force_excluded.size(); i++) {
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slices.size())
                _slice_weight[_force_excluded[i]] = 0;
        }

        SVRTK_END_TIMING("InitializeEMValues");
    }

    //-------------------------------------------------------------------

    // initialise parameters of EM robust statistics
    void Reconstruction::InitializeRobustStatistics() {
        Array<int> sigma_numbers(_slices.size());
        Array<double> sigma_values(_slices.size());

        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            RealImage slice = _slices[inputIndex];
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
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sigma += sigma_values[inputIndex];
            num += sigma_numbers[inputIndex];
        }

        //Force exclusion of slices predefined by user
        for (int i = 0; i < _force_excluded.size(); i++) {
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slices.size())
                _slice_weight[_force_excluded[i]] = 0;
        }

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

        if (_verbose)
            _verbose_log << "Initializing robust statistics: " << "sigma=" << sqrt(_sigma) << " " << "m=" << _m
            << " " << "mix=" << _mix << " " << "mix_s=" << _mix_s << endl;
    }


    //-------------------------------------------------------------------

    // class for EStep (RS)
    class ParallelEStep {
        Reconstruction *reconstructor;
        Array<double>& slice_potential;
    public:

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];

                //read current weight image
                reconstructor->_weights[inputIndex] = 0;

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
                            const int n = reconstructor->_volcoeffs[inputIndex][i][j].size();

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

        ParallelEStep(
            Reconstruction *reconstructor,
            Array<double>& slice_potential) :
            reconstructor(reconstructor), 
            slice_potential(slice_potential) {}

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };


    //-------------------------------------------------------------------

    // run EStep for calculation of voxel-wise and slice-wise posteriors (weights)
    void Reconstruction::EStep() {
        Array<double> slice_potential(_slices.size());

        ParallelEStep parallelEStep(this, slice_potential);
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

        //Calulation of slice-wise robust statistics parameters.
        //This is theoretically M-step,
        //but we want to use latest estimate of slice potentials
        //to update the parameters

        if (_verbose) {
            _verbose_log << endl << "Slice potentials: ";
            for (size_t inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
                _verbose_log << slice_potential[inputIndex] << " ";
            _verbose_log << endl;
        }

        //Calculate means of the inlier and outlier potentials
        double sum = 0, den = 0, sum2 = 0, den2 = 0, maxs = 0, mins = 1;
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
        if ((sum2 > 0) && (den2 > 0)) {
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
            _verbose_log << "Slice weights: ";
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                _verbose_log << _slice_weight[inputIndex] << " ";
            _verbose_log << endl;
        }
    }


    //-------------------------------------------------------------------

    // class for slice scale calculation
    class ParallelScale {
        Reconstruction *reconstructor;

    public:
        ParallelScale(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

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

    // run slice scaling
    void Reconstruction::Scale() {
        SVRTK_START_TIMING();

        ParallelScale parallelScale(this);
        parallelScale();

        if (_verbose) {
            _verbose_log << setprecision(3);
            _verbose_log << "Slice scale = ";
            for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                _verbose_log << _scale[inputIndex] << " ";
            _verbose_log << endl;
        }

        SVRTK_END_TIMING("Scale");
    }

    //-------------------------------------------------------------------

    // class for slice bias calculation
    class ParallelBias {
        Reconstruction *reconstructor;

    public:
        ParallelBias(Reconstruction *reconstructor) : reconstructor(reconstructor) {}

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

    // run slice bias correction
    void Reconstruction::Bias() {
        SVRTK_START_TIMING();
        ParallelBias parallelBias(this);
        parallelBias();
        SVRTK_END_TIMING("Bias");
    }


    //-------------------------------------------------------------------

    // compute difference between simulated and original slices
    void Reconstruction::SliceDifference() {
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
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

    // class for SR reconstruction
    class ParallelSuperresolution {
        Reconstruction *reconstructor;

    public:
        RealImage confidence_map;
        RealImage addon;

        ParallelSuperresolution(Reconstruction *reconstructor) : reconstructor(reconstructor) {
            //Clear addon
            addon.Initialize(reconstructor->_reconstructed.Attributes());

            //Clear confidence map
            confidence_map.Initialize(reconstructor->_reconstructed.Attributes());
        }

        ParallelSuperresolution(ParallelSuperresolution& x, split) : ParallelSuperresolution(x.reconstructor) {}

        void join(const ParallelSuperresolution& y) {
            addon += y.addon;
            confidence_map += y.confidence_map;
        }

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

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };


    //-------------------------------------------------------------------

    // run SR reconstruction step
    void Reconstruction::Superresolution(int iter) {
        SVRTK_START_TIMING();

        // save current reconstruction for edge-preserving smoothing
        RealImage original = _reconstructed;

        SliceDifference();

        ParallelSuperresolution parallelSuperresolution(this);
        parallelSuperresolution();

        RealImage& addon = parallelSuperresolution.addon;
        _confidence_map = move(parallelSuperresolution.confidence_map);
        //_confidence4mask = _confidence_map;

        if (_debug) {
            _confidence_map.Write((boost::format("confidence-map%1%.nii.gz") % iter).str().c_str());
            addon.Write((boost::format("addon%1%.nii.gz") % iter).str().c_str());
        }

        if (!_adaptive) {
            for (int i = 0; i < addon.GetX(); i++)
                for (int j = 0; j < addon.GetY(); j++)
                    for (int k = 0; k < addon.GetZ(); k++)
                        if (_confidence_map(i, j, k) > 0) {
                            // ISSUES if _confidence_map(i, j, k) is too small leading to bright pixels
                            addon(i, j, k) /= _confidence_map(i, j, k);
                            //this is to revert to normal (non-adaptive) regularisation
                            _confidence_map(i, j, k) = 1;
                        }
        }

        // update the volume with computed addon
        _reconstructed += addon * _alpha; //_average_volume_weight;

        //bound the intensities
        for (int i = 0; i < _reconstructed.GetX(); i++)
            for (int j = 0; j < _reconstructed.GetY(); j++)
                for (int k = 0; k < _reconstructed.GetZ(); k++) {
                    if (_reconstructed(i, j, k) < _min_intensity * 0.9)
                        _reconstructed(i, j, k) = _min_intensity * 0.9;
                    if (_reconstructed(i, j, k) > _max_intensity * 1.1)
                        _reconstructed(i, j, k) = _max_intensity * 1.1;
                }

        //Smooth the reconstructed image with regularisation
        AdaptiveRegularization(iter, original);

        //Remove the bias in the reconstructed volume compared to previous iteration
        if (_global_bias_correction)
            BiasCorrectVolume(original);

        SVRTK_END_TIMING("Superresolution");
    }


    //-------------------------------------------------------------------

    // class for MStep (RS)
    class ParallelMStep {
        Reconstruction *reconstructor;

    public:
        double sigma;
        double mix;
        double num;
        double min;
        double max;

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

        void join(const ParallelMStep& y) {
            if (y.min < min)
                min = y.min;
            if (y.max > max)
                max = y.max;

            sigma += y.sigma;
            mix += y.mix;
            num += y.num;
        }

        ParallelMStep(ParallelMStep& x, split) : ParallelMStep(x.reconstructor) {}

        ParallelMStep(Reconstruction *reconstructor) : reconstructor(reconstructor) {
            sigma = 0;
            mix = 0;
            num = 0;
            min = voxel_limits<RealPixel>::max();
            max = voxel_limits<RealPixel>::min();
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    // run MStep (RS)
    void Reconstruction::MStep(int iter) {
        ParallelMStep parallelMStep(this);
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
            cerr << "Something went wrong: sigma=" << sigma << " mix=" << mix << endl;
            exit(1);
        }
        if (_sigma < _step * _step / 6.28)
            _sigma = _step * _step / 6.28;
        if (iter > 1)
            _mix = mix / num;

        //Calculate m
        _m = 1 / (max - min);

        if (_verbose) {
            _verbose_log << "Voxel-wise robust statistics parameters: ";
            _verbose_log << "sigma = " << sqrt(_sigma) << " mix = " << _mix << " ";
            _verbose_log << " m = " << _m << endl;
        }
    }

    //-------------------------------------------------------------------

    // class for adaptive regularisation (Part I)
    class ParallelAdaptiveRegularization1 {
        const Reconstruction *reconstructor;
        Array<RealImage>& b;
        const Array<double>& factor;
        const RealImage& original;

    public:
        ParallelAdaptiveRegularization1(
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
    class ParallelAdaptiveRegularization2 {
        Reconstruction *reconstructor;
        const Array<RealImage>& b;
        const RealImage& original;

    public:
        ParallelAdaptiveRegularization2(
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

    // run adaptive regularisation of the SR reconstructed volume
    void Reconstruction::AdaptiveRegularization(int iter, const RealImage& original) {
        Array<double> factor(13);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }

        Array<RealImage> b(13, _reconstructed);

        ParallelAdaptiveRegularization1 parallelAdaptiveRegularization1(this, b, factor, original);
        parallelAdaptiveRegularization1();

        const RealImage original2 = _reconstructed;
        ParallelAdaptiveRegularization2 parallelAdaptiveRegularization2(this, b, original2);
        parallelAdaptiveRegularization2();

        if (_alpha * _lambda / (_delta * _delta) > 0.068)
            cerr << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068." << endl;
    }


    //-------------------------------------------------------------------

    // correct bias
    void Reconstruction::BiasCorrectVolume(const RealImage& original) {
        //remove low-frequancy component in the reconstructed image which might have accured due to overfitting of the biasfield
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
        //blur weigted residual
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
    void Reconstruction::Evaluate(int iter) {
        cout << "Iteration " << iter << ": " << endl;

        cout << "Included slices: ";
        int sum = 0;
        for (size_t i = 0; i < _slices.size(); i++) {
            if (_slice_weight[i] >= 0.5 && _slice_inside[i]) {
                cout << i << " ";
                sum++;
            }
        }
        cout << endl << "Total: " << sum << endl;

        cout << "Excluded slices: ";
        sum = 0;
        for (size_t i = 0; i < _slices.size(); i++) {
            if (_slice_weight[i] < 0.5 && _slice_inside[i]) {
                cout << i << " ";
                sum++;
            }
        }
        cout << endl << "Total: " << sum << endl;

        cout << "Outside slices: ";
        sum = 0;
        for (size_t i = 0; i < _slices.size(); i++) {
            if (!_slice_inside[i]) {
                cout << i << " ";
                sum++;
            }
        }
        cout << endl << "Total: " << sum << endl;
    }

    //-------------------------------------------------------------------

    // class for bias normalisation
    class ParallelNormaliseBias {
        Reconstruction *reconstructor;
    public:
        RealImage bias;

        ParallelNormaliseBias(Reconstruction *reconstructor) : reconstructor(reconstructor) {
            bias.Initialize(reconstructor->_reconstructed.Attributes());
        }

        ParallelNormaliseBias(ParallelNormaliseBias& x, split) : ParallelNormaliseBias(x.reconstructor) {}

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
                            const int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            //add contribution of current slice voxel to all voxel volumes to which it contributes
                            for (int k = 0; k < n; k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                bias(p.x, p.y, p.z) += p.value * b(i, j, 0);
                            }
                        }
                //end of loop for a slice inputIndex
            }
        }

        void join(const ParallelNormaliseBias& y) {
            bias += y.bias;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };


    //-------------------------------------------------------------------

    // normalise Bias
    void Reconstruction::NormaliseBias(int iter) {
        SVRTK_START_TIMING();

        ParallelNormaliseBias parallelNormaliseBias(this);
        parallelNormaliseBias();
        RealImage& bias = parallelNormaliseBias.bias;

        // normalize the volume by proportion of contributing slice voxels for each volume voxel
        bias /= _volume_weights;

        MaskImage(bias, 0);
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
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {
            if (pi[i] != -1)
                pi[i] /= exp(-(pb[i]));
        }

        SVRTK_END_TIMING("NormaliseBias");
    }

    //-------------------------------------------------------------------

    void Reconstruction::ReadTransformations(const char *folder, size_t file_count, Array<RigidTransformation>& transformations) {
        if (_slices.size() == 0) {
            cerr << "Please create slices before reading transformations!" << endl;
            exit(1);
        }

        transformations.clear();
        for (size_t i = 0; i < file_count; i++) {
            const string path = (boost::format("%1%/transformation%2%.dof") % (folder ? folder : ".") % i).str();
            unique_ptr<Transformation> transformation(Transformation::New(path.c_str()));
            RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*>(transformation.get());
            transformations.push_back(*rigidTransf);
            cout << path << endl;
        }
    }

    //-------------------------------------------------------------------

    void Reconstruction::SetReconstructed(const RealImage& reconstructed) {
        _reconstructed = reconstructed;
        _template_created = true;
    }

    //-------------------------------------------------------------------

    void Reconstruction::SetTransformations(const Array<RigidTransformation>& transformations) {
        _transformations = transformations;
    void Reconstruction::ReadTransformations(const char *folder) {
        cout << "Reading transformations:" << endl;
        ReadTransformations(folder, _slices.size(), _transformations);
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveBiasFields() {
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _bias[inputIndex].Write((boost::format("bias%1%.nii.gz") % inputIndex).str().c_str());
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveConfidenceMap() {
        _confidence_map.Write("confidence-map.nii.gz");
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveSlices() {
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _slices[inputIndex].Write((boost::format("slice%1%.nii.gz") % inputIndex).str().c_str());
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveSlicesWithTiming() {
        cout << "Saving slices with timing: ";
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _slices[inputIndex].Write((boost::format("sliceTime%1%.nii.gz") % _slice_timing[inputIndex]).str().c_str());
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveSimulatedSlices() {
        cout << "Saving simulated slices ...";
        for (size_t inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _simulated_slices[inputIndex].Write((boost::format("simslice%1%.nii.gz") % inputIndex).str().c_str());
        cout << "done." << endl;
    }


    //-------------------------------------------------------------------

    void Reconstruction::SaveWeights() {
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
        for (size_t i = 0; i < _slices.size(); i++) {
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
        for (size_t i = 0; i < _transformations.size(); i++)
            _transformations[i].Write((boost::format("transformation%1%.dof") % i).str().c_str());
    }

    //-------------------------------------------------------------------

    void Reconstruction::GetTransformations(Array<RigidTransformation>& transformations) {
        transformations = _transformations;
    }

    //-------------------------------------------------------------------

    void Reconstruction::GetSlices(Array<RealImage>& slices) {
        slices = _slices;
    }

    //-------------------------------------------------------------------

    void Reconstruction::SetSlices(const Array<RealImage>& slices) {
        _slices = slices;
    }

    //-------------------------------------------------------------------

    void Reconstruction::SaveProbabilityMap(int i) {
        _brain_probability.Write((boost::format("probability_map%1%.nii") % i).str().c_str());
    }

    //-------------------------------------------------------------------

    // save slice info
    void Reconstruction::SlicesInfo(const char *filename, const Array<string>& stack_files) {
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

        for (int i = 0; i < _slices.size(); i++) {
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

        info.close();
    }

    //-------------------------------------------------------------------

    // get slice order parameters
    void Reconstruction::GetSliceAcquisitionOrder(const Array<RealImage>& stacks, const Array<int>& pack_num, const Array<int>& order, const int step, const int rewinder) {
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            const ImageAttributes& attr = stacks[dyn].Attributes();
            const int slicesPerPackage = attr._z / pack_num[dyn];
            int z_slice_order[attr._z];
            int t_slice_order[attr._z];

            // Ascending or descending
            if (order[dyn] == 1 || order[dyn] == 2) {
                int counter = 0;
                int slice_pos_counter = 0;
                int p = 0; // package counter

                while (counter < attr._z) {
                    z_slice_order[counter] = slice_pos_counter;
                    t_slice_order[slice_pos_counter] = counter++;
                    slice_pos_counter += pack_num[dyn];

                    // start new package
                    if (order[dyn] == 1) {
                        if (slice_pos_counter >= attr._z)
                            slice_pos_counter = ++p;
                    } else {
                        if (slice_pos_counter < 0)
                            slice_pos_counter = attr._z - 1 - ++p;
                    }
                }
            } else {
                Array<int> realInterleaved;
                int rewinderFactor, stepFactor;

                if (order[dyn] == 3) {
                    rewinderFactor = 1;
                    stepFactor = 2;
                } else if (order[dyn] == 4) {
                    rewinderFactor = 1;
                } else {
                    stepFactor = step;
                    rewinderFactor = rewinder;
                }

                // pretending to do ascending within each package, and then shuffling according to interleaved acquisition
                for (int p = 0, counter = 0; p < pack_num[dyn]; p++) {
                    Array<int> fakeAscending;

                    if (order[dyn] == 4) {
                        // getting step size, from PPE
                        if (attr._z - counter > slicesPerPackage * pack_num[dyn]) {
                            stepFactor = round(sqrt(double(slicesPerPackage + 1)));
                            counter++;
                        } else
                            stepFactor = round(sqrt(double(slicesPerPackage)));
                    }

                    // middle part of the stack
                    for (int s = 0; s < slicesPerPackage; s++) {
                        const int slice_pos_counter = s * pack_num[dyn] + p;
                        fakeAscending.push_back(slice_pos_counter);
                    }

                    // last slices for larger packages
                    if (attr._z > slicesPerPackage * pack_num[dyn]) {
                        const int slice_pos_counter = slicesPerPackage * pack_num[dyn] + p;
                        if (slice_pos_counter < attr._z)
                            fakeAscending.push_back(slice_pos_counter);
                    }

                    // shuffling
                    for (int i = 0, index = 0, restart = 0; i < fakeAscending.size(); i++, index += stepFactor) {
                        if (index >= fakeAscending.size()) {
                            restart += rewinderFactor;
                            index = restart;
                        }
                        realInterleaved.push_back(fakeAscending[index]);
                    }
                }

                // saving
                for (int i = 0; i < attr._z; i++) {
                    z_slice_order[i] = realInterleaved[i];
                    t_slice_order[realInterleaved[i]] = i;
                }
            }

            // copying
            for (int i = 0; i < attr._z; i++) {
                _z_slice_order.push_back(z_slice_order[i]);
                _t_slice_order.push_back(t_slice_order[i]);
            }
        }
    }


    //-------------------------------------------------------------------

    // split images into varying N packages
    void Reconstruction::flexibleSplitImage(const Array<RealImage>& stacks, Array<RealImage>& sliceStacks, const Array<int>& pack_num, const Array<int>& sliceNums, const Array<int>& order, const int step, const int rewinder) {
        // calculate slice order
        GetSliceAcquisitionOrder(stacks, pack_num, order, step, rewinder);

        // counters
        int counter1 = 0, counter2 = 0, counter3 = 0;

        int startIterations = 0;
        int endIterations = 0;
        int sum = 0;

        // dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            // location acquisition order
            Array<int> z_internal_slice_order;

            // slice loop
            for (int sl = 0; sl < attr._z; sl++)
                z_internal_slice_order.push_back(_z_slice_order[counter1 + sl]);

            // fake packages
            while (sum < attr._z) {
                sum += sliceNums[counter2];
                counter2++;
            }
            endIterations = counter2;

            // fake package loop
            for (int iter = startIterations; iter < endIterations; iter++) {
                const int internalIterations = sliceNums[iter];
                RealImage stack(attr);

                // copying
                for (int sl = counter3; sl < internalIterations + counter3; sl++)
                    for (int j = 0; j < stack.GetY(); j++)
                        for (int i = 0; i < stack.GetX(); i++)
                            stack.Put(i, j, z_internal_slice_order[sl], image(i, j, z_internal_slice_order[sl]));

                // pushing package
                sliceStacks.push_back(move(stack));
                counter3 += internalIterations;

                // for (int i = 0; i < sliceStacks.size(); i++)
                //     sliceStacks[i].Write((boost::format("sliceStacks%1%.nii") % i).str().c_str());
            }

            // updating varialbles for next dynamic
            counter1 += attr._z;
            counter3 = 0;
            sum = 0;
            startIterations = endIterations;
        }
    }

    //-------------------------------------------------------------------

    // split images based on multi-band acquisition
    void Reconstruction::flexibleSplitImagewithMB(const Array<RealImage>& stacks, Array<RealImage>& sliceStacks, const Array<int>& pack_num, const Array<int>& sliceNums, const Array<int>& multiband_vector, const Array<int>& order, const int step, const int rewinder) {
        // initializing variables
        Array<RealImage> chuncks, chunksAll;
        Array<int> pack_num_chucks;
        int counter1, counter2, counter3, counter4, counter5;
        Array<int> sliceNumsChunks;

        int startFactor = 0, endFactor = 0;
        // dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            RealImage multibanded(attr);
            const int multiband = multiband_vector[dyn];
            const int sliceMB = attr._z / multiband;
            int sum = 0;

            for (int m = 0; m < multiband; m++) {
                RealImage chunck = image.GetRegion(0, 0, m * sliceMB, attr._x, attr._y, (m + 1) * sliceMB);
                chuncks.push_back(move(chunck));
                pack_num_chucks.push_back(pack_num[dyn]);
            }

            while (sum < sliceMB) {
                sum += sliceNums[endFactor];
                endFactor++;
            }

            for (int m = 0; m < multiband; m++)
                for (int iter = startFactor; iter < endFactor; iter++)
                    sliceNumsChunks.push_back(sliceNums[iter]);

            startFactor = endFactor;
        }

        // splitting each multiband subgroup
        flexibleSplitImage(chuncks, chunksAll, pack_num_chucks, sliceNumsChunks, order, step, rewinder);

        counter4 = 0;
        counter5 = 0;
        // new dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            Array<RealImage> chuncks_separated, chuncks_separated_reordered;
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            RealImage multibanded(attr);
            const int multiband = multiband_vector[dyn];
            const int sliceMB = attr._z / multiband;
            int sum = 0, stepFactor = 0;

            // stepping factor in vector
            while (sum < sliceMB) {
                sum += sliceNums[counter5 + stepFactor];
                stepFactor++;
            }

            // getting data from this dynamic
            for (int iter = 0; iter < multiband * stepFactor; iter++)
                chuncks_separated.push_back(chunksAll[iter + counter4]);

            counter4 += multiband * stepFactor;

            // reordering chuncks_separated
            counter1 = 0;
            counter2 = 0;
            counter3 = 0;
            while (counter1 < chuncks_separated.size()) {
                chuncks_separated_reordered.push_back(chuncks_separated[counter2]);
                counter2 += stepFactor;
                if (counter2 > chuncks_separated.size() - 1)
                    counter2 = ++counter3;
                counter1++;
            }

            // riassembling multiband packs
            counter1 = 0;
            counter2 = 0;
            while (counter1 < chuncks_separated_reordered.size()) {
                for (int m = 0; m < multiband; m++) {
                    const RealImage& toAdd = chuncks_separated_reordered[counter1];
                    for (int k = 0; k < toAdd.GetZ(); k++) {
                        for (int j = 0; j < toAdd.GetY(); j++)
                            for (int i = 0; i < toAdd.GetX(); i++)
                                multibanded.Put(i, j, counter2, toAdd(i, j, k));

                        counter2++;
                    }
                    counter1++;
                }
                sliceStacks.push_back(multibanded);
                counter2 = 0;
            }
            counter5 += stepFactor;
            stepFactor = 0;
        }
    }


    //-------------------------------------------------------------------

    // split stacks into packages based on specific order
    void Reconstruction::splitPackages(const Array<RealImage>& stacks, const Array<int>& pack_num, Array<RealImage>& packageStacks, const Array<int>& order, const int step, const int rewinder) {
        // calculate slice order
        GetSliceAcquisitionOrder(stacks, pack_num, order, step, rewinder);

        // dynamic loop
        for (int dyn = 0, counter1 = 0; dyn < stacks.size(); dyn++) {
            // location acquisition order
            Array<int> z_internal_slice_order;
            // current stack
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            const int pkg_z = attr._z / pack_num[dyn];

            // slice loop
            for (int sl = 0; sl < attr._z; sl++)
                z_internal_slice_order.push_back(_z_slice_order[counter1 + sl]);

            // package loop
            int counter2 = 0, counter3 = 0;
            for (int p = 0; p < pack_num[dyn]; p++) {
                int internalIterations;
                // slice excess for each package
                if (attr._z - counter2 > pkg_z * pack_num[dyn]) {
                    internalIterations = pkg_z + 1;
                    counter2++;
                } else {
                    internalIterations = pkg_z;
                }

                // copying
                RealImage stack(attr);
                for (int sl = counter3; sl < internalIterations + counter3; sl++)
                    for (int j = 0; j < stack.GetY(); j++)
                        for (int i = 0; i < stack.GetX(); i++)
                            stack.Put(i, j, z_internal_slice_order[sl], image(i, j, z_internal_slice_order[sl]));

                // pushing package
                packageStacks.push_back(move(stack));
                // updating varialbles for next package
                counter3 += internalIterations;
            }
            counter1 += attr._z;
        }
    }


    //-------------------------------------------------------------------

    // multi-band based
    void Reconstruction::splitPackageswithMB(const Array<RealImage>& stacks, const Array<int>& pack_num, Array<RealImage>& packageStacks, const Array<int>& multiband_vector, const Array<int>& order, const int step, const int rewinder) {
        // initializing variables
        Array<RealImage> chuncks, chunksAll;
        Array<int> pack_numAll;
        int counter1, counter2, counter3, counter4;

        // dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            RealImage multibanded(attr);
            const int multiband = multiband_vector[dyn];
            const int sliceMB = attr._z / multiband;

            for (int m = 0; m < multiband; m++) {
                RealImage chunck = image.GetRegion(0, 0, m * sliceMB, attr._x, attr._y, (m + 1) * sliceMB);
                chuncks.push_back(move(chunck));
                pack_numAll.push_back(pack_num[dyn]);
            }
        }

        // split package
        splitPackages(chuncks, pack_numAll, chunksAll, order, step, rewinder);

        counter4 = 0;
        // new dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            Array<RealImage> chuncks_separated, chuncks_separated_reordered;
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            RealImage multibanded(attr);
            const int multiband = multiband_vector[dyn];
            const int sliceMB = attr._z / multiband;

            // getting data from this dynamic
            const int stepFactor = pack_num[dyn];
            for (int iter = 0; iter < multiband * stepFactor; iter++) {
                chuncks_separated.push_back(chunksAll[iter + counter4]);
            }
            counter4 += multiband * stepFactor;

            // reordering chuncks_separated
            counter1 = 0;
            counter2 = 0;
            counter3 = 0;
            while (counter1 < chuncks_separated.size()) {
                chuncks_separated_reordered.push_back(chuncks_separated[counter2]);

                counter2 += stepFactor;
                if (counter2 > chuncks_separated.size() - 1) {
                    counter3++;
                    counter2 = counter3;
                }
                counter1++;
            }

            // riassembling multiband slices
            counter1 = 0;
            counter2 = 0;
            while (counter1 < chuncks_separated_reordered.size()) {
                for (int m = 0; m < multiband; m++) {
                    const RealImage& toAdd = chuncks_separated_reordered[counter1];
                    for (int k = 0; k < toAdd.GetZ(); k++) {
                        for (int j = 0; j < toAdd.GetY(); j++)
                            for (int i = 0; i < toAdd.GetX(); i++)
                                multibanded.Put(i, j, counter2, toAdd(i, j, k));

                        counter2++;
                    }
                    counter1++;
                }
                packageStacks.push_back(move(multibanded));
                counter2 = 0;
            }
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

        for (int w = 0; w < wrapper; w++) {
            Array<RealImage> sstacks;
            Array<int> spack_num;
            Array<int> smultiband_vector;
            Array<int> sorder;
            Array<RealImage> packages;
            int doffset = w * steps;
            // preparing input for this iterations
            for (int s = 0; s < steps; s++) {
                if (s + doffset < stacks.size()) {
                    sstacks.push_back(stacks[s + doffset]);
                    spack_num.push_back(pack_num[s + doffset]);
                    smultiband_vector.push_back(multiband_vector[s + doffset]);
                    sorder.push_back(order[s + doffset]);
                }
            }

            splitPackageswithMB(sstacks, spack_num, packages, smultiband_vector, sorder, step, rewinder);

            // other variables
            int counter1 = 0, counter2 = 0, counter3 = 0;

            for (int i = 0; i < sstacks.size(); i++) {
                Array<int> t_internal_slice_order;
                Array<RigidTransformation> internal_transformations;
                const RealImage& firstPackage = packages[counter1];
                const int multiband = smultiband_vector[i];
                int extra = (firstPackage.GetZ() / multiband) % spack_num[i];
                int startIterations = 0, endIterations = 0;

                // slice loop
                for (int sl = 0; sl < firstPackage.GetZ(); sl++) {
                    t_internal_slice_order.push_back(_t_slice_order[counter3 + sl]);
                    internal_transformations.push_back(_transformations[counter2 + sl]);
                }

                // package look
                for (int j = 0; j < spack_num[i]; j++) {
                    // performing registration
                    const RealImage& target = packages[counter1];
                    const GreyImage s = _reconstructed;
                    GreyImage t = target;

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
                        Matrix mo = offset.GetMatrix();
                        Matrix m = internal_transformations[j].GetMatrix() * mo;
                        internal_transformations[j].PutMatrix(m);

                        rigidregistration.Input(&t, &s);
                        Transformation *dofout = nullptr;
                        rigidregistration.Output(&dofout);
                        rigidregistration.InitialGuess(&internal_transformations[j]);
                        rigidregistration.GuessParameter();
                        rigidregistration.Run();

                        RigidTransformation *rigid_dofout = dynamic_cast<RigidTransformation*>(dofout);
                        internal_transformations[j] = *rigid_dofout;

                        m = internal_transformations[j].GetMatrix() * mo.Invert();
                        internal_transformations[j].PutMatrix(m);
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
                        for (int l = 0; l < t_internal_slice_order.size(); l++) {
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
            }

            //save overal slice order
            const ImageAttributes& attr = stacks[0].Attributes();
            int dyn, num;
            int slices_per_dyn = attr._z / multiband_vector[0];

            //slice order should repeat for each dynamic - only take first dynamic
            _slice_timing.clear();
            for (dyn = 0; dyn < stacks.size(); dyn++)
                for (int i = 0; i < attr._z; i++) {
                    _slice_timing.push_back(dyn * slices_per_dyn + _t_slice_order[i]);
                    cout << "slice timing = " << _slice_timing[i] << endl;
                }

            for (int i = 0; i < _z_slice_order.size(); i++)
                cout << "z(" << i << ")=" << _z_slice_order[i] << endl;

            for (int i = 0; i < _t_slice_order.size(); i++)
                cout << "t(" << i << ")=" << _t_slice_order[i] << endl;

            // save transformations and clear
            _z_slice_order.clear();
            _t_slice_order.clear();
        }
    }

    //-------------------------------------------------------------------

    // split image into N packages
    void Reconstruction::SplitImage(const RealImage& image, const int packages, Array<RealImage>& stacks) {
        //slices in package
        const int pkg_z = image.Attributes()._z / packages;
        const double pkg_dz = image.Attributes()._dz * packages;

        for (int l = 0; l < packages; l++) {
            ImageAttributes attr = image.Attributes();
            if (pkg_z * packages + l < attr._z)
                attr._z = pkg_z + 1;
            else
                attr._z = pkg_z;
            attr._dz = pkg_dz;

            //fill values in each stack
            RealImage stack(attr);
            double ox, oy, oz;
            stack.GetOrigin(ox, oy, oz);

            for (int k = 0; k < stack.GetZ(); k++)
                for (int j = 0; j < stack.GetY(); j++)
                    for (int i = 0; i < stack.GetX(); i++)
                        stack.Put(i, j, k, image(i, j, k * packages + l));

            //adjust origin
            //original image coordinates
            double x = 0;
            double y = 0;
            double z = l;
            image.ImageToWorld(x, y, z);
            //stack coordinates
            double sx = 0;
            double sy = 0;
            double sz = 0;
            stack.PutOrigin(ox, oy, oz); //adjust to original value
            stack.ImageToWorld(sx, sy, sz);
            //adjust origin
            stack.PutOrigin(ox + (x - sx), oy + (y - sy), oz + (z - sz));
            stacks.push_back(move(stack));
        }
    }

    //-------------------------------------------------------------------

    // split image into 2 packages
    void Reconstruction::SplitImageEvenOdd(const RealImage& image, const int packages, Array<RealImage>& stacks) {
        Array<RealImage> packs;
        cout << "Split Image Even Odd: " << packages << " packages." << endl;

        stacks.clear();
        SplitImage(image, packages, packs);

        for (int i = 0; i < packs.size(); i++) {
            cout << "Package " << i << ": " << endl;
            Array<RealImage> packs2;
            SplitImage(packs[i], 2, packs2);
            stacks.push_back(move(packs2[0]));
            stacks.push_back(move(packs2[1]));
        }

        cout << "done." << endl;
    }

    //-------------------------------------------------------------------

    // split image into 4 packages
    void Reconstruction::SplitImageEvenOddHalf(const RealImage& image, const int packages, Array<RealImage>& stacks, const int iter) {
        Array<RealImage> packs;

        cout << "Split Image Even Odd Half " << iter << endl;
        stacks.clear();
        if (iter > 1)
            SplitImageEvenOddHalf(image, packages, packs, iter - 1);
        else
            SplitImageEvenOdd(image, packages, packs);

        for (int i = 0; i < packs.size(); i++) {
            Array<RealImage> packs2;
            HalfImage(packs[i], packs2);
            for (int j = 0; j < packs2.size(); j++)
                stacks.push_back(move(packs2[j]));
        }
    }

    //-------------------------------------------------------------------

    // split image into 2 packages
    void Reconstruction::HalfImage(const RealImage& image, Array<RealImage>& stacks) {
        const ImageAttributes& attr = image.Attributes();
        stacks.clear();

        //We would not like single slices - that is reserved for slice-to-volume
        if (attr._z >= 4) {
            stacks.push_back(image.GetRegion(0, 0, 0, attr._x, attr._y, attr._z / 2));
            stacks.push_back(image.GetRegion(0, 0, attr._z / 2, attr._x, attr._y, attr._z));
        } else
            stacks.push_back(image);
    }

    //-------------------------------------------------------------------

    // run package-to-volume registration
    void Reconstruction::PackageToVolume(const Array<RealImage>& stacks, const Array<int>& pack_num, const Array<RigidTransformation>& stack_transformations) {
        SVRTK_START_TIMING();

        int firstSlice = 0;
        Array<int> firstSlice_array;
        for (int i = 0; i < stacks.size(); i++) {
            firstSlice_array.push_back(firstSlice);
            firstSlice += stacks[i].GetZ();
        }

        for (int i = 0; i < stacks.size(); i++) {
            Array<RealImage> packages;
            SplitImage(stacks[i], pack_num[i], packages);

            for (int j = 0; j < packages.size(); j++) {
                GenericRegistrationFilter rigidregistration;
                ParameterList params;
                Insert(params, "Transformation model", "Rigid");
                // Insert(params, "Image interpolation mode", "Linear");
                Insert(params, "Background value for image 1", 0);
                Insert(params, "Background value for image 2", -1);

                if (_nmi_bins > 0)
                    Insert(params, "No. of bins", _nmi_bins);

                rigidregistration.Parameter(params);

                if (_debug)
                    packages[j].Write((boost::format("package-%1%-%2%.nii.gz") % i % j).str().c_str());

                //packages are not masked at present

                RealImage mask = _mask;
                const RigidTransformation& mask_transform = stack_transformations[i]; //s[i];
                TransformMask(packages[j], mask, mask_transform);

                RealImage target = packages[j] * mask;
                const RealImage& source = _reconstructed;

                //find existing transformation
                double x = 0, y = 0, z = 0;
                packages[j].ImageToWorld(x, y, z);
                stacks[i].WorldToImage(x, y, z);

                int firstSliceIndex = round(z) + firstSlice_array[i];
                // cout<<"First slice index for package "<<j<<" of stack "<<i<<" is "<<firstSliceIndex<<endl;

                //put origin in target to zero
                RigidTransformation offset;
                ResetOrigin(target, offset);
                Matrix mo = offset.GetMatrix();
                Matrix m = _transformations[firstSliceIndex].GetMatrix() * mo;
                _transformations[firstSliceIndex].PutMatrix(m);

                rigidregistration.Input(&target, &source);
                Transformation *dofout = nullptr;
                rigidregistration.Output(&dofout);
                rigidregistration.InitialGuess(&_transformations[firstSliceIndex]);
                rigidregistration.GuessParameter();
                rigidregistration.Run();

                RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*>(dofout);
                _transformations[firstSliceIndex] = *rigidTransf;

                //undo the offset
                m = _transformations[firstSliceIndex].GetMatrix() * mo.Invert();
                _transformations[firstSliceIndex].PutMatrix(m);

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

                    if (sliceIndex >= _transformations.size()) {
                        cerr << "Reconstruction::PackageToVolume: sliceIndex out of range." << endl;
                        cerr << sliceIndex << " " << _transformations.size() << endl;
                        exit(1);
                    }

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

    //-------------------------------------------------------------------

    // Crops the image according to the mask
    void Reconstruction::CropImage(RealImage& image, const RealImage& mask) {
        //ROI boundaries
        //Original ROI
        int x1 = 0;
        int y1 = 0;
        int z1 = 0;
        int x2 = image.GetX();
        int y2 = image.GetY();
        int z2 = image.GetZ();

        //upper boundary for z coordinate
        int i, j, k, sum = 0;
        for (k = image.GetZ() - 1; k >= 0; k--) {
            sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        z2 = k;

        //lower boundary for z coordinate
        sum = 0;
        for (k = 0; k <= image.GetZ() - 1; k++) {
            sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        z1 = k;

        //upper boundary for y coordinate
        sum = 0;
        for (j = image.GetY() - 1; j >= 0; j--) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        y2 = j;

        //lower boundary for y coordinate
        sum = 0;
        for (j = 0; j <= image.GetY() - 1; j++) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        y1 = j;

        //upper boundary for x coordinate
        sum = 0;
        for (i = image.GetX() - 1; i >= 0; i--) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (j = image.GetY() - 1; j >= 0; j--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        x2 = i;

        //lower boundary for x coordinate
        sum = 0;
        for (i = 0; i <= image.GetX() - 1; i++) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (j = image.GetY() - 1; j >= 0; j--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        x1 = i;

        // if no intersection with mask, force exclude
        if ((x2 <= x1) || (y2 <= y1) || (z2 <= z1))
            x1 = y1 = z1 = x2 = y2 = z2 = 0;

        //Cut region of interest
        image = image.GetRegion(x1, y1, z1, x2 + 1, y2 + 1, z2 + 1);
    }

    //-------------------------------------------------------------------

    // GF 190416, useful for handling different slice orders
    void Reconstruction::CropImageIgnoreZ(RealImage& image, const RealImage& mask) {
        //Crops the image according to the mask
        // Filling slices out of mask with zeros
        int i, j, k, sum = 0;
        for (k = image.GetZ() - 1; k >= 0; k--) {
            sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0) {
                k++;
                break;
            }
        }
        int z2 = k;

        //lower boundary for z coordinate
        sum = 0;
        for (k = 0; k <= image.GetZ() - 1; k++) {
            sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int z1 = k;

        // Filling upper part
        for (int k = z2; k < image.GetZ(); k++)
            for (int j = 0; j < image.GetY(); j++)
                for (int i = 0; i < image.GetX(); i++)
                    image.Put(i, j, k, 0);

        // Filling lower part
        for (int k = 0; k < z1; k++)
            for (int j = 0; j < image.GetY(); j++)
                for (int i = 0; i < image.GetX(); i++)
                    image.Put(i, j, k, 0);

        //Original ROI
        int x1 = 0;
        int y1 = 0;
        z1 = 0;
        int x2 = image.GetX();
        int y2 = image.GetY();
        z2 = image.GetZ() - 1;

        //upper boundary for y coordinate
        sum = 0;
        for (j = image.GetY() - 1; j >= 0; j--) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        y2 = j;

        //lower boundary for y coordinate
        sum = 0;
        for (j = 0; j <= image.GetY() - 1; j++) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        y1 = j;

        //upper boundary for x coordinate
        sum = 0;
        for (i = image.GetX() - 1; i >= 0; i--) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (j = image.GetY() - 1; j >= 0; j--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        x2 = i;

        //lower boundary for x coordinate
        sum = 0;
        for (i = 0; i <= image.GetX() - 1; i++) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (j = image.GetY() - 1; j >= 0; j--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        x1 = i;

        // if no intersection with mask, force exclude
        if ((x2 <= x1) || (y2 <= y1) || (z2 <= z1))
            x1 = y1 = z1 = x2 = y2 = z2 = 0;

        //Cut region of interest
        image = image.GetRegion(x1, y1, z1, x2 + 1, y2 + 1, z2 + 1);
    }

    //-------------------------------------------------------------------

    // invert transformations
    void Reconstruction::InvertStackTransformations(Array<RigidTransformation>& stack_transformations) {
        //for each stack
        for (size_t i = 0; i < stack_transformations.size(); i++) {
            //invert transformation for the stacks
            stack_transformations[i].Invert();
            stack_transformations[i].UpdateParameter();
        }
    }

    //-------------------------------------------------------------------

    // mask reconstructed volume
    void Reconstruction::MaskVolume() {
        RealPixel *pr = _reconstructed.Data();
        const RealPixel *pm = _mask.Data();
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {
            if (pm[i] == 0)
                pr[i] = -1;
        }
    }

    //-------------------------------------------------------------------

    // mask input volume
    void Reconstruction::MaskImage(RealImage& image, double padding) {
        if (image.NumberOfVoxels() != _mask.NumberOfVoxels()) {
            cerr << "Cannot mask the image - different dimensions" << endl;
            exit(1);
        }

        RealPixel *pr = image.Data();
        const RealPixel *pm = _mask.Data();
        for (int i = 0; i < image.NumberOfVoxels(); i++) {
            if (pm[i] == 0)
                pr[i] = padding;
        }
    }

    //-------------------------------------------------------------------

    // rescale input volume
    void Reconstruction::Rescale(RealImage& img, double max) {
        // Get lower and upper bound
        RealPixel min_val, max_val;
        img.GetMinMax(&min_val, &max_val);

        RealPixel *ptr = img.Data();
        for (int i = 0; i < img.NumberOfVoxels(); i++) {
            if (ptr[i] > 0)
                ptr[i] = double(ptr[i]) / double(max_val) * max;
        }
    }

    //-------------------------------------------------------------------

    // run stack background filtering (GS based)
    void Reconstruction::BackgroundFiltering(Array<RealImage>& stacks, const double fg_sigma, const double bg_sigma) {
        GaussianBlurring<RealPixel> gb3(stacks[0].GetXSize() * fg_sigma);
        GaussianBlurring<RealPixel> gb2(stacks[0].GetXSize() * bg_sigma);

        for (int j = 0; j < stacks.size(); j++) {
            RealImage stack = stacks[j];
            stack.Write((boost::format("original-%1%.nii.gz") % j).str().c_str());

            RealImage global_blurred = stacks[j];
            gb2.Input(&global_blurred);
            gb2.Output(&global_blurred);
            gb2.Run();

            for (int i = 0; i < stacks[j].GetZ(); i++) {
                RealImage tmp_slice = stacks[j].GetRegion(0, 0, i, stacks[j].GetX(), stacks[j].GetY(), i + 1);
                RealImage tmp_slice_b = tmp_slice;

                gb3.Input(&tmp_slice_b);
                gb3.Output(&tmp_slice_b);
                gb3.Run();

                gb2.Input(&tmp_slice);
                gb2.Output(&tmp_slice);
                gb2.Run();

                for (int x = 0; x < stacks[j].GetX(); x++) {
                    for (int y = 0; y < stacks[j].GetY(); y++) {
                        stack(x, y, i) = tmp_slice_b(x, y, 0) + global_blurred(x, y, i) - tmp_slice(x, y, 0);
                        if (stack(x, y, i) < 0)
                            stack(x, y, i) = 1;
                    }
                }
            }

            stacks[j] = move(stack);

            stack.Write((boost::format("filtered-%1%.nii.gz") % j).str().c_str());
        }
    }

    //-------------------------------------------------------------------

    // another implementation of NCC between images (should be removed or the previous should be replaced)
    double Reconstruction::ComputeNCC(const RealImage& slice_1, const RealImage& slice_2, const double threshold, double *count) {
        const int slice_1_N = slice_1.NumberOfVoxels();
        const int slice_2_N = slice_2.NumberOfVoxels();

        const double *slice_1_ptr = slice_1.Data();
        const double *slice_2_ptr = slice_2.Data();

        int slice_1_n = 0;
        double slice_1_m = 0;
        for (int j = 0; j < slice_1_N; j++) {
            if (slice_1_ptr[j] > threshold && slice_2_ptr[j] > threshold) {
                slice_1_m += slice_1_ptr[j];
                slice_1_n++;
            }
        }
        slice_1_m /= slice_1_n;

        int slice_2_n = 0;
        double slice_2_m = 0;
        for (int j = 0; j < slice_2_N; j++) {
            if (slice_1_ptr[j] > threshold && slice_2_ptr[j] > threshold) {
                slice_2_m += slice_2_ptr[j];
                slice_2_n++;
            }
        }
        slice_2_m /= slice_2_n;

        if (count) {
            *count = 0;
            for (int j = 0; j < slice_1_N; j++)
                if (slice_1_ptr[j] > threshold && slice_2_ptr[j] > threshold)
                    (*count)++;
        }

        double CC_slice;
        if (slice_1_n < 5 || slice_2_n < 5) {
            CC_slice = -1;
        } else {
            double diff_sum = 0;
            double slice_1_sq = 0;
            double slice_2_sq = 0;

            for (int j = 0; j < slice_1_N; j++) {
                if (slice_1_ptr[j] > threshold && slice_2_ptr[j] > threshold) {
                    diff_sum += (slice_1_ptr[j] - slice_1_m) * (slice_2_ptr[j] - slice_2_m);
                    slice_1_sq += pow(slice_1_ptr[j] - slice_1_m, 2);
                    slice_2_sq += pow(slice_2_ptr[j] - slice_2_m, 2);
                }
            }

            if (slice_1_sq * slice_2_sq > 0)
                CC_slice = diff_sum / sqrt(slice_1_sq * slice_2_sq);
            else
                CC_slice = 0;
        }

        return CC_slice;
    }

    //-------------------------------------------------------------------

    // class for global stack similarity statists (for the stack selection function)
    class ParallelGlobalSimilarityStats {
        Reconstruction *reconstructor;
        int nStacks;
        const Array<RealImage>& stacks;
        const Array<RealImage>& masks;
        Array<double>& all_global_ncc_array;
        Array<double>& all_global_volume_array;

    public:
        ParallelGlobalSimilarityStats(
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

    // run parallel global similarity statists (for the stack selection function)
    void Reconstruction::RunParallelGlobalStackStats(const Array<RealImage>& stacks, const Array<RealImage>& masks, Array<double>& all_global_ncc_array, Array<double>& all_global_volume_array) {
        all_global_ncc_array = Array<double>(stacks.size());
        all_global_volume_array = Array<double>(stacks.size());

        cout << " start ... " << endl;

        ParallelGlobalSimilarityStats registration(this, stacks.size(), stacks, masks, all_global_ncc_array, all_global_volume_array);
        registration();
    }

    //-------------------------------------------------------------------

    // run serial global similarity statists (for the stack selection function)
    void Reconstruction::GlobalStackStats(RealImage template_stack, const RealImage& template_mask, const Array<RealImage>& stacks, const Array<RealImage>& masks, double& average_ncc, double& average_volume, Array<RigidTransformation>& current_stack_tranformations) {
        template_stack *= template_mask;

        RigidTransformation r_init;
        r_init.PutTranslationX(0.0001);
        r_init.PutTranslationY(0.0001);
        r_init.PutTranslationZ(-0.0001);

        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        Insert(params, "Background value for image 1", 0);
        Insert(params, "Background value for image 2", 0);

        average_ncc = 0;
        average_volume = 0;

        for (int i = 0; i < stacks.size(); i++) {
            RealImage input_stack = stacks[i] * masks[i];

            GenericRegistrationFilter registration;
            registration.Parameter(params);
            registration.Input(&template_stack, &input_stack);
            Transformation *dofout = nullptr;
            registration.Output(&dofout);
            registration.InitialGuess(&r_init);
            registration.GuessParameter();
            registration.Run();
            RigidTransformation *r_dofout = dynamic_cast<RigidTransformation*>(dofout);

            GenericLinearInterpolateImageFunction<RealImage> interpolator;
            double source_padding = 0;
            double target_padding = -inf;
            bool dofin_invert = false;
            bool twod = false;
            RealImage output(template_stack.Attributes());

            unique_ptr<ImageTransformation> imagetransformation(new ImageTransformation);
            imagetransformation->Input(&input_stack);
            imagetransformation->Transformation(r_dofout);
            imagetransformation->Output(&output);
            imagetransformation->TargetPaddingValue(target_padding);
            imagetransformation->SourcePaddingValue(source_padding);
            imagetransformation->Interpolator(&interpolator);
            imagetransformation->TwoD(twod);
            imagetransformation->Invert(dofin_invert);
            imagetransformation->Run();
            input_stack = move(output);

            double slice_count = 0;
            const double local_ncc = ComputeNCC(template_stack, input_stack, 0.01, &slice_count);
            average_ncc += local_ncc;
            average_volume += slice_count;
            current_stack_tranformations.push_back(*r_dofout);
        }

        average_ncc /= stacks.size();
        average_volume /= stacks.size();
        average_volume *= template_stack.GetXSize() * template_stack.GetYSize() * template_stack.GetZSize() / 1000;
    }

    //-------------------------------------------------------------------

    // compute internal stack statistics (volume and inter-slice NCC)
    void Reconstruction::StackStats(RealImage input_stack, const RealImage& mask, double& mask_volume, double& slice_ncc) {
        input_stack *= mask;

        int slice_num = 0;
        for (int z = 0; z < input_stack.GetZ() - 1; z++) {
            constexpr int sh = 1;
            const RealImage slice_1 = input_stack.GetRegion(sh, sh, z, input_stack.GetX() - sh, input_stack.GetY() - sh, z + 1);
            const RealImage slice_2 = input_stack.GetRegion(sh, sh, z + 1, input_stack.GetX() - sh, input_stack.GetY() - sh, z + 2);

            const double local_ncc = ComputeNCC(slice_1, slice_2);
            if (local_ncc > 0) {
                slice_ncc += local_ncc;
                slice_num++;
            }
        }

        if (slice_num > 0)
            slice_ncc /= slice_num;
        else
            slice_ncc = 0;

        int mask_count = 0;
        for (int x = 0; x < mask.GetX(); x++)
            for (int y = 0; y < mask.GetY(); y++)
                for (int z = 0; z < mask.GetZ(); z++)
                    if (mask(x, y, z) > 0.01)
                        mask_count++;

        mask_volume = mask_count * mask.GetXSize() * mask.GetYSize() * mask.GetZSize() / 1000;
    }

    //-------------------------------------------------------------------

} // namespace svrtk
