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

// MIRTK
#include "mirtk/GenericImage.h"
#include "mirtk/ImageTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/LinearInterpolateImageFunction.h"
#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/GaussianBlurring.h"

// SVRTK
#include "svrtk/MeanShift.h"
#include "svrtk/NLDenoising.h"

// Boost
#include <boost/format.hpp>

using namespace std;
using namespace mirtk;

// Forward declarations
namespace svrtk {
    class Reconstruction;
}

namespace svrtk::Utility {

    // Extract specific image ROI
    void bbox(const RealImage& stack, const RigidTransformation& transformation, double& min_x, double& min_y, double& min_z, double& max_x, double& max_y, double& max_z);

    // Crop to non zero ROI
    void bboxCrop(RealImage& image);

    // Find centroid
    void centroid(const RealImage& image, double& x, double& y, double& z);

    // Center stacks
    void CenterStacks(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber);

    // Crops the image according to the mask
    void CropImage(RealImage& image, const RealImage& mask);

    // GF 190416, useful for handling different slice orders
    void CropImageIgnoreZ(RealImage& image, const RealImage& mask);

    // Implementation of NCC between images
    double ComputeNCC(const RealImage& slice_1, const RealImage& slice_2, const double threshold = 0.01, double *count = nullptr);

    // compute inter-slice volume NCC (motion metric)
    double VolumeNCC(RealImage& input_stack, RealImage template_stack, const RealImage& mask);

    // compute internal stack statistics (volume and inter-slice NCC)
    void StackStats(RealImage input_stack, RealImage& mask, double& mask_volume, double& slice_ncc);

    // run serial global similarity statistics (for the stack selection function)
    void GlobalStackStats(RealImage template_stack, const RealImage& template_mask, const Array<RealImage>& stacks,
        const Array<RealImage>& masks, double& average_ncc, double& average_volume,
        Array<RigidTransformation>& current_stack_transformations);

    void RunParallelGlobalStackStats(const Array<RealImage>& stacks, const Array<RealImage>& masks,
        Array<double>& all_global_ncc_array, Array<double>& all_global_volume_array);

    //Create average image from the stacks and volumetric transformations
    RealImage CreateAverage(const Reconstruction *reconstructor, const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations);

    void CreateMaskFromBlackBackground(const Reconstruction *reconstructor, const Array<RealImage>& stacks, Array<RigidTransformation> stack_transformations, double smooth_mask);

    // Transform and resample mask to the space of the image
    void TransformMask(const RealImage& image, RealImage& mask, const RigidTransformation& transformation);

    void BackgroundFiltering(Array<RealImage>& stacks, const double fg_sigma, const double bg_sigma);

    void MaskStacks(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, const RealImage& mask);

    void MaskSlices(Array<RealImage>& slices, const RealImage& mask, function<void(size_t, double&, double&, double&)> Transform);

    void GetSliceAcquisitionOrder(const Array<RealImage>& stacks, const Array<int>& pack_num, const Array<int>& order,
        const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order);

    // split images into varying N packages
    void FlexibleSplitImage(const Array<RealImage>& stacks, Array<RealImage>& sliceStacks, const Array<int>& pack_num,
        const Array<int>& sliceNums, const Array<int>& order, const int step, const int rewinder,
        Array<int>& output_z_slice_order, Array<int>& output_t_slice_order);

    // split images based on multi-band acquisition
    void FlexibleSplitImagewithMB(const Array<RealImage>& stacks, Array<RealImage>& sliceStacks, const Array<int>& pack_num,
        const Array<int>& sliceNums, const Array<int>& multiband_vector, const Array<int>& order, const int step, const int rewinder,
        Array<int>& output_z_slice_order, Array<int>& output_t_slice_order);

    // split stacks into packages based on specific order
    void SplitPackages(const Array<RealImage>& stacks, const Array<int>& pack_num, Array<RealImage>& packageStacks, const Array<int>& order,
        const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order);

    // split stacks into packages based on specific order (multi-band based)
    void SplitPackageswithMB(const Array<RealImage>& stacks, const Array<int>& pack_num, Array<RealImage>& packageStacks, const Array<int>& multiband_vector,
        const Array<int>& order, const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order);

    // split image into N packages
    void SplitImage(const RealImage& image, const int packages, Array<RealImage>& stacks);

    // split image into 2 packages
    void SplitImageEvenOdd(const RealImage& image, const int packages, Array<RealImage>& stacks);

    // split image into 4 packages
    void SplitImageEvenOddHalf(const RealImage& image, const int packages, Array<RealImage>& stacks, const int iter);

    // split image into 2 packages
    void HalfImage(const RealImage& image, Array<RealImage>& stacks);

    ////////////////////////////////////////////////////////////////////////////////
    // Inline/template definitions
    ////////////////////////////////////////////////////////////////////////////////

    // Clear and preallocate memory for a vector
    inline void ClearAndReserve(vector<auto>& vectorVar, size_t reserveSize) {
        vectorVar.clear();
        vectorVar.reserve(reserveSize);
    }

    //-------------------------------------------------------------------

    // Clear and resize memory for vectors
    template<typename VectorType>
    inline void ClearAndResize(vector<VectorType>& vectorVar, size_t reserveSize, const VectorType& defaultValue = VectorType()) {
        vectorVar.clear();
        vectorVar.resize(reserveSize, defaultValue);
    }

    //-------------------------------------------------------------------

    // Binarise mask
    // If template image has been masked instead of creating the mask in separate
    // file, this function can be used to create mask from the template image
    inline RealImage CreateMask(RealImage image, double threshold = 0.5) {
        RealPixel *ptr = image.Data();
        #pragma omp parallel for
        for (int i = 0; i < image.NumberOfVoxels(); i++)
            ptr[i] = ptr[i] > threshold ? 1 : 0;

        return image;
    }

    //-------------------------------------------------------------------

    // normalise and threshold mask
    inline RealImage ThresholdNormalisedMask(RealImage image, double threshold) {
        RealPixel smin, smax;
        image.GetMinMax(&smin, &smax);

        if (smax > 0)
            image /= smax;

        return CreateMask(image, threshold);
    }

    //-------------------------------------------------------------------

    // reset image origin and save it into the output transformation (RealImage/GreyImage/ByteImage)
    template<typename ImageType>
    inline void ResetOrigin(GenericImage<ImageType>& image, RigidTransformation& transformation) {
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

    // perform nonlocal means filtering
    inline void NLMFiltering(Array<RealImage>& stacks) {
        #pragma omp parallel for
        for (int i = 0; i < stacks.size(); i++) {
            stacks[i] = NLDenoising::Run(stacks[i], 3, 1);
            stacks[i].Write((boost::format("denoised-%1%.nii.gz") % i).str().c_str());
        }
    }

    //-------------------------------------------------------------------

    // invert transformations
    inline void InvertStackTransformations(Array<RigidTransformation>& stack_transformations) {
        //for each stack
        #pragma omp parallel for
        for (size_t i = 0; i < stack_transformations.size(); i++) {
            //invert transformation for the stacks
            stack_transformations[i].Invert();
            stack_transformations[i].UpdateParameter();
        }
    }

    //-------------------------------------------------------------------

    // mask input volume
    inline void MaskImage(RealImage& image, const RealImage& mask, double padding = -1) {
        if (image.NumberOfVoxels() != mask.NumberOfVoxels()) {
            cerr << "Cannot mask the image - different dimensions" << endl;
            exit(1);
        }

        RealPixel *pr = image.Data();
        const RealPixel *pm = mask.Data();
        #pragma omp parallel for
        for (int i = 0; i < image.NumberOfVoxels(); i++)
            if (pm[i] == 0)
                pr[i] = padding;
    }

    //-------------------------------------------------------------------

    // Rescale image ignoring negative values
    inline void Rescale(RealImage& img, double max) {
        // Get lower and upper bound
        RealPixel min_val, max_val;
        img.GetMinMax(&min_val, &max_val);

        RealPixel *ptr = img.Data();
        #pragma omp parallel for
        for (int i = 0; i < img.NumberOfVoxels(); i++)
            if (ptr[i] > 0)
                ptr[i] = double(ptr[i]) / double(max_val) * max;
    }

    //-------------------------------------------------------------------

    // Apply Static Mask to 4D Volume
    inline void StaticMaskVolume4D(RealImage& volume, const RealImage& mask, const double padding) {
        #pragma omp parallel for
        for (int i = 0; i < volume.GetX(); i++)
            for (int j = 0; j < volume.GetY(); j++)
                for (int k = 0; k < volume.GetZ(); k++)
                    if (mask(i, j, k) == 0)
                        for (int t = 0; t < volume.GetT(); t++)
                            volume(i, j, k, t) = padding;
    }
}
