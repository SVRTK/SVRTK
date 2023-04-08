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
#include "svrtk/Common.h"

using namespace std;
using namespace mirtk;

// Forward declarations
namespace svrtk {
    class Reconstruction;
}

namespace svrtk::Utility {

    /**
     * @brief Extract specific image ROI.
     * @param stack
     * @param transformation
     * @param min_x
     * @param min_y
     * @param min_z
     * @param max_x
     * @param max_y
     * @param max_z
     */
    void bbox(const RealImage& stack, const RigidTransformation& transformation, double& min_x, double& min_y, double& min_z, double& max_x, double& max_y, double& max_z);

    /**
     * @brief Crop to non zero ROI.
     * @param image
     */
    void bboxCrop(RealImage& image);

    /**
     * @brief Find centroid.
     * @param image
     * @param x
     * @param y
     * @param z
     */
    void centroid(const RealImage& image, double& x, double& y, double& z);

    /**
     * @brief Center stacks.
     * @param stacks
     * @param stack_transformations
     * @param templateNumber
     */
    void CenterStacks(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber);

    /// Crops the image according to the mask
    void CropImage(RealImage& image, const RealImage& mask);

    /// GF 190416, useful for handling different slice orders
    void CropImageIgnoreZ(RealImage& image, const RealImage& mask);

    /**
     * @brief Compute NCC between images.
     * @param slice_1
     * @param slice_2
     * @param threshold
     * @param count
     * @return
     */
    double ComputeNCC(const RealImage& slice_1, const RealImage& slice_2, const double threshold = 0.01, double *count = nullptr);


    void RemoveNanNegative(RealImage& stack);


    double LocalSSIM(const RealImage slice, const RealImage sim_slice );

    /**
     * @brief Compute inter-slice volume NCC (motion metric).
     * @param input_stack
     * @param template_stack
     * @param mask
     * @return
     */
    double VolumeNCC(RealImage& input_stack, RealImage template_stack, const RealImage& mask);

    /**
     * @brief Compute internal stack statistics (volume and inter-slice NCC).
     * @param input_stack
     * @param mask
     * @param mask_volume
     * @param slice_ncc
     */
    void StackStats(RealImage input_stack, RealImage& mask, double& mask_volume, double& slice_ncc);

    /**
     * @brief Run serial global similarity statistics (for the stack selection function).
     * @param template_stack
     * @param template_mask
     * @param stacks
     * @param masks
     * @param average_ncc
     * @param average_volume
     * @param current_stack_transformations
     */
    void GlobalStackStats(RealImage template_stack, const RealImage& template_mask, const Array<RealImage>& stacks,
        const Array<RealImage>& masks, double& average_ncc, double& average_volume,
        Array<RigidTransformation>& current_stack_transformations);

    // Cropping of stacks based on intersection
    void StackIntersection(Array<RealImage>& stacks, RealImage template_mask);

    /**
     * @brief Run parallel global similarity statistics.
     * @param stacks
     * @param masks
     * @param all_global_ncc_array
     * @param all_global_volume_array
     */
    void RunParallelGlobalStackStats(const Array<RealImage>& stacks, const Array<RealImage>& masks,
        Array<double>& all_global_ncc_array, Array<double>& all_global_volume_array);

    /**
     * @brief Create average image from the stacks and volumetric transformations.
     * @param reconstructor
     * @param stacks
     * @param stack_transformations
     * @return
     */
    RealImage CreateAverage(const Reconstruction *reconstructor, const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations);

    /**
     * @brief Create a mask from dark/black background.
     * @param reconstructor
     * @param stacks
     * @param stack_transformations
     * @param smooth_mask
     */
    RealImage CreateMaskFromBlackBackground(const Reconstruction *reconstructor, const Array<RealImage>& stacks, Array<RigidTransformation> stack_transformations);

    /**
     * @brief Transform and resample mask to the space of the image.
     * @param image
     * @param mask
     * @param transformation
     */
    void TransformMask(const RealImage& image, RealImage& mask, const RigidTransformation& transformation);

    /**
     * @brief Run stack background filtering (GS based).
     * @param stacks
     * @param fg_sigma
     * @param bg_sigma
     */
    void BackgroundFiltering(Array<RealImage>& stacks, const double fg_sigma, const double bg_sigma);

    /**
     * @brief Mask stacks with respect to the reconstruction mask and given transformations.
     * @param stacks
     * @param stack_transformations
     * @param mask
     */
    void MaskStacks(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, const RealImage& mask);

    /**
     * @brief Mask slices based on the reconstruction mask.
     * @param slices
     * @param mask
     * @param Transform
     */
    void MaskSlices(Array<RealImage>& slices, const RealImage& mask, function<void(size_t, double&, double&, double&)> Transform);

    /**
     * @brief Get slice order parameters.
     * @param stacks
     * @param pack_num
     * @param order
     * @param step
     * @param rewinder
     * @param output_z_slice_order
     * @param output_t_slice_order
     */
    void GetSliceAcquisitionOrder(const Array<RealImage>& stacks, const Array<int>& pack_num, const Array<int>& order,
        const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order);

    /**
     * @brief Split images into varying N packages.
     * @param stacks
     * @param sliceStacks
     * @param pack_num
     * @param sliceNums
     * @param order
     * @param step
     * @param rewinder
     * @param output_z_slice_order
     * @param output_t_slice_order
     */
    void FlexibleSplitImage(const Array<RealImage>& stacks, Array<RealImage>& sliceStacks, const Array<int>& pack_num,
        const Array<int>& sliceNums, const Array<int>& order, const int step, const int rewinder,
        Array<int>& output_z_slice_order, Array<int>& output_t_slice_order);

    /**
     * @brief Split images based on multi-band acquisition.
     * @param stacks
     * @param sliceStacks
     * @param pack_num
     * @param sliceNums
     * @param multiband_vector
     * @param order
     * @param step
     * @param rewinder
     * @param output_z_slice_order
     * @param output_t_slice_order
     */
    void FlexibleSplitImagewithMB(const Array<RealImage>& stacks, Array<RealImage>& sliceStacks, const Array<int>& pack_num,
        const Array<int>& sliceNums, const Array<int>& multiband_vector, const Array<int>& order, const int step, const int rewinder,
        Array<int>& output_z_slice_order, Array<int>& output_t_slice_order);

    /**
     * @brief Split stacks into packages based on specific order.
     * @param stacks
     * @param pack_num
     * @param packageStacks
     * @param order
     * @param step
     * @param rewinder
     * @param output_z_slice_order
     * @param output_t_slice_order
     */
    void SplitPackages(const Array<RealImage>& stacks, const Array<int>& pack_num, Array<RealImage>& packageStacks, const Array<int>& order,
        const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order);

    /**
     * @brief Split stacks into packages based on specific order (multi-band based).
     * @param stacks
     * @param pack_num
     * @param packageStacks
     * @param multiband_vector
     * @param order
     * @param step
     * @param rewinder
     * @param output_z_slice_order
     * @param output_t_slice_order
     */
    void SplitPackageswithMB(const Array<RealImage>& stacks, const Array<int>& pack_num, Array<RealImage>& packageStacks, const Array<int>& multiband_vector,
        const Array<int>& order, const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order);

    /**
     * @brief Split image into N packages.
     * @param image
     * @param packages
     * @param stacks
     */
    void SplitImage(const RealImage& image, const int packages, Array<RealImage>& stacks);

    /**
     * @brief Split image into 2 packages.
     * @param image
     * @param packages
     * @param stacks
     */
    void SplitImageEvenOdd(const RealImage& image, const int packages, Array<RealImage>& stacks);

    /**
     * @brief Split image into 4 packages.
     * @param image
     * @param packages
     * @param stacks
     * @param iter
     */
    void SplitImageEvenOddHalf(const RealImage& image, const int packages, Array<RealImage>& stacks, const int iter);

    /**
     * @brief Split image into 2 packages.
     * @param image
     * @param stacks
     */
    void HalfImage(const RealImage& image, Array<RealImage>& stacks);

    ////////////////////////////////////////////////////////////////////////////////
    // Inline/template definitions
    ////////////////////////////////////////////////////////////////////////////////

    /// Clear and preallocate memory for a vector
    template<typename VectorType>
    inline void ClearAndReserve(vector<VectorType>& vectorVar, size_t reserveSize) {
        vectorVar.clear();
        vectorVar.reserve(reserveSize);
    }

    //-------------------------------------------------------------------

    /// Clear and resize memory for vectors
    template<typename VectorType>
    inline void ClearAndResize(vector<VectorType>& vectorVar, size_t reserveSize, const VectorType& defaultValue = VectorType()) {
        vectorVar.clear();
        vectorVar.resize(reserveSize, defaultValue);
    }

    //-------------------------------------------------------------------

    /**
     * @brief Binarise mask.
     * If template image has been masked instead of creating the mask in separate
     * file, this function can be used to create mask from the template image.
     * @param image
     * @param threshold
     * @return
     */
    inline RealImage CreateMask(RealImage image, double threshold = 0.5) {
        RealPixel *ptr = image.Data();
        #pragma omp parallel for
        for (int i = 0; i < image.NumberOfVoxels(); i++)
            ptr[i] = ptr[i] > threshold ? 1 : 0;

        return image;
    }

    //-------------------------------------------------------------------

    /// Normalise and threshold mask
    inline RealImage ThresholdNormalisedMask(RealImage image, double threshold) {
        RealPixel smin, smax;
        image.GetMinMax(&smin, &smax);

        if (smax > 0)
            image /= smax;

        return CreateMask(image, threshold);
    }

    //-------------------------------------------------------------------

    /// Reset image origin and save it into the output transformation (RealImage/GreyImage/ByteImage)
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

    /// Perform nonlocal means filtering
    inline void NLMFiltering(Array<RealImage>& stacks) {
        #pragma omp parallel for
        for (int i = 0; i < stacks.size(); i++) {
            stacks[i] = NLDenoising::Run(stacks[i], 3, 1);
            stacks[i].Write((boost::format("denoised-%1%.nii.gz") % i).str().c_str());
        }
    }

    //-------------------------------------------------------------------

    /// Invert stack transformations
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

    /// Mask input volume
    inline void MaskImage(RealImage& image, const RealImage& mask, double padding = -1) {
        if (image.NumberOfVoxels() != mask.NumberOfVoxels())
            throw runtime_error("Cannot mask the image - different dimensions");

        RealPixel *pr = image.Data();
        const RealPixel *pm = mask.Data();
        #pragma omp parallel for
        for (int i = 0; i < image.NumberOfVoxels(); i++)
            if (pm[i] == 0)
                pr[i] = padding;
    }

    //-------------------------------------------------------------------

    /// Rescale image ignoring negative values
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

    /// Apply static mask to 4d volume
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
