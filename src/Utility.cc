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

// SVRTK
#include "svrtk/Utility.h"
#include "svrtk/Parallel.h"

namespace svrtk::Utility {

    // Extract specific image ROI
    void bbox(const RealImage& stack, const RigidTransformation& transformation, double& min_x, double& min_y, double& min_z, double& max_x, double& max_y, double& max_z) {
        min_x = DBL_MAX;
        min_y = DBL_MAX;
        min_z = DBL_MAX;
        max_x = -DBL_MAX;
        max_y = -DBL_MAX;
        max_z = -DBL_MAX;
        // WARNING: do not search to increment by stack.GetZ()-1,
        // otherwise you would end up with a 0 increment for slices...
        for (int i = 0; i <= stack.GetX(); i += stack.GetX())
            for (int j = 0; j <= stack.GetY(); j += stack.GetY())
                for (int k = 0; k <= stack.GetZ(); k += stack.GetZ()) {
                    double x = i;
                    double y = j;
                    double z = k;

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
        int min_x = image.GetX() - 1;
        int min_y = image.GetY() - 1;
        int min_z = image.GetZ() - 1;
        int max_x = 0;
        int max_y = 0;
        int max_z = 0;
        for (int i = 0; i < image.GetX(); i++)
            for (int j = 0; j < image.GetY(); j++)
                for (int k = 0; k < image.GetZ(); k++) {
                    if (image(i, j, k) > 0) {
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
    void centroid(const RealImage& image, double& x, double& y, double& z) {
        double sum_x = 0;
        double sum_y = 0;
        double sum_z = 0;
        double norm = 0;
        for (int i = 0; i < image.GetX(); i++)
            for (int j = 0; j < image.GetY(); j++)
                for (int k = 0; k < image.GetZ(); k++) {
                    double v = image(i, j, k);
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

    // Center stacks with respect to the centre of the mask
    void CenterStacks(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber) {
        RealImage mask = stacks[templateNumber];
        RealPixel *ptr = mask.Data();
        #pragma omp parallel for
        for (int i = 0; i < mask.NumberOfVoxels(); i++)
            if (ptr[i] < 0)
                ptr[i] = 0;

        double x0, y0, z0;
        centroid(mask, x0, y0, z0);

        #pragma omp parallel for private(mask, ptr)
        for (size_t i = 0; i < stacks.size(); i++) {
            if (i == templateNumber)
                continue;

            mask = stacks[i];
            ptr = mask.Data();
            for (int i = 0; i < mask.NumberOfVoxels(); i++)
                if (ptr[i] < 0)
                    ptr[i] = 0;

            double x, y, z;
            centroid(mask, x, y, z);

            RigidTransformation translation;
            translation.PutTranslationX(x0 - x);
            translation.PutTranslationY(y0 - y);
            translation.PutTranslationZ(z0 - z);

            stack_transformations[i].PutMatrix(translation.GetMatrix() * stack_transformations[i].GetMatrix());
        }
    }

    //-------------------------------------------------------------------

    // Crops the image according to the mask
    void CropImage(RealImage& image, const RealImage& mask) {
        int i, j, k;
        //upper boundary for z coordinate
        for (k = image.GetZ() - 1; k >= 0; k--) {
            int sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int z2 = k;

        //lower boundary for z coordinate
        for (k = 0; k <= image.GetZ() - 1; k++) {
            int sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int z1 = k;

        //upper boundary for y coordinate
        for (j = image.GetY() - 1; j >= 0; j--) {
            int sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int y2 = j;

        //lower boundary for y coordinate
        for (j = 0; j <= image.GetY() - 1; j++) {
            int sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int y1 = j;

        //upper boundary for x coordinate
        for (i = image.GetX() - 1; i >= 0; i--) {
            int sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (j = image.GetY() - 1; j >= 0; j--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int x2 = i;

        //lower boundary for x coordinate
        for (i = 0; i <= image.GetX() - 1; i++) {
            int sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (j = image.GetY() - 1; j >= 0; j--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int x1 = i;

        // if no intersection with mask, force exclude
        if ((x2 <= x1) || (y2 <= y1) || (z2 <= z1))
            x1 = y1 = z1 = x2 = y2 = z2 = 0;

        //Cut region of interest
        image = image.GetRegion(x1, y1, z1, x2 + 1, y2 + 1, z2 + 1);
    }

    //-------------------------------------------------------------------

    // GF 190416, useful for handling different slice orders
    void CropImageIgnoreZ(RealImage& image, const RealImage& mask) {
        int i, j, k;
        //Crops the image according to the mask
        // Filling slices out of mask with zeros
        for (k = image.GetZ() - 1; k >= 0; k--) {
            int sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0) {
                k++;
                break;
            }
        }
        int z2 = k;

        //lower boundary for z coordinate
        for (k = 0; k <= image.GetZ() - 1; k++) {
            int sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int z1 = k;

        // Filling upper part
        for (k = z2; k < image.GetZ(); k++)
            for (j = 0; j < image.GetY(); j++)
                for (i = 0; i < image.GetX(); i++)
                    image.Put(i, j, k, 0);

        // Filling lower part
        for (k = 0; k < z1; k++)
            for (j = 0; j < image.GetY(); j++)
                for (i = 0; i < image.GetX(); i++)
                    image.Put(i, j, k, 0);

        //Original ROI
        z1 = 0;
        z2 = image.GetZ() - 1;

        //upper boundary for y coordinate
        for (j = image.GetY() - 1; j >= 0; j--) {
            int sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int y2 = j;

        //lower boundary for y coordinate
        for (j = 0; j <= image.GetY() - 1; j++) {
            int sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int y1 = j;

        //upper boundary for x coordinate
        for (i = image.GetX() - 1; i >= 0; i--) {
            int sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (j = image.GetY() - 1; j >= 0; j--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int x2 = i;

        //lower boundary for x coordinate
        for (i = 0; i <= image.GetX() - 1; i++) {
            int sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (j = image.GetY() - 1; j >= 0; j--)
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        int x1 = i;

        // if no intersection with mask, force exclude
        if ((x2 <= x1) || (y2 <= y1) || (z2 <= z1))
            x1 = y1 = z1 = x2 = y2 = z2 = 0;

        //Cut region of interest
        image = image.GetRegion(x1, y1, z1, x2 + 1, y2 + 1, z2 + 1);
    }

    //-------------------------------------------------------------------

    // crop stacks based on global intersection
    void StackIntersection(Array<RealImage>& stacks, RealImage template_mask) {
        
        ImageAttributes attr = template_mask.Attributes();
        int new_x = 2*attr._x;
        int new_y = 2*attr._y;
        int new_z = 2*attr._z;
        RealImage intersection_mask(attr);
        intersection_mask = 1;
        
        int ix, iy, iz, is;
        double sx, sy, sz;
        double wx, wy, wz;
        for (ix = 0; ix < intersection_mask.GetX(); ix++) {
            for (iy = 0; iy < intersection_mask.GetY(); iy++) {
                for (iz = 0; iz < intersection_mask.GetZ(); iz++) {
                    wx = ix; wy = iy; wz = iz;
                    intersection_mask.ImageToWorld(wx, wy, wz);
                    for (is = 0; is < stacks.size(); is++) {
                        sx = wx; sy = wy; sz = wz;
                        stacks[is].WorldToImage(sx, sy, sz);
                        if (stacks[is].IsInside(sx, sy, sz)) {
                            intersection_mask(ix,iy,iz) = intersection_mask(ix,iy,iz) * 1;
                        } else {
                            intersection_mask(ix,iy,iz) = intersection_mask(ix,iy,iz) * 0;
                        }
                    }
                    sx = wx; sy = wy; sz = wz;
                    template_mask.WorldToImage(sx, sy, sz);
                    int mx, my, mz;
                    mx = round(sx); my = round(sy); mz = round(sz);
                    if (template_mask.IsInside(mx, my, mz) && template_mask(mx, my, mz) > 0)
                            intersection_mask(ix,iy,iz) = 1;
                }
            }
        }
        ConnectivityType i_connectivity = CONNECTIVITY_26;
        Dilate<RealPixel>(&intersection_mask, 5, i_connectivity);
        
//        if (_debug) {
//            intersection_mask.Write("intersection-mask.nii.gz");
//        }
        
        for (int i=0; i<stacks.size(); i++) {
            RigidTransformation* tmp_r = new RigidTransformation();
            RealImage i_tmp = intersection_mask;
            TransformMask(stacks[i],i_tmp, *tmp_r);
            CropImage(stacks[i],i_tmp);
//            if (_debug) {
//                stacks[i].Write((boost::format("gcropped-%1%.nii.gz") % i).str().c_str());
//            }
        }

    }

    //-------------------------------------------------------------------

    // Implementation of NCC between images
    double ComputeNCC(const RealImage& slice_1, const RealImage& slice_2, const double threshold, double *count) {
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

    // Implementation of SSIM between images
    double LocalSSIM( const RealImage slice, const RealImage sim_slice ) {
        double C1 = 6.5025, C2 = 58.5225;
        double SSIM = 0, DSSIM = 0;
        
        double mu1=0, mu2=0, var1=0, var2=0, covar=0, num=0;
        double x_sq=0, y_sq=0, xy=0;
        int box = round(slice.GetX()/2);
        int y = box;
        int x = box;
                
        for (int yy=0; yy<slice.GetY(); yy++) {
            for (int xx=0; xx<slice.GetX(); xx++) {
                if (slice(xx,yy,0)>0.01 && sim_slice(xx,yy,0)>0.01) {
                    mu1 += slice(xx, yy, 0);
                    mu2 += sim_slice(xx,yy,0);
                    num += 1;
                    x_sq += pow(slice(xx, yy, 0),2.0);
                    y_sq += pow(sim_slice(xx, yy, 0),2.0);
                    xy += slice(xx, yy, 0)*sim_slice(xx, yy, 0);
                 }
            }
        }
                
        mu1 = mu1/num;
        mu2 = mu2/num;
        var1 = (x_sq/num)-pow(mu1,2.0);
        var2 = (y_sq/num)-pow(mu2,2.0);
        covar = (xy/num)-mu1*mu2;
        double curSSIM = ((2*mu1*mu2+C1)*(2*covar+C2)) / ( (pow(mu1,2.)+pow(mu2,2.)+C1) * (var1+var2+C2) );
        
        return curSSIM;
    }


    //-------------------------------------------------------------------

    // compute inter-slice volume NCC (motion metric)
    double VolumeNCC(RealImage& input_stack, RealImage template_stack, const RealImage& mask) {
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
        constexpr double source_padding = 0;
        const double target_padding = -inf;
        constexpr bool dofin_invert = false;
        constexpr bool twod = false;

        RealImage& output = template_stack;
        memset(output.Data(), 0, sizeof(RealPixel) * output.NumberOfVoxels());

        ImageTransformation imagetransformation;
        imagetransformation.Input(&input_stack);
        imagetransformation.Transformation(r_dofout.get());
        imagetransformation.Output(&output);
        imagetransformation.TargetPaddingValue(target_padding);
        imagetransformation.SourcePaddingValue(source_padding);
        imagetransformation.Interpolator(&interpolator);
        imagetransformation.TwoD(twod);
        imagetransformation.Invert(dofin_invert);
        imagetransformation.Run();

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

    // compute internal stack statistics (volume and inter-slice NCC)
    void StackStats(RealImage input_stack, RealImage& mask, double& mask_volume, double& slice_ncc) {
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
        RealPixel *pm = mask.Data();
        for (int i = 0; i < mask.NumberOfVoxels(); i++)
            if (pm[i] > 0.01)
                mask_count++;

        mask_volume = mask_count * mask.GetXSize() * mask.GetYSize() * mask.GetZSize() / 1000;
    }

    //-------------------------------------------------------------------

    // run serial global similarity statists (for the stack selection function)
    void GlobalStackStats(RealImage template_stack, const RealImage& template_mask, const Array<RealImage>& stacks, const Array<RealImage>& masks, double& average_ncc, double& average_volume, Array<RigidTransformation>& current_stack_transformations) {
        template_stack *= template_mask;

        RigidTransformation r_init;
        r_init.PutTranslationX(0.0001);
        r_init.PutTranslationY(0.0001);
        r_init.PutTranslationZ(-0.0001);

        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        Insert(params, "Background value for image 1", 0);
        Insert(params, "Background value for image 2", 0);

        constexpr double source_padding = 0;
        const double target_padding = -inf;
        constexpr bool dofin_invert = false;
        constexpr bool twod = false;
        GenericRegistrationFilter registration;
        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        ImageTransformation imagetransformation;
        imagetransformation.TargetPaddingValue(target_padding);
        imagetransformation.SourcePaddingValue(source_padding);
        imagetransformation.TwoD(twod);
        imagetransformation.Invert(dofin_invert);
        imagetransformation.Interpolator(&interpolator);

        average_ncc = 0;
        average_volume = 0;
        current_stack_transformations.resize(stacks.size());

        //#pragma omp parallel for firstprivate(imagetransformation) private(registration) reduction(+: average_ncc, average_volume)
        for (size_t i = 0; i < stacks.size(); i++) {
            RealImage input_stack = stacks[i] * masks[i];

            Transformation *dofout;
            registration.Parameter(params);
            registration.Output(&dofout);
            registration.InitialGuess(&r_init);
            registration.Input(&template_stack, &input_stack);
            registration.GuessParameter();
            registration.Run();
            unique_ptr<RigidTransformation> r_dofout(dynamic_cast<RigidTransformation*>(dofout));
            current_stack_transformations[i] = *r_dofout;

            RealImage output(template_stack.Attributes());
            imagetransformation.Input(&input_stack);
            imagetransformation.Transformation(dofout);
            imagetransformation.Output(&output);
            imagetransformation.Run();
            input_stack = move(output);

            double slice_count = 0;
            const double local_ncc = ComputeNCC(template_stack, input_stack, 0.01, &slice_count);
            average_ncc += local_ncc;
            average_volume += slice_count;
        }

        average_ncc /= stacks.size();
        average_volume /= stacks.size();
        average_volume *= template_stack.GetXSize() * template_stack.GetYSize() * template_stack.GetZSize() / 1000;
    }

    //-------------------------------------------------------------------

    // run parallel global similarity statists (for the stack selection function)
    void RunParallelGlobalStackStats(const Array<RealImage>& stacks, const Array<RealImage>& masks, Array<double>& all_global_ncc_array, Array<double>& all_global_volume_array) {
        ClearAndResize(all_global_ncc_array, stacks.size());
        ClearAndResize(all_global_volume_array, stacks.size());

        cout << " start ... " << endl;

        Parallel::GlobalSimilarityStats registration(stacks.size(), stacks, masks, all_global_ncc_array, all_global_volume_array);
        registration();
    }

    //-------------------------------------------------------------------

    // Create average of all stacks based on the input transformations
    RealImage CreateAverage(const Reconstruction *reconstructor, const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations) {
        InvertStackTransformations(stack_transformations);
        Parallel::Average p_average(reconstructor, stacks, stack_transformations, -1, 0, 0, true);
        p_average();
        const RealImage average = p_average.average / p_average.weights;
        InvertStackTransformations(stack_transformations);

        return average;
    }

    //-------------------------------------------------------------------

    // create a mask from dark / black background
    RealImage CreateMaskFromBlackBackground(const Reconstruction *reconstructor, const Array<RealImage>& stacks, Array<RigidTransformation> stack_transformations) {
        //Create average of the stack using currect stack transformations
        GreyImage average = CreateAverage(reconstructor, stacks, stack_transformations);

        GreyPixel *ptr = average.Data();
        #pragma omp parallel for
        for (int i = 0; i < average.NumberOfVoxels(); i++)
            if (ptr[i] < 0)
                ptr[i] = 0;

        //Create mask of the average from the black background
        MeanShift msh(average, 0, 256);
        msh.GenerateDensity();
        msh.SetThreshold();
        msh.RemoveBackground();
        RealImage mask = msh.ReturnMask();

        //Calculate LCC of the mask to remove disconnected structures
        MeanShift msh2(mask, 0, 256);
        msh2.SetOutput(&mask);
        msh2.Lcc(1);
        
        return mask;
    }

    //-------------------------------------------------------------------

    // transform mask to the specified stack space
    void TransformMask(const RealImage& image, RealImage& mask, const RigidTransformation& transformation) {
        //transform mask to the space of image
        unique_ptr<InterpolateImageFunction> interpolator(InterpolateImageFunction::New(Interpolation_NN));
        ImageTransformation imagetransformation;
        RealImage m(image.Attributes());

        imagetransformation.Input(&mask);
        imagetransformation.Transformation(&transformation);
        imagetransformation.Output(&m);
        //target contains zeros and ones image, need padding -1
        imagetransformation.TargetPaddingValue(-1);
        //need to fill voxels in target where there is no info from source with zeroes
        imagetransformation.SourcePaddingValue(0);
        imagetransformation.Interpolator(interpolator.get());
        imagetransformation.Run();
        mask = move(m);
    }


    void RemoveNanNegative(RealImage& stack) {
 
        for (int x = 0; x < stack.GetX(); x++) {
            for (int y = 0; y < stack.GetY(); y++) {
                for (int z = 0; z < stack.GetZ(); z++) {
                    
                    if (stack(x,y,z) < 0)
                        stack(x,y,z) = 0;
                    
                    if (stack(x,y,z) != stack(x,y,z))
                        stack(x,y,z) = 0;
                    
                }
            }
        }
               
        
    }


    //-------------------------------------------------------------------

    // run stack background filtering (GS based)
    void BackgroundFiltering(Array<RealImage>& stacks, const double fg_sigma, const double bg_sigma) {
        GaussianBlurring<RealPixel> gb2(stacks[0].GetXSize() * bg_sigma);
        GaussianBlurring<RealPixel> gb3(stacks[0].GetXSize() * fg_sigma);
        RealImage stack, global_blurred, tmp_slice, tmp_slice_b;

        // Do not parallelise: GaussianBlurring has already been parallelised!
        for (size_t j = 0; j < stacks.size(); j++) {
            stack = stacks[j];
            stack.Write((boost::format("original-%1%.nii.gz") % j).str().c_str());

            global_blurred = stacks[j];
            gb2.Input(&global_blurred);
            gb2.Output(&global_blurred);
            gb2.Run();

            // Do not parallelise: GaussianBlurring has already been parallelised!
            for (int i = 0; i < stacks[j].GetZ(); i++) {
                tmp_slice = stacks[j].GetRegion(0, 0, i, stacks[j].GetX(), stacks[j].GetY(), i + 1);
                tmp_slice_b = tmp_slice;

                gb3.Input(&tmp_slice_b);
                gb3.Output(&tmp_slice_b);
                gb3.Run();

                gb2.Input(&tmp_slice);
                gb2.Output(&tmp_slice);
                gb2.Run();

                #pragma omp parallel for
                for (int x = 0; x < stacks[j].GetX(); x++) {
                    for (int y = 0; y < stacks[j].GetY(); y++) {
                        stack(x, y, i) = tmp_slice_b(x, y, 0) + global_blurred(x, y, i) - tmp_slice(x, y, 0);
                        if (stack(x, y, i) < 0)
                            stack(x, y, i) = 1;
                    }
                }
            }

            stack.Write((boost::format("filtered-%1%.nii.gz") % j).str().c_str());

            stacks[j] = move(stack);
        }
    }

    //-------------------------------------------------------------------

    // mask stacks with respect to the reconstruction mask and given transformations
    void MaskStacks(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, const RealImage& mask) {
        for (size_t inputIndex = 0; inputIndex < stacks.size(); inputIndex++) {
            RealImage& stack = stacks[inputIndex];
            #pragma omp parallel for
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
                        mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);
                        //if the voxel is outside mask ROI set it to -1 (padding value)
                        if (mask.IsInside(x, y, z)) {
                            if (mask(x, y, z) == 0)
                                stack(i, j, k) = -1;
                        } else
                            stack(i, j, k) = -1;
                    }
                }
        }
    }

    //-------------------------------------------------------------------

    // mask slices based on the reconstruction mask
    void MaskSlices(Array<RealImage>& slices, const RealImage& mask, function<void(size_t, double&, double&, double&)> Transform) {
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < slices.size(); inputIndex++) {
            RealImage& slice = slices[inputIndex];
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

                    Transform(inputIndex, x, y, z);

                    //image coordinates in volume space
                    mask.WorldToImage(x, y, z);
                    x = round(x);
                    y = round(y);
                    z = round(z);
                    //if the voxel is outside mask ROI set it to -1 (padding value)
                    if (mask.IsInside(x, y, z)) {
                        if (mask(x, y, z) == 0)
                            slice(i, j, 0) = -1;
                    } else
                        slice(i, j, 0) = -1;
                }
        }
    }

    //-------------------------------------------------------------------

    // get slice order parameters
    void GetSliceAcquisitionOrder(const Array<RealImage>& stacks, const Array<int>& pack_num, const Array<int>& order, const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order) {
        Array<int> realInterleaved, fakeAscending;

        for (size_t dyn = 0; dyn < stacks.size(); dyn++) {
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
                    for (size_t i = 0, index = 0, restart = 0; i < fakeAscending.size(); i++, index += stepFactor) {
                        if (index >= fakeAscending.size()) {
                            restart += rewinderFactor;
                            index = restart;
                        }
                        realInterleaved.push_back(fakeAscending[index]);
                    }

                    fakeAscending.clear();
                }

                // saving
                for (int i = 0; i < attr._z; i++) {
                    z_slice_order[i] = realInterleaved[i];
                    t_slice_order[realInterleaved[i]] = i;
                }

                realInterleaved.clear();
            }

            // copying
            for (int i = 0; i < attr._z; i++) {
                output_z_slice_order.push_back(z_slice_order[i]);
                output_t_slice_order.push_back(t_slice_order[i]);
            }
        }
    }

    //-------------------------------------------------------------------

    // split images into varying N packages
    void FlexibleSplitImage(const Array<RealImage>& stacks, Array<RealImage>& sliceStacks, const Array<int>& pack_num, const Array<int>& sliceNums, const Array<int>& order, const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order) {
        // calculate slice order
        GetSliceAcquisitionOrder(stacks, pack_num, order, step, rewinder, output_z_slice_order, output_t_slice_order);

        Array<int> z_internal_slice_order;
        z_internal_slice_order.reserve(stacks[0].Attributes()._z);

        // counters
        int counter1 = 0, counter2 = 0, counter3 = 0;
        int startIterations = 0;

        // dynamic loop
        for (size_t dyn = 0; dyn < stacks.size(); dyn++) {
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            // location acquisition order

            // slice loop
            for (int sl = 0; sl < attr._z; sl++)
                z_internal_slice_order.push_back(output_z_slice_order[counter1 + sl]);

            // fake packages
            int sum = 0;
            while (sum < attr._z) {
                sum += sliceNums[counter2];
                counter2++;
            }

            // fake package loop
            const int endIterations = counter2;
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

                // for (size_t i = 0; i < sliceStacks.size(); i++)
                //     sliceStacks[i].Write((boost::format("sliceStacks%1%.nii") % i).str().c_str());
            }

            // updating variables for next dynamic
            counter1 += attr._z;
            counter3 = 0;
            startIterations = endIterations;

            z_internal_slice_order.clear();
        }
    }

    //-------------------------------------------------------------------

    // split images based on multi-band acquisition
    void FlexibleSplitImagewithMB(const Array<RealImage>& stacks, Array<RealImage>& sliceStacks, const Array<int>& pack_num, const Array<int>& sliceNums, const Array<int>& multiband_vector, const Array<int>& order, const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order) {
        // initializing variables
        Array<RealImage> chunks, chunksAll, chunks_separated, chunks_separated_reordered;
        Array<int> pack_num_chucks;
        int counter1, counter2, counter3, counter4, counter5;
        Array<int> sliceNumsChunks;

        int startFactor = 0, endFactor = 0;
        // dynamic loop
        for (size_t dyn = 0; dyn < stacks.size(); dyn++) {
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            const int multiband = multiband_vector[dyn];
            const int sliceMB = attr._z / multiband;
            int sum = 0;

            for (int m = 0; m < multiband; m++) {
                RealImage chunk = image.GetRegion(0, 0, m * sliceMB, attr._x, attr._y, (m + 1) * sliceMB);
                chunks.push_back(move(chunk));
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
        FlexibleSplitImage(chunks, chunksAll, pack_num_chucks, sliceNumsChunks, order, step, rewinder, output_z_slice_order, output_t_slice_order);

        counter4 = counter5 = 0;
        RealImage multibanded;
        // new dynamic loop
        for (size_t dyn = 0; dyn < stacks.size(); dyn++) {
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            const int multiband = multiband_vector[dyn];
            const int sliceMB = attr._z / multiband;
            int sum = 0, stepFactor = 0;
            multibanded.Initialize(attr);

            // stepping factor in vector
            while (sum < sliceMB) {
                sum += sliceNums[counter5 + stepFactor];
                stepFactor++;
            }

            // getting data from this dynamic
            for (int iter = 0; iter < multiband * stepFactor; iter++)
                chunks_separated.push_back(chunksAll[iter + counter4]);

            counter4 += multiband * stepFactor;

            // reordering chunks_separated
            counter1 = counter2 = counter3 = 0;
            while (counter1 < chunks_separated.size()) {
                chunks_separated_reordered.push_back(chunks_separated[counter2]);
                counter2 += stepFactor;
                if (counter2 > chunks_separated.size() - 1)
                    counter2 = ++counter3;
                counter1++;
            }

            // reassembling multiband packs
            counter1 = counter2 = 0;
            while (counter1 < chunks_separated_reordered.size()) {
                for (int m = 0; m < multiband; m++) {
                    const RealImage& toAdd = chunks_separated_reordered[counter1];
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

            chunks_separated.clear();
            chunks_separated_reordered.clear();
        }
    }

    //-------------------------------------------------------------------

    // split stacks into packages based on specific order
    void SplitPackages(const Array<RealImage>& stacks, const Array<int>& pack_num, Array<RealImage>& packageStacks, const Array<int>& order, const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order) {
        // calculate slice order
        GetSliceAcquisitionOrder(stacks, pack_num, order, step, rewinder, output_z_slice_order, output_t_slice_order);

        // location acquisition order
        Array<int> z_internal_slice_order;
        z_internal_slice_order.reserve(stacks[0].Attributes()._z);

        // dynamic loop
        for (size_t dyn = 0, counter1 = 0; dyn < stacks.size(); dyn++) {
            // current stack
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            const int pkg_z = attr._z / pack_num[dyn];

            // slice loop
            for (int sl = 0; sl < attr._z; sl++)
                z_internal_slice_order.push_back(output_z_slice_order[counter1 + sl]);

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
                // updating variables for next package
                counter3 += internalIterations;
            }
            counter1 += attr._z;

            z_internal_slice_order.clear();
        }
    }

    //-------------------------------------------------------------------

    // multi-band based
    void SplitPackageswithMB(const Array<RealImage>& stacks, const Array<int>& pack_num, Array<RealImage>& packageStacks, const Array<int>& multiband_vector, const Array<int>& order, const int step, const int rewinder, Array<int>& output_z_slice_order, Array<int>& output_t_slice_order) {
        // initializing variables
        Array<RealImage> chunks, chunksAll, chunks_separated, chunks_separated_reordered;
        Array<int> pack_numAll;
        int counter1, counter2, counter3, counter4;

        // dynamic loop
        for (size_t dyn = 0; dyn < stacks.size(); dyn++) {
            const RealImage& image = stacks[dyn];
            const ImageAttributes& attr = image.Attributes();
            const int& multiband = multiband_vector[dyn];
            const int sliceMB = attr._z / multiband;

            for (int m = 0; m < multiband; m++) {
                RealImage chunk = image.GetRegion(0, 0, m * sliceMB, attr._x, attr._y, (m + 1) * sliceMB);
                chunks.push_back(move(chunk));
                pack_numAll.push_back(pack_num[dyn]);
            }
        }

        // split package
        SplitPackages(chunks, pack_numAll, chunksAll, order, step, rewinder, output_z_slice_order, output_t_slice_order);

        counter4 = 0;
        RealImage multibanded;
        // new dynamic loop
        for (size_t dyn = 0; dyn < stacks.size(); dyn++) {
            const RealImage& image = stacks[dyn];
            const int& multiband = multiband_vector[dyn];
            multibanded.Initialize(image.Attributes());

            // getting data from this dynamic
            const int& stepFactor = pack_num[dyn];
            for (int iter = 0; iter < multiband * stepFactor; iter++)
                chunks_separated.push_back(chunksAll[iter + counter4]);
            counter4 += multiband * stepFactor;

            // reordering chunks_separated
            counter1 = 0;
            counter2 = 0;
            counter3 = 0;
            while (counter1 < chunks_separated.size()) {
                chunks_separated_reordered.push_back(chunks_separated[counter2]);

                counter2 += stepFactor;
                if (counter2 > chunks_separated.size() - 1) {
                    counter3++;
                    counter2 = counter3;
                }
                counter1++;
            }

            // reassembling multiband slices
            counter1 = 0;
            counter2 = 0;
            while (counter1 < chunks_separated_reordered.size()) {
                for (int m = 0; m < multiband; m++) {
                    const RealImage& toAdd = chunks_separated_reordered[counter1];
                    for (int k = 0; k < toAdd.GetZ(); k++) {
                        for (int j = 0; j < toAdd.GetY(); j++)
                            for (int i = 0; i < toAdd.GetX(); i++)
                                multibanded.Put(i, j, counter2, toAdd(i, j, k));
                        counter2++;
                    }
                    counter1++;
                }
                packageStacks.push_back(multibanded);
                counter2 = 0;
            }

            chunks_separated.clear();
            chunks_separated_reordered.clear();
        }
    }

    //-------------------------------------------------------------------

    // split image into N packages
    void SplitImage(const RealImage& image, const int packages, Array<RealImage>& stacks) {
        //slices in package
        const int pkg_z = image.Attributes()._z / packages;
        const double pkg_dz = image.Attributes()._dz * packages;

        ClearAndReserve(stacks, packages);

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
    void SplitImageEvenOdd(const RealImage& image, const int packages, Array<RealImage>& stacks) {
        cout << "Split Image Even Odd: " << packages << " packages." << endl;

        Array<RealImage> packs, packs2;
        SplitImage(image, packages, packs);

        ClearAndReserve(stacks, packs.size() * 2);

        for (size_t i = 0; i < packs.size(); i++) {
            cout << "Package " << i << ": " << endl;
            SplitImage(packs[i], 2, packs2);
            stacks.push_back(move(packs2[0]));
            stacks.push_back(move(packs2[1]));
        }

        cout << "done." << endl;
    }

    //-------------------------------------------------------------------

    // split image into 4 packages
    void SplitImageEvenOddHalf(const RealImage& image, const int packages, Array<RealImage>& stacks, const int iter) {
        cout << "Split Image Even Odd Half " << iter << endl;

        Array<RealImage> packs, packs2;
        if (iter > 1)
            SplitImageEvenOddHalf(image, packages, packs, iter - 1);
        else
            SplitImageEvenOdd(image, packages, packs);

        ClearAndReserve(stacks, packs.size() * packs2.size());
        for (size_t i = 0; i < packs.size(); i++) {
            HalfImage(packs[i], packs2);
            for (size_t j = 0; j < packs2.size(); j++)
                stacks.push_back(move(packs2[j]));
        }
    }

    //-------------------------------------------------------------------

    // split image into 2 packages
    void HalfImage(const RealImage& image, Array<RealImage>& stacks) {
        const ImageAttributes& attr = image.Attributes();
        stacks.clear();

        //We would not like single slices - that is reserved for slice-to-volume
        if (attr._z >= 4) {
            stacks.push_back(image.GetRegion(0, 0, 0, attr._x, attr._y, attr._z / 2));
            stacks.push_back(image.GetRegion(0, 0, attr._z / 2, attr._x, attr._y, attr._z));
        } else
            stacks.push_back(image);
    }

}
