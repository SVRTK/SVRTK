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
#include "svrtk/ReconstructionqMRI.h"
#include "svrtk/Profiling.h"
#include "svrtk/Parallel.h"
#include "svrtk/ParallelqMRI.h"
#include "mirtk/HistogramMatching.h"

using namespace std;
using namespace mirtk;
using namespace svrtk::Utility;

namespace svrtk {

    double ReconstructionqMRI::CreateTemplateqMRI(const Array<RealImage> stacks, double resolution, int volumetemplateNumber) {
        double dx, dy, dz, d;

        // //Get image attributes - image size and voxel size
        // ImageAttributes attr = stack.Attributes();

        // //enlarge stack in z-direction in case top of the head is cut off
        // attr._z += 2;

        // //create enlarged image
        // RealImage enlarged(attr);

        RealImage stack = stacks[volumetemplateNumber];
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

        //_epsilon = (0.05 / _lambda) * _delta * _delta;

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


        // Create volume and initialise volume values
        ImageAttributes Attrs_4D = temp.Attributes();
        // Set temporal dimension
        Attrs_4D._t = _nEchoTimes;
        Attrs_4D._dt = 1;
        RealImage volume4D(Attrs_4D);
        RealImage VolLabel4D(Attrs_4D);

        for (int ii = 0; ii < stacks.size(); ii++) {
            RealImage iistack = stacks[ii];
            RealImage iitemp(iistack.Attributes());
            RealPixel smin, smax;
            stack.GetMinMax(&smin, &smax);

            //_epsilon = (0.05 / _lambda) * _delta * _delta;

            // interpolate the input stack to the given resolution
            if (smin < -0.1) {
                GenericLinearInterpolateImageFunction<RealImage> interpolator;
                ResamplingWithPadding<RealPixel> resampler(d, d, d, -1);
                resampler.Input(&iistack);
                resampler.Output(&iitemp);
                resampler.Interpolator(&interpolator);
                resampler.Run();
            } else if (smin < 0.1) {
                GenericLinearInterpolateImageFunction<RealImage> interpolator;
                ResamplingWithPadding<RealPixel> resampler(d, d, d, 0);
                resampler.Input(&iistack);
                resampler.Output(&iitemp);
                resampler.Interpolator(&interpolator);
                resampler.Run();
            } else {
                //resample "temp" to resolution "d"
                unique_ptr<InterpolateImageFunction> interpolator(InterpolateImageFunction::New(Interpolation_Linear));
                Resampling<RealPixel> resampler(d, d, d);
                resampler.Input(&iistack);
                resampler.Output(&iitemp);
                resampler.Interpolator(interpolator.get());
                resampler.Run();
            }

            for (int x = 0; x < iitemp.GetX(); x++){
                for (int y = 0; y < iitemp.GetY(); y++){
                    for (int z = 0; z < iitemp.GetZ(); z++) {
                        volume4D(x, y, z, ii) = iitemp(x, y, z);
                        VolLabel4D(x, y, z, ii) = ii;
                    }
                }
            }

            if (ii == volumetemplateNumber) {
                //initialise reconstructed volume
                _reconstructed = move(iitemp);
                _template_created = true;
                _grey_reconstructed = _reconstructed;
                _attr_reconstructed = _reconstructed.Attributes();
            }


        }
        /*
        // Resample to specified resolution
        unique_ptr<InterpolateImageFunction> interpolator(InterpolateImageFunction::New(Interpolation_NN));

        Resampling<RealPixel> resampling(resolution, resolution, resolution);
        resampling.Input(&volume4D);
        resampling.Output(&volume4D);
        resampling.Interpolator(interpolator.get());
        resampling.Run();*/
        _reconstructed4D = move(volume4D);
        _volumeLabel4D= move(VolLabel4D);
        _attr_reconstructed4D = _reconstructed4D.Attributes();



        if (_debug) {
            _reconstructed.Write("template.nii.gz");
            _reconstructed4D.Write("template4D.nii.gz");
        }

        //return resulting resolution of the template image
        return d;
    }

    // Set given mask to the reconstruction object
    void ReconstructionqMRI::SetMasksqMRI(Array<RealImage> Masks, vector<int> MaskStackNums, int templateNumber, double sigma, double threshold) {
        if (!_template_created)
            throw runtime_error("Please create the template before setting the mask, so that the mask can be resampled to the correct dimensions.");

        _mask.Initialize(_reconstructed.Attributes());

        for (int ii = 0; ii < Masks.size(); ii++) {
            RealImage mask_ii;
            mask_ii.Initialize(_reconstructed.Attributes());
            _Masks.push_back(mask_ii);
        }

        for (int ii = 0; ii < Masks.size(); ii++) {
            RealImage *mask_ii = &Masks[ii];
            if (mask_ii != NULL) {
                //if sigma is nonzero first smooth the mask
                if (sigma > 0) {
                    //blur mask
                    GaussianBlurring<RealPixel> gb(sigma);

                    gb.Input(mask_ii);
                    gb.Output(mask_ii);
                    gb.Run();

                    //binarise mask
                    *mask_ii = CreateMask(*mask_ii, threshold);
                }

                //resample the mask according to the template volume using identity transformation
                RigidTransformation transformation;
                ImageTransformation imagetransformation;

                //GenericNearestNeighborExtrapolateImageFunction<RealImage> interpolator;
                InterpolationMode interpolation = Interpolation_NN;
                unique_ptr<InterpolateImageFunction> interpolator(InterpolateImageFunction::New(interpolation));



                imagetransformation.Input(mask_ii);
                imagetransformation.Transformation(&transformation);
                imagetransformation.Output(&_Masks[ii]);

                //target is zero image, need padding -1
                imagetransformation.TargetPaddingValue(-1);
                //need to fill voxels in target where there is no info from source with zeroes
                imagetransformation.SourcePaddingValue(0);
                imagetransformation.Interpolator(interpolator.get());
                imagetransformation.Run();
                if (MaskStackNums[ii] == templateNumber) {
                    _mask = _Masks[ii];
                }
            } else {
                //fill the mask with ones
                _Masks[ii] = 1;
                if (MaskStackNums[ii] == templateNumber) {
                    _mask = 1;
                }
            }
        }
        //set flag that mask was created
        _have_mask = true;
        _evaluation_mask = _mask;

        // compute mask volume
        double vol = 0;
        RealPixel *pm = _evaluation_mask.Data();
        #pragma omp parallel for reduction(+: vol)
        for (int i = 0; i < _mask.NumberOfVoxels(); i++)
            if (pm[i] > 0.1)
                vol++;

        vol *= _reconstructed.GetXSize() * _reconstructed.GetYSize() * _reconstructed.GetZSize() / 1000;

        cout << "ROI volume : " << vol << " cc " << endl;

        if (_debug) {
            _mask.Write("mask.nii.gz");
            for (int ii = 0; ii < Masks.size(); ii++) {
                _Masks[ii].Write((boost::format("masks_stack_%1%.nii.gz") % MaskStackNums[ii]).str().c_str());
            }
        }
    }

    //-------------------------------------------------------------------


    // run SR reconstruction step
    void ReconstructionqMRI::SuperresolutionqMRI(int iter) {
        SVRTK_START_TIMING();

        // save current reconstruction for edge-preserving smoothing
        const RealImage original = _reconstructed4D;

        SliceDifferenceqMRI();

        ParallelqMRI::SuperresolutionqMRI parallelSuperresolution(this);
        parallelSuperresolution();

        RealImage &addon4D = parallelSuperresolution.addon;

        _confidence_map4D = move(parallelSuperresolution.confidence_map);
        //_confidence4mask = _confidence_map;



        if (_debug) {
            _confidence_map4D.Write((boost::format("confidence-map%1%.nii.gz") % iter).str().c_str());
        }

        _SR_iter = iter;

        if (!_adaptive) {
            RealPixel *pa = addon4D.Data();
            RealPixel *pcm = _confidence_map4D.Data();
            #pragma omp parallel for
            for (int i = 0; i < addon4D.NumberOfVoxels(); i++) {
                if (pcm[i] > 0) {
                    // ISSUES if pcm[i] is too small leading to bright pixels
                    pa[i] /= pcm[i];
                    //this is to revert to normal (non-adaptive) regularisation
                    pcm[i] = 1;
                }
            }
        }

        if (_debug) {
            addon4D.Write((boost::format("addon%1%.nii.gz") % iter).str().c_str());
        }

        // update the volume with computed addon
        //_reconstructed4D += addon * _alpha; //_average_volume_weight;

        //bound the intensities_average_volume_weight
        /*RealPixel *pr = _reconstructed4D.Data();
        #pragma omp parallel for
        for (int i = 0; i < _reconstructed4D.NumberOfVoxels(); i++) {
            if (pr[i] < _min_intensity * 0.9)
                pr[i] = _min_intensity * 0.9;
            if (pr[i] > _max_intensity * 1.1)
                pr[i] = _max_intensity * 1.1;
        }*/

        /*if (_debug){
            cout << "addon: " << addon4D(51,49,57,1) << " reconstructed: " << _reconstructed4D(51,49,57,1) << endl;
        }*/





        // Regularise with the model fit
        if (_jointSR){
            SVRTK_END_TIMING("SuperresolutionqMRI");
            SVRTK_RESET_TIMING();
            ParallelqMRI::ModelFitqMRI ModelFit(this);
            ModelFit();
            RealImage& addon2_4D = ModelFit.addon2;
            RealImage& T2MapIterm = ModelFit.T2Map;

           /* if (_debug){
                cout << "addon: " << addon4D(51,49,57,1) <<
                     " reconstructed: " << _reconstructed4D(51,49,57,1) <<
                     " addon2: " << addon2_4D(51,49,57,1) << endl;
            }*/

            //cout << T2MapIterm.GetX() << " " << T2MapIterm.GetY() << " " << T2MapIterm.GetZ() << endl;

            if (_debug) {
                addon2_4D.Write((boost::format("ModelAddon%1%.nii.gz") % iter).str().c_str());
                _reconstructed4D.Write((boost::format("IntermediateReconsTest%1%.nii.gz") % iter).str().c_str());
                T2MapIterm.Write((boost::format("IntermediateT2MapTest%1%.nii.gz") % iter).str().c_str());
            }

            // update the volume with computed addon
            _reconstructed4D += addon4D * _alpha; //_average_volume_weight;
            /*RealPixel *pa4d = addon4D.Data();
            RealPixel *pcr4d = _reconstructed4D.Data();
            #pragma omp parallel for
            for (int i = 0; i < _reconstructed4D.NumberOfVoxels(); i++) {
                pcr4d[i] += _alpha*pa4d[i];
            }*/

            if (_debug) {
                _reconstructed4D.Write((boost::format("IntermediateRecons2Test%1%.nii.gz") % iter).str().c_str());
            }

            /*if (_debug){
                cout << "addon: " << addon4D(51,49,57,1) <<
                     " reconstructed: " << _reconstructed4D(51,49,57,1) <<
                     " addon2: " << addon2_4D(51,49,57,1) << endl;
            }*/

            // update the volume with computed addon for regularisation with model
            _reconstructed4D -= addon2_4D * _alpha * _epsilon; // _Step; //_average_volume_weight;

            /*RealPixel *pa4d2 = addon2_4D.Data();
            RealPixel *pcr4d2 = _reconstructed4D.Data();
            #pragma omp parallel for
            for (int i = 0; i < _reconstructed4D.NumberOfVoxels(); i++) {
                pcr4d2[i] -= _alpha*_epsilon*pa4d2[i];
            }*/

            /*if (_debug){
                cout << "addon: " << addon4D(51,49,57,1) <<
                     " reconstructed: " << _reconstructed4D(51,49,57,1) <<
                     " addon2: " << addon2_4D(51,49,57,1) << endl;
            }*/

            if (_debug) {
                _reconstructed4D.Write((boost::format("IntermediateRecons3Test%1%.nii.gz") % iter).str().c_str());
            }

            //bound the intensities
            RealPixel *pr = _reconstructed4D.Data();
            RealPixel *pi = _volumeLabel4D.Data();
            //#pragma omp parallel for
            for (int i = 0; i < _reconstructed4D.NumberOfVoxels(); i++) {
                if (pr[i] < _min_intensitiesqMRI[pi[i]] * 0.9)
                    pr[i] = _min_intensitiesqMRI[pi[i]] * 0.9;
                if (pr[i] > _max_intensitiesqMRI[pi[i]] * 1.1)
                    pr[i] = _max_intensitiesqMRI[pi[i]] * 1.1;
            }

            /*
            const int dx = _reconstructed4D.GetX();
            const int dy = _reconstructed4D.GetY();
            const int dz = _reconstructed4D.GetZ();
            const int dt = _reconstructed4D.GetT();
            //#pragma omp parallel for
            for (int x = 0; x < dx; x++){
                for (int y = 0; y < dy; y++) {
                    for (int z = 0; z < dz; z++) {
                        for (int t = 0; t < dt; t++) {
                            if (_reconstructed4D(x, y, z, t) > _max_intensitiesqMRI[t] * 1.1)
                                _reconstructed4D(x, y, z, t) = _max_intensitiesqMRI[t] * 1.1;
                            if (_reconstructed4D(x, y, z, t) < _min_intensitiesqMRI[t] * 0.9)
                                _reconstructed4D(x, y, z, t) = _min_intensitiesqMRI[t] * 0.9;

                        }
                    }
                }
            }
            */
            SVRTK_END_TIMING("ModelFitqMRI");
            SVRTK_RESET_TIMING();
            //Smooth the reconstructed image with regularisation
            AdaptiveRegularizationqMRI(iter, original);
            SVRTK_END_TIMING("AdaptiveRegularizationqMRI");

            if (_debug) {
                _reconstructed4D.Write((boost::format("IntermediateRecons4Test%1%.nii.gz") % iter).str().c_str());
            }

        }
        else {

            //_reconstructed4D.Write((boost::format("BeforeAddon%1%.nii.gz") % iter).str().c_str());

            // update the volume with computed addon
            _reconstructed4D += addon4D * _alpha; //_average_volume_weight;

            //_reconstructed4D.Write((boost::format("AfterAddon%1%.nii.gz") % iter).str().c_str());

            //bound the intensities
            RealPixel *pr = _reconstructed4D.Data();
            RealPixel *pi = _volumeLabel4D.Data();
            #pragma omp parallel for
            for (int i = 0; i < _reconstructed4D.NumberOfVoxels(); i++) {
                if (pr[i] < _min_intensitiesqMRI[pi[i]] * 0.9)
                    pr[i] = _min_intensitiesqMRI[pi[i]] * 0.9;
                if (pr[i] > _max_intensitiesqMRI[pi[i]] * 1.1)
                    pr[i] = _max_intensitiesqMRI[pi[i]] * 1.1;
            }

            /*
            const int dx = _reconstructed4D.GetX();
            const int dy = _reconstructed4D.GetY();
            const int dz = _reconstructed4D.GetZ();
            const int dt = _reconstructed4D.GetT();
            //#pragma omp parallel for
            for (int x = 0; x < dx; x++){
                for (int y = 0; y < dy; y++) {
                    for (int z = 0; z < dz; z++) {
                        for (int t = 0; t < dt; t++) {
                            if (_reconstructed4D(x, y, z, t) > _max_intensitiesqMRI[t] * 1.1)
                                _reconstructed4D(x, y, z, t) = _max_intensitiesqMRI[t] * 1.1;
                            if (_reconstructed4D(x, y, z, t) < _min_intensitiesqMRI[t] * 0.9)
                                _reconstructed4D(x, y, z, t) = _min_intensitiesqMRI[t] * 0.9;

                        }
                    }
                }
            }*/

            SVRTK_END_TIMING("SuperresolutionqMRI");
            SVRTK_RESET_TIMING();

            //Smooth the reconstructed image with regularisation
            AdaptiveRegularizationqMRI(iter, original);

            //_reconstructed4D.Write((boost::format("AfterAdaptive%1%.nii.gz") % iter).str().c_str());

            SVRTK_END_TIMING("AdaptiveRegularizationqMRI");
        }



        //Remove the bias in the reconstructed volume compared to previous iteration
        if (_global_bias_correction)
            BiasCorrectVolumeqMRI(original);

        // Update the template reconstruction
        /*_reconstructed = _reconstructed4D.GetRegion(0, 0, 0, _volume_index[_templateNumber], _reconstructed4D.GetX(),
                                                    _reconstructed4D.GetY(), _reconstructed4D.GetZ(), _volume_index[_templateNumber] + 1)*/;


    }


    // run adaptive regularisation of the SR reconstructed volume
    void ReconstructionqMRI::AdaptiveRegularizationqMRI(int iter, const RealImage& original) {
        Array<double> factor(13);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }

        Array<RealImage> b(13, _reconstructed4D);

        ParallelqMRI::AdaptiveRegularization1qMRI parallelAdaptiveRegularization1(this, b, factor, original);
        parallelAdaptiveRegularization1();

        const RealImage original2 = _reconstructed4D;
        ParallelqMRI::AdaptiveRegularization2qMRI parallelAdaptiveRegularization2(this, b, original2);
        parallelAdaptiveRegularization2();

        if (_alpha * _lambda / (_delta * _delta) > 0.068)
            cerr << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068." << endl;
    }


    // run adaptive regularisation of the SR reconstructed volume
    void ReconstructionqMRI::AdaptiveRegularizationT2Map(int iter, const RealImage& original) {
        Array<double> factor(13);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }

        Array<RealImage> b(13, _T2Map);

        ParallelqMRI::AdaptiveRegularization1T2Map parallelAdaptiveRegularization1(this, b, factor, original);
        parallelAdaptiveRegularization1();

        const RealImage original2 = _T2Map;
        ParallelqMRI::AdaptiveRegularization2T2Map parallelAdaptiveRegularization2(this, b, original2);
        parallelAdaptiveRegularization2();

        if (_alpha * _lambda / (_delta * _delta) > 0.068)
            cerr << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068." << endl;
    }


    void ReconstructionqMRI::GaussianReconstructionqMRI(int iter){
        SVRTK_START_TIMING();

        RealImage slice;
        Array<Array<int>> voxel_num_qMRI(_nEchoTimes);
        for (int ii = 0; ii < _nEchoTimes; ii++){
            voxel_num_qMRI[ii].reserve(_NumSlicesPerVolume[ii]);
        }
        Array<int> voxel_num;
        voxel_num.reserve(_slicesqMRI.size());

        //clear _reconstructed image
        memset(_reconstructed4D.Data(), 0, sizeof(RealPixel) * _reconstructed4D.NumberOfVoxels());
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            int indstack = _stack_index[inputIndex];
            int outputIndex = _volume_index[indstack];
            if (_volcoeffs[inputIndex].empty())
                continue;

            int slice_vox_num = 0;
            //copy the current slice
            slice = _slicesqMRI[inputIndex];
            //alias the current bias image
            const RealImage& b = _biasqMRI[inputIndex];
            //read current scale factor
            const double scale = _scaleqMRI[inputIndex];

            //Distribute slice intensities to the volume
            for (size_t i = 0; i < _volcoeffs[inputIndex].size(); i++)
                for (size_t j = 0; j < _volcoeffs[inputIndex][i].size(); j++)
                    if (slice(i, j, 0) > -0.01) {

                        double jac = 1;
                        if (_ffd) {
                            double x = i, y = j, z = 0;
                            _slicesqMRI[inputIndex].ImageToWorld(x, y, z);
                            jac = _mffd_transformations[inputIndex]->Jacobian(x, y, z, 0, 0);
                        } else {
                            jac = 1;
                        }

                        if ((100*jac) > 50) {

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
                                _reconstructed4D(p.x, p.y, p.z, outputIndex) += p.value * slice(i, j, 0);
                            }
                        }
                    }
            voxel_num.push_back(slice_vox_num);
            voxel_num_qMRI[outputIndex].push_back(slice_vox_num);
            //end of loop for a slice inputIndex
        }

        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxel
        _reconstructed4D /= _volume_weightsqMRI;

        for (int t = 0; t < _reconstructed4D.GetT(); t++) {
            RealImage init = _reconstructed4D.GetRegion(0, 0, 0, t, _reconstructed4D.GetX(), _reconstructed4D.GetY(),
                                                        _reconstructed4D.GetZ(), t + 1);
            double TEtoWrite = _echoTimes[t];
            if (_debug)
                init.Write((boost::format("init_TE_%1%ms.nii.gz") % TEtoWrite).str().c_str());
            /*if (t == _volume_index[_templateNumber]) {
                _reconstructed = _reconstructed4D.GetRegion(0, 0, 0, t, _reconstructed4D.GetX(),
                                                            _reconstructed4D.GetY(), _reconstructed4D.GetZ(), t + 1);
            }*/
        }

        // find slices with small overlap with ROI and exclude them.
        //find median
        Array<int> voxel_num_tmp = voxel_num;
        int median = round(voxel_num_tmp.size() * 0.5) - 1;
        nth_element(voxel_num_tmp.begin(), voxel_num_tmp.begin() + median, voxel_num_tmp.end());
        median = voxel_num_tmp[median];
        Array<int> medianqMRI;
        for (int ii = 0; ii < _nEchoTimes; ii++){
            Array<int> voxel_num_tmp2 = voxel_num_qMRI[ii];
            int median2 = round(voxel_num_tmp2.size() * 0.5) - 1;
            nth_element(voxel_num_tmp2.begin(), voxel_num_tmp2.begin() + median2, voxel_num_tmp2.end());
            medianqMRI.push_back(voxel_num_tmp2[median2]);
        }


        //remember slices with small overlap with ROI
        ClearAndReserve(_small_slices, voxel_num.size());
        /*for (size_t i = 0; i < voxel_num.size(); i++)
            if (voxel_num[i] < 0.1 * median)
                _small_slices.push_back(i);*/
        int ii = 0;
        for(size_t i = 0; i < voxel_num_qMRI.size();i++){
            for(size_t j = 0; j < voxel_num_qMRI[i].size(); j++){
                if (voxel_num_qMRI[i][j] < 0.1 * medianqMRI[i])
                    _small_slices.push_back(ii);
                ii++;
            }
        }


        if (_verbose) {
            _verbose_log << "Small slices:";
            for (size_t i = 0; i < _small_slices.size(); i++)
                _verbose_log << " " << _small_slices[i];
            _verbose_log << endl;
        }

        SVRTK_END_TIMING("GaussianReconstructionqMRI");

    }

    //-------------------------------------------------------------------

    // run calculation of transformation matrices
    void ReconstructionqMRI::CoeffInitqMRI() {
        SVRTK_START_TIMING();

        //resize slice-volume matrix from previous iteration
        ClearAndResize(_volcoeffs, _slicesqMRI.size());

        //resize indicator of slice having and overlap with volumetric mask
        ClearAndResize(_slice_inside, _slicesqMRI.size());
        _attr_reconstructed4D = _reconstructed4D.Attributes();
        _attr_reconstructed = _reconstructed.Attributes();

        Parallel::CoeffInit coeffinit(this);
        coeffinit();


        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        _volume_weights.Initialize(_reconstructed.Attributes());
        _volume_weightsqMRI.Initialize(_reconstructed4D.Attributes());

        // Do not parallelise: It would cause data inconsistencies
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            bool excluded = false;

            for (size_t fe = 0; fe < _force_excluded.size(); fe++) {
                if (inputIndex == _force_excluded[fe]) {
                    excluded = true;
                    break;
                }
            }

            if (_structural_slice_weight[inputIndex] < 0.5)
                excluded = true;

            if (!excluded) {

                // Do not parallelise: It would cause data inconsistencies
                for (int i = 0; i < _slicesqMRI[inputIndex].GetX(); i++)
                    for (int j = 0; j < _slicesqMRI[inputIndex].GetY(); j++)
                        for (size_t k = 0; k < _volcoeffs[inputIndex][i][j].size(); k++) {
                            const POINT3D& p = _volcoeffs[inputIndex][i][j][k];

                            if (_ffd) {
                                double x = i, y = j, z = 0;
                                _slicesqMRI[inputIndex].ImageToWorld(x, y, z);
                                double jac = _mffd_transformations[inputIndex]->Jacobian(x, y, z, 0, 0);
                                if ((100*jac) > 50) {
                                    int indstack = _stack_index[inputIndex];
                                    int outputIndex = _volume_index[indstack];
                                    _volume_weights(p.x, p.y, p.z) += p.value;
                                    _volume_weightsqMRI(p.x, p.y, p.z,outputIndex) += p.value;
                                }
                            } else {
                                int indstack = _stack_index[inputIndex];
                                int outputIndex = _volume_index[indstack];
                                _volume_weights(p.x, p.y, p.z) += p.value;
                                _volume_weightsqMRI(p.x, p.y, p.z,outputIndex) += p.value;
                            }

                        }
            }

        }

        if (_ffd) {
            for (int z = 0; z < _volume_weights.GetZ(); z++)
                for (int y = 0; y < _volume_weights.GetY(); y++)
                    for (int x = 0; x < _volume_weights.GetX(); x++)
                        if (_volume_weights(x,y,z) < 0.0001) {
                            _volume_weights(x,y,z) = 1;
                        }
        }


        if (_debug) {
            _volume_weightsqMRI.Write("volume_weights4D.nii.gz");
            _volume_weights.Write("volume_weights.nii.gz");
        }

        //find average volume weight to modify alpha parameters accordingly
        const RealPixel *ptr = _volume_weightsqMRI.Data();
        const RealPixel *pm = _mask.Data();
        double sum = 0;
        int num = 0;
        #pragma omp parallel for reduction(+: sum, num)
        for (int i = 0; i < _volume_weightsqMRI.NumberOfVoxels(); i++) {
            if (pm[i] == 1) {
                sum += ptr[i];
                num++;
            }
        }
        _average_volume_weight = sum / num;

        if (_verbose)
            _verbose_log << "Average volume weight is " << _average_volume_weight << endl;

        SVRTK_END_TIMING("CoeffInitqMRI");
    }

    //-------------------------------------------------------------------

    void ReconstructionqMRI::SimulateSlicesqMRI() {
        SVRTK_START_TIMING();
        ParallelqMRI::SimulateSlicesqMRI p_sim(this);
        p_sim();
        SVRTK_END_TIMING("SimulateSlicesqMRI");
    }

    //-------------------------------------------------------------------

    void ReconstructionqMRI::SaveSimulatedSlicesqMRI() {
        cout << "Saving simulated slices ... ";
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++)
            _simulated_slicesqMRI[inputIndex].Write((boost::format("simsliceqMRI%1%.nii.gz") % inputIndex).str().c_str());
        cout << "done." << endl;
    }


    //-------------------------------------------------------------------

    // Normalise Bias
    void ReconstructionqMRI::NormaliseBiasqMRI(int iter) {
        SVRTK_START_TIMING();

        ParallelqMRI::NormaliseBiasqMRI parallelNormaliseBias(this);
        parallelNormaliseBias();
        RealImage& bias = parallelNormaliseBias.bias;

        // normalize the volume by proportion of contributing slice voxels for each volume voxel
        bias /= _volume_weightsqMRI;

        StaticMaskVolume4D(bias, _mask, 0);
        RealImage m = _mask;
        GaussianBlurring<RealPixel> gb(_sigma_biasqMRI,_sigma_biasqMRI,_sigma_biasqMRI,0);
        GaussianBlurring<RealPixel> gb2(_sigma_biasqMRI);

        gb.Input(&bias);
        gb.Output(&bias);
        gb.Run();

        gb2.Input(&m);
        gb2.Output(&m);
        gb2.Run();
        //m.Write("MaskBiasCorrection.nii.gz");
        for (int t = 0; t < bias.GetT(); t++) {
            for (int i = 0; i < bias.GetX(); i++) {
                for (int j = 0; j < bias.GetY(); j++) {
                    for (int k = 0; k < bias.GetZ(); k++) {
                        bias(i, j, k, t) /= m(i, j, k);
                    }
                }
            }
        }

        if (_debug)
            bias.Write((boost::format("averagebias%1%.nii.gz") % iter).str().c_str());

        RealPixel *pi = _reconstructed4D.Data();
        const RealPixel *pb = bias.Data();
        #pragma omp parallel for
        for (int i = 0; i < _reconstructed4D.NumberOfVoxels(); i++)
            if (pi[i] != -1)
                pi[i] /= exp(-(pb[i]));

        SVRTK_END_TIMING("NormaliseBiasqMRI");
    }

    //-------------------------------------------------------------------

    //-------------------------------------------------------------------

    void ReconstructionqMRI::FindStackAverage(Array<RealImage>& stacks, const Array<RigidTransformation>& stack_transformations){
        _stack_averages.reserve(stacks.size());

        //Calculate the averages of intensities for all stacks in the mask ROI
        for (size_t ind = 0; ind < stacks.size(); ind++) {
            double sum = 0, num = 0;

            //#pragma omp parallel for reduction(+: sum, num)
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
                                sum += stacks[ind](i, j, k);
                                num++;
                            }
                        }
                    }

            //calculate average for the stack
            if (num > 0) {
                _stack_averages.push_back(sum / num);
            } else {
                throw runtime_error("Stack " + to_string(ind) + " has no overlap with ROI");
            }
        }
    }

    //-------------------------------------------------------------------

    // match stack intensities with respect to the masked ROI
    void ReconstructionqMRI::MatchStackIntensitiesWithMaskingqMRI(Array<RealImage>& stacks, const Array<RigidTransformation>& stack_transformations, double averageValue, bool together) {
        SVRTK_START_TIMING();

        RealImage m;
        Array<double> stack_average;
        stack_average.reserve(stacks.size());
        _stack_averages.reserve(stacks.size());

        //remember the set average value
        _average_value = averageValue;

        //Calculate the averages of intensities for all stacks in the mask ROI
        for (size_t ind = 0; ind < stacks.size(); ind++) {
            double sum = 0, num = 0;

            if (_debug)
                m = stacks[ind];

            //#pragma omp parallel for reduction(+: sum, num)
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
                _stack_averages.push_back(sum / num);
            } else {
                throw runtime_error("Stack " + to_string(ind) + " has no overlap with ROI");
            }
        }


        vector<double> global_averages;
        for (size_t i = 0; i < _nEchoTimes; i++)
            global_averages.push_back(0);

        if (together) {
            for (size_t i = 0; i < stack_average.size(); i++) {
                int volind = _volume_index[i];
                global_averages[volind] = global_averages[volind] + stack_average[i];
            }
            for (size_t i = 0; i < global_averages.size(); i++)
                global_averages[i] /= _StacksPerEchoTime[i];
        }

        if (_verbose) {
            _verbose_log << "Stack average intensities are ";
            for (size_t ind = 0; ind < stack_average.size(); ind++)
                _verbose_log << stack_average[ind] << " ";
            _verbose_log << endl;
            _verbose_log << "The new average value is " << averageValue << endl;
        }

        //Rescale stacks
        ClearAndReserve(_stack_factorqMRI, stacks.size());
        for (size_t ind = 0; ind < stacks.size(); ind++) {
            int volind = _volume_index[ind];
            double factor = averageValue / (together ? global_averages[volind] : stack_average[ind]);
            _stack_factorqMRI.push_back(factor);

            RealPixel *ptr = stacks[ind].Data();
            #pragma omp parallel for
            for (int i = 0; i < stacks[ind].NumberOfVoxels(); i++)
                if (ptr[i] > 0)
                    ptr[i] *= factor;
        }

        if (_debug) {
            #pragma omp parallel for
            for (size_t ind = 0; ind < stacks.size(); ind++)
                stacks[ind].Write((boost::format("rescaled-stackqMRI%1%.nii.gz") % ind).str().c_str());
        }

        if (_verbose) {
            _verbose_log << "Slice intensity factors are ";
            for (size_t ind = 0; ind < stack_average.size(); ind++)
                _verbose_log << _stack_factorqMRI[ind] << " ";
            _verbose_log << endl;
            _verbose_log << "The new average value is " << averageValue << endl;
        }

        SVRTK_END_TIMING("MatchStackIntensitiesWithMaskingqMRI");
    }

    //-------------------------------------------------------------------

    // restore the original slice intensities
    void ReconstructionqMRI::RestoreSliceIntensitiesqMRI() {
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            //calculate scaling factor
            const double& factor = _stack_factorqMRI[_stack_index[inputIndex]];//_average_value;
            //_slices[inputIndex].Write((boost::format("Scaledslice%1%.nii.gz") % inputIndex).str().c_str());
            // read the pointer to current slice
            RealPixel *p = _slicesqMRI[inputIndex].Data();
            for (int i = 0; i < _slicesqMRI[inputIndex].NumberOfVoxels(); i++)
                if (p[i] > 0)
                    p[i] /= factor;
        }
    }

    //-------------------------------------------------------------------

    // scale the reconstructed volume
    vector<double> ReconstructionqMRI::ScaleVolumeqMRI(RealImage& reconstructed,bool midRecon) {
        Array<double> scalenum, scaleden;
        scalenum.reserve(_nEchoTimes);
        scaleden.reserve(_nEchoTimes);

        for (int i = 0; i < _nEchoTimes; i++)
        {
            scalenum.push_back(0);
            scaleden.push_back(0);
        }

        //#pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {

            int indstack = _stack_index[inputIndex];
            int outputIndex = _volume_index[indstack];

            // alias for the current slice
            RealImage slice = _slicesqMRI[inputIndex];

            const double& factor = _stack_factorqMRI[_stack_index[inputIndex]];//_average_value;
            if (midRecon){
                //_slices[inputIndex].Write((boost::format("Scaledslice%1%.nii.gz") % inputIndex).str().c_str());
                // read the pointer to current slice
                RealPixel *p = slice.Data();
                for (int i = 0; i < slice.NumberOfVoxels(); i++)
                    if (p[i] > 0)
                        p[i] /= factor;
            }

            //alias for the current weight image
            const RealImage& w = _weightsqMRI[inputIndex];

            // alias for the current simulated slice
            const RealImage& sim = _simulated_slicesqMRI[inputIndex];

            for (int i = 0; i < slice.GetX(); i++)
                for (int j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {
                        //scale - intensity matching
                        if (_simulated_weightsqMRI[inputIndex](i, j, 0) > 0.99) {
                            scalenum[outputIndex] += w(i, j, 0) * _slice_weightqMRI[inputIndex] * slice(i, j, 0) * sim(i, j, 0);
                            scaleden[outputIndex] += w(i, j, 0) * _slice_weightqMRI[inputIndex] * sim(i, j, 0) * sim(i, j, 0);
                        }
                    }

        } //end of loop for a slice inputIndex

        //calculate scale for the volume
        vector<double> scale;
        scale.reserve(_nEchoTimes);

        for (int i = 0; i < _nEchoTimes; i++) {
            if (scaleden[i] > 0)
                scale.push_back(scalenum[i] / scaleden[i]);
        }

        if (_verbose) {
            _verbose_log << "scales : ";
            for (int i = 0; i < _nEchoTimes; i++)
                _verbose_log << scale[i] << " ";
            _verbose_log << endl;
        }

        //RealPixel *ptr = reconstructed.Data();
        //#pragma omp parallel for

        RealPixel *ptr = reconstructed.Data();
        RealPixel *pi = _volumeLabel4D.Data();
        #pragma omp parallel for
        for (int i = 0; i < reconstructed.NumberOfVoxels(); i++)
            if (ptr[i] > 0)
                ptr[i] *= scale[pi[i]];


        /*for (int i = 0; i < reconstructed.GetX(); i++) {
            for (int j = 0; j < reconstructed.GetY(); j++) {
                for (int k = 0; k < reconstructed.GetZ(); k++) {
                    for (int t = 0; t < reconstructed.GetT(); t++) {
                        reconstructed(i, j, k, t) *= scale[t];
                    }
                }
            }
        }*/
        /*
        for (int i = 0; i < reconstructed.NumberOfVoxels(); i++)
            if (ptr[i] > 0)
                ptr[i] *= scale;*/
        return scale;
    }

    //-------------------------------------------------------------------

    // correct bias
    void ReconstructionqMRI::BiasCorrectVolumeqMRI(const RealImage& original) {
        //remove low-frequency component in the reconstructed image which might have occurred due to overfitting of the biasfield
        RealImage residual = _reconstructed4D;
        RealImage weights = _mask;

        //calculate weighted residual
        const RealPixel *po = original.Data();
        RealPixel *pr = residual.Data();
        RealPixel *pw = weights.Data();
        #pragma omp parallel for
        for (int i = 0; i < _reconstructed4D.NumberOfVoxels(); i++) {
            //second and term to avoid numerical problems
            if ((pw[i] == 1) && (po[i] > _low_intensity_cutoff * _max_intensity) && (pr[i] > _low_intensity_cutoff * _max_intensity)) {
                pr[i] = log(pr[i] / po[i]);
            } else {
                pw[i] = pr[i] = 0;
            }
        }

        //blurring needs to be same as for slices
        GaussianBlurring<RealPixel> gb(_sigma_biasqMRI);
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
        RealPixel *pi = _reconstructed4D.Data();
        #pragma omp parallel for
        for (int i = 0; i < _reconstructed4D.NumberOfVoxels(); i++) {
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

    // initialise slice EM step
    void ReconstructionqMRI::InitializeEMqMRI() {
        ClearAndReserve(_weights, _slicesqMRI.size());
        ClearAndReserve(_bias, _slicesqMRI.size());
        ClearAndReserve(_scale, _slicesqMRI.size());
        ClearAndReserve(_weightsqMRI, _slicesqMRI.size());
        ClearAndReserve(_biasqMRI, _slicesqMRI.size());
        ClearAndReserve(_scaleqMRI, _slicesqMRI.size());
        ClearAndReserve(_slice_weight, _slicesqMRI.size());
        ClearAndReserve(_slice_weightqMRI, _slicesqMRI.size());
        ClearAndReserve(_max_intensitiesqMRI,_nEchoTimes);
        ClearAndReserve(_min_intensitiesqMRI,_nEchoTimes);

        for (size_t i = 0; i < _slicesqMRI.size(); i++) {
            //Create images for voxel weights and bias fields
            _weights.push_back(_slicesqMRI[i]);
            _bias.push_back(_slicesqMRI[i]);
            _weightsqMRI.push_back(_slicesqMRI[i]);
            _biasqMRI.push_back(_slicesqMRI[i]);

            //Create and initialize scales
            _scale.push_back(1);
            _scaleqMRI.push_back(1);

            //Create and initialize slice weights
            _slice_weight.push_back(1);
            _slice_weightqMRI.push_back(1);
        }

        //Find the range of intensities
        for (size_t i = 0; i < _nEchoTimes; i++) {
            _max_intensitiesqMRI[i] = voxel_limits<RealPixel>::min();
            _min_intensitiesqMRI[i] = voxel_limits<RealPixel>::max();
        }

        //Find the range of intensities
        _max_intensity = voxel_limits<RealPixel>::min();
        _min_intensity = voxel_limits<RealPixel>::max();

        //#pragma omp parallel for reduction(min: _min_intensity) reduction(max: _max_intensity)
        for (size_t i = 0; i < _slicesqMRI.size(); i++) {
            //to update minimum we need to exclude padding value
            int indstack = _stack_index[i];
            int t = _volume_index[indstack];
            const RealPixel *ptr = _slicesqMRI[i].Data();
            for (int ind = 0; ind < _slicesqMRI[i].NumberOfVoxels(); ind++) {
                if (ptr[ind] > 0) {
                    if (ptr[ind] > _max_intensity) {
                        _max_intensity = ptr[ind];
                    }
                    if (ptr[ind] > _max_intensitiesqMRI[t]){
                        _max_intensitiesqMRI[t] = ptr[ind];
                    }
                    if (ptr[ind] < _min_intensity) {
                        _min_intensity = ptr[ind];
                    }
                    if (ptr[ind] < _min_intensitiesqMRI[t]){
                        _min_intensitiesqMRI[t] = ptr[ind];
                    }
                }
            }
        }
    }

    //-------------------------------------------------------------------

    // initialise / reset EM values
    void ReconstructionqMRI::InitializeEMValuesqMRI() {
        SVRTK_START_TIMING();

        #pragma omp parallel for
        for (size_t i = 0; i < _slicesqMRI.size(); i++) {
            //Initialise voxel weights and bias values
            _weightsqMRI[i] = _weights[i];
            _biasqMRI[i] = _bias[i];
            const RealPixel *pi = _slices[i].Data();
            RealPixel *pw = _weightsqMRI[i].Data();
            RealPixel *pb = _biasqMRI[i].Data();
            for (int j = 0; j < _weightsqMRI[i].NumberOfVoxels(); j++) {
                pw[j] = pi[j] > -0.01 ? 1 : 0;
                pb[j] = 0;
            }

            //Initialise slice weights
            _slice_weightqMRI[i] = 1;

            //Initialise scaling factors for intensity matching
            _scaleqMRI[i] = 1;
        }

        //Force exclusion of slices predefined by user
        for (size_t i = 0; i < _force_excluded.size(); i++)
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slicesqMRI.size())
                _slice_weightqMRI[_force_excluded[i]] = 0;

        SVRTK_END_TIMING("InitializeEMValues");
    }

    void ReconstructionqMRI::RemoveNegativeBackground(RealImage& image){
        RealPixel *pi = image.Data();
        #pragma omp parallel for
        for (int i = 0; i < image.NumberOfVoxels(); i++) {
            if (pi[i] < 0) {
                pi[i] = 0;
            }
        }
    }

    //-------------------------------------------------------------------

    // create slices from the input stacks
    void ReconstructionqMRI::CreateSlicesAndTransformationsqMRI(const Array<RealImage>& stacks, const Array<RigidTransformation>& stack_transformations,
                                                                const Array<double>& thickness, const Array<RealImage>& probability_maps, RealImage templateStack,
                                                                bool HistogramMatchingB,double averageValue) {
        double average_thickness = 0;

        Array<RealImage> greyStacksHistMatched;
        // Reset and allocate memory
        size_t reserve_size = 0;
        for (size_t i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            reserve_size += stacks[i].Attributes()._z;;
            RealImage greyStackHistMatched;
            HistogramMatching<RealPixel> match;
            if (HistogramMatchingB){
                match.Input(&stacks[i]);
                match.Reference(&templateStack);
                match.Output(&greyStackHistMatched);
                match.Run();
                if (_debug)
                    greyStackHistMatched.Write((boost::format("histmatchedStack%1%.nii.gz") % i).str().c_str());
            }
            greyStacksHistMatched.push_back(greyStackHistMatched);

        }
        if (HistogramMatchingB && _MatchIntensity) {
            MatchStackIntensitiesWithMasking(greyStacksHistMatched, stack_transformations, averageValue);
        }
        ClearAndReserve(_zero_slices, reserve_size);
        ClearAndReserve(_slices, reserve_size);
        ClearAndReserve(_slicesqMRI, reserve_size);
        ClearAndReserve(_slice_masks, reserve_size);
        ClearAndReserve(_slice_ssim_maps, reserve_size);
        ClearAndReserve(_package_index, reserve_size);
        ClearAndReserve(_slice_attributes, reserve_size);
        ClearAndReserve(_grey_slices, reserve_size);
        ClearAndReserve(_slice_dif, reserve_size);
        ClearAndReserve(_simulated_slices, reserve_size);
        ClearAndReserve(_structural_slice_weight, reserve_size);
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

//                if (excluded)
//                    continue;

                //create slice by selecting the appropriate region of the stack
                RealImage slice = stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                RealImage greyslice = stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                if (HistogramMatchingB){
                    greyslice = greyStacksHistMatched[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                    greyslice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                }
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

                if (_blurring) {
                    GaussianBlurring<RealPixel> gbt(0.6 * slice.GetXSize());
                    gbt.Input(&greyslice);
                    gbt.Output(&greyslice);
                    gbt.Run();
                }

                if (HistogramMatchingB){
                    _slices.push_back(greyslice);
                    _grey_slices.push_back(greyslice);
                }
                else{
                    _slices.push_back(slice);
                    _grey_slices.push_back(slice);
                }
                _slicesqMRI.push_back(slice);
                _package_index.push_back(current_package);
                _slice_attributes.push_back(slice.Attributes());


                memset(slice.Data(), 0, sizeof(RealPixel) * slice.NumberOfVoxels());
                _slice_dif.push_back(slice);
                _simulated_slices.push_back(slice);
                _slice_masks.push_back(slice);
                _slice_ssim_maps.push_back(slice);
                _structural_slice_weight.push_back(1);
                _slice_pos.push_back(j);
                slice = 1;
                _simulated_weights.push_back(slice);
                _simulated_inside.push_back(move(slice));
                //remember stack index for this slice
                _stack_index.push_back(i);
                //initialize slice transformation with the stack transformation
                _transformations.push_back(stack_transformations[i]);

                // if non-rigid FFD registartion option was selected
                if (_ffd) {
                    MultiLevelFreeFormTransformation* mffd = new MultiLevelFreeFormTransformation();
                    _mffd_transformations.push_back(mffd);
                }

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

    // mask slices based on the reconstruction mask
    void ReconstructionqMRI::MaskSlicesqMRI() {
        //Check whether we have a mask
        if (!_have_mask) {
            cerr << "Could not mask slices because no mask has been set." << endl;
            return;
        }

        Utility::MaskSlices(_slicesqMRI, _mask, [&](size_t index, double& x, double& y, double& z) {
            if (!_ffd)
                _transformations[index].Transform(x, y, z);
            else
                _mffd_transformations[index]->Transform(-1, 1, x, y, z);
        });
    }

    //-------------------------------------------------------------------

    void ReconstructionqMRI::CreateSimulatedSlices(const Array<RealImage> stacks, const Array<double> thickness) {
        double average_thickness = 0;
        // Reset and allocate memory
        size_t reserve_size = 0;
        for (size_t i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            reserve_size += stacks[i].Attributes()._z;;
        }

        ClearAndReserve(_slice_difqMRI, reserve_size);
        ClearAndReserve(_simulated_slicesqMRI, reserve_size);
        ClearAndReserve(_simulated_weightsqMRI, reserve_size);
        ClearAndReserve(_simulated_insideqMRI, reserve_size);

        ClearAndReserve(_NumSlicesPerVolume,_nEchoTimes);

        int curVol = 0;
        int n_slices_in_vol = 0;

        for (size_t i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            const ImageAttributes &attr = stacks[i].Attributes();

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
                //create slice by selecting the appropriate region of the stack
                RealImage slice = stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                //set correct voxel size in the stack. Z size is equal to slice thickness.
                slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                //remember the slice

                _volume_index_for_slices.push_back(_volume_index[i]);

                _slice_difqMRI.push_back(slice);
                _simulated_slicesqMRI.push_back(slice);
                slice = 1;
                _simulated_weightsqMRI.push_back(slice);
                _simulated_insideqMRI.push_back(move(slice));

                n_slices_in_vol++;

            }
            if (i != stacks.size() - 1){
                if (_volume_index[i+1] != curVol) {
                    _NumSlicesPerVolume[curVol] = n_slices_in_vol;
                    n_slices_in_vol = 0;
                    curVol = _volume_index[i+1];
                }
            }
            else
                _NumSlicesPerVolume[curVol] = n_slices_in_vol;
        }
    }

    //-------------------------------------------------------------------

    // compute difference between simulated and original slices
    void ReconstructionqMRI::SliceDifferenceqMRI() {
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            _slice_difqMRI[inputIndex] = _slicesqMRI[inputIndex];

            for (int i = 0; i < _slicesqMRI[inputIndex].GetX(); i++) {
                for (int j = 0; j < _slicesqMRI[inputIndex].GetY(); j++) {
                    if (_slicesqMRI[inputIndex](i, j, 0) > -0.01) {
                        _slice_difqMRI[inputIndex](i, j, 0) *= exp(-(_biasqMRI[inputIndex])(i, j, 0)) * _scaleqMRI[inputIndex];
                        _slice_difqMRI[inputIndex](i, j, 0) -= _simulated_slicesqMRI[inputIndex](i, j, 0);
                    } else
                        _slice_difqMRI[inputIndex](i, j, 0) = 0;
                }
            }
        }
    }

    //-------------------------------------------------------------------

    // run slice bias correction
    void ReconstructionqMRI::BiasqMRI() {
        SVRTK_START_TIMING();
        ParallelqMRI::BiasqMRI parallelBias(this);
        parallelBias();
        SVRTK_END_TIMING("BiasqMRI");
    }

    //-------------------------------------------------------------------

    // run slice scaling
    void ReconstructionqMRI::ScaleqMRI() {
        SVRTK_START_TIMING();

        ParallelqMRI::ScaleqMRI parallelScale(this);
        parallelScale();

        if (_verbose) {
            _verbose_log << setprecision(3);
            _verbose_log << "Slice scale =";
            for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++)
                _verbose_log << " " << _scaleqMRI[inputIndex];
            _verbose_log << endl;
        }

        SVRTK_END_TIMING("ScaleqMRI");
    }

    //-------------------------------------------------------------------

    // initialise parameters of EM robust statistics
    void ReconstructionqMRI::InitializeRobustStatisticsqMRI() {
        Array<int> sigma_numbers(_slicesqMRI.size());
        Array<double> sigma_values(_slicesqMRI.size());
        RealImage slice;

        ClearAndReserve(_sigma_qMRI,_nEchoTimes);
        ClearAndReserve(_mix_qMRI,_nEchoTimes);
        ClearAndReserve(_mean_s_qMRI,_nEchoTimes);
        ClearAndReserve(_sigma_s_qMRI,_nEchoTimes);
        ClearAndReserve(_mean_s2_qMRI,_nEchoTimes);
        ClearAndReserve(_sigma_s2_qMRI,_nEchoTimes);
        ClearAndReserve(_mix_s_qMRI,_nEchoTimes);

        #pragma omp parallel for private(slice)
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            slice = _slicesqMRI[inputIndex];
            //Voxel-wise sigma will be set to stdev of volumetric errors
            //For each slice voxel
            for (int i = 0; i < slice.GetX(); i++)
                for (int j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) > -0.01) {
                        //calculate stdev of the errors
                        if (_simulated_insideqMRI[inputIndex](i, j, 0) == 1 && _simulated_weightsqMRI[inputIndex](i, j, 0) > 0.99) {
                            slice(i, j, 0) -= _simulated_slicesqMRI[inputIndex](i, j, 0);
                            sigma_values[inputIndex] += slice(i, j, 0) * slice(i, j, 0);
                            sigma_numbers[inputIndex]++;
                        }
                    }

            //if slice does not have an overlap with ROI, set its weight to zero
            if (!_slice_inside[inputIndex])
                _slice_weightqMRI[inputIndex] = 0;
        }

        double sigma = 0;
        int num = 0;
        vector<double> sigma_qMRI;
        vector<int> num_qMRI;
        for (int ii = 0; ii < _nEchoTimes; ii++){
            num_qMRI.push_back(0);
            sigma_qMRI.push_back(0);
        }

        #pragma omp simd
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            int indstack = _stack_index[inputIndex];
            int t = _volume_index[indstack];
            sigma += sigma_values[inputIndex];
            num += sigma_numbers[inputIndex];
            sigma_qMRI[t] += sigma_values[inputIndex];
            num_qMRI[t] += sigma_numbers[inputIndex];
        }



        //Force exclusion of slices predefined by user
        for (size_t i = 0; i < _force_excluded.size(); i++)
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slicesqMRI.size())
                _slice_weightqMRI[_force_excluded[i]] = 0;

        //initialize sigma for voxel-wise robust statistics
        _sigma = sigma / num;

        for (int ii = 0; ii < _nEchoTimes; ii++)
            _sigma_qMRI[ii] = sigma_qMRI[ii]/num_qMRI[ii];

        //cout << _sigma_qMRI[1] << endl;

        //initialize sigma for slice-wise robust statistics
        _sigma_s = 0.025;
        for (int ii = 0; ii < _nEchoTimes; ii++)
            _sigma_s_qMRI[ii] = 0.025;

        //initialize mixing proportion for inlier class in voxel-wise robust statistics
        _mix = 0.9;
        for (int ii = 0; ii < _nEchoTimes; ii++)
            _mix_qMRI[ii] = 0.9;

        //initialize mixing proportion for outlier class in slice-wise robust statistics
        _mix_s = 0.9;
        for (int ii = 0; ii < _nEchoTimes; ii++)
            _mix_s_qMRI[ii] = 0.9;

        //Initialise value for uniform distribution according to the range of intensities
        _m = 1 / (2.1 * _max_intensity - 1.9 * _min_intensity);

        //Initialise value for uniform distribution according to the range of intensities for each channel
        _m_qMRI.reserve(_nEchoTimes);
        for (int ii = 0; ii < _nEchoTimes; ii++)
            _m_qMRI[ii] = 1 / (2.1 * _max_intensitiesqMRI[ii] - 1.9 * _min_intensitiesqMRI[ii]);

        //cout << _m_qMRI[1] << endl;


        if (_verbose) {
            _verbose_log << "Initializing robust statistics: sigma=" << sqrt(_sigma) << " m=" << _m << " mix=" << _mix
                         << " mix_s=" << _mix_s << endl;
            for (int ii = 0; ii < _nEchoTimes; ii++){
                _verbose_log << "Initializing robust statistics: sigma[" << ii << "]=" << sqrt(_sigma_qMRI[ii]) <<
                " m[" << ii << "]=" << _m_qMRI[ii] << " mix[" << ii << "]=" << _mix_qMRI[ii]
                             << " mix_s[" << ii << "]=" << _mix_s_qMRI[ii] << endl;
            }
        }
    }

    //-------------------------------------------------------------------

    // run EStep for calculation of voxel-wise and slice-wise posteriors (weights)
    void ReconstructionqMRI::EStepqMRI() {
        Array<double> slice_potential(_slicesqMRI.size());

        ParallelqMRI::EStepqMRI parallelEStep(this, slice_potential);
        parallelEStep();

        //To force-exclude slices predefined by a user, set their potentials to -1
        for (size_t i = 0; i < _force_excluded.size(); i++)
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slicesqMRI.size())
                slice_potential[_force_excluded[i]] = -1;

        //exclude slices identified as having small overlap with ROI, set their potentials to -1
        for (size_t i = 0; i < _small_slices.size(); i++)
            slice_potential[_small_slices[i]] = -1;

        //these are unrealistic scales pointing at misregistration - exclude the corresponding slices
        for (size_t inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
            if (_scaleqMRI[inputIndex] < 0.2 || _scaleqMRI[inputIndex] > 5)
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
        vector<double> sum_qMRI, den_qMRI, sum2_qMRI, den2_qMRI, maxs_qMRI, mins_qMRI;
        for (int ii = 0; ii < _nEchoTimes; ii++){
            sum_qMRI.push_back(0);
            den_qMRI.push_back(0);
            sum2_qMRI.push_back(0);
            den2_qMRI.push_back(0);
            maxs_qMRI.push_back(0);
            mins_qMRI.push_back(1);
        }
        //#pragma omp parallel for reduction(+: sum, sum2, den, den2) reduction(max: maxs) reduction(min: mins)
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            if (slice_potential[inputIndex] >= 0) {
                int indstack = _stack_index[inputIndex];
                int t = _volume_index[indstack];
                //calculate means
                sum += slice_potential[inputIndex] * _slice_weightqMRI[inputIndex];
                den += _slice_weightqMRI[inputIndex];
                sum2 += slice_potential[inputIndex] * (1 - _slice_weightqMRI[inputIndex]);
                den2 += 1 - _slice_weightqMRI[inputIndex];
                sum_qMRI[t] += slice_potential[inputIndex] * _slice_weightqMRI[inputIndex];
                den_qMRI[t] += _slice_weightqMRI[inputIndex];
                sum2_qMRI[t] += slice_potential[inputIndex] * (1 - _slice_weightqMRI[inputIndex]);
                den2_qMRI[t] += 1 - _slice_weightqMRI[inputIndex];

                //calculate min and max of potentials in case means need to be initalized
                if (slice_potential[inputIndex] > maxs)
                    maxs = slice_potential[inputIndex];
                if (slice_potential[inputIndex] < mins)
                    mins = slice_potential[inputIndex];
                if (slice_potential[inputIndex] > maxs_qMRI[t])
                    maxs_qMRI[t] = slice_potential[inputIndex];
                if (slice_potential[inputIndex] < mins_qMRI[t])
                    mins_qMRI[t] = slice_potential[inputIndex];
            }

        }

        _mean_s = den > 0 ? sum / den : mins;
        _mean_s2 = den2 > 0 ? sum2 / den2 : (maxs + _mean_s) / 2;
        for (int ii = 0; ii < _nEchoTimes; ii++){
            _mean_s_qMRI[ii] = den_qMRI[ii] > 0 ? sum_qMRI[ii] / den_qMRI[ii] : mins_qMRI[ii];
            _mean_s2_qMRI[ii] = den2_qMRI[ii] > 0 ? sum2_qMRI[ii] / den2_qMRI[ii] : (maxs_qMRI[ii] + _mean_s_qMRI[ii]) / 2;
        }


        //Calculate the variances of the potentials
        sum = 0; den = 0; sum2 = 0; den2 = 0;
        for (int ii = 0; ii < _nEchoTimes; ii++){
            sum_qMRI[ii] = 0;
            den_qMRI[ii] = 0;
            sum2_qMRI[ii] = 0;
            den2_qMRI[ii] = 0;
        }
        //#pragma omp parallel for reduction(+: sum, sum2, den, den2)
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            if (slice_potential[inputIndex] >= 0) {
                int indstack = _stack_index[inputIndex];
                int t = _volume_index[indstack];
                sum += (slice_potential[inputIndex] - _mean_s) * (slice_potential[inputIndex] - _mean_s) *
                       _slice_weightqMRI[inputIndex];
                den += _slice_weightqMRI[inputIndex];
                sum2 += (slice_potential[inputIndex] - _mean_s2) * (slice_potential[inputIndex] - _mean_s2) *
                        (1 - _slice_weightqMRI[inputIndex]);
                den2 += 1 - _slice_weightqMRI[inputIndex];
                sum_qMRI[t] += (slice_potential[inputIndex] - _mean_s_qMRI[t]) *
                               (slice_potential[inputIndex] - _mean_s_qMRI[t]) * _slice_weightqMRI[inputIndex];
                den_qMRI[t] += _slice_weightqMRI[inputIndex];
                sum2_qMRI[t] += (slice_potential[inputIndex] - _mean_s2_qMRI[t]) *
                                (slice_potential[inputIndex] - _mean_s2_qMRI[t]) * (1 - _slice_weightqMRI[inputIndex]);
                den2_qMRI[t] += 1 - _slice_weightqMRI[inputIndex];
            }
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

        for (int ii = 0; ii < _nEchoTimes; ii++){
            if (sum_qMRI[ii] > 0 && den_qMRI[ii] > 0) {
                //do not allow too small sigma
                _sigma_s_qMRI[ii] = max(sum_qMRI[ii] / den_qMRI[ii], _step * _step / 6.28);
            } else {
                _sigma_s_qMRI[ii] = 0.025;
                if (_verbose) {
                    double TEtoWrite = _echoTimes[ii];
                    if (sum_qMRI[ii] <= 0)
                        _verbose_log << (boost::format("All slices in TE = %1%ms are equal. ") % TEtoWrite).str().c_str();
                    if (den_qMRI[ii] < 0) //this should not happen
                        _verbose_log << (boost::format("All slices in TE = %1%ms are outliers. ") % TEtoWrite).str().c_str();
                    _verbose_log << "Setting sigma in TE = " << TEtoWrite << "ms  to " << sqrt(_sigma_s_qMRI[ii]) << endl;
                }
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

        for (int ii = 0; ii < _nEchoTimes; ii++){
            //sigma_s2
            if (sum2_qMRI[ii] > 0 && den2_qMRI[ii] > 0) {
                //do not allow too small sigma
                _sigma_s2_qMRI[ii] = max(sum2_qMRI[ii] / den2_qMRI[ii], _step * _step / 6.28);
            } else {
                //do not allow too small sigma
                _sigma_s2_qMRI[ii] = max((_mean_s2_qMRI[ii] - _mean_s_qMRI[ii]) * (_mean_s2_qMRI[ii] - _mean_s_qMRI[ii]) / 4, _step * _step / 6.28);

                if (_verbose) {
                    double TEtoWrite = _echoTimes[ii];
                    if (sum2_qMRI[ii] <= 0)
                        _verbose_log << "All slices in TE = " << TEtoWrite << "ms are equal. ";
                    if (den2_qMRI[ii] <= 0)
                        _verbose_log << "All slices in TE = " << TEtoWrite << "ms inliers. ";
                    _verbose_log << "Setting sigma_s2 in TE = " << TEtoWrite << "ms to " << sqrt(_sigma_s2_qMRI[ii]) << endl;
                }
            }
        }

        //Calculate slice weights
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            int indstack = _stack_index[inputIndex];
            int t = _volume_index[indstack];

            //Slice does not have any voxels in volumetric ROI
            if (slice_potential[inputIndex] == -1) {
                _slice_weightqMRI[inputIndex] = 0;
                continue;
            }

            //All slices are outliers or the means are not valid
            if (den <= 0 || _mean_s2_qMRI[t] <= _mean_s_qMRI[t]) {
                _slice_weightqMRI[inputIndex] = 1;
                continue;
            }

            //likelihood for inliers
            const double gs1 = slice_potential[inputIndex] < _mean_s2_qMRI[t] ? G(slice_potential[inputIndex] - _mean_s_qMRI[t], _sigma_s_qMRI[t]) : 0;

            //likelihood for outliers
            const double gs2 = slice_potential[inputIndex] > _mean_s_qMRI[t] ? G(slice_potential[inputIndex] - _mean_s2_qMRI[t], _sigma_s2_qMRI[t]) : 0;

            //calculate slice weight
            const double likelihood = gs1 * _mix_s_qMRI[t] + gs2 * (1 - _mix_s_qMRI[t]);
            if (likelihood > 0)
                _slice_weightqMRI[inputIndex] = gs1 * _mix_s_qMRI[t] / likelihood;
            else {
                if (slice_potential[inputIndex] <= _mean_s_qMRI[t])
                    _slice_weightqMRI[inputIndex] = 1;
                if (slice_potential[inputIndex] >= _mean_s2_qMRI[t])
                    _slice_weightqMRI[inputIndex] = 0;
                if (slice_potential[inputIndex] < _mean_s2_qMRI[t] && slice_potential[inputIndex] > _mean_s_qMRI[t]) //should not happen
                    _slice_weightqMRI[inputIndex] = 1;
            }
        }

        //Update _mix_s this should also be part of MStep
        int num = 0; sum = 0;
        vector<double> num_qMRI;
        for (int ii = 0; ii < _nEchoTimes; ii++) {
            sum_qMRI[ii] = 0;
            num_qMRI.push_back(0);
        }
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++)
            if (slice_potential[inputIndex] >= 0) {
                int indstack = _stack_index[inputIndex];
                int t = _volume_index[indstack];
                sum += _slice_weightqMRI[inputIndex];
                num++;
                sum_qMRI[t] += _slice_weightqMRI[inputIndex];
                num_qMRI[t]++;
            }

        if (num > 0)
            _mix_s = sum / num;
        else {
            cout << "All slices are outliers. Setting _mix_s to 0.9." << endl;
            _mix_s = 0.9;
        }
        for (int ii = 0; ii < _nEchoTimes; ii++) {
            if (num > 0)
                _mix_s_qMRI[ii] = sum_qMRI[ii] / num_qMRI[ii];
            else {
                double TEtoWrite = _echoTimes[ii];
                cout << "All slices are outliers. Setting _mix_s in TE = " << TEtoWrite << "ms to 0.9." << endl;
                _mix_s_qMRI[ii] = 0.9;
            }
        }

        if (_verbose) {
            _verbose_log << setprecision(3);
            _verbose_log << "Slice robust statistics parameters: ";
            _verbose_log << "means: " << _mean_s << " " << _mean_s2 << "  ";
            _verbose_log << "sigmas: " << sqrt(_sigma_s) << " " << sqrt(_sigma_s2) << "  ";
            _verbose_log << "proportions: " << _mix_s << " " << 1 - _mix_s << endl;
            _verbose_log << "Slice weights:";
            for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++)
                _verbose_log << " " << _slice_weightqMRI[inputIndex];
            _verbose_log << endl;
            for (int ii = 0; ii < _nEchoTimes; ii++){
                _verbose_log << "means[" << ii << "]: " << _mean_s_qMRI[ii] << " " << _mean_s2_qMRI[ii] << "  ";
                _verbose_log << "sigmas[" << ii << "]: " << sqrt(_sigma_s_qMRI[ii]) << " " << sqrt(_sigma_s2_qMRI[ii]) << "  ";
                _verbose_log << "proportions[" << ii << "]: " << _mix_s_qMRI[ii] << " " << 1 - _mix_s_qMRI[ii] << endl;
            }

        }
    }

    //-------------------------------------------------------------------

    // run MStep (RS)
    void ReconstructionqMRI::MStepqMRI(int iter) {
        ParallelqMRI::MStepqMRI parallelMStep(this);
        parallelMStep();
        const double sigma = parallelMStep.sigma;
        const double mix = parallelMStep.mix;
        const double num = parallelMStep.num;
        const double min = parallelMStep.min;
        const double max = parallelMStep.max;
        vector<double> sigmaqMRI;
        vector<double> mixqMRI;
        vector<double> numqMRI;
        vector<double> minqMRI;
        vector<double> maxqMRI;
        ClearAndReserve(sigmaqMRI,_nEchoTimes);
        ClearAndReserve(mixqMRI,_nEchoTimes);
        ClearAndReserve(numqMRI,_nEchoTimes);
        ClearAndReserve(maxqMRI,_nEchoTimes);
        ClearAndReserve(minqMRI,_nEchoTimes);
        for (int i = 0; i < _nEchoTimes; i++){
            sigmaqMRI[i] = parallelMStep.sigmaqMRI[i];
            mixqMRI[i] = parallelMStep.mixqMRI[i];
            numqMRI[i] = parallelMStep.numqMRI[i];
            minqMRI[i] = parallelMStep.minqMRI[i];
            maxqMRI[i] = parallelMStep.maxqMRI[i];

            //Calculate sigma and mix
            if (mixqMRI[i] > 0) {
                _sigma_qMRI[i] = sigmaqMRI[i] / mixqMRI[i];
            } else {
                throw runtime_error("Something went wrong: sigma=" + to_string(sigmaqMRI[i]) + " mix=" + to_string(mixqMRI[i]));
            }
            if (_sigma_qMRI[i] < _step * _step / 6.28)
                _sigma_qMRI[i]= _step * _step / 6.28;
            if (iter > 1)
                _mix_qMRI[i] = mixqMRI[i] / numqMRI[i];
        }

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

        for (int ii = 0; ii < _nEchoTimes; ii++)
            _m_qMRI[ii] = 1 / (maxqMRI[ii] - minqMRI[ii]);

        if (_verbose)
            _verbose_log << "Voxel-wise robust statistics parameters: sigma=" << sqrt(_sigma) << " mix=" << _mix << " m=" << _m << endl;
        for (int i = 0; i < _nEchoTimes; i++){
            if (_verbose)
                _verbose_log << "Voxel-wise robust statistics parameters: sigma=" << sqrt(_sigma_qMRI[i]) << " mix=" << _mix_qMRI[i] << " m=" << _m_qMRI[i] << endl;
        }
    }

    //-------------------------------------------------------------------

    void ReconstructionqMRI::SaveSlicesqMRI() {
        #pragma omp parallel for
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++)
            _slicesqMRI[inputIndex].Write((boost::format("sliceqMRI%1%.nii.gz") % inputIndex).str().c_str());
    }

    //-------------------------------------------------------------------

    // evaluate reconstruction quality (NRMSE)
    double ReconstructionqMRI::EvaluateReconQualityqMRI(int stackIndex) {
        Array<double> rmse_values(_slicesqMRI.size());
        Array<int> rmse_numbers(_slicesqMRI.size());
        RealImage nt;

        // compute NRMSE between the simulated and original slice for non-zero voxels
        #pragma omp parallel for private(nt)
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            nt = _slicesqMRI[inputIndex];
            const RealImage& ns = _simulated_slicesqMRI[inputIndex];
            double s_diff = 0;
            double s_t = 0;
            int s_n = 0;
            for (int x = 0; x < nt.GetX(); x++) {
                for (int y = 0; y < nt.GetY(); y++) {
                    if (nt(x, y, 0) > 0 && ns(x, y, 0) > 0) {
                        nt(x, y, 0) *= exp(-(_biasqMRI[inputIndex])(x, y, 0)) * _scaleqMRI[inputIndex];
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
        for (size_t inputIndex = 0; inputIndex < _slicesqMRI.size(); inputIndex++) {
            rmse_total += rmse_values[inputIndex];
            slice_n += rmse_numbers[inputIndex];
        }

        if (slice_n > 0)
            rmse_total /= slice_n;
        else
            rmse_total = 0;

        return rmse_total;
    }


} // namespace svrtk
