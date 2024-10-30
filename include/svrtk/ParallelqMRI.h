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
#include "svrtk/ReconstructionqMRI.h"

using namespace mirtk;
using namespace svrtk;
using namespace svrtk::Utility;

namespace svrtk::ParallelqMRI {

    /// Class for simulation of slices from the current reconstructed 4D volume - qMRI
    class SimulateSlicesqMRI {
        ReconstructionqMRI *reconstructor;

    public:
        SimulateSlicesqMRI(SimulateSlicesqMRI& x, split) : SimulateSlicesqMRI(x.reconstructor) {}

        SimulateSlicesqMRI(ReconstructionqMRI *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                //Calculate simulated slice
                RealImage& sim_slice = reconstructor->_simulated_slicesqMRI[inputIndex];
                RealImage& sim_weight = reconstructor->_simulated_weightsqMRI[inputIndex];
                RealImage& sim_inside = reconstructor->_simulated_insideqMRI[inputIndex];

                memset(sim_slice.Data(), 0, sizeof(RealPixel) * sim_slice.NumberOfVoxels());
                memset(sim_weight.Data(), 0, sizeof(RealPixel) * sim_weight.NumberOfVoxels());
                memset(sim_inside.Data(), 0, sizeof(RealPixel) * sim_inside.NumberOfVoxels());
                reconstructor->_slice_inside[inputIndex] = false;

                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (reconstructor->_slicesqMRI[inputIndex](i, j, 0) > -0.01) {
                            double weight = 0;
                            for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                int indstack =  reconstructor->_stack_index[inputIndex];
                                int outputIndex =  reconstructor->_volume_index[indstack];
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                sim_slice(i, j, 0) += p.value * reconstructor->_reconstructed4D(p.x, p.y, p.z,outputIndex);
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

        void join(const SimulateSlicesqMRI& y) {}

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for EStep (RS)
    class EStepqMRI {
        ReconstructionqMRI *reconstructor;
        Array<double>& slice_potential;

    public:
        EStepqMRI(ReconstructionqMRI *reconstructor,
              Array<double>& slice_potential) :
                reconstructor(reconstructor),
                slice_potential(slice_potential) {}

        void operator()(const blocked_range<size_t>& r) const {
            RealImage slice;
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                slice = reconstructor->_slicesqMRI[inputIndex];

                int indstack = reconstructor->_stack_index[inputIndex];
                int outputIndex = reconstructor->_volume_index[indstack];

                //read current weight image
                RealImage& weight = reconstructor->_weightsqMRI[inputIndex];
                memset(weight.Data(), 0, sizeof(RealPixel) * weight.NumberOfVoxels());

                double num = 0;
                //Calculate error, voxel weights, and slice potential
                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-reconstructor->_biasqMRI[inputIndex](i, j, 0)) * reconstructor->_scaleqMRI[inputIndex];

                            //number of volumetric voxels to which
                            // current slice voxel contributes
                            const size_t n = reconstructor->_volcoeffs[inputIndex][i][j].size();

                            // if n == 0, slice voxel has no overlap with volumetric ROI, do not process it

                            if (n > 0 && reconstructor->_simulated_weightsqMRI[inputIndex](i, j, 0) > 0) {
                                slice(i, j, 0) -= reconstructor->_simulated_slicesqMRI[inputIndex](i, j, 0);

                                //calculate norm and voxel-wise weights

                                //Gaussian distribution for inliers (likelihood)
                                const double g = reconstructor->G(slice(i, j, 0), reconstructor->_sigma_qMRI[outputIndex]);

                                //Uniform distribution for outliers (likelihood)
                                const double m = reconstructor->M(reconstructor->_m_qMRI[outputIndex]);

                                //voxel_wise posterior
                                const double w = g * reconstructor->_mix_qMRI[outputIndex] / (g * reconstructor->_mix_qMRI[outputIndex] + m * (1 - reconstructor->_mix_qMRI[outputIndex]));
                                weight.PutAsDouble(i, j, 0, w);

                                //calculate slice potentials
                                if (reconstructor->_simulated_weightsqMRI[inputIndex](i, j, 0) > 0.99) {
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

    /// Class for slice scale calculation
    class ScaleqMRI {
        ReconstructionqMRI *reconstructor;

    public:
        ScaleqMRI(ReconstructionqMRI *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            for (size_t inputIndex = r.begin(); inputIndex != r.end(); inputIndex++) {
                //alias the current bias image
                const RealImage& b = reconstructor->_biasqMRI[inputIndex];

                //initialise calculation of scale
                double scalenum = 0;
                double scaleden = 0;

                for (int i = 0; i < reconstructor->_slicesqMRI[inputIndex].GetX(); i++)
                    for (int j = 0; j < reconstructor->_slicesqMRI[inputIndex].GetY(); j++)
                        if (reconstructor->_slicesqMRI[inputIndex](i, j, 0) > -0.01) {
                            if (reconstructor->_simulated_weightsqMRI[inputIndex](i, j, 0) > 0.99) {
                                //scale - intensity matching
                                const double eb = exp(-b(i, j, 0));
                                scalenum += reconstructor->_weightsqMRI[inputIndex](i, j, 0) * reconstructor->_slicesqMRI[inputIndex](i, j, 0) * eb * reconstructor->_simulated_slicesqMRI[inputIndex](i, j, 0);
                                scaleden += reconstructor->_weightsqMRI[inputIndex](i, j, 0) * reconstructor->_slicesqMRI[inputIndex](i, j, 0) * eb * reconstructor->_slicesqMRI[inputIndex](i, j, 0) * eb;
                            }
                        }

                //calculate scale for this slice
                reconstructor->_scaleqMRI[inputIndex] = scaleden > 0 ? scalenum / scaleden : 1;
            } //end of loop for a slice inputIndex
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slicesqMRI.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for slice bias calculation
    class BiasqMRI {
        ReconstructionqMRI *reconstructor;

    public:
        BiasqMRI(ReconstructionqMRI *reconstructor) : reconstructor(reconstructor) {}

        void operator()(const blocked_range<size_t>& r) const {
            RealImage slice, wb, wresidual, checkerslice, wbchecker, wresidualchecker;
            GaussianBlurring<RealPixel> gb(reconstructor->_sigma_biasqMRI);

            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                slice = reconstructor->_slicesqMRI[inputIndex];
                checkerslice.Initialize(slice.Attributes());

                //alias the current bias image
                RealImage& b = reconstructor->_biasqMRI[inputIndex];
                RealImage bchecker;
                bchecker.Initialize(slice.Attributes());
                //b.Write((boost::format("currentbias%1%.nii.gz") % inputIndex).str().c_str());

                //prepare weight image for bias field
                wb = reconstructor->_weightsqMRI[inputIndex];
                wbchecker.Initialize(slice.Attributes());

                wresidual.Initialize(slice.Attributes());
                wresidualchecker.Initialize(slice.Attributes());

                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            if (reconstructor->_simulated_weightsqMRI[inputIndex](i, j, 0) > 0.99) {
                                //bias-correct and scale current slice
                                const double eb = exp(-b(i, j, 0));
                                slice(i, j, 0) *= eb * reconstructor->_scaleqMRI[inputIndex];
                                checkerslice(i, j, 0) = 2.0 * eb * reconstructor->_scaleqMRI[inputIndex];

                                //calculate weight image
                                wb(i, j, 0) *= slice(i, j, 0);
                                wbchecker(i, j, 0) = checkerslice(i, j, 0)/4;

                                //calculate weighted residual image make sure it is far from zero to avoid numerical instability
                                if ((reconstructor->_simulated_slicesqMRI[inputIndex](i, j, 0) > 1) && (slice(i, j, 0) > 1)){
                                    wresidual(i, j, 0) = log(slice(i, j, 0) / reconstructor->_simulated_slicesqMRI[inputIndex](i, j, 0)) * wb(i, j, 0);
                                }
                                if (checkerslice(i, j, 0) > 1){
                                    wresidualchecker(i, j, 0) = log(checkerslice(i, j, 0) / checkerslice(i, j, 0)) * wbchecker(i, j, 0);
                                }
                            } else {
                                //do not take into account this voxel when calculating bias field
                                wresidual(i, j, 0) = 0;
                                wb(i, j, 0) = 0;
                                wresidualchecker(i, j, 0) = 0;
                                wbchecker(i, j, 0) = 0;
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

                //calculate bias field for this slice
                //smooth weighted residual
                gb.Input(&wresidualchecker);
                gb.Output(&wresidualchecker);
                gb.Run();

                //smooth weight image
                gb.Input(&wbchecker);
                gb.Output(&wbchecker);
                gb.Run();

                //update bias field
                double sum = 0;
                double sumchecker = 0;
                double num = 0;
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            if (wb(i, j, 0) > 0) {
                                b(i, j, 0) += wresidual(i, j, 0) / wb(i, j, 0);
                                bchecker(i, j, 0) += wresidualchecker(i, j, 0) / wbchecker(i, j, 0);
                            }
                            sum += b(i, j, 0);
                            sumchecker += bchecker(i, j, 0);
                            num++;
                        }

                //normalize bias field to have zero mean
                if (!reconstructor->_global_bias_correction && num > 0) {
                    const double mean = sum / num;
                    const double meanchecker = sumchecker / num;
                    for (int i = 0; i < slice.GetX(); i++)
                        for (int j = 0; j < slice.GetY(); j++)
                            if (slice(i, j, 0) > -0.01){
                                b(i, j, 0) -= mean;
                                bchecker(i, j, 0) -= mean;
                            }
                }
                /*if (reconstructor->_debug) {
                    bchecker.Write((boost::format("checkerbias%1%.nii.gz") % inputIndex).str().c_str());
                    b.Write((boost::format("comparitivebias%1%.nii.gz") % inputIndex).str().c_str());
                    wbchecker.Write((boost::format("wbchecker%1%.nii.gz") % inputIndex).str().c_str());
                    wresidualchecker.Write((boost::format("wresidualchecker%1%.nii.gz") % inputIndex).str().c_str());
                    checkerslice.Write((boost::format("checkerslice%1%.nii.gz") % inputIndex).str().c_str());
                }*/
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for SR qMRI reconstruction
    class SuperresolutionqMRI {
        ReconstructionqMRI *reconstructor;

    public:
        RealImage confidence_map;
        RealImage addon;

        SuperresolutionqMRI(ReconstructionqMRI *reconstructor) : reconstructor(reconstructor) {
            //Clear addon
            addon.Initialize(reconstructor->_reconstructed4D.Attributes());

            //Clear confidence map
            confidence_map.Initialize(reconstructor->_reconstructed4D.Attributes());

        }

        SuperresolutionqMRI(SuperresolutionqMRI& x, split) : SuperresolutionqMRI(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            //Update reconstructed volume using current slice
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                //Distribute error to the volume
                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (reconstructor->_slicesqMRI[inputIndex](i, j, 0) > -0.01) {
                            if (reconstructor->_simulated_slicesqMRI[inputIndex](i, j, 0) < 0.01)
                                reconstructor->_slice_difqMRI[inputIndex](i, j, 0) = 0;
                            #pragma omp simd
                            for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                const auto multiplier = reconstructor->_robust_slices_only ? 1 : reconstructor->_weightsqMRI[inputIndex](i, j, 0);
                                double ssim_weight = 1;

                                int indstack = reconstructor->_stack_index[inputIndex];
                                int outputIndex = reconstructor->_volume_index[indstack];

                                addon(p.x, p.y, p.z,outputIndex) += ssim_weight * multiplier * p.value * reconstructor->_slice_weightqMRI[inputIndex] * reconstructor->_slice_difqMRI[inputIndex](i, j, 0);
                                confidence_map(p.x, p.y, p.z,outputIndex) += ssim_weight * multiplier * p.value * reconstructor->_slice_weightqMRI[inputIndex];
                            }
                        }
            } //end of loop for a slice inputIndex
        }

        void join(const SuperresolutionqMRI& y) {
            addon += y.addon;
            confidence_map += y.confidence_map;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slicesqMRI.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for SR qMRI reconstruction
    class ModelFitqMRI {
        ReconstructionqMRI *reconstructor;

    public:
        RealImage T2Map;
        RealImage addon2;
        vector<double> scales;
        RealImage CopyRecon4D;
        //Find the range of intensities
        double maxVal = voxel_limits<RealPixel>::min();
        double minVal = voxel_limits<RealPixel>::max();
        double origRange;
        double standRange;
        double ratioInts;

        ModelFitqMRI(ReconstructionqMRI *reconstructor) : reconstructor(reconstructor) {
            //Clear addon
            addon2.Initialize(reconstructor->_reconstructed4D.Attributes());

            // Clear T2 map
            T2Map.Initialize(reconstructor->_reconstructed.Attributes());

            CopyRecon4D = reconstructor->_reconstructed4D;

            if (reconstructor->IsIntensityMatching()) {
                scales = reconstructor->ScaleVolumeqMRI(CopyRecon4D, true);
            }
            else {
                scales = {1,1,1};
            }

            const RealPixel *ptr = CopyRecon4D.Data();
            for (int i = 0; i < CopyRecon4D.NumberOfVoxels(); i++){
                if (ptr[i] > maxVal){
                    maxVal = ptr[i];
                }
                if (ptr[i] < minVal) {
                    minVal = ptr[i];
                }
            }

            origRange = maxVal-minVal;
            standRange = reconstructor->_max_Standard_intensity-reconstructor->_min_Standard_intensity;

            ratioInts = standRange/origRange;
        }

        ModelFitqMRI(ModelFitqMRI& x, split) : ModelFitqMRI(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {

            Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

            Eigen::MatrixXd  DictionaryMatrix = reconstructor->_GivenDictionary.GetDictionaryMatrix();

            // Normalise Dictionary
            Eigen::MatrixXd NormDictMat = DictionaryMatrix.rowwise().normalized();

            int sh = 0;

            Eigen::VectorXd T2Vals = reconstructor->_GivenDictionary.GetT2Vals();

            bool UseFD = false;
            double StepSize = 0;


            if (reconstructor->UseFiniteDifference()) {
                StepSize = reconstructor->GetFiniteDiffStep();
                UseFD = true;
            }


            // Loop through and assign T2 value to each voxel
            for (size_t x = r.begin(); x < r.end(); x++) {

                for (size_t y = sh; y < reconstructor->_reconstructed4D.GetY() - sh; y++) {
                    #pragma omp simd
                    for (size_t z = sh; z < reconstructor->_reconstructed4D.GetZ() - sh; z++) {
                        // Initialise Sample array for voxel
                        Eigen::VectorXd SampleIntensities(reconstructor->_nEchoTimes);
                        Eigen::VectorXd NormSampleIntensities(reconstructor->_nEchoTimes);

                        Eigen::VectorXd CheckerInts(reconstructor->_nEchoTimes);

                        // Fill sample array
                        for (int t = 0; t < reconstructor->_nEchoTimes; t++) {
                            CheckerInts[t] = CopyRecon4D(x,y,z,t);
                            SampleIntensities[t] = (CopyRecon4D(x,y,z,t) - minVal) * ratioInts ;
                            NormSampleIntensities[t] = (CopyRecon4D(x,y,z,t) - minVal) * ratioInts;
                        }

                        /*if (x == 51 && y ==49 && z ==57 && reconstructor->_debug)
                            cout << setprecision(7) << "Original Sample Intensities: " << CheckerInts.format(CommaInitFmt) << endl;*/

                        if (SampleIntensities.sum() <= 0.001*reconstructor->_stack_averages[0]) {
                            for (int ii = 0; ii < reconstructor->_nEchoTimes; ii++) {
                                addon2(x, y, z, ii) = 0;
                                T2Map(x, y, z) = 0;
                            }
                        } else {

                            for (int t = 0; t < reconstructor->_nEchoTimes; t++) {
                                //cout << SampleIntensities[t] << " " << CopyRecon4D(x, y, z, t) << " ";
                            }

                            // Normalise sample array:
                            NormSampleIntensities.normalize();

                            // Perform scalar product and obtain T2 for max value
                            Eigen::VectorXd NormScalarProducts;
                            NormScalarProducts = NormDictMat * NormSampleIntensities;
                            Eigen::Index indx;
                            NormScalarProducts.maxCoeff(&indx);
                            Eigen::VectorXd DictRow = DictionaryMatrix.row(indx);

                            // Find scale between dictionary entry and intensities using the minimum value of mean
                            // squared error
                            double Scale;
                            Scale = DictRow.dot((SampleIntensities)) / DictRow.dot(DictRow);

                            for (int t = 0; t < reconstructor->_nEchoTimes; t++) {
                                if (UseFD) {
                                    Eigen::VectorXd FD_SampleIntensities1(reconstructor->_nEchoTimes);
                                    FD_SampleIntensities1 = SampleIntensities;
                                    FD_SampleIntensities1[t] = FD_SampleIntensities1[t] + (StepSize);
                                    FD_SampleIntensities1.normalize();
                                    Eigen::VectorXd FD_NormScalarProducts1;
                                    FD_NormScalarProducts1 = NormDictMat * FD_SampleIntensities1;
                                    Eigen::Index FD_indx1;
                                    FD_NormScalarProducts1.maxCoeff(&FD_indx1);
                                    Eigen::VectorXd FD_DictRow1 = DictionaryMatrix.row(FD_indx1);

                                    double SampNorm = SampleIntensities.norm();
                                    double DictRowNorm = DictRow.norm();
                                    double FD_DictRow1Norm = FD_DictRow1.norm();
                                    double Term1 = (SampleIntensities[t]/SampNorm - DictRow[t]/DictRowNorm);
                                    int sqrind = 0;
                                    double denom1 = 0;
                                    double subsampsSqr = 0;
                                    for (int sind = 0; sind < reconstructor->_nEchoTimes; sind++){
                                        denom1 += SampleIntensities[sind]*SampleIntensities[sind];
                                        if (sind != t)
                                            subsampsSqr += SampleIntensities[sind]*SampleIntensities[sind];
                                    }

                                    double DiffTerm1 = subsampsSqr/(pow(denom1,1.5));
                                    double DiffTerm2 = (DictRow[t]/DictRowNorm - FD_DictRow1[t]/FD_DictRow1Norm)/StepSize;
                                    double Term2 = DiffTerm1 - DiffTerm2;
                                    addon2(x, y, z, t) += Term1*Term2/(ratioInts*scales[t]);
                                    /*if (x == 51 && y ==49 && z ==57 && reconstructor->_debug) {
                                        cout << setprecision(7) << "Sample Intensities: " << SampleIntensities.format(CommaInitFmt) << endl;
                                        cout << "scales: " << scales[0] << " " << scales[1] << " " << scales[2] << endl;
                                        cout << "SampNorm: " << SampNorm << endl;
                                        cout << "Addon: " << addon2(x, y, z, t) << endl;
                                        cout << "DiffTerm1: " << DiffTerm1 << endl;
                                        cout << "DiffTerm2: " << DiffTerm2 << endl;
                                        cout << "reconstructor->_max_Standard_intensity: " << reconstructor->_max_Standard_intensity << endl;
                                        cout << "Term1: " << Term1 << endl;
                                        cout << "Term2: " << Term2 << endl;
                                        cout << "x, y, z: " << x << ", " << y << ", " << z << ", Addon: " << Term1 * Term2 << endl;
                                        cout << "maxVal: " << maxVal << ", minVal: " << minVal << endl;
                                    }*/
                                }
                                else
                                {
                                    addon2(x, y, z, t) += (SampleIntensities(t) - Scale * DictRow(t));// / scales[t];
                                }
                            }
                            T2Map(x,y,z) += T2Vals[indx];
                        }
                    }
                }
            } //end of loop for a slice inputIndex
        }

        void join(const ModelFitqMRI& y) {
            addon2 += y.addon2;
            T2Map += y.T2Map;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_reconstructed4D.GetX()), *this);
        }
    };


    //-------------------------------------------------------------------


    /// Class for Dictionary fit
    class DictionaryFit {
        Dictionary *dictionary;

    public:
        RealImage T2Map;

        DictionaryFit(Dictionary *dictionary) : dictionary(dictionary) {

            // Clear T2 map
            T2Map.Initialize(dictionary->_images[1].Attributes());
        }

        DictionaryFit(DictionaryFit& x, split) : DictionaryFit(x.dictionary) {}

        void operator()(const blocked_range<size_t>& r) {

            int sh = 1;


            // Number of input images
            int nImages = dictionary->_images.size();

            // Normalise Dictionary
            Eigen::MatrixXd NormDictMat = dictionary->_DictionaryMatrix.rowwise().normalized();

            // Loop through and assign T2 value to each voxel
            for (size_t x = r.begin(); x < r.end(); x++) {
                for (int y = sh; y < dictionary->_images[0].GetY()-sh; y++) {
                    for (int z = sh; z < dictionary->_images[0].GetZ()-sh; z++) {

                        // Initialise Sample array for voxel
                        Eigen::VectorXd SampleIntensities(nImages);

                        // Fill sample array
                        for (int ii = 0; ii < nImages; ii++){
                            SampleIntensities[ii] = dictionary->_images[ii](x,y,z);
                        }

                        //T2Map(x,y,z) += SampleIntensities[0];
                        if (SampleIntensities.sum() <= 0){
                            T2Map(x,y,z) = 0;
                        }
                        else {
                            SampleIntensities.normalize();
                            Eigen::VectorXd T2Vals = dictionary->GetT2Vals();

                            // Perform scalar product and obtain T2 for max value
                            Eigen::VectorXd NormScalarProducts;
                            NormScalarProducts = NormDictMat * SampleIntensities;
                            Eigen::Index indx;
                            NormScalarProducts.maxCoeff(&indx);
                            T2Map(x,y,z) +=  T2Vals[indx];
                        }
                    }
                }
            }
        }

        void join(const DictionaryFit& y) {
            T2Map += y.T2Map;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(1, dictionary->_images[0].GetX()), *this);
        }
    };


    //-------------------------------------------------------------------

    /// Class for MStep (RS)
    class MStepqMRI {
        ReconstructionqMRI *reconstructor;

    public:
        double sigma;
        double mix;
        double num;
        double min;
        double max;
        vector<double> sigmaqMRI;
        vector<double> mixqMRI;
        vector<double> numqMRI;
        vector<double> minqMRI;
        vector<double> maxqMRI;

        MStepqMRI(ReconstructionqMRI *reconstructor) : reconstructor(reconstructor) {
            sigma = 0;
            mix = 0;
            num = 0;
            min = voxel_limits<RealPixel>::max();
            max = voxel_limits<RealPixel>::min();
            for (size_t i = 0; i < reconstructor->_nEchoTimes; i++) {
                sigmaqMRI.push_back(0);
                mixqMRI.push_back(0);
                numqMRI.push_back(0);
                minqMRI.push_back(voxel_limits<RealPixel>::max());
                maxqMRI.push_back(voxel_limits<RealPixel>::min());
            }
        }

        MStepqMRI(MStepqMRI& x, split) : MStepqMRI(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            RealImage slice;

            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                // read the current slice
                slice = reconstructor->_slicesqMRI[inputIndex];
                int indstack = reconstructor->_stack_index[inputIndex];
                int outputIndex = reconstructor->_volume_index[indstack];
                //calculate error
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-reconstructor->_biasqMRI[inputIndex](i, j, 0)) * reconstructor->_scaleqMRI[inputIndex];

                            //otherwise the error has no meaning - it is equal to slice intensity
                            if (reconstructor->_simulated_weightsqMRI[inputIndex](i, j, 0) > 0.99) {
                                slice(i, j, 0) -= reconstructor->_simulated_slicesqMRI[inputIndex](i, j, 0);

                                //sigma and mix
                                const double e = slice(i, j, 0);
                                sigma += e * e * reconstructor->_weightsqMRI[inputIndex](i, j, 0);
                                mix += reconstructor->_weightsqMRI[inputIndex](i, j, 0);
                                sigmaqMRI[outputIndex] += e * e * reconstructor->_weightsqMRI[inputIndex](i, j, 0);
                                mixqMRI[outputIndex] += reconstructor->_weightsqMRI[inputIndex](i, j, 0);

                                //_m
                                if (e < min)
                                    min = e;
                                if (e > max)
                                    max = e;

                                if (e < minqMRI[outputIndex])
                                    minqMRI[outputIndex] = e;
                                if (e > maxqMRI[outputIndex])
                                    maxqMRI[outputIndex] = e;


                                num++;
                                numqMRI[outputIndex]++;
                            }
                        }
            } //end of loop for a slice inputIndex
        }

        void join(const MStepqMRI& y) {
            if (y.min < min)
                min = y.min;
            if (y.max > max)
                max = y.max;
            sigma += y.sigma;
            mix += y.mix;
            num += y.num;
            for (int ii = 0; ii < reconstructor->_nEchoTimes; ii++) {
                if (y.minqMRI[ii] < minqMRI[ii])
                    minqMRI[ii] = y.minqMRI[ii];
                if (y.maxqMRI[ii] > maxqMRI[ii])
                    maxqMRI[ii] = y.maxqMRI[ii];
                sigmaqMRI[ii] += y.sigmaqMRI[ii];
                mixqMRI[ii] += y.mixqMRI[ii];
                numqMRI[ii] += y.numqMRI[ii];
            }

        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for adaptive regularisation (Part I)
    class AdaptiveRegularization1qMRI {
        const ReconstructionqMRI *reconstructor;
        Array<RealImage>& b;
        const Array<double>& factor;
        const RealImage& original;

    public:
        AdaptiveRegularization1qMRI(
                const ReconstructionqMRI *reconstructor,
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
                for (int t = 0; t < dt; t++) {
                    for (int z = 0; z < dz; z++) {
                        for (int y = 0; y < dy; y++) {
                            for (int x = 0; x < dx; x++) {

                                const int xx = x + reconstructor->_directions[i][0];
                                const int yy = y + reconstructor->_directions[i][1];
                                const int zz = z + reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                    && (reconstructor->_confidence_map4D(x, y, z, t) > 0) &&
                                    (reconstructor->_confidence_map4D(xx, yy, zz, t) > 0)) {
                                    const double diff =
                                            (original(xx, yy, zz, t) - original(x, y, z, t)) * sqrt(factor[i]) /
                                            reconstructor->_delta;
                                    b[i](x, y, z, t) = factor[i] / sqrt(1 + diff * diff);
                                } else
                                    b[i](x, y, z, t) = 0;
                            }
                        }
                    }
                }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, 13), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for adaptive regularisation (Part II)
    class AdaptiveRegularization2qMRI {
        ReconstructionqMRI *reconstructor;
        const Array<RealImage>& b;
        const RealImage& original;

    public:
        AdaptiveRegularization2qMRI(
                ReconstructionqMRI *reconstructor,
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
            for (size_t z = r.begin(); z != r.end(); ++z) {
                for (int t = 0; t < dt; t++) {
                    for (int y = 0; y < dy; y++) {
                        for (int x = 0; x < dx; x++) {
                            if (reconstructor->_confidence_map4D(x, y, z, t) > 0) {
                                double val = 0;
                                double sum = 0;

                                // Don't combine the loops, it breaks the compiler optimisation (GCC 9)
                                for (int i = 0; i < 13; i++) {
                                    const int xx = x + reconstructor->_directions[i][0];
                                    const int yy = y + reconstructor->_directions[i][1];
                                    const int zz = z + reconstructor->_directions[i][2];
                                    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                        && reconstructor->_confidence_map4D(xx, yy, zz, t) > 0) {
                                        val += b[i](x, y, z, t) * original(xx, yy, zz, t);
                                        sum += b[i](x, y, z, t);
                                    }
                                }

                                for (int i = 0; i < 13; i++) {
                                    const int xx = x - reconstructor->_directions[i][0];
                                    const int yy = y - reconstructor->_directions[i][1];
                                    const int zz = z - reconstructor->_directions[i][2];
                                    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                        && reconstructor->_confidence_map4D(xx, yy, zz, t) > 0) {
                                        val += b[i](x, y, z, t) * original(xx, yy, zz, t);
                                        sum += b[i](x, y, z, t);
                                    }
                                }

                                val -= sum * original(x, y, z, t);
                                val = original(x, y, z, t) + reconstructor->_alpha * reconstructor->_lambda /
                                                             (reconstructor->_delta * reconstructor->_delta) * val;
                                reconstructor->_reconstructed4D(x, y, z, t) = val;
                            }
                        }
                    }
                }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_reconstructed4D.GetZ()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for adaptive regularisation (Part I)
    class AdaptiveRegularization1T2Map {
        const ReconstructionqMRI *reconstructor;
        Array<RealImage>& b;
        const Array<double>& factor;
        const RealImage& original;

    public:
        AdaptiveRegularization1T2Map(
                const ReconstructionqMRI *reconstructor,
                Array<RealImage>& b,
                const Array<double>& factor,
                const RealImage& original) :
                reconstructor(reconstructor),
                b(b),
                factor(factor),
                original(original) {}

        void operator()(const blocked_range<size_t>& r) const {
            const int dx = reconstructor->_T2Map.GetX();
            const int dy = reconstructor->_T2Map.GetY();
            const int dz = reconstructor->_T2Map.GetZ();
            for (size_t i = r.begin(); i != r.end(); i++) {
                for (int z = 0; z < dz; z++)
                    for (int y = 0; y < dy; y++)
                        for (int x = 0; x < dx; x++) {
                                const int xx = x + reconstructor->_directions[i][0];
                                const int yy = y + reconstructor->_directions[i][1];
                                const int zz = z + reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                    && (reconstructor->_confidence_map4D(x, y, z, 0) > 0) &&
                                    (reconstructor->_confidence_map4D(xx, yy, zz, 0) > 0)) {
                                    const double diff = (original(xx, yy, zz) - original(x, y, z)) * sqrt(factor[i]) /
                                                        reconstructor->_delta;
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

    /// Class for adaptive regularisation (Part II)
    class AdaptiveRegularization2T2Map {
        ReconstructionqMRI *reconstructor;
        const Array<RealImage>& b;
        const RealImage& original;

    public:
        AdaptiveRegularization2T2Map(
                ReconstructionqMRI *reconstructor,
                const Array<RealImage>& b,
                const RealImage& original) :
                reconstructor(reconstructor),
                b(b),
                original(original) {}

        void operator()(const blocked_range<size_t>& r) const {
            const int dx = reconstructor->_T2Map.GetX();
            const int dy = reconstructor->_T2Map.GetY();
            const int dz = reconstructor->_T2Map.GetZ();
            for (size_t z = r.begin(); z != r.end(); ++z) {
                for (int y = 0; y < dy; y++) {
                    for (int x = 0; x < dx; x++) {
                        if (reconstructor->_confidence_map4D(x, y, z, 0) > 0) {
                            double val = 0;
                            double sum = 0;

                            // Don't combine the loops, it breaks the compiler optimisation (GCC 9)
                            for (int i = 0; i < 13; i++) {
                                const int xx = x + reconstructor->_directions[i][0];
                                const int yy = y + reconstructor->_directions[i][1];
                                const int zz = z + reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                    && reconstructor->_confidence_map4D(xx, yy, zz, 0) > 0) {
                                    val += b[i](x, y, z) * original(xx, yy, zz);
                                    sum += b[i](x, y, z);
                                }
                            }

                            for (int i = 0; i < 13; i++) {
                                const int xx = x - reconstructor->_directions[i][0];
                                const int yy = y - reconstructor->_directions[i][1];
                                const int zz = z - reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                    && reconstructor->_confidence_map4D(xx, yy, zz, 0) > 0) {
                                    val += b[i](x, y, z) * original(xx, yy, zz);
                                    sum += b[i](x, y, z);
                                }
                            }

                            val -= sum * original(x, y, z);
                            val = original(x, y, z) + reconstructor->_alpha * reconstructor->_lambda /
                                                         (reconstructor->_delta * reconstructor->_delta) * val;
                            reconstructor->_T2Map(x, y, z) = val;
                        }
                    }
                }
            }
        }

        void operator()() const {
            parallel_for(blocked_range<size_t>(0, reconstructor->_T2Map.GetZ()), *this);
        }
    };

    //-------------------------------------------------------------------

    /// Class for bias normalisation
    class NormaliseBiasqMRI {
        ReconstructionqMRI *reconstructor;

    public:
        RealImage bias;

        NormaliseBiasqMRI(ReconstructionqMRI *reconstructor) : reconstructor(reconstructor) {
            bias.Initialize(reconstructor->_reconstructed4D.Attributes());
        }

        NormaliseBiasqMRI(NormaliseBiasqMRI& x, split) : NormaliseBiasqMRI(x.reconstructor) {}

        void operator()(const blocked_range<size_t>& r) {
            RealImage b;

            for (size_t inputIndex = r.begin(); inputIndex < r.end(); inputIndex++) {
                if (reconstructor->_volcoeffs[inputIndex].empty())
                    continue;

                if (reconstructor->_verbose)
                    reconstructor->_verbose_log << inputIndex << " ";

                // alias to the current slice
                const RealImage& slice = reconstructor->_slicesqMRI[inputIndex];

                //read the current bias image
                b = reconstructor->_biasqMRI[inputIndex];

                //read current scale factor
                const double scale = reconstructor->_scaleqMRI[inputIndex];

                const RealPixel *pi = slice.Data();
                RealPixel *pb = b.Data();
                for (int i = 0; i < slice.NumberOfVoxels(); i++)
                    if (pi[i] > -1 && scale > 0)
                        pb[i] -= log(scale);

                int indstack = reconstructor->_stack_index[inputIndex];
                int outputIndex = reconstructor->_volume_index[indstack];

                //Distribute slice intensities to the volume
                for (size_t i = 0; i < reconstructor->_volcoeffs[inputIndex].size(); i++)
                    for (size_t j = 0; j < reconstructor->_volcoeffs[inputIndex][i].size(); j++)
                        if (slice(i, j, 0) > -0.01) {
                            //add contribution of current slice voxel to all voxel volumes to which it contributes
                            for (size_t k = 0; k < reconstructor->_volcoeffs[inputIndex][i][j].size(); k++) {
                                const POINT3D& p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                bias(p.x, p.y, p.z,outputIndex) += p.value * b(i, j, 0);
                            }
                        }
                //end of loop for a slice inputIndex
            }
        }

        void join(const NormaliseBiasqMRI& y) {
            bias += y.bias;
        }

        void operator()() {
            parallel_reduce(blocked_range<size_t>(0, reconstructor->_slices.size()), *this);
        }
    };

    //-------------------------------------------------------------------
}

