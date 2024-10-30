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

#pragma once

// SVRTK
#include "svrtk/Reconstruction.h"
#include "svrtk/Dictionary.h"
#include "mirtk/HistogramMatching.h"

using namespace mirtk;

namespace svrtk {

    // Forward declarations
    namespace Parallel {/*
        class CoeffInitqMRI;
        class SliceToVolumeRegistrationqMRI;
        class SimulateStacksqMRI;*/
        class CalculateError;
        class CalculateCorrectedSlices;
    }
    namespace ParallelqMRI {/*
        class CoeffInitqMRI;
        class SliceToVolumeRegistrationqMRI;
        class SimulateStacksqMRI;*/
        class NormaliseBiasqMRI;
        class AdaptiveRegularization1qMRI;
        class AdaptiveRegularization2qMRI;
        class AdaptiveRegularization1T2Map;
        class AdaptiveRegularization2T2Map;
        class SimulateSlicesqMRI;
        class SuperresolutionqMRI;
        class MStepqMRI;
        class EStepqMRI;
        class ModelFitqMRI;
        class ScaleqMRI;
        class BiasqMRI;
    }

    class ReconstructionqMRI: public Reconstruction {
    protected:
        // Dictionary object
        Dictionary _GivenDictionary;
        int _nEchoTimes = 3;
        int _templateNumber = 3;
        Array<int> _volume_index;
        Array<int> _volume_index_for_slices;

        vector<double> _echoTimes;
        vector<int> _StacksPerEchoTime;
        vector<int> _NumSlicesPerVolume;

        // Finite difference step size
        double _Step = 1;

        ImageAttributes _attr_reconstructed4D;

        Array<RealImage> _slicesqMRI;
        Array<RealImage> _simulated_slicesqMRI;
        Array<RealImage> _simulated_weightsqMRI;
        Array<RealImage> _simulated_insideqMRI;
        Array<RealImage> _slice_difqMRI;


        /// Voxel posteriors
        Array<RealImage> _weightsqMRI;
        /// Slice posteriors
        Array<double> _slice_weightqMRI;

        /// Indicator whether slice has an overlap with volumetric mask
        Array<bool> _slice_insideqMRI;

        //Bias field
        /// Variance for bias field
        double _sigma_biasqMRI;

        /// Slice-dependent bias fields
        Array<RealImage> _biasqMRI;

        /// Slice-dependent scales
        Array<double> _scaleqMRI;

        /// Profile output mode
        bool _profile = false;

        // EM algorithm
        /// Variance for inlier voxel errors
        vector<double> _sigma_qMRI;
        /// Proportion of inlier voxels
        vector<double> _mix_qMRI;
        /// Mean for inlier slice errors
        vector<double> _mean_s_qMRI;
        /// Variance for inlier slice errors
        vector<double> _sigma_s_qMRI;
        /// Mean for outlier slice errors
        vector<double> _mean_s2_qMRI;
        /// Variance for outlier slice errors
        vector<double> _sigma_s2_qMRI;
        /// Proportion of inlier slices
        vector<double> _mix_s_qMRI;
        /// Uniform distribution for outlier voxels
        vector<double> _m_qMRI;

        // Flag for Finite difference method
        bool _FiniteDiff = true;

        // Flag for intensity matching
        bool _MatchIntensity;

        // Array of Stack Averages
        Array<double> _stack_averages;

        //Gradient descent and regularisation parameters
        /// Step for gradient descent for model
        double _epsilon;

        /// Weights for Gaussian reconstruction
        RealImage _volume_weightsqMRI;

        /// Weights for regularization
        RealImage _confidence_map4D;

        /// Average voxel wights to modulate parameter alpha
        vector<double> _average_volume_weightqMRI;

        bool _jointSR = true;

        // All reconstructed 3D Images from the Signal in one image object
        RealImage _reconstructed4D;

        // For multiple masks:
        Array<RealImage> _arrayVols;

        double _maxVal;
        double _minVal;

        vector<double> _volscales;

        // For multiple masks:
        Array<RealImage> _Masks;

        // All reconstructed 3D Images from the Signal in one image object
        RealImage _volumeLabel4D;

        // Final T2 Map
        RealImage _T2Map;

        // Stack factor for volumes
        RealImage _volumestackfactor;

        // Vector of TEs for each stack.
        vector<double> _echoTimesforStacks;

        /// Scale volume common function
        vector<double>  ScaleVolumeqMRI(RealImage&, bool);

        /// Intensity min and max
        vector<double> _max_intensitiesqMRI;
        vector<double> _min_intensitiesqMRI;

        ///
        double _max_Standard_intensity;
        double _min_Standard_intensity;

        //to restore original signal intensity of the MRI slices
        Array<double> _stack_factorqMRI;
    
    public:

        int _SR_iter = 1;

        /// ReconstructionqMRI Constructor
        ReconstructionqMRI() : Reconstruction() {}

        //ReconstructionqMRI() : Reconstruction() {}

        /// ReconstructionqMRI Destructor
        ~ReconstructionqMRI() {}

        // Set _epsilon
        void SetEpsilon(const double & epsilon){
            _epsilon = epsilon;
        }

        // Instantiate Dictionary

        void InstantiateDictionary(const Dictionary& inputDictionary){
            _GivenDictionary = inputDictionary;
        }

        // Make TE vector
        void MakeTEVect(vector<int> StacksPerEchoTime, vector<double> echoTimes, int templateNumber) {
            _StacksPerEchoTime = StacksPerEchoTime;
            _echoTimes = echoTimes;
            _nEchoTimes = echoTimes.size();
            _templateNumber = templateNumber;
            for (int i = 0; i < StacksPerEchoTime.size(); i++) {
                for (int j = 0; j < StacksPerEchoTime[i]; j++) {
                    _echoTimesforStacks.push_back(echoTimes[i]);
                    _volume_index.push_back(i);
                }
            }
        }

        void SetStandardMinMax(double minval, double maxval){
            _min_Standard_intensity = minval;
            _max_Standard_intensity = maxval;
        }

        // Function to set Intensity Matching:
        void IntensityMatchingSwitch(bool IntensityMatching){
            _MatchIntensity = IntensityMatching;
        }

        // Function to set Joint Resolution:
        void SuperResolutionSwitch(bool jointSR){
            _jointSR = jointSR;
        }

        // Function to set Finite Difference method flag:
        void FiniteDifferenceSwitch(bool FiniteDifference){
            _FiniteDiff  = FiniteDifference;
        }

        // Function to set Finite Difference method step size:
        void SetFiniteDiffStep(double step){
            _Step  = step;
        }

        // Function to set Finite Difference method step size:
        void MultiplyAlpha(double multiplier){
            _alpha  *= multiplier;
        }

        // Function to get Finite Difference method step size:
        double GetFiniteDiffStep(){
            return _Step;
        }

        /// Enable profiling mode
        inline void ProfileOn() {
            _profile = true;
        }

        /// Disable profiling mode
        inline void ProfileOff() {
            _profile = false;
        }

        // Function to get T2 Map:
        RealImage GetT2Map(){
            return _T2Map;
        }

        /// Set sigma flag
        inline void SetSigmaqMRI(double sigma) {
            _sigma_biasqMRI = sigma;
        }

        // Function to carry out Intensity Matching:
        bool IsIntensityMatching(){
            return _MatchIntensity;
        }

        // Function to carry out Finite Difference Method:
        bool UseFiniteDifference(){
            return _FiniteDiff;
        }

        // Function to set Finite Difference method step size:
        void SetT2Map(RealImage T2Map){
            _T2Map  = T2Map;
        }

        double CreateTemplateqMRI(const Array<RealImage> stack, double resolution = 0,  int volumetemplateNumber = 0);

        void SetMasksqMRI(Array<RealImage> Masks, vector<int> MaskStackNums, int templateNumber, double sigma, double threshold = 0.5);

        /// Run superresolution reconstruction step
        void SuperresolutionqMRI(int iter);

        void GaussianReconstructionqMRI(int iter);

        /// Calculate transformation matrix between slices and voxels
        void CoeffInitqMRI();

        /// Scale volume to match the slice intensities
        inline void ScaleVolumeqMRI() { vector<double> scales = ScaleVolumeqMRI(_reconstructed4D, false); }

        /// Initalise parameters of EM robust statistics
        void InitializeRobustStatisticsqMRI();

        /// Calculate slice-dependent scale
        void ScaleqMRI();

        /// Perform EStep for calculation of voxel-wise and slice-wise posteriors (weights)
        void EStepqMRI();

        /// Run MStep
        void MStepqMRI(int iter);

        /// Return mask
        inline const RealImage& GetMaskqMRI(int ii) {
            return _Masks[ii];
        }

        /// Return mask
        inline const Array<RealImage>& GetMasksqMRI() {
            return _Masks;
        }

        // Calculate slice-dependent bias fields
        /// Run slice bias correction
        void BiasqMRI();
        /// Normalise bias
        void NormaliseBiasqMRI(int iter);

        /// Run simulation of slices from the reconstruction volumes
        void SimulateSlicesqMRI();

        void CreateSimulatedSlices(const Array<RealImage>, const Array<double>);

        /**
         * @brief Create slices from the stacks and slice-dependent transformations from stack transformations.
         * @param stacks
         * @param stack_transformations
         * @param thickness
         * @param probability_maps
         */
        void CreateSlicesAndTransformationsqMRI(const Array<RealImage>& stacks, const Array<RigidTransformation>& stack_transformations,
                                                const Array<double>& thickness, const Array<RealImage>& probability_maps, RealImage templateStack,
                                                bool HistogramMatchingB,double averageValue);


        /**
         * @brief Evaluate reconstruction quality (errors per slices).
         * @param index
         * @return Reconstruction quality.
         */
        double EvaluateReconQualityqMRI(int index);


            /**
             * @brief Run edge-preserving regularization with confidence map.
             * @param iter
             * @param original
             */
        void AdaptiveRegularizationqMRI(int iter, const RealImage& original);

        /**
         * @brief Run edge-preserving regularization with confidence map.
         * @param iter
         * @param original
         */
        void AdaptiveRegularizationT2Map(int iter, const RealImage& original);


        /**
         * @brief Match stack intensities with respect to the masked ROI.
         * @param stacks
         * @param stack_transformations
         * @param averageValue
         * @param together
         */
        void MatchStackIntensitiesWithMaskingqMRI(Array<RealImage>& stacks,
                                              const Array<RigidTransformation>& stack_transformations, double averageValue, bool together = false);

        /// Mask slices based on the reconstruction mask
        void MaskSlicesqMRI();

        /// Compute difference between simulated and original slices
        void SliceDifferenceqMRI();

        void FindStackAverage(Array<RealImage>& stacks,const Array<RigidTransformation>& stack_transformations);

        /// Return reconstructed volume
        inline const RealImage& GetReconstructedqMRI() {
            return _reconstructed4D;
        }

        Array<RealImage> GetReconstructedIMGs(){
            Array<RealImage> OutputIMGs;
            for (int t = 0; t < _reconstructed4D.GetT(); t++) {
                RealImage IMG = _reconstructed4D.GetRegion(0, 0, 0, t, _reconstructed4D.GetX(),
                                                            _reconstructed4D.GetY(),
                                                            _reconstructed4D.GetZ(), t + 1);
                OutputIMGs.push_back(IMG);
            }
            return OutputIMGs;
        }

        RealImage MaskAndGetMaskSlice(RealImage &stack){
            RealImage mask(stack.Attributes());
            RealPixel *ps = stack.Data();
            RealPixel *pm = mask.Data();
            #pragma omp parallel for
            for (int i = 0; i < stack.NumberOfVoxels(); i++) {
                if (ps[i] <= 0) {
                    ps[i] = -1;
                    pm[i] = 0;
                }
                else
                    pm[i] = 1;
            }
            return mask;
        }


        void WriteReconstructedIMGs(){
            for (int t = 0; t < _reconstructed4D.GetT(); t++) {
                RealImage IMG = _reconstructed4D.GetRegion(0, 0, 0, t, _reconstructed4D.GetX(),
                                                           _reconstructed4D.GetY(),
                                                           _reconstructed4D.GetZ(), t + 1);
                double TEtoWrite = _echoTimes[t];
                IMG.Write((boost::format("Recon_TE_%1%ms.nii.gz") % TEtoWrite).str().c_str());
            }
        }


        void WriteReconstructedIMGs(const string& Fname){
            for (int t = 0; t < _reconstructed4D.GetT(); t++) {
                RealImage IMG = _reconstructed4D.GetRegion(0, 0, 0, t, _reconstructed4D.GetX(),
                                                           _reconstructed4D.GetY(),
                                                           _reconstructed4D.GetZ(), t + 1);
                string writeName = Fname;
                double TEtoWrite = _echoTimes[t];
                string b = ".nii.gz";
                string PostName = "_Recon_TE_%1%ms";
                size_t pos = Fname.find(b);
                writeName.insert(pos,PostName);
                //PreName.append(Fname);
                IMG.Write((boost::format(writeName) % TEtoWrite).str().c_str());
            }
        }

        void WriteHistogramMatchedIMGs(RealImage SingIMG){
            RealImage ImageMatched;
            HistogramMatching<RealPixel> match;
            for (int t = 0; t < _reconstructed4D.GetT(); t++) {
                RealImage IMG = _reconstructed4D.GetRegion(0, 0, 0, t, _reconstructed4D.GetX(),
                                                           _reconstructed4D.GetY(),
                                                           _reconstructed4D.GetZ(), t + 1);
                match.Input(&SingIMG);
                match.Reference(&IMG);
                match.Output(&ImageMatched);
                match.Run();
                //PreName.append(Fname);double
                double TEtoWrite = _echoTimes[t];
                ImageMatched.Write((boost::format("ReconHistMatched_TE_%1%ms.nii.gz") % TEtoWrite).str().c_str());
            }
        }

        /// Save simulated slices
        void SaveSimulatedSlicesqMRI();

        /// Save non-histogram-matched slices
        void SaveSlicesqMRI();

        /// Mask the volume
        inline void MaskVolumeqMRI() { StaticMaskVolume4D(_reconstructed4D, _mask, -1); }

        /**
         * @brief Correct bias in the reconstructed volume.
         * @param original
         */
        void BiasCorrectVolumeqMRI(const RealImage& original);

        /// Initialise variables and parameters for EM
        void InitializeEMqMRI();

        /// Initialise values of variables and parameters for EM
        void InitializeEMValuesqMRI();

        /// Restore slice intensities to their original values to recover original scaling for copy slices
        void RestoreSliceIntensitiesqMRI();

        /// Remove less than 0 intensities
        void RemoveNegativeBackground(RealImage&);

        friend class ParallelqMRI::SuperresolutionqMRI;
        friend class ParallelqMRI::SimulateSlicesqMRI;
        friend class ParallelqMRI::AdaptiveRegularization1qMRI;
        friend class ParallelqMRI::AdaptiveRegularization2qMRI;
        friend class ParallelqMRI::AdaptiveRegularization1T2Map;
        friend class ParallelqMRI::AdaptiveRegularization2T2Map;
        friend class ParallelqMRI::NormaliseBiasqMRI;
        friend class ParallelqMRI::ModelFitqMRI;
        friend class ParallelqMRI::MStepqMRI;
        friend class ParallelqMRI::EStepqMRI;
        friend class ParallelqMRI::ScaleqMRI;
        friend class ParallelqMRI::BiasqMRI;
    
    };  // end of ReconstructionqMRI class definition

} // namespace svrtk
