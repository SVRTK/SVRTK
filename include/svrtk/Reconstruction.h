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

#pragma once

// SVRTK
#include "svrtk/Common.h"

using namespace std;
using namespace mirtk;
using namespace svrtk::Utility;

namespace svrtk {

    // Forward declarations
    namespace Parallel {
        class GlobalSimilarityStats;
        class QualityReport;
        class StackRegistrations;
        class SliceToVolumeRegistration;
        class SliceToVolumeRegistrationFFD;
        class RemoteSliceToVolumeRegistration;
        class CoeffInit;
        class CoeffInitSF;
        class Superresolution;
        class SStep;
        class MStep;
        class EStep;
        class Bias;
        class Scale;
        class NormaliseBias;
        class SimulateSlices;
        class SimulateMasks;
        class Average;
        class AdaptiveRegularization1;
        class AdaptiveRegularization2;
        class AdaptiveRegularization1MC;
        class AdaptiveRegularization2MC;
    }

    /**
     * @brief Reconstruction class used reconstruction.
     */
    class Reconstruction {
    protected:
        /// Reconstruction type
        RECON_TYPE _recon_type;

        /// Structures to store the matrix of transformation between volume and slices
        Array<SLICECOEFFS> _volcoeffs;
        Array<SLICECOEFFS> _volcoeffsSF;

        /// flags
        int _slicePerDyn;
        bool _ffd;
        bool _blurring;
        bool _structural;
        bool _ncc_reg;
        bool _template_flag;
        bool _no_sr;
        bool _reg_log;
        bool _masked_stacks;
        bool _ffd_global_only;
        bool _ffd_global_ncc;
        bool _no_masking_background;
        bool _no_offset_registration;

        double _global_NCC_threshold;
        int _local_SSIM_window_size;
        double _local_SSIM_threshold;
        double _global_JAC_threshold;
        
        Array<int> _n_packages;

        ImageAttributes _attr_reconstructed;

        Array<Matrix> _offset_matrices;
        bool _saved_slices;
        Array<int> _zero_slices;

        int _current_iteration;
        Array<int> _cp_spacing;
        int _global_cp_spacing;
        bool _filtered_cmp_flag;
        bool _bg_flag;

        /// Slices
        Array<RealImage> _slices;
        Array<RealImage> _slicesRwithMB;
        Array<RealImage> _simulated_slices;
        Array<RealImage> _simulated_weights;
        Array<RealImage> _simulated_inside;
        Array<RealImage> _slice_masks;
        Array<RealImage> _slice_ssim_maps;
        Array<ImageAttributes> _slice_attributes;
        Array<RealImage> _slice_dif;
        
        Array<RealImage> _not_masked_slices;

        Array<GreyImage> _grey_slices;
        GreyImage _grey_reconstructed;

        Array<int> _structural_slice_weight;
        Array<int> _package_index;
        Array<int> _slice_pos;

        /// Transformations
        Array<RigidTransformation> _transformations;
        Array<RigidTransformation> _previous_transformations;
        Array<RigidTransformation> _transformationsRwithMB;
        Array<MultiLevelFreeFormTransformation*> _mffd_transformations;
        Array<MultiLevelFreeFormTransformation*> _global_mffd_transformations;

        /// Indicator whether slice has an overlap with volumetric mask
        Array<bool> _slice_inside;
        Array<bool> _slice_insideSF;

        /// Flag to say whether the template volume has been created
        bool _template_created;
        /// Flag to say whether we have a mask
        bool _have_mask;

        /// Volumes
        RealImage _reconstructed;
        RealImage _mask;
        RealImage _evaluation_mask;
        RealImage _target;
        RealImage _brain_probability;
        Array<RealImage> _probability_maps;

        /// Weights for Gaussian reconstruction
        RealImage _volume_weights;
        RealImage _volume_weightsSF;

        /// Weights for regularization
        RealImage _confidence_map;
        
        
        int _number_of_channels;
        bool _multiple_channels_flag;
        
        bool _combined_rigid_ffd;
        
        Array<Array<RealImage*>> _mc_slices;
        Array<Array<RealImage*>> _mc_simulated_slices;
        Array<Array<RealImage*>> _mc_slice_dif;

        Array<RealImage> _mc_reconstructed;
        
        Array<double> _max_intensity_mc;
        Array<double> _min_intensity_mc;
        

        // EM algorithm
        /// Variance for inlier voxel errors
        double _sigma;
        /// Proportion of inlier voxels
        double _mix;
        /// Uniform distribution for outlier voxels
        double _m;
        /// Mean for inlier slice errors
        double _mean_s;
        /// Variance for inlier slice errors
        double _sigma_s;
        /// Mean for outlier slice errors
        double _mean_s2;
        /// Variance for outlier slice errors
        double _sigma_s2;
        /// Proportion of inlier slices
        double _mix_s;
        /// Step size for likelihood calculation
        double _step;
        /// Voxel posteriors
        Array<RealImage> _weights;
        /// Slice posteriors
        Array<double> _slice_weight;

        /// NMI registration bins
        int _nmi_bins;

        //Bias field
        /// Variance for bias field
        double _sigma_bias;
        /// Slice-dependent bias fields
        Array<RealImage> _bias;

        /// Slice-dependent scales
        Array<double> _scale;

        /// Quality factor - higher means slower and better
        double _quality_factor;
        /// Intensity min and max
        double _max_intensity;
        double _min_intensity;

        //Gradient descent and regularisation parameters
        /// Step for gradient descent
        double _alpha;
        /// Determine what is en edge in edge-preserving smoothing
        double _delta;
        /// Amount of smoothing
        double _lambda;
        /// Average voxel wights to modulate parameter alpha
        double _average_volume_weight;
        double _average_volume_weightSF;

        //global bias field correction
        /// global bias correction flag
        bool _global_bias_correction;
        /// low intensity cutoff for bias field estimation
        double _low_intensity_cutoff;

        //to restore original signal intensity of the MRI slices
        Array<double> _stack_factor;
        double _average_value;
        Array<int> _stack_index;

        Array<Array<double>> _mc_stack_factor;
        
        /// GF 200416 Handling slice acquisition order
        Array<int> _slice_timing;

        //forced excluded slices
        Array<int> _force_excluded;

        //slices identify as too small to be used
        Array<int> _small_slices;

        /// use adaptive or non-adaptive regularisation (default:false)
        bool _adaptive;

        //utility
        /// Debug mode
        bool _debug;

        /// Profile output mode
        bool _profile;

        /// Verbose mode
        bool _verbose;
        ostream _verbose_log {cout.rdbuf()};
        ofstream _verbose_log_stream_buf;

        //do not exclude voxels, only whole slices
        bool _robust_slices_only;

        bool _withMB;

        int _directions[13][3] = {
            {1, 0, -1},
            {0, 1, -1},
            {1, 1, -1},
            {1, -1, -1},
            {1, 0, 0},
            {0, 1, 0},
            {1, 1, 0},
            {1, -1, 0},
            {1, 0, 1},
            {0, 1, 1},
            {1, 1, 1},
            {1, -1, 1},
            {0, 0, 1}
        };

        /// Gestational age (to compute expected brain volume)
        double _GA;

        /// Read Transformations common function
        void ReadTransformations(const char *folder, size_t file_count, Array<RigidTransformation>& transformations);

        /// Scale volume common function
        void ScaleVolume(RealImage& reconstructed);

    public:
        /// Reconstruction constructor
        Reconstruction();
        /// Reconstruction destructor
        ~Reconstruction() {}

        int _number_of_slices_org;
        double _average_thickness_org;

        /**
         * @brief Create zero image as a template for reconstructed volume.
         * @param stack Stack to be set as template.
         * @param resolution Resolution to interpolate the input stack.
         * @return The resulting resolution of the template image.
         */
        double CreateTemplate(const RealImage& stack, double resolution = 0);

        /**
         * @brief Create anisotropic template.
         * @param stack Stack to be set as template.
         * @return The resulting resolution of the template image.
         */
        double CreateTemplateAniso(const RealImage& stack);

        /// Array of excluded slices.
        Array<int> _excluded_entirely;

        /**
         * @brief Remember volumetric mask and smooth it if necessary.
         * @param mask Mask to be set.
         * @param sigma
         * @param threshold
         * @return The resulting resolution of the template image.
         */
        void SetMask(RealImage *mask, double sigma, double threshold = 0.5);

        /// Compute difference between simulated and original slices
        void SliceDifference();

        /**
         * @brief Save slice info in a CSV file.
         * @param current_iteration Iteration number will be put into the filename.
         */
        void SaveSliceInfo(int current_iteration = -1);

        /**
         * @brief Create average image from the stacks and volumetric transformations.
         * @param stacks
         * @param stack_transformations
         * @return
         */
        RealImage CreateAverage(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations);

        /**
         * @brief Calculate initial registrations.
         * @param stacks
         * @param stack_transformations
         * @param templateNumber
         */
        void StackRegistrations(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber, RealImage *input_template);

        /**
         * @brief Create slices from the stacks and slice-dependent transformations from stack transformations.
         * @param stacks
         * @param stack_transformations
         * @param thickness
         * @param probability_maps
         */
        void CreateSlicesAndTransformations(const Array<RealImage>& stacks, const Array<RigidTransformation>& stack_transformations,
            const Array<double>& thickness, const Array<RealImage>& probability_maps);
        
        void CreateSlicesAndTransformationsMC(const Array<RealImage>& stacks, Array<Array<RealImage>> mc_stacks, const Array<RigidTransformation>& stack_transformations, const Array<double>& thickness, const Array<RealImage>& probability_maps);

        /**
         * @brief Set slices and transformations from the given array.
         * @param slices
         * @param slice_transformations
         * @param stack_ids
         * @param thickness
         */
        void SetSlicesAndTransformations(const Array<RealImage>& slices,
            const Array<RigidTransformation>& slice_transformations, const Array<int>& stack_ids, const Array<double>& thickness);

        /**
         * @brief Reset slices array and read them from the input stacks.
         * @param stacks
         * @param thickness
         */
        void ResetSlices(Array<RealImage>& stacks, Array<double>& thickness);

        /**
         * @brief Update slices based on the given stacks.
         * @param stacks
         * @param thickness
         */
        void UpdateSlices(Array<RealImage>& stacks, Array<double>& thickness);

        /**
         * @brief Saves probability map.
         * @param i
         */
        void SaveProbabilityMap(int i);

        /**
         * @brief Match stack intensities with respect to the average.
         * @param stacks
         * @param stack_transformations
         * @param averageValue
         * @param together
         */
        void MatchStackIntensities(Array<RealImage>& stacks,
            const Array<RigidTransformation>& stack_transformations, double averageValue, bool together = false);

        /**
         * @brief Match stack intensities with respect to the masked ROI.
         * @param stacks
         * @param stack_transformations
         * @param averageValue
         * @param together
         */
        void MatchStackIntensitiesWithMasking(Array<RealImage>& stacks,
            const Array<RigidTransformation>& stack_transformations, double averageValue, bool together = false);
        
        void MatchStackIntensitiesWithMaskingMC(Array<RealImage>& stacks,
            const Array<RigidTransformation>& stack_transformations, double averageValue, bool together = false);

        /// Mask slices based on the reconstruction mask
        void MaskSlices();
        
        // Create slice masks for the current iterations
        void CreateSliceMasks();
        
        void GlobalReconMask();

        /// Set reconstructed image
        void SetTemplate(RealImage tempImage);

        // void SetFFDGlobalOnly();

        /// Calculate transformation matrix between slices and voxels
        void CoeffInit();
        /// Calculate transformation matrix between slices and voxels
        void CoeffInitSF(int begin, int end);

        /// Run gaussian reconstruction based on SVR & coeffinit outputs
        void GaussianReconstruction();
        /// Run gaussian reconstruction based on SVR & coeffinit outputs
        void GaussianReconstructionSF(const Array<RealImage>& stacks);

        /// Initialise variables and parameters for EM
        void InitializeEM();

        /// Initialise values of variables and parameters for EM
        void InitializeEMValues();

        /// Initalise parameters of EM robust statistics
        void InitializeRobustStatistics();

        /// Perform EStep for calculation of voxel-wise and slice-wise posteriors (weights)
        void EStep();

        /// Calculate slice-dependent scale
        void Scale();

        // Calculate slice-dependent bias fields
        /// Run slice bias correction
        void Bias();
        /// Normalise bias
        void NormaliseBias(int iter);

        /// Run superresolution reconstruction step
        void Superresolution(int iter);

        /// Run MStep
        void MStep(int iter);
        
        /// Run SStep 
        void SStep();

        /**
         * @brief Run edge-preserving regularization with confidence map.
         * @param iter
         * @param original
         */
        void AdaptiveRegularization(int iter, const RealImage& original);
        
        void AdaptiveRegularizationMC( int iter, Array<RealImage>& mc_originals);

        /// Run slice to volume registration
        void SliceToVolumeRegistration();

        /**
         * @brief Run remote slice to volume registration.
         * @param iter
         * @param str_mirtk_path
         * @param str_current_exchange_file_path
         */
        void RemoteSliceToVolumeRegistration(int iter, const string& str_mirtk_path, const string& str_current_exchange_file_path);

        /// Save the current reconstruction model
        void SaveModelRemote(const string& str_current_exchange_file_path, int status_flag, int current_iteration);
        /// Load the current reconstruction model
        void LoadModelRemote(const string& str_current_exchange_file_path, int current_number_of_slices, double average_thickness, int current_iteration);
        /// Load remotely reconstructed volume
        void LoadResultsRemote(const string& str_current_exchange_file_path, int current_number_of_slices, int current_iteration);

        /**
         * @brief Correct bias in the reconstructed volume.
         * @param original
         */
        void BiasCorrectVolume(const RealImage& original);

        /// Mask the volume
        inline void MaskVolume() { MaskImage(_reconstructed, _mask, -1); }

        /// Save slices
        void SaveSlices();
        /// Save slices with timing info
        void SaveSlicesWithTiming();
        /// Save slice info
        void SaveSliceInfo(const char *filename, const Array<string>& stack_filenames);

        /// Save simulated slices
        void SaveSimulatedSlices();

        /// Save weights
        void SaveWeights();
        /// Save registration step
        void SaveRegistrationStep(const Array<RealImage>& stacks, const int step);

        /// Save transformations
        void SaveTransformations();
        /// Save transformations with timing info
        void SaveTransformationsWithTiming(const int iter = -1);

        /// Save confidence map
        void SaveConfidenceMap();

        /// Save bias fields
        void SaveBiasFields();

        /**
         * @brief Generate reconstruction quality report / metrics.
         * @param out_ncc
         * @param out_nrmse
         * @param average_weight
         * @param ratio_excluded
         */
        void ReconQualityReport(double& out_ncc, double& out_nrmse, double& average_weight, double& ratio_excluded);

        /**
         * @brief Evaluation based on the number of excluded slices.
         * @param iter
         * @param outstr
         */
        void Evaluate(int iter, ostream& outstr = cout);

        /// Read transformations
        void ReadTransformations(const char *folder);

        /// Restore slice intensities to their original values to recover original scaling
        void RestoreSliceIntensities();

        /// Scale volume to match the slice intensities
        inline void ScaleVolume() { ScaleVolume(_reconstructed); }

        /**
         * @brief Simulate stacks from the reconstructed volume to compare how simulation
         * from the reconstructed volume matches the original stacks.
         * @param stacks
         */
        void SimulateStacks(Array<RealImage>& stacks);

        /// Run simulation of slices from the reconstruction volume
        void SimulateSlices();

        /**
         * @brief Run package-to-volume registration.
         * @param stacks
         * @param pack_num
         * @param stack_transformations
         */
        void PackageToVolume(const Array<RealImage>& stacks, const Array<int>& pack_num,
            const Array<RigidTransformation>& stack_transformations);

        // Performs package registration
        /**
         * @brief Run updated package-to-volume registration.
         * @param stacks
         * @param pack_num
         * @param multiband
         * @param order
         * @param step
         * @param rewinder
         * @param iter
         * @param steps
         */
        void newPackageToVolume(const Array<RealImage>& stacks, const Array<int>& pack_num, const Array<int>& multiband,
            const Array<int>& order, const int step, const int rewinder, const int iter, const int steps);

        /// Run structure-based rejection of outliers
        void GlobalStructuralExclusion();
        
        // Compute local SSIM map between simulated and original slices
        void SliceSSIMMap(int inputIndex);

        /**
         * @brief Evaluate reconstruction quality (errors per slices).
         * @param index
         * @return Reconstruction quality.
         */
        double EvaluateReconQuality(int index);

        /// Initialisation slice transformations with stack transformations
        void InitialiseWithStackTransformations(const Array<RigidTransformation>& stack_transformations);

        /**
         * @brief Transform slice coordinates to the reconstructed space.
         * @param inputIndex
         * @param i
         * @param j
         * @param k
         * @param mode
         */
        void Transform2Reconstructed(const int inputIndex, int& i, int& j, int& k, const int mode);

        friend class Parallel::GlobalSimilarityStats;
        friend class Parallel::QualityReport;
        friend class Parallel::StackRegistrations;
        friend class Parallel::SliceToVolumeRegistration;
        friend class Parallel::SliceToVolumeRegistrationFFD;
        friend class Parallel::RemoteSliceToVolumeRegistration;
        friend class Parallel::CoeffInit;
        friend class Parallel::CoeffInitSF;
        friend class Parallel::Superresolution;
        friend class Parallel::MStep;
        friend class Parallel::EStep;
        friend class Parallel::SStep;
        friend class Parallel::Bias;
        friend class Parallel::Scale;
        friend class Parallel::NormaliseBias;
        friend class Parallel::SimulateSlices;
        friend class Parallel::SimulateMasks;
        friend class Parallel::Average;
        friend class Parallel::AdaptiveRegularization1;
        friend class Parallel::AdaptiveRegularization2;
        friend class Parallel::AdaptiveRegularization1MC;
        friend class Parallel::AdaptiveRegularization2MC;

        ////////////////////////////////////////////////////////////////////////////////
        // Inline/template definitions
        ////////////////////////////////////////////////////////////////////////////////

        /// Zero-mean Gaussian PDF
        inline double G(double x, double s) {
            return _step * exp(-x * x / (2 * s)) / (sqrt(6.28 * s));
        }

        /// Uniform PDF
        inline double M(double m) {
            return m * _step;
        }
        
        inline void SetMCSettings(int num)
        {
            _number_of_channels = num;
            _multiple_channels_flag = true;
        }

        
        
        inline void SetRigidFFDSettings(bool combined_rigid_ffd)
        {
            _combined_rigid_ffd = combined_rigid_ffd;
        }

        inline void SaveMCReconstructed()
        {
            char buffer[256];
            if (_multiple_channels_flag && (_number_of_channels > 0)) {
                for (int n=0; n<_number_of_channels; n++) {
                    sprintf(buffer,"mc-output-%i.nii.gz", n);
                    _mc_reconstructed[n].Write(buffer);
                    cout << "- " << buffer << endl;
                }
            }
        }

        /// Return reconstructed volume
        inline const RealImage& GetReconstructed() {
            return _reconstructed;
        }

        /// Set reconstructed volume
        inline void SetReconstructed(const RealImage& reconstructed) {
            _reconstructed = reconstructed;
            _template_created = true;
        }

        inline void SetCurrentIteration(int current_iteration) {
            _current_iteration = current_iteration;
        }
        
        /// Return mask
        inline const RealImage& GetMask() {
            return _mask;
        }

        /// Set mask
        inline void PutMask(const RealImage& mask) {
            _mask = mask;
        }

        /// Set FFD
        inline void SetFFD(bool flag_ffd) {
            _ffd = flag_ffd;
        }

        /// Set NCC
        inline void SetNCC(bool flag_ncc) {
            _ncc_reg = flag_ncc;
        }

        /// Set reglog
        inline void SetRegLog(bool flag_reg_log) {
            _reg_log = flag_reg_log;
        }

        /// Set SR
        inline void SetSR(bool flag_sr) {
            _no_sr = flag_sr;
        }

        /// Set template flag
        inline void SetTemplateFlag(bool template_flag) {
            _template_flag = template_flag;
        }

        /// Enable debug mode
        inline void DebugOn() {
            _debug = true;
        }

        /// Disable debug mode
        inline void DebugOff() {
            _debug = false;
        }

        /// Enable profiling mode
        inline void ProfileOn() {
            _profile = true;
        }

        /// Disable profiling mode
        inline void ProfileOff() {
            _profile = false;
        }

        /// Enable verbose mode
        inline void VerboseOn(const string& log_file_name = "") {
            VerboseOff();

            if (!log_file_name.empty()) {
                _verbose_log_stream_buf.open(log_file_name, ofstream::out | ofstream::trunc);
                _verbose_log.rdbuf(_verbose_log_stream_buf.rdbuf());
            }

            _verbose = true;
        }

        /// Disable verbose mode
        inline void VerboseOff() {
            if (_verbose_log_stream_buf.is_open()) {
                _verbose_log_stream_buf.close();
                _verbose_log.rdbuf(cout.rdbuf());
            }

            _verbose = false;
        }

        /// Return cout-like verbose log variable for printing out to file or screen
        inline ostream& GetVerboseLog() {
            return _verbose_log;
        }

        /// Set blurring flag
        inline void SetBlurring(bool flag_blurring) {
            _blurring = flag_blurring;
        }
        
        
        inline void SetGlobalNCC(double global_NCC_threshold) {
            _global_NCC_threshold = global_NCC_threshold;
        }
        
        inline void SetGlobalJAC(double global_JAC_threshold) {
            _global_JAC_threshold = global_JAC_threshold;
        }
        
        inline void SetLocalSSIM(double local_SSIM_threshold) {
            _local_SSIM_threshold = local_SSIM_threshold;
        }
        
        

        /// Set structural flag
        inline void SetStructural(bool flag_structural) {
            _structural = flag_structural;
        }

        /// Enable adaptive regularisation
        inline void UseAdaptiveRegularisation() {
            _adaptive = true;
        }

        /// Set masked stacks flag
        inline void SetMaskedStacks() {
            _masked_stacks = true;
        }

        inline void SetFFDGlobalOnly() {
            _ffd_global_only = true; 
        }

        inline void SetFFDGlobalNCC() {
            _ffd_global_ncc = true; 
        }
        
        
        inline void SetMCRecon(int number_of_channels) {
            _multiple_channels_flag = true;
            _number_of_channels = number_of_channels;
        }
        
        
        inline void SetNoMaskingBackground() {
            _no_masking_background = true;
        }

        inline void SetNoOffsetRegistration() {
            _no_offset_registration = true;
        }

        
        /// Set sigma flag
        inline void SetSigma(double sigma) {
            _sigma_bias = sigma;
        }

        /// Set gestational age (to compute expected brain volume)
        inline void SetGA(double ga) {
            _GA = ga;
        }

        /// Set n packages flag
        inline void SetNPackages(const Array<int>& N) {
            _n_packages = N;
        }

        /// Use faster lower quality reconstruction
        inline void SpeedupOn() {
            _quality_factor = 1;
        }

        /// Use slower better quality reconstruction
        inline void SpeedupOff() {
            _quality_factor = 2;
        }

        /// Enable global bias correction
        inline void GlobalBiasCorrectionOn() {
            _global_bias_correction = true;
        }

        /// Disable global bias correction
        inline void GlobalBiasCorrectionOff() {
            _global_bias_correction = false;
        }

        /// Set lower threshold for low intensity cutoff during bias estimation
        inline void SetLowIntensityCutoff(double cutoff) {
            if (cutoff > 1) cutoff = 1;
            else if (cutoff < 0) cutoff = 0;
            _low_intensity_cutoff = cutoff;
        }

        /// Enable robust slices mode
        inline void ExcludeWholeSlicesOnly() {
            _robust_slices_only = true;
            cout << "Exclude only whole slices." << endl;
        }

        /// Set smoothing parameters
        inline void SetSmoothingParameters(double delta, double lambda) {
            _delta = delta;
            _lambda = lambda * delta * delta;
            _alpha = 0.05 / lambda;
            if (_alpha > 1)
                _alpha = 1;

            if (_verbose)
                _verbose_log << "delta = " << _delta << " lambda = " << lambda << " alpha = " << _alpha << endl;
        }

        /// Set slices which need to be excluded by default
        inline void SetForceExcludedSlices(const Array<int>& force_excluded) {
            _force_excluded = force_excluded;
        }

        /// Set NMI bins
        inline void SetNMIBins(int nmi_bins) {
            _nmi_bins = nmi_bins;
        }

        /// Set reconstruction type to 3D
        inline void Set3DRecon() {
            _recon_type = _3D;
        }

        /// Set reconstruction type to 1D
        inline void Set1DRecon() {
            _recon_type = _1D;
        }

        /// Set reconstruction type to interpolation
        inline void SetInterpolationRecon() {
            _recon_type = _interpolate;
        }

        /// Set slices per dyn
        inline void SetSlicesPerDyn(int slices) {
            _slicePerDyn = slices;
        }

        /// Enable multiband mode
        inline void SetMultiband(bool withMB) {
            _withMB = withMB;
        }

        /// Return transformation size
        inline int GetNumberOfTransformations() {
            return _transformations.size();
        }

        /// Return specified transformation
        inline RigidTransformation GetTransformation(int n) {
            if (_transformations.size() <= n)
                throw runtime_error("GetTransformation: too large n = " + to_string(n));
            return _transformations[n];
        }

        /// Set BG flag
        inline void SetBG(bool flag_bg) {
            _bg_flag = flag_bg;
        }

        /// Set CP spacing
        inline void SetCP(int cp_spacing_value, int step) {
            if (cp_spacing_value > 4) {
                _cp_spacing.clear();
                for (int i=0; i<5; i++) {
                    if (cp_spacing_value > 4) {
                        _cp_spacing.push_back(cp_spacing_value);
                        cp_spacing_value -= step;
                    } else {
                        _cp_spacing.push_back(5);
                    }
                }
            }
        }
        
        inline void SetGlobalCP(int cp_spacing_value) {
            
            _global_cp_spacing = cp_spacing_value;
            
        }
        

        /// Return rigid transformations
        inline void GetTransformations(Array<RigidTransformation>& transformations) {
            transformations = _transformations;
        }

        /// Set rigid transformations
        inline void SetTransformations(const Array<RigidTransformation>& transformations) {
            _transformations = transformations;
        }

        /// Return slices
        inline void GetSlices(Array<RealImage>& slices) {
            slices = _slices;
        }

        /// Set slices
        inline void SetSlices(const Array<RealImage>& slices) {
            _slices = slices;
        }

        /// Mask stacks with respect to the reconstruction mask and given transformations
        inline void MaskStacks(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations) {
            if (!_have_mask)
                cerr << "Could not mask slices because no mask has been set." << endl;
            else
                Utility::MaskStacks(stacks, stack_transformations, _mask);
        }

    };  // end of Reconstruction class definition

} // namespace svrtk
