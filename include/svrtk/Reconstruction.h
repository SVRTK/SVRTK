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

// MIRTK
#include "mirtk/Common.h"
#include "mirtk/Options.h"
#include "mirtk/Array.h"
#include "mirtk/Point.h"
#include "mirtk/GenericImage.h"
#include "mirtk/GaussianBlurring.h"
#include "mirtk/Resampling.h"
#include "mirtk/ResamplingWithPadding.h"
#include "mirtk/LinearInterpolateImageFunction.hxx"
#include "mirtk/GaussianBlurringWithPadding.h"
#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageTransformation.h"
#include "mirtk/MultiLevelFreeFormTransformation.h"
#include "mirtk/FreeFormTransformation.h"
#include "mirtk/LinearFreeFormTransformation3D.h"

// SVRTK
#include "svrtk/MeanShift.h"
#include "svrtk/NLDenoising.h"

// C++ Standard
#include <set>
#include <pthread.h>
#include <thread>

using namespace mirtk;

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
        class MStep;
        class EStep;
        class Bias;
        class Scale;
        class NormaliseBias;
        class SimulateSlices;
        class Average;
        class AdaptiveRegularization1;
        class AdaptiveRegularization2;
    }

    struct POINT3D {
        short x;
        short y;
        short z;
        double value;
    };

    enum RECON_TYPE { _3D, _1D, _interpolate };

    typedef Array<POINT3D> VOXELCOEFFS;
    typedef Array<Array<VOXELCOEFFS>> SLICECOEFFS;

    class Reconstruction {
    protected:
        // Reconstruction type
        RECON_TYPE _recon_type;

        // Structures to store the matrix of transformation between volume and slices
        Array<SLICECOEFFS> _volcoeffs;
        Array<SLICECOEFFS> _volcoeffsSF;

        // flags
        int _slicePerDyn;
        bool _ffd;
        bool _blurring;
        bool _structural;
        bool _ncc_reg;
        bool _template_flag;
        bool _no_sr;
        bool _reg_log;
        bool _masked_stacks;

        double _global_NCC_threshold;
        Array<int> _n_packages;

        ImageAttributes _attr_reconstructed;

        Array<Matrix> _offset_matrices;
        bool _saved_slices;
        Array<int> _zero_slices;

        int _cp_spacing;
        bool _filtered_cmp_flag;
        bool _bg_flag;

        // Slices
        Array<RealImage> _slices;
        Array<RealImage> _slicesRwithMB;
        Array<RealImage> _simulated_slices;
        Array<RealImage> _simulated_weights;
        Array<RealImage> _simulated_inside;

        Array<ImageAttributes> _slice_attributes;
        Array<RealImage> _slice_dif;

        Array<GreyImage> _grey_slices;
        GreyImage _grey_reconstructed;

        Array<int> _reg_slice_weight;
        Array<int> _package_index;
        Array<int> _slice_pos;

        // Transformations
        Array<RigidTransformation> _transformations;
        Array<RigidTransformation> _previous_transformations;
        Array<RigidTransformation> _transformationsRwithMB;
        Array<MultiLevelFreeFormTransformation> _mffd_transformations;

        // Indicator whether slice has an overlap with volumetric mask
        Array<bool> _slice_inside;
        Array<bool> _slice_insideSF;


        // Flag to say whether the template volume has been created
        bool _template_created;
        // Flag to say whether we have a mask
        bool _have_mask;

        // Volumes
        RealImage _reconstructed;
        RealImage _mask;
        RealImage _target;
        RealImage _brain_probability;
        Array<RealImage> _probability_maps;

        // Weights for Gaussian reconstruction
        RealImage _volume_weights;
        RealImage _volume_weightsSF;

        // Weights for regularization
        RealImage _confidence_map;

        // EM algorithm
        // Variance for inlier voxel errors
        double _sigma;
        // Proportion of inlier voxels
        double _mix;
        // Uniform distribution for outlier voxels
        double _m;
        // Mean for inlier slice errors
        double _mean_s;
        // Variance for inlier slice errors
        double _sigma_s;
        // Mean for outlier slice errors
        double _mean_s2;
        // Variance for outlier slice errors
        double _sigma_s2;
        // Proportion of inlier slices
        double _mix_s;
        // Step size for likelihood calculation
        double _step;
        // Voxel posteriors
        Array<RealImage> _weights;
        // Slice posteriors
        Array<double> _slice_weight;

        // NMI registration bins
        int _nmi_bins;

        //Bias field
        // Variance for bias field
        double _sigma_bias;
        // Slice-dependent bias fields
        Array<RealImage> _bias;

        // Slice-dependent scales
        Array<double> _scale;

        // Quality factor - higher means slower and better
        double _quality_factor;
        // Intensity min and max
        double _max_intensity;
        double _min_intensity;

        //Gradient descent and regularisation parameters
        // Step for gradient descent
        double _alpha;
        // Determine what is en edge in edge-preserving smoothing
        double _delta;
        // Amount of smoothing
        double _lambda;
        // Average voxel wights to modulate parameter alpha
        double _average_volume_weight;
        double _average_volume_weightSF;

        //global bias field correction
        // global bias correction flag
        bool _global_bias_correction;
        // low intensity cutoff for bias field estimation
        double _low_intensity_cutoff;

        //to restore original signal intensity of the MRI slices
        Array<double> _stack_factor;
        double _average_value;
        Array<int> _stack_index;

        // GF 200416 Handling slice acquisition order
        // Array containing slice acquisition order
        Array<int> _z_slice_order;
        Array<int> _t_slice_order;
        Array<int> _slice_timing;

        //forced excluded slices
        Array<int> _force_excluded;

        //slices identify as too small to be used
        Array<int> _small_slices;

        // use adaptive or non-adaptive regularisation (default:false)
        bool _adaptive;

        //utility
        // Debug mode
        bool _debug;

        // Verbose mode
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

        // Gestational age (to compute expected brain volume)
        double _GA;

        // Read Transformations common function
        void ReadTransformations(const char *folder, size_t file_count, Array<RigidTransformation>& transformations);

        // Scale volume common function
        void ScaleVolume(RealImage& reconstructed);

    public:
        // Constructor
        Reconstruction();
        // Destructor
        ~Reconstruction();

        int _number_of_slices_org;
        double _average_thickness_org;

        // Create zero image as a template for reconstructed volume
        double CreateTemplate(const RealImage& stack, double resolution = 0);

        double CreateTemplateAniso(const RealImage& stack);

        Array<int> _excluded_entirely;

        // Remember volumetric mask and smooth it if necessary
        void SetMask(RealImage *mask, double sigma, double threshold = 0.5);

        void SliceDifference();

        void SaveSliceInfo(int current_iteration);

        // Center stacks
        void CenterStacks(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber);

        //Create average image from the stacks and volumetric transformations
        RealImage CreateAverage(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations);

        // Transform and resample mask to the space of the image
        void TransformMask(const RealImage& image, RealImage& mask, const RigidTransformation& transformation);

        // Rescale image ignoring negative values
        void Rescale(RealImage& img, double max);

        // Calculate initial registrations
        void StackRegistrations(const Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber);

        // Create slices from the stacks and slice-dependent transformations from
        // stack transformations
        void CreateSlicesAndTransformations(const Array<RealImage> &stacks, const Array<RigidTransformation> &stack_transformations,
            const Array<double> &thickness, const Array<RealImage> &probability_maps);

        void SetSlicesAndTransformations(const Array<RealImage>& slices,
            const Array<RigidTransformation>& slice_transformations, const Array<int>& stack_ids, const Array<double>& thickness);

        void ResetSlices(Array<RealImage>& stacks, Array<double>& thickness);

        // Update slices if stacks have changed
        void UpdateSlices(Array<RealImage>& stacks, Array<double>& thickness);

        void SaveProbabilityMap(int i);

        // Invert all stack transformation
        void InvertStackTransformations(Array<RigidTransformation>& stack_transformations);

        // Match stack intensities
        void MatchStackIntensities(Array<RealImage>& stacks,
            const Array<RigidTransformation>& stack_transformations, double averageValue, bool together = false);

        // Match stack intensities with masking
        void MatchStackIntensitiesWithMasking(Array<RealImage>& stacks,
            const Array<RigidTransformation>& stack_transformations, double averageValue, bool together = false);

        // If template image has been masked instead of creating the mask in separate
        // file, this function can be used to create mask from the template image
        RealImage CreateMask(RealImage image);

        void CreateMaskFromBlackBackground(const Array<RealImage>& stacks,
            Array<RigidTransformation> stack_transformations, double smooth_mask);

        // Mask all stacks
        void MaskStacks(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations);

        // Mask all slices
        void MaskSlices();

        // Set reconstructed image
        void SetTemplate(RealImage tempImage);

        // Calculate transformation matrix between slices and voxels
        void CoeffInit();
        void CoeffInitSF(int begin, int end);

        // Reconstruction using weighted Gaussian PSF
        void GaussianReconstruction();
        void GaussianReconstructionSF(const Array<RealImage>& stacks);

        // NLM filtering
        void NLMFiltering(Array<RealImage>& stacks);

        // NCC stats
        double VolumeNCC(RealImage& input_stack, RealImage template_stack, const RealImage& mask);

        // Initialise variables and parameters for EM
        void InitializeEM();

        // Initialise values of variables and parameters for EM
        void InitializeEMValues();

        // Initalize robust statistics
        void InitializeRobustStatistics();

        // Perform E-step
        void EStep();

        // Calculate slice-dependent scale
        void Scale();

        // Calculate slice-dependent bias fields
        void Bias();
        void NormaliseBias(int iter);

        // Superresolution
        void Superresolution(int iter);

        // Calculation of voxel-vise robust statistics
        void MStep(int iter);

        // Edge-preserving regularization
        void Regularization(int iter);

        // Edge-preserving regularization with confidence map
        void AdaptiveRegularization(int iter, const RealImage& original);

        // Slice to volume registrations
        void SliceToVolumeRegistration();
        void RemoteSliceToVolumeRegistration(int iter, string str_mirtk_path, string str_current_main_file_path, string str_current_exchange_file_path);

        // remote recon functions
        void SaveModelRemote(string str_current_exchange_file_path, int status_flag, int current_iteration);
        void LoadModelRemote(string str_current_exchange_file_path, int current_number_of_slices, double average_thickness, int current_iteration);
        void LoadResultsRemote(string str_current_exchange_file_path, int current_number_of_slices, int current_iteration);

        // Correct bias in the reconstructed volume
        void BiasCorrectVolume(const RealImage& original);

        // Mask the volume
        void MaskVolume();
        void MaskImage(RealImage& image, double padding = -1);

        // Save slices
        void SaveSlices();
        void SaveSlicesWithTiming();
        void SlicesInfo(const char *filename, const Array<string>& stack_filenames);

        // Save simulated slices
        void SaveSimulatedSlices();

        // Save weights
        void SaveWeights();
        void SaveRegistrationStep(const Array<RealImage>& stacks, const int step);

        // Save transformations
        void SaveTransformations();
        void SaveTransformationsWithTiming(const int iter = -1);

        // Save confidence map
        void SaveConfidenceMap();

        // Save bias field
        void SaveBiasFields();

        void ReconQualityReport(double& out_ncc, double& out_nrmse, double& average_weight, double& ratio_excluded);

        RealImage ThresholdNormalisedMask(RealImage image, double threshold);

        double ComputeNCC(const RealImage& slice_1, const RealImage& slice_2, const double threshold = 0.01, double *count = nullptr);

        void GlobalStackStats(RealImage template_stack, const RealImage& template_mask, const Array<RealImage>& stacks,
            const Array<RealImage>& masks, double& average_ncc, double& average_volume,
            Array<RigidTransformation>& current_stack_tranformations);

        void StackStats(RealImage input_stack, const RealImage& mask, double& mask_volume, double& slice_ncc);

        void RunParallelGlobalStackStats(const Array<RealImage>& stacks, const Array<RealImage>& masks,
            Array<double>& all_global_ncc_array, Array<double>& all_global_volume_array);

        // evaluation based on the number of excluded slices
        void Evaluate(int iter, ostream& outstr = cout);

        // Read transformations
        void ReadTransformations(const char *folder);

        //To recover original scaling
        // Restore slice intensities to their original values
        void RestoreSliceIntensities();
        // Scale volume to match the slice intensities
        void ScaleVolume();

        // To compare how simulation from the reconstructed volume matches the original stacks
        void SimulateStacks(Array<RealImage>& stacks);

        void SimulateSlices();

        // Puts origin of the image into origin of world coordinates
        void ResetOrigin(GreyImage& image, RigidTransformation& transformation);
        void ResetOrigin(RealImage& image, RigidTransformation& transformation);

        // Packages to volume registrations
        void PackageToVolume(const Array<RealImage>& stacks, const Array<int>& pack_num,
            const Array<RigidTransformation>& stack_transformations);

        // Calculate Slice acquisition order
        void GetSliceAcquisitionOrder(const Array<RealImage>& stacks,
            const Array<int>& pack_num, const Array<int>& order, const int step, const int rewinder);

        // Split image in a flexible manner
        void flexibleSplitImage(const Array<RealImage>& stacks, Array<RealImage>& sliceStacks,
            const Array<int>& pack_num, const Array<int>& sliceNums, const Array<int>& order, const int step, const int rewinder);

        // Create Multiband replica for flexibleSplitImage
        void flexibleSplitImagewithMB(const Array<RealImage>& stacks, Array<RealImage>& sliceStacks,
            const Array<int>& pack_num, const Array<int>& sliceNums, const Array<int>& multiband,
            const Array<int>& order, const int step, const int rewinder);

        // Split images into packages
        void splitPackages(const Array<RealImage>& stacks, const Array<int>& pack_num,
            Array<RealImage>& packageStacks, const Array<int>& order, const int step, const int rewinder);

        // Create Multiband replica for splitPackages
        void splitPackageswithMB(const Array<RealImage>& stacks, const Array<int>& pack_num, Array<RealImage>& packageStacks,
            const Array<int>& multiband_vector, const Array<int>& order, const int step, const int rewinder);

        // Performs package registration
        void newPackageToVolume(const Array<RealImage>& stacks, const Array<int>& pack_num, const Array<int>& multiband,
            const Array<int>& order, const int step, const int rewinder, const int iter, const int steps);

        // Filter backgound
        void BackgroundFiltering(Array<RealImage>& stacks, const double fg_sigma, const double bg_sigma);

        // Splits stacks into packages
        void SplitImage(const RealImage& image, const int packages, Array<RealImage>& stacks);

        // Splits stacks into packages and each package into even and odd slices
        void SplitImageEvenOdd(const RealImage& image, const int packages, Array<RealImage>& stacks);

        // Splits image into top and bottom half roi according to z coordinate
        void HalfImage(const RealImage& image, Array<RealImage>& stacks);

        // Splits stacks into packages and each package into even and odd slices and top and bottom roi
        void SplitImageEvenOddHalf(const RealImage& image, const int packages, Array<RealImage>& stacks, const int iter = 1);

        // Crop image according to the given mask
        void CropImage(RealImage& image, const RealImage& mask);
        void CropImageIgnoreZ(RealImage& image, const RealImage& mask);

        // Run structure-based rejection of outliers
        void StructuralExclusion();

        // Evaluated reconstruction quality (errors per slices)
        double EvaluateReconQuality(int index);

        // Initialisation with stack transformations
        void InitialiseWithStackTransformations(const Array<RigidTransformation>& stack_transformations);

        // transformation to the reconstructed image space
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
        friend class Parallel::Bias;
        friend class Parallel::Scale;
        friend class Parallel::NormaliseBias;
        friend class Parallel::SimulateSlices;
        friend class Parallel::Average;
        friend class Parallel::AdaptiveRegularization1;
        friend class Parallel::AdaptiveRegularization2;

        ////////////////////////////////////////////////////////////////////////////////
        // Inline/template definitions
        ////////////////////////////////////////////////////////////////////////////////

        // Zero-mean Gaussian PDF
        inline double G(double x, double s) {
            return _step * exp(-x * x / (2 * s)) / (sqrt(6.28 * s));
        }

        // Uniform PDF
        inline double M(double m) {
            return m * _step;
        }

        inline const RealImage& GetReconstructed() {
            return _reconstructed;
        }

        inline void SetReconstructed(const RealImage& reconstructed) {
            _reconstructed = reconstructed;
            _template_created = true;
        }

        inline const RealImage& GetMask() {
            return _mask;
        }

        inline void PutMask(const RealImage& mask) {
            _mask = mask;
        }

        inline void SetFFD(bool flag_ffd) {
            _ffd = flag_ffd;
        }

        inline void SetNCC(bool flag_ncc) {
            _ncc_reg = flag_ncc;
        }

        inline void SetRegLog(bool flag_reg_log) {
            _reg_log = flag_reg_log;
        }

        inline void SetSR(bool flag_sr) {
            _no_sr = flag_sr;
        }

        inline void SetTemplateFlag(bool template_flag) {
            _template_flag = true;
        }

        // Whether to save intermediate results
        inline void DebugOn() {
            _debug = true;
        }

        inline void DebugOff() {
            _debug = false;
        }

        // Whether to print out results
        inline void VerboseOn(const string& log_file_name = "") {
            VerboseOff();

            if (!log_file_name.empty()) {
                _verbose_log_stream_buf.open(log_file_name, ofstream::out | ofstream::trunc);
                _verbose_log.rdbuf(_verbose_log_stream_buf.rdbuf());
            }

            _verbose = true;
        }

        inline void VerboseOff() {
            if (_verbose_log_stream_buf.is_open()) {
                _verbose_log_stream_buf.close();
                _verbose_log.rdbuf(cout.rdbuf());
            }

            _verbose = false;
        }

        inline ostream& GetVerboseLog() {
            return _verbose_log;
        }

        inline void SetBlurring(bool flag_blurring) {
            _blurring = flag_blurring;
        }

        inline void SetStructural(bool flag_structural) {
            _structural = flag_structural;
        }

        inline void UseAdaptiveRegularisation() {
            _adaptive = true;
        }

        inline void SetMaskedStacks() {
            _masked_stacks = true;
        }

        inline void SetSigma(double sigma) {
            _sigma_bias = sigma;
        }

        // Set gestational age (to compute expected brain volume)
        inline void SetGA(double ga) {
            _GA = ga;
        }

        inline void SetNPackages(const Array<int>& N) {
            _n_packages = N;
        }

        // Use faster lower quality reconstruction
        inline void SpeedupOn() {
            _quality_factor = 1;
        }

        // Use slower better quality reconstruction
        inline void SpeedupOff() {
            _quality_factor = 2;
        }

        inline void GlobalBiasCorrectionOn() {
            _global_bias_correction = true;
        }

        inline void GlobalBiasCorrectionOff() {
            _global_bias_correction = false;
        }

        // Set lower threshold for low intensity cutoff during bias estimation
        inline void SetLowIntensityCutoff(double cutoff) {
            if (cutoff > 1) cutoff = 1;
            else if (cutoff < 0) cutoff = 0;
            _low_intensity_cutoff = cutoff;
        }

        inline void ExcludeWholeSlicesOnly() {
            _robust_slices_only = true;
            cout << "Exclude only whole slices." << endl;
        }

        inline void SetSmoothingParameters(double delta, double lambda) {
            _delta = delta;
            _lambda = lambda * delta * delta;
            _alpha = 0.05 / lambda;
            if (_alpha > 1)
                _alpha = 1;

            if (_verbose)
                _verbose_log << "delta = " << _delta << " lambda = " << lambda << " alpha = " << _alpha << endl;
        }

        // Set slices which need to be excluded by default
        inline void SetForceExcludedSlices(const Array<int>& force_excluded) {
            _force_excluded = force_excluded;
        }

        inline void SetNMIBins(int nmi_bins) {
            _nmi_bins = nmi_bins;
        }

        inline void Set3DRecon() {
            _recon_type = _3D;
        }

        inline void Set1DRecon() {
            _recon_type = _1D;
        }

        inline void SetInterpolationRecon() {
            _recon_type = _interpolate;
        }

        inline void SetSlicesPerDyn(int slices) {
            _slicePerDyn = slices;
        }

        inline void SetMultiband(bool withMB) {
            _withMB = withMB;
        }

        inline int GetNumberOfTransformations() {
            return _transformations.size();
        }

        inline RigidTransformation GetTransformation(int n) {
            if (_transformations.size() <= n) {
                cerr << "GetTransformation: too large n = " << n << endl;
                exit(1);
            }
            return _transformations[n];
        }

        inline void SetBG(bool flag_bg) {
            _bg_flag = flag_bg;
        }

        inline void SetCP(int cp_spacing) {
            _cp_spacing = cp_spacing;
        }

        inline void GetTransformations(Array<RigidTransformation>& transformations) {
            transformations = _transformations;
        }

        inline void SetTransformations(const Array<RigidTransformation>& transformations) {
            _transformations = transformations;
        }

        inline void GetSlices(Array<RealImage>& slices) {
            slices = _slices;
        }

        inline void SetSlices(const Array<RealImage>& slices) {
            _slices = slices;
        }

    };  // end of Reconstruction class definition

} // namespace svrtk
