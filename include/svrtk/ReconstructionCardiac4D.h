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

using namespace mirtk;

namespace svrtk {

    // Forward declarations
    namespace Parallel {
        class CoeffInitCardiac4D;
        class SliceToVolumeRegistrationCardiac4D;
        class SimulateSlicesCardiac4D;
        class SimulateStacksCardiac4D;
        class NormaliseBiasCardiac4D;
        class SuperresolutionCardiac4D;
        class AdaptiveRegularization1Cardiac4D;
        class AdaptiveRegularization2Cardiac4D;
        class CalculateError;
        class CalculateCorrectedSlices;
    }

    class ReconstructionCardiac4D: public Reconstruction {
    protected:
        // Stacks
        Array<int> _loc_index;        // running index of all 2D slice locations
        Array<int> _stack_loc_index;  // index of 2D slice location in M2D stack
        Array<int> _stack_dyn_index;  // index of dynamic in M2D stack

        bool _no_ts;
        bool _no_sr;

        // Images
        Array<RealImage> _error;
        Array<RealImage> _corrected_slices;

        // Reconstructed 4D Cardiac Cine Images
        RealImage _reconstructed4D;
        // TODO: replace _reconstructed4D with _reconstructed and fix conflicts between irtkReconstruction and irtkReconstructionCardiac4D use of _reconstruction and_reconstruction4D

        // Reconstructed Cardiac Phases
        Array<double> _reconstructed_cardiac_phases;

        // Reconstructed R-R Interval
        double _reconstructed_rr_interval;

        // Reconstructed Temporal Resolution
        double _reconstructed_temporal_resolution;

        // Slice Acquisition Time
        Array<double> _slice_time;

        // Slice Temporal Resolution
        Array<double> _slice_dt;

        // Slice R-R interval
        Array<double> _slice_rr;

        // Slice Cardiac Phase
        Array<double> _slice_cardphase;

        // Slice Temporal Weight
        // access as: _slice_temporal_weight[iReconstructedCardiacPhase][iSlice]
        Array<Array<double>> _slice_temporal_weight;

        // Slice SVR Target Cardiac Phase
        Array<int> _slice_svr_card_index;

        // Displacement
        Array<double> _slice_displacement;
        Array<double> _slice_tx;
        Array<double> _slice_ty;
        Array<double> _slice_tz;
        Array<double> _slice_weighted_displacement;

        // Target Registration Error
        Array<RigidTransformation> _ref_transformations;
        Array<double> _slice_tre;

        // Force exclusion of stacks
        Array<int> _force_excluded_stacks;

        // Force exclusion of slice-locations
        Array<int> _force_excluded_locs;

        // Slice Excluded
        Array<bool> _slice_excluded;

        // Temporal Weighting Tukey Window Edge Percent
        // applies to sinc temporal weighting only
        static constexpr double _wintukeypct = 0.3;

        // Flag for Sinc Temporal Weighting
        // use Gaussian weighting if false
        bool _is_temporalpsf_gauss;

        // Common calculate function
        double Calculate(const RigidTransformation& drift, double (ReconstructionCardiac4D::*)());

        // Calculate Temporal Weight
        double CalculateTemporalWeight(double cardphase0, double cardphase, double dt, double rr, double alpha);

        // Common save function
        void Save(const Array<RealImage>& save_stacks, const char *message, const char *filename_prefix);
        void Save(const Array<RealImage>& source, const Array<RealImage>& stacks, int iter, int rec_iter, const char *message, const char *filename_prefix);

        // Angular difference between output cardiac phase (cardphase0) and slice cardiac phase (cardphase)
        inline double CalculateAngularDifference(double cardphase0, double cardphase) {
            const double angdiff = (cardphase - cardphase0) - (2 * PI) * floor((cardphase - cardphase0) / (2 * PI));
            return angdiff <= PI ? angdiff : -(2 * PI - angdiff);
        }

        // Sinc Function
        inline double sinc(double x) {
            return x == 0 ? 1 : sin(x) / x;
        }

        // Tukey Window Function
        inline double wintukey(double angdiff, double alpha) {
            // angdiff = angular difference (-PI to +PI)
            // alpha   = amount of window with tapered cosine edges (0 to 1)
            if (fabs(angdiff) > PI * (1 - alpha))
                return (1 + cos((fabs(angdiff) - PI * (1 - alpha)) / alpha)) / 2;
            return 1;
        }

    public:
        /// ReconstructionCardiac4D Constructor
        ReconstructionCardiac4D() : Reconstruction() {
            _recon_type = _3D;
            _no_sr = false;
            _no_ts = false;
        }

        /// ReconstructionCardiac4D Destructor
        ~ReconstructionCardiac4D() {}

        /// Get slice-location transformations
        void ReadSliceTransformations(const char *folder);

        /// Determine resolution of volume to reconstruct
        double GetReconstructedResolutionFromTemplateStack(const RealImage& stack);

        /**
         * @brief Initialise reconstructed volume from static mask.
         * @param mask
         * @param resolution
         */
        void CreateTemplateCardiac4DFromStaticMask(const RealImage& mask, double resolution);

        /**
         * @brief Match stack intensities with respect to the masked ROI.
         * @param stacks
         * @param stack_transformations
         * @param averageValue
         * @param together
         */
        void MatchStackIntensitiesWithMasking(Array<RealImage>& stacks,
            const Array<RigidTransformation>& stack_transformations,
            double averageValue,
            bool together = false);

        /**
         * @brief Create slices from the stacks and slice-dependent transformations from stack transformations
         * @param stacks
         * @param stack_transformations
         * @param thickness
         * @param probability_maps
         */
        void CreateSlicesAndTransformationsCardiac4D(const Array<RealImage>& stacks,
            const Array<RigidTransformation>& stack_transformations,
            const Array<double>& thickness,
            const Array<RealImage>& probability_maps = Array<RealImage>());

        /// Calculate Slice Temporal Weights
        void CalculateSliceTemporalWeights();

        /// Calculate transformation matrix between slices and voxels
        void CoeffInitCardiac4D();

        /// PSF-weighted reconstruction
        void GaussianReconstructionCardiac4D();

        /// Calculate corrected slices
        void CalculateCorrectedSlices();

        /// Simulate slices
        void SimulateSlicesCardiac4D();

        /// Simulate stacks
        void SimulateStacksCardiac4D();

        /// Normalise bias
        void NormaliseBiasCardiac4D(int iter, int rec_inter);

        /// Calculate target cardiac phase in reconstructed volume for slice-to-volume registration.
        void CalculateSliceToVolumeTargetCardiacPhase();
        /// Slice-to-volume registration
        void SliceToVolumeRegistrationCardiac4D();
        /**
         * @brief Remote slice-to-volume registration.
         * @param iter
         * @param str_mirtk_path
         * @param str_current_exchange_file_path
         */
        void RemoteSliceToVolumeRegistrationCardiac4D(int iter, const string& str_mirtk_path, const string& str_current_exchange_file_path);

        /**
         * @brief Volume-to-volume Registration
         * @param target
         * @param source
         * @param rigidTransf
         */
        void VolumeToVolumeRegistration(const GreyImage& target, const GreyImage& source, RigidTransformation& rigidTransf);

        /// Calculate displacement
        double CalculateDisplacement();
        /// Calculate displacement relative to alignment
        inline double CalculateDisplacement(const RigidTransformation& drift) { return Calculate(drift, &ReconstructionCardiac4D::CalculateDisplacement); }
        /// Calculate weighted displacement
        double CalculateWeightedDisplacement();
        /// Calculate weighted displacement relative to alignment
        inline double CalculateWeightedDisplacement(const RigidTransformation& drift) { return Calculate(drift, &ReconstructionCardiac4D::CalculateWeightedDisplacement); }

        /// Initialise target registration error (TRE)
        inline void InitTRE() { _slice_tre = Array<double>(_slices.size(), -1); }
        /// Calculate target registration error (TRE)
        double CalculateTRE();
        /// Calculate target registration error (TRE) relative to alignment
        inline double CalculateTRE(const RigidTransformation& drift) { return Calculate(drift, &ReconstructionCardiac4D::CalculateTRE); }

        /**
         * @brief Smooth transformations.
         * @param sigma
         * @param niter
         * @param use_slice_inside
         */
        void SmoothTransformations(double sigma, int niter = 10, bool use_slice_inside = false);

        /// Scale transformations
        void ScaleTransformations(double scale);

        /// Apply static mask to reconstructed 4d volume
        inline void StaticMaskReconstructedVolume4D() { StaticMaskVolume4D(_reconstructed4D, _mask, -1); }

        /// Calculate error
        void CalculateError();

        /// Run superresolution reconstruction step
        void SuperresolutionCardiac4D(int iter);

        /// Edge-preserving regularization with confidence map
        void AdaptiveRegularizationCardiac4D(int iter, const RealImage& original);

        /// Read reference transformations
        void ReadRefTransformations(const char *folder);

        /// Calculate entropy
        double CalculateEntropy();

        /// Save slices
        void SaveSlices(const Array<RealImage>& stacks);

        /// Save slice info
        void SaveSliceInfoCardiac4D(const char *filename, const Array<string>& stack_filenames);

        friend class Parallel::CoeffInitCardiac4D;
        friend class Parallel::SliceToVolumeRegistrationCardiac4D;
        friend class Parallel::SimulateSlicesCardiac4D;
        friend class Parallel::SimulateStacksCardiac4D;
        friend class Parallel::NormaliseBiasCardiac4D;
        friend class Parallel::SuperresolutionCardiac4D;
        friend class Parallel::AdaptiveRegularization1Cardiac4D;
        friend class Parallel::AdaptiveRegularization2Cardiac4D;
        friend class Parallel::CalculateError;
        friend class Parallel::CalculateCorrectedSlices;

        ////////////////////////////////////////////////////////////////////////////////
        // Inline/template definitions
        ////////////////////////////////////////////////////////////////////////////////

        /// Set temporal weight gaussian
        inline void SetTemporalWeightGaussian() {
            _is_temporalpsf_gauss = true;
            cout << "Temporal PSF = Gaussian()" << endl;
        }

        /// Set temporal weight sinc
        inline void SetTemporalWeightSinc() {
            _is_temporalpsf_gauss = false;
            cout << "Temporal PSF = sinc() * Tukey_window()" << endl;
        }

        /// Return reconstructed 4d volume
        inline const RealImage& GetReconstructedCardiac4D() {
            return _reconstructed4D;
        }

        /// Set reconstructed 4d volume
        inline void SetReconstructedCardiac4D(const RealImage& reconstructed4D) {
            _reconstructed4D = reconstructed4D;

            ImageAttributes attr = reconstructed4D.Attributes();
            attr._t = 1;

            _reconstructed = RealImage(attr);

            _mask = RealImage(attr);
            _mask = 1;

            _template_created = true;
        }

        /// Return volume weights
        inline const RealImage& GetVolumeWeights() {
            return _volume_weights;
        }

        /// Set force excluded stacks
        inline void SetForceExcludedStacks(const Array<int>& force_excluded_stacks) {
            _force_excluded_stacks = force_excluded_stacks;
        }

        /// Set force excluded locations
        inline void SetForceExcludedLocs(const Array<int>& force_excluded_locs) {
            _force_excluded_locs = force_excluded_locs;
        }

        /// Set slice R-R interval
        inline void SetSliceRRInterval(const Array<double>& rr) {
            _slice_rr = rr;
        }

        /// Set slice R-R interval
        inline void SetSliceRRInterval(double rr) {
            _slice_rr = Array<double>(_slices.size(), rr);
        }

        /// Set location R-R interval
        inline void SetLocRRInterval(const Array<double>& rr) {
            Array<double> slice_rr;
            for (size_t i = 0; i < _slices.size(); i++)
                slice_rr.push_back(rr[_loc_index[i]]);
            _slice_rr = move(slice_rr);
        }

        /// Set slice cardiac phase
        inline void SetSliceCardiacPhase(const Array<double>& cardiacphases) {
            _slice_cardphase = cardiacphases;
        }

        /// Set slice cardiac phase
        inline void SetSliceCardiacPhase() {
            _slice_cardphase = Array<double>(_slices.size());
        }

        /// Set reconstructed cardiac phase
        inline void SetReconstructedCardiacPhase(const Array<double>& cardiacphases) {
            _reconstructed_cardiac_phases = cardiacphases;
        }

        /// Set reconstructed R-R interval
        inline void SetReconstructedRRInterval(double rrinterval) {
            _reconstructed_rr_interval = rrinterval;
            if (_verbose)
                _verbose_log << "Reconstructed R-R interval = " << _reconstructed_rr_interval << " s." << endl;
        }

        /// Set reconstructed temporal resolution
        inline void SetReconstructedTemporalResolution(double temporalresolution) {
            _reconstructed_temporal_resolution = temporalresolution;
            if (_verbose)
                _verbose_log << "Reconstructed temporal resolution = " << _reconstructed_temporal_resolution << " s." << endl;
        }

        /// Initialise stack factor
        inline void InitStackFactor(const Array<RealImage>& stacks) {
            _stack_factor = Array<double>(stacks.size(), 1);
        }

        /// Reset slices and transformations
        void ResetSlicesAndTransformationsCardiac4D() {
            _slice_time.clear();
            _slice_dt.clear();
            _slices.clear();
            _simulated_slices.clear();
            _simulated_weights.clear();
            _simulated_inside.clear();
            _stack_index.clear();
            _loc_index.clear();
            _stack_loc_index.clear();
            _stack_dyn_index.clear();
            _transformations.clear();
            _slice_excluded.clear();
            _probability_maps.clear();
        }

        /// Initialise corrected slices
        inline void InitCorrectedSlices() {
            _corrected_slices = _slices;
        }

        /// Initialise error
        inline void InitError() {
            _error = _slices;
        }

        /// Initialise slice temporal weights
        inline void InitSliceTemporalWeights() {
            ClearAndResize(_slice_temporal_weight, _reconstructed_cardiac_phases.size(), Array<double>(_slices.size()));
        }

        /// Scale volume to match the slice intensities
        inline void ScaleVolumeCardiac4D() {
            if (_verbose)
                _verbose_log << "Scaling volume: ";

            ScaleVolume(_reconstructed4D);
        }

        /// Save bias fields
        inline void SaveBiasFields(const Array<RealImage>& stacks, int iter = -1, int rec_iter = -1) {
            Save(_bias, stacks, iter, rec_iter, "SaveBiasFields as stacks", "bias");
        }

        /// Save simulated slices
        inline void SaveSimulatedSlices() {
            Save(_simulated_slices, "Saving simulated images", "simimage");
        }

        /// Save simulated slices
        inline void SaveSimulatedSlices(const Array<RealImage>& stacks, int iter = -1, int rec_iter = -1) {
            Save(_simulated_slices, stacks, iter, rec_iter, "Saving simulated images as stacks", "simstack");
        }

        /// Save simulated slices
        void SaveSimulatedSlices(const Array<RealImage>& stacks, int stack_no) {
            RealImage simstack(stacks[stack_no].Attributes());

            for (size_t inputIndex = 0; inputIndex < _simulated_slices.size(); inputIndex++)
                for (int i = 0; i < _simulated_slices[inputIndex].GetX(); i++)
                    for (int j = 0; j < _simulated_slices[inputIndex].GetY(); j++)
                        if (stack_no == _stack_index[inputIndex])
                            simstack(i, j, _stack_loc_index[inputIndex], _stack_dyn_index[inputIndex]) = _simulated_slices[inputIndex](i, j, 0);

            simstack.Write((boost::format("simstack%03i.nii.gz") % stack_no).str().c_str());
        }

        /// Save simulated weights
        inline void SaveSimulatedWeights() {
            Save(_simulated_weights, "Saving simulated weights", "simweight");
        }

        /// Save simulated weights
        inline void SaveSimulatedWeights(const Array<RealImage>& stacks, int iter = -1, int rec_iter = -1) {
            Save(_simulated_weights, stacks, iter, rec_iter, "Saving simulated weights as stacks", "simweightstack");
        }

        /// Save slices
        inline void SaveSlices() {
            Save(_slices, "Saving slices", "image");
        }

        /// Save corrected slices
        inline void SaveCorrectedSlices() {
            Save(_corrected_slices, "Saving corrected images", "correctedimage");
        }

        /// Save corrected slices
        inline void SaveCorrectedSlices(const Array<RealImage>& stacks, int iter = -1, int rec_iter = -1) {
            Save(_corrected_slices, stacks, iter, rec_iter, "Saving error images as stacks", "correctedstack");
        }

        /// Save error
        inline void SaveError() {
            Save(_error, "Saving error images", "errorimage");
        }

        /// Save error
        inline void SaveError(const Array<RealImage>& stacks, int iter = -1, int rec_iter = -1) {
            Save(_error, stacks, iter, rec_iter, "Saving error images as stacks", "errorstack");
        }

        // Save weights
        inline void SaveWeights(const Array<RealImage>& stacks, int iter = -1, int rec_iter = -1) {
            Save(_weights, stacks, iter, rec_iter, "Saving weights as stacks", "weight");
        }

    };  // end of ReconstructionCardiac4D class definition

} // namespace svrtk
