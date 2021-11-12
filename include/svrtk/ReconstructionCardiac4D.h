/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2018-2020 King's College London
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

    class ReconstructionCardiac4D : public Reconstruction
    {
       
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
        

        // PI
        const double PI = 3.14159265358979323846;
        
        // Reconstructed 4D Cardiac Cine Images
        RealImage _reconstructed4D;
        // TODO: replace _reconstructed4D with _reconstructed and fix conflicts between irtkReconstruction and irtkReconstructionCardiac4D use of _reconstruction and_reconstruction4D
                
        // Reconstructed Cardiac Phases
        Array<double> _reconstructed_cardiac_phases;
        
        // Reconstructe R-R Interval
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
        Array< Array<double> > _slice_temporal_weight;
        
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
        double _wintukeypct = 0.3;
        
        // Flag for Sinc Temporal Weighting
        // use Gaussian weighting if false
        bool _is_temporalpsf_gauss;
        

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
        // Constructor
        ReconstructionCardiac4D() : Reconstruction() {
            _recon_type = _3D;
            _no_sr = false;
            _no_ts = false;
        }

        // Destructor
        ~ReconstructionCardiac4D() {}

        // Get Slice-Location transformations
        void ReadSliceTransformation(const char *folder);

        // Set Reconstructed Volume Spatial Resolution
        double GetReconstructedResolutionFromTemplateStack( RealImage stack );
        
        // Initialise Reconstructed Volume from Static Mask
        void CreateTemplateCardiac4DFromStaticMask( RealImage mask, double resolution );
        
        ///Match stack intensities with masking
        void MatchStackIntensitiesWithMasking (Array<RealImage>& stacks,
                                               Array<RigidTransformation>& stack_transformations,
                                               double averageValue,
                                               bool together=false);
        
        // Create slices from the stacks and
        // slice-dependent transformations from stack transformations
        void CreateSlicesAndTransformationsCardiac4D( Array<RealImage>& stacks,
                                                     Array<RigidTransformation>& stack_transformations,
                                                     Array<double>& thickness,
                                                     const Array<RealImage> &probability_maps=Array<RealImage>() );
        
        // Calculate Slice Temporal Weights
        void CalculateSliceTemporalWeights();

        // Calculate Transformation Matrix Between Slices and Voxels
        void CoeffInitCardiac4D();
        
        // PSF-Weighted Reconstruction
        void GaussianReconstructionCardiac4D();

        // Calculate Corrected Slices
        void CalculateCorrectedSlices();
        
        // Simulate Slices
        void SimulateSlicesCardiac4D();
        
        // Simulate stacks
        void SimulateStacksCardiac4D(Array<bool> stack_excluded);
        
        // Normalise Bias
        void NormaliseBiasCardiac4D(int iter, int rec_inter);
        
        // Slice-to-Volume Registration
        void CalculateSliceToVolumeTargetCardiacPhase();
        void SliceToVolumeRegistrationCardiac4D();
        void RemoteSliceToVolumeRegistrationCardiac4D(int iter, string str_mirtk_path, string str_current_main_file_path, string str_current_exchange_file_path);
        
        // Volume-to-Volume Registration
        void VolumeToVolumeRegistration(GreyImage target, GreyImage source, RigidTransformation& rigidTransf);
        
        // Calculate Displacement
        double CalculateDisplacement();
        double CalculateDisplacement(RigidTransformation drift);
        double CalculateWeightedDisplacement();
        double CalculateWeightedDisplacement(RigidTransformation drift);
        
        // Calculate Target Registration Error (TRE)
        inline void InitTRE() { _slice_tre = Array<double>(_slices.size(), -1); }
        double CalculateTRE();
        double CalculateTRE(RigidTransformation drift);
        
        // Smooth Transformationss
        void SmoothTransformations(double sigma, int niter=10, bool use_slice_inside=false);
        
        // Scale Transformationss
        void ScaleTransformations(double scale);
        
        // Apply Static Mask to Reconstructed 4D Volume
        inline void StaticMaskReconstructedVolume4D() { StaticMaskVolume4D(_reconstructed4D, -1); }

        // Apply Static Mask 4D Volume
        RealImage StaticMaskVolume4D(RealImage volume, double padding);
        
        // Calculate Error
        void CalculateError();
        
        // Superresolution
        void SuperresolutionCardiac4D( int iter );
        
        /// Edge-Preserving Regularization with Confidence Map
        void AdaptiveRegularizationCardiac4D(int iter, RealImage& original);
        
        /// Read Reference Transformations
        void ReadRefTransformations(const char *folder);

        /// Calculate Entropy
        double CalculateEntropy();
        
        ///Save transformations
        void SaveTransformations();
        
        // Save Bias Fields
        void SaveBiasFields();
        void SaveBiasFields(Array<RealImage>& stacks);
        void SaveBiasFields(Array<RealImage>& stacks, int iter, int rec_iter);
        
        // Save Slices
        void SaveSimulatedSlices();
        void SaveSimulatedSlices(Array<RealImage>& stacks);
        void SaveSimulatedSlices(Array<RealImage>& stacks, int iter, int rec_iter);
        void SaveSimulatedSlices(Array<RealImage>& stacks, int stack_no);
        
        // Save Simulated Weights
        void SaveSimulatedWeights();
        void SaveSimulatedWeights(Array<RealImage>& stacks);
        void SaveSimulatedWeights(Array<RealImage>& stacks, int iter, int rec_iter);
        
        // Save Slices
        void SaveSlices();
        void SaveSlices(Array<RealImage>& stacks);
        
        // Save Corrected Slices
        void SaveCorrectedSlices();
        void SaveCorrectedSlices(Array<RealImage>& stacks);
        void SaveCorrectedSlices(Array<RealImage>& stacks, int iter, int rec_iter);
        
        // Save Error
        void SaveError();
        void SaveError(Array<RealImage>& stacks);
        void SaveError(Array<RealImage>& stacks, int iter, int rec_iter);
        
        // Save Weights
        void SaveWeights();
        void SaveWeights(Array<RealImage>& stacks);
        void SaveWeights(Array<RealImage>& stacks, int iter, int rec_iter);
        
        // Slice Info
        void SlicesInfoCardiac4D( const char* filename, Array<string> &stack_filenames );
        
        // Access to Parallel Processing Classes
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

        inline void SetTemporalWeightGaussian() {
            _is_temporalpsf_gauss = true;
            cout << "Temporal PSF = Gaussian()" << endl;
        }

        inline void SetTemporalWeightSinc() {
            _is_temporalpsf_gauss = false;
            cout << "Temporal PSF = sinc() * Tukey_window()" << endl;
        }

        inline const RealImage& GetReconstructedCardiac4D() {
            return _reconstructed4D;
        }

        inline void SetReconstructedCardiac4D(const RealImage& reconstructed4D) {
            _reconstructed4D = reconstructed4D;

            ImageAttributes attr = reconstructed4D.Attributes();
            attr._t = 1;

            _reconstructed = RealImage(attr);

            _mask = RealImage(attr);
            _mask = 1;

            _template_created = true;
        }

        inline const RealImage& GetVolumeWeights() {
            return _volume_weights;
        }

        inline void SetForceExcludedStacks(const Array<int>& force_excluded_stacks) {
            _force_excluded_stacks = force_excluded_stacks;
        }

        inline void SetForceExcludedLocs(const Array<int>& force_excluded_locs) {
            _force_excluded_locs = force_excluded_locs;
        }

        inline void SetSliceRRInterval(const Array<double>& rr) {
            _slice_rr = rr;
        }

        inline void SetSliceRRInterval(double rr) {
            _slice_rr = Array<double>(_slices.size(), rr);
        }

        inline void SetLocRRInterval(const Array<double>& rr) {
            Array<double> slice_rr;
            for (size_t i = 0; i < _slices.size(); i++)
                slice_rr.push_back(rr[_loc_index[i]]);
            _slice_rr = move(slice_rr);
        }

        inline void SetSliceCardiacPhase(const Array<double>& cardiacphases) {
            _slice_cardphase = cardiacphases;
        }

        inline void SetSliceCardiacPhase() {
            _slice_cardphase = Array<double>(_slices.size());
        }

        inline void SetReconstructedCardiacPhase(const Array<double>& cardiacphases) {
            _reconstructed_cardiac_phases = cardiacphases;
        }

        inline void SetReconstructedRRInterval(double rrinterval) {
            _reconstructed_rr_interval = rrinterval;
            if (_verbose)
                _verbose_log << "Reconstructed R-R interval = " << _reconstructed_rr_interval << " s." << endl;
        }

        inline void SetReconstructedTemporalResolution(double temporalresolution) {
            _reconstructed_temporal_resolution = temporalresolution;
            if (_verbose)
                _verbose_log << "Reconstructed temporal resolution = " << _reconstructed_temporal_resolution << " s." << endl;
        }

        inline void InitStackFactor(const Array<RealImage>& stacks) {
            _stack_factor = Array<double>(stacks.size(), 1);
        }

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

        inline void InitCorrectedSlices() {
            _corrected_slices = _slices;
        }

        inline void InitError() {
            _error = _slices;
        }

        inline void InitSliceTemporalWeights() {
            _slice_temporal_weight.clear();
            _slice_temporal_weight.resize(_reconstructed_cardiac_phases.size());
            for (size_t i = 0; i < _reconstructed_cardiac_phases.size(); i++)
                _slice_temporal_weight[i].resize(_slices.size());
        }

        // Scale volume to match the slice intensities
        inline void ScaleVolumeCardiac4D() {
            if (_verbose)
                _verbose_log << "Scaling volume: ";

            ScaleVolume(_reconstructed4D);
        }
    };  // end of ReconstructionCardiac4D class definition

} // namespace svrtk
