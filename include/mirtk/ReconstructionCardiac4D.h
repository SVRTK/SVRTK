/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2008-2017 Imperial College London
 * Copyright 2018-2019 King's College London
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

#ifndef MIRTK_ReconstructionCardiac4D_H
#define MIRTK_ReconstructionCardiac4D_H

#include "mirtk/Reconstruction.h"


namespace mirtk {



class ReconstructionCardiac4D : public Reconstruction 
{

protected: 


  // Stacks
  Array<int> _loc_index;        // running index of all 2D slice locations
  Array<int> _stack_loc_index;  // index of 2D slice location in M2D stack
  Array<int> _stack_dyn_index;  // index of dynamic in M2D stack
  
  // Images
  Array<RealImage> _error;
  Array<RealImage> _corrected_slices;

   // PI
   const double PI = 3.14159265358979323846;

   // Reconstructed 4D Cardiac Cine Image
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
  
   // Initialise Slice Temporal Weights
   void InitSliceTemporalWeights();
   
   // Calculate Angular Difference
   double CalculateAngularDifference( double cardphase0, double cardphase );

   // Temporal Weighting Tukey Window Edge Percent
   // applies to sinc temporal weighting only
   double _wintukeypct = 0.3;
   
   // Flag for Sinc Temporal Weighting
   // use Gaussian weighting if false
   bool _is_temporalpsf_gauss;
   
   // Sinc Function
   double sinc( double x );

   // Tukey Window Function
   double wintukey( double angdiff, double alpha );
   
   // Calculate Temporal Weight
   double CalculateTemporalWeight( double cardphase0, double cardphase, double dt, double rr, double alpha );



 public:
   
   // Constructor
   ReconstructionCardiac4D();
   
   // Destructor
   ~ReconstructionCardiac4D();
   
   // Get/Set Reconstructed 4D Volume
   inline RealImage GetReconstructedCardiac4D();
   inline void SetReconstructedCardiac4D(RealImage &reconstructed4D);

   // Get Volume Weights
   inline RealImage GetVolumeWeights();

   // Get Slice-Location transformations
   void ReadSliceTransformation(char* slice_transformations_folder);

   ///Set stacks to be excluded
   inline void SetForceExcludedStacks( Array<int>& force_excluded_stacks );
   
   ///Set slice-locations to be excluded
   inline void SetForceExcludedLocs( Array<int>& force_excluded_locs );

   // Set Slice R-R Intervals
   void SetSliceRRInterval( Array<double> rr );
   void SetSliceRRInterval( double rr );
   
   // Set Loc R-R Intervals
   void SetLocRRInterval( Array<double> rr );

   // Set Slice Cardiac Phases
   void SetSliceCardiacPhase( Array<double> cardiacphases );
   void SetSliceCardiacPhase();
   
   // Set Reconstructed R-R Interval
   void SetReconstructedRRInterval( double rrinterval );
   
   // Set Reconstructed Temporal Resolution
   void SetReconstructedTemporalResolution( double temporalresolution );
   
   // Set Reconstructed Cardiac Phases
   void SetReconstructedCardiacPhase( vector<double> cardiacphases );

   // Set Reconstructed Volume Spatial Resolution
   double GetReconstructedResolutionFromTemplateStack( RealImage stack );
   
   // Initialise Reconstructed Volume from Static Mask
   void CreateTemplateCardiac4DFromStaticMask( RealImage mask, double resolution );
  
   ///Match stack intensities with masking
   void MatchStackIntensitiesWithMasking (Array<RealImage>& stacks,
                               Array<RigidTransformation>& stack_transformations,
                               double averageValue,
                               bool together=false);
                               
   //Set stack factors
   void InitStackFactor(Array<RealImage>& stacks);
   
   // Create slices from the stacks and 
   // slice-dependent transformations from stack transformations
   void CreateSlicesAndTransformationsCardiac4D( Array<RealImage>& stacks,
                                        Array<RigidTransformation>& stack_transformations,
                                        Array<double>& thickness,
                                        const Array<RealImage> &probability_maps=Array<RealImage>() );

  void ResetSlicesAndTransformationsCardiac4D();
  
   /// Init Corrected Slices
   void InitCorrectedSlices();
  
   /// Init Error
   void InitError();
 
   // Calculate Slice Temporal Weights
   void CalculateSliceTemporalWeights();
   inline void SetTemporalWeightGaussian();
   inline void SetTemporalWeightSinc();
   
   // Calculate Transformation Matrix Between Slices and Voxels
   void CoeffInitCardiac4D();

   // PSF-Weighted Reconstruction
   void GaussianReconstructionCardiac4D();
   
   ///Scale volume to match the slice intensities
   void ScaleVolumeCardiac4D();
   
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
   
   // Volume-to-Volume Registration
   void VolumeToVolumeRegistration(GreyImage target, GreyImage source, RigidTransformation& rigidTransf);
  
   // Calculate Displacement
   double CalculateDisplacement();
   double CalculateDisplacement(RigidTransformation drift);
   double CalculateWeightedDisplacement();
   double CalculateWeightedDisplacement(RigidTransformation drift);
   
   // Calculate Target Registration Error (TRE)
   void InitTRE();
   double CalculateTRE();
   double CalculateTRE(RigidTransformation drift);
   
   // Smooth Transformationss
   void SmoothTransformations(double sigma, int niter=10, bool use_slice_inside=false);
   
   // Scale Transformationss
   void ScaleTransformations(double scale);

   // Apply Static Mask to Reconstructed 4D Volume
   void StaticMaskReconstructedVolume4D();
   
   // Apply Static Mask 4D Volume
   RealImage StaticMaskVolume4D(RealImage volume, double padding);
   
   // Calculate Error
   void CalculateError();
   
   // Superresolution 
   void SuperresolutionCardiac4D( int iter );

   /// Edge-Preserving Regularization with Confidence Map
   void AdaptiveRegularizationCardiac4D(int iter, RealImage& original);

   
   /// Read Transformations
   void ReadTransformation( char* folder );
   
   /// Read Reference Transformations
   void ReadRefTransformation( char* folder );

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
   friend class ParallelCoeffInitCardiac4D;
   friend class ParallelSliceToVolumeRegistrationCardiac4D;
   friend class ParallelSimulateSlicesCardiac4D;
   friend class ParallelSimulateStacksCardiac4D;
   friend class ParallelNormaliseBiasCardiac4D;
   friend class ParallelSuperresolutionCardiac4D;   
   friend class ParallelAdaptiveRegularization1Cardiac4D;
   friend class ParallelAdaptiveRegularization2Cardiac4D;
   friend class ParallelCalculateError;
   friend class ParallelCalculateCorrectedSlices;
   
};  // end of ReconstructionCardiac4D class definition






////////////////////////////////////////////////////////////////////////////////
// Inline/template definitions
////////////////////////////////////////////////////////////////////////////////


// -----------------------------------------------------------------------------
// Set Temporal Weighting
// -----------------------------------------------------------------------------
inline void ReconstructionCardiac4D::SetTemporalWeightGaussian()
{
    _is_temporalpsf_gauss = true;
    cout << "Temporal PSF = Gaussian()" << endl;
}

inline void ReconstructionCardiac4D::SetTemporalWeightSinc()
{
    _is_temporalpsf_gauss = false;
    cout << "Temporal PSF = sinc() * Tukey_window()" << endl;
}

// -----------------------------------------------------------------------------
// Get/Set Reconstructed 4D Volume
// -----------------------------------------------------------------------------
inline RealImage ReconstructionCardiac4D::GetReconstructedCardiac4D()
{
    return _reconstructed4D;
}

inline void ReconstructionCardiac4D::SetReconstructedCardiac4D(RealImage &reconstructed4D)
{
  
    _reconstructed4D = reconstructed4D; 
    
    ImageAttributes attr = reconstructed4D.GetImageAttributes(); 
    attr._t = 1; 
    RealImage volume3D(attr); 
    
    _reconstructed = volume3D; 
    _reconstructed = 0; 
    
    _mask = volume3D; 
    _mask = 1; 
    
    _template_created = true; 

}


// -----------------------------------------------------------------------------
// Get Volume Weights
// -----------------------------------------------------------------------------
inline RealImage ReconstructionCardiac4D::GetVolumeWeights()
{
    return _volume_weights;
}


// -----------------------------------------------------------------------------
// Force Exclusion of Stacks
// -----------------------------------------------------------------------------
inline void ReconstructionCardiac4D::SetForceExcludedStacks(Array<int>& force_excluded_stacks)
{
    _force_excluded_stacks = force_excluded_stacks;  
}


// -----------------------------------------------------------------------------
// Force Exclusion of Stacks
// -----------------------------------------------------------------------------
inline void ReconstructionCardiac4D::SetForceExcludedLocs(Array<int>& force_excluded_locs)
{
    _force_excluded_locs = force_excluded_locs;  
}





} // namespace mirtk


#endif // MIRTK_ReconstructionCardiac4D_H
