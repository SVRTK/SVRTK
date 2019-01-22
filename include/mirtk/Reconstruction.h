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

#ifndef MIRTK_Reconstruction_H
#define MIRTK_Reconstruction_H

#include "mirtk/Common.h"
#include "mirtk/Options.h"

#include "mirtk/Array.h"
#include "mirtk/Point.h"

#include "mirtk/GenericImage.h"
#include "mirtk/GaussianBlurring.h"
#include "mirtk/Resampling.h"
#include "mirtk/ResamplingWithPadding.h"
#include "mirtk/LinearInterpolateImageFunction.hxx"

#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageTransformation.h"


#include "mirtk/MeanShift.h"


namespace mirtk {


struct POINT3D
{
    short x;
    short y;
    short z;
    double value;
};

enum RECON_TYPE {_3D, _1D, _interpolate};

typedef Array<POINT3D> VOXELCOEFFS; 
typedef Array<Array<VOXELCOEFFS> > SLICECOEFFS;
    

class Reconstruction 
{

protected: 
    
    // ---------------------------------------------------------------------------


    //Reconstruction type
    RECON_TYPE _recon_type;
    
    /// Structures to store the matrix of transformation between volume and slices
    Array<SLICECOEFFS> _volcoeffs;
    Array<SLICECOEFFS> _volcoeffsSF;
    
    int _slicePerDyn;


    /// Slices
    Array<RealImage> _slices;
    Array<RealImage> _slicesRwithMB;
    Array<RealImage> _simulated_slices;
    Array<RealImage> _simulated_weights;
    Array<RealImage> _simulated_inside;

    /// Transformations
    Array<RigidTransformation> _transformations;
    Array<RigidTransformation> _previous_transformations;
    Array<RigidTransformation> _transformationsRwithMB;


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
    RealImage _target;
    RealImage _brain_probability;
    Array<RealImage> _probability_maps;

    /// Weights for Gaussian reconstruction
    RealImage _volume_weights;
    RealImage _volume_weightsSF;

    /// Weights for regularization
    RealImage _confidence_map;
    
    //EM algorithm
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
    ///Variance for bias field
    double _sigma_bias;
    /// Slice-dependent bias fields
    Array<RealImage> _bias;

    ///Slice-dependent scales
    Array<double> _scale;
  
    ///Quality factor - higher means slower and better
    double _quality_factor;
    ///Intensity min and max
    double _max_intensity;
    double _min_intensity;
  
    //Gradient descent and regulatization parameters
    ///Step for gradient descent
    double _alpha;
    ///Determine what is en edge in edge-preserving smoothing
    double _delta;
    ///Amount of smoothing
    double _lambda;
    ///Average voxel wights to modulate parameter alpha
    double _average_volume_weight;
    double _average_volume_weightSF;

    
    //global bias field correction
    ///global bias correction flag
    bool _global_bias_correction;
    ///low intensity cutoff for bias field estimation
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

    /// use adaptive or non-adaptive regularisation (default:false)
    bool _adaptive;
    
    //utility
    ///Debug mode
    bool _debug;

    //do not exclude voxels, only whole slices
    bool _robust_slices_only;

    bool _withMB;
  
    //Probability density functions
    ///Zero-mean Gaussian PDF
    inline double G(double x,double s);
    ///Uniform PDF
    inline double M(double m);

    int _directions[13][3];

    /// Gestational age (to compute expected brain volume)
    double _GA;
    
    
    // ---------------------------------------------------------------------------
    
public: 
    
    ///Constructor
    Reconstruction();
    ///Destructor
    ~Reconstruction();

    ///Create zero image as a template for reconstructed volume
    double CreateTemplate( RealImage stack,
                           double resolution=0 );
    
    double CreateTemplateAniso(RealImage stack);

    
    double CreateLargeTemplate( Array<RealImage>& stacks,
                                Array<RigidTransformation>& stack_transformations,
                                ImageAttributes &templateAttr,
                                double resolution,
                                double smooth_mask,
                                double threshold_mask,
                                double expand=0 );

    ///Remember volumetric mask and smooth it if necessary
    void SetMask( RealImage* mask,
                  double sigma,
                  double threshold=0.5 );
    
    /// Set gestational age (to compute expected brain volume)
    void SetGA( double ga );
    
    
    ///Center stacks
    void CenterStacks( Array<RealImage>& stacks,
                       Array<RigidTransformation>& stack_transformations,
                       int templateNumber );
    
    //Create average image from the stacks and volumetric transformations
    RealImage CreateAverage( Array<RealImage>& stacks,
                                Array<RigidTransformation>& stack_transformations );
    
    
    ///Transform and resample mask to the space of the image
    void TransformMask( RealImage& image,
                        RealImage& mask,
                        RigidTransformation& transformation );

    /// Rescale image ignoring negative values
    void Rescale( RealImage &img,
                  double max);
    
    ///Calculate initial registrations
    void StackRegistrations( Array<RealImage>& stacks,
                             Array<RigidTransformation>& stack_transformations,
                             int templateNumber);
    
    ///Create slices from the stacks and slice-dependent transformations from
    ///stack transformations
    void CreateSlicesAndTransformations( Array<RealImage> &stacks,
                                        Array<RigidTransformation> &stack_transformations,
                                        Array<double> &thickness,
                                        const Array<RealImage> &probability_maps );
    
    void SetSlicesAndTransformations( Array<RealImage>& slices,
                                      Array<RigidTransformation>& slice_transformations,
                                      Array<int>& stack_ids,
                                      Array<double>& thickness );
    
    void ResetSlices( Array<RealImage>& stacks,
                      Array<double>& thickness );

    ///Update slices if stacks have changed
    void UpdateSlices( Array<RealImage>& stacks,
                       Array<double>& thickness );
  
    void GetSlices( Array<RealImage>& slices );
    
    void SetSlices( Array<RealImage>& slices );
    
    void SaveProbabilityMap( int i );
  
    ///Invert all stack transformation
    void InvertStackTransformations( Array<RigidTransformation>& stack_transformations );

    ///Match stack intensities
    void MatchStackIntensities ( Array<RealImage>& stacks,
                                 Array<RigidTransformation>& stack_transformations,
                                 double averageValue,
                                 bool together=false);
 
    ///Match stack intensities with masking
    void MatchStackIntensitiesWithMasking ( Array<RealImage>& stacks,
                                            Array<RigidTransformation>& stack_transformations,
                                            double averageValue,
                                            bool together=false);
 
    ///If template image has been masked instead of creating the mask in separate
    ///file, this function can be used to create mask from the template image
    RealImage CreateMask( RealImage image );
    
    void CreateMaskFromBlackBackground( Array<RealImage> stacks,
                                        Array<RigidTransformation> stack_transformations,
                                        double smooth_mask );
   
    
    ///Mask all stacks
    void MaskStacks(Array<RealImage>& stacks,Array<RigidTransformation>& stack_transformations);

    ///Mask all slices
    void MaskSlices();
 
    ///Set reconstructed image
    void SetTemplate(RealImage tempImage);
  
    ///Calculate transformation matrix between slices and voxels
    void CoeffInit();
    void CoeffInitSF(int begin, int end);

    ///Reconstruction using weighted Gaussian PSF
    void GaussianReconstruction();
    void GaussianReconstructionSF(Array<RealImage>& stacks);
    
    
    ///Initialise variables and parameters for EM
    void InitializeEM();
  
    ///Initialise values of variables and parameters for EM
    void InitializeEMValues();
  
    ///Initalize robust statistics
    void InitializeRobustStatistics();
  
    ///Perform E-step 
    void EStep();
  
    ///Calculate slice-dependent scale
    void Scale();
  
    ///Calculate slice-dependent bias fields
    void Bias();
    void NormaliseBias( int iter );
  
    ///Superresolution
    void Superresolution( int iter );
  
    ///Calculation of voxel-vise robust statistics
    void MStep( int iter );
  
    ///Edge-preserving regularization
    void Regularization( int iter );
  
    ///Edge-preserving regularization with confidence map
    void AdaptiveRegularization( int iter, RealImage& original );
  
    ///Slice to volume registrations
    void SliceToVolumeRegistration();
  
    ///Correct bias in the reconstructed volume
    void BiasCorrectVolume( RealImage& original );
    
    ///Mask the volume
    void MaskVolume();
    void MaskImage( RealImage& image, double padding=-1);
  
    ///Save slices
    void SaveSlices();
    void SaveSlicesWithTiming();
    void SlicesInfo( const char* filename,
                     Array<string> &stack_filenames );

    ///Save simulated slices
    void SaveSimulatedSlices();
  
    ///Save weights
    void SaveWeights();
    void SaveRegistrationStep(Array<RealImage>& stacks,int step);
  
    ///Save transformations
    void SaveTransformations();
    void SaveTransformationsWithTiming();
    void SaveTransformationsWithTiming(int iter);
    void GetTransformations( Array<RigidTransformation> &transformations );
    void SetTransformations( Array<RigidTransformation> &transformations );
  
    ///Save confidence map
    void SaveConfidenceMap();
  
    ///Save bias field
    void SaveBiasFields();
  
    ///Remember stdev for bias field
    inline void SetSigma( double sigma );
  
    ///Return reconstructed volume
    inline RealImage GetReconstructed();
    
    void SetReconstructed( RealImage &reconstructed );
  
    ///Return resampled mask
    inline RealImage GetMask();
    
    ///Remember volumetric mask 
    inline void PutMask( RealImage mask );
  
    ///Set smoothing parameters
    inline void SetSmoothingParameters( double delta, double lambda );
  
    ///Use faster lower quality reconstruction
    inline void SpeedupOn();
  
    ///Use slower better quality reconstruction
    inline void SpeedupOff();

    ///Switch on global bias correction
    inline void GlobalBiasCorrectionOn();
  
    ///Switch off global bias correction
    inline void GlobalBiasCorrectionOff();
  
    ///Set lower threshold for low intensity cutoff during bias estimation
    inline void SetLowIntensityCutoff( double cutoff );
  
    ///Set slices which need to be excluded by default
    inline void SetForceExcludedSlices( Array<int>& force_excluded );

    inline void Set3DRecon();
    inline void Set1DRecon();
    inline void SetInterpolationRecon();
    inline void SetSlicesPerDyn(int slices);
    inline void SetMultiband(bool withMB);
    
    inline int GetNumberOfTransformations();
    inline RigidTransformation GetTransformation(int n);

    
    inline void SetNMIBins( int nmi_bins );


    //utility
    ///Save intermediate results
    inline void DebugOn();
    ///Do not save intermediate results
    inline void DebugOff();

    inline void UseAdaptiveRegularisation();
    
    inline void ExcludeWholeSlicesOnly();
    
    ///Write included/excluded/outside slices
    void Evaluate( int iter );
    void EvaluateWithTiming( int iter );
  
    /// Read Transformations
    void ReadTransformation( char* folder );
  
    //To recover original scaling
    ///Restore slice intensities to their original values
    void RestoreSliceIntensities();
    ///Scale volume to match the slice intensities
    void ScaleVolume();
  
    ///To compare how simulation from the reconstructed volume matches the original stacks
    void SimulateStacks( Array<RealImage>& stacks );

    void SimulateSlices();
    
    ///Puts origin of the image into origin of world coordinates
    void ResetOrigin( GreyImage &image, RigidTransformation& transformation);
    
    ///Puts origin of the image into origin of world coordinates
    void ResetOrigin( RealImage &image, RigidTransformation& transformation);
  
    ///Packages to volume registrations
    void PackageToVolume( Array<RealImage>& stacks,
                          Array<int> &pack_num,
                          int iter,
                          bool evenodd=false,
                          bool half=false,
                          int half_iter=1);
  
    // Calculate Slice acquisition order
    void GetSliceAcquisitionOrder(Array<RealImage>& stacks,
    		Array<int> &pack_num, Array<int> order, int step, int rewinder);
    
    // Split image in a flexible manner
    void flexibleSplitImage(Array<RealImage>& stacks, Array<RealImage>& sliceStacks,
    		Array<int> &pack_num, Array<int> sliceNums, Array<int> order, int step, int rewinder);
    
    // Create Multiband replica for flexibleSplitImage
    void flexibleSplitImagewithMB(Array<RealImage>& stacks, Array<RealImage>& sliceStacks,
    		Array<int> &pack_num, Array<int> sliceNums, Array<int> multiband, Array<int> order, int step, int rewinder);
    
    // Split images into packages
    void splitPackages(Array<RealImage>& stacks, Array<int> &pack_num,
    		Array<RealImage>& packageStacks, Array<int> order, int step, int rewinder);
    
    // Create Multiband replica for splitPackages
    void splitPackageswithMB(Array<RealImage>& stacks, Array<int> &pack_num,
    		Array<RealImage>& packageStacks, Array<int> multiband, Array<int> order,
    		int step, int rewinder);
    
    // Performs package registration
    void newPackageToVolume( Array<RealImage>& stacks, Array<int> &pack_num,
    		Array<int> multiband, Array<int> order, int step,
    		int rewinder, int iter, int steps);
    
    // Perform subpackage registration for flexibleSplitImage
    void ChunkToVolume( Array<RealImage>& stacks, Array<int> &pack_num,
    		Array<int> sliceNums, Array<int> multiband, Array<int> order,
    		int step, int rewinder, int iter, int steps);
    
    // Calculate number of iterations needed for subpacking stages
    int giveMeDepth(Array<RealImage>& stacks, Array<int> &pack_num,
    		Array<int> multiband);
    
    // Calculate subpacking needed for tree like structure
    Array<int> giveMeSplittingArray(Array<RealImage>& stacks, Array<int> &pack_num,
    		Array<int> multiband, int iterations, bool last);
    
    void WriteSliceOrder();
    
    // Calculate relative change of displacement field across different iterations
    double calculateResidual(int padding);



    ///Splits stacks into packages
    void SplitImage( RealImage image,
                     int packages,
                     Array<RealImage>& stacks );
    
    ///Splits stacks into packages and each package into even and odd slices
    void SplitImageEvenOdd( RealImage image,
                            int packages,
                            Array<RealImage>& stacks );
    
    ///Splits image into top and bottom half roi according to z coordinate
    void HalfImage( RealImage image,
                    Array<RealImage>& stacks );
    
    ///Splits stacks into packages and each package into even and odd slices and top and bottom roi
    void SplitImageEvenOddHalf( RealImage image,
                                int packages,
                                Array<RealImage>& stacks,
                                int iter=1);
    
    ///Crop image according to the given mask
    void CropImage( RealImage& image, RealImage& mask );
    
    void CropImageIgnoreZ( RealImage& image, RealImage& mask );


    friend class ParallelStackRegistrations;
    friend class ParallelSliceToVolumeRegistration;
    friend class ParallelCoeffInit;
    friend class ParallelCoeffInitSF;
    friend class ParallelSuperresolution;
    friend class ParallelMStep;
    friend class ParallelEStep;
    friend class ParallelBias;
    friend class ParallelScale;
    friend class ParallelNormaliseBias;
    friend class ParallelSimulateSlices;
    friend class ParallelAverage;
    friend class ParallelSliceAverage;
    friend class ParallelAdaptiveRegularization1;
    friend class ParallelAdaptiveRegularization2;
    
    

   /*
  // Input images

  /// Set input images of the registration filter
  void Input(int, const RealImage **);

  /// Set input images of the registration filter
  template <class TVoxel> void Input(int, const RealImage<TVoxel> **);
  */

};

////////////////////////////////////////////////////////////////////////////////
// Inline/template definitions
////////////////////////////////////////////////////////////////////////////////


inline double Reconstruction::G(double x,double s)
{
    return _step*exp(-x*x/(2*s))/(sqrt(6.28*s));
}

inline double Reconstruction::M(double m)
{
    return m*_step;
}

inline RealImage Reconstruction::GetReconstructed()
{
    return _reconstructed;
}

inline RealImage Reconstruction::GetMask()
{
    return _mask;
}

inline void Reconstruction::PutMask(RealImage mask)
{
    _mask=mask;;
}


inline void Reconstruction::DebugOn()
{
    _debug=true;
}

inline void Reconstruction::UseAdaptiveRegularisation()
{
    _adaptive = true;
}

inline void Reconstruction::DebugOff()
{
    _debug=false;
}

inline void Reconstruction::SetSigma(double sigma)
{
    _sigma_bias=sigma;
}


inline void Reconstruction::SetGA(double ga)
{
    _GA = ga;
}


inline void Reconstruction::SpeedupOn()
{
    _quality_factor=1;
}

inline void Reconstruction::SpeedupOff()
{
    _quality_factor=2;
}

inline void Reconstruction::GlobalBiasCorrectionOn()
{
    _global_bias_correction=true;
}

inline void Reconstruction::GlobalBiasCorrectionOff()
{
    _global_bias_correction=false;
}

inline void Reconstruction::SetLowIntensityCutoff(double cutoff)
{
    if (cutoff>1) cutoff=1;
    if (cutoff<0) cutoff=0;
    _low_intensity_cutoff = cutoff;
}

inline void Reconstruction::ExcludeWholeSlicesOnly()
{
    _robust_slices_only=true;
    cout<<"Exclude only whole slices."<<endl;
}


inline void Reconstruction::SetSmoothingParameters(double delta, double lambda)
{
    _delta=delta;
    _lambda=lambda*delta*delta;
    _alpha = 0.05/lambda;
    if (_alpha>1) 
        _alpha= 1;
    cout<<"delta = "<<_delta<<" lambda = "<<lambda<<" alpha = "<<_alpha<<endl;
}

inline void Reconstruction::SetForceExcludedSlices(Array<int>& force_excluded)
{
    _force_excluded = force_excluded;  
}

inline void Reconstruction::SetNMIBins( int nmi_bins )
{
    _nmi_bins = nmi_bins;
    
}


inline void Reconstruction::Set3DRecon()
{
    _recon_type = _3D;
}

inline void Reconstruction::Set1DRecon()
{
    _recon_type = _1D;
}

inline void Reconstruction::SetInterpolationRecon()
{
    _recon_type = _interpolate;
}

inline void Reconstruction::SetSlicesPerDyn(int slices) 
{
	_slicePerDyn = slices;
}

inline void Reconstruction::SetMultiband(bool withMB) 
{
	_withMB = withMB;
}

inline int Reconstruction::GetNumberOfTransformations()
{
  return _transformations.size();
}

inline RigidTransformation Reconstruction::GetTransformation(int n)
{
  if(_transformations.size()<=n)
  {
    cerr<<"Reconstruction::GetTransformation: too large n = "<<n<<endl;
    exit(1);
  }
  return _transformations[n];
}




} // namespace mirtk


#endif // MIRTK_Reconstruction_H
