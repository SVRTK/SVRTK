/*
 *
 */

#ifndef MIRTK_Reconstruction_H
#define MIRTK_Reconstruction_H

#include "mirtk/Common.h"
#include "mirtk/Options.h"

#include "mirtk/Array.h"
#include "mirtk/Point.h"

#include "mirtk/GenericImage.h"
#include "mirtk/GaussianBlurring.h"
#include "mirtk/GaussianBlurringWithPadding.h"

#include "mirtk/Resampling.h"
#include "mirtk/ResamplingWithPadding.h"
#include "mirtk/LinearInterpolateImageFunction.hxx"

#include "mirtk/GaussianBlurring2D.h"

#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/Transformation.h" 
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageTransformation.h"

#include "mirtk/MultiLevelFreeFormTransformation.h"
#include "mirtk/FreeFormTransformation.h"
#include "mirtk/LinearFreeFormTransformation3D.h"

#include "mirtk/IOConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/VoxelFunction.h"
#include "mirtk/Transformations.h"

#include "mirtk/SimilarityMeasure.h"
#include "mirtk/ImageSimilarity.h"
#include "mirtk/HistogramImageSimilarity.h"

#include "mirtk/Dilation.h"
#include "mirtk/Erosion.h"

#include "mirtk/NLDenoising.h"

#include "mirtk/HistogramMatching.h"

#include <pthread.h>



namespace mirtk {
    
    
    struct POINT3D
    {
        short x;
        short y;
        short z;
        double value;
    };
    
    typedef Array<POINT3D> VOXELCOEFFS;
    typedef Array<Array<VOXELCOEFFS> > SLICECOEFFS;
    
    
    
    class ReconstructionFFD //: public RegistrationFilter
    {
        
    protected:
        
        // ---------------------------------------------------------------------------
        
        /// Structures to store the matrix of transformation between volume and slices
        Array<SLICECOEFFS> _volcoeffs;
        
        /// Slices
        Array<RealImage*> _slices;
        Array<RealImage*> _simulated_slices;
        Array<RealImage*> _simulated_weights;
        Array<RealImage*> _simulated_inside;
        Array<RealImage*> _denoised_slices;
        
        Array<GreyImage*> _slice_masks;
        Array<ImageAttributes> _slice_attr;
        ImageAttributes _reconstructed_attr;
        
        GreyImage _grey_reconstructed;
        

        Array<GreyImage*> _grey_slices;
        Array<int> _stack_locs;
        Array<int> _package_index;
        
        
        GreyImage* _p_grey_reconstructed;
        
//        Array<GreyImage*> _p_grey_slices;
        
        
        double _global_NCC_threshold;
        double _local_NCC_threshold;
        int _local_window_size;
        
        
        double _global_SSIM_threshold;
        double _local_SSIM_threshold;
        
        Array<int> _cp_spacing;
        
        
        int _template_slice_number;
        
        bool _structural_exclusion;
        bool _global_init_ffd;
        bool _masked;
        
        
        Array<double> _reg_slice_weight;
        
        Array<double> _structural_slice_weight;
        Array<double> _slice_ssim;
        Array<double> _negative_J_values;
        Array<double> _slice_cc;
        Array<int> _excluded_points;
        
        Array<double> _min_dispacements;
        Array<double> _max_dispacements;

        Array<double> _min_val;
        Array<double> _max_val;
        
        
        Array<RealImage*> _slice_displacements;
        Array<RealImage*> _slice_jacobians;
        Array<RealImage*> _slice_2dssim;
        Array<RealImage*> _slice_2dncc;
        Array<RealImage*> _slice_dif;
        
        int _current_iteration;
        int _current_round;
        
        int _excluded_stack;
        
        /// Transformations
        Array<RigidTransformation> _transformations;
        Array<RigidTransformation> _package_transformations;
        Array<MultiLevelFreeFormTransformation*> _mffd_transformations;
        Array<MultiLevelFreeFormTransformation*> _global_mffd_transformations;
        
        
        // ---------------------------------------------------------------------------
        
        int _directions2D[5][2];
        double _alpha2D;
        double _lambda2D;
        double _delta2D;
        RealImage _slice;
        
        
        // ---------------------------------------------------------------------------
        
        
        /// Indicator whether slice has an overlap with volumetric mask
        Array<bool> _slice_inside;
        /// Flag to say whether the template volume has been created
        bool _template_created;
        /// Flag to say whether we have a mask
        bool _have_mask;
        
        /// Volumes
        RealImage _reconstructed;
        RealImage _mask;
        RealImage _brain_probability;
        Array<RealImage> _probability_maps;
        /// Weights for Gaussian reconstruction
        RealImage _volume_weights;
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
        Array<RealImage*> _weights;
        ///Slice posteriors
        Array<double> _slice_weight;
        
        //Bias field
        ///Variance for bias field
        double _sigma_bias;
        /// Slice-dependent bias fields
        Array<RealImage*> _bias;
        
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
        
        
        //global bias field correction_stack_locs
        ///global bias correction flag_stack_locs
        bool _global_bias_correction;
        ///low intensity cutoff for bi_stack_locs
        double _low_intensity_cutoff;
        
        //to restore original signal i_stack_locs
        Array<double> _stack_factor;
        double _average_value;
        Array<int> _stack_index;
        
        //forced excluded slices
        Array<int> _force_excluded;
        
        //slices identify as too small_stack_locs
        Array<int> _small_slices;
        
        /// use adaptive or non-adaptive regularisation (default:false)
        bool _adaptive;
        
        //utility
        ///Debug mode
        bool _debug;
        
        
        //Probability density functions
        ///Zero-mean Gaussian PDF
        inline double G(double x,double s);
        ///Uniform PDF
        inline double M(double m);
        
        int _directions[13][3];
        
        /// Additional FFD registration of slices
        bool _mffd_mode;
        
        /// Flag for RS: not exclude voxels, only whole slices
        bool _robust_slices_only;
        
        // ---------------------------------------------------------------------------
        
    public:
        
        ///Constructor
        ReconstructionFFD();
        ///Destructor
        ~ReconstructionFFD();
        
        ///Create zero image as a template for reconstructed volume
        double CreateTemplate( RealImage* stack,
                              double resolution=0 );
        
        
        ///Remember volumetric mask and smooth it if necessary
        void SetMask( RealImage* mask,
                     double sigma,
                     double threshold=0.5 );
        
        
        ///Transform and resample mask to the space of the image
        void TransformMask( RealImage& image,
                           RealImage& mask,
                           RigidTransformation& transformation );
        
        /// Rescale image ignoring negative values
        void Rescale( RealImage *img,
                     double max);
        
        
        void SaveProbabilityMap( int i );
        
        ///Invert all stack transformation
        void InvertStackTransformations( Array<RigidTransformation>& stack_transformations );
        
        ///Match stack intensities
        void MatchStackIntensities ( Array<RealImage>& stacks,
                                    Array<RigidTransformation>& stack_transformations,
                                    double averageValue,
                                    bool together=false);
        
        ///Match stack intensities with masking
        void MatchStackIntensitiesWithMasking ( Array<RealImage*> stacks,
                                               Array<RigidTransformation>& stack_transformations,
                                               double averageValue,
                                               bool together=false);
        
        RealImage CreateMask( RealImage image );
        
        
        ///Calculate transformation matrix between slices and voxels
        void CoeffInit();
        
        ///Reconstruction using weighted Gaussian PSF
        void GaussianReconstruction();
        
        
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
        void FastSliceToVolumeRegistration( int iter, int round, int number_of_stacks);
        
        
        ///Correct bias in the reconstructed volume
        void BiasCorrectVolume( RealImage& original );
        
        ///Mask the volume
        void MaskVolume();
        void MaskImage( RealImage& image, double padding=-1);
        
        ///Save slices
        void SaveSlices();
        void SlicesInfo( const char* filename, Array<string> &stack_filenames );
        
        ///Save weights
        void SaveWeights();
        
        ///Save transformations
        void SaveTransformations();
        
        ///Save confidence map
        void SaveConfidenceMap();
        
        ///Save bias field
        void SaveBiasFields();
        
        ///Remember stdev for bias field
        inline void SetSigma( double sigma );
        
        ///Return reconstructed volume
        inline RealImage GetReconstructed();
        
        ///Return resampled mask
        inline RealImage GetMask();
        
        inline void PutMask( RealImage mask );
        
        inline void SetMaskedFlag( bool flag );
        
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
        
        ///Set FFD registration flag
        inline void FFDRegistrationOn();
        inline void FFDRegistrationOff();
        
        inline void IncludeTemplateOn();
        inline void IncludeTemplateOff();
        
        inline void ExcludeStack(int excluded_stack);
        
        inline void SetStructural(bool flag);
        
        //utility
        ///Save intermediate results
        inline void DebugOn();
        ///Do not save intermediate results
        inline void DebugOff();
        
        inline void UseAdaptiveRegularisation();
        
        inline void ExcludeWholeSlicesOnly();
        
        ///Write included/excluded/outside slices
        void Evaluate( int iter );
        
        //To recover original scaling
        ///Restore slice intensities to their original values
        void RestoreSliceIntensities();
        ///Scale volume to match the slice intensities
        void ScaleVolume();
        
        
        void SliceToReconstuced( int inputIndex, double& x, double& y, double& z, int& i, int& j, int& k, int mode );
        
        ///To compare how simulation from the reconstructed volume matches the original stacks
        void SimulateStacks( Array<RealImage*>& stacks, bool simulate_excluded );
        
        void SimulateSlices();
        
        ///Puts origin of the image into origin of world coordinates
        void ResetOrigin( GreyImage &image, RigidTransformation& transformation);
        
        ///Puts origin of the image into origin of world coordinates
        void ResetOrigin( RealImage &image, RigidTransformation& transformation);
        
        ///Splits stacks into packages
        void SplitImage( RealImage image,
                        int packages,
                        Array<RealImage>& stacks );
        
        
        ///Crop image according to the given mask
        void CropImage( RealImage& image, RealImage& mask );
        
        
        // ---------------------------------------------------------------------------
        
        void SliceDifference();
        
        
        void ResetSlicesAndTransformationsFFD();
        
        
        void CreateSlicesAndTransformationsFFD( Array<RealImage*> stacks, Array<RigidTransformation> &stack_transformations, Array<double> thickness, int d_packages, Array<int> selected_slices, int template_number );
        
        
        void SimulateMotion( RealImage input_volume, Array<RealImage*> stacks, int iter );
        
        
        void SetCP( int value, int step );
        
        
        double SliceCCMap2(int inputIndex, int box_size, double threshold);
        
        
        void SlowSliceToVolumeRegistrationFFD( int iter, int round, int number_of_stacks );
        
        
        void SVRStepStack( int stackIndex );
        
        
        void Save3DNCC( Array<RealImage*> stacks, int iteration);
        
        
        void CStep2D();
        
        
        void SaveWeights3D(Array<RealImage> stacks, int iter);
        
        
        void SliceDisplacement( Array<RealImage> stacks, int iter );
        
        
        void RigidPackageRegistration( Array<GreyImage*> stacks, GreyImage* template_stack, int nPackages, Array<RigidTransformation>& stack_transformations, bool init_reset );
        
        
        void GenerateGlobalDisplacement(Array<RealImage> stacks);
        
        
        void GenerateStackDisplacement(RealImage stack, int stack_number);
        
        
        RealImage GenerateSliceDisplacement(int i);
        
        
        RealImage GenerateSliceMotionModel(int i, double& j_min, double& j_max, double& j_avg);
        
        
        void GenerateMotionModel(RealImage stack, int stack_number);
        
        
        void JStep(int iteration);
        
        
        int SliceJNegativeCount(int i);
        
        
        void EvaluateRegistration();
        
        
        void CreateSliceMasks(RealImage main_mask, int iter);
        
        
        double LocalSSIM( RealImage slice, RealImage sim_slice );
        
        
        void LoadGlobalTransformations( int number_of_stacks, Array<MultiLevelFreeFormTransformation*>& global_mffd_transformations );
        
        
        void NLMFiltering(Array<RealImage*> stacks);
        
        
        double SliceCC(RealImage slice_1, RealImage slice_2);
        
        
        void FFDStackRegistration( Array<GreyImage*> stacks, GreyImage* template_stack, Array<RigidTransformation> stack_transformations );
        
        
        void RigidStackRegistration( Array<GreyImage*> stacks, GreyImage* template_stack, Array<RigidTransformation>& stack_transformations, bool init_reset );
        
        
        void FilterSlices( double alpha2D, double lambda2D, double delta2D );
        
        void AdaptiveRegularization2D( int inputIndex, double alpha, double lambda, double delta );
        
        
        void FastCreateSliceMasks( GreyImage main_mask, int iter );


        void SVRStep( int inputIndex );

        RealImage StackIntersection( Array<RealImage*>& p_stacks, int templateNumber, RealImage template_mask );
        
        void CropImageP(RealImage* image, RealImage* mask);
        
        void TransformMaskP(RealImage* image, RealImage& mask);
        
        
        
        
        void SetSVROutput( Array<MultiLevelFreeFormTransformation*>& slice_mffd_transformations );
        
        void GetSVRInput( RealImage& grey_reconstructed, Array<RealImage>& grey_slices, Array<MultiLevelFreeFormTransformation*>& global_mffd_transformations, Array<MultiLevelFreeFormTransformation*>& slice_mffd_transformations, Array<RigidTransformation>& package_transformations );
        
        
        // ---------------------------------------------------------------------------
        
        
        friend class Parallel2DAdaptiveRegularization2FFD;
        friend class Parallel2DAdaptiveRegularization1FFD;
        
        
        friend class ParallelCreateSliceMasks;
        
        friend class ParallelSVRFFD; 

        
        friend class ParallelStackRegistrationsFFD;
        friend class ParallelSliceToVolumeRegistrationFFD;
        friend class ParallelSliceToVolumeRegistrationInitFFD;
        
        friend class ParallelCCMap;
        
        friend class ParallelCoeffInitFFD;
        friend class ParallelSuperresolutionFFD;
        friend class ParallelMStepFFD;
        friend class ParallelEStepFFD;
        friend class ParallelBiasFFD;
        friend class ParallelScaleFFD;
        friend class ParallelNormaliseBiasFFD;
        friend class ParallelSimulateSlicesFFD;
        
        friend class ParallelAdaptiveRegularization1FFD;
        friend class ParallelAdaptiveRegularization2FFD;
        
        
        
    };
    
    ////////////////////////////////////////////////////////////////////////////////
    // Inline/template definitions
    ////////////////////////////////////////////////////////////////////////////////
    
    inline void ReconstructionFFD::SetStructural(bool flag)
    {
        _structural_exclusion = flag;
        _global_init_ffd = true;
        
    }
    
    inline void ReconstructionFFD::SetMaskedFlag(bool flag)
    {
        _masked = true;
        _global_NCC_threshold = 0.7;
    }
    
    
    inline double ReconstructionFFD::G(double x,double s)
    {
        return _step*exp(-x*x/(2*s))/(sqrt(6.28*s));
    }
    
    inline double ReconstructionFFD::M(double m)
    {
        return m*_step;
    }
    
    inline RealImage ReconstructionFFD::GetReconstructed()
    {
        return _reconstructed;
    }
    
    inline RealImage ReconstructionFFD::GetMask()
    {
        return _mask;
    }
    
    inline void ReconstructionFFD::PutMask(RealImage mask)
    {
        _mask=mask;
    }
    
    
    inline void ReconstructionFFD::FFDRegistrationOn()
    {
        _mffd_mode = true;
    }
    
    inline void ReconstructionFFD::FFDRegistrationOff()
    {
        _mffd_mode = false;
    }
    
    
    inline void ReconstructionFFD::DebugOn()
    {
        _debug=true;
    }
    
    inline void ReconstructionFFD::UseAdaptiveRegularisation()
    {
        _adaptive = true;
    }
    
    inline void ReconstructionFFD::DebugOff()
    {
        _debug=false;
    }
    
    inline void ReconstructionFFD::SetSigma(double sigma)
    {
        _sigma_bias=sigma;
    }
    
    
    inline void ReconstructionFFD::ExcludeStack(int excluded_stack)
    {
        _excluded_stack=excluded_stack;
    }
    
    
    inline void ReconstructionFFD::SpeedupOn()
    {
        _quality_factor=1;
    }
    
    inline void ReconstructionFFD::SpeedupOff()
    {
        _quality_factor=2;
    }
    
    inline void ReconstructionFFD::GlobalBiasCorrectionOn()
    {
        _global_bias_correction=true;
    }
    
    inline void ReconstructionFFD::GlobalBiasCorrectionOff()
    {
        _global_bias_correction=false;
    }
    
    inline void ReconstructionFFD::SetLowIntensityCutoff(double cutoff)
    {
        if (cutoff>1) cutoff=1;
        if (cutoff<0) cutoff=0;
        _low_intensity_cutoff = cutoff;
    }
    
    
    inline void ReconstructionFFD::SetSmoothingParameters(double delta, double lambda)
    {
        _delta=delta;
        _lambda=lambda*delta*delta;
        _alpha = 0.05/lambda;
        if (_alpha>1)
            _alpha= 1;
        
        cout << " - parameters : d = " << delta << ", l = " << lambda << " a = " << _alpha << endl;
        
    }
    
    inline void ReconstructionFFD::SetForceExcludedSlices(Array<int>& force_excluded)
    {
        _force_excluded = force_excluded;
    }
    
    inline void ReconstructionFFD::ExcludeWholeSlicesOnly()
    {
        _robust_slices_only=true;
        cout<<"Exclude only whole slices."<<endl;
    }
    
    
    
    
    
    
    
} // namespace mirtk


#endif // MIRTK_Reconstruction_H

