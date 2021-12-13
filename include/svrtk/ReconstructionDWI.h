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
#include "svrtk/Common.h"

using namespace mirtk;

namespace svrtk {

    class ReconstructionDWI
    {

    protected:

        RECON_TYPE _recon_type;

        Array<SLICECOEFFS> _volcoeffs;
        Array<SLICECOEFFS> _volcoeffsSF;


        Array<RealImage> _original_slices;

        Array<RealImage> _slices;
        Array<RealImage> _simulated_slices;
        Array<RealImage> _simulated_weights;
        Array<RealImage> _simulated_inside;

        Array<RigidTransformation> _transformations;

        Array<bool> _slice_inside;

        RealImage _reconstructed;

        bool _template_created;

        RealImage _mask;
        RealImage _mask_internal;
        RealImage _brain_probability;
        Array<RealImage> _probability_maps;

        bool _have_mask;

        RealImage _volume_weights;

        RealImage _confidence_map;

        double _sigma;

        double _mix;

        double _m;

        double _mean_s;

        double _sigma_s;

        double _mean_s2;

        double _sigma_s2;

        double _mix_s;

        double _step;

        Array<RealImage> _weights;

        Array<double> _slice_weight;

        Array<int> _stack_locs;

        Array<int> _intensity_weights;

        double _sigma_bias;

        Array<RealImage> _bias;

        Array<double> _scale;

        double _quality_factor;

        double _max_intensity;
        double _min_intensity;

        double _alpha;

        double _delta;

        double _lambda;

        double _average_volume_weight;
        int _regul_steps;

        double _alpha_bias;

        bool _global_bias_correction;

        double _low_intensity_cutoff;

        bool _intensity_matching_GD;

        Array<double> _stack_factor;
        double _average_value;
        Array<int> _stack_index;

        Array<int> _force_excluded;

        Array<int> _small_slices;

        bool _adaptive;

        bool _debug;

        bool _robust_slices_only;

        inline double G(double x,double s);

        inline double M(double m);

        int _directions[13][3];

        double _GA;



        Array<Array<double> > _directionsDTI;

        Array<double> _bvalues;

        RealImage _simulated_signal;

        RealImage _SH_coeffs;

        int _order;

        int _coeffNum;

        Matrix _dirs;

        SphericalHarmonics _sh;

        SphericalHarmonics _sh_vol;

        double _motion_sigma;

        double _lambdaLB;

        Array<int> _slice_order;



    public:

        ReconstructionDWI();

        ~ReconstructionDWI();


        double CreateTemplate( RealImage stack,
                              double resolution=0 );


        void SetMask(RealImage * mask, double sigma, double threshold=0.5 );

        void SetMaskOrient(RealImage * mask, RigidTransformation transformation);

        void StackRegistrations(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations);

        void SetTemplateImage(RealImage t, RigidTransformation);


        RealImage ReturnSimulatedSignal();

        void SetGA(double ga);

        void PutMask(RealImage mask);

        void CreateInternalMask();


        RealImage CreateAverage( Array<RealImage>& stacks,
                                    Array<RigidTransformation>& stack_transformations );

        void CropImage( RealImage& image, RealImage& mask, bool cropz = true );

        void TransformMask( RealImage& image,
                           RealImage& mask,
                           RigidTransformation& transformation );

        void Rescale( RealImage &img, double max);

        void StackRegistrations( Array<RealImage>& stacks,
                                Array<RigidTransformation>& stack_transformations,
                                int templateNumber);

        void CreateSlicesAndTransformations( Array<RealImage>& stacks,
                                            Array<RigidTransformation>& stack_transformations,
                                            Array<double>& thickness );

        void SetSlicesAndTransformations( Array<RealImage>& slices,
                                         Array<RigidTransformation>& slice_transformations,
                                         Array<int>& stack_ids,
                                         Array<double>& thickness );

        void SetForceExcludedStacks(Array<int> &force_excluded_stacks);


        void UpdateSlices(Array<RealImage>& stacks, Array<double>& thickness);

        void GetSlices( Array<RealImage>& slices );

        void SetSlices( Array<RealImage>& slices );

        void InvertStackTransformations( Array<RigidTransformation>& stack_transformations );

        void MatchStackIntensities (Array<RealImage>& stacks,
                                    Array<RigidTransformation>& stack_transformations,
                                    double averageValue,
                                    bool together=false);

        void MatchStackIntensitiesWithMasking (Array<RealImage>& stacks,
                                               Array<RigidTransformation>& stack_transformations,
                                               double averageValue,
                                               bool together=false);

        void MaskStacks(Array<RealImage>& stacks,Array<RigidTransformation>& stack_transformations);

        void MaskSlices();

        void SetTemplate(RealImage tempImage);

        void CoeffInit();

        int SliceCount(int inputIndex);

        void GaussianReconstruction(double small_slices_threshold = 0.1);

        void InitializeEM();

        void InitializeEMValues();

        void ResetBiasAndScale(Array<RealImage>& bias, Array<double>&);

        void AddBiasAndScale(Array<RealImage>& bias, Array<double>&);

        void InitializeRobustStatistics();

        void EStep();

        void Scale();

        void ExcludeSlicesScale();

        void Bias();

        void NormaliseBias(int iter);

        void NormaliseBias(int iter, RealImage& image);

        void SetBiasAlpha();

        void Superresolution(int iter);

        void MStep(int iter);

        void Regularization(int iter);

        void AdaptiveRegularization(int iter, RealImage& original);

        void L22Regularization(int iter, RealImage& original);

        void LaplacianRegularization(int iter, int t, RealImage& original);

        double LaplacianSmoothness(RealImage& original);

        void SliceToVolumeRegistration();

        void BiasCorrectVolume(RealImage& original);

        void MaskVolume();

        void MaskImage( RealImage& image, double padding=-1);

        void SaveSlices();

        void SlicesInfo( const char* filename, Array<string> &stack_filenames );

        void SaveSimulatedSlices();

        void SaveWeights();

        void SaveTransformations();

        void GetTransformations( Array<RigidTransformation> &transformations );

        void SetTransformations( Array<RigidTransformation> &transformations );

        inline RigidTransformation GetTransformation(int i);

        void SaveConfidenceMap();

        void SaveBiasFields();

        inline void SetSigma( double sigma );

        inline RealImage GetReconstructed();

        void SetReconstructed(RealImage &reconstructed);

        inline RealImage GetMask();

        inline void SetSmoothingParameters( double delta, double lambda );

        inline void SetRegulSteps(int steps);

        inline void SpeedupOn();

        inline void SpeedupOff();

        inline void GlobalBiasCorrectionOn();

        inline void GlobalBiasCorrectionOff();

        inline void SetLowIntensityCutoff( double cutoff );

        inline void SetForceExcludedSlices( Array<int>& force_excluded );

        inline void Set3DRecon();

        inline void Set1DRecon();

        inline void SetInterpolationRecon();

        inline void DebugOn();

        inline void DebugOff();

        inline void UseAdaptiveRegularisation();

        inline void ExcludeWholeSlicesOnly();

        inline int GetNumberOfTransformations();

        inline double GetAverageVolumeWeight();

        inline void SetIntensityMatchingGD();

        void Evaluate( int iter );

        void ReadTransformations( char* folder );

        void RestoreSliceIntensities();

        void ScaleVolume();

        void SimulateStacks(Array<RealImage>& stacks, bool threshold = false);

        void SimulateSlices();

        double Consistency();

        void ResetOrigin( GreyImage &image,
                         RigidTransformation& transformation);

        void ResetOrigin( RealImage &image,
                         RigidTransformation& transformation);


        void PackageToVolume( Array<RealImage>& stacks,
                             Array<int> &pack_num,
                             Array<RigidTransformation> stack_transformations);

        void SplitImage( RealImage image,
                        int packages,
                        Array<RealImage>& stacks );






        void GaussianReconstruction4D(int nStacks, Array<RigidTransformation> &stack_transformations, int order);
        void GaussianReconstruction4D2(int nStacks);
        void GaussianReconstruction4D3();
        void Init4D(int nStacks);
        void Init4DGauss(int nStacks, Array<RigidTransformation> &stack_transformations, int order);
        void Init4DGaussWeighted(int nStacks, Array<RigidTransformation> &stack_transformations, int order);
        void RotateDirections(double &dx, double &dy, double &dz, int i);
        void RotateDirections(double &dx, double &dy, double &dz, RigidTransformation &t);
        void CreateSliceDirections(Array< Array<double> >& directions, Array<double>& bvalues);
        void InitSH(Matrix dirs,int order);
        void InitSHT(Matrix dirs,int order);
        void SimulateSlicesDTI();
        void SimulateStacksDTI(Array<RealImage>& stacks, bool simulate_excluded=false);
        void SimulateStacksDTIIntensityMatching(Array<RealImage>& stacks, bool simulate_excluded=false);
        void SuperresolutionDTI(int iter, bool tv = false, double sh_alpha = 5);
        double LaplacianSmoothnessDTI();
        void SaveSHcoeffs(int iteration);
        void SimulateSignal();
        void SetLambdaLB(double lambda);
        void SetSliceOrder(Array<int> slice_order);
        void PostProcessMotionParameters(RealImage target, bool empty=false);
        void PostProcessMotionParameters2(RealImage target, bool empty=false);
        void PostProcessMotionParametersHS(RealImage target, bool empty=false);
        void PostProcessMotionParametersHS2(RealImage target, bool empty=false);
        void SetMotionSigma(double sigma);
        void SliceToVolumeRegistrationSH();
        void SetSimulatedSignal(RealImage signal);
        void CorrectStackIntensities(Array<RealImage>& stacks);

        void NormaliseBiasSH(int iter);
        void NormaliseBiasDTI(int iter, Array<RigidTransformation> &stack_transformations, int order);
        void NormaliseBiasSH2(int iter, Array<RigidTransformation> &stack_transformations, int order);

        void SlicesInfo2(Array<RigidTransformation> stack_transformations, RealImage mask);
        void SliceAverage(Array<RigidTransformation> stack_transformations, RealImage mask, Array<double>& average_values, Array<int>& areas, Array<double>& errors);

        void StructuralExclusion(Array<RigidTransformation> stack_transformations, RealImage mask, Array<RealImage> stacks, double intensity_threshold);

        inline void WriteSimulatedSignal(char * name);

        inline void SetSH(RealImage sh);

        double ConsistencyDTI();








        friend class ParallelStackRegistrations_DWI;
        friend class ParallelSliceToVolumeRegistration_DWI;
        friend class ParallelCoeffInit_DWI;
        friend class ParallelSuperresolution_DWI;
        friend class ParallelMStep_DWI;
        friend class ParallelEStep_DWI;
        friend class ParallelBias_DWI;
        friend class ParallelScale_DWI;
        friend class ParallelNormaliseBias_DWI;
        friend class ParallelSimulateSlices_DWI;
        friend class ParallelAverage_DWI;
        friend class ParallelSliceAverage_DWI;
        friend class ParallelAdaptiveRegularization1_DWI;
        friend class ParallelAdaptiveRegularization2_DWI;
        friend class ParallelL22Regularization_DWI;
        friend class ParallelLaplacianRegularization1_DWI;
        friend class ParallelLaplacianRegularization2_DWI;


        friend class ParallelSimulateSlicesDTI;
        friend class ParallelSuperresolutionDTI;
        friend class ParallelSliceToVolumeRegistrationSH;
        friend class ParallelNormaliseBiasDTI;
        friend class ParallelNormaliseBiasDTI2;



    };  // end of ReconstructionDWI class definition

    ////////////////////////////////////////////////////////////////////////////////
    // Inline/template definitions
    ////////////////////////////////////////////////////////////////////////////////


    inline double ReconstructionDWI::G(double x,double s)
    {
        return _step*exp(-x*x/(2*s))/(sqrt(6.28*s));
    }

    inline double ReconstructionDWI::M(double m)
    {
        return m*_step;
    }

    inline RealImage ReconstructionDWI::GetReconstructed()
    {
        return _reconstructed;
    }

    inline RealImage ReconstructionDWI::GetMask()
    {
        return _mask;
    }

    inline void ReconstructionDWI::PutMask(RealImage mask)
    {
        _mask=mask;;
    }


    inline void ReconstructionDWI::DebugOn()
    {
        _debug=true;
        cout<<"Debug mode."<<endl;
    }

    inline void ReconstructionDWI::ExcludeWholeSlicesOnly()
    {
        _robust_slices_only=true;
        cout<<"Exclude only whole slices."<<endl;
    }

    inline void ReconstructionDWI::UseAdaptiveRegularisation()
    {
        _adaptive = true;
    }

    inline void ReconstructionDWI::DebugOff()
    {
        _debug=false;
    }

    inline void ReconstructionDWI::SetSigma(double sigma)
    {
        _sigma_bias=sigma;
    }

    inline void ReconstructionDWI::SetGA(double ga)
    {
        _GA = ga;
    }

    inline void ReconstructionDWI::SpeedupOn()
    {
        _quality_factor=1;
    }

    inline void ReconstructionDWI::SpeedupOff()
    {
        _quality_factor=2;
    }

    inline void ReconstructionDWI::GlobalBiasCorrectionOn()
    {
        _global_bias_correction=true;
    }

    inline void ReconstructionDWI::GlobalBiasCorrectionOff()
    {
        _global_bias_correction=false;
    }

    inline void ReconstructionDWI::SetLowIntensityCutoff(double cutoff)
    {
        if (cutoff>1) cutoff=1;
        if (cutoff<0) cutoff=0;
        _low_intensity_cutoff = cutoff;
    }


    inline void ReconstructionDWI::SetSmoothingParameters(double delta, double lambda)
    {
        _delta=delta;
        _lambda=lambda*delta*delta;
        _alpha = 0.05/lambda;
        if (_alpha>1) _alpha= 1;
        cout<<"delta = "<<_delta<<" lambda = "<<lambda<<" alpha = "<<_alpha<<endl;
    }

    inline void ReconstructionDWI::SetForceExcludedSlices(Array<int>& force_excluded)
    {
        _force_excluded = force_excluded;
    }

    inline void ReconstructionDWI::Set3DRecon()
    {
        _recon_type = _3D;
    }

    inline void ReconstructionDWI::Set1DRecon()
    {
        _recon_type = _1D;
    }

    inline void ReconstructionDWI::SetInterpolationRecon()
    {
        _recon_type = _interpolate;
    }

    inline int ReconstructionDWI::GetNumberOfTransformations()
    {
        return _transformations.size();
    }

    inline RigidTransformation ReconstructionDWI::GetTransformation(int i)
    {
        return _transformations[i];
    }

    inline void ReconstructionDWI::SetRegulSteps(int steps)
    {
        _regul_steps=steps;
    }

    inline double ReconstructionDWI::GetAverageVolumeWeight()
    {
        return _average_volume_weight;
    }

    inline void ReconstructionDWI::SetIntensityMatchingGD()
    {
        _intensity_matching_GD = true;
    }

    inline void ReconstructionDWI::SetLambdaLB(double lambda)
    {
        _lambdaLB=lambda;
    }

    inline void ReconstructionDWI::SetSliceOrder(vector<int> slice_order)
    {
        _slice_order=slice_order;
    }

    inline void ReconstructionDWI::SetMotionSigma(double sigma)
    {
        _motion_sigma=sigma;
    }

    inline void ReconstructionDWI::SetSimulatedSignal(RealImage signal)
    {
        _simulated_signal=signal;
    }

    inline void ReconstructionDWI::WriteSimulatedSignal(char * name)
    {
        _simulated_signal.Write(name);
    }


    inline void ReconstructionDWI::SetSH(RealImage sh)
    {
        _SH_coeffs=sh;
    }

} // namespace svrtk
