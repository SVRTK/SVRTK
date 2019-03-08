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

#ifndef MIRTK_ReconstructionCardiacVelocity4D_H
#define MIRTK_ReconstructionCardiacVelocity4D_H

#include "mirtk/ReconstructionCardiac4D.h"

#include "mirtk/Arith.h"


namespace mirtk {
    
    
    
    class ReconstructionCardiacVelocity4D : public ReconstructionCardiac4D
    {
        
    protected:
        
        // RealImage _reconstructed4DVelocity;
        RealImage _confidence_map_velocity;
        
        Array<Array<double> > _g_directions;
        Array<double> _g_values;
        Array<Array<double> > _v_directions;
        
        
        RealImage _simulated_signal;
        
        
        // Reconstructed 4D cardiac cine velocity images (for X, Y and Z components)
        Array<RealImage> _reconstructed5DVelocity;
        Array<RealImage> _confidence_maps_velocity;
        
        Array<Array<RealImage> > _globalReconstructed4DVelocityArray;
        Array<RealImage> _globalReconstructed4DArray;
        
        
        double _min_phase;
        double _max_phase;
        
        double _min_velocity;
        double _max_velocity;
        
        bool _adaptive_regularisation;
        bool _limit_intensities;
        
        
        // do we need it ?
        double _VENC;
        
        
        double _velocity_scale;
        
        // what is the correct value / units?
        const double gamma = 1; //42577; //1
        
        Array<RigidTransformation> _random_transformations;
        
        
    public:
        
        int current_stack_for_processing;
        int current_velocity_for_processing;
        
        Array<RealImage> _dif_stacks;
        
        ReconstructionCardiacVelocity4D();
        ~ReconstructionCardiacVelocity4D();
        
        
        void InitializeGradientMoments(Array<Array<double> > g_directions, Array<double> g_values);
        void InitializeVelocityVolumes();
        void RotateDirections(double &dx, double &dy, double &dz, int i);
        
        void SimulateSlicesCardiacVelocity4D();
        void SuperresolutionCardiacVelocity4D(int iter);
        void AdaptiveRegularizationCardiacVelocity4D(int iter, Array<RealImage>& originals);
        
        void GaussianReconstructionCardiacVelocity4DxT();
        void GaussianReconstructionCardiac4DxT();
        
        
        void InitialiseInverse(Array<RealImage> stacks);
        
        
        void InitialisationCardiacVelocity4D(Array<int> stack_numbers);
        Array<double> InverseVelocitySolution(Array<double> p_values, Array<Array<double>> g_values);
        
        void RandomRotations();
        
        void MaskSlicesPhase();
        void ResetValues();
        
        void SimulateSignal(int iter);
        double Consistency();
        
        
        inline void SetAlpha(double alpha);
        
        
        inline void SaveReconstructedVelocity4D(int iter);
        inline void SaveReconstructedVolume4D(int iter);
        
        inline void SetVelocityScale(double scale);
        inline void SetAdaptiveRegularisation(bool flag);
        inline void SetLimitIntensities(bool flag);
        
        inline void ItinialiseVelocityBounds();
        
        void StaticMaskReconstructedVolume5D();
        
        
        void InitializeEMVelocity4D();
        void InitializeEMValuesVelocity4D();
        void InitializeRobustStatisticsVelocity4D();
        void EStepVelocity4D();
        void MStepVelocity4D(int iter);

        friend class ParallelEStepardiacVelocity4D;
        friend class ParallelMStepCardiacVelocity4D;
        
        
        
        friend class ParallelSimulateSlicesCardiacVelocity4D;
        friend class ParallelSuperresolutionCardiacVelocity4D;
        friend class ParallelAdaptiveRegularization1CardiacVelocity4D;
        friend class ParallelAdaptiveRegularization2CardiacVelocity4D;
        
        friend class ParallelGaussianReconstructionCardiacVelocity4D;
        friend class ParallelGaussianReconstructionCardiacVelocity4DxT;
        friend class ParallelGaussianReconstructionCardiac4DxT;
        
        
    };  // end of ReconstructionCardiacVelocity4D class definition
    
    
    
    ////////////////////////////////////////////////////////////////////////////////
    // Inline/template definitions
    ////////////////////////////////////////////////////////////////////////////////
    
    // -----------------------------------------------------------------------------
    // ...
    // -----------------------------------------------------------------------------
    
    
    inline void ReconstructionCardiacVelocity4D::ResetValues()
    {
        
        for (int i=0; i<_slices.size(); i++) {
            _bias[i] = 0;
            _scale[i] = 1;
        }
        
    }

    
    // -----------------------------------------------------------------------------
    // ...
    // -----------------------------------------------------------------------------
    
    
    inline void ReconstructionCardiacVelocity4D::SetAlpha(double alpha)
    {
//        if (alpha > 1)
//            alpha = 1;
        
        _alpha = alpha;
        
    }
    
    
    // -----------------------------------------------------------------------------
    // ...
    // -----------------------------------------------------------------------------
    
    
    inline void ReconstructionCardiacVelocity4D::SetAdaptiveRegularisation(bool flag)
    {
        _adaptive_regularisation = flag;
    }
    
    // -----------------------------------------------------------------------------
    // ...
    // -----------------------------------------------------------------------------

    
    inline void ReconstructionCardiacVelocity4D::SetLimitIntensities(bool flag)
    {
        _limit_intensities = flag;
    }
    
    // -----------------------------------------------------------------------------
    // ...
    // -----------------------------------------------------------------------------
    
    
    inline void ReconstructionCardiacVelocity4D::ItinialiseVelocityBounds()
    {
        
        _min_phase = -3.14;
        _max_phase = 3.14;
        
        int templateNumber = 1;
        
        _min_velocity = _min_phase / (_g_values[templateNumber]*gamma);
        _max_velocity = _max_phase / (_g_values[templateNumber]*gamma);
        
        cout << " - velocity limits : [ " << _min_velocity << " ; " << _max_velocity << " ] " << endl;
        
        
    }
    
    // -----------------------------------------------------------------------------
    // ...
    // -----------------------------------------------------------------------------
    
    
    inline void ReconstructionCardiacVelocity4D::SetVelocityScale(double scale)
    {
        _velocity_scale = scale;
    }
    
    // -----------------------------------------------------------------------------
    // ...
    // -----------------------------------------------------------------------------
    
    
    inline void ReconstructionCardiacVelocity4D::SaveReconstructedVolume4D(int iter)
    {
        
        char buffer[256];
        
        sprintf(buffer,"phase-%i.nii.gz", iter);
        _reconstructed4D.Write(buffer);
        
    }
    
    
    // -----------------------------------------------------------------------------
    // ...
    // -----------------------------------------------------------------------------
    
    
    
    inline void ReconstructionCardiacVelocity4D::SaveReconstructedVelocity4D(int iter)
    {
        
        char buffer[256];
        
        
        cout << ".............................................." << endl;
        cout << ".............................................." << endl;
        cout << " - Reconstructed velocity : " << endl;
        
        
        if (_reconstructed5DVelocity[0].GetT() == 1) {
        
            ImageAttributes attr = _reconstructed5DVelocity[0].GetImageAttributes();
            attr._t = 3;
            
            RealImage output_4D(attr);
            
            for (int t=0; t<output_4D.GetT(); t++)
                for (int z=0; z<output_4D.GetZ(); z++)
                    for (int y=0; y<output_4D.GetY(); y++)
                        for (int x=0; x<output_4D.GetX(); x++)
                            output_4D(x,y,z,t) = _reconstructed5DVelocity[t](x,y,z,0);
            
            if (iter < 0) {
                sprintf(buffer,"velocity-vector-init.nii.gz");
            }
            else {
            if (iter < 50)
                    sprintf(buffer,"velocity-vector-%i.nii.gz", iter);
                else
                    sprintf(buffer,"velocity-vector-final.nii.gz");
            }
            
            output_4D.Write(buffer);
            
            
            attr._t = 1;
            RealImage output_sum(attr);
            output_sum = 0;
            
            for (int v=0; v<output_4D.GetT(); v++)
                output_sum += output_4D.GetRegion(0, 0, 0, v, attr._x, attr._y, attr._z, (v+1));
            
            
            output_sum.Write("sum-velocity.nii.gz");
            
            
            cout << "        " << buffer << endl;
            
        }
        else {
            
            for (int i=0; i<_reconstructed5DVelocity.size(); i++) {
                
                if (iter < 50)
                sprintf(buffer,"velocity-%i-%i.nii.gz", i, iter);
                    else
                sprintf(buffer,"velocity-final-%i.nii.gz", i);
                
                _reconstructed5DVelocity[i].Write(buffer);
                cout << "     " << buffer << endl;
            }
        }
        

        cout << ".............................................." << endl;
        cout << ".............................................." << endl;
        
        /*
        
        for (int i=0; i<_reconstructed5DVelocity.size(); i++) {
            
            RealImage subtracted4D = _reconstructed5DVelocity[i];
            
            RealImage average3D = subtracted4D.GetRegion(0,0,0,0,subtracted4D.GetX(),subtracted4D.GetY(),subtracted4D.GetZ(),1);
            average3D = 0;
            
            
            for (int t=0; t<subtracted4D.GetT(); t++)
                average3D = average3D + subtracted4D.GetRegion(0,0,0,(t),subtracted4D.GetX(),subtracted4D.GetY(),subtracted4D.GetZ(),(t+1));
            
            average3D /= subtracted4D.GetT();
            
            for (int t=0; t<subtracted4D.GetT(); t++)
                for (int z=0; z<subtracted4D.GetZ(); z++)
                    for (int y=0; y<subtracted4D.GetY(); y++)
                        for (int x=0; x<subtracted4D.GetX(); x++)
                            subtracted4D(x,y,z,t) = subtracted4D(x,y,z,t) - average3D(x,y,z);
            
            subtracted4D *= _velocity_scale;
            
            sprintf(buffer,"subtracted-velocity-%i-%i.nii.gz", i, iter);
            subtracted4D.Write(buffer);
            
        }
        
        */
    }
    
    
    // -----------------------------------------------------------------------------
    // ...
    // -----------------------------------------------------------------------------
    
    
} // namespace mirtk


#endif // MIRTK_Reconstruction_H

