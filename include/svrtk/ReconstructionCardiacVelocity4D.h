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

// MIRTK
#include "mirtk/Arith.h" 
#include "mirtk/Math.h"

// SVRTK
#include "svrtk/ReconstructionCardiac4D.h"

using namespace mirtk;

namespace svrtk {
    

    class ReconstructionCardiacVelocity4D : public ReconstructionCardiac4D
    {
        
    protected:
        
        // Arrays of gradient moment values
        Array<Array<double> > _g_directions;
        Array<double> _g_values;
        
        // Array of velocity directions
        Array<Array<double> > _v_directions;
        
        // Reconstructed 4D cardiac cine velocity images (for X, Y and Z components)
        Array<RealImage> _reconstructed5DVelocity;
        Array<RealImage> _confidence_maps_velocity;

        
        double _min_phase;
        double _max_phase;
        
        double _min_velocity;
        double _max_velocity;
        
        bool _adaptive_regularisation;
        bool _limit_intensities;
        
        const double PI = 3.14159265358979323846;
        
        const double gamma = 2*PI*0.042577;
        
        Array<RigidTransformation> _random_transformations;
        Array<Array<double> > _slice_g_directions;
        Array<Array<RealImage>> _simulated_velocities;
        
        
    public:
        
        ReconstructionCardiacVelocity4D();
        ~ReconstructionCardiacVelocity4D();
        
        void InitializeGradientMoments(Array<Array<double> > g_directions, Array<double> g_values);
        void InitializeVelocityVolumes();
        void InitializeSliceGradients4D();
        
        void SimulateSlicesCardiacVelocity4D();
        void SuperresolutionCardiacVelocity4D(int iter);
        void AdaptiveRegularizationCardiacVelocity4D(int iter, Array<RealImage>& originals);

        void SaveSliceInfo();
        void SaveOuput( Array<RealImage> stacks );
        void StaticMaskReconstructedVolume5D();
        
        inline void SaveReconstructedVelocity4D(int iter);
        inline void SaveReconstructedVolume4D(int iter);

        void MaskSlicesPhase();
        double Consistency();
        
        
        inline void SetAlpha(double alpha);
        inline void SetAdaptiveRegularisation(bool flag);
        inline void SetLimitIntensities(bool flag);
        inline void LimitTimeWindow();
        
        inline void ItinialiseVelocityLimits();
        
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


    };  // end of ReconstructionCardiacVelocity4D class definition
    
    
    
    ////////////////////////////////////////////////////////////////////////////////
    // Inline/template definitions
    ////////////////////////////////////////////////////////////////////////////////
    
    // -----------------------------------------------------------------------------
    // Set flag for cropping temporal PSF
    // -----------------------------------------------------------------------------

    
    inline void ReconstructionCardiacVelocity4D::LimitTimeWindow()
    {
        _no_ts = true;
    }
    
    
    // -----------------------------------------------------------------------------
    // Set alpha for gradient descent
    // -----------------------------------------------------------------------------
    
    
    inline void ReconstructionCardiacVelocity4D::SetAlpha(double alpha)
    {
        _alpha = alpha;
    }

    
    // -----------------------------------------------------------------------------
    // Set adaptive regularisation flag
    // -----------------------------------------------------------------------------
    
    
    inline void ReconstructionCardiacVelocity4D::SetAdaptiveRegularisation(bool flag)
    {
        _adaptive_regularisation = flag;
    }
    
    
    // -----------------------------------------------------------------------------
    // Set flag for limiting intensities
    // -----------------------------------------------------------------------------

    
    inline void ReconstructionCardiacVelocity4D::SetLimitIntensities(bool flag)
    {
        _limit_intensities = flag;
    }
    
    // -----------------------------------------------------------------------------
    // Compute and set velocity limits
    // -----------------------------------------------------------------------------
    
    inline void ReconstructionCardiacVelocity4D::ItinialiseVelocityLimits()
    {
        
        _min_phase = -3.14;
        _max_phase = 3.14;
        
        int templateNumber = 1;
        
        _min_velocity = _min_phase / (_g_values[templateNumber]*gamma);
        _max_velocity = _max_phase / (_g_values[templateNumber]*gamma);
        
        if (_debug)
            cout << " - velocity limits : [ " << _min_velocity << " ; " << _max_velocity << " ] " << endl;
        
    }

    
    // -----------------------------------------------------------------------------
    // Save reconstructed velocity volumes
    // -----------------------------------------------------------------------------

    inline void ReconstructionCardiacVelocity4D::SaveReconstructedVelocity4D(int iter)
    {
        char buffer[256];
        
        
        cout << " - Reconstructed velocity : " << endl;
        
        if (_reconstructed5DVelocity[0].GetT() == 1) {
        
            ImageAttributes attr = _reconstructed5DVelocity[0].Attributes();
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
                    sprintf(buffer,"../velocity-vector-final.nii.gz");
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
            
            ImageAttributes attr = _reconstructed5DVelocity[0].Attributes();
            RealImage output_sum(attr);
            output_sum = 0;
            
            for (int i=0; i<_reconstructed5DVelocity.size(); i++) {
                
                output_sum += _reconstructed5DVelocity[i];
                
                if (iter < 50)
                    sprintf(buffer,"velocity-%i-%i.nii.gz", i, iter);
                else
                    sprintf(buffer,"velocity-final-%i.nii.gz", i);
                
                _reconstructed5DVelocity[i].Write(buffer);
                cout << "     " << buffer << endl;
            }
            

            if (iter < 50)
                sprintf(buffer,"sum-velocity-%i.nii.gz", iter);
            else
                sprintf(buffer,"sum-velocity-final.nii.gz");
            
            output_sum.Write(buffer);
            
        }
        
        cout << ".............................................." << endl;
        cout << ".............................................." << endl;
    }
    
} // namespace svrtk
