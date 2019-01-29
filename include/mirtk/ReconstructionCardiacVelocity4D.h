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


namespace mirtk {



class ReconstructionCardiacVelocity4D : public ReconstructionCardiac4D 
{

protected: 

  // RealImage _reconstructed4DVelocity;
  RealImage _confidence_map_velocity;

  Array<Array<double> > _g_directions;
  Array<double> _g_values;
  Array<Array<double> > _v_directions;

  Array<RealImage> _simulated_velocity;


  // Reconstructed 4D cardiac cine velocity images (for X, Y and Z components)
  Array<RealImage> _reconstructed5DVelocity;
  Array<RealImage> _confidence_maps_velocity;

  

  // do we need it ?
  double _VENC; 

  // what is the correct value / units? 
  const double gamma = 42577; 


 public:
   
    ReconstructionCardiacVelocity4D();
    ~ReconstructionCardiacVelocity4D();


    void InitializeGradientMoments(Array<Array<double> > g_directions, Array<double> g_values);
    void InitializeVelocityVolumes();
    void RotateDirections(double &dx, double &dy, double &dz, int i);


    void GaussianReconstructionCardiacVelocity4D();    
    void SimulateSlicesCardiacVelocity4D();
    void SuperresolutionCardiacVelocity4D(int iter);
    void AdaptiveRegularizationCardiacVelocity4D(int iter, Array<RealImage>& originals);
    
    void FastGaussianReconstructionCardiacVelocity4D();

    void  MaskSlicesPhase();

    inline void SaveReconstructedVelocity4D(); 

   
    friend class ParallelSimulateSlicesCardiacVelocity4D;
    friend class ParallelSuperresolutionCardiacVelocity4D;
    friend class ParallelAdaptiveRegularization1CardiacVelocity4D;
    friend class ParallelAdaptiveRegularization2CardiacVelocity4D;

    friend class ParallelGaussianReconstructionCardiacVelocity4D;

    
};  // end of ReconstructionCardiacVelocity4D class definition



////////////////////////////////////////////////////////////////////////////////
// Inline/template definitions
////////////////////////////////////////////////////////////////////////////////


// -----------------------------------------------------------------------------
// ...
// -----------------------------------------------------------------------------

inline void ReconstructionCardiacVelocity4D::SaveReconstructedVelocity4D()
{

  char buffer[256];

  cout << ".............................................." << endl;
  cout << ".............................................." << endl;
  cout << "Reconstructed velocity files : " << endl;

  for (int i=0; i<_reconstructed5DVelocity.size(); i++) {

    sprintf(buffer,"velocity-%i.nii.gz",i);
    _reconstructed5DVelocity[i].Write(buffer);
    cout << " - " << buffer << endl;
  }

  cout << ".............................................." << endl;
  cout << ".............................................." << endl;


}


// -----------------------------------------------------------------------------
// ...
// -----------------------------------------------------------------------------


} // namespace mirtk


#endif // MIRTK_Reconstruction_H
