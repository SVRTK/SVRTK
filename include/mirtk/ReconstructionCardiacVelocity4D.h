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

  Array<Array<RealImage> > _globalReconstructed4DVelocityArray;
  Array<RealImage> _globalReconstructed4DArray;
  

  // do we need it ?
  double _VENC; 


  double _velocity_scale;

  // what is the correct value / units? 
  const double gamma = 42.577; 


 public:
   
    ReconstructionCardiacVelocity4D();
    ~ReconstructionCardiacVelocity4D();


    void InitializeGradientMoments(Array<Array<double> > g_directions, Array<double> g_values);
    void InitializeVelocityVolumes();
    void RotateDirections(double &dx, double &dy, double &dz, int i);


    void GaussianReconstructionCardiacVelocity4D();    
    void SimulateSlicesCardiacVelocity4D();
    void SuperresolutionCardiacVelocity4D(int iter);
    void AdaptiveRegularizationCardiacVelocity4D(int iter, Array<RealImage>& originals, RealImage& original_main);
    
    void GaussianReconstructionCardiacVelocity4Dx3();
    void GaussianReconstructionCardiacVelocity4DxT();
    void GaussianReconstructionCardiac4DxT();

    void  MaskSlicesPhase();

    inline void SaveReconstructedVelocity4D(); 
    inline void SetVelocityScale(double scale);

   
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

inline void ReconstructionCardiacVelocity4D::SetVelocityScale(double scale)
{
  _velocity_scale = scale;
}

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

    RealImage scaled =  _reconstructed5DVelocity[i];
    scaled *= _velocity_scale;
    sprintf(buffer,"scaled-velocity-%i.nii.gz",i);
    scaled.Write(buffer);

  }

  cout << ".............................................." << endl;
  cout << ".............................................." << endl;

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

    sprintf(buffer,"subtracted-velocity-%i.nii.gz",i);
    subtracted4D.Write(buffer);

  }

}


// -----------------------------------------------------------------------------
// ...
// -----------------------------------------------------------------------------


} // namespace mirtk


#endif // MIRTK_Reconstruction_H
