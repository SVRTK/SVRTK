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


  // ...

 public:
   
   // Constructor
   ReconstructionCardiacVelocity4D();
   
   // Destructor
   ~ReconstructionCardiacVelocity4D();


   void SimulateSlicesCardiacVelocity4D();


   void InitGradientMoments( Array< Array<double> > gm_dirs, Array<double> gm_vals );


   friend class ParallelSimulateSlicesCardiacVelocity4D;
   
   
   
};  // end of ReconstructionCardiacVelocity4D class definition






////////////////////////////////////////////////////////////////////////////////
// Inline/template definitions
////////////////////////////////////////////////////////////////////////////////


// -----------------------------------------------------------------------------
// ...
// -----------------------------------------------------------------------------





} // namespace mirtk


#endif // MIRTK_Reconstruction_H
