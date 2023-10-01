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

    class ReconstructionFFD: public Reconstruction {
    protected:


    public:
        // Constructor
        ReconstructionFFD() : Reconstruction() {
            _recon_type = _3D;
            _ffd = true;
            _ffd_global_only = false;
            
            _cp_spacing.push_back(15);
            _cp_spacing.push_back(10);
            _cp_spacing.push_back(5);
            _cp_spacing.push_back(5);
            _cp_spacing.push_back(5);
            
            _global_NCC_threshold = 0.6;
//            _local_SSIM_threshold = 0.4;
            _local_SSIM_window_size = 20;
            _global_JAC_threshold = 30;
            
            _global_cp_spacing = 10; 
            
            _local_SSIM_threshold = 0.3;
            
            _combined_rigid_ffd = false;
            
            _nmi_bins = -1;

        }

        // Destructor
        ~ReconstructionFFD() {}
        
        
        // Run FFD stack registrations
        void FFDStackRegistrations(Array<RealImage>& stacks, Array<Array<RealImage>>& mc_stacks, RealImage template_image, RealImage mask);
 

        // Access to Parallel Processing Classes

        ////////////////////////////////////////////////////////////////////////////////
        // Inline/template definitions
        ////////////////////////////////////////////////////////////////////////////////


    };  // end of ReconstructionFFD class definition

} // namespace svrtk
