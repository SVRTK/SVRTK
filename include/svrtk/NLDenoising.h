/*=========================================================================

 The code was adopted from : https://github.com/djkwon/naonlm3d
 originally developed by Jose V. Manjon, Pierrick Coupe and Dongjin Kwon

 based on:
 "Adaptive Non-Local Means Denoising of MR Images
 With Spatially Varying Noise Levels"

 Jose V. Manjon, Pierrick Coupe, Luis Marti-Bonmati,
 D. Louis Collins and Montserrat Robles

 J Magn Reson Imaging. 2010 Jan;31(1):192-203.

 =========================================================================*/

#pragma once

// MIRTK
#include "mirtk/BaseImage.h"
#include "mirtk/GenericImage.h"

using namespace std;
using namespace mirtk;

namespace svrtk {

    class NLDenoising {
    public:
        static RealImage Run(const RealImage& image);
        static RealImage Run(const RealImage& image, int input_param_w, int input_param_f);
    };

} // namespace svrtk
