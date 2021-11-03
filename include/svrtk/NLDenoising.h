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


#ifndef _MIRTK_NLDenoising_H
#define _MIRTK_NLDenoising_H

// MIRTK
#include "mirtk/BaseImage.h"
#include "mirtk/GenericImage.h"

// C++ Standard
#include <stdio.h>
#include <vector>
#include <pthread.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <float.h>

using namespace std;
using namespace mirtk;

namespace svrtk {

    typedef struct {
        int rows;
        int cols;
        int slices;
        double *in_image;
        double *means_image;
        double *var_image;
        double *estimate;
        double *label;
        double *bias;
        int ini;
        int fin;
        int radioB;
        int radioS;
        bool rician;
        double max_val;
    } myargument;

    class NLDenoising {
    public:
        NLDenoising() {}
        ~NLDenoising() {}

        RealImage Run(const RealImage& image);
        RealImage Run(const RealImage& image, int input_param_w, int input_param_f);
    };

#endif 
} // namespace svrtk
