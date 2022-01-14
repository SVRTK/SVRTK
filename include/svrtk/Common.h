/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2021- King's College London
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
#include "mirtk/Common.h"
#include "mirtk/Options.h"
#include "mirtk/IOConfig.h"
#include "mirtk/Array.h"
#include "mirtk/Arith.h"
#include "mirtk/Math.h"
#include "mirtk/Point.h"
#include "mirtk/BaseImage.h"
#include "mirtk/GenericImage.h"
#include "mirtk/ImageReader.h"
#include "mirtk/Dilation.h"
#include "mirtk/Erosion.h"
#include "mirtk/GaussianBlurring.h"
#include "mirtk/Resampling.h"
#include "mirtk/ResamplingWithPadding.h"
#include "mirtk/LinearInterpolateImageFunction.hxx"
#include "mirtk/GaussianBlurringWithPadding.h"
#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageTransformation.h"
#include "mirtk/MultiLevelFreeFormTransformation.h"
#include "mirtk/FreeFormTransformation.h"
#include "mirtk/LinearFreeFormTransformation3D.h"
#include "mirtk/BSplineFreeFormTransformationSV.h"

// C++ Standard
#include <algorithm>
#include <pthread.h>
#include <queue>
#include <set>
#include <thread>

// Boost
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

// OpenMP
#include <omp.h>

// SVRTK
#include "svrtk/MeanShift.h"
#include "svrtk/NLDenoising.h"
#include "svrtk/SphericalHarmonics.h"
#include "svrtk/Utility.h"

namespace svrtk {
    struct POINT3D {
        short x;
        short y;
        short z;
        double value;
    };

    enum RECON_TYPE { _3D, _1D, _interpolate };

    typedef Array<POINT3D> VOXELCOEFFS;
    typedef Array<Array<VOXELCOEFFS>> SLICECOEFFS;

    /// PI
    constexpr double PI = 3.14159265358979323846;
}
