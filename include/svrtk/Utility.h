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
#include "mirtk/GenericImage.h"
#include "mirtk/ImageTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/LinearInterpolateImageFunction.h"
#include "mirtk/GenericRegistrationFilter.h"

// SVRTK
#include "svrtk/NLDenoising.h"

// Boost
#include <boost/format.hpp>

using namespace std;
using namespace mirtk;

namespace svrtk::Utility {

    // Clear and preallocate memory for a vector
    inline void ClearAndReserve(vector<auto>& vectorVar, size_t reserveSize) {
        vectorVar.clear();
        vectorVar.reserve(reserveSize);
    }

    //-------------------------------------------------------------------

    // Clear and resize memory for vectors
    template<typename VectorType>
    inline void ClearAndResize(vector<VectorType>& vectorVar, size_t reserveSize, const VectorType& defaultValue = VectorType()) {
        vectorVar.clear();
        vectorVar.resize(reserveSize, defaultValue);
    }

}
