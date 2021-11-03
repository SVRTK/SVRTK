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
#include "mirtk/Common.h"
#include "mirtk/Options.h"
#include "mirtk/Array.h"
#include "mirtk/Point.h"
#include "mirtk/BaseImage.h"
#include "mirtk/GenericImage.h"

// C++ Standard
#include <algorithm>
#include <queue>

using namespace std;
using namespace mirtk;

namespace svrtk {

    class SphericalHarmonics {
    public:
        SphericalHarmonics();
        ~SphericalHarmonics();
        
        Matrix _SHT;
        Matrix _iSHT;
        
        Matrix shTransform(Matrix directions, int lmax);
        void LegendrePolynomials(Vector& array, const int lmax, const int m, const double x);
        
        Matrix cartesian2spherical (Matrix xyz);
        
        Vector LSFit(Vector signal, Matrix dirs, int lmax);
        
        void InitSHT(Matrix dirs, int lmax);
        
        Matrix SHbasis(Matrix dirs, int lmax);
        
        Vector Coeff2Signal(Vector c);
        
        Vector Signal2Coeff(Vector s);
        
        RealImage Signal2Coeff(RealImage signal);
        
        RealImage Coeff2Signal(RealImage coeff);
        
        Matrix LaplaceBeltramiMatrix(int lmax);
        
        void InitSHTRegul(Matrix dirs, double lambda, int lmax);
        
        inline int NforL (int lmax);
        inline double LegendrePolynomialsHelper (const double x, const double m);
        inline int shIndex (int l, int m);
        
    };
    
    
    ////////////////////////////////////////////////////////////////////////////////
    // Inline/template definitions
    ////////////////////////////////////////////////////////////////////////////////
    
    // -----------------------------------------------------------------------------
    // ...
    // -----------------------------------------------------------------------------
    
    
    inline int SphericalHarmonics::NforL (int lmax)
    {
        return (lmax+1) * (lmax+2) /2;
    }
    
    inline double SphericalHarmonics::LegendrePolynomialsHelper (const double x, const double m)
    {
        return (m < 1.0 ? 1.0 : x * (m-1.0) / m * LegendrePolynomialsHelper(x, m-2.0));
    }
    
    inline int SphericalHarmonics::shIndex (int l, int m)
    {
        return l * (l+1) /2 + m;
    }

} // namespace svrtk
