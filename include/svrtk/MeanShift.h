/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2008-2017 Imperial College London
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
#include "svrtk/Common.h"

using namespace std;
using namespace mirtk;

namespace svrtk {

    class MeanShift {
    protected:
        int _nBins;
        int _padding;
        RealImage _image;
        queue<Point> _q;
        RealImage _map, _orig_image;
        RealImage *_brain;
        RealImage *_output;
        RealPixel _imin, _imax;
        double _limit1, _limit2, _limit, _threshold;
        double _bin_width;
        double * _density;
        int _clusterSize;

    public:
        double _bg, _wm, _gm, _split1, _split2;

        MeanShift(const RealImage& image, int padding = -1, int nBins = 256);
        ~MeanShift();
        void SetOutput(RealImage *_output);
        double ValueToBin(double value);
        double BinToValue(int bin);
        void AddPoint(int x, int y, int z);
        void AddPoint(int x, int y, int z, int label);
        double msh(double y, double h);
        double findMax(double tr1, double tr2);
        double findMin(double tr1, double tr2);
        double findGMvar();
        double split(double pos1, double pos2, double bw, double h1, double h2);
        double GenerateDensity(double cut_off = 0.02);
        void Grow(int x, int y, int z, int label);
        int Lcc(int label, bool add_second = false);
        int LccS(int label, double threshold = 0.5);
        void RemoveBackground();
        void RegionGrowing();
        void FindWMGMmeans();
        void Write(char *output_name);
        void WriteMap(char *output_name);
        void SetThreshold();
        void SetThreshold(double threshold);
        RealImage ReturnMask();
    };

} // namespace svrtk
