/*
 */ 

#ifndef _MIRTK_MeanShift_H
#define _MIRTK_MeanShift_H

#include "mirtk/Common.h"
#include "mirtk/Options.h"

#include "mirtk/Array.h"
#include "mirtk/Point.h"

#include "mirtk/BaseImage.h"
#include "mirtk/GenericImage.h"

#include "mirtk/Dilation.h"
#include "mirtk/Erosion.h"
#include "mirtk/GaussianBlurring.h"
#include <algorithm>
#include <queue>

using namespace std; 

namespace mirtk {

    class MeanShift {
        
    protected: 
        int _nBins;
        int _padding;
        GreyImage _image;
        queue<Point> _q;
        GreyImage _map, _orig_image;
        GreyImage *_brain;
        GreyImage *_output;
        GreyPixel _imin, _imax;
        double _limit1, _limit2, _limit, _treshold;
        double _bin_width;
        double * _density;
        int _clusterSize;
    public:
        double _bg, _wm, _gm, _split1, _split2;


    public:

        MeanShift(GreyImage& image, int padding = -1, int nBins = 256);
        ~MeanShift();
        void SetOutput(GreyImage *_output);
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
        int LccS(int label, double treshold = 0.5);
        void RemoveBackground();
        void RegionGrowing();
        void FindWMGMmeans();
        void Write(char *output_name);
        void WriteMap(char *output_name);
        void SetTreshold();
        void SetTreshold(double treshold);

        RealImage ReturnMask();
    };

} // namespace mirtk

#endif  // MIRTK_MeanShift_H

