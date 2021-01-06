/*
* SVRTK : SVR reconstruction based on MIRTK
*
* Copyright 2018-2021 King's College London
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


#include "mirtk/Common.h"
#include "mirtk/Options.h" 

#include "mirtk/IOConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/ImageReader.h"


#include "mirtk/Resampling.h"
#include "mirtk/ResamplingWithPadding.h"
#include "mirtk/LinearInterpolateImageFunction.hxx"
#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageTransformation.h"
#include "mirtk/MultiLevelFreeFormTransformation.h"
#include "mirtk/FreeFormTransformation.h"
#include "mirtk/LinearFreeFormTransformation3D.h"


using namespace mirtk; 
using namespace std;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: pad-3d [input] [output] [isotropic-3d-voxel-grid-size] [interpolation mode: 0 - NN (for masks), 1 - Linear (for images)] \n" << endl;
    exit(0);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    
     //if not enough arguments print help
    if (argc < 5)
    usage();
    
    
    
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();


    //-------------------------------------------------------------------
   
    RealImage main_stack;

    tmp_fname = argv[1];
    image_reader.reset(ImageReader::TryNew(tmp_fname));
    tmp_image.reset(image_reader->Run());
    main_stack = *tmp_image;
    argc--;
    argv++;
    
    
    const char *out_fname;
    out_fname = argv[1];
    argc--;
    argv++;
    
    
    int grid_dim = 128;
    grid_dim = atoi(argv[1]);
    argc--;
    argv++;
    
    
    int lin_interp_mode = 1;
    lin_interp_mode = atoi(argv[1]);
    argc--;
    argv++;
    
    
    
    int x_max = main_stack.GetX();
    int y_max = main_stack.GetY();
    int z_max = main_stack.GetZ();
    
    int dim_max = x_max;
    if (y_max > dim_max)
        dim_max = y_max;
    if (z_max > dim_max)
        dim_max = z_max;
    
    
    double new_res = 1.05*(main_stack.GetXSize() * dim_max)/grid_dim;
    

    InterpolationMode interpolation = Interpolation_Linear;
    UniquePtr<InterpolateImageFunction> interpolator;
    interpolator.reset(InterpolateImageFunction::New(interpolation));
    
    if (lin_interp_mode < 1) {
        InterpolationMode interpolation_nn = Interpolation_NN;
        interpolator.reset(InterpolateImageFunction::New(interpolation_nn));
    }
    

    RealImage res_stack;
    Resampling<RealPixel> resampler(new_res,new_res,new_res);
    resampler.Input(&main_stack);
    resampler.Output(&res_stack);
    resampler.Interpolator(interpolator.get());
    resampler.Run();
    
    
    ImageAttributes attr = res_stack.GetImageAttributes();
    attr._x = grid_dim;
    attr._y = grid_dim;
    attr._z = grid_dim;
    
    RealImage fin_stack(attr);
    
    
    
    
    double source_padding = 0;
    double target_padding = -inf;
    bool dofin_invert = false;
    bool twod = false;
    
    
    
    MultiLevelFreeFormTransformation *mffd_init = new MultiLevelFreeFormTransformation;
    
    ImageTransformation *imagetransformation = new ImageTransformation;
    imagetransformation->Input(&main_stack);
    imagetransformation->Transformation(mffd_init);
    imagetransformation->Output(&fin_stack);
    imagetransformation->TargetPaddingValue(target_padding);
    imagetransformation->SourcePaddingValue(source_padding);
    imagetransformation->Interpolator(interpolator.get());  // &nn);
    imagetransformation->TwoD(twod);
    imagetransformation->Invert(dofin_invert);
    imagetransformation->Run();
    

    
    fin_stack.Write(out_fname);
    
    
    
    return 0;
}



