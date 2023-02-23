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

// MIRTK
#include "mirtk/Common.h"
#include "mirtk/Options.h"
#include "mirtk/NumericsConfig.h"
#include "mirtk/IOConfig.h"
#include "mirtk/TransformationConfig.h"
#include "mirtk/RegistrationConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageReader.h"

// SVRTK
#include "svrtk/Reconstruction.h"

using namespace std;
using namespace mirtk;
using namespace svrtk;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: mirtk combine-patches [reference_image] [output_resolution] [N] [stack_1] .. [stack_N] " << endl;
    cout << endl;
    cout << "Function for mapping a set of input image patches on the common reference images space." << endl;
    cout << endl;
    cout << "\t" << endl;
    cout << "\t" << endl;

    exit(1);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{

    cout << "---------------------------------------------------------------------" << endl;

    char buffer[256];
    RealImage stack;
    char * output_name = NULL;
    /// Slice stacks
    Array<RealImage> stacks;
    Array<string> stack_files;


    //if not enough arguments print help
    if (argc < 5)
    usage();


    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();


    //-------------------------------------------------------------------

    RealImage target_volume;

    int nStacks;

    //read name of the target volume
    target_volume.Read(argv[1]);
    cout<<"Original reference image: "<<argv[1]<<endl;
    argc--;
    argv++;



    double output_resolution;

    //resolution
    output_resolution = atof(argv[1]);
    cout<<"Output resolution : "<<output_resolution<<endl;
    argc--;
    argv++;



    //read number of stacks
    nStacks = atoi(argv[1]);
    cout<<"Number of images : "<<nStacks<<endl;
    argc--;
    argv++;

    //-------------------------------------------------------------------

    // Read stacks
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;

    for (int i=0; i<nStacks; i++) {

        stack_files.push_back(argv[1]);

        cout<<"Reading image : "<<argv[1]<<endl;

        tmp_fname = argv[1];
        image_reader.reset(ImageReader::TryNew(tmp_fname));
        tmp_image.reset(image_reader->Run());

        stack = *tmp_image;

        double smin, smax;
        stack.GetMinMax(&smin, &smax);

        // if (smin < 0 || smax < 0) {

        //     stack.PutMinMaxAsDouble(0, 1000);
        // }

        argc--;
        argv++;
        stacks.push_back(stack);
    }


    InterpolationMode interpolation = Interpolation_Linear;
    UniquePtr<InterpolateImageFunction> interpolator;
    interpolator.reset(InterpolateImageFunction::New(interpolation));

    Resampling<RealPixel> resampler(output_resolution, output_resolution, output_resolution);
    resampler.Interpolator(interpolator.get());

    resampler.Input(&target_volume);
    resampler.Output(&target_volume);
    resampler.Run();




    RealImage output_volume;
    output_volume.Initialize(target_volume.Attributes());


    //-------------------------------------------------------------------

    double patch_resolution = output_resolution;

    Resampling<RealPixel> resampler2(patch_resolution, patch_resolution, patch_resolution);
    resampler2.Interpolator(interpolator.get());

    for (int i=0; i<stacks.size(); i++) {

        resampler2.Input(&stacks[i]);
        resampler2.Output(&stacks[i]);
        resampler2.Run();
    }


    //-------------------------------------------------------------------


    double wx, wy, wz;
    int rx, ry, rz;

    RealImage weights = output_volume;
    weights = 0;
    output_volume = 0;

    double val, num;

    for (int z=0; z<output_volume.GetZ(); z++) {
        for (int y=0; y<output_volume.GetY(); y++) {
            for (int x=0; x<output_volume.GetX(); x++) {

                val = 0;
                num = 0;

                for (int i=0; i<stacks.size(); i++) {

                    wx = x;
                    wy = y;
                    wz = z;

                    output_volume.ImageToWorld(wx, wy, wz);
                    stacks[i].WorldToImage(wx, wy, wz);

                    rx = round(wx);
                    ry = round(wy);
                    rz = round(wz);

                    if (stacks[i].IsInside(rx, ry, rz)) {
                        val += stacks[i](rx, ry, rz);
                        num++;
                    }
                }

                if (num >0)
                val = val/num;

                output_volume(x, y, z) = val;
                weights(x, y, z) = num;


            }
        }
    }




    cout << "---------------------------------------------------------------------" << endl;

    cout<<"Output image : combined.nii.gz "<<endl;

    output_volume.Write("combined.nii.gz");

    weights.Write("overlap-weights.nii.gz");

    cout << "---------------------------------------------------------------------" << endl;



    return 0;
}
