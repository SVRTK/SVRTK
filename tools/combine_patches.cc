/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2008-2017 Imperial College London
 * Copyright 2018-2019 King's College London
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


#include "mirtk/Reconstruction.h"


using namespace mirtk;
using namespace std;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: combine_patches [target_volume] [output_resolution] [N] [stack_1] .. [stack_N] \n" << endl;
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
    argc--;
    argv++;
    cout<<"Original volume: "<<argv[1]<<endl;
    
    
    double output_resolution;
    
    //resolution
    output_resolution = atoi(argv[1]);
    argc--;
    argv++;
    cout<<"Output resolution : "<<output_resolution<<endl;
    
    
    //read number of stacks
    nStacks = atoi(argv[1]);
    argc--;
    argv++;
    cout<<"Number of stacks : "<<nStacks<<endl;
    
    
    
    
    //-------------------------------------------------------------------
    
    // Read stacks
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;

    for (int i=0; i<nStacks; i++) {
        
        stack_files.push_back(argv[1]);
        
        cout<<"Reading stack : "<<argv[1]<<endl;
        
        tmp_fname = argv[1];
        image_reader.reset(ImageReader::TryNew(tmp_fname));
        tmp_image.reset(image_reader->Run());
        
        stack = *tmp_image;
        
        double smin, smax;
        stack.GetMinMax(&smin, &smax);
        
        if (smin < 0 || smax < 0) {
            
            stack.PutMinMaxAsDouble(0, 1000);
        }
        
        argc--;
        argv++;
        stacks.push_back(stack);
    }
    
    
    
    RealImage output_volume = target_volume;
    target_volume = 0;

    InterpolationMode interpolation = Interpolation_Linear;
    UniquePtr<InterpolateImageFunction> interpolator;
    interpolator.reset(InterpolateImageFunction::New(interpolation));
    
    Resampling<RealPixel> resampler(output_resolution, output_resolution, output_resolution);
    resampler.Input(&output_volume);
    resampler.Output(&output_volume);
    resampler.Interpolator(interpolator.get());
    resampler.Run();
    
    //-------------------------------------------------------------------
    
    
    double wx, wy, wz;
    int rx, ry, rz;
    
    for (int i=0; i<stacks.size(); i++) {
        
        for (int z=0; z<stacks[i].GetZ(); z++) {
            
            for (int y=0; y<stacks[i].GetY(); y++) {
                for (int x=0; x<stacks[i].GetX(); x++) {
                    
                    wx = x;
                    wy = y;
                    wz = z;

                    stacks[i].ImageToWorld(wx, wy, wz);
                    output_volume.WorldToImage(wx, wy, wz);
                    
                    rx = round(wx);
                    ry = round(wy);
                    rz = round(wz);
                    
                    if (rx > -1 && ry > -1 && rz > -1 && rx < output_volume.GetX() && ry < output_volume.GetY() && rz < output_volume.GetZ() ) {
                    
                        output_volume(rx, ry, rz) = stacks[i](x, y, z);
                    }
                    
                }
            }
        }
        
        
    }
    
    cout << "---------------------------------------------------------------------" << endl;
    
    cout<<"Output volume : combined.nii.gz "<<endl;
    
    output_volume.Write("combined.nii.gz");
    
    cout << "---------------------------------------------------------------------" << endl;
    
    
    
        return 0;
    }

    
