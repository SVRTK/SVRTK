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
#include "mirtk/ResamplingWithPadding.h"
#include "mirtk/GenericImage.h"
#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/LinearInterpolateImageFunction.hxx"
#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageReader.h"

using namespace std;
using namespace mirtk;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: mirtk combine-masks [reference_image] [input_1] ... [input_n] [output] " << endl;
    cout << endl;
    cout << "Function for computing an average mask from multiple input files in the reference space (transferred from IRTK library: https://biomedia.doc.ic.ac.uk/software/irtk/)." << endl;
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
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();


    // Determine how many volumes we have
    int number_of_volumes = argc-3;

    if (number_of_volumes < 1) usage();

    cout << "Combine images from " << number_of_volumes << " images" << endl;

    RealImage reference;
    reference.Read(argv[1]);

    double xsize, ysize, zsize, size;
    reference.GetPixelSize(&xsize, &ysize, &zsize);
    size = xsize;
    size = (size < ysize) ? size : ysize;
    size = (size < zsize) ? size : zsize;

    ResamplingWithPadding<RealPixel> resampling(size, size, size,-1);
    GenericLinearInterpolateImageFunction<RealImage> interpolator;

    resampling.Input(&reference);
    resampling.Output(&reference);
    resampling.Interpolator(&interpolator);
    resampling.Run();


    RealImage output_volume = reference;
    output_volume = 0;

    Array<GreyImage> input_volumes;

    // Read remaining images
    for (int i = 0; i < number_of_volumes; i++) {

        cout << "Reading " << argv[i+2] << endl;
        GreyImage stack;
        stack.Read(argv[i+2]);
        input_volumes.push_back(stack);
    }


    double wx, wy, wz;
    int rx, ry, rz;
    double val, num;

    for (int z=0; z<output_volume.GetZ(); z++) {
        for (int y=0; y<output_volume.GetY(); y++) {
            for (int x=0; x<output_volume.GetX(); x++) {

                val = 0;
                num = 0;

                for (int i=0; i<input_volumes.size(); i++) {

                    wx = x;
                    wy = y;
                    wz = z;

                    output_volume.ImageToWorld(wx, wy, wz);
                    input_volumes[i].WorldToImage(wx, wy, wz);

                    rx = round(wx);
                    ry = round(wy);
                    rz = round(wz);

                    if (input_volumes[i].IsInside(rx, ry, rz)) {
                        if (input_volumes[i](rx, ry, rz)>0) {
                            val += input_volumes[i](rx, ry, rz);
                            num++;
                        }
                    }
                }

                if (num > 1)
                    val = 1; //val/num;
                else
                    val = -1;

                output_volume(x, y, z) = val;

            }
        }
    }




    cout << "---------------------------------------------------------------------" << endl;

    cout<<"Output image : " << argv[number_of_volumes+2] <<endl;
    output_volume.Write(argv[number_of_volumes+2]);

    cout << "---------------------------------------------------------------------" << endl;



    return 0;
}
