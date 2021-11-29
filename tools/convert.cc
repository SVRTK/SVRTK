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
#include "mirtk/IOConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/ImageReader.h"

using namespace std;
using namespace mirtk;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: mirtk convert [input_image] [reference_image] [output_image] [padding_value] " << endl;
    cout << endl;
    cout << "Function for conversion between different .nii file formats based on per-voxel copying." << endl;
    cout << endl;
    cout << "\t" << endl;
    cout << "\t" << endl;
    exit(0);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{

    cout << "---------------------------------------------------------------------" << endl;

    if (argc < 5 || argc > 5)
        usage();

    RealImage stack_in, stack_ref, stack_out;
    char * file_name = NULL;


    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();



    file_name = argv[1];
    argc--;
    argv++;
    cout << "Input image : " << file_name << endl;
    stack_in.Read(file_name);



    file_name = argv[1];
    argc--;
    argv++;
    cout << "Reference image : " << file_name << endl;
    stack_ref.Read(file_name);




    file_name = argv[1];
    argc--;
    argv++;
    cout << "Output image : " << file_name << endl;



    double padding_val = atof(argv[1]);
    argc--;
    argv++;
    cout << "Padding value : " << padding_val << endl;


    stack_out = stack_ref;
    stack_out = padding_val;


    double wx, wy, wz;
    int ix, iy, iz;
    double val;

    for (int t=0; t<stack_out.GetT(); t++) {
        for (int z=0; z<stack_out.GetZ(); z++) {
            for (int y=0; y<stack_out.GetY(); y++) {
                for (int x=0; x<stack_out.GetX(); x++) {

                    wx = x;
                    wy = y;
                    wz = z;
                    stack_ref.ImageToWorld(wx, wy, wz);
                    stack_in.WorldToImage(wx, wy, wz);
                    ix = round(wx);
                    iy = round(wy);
                    iz = round(wz);

                    if (stack_in.IsInside(ix, iy, iz))
                        val = stack_in(ix, iy, iz, t);
                    else
                        val = padding_val;

                    stack_ref(x, y, z, t) = val;


                }
            }
        }
    }

    stack_ref.Write(file_name);

    cout << "---------------------------------------------------------------------" << endl;

    return 0;
}
