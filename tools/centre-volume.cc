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
#include "mirtk/Dilation.h"

// SVRTK
#include "svrtk/Utility.h"
#define SVRTK_TOOL

using namespace std;
using namespace mirtk;
using namespace svrtk;
using namespace svrtk::Utility;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------
void usage()
{
    cout << "Usage: mirtk centre-volume [input image] [mask] ... [output image] " << endl;
    cout << endl;
    cout << "Function that changes origin of the image with respect to the centre of the mask (world coordinates). ." << endl;
    cout << endl;
    cout << "\t" << endl;
    cout << "\t" << endl;

}
// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{

    char buffer[256];
    RealImage stack;

    char *output_name = NULL;

    RealImage input_stack, input_mask, output_stack;


    char *tmp_fname = NULL;
    UniquePtr<BaseImage> tmp_image;

    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();



    //read input name
    tmp_fname = argv[1];
    input_stack.Read(tmp_fname);
    argc--;
    argv++;

    //read mask name
    tmp_fname = argv[1];
    input_mask.Read(tmp_fname);
    argc--;
    argv++;

    //read output name
    output_name = argv[1];
    argc--;
    argv++;



    RigidTransformation *rigidTransf_mask = new RigidTransformation;
    TransformMask(input_stack, input_mask, *rigidTransf_mask);

    int sh = 0;

    double mask_centre_x = 0;
    double mask_centre_y = 0;
    double mask_centre_z = 0;
    int N = 0;

    for (int t = 0; t < input_stack.GetT(); t++) {
        for (int x = sh; x < input_stack.GetX()-sh; x++) {
           for (int y = sh; y < input_stack.GetY()-sh; y++) {
               for (int z = sh; z < input_stack.GetZ()-sh; z++) {

                   if (input_mask(x,y,z)>0.5) {

                       mask_centre_x = mask_centre_x + x;
                       mask_centre_y = mask_centre_y + y;
                       mask_centre_z = mask_centre_z + z;
                       N = N + 1;

                   }

               }
           }
        }
    }

    if (N > 0) {
        mask_centre_x = mask_centre_x / N;
        mask_centre_y = mask_centre_y / N;
        mask_centre_z = mask_centre_z / N;
    }

    input_stack.ImageToWorld(mask_centre_x, mask_centre_y, mask_centre_z);

    double org_centre_x, org_centre_y, org_centre_z;
    input_stack.GetOrigin(org_centre_x, org_centre_y, org_centre_z);


    double final_centre_x, final_centre_y, final_centre_z;

    final_centre_x = org_centre_x - mask_centre_x;
    final_centre_y = org_centre_y - mask_centre_y;
    final_centre_z = org_centre_z - mask_centre_z;



    input_stack.PutOrigin(final_centre_x, final_centre_y, final_centre_z);
    input_stack.Write(output_name);


    return 0;
}
