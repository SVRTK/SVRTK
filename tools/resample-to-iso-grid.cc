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
#include "mirtk/Dilation.h"
#include "mirtk/ImageReader.h"

// SVRTK
#include "svrtk/ReconstructionFFD.h"

using namespace std;
using namespace mirtk;
using namespace svrtk;
 
// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------
void usage()
{
    cout << "Usage: mirtk resample-to-iso-grid [input_image] [output_image] [mode: 2 for XY or 3 for XYZ]" << endl;
    cout << endl;
    cout << "Function for resampling images to the isotropic 2D XY or 3D XYZ grid (based on per voxel copying)." << endl;
    cout << "The grid size is computed as the largest along the X, Y or Z size." << endl;
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
    
    
    
    RealImage stack_in, stack_out;
    char * file_name = NULL;
    
    
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    
    
    file_name = argv[1];
    argc--;
    argv++;
    stack_in.Read(file_name);
    
    
    file_name = argv[1];
    argc--;
    argv++;
    
    
    int mode = atoi(argv[1]);
    argc--;
    argv++;
    
    
    int x = stack_in.GetX();
    int y = stack_in.GetY();
    int z = stack_in.GetZ();
    
    int shift_x = 0;
    int shift_y = 0;
    int shift_z = 0;
    
    int max = 0;
    
    if (x > y) {
        max = x;
        shift_y = floor(abs(max-y)/2);
    } else {
        max = y;
        shift_x = floor(abs(max-x)/2);
    }
    
    
    if (mode == 3) {
        if (z > max) {
            max = z;
            shift_x = floor(abs(max-x)/2);
            shift_y = floor(abs(max-y)/2);
        } else {
            shift_z = floor(abs(max-z)/2);
        }
    }
    
    
    
    
    ImageAttributes attr = stack_in.GetImageAttributes();
    double ox,oy,oz;
    stack_in.GetOrigin(ox,oy,oz);
    
    stack_out = stack_in;
    stack_out = 0;
    
    
    attr._x = max;
    attr._y = max;
    if (mode == 3)
    attr._z = max;
    stack_out.Initialize(attr);
    stack_out.PutOrigin(ox,oy,oz);
    stack_out = 0;
    
    
    for (int z=0; z<stack_in.GetZ(); z++) {
        for (int y=0; y<stack_in.GetY(); y++) {
            for (int x=0; x<stack_in.GetX(); x++) {
                stack_out(x+shift_x, y+shift_y, z+shift_z) = stack_in(x,y,z);
            }
        }
    }
    
    stack_out.Write(file_name);
    
    
    return 0;
}
