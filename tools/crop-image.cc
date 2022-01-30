/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2018- King's College London
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

#include "svrtk/Utility.h"

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
    cout << "Usage: mirtk crop-image [input_image] [input_mask] [output_image] " << endl;
    cout << endl;
    cout << "Function for cropping images based on the input mask (the mask does not have to be binary / have the same orientation and size as the input image)." << endl;
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

    output_stack = input_stack;

    
    RigidTransformation *rigidTransf_mask = new RigidTransformation;
    TransformMask(input_stack, input_mask, *rigidTransf_mask);
    
    
    RealPixel smin, smax;
    input_mask.GetMinMax(&smin, &smax);
    
    
    if (smax > 0.5) {
        CropImage(output_stack,input_mask);
    } else {
        cout << "Warning: zero mask ..." << endl;
    }
    
    output_stack.Write(output_name);

    
    return 0;
}
