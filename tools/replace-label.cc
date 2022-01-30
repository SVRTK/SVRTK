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

using namespace mirtk;
using namespace std;
 
// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------
void usage()
{
    cout << "Usage: mirtk replate-label [input_parcellation] [reference_parcellation]  [output_parcellation] [label_number_in_reference] [label_number_to_replace_in_input] \n" << endl;
    cout << endl;
    cout << "Function for replacing one label to another based on the provided additional reference parcellation. " << endl;
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

    char buffer[256];
    RealImage stack;

    char *output_name = NULL;

    
    RealImage input_mask, output_mask, reference_mask;
    

    char *tmp_fname = NULL;
    UniquePtr<BaseImage> tmp_image;
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();


    tmp_fname = argv[1];
    input_mask.Read(tmp_fname);
    argc--;
    argv++;
    

    tmp_fname = argv[1];
    reference_mask.Read(tmp_fname);
    argc--;
    argv++;
    
    output_name = argv[1];
    argc--;
    argv++;
    
    
    int num_l_ref = atoi(argv[1]);
    
    argc--;
    argv++;
    
    int num_l_in = atoi(argv[1]);
    
    argc--;
    argv++;
    

    output_mask = input_mask;
 
    int sh = 1;

     for (int x = sh; x < input_mask.GetX()-sh; x++) {
        for (int y = sh; y < input_mask.GetY()-sh; y++) {
            for (int z = sh; z < input_mask.GetZ()-sh; z++) {

                if ( reference_mask(x,y,z) == num_l_ref && input_mask(x,y,z) == num_l_in ) {
                    output_mask(x,y,z) = num_l_ref;
                }
 
            }
        }
    }

    
    output_mask.Write(output_name);

    
    return 0;
    
}


