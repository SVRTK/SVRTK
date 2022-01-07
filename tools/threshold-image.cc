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

using namespace std;
using namespace mirtk;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: mirtk threshold-image [input_image] [output_image] [threshold] \n" << endl;
    cout << endl;
    cout << "Function for binary thresholding: the output will be in [0; 1] range. "<< endl;
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

    
    //if not enough arguments print help
    if (argc < 4)
    usage();
    
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    
    //-------------------------------------------------------------------
    
    RealImage input_volume, output_volume;

    
    input_volume.Read(argv[1]);
    cout<<"Input volume: "<<argv[1]<<endl;
    argc--;
    argv++;
    
    output_name = argv[1];
    cout<<"Ouput volume: "<<output_name<<endl;
    argc--;
    argv++;
    
    
    double threshold = 0;
    
    
    threshold = atof(argv[1]);
    cout<<"threshold : "<<threshold<<endl;
    argc--;
    argv++;
    
    
    //-------------------------------------------------------------------
    
    
    output_volume = input_volume;
    
    for (int t=0; t<output_volume.GetT(); t++) {
        for (int z=0; z<output_volume.GetZ(); z++) {
            for (int y=0; y<output_volume.GetY(); y++) {
                for (int x=0; x<output_volume.GetX(); x++) {
    
                    if (output_volume(x,y,z,t) < threshold) {
                        output_volume(x,y,z,t) = 0;
                    } else {
                        output_volume(x,y,z,t) = 1;
                    }
    
                }
            }
        }
    }
                                     
    output_volume.Write(output_name);
    
    
    cout << "---------------------------------------------------------------------" << endl;
    
    
    
    return 0;
}
