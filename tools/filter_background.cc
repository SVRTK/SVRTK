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
    cout << "Usage: filter_background [input_volume] [output_volume] [foreground_sigma] [background_sigma] \n" << endl;
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

    
    //if not enough arguments print help
    if (argc < 5)
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
    
    
    double bg_sigma, fg_sigma;
    
    
    fg_sigma = atof(argv[1]);
    cout<<"foreground sigma : "<<fg_sigma<<endl;
    argc--;
    argv++;
    
    bg_sigma = atof(argv[1]);
    cout<<"background sigma : "<<bg_sigma<<endl;
    argc--;
    argv++;
    
    
    //-------------------------------------------------------------------
    
    Reconstruction *reconstruction = new Reconstruction();
    
    Array<RealImage> stacks;
    stacks.push_back(input_volume);
    
    reconstruction->BackgroundFiltering(stacks, fg_sigma, bg_sigma);
    
    output_volume = stacks[0];
    
    output_volume.Write(output_name);
    
    
    cout << "---------------------------------------------------------------------" << endl;
    
    
    
    return 0;
}
