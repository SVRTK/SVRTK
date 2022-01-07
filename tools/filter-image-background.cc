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
    cout << "Usage: mirtk filter-image-background [input_image] [output_image] [foreground_sigma] [background_sigma] \n" << endl;
    cout << endl;
    cout << "Function for filtering on the image background based on the difference between the outputs of gaussian blurring with differet kernels." << endl;
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
    cout<<"Input image: "<<argv[1]<<endl;
    argc--;
    argv++;

    double smin, smax;
    input_volume.GetMinMax(&smin, &smax);

    if (smin < 0 || smax < 0) {
        input_volume.PutMinMaxAsDouble(0, 1000);
    }


    output_name = argv[1];
    cout<<"Ouput image: "<<output_name<<endl;
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

    BackgroundFiltering(stacks, fg_sigma, bg_sigma);

    output_volume = stacks[0];

    output_volume.Write(output_name);


    cout << "---------------------------------------------------------------------" << endl;



    return 0;
}
