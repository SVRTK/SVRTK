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
    cout << "Usage: mirtk enlarge-image [input_image] [output_image] <-x voxels> <-y voxels> <-z voxels> <-percent> <-value value>" <<endl;
    cout << endl;
    cout << "Function for changing the image grid size with padding (transferred from IRTK library: https://biomedia.doc.ic.ac.uk/software/irtk/)." << endl;
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


    if (argc < 3) {
        usage();
        exit(1);
    }

    int ex=0,ey=0,ez=0,percent=0,value=0;
    int ok;

    char *output_name;
    RealImage input_volume, output_volume;


    input_volume.Read(argv[1]);
    cout<<"Input volume: "<<argv[1]<<endl;
    argc--;
    argv++;

    output_name = argv[1];
    cout<<"Ouput volume: "<<output_name<<endl;
    argc--;
    argv++;


    // Parse remaining parameters
    while (argc > 1){
        ok = false;
        if ((ok == false) && (strcmp(argv[1], "-x") == 0)){
            argc--;
            argv++;
            ex = atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-y") == 0)){
            argc--;
            argv++;
            ey = atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-z") == 0)){
            argc--;
            argv++;
            ez = atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-percent") == 0)){
            argc--;
            argv++;
            percent = 1;
            ok = true;
        }
        if ((ok == false) && (strcmp(argv[1], "-value") == 0)){
            argc--;
            argv++;
            value = atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        if (ok == false){
            cerr << "Can not parse argument " << argv[1] << endl;
            usage();
        }
    }


    ImageAttributes attr = input_volume.Attributes();
    if(percent == 1){
        ex = round(double(input_volume.GetX()*ex)/100.0);
        ey = round(double(input_volume.GetY()*ey)/100.0);
        ez = round(double(input_volume.GetZ()*ez)/100.0);
    }
    attr._x += 2*ex;
    attr._y += 2*ey;
    attr._z += 2*ez;

    output_volume.Initialize(attr);

    output_volume = 0;

    for(int l=0;l<input_volume.GetT();l++)
    for(int i=0; i<input_volume.GetX();i++)
    for(int j=0; j<input_volume.GetY();j++)
    for(int k=0; k<input_volume.GetZ();k++)
    {
        output_volume.Put(i+ex,j+ey,k+ez,l,input_volume(i,j,k,l));
    }

    output_volume.Write(output_name);


    cout << "---------------------------------------------------------------------" << endl;



    return 0;
}
