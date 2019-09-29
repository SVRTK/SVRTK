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


using namespace mirtk;
using namespace std;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: pad_image [input_A_volume] [input_B_volume] [output_volume] [Value in imageB] [Padding Value in output] \n" << endl;
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
    if (argc < 6)
    usage();
    
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    
    //-------------------------------------------------------------------
    
    RealImage input_volume_A, input_volume_B, output_volume;

    
    input_volume_A.Read(argv[1]);
    cout<<"Input volume A: "<<argv[1]<<endl;
    argc--;
    argv++;
    
    
    input_volume_B.Read(argv[1]);
    cout<<"Input volume B: "<<argv[1]<<endl;
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
    
    double padding = 0;
    padding = atof(argv[1]);
    cout<<"padding : "<<padding<<endl;
    argc--;
    argv++;
    
    
    
    //-------------------------------------------------------------------
    
    
    output_volume = input_volume_A;
    
    int invert = 0;
    
    for (int t = 0; t < output_volume.GetT(); t++) {
        for (int z = 0; z < output_volume.GetZ(); z++) {
            for (int y = 0; y < output_volume.GetY(); y++) {
                for (int x = 0; x < output_volume.GetX(); x++) {
                    
                    double i = x;
                    double j = y;
                    double k = z;
                    input_volume_A.ImageToWorld(i,j,k);
                    input_volume_B.WorldToImage(i,j,k);
                    i = round(i);
                    j = round(j);
                    k = round(k);
                    
                    if(i >= 0 && i < input_volume_B.GetX() && j >= 0 && j < input_volume_B.GetY() && k >= 0 && k < input_volume_B.GetZ() && t >= 0 && t < input_volume_B.GetT()){
                        if(invert){
                            if (input_volume_B(i, j, k, t) != threshold) output_volume(x, y, z, t) = padding;
                        }else{
                            if (input_volume_B(i, j, k, t) == threshold) output_volume(x, y, z, t) = padding;
                        }
                    }
                }
            }
        }
    }
    
                                     
    output_volume.Write(output_name);
    
    
    cout << "---------------------------------------------------------------------" << endl;
    
    
    
    return 0;
}
