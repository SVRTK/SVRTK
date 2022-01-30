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

#include "mirtk/IOConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/ImageReader.h"


using namespace mirtk;
using namespace std;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: nan [stack_name] [threshold] \n" << endl;
    cout << endl;
    cout << "Function for setting voxels with large (abs) and NaN values to 0." << endl;
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
    
    
    if (argc < 3)
        usage();
    
    
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();


    //-------------------------------------------------------------------
   
    RealImage main_stack;

    tmp_fname = argv[1];
    image_reader.reset(ImageReader::TryNew(tmp_fname));
    tmp_image.reset(image_reader->Run());
        
    main_stack = *tmp_image;
    
    
    argc--;
    argv++;
    
    double lower_threshold = 0;
    lower_threshold = atof(argv[1]);
    argc--;
    argv++;
    
    
    for (int z = 0; z < main_stack.GetZ(); z++) {
        
        for (int y = 0; y < main_stack.GetY(); y++) {
            
            for (int x = 0; x < main_stack.GetX(); x++) {
                
                for (int t = 0; t < main_stack.GetT(); t++) {
                
                    if (main_stack(x,y,z,t) != main_stack(x,y,z,t) || abs(main_stack(x,y,z,t)) > lower_threshold  || main_stack(x,y,z,t) < 0) {
                        main_stack(x,y,z,t) = 0;
                    }
                                    
                }
            }
        }
    }

    
    main_stack.Write(tmp_fname);
    

    //-------------------------------------------------------------------

    
    return 0;
}




