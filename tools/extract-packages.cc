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

#include "mirtk/Common.h"
#include "mirtk/Options.h" 

#include "mirtk/IOConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/ImageReader.h"

#include "mirtk/ReconstructionFFD.h"

using namespace mirtk; 
using namespace std;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: extract-packages [stack_name] [number_of_packages] \n" << endl;
    exit(0);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    
    
    if (argc != 3)
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
    
    int number_of_packages = atoi(argv[1]);


    //-------------------------------------------------------------------
    
    
    if (number_of_packages > 1) {
        
        ReconstructionFFD reconstruction;
        
        
        Array<RealImage> packages;
        reconstruction.SplitImage(main_stack, number_of_packages, packages);
        
        string org_name(tmp_fname);
        std::size_t pos = org_name.find(".nii");
        std::string main_name = org_name.substr (0, pos);
        std::string end_name = org_name.substr (pos, org_name.length());

        for (int t=0; t<number_of_packages; t++) {
        
            std::string stack_index = "-p" + to_string(t);
            string new_name = main_name + stack_index + end_name;
            char *c_new_name = &new_name[0];
            
            RealImage stack = packages[t];

            stack.Write(c_new_name);
            cout << c_new_name << endl;
            
        }

    }
    

    //-------------------------------------------------------------------

    
    return 0;
}



