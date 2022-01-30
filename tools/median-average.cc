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


// C++ Standard
#include <iostream>
#include <chrono>
#include <ctime>
#include <fstream>
#include <cmath>
#include <set>
#include <algorithm>
#include <thread>
#include <functional>
#include <vector>
#include <cstdlib>
#include <pthread.h>
#include <string>

#include "svrtk/Utility.h"

using namespace std;
using namespace mirtk;
using namespace svrtk;
using namespace svrtk::Utility;

// =============================================================================
//
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: mirtk median-average [reference_image] [output_image] [number_of_input_images] [input_1] ... [input_n] " << endl;
    cout << endl;
    cout << "Function for computing an median average from multiple input files in the reference space." << endl;
    cout << endl;
    cout << "\t" << endl;
    cout << "\t" << endl;
    
    exit(1);
}



double median_val(Array<double> in_vector)
{
  size_t size = in_vector.size();

  if (size == 0) {
    return 0;  
  } else {
    sort(in_vector.begin(), in_vector.end());
    if (size % 2 == 0) {
      return (in_vector[size / 2 - 1] + in_vector[size / 2]) / 2;
    } else {
      return in_vector[size / 2];
    }
  }
}



// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    char buffer[256];
    
    Array<RealImage> stacks;
    
    
    cout << "------------------------------------------------------" << endl;
    
    tmp_fname = argv[1];
    //image_reader.reset(ImageReader::TryNew(tmp_fname));
    //tmp_image.reset(image_reader->Run());
    
    RealImage target(tmp_fname); // = *tmp_image;
    
    argc--;
    argv++;
    
    cout << "Target space : " << tmp_fname << endl;
    
    
    cout << "------------------------------------------------------" << endl;
    
    const char *average_fname;
    average_fname = argv[1];
    argc--;
    argv++;
    
    
    cout << "Output median stack : " << average_fname << endl;
    
    ImageAttributes attr = target.Attributes();
    
    RealImage output_average_stack;
    output_average_stack.Initialize(attr);
    
    cout << "------------------------------------------------------" << endl;
    
    GenericLinearInterpolateImageFunction<RealImage> interpolator;
    double source_padding = 0;
    double target_padding = -inf;
    bool dofin_invert = false;
    bool twod = false;
    

    int nStacks = atoi(argv[1]);
    argc--;
    argv++;
    cout << "Number of stacks : " << nStacks << endl;
    cout << endl;
    
    for (int i=0; i<nStacks; i++) {
        
        cout<<" - stack " << i << " : " << argv[1] << endl;
        
        tmp_fname = argv[1];
        //image_reader.reset(ImageReader::TryNew(tmp_fname));
        //tmp_image.reset(image_reader->Run());
        
        RealImage tmp_stack(tmp_fname); // = *tmp_image;
        tmp_stack.PutMinMaxAsDouble(0, 1000);

        RealImage stack = target;
        
        RigidTransformation *r_tmp = new RigidTransformation;
        ImageTransformation *imagetransformation = new ImageTransformation;
        imagetransformation->Input(&tmp_stack);
        imagetransformation->Transformation(r_tmp);
        imagetransformation->Output(&stack);
        imagetransformation->TargetPaddingValue(target_padding);
        imagetransformation->SourcePaddingValue(source_padding);
        imagetransformation->Interpolator(&interpolator);
        imagetransformation->TwoD(twod);
        imagetransformation->Invert(dofin_invert);
        imagetransformation->Run();
        delete imagetransformation;

        stacks.push_back(stack);
        
        argc--;
        argv++;
        
    }
    
    cout << "------------------------------------------------------" << endl;
    
    cout << endl;
    cout << " ... " << endl;
    cout << endl;
    
    for (int z = 0; z < output_average_stack.GetZ(); z++) {
        for (int y = 0; y < output_average_stack.GetY(); y++) {
            for (int x = 0; x < output_average_stack.GetX(); x++) {

                Array<double> all_voxels;
                for (int s = 0; s < stacks.size(); s++) {
                    all_voxels.push_back(stacks[s](x,y,z));
                }

                output_average_stack(x,y,z) = median_val(all_voxels);

            }
        }
    }

    output_average_stack.Write(average_fname);

    cout << "------------------------------------------------------" << endl;
    


    return 0;
}




