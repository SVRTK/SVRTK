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
#include "mirtk/Dilation.h"

// SVRTK
#include "svrtk/ReconstructionFFD.h"

// C++ Standard
#include <set>

using namespace std;
using namespace mirtk;
using namespace svrtk;
 
// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------
void usage()
{
    cout << "Usage: mirtk image-label-statistics [input_label_mask] [start_label_number] [end_label_number] [std_threshold_for_mean] [number_of_images] [image_1] ... [image_n]  " << endl;
    cout << endl;
    cout << "Function for calculating label-based statistics (from start to end label ie) in an array of images: mask volume in mm3, mean image intensity value within the label ROI, intensity st.dev within the label ROI. " << endl;
    cout << "The [std_threshold_for_mean] value specifies the st.dev factor for computing the mean intensity value within the label ROI." << endl;
    cout << "The outputs are printed and saved as .csv files: stats-volume.csv, stats-mean.csv, stats-stdev.csv. " << endl;
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
    
    char *output_name = NULL;
    
    ReconstructionFFD *reconstruction = new ReconstructionFFD();
    
    RealImage input_stack, input_mask;
    
    
    char *tmp_fname = NULL;
    UniquePtr<BaseImage> tmp_image;
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    RealImage org_input_mask;
    
    
    tmp_fname = argv[1];
    org_input_mask.Read(tmp_fname);
    argc--;
    argv++;
    
    input_mask = org_input_mask;
    
    int num_l_start = 1;
    int num_l_stop = 2;
    
    num_l_start = atoi(argv[1]);
    argc--;
    argv++;
    
    
    num_l_stop = atoi(argv[1]);
    argc--;
    argv++;
    
    
    double std_dev_limit = 1.5;
    std_dev_limit = atof(argv[1]);
    argc--;
    argv++;
    
    

    
    
    Array<RealImage> stacks;
    Array<string> stack_files;
    RealImage stack;
    
    
    int nStacks = atoi(argv[1]);
    argc--;
    argv++;
    
    
    for (int i = 0; i < nStacks; i++) {
        
        stack_files.push_back(argv[1]);
        
        cout<<" - " << argv[1] << endl;
        
        tmp_fname = argv[1];
        image_reader.reset(ImageReader::TryNew(tmp_fname));
        tmp_image.reset(image_reader->Run());
        
        stack = *tmp_image;
        
        argc--;
        argv++;
        stacks.push_back(stack);
    }
    
    
    
    sprintf(buffer, "stats-volume.csv");
    ofstream info_volume;
    info_volume.open( buffer );
    
    
    sprintf(buffer, "stats-mean.csv");
    ofstream info_mean;
    info_mean.open( buffer );
    
    
    sprintf(buffer, "stats-stdev.csv");
    ofstream info_stddev;
    info_stddev.open( buffer );
    
    
    info_mean << "id" << ",";
    info_stddev << "id" << ",";
    info_volume << "id" << ",";
    
    for (int num_l = num_l_start; num_l < (num_l_stop+1); num_l++) {
        
        info_mean << "l" << num_l << ",";
        info_stddev << "l" << num_l << ",";
        info_volume << "l" << num_l << ",";
        
    }
    
    info_mean << endl;
    info_stddev << endl;
    info_volume << endl;
    
    
    
    for (int s = 0; s < stacks.size(); s++) {
        
        
        cout << " - " << s << endl;
        
        input_stack = stacks[s];
        
        input_mask = org_input_mask;
        
        RigidTransformation *rigidTransf_mask = new RigidTransformation;
        reconstruction->TransformMask(input_stack, input_mask, *rigidTransf_mask);
        
        info_volume << stack_files[s] << ",";
        info_mean << stack_files[s] << ",";
        info_stddev << stack_files[s] << ",";
        
        
        for (int num_l = num_l_start; num_l < (num_l_stop+1); num_l++) {
            
            
            
            double mask_volume = 0;
            int mask_count = 0;
            
            double mean_value = 0;
            
            for (int x = 0; x < input_stack.GetX(); x++) {
                for (int y = 0; y < input_stack.GetY(); y++) {
                    for (int z = 0; z < input_stack.GetZ(); z++) {
                        
                        if (input_mask(x,y,z) == num_l ) {
                            mask_count+=1;
                            mean_value = mean_value + input_stack(x,y,z);
                        }
                        
                    }
                }
            }
            
            mean_value = mean_value / mask_count;
            
            double stddev_value = 0;
            
            for (int x = 0; x < input_stack.GetX(); x++) {
                for (int y = 0; y < input_stack.GetY(); y++) {
                    for (int z = 0; z < input_stack.GetZ(); z++) {
                        
                        if (input_mask(x,y,z) == num_l ) {
                            stddev_value = stddev_value + (mean_value - input_stack(x,y,z))*(mean_value - input_stack(x,y,z));
                        }
                        
                    }
                }
            }
            
            stddev_value = sqrt(stddev_value / mask_count);
            
            
            mask_volume = mask_count * input_stack.GetXSize() * input_stack.GetYSize() * input_stack.GetZSize();
            
            
            double mean_value_final = 0;
            mask_count = 0;
            
            for (int x = 0; x < input_stack.GetX(); x++) {
                for (int y = 0; y < input_stack.GetY(); y++) {
                    for (int z = 0; z < input_stack.GetZ(); z++) {
                        
                        
                        double test_val = abs(input_stack(x,y,z) - mean_value);
                        
                        if (input_mask(x,y,z) == num_l && test_val < std_dev_limit*stddev_value) {           // && abs(input_stack(x,y,z,t)) < 1.00
                            
                            mask_count+=1;
                            mean_value_final = mean_value_final + input_stack(x,y,z);
                        }
                        
                    }
                }
            }
            
            mean_value_final = mean_value_final / mask_count;
            
            cout << num_l << " : " << mask_count << " , " << mask_volume << " , " << mean_value_final << " , " << stddev_value << endl;
            
            info_mean << mean_value_final << ",";
            info_stddev << stddev_value << ",";
            info_volume << mask_volume << ",";
            
        }
        
        info_mean << endl;
        info_stddev << endl;
        info_volume << endl;
        
    }
    

    info_mean.close();
    info_stddev.close();
    info_volume.close();
    
    cout << " - outputs are in: stats-volume.csv, stats-mean.csv, stats-stdev.csv" << endl; 
    
    

    
    return 0;
}




