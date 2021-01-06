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

#include <set>

using namespace mirtk;
using namespace std;
 
// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------
void usage()
{
    cout << "..." << endl;
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
    RealImage stack;

    char *output_name = NULL;

    RealImage input_stack;
    

    char *tmp_fname = NULL;
    UniquePtr<BaseImage> tmp_image;
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    

    tmp_fname = argv[1];
    input_stack.Read(tmp_fname); 
    argc--;
    argv++;


    RealImage template_mask, map_stack, jac_stack;

    tmp_fname = argv[1];
    map_stack.Read(tmp_fname); 
    argc--;
    argv++;


    tmp_fname = argv[1];
    jac_stack.Read(tmp_fname); 
    argc--;
    argv++;


    int th_val = 1000.0;
    th_val = atof(argv[1]);
    argc--;
    argv++;


    int tr_val = 1.0;
    tr_val = atof(argv[1]);
    argc--;
    argv++;


    int label_number = 1;
    
    double mean_val = 0;
    double mean_count = 0;
    double mean_val_recon = 0;
    double mean_count_recon = 0;
    
    double mean_val_jac = 0;
    double mean_count_jac = 0;

    
    double dx = input_stack.GetXSize();
    double dy = input_stack.GetYSize();
    double dz = input_stack.GetZSize();
    
    
    int sh = 0;
    
    
	double current_dt = 0;
	double dt = input_stack.GetTSize();


	dt = tr_val;


    sprintf(buffer, "displ-stats.csv");

    ofstream GD_csv_file;
    GD_csv_file.open(buffer);

    GD_csv_file << "#" << "," << "t [ms]" << "," << "v [mm^3]" << "," << "t2* [ms]" << "," << "mean jac" << endl;
        
        for (int t = 0; t < input_stack.GetT(); t++) {

            mean_count = 0;
            mean_val = 0;
            for (int x = sh; x < input_stack.GetX()-sh; x++) {
                for (int y = sh; y < input_stack.GetY()-sh; y++) {
                    for (int z = sh; z < input_stack.GetZ()-sh; z++) {
                        if (input_stack(x,y,z,t) > 0.05) {
                            mean_count = mean_count + 1;
                        }
                    }
                }
            }
            double volume_mm = mean_count * dx * dy * dz;

                
            mean_count = 0;
            mean_val = 0;
            for (int x = sh; x < input_stack.GetX()-sh; x++) {
                for (int y = sh; y < input_stack.GetY()-sh; y++) {
                    for (int z = sh; z < input_stack.GetZ()-sh; z++) {
                        if (input_stack(x,y,z,t) > 0.05 && input_stack(x,y,z,t) < th_val) {
                            mean_val = mean_val + input_stack(x,y,z,t);
                            mean_count = mean_count + 1;
                        }
                    }
                }
            }
            if (mean_count>0) {
                mean_val = mean_val / mean_count;
            }
            mean_val_recon = 0;
            mean_count_recon = 0;
     

            for (int x = sh; x < map_stack.GetX()-sh; x++) {
                for (int y = sh; y < map_stack.GetY()-sh; y++) {
                    for (int z = sh; z < map_stack.GetZ()-sh; z++) {
                        if (map_stack(x,y,z,t) > 0.05 && map_stack(x,y,z,t) < th_val) {
                            mean_val_recon = mean_val_recon + map_stack(x,y,z,t);
                            mean_count_recon = mean_count_recon + 1;
                        }
                    }
                }
            }
            if (mean_count_recon>0) {
                mean_val_recon = mean_val_recon / mean_count_recon;
            }


            mean_val_jac = 0;
            mean_count_jac = 0;
            for (int x = sh; x < jac_stack.GetX()-sh; x++) {
                for (int y = sh; y < jac_stack.GetY()-sh; y++) {
                    for (int z = sh; z < jac_stack.GetZ()-sh; z++) {
                        if (jac_stack(x,y,z,t) > 0.05) {
                            mean_val_jac = mean_val_jac + jac_stack(x,y,z,t);
                            mean_count_jac = mean_count_jac  + 1;
                        }
                    }
                }
            }
            if (mean_count_jac>0) {
                mean_val_jac = mean_val_jac / mean_count_jac;
            }


            double volume_mm_recon = mean_count_recon * map_stack.GetXSize() * map_stack.GetYSize() * map_stack.GetZSize();
            
            GD_csv_file << t << "," << current_dt << "," << volume_mm << "," << mean_val_recon << "," << mean_val_jac << endl;

            current_dt = current_dt + dt;

        }

        GD_csv_file.close();


    
    return 0;
}


