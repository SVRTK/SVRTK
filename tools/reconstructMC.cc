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

#include "svrtk/ReconstructionFFD.h"
#include "mirtk/ImageReader.h"
#include "mirtk/Dilation.h"

#include <iostream>
#include <cstdio>
#include <ctime> 
#include <chrono>
 
#include <thread>
#include <vector> 

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include <pthread.h>


using namespace mirtk;
using namespace std;
 
// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: mirtk reconstructMC [reconstructed] [N] [stack_1] .. [stack_N] <options>\n" << endl;
    cout << endl;
    
    cout << "\t[reconstructed]         Name for the reconstructed volume (Nifti format)." << endl;
    cout << "\t[N]                     Number of stacks." << endl;
    cout << "\t[stack_1] .. [stack_N]  The input stacks (Nifti format)." << endl;
    cout << "\t" << endl;
    cout << "Options:" << endl;
    //cout << "\t-dofin [dof_1] .. [dof_N]    The transformations of the template stack to all input stacks in .dof format." <<endl;
    cout << "\t-channels M [c1_1] .. [c1_N] .. [cM_1] .. [cM_N]  Multiple [M] channel stacks with the same size as the main channel." <<endl;
    cout << "\t-thickness [d_thickness]  Slice thickness.[Default: 2x voxel size in z direction]"<<endl;
    cout << "\t-mask [mask]              Binary mask to define the region od interest. [Default: whole image]"<<endl;
    cout << "\t-iterations [iter]        Number of registration-reconstruction iterations. [Default: 3]"<<endl;
    cout << "\t-sigma [sigma]            Stdev for bias field. [Default: 12mm]"<<endl;
    cout << "\t-resolution [res]         Isotropic resolution of the volume. [Default: 0.85mm]"<<endl;
    cout << "\t-multires [levels]        Multiresolution smooting with given number of levels. [Default: 3]"<<endl;
    cout << "\t-average [average]        Average intensity value for stacks [Default: 700]"<<endl;
    cout << "\t-delta [delta]            Parameter to define what is an edge. [Default: 175]"<<endl;
    cout << "\t-lambda [lambda]          Smoothing parameter. [Default: 0.0225]"<<endl;
    cout << "\t-lastIter [lambda]        Smoothing parameter for last iteration. [Default: 0.0175]"<<endl;
    cout << "\t-denoise                  Preprocess stacks using NLM denoising. [Default: false]"<<endl;
    cout << "\t-filter                   Filter slices using anisotropic diffusion. [Default: false]"<<endl;
    cout << "\t-smooth_mask [sigma]      Smooth the mask to reduce artefacts of manual segmentation. [Default: 4mm]"<<endl;
    cout << "\t-global_bias_correction   Correct the bias in reconstructed image against previous estimation."<<endl;
    cout << "\t-low_intensity_cutoff     Lower intensity threshold for inclusion of voxels in global bias correction."<<endl;
    cout << "\t-force_exclude [number of slices] [ind1] ... [indN]  Force exclusion of slices with these indices."<<endl;
    cout << "\t-no_intensity_matching    Switch off intensity matching."<<endl;
    cout << "\t-no_packages              Switch off package-based registration adjustment."<<endl;
    cout << "\t-log_prefix [prefix]      Prefix for the log file."<<endl;
    cout << "\t-template [template]      Volume to used as a template for registration. [Default: 1st stack]"<<endl;
    cout << "\t-dilation [dilation]      Degree of mask dilation for ROI. [Default: 5]"<<endl;
    cout << "\t-cp [init_cp] [cp_step]   Initial CP spacing and step per SVR iteration. [Default: 15 5]"<<endl;
    cout << "\t-exclude_slices_only      Only slice-level exlusion of outliers based on robust statistics. [Default: false]"<<endl;
    cout << "\t-structural               Structural outlier rejection step. [Default: false]"<<endl;
    cout << "\t-full                     Additional SVR and SR iterations (for severe motion cases). [Default: false]"<<endl;
    cout << "\t-masked                   Use masking for registration and reconstruction (faster version). [Default: false]"<<endl;
    cout << "\t-intersection             Use preliminary stack intersection. [Default: false]"<<endl;
    cout << "\t-default                  Optimal delault options (-intersection, -structural, -rigid_init, -dilation 7)."<<endl;
    cout << "\t-remote                   Run SVR registration as remote functions in case of memory issues [Default: false]."<<endl;
    cout << "\t-debug                    Debug mode - save intermediate results."<<endl;
    cout << "\t-no_log                   Do not redirect cout and cout to log files."<<endl;
    cout << "\t" << endl;
    cout << "\t" << endl;
    
    exit(1);
}




// -----------------------------------------------------------------------------

bool IsIdentity(const string &name)
{
    return name.empty() || name == "identity" || name == "Identity" || name == "Id";
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    
    

    const char *current_mirtk_path = argv[0];
    
    int i, ok;
    char buffer[256];
    RealImage stack;
    
    char * output_name = NULL;
    Array<RealImage*> stacks;
    Array<string> stack_files;
    Array<RigidTransformation> stack_transformations;
    bool have_stack_transformations = false;
    Array<double> thickness;
    int nStacks;
    Array<int> packages;
    


    int templateNumber=0;
    RealImage *mask=NULL;
    int iterations = 3;
    bool debug = false;
    double sigma=20;
    double resolution = 0.85;
    double lambda = 0.020;
    double delta = 120;
    int levels = 3;
    double lastIterLambda = 0.0125;
    int rec_iterations;
    double averageValue = 700;
    double smooth_mask = 4;
    bool global_bias_correction = false;
    double low_intensity_cutoff = 0.01;


    bool no_longsr_flag = false;
    
    bool structural_after_1_flag = false;


    bool intensity_matching = true;
    bool rescale_stacks = false;
    
    bool robust_statistics = true;
    bool robust_slices_only = false;
    
    bool ffd_flag = true;
    bool structural_exclusion = false;
    bool fast_flag = true;
    bool flag_full = false;
    int mask_dilation = 5;

    bool filter_flag = false;
    int excluded_stack = 99;
    bool masked_flag = false;

    bool packages_flag = false;
    bool intersection_flag = false;

    bool flag_denoise = false;
    
    bool no_packages_flag = false;
    
    bool no_intensity_absolute = false;
    
    bool remote_flag = false;
    
    int d_packages = -1;
    double d_thickness = -1;

    
    RealImage main_mask;
    RealImage template_stack, template_mask;
    RealImage real_phantom_volume;

    Array<int> selected_slices;
    bool selected_flag = false;
    bool no_registration_flag = false;
    bool rigid_init_flag = false;
    bool real_phantom_flag = false;

    
    bool multiple_channels_flag = false;
    Array<Array<RealImage*>> multi_channel_stacks;
    int number_of_channels = 0;

    
    auto start_main = std::chrono::system_clock::now();
    auto end_main = std::chrono::system_clock::now();
    
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    
    std::chrono::duration<double> elapsed_seconds;
    

    RealImage average;
    
    string log_id;
    bool no_log = false;
    
    int number_of_force_excluded_slices = 0;
    Array<int> force_excluded;

    ReconstructionFFD *reconstruction = new ReconstructionFFD();

 
    cout << ".........................................................." << endl;
    cout << ".........................................................." << endl;
    
    if (argc < 5)
        usage();
    
    output_name = argv[1];
    argc--;
    argv++;
    cout << "Recontructed volume : " << output_name << endl;
    
    nStacks = atoi(argv[1]);
    argc--;
    argv++;
    cout << "Number of stacks : " << nStacks << endl;
    
  
    const char *tmp_fname;
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    

    RealImage* p_stack;

    
    
    for (i=0; i<nStacks; i++) {
        
        stack_files.push_back(argv[1]);
        
        cout << "Input stack " << i << " : " << argv[1];
        
        
        RealImage *tmp_p_stack = new RealImage(argv[1]);
        
        double min, max;

        tmp_p_stack->GetMinMaxAsDouble(min, max);
        
//        // only for some of the cases (e.g., angio )
//        if (min < 0) {
//           tmp_p_stack->PutMinMaxAsDouble(0, 800);
//        }
        
        double tmp_dx = tmp_p_stack->GetXSize();
        double tmp_dy = tmp_p_stack->GetYSize();
        double tmp_dz = tmp_p_stack->GetZSize();
        double tmp_dt = tmp_p_stack->GetTSize();
        
        double tmp_sx = tmp_p_stack->GetX();
        double tmp_sy = tmp_p_stack->GetY();
        double tmp_sz = tmp_p_stack->GetZ();
        double tmp_st = tmp_p_stack->GetT();
       
        cout << "  ;  " << tmp_sx << " - " << tmp_sy << " - " << tmp_sz << " - " << tmp_st  << "  ;";
        cout << "  " << tmp_dx << " - " << tmp_dy << " - " << tmp_dz << " - " << tmp_dt  << "  ;";
        cout << "  [" << min << "; " << max << "]" << endl;
        
        argc--;
        argv++;
        stacks.push_back(tmp_p_stack);
        
        
    }


    
    int nPackages = 1;
    templateNumber = 0;
    
    Array<RealImage> stacks_before_division;
    Array<int> stack_package_index;

    for (int i=0; i<stacks.size(); i++)
        stack_package_index.push_back(i);


  
    template_stack = stacks[templateNumber]->GetRegion(0,0,0,0,stacks[templateNumber]->GetX(),stacks[templateNumber]->GetY(),stacks[templateNumber]->GetZ(),(1));

    

    while (argc > 1) {
        ok = false;
        
        if ((ok == false) && (strcmp(argv[1], "-dofin") == 0)){
            argc--;
            argv++;
            
            for (i=0; i<nStacks; i++) {
                
                
                cout << "Rigid ransformation : " << argv[1] << endl;

                Transformation *t_tmp = Transformation::New(argv[1]);
                RigidTransformation *r_transformation = dynamic_cast<RigidTransformation*> (t_tmp);

                argc--;
                argv++;
                
                stack_transformations.push_back(*r_transformation);
                
            }
            reconstruction->InvertStackTransformations(stack_transformations);
            have_stack_transformations = true;
        }
        
        
        
        
        if ((ok == false) && (strcmp(argv[1], "-channels") == 0)){
            argc--;
            argv++;
            
            number_of_channels = atoi(argv[1]);
            
            argc--;
            argv++;
            
//            cout << number_of_channels << endl;
            
            
            if (number_of_channels > 0) {
            
                for (int n=0; n<number_of_channels; n++) {
                
                    Array<RealImage*> tmp_mc_array;
                
                    for (i=0; i<nStacks; i++) {
                        
                        cout << "MC stack " << i << " (" << n  << ") : " << argv[1] << endl;
                        
                        RealImage *tmp_mc_stack = new RealImage(argv[1]);
                        
                        
                        if (tmp_mc_stack->GetX() != stacks[i]->GetX() || tmp_mc_stack->GetY() != stacks[i]->GetY() || tmp_mc_stack->GetZ() != stacks[i]->GetZ() || tmp_mc_stack->GetT() != stacks[i]->GetT()) {
                            
                            cout << "Dimensions of MC stacks should the same as the main channel." << endl;
                            exit(1);
                        }
                        
                        tmp_mc_array.push_back(tmp_mc_stack);
                       
                        argc--;
                        argv++;
                        
                    }
                    
                    multi_channel_stacks.push_back(tmp_mc_array);
                    
                }

                multiple_channels_flag = true;
                
                reconstruction->SetMCSettings(number_of_channels);
                
            }
            
            ok = true;

        }
        
        
        
        if ((ok == false) && (strcmp(argv[1], "-select") == 0)) {
            argc--;
            argv++;

            
            
            int slice_number;
            string line;
            ifstream selection_file (argv[1]);
            
            if (selection_file.is_open()) {
                selected_slices.clear();
                while ( getline (selection_file, line) ) {
                    slice_number = atoi(line.c_str());
                    selected_slices.push_back(slice_number);
                }
                selection_file.close();
                selected_flag = true;
            }
            else {
                cout << "File with selected slices could not be open." << endl;
                exit(1);
            }
            
            cout <<"Number of selected slices : " << selected_slices.size() << endl;
            
            argc--;
            argv++;
        }
        

        if ((ok == false) && (strcmp(argv[1], "-thickness") == 0)) {
            argc--;
            argv++;
            
            d_thickness = atof(argv[1]);
            
            cout<< "Slice thickness : " << d_thickness << endl;

            ok = true;
            argc--;
            argv++;
        }
        

        
        
        if ((ok == false) && (strcmp(argv[1], "-no_overlap_thickness") == 0)) {

            d_thickness = stacks[0]->GetZSize()*1.1;
            
            cout<< "Interpolation : " << d_thickness << endl;

            ok = true;
            argc--;
            argv++;
        }
        
        

        if ((ok == false) && (strcmp(argv[1], "-single_stack_recon") == 0)) {

            resolution = stacks[0]->GetXSize()*0.85;

            cout<< "Resolution : " << resolution << endl;

            ok = true;
            argc--;
            argv++;
        }
        


        if ((ok == false) && (strcmp(argv[1], "-packages") == 0)) {
            argc--;
            argv++;

            d_packages = atoi(argv[1]);

            cout<< "Packages : " << d_packages << endl;

            ok = true;
            argc--;
            argv++;
        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-mask") == 0)) {
            argc--;
            argv++;
            mask= new RealImage(argv[1]);
            
            main_mask = (*mask);

            cout<< "Mask : " << argv[1] << endl;
            
            template_mask = main_mask;
            
            ok = true;
            argc--;
            argv++;
        }
        
        
        
        if ((ok == false) && (strcmp(argv[1], "-template") == 0)) {
            argc--;
            argv++;
            
            RealImage *tmp_p_stack = new RealImage(argv[1]);
            template_stack = *tmp_p_stack;
            
            
            
            ok = true;
            argc--;
            argv++;
        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-real_phantom") == 0)) {
            argc--;
            argv++;
            
            real_phantom_volume.Read(argv[1]);
            real_phantom_flag = true;
            
            ok = true;
            argc--;
            argv++;
        }
        
        

        if ((ok == false) && (strcmp(argv[1], "-mask_template") == 0)) {
            argc--;
            argv++;
            
            RealImage *tmp_m_stack = new RealImage(argv[1]);
            template_mask = *tmp_m_stack;

            ok = true;
            argc--;
            argv++;
        }
        
 
        
        if ((ok == false) && (strcmp(argv[1], "-dilation") == 0)) {
            argc--;
            argv++;
            mask_dilation=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }



        if ((ok == false) && (strcmp(argv[1], "-no_long_sr") == 0)) {
                argc--;
                argv++;
                no_longsr_flag=true;
                ok = true;
        }

        
        if ((ok == false) && (strcmp(argv[1], "-no_global") == 0)) {
            argc--;
            argv++;
            reconstruction->_no_global_ffd_flag = true;
            ok = true;
        }
        
        

        if ((ok == false) && (strcmp(argv[1], "-cp") == 0)) {
            argc--;
            argv++;
            
            int value, step;

            value=atoi(argv[1]);
            
            argc--;
            argv++;
            
            step=atoi(argv[1]);
            
            reconstruction->SetCP(value, step);
            
            ok = true;
            argc--;
            argv++;
        }
        


        if ((ok == false) && (strcmp(argv[1], "-iterations") == 0)) {
            argc--;
            argv++;
            iterations=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-sigma") == 0)) {
            argc--;
            argv++;
            sigma=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-lambda") == 0)) {
            argc--;
            argv++;
            lambda=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-lastIter") == 0)) {
            argc--;
            argv++;
            lastIterLambda=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        
        
        if ((ok == false) && (strcmp(argv[1], "-delta") == 0)) {
            argc--;
            argv++;
            delta=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        

        
        
        if ((ok == false) && (strcmp(argv[1], "-resolution") == 0)) {
            argc--;
            argv++;
            resolution=atof(argv[1]);
            ok = true;
            argc--;
            argv++; 
        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-multires") == 0)) {
            argc--;
            argv++;
            levels=atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-exclude_stack") == 0)) {
            argc--;
            argv++;
            excluded_stack=atoi(argv[1]);
            cout << " .. excluded stack : " << excluded_stack << endl;
            argc--;
            argv++;
            ok = true;
        }
        

        if ((ok == false) && (strcmp(argv[1], "-smooth_mask") == 0)) {
            argc--;
            argv++;
            smooth_mask=atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-no_intensity_matching") == 0)) {
            argc--;
            argv++;
            intensity_matching=false;
            ok = true;
        }
        

        if ((ok == false) && (strcmp(argv[1], "-no_intensity_absolute") == 0)) {
            argc--;
            argv++;
            intensity_matching=false;
            no_intensity_absolute=true;
            ok = true;
        }
        
        

        if ((ok == false) && (strcmp(argv[1], "-no_robust_statistics") == 0)) {
            argc--;
            argv++;
            robust_statistics=false;
            ok = true;
        }
        

        if ((ok == false) && (strcmp(argv[1], "-remote") == 0)) {
            argc--;
            argv++;
            remote_flag=true;
            ok = true;
        }

        
//        if ((ok == false) && (strcmp(argv[1], "-ncc") == 0)) {
//            argc--;
//            argv++;
//            reconstruction->_ncc_flag=true;
//            ok = true;
//        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-structural_after_1") == 0)) {
            argc--;
            argv++;
            structural_exclusion = true;
            structural_after_1_flag = true;
            
            ok = true;
        }

        
        if ((ok == false) && (strcmp(argv[1], "-no_packages") == 0)) {
            argc--;
            argv++;
            no_packages_flag = true;
            ok = true;
        }
        
        

        if ((ok == false) && (strcmp(argv[1], "-exclude_slices_only") == 0)) {
            argc--;
            argv++;
            robust_slices_only = true;
            ok = true;
        }
        
        
        if ((ok == false) && (strcmp(argv[1], "-structural") == 0)) {
            argc--;
            argv++;
            structural_exclusion = true;
            ok = true;
        }
        
        
        
        if ((ok == false) && (strcmp(argv[1], "-fast") == 0)) {
            argc--;
            argv++;
            fast_flag = true;
            ok = true;
        }
        
        
        if ((ok == false) && (strcmp(argv[1], "-default") == 0)) {
            argc--;
            argv++;
            
            intersection_flag = true;
            fast_flag = true;
            structural_exclusion = true;
            rigid_init_flag = true;
            
            ok = true;
        }

        
        if ((ok == false) && (strcmp(argv[1], "-denoise") == 0)){
            argc--;
            argv++;
            flag_denoise=true;
            ok = true;
        }
        

        if ((ok == false) && (strcmp(argv[1], "-rigid_init") == 0)) {
            argc--;
            argv++;
            rigid_init_flag = true;
            ok = true;
        }
        
        
        
        if ((ok == false) && (strcmp(argv[1], "-full") == 0)) {
            argc--;
            argv++;
            flag_full = true;
            ok = true;
        }

        
        
        if ((ok == false) && (strcmp(argv[1], "-masked") == 0)) {
            argc--;
            argv++;
            masked_flag=true;
            reconstruction->SetMaskedFlag(masked_flag);
            
            ok = true;
        }
        
        
        
        if ((ok == false) && (strcmp(argv[1], "-no_registration") == 0)) {
            argc--;
            argv++;
            no_registration_flag=true;
            ok = true;
        }


        if ((ok == false) && (strcmp(argv[1], "-intersection") == 0)) {
            argc--;
            argv++;
            intersection_flag=true;
            ok = true;
        }
        
        
        
        if ((ok == false) && (strcmp(argv[1], "-filter") == 0)) {
            argc--;
            argv++;
            filter_flag=true;
            ok = true;
        }
        


        
        if ((ok == false) && (strcmp(argv[1], "-global_bias_correction") == 0)) {
            argc--;
            argv++;
            global_bias_correction=true;
            ok = true;
        }
        
        if ((ok == false) && (strcmp(argv[1], "-low_intensity_cutoff") == 0)) {
            argc--;
            argv++;
            low_intensity_cutoff=atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        

        if ((ok == false) && (strcmp(argv[1], "-debug") == 0)) {
            argc--;
            argv++;
            debug=true;
            ok = true;
        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-log_prefix") == 0)) {
            argc--;
            argv++;
            log_id=argv[1];
            ok = true;
            argc--;
            argv++;
        }
        

        
        if ((ok == false) && (strcmp(argv[1], "-no_log") == 0)) {
            argc--;
            argv++;
            no_log=true;
            ok = true;
        }
        

        if ((ok == false) && (strcmp(argv[1], "-rescale_stacks") == 0)) {
            argc--;
            argv++;
            rescale_stacks=true;
            ok = true;
        }
        
        
        if ((ok == false) && (strcmp(argv[1], "-force_exclude") == 0)) {
            argc--;
            argv++;
            number_of_force_excluded_slices = atoi(argv[1]);
            argc--;
            argv++;
            
            cout<< number_of_force_excluded_slices<< " force excluded slices: ";
            for (i=0;i<number_of_force_excluded_slices;i++) {
                force_excluded.push_back(atoi(argv[1]));
                cout<<force_excluded[i]<<" ";
                argc--;
                argv++;
            }
            cout<<"."<<endl;
            
            ok = true;
        }

        if (ok == false) {
            cerr << "Can not parse argument " << argv[1] << endl;
            usage();
        }
    }

    
     
     

    
    string str_mirtk_path;
    string str_current_main_file_path;
    string str_current_exchange_file_path;

    string str_recon_path(current_mirtk_path);
    size_t pos = str_recon_path.find_last_of("/");
    str_mirtk_path = str_recon_path.substr (0, pos);
    
    system("pwd > pwd.txt ");
    ifstream pwd_file("pwd.txt");
    
    if (pwd_file.is_open()) {
        getline(pwd_file, str_current_main_file_path);
        pwd_file.close();
    } else {
        cout << "System error: no rights to write in the current folder" << endl;
        exit(1);
    }
    
    str_current_exchange_file_path = str_current_main_file_path + "/tmp-file-exchange";
    
    if (str_current_exchange_file_path.length() > 0) {
//        string remove_folder_cmd = "rm -r " + str_current_exchange_file_path + " > tmp-log.txt ";
//        int tmp_log_rm = system(remove_folder_cmd.c_str());

        string create_folder_cmd = "mkdir " + str_current_exchange_file_path + " > tmp-log.txt ";
        int tmp_log_mk = system(create_folder_cmd.c_str());
        
    } else {
        cout << "System error: could not create a folder for file exchange" << endl;
        exit(1);
    }


    
    
    RealImage main_stack;
    
    if (stacks[templateNumber]->GetT() > 1) {
        main_stack = stacks[templateNumber]->GetRegion(0,0,0,0,stacks[0]->GetX(),stacks[0]->GetY(),stacks[0]->GetZ(),(1));
    } else {
        main_stack = *stacks[templateNumber];
    }

    
    bool has_4D_stacks = false;
    
    for (i=0; i<stacks.size(); i++) {
        
        if (stacks[i]->GetT()>1) {
            has_4D_stacks = true;
            break;
        }
        
    }

    

    
    if (has_4D_stacks) {
        
        
        Array<Array<RealImage>> new_multi_channel_stacks;
        Array<RealImage> new_stacks;

        
        if (number_of_channels < 1) {
        
            for (i=0; i<stacks.size(); i++) {
                
                if (stacks[i]->GetT() == 1) {
                    new_stacks.push_back(*stacks[i]);
     
                }
                else {
                    for (int t=0; t<stacks[i]->GetT(); t++) {
                        
                        stack = stacks[i]->GetRegion(0,0,0,t,stacks[i]->GetX(),stacks[i]->GetY(),stacks[i]->GetZ(),(t+1));
                        new_stacks.push_back(stack);
                    }
                    
                }
            }
        
        } else {


            for (i=0; i<stacks.size(); i++) {
                
                if (stacks[i]->GetT() == 1) {
                    new_stacks.push_back(*stacks[i]);
                }
                else {
                    for (int t=0; t<stacks[i]->GetT(); t++) {
                        stack = stacks[i]->GetRegion(0,0,0,t,stacks[i]->GetX(),stacks[i]->GetY(),stacks[i]->GetZ(),(t+1));
                        new_stacks.push_back(stack);
                    }
                    
                }
            }
            
            
            for (int n=0; n<number_of_channels; n++) {
                
                Array<RealImage> tmp_mc_array;
                
                for (i=0; i<stacks.size(); i++) {
                    
                    if (stacks[i]->GetT() == 1) {
                        tmp_mc_array.push_back(*multi_channel_stacks[n][i]);
                        
                    }
                    else {
                        for (int t=0; t<stacks[i]->GetT(); t++) {
                            
                            stack = multi_channel_stacks[n][i]->GetRegion(0,0,0,t,stacks[i]->GetX(),stacks[i]->GetY(),stacks[i]->GetZ(),(t+1));
                            tmp_mc_array.push_back(stack);
                            
                        }
                        
                    }
                }
                
                new_multi_channel_stacks.push_back(tmp_mc_array);
                
            }
            
        }
        

        
        nStacks = new_stacks.size();
        
        stacks.clear();

        
        for (i=0; i<new_stacks.size(); i++) {
            stacks.push_back(new RealImage(new_stacks[i]));
        }
        
        if (number_of_channels > 0) {
            
            multi_channel_stacks.clear();
            
            for (int n=0; n<number_of_channels; n++) {

                Array<RealImage*> tmp_mc_array;
                
                for (i=0; i<new_stacks.size(); i++) {
                    tmp_mc_array.push_back(new RealImage(new_multi_channel_stacks[n][i]));
                    
                }
                
                multi_channel_stacks.push_back(tmp_mc_array);
                
            }
            
            new_multi_channel_stacks.clear();

        }
        
        
        new_stacks.clear();

    }
    


    

    RealImage i_mask = *stacks[templateNumber];
    i_mask = 1;

    if (intersection_flag) {

        cout << "Intersection" << endl;

        start = std::chrono::system_clock::now();

        i_mask = reconstruction->StackIntersection(stacks, templateNumber, template_mask);

        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
        
        

        for (int i=0; i<stacks.size(); i++) {
            
            RigidTransformation* tmp_r = new RigidTransformation();
            
            RealImage i_tmp = i_mask;
            reconstruction->TransformMask(*stacks[i],i_tmp, *tmp_r);
            RealImage stack_tmp = *stacks[i];

            reconstruction->CropImage(stack_tmp,i_tmp);
            
            delete stacks[i];
            stacks[i] = new RealImage(stack_tmp);
            
            if (debug) {
                sprintf(buffer,"cropped-%i.nii.gz",i);
                stacks[i]->Write(buffer);
            }
            
        }
        
        
        if (number_of_channels > 0) {
            
            for (int n=0; n<number_of_channels; n++) {
                
                for (int i=0; i<stacks.size(); i++) {
                    
                    RigidTransformation* tmp_r = new RigidTransformation();
                    
                    RealImage i_tmp = i_mask;
                    reconstruction->TransformMask(*multi_channel_stacks[n][i],i_tmp, *tmp_r);
                    RealImage stack_tmp = *multi_channel_stacks[n][i];
                    
                    reconstruction->CropImage(stack_tmp,i_tmp);
                    
                    delete multi_channel_stacks[n][i];
                    multi_channel_stacks[n][i] = new RealImage(stack_tmp);
                    
                    if (debug) {
                        sprintf(buffer,"mc-cropped-%i-%i.nii.gz",n,i);
                        multi_channel_stacks[n][i]->Write(buffer);
                    }
   
                }
                
                
            }
            
        }
        
        

    }

 
    
    if (flag_denoise) {
        
        cout << "NLMFiltering" << endl;
        reconstruction->NLMFiltering(stacks);
        
    }
    

    if (rescale_stacks) {
        for (i=0; i<nStacks; i++)
            reconstruction->Rescale(stacks[i], 1000);
    }
    
    cout << ".........................................................." << endl;

    
    
    if (d_thickness>0) {
        for (i=0; i<nStacks; i++) {
            thickness.push_back(d_thickness);
        }
    }
    else {
        
        cout<< "Slice thickness : ";
        
        for (i=0; i<nStacks; i++) {
            double dx,dy,dz;
            stacks[i]->GetPixelSize(&dx, &dy, &dz);
            thickness.push_back(dz*2);
            cout << thickness[i] <<" ";
        }
        cout << endl;
    }


    if (d_packages>0) {
        for (i=0; i<nStacks; i++) {
            packages.push_back(d_packages);
        }
        packages_flag = true;
    }
    else {
        d_packages = 1;
    }



    if (!have_stack_transformations) {
        for (i=0; i<nStacks; i++) {
            RigidTransformation *rigidTransf = new RigidTransformation;
            stack_transformations.push_back(*rigidTransf);
            delete rigidTransf;
        }
    }

    


    if(ffd_flag)
        reconstruction->FFDRegistrationOn();
    else
        reconstruction->FFDRegistrationOff();
    

    
    reconstruction->SetStructural(structural_exclusion);
    
    reconstruction->ExcludeStack(excluded_stack);
    
    


    
    if (!selected_flag) {
        selected_slices.clear();
        for (i=0; i<stacks[0]->GetZ(); i++)
            selected_slices.push_back(i);
        selected_flag = true;
        
    }

    //Output volume
    RealImage reconstructed;
    
    
    //Set debug mode
    if (debug) reconstruction->DebugOn();
    else reconstruction->DebugOff();
    
    //Set force excluded slices
    reconstruction->SetForceExcludedSlices(force_excluded);
    
    //Set low intensity cutoff for bias estimation
    reconstruction->SetLowIntensityCutoff(low_intensity_cutoff);
    
    
    // Check whether the template stack can be identified
    if (templateNumber<0) {
        cerr<<"Please identify the template by assigning id transformation."<<endl;
        exit(1);
    }

    
    RealImage maskedTemplate;
    GreyImage grey_maskedTemplate;
    
    //Before creating the template we will crop template stack according to the given mask
    if (mask !=NULL)
    {

        RealImage transformed_mask = *mask;

        reconstruction->TransformMaskP(&template_stack, transformed_mask);
        
        //Crop template stack
        maskedTemplate = template_stack * transformed_mask;
        reconstruction->CropImage(maskedTemplate, transformed_mask);
        maskedTemplate.Write("masked.nii.gz");
        
        grey_maskedTemplate = maskedTemplate;

    }
    else {
        cout << "Mask was not specified." << endl;
        exit(1);
    }
    
    
    cout << ".........................................................." << endl;
    cout << endl;
    
    
    // rigid stack registration
    
    Array<GreyImage*> p_grey_stacks;
    for (int i=0; i<stacks.size(); i++) {
        GreyImage tmp_s = *stacks[i];
//        GreyImage *tmp_grey_stack = new GreyImage(tmp_s);
        p_grey_stacks.push_back(new GreyImage(tmp_s)); //tmp_grey_stack);
        
    }
    
    
    bool init_reset = false;
    
    if (!have_stack_transformations && rigid_init_flag) {
        
        start = std::chrono::system_clock::now();
        
        init_reset = true;

//        grey_stacks[0]->Write("../z.nii.gz");
        
        cout << "RigidStackRegistration" << endl;
        reconstruction->RigidStackRegistration(p_grey_stacks, &grey_maskedTemplate, stack_transformations, init_reset);

        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;


    }

    
    // stack template preparation
    
    RealImage blurred_template_stack = template_stack;
    
    {
        RealImage template_mask_transformed = template_mask;
        reconstruction->TransformMaskP(&blurred_template_stack, template_mask_transformed);
        reconstruction->CropImage(blurred_template_stack, template_mask_transformed);
        
        GaussianBlurring<RealPixel> gb(stacks[0]->GetXSize()*1.25);
        gb.Input(&blurred_template_stack);
        gb.Output(&blurred_template_stack);
        gb.Run();
    
    }
    
    
    if (debug)
        blurred_template_stack.Write("blurred-template.nii.gz");

    
    // Reorientation of stacks
    if (have_stack_transformations || rigid_init_flag) {
        
        if (number_of_channels > 0) {
            
            for (int n=0; n<number_of_channels; n++) {
                
                for (i=0; i<stacks.size(); i++) {
                    
                    if (i != templateNumber) {
                        
                        Matrix m = stack_transformations[i].GetMatrix();
                        multi_channel_stacks[n][i]->PutAffineMatrix(m, true);
                        
                    }
                    
                }
                
            }
            
        }
        
        for (i=0; i<stacks.size(); i++) {
        
            if (i != templateNumber) {
            
                Matrix m = stack_transformations[i].GetMatrix();
                stacks[i]->PutAffineMatrix(m, true);
                
                RigidTransformation *rigidTransf = new RigidTransformation;
                stack_transformations[i] = *rigidTransf;

            }

        }
        
    }
    
    
    

    if (ffd_flag) {

        start = std::chrono::system_clock::now();
        
        GreyImage grey_blurred_template_stack = blurred_template_stack;

        cout << "FFDStackRegistration" << endl;
        reconstruction->FFDStackRegistration(p_grey_stacks, &grey_blurred_template_stack, stack_transformations);

        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
        
    }


    RealImage dilated_main_mask = main_mask;
    
//    {
    
        ConnectivityType connectivity = CONNECTIVITY_26;
        Dilate<RealPixel>(&dilated_main_mask, mask_dilation, connectivity); // 5

        start = std::chrono::system_clock::now();
        cout << "Masking" << endl;
        
        for (i=0; i<stacks.size(); i++) {
            
                
                sprintf(buffer, "ms-%i.dof", i);

                Transformation *tt = Transformation::New(buffer);
                MultiLevelFreeFormTransformation *mffd_init = dynamic_cast<MultiLevelFreeFormTransformation*> (tt);

                InterpolationMode interpolation = Interpolation_NN;
                UniquePtr<InterpolateImageFunction> interpolator;
                interpolator.reset(InterpolateImageFunction::New(interpolation));

                RealImage transformed_main_mask = *stacks[i];

                double source_padding = 0;
                double target_padding = -inf;
                bool dofin_invert = false;
                bool twod = false;

                ImageTransformation *imagetransformation = new ImageTransformation;

                imagetransformation->Input(&dilated_main_mask);
                imagetransformation->Transformation(mffd_init);
                imagetransformation->Output(&transformed_main_mask);
                imagetransformation->TargetPaddingValue(target_padding);
                imagetransformation->SourcePaddingValue(source_padding);
                imagetransformation->Interpolator(interpolator.get());  // &nn);
                imagetransformation->TwoD(twod);
                imagetransformation->Invert(dofin_invert);
                imagetransformation->Run();


                delete imagetransformation;

            
                RealImage stack_tmp2 = *stacks[i];
                reconstruction->CropImage(stack_tmp2, transformed_main_mask);

                stacks[i] = new RealImage(stack_tmp2);   //tmp_real_stack;
                
                sprintf(buffer,"fcropped-%i.nii.gz",i);
                stacks[i]->Write(buffer);
            
            
                if (number_of_channels > 0) {
                    
                    for (int n=0; n<number_of_channels; n++) {
                    
                        stack_tmp2 = *multi_channel_stacks[n][i];
                        reconstruction->CropImage(stack_tmp2, transformed_main_mask);
                        
                        multi_channel_stacks[n][i] = new RealImage(stack_tmp2);
                        
                        sprintf(buffer,"mc-fcropped-%i-%i.nii.gz",n,i);
                        multi_channel_stacks[n][i]->Write(buffer);
                        
                    }
                    
                }

        }
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
    

        blurred_template_stack = template_stack;
        GaussianBlurring<RealPixel> gb4(stacks[0]->GetXSize()*1.25);
        
        gb4.Input(&blurred_template_stack);
        gb4.Output(&blurred_template_stack);
        gb4.Run();
    

        
        reconstruction->TransformMaskP(&blurred_template_stack, dilated_main_mask);
        reconstruction->CropImage(blurred_template_stack, dilated_main_mask);
    
    
        
        if (masked_flag) {

            RealImage tmp_mask = dilated_main_mask;
            reconstruction->CropImage(tmp_mask, tmp_mask);
            blurred_template_stack *= tmp_mask;
            blurred_template_stack.Write("masked_template.nii.gz");
        }
        
        cout << "CreateTemplate : " << resolution << endl;
        resolution = reconstruction->CreateTemplate(&blurred_template_stack, resolution);

    
        if (masked_flag) {
            reconstruction->SetMask(&dilated_main_mask, smooth_mask);
            
        }
        else {
            dilated_main_mask = 1;
            reconstruction->SetMask(&dilated_main_mask, smooth_mask);
            
        }



    streambuf* strm_buffer = cout.rdbuf();
    streambuf* strm_buffer_e = cerr.rdbuf();

    string name;
    name = log_id+"log-registration.txt";
    ofstream file(name.c_str());
    name = log_id+"log-registration-error.txt";
    ofstream file_e(name.c_str());

    name = log_id+"log-reconstruction->txt";
    ofstream file2(name.c_str());
    name = log_id+"log-evaluation.txt";
    ofstream fileEv(name.c_str());
    
    cout<<setprecision(3);
    cerr<<setprecision(3);
    

    
    if (!no_intensity_absolute) {
    
        start = std::chrono::system_clock::now();
    
        cout << "MatchStackIntensitiesWithMasking" << endl;
        if (intensity_matching)
            reconstruction->MatchStackIntensitiesWithMasking(stacks, stack_transformations, averageValue);
        else
            reconstruction->MatchStackIntensitiesWithMasking(stacks, stack_transformations, averageValue, true);
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
        
    }
    
    
    
    

    double rmse_out;

    start = std::chrono::system_clock::now();

    cout << "CreateSlicesAndTransformations" << endl;
    selected_slices.clear();
    reconstruction->ResetSlicesAndTransformationsFFD();

    for (int ii=0; ii<stacks.size(); ii++) {
        
        selected_slices.clear();
        for (i = 0; i < stacks[ii]->GetZ(); i++)
            selected_slices.push_back(i);

        if (number_of_channels > 0) {
            
            reconstruction->CreateSlicesAndTransformationsFFDMC(stacks, multi_channel_stacks, stack_transformations, thickness, d_packages, selected_slices, ii);
            
        } else {
            
            reconstruction->CreateSlicesAndTransformationsFFD(stacks, stack_transformations, thickness, d_packages, selected_slices, ii);
            
        }
 
    }
    
    
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    if (debug)
        cout << "- duration : " << elapsed_seconds.count() << " s " << endl;

    
    cout << ".........................................................." << endl;
    
    
    if (filter_flag) {
        double delta2D = 400;
        double lambda2D = 0.1;
        double alpha2D = (0.05 / lambda2D) * delta2D * delta2D;
        
        start = std::chrono::system_clock::now();
        
        cout << "FilterSlices : " << alpha2D << " - " << lambda2D << " - " << delta2D << endl;
        reconstruction->FilterSlices(alpha2D, lambda2D, delta2D);
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
        
    }

    
    cout << ".........................................................." << endl;


    if (!have_stack_transformations && !rigid_init_flag)
        init_reset = true;
    else
        init_reset = false;
    
    
    
    
    if (no_packages_flag) {
        nPackages = 1;
    }
    
    
    start = std::chrono::system_clock::now();

    for (int i=0; i<stacks.size(); i++) {
        
        delete p_grey_stacks[i];
    }
    p_grey_stacks.clear();
    for (int i=0; i<stacks.size(); i++) {
        GreyImage tmp_s = *stacks[i];
        GreyImage *tmp_grey_stack = new GreyImage(tmp_s);
        p_grey_stacks.push_back(tmp_grey_stack);
    }
    
    

    
    cout << "RigidPackageRegistration" << endl;
    reconstruction->RigidPackageRegistration(p_grey_stacks, &grey_maskedTemplate, nPackages, stack_transformations, init_reset);
    
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    if (debug)
        cout << "- duration : " << elapsed_seconds.count() << " s " << endl;

    start = std::chrono::system_clock::now();
    
    
    if (structural_exclusion) {
        cout << "CreateSliceMasks" << endl;
        reconstruction->CreateSliceMasks(main_mask, -1);
    }
    

    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    if (debug)
        cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
    

    if (sigma>0)
        reconstruction->SetSigma(sigma);
    else
        reconstruction->SetSigma(20);
    

    if (global_bias_correction)
        reconstruction->GlobalBiasCorrectionOn();
    else
        reconstruction->GlobalBiasCorrectionOff();
    
    start = std::chrono::system_clock::now();
    

    cout << "InitializeEM" << endl;
    reconstruction->InitializeEM();
    
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    if (debug)
        cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
    
    
    cout << ".........................................................." << endl;
    cout << ".........................................................." << endl;
    cout << "Reconstruction loop " << endl;
    cout << ".........................................................." << endl;
    
    


    
    for (int iter=0;iter<iterations;iter++) {

        if ( ! no_log ) {
            cout.rdbuf (strm_buffer);
        }
        cout << ".........................................................." << endl;
        cout << "- Main iteration : " << iter << endl;
        
        

        if (iter > -1) {
            
            if (structural_exclusion) {
                
                reconstruction->SetStructural(structural_exclusion);
                
            }

            
            if (structural_after_1_flag) {
                
                if (iter == 0) {
                    structural_exclusion = false;
                    reconstruction->SetStructural(false);
                } else {
                    structural_exclusion = true;
                    reconstruction->SetStructural(true);
                }
                
            }
            
            

            if (iter > -1) {

                if (!no_registration_flag) {
                    
                    start = std::chrono::system_clock::now();

                    cout << "SliceToVolumeRegistration" << endl;
                    
                    
                    if (remote_flag) {

                        reconstruction->RemoteSliceToVolumeRegistration(iter, 1, stacks.size(), str_mirtk_path, str_current_main_file_path, str_current_exchange_file_path);
                        
                    } else {
                        
                        reconstruction->FastSliceToVolumeRegistration(iter, 1, stacks.size());
                    }
                    
                    
                    end = std::chrono::system_clock::now();
                    elapsed_seconds = end-start;
                    if (debug)
                        cout << "- duration : " << elapsed_seconds.count() << " s " << endl;

                        start = std::chrono::system_clock::now();

                        cout << "CreateSliceMasks" << endl;
                       reconstruction->CreateSliceMasks(main_mask, iter);
                        
                        end = std::chrono::system_clock::now();
                        elapsed_seconds = end-start;
                        if (debug)
                            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
                    
                    

                }
            
            }

            if ( ! no_log ) {
                cerr.rdbuf (strm_buffer_e);
            }
        }

        
        if (structural_after_1_flag) {
            
            structural_exclusion = true;
            reconstruction->SetStructural(true);
            
        }
        

        if(iter==(iterations-1))
            reconstruction->SetSmoothingParameters(delta,lastIterLambda);
        else
        {
            double l=lambda;
            for (i=0;i<levels;i++) {

                if (iter==iterations*(levels-i-1)/levels)
                    reconstruction->SetSmoothingParameters(delta, l);
                l*=2;
            }
            
        }


        if ( iter<(iterations-1) )
            reconstruction->SpeedupOn();
        else
            reconstruction->SpeedupOff();
        
        
        if(robust_slices_only)
            reconstruction->ExcludeWholeSlicesOnly();
        
        
        start = std::chrono::system_clock::now();

        cout << "InitializeEMValues" << endl;
        reconstruction->InitializeEMValues();
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
        
        
        
        
        start = std::chrono::system_clock::now();

        cout << "CoeffInit" << endl;
        reconstruction->CoeffInit();
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
        
        
        
        start = std::chrono::system_clock::now();

        cout << "GaussianReconstruction" << endl;
        reconstruction->GaussianReconstruction();
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
        
        
        
        
        if (iter == 0 && structural_exclusion && flag_full) {
        
            cout << "SliceToVolumeRegistration" << endl;
            
            if (remote_flag) {
                
                reconstruction->RemoteSliceToVolumeRegistration(iter, 1, stacks.size(), str_mirtk_path, str_current_main_file_path, str_current_exchange_file_path);
            } else {
            
                reconstruction->FastSliceToVolumeRegistration(iter, 2, stacks.size());
            }
            
            
            cout << "CreateSliceMasks" << endl;
            reconstruction->CreateSliceMasks(main_mask, iter);
            
            cout << "CoeffInit" << endl;
            reconstruction->CoeffInit();
            
            cout << "GaussianReconstruction" << endl;
            reconstruction->GaussianReconstruction();
 
        }
        
        

        start = std::chrono::system_clock::now();
        
        cout << "SimulateSlices" << endl;
        reconstruction->SimulateSlices();
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
        
        
        

        start = std::chrono::system_clock::now();
        
        cout << "InitializeRobustStatistics" << endl;
        reconstruction->InitializeRobustStatistics();
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
        


        start = std::chrono::system_clock::now();
        
        cout << "EStep" << endl;
        if(robust_statistics)
            reconstruction->EStep();
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        if (debug)
            cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
        
        
        
        if ( iter==(iterations-1) ) {
            rec_iterations = 20;
            
            if (flag_full)
                rec_iterations = 30;
        }
        else {
            rec_iterations = 7;
            
            if (flag_full)
                rec_iterations = 10;
            
        }



	if (no_longsr_flag) {



		if ( iter==(iterations-1) ) {
		    rec_iterations = 10; 
		}
		else {
		    rec_iterations = 2; 
		}
	}



        
        if (structural_exclusion) {
            
            start = std::chrono::system_clock::now();
            
            cout << "CStep" << endl;
            reconstruction->CStep2D();
            
            end = std::chrono::system_clock::now();
            elapsed_seconds = end-start;
            if (debug)
                cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
            
        }
        
        i=0;
        for (i=0; i<rec_iterations; i++) {
            
            cout << ".........................................................." << endl;
            cout<<endl<<"- Reconstruction iteration : "<<i<<" "<<endl;
            
            if (intensity_matching) {
                
                start = std::chrono::system_clock::now();
                
                cout.rdbuf (fileEv.rdbuf());
                cout << "Bias correction & scaling" << endl;
                if (sigma>0)
                    reconstruction->Bias();

                reconstruction->Scale();
                cout.rdbuf (strm_buffer);
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                if (debug)
                    cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
                
            }
            
            {
                start = std::chrono::system_clock::now();
                
                cout << "Superresolution" << endl;
                reconstruction->Superresolution(i+1);
                
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                if (debug)
                    cout << "- duration : " << elapsed_seconds.count() << " s " << endl;

            }
            
            if (intensity_matching) {
                
                start = std::chrono::system_clock::now();
                
                cout << "NormaliseBias" << endl;
                cout.rdbuf (fileEv.rdbuf());
                if((sigma>0)&&(!global_bias_correction))
                    reconstruction->NormaliseBias(i);
                cout.rdbuf (strm_buffer);
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                if (debug)
                    cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
            }
            
            
            start = std::chrono::system_clock::now();
            
            cout << "Simulateslices" << endl;
            reconstruction->SimulateSlices();
            
            end = std::chrono::system_clock::now();
            elapsed_seconds = end-start;
            if (debug)
                cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
            
            
            
            if(robust_statistics) {
                
                start = std::chrono::system_clock::now();
                
                cout << "MStep" << endl;
                reconstruction->MStep(i+1);
                
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                if (debug)
                    cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
                
            }
            
            if(robust_statistics) {
                
                start = std::chrono::system_clock::now();
                
                cout << "EStep" << endl;
                reconstruction->EStep();
                
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                if (debug)
                    cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
                
            }
        
            if (structural_exclusion && (i<3)) {
                
                start = std::chrono::system_clock::now();
                
                cout << "CStep" << endl;
                reconstruction->CStep2D();
                
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                if (debug)
                    cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
                
            }

            if (iter>1 && debug) {
                reconstructed=reconstruction->GetReconstructed();
                sprintf(buffer,"super-%i.nii.gz",i);
                reconstructed.Write(buffer);
            }
            
            
        }
        
        reconstruction->MaskVolume();
        
        reconstructed=reconstruction->GetReconstructed();
        sprintf(buffer,"image%i.nii.gz",iter);
        reconstructed.Write(buffer);

        if ( ! no_log ) {
            cout.rdbuf (fileEv.rdbuf());
        }
        
        reconstruction->Evaluate(iter);
        
        cout<<endl;
        
        if ( ! no_log ) {
            cout.rdbuf (strm_buffer);
        }
        
    }
    


    cout << ".........................................................." << endl;
    cout << ".........................................................." << endl;

    // reconstruction->SaveSSIM(stacks, 99);

//    if (debug) {
//     reconstruction->SaveTransformations();
//     reconstruction->SaveSlices();
//    }
    
    if (str_current_exchange_file_path.length() > 0) {
        string remove_folder_cmd = "rm -r " + str_current_exchange_file_path + " > tmp-log.txt ";
        int tmp_log_rm = system(remove_folder_cmd.c_str());
    }
    
    if (debug) {
        
        reconstruction->SimulateStacks(stacks, true);
        
        for (unsigned int i=0;i<stacks.size();i++) {
            sprintf(buffer,"again-simulated-%i.nii.gz",i);
            stacks[i]->Write(buffer);
        }
        
    }


    cout << "RestoreSliceIntensities" << endl;
    
    start = std::chrono::system_clock::now();
    
    
    
    if (!no_intensity_absolute) {
        
        reconstruction->RestoreSliceIntensities();
        reconstruction->ScaleVolume();
    
    }
    
    
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    if (debug)
        cout << "- duration : " << elapsed_seconds.count() << " s " << endl;
    
    
    reconstructed=reconstruction->GetReconstructed();
    reconstructed.Write(output_name);
    
    cout << "Recontructed volume : " << output_name << endl;
    cout << ".........................................................." << endl;

    
    if (number_of_channels > 0) {
        
        cout << "Recontructed MC volumes : " << endl;
        reconstruction->SaveMCReconstructed();
        
        cout << ".........................................................." << endl;
    }
    
    
    if (debug) {
        
        reconstruction->SimulateStacks(stacks, true);
        
        for (unsigned int i=0;i<stacks.size();i++) {
            sprintf(buffer,"simulated-%i.nii.gz",i);
            stacks[i]->Write(buffer);
        }
        
    }
     
     

    

    return 0;
}
