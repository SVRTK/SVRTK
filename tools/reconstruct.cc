/*
* SVRTK : SVR reconstruction based on MIRTK
*
* Copyright 2008-2017 Imperial College London
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

#include "mirtk/Reconstruction.h"

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



using namespace mirtk;
using namespace std;

// =============================================================================
//
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: reconstruct [reconstructed] [N] [stack_1] .. [stack_N] <options>\n" << endl;
    cout << endl;
    
    cout << "\t[reconstructed]         Name for the reconstructed volume. Nifti or Analyze format." << endl;
    cout << "\t[N]                     Number of stacks." << endl;
    cout << "\t[stack_1] .. [stack_N]  The input stacks. Nifti or Analyze format." << endl;
    cout << "\t" << endl;
    cout << "Options:" << endl;
    cout << "\t-dofin [dof_1]   .. [dof_N]    The transformations of the input stack to template" << endl;
    cout << "\t                        in \'dof\' format used in IRTK." <<endl;
    cout << "\t                        Only rough alignment with correct orienation and " << endl;
    cout << "\t                        some overlap is needed." << endl;
    cout << "\t                        Use \'id\' for an identity transformation for at least" << endl;
    cout << "\t                        one stack. The first stack with \'id\' transformation" << endl;
    cout << "\t                        will be resampled as template." << endl;
    cout << "\t-template [volume]        Template for registration" << endl;
    cout << "\t-thickness [th_1] .. [th_N] Give slice thickness.[Default: twice voxel size in z direction]"<<endl;
    cout << "\t-default_thickness [d_th] Default thickness for all stacks. [Default: twice voxel size in z direction]"<<endl;
    cout << "\t-mask [mask]              Binary mask to define the region od interest. [Default: whole image]"<<endl;
    cout << "\t-packages [num_1] .. [num_N] Give number of packages used during acquisition for each stack."<<endl;
    cout << "\t                          The stacks will be split into packages during registration iteration 1"<<endl;
    cout << "\t                          and then into odd and even slices within each package during "<<endl;
    cout << "\t                          registration iteration 2. The method will then continue with slice to"<<endl;
    cout << "\t                          volume approach. [Default: slice to volume registration only]"<<endl;
    cout << "\t-template_number          Number of the template stack. [Default: 0]"<<endl;
    cout << "\t-iterations [iter]        Number of registration-reconstruction iterations. [Default: 3]"<<endl;
    cout << "\t-sr_iterations [sr_iter]  Number of SR reconstruction iterations. [Default: 7,...,7,7*3]"<<endl;
    cout << "\t-sigma [sigma]            Stdev for bias field. [Default: 12mm]"<<endl;
    cout << "\t-resolution [res]         Isotropic resolution of the volume. [Default: 0.75mm]"<<endl;
    cout << "\t-multires [levels]        Multiresolution smooting with given number of levels. [Default: 3]"<<endl;
    cout << "\t-average [average]        Average intensity value for stacks [Default: 700]"<<endl;
    cout << "\t-delta [delta]            Parameter to define what is an edge. [Default: 150]"<<endl;
    cout << "\t-lambda [lambda]          Smoothing parameter. [Default: 0.02]"<<endl;
    cout << "\t-lastIter [lambda]        Smoothing parameter for last iteration. [Default: 0.01]"<<endl;
    cout << "\t-smooth_mask [sigma]      Smooth the mask to reduce artefacts of manual segmentation. [Default: 4mm]"<<endl;
    cout << "\t-global_bias_correction   Correct the bias in reconstructed image against previous estimation."<<endl;
    cout << "\t-no_intensity_matching    Switch off intensity matching."<<endl;
    cout << "\t-no_robust_statistics     Switch off robust statistics."<<endl;
    cout << "\t-no_robust_statistics     Switch off robust statistics."<<endl;
    cout << "\t-exclude_wrong_stacks     Automated exclusion of misregistered stacks."<<endl;
    cout << "\t-rescale_stacks           Rescale stacks to avoid nan pixel errors. [Default: False]"<<endl;
    cout << "\t-svr_only                 Only SVR registration to a template stack."<<endl;
    cout << "\t-no_global                No global stack registration."<<endl;
    cout << "\t-exact_thickness          Exact slice thickness without negative gap. [Default: False]"<<endl;
    cout << "\t-ncc                      Use global NCC similarity for SVR steps. [Default: NMI]"<<endl;
    cout << "\t-nmi_bins [nmi_bins]      Number of NMI bins for registration. [Default: 16]"<<endl;
    cout << "\t-structural               Use structrural exclusion of slices at the last iteration."<<endl;
    cout << "\t-exclude_slices_only      Robust statistics for exclusion of slices only."<<endl;
    cout << "\t-template [template]      Use template for initialisation of registration loop. [Default: average of stack registration]"<<endl;
    cerr << "\t-remove_black_background  Create mask from black background."<<endl;
    cerr << "\t-transformations [folder] Use existing slice-to-volume transformations to initialize the reconstruction->"<<endl;
    cerr << "\t-force_exclude [number of slices] [ind1] ... [indN]  Force exclusion of slices with these indices."<<endl;
    cout << "\t-remote                   Run SVR registration as remote functions in case of memory issues [Default: false]."<<endl;
    cout << "\t-full_remote              Run SR+SVR steps as remote functions in case of memory issues (slower option) [Default: false]."<<endl;
    cout << "\t-debug                    Debug mode - save intermediate results."<<endl;
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
    
    
    // -----------------------------------------------------------------------------
    // INPUT VARIABLES, FLAG AND DEFAULT VALUES
    // -----------------------------------------------------------------------------
    
    // path to MIRTK executables (required for calling registration function outside reconstruction )
    const char *current_mirtk_path = argv[0];
    
    // initialisation of MIRTK image reader library
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    // utility variables for time estimation
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    auto start_total = std::chrono::system_clock::now();
    auto end_total = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds;
    std::time_t end_time;
    
    // general utility variables
    int i, j, x, y, z, ok;
    char buffer[256];
    RealImage stack;
    
    // names for output info files
    string info_filename = "slice_info.tsv";
    string log_id;
    string str_flags = " ";
    
    
    // main variables:
    
    // name for output volume
    char * output_name = NULL;
    
    // folder for slice-to-volume registrations, if given
    char * folder=NULL;
    
    // array of stacks and stack names
    Array<RealImage> stacks;
    Array<string> stack_files;
    
    // output volume
    RealImage reconstructed;
    
    // template stack
    RealImage template_stack;
    
    // input mask
    RealImage *mask=NULL;

    // array of stack stack transformation to the template space
    Array<RigidTransformation> stack_transformations;
    
    // array of stack slice thickness
    Array<double> thickness;
    
    // number of input stacks
    int nStacks = 0;
    
    // array of number of packages for each stack
    Array<int> packages;
    
    // masked template stack
    RealImage maskedTemplate;
    
    // output of stack average
    RealImage average;
    
    // variables for forced exclusion of slices
    int number_of_force_excluded_slices = 0;
    vector<int> force_excluded;
    
    
    // default values for reconstruction variables:
    
    int templateNumber=0;
    int iterations = 3;
    int sr_iterations = 7;
    bool debug = false;
    double sigma = 20;
    double resolution = 0.75;
    double lambda = 0.02;
    double delta = 150;
    int levels = 3;
    double lastIterLambda = 0.01;
    int rec_iterations;
    double averageValue = 700;
    double smooth_mask = 4;
    bool global_bias_correction = false;
    double low_intensity_cutoff = 0.01;
    
    
    // flags for reconstruction options:
    
    // flag whether the template (-template option)
    bool use_template = false;
    
    // flag if the exact should be used
    bool flag_no_overlap_thickness = false;
    
    // flag whether the user provided input transformations
    bool have_stack_transformations = false;
    
    //flag to remove black background, e.g. when neonatal motion correction is performed
    bool remove_black_background = false;
    
    //flag to switch the intensity matching on and off
    bool intensity_matching = true;
    
    bool rescale_stacks = false;
    
    // flag whether registration should be used
    bool registration_flag = true;
    
    // flags to switch the robust statistics on and off
    bool robust_statistics = true;
    bool robust_slices_only = false;
    
    // no log flag
    bool no_log = false;
    
    // flag for SVR registration to the template (skips 1st averaging SR iteration)
    bool svr_only = false;
    
    // flag for struture-based exclusion of slices
    bool structural = false;
    
    // flag for switching off NMI and using NCC for SVR step
    bool ncc_reg_flag = false;
    
    // flag for running registration step outside
    bool remote_flag = false;
    
    // flag for no global registration
    bool no_global_flag = false;
    
    // flag that sets slice thickness to 1.5 of spacing (for testing purposes)
    bool thin_flag = false;
    
    // flag for longer reconstruction
    bool full_remote_recon = false;
    
    // flag for automated exclusion of low quality / similarity stacks
    bool exclude_wrong_stacks = false;
    int best_selected_stacks = -1;
    
    
    
    // -----------------------------------------------------------------------------
    // READ INPUT DATA AND OPTIONS
    // -----------------------------------------------------------------------------
    
    // create reconstruction object
    Reconstruction *reconstruction = new Reconstruction();
    
    // if not enough arguments print help
    if (argc < 5)
        usage();

    cout << "------------------------------------------------------" << endl;
    
    // read output name
    output_name = argv[1];
    argc--;
    argv++;
    cout<<"Recontructed volume name : "<<output_name<<endl;
    
    // read number of stacks
    nStacks = atoi(argv[1]);
    argc--;
    argv++;
    cout<<"Number of stacks : "<<nStacks<<endl;
    
    // read input stacks
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    for (i=0; i<nStacks; i++) {
        
        stack_files.push_back(argv[1]);
        cout<<"Stack " << i << " : "<<argv[1];
        tmp_fname = argv[1];
        image_reader.reset(ImageReader::TryNew(tmp_fname));
        tmp_image.reset(image_reader->Run());
        stack = *tmp_image;
        
        // check if the intensity is not negative and correct if so
        double smin, smax;
        stack.GetMinMax(&smin, &smax);
        if (smin < 0 || smax < 0) {
            stack.PutMinMaxAsDouble(0, 1000);
        }
        
        // print stack info
        double tmp_dx = stack.GetXSize(); double tmp_dy = stack.GetYSize();
        double tmp_dz = stack.GetZSize(); double tmp_dt = stack.GetTSize();
        double tmp_sx = stack.GetX(); double tmp_sy = stack.GetY();
        double tmp_sz = stack.GetZ(); double tmp_st = stack.GetT();
        
        cout << "  ;  size : " << tmp_sx << " - " << tmp_sy << " - " << tmp_sz << " - " << tmp_st  << "  ;";
        cout << "  voxel : " << tmp_dx << " - " << tmp_dy << " - " << tmp_dz << " - " << tmp_dt  << "  ;";
        cout << "  range : [" << smin << "; " << smax << "]" << endl;
        
        argc--;
        argv++;
        stacks.push_back(stack);
    }
    
    // default template stack
    template_stack = stacks[0];
    
    // Parse options:
    while (argc > 1) {
        ok = false;
        
        // input stack transformations to the template space
        if ((ok == false) && (strcmp(argv[1], "-dofin") == 0)){
            argc--;
            argv++;
            
            for (i=0;i<nStacks;i++) {
                
                cout<<"Transformation " << i << " : "<<argv[1]<<endl;
                Transformation *t = Transformation::New(argv[1]);
                RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*> (t);
                
                stack_transformations.push_back(*rigidTransf);
                delete rigidTransf;
                
                argc--;
                argv++;
            }
            reconstruction->InvertStackTransformations(stack_transformations);
            have_stack_transformations = true;
        }
        
        // slice thickness per stack
        if ((ok == false) && (strcmp(argv[1], "-thickness") == 0)) {
            argc--;
            argv++;
            cout<< "Slice thickness : ";
            for (i=0;i<nStacks;i++) {
                thickness.push_back(atof(argv[1]));
                cout << thickness[i]<<" ";
                argc--;
                argv++;
            }
            cout<<endl;
            ok = true;
        }

        // number of packages for each stack
        if ((ok == false) && (strcmp(argv[1], "-packages") == 0)) {
            argc--;
            argv++;
            cout << "Package number : ";
            for (i=0; i<nStacks; i++) {
                packages.push_back(atoi(argv[1]));
                cout << packages[i] << " ";
                argc--;
                argv++;
            }
            cout << endl;
            reconstruction->SetNPackages(packages);
            ok = true;
        }
        
        // template for initialiation of registration
        if ((ok == false) && (strcmp(argv[1], "-template") == 0)) {
            argc--;
            argv++;
            cout << "Template : " << argv[1] << endl;
            template_stack.Read(argv[1]);
            
            // check intensities and rescale if required
            double smin, smax;
            template_stack.GetMinMax(&smin, &smax);
            if (smin < 0 || smax < 0) {
                template_stack.PutMinMaxAsDouble(0, 1000);
            }
            
            // extract 1st dynamic
            if (template_stack.GetT() > 1) {
                template_stack = template_stack.GetRegion(0,0,0,0,template_stack.GetX(),template_stack.GetY(),template_stack.GetZ(),1);  
            }

            use_template = true;
            reconstruction->SetTemplateFlag(use_template);
            
            ok = true;
            argc--;
            argv++;
        }
        
        // binary mask for reconsturction / final volume
        if ((ok == false) && (strcmp(argv[1], "-mask") == 0)) {
            argc--;
            argv++;
            cout<<"Mask : "<<argv[1]<<endl;
            mask= new RealImage(argv[1]);
            ok = true;
            argc--;
            argv++;
        }

        // number of registration-reconstruction iterations
        if ((ok == false) && (strcmp(argv[1], "-iterations") == 0)) {
            argc--;
            argv++;
            iterations=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
            str_flags = str_flags + " -iterations " + to_string(iterations) + " ";
        }
        
        // the same thickness for all stacks
        if ((ok == false) && (strcmp(argv[1], "-default_thickness") == 0)) {
            argc--;
            argv++;
            double default_thickness=atof(argv[1]);
            cout << "Slice thickness (default for all stacks): ";
            for (i=0; i<nStacks; i++) {
                thickness.push_back(default_thickness);
                cout << thickness[i] << " ";
            }
            cout << endl;
            ok = true;
            argc--;
            argv++;
        }
        
        // number of SR iterations
        if ((ok == false) && (strcmp(argv[1], "-sr_iterations") == 0)) {
            argc--;
            argv++;
            sr_iterations=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
            str_flags = str_flags + " -sr_iterations " + to_string(sr_iterations) + " ";
        }
        
        // template stack number
        if ((ok == false) && (strcmp(argv[1], "-template_number") == 0)) {
            argc--;
            argv++;
            templateNumber=atoi(argv[1]);
            template_stack = stacks[templateNumber];
            ok = true;
            argc--;
            argv++;
        }
    
        // variance of Gaussian kernel to smooth the bias field
        if ((ok == false) && (strcmp(argv[1], "-sigma") == 0)) {
            argc--;
            argv++;
            sigma=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
            str_flags = str_flags + " -sigma " + to_string(sigma) + " ";
        }
        
        // SR smoothing parameter
        if ((ok == false) && (strcmp(argv[1], "-lambda") == 0)) {
            argc--;
            argv++;
            lambda=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
            str_flags = str_flags + " -lambda " + to_string(lambda) + " ";
        }
        
        // smoothing parameter for last iteration
        if ((ok == false) && (strcmp(argv[1], "-lastIter") == 0)) {
            argc--;
            argv++;
            lastIterLambda=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
            str_flags = str_flags + " -lastIter " + to_string(lastIterLambda) + " ";
        }
        
        // SR parameter to define what is an edge
        if ((ok == false) && (strcmp(argv[1], "-delta") == 0)) {
            argc--;
            argv++;
            delta=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
            str_flags = str_flags + " -delta " + to_string(delta) + " ";
        }
        
        // isotropic resolution for the reconstructed volume
        if ((ok == false) && (strcmp(argv[1], "-resolution") == 0)) {
            argc--;
            argv++;
            resolution=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
            str_flags = str_flags + " -resolution " + to_string(resolution) + " ";
        }
        
        // number of resolution levels
        if ((ok == false) && (strcmp(argv[1], "-multires") == 0)) {
            argc--;
            argv++;
            levels=atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        
        // run registration as remote functions
        if ((ok == false) && (strcmp(argv[1], "-remote") == 0)) {
            argc--;
            argv++;
            remote_flag=true;
            ok = true;
            str_flags = str_flags + " -remote ";
        }
        
        // long reconstruction (10x10x30 iterations)
        if ((ok == false) && (strcmp(argv[1], "-full_remote") == 0)) {
            argc--;
            argv++;
            full_remote_recon=true;
            ok = true;
            remote_flag=true;
            str_flags = str_flags + " -remote ";
        }
        
        // use exact dz as slice thickness
        if ((ok == false) && (strcmp(argv[1], "-exact_thickness") == 0)) {
            argc--;
            argv++;
            flag_no_overlap_thickness=true;
            ok = true;
        }
        
        // rescale input stacks
        if ((ok == false) && (strcmp(argv[1], "-rescale_stacks") == 0)) {
            argc--;
            argv++;
            rescale_stacks=true;
            ok = true;
        }
        
        // use only SVR to the template (skip 1st SR averaging iteration)
        if ((ok == false) && (strcmp(argv[1], "-svr_only") == 0)) {
            argc--;
            argv++;
            svr_only=true;
            ok = true;
            str_flags = str_flags + " -svr_only ";
        }
        
        // no global stack registration
        if ((ok == false) && (strcmp(argv[1], "-no_global") == 0)) {
            argc--;
            argv++;
            no_global_flag=true;
            ok = true;
        }

        // option for 1.5 x dz slice thickness (testing)
        if ((ok == false) && (strcmp(argv[1], "-thin") == 0)) {
            argc--;
            argv++;
            thin_flag=true;
            ok = true;
        }

        // exclude low quality / similarity stacks prior to reconstruction automatically
        if ((ok == false) && (strcmp(argv[1], "-exclude_wrong_stacks") == 0)) {
            argc--;
            argv++;
            exclude_wrong_stacks = true;
            ok = true;
        }
        
        // use NCC similarity metric for SVR
        if ((ok == false) && (strcmp(argv[1], "-ncc") == 0)) {
            argc--;
            argv++;
            ncc_reg_flag=true;
            reconstruction->SetNCC(ncc_reg_flag);
            ok = true;
            str_flags = str_flags + " -ncc ";
        }
        
        
        // read transformations from this folder
        if ((ok == false) && (strcmp(argv[1], "-transformations") == 0)){
            argc--;
            argv++;
            folder=argv[1];
            ok = true;
            argc--;
            argv++;
        }
        
        // remove black background
        if ((ok == false) && (strcmp(argv[1], "-remove_black_background") == 0)){
            argc--;
            argv++;
            remove_black_background=true;
            ok = true;
        }
        
        // force removal of certain slices
        if ((ok == false) && (strcmp(argv[1], "-force_exclude") == 0)){
            argc--;
            argv++;
            number_of_force_excluded_slices = atoi(argv[1]);
            argc--;
            argv++;
            
            cout << number_of_force_excluded_slices << " force excluded slices: ";
            for (i=0; i<number_of_force_excluded_slices; i++)
            {
                force_excluded.push_back(atoi(argv[1]));
                cout << force_excluded[i] << " ";
                argc--;
                argv++;
            }
            cout << endl;
            ok = true;
        }
        
        // switch off intensity matching
        if ((ok == false) && (strcmp(argv[1], "-no_intensity_matching") == 0)) {
            argc--;
            argv++;
            intensity_matching=false;
            ok = true;
            str_flags = str_flags + " -no_intensity_matching ";
        }
        
        // switch off robust statistics
        if ((ok == false) && (strcmp(argv[1], "-no_robust_statistics") == 0)) {
            argc--;
            argv++;
            robust_statistics=false;
            ok = true;
            str_flags = str_flags + " -no_robust_statistics ";
        }

        // sse structural exclusion of slices (after 2nd iteration)
        if ((ok == false) && (strcmp(argv[1], "-structural") == 0)) {
            argc--;
            argv++;
            structural=true;
            ok = true;
            str_flags = str_flags + " -structural ";
        }
        
        // use robust statistics for slices only
        if ((ok == false) && (strcmp(argv[1], "-exclude_slices_only") == 0)) {
            argc--;
            argv++;
            robust_slices_only = true;
            ok = true;
            str_flags = str_flags + " -exclude_slices_only ";
        }
        
        // switch off registration
        if ((ok == false) && (strcmp(argv[1], "-no_registration") == 0)) {
            argc--;
            argv++;
            registration_flag=false;
            ok = true;
            iterations = 1;
            str_flags = str_flags + " -no_registration ";
        }
        
        // perform bias correction of the reconstructed image agains the GW image in the same motion correction iteration
        if ((ok == false) && (strcmp(argv[1], "-global_bias_correction") == 0)) {
            argc--;
            argv++;
            global_bias_correction=true;
            ok = true;
            str_flags = str_flags + " -global_bias_correction ";
        }
        
        // debug mode
        if ((ok == false) && (strcmp(argv[1], "-debug") == 0)) {
            argc--;
            argv++;
            debug=true;
            ok = true;
            str_flags = str_flags + " -debug ";
        }
        
        if (ok == false) {
            cerr << "Cannot parse argument " << argv[1] << endl;
            usage();
        }
    }
    
    
    // -----------------------------------------------------------------------------
    // SET RECONSTRUCTION OPTIONS AND PERFORM PREPROCESSING
    // -----------------------------------------------------------------------------
    
    // set thickess to the exact dz value if specified
    if (flag_no_overlap_thickness && thickness.size() < 1) {
        for (i=0; i<stacks.size(); i++) {
            thickness.push_back(stacks[i].GetZSize());
        }
    }
    
    // check if stacks have multiple dynamics and spit them if it is the case
    bool has_4D_stacks = false;
    for (i=0; i<stacks.size(); i++) {
        if (stacks[i].GetT()>1) {
            has_4D_stacks = true;
            break;
        }
    }

    if (has_4D_stacks) {
        cout << "Splitting stacks into dynamincs ... ";
        
        Array<double> new_thickness;
        Array<int> new_packages;
        Array<RigidTransformation> new_stack_transformations;
        Array<RealImage> new_stacks;
        
        for (i=0; i<stacks.size(); i++) {
            
            if (stacks[i].GetT() == 1) {
                new_stacks.push_back(stacks[i]);
                if (stack_transformations.size()>0)
                    new_stack_transformations.push_back(stack_transformations[i]);
                if (packages.size()>0)
                    new_packages.push_back(packages[i]);
                if (thickness.size()>0)
                    new_thickness.push_back(thickness[i]);
            }
            else {
                for (int t=0; t<stacks[i].GetT(); t++) {
                    stack = stacks[i].GetRegion(0,0,0,t,stacks[i].GetX(),stacks[i].GetY(),stacks[i].GetZ(),(t+1));
                    new_stacks.push_back(stack);
                    if (stack_transformations.size()>0)
                        new_stack_transformations.push_back(stack_transformations[i]);
                    if (packages.size()>0)
                        new_packages.push_back(packages[i]);
                    if (thickness.size()>0)
                        new_thickness.push_back(thickness[i]);
                }
            }
        }
        nStacks = new_stacks.size();
        stacks.clear();
        thickness.clear();
        packages.clear();
        stack_transformations.clear();
        
        cout << "New number of stacks : " << nStacks << endl;
        for (i=0; i<new_stacks.size(); i++) {
            stacks.push_back(new_stacks[i]);
            if (new_thickness.size()>0)
                thickness.push_back(new_thickness[i]);
            if (new_packages.size()>0)
                packages.push_back(new_packages[i]);
            if (new_stack_transformations.size()>0)
                stack_transformations.push_back(new_stack_transformations[i]);
        }
        
        new_stacks.clear();
        new_thickness.clear();
        new_packages.clear();
        new_stack_transformations.clear();
    }

    
    // read path to MIRTK executables for remote registration
    string str_mirtk_path;
    string str_current_main_file_path;
    string str_current_exchange_file_path;
    
    string str_recon_path(current_mirtk_path);
    str_mirtk_path = str_recon_path.substr (0, pos);
    
    size_t pos = str_recon_path.find_last_of("/");
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
        string remove_folder_cmd = "rm -r " + str_current_exchange_file_path + " > tmp-log.txt ";
        int tmp_log_rm = system(remove_folder_cmd.c_str());
        
        string create_folder_cmd = "mkdir " + str_current_exchange_file_path + " > tmp-log.txt ";
        int tmp_log_mk = system(create_folder_cmd.c_str());
        
    } else {
        cout << "System error: could not create a folder for file exchange" << endl;
        exit(1);
    }
   
    
    // rescale stack if specified
    if (rescale_stacks) {
        for (i=0;i<nStacks;i++)
            reconstruction->Rescale(stacks[i],1000);
    }

    // if transformations were not defined by user, set them to identity
    for (i=0;i<nStacks;i++) {
        RigidTransformation *rigidTransf = new RigidTransformation;
        stack_transformations.push_back(*rigidTransf);
        delete rigidTransf;
    }
    
    // initialise 2*slice thickness if not given by user
    if (thickness.size()==0) {
        cout << "Slice thickness : ";
        for (i=0;i<nStacks;i++) {
            double dx,dy,dz;
            stacks[i].GetPixelSize(&dx,&dy,&dz);
            if (!thin_flag) {
                thickness.push_back(dz*2);
            } else {
                thickness.push_back(dz*1.5);
            }
            cout << thickness[i] << " ";
        }
        cout << endl;
    }

    // set debug mode option
    if (debug) reconstruction->DebugOn();
        else reconstruction->DebugOff();
    
    // set force excluded slices option
    reconstruction->SetForceExcludedSlices(force_excluded);
    
    // set low intensity cutoff for bias estimation
    reconstruction->SetLowIntensityCutoff(low_intensity_cutoff)  ;
    
    // if no mask was given - try to create mask from the template image in case it was padded
    if (mask == NULL) {
        mask = new RealImage(stacks[templateNumber]);
        *mask = reconstruction->CreateMask(*mask);
        cout << "Warning : no mask was provided " << endl;
    }
    
    // before creating the template we will crop template stack according to the given mask
    if (mask != NULL)
    {
        // first resample the mask to the space of the stack
        // for template stact the transformation is identity
        RealImage m = *mask;
        reconstruction->TransformMask(stacks[templateNumber],m,stack_transformations[templateNumber]);
        
        // crop template stack and prepare template for global volumetric registration
        maskedTemplate = stacks[templateNumber]*m;
        reconstruction->CropImage(stacks[templateNumber],m);
        reconstruction->CropImage(maskedTemplate,m);
        
        if (debug) {
            m.Write("maskforTemplate.nii.gz");
            stacks[templateNumber].Write("croppedTemplate.nii.gz");
        }
    }
    
    // if the template was provided separately - crop and mask the template with the given mask
    if (use_template) {
        RealImage m = *mask;
        RigidTransformation *rigidTransf = new RigidTransformation;
        reconstruction->TransformMask(template_stack,m,*rigidTransf);
        
        // crop template stack and prepare template for global volumetric registration
        maskedTemplate = template_stack*m;
        reconstruction->CropImage(maskedTemplate,m);
        reconstruction->CropImage(template_stack,m);
    }
 
    // create template volume with isotropic resolution
    // if resolution==0 it will be determined from in-plane resolution of the image
    resolution = reconstruction->CreateTemplate(maskedTemplate, resolution);
    
    // set mask to reconstruction object
    reconstruction->SetMask(mask,smooth_mask);
    
    // if remove_black_background flag is set, create mask from black background of the stacks
    if (remove_black_background) {
        reconstruction->CreateMaskFromBlackBackground(stacks, stack_transformations, smooth_mask);
    }
    
    // options to redirect output from screen to text files
    
    // cout and cerr buffer variables
    streambuf* strm_buffer = cout.rdbuf();
    streambuf* strm_buffer_e = cerr.rdbuf();
    //files for registration output
    string name;
    name = log_id+"log-registration.txt";
    ofstream file(name.c_str());
    name = log_id+"log-registration-error.txt";
    ofstream file_e(name.c_str());
    //files for reconstruction output
    name = log_id+"log-reconstruction.txt";
    ofstream file2(name.c_str());
    name = log_id+"log-evaluation.txt";
    ofstream fileEv(name.c_str());
    
    //set precision
    cout<<setprecision(3);
    cerr<<setprecision(3);
    
    
    // -----------------------------------------------------------------------------
    // RUN GLOBAL STACK REGISTRATION AND FURTHER PREPROCESSING
    // -----------------------------------------------------------------------------
    
    cout << "------------------------------------------------------" << endl;

    // volumetric stack to template registration
    if (!no_global_flag) {
        if (debug) start = std::chrono::system_clock::now();
        cout.rdbuf (file.rdbuf());
        reconstruction->StackRegistrations(stacks,stack_transformations,templateNumber);
        cout.rdbuf (strm_buffer);
        if (debug) {
            end = std::chrono::system_clock::now();
            elapsed_seconds = end-start;
            cout << "StackRegistrations ";
            cout << "- " << elapsed_seconds.count() << "s " << endl;
        }
    }
    
    
    // create average volume
    if (debug) start = std::chrono::system_clock::now();
    average = reconstruction->CreateAverage(stacks,stack_transformations);
    if (debug)
        average.Write("average1.nii.gz");
    
    if (debug) {
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        
        cout << "CreateAverage ";
        cout << "- " << elapsed_seconds.count() << "s " << endl;
    }
    
    // mask is transformed to the all other stacks and they are cropped
    for (i=0; i<nStacks; i++)
    {
        //template stack has been cropped already
        if ((i==templateNumber))
            continue;
        //transform the mask
        RealImage m=reconstruction->GetMask();
        reconstruction->TransformMask(stacks[i],m,stack_transformations[i]);
        //Crop template stack
        reconstruction->CropImage(stacks[i],m);
        
        if (debug) {
            sprintf(buffer,"mask%i.nii.gz",i);
            m.Write(buffer);
            sprintf(buffer,"cropped%i.nii.gz",i);
            stacks[i].Write(buffer);
        }
    }
    
    // remove small stacks (no intersection with ROI)
    Array<RealImage> selected_stacks;
    Array<RigidTransformation> selected_stack_transformations;
    int new_nStacks = 0;
    int new_templateNumber = 0;
    for (i=0; i<nStacks; i++) {
        if (stacks[i].GetX() == 1) {
            cerr << "stack " << i << " has no intersection with ROI" << endl;
            continue;
        }
        selected_stacks.push_back(stacks[i]);
        selected_stack_transformations.push_back(stack_transformations[i]);
        if (i == templateNumber)
            new_templateNumber = templateNumber - (i-new_nStacks);
        new_nStacks++;
    }
    stacks.clear();
    stack_transformations.clear();
    nStacks = new_nStacks;
    templateNumber = new_templateNumber;
    for (i=0; i<nStacks; i++) {
        stacks.push_back(selected_stacks[i]);
        stack_transformations.push_back(selected_stack_transformations[i]);
    }
    
    // perform volumetric registration again
    if (debug) start = std::chrono::system_clock::now();
    if (!no_global_flag) {
        cout.rdbuf (file.rdbuf());
        reconstruction->StackRegistrations(stacks,stack_transformations,templateNumber);
        cout.rdbuf (strm_buffer);
    }
    if (debug) {
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        cout << "StackRegistrations ";
        cout << "- " << elapsed_seconds.count() << "s " << endl;
    }
    
    // exclude low quality / similarity stacks (should be transferred to a separate function)
    if (exclude_wrong_stacks) {

        best_selected_stacks = stacks.size();
        cout << "Selecting stacks : " << " " << endl;
    
        RealImage transformed_template_mask = *mask;
        RealImage template_to_check = template_stack;
        RigidTransformation *tmp_rreg = new RigidTransformation();
        reconstruction->TransformMask(template_to_check, transformed_template_mask, *tmp_rreg);
        reconstruction->CropImage(template_to_check, transformed_template_mask);
        reconstruction->CropImage(transformed_template_mask, transformed_template_mask);

        Array<double> all_ncc_array;
        Array<double> all_slice_ncc_array;
        Array<int> all_indices_array;
        Array<double> all_count_array;
        Array<double> sorted_ncc_array;
        Array<int> sorted_indices_array;
        Array<double> sorted_count_array;
        Array<double> selected_ncc_array;
        Array<int> selected_indices_array;
        double max_ncc = -1.0;
        int max_index = -1;
        int average_count_ncc = 0;
        double average_slice_ncc = 0;
        double average_volume_ncc = 0;
        Array<RealImage> new_stacks;
        Array<double> new_thickness;
        Array<int> new_packages;
        Array<RigidTransformation> new_stack_transformations;

        for (i=0; i<stacks.size(); i++) {
            RealImage stack_to_check = stacks[i];
            Matrix m = stack_transformations[i].GetMatrix();
            stack_to_check.PutAffineMatrix(m, true);
            
            double tx, ty, tz, rx, ry, rz;
            tx = stack_transformations[i].GetTranslationX();
            ty = stack_transformations[i].GetTranslationY();
            tz = stack_transformations[i].GetTranslationZ();
            rx = stack_transformations[i].GetRotationX();
            ry = stack_transformations[i].GetRotationY();
            rz = stack_transformations[i].GetRotationZ();

            double slice_ncc = 0.0;
            slice_ncc = reconstruction->VolumeNCC(stack_to_check, template_to_check, transformed_template_mask);
            double count_ncc = -1;
            double volume_ncc = reconstruction->GlobalNCC(stack_to_check, template_to_check, count_ncc);
            
            average_count_ncc = average_count_ncc + count_ncc;
            average_slice_ncc = average_slice_ncc + slice_ncc;
            average_volume_ncc = average_volume_ncc + volume_ncc;
      
            all_ncc_array.push_back(volume_ncc);
            all_indices_array.push_back(i);
            all_count_array.push_back(count_ncc);
            all_slice_ncc_array.push_back(slice_ncc);
                        
            if (volume_ncc > max_ncc) {
                max_ncc = volume_ncc;
                max_index = i;
            }
        }

        average_count_ncc = ((average_count_ncc/stacks.size()));
        average_slice_ncc = ((average_slice_ncc/stacks.size()));
        average_volume_ncc = ((average_volume_ncc/stacks.size()));
        
        double std_count_ncc = 0;
        double std_slice_ncc = 0;
        double std_volume_ncc = 0;

        for (i=0; i<stacks.size(); i++) {
            std_count_ncc = std_count_ncc + pow((all_count_array[i] - average_count_ncc),2);
            std_slice_ncc = std_slice_ncc + pow((all_slice_ncc_array[i] - average_slice_ncc),2);
            std_volume_ncc = average_volume_ncc + pow((all_ncc_array[i] - average_volume_ncc),2);
        }
            
        std_slice_ncc = std_slice_ncc / stacks.size();
        std_volume_ncc = std_volume_ncc / stacks.size();
        std_count_ncc = std_count_ncc / stacks.size();

        cout << " - average values : volume ncc = " << average_volume_ncc << " +/- " << std_volume_ncc << " ; slice ncc = " << average_slice_ncc << " +/- " << std_slice_ncc  << " ; volume [mm^3] = " << average_count_ncc << "+/- " << std_count_ncc << endl;

        sorted_ncc_array = all_ncc_array;
        sorted_indices_array = all_indices_array;
        
        std::sort(sorted_ncc_array.begin(), sorted_ncc_array.end());
        std::reverse(sorted_ncc_array.begin(), sorted_ncc_array.end());
        
        cout << " - selected : " << endl;
        int total_selected = 0;
        for (int j=0; j<sorted_ncc_array.size(); j++) {
            
            for (i=0; i<stacks.size(); i++) {

                if (total_selected < best_selected_stacks) {
                    if (sorted_ncc_array[j] == all_ncc_array[i] && all_count_array[i] > 0.65*average_count_ncc && all_ncc_array[i] > (average_volume_ncc - 2*std_volume_ncc)) {
                        
                        selected_ncc_array.push_back(all_ncc_array[i]);
                        selected_indices_array.push_back(all_indices_array[i]);
                        total_selected++;
                        new_stacks.push_back(stacks[all_indices_array[i]]);
                        cout << "" << all_indices_array[i] << " : volume ncc = " << all_ncc_array[i] << " ; slice ncc = " << all_slice_ncc_array[i] << endl;
                        
                        if (stack_transformations.size()>0)
                            new_stack_transformations.push_back(stack_transformations[i]);
                        if (packages.size()>0)
                            new_packages.push_back(packages[i]);
                        if (thickness.size()>0)
                            new_thickness.push_back(thickness[i]);
                    }
                }
            }
        }

        nStacks = new_stacks.size();
        stacks.clear();
        thickness.clear();
        packages.clear();
        stack_transformations.clear();

        for (i=0; i<new_stacks.size(); i++) {
            stacks.push_back(new_stacks[i]);
            
            if (new_thickness.size()>0)
                thickness.push_back(new_thickness[i]);
            if (new_packages.size()>0)
                packages.push_back(new_packages[i]);
            if (new_stack_transformations.size()>0)
                stack_transformations.push_back(new_stack_transformations[i]);
        }
        new_stacks.clear();
        new_thickness.clear();
        new_packages.clear();
        new_stack_transformations.clear();
    }

    cout << "------------------------------------------------------" << endl;

    // rescale intensities of the stacks to have the same average
    if (intensity_matching) {
        if (debug) start = std::chrono::system_clock::now();
        cout.rdbuf (file.rdbuf());
        if (intensity_matching)
            reconstruction->MatchStackIntensitiesWithMasking(stacks,stack_transformations,averageValue);
        else
            reconstruction->MatchStackIntensitiesWithMasking(stacks,stack_transformations,averageValue,true);
        cout.rdbuf (strm_buffer);
        if (debug) {
            end = std::chrono::system_clock::now();
            elapsed_seconds = end-start;
            cout << "MatchStackIntensitiesWithMasking ";
            cout << "- " << elapsed_seconds.count() << "s " << endl;
        }
    }

    // create average of the registered stacks
    if (debug) start = std::chrono::system_clock::now();
    cout.rdbuf (file.rdbuf());
    average = reconstruction->CreateAverage(stacks,stack_transformations);
    cout.rdbuf (strm_buffer);
    if (debug)
        average.Write("average2.nii.gz");
    if (debug) {
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        cout << "CreateAverage ";
        cout << "- " << elapsed_seconds.count() << "s " << endl;
    }
    
    // create slices and slice-dependent transformations
    Array<RealImage> probability_maps;
    reconstruction->CreateSlicesAndTransformations(stacks, stack_transformations, thickness, probability_maps);
    
    // if full remote reconstruction option is used - save the model and all outputs
    if (full_remote_recon) {
        reconstruction->SaveModelRemote(str_current_exchange_file_path, 1, 0);
    }
    
    // mask all the slices
    cout.rdbuf (file.rdbuf());
    reconstruction->MaskSlices();
    cout.rdbuf (strm_buffer);
    
    // set sigma for the bias field smoothing
    if (sigma>0)
        reconstruction->SetSigma(sigma);
    else
        reconstruction->SetSigma(20);
    
    // set global bias correction flag
    if (global_bias_correction)
        reconstruction->GlobalBiasCorrectionOn();
    else
        reconstruction->GlobalBiasCorrectionOff();
    
    // if given read slice-to-volume registrations
    if (folder!=NULL)
        reconstruction->ReadTransformation(folder);
    
    // initialise data structures for EM
    reconstruction->InitializeEM();
    
    // if registration was switched off - only 1 iteration is required
    cout.rdbuf (strm_buffer);
    if (!registration_flag) {
        iterations = 1;
    }
    
    // -----------------------------------------------------------------------------
    // RUN INTERLEAVED SVR-SR RECONSTRUCTION
    // -----------------------------------------------------------------------------
    
    int current_iteration = 0;

    if (full_remote_recon && (packages.size()<1) ) {
    
        // run reconstruction remotely ("reconstruct-round" function) - can be removed
        for (int iter=0; iter<iterations; iter++) {
            current_iteration = iter;
            int current_iteration = iter;
            string str_iter = to_string(current_iteration);
            string str_num_slices = to_string(reconstruction->_number_of_slices_org);
            string str_thickness = to_string(reconstruction->_average_thickness_org);
            string reconstruct_cmd = str_mirtk_path + "/reconstruct-round " + " " + str_mirtk_path + " " + str_current_main_file_path + " " + str_current_exchange_file_path + " " + str_iter + " " + str_num_slices + " " + str_thickness + " " + str_flags;
            int tmp_log = system(reconstruct_cmd.c_str());
        }

    } else {

        //interleaved registration-reconstruction iterations
        for (int iter=0; iter<iterations; iter++) {

            cout << "------------------------------------------------------" << endl;
            cout<<"Iteration : " << iter << endl;
            
            reconstruction->MaskVolume();
            
            // if only SVR option is used - skip 1st SR only averaging
            if (svr_only) {
                if (debug) start = std::chrono::system_clock::now();

                cout.rdbuf (file.rdbuf());
                if (remote_flag) {
                    reconstruction->RemoteSliceToVolumeRegistration(iter, str_mirtk_path, str_current_main_file_path, str_current_exchange_file_path);
                } else {
                    reconstruction->SliceToVolumeRegistration();
                }
                cout.rdbuf (strm_buffer);
                
                if (debug) {
                    end = std::chrono::system_clock::now();
                    elapsed_seconds = end-start;
                    cout << "SliceToVolumeRegistration ";
                    cout << "- " << elapsed_seconds.count() << "s " << endl;
                }
            }
            else {
                
                if (iter>0) {
                    // run package-based registartion is the number of packages was given
                    if((packages.size()>0)&&(iter<iterations-1)) {
                        
                        if (debug) start = std::chrono::system_clock::now();
                        cout.rdbuf (strm_buffer);
                        reconstruction->PackageToVolume(stacks, packages, stack_transformations);
                        if (debug) {
                            end = std::chrono::system_clock::now();
                            elapsed_seconds = end-start;
                            cout << "PackageToVolume ";
                            cout << "- " << elapsed_seconds.count() << "s " << endl;
                        }
                    }
                    else {
                        
                        // run
                        if (debug) start = std::chrono::system_clock::now();
                        cout.rdbuf (file.rdbuf());
                        if (remote_flag) {
                            reconstruction->RemoteSliceToVolumeRegistration(iter, str_mirtk_path, str_current_main_file_path, str_current_exchange_file_path);
                        } else {
                            reconstruction->SliceToVolumeRegistration();
                        }
                        cout.rdbuf (strm_buffer);
                        
                        if (debug) {
                            end = std::chrono::system_clock::now();
                            elapsed_seconds = end-start;
                            cout << "SliceToVolumeRegistration ";
                            cout << "- " << elapsed_seconds.count() << "s " << endl;
                        }
                        
                    }

                }
                
            }
            
            // run structure-based outlier rejection if specified
            if (structural && iter>1) {
                 cout.rdbuf (strm_buffer);
                if (debug) start = std::chrono::system_clock::now();
                reconstruction->StructuralExclusion();
                if (debug) {
                    end = std::chrono::system_clock::now();
                    elapsed_seconds = end-start;
                    cout << "StructuralExclusion ";
                    cout << "- " << elapsed_seconds.count() << "s " << endl;
                }
                cout.rdbuf (file.rdbuf());
            }
 
            // set smoothing parameters
            // amount of smoothing (given by lambda) is decreased with improving alignment
            // delta (to determine edges) stays constant throughout
            cout.rdbuf (file.rdbuf());
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
            cout.rdbuf (strm_buffer);
            
            // use faster reconstruction during iterations and slower for final reconstruction
            if ( iter<(iterations-1) )
                reconstruction->SpeedupOn();
            else
                reconstruction->SpeedupOff();
            
            if(robust_slices_only)
                reconstruction->ExcludeWholeSlicesOnly();
            
            if (debug) start = std::chrono::system_clock::now();
            cout.rdbuf (file.rdbuf());
            reconstruction->InitializeEMValues();
            cout.rdbuf (strm_buffer);
            if (debug) {
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                cout << "InitializeEMValues ";
                cout << "- " << elapsed_seconds.count() << "s " << endl;
            }

            // calculate matrix of transformation between voxels of slices and volume
            if (debug) start = std::chrono::system_clock::now();
            reconstruction->CoeffInit();
            if (debug) {
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                cout << "CoeffInit ";
                cout << "- " << elapsed_seconds.count() << "s " << endl;
            }
            
            if (debug) start = std::chrono::system_clock::now();
            // initialize reconstructed image with Gaussian weighted reconstruction
            
            cout.rdbuf (file.rdbuf());
            reconstruction->GaussianReconstruction();
            cout.rdbuf (strm_buffer);
            if (debug) {
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                cout << "GaussianReconstruction ";
                cout << "- " << elapsed_seconds.count() << "s " << endl;
            }
            
            // simulate slices (needs to be done after Gaussian reconstruction)
            if (debug) start = std::chrono::system_clock::now();
            cout.rdbuf (file.rdbuf());
            reconstruction->SimulateSlices();
            cout.rdbuf (strm_buffer);
            if (debug) {
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                cout << "SimulateSlices ";
                cout << "- " << elapsed_seconds.count() << "s " << endl;
            }
            
            // initialize robust statistics parameters
            cout.rdbuf (file.rdbuf());
            reconstruction->InitializeRobustStatistics();
            cout.rdbuf (strm_buffer);
            
            // EStep
            if(robust_statistics) {
                cout.rdbuf (file.rdbuf());
                reconstruction->EStep();
                cout.rdbuf (strm_buffer);
            }
            
            // set number of reconstruction iterations
            if ( iter==(iterations-1) )
                rec_iterations = sr_iterations*3;
            else
                rec_iterations = sr_iterations;
            
            // SR reconstruction loop
            for (i=0; i<rec_iterations; i++) {
                
                if (debug) {
                    cout << "------------------------------------------------------" << endl;
                    cout<<"Reconstruction iteration : "<<i<<endl;
                }
                
                if (intensity_matching) {
                    
                    //calculate bias fields
                    if (debug) start = std::chrono::system_clock::now();
                    cout.rdbuf (file.rdbuf());
                    if (sigma>0)
                        reconstruction->Bias();
                    cout.rdbuf (strm_buffer);
                    if (debug) {
                        end = std::chrono::system_clock::now();
                        elapsed_seconds = end-start;
                        cout << "Bias ";
                        cout << "- " << elapsed_seconds.count() << "s " << endl;
                    }

                    //calculate scales
                    if (debug) start = std::chrono::system_clock::now();
                    cout.rdbuf (file.rdbuf());
                    reconstruction->Scale();
                    cout.rdbuf (strm_buffer);
                    if (debug) {
                        end = std::chrono::system_clock::now();
                        elapsed_seconds = end-start;
                        cout << "Scale ";
                        cout << "- " << elapsed_seconds.count() << "s " << endl;
                    }
                    
                }
                
                // update reconstructed volume - super-resolution reconstruction
                if (debug) start = std::chrono::system_clock::now();
                cout.rdbuf (file.rdbuf());
                reconstruction->Superresolution(i+1);
                cout.rdbuf (strm_buffer);
                if (debug) {
                    end = std::chrono::system_clock::now();
                    elapsed_seconds = end-start;
                    cout << "Superresolution ";
                    cout << "- " << elapsed_seconds.count() << "s " << endl;
                }
                
                // run bias normalisation
                if (intensity_matching) {
                    if (debug) start = std::chrono::system_clock::now();
                    cout.rdbuf (file.rdbuf());
                    if((sigma>0)&&(!global_bias_correction))
                        reconstruction->NormaliseBias(i);
                    cout.rdbuf (strm_buffer);
                    if (debug) {
                        end = std::chrono::system_clock::now();
                        elapsed_seconds = end-start;
                        cout << "NormaliseBias ";
                        cout << "- " << elapsed_seconds.count() << "s " << endl;
                    }
                }
                
                // simulate slices (needs to be done after the update of the reconstructed volume)
                if (debug) start = std::chrono::system_clock::now();
                
                cout.rdbuf (file.rdbuf());
                reconstruction->SimulateSlices();
                cout.rdbuf (strm_buffer);
                if (debug) {
                    end = std::chrono::system_clock::now();
                    elapsed_seconds = end-start;
                    cout << "SimulateSlices ";
                    cout << "- " << elapsed_seconds.count() << "s " << endl;
                }
                
                // run robust statistics for rejection of outliers
                if(robust_statistics) {
                    if (debug) start = std::chrono::system_clock::now();
                    
                    cout.rdbuf (file.rdbuf());
                    reconstruction->MStep(i+1);
                    cout.rdbuf (strm_buffer);
                    
                    cout.rdbuf (file.rdbuf());
                    reconstruction->EStep();
                    cout.rdbuf (strm_buffer);
                    
                    if (debug) {
                        end = std::chrono::system_clock::now();
                        elapsed_seconds = end-start;
                        cout << "Robust statistics ";
                        cout << "- " << elapsed_seconds.count() << "s " << endl;
                    }
                }
                
                // save intermediate reconstructed image
                if (debug) {
                    reconstructed=reconstruction->GetReconstructed();
                    sprintf(buffer,"super%i.nii.gz",i);
                    reconstructed.Write(buffer);
                }
                
                // evaluate reconstruction quality
                if (debug) {
                    double error = reconstruction->EvaluateReconQuality(1);
                    cout << "Total reconstruction error : " << error << endl;
                }

            } //end of SR reconstruction iterations
            
            // mask reconstructed image to ROI given by the mask
            reconstruction->MaskVolume();
            
            // save reconstructed image
            reconstructed = reconstruction->GetReconstructed();
            sprintf(buffer, "image%i.nii.gz", iter);
            reconstructed.Write(buffer);

            // compute and save quality metrics
            double out_ncc = 0;
            double out_nrmse = 0;
            double average_volume_weight = 0;
            double ratio_excluded = 0;
            reconstruction->ReconQualityReport(out_ncc, out_nrmse, average_volume_weight, ratio_excluded);
            cout << " - global metrics: ncc = " << out_ncc << " ; nrmse = " << out_nrmse << " ; average weight = " << average_volume_weight << " ; excluded slices = " << ratio_excluded << endl;
            {
                name = "output-metric-ncc.txt";
                ofstream f_out_ncc(name.c_str());
                name = "output-metric-nrmse.txt";
                ofstream f_out_nrmse(name.c_str());
                name = "output-metric-average-weight.txt";
                ofstream f_out_weight(name.c_str());
                name = "output-metric-excluded-ratio.txt";
                ofstream f_out_excluded(name.c_str());
                
                cout.rdbuf (f_out_ncc.rdbuf());
                cout << out_ncc << endl;
                
                cout.rdbuf (f_out_nrmse.rdbuf());
                cout << out_nrmse << endl;
                
                cout.rdbuf (f_out_weight.rdbuf());
                cout << average_volume_weight << endl;
                
                cout.rdbuf (f_out_excluded.rdbuf());
                cout << ratio_excluded << endl;
                
                cout.rdbuf (strm_buffer);
            }

//            reconstruction->SaveSliceInfo(iter);

            //Evaluate - write number of included/excluded/outside/zero slices in each iteration in the file
            if (debug) {
                if ( ! no_log ) {
                    cout.rdbuf (fileEv.rdbuf());
                }
                reconstruction->Evaluate(iter);
                if ( ! no_log ) {
                    cout.rdbuf (strm_buffer);
                }
            }

        } // end of interleaved registration-reconstruction iterations
        
        
        // -----------------------------------------------------------------------------
        // SAVE RESULTS
        // -----------------------------------------------------------------------------
        
        cout << "------------------------------------------------------" << endl;
        
        reconstruction->RestoreSliceIntensities();
        
        if (debug) {
            reconstruction->SaveTransformations();
            reconstruction->SaveSlices();
        }

            if(debug)
            {
                reconstruction->SaveWeights();
                reconstruction->SaveBiasFields();
                reconstruction->SimulateStacks(stacks);
                for (unsigned int i=0;i<stacks.size();i++)
                {
                    sprintf(buffer,"simulated%i.nii.gz",i);
                    stacks[i].Write(buffer);
                }
            }
        
        reconstruction->ScaleVolume();
    }
        
    // load results if remore reconstruction was used - this optin can be removed
    if (full_remote_recon) {
        reconstruction->LoadResultsRemote(str_current_exchange_file_path, reconstruction->_number_of_slices_org, current_iteration);
        reconstruction->ScaleVolume();
    }

    // save final result
    if (str_current_exchange_file_path.length() > 0) {
        string remove_folder_cmd = "rm -r " + str_current_exchange_file_path + " > tmp-log.txt ";
        int tmp_log_rm = system(remove_folder_cmd.c_str());
    }

    reconstructed=reconstruction->GetReconstructed();
    reconstructed.Write(output_name);
    
    cout << "Output volume : " << output_name << endl;
    
    end_total = std::chrono::system_clock::now();
    elapsed_seconds = end_total-start_total;
    cout << "Total time:" << elapsed_seconds.count() << "s " << endl;

    cout << "------------------------------------------------------" << endl;
    
    return 0;
}

