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

using namespace std;
using namespace mirtk;
using namespace svrtk;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: reconstructAngio [reconstructed] [N] [stack_1] .. [stack_N] <options>\n" << endl;
    cout << endl;
    
    cout << "\t[reconstructed]         Name for the reconstructed volume. Nifti or Analyze format." << endl;
    cout << "\t[N]                     Number of stacks." << endl;
    cout << "\t[stack_1] .. [stack_N]  The input stacks. Nifti or Analyze format." << endl;
    cout << "\t" << endl;
    cout << "Options:" << endl;
    cout << "\t-template [volume]        Template for registration" << endl;
    cout << "\t-dofin [dof_1]   .. [dof_N]    The transformations of the input stack to template" << endl;
    cout << "\t                        in \'dof\' format used in IRTK." <<endl;
    cout << "\t                        Only rough alignment with correct orienation and " << endl;
    cout << "\t                        some overlap is needed." << endl;
    cout << "\t                        Use \'id\' for an identity transformation for at least" << endl;
    cout << "\t                        one stack. The first stack with \'id\' transformation" << endl;
    cout << "\t                        will be resampled as template." << endl;
    cout << "\t-thickness [th_1] .. [th_N] Give slice thickness.[Default: twice voxel size in z direction]"<<endl;
    cout << "\t-mask [mask]              Binary mask to define the region od interest. [Default: whole image]"<<endl;
    cout << "\t-packages [num_1] .. [num_N] Give number of packages used during acquisition for each stack."<<endl;
    cout << "\t                          The stacks will be split into packages during registration iteration 1"<<endl;
    cout << "\t                          and then into odd and even slices within each package during "<<endl;
    cout << "\t                          registration iteration 2. The method will then continue with slice to"<<endl;
    cout << "\t                          volume approach. [Default: slice to volume registration only]"<<endl;
    cout << "\t-template_number          Number of the template stack. [Default: 0]"<<endl;
    cout << "\t-iterations [iter]        Number of registration-reconstruction iterations. [Default: 3]"<<endl;
    cout << "\t-sigma [sigma]            Stdev for bias field. [Default: 12mm]"<<endl;
    cout << "\t-resolution [res]         Isotropic resolution of the volume. [Default: 0.75mm]"<<endl;
    cout << "\t-multires [levels]        Multiresolution smooting with given number of levels. [Default: 3]"<<endl;
    cout << "\t-average [average]        Average intensity value for stacks [Default: 700]"<<endl;
    cout << "\t-delta [delta]            Parameter to define what is an edge. [Default: 150]"<<endl;
    cout << "\t-lambda [lambda]          Smoothing parameter. [Default: 0.02]"<<endl;
    cout << "\t-lastIter [lambda]        Smoothing parameter for last iteration. [Default: 0.01]"<<endl;
    cout << "\t-smooth_mask [sigma]      Smooth the mask to reduce artefacts of manual segmentation. [Default: 4mm]"<<endl;
    cout << "\t-nmi_bins [nmi_bins]      Number of NMI bins for registration. [Default: 16]"<<endl;
    cout << "\t-ncc                      Use global NCC similarity for SVR steps. [Default: NMI]"<<endl;
    cout << "\t-svr_only                 Only SVR registration to a template stack."<<endl;
    cout << "\t-no_sr                    Switch off SR PSF."<<endl;
    cout << "\t-global_bias_correction   Correct the bias in reconstructed image against previous estimation."<<endl;
    cout << "\t-no_intensity_matching    Switch off intensity matching."<<endl;
    cout << "\t-no_robust_statistics     Switch off robust statistics."<<endl;
    cout << "\t-exclude_slices_only      Robust statistics for exclusion of slices only."<<endl;
    cout << "\t-remove_black_background  Create mask from black background."<<endl;
    cout << "\t-transformations [folder] Use existing slice-to-volume transformations to initialize the reconstruction->"<<endl;
    cout << "\t-force_exclude [number of slices] [ind1] ... [indN]  Force exclusion of slices with these indices."<<endl;
    cout << "\t-excluded_file [file]     .txt file with the excluded slice numbers."<<endl;
    cout << "\t-exclude_entirely [number of slices] [ind1] ... [indN]  Entirely exlude slices from processing."<<endl;
    cout << "\t-gaussian_only            Only Gaussian PSF interpolation."<<endl;
    cout << "\t-denoise                  Apply NLM denoising."<<endl;
    cout << "\t-blur                     Apply Gaussian filtering for slices with sigma = 0.75 x voxel size."<<endl;
    cout << "\t-filter [sigma]           Apply background filtering (based on non-uniform lighting correction) with sigma defining "<< endl;
    cout << "\t                          background features (use values from [5; 10] range)."<<endl;
    cout << "\t-ffd                      Use FFD registration for SVR."<<endl;
    cout << "\t-cp_spacing [spacing]     Specify CP spacing (in mm) for FFD registration [Default: 5 * min voxel size]."<<endl;
    cout << "\t-sr_iterations            Number of SR iterations in the last round."<<endl;
    cout << "\t-reg_log                  Print registration log."<<endl;
    cout << "\t-remote                   Run SVR registration as remote functions in case of memory issues [Default: false]."<<endl;
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
    const char *current_mirtk_path = argv[0];
    
    cout << "---------------------------------------------------------------------" << endl;
    
    //utility variables
    int i, j, x, y, z, ok;
    char buffer[256];
    RealImage stack;
    RealImage maskedTemplate;
    
    //declare variables for input
    /// Name for output volume
    char * output_name = NULL;
    /// Slice stacks
    Array<RealImage> stacks;
    Array<string> stack_files;
    /// Stack transformation
    Array<RigidTransformation> stack_transformations;
    /// user defined transformations
    bool have_stack_transformations = false;
    /// Stack thickness
    Array<double> thickness;
    ///number of stacks
    int nStacks;
    /// number of packages for each stack
    Array<int> packages;
    
    // Numbers of NMI bins for registration
    int nmi_bins = -1; //16;
    
    // Default values.
    int templateNumber=0;
    RealImage *mask=NULL;
    int iterations = 3;
    bool debug = false;
    double sigma = 20;
    double resolution = 0.75;
    double lambda = 0.025;
    double delta = 200;
    int levels = 3;
    double lastIterLambda = 0.02;
    int rec_iterations;
    double averageValue = 500;
    double smooth_mask = 4;
    bool global_bias_correction = false;
    double low_intensity_cutoff = 0.01;
    //folder for slice-to-volume registrations, if given
    char * folder=NULL;
    //flag to remove black background, e.g. when neonatal motion correction is performed
    bool remove_black_background = false;
    //flag to switch the intensity matching on and off
    bool intensity_matching = true;
    bool rescale_stacks = false;
    bool registration_flag = true;
    
    //flags to switch the robust statistics on and off
    bool robust_statistics = true;
    bool robust_slices_only = false;
    
    //flag for SVR registration to a template (without packages)
    bool svr_only = true;
    
    // flag for filtering
    double fg_sigma = 0.3;
    double bg_sigma = 5; //10;
    
    bool flag_filter = false;
    bool flag_ffd = false;
    bool flag_blurring = false;
    bool flag_no_sr = false;
    
    //flag for switching off NMI and using NCC for SVR step
    bool ncc_reg_flag = false;
    
    //flag to replace super-resolution reconstruction by multilevel B-spline interpolation
    bool bspline = false;
    
    bool remote_flag = false;
    bool template_flag = false;
    
    int cp_spacing = -1;
    
    RealImage average;
    
    string info_filename = "slice_info.tsv";
    string log_id;
    bool no_log = false;
    
    //forced exclusion of slices
    int number_of_force_excluded_slices = 0;
    vector<int> force_excluded;
    
    
    RealImage template_stack;
    
    bool gaussian_only = false;
    
    bool flag_denoise = false;
    
    
    int last_rec_iterations = 10;
    
    
    //Create reconstruction object
    Reconstruction *reconstruction = new Reconstruction();
    
    //if not enough arguments print help
    if (argc < 5)
        usage();
    
    //read output name
    output_name = argv[1];
    argc--;
    argv++;
    cout<<"Recontructed volume name : "<<output_name<<endl;
    
    //read number of stacks
    nStacks = atoi(argv[1]);
    argc--;
    argv++;
    cout<<"Number of stacks : "<<nStacks<<endl;
    
    // Read stacks
    
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    for (i=0;i<nStacks;i++) {
        
        stack_files.push_back(argv[1]);
        
        cout<<"Reading stack : "<<argv[1]<<endl;
        
        tmp_fname = argv[1];
        image_reader.reset(ImageReader::TryNew(tmp_fname));
        tmp_image.reset(image_reader->Run());
        
        stack = *tmp_image;
        
        double smin, smax;
        stack.GetMinMax(&smin, &smax);
        
        if (smin < 0 || smax < 0) {
            stack.PutMinMaxAsDouble(0, 1000);
        }
        
        argc--;
        argv++;
        stacks.push_back(stack);
    }
    
    template_stack = stacks[0];
    
    
    // Parse options.
    while (argc > 1) {
        ok = false;
        
        //Read stack transformations
        if ((ok == false) && (strcmp(argv[1], "-dofin") == 0)){
            argc--;
            argv++;
            
            for (i=0;i<nStacks;i++) {
                
                cout<<"Reading transformation : "<<argv[1]<<endl;
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
        
        //Read slice thickness
        if ((ok == false) && (strcmp(argv[1], "-thickness") == 0)) {
            argc--;
            argv++;
            cout<< "Slice thickness : ";
            for (i=0;i<nStacks;i++) {
                thickness.push_back(atof(argv[1]));
                cout<<thickness[i]<<" ";
                argc--;
                argv++;
            }
            cout<<endl;
            ok = true;
        }
        
        //Read number of packages for each stack
        if ((ok == false) && (strcmp(argv[1], "-packages") == 0)) {
            argc--;
            argv++;
            cout<< "Package number : ";
            for (i=0;i<nStacks;i++) {
                packages.push_back(atoi(argv[1]));
                cout<<packages[i]<<" ";
                argc--;
                argv++;
            }
            cout<<endl;
            ok = true;
        }
        
        //Read binary mask for final volume
        if ((ok == false) && (strcmp(argv[1], "-mask") == 0)) {
            argc--;
            argv++;
            
            cout<<"Reading mask : "<<argv[1]<<endl;
            
            mask= new RealImage(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Read template for registration
        if ((ok == false) && (strcmp(argv[1], "-template") == 0)) {
            argc--;
            argv++;
            
            cout<<"Reading template : "<<argv[1]<<endl;
            
            template_stack.Read(argv[1]);
            
            double smin, smax;
            template_stack.GetMinMax(&smin, &smax);
            
            if (smin < 0 || smax < 0) {
                template_stack.PutMinMaxAsDouble(0, 1000);
            }
            
            template_flag = true;
            
            ok = true;
            argc--;
            argv++;
        }
        
        
        //Read number of registration-reconstruction iterations
        if ((ok == false) && (strcmp(argv[1], "-iterations") == 0)) {
            argc--;
            argv++;
            iterations=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        if ((ok == false) && (strcmp(argv[1], "-cp_spacing") == 0)) {
            argc--;
            argv++;
            
            cp_spacing=atoi(argv[1]);
            reconstruction->SetCP(cp_spacing);
            cout << "CP spacing for FFD : " << cp_spacing << endl;
            
            ok = true;
            argc--;
            argv++;
        }
        
        
        //Read template number
        if ((ok == false) && (strcmp(argv[1], "-template_number") == 0)) {
            argc--;
            argv++;
            templateNumber=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Variance of Gaussian kernel to smooth the bias field.
        if ((ok == false) && (strcmp(argv[1], "-sigma") == 0)) {
            argc--;
            argv++;
            sigma=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Smoothing parameter
        if ((ok == false) && (strcmp(argv[1], "-lambda") == 0)) {
            argc--;
            argv++;
            lambda=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Smoothing parameter for last iteration
        if ((ok == false) && (strcmp(argv[1], "-lastIter") == 0)) {
            argc--;
            argv++;
            lastIterLambda=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Parameter to define what is an edge
        if ((ok == false) && (strcmp(argv[1], "-delta") == 0)) {
            argc--;
            argv++;
            delta=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Isotropic resolution for the reconstructed volume
        if ((ok == false) && (strcmp(argv[1], "-resolution") == 0)) {
            argc--;
            argv++;
            resolution=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        
        //Number of resolution levels
        if ((ok == false) && (strcmp(argv[1], "-multires") == 0)) {
            argc--;
            argv++;
            levels=atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        
        
        //Bins for NMI registration
        if ((ok == false) && (strcmp(argv[1], "-nmi_bins") == 0)){
            argc--;
            argv++;
            nmi_bins=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }

        
        //Use NCC similarity metric for SVR
        if ((ok == false) && (strcmp(argv[1], "-ncc") == 0)) {
            argc--;
            argv++;
            ncc_reg_flag=true;
            reconstruction->SetNCC(ncc_reg_flag);
            ok = true;
        }


        //Print registration log
        if ((ok == false) && (strcmp(argv[1], "-reg_log") == 0)) {
            argc--;
            argv++;
            bool flag_reg_log=true;
            reconstruction->SetRegLog(flag_reg_log);
            ok = true;
        }
        
        
        //Switch off SR for reconstruction
        if ((ok == false) && (strcmp(argv[1], "-no_sr") == 0)) {
            argc--;
            argv++;
            flag_no_sr=true;
            reconstruction->SetSR(flag_no_sr);
            ok = true;
        }
        
        //Read transformations from this folder
        if ((ok == false) && (strcmp(argv[1], "-transformations") == 0)){
            argc--;
            argv++;
            folder=argv[1];
            ok = true;
            argc--;
            argv++;
        }
        
        //Remove black background
        if ((ok == false) && (strcmp(argv[1], "-remove_black_background") == 0)){
            argc--;
            argv++;
            remove_black_background=true;
            ok = true;
        }
        
        
        //Gaussian only
        if ((ok == false) && (strcmp(argv[1], "-gaussian_only") == 0)){
            argc--;
            argv++;
            gaussian_only=true;
            ok = true;
        }
        
        //Registration log
        if ((ok == false) && (strcmp(argv[1], "-reg_log") == 0)){
            argc--;
            argv++;
            gaussian_only=true;
            ok = true;
        }
        
        
        //Blurring filtering
        if ((ok == false) && (strcmp(argv[1], "-blur") == 0)){
            argc--;
            argv++;
            flag_blurring=true;
            
            reconstruction->SetBlurring(flag_blurring);
            
            ok = true;
        }
        
        
        //Apply NLM denoising as preprocessing
        if ((ok == false) && (strcmp(argv[1], "-denoise") == 0)){
            argc--;
            argv++;
            flag_denoise=true;
            ok = true;
        }
        
        
        //Apply FFD registration
        if ((ok == false) && (strcmp(argv[1], "-ffd") == 0)){
            argc--;
            argv++;
            flag_ffd=true;
            
            reconstruction->SetFFD(flag_ffd);
            cout << "Registration type : FFD" << endl;

            ok = true;
        }
        
        
        //Read excluded slices from file
        if ((ok == false) && (strcmp(argv[1], "-excluded_file") == 0)) {
            
            argc--;
            argv++;
            
            //Read the name of the text file with gradient values
            char *e_file = argv[1];
            
            reconstruction->_excluded_entirely.clear();
            
            //Read the gradient values from the text file
            ifstream in_e_file(e_file);
            double num;
            i = 0;
            cout<<"Reading excluded slices from " << e_file << " : "<<endl;
            if (in_e_file.is_open()) {

                while (!in_e_file.eof()) {
                    in_e_file >> num;
                    reconstruction->_excluded_entirely.push_back(num);
                    
                }
                in_e_file.close();
                
                for (i=0; i<reconstruction->_excluded_entirely.size(); i++)
                    
                    cout << reconstruction->_excluded_entirely[i] << " ";
                cout << endl;
            }
            else {
                cout << "Unable to open file " << e_file << endl;
                exit(1);
            }
            
            number_of_force_excluded_slices = reconstruction->_excluded_entirely.size();
            
            cout <<"Number of excluded slices : " << number_of_force_excluded_slices << endl;
            
            argc--;
            argv++;
            
            ok = true;
        }
        
        //entire removal of certain slices from the pipeline
        if ((ok == false) && (strcmp(argv[1], "-exclude_entirely") == 0)){
            argc--;
            argv++;
            int number_of_excluded_slices = atoi(argv[1]);
            argc--;
            argv++;
            
            cout<< number_of_excluded_slices<< " entirely excluded slices: ";
            for (i=0;i<number_of_excluded_slices;i++)
            {
                int num = atoi(argv[1]);
                reconstruction->_excluded_entirely.push_back(num);
                cout<<num<<" ";
                argc--;
                argv++;
            }
            cout<<"."<<endl;
            
            ok = true;
        }

        //Force removal of certain slices
        if ((ok == false) && (strcmp(argv[1], "-force_exclude") == 0)){
            argc--;
            argv++;
            number_of_force_excluded_slices = atoi(argv[1]);
            argc--;
            argv++;
            
            cout<< number_of_force_excluded_slices<< " force excluded slices: ";
            for (i=0;i<number_of_force_excluded_slices;i++)
            {
                force_excluded.push_back(atoi(argv[1]));
                cout<<force_excluded[i]<<" ";
                argc--;
                argv++;
            }
            cout<<"."<<endl;
            
            ok = true;
        }
        
        //Switch off intensity matching
        if ((ok == false) && (strcmp(argv[1], "-no_intensity_matching") == 0)) {
            argc--;
            argv++;
            intensity_matching=false;
            ok = true;
        }
        
        
        //SVR reconstruction as remote functions
        if ((ok == false) && (strcmp(argv[1], "-remote") == 0)) {
            argc--;
            argv++;
            remote_flag=true;
            ok = true;
        }
        
        
        //Switch off robust statistics
        if ((ok == false) && (strcmp(argv[1], "-no_robust_statistics") == 0)) {
            argc--;
            argv++;
            robust_statistics=false;
            ok = true;
        }
        
        //Robust statistics for slices only
        if ((ok == false) && (strcmp(argv[1], "-exclude_slices_only") == 0)) {
            argc--;
            argv++;
            robust_slices_only = true;
            ok = true;
        }
        
        //Switch off registration
        if ((ok == false) && (strcmp(argv[1], "-no_registration") == 0)) {
            argc--;
            argv++;
            registration_flag=false;
            ok = true;
        }
        
        //Use only SVR to a template
        if ((ok == false) && (strcmp(argv[1], "-svr_only") == 0)) {
            argc--;
            argv++;
            svr_only=true;
            ok = true;
        }

        //Use additional filtereing
        if ((ok == false) && (strcmp(argv[1], "-filter") == 0)) {
            
            argc--;
            argv++;
            bg_sigma=atof(argv[1]);
            flag_filter=true;
            ok = true;
            argc--;
            argv++;
        }

        //Use additional filtereing
        if ((ok == false) && (strcmp(argv[1], "-sr_iterations") == 0)) {
            
            argc--;
            argv++;
            last_rec_iterations=atoi(argv[1]);
            
            ok = true;
            argc--;
            argv++;
        }
        
        //Perform bias correction of the reconstructed image agains the GW image in the same motion correction iteration
        if ((ok == false) && (strcmp(argv[1], "-global_bias_correction") == 0)) {
            argc--;
            argv++;
            global_bias_correction=true;
            ok = true;
        }
        
        //Debug mode
        if ((ok == false) && (strcmp(argv[1], "-debug") == 0)) {
            argc--;
            argv++;
            debug=true;
            ok = true;
        }
        
        if (ok == false) {
            cerr << "Can not parse argument " << argv[1] << endl;
            usage();
        }
    }
    
    
    // -----------------------------------------------------------------------------
    
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
        string remove_folder_cmd = "rm -r " + str_current_exchange_file_path + " > tmp-log.txt ";
        int tmp_log_rm = system(remove_folder_cmd.c_str());
        
        string create_folder_cmd = "mkdir " + str_current_exchange_file_path + " > tmp-log.txt ";
        int tmp_log_mk = system(create_folder_cmd.c_str());
        
    } else {
        cout << "System error: could not create a folder for file exchange" << endl;
        exit(1);
    }
    
    //---------------------------------------------------------------------------------------------
    
    
    
    if (rescale_stacks) {
        for (i=0;i<nStacks;i++)
            reconstruction->Rescale(stacks[i],1000);
    }
    
    
    //If transformations were not defined by user, set them to identity
    for (i=0;i<nStacks;i++) {
        RigidTransformation *rigidTransf = new RigidTransformation;
        stack_transformations.push_back(*rigidTransf);
        delete rigidTransf;
    }
    
    
    //Initialise 2*slice thickness if not given by user
    if (thickness.size()==0) {
        cout<< "Slice thickness : ";
        
        for (i=0;i<nStacks;i++) {
            double dx,dy,dz;
            stacks[i].GetPixelSize(&dx,&dy,&dz);
            thickness.push_back(dz*2);
            cout<<thickness[i]<<" ";
        }
        cout<<endl;
    }
    
    //Output volume
    RealImage reconstructed;
    
    
    //Set debug mode
    if (debug) reconstruction->DebugOn();
    else reconstruction->DebugOff();
    
    //Set NMI bins for registration
    reconstruction->SetNMIBins(nmi_bins);

    //Set force excluded slices
    reconstruction->SetForceExcludedSlices(force_excluded);
    
    //Set low intensity cutoff for bias estimation
    reconstruction->SetLowIntensityCutoff(low_intensity_cutoff)  ;
    
    
    // Check whether the template stack can be indentified
    if (templateNumber<0) {
        cerr<<"Please identify the template by assigning id transformation."<<endl;
        exit(1);
    }
    
    //If no mask was given - try to create mask from the template image in case it was padded
    if (mask == NULL) {
        
        RealImage tmp_mask = stacks[templateNumber];
        tmp_mask = 1;
        
        mask = new RealImage(tmp_mask);
        *mask = reconstruction->CreateMask(*mask);
        cout << "Warning : no mask was provided" << endl;
    }
    
    
    //Before creating the template we will crop template stack according to the given mask
    if (mask != NULL)
    {
        //first resample the mask to the space of the stack
        //for template stact the transformation is identity
        RealImage m = *mask;
        reconstruction->TransformMask(template_stack,m,stack_transformations[templateNumber]);
        
        //Crop template stack and prepare template for global volumetric registration
        
        maskedTemplate = template_stack*m;
        reconstruction->CropImage(template_stack, m);
        reconstruction->CropImage(maskedTemplate,m);
        
        if (debug) {
            m.Write("maskforTemplate.nii.gz");
            template_stack.Write("croppedTemplate.nii.gz");
        }
    }
    
    cout << "---------------------------------------------------------------------" << endl;
    
    
    if (flag_denoise) {
        
        cout << "NLMFiltering" << endl;
        reconstruction->NLMFiltering(stacks);
    }
    
    if (flag_filter) {
        
        cout << "BackgroundFiltering" << endl;
        reconstruction->BackgroundFiltering(stacks, fg_sigma, bg_sigma);
    }
    
    GaussianBlurring<RealPixel> gbt(1.2*template_stack.GetXSize());
    gbt.Input(&template_stack);
    gbt.Output(&template_stack);
    gbt.Run();
    
    
    //Create template volume with isotropic resolution
    //if resolution==0 it will be determined from in-plane resolution of the image
    resolution = reconstruction->CreateTemplate(template_stack, resolution);
    
    //Set mask to reconstruction object.
    reconstruction->SetMask(mask,smooth_mask);
    
    //to redirect output from screen to text files
    
    //to remember cout and cerr buffer
    streambuf* strm_buffer = cout.rdbuf();
    streambuf* strm_buffer_e = cerr.rdbuf();
    //files for registration output
    string name;
    name = log_id+"log-registration.txt";
    ofstream file(name.c_str());
    name = log_id+"log-registration-error.txt";
    ofstream file_e(name.c_str());
    //files for reconstruction output
    name = log_id+"log-reconstruction->txt";
    ofstream file2(name.c_str());
    name = log_id+"log-evaluation.txt";
    ofstream fileEv(name.c_str());
    
    //set precision
    cout<<setprecision(3);
    cerr<<setprecision(3);
    
    cout << "---------------------------------------------------------------------" << endl;
    
    //perform volumetric registration of the stacks
    //redirect output to files
    
    if ( ! no_log ) {
        cerr.rdbuf(file_e.rdbuf());
        cout.rdbuf (file.rdbuf());
    }
    
    if (stacks.size() > 1) {
        //volumetric registration
        reconstruction->StackRegistrations(stacks,stack_transformations,templateNumber);
        cout<<endl;
    }

    //redirect output back to screen
    if ( ! no_log ) {
        cout.rdbuf (strm_buffer);
        cerr.rdbuf (strm_buffer_e);
    }

    average = reconstruction->CreateAverage(stacks,stack_transformations);
    if (debug)
        average.Write("average1.nii.gz");
    

//     //Rescale intensities of the stacks to have the same average
//     cout << "MatchStackIntensities" << endl;
//     if (intensity_matching)
//         reconstruction->MatchStackIntensitiesWithMasking(stacks,stack_transformations,averageValue);
//     else
//         reconstruction->MatchStackIntensitiesWithMasking(stacks,stack_transformations,averageValue,true);
    
    
    
    average = reconstruction->CreateAverage(stacks,stack_transformations);
    if (debug)
        average.Write("average2.nii.gz");
    
    
    for (i=0; i<stacks.size(); i++)
    {
        sprintf(buffer,"input%i.nii.gz",i);
        stacks[i].Write(buffer);
    }
    
    
    //Create slices and slice-dependent transformations
    Array<RealImage> probability_maps;
    cout << "CreateSlicesAndTransformations" << endl;
    reconstruction->CreateSlicesAndTransformations(stacks,stack_transformations,thickness,probability_maps);
    
    //Mask all the slices
    reconstruction->MaskSlices();
    
    //Set sigma for the bias field smoothing
    if (sigma>0)
        reconstruction->SetSigma(sigma);
    else
        reconstruction->SetSigma(20);
    
    //Set global bias correction flag
    if (global_bias_correction)
        reconstruction->GlobalBiasCorrectionOn();
    else
        reconstruction->GlobalBiasCorrectionOff();
    
    //if given read slice-to-volume registrations
    if (folder!=NULL)
        reconstruction->ReadTransformations(folder);
    
    //Initialise data structures for EM
    cout << "InitializeEM" << endl;
    reconstruction->InitializeEM();
    
    //If registration was switched off - only 1 iteration is required
    cout.rdbuf (strm_buffer);
    if (!registration_flag) {
        iterations = 1;
    }
    
    //interleaved registration-reconstruction iterations
    for (int iter=0;iter<iterations;iter++) {
        //Print iteration number on the screen
        if ( ! no_log ) {
            cout.rdbuf (strm_buffer);
        }
        
        cout << "---------------------------------------------------------------------" << endl;
        cout<<"Iteration : "<<iter<<endl;
        
        
        if (svr_only || template_flag) {
            cout<< "SliceToVolumeRegistration" << endl;
            if (remote_flag) {
                reconstruction->RemoteSliceToVolumeRegistration(iter, str_mirtk_path, str_current_main_file_path, str_current_exchange_file_path);
            } else {
                reconstruction->SliceToVolumeRegistration();
            }
            
        }
        else {
            //perform slice-to-volume registrations - skip the first iteration
            if (iter>0) {
                
                if (registration_flag) {
                    // cout<<"SVR iteration : "<<iter<<endl;
                    if((packages.size()>0)&&(iter<=iterations*(levels-1)/levels)&&(iter<(iterations-1)))
                    {
                        reconstruction->PackageToVolume(stacks,packages,stack_transformations);
                    }
                    else {
                        if (remote_flag) {
                            reconstruction->RemoteSliceToVolumeRegistration(iter, str_mirtk_path, str_current_main_file_path, str_current_exchange_file_path);
                        } else {
                            reconstruction->SliceToVolumeRegistration();
                        }
                    }
                        
                    
                }
                
            }
        }
        
        cout << "---------------------------------------------------------------------" << endl;
        
        
        //Set smoothing parameters
        //amount of smoothing (given by lambda) is decreased with improving alignment
        //delta (to determine edges) stays constant throughout
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
        
        //Use faster reconstruction during iterations and slower for final reconstruction
        if ( iter<(iterations-1) )
            reconstruction->SpeedupOn();
        else
            reconstruction->SpeedupOff();
        
        if(robust_slices_only)
            reconstruction->ExcludeWholeSlicesOnly();
        
        //Initialise values of weights, scales and bias fields
        cout << "InitializeEMValues" << endl;
        reconstruction->InitializeEMValues();
        
        
        //Calculate matrix of transformation between voxels of slices and volume
        cout << "CoeffInit" << endl;
        reconstruction->CoeffInit();
        
        //Initialize reconstructed image with Gaussian weighted reconstruction
        cout << "GaussianReconstruction" << endl;
        reconstruction->GaussianReconstruction();
        
        //Simulate slices (needs to be done after Gaussian reconstruction)
        cout << "SimulateSlices" << endl;
        reconstruction->SimulateSlices();
        
        //Initialize robust statistics parameters
        cout << "InitializeRobustStatistics" << endl;
        reconstruction->InitializeRobustStatistics();

        if (!gaussian_only) {
            
            //EStep
            if(robust_statistics) {
                cout << "EStep" << endl;
                reconstruction->EStep();
            }
            
            //number of reconstruction iterations
            if ( iter==(iterations-1) )
                rec_iterations = 10;
            else
                rec_iterations = last_rec_iterations;
            
            
            //reconstruction iterations
            i=0;
            for (i=0;i<rec_iterations;i++) {
                
                cout << "---------------------------------------------------------------------" << endl;
                cout<<endl<<"SR iteration : "<<i<<endl;
                
                if (intensity_matching) {
                    //calculate bias fields
                    
                    cout << "Bias" << endl;
                    if (sigma>0)
                        reconstruction->Bias();
                    //calculate scales
                    cout << "Scale" << endl;
                    reconstruction->Scale();
                }
                
                //Update reconstructed volume
                cout << "Superresolution" << endl;
                reconstruction->Superresolution(i+1);
                
                if (intensity_matching) {
                    cout << "NormaliseBias" << endl;
                    if((sigma>0)&&(!global_bias_correction))
                        reconstruction->NormaliseBias(i);
                }
                
                // Simulate slices (needs to be done
                // after the update of the reconstructed volume)
                cout << "SimulateSlices" << endl;
                reconstruction->SimulateSlices();
                
                if(robust_statistics) {
                    cout << "MStep" << endl;
                    reconstruction->MStep(i+1);
                }
                
                //E-step
                if(robust_statistics) {
                    cout << "EStep" << endl;
                    reconstruction->EStep();
                }
                
                //Save intermediate reconstructed image
                if (debug) {
                    reconstructed=reconstruction->GetReconstructed();
                    sprintf(buffer,"super%i.nii.gz",i);
                    reconstructed.Write(buffer);
                }
                
                
            }//end of reconstruction iterations
            
            //Mask reconstructed image to ROI given by the mask
            reconstruction->MaskVolume();

            // //Evaluate - write number of included/excluded/outside/zero slices in each iteration in the file
            reconstruction->Evaluate(iter, fileEv);
        }
        
        //Save reconstructed image
        //            if (debug)
        //            {
        reconstructed = reconstruction->GetReconstructed();
        sprintf(buffer, "image%i.nii.gz", iter);
        reconstructed.Write(buffer);
        //            }
        
        
        
    } // end of interleaved registration-reconstruction iterations
    
    
    if (str_current_exchange_file_path.length() > 0) {
        string remove_folder_cmd = "rm -r " + str_current_exchange_file_path + " > tmp-log.txt ";
        int tmp_log_rm = system(remove_folder_cmd.c_str());
    }
    
    //save final result
//     reconstruction->RestoreSliceIntensities();
//     reconstruction->ScaleVolume();
    reconstructed=reconstruction->GetReconstructed();
    reconstructed.Write(output_name);
    
    cout << "Reconstructed volume : " << output_name << endl;
    
    cout << "---------------------------------------------------------------------" << endl;
    
    if ( info_filename.length() > 0 )
        reconstruction->SlicesInfo( info_filename.c_str(),
                                  stack_files );
    
    if(debug)
    {
        reconstruction->SaveWeights();
        reconstruction->SaveBiasFields();
        reconstruction->SimulateStacks(stacks);
        for (i=0; i<stacks.size(); i++)
        {
            sprintf(buffer,"simulated%i.nii.gz",i);
            stacks[i].Write(buffer);
        }
    }
    
    //The end of main()
    
    return 0;
}



