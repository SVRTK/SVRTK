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
//
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    
    cout << "Internal function for memory-efficient remote reconstruction." << endl;
    cout << "Usage: reconstruct-round [reconstructed] [path to mirtk] [path to the main folder] [path to the tmp file exchange folder] [current iteration] [number of slices] [thickness] [flags] <options>\n" << endl;
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
    cout << "\t-ncc                      Use global NCC similarity for SVR steps. [Default: NMI]"<<endl;
    cout << "\t-nmi_bins [nmi_bins]      Number of NMI bins for registration. [Default: 16]"<<endl;
    cout << "\t-structural               Use structrural exclusion of slices at the last iteration."<<endl;
    cout << "\t-exclude_slices_only      Robust statistics for exclusion of slices only."<<endl;
    cout << "\t-template [template]      Use template for initialisation of registration loop. [Default: average of stack registration]"<<endl;
    cerr << "\t-remove_black_background  Create mask from black background."<<endl;
    cerr << "\t-transformations [folder] Use existing slice-to-volume transformations to initialize the reconstruction->"<<endl;
    cerr << "\t-force_exclude [number of slices] [ind1] ... [indN]  Force exclusion of slices with these indices."<<endl;
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
    
//    const char *current_mirtk_path = argv[0];
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    
    auto start_total = std::chrono::system_clock::now();
    auto end_total = std::chrono::system_clock::now();
    
    std::chrono::duration<double> elapsed_seconds;
    std::time_t end_time;
    
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
    
    
    bool use_template = false;
    RealImage template_stack;
    
    bool flag_no_overlap_thickness = false;
    
    
    // Default values.
    int templateNumber=0;
    RealImage *mask=NULL;
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
    //folder for slice-to-volume registrations, if given
    char * folder=NULL;

    //flag to switch the intensity matching on and off
    bool intensity_matching = true;
    bool rescale_stacks = false;
    bool registration_flag = true;
    
    //flags to switch the robust statistics on and off
    bool robust_statistics = true;
    bool robust_slices_only = false;
    
    //flag to replace super-resolution reconstruction by multilevel B-spline interpolation
    bool bspline = false;
    
    RealImage average;
    
    string info_filename = "slice_info.tsv";
    string log_id;
    bool no_log = false;
    
    //forced exclusion of slices
    int number_of_force_excluded_slices = 0;
    vector<int> force_excluded;
    
    //flag for SVR registration to a template (without packages)
    bool svr_only = false;
    
    //flag for struture-based exclusion of slices
    bool structural = false;
    
    //flag for switching off NMI and using NCC for SVR step
    bool ncc_reg_flag = false;
    
    bool remote_flag = false;
    
    
    //Create reconstruction object
    Reconstruction *reconstruction = new Reconstruction();
    

    // Paths to the files stored by the original reconstruct command
    
    
    int current_iteration = 0;
    int current_number_of_slices = 0;
    double average_thickness = 0;
    
    
    char * param_in = NULL;
    
    
    param_in = argv[1];
    argc--;
    argv++;
    string str_mirtk_path(param_in);
   
    
    param_in = argv[1];
    argc--;
    argv++;
    string str_current_main_file_path(param_in);
    
    
    param_in = argv[1];
    argc--;
    argv++;
    string str_current_exchange_file_path(param_in);

    
    current_iteration = atoi(argv[1]);
    argc--;
    argv++;
    
    
    current_number_of_slices = atoi(argv[1]);
    argc--;
    argv++;
    
    
    average_thickness = atof(argv[1]);
    argc--;
    argv++;

    
    int load_iteration = current_iteration-1;
    if (load_iteration < 0)
        load_iteration = 0;
    
    
     
//    reconstruction->LoadResultsRemote(str_current_exchange_file_path, current_number_of_slices);
    

    
    // Parse options.
    while (argc > 1) {
        ok = false;
        
                    
        
        //Read number of registration-reconstruction iterations
        if ((ok == false) && (strcmp(argv[1], "-iterations") == 0)) {
            argc--;
            argv++;
            iterations=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        
        if ((ok == false) && (strcmp(argv[1], "-sr_iterations") == 0)) {
            argc--;
            argv++;
            sr_iterations=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        
        
        //Read template number
        if ((ok == false) && (strcmp(argv[1], "-template_number") == 0)) {
            argc--;
            argv++;
            templateNumber=atoi(argv[1]);
            
            template_stack = stacks[templateNumber];
            
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
        
        //SVR reconstruction as remote functions
        if ((ok == false) && (strcmp(argv[1], "-remote") == 0)) {
            argc--;
            argv++;
            remote_flag=true;
            ok = true;
        }
        
        if ((ok == false) && (strcmp(argv[1], "-exact-thickness") == 0)) {
            argc--;
            argv++;
            flag_no_overlap_thickness=true;
            ok = true;
        }

        
        //Use only SVR to a template
        if ((ok == false) && (strcmp(argv[1], "-svr_only") == 0)) {
            argc--;
            argv++;
            svr_only=true;
            ok = true;
        }
        
        
        //Use NCC similarity metric for SVR
        if ((ok == false) && (strcmp(argv[1], "-ncc") == 0)) {
            argc--;
            argv++;
            ncc_reg_flag=true;
            reconstruction->SetNCC(ncc_reg_flag);
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
        

        //Switch off intensity matching
        if ((ok == false) && (strcmp(argv[1], "-no_intensity_matching") == 0)) {
            argc--;
            argv++;
            intensity_matching=false;
            ok = true;
        }
        
        //Switch off robust statistics
        if ((ok == false) && (strcmp(argv[1], "-no_robust_statistics") == 0)) {
            argc--;
            argv++;
            robust_statistics=false;
            ok = true;
        }

        //Use structural exclusion of slices
        if ((ok == false) && (strcmp(argv[1], "-structural") == 0)) {
            argc--;
            argv++;
            structural=true;
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
    
    
    
    
    
     int iter = current_iteration;
    
    
    cout << "------------------------------------------------------" << endl;
    cout<<"Iteration (remote) : " << iter << endl;
    
    
    
     reconstruction->LoadModelRemote(str_current_exchange_file_path, current_number_of_slices, average_thickness, load_iteration);

  

    
    //Output volume
    RealImage reconstructed;
    
    
    //Set debug mode
    if (debug) reconstruction->DebugOn();
    else reconstruction->DebugOff();
    
    //Set force excluded slices
//    reconstruction->SetForceExcludedSlices(force_excluded);
    
    //Set low intensity cutoff for bias estimation
    reconstruction->SetLowIntensityCutoff(low_intensity_cutoff)  ;
    
    
    
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
    name = log_id+"log-reconstruction.txt";
    ofstream file2(name.c_str());
    name = log_id+"log-evaluation.txt";
    ofstream fileEv(name.c_str());
    
    //set precision
    cout<<setprecision(3);
    cerr<<setprecision(3);
    

    

//    //Create template volume with isotropic resolution
//    //if resolution==0 it will be determined from in-plane resolution of the image
//    resolution = reconstruction->CreateTemplate(maskedTemplate, resolution);
//
//    //Set mask to reconstruction object.
//    reconstruction->SetMask(mask,smooth_mask);


    
    
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
    
     
    //Initialise data structures for EM
    reconstruction->InitializeEM();
    
    
    
    //interleaved registration-reconstruction iterations
        
        
        reconstruction->MaskVolume();
        
        if (svr_only) {
            if (debug)
                start = std::chrono::system_clock::now();
            
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
                 {
                    if (debug)
                        start = std::chrono::system_clock::now();

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
        
        
        if (structural && iter>1) {
            
             cout.rdbuf (strm_buffer);
            
            if (debug)
                start = std::chrono::system_clock::now();
            
            reconstruction->StructuralExclusion();
            
            if (debug) {
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                cout << "StructuralExclusion ";
                cout << "- " << elapsed_seconds.count() << "s " << endl;
            }
            
            cout.rdbuf (file.rdbuf());
            
        }
    
        
        //Set smoothing parameters
        //amount of smoothing (given by lambda) is decreased with improving alignment
        //delta (to determine edges) stays constant throughout
        cout.rdbuf (file.rdbuf());
        if(iter==(iterations-1)) {
            reconstruction->SetSmoothingParameters(delta,lastIterLambda);
            
        } else {
            
            double l=lambda;
            for (i=0;i<levels;i++) {
                if (iter==iterations*(levels-i-1)/levels)
                    reconstruction->SetSmoothingParameters(delta, l);
                l*=2;
            }
        }

    
    
        cout.rdbuf (strm_buffer);
        
        //Use faster reconstruction during iterations and slower for final reconstruction
        if ( iter<(iterations-1) )
            reconstruction->SpeedupOn();
        else
            reconstruction->SpeedupOff();
        
        if(robust_slices_only)
            reconstruction->ExcludeWholeSlicesOnly();
        
        if (debug)
            start = std::chrono::system_clock::now();
        cout.rdbuf (file.rdbuf());
        reconstruction->InitializeEMValues();
        cout.rdbuf (strm_buffer);
        if (debug) {
            end = std::chrono::system_clock::now();
            elapsed_seconds = end-start;
            cout << "InitializeEMValues ";
            cout << "- " << elapsed_seconds.count() << "s " << endl;
        }
        
        
        if (debug)
            start = std::chrono::system_clock::now();
        //Calculate matrix of transformation between voxels of slices and volume
        
        // cout.rdbuf (file.rdbuf());
        reconstruction->CoeffInit();
        // cout.rdbuf (strm_buffer);
        if (debug) {
            end = std::chrono::system_clock::now();
            elapsed_seconds = end-start;
            cout << "CoeffInit ";
            cout << "- " << elapsed_seconds.count() << "s " << endl;
        }
        
        if (debug)
            start = std::chrono::system_clock::now();
        //Initialize reconstructed image with Gaussian weighted reconstruction
        
        cout.rdbuf (file.rdbuf());
        reconstruction->GaussianReconstruction();
        cout.rdbuf (strm_buffer);
        if (debug) {
            end = std::chrono::system_clock::now();
            elapsed_seconds = end-start;
            cout << "GaussianReconstruction ";
            cout << "- " << elapsed_seconds.count() << "s " << endl;
        }
        
        if (debug)
            start = std::chrono::system_clock::now();
        //Simulate slices (needs to be done after Gaussian reconstruction)
        
        cout.rdbuf (file.rdbuf());
        reconstruction->SimulateSlices();
        cout.rdbuf (strm_buffer);
        if (debug) {
            end = std::chrono::system_clock::now();
            elapsed_seconds = end-start;
            cout << "SimulateSlices ";
            cout << "- " << elapsed_seconds.count() << "s " << endl;
        }
        
        //Initialize robust statistics parameters
        cout.rdbuf (file.rdbuf());
        reconstruction->InitializeRobustStatistics();
        cout.rdbuf (strm_buffer);
        
        //EStep
        if(robust_statistics) {
            
            cout.rdbuf (file.rdbuf());
            reconstruction->EStep();
            cout.rdbuf (strm_buffer);
        }
               
    
        //number of reconstruction iterations
        if ( iter==(iterations-1) )
            rec_iterations = sr_iterations*3;
        else
            rec_iterations = sr_iterations;
        
        
        
        //reconstruction iterations
        for (int i=0;i<rec_iterations;i++) {
            

            if (intensity_matching) {
                
                //calculate bias fields
                if (debug)
                    start = std::chrono::system_clock::now();
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
                if (debug)
                    start = std::chrono::system_clock::now();
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
            
            //Update reconstructed volume
            if (debug)
                start = std::chrono::system_clock::now();
            
            cout.rdbuf (file.rdbuf());
            reconstruction->Superresolution(i+1);
            cout.rdbuf (strm_buffer);
            if (debug) {
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                cout << "Superresolution ";
                cout << "- " << elapsed_seconds.count() << "s " << endl;
            }
            
            if (intensity_matching) {
                if (debug)
                    start = std::chrono::system_clock::now();
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
            
            // Simulate slices (needs to be done
            // after the update of the reconstructed volume)
            if (debug)
                start = std::chrono::system_clock::now();
            
            cout.rdbuf (file.rdbuf());
            reconstruction->SimulateSlices();
            cout.rdbuf (strm_buffer);
            if (debug) {
                end = std::chrono::system_clock::now();
                elapsed_seconds = end-start;
                cout << "SimulateSlices ";
                cout << "- " << elapsed_seconds.count() << "s " << endl;
            }
            
            if(robust_statistics) {
                if (debug)
                    start = std::chrono::system_clock::now();
                
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
            
            
        }//end of reconstruction iterations
        
        //Mask reconstructed image to ROI given by the mask
        reconstruction->MaskVolume();
        
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
        

        //Evaluate - write number of included/excluded/outside/zero slices in each iteration in the file
        if (debug)
            reconstruction->Evaluate(iter, fileEv);
        
//        reconstruction->SaveSliceInfo(iter);
    
    
    reconstructed = reconstruction->GetReconstructed();
    sprintf(buffer, "image%i.nii.gz", iter);
    reconstructed.Write(buffer);
    
    
//    cout << "------------------------------------------------------" << endl;
        
//    reconstructed = reconstruction->GetReconstructed();
//    sprintf(buffer, "latest-out-recon.nii.gz", iter);
//    reconstructed.Write(buffer);
//
    
    
    if (iter == (iterations-1))
        reconstruction->ScaleVolume();
    
    
    reconstruction->SaveModelRemote(str_current_exchange_file_path, -1, current_iteration);
    
    
    return 0;
}

