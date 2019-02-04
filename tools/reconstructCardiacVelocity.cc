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


#include "mirtk/ReconstructionCardiacVelocity4D.h"
#include <string>


using namespace mirtk;
using namespace std;

//Application to perform reconstruction of volumetric cardiac cine phase / velocity MRI from thick-slice dynamic 2D MRI

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cerr << "Usage: reconstructionCardiacVelocity [reconstructed] [N] [stack_1] .. [stack_N] [g_values] [g_directions] <options>\n" << endl;
    cerr << endl;
    
    cerr << "\t[reconstructed]            Name for the reconstructed volume. Nifti or Analyze format." << endl;
    cerr << "\t[N]                        Number of stacks." << endl;
    cerr << "\t[stack_1]..[stack_N]       The input stacks. Nifti or Analyze format." << endl;
    cerr << "\t" << endl;
    cerr << "Options:" << endl;
    cerr << "\t-stack_registration        Perform stack-stack regisrtation." << endl;
    cerr << "\t-target_stack [stack_no]   Stack number of target for stack-stack registration." << endl;
    cerr << "\t-dofin [dof_1]..[dof_N]    The transformations of the input stack to template" << endl;
    cerr << "\t                           in \'dof\' format used in IRTK/MIRTK." <<endl;
    // cerr << "\t                          Only rough alignment with correct orienation and " << endl;
    // cerr << "\t                          some overlap is needed." << endl;
    cerr << "\t                           Use \'id\' for an identity transformation." << endl;
    cerr << "\t-thickness [th_1]..[th_N]  Give slice thickness.[Default: twice voxel size in z direction]"<<endl;
    cerr << "\t-mask [mask]               Binary mask to define the region of interest. [Default: whole image]"<<endl;
    cerr << "\t-masks [stack_1]..[stack_N]  Binary masks for the ROI in each of the stacks. [Default: none]"<<endl;
    cerr << "\t-transformations [folder]  Use existing image-frame to volume transformations to initialize the reconstruction."<<endl;
    cerr << "\t-slice_transformations [folder]  Use existing slice-location transformations to initialize the reconstruction."<<endl;
    cerr << "\t-motion_sigma [sigma]      Stdev for smoothing transformations. [Default: 0s, no smoothing]"<<endl;
    cerr << "\t-rrintervals [L] [rr_1]..[rr_L]  R-R interval for slice-locations 1-L in input stacks. [Default: 1 s]."<<endl;
    cerr << "\t-cardphase [K] [num_1]..[num_K]  Cardiac phase (0-2PI) for each image-frames 1-K. [Default: 0]."<<endl;
    cerr << "\t-temporalpsfgauss          Use Gaussian temporal point spread function. [Default: temporal PSF = sinc()*Tukey_window()]" << endl;
    cerr << "\t-resolution [res]          Isotropic resolution of the volume. [Default: 0.75mm]"<<endl;
    cerr << "\t-numcardphase              Number of cardiac phases to reconstruct. [Default: 15]."<<endl;
    cerr << "\t-rrinterval [rr]           R-R interval of reconstructed cine volume. [Default: 1 s]."<<endl;
    cerr << "\t-iterations [n]            Number of registration-reconstruction iterations. [Default: 4]"<<endl;
    cerr << "\t-rec_iterations [n]        Number of super-resolution reconstruction iterations. [Default: 10]"<<endl;
    cerr << "\t-rec_iterations_last [n]   Number of super-resolution reconstruction iterations for last iteration. [Default: 2 x rec_iterations]"<<endl;
    cerr << "\t-sigma [sigma]             Stdev for bias field. [Default: 12mm]"<<endl;
    cerr << "\t-nmi_bins [nmi_bins]       Number of NMI bins for registration. [Default: 16]"<<endl;
    cerr << "\t-average [average]         Average intensity value for stacks [Default: 700]"<<endl;
    cerr << "\t-delta [delta]             Parameter to define what is an edge. [Default: 150]"<<endl;
    cerr << "\t-lambda [lambda]           Smoothing parameter. [Default: 0.02]"<<endl;
    cerr << "\t-lastIter [lambda]         Smoothing parameter for last iteration. [Default: 0.01]"<<endl;
    cerr << "\t-multires [levels]         Multiresolution smooting with given number of levels. [Default: 3]"<<endl;
    cerr << "\t-smooth_mask [sigma]       Smooth the mask to reduce artefacts of manual segmentation. [Default: 4mm]"<<endl;
    cerr << "\t-force_exclude [n] [ind1]..[indN]  Force exclusion of image-frames with these indices."<<endl;
    cerr << "\t-force_exclude_sliceloc [n] [ind1]..[indN]  Force exclusion of slice-locations with these indices."<<endl;
    cerr << "\t-force_exclude_stack [n] [ind1]..[indN]  Force exclusion of stacks with these indices."<<endl;
    cerr << "\t-no_stack_intensity_matching  Switch off stack intensity matching."<<endl;
    cerr << "\t-no_intensity_matching     Switch off intensity matching."<<endl;
    cerr << "\t-no_robust_statistics      Switch off robust statistics."<<endl;
    cerr << "\t-no_regularisation         Switch off adaptive regularisation."<<endl;
    cerr << "\t-exclude_slices_only       Do not exclude individual voxels."<<endl;
    cerr << "\t-ref_vol                   Reference volume for adjustment of spatial position of reconstructed volume."<<endl;
    cerr << "\t-rreg_recon_to_ref         Register reconstructed volume to reference volume [Default: recon to ref]"<<endl;
    cerr << "\t-ref_transformations [folder]  Reference slice-to-volume transformation folder."<<endl;
    cerr << "\t-log_prefix [prefix]       Prefix for the log file."<<endl;
    cerr << "\t-info [filename]           Filename for slice information in tab-sparated columns."<<endl;
    cerr << "\t-debug                     Debug mode - save intermediate results."<<endl;
    cerr << "\t-no_log                    Do not redirect cout and cerr to log files."<<endl;
    // cerr << "\t-global_bias_correction   Correct the bias in reconstructed image against previous estimation."<<endl;
    // cerr << "\t-low_intensity_cutoff     Lower intensity threshold for inclusion of voxels in global bias correction."<<endl;
    // cerr << "\t-remove_black_background  Create mask from black background."<<endl;
    //cerr << "\t-multiband [num_1] .. [num_N]  Multiband factor for each stack for each stack. [Default: 1]"<<endl;
    //cerr << "\t-packages [num_1] .. [num_N]   Give number of packages used during acquisition for each stack. [Default: 1]"<<endl;
    //cerr << "\t                          The stacks will be split into packages during registration iteration 1"<<endl;
    // cerr << "\t                          and then following the specific slice ordering "<<endl;
    // cerr << "\t                          from iteration 2. The method will then perform slice to"<<endl;
    // cerr << "\t                          volume (or multiband registration)."<<endl;
    // cerr << "\t-order                    Array of slice acquisition orders used at acquisition. [Default: (1)]"<<endl;
    // cerr << "\t                          Possible values: 1 (ascending), 2 (descending), 3 (default), 4 (interleaved)"<<endl;
    // cerr << "\t                          and 5 (Customized)."<<endl;
    // cerr << "\t-step                     Forward slice jump for customized (C) slice ordering [Default: 1]"<<endl;
    // cerr << "\t-rewinder              Rewinder for customized slice ordering [Default: 1]"<<endl;
    // cerr << "\t-bspline                  Use multi-level bspline interpolation instead of super-resolution."<<endl;
    // cerr << "\t" << endl;
    // cerr << "\tNOTE: work in progress, use of following options is not recommended..." << endl;
    // cerr << "\t\tmultiband, packages, order, step, rewinder, rescale_stacks" << endl;
    cerr << "\t" << endl;
    cerr << "\t" << endl;
    exit(1);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------


int main(int argc, char **argv)
{
    //utility variables
    int i, j, ok;
    char buffer[256];
    RealImage stack;
    const double PI = 3.14159265358979323846;
    
    //declare variables for input
    /// Name for output volume
    char * output_name = NULL;
    /// Reference volume
    char * ref_vol_name = NULL;
    bool have_ref_vol = false;
    RigidTransformation transformation_recon_to_ref;
    bool rreg_recon_to_ref = false;
    /// Slice stacks
    Array<RealImage> stacks;
    Array<string> stack_files;
    /// Stack transformation
    Array<RigidTransformation> stack_transformations;
    /// Stack heart masks
    Array<RealImage> masks;
    
    /// user defined transformations
    bool have_stack_transformations = false;
    /// Stack thickness
    Array<double > thickness;
    ///number of stacks
    int nStacks;
    /// number of packages for each stack
    Array<int> packages;
    Array<int> order_Array;
    // Locations of R-R intervals;
    Array<double> rr_loc;
    // Slice R-R intervals
    Array<double> rr;
    // Slice cardiac phases
    Array<double> cardPhase;
    // Mean displacement
    Array<double> mean_displacement;
    Array<double> mean_weighted_displacement;
    // Mean target registration error
    Array<double> mean_tre;
    
    // int step = 1;
    // int rewinder = 1;
    
    
    // Numbers of NMI bins for registration
    int nmi_bins = 16;
    double velocity_scale = 1;
    
    
    // Default values
    int templateNumber = 0;
    RealImage *mask=NULL;
    int iterations = 4;
    bool debug = false;
    double sigma=20;
    double motion_sigma = 0;
    double resolution = 0.75;
    int numCardPhase = 15;
    double rrDefault = 1;
    double rrInterval = rrDefault;
    bool is_temporalpsf_gauss = false;
    double lambda = 0.015; // 0.02;
    double delta = 80; // 150;
    int levels = 3;
    double lastIterLambda = 0.01;
    int rec_iterations = 10;
    int rec_iterations_first = 10;
    int rec_iterations_last = -1;
    double averageValue = 700;
    double smooth_mask = 4;
    bool global_bias_correction = false;
    // double low_intensity_cutoff = 0.01;
    //folder for slice-location registrations, if given
    char *slice_transformations_folder=NULL;
    //folder for slice-to-volume registrations, if given
    char *folder=NULL;
    //folder for reference slice-to-volume registrations, if given
    char *ref_transformations_folder=NULL;
    bool have_ref_transformations = false;
    //flag to remove black background, e.g. when neonatal motion correction is performed
    bool remove_black_background = false;
    //flag to swich the intensity matching on and off
    bool stack_intensity_matching = false; //true;
    bool intensity_matching = false; //true;
    bool rescale_stacks = false;
    bool stack_registration = false;

    bool adaptive_regularisation = true;
    
    //flag to swich the robust statistics on and off
    bool robust_statistics = false; //true;
    bool robust_slices_only = false;
    //flag to replace super-resolution reconstruction by multilevel B-spline interpolation
    bool bspline = false;
    Array<int> multiband_Array;
    // int multiband_factor=1;
    
    RealImage average;
    
    string info_filename = "info.tsv";
    string log_id;
    bool no_log = false;
    
    //forced exclusion of slices
    int number_of_force_excluded_slices = 0;
    Array<int> force_excluded;
    int number_of_force_excluded_stacks = 0;
    Array<int> force_excluded_stacks;
    int number_of_force_excluded_locs = 0;
    Array<int> force_excluded_locs;
    
    //Create reconstruction object
    ReconstructionCardiacVelocity4D reconstruction;
    
    //Entropy of reconstructed volume
    Array<double> e;
    Array< Array<double> > entropy;
    
    //if not enough arguments print help
    if (argc < 5)
        usage();
    
    //read output name
    output_name = argv[1];
    argc--;
    argv++;
    cout<<"Recontructed volume name ... "<<output_name<<endl;
    
    //read number of stacks
    nStacks = atoi(argv[1]);
    argc--;
    argv++;
    cout<<"Number of stacks ... "<<nStacks<<endl;
    
    // Read stacks
    
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    
    for (i=0; i<nStacks; i++)
    {
        stack_files.push_back(argv[1]);
        
        //stack.Read(argv[1]);
        
        cout<<"Reading stack ... "<<argv[1]<<endl;
        
        tmp_fname = argv[1];
        image_reader.reset(ImageReader::TryNew(tmp_fname));
        tmp_image.reset(image_reader->Run());
        
        stack = *tmp_image;
        
        //        stack = stack;
        
        argc--;
        argv++;
        stacks.push_back(stack);
    }
    
    
    Array<Array<double> > g_directions;
    Array<double> g_values;
    
    Array<double> tmp_array;
    for (i=0; i<3; i++)
        tmp_array.push_back(0);
    
    for (i=0; i<nStacks; i++) {
        g_values.push_back(0);
        g_directions.push_back(tmp_array);
        
    }
    
    
    //Read the name of the text file with gradient values
    char *g_val_file = argv[1];
    argc--;
    argv++;
    
    //Read the gradient values from the text file
    ifstream in_g_val(g_val_file);
    double num;
    i = 0;
    cout<<" - Reading Gradient values from " << g_val_file << " : "<<endl;
    if (in_g_val.is_open()) {
        while (!in_g_val.eof()) {
            in_g_val >> num;
            if (i<nStacks)
                g_values[i]=num;
            i++;
        }
        in_g_val.close();
        
        for (i=0; i<g_values.size(); i++)
            cout << g_values[i] << " ";
        cout << endl;
    }
    else {
        cout << " - Unable to open file " << g_val_file << endl;
        exit(1);
    }
    
    
    //Read the gradient directions from the text file
    char *g_dir_file = argv[1];
    argc--;
    argv++;
    
    ifstream in_g_dir(g_dir_file);
    int coord = 0;
    int dir = 0;
    cout<<" - Reading Gradient directions from " << g_dir_file << " : "<<endl;
    if (in_g_dir.is_open()) {
        while (!in_g_dir.eof()) {
            in_g_dir >> num;
            if ((coord<nStacks)&&(dir<3))
                g_directions[coord][dir]=num;
            dir++;
            if (dir>=3) {
                dir=0;
                coord++;
            }
        }
        in_g_dir.close();
        
        for (i=0; i<g_directions.size(); i++) {
            Array<double> g_direction = g_directions[i];
            for (int k=0; k<g_direction.size(); k++)
                cout << g_direction[k] << " ";
            cout << endl;
        }
        
    }
    else {
        cout << " - Unable to open file " << g_dir_file << endl;
        exit(1);
    }
    
    
    // Parse options.
    while (argc > 1){
        ok = false;
        
        // Target stack
        if ((ok == false) && (strcmp(argv[1], "-target_stack") == 0)){
            argc--;
            argv++;
            templateNumber=atof(argv[1])-1;
            cout<<"Target stack no. is "<<atof(argv[1])<<" (zero-indexed stack no. "<<templateNumber<<")."<<endl;
            argc--;
            argv++;
            ok = true;
        }
        
        //Read stack transformations
        if ((ok == false) && (strcmp(argv[1], "-dofin") == 0)){
            argc--;
            argv++;
            
            for (i=0;i<nStacks;i++)
            {
                Transformation *transformation;
                cout<<"Reading transformation ... "<<argv[1]<<" ... ";
                cout.flush();
                if (strcmp(argv[1], "id") == 0)
                {
                    transformation = new RigidTransformation;
                }
                else
                {
                    transformation = Transformation::New(argv[1]);
                }
                cout<<" done."<<endl;
                
                argc--;
                argv++;
                RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*> (transformation);
                stack_transformations.push_back(*rigidTransf);
                delete rigidTransf;
            }
            reconstruction.InvertStackTransformations(stack_transformations);
            have_stack_transformations = true;
        }
        
        //Stack registration
        if ((ok == false) && (strcmp(argv[1], "-stack_registration") == 0)){
            argc--;
            argv++;
            stack_registration=true;
            ok = true;
            cout << "Stack-stack registrations, if possible."<<endl;
        }
        
        //Read slice thickness
        if ((ok == false) && (strcmp(argv[1], "-thickness") == 0)){
            argc--;
            argv++;
            cout<< "Slice thickness is ";
            for (i=0;i<nStacks;i++)
            {
                thickness.push_back(atof(argv[1]));
                cout<<thickness[i]<<" ";
                argc--;
                argv++;
            }
            cout<<"."<<endl;
            ok = true;
        }
        
        //Read stack location R-R Intervals
        if ((ok == false) && (strcmp(argv[1], "-rrintervals") == 0)){
            argc--;
            argv++;
            int nLocs = atoi(argv[1]);
            cout<<"Reading R-R intervals for "<<nLocs<<" slice locations"<<endl;
            argc--;
            argv++;
            cout<< "R-R intervals are ";
            for (i=0;i<nLocs;i++)
            {
                rr_loc.push_back(atof(argv[1]));
                cout<<i<<":"<<rr_loc[i]<<", ";
                argc--;
                argv++;
            }
            cout<<"\b\b."<<endl;
            ok = true;
        }
        
        //Read cardiac phases
        if ((ok == false) && (strcmp(argv[1], "-cardphase") == 0)){
            argc--;
            argv++;
            int nSlices = atoi(argv[1]);
            cout<<"Reading cardiac phase for "<<nSlices<<" images."<<endl;
            argc--;
            argv++;
            for (i=0;i<nSlices;i++)
            {
                cardPhase.push_back(atof(argv[1]));
                argc--;
                argv++;
            }
            ok = true;
        }
        
        // Number of cardiac phases in reconstructed volume
        if ((ok == false) && (strcmp(argv[1], "-numcardphase") == 0)){
            argc--;
            argv++;
            numCardPhase=atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        
        // R-R Interval of Reconstructed Volume
        if ((ok == false) && (strcmp(argv[1], "-rrinterval") == 0)){
            argc--;
            argv++;
            rrInterval=atof(argv[1]);
            argc--;
            argv++;
            cout<<"R-R interval of reconstructed volume is "<<rrInterval<<" s."<<endl;
            ok = true;
            reconstruction.SetReconstructedRRInterval(rrInterval);
        }
        
        // Use Gaussian Temporal Point Spread Function
        if ((ok == false) && (strcmp(argv[1], "-temporalpsfgauss") == 0)){
            argc--;
            argv++;
            is_temporalpsf_gauss=true;
            ok = true;
        }
        
        
        //Read binary mask for final volume
        if ((ok == false) && (strcmp(argv[1], "-mask") == 0)){
            argc--;
            argv++;
            mask = new RealImage(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Read binary masks for all stacks
        if ((ok == false) && (strcmp(argv[1], "-masks") == 0)){
            
            argc--;
            argv++;
            cout<< "Reading heart masks.";
            for (i=0;i<nStacks;i++)
            {
                RealImage *tmp_mask_p;
                tmp_mask_p = new RealImage(argv[1]);
                
                RealImage tmp_mask = *tmp_mask_p;
                
                // dilate the stack heart masks (tmp)
                // ConnectivityType connectivity = CONNECTIVITY_26;
                // Dilate<RealPixel>(&tmp_mask, 9, connectivity);
                
                masks.push_back(tmp_mask);
                
                argc--;
                argv++;
            }
            
            ok = true;
            
        }
        
        
        //Read number of registration-reconstruction iterations
        if ((ok == false) && (strcmp(argv[1], "-iterations") == 0)){
            argc--;
            argv++;
            iterations=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Read number of reconstruction iterations
        if ((ok == false) && (strcmp(argv[1], "-rec_iterations") == 0)){
            argc--;
            argv++;
            rec_iterations_first=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Read number of reconstruction iterations for last registration-reconstruction iteration
        if ((ok == false) && (strcmp(argv[1], "-rec_iterations_last") == 0)){
            argc--;
            argv++;
            rec_iterations_last=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Bins for NMI registration.
        if ((ok == false) && (strcmp(argv[1], "-nmi_bins") == 0)){
            argc--;
            argv++;
            nmi_bins=atoi(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Variance of Gaussian kernel to smooth the bias field.
        if ((ok == false) && (strcmp(argv[1], "-sigma") == 0)){
            argc--;
            argv++;
            sigma=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Variance of Gaussian kernel to smooth the motion
        if ((ok == false) && (strcmp(argv[1], "-motion_sigma") == 0)){
            argc--;
            argv++;
            motion_sigma=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Smoothing parameter
        if ((ok == false) && (strcmp(argv[1], "-lambda") == 0)){
            argc--;
            argv++;
            lambda=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Smoothing parameter for last iteration
        if ((ok == false) && (strcmp(argv[1], "-lastIter") == 0)){
            argc--;
            argv++;
            lastIterLambda=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Parameter to define what is an edge
        if ((ok == false) && (strcmp(argv[1], "-delta") == 0)){
            argc--;
            argv++;
            delta=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Isotropic resolution for the reconstructed volume
        if ((ok == false) && (strcmp(argv[1], "-resolution") == 0)){
            argc--;
            argv++;
            resolution=atof(argv[1]);
            ok = true;
            argc--;
            argv++;
        }
        
        //Number of resolution levels
        if ((ok == false) && (strcmp(argv[1], "-multires") == 0)){
            argc--;
            argv++;
            levels=atoi(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        
        //Smooth mask to remove effects of manual segmentation
        if ((ok == false) && (strcmp(argv[1], "-smooth_mask") == 0)){
            argc--;
            argv++;
            smooth_mask=atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        
        //Scale for output velocity volumes
        if ((ok == false) && (strcmp(argv[1], "-velocity_scale") == 0)){
            argc--;
            argv++;
            velocity_scale=atof(argv[1]);
            argc--;
            argv++;
            ok = true;
        }
        
        //No adaptive regularisation
        if ((ok == false) && (strcmp(argv[1], "-no_regularisation") == 0)){
            argc--;
            argv++;
            adaptive_regularisation=false;

            ok = true;
        }
        
        //Match stack intensities
        if ((ok == false) && (strcmp(argv[1], "-no_stack_intensity_matching") == 0)){
            argc--;
            argv++;
            stack_intensity_matching=false;
            ok = true;
            cout << "No stack intensity matching."<<endl;
        }
        
        //Switch off intensity matching
        if ((ok == false) && (strcmp(argv[1], "-no_intensity_matching") == 0)){
            argc--;
            argv++;
            intensity_matching=false;
            ok = true;
            cout << "No intensity matching."<<endl;
        }
        
        //Switch off robust statistics
        if ((ok == false) && (strcmp(argv[1], "-no_robust_statistics") == 0)){
            argc--;
            argv++;
            robust_statistics=false;
            ok = true;
        }
        
        //Switch off robust statistics
        if ((ok == false) && (strcmp(argv[1], "-exclude_slices_only") == 0)){
            argc--;
            argv++;
            robust_slices_only=true;
            ok = true;
        }
        
        //Use multilevel B-spline interpolation instead of super-resolution
        // if ((ok == false) && (strcmp(argv[1], "-bspline") == 0)){
        //   argc--;
        //   argv++;
        //   bspline=true;
        //   ok = true;
        // }
        
        //Perform bias correction of the reconstructed image agains the GW image in the same motion correction iteration
        // if ((ok == false) && (strcmp(argv[1], "-global_bias_correction") == 0)){
        //   argc--;
        //   argv++;
        //   global_bias_correction=true;
        //   ok = true;
        // }
        
        // if ((ok == false) && (strcmp(argv[1], "-low_intensity_cutoff") == 0)){
        //   argc--;
        //   argv++;
        //   low_intensity_cutoff=atof(argv[1]);
        //   argc--;
        //   argv++;
        //   ok = true;
        // }
        
        //Debug mode
        if ((ok == false) && (strcmp(argv[1], "-debug") == 0)){
            argc--;
            argv++;
            debug=true;
            ok = true;
        }
        
        //Prefix for log files
        if ((ok == false) && (strcmp(argv[1], "-log_prefix") == 0)){
            argc--;
            argv++;
            log_id=argv[1];
            ok = true;
            argc--;
            argv++;
        }
        
        //No log files
        if ((ok == false) && (strcmp(argv[1], "-no_log") == 0)){
            argc--;
            argv++;
            no_log=true;
            ok = true;
        }
        
        // rescale stacks to avoid error:
        // irtkImageRigidRegistrationWithPadding::Initialize: Dynamic range of source is too large
        if ((ok == false) && (strcmp(argv[1], "-rescale_stacks") == 0)){
            argc--;
            argv++;
            rescale_stacks=true;
            ok = true;
        }
        
        
        
        // Save slice info
        if ((ok == false) && (strcmp(argv[1], "-info") == 0)) {
            argc--;
            argv++;
            info_filename=argv[1];
            ok = true;
            argc--;
            argv++;
        }
        
        //Read slice-location transformations from this folder
        if ((ok == false) && (strcmp(argv[1], "-slice_transformations") == 0)){
            argc--;
            argv++;
            slice_transformations_folder=argv[1];
            ok = true;
            argc--;
            argv++;
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
        
        //Force removal of certain stacks
        if ((ok == false) && (strcmp(argv[1], "-force_exclude_stack") == 0)){
            argc--;
            argv++;
            number_of_force_excluded_stacks = atoi(argv[1]);
            argc--;
            argv++;
            
            cout<< number_of_force_excluded_stacks<< " force excluded stacks: ";
            for (i=0;i<number_of_force_excluded_stacks;i++)
            {
                force_excluded_stacks.push_back(atoi(argv[1]));
                cout<<force_excluded_stacks[i]<<" ";
                argc--;
                argv++;
            }
            cout<<"."<<endl;
            ok = true;
        }
        
        //Force removal of certain slice-locations
        if ((ok == false) && (strcmp(argv[1], "-force_exclude_sliceloc") == 0)){
            argc--;
            argv++;
            number_of_force_excluded_locs = atoi(argv[1]);
            argc--;
            argv++;
            
            cout<< number_of_force_excluded_locs << " force excluded slice-locations: ";
            for (i=0;i<number_of_force_excluded_locs;i++)
            {
                force_excluded_locs.push_back(atoi(argv[1]));
                cout<<force_excluded_locs[i]<<" ";
                argc--;
                argv++;
            }
            cout<<"."<<endl;
            ok = true;
        }
        
        //Get name of reference volume for adjustment of spatial position of reconstructed volume
        if ((ok == false) && (strcmp(argv[1], "-ref_vol") == 0)){
            argc--;
            argv++;
            ref_vol_name = argv[1];
            argc--;
            argv++;
            have_ref_vol = true;
            ok = true;
        }
        
        //Register reconstructed volume to reference volume
        if ((ok == false) && (strcmp(argv[1], "-rreg_recon_to_ref") == 0)){
            argc--;
            argv++;
            rreg_recon_to_ref = true;
            ok = true;
        }
        
        //Read transformations from this folder
        if ((ok == false) && (strcmp(argv[1], "-ref_transformations") == 0)){
            argc--;
            argv++;
            ref_transformations_folder=argv[1];
            argc--;
            argv++;
            have_ref_transformations = true;
            ok = true;
        }
        
        if (ok == false){
            cerr << "Can not parse argument " << argv[1] << endl;
            usage();
        }
    }
    
    
    // if (masks.size()>0) {
    //     for (i=0;i<nStacks;i++)    {
    //         RealImage m=masks[i];
    //         RigidTransformation *rigidTransf3 = new RigidTransformation;
    //         reconstruction.TransformMask(stacks[i],m,*rigidTransf3);
    //         reconstruction.CropImageIgnoreZ(stacks[i],m);
    //     }
    // }
    
    
    // check that conflicting transformation folders haven't been given
    if ((folder!=NULL)&(slice_transformations_folder!=NULL))
    {
        cerr << "Can not use both -transformations and -slice_transformations arguments." << endl;
        exit(1);
    }
    
    if (rescale_stacks)
    {
        for (i=0;i<nStacks;i++)
            reconstruction.Rescale(stacks[i],1000);
    }
    
    // set packages to 1 if not given by user
    if (packages.size() == 0)
        for (i=0;i<nStacks;i++)    {
            packages.push_back(1);
            cout<<"All packages set to 1"<<endl;
        }
    
    // set multiband to 1 if not given by user
    if (multiband_Array.size() == 0)
        for (i=0;i<nStacks;i++)    {
            multiband_Array.push_back(1);
            cout<<"Multiband set to 1 for all stacks"<<endl;
        }
    
    // set ascending if not given by user
    if (order_Array.size() == 0)
        for (i=0;i<nStacks;i++)    {
            order_Array.push_back(1);
            cout<<"Slice order set to ascending for all stacks"<<endl;
        }
    
    //If transformations were not defined by user, set them to identity
    if(!have_stack_transformations)
    {
        for (i=0;i<nStacks;i++)
        {
            RigidTransformation *rigidTransf = new RigidTransformation;
            stack_transformations.push_back(*rigidTransf);
            delete rigidTransf;
        }
    }
    
    //Initialise 2*slice thickness if not given by user
    if (thickness.size()==0)
    {
        cout<< "Slice thickness is ";
        for (i=0;i<nStacks;i++)
        {
            double dx,dy,dz;
            stacks[i].GetPixelSize(&dx,&dy,&dz);
            thickness.push_back(dz*2);
            cout<<thickness[i]<<" ";
        }
        cout<<"."<<endl;
    }
    
    //Heart mask-based centering (not required for velocity)
    if (masks.size()>0) {
        
        for (i=0;i<masks.size();i++) {
            reconstruction.CropImage(masks[i],masks[i]);
        }
        
        reconstruction.CenterStacks(masks, stack_transformations, templateNumber);
        
        for (i=0;i<masks.size();i++) {
            sprintf(buffer, "centered-mask-transformations-%i.dof", i);
            stack_transformations[i].Write(buffer);
        }
        
    }
    
    
    //Set temporal point spread function
    if (is_temporalpsf_gauss)
        reconstruction.SetTemporalWeightGaussian();
    else
        reconstruction.SetTemporalWeightSinc();
    
    //Output volume
    RealImage reconstructed;
    RealImage volumeweights;
    Array<double> reconstructedCardPhase;
    cout<<setprecision(3);
    cout<<"Reconstructing "<<numCardPhase<<" cardiac phases: ";
    for (i=0;i<numCardPhase;i++)
    {
        reconstructedCardPhase.push_back(2*PI*i/numCardPhase);
        cout<<" "<<reconstructedCardPhase[i]/PI<<",";
    }
    cout<<"\b x PI."<<endl;
    reconstruction.SetReconstructedCardiacPhase( reconstructedCardPhase );
    reconstruction.SetReconstructedTemporalResolution( rrInterval/numCardPhase );
    
    //Reference volume for adjustment of spatial position of reconstructed volume
    RealImage ref_vol;
    if(have_ref_vol){
        cout<<"Reading reference volume: "<<ref_vol_name<<endl;
        ref_vol.Read(ref_vol_name);
    }
    
    //Set debug mode
    if (debug) reconstruction.DebugOn();
    else reconstruction.DebugOff();
    
    //Set NMI bins for registration
    reconstruction.SetNMIBins(nmi_bins);
    
    //Set adaptive regularisation option flag
    reconstruction.SetAdaptiveRegularisation(adaptive_regularisation);
    
    //Set scale for ouput velocity volumes
    reconstruction.SetVelocityScale(velocity_scale);
    
    //Set force excluded slices
    reconstruction.SetForceExcludedSlices(force_excluded);
    
    //Set force excluded stacks
    reconstruction.SetForceExcludedStacks(force_excluded_stacks);
    
    //Set force excluded stacks
    reconstruction.SetForceExcludedLocs(force_excluded_locs);
    
    //Set low intensity cutoff for bias estimation
    //reconstruction.SetLowIntensityCutoff(low_intensity_cutoff)  ;
    
    // Check whether the template stack can be indentified
    if (templateNumber<0)
    {
        cerr<<"Please identify the template by assigning id transformation."<<endl;
        exit(1);
    }
    
    // Initialise Reconstructed Volume
    // Check that mask is provided
    if (mask==NULL)
    {
        cerr<<"5D cardiac velocity MRI reconstruction requires mask to initilise reconstructed volume."<<endl;
        exit(1);
    }
    
    // Crop mask
    RealImage maskCropped = *mask;
    reconstruction.CropImage(maskCropped,*mask);  // TODO: TBD: use CropImage or CropImageIgnoreZ
    // Initilaise reconstructed volume with isotropic resolution
    // if resolution==0 it will be determined from in-plane resolution of the image
    if (resolution <= 0)
    {
        resolution = reconstruction.GetReconstructedResolutionFromTemplateStack( stacks[templateNumber] );
    }
    if (debug)
        cout << "Initialising volume with isotropic voxel size " << resolution << "mm" << endl;
    
    // Create template 4D volume
    reconstruction.CreateTemplateCardiac4DFromStaticMask( maskCropped, resolution );
    
    // Set mask to reconstruction object
    reconstruction.SetMask(mask,smooth_mask);
    
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
    name = log_id+"log-reconstruction.txt";
    ofstream file2(name.c_str());
    name = log_id+"log-evaluation.txt";
    ofstream fileEv(name.c_str());
    
    //set precision
    cout<<setprecision(3);
    cerr<<setprecision(3);
    
    
    average = reconstruction.CreateAverage(stacks,stack_transformations);
    //    if (debug)
    //    average.Write("average1.nii.gz");
    
    //Mask is transformed to the all stacks and they are cropped
    for (i=0; i<nStacks; i++)
    {
        //transform the mask
        RealImage m=reconstruction.GetMask();
        reconstruction.TransformMask(stacks[i],m,stack_transformations[i]);
        //Crop template stack
        
        // ConnectivityType connectivity2 = CONNECTIVITY_26;
        // Dilate<RealPixel>(&m, 5, connectivity2);
        
        
        reconstruction.CropImageIgnoreZ(stacks[i],m);
        if (debug)
        {
            sprintf(buffer,"mask%03i.nii.gz",i);
            m.Write(buffer);
            sprintf(buffer,"cropped%03i.nii.gz",i);
            stacks[i].Write(buffer);
        }
    }
    
    /*
     //Rescale intensities of the stacks to have the same average
     if (stack_intensity_matching)
     reconstruction.MatchStackIntensitiesWithMasking(stacks,stack_transformations,averageValue);
     else
     reconstruction.MatchStackIntensitiesWithMasking(stacks,stack_transformations,averageValue,true);
     if (debug) {
     for (i=0; i<nStacks; i++) {
     sprintf(buffer,"rescaledstack%03i.nii.gz",i);
     stacks[i].Write(buffer);
     }
     }
     average = reconstruction.CreateAverage(stacks,stack_transformations);
     if (debug)
     average.Write("average2.nii.gz");
     */
    
    
    //Create slices and slice-dependent transformations
    reconstruction.CreateSlicesAndTransformationsCardiac4D(stacks,stack_transformations,thickness);
    // reconstruction.CreateSliceDirections(directions,bvalues);
    
    if(debug){
        reconstruction.InitCorrectedSlices();
        reconstruction.InitError();
    }
    
    //if given, read transformations
    if (folder!=NULL)
        reconstruction.ReadTransformation(folder);  // image-frame to volume registrations
    else {
        if (slice_transformations_folder!=NULL)     // slice-location to volume registrations
            reconstruction.ReadSliceTransformation(slice_transformations_folder);
        else {
            cerr<<"5D cardiac velocity MRI reconstruction requires slice transformations from structural reconstruction."<<endl;
            exit(1);
        }
    }
    
    //if given, read reference transformations
    if ((have_ref_transformations)&(ref_transformations_folder!=NULL))
        reconstruction.ReadRefTransformation(ref_transformations_folder);
    else
        have_ref_transformations = false;
    if (!have_ref_transformations)
        reconstruction.InitTRE();
    
    //Mask all the slices
    reconstruction.MaskSlicesPhase();
    
    // Set R-R for each image
    if (rr_loc.empty())
    {
        cerr<<"5D cardiac velocity MRI reconstruction requires specification of R-R intervals."<<endl;
        exit(1);
        // reconstruction.SetSliceRRInterval(rrInterval);
        // if (debug)
        // cout<<"No R-R intervals specified. All R-R intervals set to "<<rrInterval<<" s."<<endl;
    }
    else
        reconstruction.SetLocRRInterval(rr_loc);
    
    //Set sigma for the bias field smoothing
    if (sigma>0)
        reconstruction.SetSigma(sigma);
    else
    {
        //cerr<<"Please set sigma larger than zero. Current value: "<<sigma<<endl;
        //exit(1);
        reconstruction.SetSigma(20);
    }
    
    //Set global bias correction flag
    if (global_bias_correction)
        reconstruction.GlobalBiasCorrectionOn();
    else
        reconstruction.GlobalBiasCorrectionOff();
    
    //Initialise data structures for EM
    reconstruction.InitializeEM();
    
    // Calculate Cardiac Phase of Each Slice
    if ( cardPhase.size() == 0 ) {  // no cardiac phases specified
        
        cerr<<"5D cardiac velocity MRI reconstruction requires specification of cardiac phases."<<endl;
        exit(1);
        
    }
    else {
        reconstruction.SetSliceCardiacPhase( cardPhase );   // set all cardiac phases to given values
    }
    // Calculate Target Cardiac Phase in Reconstructed Volume for Slice-To-Volume Registration
    reconstruction.CalculateSliceToVolumeTargetCardiacPhase();
    // Calculate Temporal Weight for Each Slice
    reconstruction.CalculateSliceTemporalWeights();
    
    
    // ...........................................................................
    
    // Velocity reconstruction (draft version)
    
    reconstruction.InitializeVelocityVolumes();
    
    reconstruction.InitializeGradientMoments(g_directions, g_values);
    
    
    
    reconstruction.SetSmoothingParameters(delta,lastIterLambda);
    reconstruction.SpeedupOff();
    
    
    if(robust_slices_only)
        reconstruction.ExcludeWholeSlicesOnly();
    
    reconstruction.InitializeEMValues();
    
    reconstruction.CoeffInitCardiac4D();
    
    reconstruction.Set3DRecon();
    
    // Initialise velocity and phase limits
    reconstruction.ItinialiseVelocityBounds();
    
    
    
    //-------------------------------------------------------------------------------------
    // Main velocity reconstruciton steps
 
    
    // STEP 1: Gaussian reconstruction of phase volume
    reconstruction.GaussianReconstructionCardiac4DxT();


    // STEP 2: Gaussian recostruction of velocity volumes
    reconstruction.GaussianReconstructionCardiacVelocity4DxT();

 
    // STEP 3: Simulate slices (should be done after Gaussian reconstruction)
    reconstruction.SimulateSlicesCardiacVelocity4D();
    
    
    //Initialize robust statistics parameters
    reconstruction.InitializeRobustStatistics();
    
    //EStep
    if(robust_statistics)
        reconstruction.EStep();
    
     
    rec_iterations = rec_iterations_first;      // number of iterations is controlled by '-rec_iterations' flag

    // main reconstruction loop
    for(int iteration = 0; iteration < rec_iterations; iteration++ ) {
        
        cout<<endl<<" - Reconstruction iteration : "<< iteration <<" ... "<<endl;
        
        
        reconstruction.SimulateSignal(iteration);
        
        
        if (intensity_matching)
        {
            //calculate bias fields
            if (sigma>0)
                reconstruction.Bias();
            //calculate scales
            reconstruction.Scale();
        }
        
        
        // STEPS 4-5: super-resolution reconstruction and simulation of slices (optimisation loop)
        reconstruction.SuperresolutionCardiacVelocity4D(iteration);
        reconstruction.SimulateSlicesCardiacVelocity4D();
        
        
        double consistency = reconstruction.Consistency();
        
        
        if ((i+1)<rec_iterations) {
            
            if(robust_statistics)
                reconstruction.MStep(i+1);
            
            //E-step
            if(robust_statistics)
                reconstruction.EStep();
            
        }
        
        reconstruction.SaveReconstructedVelocity4D(iteration);
        reconstruction.SaveReconstructedVolume4D(iteration);
        
    } // end of reconstruction loop
    
    //-------------------------------------------------------------------------------------
    
    
    reconstruction.StaticMaskReconstructedVolume5D();
    reconstruction.SaveReconstructedVelocity4D(100);
    
    
    reconstruction.StaticMaskReconstructedVolume4D();
    // reconstruction.RestoreSliceIntensities();
    // reconstruction.ScaleVolumeCardiac4D();
    reconstructed=reconstruction.GetReconstructedCardiac4D();
    reconstructed.Write(output_name);
    
    reconstruction.SaveSlices(stacks);
    // reconstruction.SaveTransformations();
    
    reconstruction.SaveSimulatedSlices(stacks);
    
    // reconstruction.SaveSimulatedSlices();
    
    
    //The end of main()
    
}


// -----------------------------------------------------------------------------




