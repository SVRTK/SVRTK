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


#include "mirtk/ReconstructionCardiac4D.h"
#include <string>


using namespace mirtk;
using namespace std;

//Application to perform reconstruction of volumetric cardiac cine MRI from thick-slice dynamic 2D MRI

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cerr << "Usage: reconstructCardiac [reconstructed] [N] [stack_1] .. [stack_N] <options>\n" << endl;
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
    // Location R-R Intervals;
    Array<double> rr_loc;
    // Slice R-R Intervals
    Array<double> rr;
    // Slice cardiac phases
    Array<double> cardPhase;
    // Mean Displacement
    Array<double> mean_displacement;
    Array<double> mean_weighted_displacement;
    // Mean Target Registration Error
    Array<double> mean_tre;
    
    // int step = 1;
    // int rewinder = 1;
    
    // Numbers of NMI bins for registration
    int nmi_bins = 16;
    
    // Default values.
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
    double lambda = 0.02;
    double delta = 150;
    int levels = 3;
    double lastIterLambda = 0.01;
    int rec_iterations;
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
    bool stack_intensity_matching = true;
    bool intensity_matching = true;
    bool rescale_stacks = false;
    bool stack_registration = false;
    
    //flag to swich the robust statistics on and off
    bool robust_statistics = true;
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
    ReconstructionCardiac4D reconstruction;
    
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
    
    
    for (i=0;i<nStacks;i++)
    {
        stack_files.push_back(argv[1]);
        
        //stack.Read(argv[1]);
        
        cout<<"Reading stack ... "<<argv[1]<<endl;
        
        tmp_fname = argv[1];
        image_reader.reset(ImageReader::TryNew(tmp_fname));
        tmp_image.reset(image_reader->Run());
        
        stack = *tmp_image;
        
        argc--;
        argv++;
        stacks.push_back(stack);
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
            cout<< "Reading stack masks.";
            for (i=0;i<nStacks;i++)
            {
                RealImage *tmp_mask_p;
                tmp_mask_p = new RealImage(argv[1]);
                
                RealImage tmp_mask = *tmp_mask_p;
                masks.push_back(tmp_mask);
                
                argc--;
                argv++;
            }

            reconstruction.SetMaskedStacks();
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
        
        //Remove black background
        // if ((ok == false) && (strcmp(argv[1], "-remove_black_background") == 0)){
        //   argc--;
        //   argv++;
        //   remove_black_background=true;
        //   ok = true;
        // }
        
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

    
    
    
    
    Array<RealImage> masked_stacks;
    
    for (i=0;i<nStacks;i++)
    {
        masked_stacks.push_back(stacks[i]);
    }
    
    
    if (masks.size()>0) {
        
        for (i=0;i<masks.size();i++) {
            
            cout << i << endl;
            
            RigidTransformation* tmp_rreg = new RigidTransformation;
            RealImage stack_mask = masks[i];
            reconstruction.TransformMask(stacks[i], stack_mask, *tmp_rreg);
            
//            ConnectivityType i_connectivity = CONNECTIVITY_26;
//            Dilate<RealPixel>(&stack_mask, 7, i_connectivity);
            
            RealImage stack = masked_stacks[i]*stack_mask;
            
            masked_stacks[i] = stack;

            
            sprintf(buffer, "masked-%i.nii.gz", i);
            masked_stacks[i].Write(buffer);
            
        }
        
//        reconstruction.CenterStacks(masks, stack_transformations, templateNumber);
        
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
        cerr<<"Reconstruction of volumetric cardiac cine MRI from thick-slice dynamic 2D MRI requires mask to initilise reconstructed volume."<<endl;
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
    
    //redirect output to files
    if ( ! no_log ) {
        cerr.rdbuf(file_e.rdbuf());
        cout.rdbuf (file.rdbuf());
    }
    
    //volumetric registration if input stacks are single time frame
    if (stack_registration)
    {
        ImageAttributes attr = stacks[templateNumber].GetImageAttributes();
        if (attr._t > 1)
        cout << "Skipping stack-stack registration; target stack has more than one time frame." << endl;
        else
        {
            if (debug)
            cout << "StackRegistrations" << endl;
//            reconstruction.StackRegistrations(stacks, stack_transformations, templateNumber);
            reconstruction.StackRegistrations(masked_stacks, stack_transformations, templateNumber);
            
            if (debug)
            {
                reconstruction.InvertStackTransformations(stack_transformations);
                for (i=0;i<nStacks;i++)
                {
                    sprintf(buffer, "stack-transformation%03i.dof", i);
                    stack_transformations[i].Write(buffer);
                }
                reconstruction.InvertStackTransformations(stack_transformations);
            }
        }
        
    }


        
    //if remove_black_background flag is set, create mask from black background of the stacks
    if (remove_black_background)
        reconstruction.CreateMaskFromBlackBackground(stacks, stack_transformations, smooth_mask);
    
    cout<<endl;
    //redirect output back to screen
    if ( ! no_log ) {
        cout.rdbuf (strm_buffer);
        cerr.rdbuf (strm_buffer_e);
    }
    
    average = reconstruction.CreateAverage(stacks,stack_transformations);
    if (debug)
    average.Write("average1.nii.gz");
    
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
    
    //Create slices and slice-dependent transformations
    reconstruction.CreateSlicesAndTransformationsCardiac4D(stacks,stack_transformations,thickness);
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
    }
    
    //if given, read reference transformations
    if ((have_ref_transformations)&(ref_transformations_folder!=NULL))
    reconstruction.ReadRefTransformation(ref_transformations_folder);
    else
    have_ref_transformations = false;
    if (!have_ref_transformations)
    reconstruction.InitTRE();
    
    //Mask all the slices
    reconstruction.MaskSlices();
    
    // Set R-R for each image
    if (rr_loc.empty())
    {
        reconstruction.SetSliceRRInterval(rrInterval);
        if (debug)
        cout<<"No R-R intervals specified. All R-R intervals set to "<<rrInterval<<" s."<<endl;
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
        if ( numCardPhase != 1 ) {    // reconstructing cine volume
            ImageAttributes attr = stacks[templateNumber].GetImageAttributes();
            if (attr._t > 1) {
                cerr<<"Cardiac 4D reconstruction requires cardiac phase for each slice."<<endl;
                exit(1);
            }
        }
        else {                        // reconstructing single cardiac phase volume
            reconstruction.SetSliceCardiacPhase();    // set all cardiac phases to zero
        }
    }
    else {
        reconstruction.SetSliceCardiacPhase( cardPhase );   // set all cardiac phases to given values
    }
    // Calculate Target Cardiac Phase in Reconstructed Volume for Slice-To-Volume Registration
    reconstruction.CalculateSliceToVolumeTargetCardiacPhase();
    // Calculate Temporal Weight for Each Slice
    reconstruction.CalculateSliceTemporalWeights();
    
    //interleaved registration-reconstruction iterations
    if(debug)
    cout<<"Number of iterations is :"<<iterations<<endl;
    
    for (int iter=0;iter<iterations;iter++)
    {
        //Print iteration number on the screen
        if ( ! no_log ) {
            cout.rdbuf (strm_buffer);
        }
        
        cout<<"Iteration"<<iter<<". "<<endl;
        
        //perform slice-to-volume registrations
        if ( iter > 0 ) 
        {
            
           if ( ! no_log ) {
               cerr.rdbuf(file_e.rdbuf());
               cout.rdbuf (file.rdbuf());
           }
            cout<<endl<<endl<<"Iteration "<<iter<<": "<<endl<<endl;
            reconstruction.SliceToVolumeRegistrationCardiac4D();
            cout<<endl;
            
           if ( ! no_log ) {
               cerr.rdbuf (strm_buffer_e);
           }
            
            // if ((iter>0) && (debug))
            //       reconstruction.SaveRegistrationStep(stacks,iter);
            
           if ( ! no_log ) {
               cerr.rdbuf (strm_buffer_e);
           }
            
            // process transformations
            if(motion_sigma>0)
            reconstruction.SmoothTransformations(motion_sigma);
            
        }  // if ( iter > 0 )
        
        
       //Write to file
       if ( ! no_log ) {
           cout.rdbuf (file2.rdbuf());
       }
        cout<<endl<<endl<<"Iteration "<<iter<<": "<<endl<<endl;
        
        //Set smoothing parameters
        //amount of smoothing (given by lambda) is decreased with improving alignment
        //delta (to determine edges) stays constant throughout
        if(iter==(iterations-1))
        reconstruction.SetSmoothingParameters(delta,lastIterLambda);
        else
        {
            double l=lambda;
            for (i=0;i<levels;i++)
            {
                if (iter==iterations*(levels-i-1)/levels)
                reconstruction.SetSmoothingParameters(delta, l);
                l*=2;
            }
        }
        
        //Use faster reconstruction during iterations and slower for final reconstruction
        if ( iter<(iterations-1) )
        reconstruction.SpeedupOn();
        else
        reconstruction.SpeedupOff();
        
        //Exclude whole slices only
        if(robust_slices_only)
        reconstruction.ExcludeWholeSlicesOnly();
        
        //Initialise values of weights, scales and bias fields
        reconstruction.InitializeEMValues();
        
        //Calculate matrix of transformation between voxels of slices and volume
        if (bspline)
        {
            cerr<<"Cannot currently initalise b-spline for cardiac 4D reconstruction."<<endl;
            exit(1);
        }
        else
        reconstruction.CoeffInitCardiac4D();
        
        //Initialize reconstructed image with Gaussian weighted reconstruction
        if (bspline)
        {
            cerr<<"Cannot currently reconstruct b-spline for cardiac 4D reconstruction."<<endl;
            exit(1);
        }
        else
        reconstruction.GaussianReconstructionCardiac4D();
        
        // Calculate Entropy
        e.clear();
        e.push_back(reconstruction.CalculateEntropy());
        
        // Save Initialised Volume to File
        if (debug)
        {
            reconstructed = reconstruction.GetReconstructedCardiac4D();
            sprintf(buffer,"init_mc%02i.nii.gz",iter);
            reconstructed.Write(buffer);
            volumeweights = reconstruction.GetVolumeWeights();
            sprintf(buffer,"volumeweights_mc%02i.nii.gz",iter);
            volumeweights.Write(buffer);
        }
        
        //Simulate slices (needs to be done after Gaussian reconstruction)
        reconstruction.SimulateSlicesCardiac4D();
        
        //Save intermediate simulated slices
        if(debug)
        {
            reconstruction.SaveSimulatedSlices(stacks,iter,0);
            reconstruction.SaveSimulatedWeights(stacks,iter,0);
            reconstruction.CalculateError();
            reconstruction.SaveError(stacks,iter,0);
        }
        
        //Initialize robust statistics parameters
        reconstruction.InitializeRobustStatistics();
        
        //EStep
        if(robust_statistics)
        reconstruction.EStep();
        
        if(debug)
        reconstruction.SaveWeights(stacks,iter,0);
        
        //number of reconstruction iterations
        if ( iter==(iterations-1) )
        {
            if (rec_iterations_last<0)
            rec_iterations_last = 2 * rec_iterations_first;
            rec_iterations = rec_iterations_last;
        }
        else {
            rec_iterations = rec_iterations_first;
        }
        if (debug)
        cout << "rec_iterations = " << rec_iterations << endl;
        
        if ((bspline)&&(!robust_statistics)&&(!intensity_matching))
        rec_iterations=0;
        
        //reconstruction iterations
        for (i=0;i<rec_iterations;i++)
        {
            cout<<endl<<"  Reconstruction iteration "<<i<<". "<<endl;
            
            if (intensity_matching)
            {
                //calculate bias fields
                if (sigma>0)
                reconstruction.Bias();
                //calculate scales
                reconstruction.Scale();
            }
            
            //Update reconstructed volume
            if (!bspline)
            reconstruction.SuperresolutionCardiac4D(i);
            
            if (intensity_matching)
            {
                if((sigma>0)&&(!global_bias_correction))
                reconstruction.NormaliseBiasCardiac4D(iter,i);
            }
            
            //Save intermediate reconstructed volume
            if (debug)
            {
                reconstructed=reconstruction.GetReconstructedCardiac4D();
                reconstructed=reconstruction.StaticMaskVolume4D(reconstructed,-1);
                sprintf(buffer,"super_mc%02isr%02i.nii.gz",iter,i);
                reconstructed.Write(buffer);
            }
            
            // Calculate Entropy
            e.push_back(reconstruction.CalculateEntropy());
            
            // Simulate slices (needs to be done
            // after the update of the reconstructed volume)
            reconstruction.SimulateSlicesCardiac4D();
            
            if ((i+1)<rec_iterations)
            {
                //Save intermediate simulated slices
                if(debug)
                {
                    if (intensity_matching) {
                        reconstruction.CalculateCorrectedSlices();
                        reconstruction.SaveCorrectedSlices(stacks,iter,i+1);
                        if (sigma>0)
                        reconstruction.SaveBiasFields(stacks,iter,i+1);
                    }
                    reconstruction.SaveSimulatedSlices(stacks,iter,i+1);
                    reconstruction.CalculateError();
                    reconstruction.SaveError(stacks,iter,i+1);
                }
                
                if(robust_statistics)
                reconstruction.MStep(i+1);
                
                //E-step
                if(robust_statistics)
                reconstruction.EStep();
                
                //Save intermediate weights
                if(debug)
                reconstruction.SaveWeights(stacks,iter,i+1);
            }
            
        }//end of reconstruction iterations
        
        //Mask reconstructed image to ROI given by the mask
        if(!bspline)
        reconstruction.StaticMaskReconstructedVolume4D();
        
        //Save reconstructed image
        reconstructed=reconstruction.GetReconstructedCardiac4D();
        reconstructed=reconstruction.StaticMaskVolume4D(reconstructed,-1);
        sprintf(buffer,"reconstructed_mc%02i.nii.gz",iter);
        reconstructed.Write(buffer);
        
        //Save Calculated Entropy
        entropy.push_back(e);
        
        //Evaluate - write number of included/excluded/outside/zero slices in each iteration in the file
        if ( ! no_log )
        cout.rdbuf (fileEv.rdbuf());
        reconstruction.Evaluate(iter);
        cout<<endl;
        if ( ! no_log )
        cout.rdbuf (strm_buffer);
        
        // Calculate Displacements
        if(have_ref_vol){
            
            // Change logging
            if ( ! no_log ) {
                cerr.rdbuf(file_e.rdbuf());
                cout.rdbuf (file.rdbuf());
            }
            
            // Get Current Reconstructed Volume
            reconstructed=reconstruction.GetReconstructedCardiac4D();
            reconstructed=reconstruction.StaticMaskVolume4D(reconstructed,-1);
            
            // Invert to get recon to ref transformation
            if(rreg_recon_to_ref) {
                reconstruction.VolumeToVolumeRegistration(ref_vol,reconstructed,transformation_recon_to_ref);
                Matrix m = transformation_recon_to_ref.GetMatrix();
                m.Invert();  // Invert to get recon to ref transformation
                transformation_recon_to_ref.PutMatrix(m);
            }
            else {
                reconstruction.VolumeToVolumeRegistration(reconstructed,ref_vol,transformation_recon_to_ref);
            }
            
            // Change logging
            if ( ! no_log ) {
                cout.rdbuf (strm_buffer);
                cerr.rdbuf (strm_buffer_e);
            }
            
            // Save Transformation
            sprintf(buffer, "recon_to_ref_mc%02i.dof",iter);
            transformation_recon_to_ref.Write(buffer);
            
            // Calculate Displacements Relative to Alignment
            mean_displacement.push_back(reconstruction.CalculateDisplacement(transformation_recon_to_ref));
            mean_weighted_displacement.push_back(reconstruction.CalculateWeightedDisplacement(transformation_recon_to_ref));
            
            // Calculate TRE Relative to Alignment
            if(have_ref_transformations)
            mean_tre.push_back(reconstruction.CalculateTRE(transformation_recon_to_ref));
            
        }
        else {
            
            // Calculate Displacement
            mean_displacement.push_back(reconstruction.CalculateDisplacement());
            mean_weighted_displacement.push_back(reconstruction.CalculateWeightedDisplacement());
            
            // Calculate TRE
            if(have_ref_transformations)
            mean_tre.push_back(reconstruction.CalculateTRE());
            
        }
        
        // Display Displacements and TRE
        if (debug) {
            cout<<"Mean Displacement (iter "<<iter<<") = "<<mean_displacement[iter]<<" mm."<<endl;
            cout<<"Mean Weighted Displacement (iter "<<iter<<") = "<<mean_weighted_displacement[iter]<<" mm."<<endl;
            if(have_ref_transformations)
            cout<<"Mean TRE (iter "<<iter<<") = "<<mean_tre[iter]<<" mm."<<endl;
        }
        
        // Save Info for Iteration
        if(debug)
        {
            cout<<"SlicesInfoCardiac4D"<<endl;
            sprintf(buffer,"info_mc%02i.tsv",iter);
            reconstruction.SlicesInfoCardiac4D( buffer, stack_files );
        }
        
    }// end of interleaved registration-reconstruction iterations
    
    //Display Entropy Values
    if(debug)
    {
        cout<<setprecision(9);
        cout << "Calculated Entropy:" << endl;
        for(unsigned int iter_mc=0; iter_mc<entropy.size(); iter_mc++)
        {
            cout << iter_mc << ": ";
            for(unsigned int iter_sr=0; iter_sr<entropy[iter_mc].size(); iter_sr++)
            {
                cout << entropy[iter_mc][iter_sr] << " ";
            }
            cout << endl;
        }
        cout<<setprecision(3);
    }
    
    //Display Mean Displacements and TRE
    if (debug) {
        cout<<"Mean Displacement:";
        for(unsigned int iter_mc=0; iter_mc<mean_displacement.size(); iter_mc++) {
            cout<<" "<<mean_displacement[iter_mc];
        }
        cout<<" mm."<<endl;
        cout<<"Mean Weighted Displacement:";
        for(unsigned int iter_mc=0; iter_mc<mean_weighted_displacement.size(); iter_mc++) {
            cout<<" "<<mean_weighted_displacement[iter_mc];
        }
        cout<<" mm."<<endl;
        if(have_ref_transformations) {
            cout<<"Mean TRE:";
            for(unsigned int iter_mc=0; iter_mc<mean_tre.size(); iter_mc++) {
                cout<<" "<<mean_tre[iter_mc];
            }
            cout<<" mm."<<endl;
        }
    }
    
    //save final result
    if(debug)
    cout<<"RestoreSliceIntensities"<<endl;
    reconstruction.RestoreSliceIntensities();
    if(debug)
    cout<<"ScaleVolumeCardiac4D"<<endl;
    reconstruction.ScaleVolumeCardiac4D();
    if(debug)
    cout<<"Saving Reconstructed Volume"<<endl;
    reconstructed=reconstruction.GetReconstructedCardiac4D();
    reconstructed.Write(output_name);
    if(debug)
    cout<<"SaveSlices"<<endl;
    reconstruction.SaveSlices(stacks);
    if(debug)
    cout<<"SaveTransformations"<<endl;
    reconstruction.SaveTransformations();
    
    //save final transformation to reference volume
    if(have_ref_vol) {
        sprintf(buffer, "recon_to_ref.dof");
        transformation_recon_to_ref.Write(buffer);
    }
    
    if ( info_filename.length() > 0 )
    {
        if(debug)
        cout<<"SlicesInfoCardiac4D"<<endl;
        reconstruction.SlicesInfoCardiac4D( info_filename.c_str(), stack_files );
    }
    
    if(debug)
    {
        cout<<"SaveWeights"<<endl;
        reconstruction.SaveWeights(stacks);
        cout<<"SaveBiasFields"<<endl;
        reconstruction.SaveBiasFields(stacks);
        cout<<"SaveSimulatedSlices"<<endl;
        reconstruction.SaveSimulatedSlices(stacks);
        cout<<"ReconstructionCardiac complete."<<endl;
    }
    //The end of main()
    
}


// -----------------------------------------------------------------------------



