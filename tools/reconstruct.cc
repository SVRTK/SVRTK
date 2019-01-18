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


#include "mirtk/Reconstruction.h"


using namespace mirtk; 
using namespace std;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: reconstruction [reconstructed] [N] [stack_1] .. [stack_N] <options>\n" << endl;
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
    cout << "\t-thickness [th_1] .. [th_N] Give slice thickness.[Default: twice voxel size in z direction]"<<endl;
    cout << "\t-mask [mask]              Binary mask to define the region od interest. [Default: whole image]"<<endl;
    cout << "\t-packages [num_1] .. [num_N] Give number of packages used during acquisition for each stack."<<endl;
    cout << "\t                          The stacks will be split into packages during registration iteration 1"<<endl;
    cout << "\t                          and then into odd and even slices within each package during "<<endl;
    cout << "\t                          registration iteration 2. The method will then continue with slice to"<<endl;
    cout << "\t                          volume approach. [Default: slice to volume registration only]"<<endl;
    cout << "\t-template_number          Number of the template stack. [Default: 0]"<<endl;
    cout << "\t-iterations [iter]        Number of registration-reconstruction iterations. [Default: 3]"<<endl;
    cout << "\t-resolution [res]         Isotropic resolution of the volume. [Default: 0.75mm]"<<endl;
    cout << "\t-global_bias_correction   Correct the bias in reconstructed image against previous estimation."<<endl;
    cout << "\t-no_intensity_matching    Switch off intensity matching."<<endl;
    cout << "\t-no_robust_statistics     Switch off robust statistics."<<endl;
    cout << "\t-no_robust_statistics     Switch off robust statistics."<<endl;
    cout << "\t-exclude_slices_only      Robust statistics for exclusion of slices only."<<endl;
    cerr << "\t-remove_black_background  Create mask from black background."<<endl;
    cerr << "\t-transformations [folder] Use existing slice-to-volume transformations to initialize the reconstruction."<<endl;
    cerr << "\t-force_exclude [number of slices] [ind1] ... [indN]  Force exclusion of slices with these indices."<<endl;
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
    
    
    // Default values.
    int templateNumber=0;
    RealImage *mask=NULL;
    int iterations = 3;
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
    //flag to remove black background, e.g. when neonatal motion correction is performed
    bool remove_black_background = false;
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
    
    
    //Create reconstruction object
    Reconstruction reconstruction;
    
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
        
        argc--;
        argv++;
        stacks.push_back(stack);
    }
    
    // Parse options.
    while (argc > 1) {
        ok = false;
        
        //Read stack transformations
        if ((ok == false) && (strcmp(argv[1], "-dofin") == 0)){
            argc--;
            argv++;
            
            for (i=0;i<nStacks;i++) {
                
                
                cout<<"Reading transformation : "<<argv[1]<<endl;

                UniquePtr<Transformation> t(Transformation::New(argv[1]));

                argc--;
                argv++;
                
                RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*> (t.get());

                stack_transformations.push_back(*rigidTransf);
                delete rigidTransf;
            }
            reconstruction.InvertStackTransformations(stack_transformations);
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
        
        //Read number of registration-reconstruction iterations
        if ((ok == false) && (strcmp(argv[1], "-iterations") == 0)) {
            argc--;
            argv++;
            iterations=atoi(argv[1]);
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
          
        //Isotropic resolution for the reconstructed volume
        if ((ok == false) && (strcmp(argv[1], "-resolution") == 0)) {
            argc--;
            argv++;
            resolution=atof(argv[1]);
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
        if ((ok == false) && (strcmp(argv[1], "-remove_black_background") == 0)){
            argc--;
            argv++;
            remove_black_background=true;
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
    
    if (rescale_stacks) {
        for (i=0;i<nStacks;i++)
        reconstruction.Rescale(stacks[i],1000);
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
    if (debug) reconstruction.DebugOn();
    else reconstruction.DebugOff();
    
    //Set force excluded slices
    reconstruction.SetForceExcludedSlices(force_excluded);
    
    //Set low intensity cutoff for bias estimation
    reconstruction.SetLowIntensityCutoff(low_intensity_cutoff)  ;
    
    
    // Check whether the template stack can be indentified
    if (templateNumber<0) {
        cerr<<"Please identify the template by assigning id transformation."<<endl;
        exit(1);
    }
    
    //If no mask was given - try to create mask from the template image in case it was padded
    if (mask == NULL) {
        mask = new RealImage(stacks[templateNumber]);
        *mask = reconstruction.CreateMask(*mask);
        cout << "Warning : no mask was provided (reconstruction might take too long)" << endl;
    }
    

    //Before creating the template we will crop template stack according to the given mask
    if (mask != NULL)
    {
        //first resample the mask to the space of the stack
        //for template stact the transformation is identity
        RealImage m = *mask;
        reconstruction.TransformMask(stacks[templateNumber],m,stack_transformations[templateNumber]);
        
        //Crop template stack and prepare template for global volumetric registration
        
        maskedTemplate = stacks[templateNumber]*m;
        reconstruction.CropImage(stacks[templateNumber],m);
        reconstruction.CropImage(maskedTemplate,m);
 
        if (debug) {
            m.Write("maskforTemplate.nii.gz");
            stacks[templateNumber].Write("croppedTemplate.nii.gz");
        }
    }
 
    //Create template volume with isotropic resolution
    //if resolution==0 it will be determined from in-plane resolution of the image
    resolution = reconstruction.CreateTemplate(maskedTemplate, resolution);
    
    //Set mask to reconstruction object.
    reconstruction.SetMask(mask,smooth_mask);
    
    //if remove_black_background flag is set, create mask from black background of the stacks
    if (remove_black_background) {
        reconstruction.CreateMaskFromBlackBackground(stacks, stack_transformations, smooth_mask);
        
    }
    
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
    
    //perform volumetric registration of the stacks
    //redirect output to files
    
    if ( ! no_log ) {
        cerr.rdbuf(file_e.rdbuf());
        cout.rdbuf (file.rdbuf());
    }
    
    //volumetric registration
    reconstruction.StackRegistrations(stacks,stack_transformations,templateNumber);
    
    cout<<endl;
    
    
    //redirect output back to screen
    
    if ( ! no_log ) {
        cout.rdbuf (strm_buffer);
        cerr.rdbuf (strm_buffer_e);
    }


    
    average = reconstruction.CreateAverage(stacks,stack_transformations);
    if (debug)
        average.Write("average1.nii.gz");
    
    
    //Mask is transformed to the all other stacks and they are cropped
    for (i=0; i<nStacks; i++)
    {
        //template stack has been cropped already
        if ((i==templateNumber))
            continue;
        //transform the mask
        RealImage m=reconstruction.GetMask();
        reconstruction.TransformMask(stacks[i],m,stack_transformations[i]);
        //Crop template stack
        reconstruction.CropImage(stacks[i],m);
        
        if (debug) {
            sprintf(buffer,"mask%i.nii.gz",i);
            m.Write(buffer);
            sprintf(buffer,"cropped%i.nii.gz",i);
            stacks[i].Write(buffer);
        }
    }

    // we remove stacks of size 1 voxel (no intersection with ROI)
    Array<RealImage> selected_stacks;
    Array<RigidTransformation> selected_stack_transformations;
    int new_nStacks = 0;
    int new_templateNumber = 0;
    
    for (i=0; i<nStacks; i++) {
        if (stacks[i].GetX() == 1) {
            cerr << "stack " << i << " has no intersection with ROI" << endl;
            continue;
        }
        
        // we keep it
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
    
    //Repeat volumetric registrations with cropped stacks
    //redirect output to files
    if ( ! no_log ) {
        cerr.rdbuf(file_e.rdbuf());
        cout.rdbuf (file.rdbuf());
    }
    //volumetric registration
    reconstruction.StackRegistrations(stacks,stack_transformations,templateNumber);
    cout<<endl;
    
    //redirect output back to screen
    if ( ! no_log ) {
        cout.rdbuf (strm_buffer);
        cerr.rdbuf (strm_buffer_e);
    }
    
    //Rescale intensities of the stacks to have the same average
    if (intensity_matching)
        reconstruction.MatchStackIntensitiesWithMasking(stacks,stack_transformations,averageValue);
    else
        reconstruction.MatchStackIntensitiesWithMasking(stacks,stack_transformations,averageValue,true);
    
    average = reconstruction.CreateAverage(stacks,stack_transformations);
    if (debug)
        average.Write("average2.nii.gz");

    //Create slices and slice-dependent transformations
    Array<RealImage> probability_maps;
    reconstruction.CreateSlicesAndTransformations(stacks,stack_transformations,thickness,probability_maps);
    
    //Mask all the slices
    reconstruction.MaskSlices();
    
    //Set sigma for the bias field smoothing
    if (sigma>0)
        reconstruction.SetSigma(sigma);
    else
        reconstruction.SetSigma(20);
    
    //Set global bias correction flag
    if (global_bias_correction)
        reconstruction.GlobalBiasCorrectionOn();
    else
        reconstruction.GlobalBiasCorrectionOff();
    
    //if given read slice-to-volume registrations
    if (folder!=NULL)
        reconstruction.ReadTransformation(folder);
    
    //Initialise data structures for EM
    reconstruction.InitializeEM();
    
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
        cout<<"Iteration : "<<iter<<endl;
        
        //perform slice-to-volume registrations - skip the first iteration
        if (iter>-1) {
            if ( ! no_log ) {
                cerr.rdbuf(file_e.rdbuf());
                cout.rdbuf (file.rdbuf());
            }
            if (registration_flag) {
                cout<<"SVR iteration : "<<iter<<endl;
                reconstruction.SliceToVolumeRegistration();
            }
            
            cout<<endl;
            if ( ! no_log ) {
                cerr.rdbuf (strm_buffer_e);
            }
        }
        
        //Write to file
        if ( ! no_log ) {
            cout.rdbuf (file2.rdbuf());
        }
        cout<<endl<<endl<<"Iteration : "<<iter<<endl<<endl;
        
        //Set smoothing parameters
        //amount of smoothing (given by lambda) is decreased with improving alignment
        //delta (to determine edges) stays constant throughout
        if(iter==(iterations-1))
            reconstruction.SetSmoothingParameters(delta,lastIterLambda);
        else
        {
            double l=lambda;
            for (i=0;i<levels;i++) {
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
        
        if(robust_slices_only)
            reconstruction.ExcludeWholeSlicesOnly();
        
        //Initialise values of weights, scales and bias fields
        reconstruction.InitializeEMValues();
        
        //Calculate matrix of transformation between voxels of slices and volume
        reconstruction.CoeffInit();
        
        //Initialize reconstructed image with Gaussian weighted reconstruction
        reconstruction.GaussianReconstruction();
        
        //Simulate slices (needs to be done after Gaussian reconstruction)
        reconstruction.SimulateSlices();
        
        //Initialize robust statistics parameters
        reconstruction.InitializeRobustStatistics();
        
        //EStep
        if(robust_statistics)
            reconstruction.EStep();
        
        //number of reconstruction iterations
        if ( iter==(iterations-1) )
            rec_iterations = 30;
        else
            rec_iterations = 10;
    
        
        //reconstruction iterations
        i=0;
        for (i=0;i<rec_iterations;i++) {
            cout<<endl<<"SR iteration : "<<i<<endl;
            
            if (intensity_matching) {
                //calculate bias fields
                if (sigma>0)
                    reconstruction.Bias();
                //calculate scales
                reconstruction.Scale();
            }
            
            //Update reconstructed volume
            reconstruction.Superresolution(i+1);
            
            if (intensity_matching) {
                if((sigma>0)&&(!global_bias_correction))
                    reconstruction.NormaliseBias(i);
            }
            
            // Simulate slices (needs to be done
            // after the update of the reconstructed volume)
            reconstruction.SimulateSlices();
            
            if(robust_statistics)
                reconstruction.MStep(i+1);
            
            //E-step
            if(robust_statistics)
                reconstruction.EStep();
            
            //Save intermediate reconstructed image
            if (debug) {
                reconstructed=reconstruction.GetReconstructed();
                sprintf(buffer,"super%i.nii.gz",i);
                reconstructed.Write(buffer);
            }
            
            
        }//end of reconstruction iterations
        
        //Mask reconstructed image to ROI given by the mask
        reconstruction.MaskVolume();
        
        //Save reconstructed image
        if (debug)
        {
            reconstructed = reconstruction.GetReconstructed();
            sprintf(buffer, "image%i.nii.gz", iter);
            reconstructed.Write(buffer);
        }
        
        //Evaluate - write number of included/excluded/outside/zero slices in each iteration in the file
        if ( ! no_log ) {
            cout.rdbuf (fileEv.rdbuf());
        }
        
        reconstruction.Evaluate(iter);
        
        cout<<endl;
        
        if ( ! no_log ) {
            cout.rdbuf (strm_buffer);
        }
        
    } // end of interleaved registration-reconstruction iterations
    
    //save final result
    reconstruction.RestoreSliceIntensities();
    reconstruction.ScaleVolume();
    reconstructed=reconstruction.GetReconstructed();
    reconstructed.Write(output_name);
    
    cout << output_name << endl;
    
    if (debug) {
        reconstruction.SaveTransformations();
        reconstruction.SaveSlices();
    }
    
    if ( info_filename.length() > 0 )
    reconstruction.SlicesInfo( info_filename.c_str(),
                              stack_files );
    
    if(debug)
    {
        reconstruction.SaveWeights();
        reconstruction.SaveBiasFields();
        reconstruction.SimulateStacks(stacks);
        for (unsigned int i=0;i<stacks.size();i++)
        {
            sprintf(buffer,"simulated%i.nii.gz",i);
            stacks[i].Write(buffer);
        }
    }
    
    //The end of main()
    
    return 0;
}
