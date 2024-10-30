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

// SVRTK Reconstruct qMRI T2 map code
#include "svrtk/ReconstructionqMRI.h"
#include "svrtk/Dictionary.h"
#define SVRTK_TOOL
#include "svrtk/Profiling.h"
#include <Eigen/Dense>
#include<iostream>
#include<fstream>

using namespace std;
using namespace mirtk;
using namespace svrtk;
using namespace boost::program_options;
// using namespace Eigen;

// =============================================================================
//
// =============================================================================

// -----------------------------------------------------------------------------

void PrintUsage(const options_description& opts) {
    // Print positional arguments
    cout << "Usage: reconstructqMRI [MapName] [Dictionary] [Stacks] " << endl;
    cout << "  [MapName]            Name for the reconstructed map (Nifti format)" << endl;
    cout << "  [Dictionary]            The input Dictionary (.txt tab delimited format)" << endl;
    cout << "  [stack_1] .. [stack_N]            The input stacks (Nifti format)" << endl<<endl;
    cout << opts << endl;
}



// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv) {


    // -----------------------------------------------------------------------------
    // INPUT VARIABLES, FLAG AND DEFAULT VALUES
    // -----------------------------------------------------------------------------

    // Initialisation of MIRTK image reader library
    InitializeIOLibrary();

    // Initialise profiling
    SVRTK_START_TIMING();

    // Variable for flags
    string strFlags;



    // Name for output volume
    string outputName;

    // Output Map
    RealImage OutputMap;
    
    // Array of number of packages for each stack
    vector<int> packages;
    
    // Masked template stack
    RealImage maskedTemplate;
    
    // Template stack
    RealImage templateStack;

    // Template Stack Arrays
    Array<RealImage> qMRItemplateStacks;

    // Array of stack transformation to the template space
    Array<RigidTransformation> stackTransformations;

    
    // Output of stack average
    RealImage average;

    // Default optional values
    int templateNumber = 3;
    int volumetemplateNumber = 0;
    vector<int> templateNumberPerVol;
    vector<int> templateNumbersqMRI;
    int iterations = 3;
    int srIterations = 7;
    int finalsrIterations = 7;
    double sigma = 20;
    double resolution = 0.75;
    double lambda = 0.02;
    double epsilon = 1;
    double delta = 150;
    int levels = 3;
    double lastIterLambda = 0.01;
    double averageValue = 700;
    double minStandVal = 0;
    double maxStandVal = 5;
    double smoothMask = 4;
    bool globalBiasCorrection = false;
    double lowIntensityCutoff = 0.01;
    bool debug = false;
    bool jointSR = true;
    bool FiniteDiff = true;
    double FD_Step = 0.01;
    double Multiplier = 1;
    bool GenerateMap = true;
    bool profile = false;
    bool AdaptiveRegularMap = true;
    bool HistogramMatching = true;
    bool HistogramVolMatching = false;
    
    // Flags for reconstruction options:
    // Flag whether the template (-template option)
    bool useTemplate = false;

    // Flag if the exact should be used
    bool flagNoOverlapThickness = false;

    // Flag to remove black background, e.g. when neonatal motion correction is performed
    bool removeBlackBackground = false;

    // Flag to switch the intensity matching on and off
    bool intensityMatching = true;

    bool rescaleStacks = false;

    // Flag whether registration should be used
    bool registrationFlag = true;

    // Flags to switch the robust statistics on and off
    bool robustStatistics = true;
    bool robustSlicesOnly = false;

    // No log flag
    bool noLog = false;

    // Flag for SVR registration to the template (skips 1st averaging SR iteration)
    bool svrOnly = false;

    // Flag for struture-based exclusion of slices
    bool structural = false;

    // Flag for switching off NMI and using NCC for SVR step
    bool nccRegFlag = false;

    // Flag for running registration step outside
    bool remoteFlag = false;

    // Flag for no global registration
    bool noGlobalFlag = false;

    // Flag that sets slice thickness to 1.5 of spacing (for testing purposes)
    bool thinFlag = false;
    
    // Flag for saving slices
    bool saveSlicesFlag = false;
    
    bool compensateFlag = false;
    
    // Number of input stacks
    int nStacks = 0;

    // Array of stacks and stack names
    Array<RealImage> stacks;
    vector<string> stackFiles;
    
    // Input mask
    unique_ptr<RealImage> mask;

    // Array of Masks and Mask names
    Array<RealImage> Masks;
    vector<string> MasksFileNames;
    // Array of Mask Stack numbers
    vector<int> MaskStackNums = {0,3,6};
    bool MultiMask = false;

    // If we want to skip masking
    bool SkipMasking = false;

    // Dictionary filename 
    string DictFile;
    
    // Array of stack slice thickness
    vector<double> thickness;
    
    // Array of stack echo times
    vector<double> echoTimes = {80,180,400};
    
    // Array of T2 ranges for the dictionary
    vector<double> T2ValsStartStopEnd = {25,3000};
    
    // Number of stacks per echo time
    vector<int> StacksPerEchoTime = {3,3,3};
       
    
    // Create reconstruction object
    ReconstructionqMRI reconstruction;
    
    // Paths of 'dofin' arguments
    vector<string> dofinPaths;
    
    
    cout << "------------------------------------------------------" << endl;
    
    
    
    // -----------------------------------------------------------------------------
    // READ INPUT DATA AND OPTIONS
    // -----------------------------------------------------------------------------

    // Define required options
    options_description reqOpts;
    reqOpts.add_options()
    	("map", value<string>(&outputName)->required(), "Name for the T2 Map (Nifti format)")
        ("dictionary", value<string>(&DictFile)->required(), "The input Dictionary (.txt tab delimited format)")
        ("stacks", value<vector<string>>(&stackFiles)->multitoken()->required(), "The input stacks (Nifti format)");
        
    // Define positional options
    positional_options_description posOpts;
    posOpts.add("map", 1).add("dictionary", 1).add("stacks", -1);
    
    // Define optional options
    options_description opts("Options");
    opts.add_options()
        ("dofin", value<vector<string>>(&dofinPaths)->multitoken(), "The transformations of the input stack to template in \'dof\' format used in IRTK. Only rough alignment with correct orientation and some overlap is needed. Use \'id\' for an identity transformation for at leastone stack. The first stack with \'id\' transformationwill be resampled as template.")
        ("template", value<string>(), "Use template for initialisation of registration loop. [Default: average of stack registration]")
        ("thickness", value<vector<double>>(&thickness)->multitoken(), "Give slice thickness. [Default: twice voxel size in z direction]")
        ("default_thickness", value<double>(), "Default thickness for all stacks. [Default: twice voxel size in z direction]")
        ("default_packages", value<int>(), "Default package number for all stacks. [Default: 1]")
        ("mask", value<string>(), "Binary mask to define the region of interest. [Default: whole image]")
        ("multi_masks", value<vector<string>>(&MasksFileNames)->multitoken(), "Multiple binary masks to define the regions of interest for different TEs.")
        ("masks_stack_no", value<vector<int>>(&MaskStackNums)->multitoken(), "The stacks from which the multiple masks have been created.")
        ("packages", value<vector<int>>(&packages)->multitoken(), "Give number of packages used during acquisition for each stack. The stacks will be split into packages during registration iteration 1 and then into odd and even slices within each package during registration iteration 2. The method will then continue with slice to volume approach. [Default: slice to volume registration only]")
        ("template_number", value<int>(&templateNumber), "Number of the template stack [Default: 0]")
        ("T2_vals", value<vector<double>>(&T2ValsStartStopEnd)->multitoken(), "Give the start and end values in the dictionaries (and optional step increment value). Default is integers in the range [0,3000]ms")
        ("echo_times", value<vector<double>>(&echoTimes)->multitoken(), "Give the unique echo times for the stacks. [Default: 80, 180, 400ms]")
        ("n_per_echo_times", value<vector<int>>(&StacksPerEchoTime)->multitoken(), "Give number of stacks per echo time. [Default: 3,3,3 (or 3 stacks per echo time)]. User can enter a single number if constant number of stacks per echo time.")
        ("iterations", value<int>(&iterations), "Number of registration-reconstruction iterations [Default: 3]")
        ("sr_iterations", value<int>(&srIterations), "Number of SR reconstruction iterations [Default: 7,...,7,7*3]")
        ("final_sr_iterations", value<int>(&finalsrIterations), "Number of SR reconstruction iterations [Default: 7,...,7,7*3]")
        ("sigma", value<double>(&sigma), "Stdev for bias field [Default: 20mm]")
        ("resolution", value<double>(&resolution), "Isotropic resolution of the volume [Default: 0.75mm]")
        ("multires", value<int>(&levels), "Multiresolution smoothing with given number of levels [Default: 3]")
        ("svr_only", bool_switch(&svrOnly), "Only SVR registration to a template stack")
        ("no_global", bool_switch(&noGlobalFlag), "No global stack registration")
        ("average", value<double>(&averageValue), "Average intensity value for stacks [Default: 700]")
        ("delta", value<double>(&delta), "Parameter to define what is an edge [Default: 150]")
        ("epsilon", value<double>(&epsilon), "Parameter to define balance of model regularisation on the joint Super resolution.")
        ("no_joint_sr", "Switch off joint Super Resolution.")
        ("no_histogram_matching", "Don't match histograms for stacks of different TEs")
        ("volume_matching", "Don't match histograms for volumes with combined volume of different TEs")
        ("no_finite_difference", "Switch off joint Super Resolution.")
        ("fd_step", value<double>(&FD_Step), "Step size for finite difference in model fit.")
        ("multiplier", value<double>(&Multiplier), "Multiplier of alpha of normal superresolution to mix with model fit.")
        ("no_fitting", "Switch off Dictionary fitting")
        ("lambda", value<double>(&lambda), "Smoothing parameter [Default: 0.02]")
        ("lastIter", value<double>(&lastIterLambda), "Smoothing parameter for last iteration [Default: 0.01]")   
        ("smooth_mask", value<double>(&smoothMask), "Smooth the mask to reduce artefacts of manual segmentation [Default: 4mm]")
        ("global_bias_correction", bool_switch(&globalBiasCorrection), "Correct the bias in reconstructed image against previous estimation")
        ("no_intensity_matching", "Switch off intensity matching")
        ("no_adaptive_map", "Switch off adaptive regularisation for map")
        ("no_robust_statistics", "Switch off robust statistics")
        ("exact_thickness", bool_switch(&flagNoOverlapThickness), "Exact slice thickness without negative gap [Default: false]")
        ("ncc", bool_switch(&nccRegFlag), "Use global NCC similarity for SVR steps [Default: NMI]")
        ("skip_masking", bool_switch(&SkipMasking), "Use global NCC similarity for SVR steps [Default: NMI]")
        ("save_slices", bool_switch(&saveSlicesFlag), "Save slices for future exclusion [Default: false]")
        ("structural", bool_switch(&structural), "Use structural exclusion of slices at the last iteration")
        ("exclude_slices_only", bool_switch(&robustSlicesOnly), "Robust statistics for exclusion of slices only")
        ("no_registration", "Switch off registration")
        ("debug", bool_switch(&debug), "Debug mode - save intermediate results");
    
    // Combine all options
    options_description allOpts("Allowed options");
    allOpts.add(reqOpts).add(opts);
        
    // Parse arguments and catch errors
    variables_map vm;
    
    try {
        store(command_line_parser(argc, argv).options(allOpts).positional(posOpts)
            // Allow single dash (-) for long arguments
            .style(command_line_style::unix_style | command_line_style::allow_long_disguise).run(), vm);
        notify(vm);
        nStacks = accumulate(StacksPerEchoTime.begin(),StacksPerEchoTime.end(),0);
        cout << "Just after command line parsing are stack files seen? StackFiles size: " << stackFiles.size() << endl;
        
        if (stackFiles.size() < nStacks)
            throw error("Total for number of stacks for each echo time should be equal to the stack count!");
        if (StacksPerEchoTime.size() == 1){
            int allNumberOfStacks = StacksPerEchoTime[0];
            for (int i = 1; i < echoTimes.size(); i++)
                StacksPerEchoTime.push_back(allNumberOfStacks);
        }
        if (!packages.empty() && packages.size() < nStacks)
            throw error("Count of package values should equal to stack count!");
        if (echoTimes.size() != StacksPerEchoTime.size())
            throw error("Please give the appropriate number of stacks per echo time for all echo times given.");
    } catch (error& e) {
        // Delete -- from the argument name in the error message
        string err = e.what();
        size_t dashIndex = err.find("\'--");
        if (dashIndex != string::npos)
            err.erase(dashIndex + 1, 2);
        cerr << "Argument parsing error: " << err << "\n\n";
        PrintUsage(opts);
        return 1;
    }

    cout << "Reconstructed T2 map name : " << outputName << endl;
    cout << "Number of stacks : " << nStacks << endl;
    cout << "Dictionary text filename: " << DictFile << endl; 
    vector<double> echoTimeforStack;
    vector<int> VolumeIndForStack;
    int cummulativeTEstacksCount = 0;
    for (int i = 0; i < StacksPerEchoTime.size(); i++){
    	for (int j = 0; j < StacksPerEchoTime[i]; j++) {
            echoTimeforStack.push_back(echoTimes[i]);
            VolumeIndForStack.push_back(i);
        }
        cummulativeTEstacksCount += StacksPerEchoTime[i];
        int theonebefore = cummulativeTEstacksCount - StacksPerEchoTime[i];
        if (cummulativeTEstacksCount > templateNumber & theonebefore <= templateNumber) {
            templateNumberPerVol.push_back(templateNumber - (theonebefore));
            volumetemplateNumber = i;
            templateNumbersqMRI.push_back(templateNumber);
        }
        else {
            templateNumberPerVol.push_back(0);
            templateNumbersqMRI.push_back(theonebefore + 0);
        }
    }

    if (echoTimes.size() == 1){
        GenerateMap = false;
        templateNumber = 0;
    }
    else if (!vm.count("template_number")) {
        templateNumber = StacksPerEchoTime[0];
    }

    // Generate echo times for images
    reconstruction.MakeTEVect(StacksPerEchoTime, echoTimes, templateNumber);
    
    // Read input stacks
    for (int i = 0; i < nStacks; i++) {
        cout << "Stack" << i << " with echo time " << echoTimeforStack[i] << "ms: " << stackFiles[i];
        RealImage stack(stackFiles[i].c_str());


        // Check if the intensity is not negative and correct if so
        double smin, smax;
        stack.GetMinMax(&smin, &smax);
        if (smin < 0 || smax < 0)
            stack.PutMinMaxAsDouble(0, 1000);

        // Print stack info
        double dx = stack.GetXSize();
        double dy = stack.GetYSize();
        double dz = stack.GetZSize();
        double dt = stack.GetTSize();
        double sx = stack.GetX();
        double sy = stack.GetY();
        double sz = stack.GetZ();
        double st = stack.GetT();

        cout << "  ;  size : " << sx << " - " << sy << " - " << sz << " - " << st << "  ;";
        cout << "  voxel : " << dx << " - " << dy << " - " << dz << " - " << dt << "  ;";
        cout << "  range : [" << smin << "; " << smax << "]" << endl << endl;

        // Push individual stack into stacks array
        stacks.push_back(move(stack));
    }

    // Input stack transformations to the template space
    if (!dofinPaths.empty()) {
        for (size_t i = 0; i < stacks.size(); i++) {
            cout << "Transformation " << i << " : " << dofinPaths[i] << endl;
            Transformation *t = Transformation::New(dofinPaths[i].c_str());
            unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(t));
            stackTransformations.push_back(*rigidTransf);
        }
        InvertStackTransformations(stackTransformations);
    }

    // Slice thickness per stack
    if (!thickness.empty()) {
        cout << "Slice thickness : ";
        for (size_t i = 0; i < stacks.size(); i++)
            cout << thickness[i] << " ";
        cout << endl;
    }

    // Number of packages for each stack
    if (!packages.empty()) {
        cout << "Package number : ";
        for (size_t i = 0; i < stacks.size(); i++)
            cout << packages[i] << " ";
        cout << endl;
        reconstruction.SetNPackages(packages);
    }

    // Template for initialisation of registration
    if (vm.count("template")) {
        cout << "Template : " << vm["template"].as<string>() << endl;
        templateStack.Read(vm["template"].as<string>().c_str());

        // Check intensities and rescale if required
        double smin, smax;
        templateStack.GetMinMax(&smin, &smax);

        if (smin == -1)
            reconstruction.RemoveNegativeBackground(templateStack);

        templateStack.GetMinMax(&smin, &smax);

        if (smin < 0)
            templateStack.PutMinMaxAsDouble(0, smax);

        if (smax < 0)
            templateStack.PutMinMaxAsDouble(0, 1000);

        // Extract 1st dynamic
        if (templateStack.GetT() > 1)
            templateStack = templateStack.GetRegion(0, 0, 0, 0, templateStack.GetX(), templateStack.GetY(), templateStack.GetZ(), 1);

        useTemplate = true;
        reconstruction.SetTemplateFlag(useTemplate);
    } else if (vm.count("template_number"))
        templateStack = stacks[templateNumber];

    // Binary mask for reconstruction / final volume
    if (vm.count("mask")) {
        cout << "Mask : " << vm["mask"].as<string>() << endl;
        mask = unique_ptr<RealImage>(new RealImage(vm["mask"].as<string>().c_str()));
    }

    // Number of registration-reconstruction iterations
    if (vm.count("iterations"))
        strFlags += " -iterations " + to_string(iterations);
    
    // The same thickness for all stacks
    if (vm.count("default_thickness")) {
        double defaultThickness = vm["default_thickness"].as<double>();
        cout << "Slice thickness (default for all stacks): ";
        for (size_t i = 0; i < stacks.size(); i++) {
            thickness.push_back(defaultThickness);
            cout << thickness[i] << " ";
        }
        cout << endl;
    }
    
    // The same number of packages for all stacks
    if (vm.count("default_packages")) {
        int defaultPackageNumber = vm["default_packages"].as<int>();
        cout << "Number of packages (default for all stacks): ";
        for (size_t i = 0; i < stacks.size(); i++) {
            packages.push_back(defaultPackageNumber);
            cout << packages[i] << " ";
        }
        cout << endl;
    }

    // If Masks are given
    if (vm.count("multi_masks")) {
        MultiMask = true;
        cout << "Using multiple masks: ";
        for (size_t i = 0; i < MasksFileNames.size(); i++) {
            RealImage mask_i(MasksFileNames[i].c_str());
            Masks.push_back(mask_i);
            cout << "Mask name " << MasksFileNames[i] << " for echo time " << echoTimes[i] << endl;
            if (MaskStackNums[i] == templateNumber) {
                mask = unique_ptr<RealImage>(new RealImage(MasksFileNames[i].c_str()));
                cout << "Main mask : " << MasksFileNames[i].c_str() << endl;
            }
        }
    }

    // Number of SR iterations
    if (vm.count("sr_iterations"))
        strFlags += " -sr_iterations " + to_string(srIterations);

    // Number of final SR iterations
    if (vm.count("final_sr_iterations"))
        strFlags += " -final_sr_iterations " + to_string(finalsrIterations);

    // Variance of Gaussian kernel to smooth the bias field
    if (vm.count("sigma"))
        strFlags += " -sigma " + to_string(sigma);

    // SR smoothing parameter
    if (vm.count("lambda"))
        strFlags += " -lambda " + to_string(lambda);

    // Model regularisation parameter
    if (vm.count("epsilon"))
        strFlags += " -epsilon " + to_string(epsilon);

    // Multiplier for alpha parameter
    if (vm.count("multiplier"))
        strFlags += " -multiplier " + to_string(Multiplier);

    // Smoothing parameter for last iteration
    if (vm.count("lastIter"))
        strFlags += " -lastIter " + to_string(lastIterLambda);

    // SR parameter to define what is an edge
    if (vm.count("delta"))
        strFlags += " -delta " + to_string(delta);

    // Isotropic resolution for the reconstructed volume
    if (vm.count("resolution"))
        strFlags += " -resolution " + to_string(resolution);

    // Number of resolution levels
    if (vm.count("multires"))
        strFlags += " -multires " + to_string(levels);

//    // Run registration as remote functions
//    if (remoteFlag)
//        strFlags += " -remote";

    // Use only SVR to the template (skip 1st SR averaging iteration)
    if (svrOnly)
        strFlags += " -svr_only";

    // Use NCC similarity metric for SVR
    if (nccRegFlag) {
        reconstruction.SetNCC(nccRegFlag);
        strFlags += " -ncc";
    }

    // Switch off intensity matching
    if (vm.count("no_intensity_matching")) {
        intensityMatching = false;
        strFlags += " -no_intensity_matching";
    }

    // Switch off histogram matching
    if (vm.count("no_histogram_matching")) {
        HistogramMatching = false;
        strFlags += " -no_histogram_matching";
    }

    // Switch off histogram matching
    if (vm.count("volume_matching")) {
        HistogramVolMatching = true;
        strFlags += " -volume_matching";
    }

    // Switch off intensity matching
    if (vm.count("no_adaptive_map")) {
        AdaptiveRegularMap = false;
        strFlags += " -no_adaptive_map";
    }


    // Switch off joint reconstruction
    if (vm.count("no_joint_sr")) {
        jointSR = false;
        strFlags += " -no_joint_sr";
    }

    // Switch off finite difference in joint SR
    if (vm.count("no_finite_difference")) {
        FiniteDiff = false;
        strFlags += " -no_finite_difference";
    }

    // Finite Difference step parameter
    if (vm.count("fd_step"))
        strFlags += " -fd_step " + to_string(FD_Step);

    // Switch off dictionary fitting
    if (vm.count("no_fitting")) {
        GenerateMap = false;
        strFlags += " -no_fitting";
    }

    // Switch off robust statistics
    if (vm.count("no_robust_statistics")) {
        robustStatistics = false;
        strFlags += " -no_robust_statistics";
    }

    // Use structural exclusion of slices (after 2nd iteration)
    if (structural)
        strFlags += " -structural";

    // Use robust statistics for slices only
    if (robustSlicesOnly)
        strFlags += " -exclude_slices_only";

    // Switch off registration
    if (vm.count("no_registration")) {
        registrationFlag = false;
        strFlags += " -no_registration";
    }

    // Perform bias correction of the reconstructed image agains the GW image in the same motion correction iteration
    if (globalBiasCorrection)
        strFlags += " -global_bias_correction";

    // Debug mode
    if (debug)
        strFlags += " -debug";





    // -----------------------------------------------------------------------------
    // SET RECONSTRUCTION OPTIONS AND PERFORM PREPROCESSING
    // -----------------------------------------------------------------------------

    reconstruction.SetStructural(structural);

    reconstruction.SetEpsilon(epsilon);

    if (FiniteDiff)
        reconstruction.SetFiniteDiffStep(FD_Step);
    else {
        reconstruction.SetFiniteDiffStep(1);
        reconstruction.FiniteDifferenceSwitch(FiniteDiff);
    }

    reconstruction.MultiplyAlpha(Multiplier);

    // Set thickness to the exact dz value if specified
    if (flagNoOverlapThickness && thickness.size() < 1) {
        for (size_t i = 0; i < stacks.size(); i++)
            thickness.push_back(stacks[i].GetZSize());
    }

    // Check if stacks have multiple dynamics and spit them if it is the case
    bool has4DStacks = false;
    for (size_t i = 0; i < stacks.size(); i++) {
        if (stacks[i].GetT() > 1) {
            has4DStacks = true;
            break;
        }
    }

    // Initialise 2*slice thickness if not given by user
    if (thickness.empty()) {
        cout << "Slice thickness : ";
        for (size_t i = 0; i < stacks.size(); i++) {
            double dz = stacks[i].GetZSize();
            if (thinFlag)
                thickness.push_back(dz * 1.2);
            else
                thickness.push_back(dz * 2);
            cout << thickness[i] << " ";
        }
        cout << endl;
    }

    if (compensateFlag)  {

        double average_thickness = 0;
        double average_spacing = 0;
        for (size_t i = 0; i < stacks.size(); i++) {
            average_thickness = average_thickness + thickness[i];
            average_spacing = average_spacing + stacks[i].GetZSize();
        }
        average_thickness  = average_thickness / stacks.size();
        average_spacing = average_spacing / stacks.size();

        if (stacks.size() < 16 && average_thickness/average_spacing < 0.75 ) {
            cout << "Adjusting for undersampling : " << average_thickness/average_spacing << endl;
            for (size_t i = 0; i < stacks.size(); i++) {
                thickness[i] = thickness[i] * 1.5;
            }
        }

    }


    if (has4DStacks) {
        cout << "Splitting stacks into dynamics ... ";

        Array<double> newThickness;
        Array<int> newPackages;
        Array<RigidTransformation> newStackTransformations;
        Array<RealImage> newStacks;

        for (size_t i = 0; i < stacks.size(); i++) {
            if (stacks[i].GetT() == 1) {
                newStacks.push_back(stacks[i]);
                if (!stackTransformations.empty())
                    newStackTransformations.push_back(stackTransformations[i]);
                if (!packages.empty())
                    newPackages.push_back(packages[i]);
                if (!thickness.empty())
                    newThickness.push_back(thickness[i]);
            } else {
                for (int t = 0; t < stacks[i].GetT(); t++) {
                    RealImage stack = stacks[i].GetRegion(0, 0, 0, t, stacks[i].GetX(), stacks[i].GetY(), stacks[i].GetZ(), t + 1);
                    newStacks.push_back(move(stack));
                    if (!stackTransformations.empty())
                        newStackTransformations.push_back(stackTransformations[i]);
                    if (!packages.empty())
                        newPackages.push_back(packages[i]);
                    if (!thickness.empty())
                        newThickness.push_back(thickness[i]);
                }
            }
        }

        stacks = move(newStacks);
        thickness = move(newThickness);
        packages = move(newPackages);
        stackTransformations = move(newStackTransformations);

        cout << "New number of stacks : " << stacks.size() << endl;
    }

    if (packages.size() > 0) {
        cout << "Splitting stacks into packages ... ";

        Array<double> newThickness;
        Array<int> newPackages;
        Array<RigidTransformation> newStackTransformations;
        Array<RealImage> newStacks;

        int q = 0;
        for (size_t i = 0; i < stacks.size(); i++) {
            Array<RealImage> out_packages;
            if (packages[i] > 1) {
                SplitImage(stacks[i], packages[i], out_packages);
                for (size_t j = 0; j < out_packages.size(); j++) {
                    newStacks.push_back(out_packages[j]);
                    q = q + 1;
                    if (thickness.size() > 0)
                        newThickness.push_back(thickness[i]);
                    newPackages.push_back(1);
                    if (!stackTransformations.empty())
                        newStackTransformations.push_back(stackTransformations[i]);
                }
            } else {
                newStacks.push_back(stacks[i]);
                if (thickness.size() > 0)
                    newThickness.push_back(thickness[i]);
                newPackages.push_back(1);
                if (!stackTransformations.empty())
                    newStackTransformations.push_back(stackTransformations[i]);
            }
        }

        stacks = move(newStacks);
        thickness = move(newThickness);
        packages = move(newPackages);
        stackTransformations = move(newStackTransformations);

        cout << "New number of stacks : " << stacks.size() << endl;
    }


    // Read path to MIRTK executables for remote registration
    string strMirtkPath(argv[0]);
    strMirtkPath = strMirtkPath.substr(0, strMirtkPath.find_last_of("/"));
    const string strCurrentMainFilePath = boost::filesystem::current_path().string();

    // Create an empty file exchange directory
    const string strCurrentExchangeFilePath = strCurrentMainFilePath + "/tmp-file-exchange";
    boost::filesystem::remove_all(strCurrentExchangeFilePath.c_str());
    boost::filesystem::create_directory(strCurrentExchangeFilePath.c_str());

    // If transformations were not defined by user, set them to identity
    if (dofinPaths.empty())
        stackTransformations = Array<RigidTransformation>(stacks.size());

    // Set debug mode option
    if (debug) reconstruction.DebugOn();
    else reconstruction.DebugOff();

    // Set verbose mode on with file
    reconstruction.VerboseOn("log-registration.txt");


    // If no mask was given - try to create mask from the template image in case it was padded
    if (mask == NULL) {
        mask = unique_ptr<RealImage>(new RealImage(stacks[templateNumber]));
        *mask = CreateMask(*mask);
        cout << "Warning : no mask was provided " << endl;
        if (debug) {
            mask->Write("mask.nii.gz");
        }
    }

    // Set low intensity cutoff for bias estimation
    reconstruction.SetLowIntensityCutoff(lowIntensityCutoff);

    // If no separate template was provided - use the selected template stack
    if (!useTemplate) {
        templateStack = stacks[templateNumber];
    }

    // Before creating the template we will crop template stack according to the given mask
    // What is the point of this if-statement, it's always going to mask?
    if (mask != NULL) {

        RealImage m = *mask;
        TransformMask(stacks[templateNumber], m, stackTransformations[templateNumber]);

        // Crop template stack and prepare template for global volumetric registration
        maskedTemplate = stacks[templateNumber] * m;
        CropImage(stacks[templateNumber], m);
        CropImage(maskedTemplate, m);

        if (debug) {
            m.Write("maskforTemplate.nii.gz");
            maskedTemplate.Write("croppedTemplate.nii.gz");
        }

    }

    //for(int ii = 0; ii < stacks.size(); ii++)
    //    cout << stacks[ii].GetZ() << endl;

    // If the template was provided separately - crop and mask the template with the given mask
    if (useTemplate) {
        RealImage m = *mask;
        TransformMask(templateStack, m, RigidTransformation());

        // Crop template stack and prepare template for global volumetric registration
        maskedTemplate = templateStack * m;
        CropImage(maskedTemplate, m);
        CropImage(templateStack, m);

        if (debug) {
            m.Write("maskforTemplate.nii.gz");
            maskedTemplate.Write("croppedTemplate.nii.gz");
        }
        for (int ii = 0; ii < echoTimes.size(); ii++){
            qMRItemplateStacks.push_back(maskedTemplate);
        }
    }
    else{
        for (int ii = 0; ii < echoTimes.size(); ii++){
            RealImage m = Masks[ii];
            int tnum = templateNumberPerVol[ii];
            RealImage TImage = stacks[tnum];
            TransformMask(TImage, m, stackTransformations[tnum]);
            RealImage TMTemp = TImage * m;
            CropImage(TImage, m);
            CropImage(TMTemp, m);
            qMRItemplateStacks.push_back(TMTemp);
        }

    }

    // Create template volume with isotropic resolution
    // If resolution==0 it will be determined from in-plane resolution of the image
    resolution = reconstruction.CreateTemplateqMRI(qMRItemplateStacks, resolution, volumetemplateNumber);

    if (debug) {
        // Save intermediate reconstructed image
        int superit = 0;
        reconstruction.GetReconstructedqMRI().Write("AfterCreateTemplate.nii.gz");
    }

    // Set mask to reconstruction object

    if (MultiMask)
        reconstruction.SetMasksqMRI(Masks, MaskStackNums, templateNumber, smoothMask);
    else
        reconstruction.SetMask(mask.get(), smoothMask);


    // If remove_black_background flag is set, create mask from black background of the stacks
    if (removeBlackBackground)
        CreateMaskFromBlackBackground(&reconstruction, stacks, stackTransformations);

    // Set precision
    cout << setprecision(3);
    cerr << setprecision(3);


    // -----------------------------------------------------------------------------
    // RUN GLOBAL STACK REGISTRATION AND FURTHER PREPROCESSING
    // -----------------------------------------------------------------------------

    cout << "------------------------------------------------------" << endl;

    // Volumetric stack to template registration
    if (!noGlobalFlag)
        reconstruction.StackRegistrations(stacks, stackTransformations, templateNumber, &templateStack);

    // Create average volume
    average = reconstruction.CreateAverage(stacks, stackTransformations);
    if (debug)
        average.Write("average1.nii.gz");

    // Mask is transformed to all the other stacks and they are cropped

    for (size_t i = 0; i < stacks.size(); i++) {
        // Template stack has been cropped already
        if (i == templateNumber)
            continue;
        // Transform the mask

        if(MultiMask){
            RealImage m = reconstruction.GetMaskqMRI(VolumeIndForStack[i]);
            TransformMask(stacks[i], m, stackTransformations[i]);
            // Crop template stack
            CropImage(stacks[i], m);

            if (debug) {
                m.Write((boost::format("mask%1%.nii.gz") % i).str().c_str());
                stacks[i].Write((boost::format("cropped%1%.nii.gz") % i).str().c_str());
            }
        }
        else {
            if (SkipMasking) {
                RealImage m = reconstruction.MaskAndGetMaskSlice(stacks[i]);
                CropImage(stacks[i], m);


                if (debug) {
                    m.Write((boost::format("mask%1%.nii.gz") % i).str().c_str());
                    stacks[i].Write((boost::format("cropped%1%.nii.gz") % i).str().c_str());
                }

            }
            else {
                RealImage m = reconstruction.GetMask();
                TransformMask(stacks[i], m, stackTransformations[i]);
                // Crop template stack
                CropImage(stacks[i], m);


                if (debug) {
                    m.Write((boost::format("mask%1%.nii.gz") % i).str().c_str());
                    stacks[i].Write((boost::format("cropped%1%.nii.gz") % i).str().c_str());
                }
            }
        }


    }

    // Remove small stacks (no intersection with ROI)
    Array<RealImage> selectedStacks;
    Array<RigidTransformation> selectedStackTransformations;
    int nNewStacks = 0;
    int newTemplateNumber = 0;
    for (size_t i = 0; i < stacks.size(); i++) {
        if (stacks[i].GetX() == 1) {
            cerr << "stack " << i << " has no intersection with ROI" << endl;
            continue;
        }
        selectedStacks.push_back(stacks[i]);
        selectedStackTransformations.push_back(stackTransformations[i]);
        if (i == templateNumber)
            newTemplateNumber = nNewStacks;
        nNewStacks++;
    }
    templateNumber = newTemplateNumber;
    stacks = move(selectedStacks);
    stackTransformations = move(selectedStackTransformations);

    // Perform volumetric registration again
    if (!noGlobalFlag)
        reconstruction.StackRegistrations(stacks, stackTransformations, templateNumber, &templateStack);

    cout << "------------------------------------------------------" << endl;

    // Switch on/off intensity matching
    reconstruction.IntensityMatchingSwitch(intensityMatching);

    // Switch on/off Super Resolution
    reconstruction.SuperResolutionSwitch(jointSR);
    if (jointSR)
        reconstruction.SuperResolutionSwitch(GenerateMap);

    // Rescale intensities of the stacks to have the same average
    if (intensityMatching) {
        reconstruction.MatchStackIntensitiesWithMaskingqMRI(stacks, stackTransformations, averageValue);
    }
    else
        reconstruction.FindStackAverage(stacks, stackTransformations);

    reconstruction.SetStandardMinMax(minStandVal,maxStandVal);

    // Create average of the registered stacks -> Might need to fix this
    average = reconstruction.CreateAverage(stacks, stackTransformations);
    if (debug)
        average.Write("average2.nii.gz");

    // Create slices and slice-dependent transformations
    Array<RealImage> probabilityMaps;
    /*if (!HistogramMatching || echoTimes.size() == 1){
        reconstruction.CreateSlicesAndTransformations(stacks, stackTransformations, thickness, probabilityMaps);
    }
    else {
        reconstruction.CreateSlicesAndTransformationsqMRI(stacks, stackTransformations, thickness, probabilityMaps,templateStack,HistogramMatching);
    }*/
    reconstruction.CreateSlicesAndTransformationsqMRI(stacks, stackTransformations, thickness,
                                                      probabilityMaps,templateStack,HistogramMatching, averageValue);
    reconstruction.CreateSimulatedSlices(stacks, thickness);

    //reconstruction.SaveSlices();

    // Save slices for future exclusion
    if(saveSlicesFlag) {
        reconstruction.SaveSlices();
        reconstruction.SaveSlicesqMRI();
    }

    // Mask all the slices
    if (!SkipMasking){
        reconstruction.MaskSlices();
        reconstruction.MaskSlicesqMRI();
    }



    // Set sigma for the bias field smoothing
    if (sigma > 0){
        reconstruction.SetSigma(sigma);
        reconstruction.SetSigmaqMRI(sigma);
    }
    else {
        reconstruction.SetSigma(20);
        reconstruction.SetSigmaqMRI(20);
    }

    // Set global bias correction flag
    if (globalBiasCorrection)
        reconstruction.GlobalBiasCorrectionOn();
    else
        reconstruction.GlobalBiasCorrectionOff();

    /*// If given read slice-to-volume registrations
    if (!folder.empty())
        reconstruction.ReadTransformations((char*)folder.c_str());*/

    // Initialise data structures for EM
    reconstruction.InitializeEMqMRI();

    // If registration was switched off - only 1 iteration is required
    if (!registrationFlag)
        iterations = 1;


    // Instantiate dictionary for the model regularisation and/or building the final T2 map
    Dictionary T2Dictionary(DictFile.c_str());
    if (T2ValsStartStopEnd.size() == 2)
        T2Dictionary.SetT2Vals(T2ValsStartStopEnd[0], T2ValsStartStopEnd[1]);
    if (T2ValsStartStopEnd.size() == 3)
        T2Dictionary.SetT2Vals(T2ValsStartStopEnd[0],T2ValsStartStopEnd[1],T2ValsStartStopEnd[3]);
    reconstruction.InstantiateDictionary(T2Dictionary);

    // -----------------------------------------------------------------------------
    // RUN INTERLEAVED SVR-SR RECONSTRUCTION
    // -----------------------------------------------------------------------------


    int currentIteration = 0;

    {
        // Interleaved registration-reconstruction iterations
        for (int iter = 0; iter < iterations; iter++) {
            if (iter == iterations-1)
                reconstruction.MultiplyAlpha(Multiplier);
            cout << "------------------------------------------------------" << endl;
            cout << "Iteration : " << iter << endl;

            reconstruction.SetCurrentIteration(iter);

            reconstruction.MaskVolume();
            reconstruction.MaskVolumeqMRI();

            // If only SVR option is used - skip 1st SR only averaging
            if (svrOnly || iter > 0) {
                if (remoteFlag)
                    reconstruction.RemoteSliceToVolumeRegistration(iter, strMirtkPath, strCurrentExchangeFilePath);
                else{
                    reconstruction.SliceToVolumeRegistration();
                }
            }

            // Run global NNC structure-based outlier rejection of slices
            if (structural)
                reconstruction.GlobalStructuralExclusion();

            // Set smoothing parameters
            // Amount of smoothing (given by lambda) is decreased with improving alignment
            // Delta (to determine edges) stays constant throughout
            if (iter == (iterations - 1))
                reconstruction.SetSmoothingParameters(delta, lastIterLambda);
            else {
                double l = lambda;
                for (int i = 0; i < levels; i++) {
                    if (iter == iterations * (levels - i - 1) / levels)
                        reconstruction.SetSmoothingParameters(delta, l);
                    l *= 2;
                }
            }

            // Use faster reconstruction during iterations and slower for final reconstruction
            if (iter < iterations - 1)
                reconstruction.SpeedupOn();
            else
                reconstruction.SpeedupOff();

            if (robustSlicesOnly)
                reconstruction.ExcludeWholeSlicesOnly();

            if (iter < iterations - 1)
                reconstruction.InitializeEMValues();
            else {
                if (HistogramVolMatching || iterations == 1)
                    reconstruction.InitializeEMValues();
                reconstruction.InitializeEMValuesqMRI();
            }

            if (debug) {
                // Save intermediate reconstructed image
                int superit = 0;
                reconstruction.GetReconstructed().Write((boost::format("AfterInitializeEM%1%.nii.gz") % iter).str().c_str());
                reconstruction.GetReconstructedqMRI().Write((boost::format("AfterInitializeEM4D_%1%.nii.gz") % iter).str().c_str());
            }

            // Calculate matrix of transformation between voxels of slices and volume
            reconstruction.CoeffInitqMRI();

            if (debug) {
                reconstruction.GetReconstructed().Write((boost::format("AfterCoeffInit%1%.nii.gz") % iter).str().c_str());
                reconstruction.GetReconstructedqMRI().Write((boost::format("AfterCoeffInit4D_%1%.nii.gz") % iter).str().c_str());
            }

            // Initialise template reconstruction for SVR using normal Gaussian Recon function
            if (iter < iterations - 1 ){
                reconstruction.GaussianReconstruction();
            }
            else {
                // Initialise reconstructed image with Gaussian weighted reconstruction
                if (HistogramVolMatching)
                    reconstruction.GaussianReconstruction();
                reconstruction.GaussianReconstructionqMRI(iter);
            }

            if (debug) {
                // Save intermediate reconstructed image
                reconstruction.GetReconstructed().Write((boost::format("AfterGaussian%1%.nii.gz") % iter).str().c_str());
                reconstruction.GetReconstructedqMRI().Write((boost::format("AfterGaussian4D_%1%.nii.gz") % iter).str().c_str());
                int superit = 0;
                reconstruction.GetReconstructed().Write((boost::format("super%1%.nii.gz") % superit).str().c_str());
                reconstruction.GetReconstructedqMRI().Write((boost::format("super4D%1%.nii.gz") % superit).str().c_str());
            }

            // Simulate slices (needs to be done after Gaussian reconstruction)
            if (iter < iterations - 1){
                reconstruction.SimulateSlices();
                /*if (iter == iterations-2)
                    reconstruction.SimulateSlicesqMRI();*/
                }
            else{
                if (HistogramVolMatching)
                    reconstruction.SimulateSlices();
                reconstruction.SimulateSlicesqMRI();
            }

            // Initialize robust statistics parameters
            if (iter < iterations - 1)
                reconstruction.InitializeRobustStatistics();
            else {
                if (HistogramVolMatching)
                    reconstruction.InitializeRobustStatistics();
                reconstruction.InitializeRobustStatisticsqMRI();
            }

            // EStep
            if (robustStatistics)
                if (iter < iterations - 1)
                    reconstruction.EStep();
                else {
                    if (HistogramVolMatching)
                        reconstruction.EStep();
                    reconstruction.EStepqMRI();
                }

            // Run local SSIM structure-based outlier rejection
            if (structural) {
                reconstruction.CreateSliceMasks();
                reconstruction.SStep();
            }

            int recIterations;

            // Set number of reconstruction iterations
            if (srIterations != finalsrIterations)
                recIterations = iter == iterations - 1 ? finalsrIterations * 3 : srIterations;
            else
                recIterations = iter == iterations - 1 ? srIterations * 3 : srIterations;

            // SR reconstruction loop
            for (int i = 0; i < recIterations; i++) {
                if (debug) {
                    cout << "------------------------------------------------------" << endl;
                    cout << "Reconstruction iteration : " << i << endl;
                }

                if (intensityMatching) {
                    // Calculate bias fields
                    if (sigma > 0) {
                        if (iter < iterations - 1) {
                            reconstruction.Bias();
                            /*if (iter == iterations - 2 && i == recIterations - 1)
                                reconstruction.BiasqMRI();*/
                        }
                        else {
                            if (HistogramVolMatching)
                                reconstruction.Bias();
                            reconstruction.BiasqMRI();
                        }
                    }

                    // Calculate scales
                    if (iter < iterations - 1) {
                        reconstruction.Scale();
                        /*if (iter == iterations - 2 && i == recIterations - 1)
                            reconstruction.ScaleqMRI();*/
                    }
                    else {
                        if (HistogramVolMatching)
                            reconstruction.Scale();
                        reconstruction.ScaleqMRI();
                    }
                }



                // Update reconstructed volume - super-resolution reconstruction
                if (iter < iterations - 1)
                    reconstruction.Superresolution(i + 1);
                else {
                    if (HistogramVolMatching)
                        reconstruction.Superresolution(i + 1);
                    reconstruction.SuperresolutionqMRI(i + 1);
                }

                if (debug) {
                    int superit1 = i + 1;
                    reconstruction.GetReconstructedqMRI().Write((boost::format("AfterSuperresolution%1%.nii.gz") % superit1).str().c_str());
                }



                // Run bias normalisation
                if (intensityMatching && sigma > 0 && !globalBiasCorrection) {
                    if (iter < iterations - 1)
                        reconstruction.NormaliseBias(i);
                    else {
                        if (HistogramVolMatching)
                            reconstruction.NormaliseBias(i);
                        reconstruction.NormaliseBiasqMRI(i);
                    }
                }

                // Simulate slices (needs to be done after the update of the reconstructed volume)
                if (iter < iterations - 1){
                    reconstruction.SimulateSlices();
                    /*if (iter == iterations - 2 && i == recIterations - 1)
                        reconstruction.SimulateSlicesqMRI();*/
                    }
                else {
                    if (HistogramVolMatching)
                        reconstruction.SimulateSlices();
                    reconstruction.SimulateSlicesqMRI();
                }

                // Run robust statistics for rejection of outliers
                if (robustStatistics) {
                    if (iter < iterations - 1) {
                        reconstruction.MStep(i + 1);
                        reconstruction.EStep();
                    }
                    else {
                        if (HistogramVolMatching){
                            reconstruction.MStep(i + 1);
                            reconstruction.EStep();
                        }
                        reconstruction.MStepqMRI(i + 1);
                        reconstruction.EStepqMRI();
                    }
                }

                // Run local SSIM structure-based outlier rejection
                if (structural) {
                    reconstruction.SStep();
                }

                if (debug) {
                    // Save intermediate reconstructed image
                    int superit = i + 1;
                    reconstruction.GetReconstructedqMRI().Write((boost::format("super4D%1%.nii.gz") % superit).str().c_str());
                    reconstruction.GetReconstructed().Write((boost::format("super%1%.nii.gz") % superit).str().c_str());

                    // Evaluate reconstruction quality
                    double error;
                    if (iter < iterations - 1)
                        error = reconstruction.EvaluateReconQuality(1);
                    else
                        error = reconstruction.EvaluateReconQualityqMRI(1);
                    cout << "Total reconstruction error : " << error << endl;
                }

            } // End of SR reconstruction iterations


            // Mask reconstructed image to ROI given by the mask
            reconstruction.MaskVolume();
            reconstruction.MaskVolumeqMRI();

            // Save reconstructed image
            reconstruction.GetReconstructed().Write((boost::format("image%1%.nii.gz") % iter).str().c_str());

            // Compute and save quality metrics
            double outNcc = 0;
            double outNrmse = 0;
            double averageVolumeWeight = 0;
            double ratioExcluded = 0;
            reconstruction.ReconQualityReport(outNcc, outNrmse, averageVolumeWeight, ratioExcluded);
            cout << " - global metrics: ncc = " << outNcc << " ; nrmse = " << outNrmse << " ; average weight = " << averageVolumeWeight << " ; excluded slices = " << ratioExcluded << endl;

            ofstream ofsNcc("output-metric-ncc.txt");
            ofstream ofsNrmse("output-metric-nrmse.txt");
            ofstream ofsWeight("output-metric-average-weight.txt");
            ofstream ofsExcluded("output-metric-excluded-ratio.txt");

            ofsNcc << outNcc << endl;
            ofsNrmse << outNrmse << endl;
            ofsWeight << averageVolumeWeight << endl;
            ofsExcluded << ratioExcluded << endl;

        } // End of interleaved registration-reconstruction iterations


        // -----------------------------------------------------------------------------
        // SAVE RESULTS
        // -----------------------------------------------------------------------------

        cout << "------------------------------------------------------" << endl;

        if (intensityMatching) {
            //reconstruction.RestoreSliceIntensities();
            reconstruction.RestoreSliceIntensitiesqMRI();
        }
        if (debug) {
            reconstruction.SaveTransformations();
            reconstruction.SaveSlices();
            reconstruction.SaveSlicesqMRI();
            reconstruction.SaveWeights();
            reconstruction.SaveBiasFields();
            reconstruction.SaveSimulatedSlices();
            reconstruction.SaveSimulatedSlicesqMRI();
            reconstruction.SimulateStacks(stacks);
            reconstruction.WriteReconstructedIMGs();
            for (size_t i = 0; i < stacks.size(); i++)
                stacks[i].Write((boost::format("simulated%1%.nii.gz") % i).str().c_str());
        }

        if (intensityMatching) {
            //reconstruction.ScaleVolume();
            reconstruction.ScaleVolumeqMRI();
        }
    }



    string ReconName = "Reconstructions_";
    ReconName = ReconName.append(outputName);
    //reconstruction.GetReconstructedqMRI().Write(ReconName.c_str());
    reconstruction.WriteReconstructedIMGs(outputName);
    if ((echoTimes.size() > 1) || GenerateMap) {
        OutputMap = T2Dictionary.FitDictionaryToMap(reconstruction.GetReconstructedIMGs());
        if (AdaptiveRegularMap) {
            reconstruction.SetT2Map(OutputMap);
            //OutputMap.Write("TestMap.nii.gz");
            reconstruction.AdaptiveRegularizationT2Map(21, OutputMap);
            reconstruction.GetT2Map().Write(outputName.c_str());
        }
        else
            OutputMap.Write(outputName.c_str());
    }
    if (HistogramVolMatching) {
        reconstruction.GetReconstructed().Write("Combined.nii.gz");
        reconstruction.WriteHistogramMatchedIMGs(reconstruction.GetReconstructed());
    }

    // Remove the file exchange directory
    boost::filesystem::remove_all(strCurrentExchangeFilePath.c_str());

    cout << "Output volume : " << outputName << endl;

    SVRTK_END_TIMING("all");

    cout << "------------------------------------------------------" << endl;

    return 0;
}
