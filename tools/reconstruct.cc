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

// SVRTK
#include "svrtk/Reconstruction.h"
#define SVRTK_TOOL
#include "svrtk/Profiling.h"

using namespace std;
using namespace mirtk;
using namespace svrtk;
using namespace boost::program_options;

// =============================================================================
//
// =============================================================================

// -----------------------------------------------------------------------------

void PrintUsage(const options_description& opts) {
    // Print positional arguments
    cout << "Usage: reconstruct [reconstructed] [N] [stack_1] .. [stack_N] <options>\n" << endl;
    cout << "  [reconstructed]            Name for the reconstructed volume (Nifti format)" << endl;
    cout << "  [N]                        Number of stacks" << endl;
    cout << "  [stack_1] .. [stack_N]     The input stacks (Nifti format)" << endl << endl;
    // Print optional arguments
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

    // Folder for slice-to-volume registrations, if given
    string folder;

    // Array of stacks and stack names
    Array<RealImage> stacks;
    vector<string> stackFiles;

    // Template stack
    RealImage templateStack;

    // Input mask
    unique_ptr<RealImage> mask;

    // Array of stack transformation to the template space
    Array<RigidTransformation> stackTransformations;

    // Array of stack slice thickness
    vector<double> thickness;

    // Number of input stacks
    int nStacks = 0;

    // Array of number of packages for each stack
    vector<int> packages;

    // Masked template stack
    RealImage maskedTemplate;

    // Output of stack average
    RealImage average;

    // Variables for forced exclusion of slices
    vector<int> forceExcluded;

    // Default values for reconstruction variables:
    int templateNumber = 0;
    int iterations = 3;
    int srIterations = 7;
    bool debug = false;
    double sigma = 20;
    double resolution = 0.75;
    double lambda = 0.02;
    double delta = 150;
    int levels = 3;
    double lastIterLambda = 0.01;
    double averageValue = 700;
    double smoothMask = 4;
    bool globalBiasCorrection = false;
    double lowIntensityCutoff = 0.01;

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

    // Paths of 'dofin' arguments
    vector<string> dofinPaths;

    // Create reconstruction object
    Reconstruction reconstruction;

    cout << "------------------------------------------------------" << endl;

    // -----------------------------------------------------------------------------
    // READ INPUT DATA AND OPTIONS
    // -----------------------------------------------------------------------------

    // Define required options
    options_description reqOpts;
    reqOpts.add_options()
        ("reconstructed", value<string>(&outputName)->required(), "Name for the reconstructed volume (Nifti format)")
        ("N", value<int>(&nStacks)->required(), "Number of stacks")
        ("stack", value<vector<string>>(&stackFiles)->multitoken()->required(), "The input stacks (Nifti format)");

    // Define positional options
    positional_options_description posOpts;
    posOpts.add("reconstructed", 1).add("N", 1).add("stack", -1);

    // Define optional options
    options_description opts("Options");
    opts.add_options()
        ("dofin", value<vector<string>>(&dofinPaths)->multitoken(), "The transformations of the input stack to template in \'dof\' format used in IRTK. Only rough alignment with correct orientation and some overlap is needed. Use \'id\' for an identity transformation for at leastone stack. The first stack with \'id\' transformationwill be resampled as template.")
        ("template", value<string>(), "Use template for initialisation of registration loop. [Default: average of stack registration]")
        ("thickness", value<vector<double>>(&thickness)->multitoken(), "Give slice thickness. [Default: twice voxel size in z direction]")
        ("default_thickness", value<double>(), "Default thickness for all stacks. [Default: twice voxel size in z direction]")
        ("default_packages", value<int>(), "Default package number for all stacks. [Default: 1]")
        ("mask", value<string>(), "Binary mask to define the region of interest. [Default: whole image]")
        ("packages", value<vector<int>>(&packages)->multitoken(), "Give number of packages used during acquisition for each stack. The stacks will be split into packages during registration iteration 1 and then into odd and even slices within each package during registration iteration 2. The method will then continue with slice to volume approach. [Default: slice to volume registration only]")
        ("template_number", value<int>(&templateNumber), "Number of the template stack [Default: 0]")
        ("iterations", value<int>(&iterations), "Number of registration-reconstruction iterations [Default: 3]")
        ("sr_iterations", value<int>(&srIterations), "Number of SR reconstruction iterations [Default: 7,...,7,7*3]")
        ("sigma", value<double>(&sigma), "Stdev for bias field [Default: 20mm]")
        ("resolution", value<double>(&resolution), "Isotropic resolution of the volume [Default: 0.75mm]")
        ("multires", value<int>(&levels), "Multiresolution smoothing with given number of levels [Default: 3]")
        ("average", value<double>(&averageValue), "Average intensity value for stacks [Default: 700]")
        ("delta", value<double>(&delta), "Parameter to define what is an edge [Default: 150]")
        ("lambda", value<double>(&lambda), "Smoothing parameter [Default: 0.02]")
        ("lastIter", value<double>(&lastIterLambda), "Smoothing parameter for last iteration [Default: 0.01]")
        ("smooth_mask", value<double>(&smoothMask), "Smooth the mask to reduce artefacts of manual segmentation [Default: 4mm]")
        ("global_bias_correction", bool_switch(&globalBiasCorrection), "Correct the bias in reconstructed image against previous estimation")
        ("no_intensity_matching", "Switch off intensity matching")
        ("no_robust_statistics", "Switch off robust statistics")
        ("rescale_stacks", bool_switch(&rescaleStacks), "Rescale stacks to avoid nan pixel errors [Default: false]")
        ("svr_only", bool_switch(&svrOnly), "Only SVR registration to a template stack")
        ("no_global", bool_switch(&noGlobalFlag), "No global stack registration")
        ("compensate", bool_switch(&compensateFlag), "Compensate for undersampling")
        ("exact_thickness", bool_switch(&flagNoOverlapThickness), "Exact slice thickness without negative gap [Default: false]")
        ("ncc", bool_switch(&nccRegFlag), "Use global NCC similarity for SVR steps [Default: NMI]")
        ("save_slices", bool_switch(&saveSlicesFlag), "Save slices for future exclusion [Default: false]")
        ("structural", bool_switch(&structural), "Use structural exclusion of slices at the last iteration")
        ("exclude_slices_only", bool_switch(&robustSlicesOnly), "Robust statistics for exclusion of slices only")
        ("remove_black_background", bool_switch(&removeBlackBackground), "Create mask from black background")
        ("transformations", value<string>(&folder), "Use existing slice-to-volume transformations to initialize the reconstruction")
        ("force_exclude", value<vector<int>>(&forceExcluded)->multitoken(), "Force exclusion of slices with these indices")
//        ("remote", bool_switch(&remoteFlag), "Run SVR registration as remote functions in case of memory issues [Default: false]")
        ("no_registration", "Switch off registration")
        ("thin", bool_switch(&thinFlag), "Option for 1.5 x dz slice thickness (testing)")
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

        if (stackFiles.size() < nStacks)
            throw error("Count of input stacks should equal to stack count!");
        if (!dofinPaths.empty() && dofinPaths.size() < nStacks)
            throw error("Count of dof files should equal to stack count!");
        if (!thickness.empty() && thickness.size() < nStacks)
            throw error("Count of thickness values should equal to stack count!");
        if (!packages.empty() && packages.size() < nStacks)
            throw error("Count of package values should equal to stack count!");
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
    
    cout << "Reconstructed volume name : " << outputName << endl;
    cout << "Number of stacks : " << nStacks << endl;

    // Read input stacks
    for (int i = 0; i < nStacks; i++) {
        cout << "Stack " << i << " : " << stackFiles[i];
        RealImage stack(stackFiles[i].c_str());

        // Check if the intensity is not negative and correct if so
        double smin, smax;
        stack.GetMinMax(&smin, &smax);
        if (smin < 0 || smax < 0)
            stack.PutMinMaxAsDouble(0, 1000);

        // Print stack info
        double dx = stack.GetXSize(); double dy = stack.GetYSize();
        double dz = stack.GetZSize(); double dt = stack.GetTSize();
        double sx = stack.GetX(); double sy = stack.GetY();
        double sz = stack.GetZ(); double st = stack.GetT();

        cout << "  ;  size : " << sx << " - " << sy << " - " << sz << " - " << st << "  ;";
        cout << "  voxel : " << dx << " - " << dy << " - " << dz << " - " << dt << "  ;";
        cout << "  range : [" << smin << "; " << smax << "]" << endl;

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
        if (smin < 0 || smax < 0)
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

    // Number of SR iterations
    if (vm.count("sr_iterations"))
        strFlags += " -sr_iterations " + to_string(srIterations);

    // Variance of Gaussian kernel to smooth the bias field
    if (vm.count("sigma"))
        strFlags += " -sigma " + to_string(sigma);

    // SR smoothing parameter
    if (vm.count("lambda"))
        strFlags += " -lambda " + to_string(lambda);

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

    // Force removal of certain slices
    if (forceExcluded.size() > 0) {
        cout << forceExcluded.size() << " force excluded slices: ";
        for (size_t i = 0; i < forceExcluded.size(); i++)
            cout << forceExcluded[i] << " ";
        cout << endl;
    }

    // Switch off intensity matching
    if (vm.count("no_intensity_matching")) {
        intensityMatching = false;
        strFlags += " -no_intensity_matching";
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

        if (stacks.size() < 8 && average_thickness/average_spacing < 0.75 ) {
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

    // Rescale stack if specified
    if (rescaleStacks) {
        for (size_t i = 0; i < stacks.size(); i++)
            Rescale(stacks[i], 1000);
    }

    // If transformations were not defined by user, set them to identity
    if (dofinPaths.empty())
        stackTransformations = Array<RigidTransformation>(stacks.size());

    // Set debug mode option
    if (debug) reconstruction.DebugOn();
    else reconstruction.DebugOff();

    // Set verbose mode on with file
    reconstruction.VerboseOn("log-registration.txt");

    // Set force excluded slices option
    reconstruction.SetForceExcludedSlices(forceExcluded);

    // Set low intensity cutoff for bias estimation
    reconstruction.SetLowIntensityCutoff(lowIntensityCutoff);

    // If no mask was given - try to create mask from the template image in case it was padded
    if (mask == NULL) {
        mask = unique_ptr<RealImage>(new RealImage(stacks[templateNumber]));
        *mask = CreateMask(*mask);
        cout << "Warning : no mask was provided " << endl;
    }
    
    // If no separate template was provided - use the selected template stack
    if (!useTemplate) {
        templateStack = stacks[templateNumber];
    }

    // Before creating the template we will crop template stack according to the given mask
    if (mask != NULL) {
        // First resample the mask to the space of the stack
        // For template stack the transformation is identity
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
    }

    // Create template volume with isotropic resolution
    // If resolution==0 it will be determined from in-plane resolution of the image
    resolution = reconstruction.CreateTemplate(maskedTemplate, resolution);

    // Set mask to reconstruction object
    reconstruction.SetMask(mask.get(), smoothMask);

    // If remove_black_background flag is set, create mask from black background of the stacks
    if (removeBlackBackground)
        CreateMaskFromBlackBackground(&reconstruction, stacks, stackTransformations, smoothMask);

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

    // Mask is transformed to the all other stacks and they are cropped
    for (size_t i = 0; i < stacks.size(); i++) {
        // Template stack has been cropped already
        if (i == templateNumber)
            continue;
        // Transform the mask
        RealImage m = reconstruction.GetMask();
        TransformMask(stacks[i], m, stackTransformations[i]);
        // Crop template stack
        CropImage(stacks[i], m);

        if (debug) {
            m.Write((boost::format("mask%1%.nii.gz") % i).str().c_str());
            stacks[i].Write((boost::format("cropped%1%.nii.gz") % i).str().c_str());
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

    // Rescale intensities of the stacks to have the same average
    if (intensityMatching)
        reconstruction.MatchStackIntensitiesWithMasking(stacks, stackTransformations, averageValue);

    // Create average of the registered stacks
    average = reconstruction.CreateAverage(stacks, stackTransformations);
    if (debug)
        average.Write("average2.nii.gz");

    // Create slices and slice-dependent transformations
    Array<RealImage> probabilityMaps;
    reconstruction.CreateSlicesAndTransformations(stacks, stackTransformations, thickness, probabilityMaps);

    // Save slices for future exclusion
    if(saveSlicesFlag)
        reconstruction.SaveSlices();
    
    // Mask all the slices
    reconstruction.MaskSlices();

    // Set sigma for the bias field smoothing
    if (sigma > 0)
        reconstruction.SetSigma(sigma);
    else
        reconstruction.SetSigma(20);

    // Set global bias correction flag
    if (globalBiasCorrection)
        reconstruction.GlobalBiasCorrectionOn();
    else
        reconstruction.GlobalBiasCorrectionOff();

    // If given read slice-to-volume registrations
    if (!folder.empty())
        reconstruction.ReadTransformations((char*)folder.c_str());

    // Initialise data structures for EM
    reconstruction.InitializeEM();

    // If registration was switched off - only 1 iteration is required
    if (!registrationFlag)
        iterations = 1;

    // -----------------------------------------------------------------------------
    // RUN INTERLEAVED SVR-SR RECONSTRUCTION
    // -----------------------------------------------------------------------------

    int currentIteration = 0;

    {
        // Interleaved registration-reconstruction iterations
        for (int iter = 0; iter < iterations; iter++) {
            cout << "------------------------------------------------------" << endl;
            cout << "Iteration : " << iter << endl;
            
            reconstruction.SetCurrentIteration(iter);

            reconstruction.MaskVolume();

            // If only SVR option is used - skip 1st SR only averaging
            if (svrOnly || iter > 0) {
                if (remoteFlag)
                    reconstruction.RemoteSliceToVolumeRegistration(iter, strMirtkPath, strCurrentExchangeFilePath);
                else
                    reconstruction.SliceToVolumeRegistration();
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

            reconstruction.InitializeEMValues();

            // Calculate matrix of transformation between voxels of slices and volume
            reconstruction.CoeffInit();

            // Initialise reconstructed image with Gaussian weighted reconstruction
            reconstruction.GaussianReconstruction();

            // Simulate slices (needs to be done after Gaussian reconstruction)
            reconstruction.SimulateSlices();

            // Initialize robust statistics parameters
            reconstruction.InitializeRobustStatistics();

            // EStep
            if (robustStatistics)
                reconstruction.EStep();
            
            // Run local SSIM structure-based outlier rejection
            if (structural) {
                reconstruction.CreateSliceMasks();
                reconstruction.SStep();
            }

            // Set number of reconstruction iterations
            const int recIterations = iter == iterations - 1 ? srIterations * 3 : srIterations;

            // SR reconstruction loop
            for (int i = 0; i < recIterations; i++) {
                if (debug) {
                    cout << "------------------------------------------------------" << endl;
                    cout << "Reconstruction iteration : " << i << endl;
                }

                if (intensityMatching) {
                    // Calculate bias fields
                    if (sigma > 0)
                        reconstruction.Bias();

                    // Calculate scales
                    reconstruction.Scale();
                }

                // Update reconstructed volume - super-resolution reconstruction
                reconstruction.Superresolution(i + 1);

                // Run bias normalisation
                if (intensityMatching && sigma > 0 && !globalBiasCorrection)
                    reconstruction.NormaliseBias(i);

                // Simulate slices (needs to be done after the update of the reconstructed volume)
                reconstruction.SimulateSlices();

                // Run robust statistics for rejection of outliers
                if (robustStatistics) {
                    reconstruction.MStep(i + 1);
                    reconstruction.EStep();
                }

                // Run local SSIM structure-based outlier rejection
                if (structural) {
                    reconstruction.SStep();
                }
                
                if (debug) {
                    // Save intermediate reconstructed image
                    reconstruction.GetReconstructed().Write((boost::format("super%1%.nii.gz") % i).str().c_str());

                    // Evaluate reconstruction quality
                    double error = reconstruction.EvaluateReconQuality(1);
                    cout << "Total reconstruction error : " << error << endl;
                }

            } // End of SR reconstruction iterations

            // Mask reconstructed image to ROI given by the mask
            reconstruction.MaskVolume();

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

        if (intensityMatching)
            reconstruction.RestoreSliceIntensities();

        if (debug) {
            reconstruction.SaveTransformations();
            reconstruction.SaveSlices();
            reconstruction.SaveWeights();
            reconstruction.SaveBiasFields();
            reconstruction.SimulateStacks(stacks);
            for (size_t i = 0; i < stacks.size(); i++)
                stacks[i].Write((boost::format("simulated%1%.nii.gz") % i).str().c_str());
        }

        if (intensityMatching)
            reconstruction.ScaleVolume();
    }

    // Remove the file exchange directory
    boost::filesystem::remove_all(strCurrentExchangeFilePath.c_str());

    reconstruction.GetReconstructed().Write(outputName.c_str());

    cout << "Output volume : " << outputName << endl;

    SVRTK_END_TIMING("all");

    cout << "------------------------------------------------------" << endl;

    return 0;
}
