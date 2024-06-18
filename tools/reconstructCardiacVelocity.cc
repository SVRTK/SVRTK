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

// SVRTK
#include "svrtk/ReconstructionCardiacVelocity4D.h"
#define SVRTK_TOOL
#include "svrtk/Profiling.h"

using namespace std;
using namespace mirtk;
using namespace svrtk;
using namespace boost::program_options;

//Application to perform reconstruction of volumetric cardiac cine phase / velocity MRI from thick-slice dynamic 2D MRI

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void PrintUsage(const options_description& opts) {
    cout << "Usage: reconstructCardiacVelocity [N] [stack_1] .. [stack_N] [g_values] [g_directions] <options>\n" << endl;
    cout << "- the output files will be in velocity-*.nii.gz files" << endl;
    cout << "  [N]                     Number of stacks" << endl;
    cout << "  [stack_1]..[stack_N]    The input stacks (Nifti format)" << endl;
    cout << "  [g_values]              .txt file containing magnitudes of gradient first moments associated with each stack [g1]…[gN]" << endl;
    cout << "  [g_directions]          .txt file containing components of unit vector associated with " << endl;
    cout << "                          gradient first moment direction of each stack [gd_1_x gd_1_y gd_1_z]...[gd_N_x gd_N_y gd_N_z]" << endl;
    cout << endl;
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
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();

    // Initialise profiling
    SVRTK_START_TIMING();

    //declare variables for input
    /// Reference volume
    string refVolName;
    bool regReconToRef = false;

    /// Slice stacks
    Array<RealImage> stacks;
    Array<string> stackFiles;

    /// Stack transformation
    Array<RigidTransformation> stackTransformations;

    /// Stack thickness
    Array<double> thickness;
    ///number of stacks
    int nStacks;
    // Locations of R-R intervals;
    Array<double> rrLocs;
    // Slice R-R intervals
    Array<double> rr;
    // Slice cardiac phases
    Array<double> cardPhases;

    // alpha for gradient descend step
    double alpha = 3;

    // Default values
    int templateNumber = 0;
    unique_ptr<RealImage> mask;
    bool debug = false;
    bool profile = false;
    double sigma = 20;
    double resolution = 1.25;
    int numCardPhase = 15;
    constexpr double rrDefault = 1;
    double rrInterval = rrDefault;
    bool isTemporalPSFGauss = false;
    double lambda = 0.01;
    double delta = 50;
    int recIterations = 40;
    double smoothMask = 4;
    bool globalBiasCorrection = false;
    // double lowIntensityCutoff = 0.01;
    //folder for slice-location registrations, if given
    string sliceTransformationsFolder;
    //folder for slice-to-volume registrations, if given
    string folder;
    //folder for reference slice-to-volume registrations, if given
    string refTransformationsFolder;

    //flags for adaptive regularisation and velocity limits
    bool adaptiveRegularisation = true;
    bool limitIntensities = false;

    //flag to swich the robust statistics on and off
    bool robustStatistics = false;
    bool robustSlicesOnly = false;

    string logID;
    bool noLog = false;

    //forced exclusion of slices
    Array<int> forceExcludedSlices;
    Array<int> forceExcludedStacks;
    Array<int> forceExcludedLocs;

    // Paths of 'dofin' arguments
    vector<string> dofinPaths;

    // Gradient values
    Array<Array<double>> gDirections;
    Array<double> gValues;
    string gValFileName, gDirFileName;

    //Create reconstruction object
    ReconstructionCardiacVelocity4D reconstruction;

    // -----------------------------------------------------------------------------
    // READ INPUT DATA AND OPTIONS
    // -----------------------------------------------------------------------------

    // Define required options
    options_description reqOpts;
    reqOpts.add_options()
        ("N", value<int>(&nStacks)->required(), "Number of stacks")
        ("stack", value<vector<string>>(&stackFiles)->multitoken()->required(), "The input stacks (Nifti format)")
        ("g_values", value<string>(&gValFileName)->required(), ".txt file containing magnitudes of gradient first moments associated with each stack [gv_1]...[gv_N]")
        ("g_directions", value<string>(&gDirFileName)->required(), ".txt file containing components of unit vector associated with gradient first moment direction of each stack [gd_1_x gd_1_y gd_1_z]...[gd_N_x gd_N_y gd_N_z]");

    // Define positional options
    positional_options_description posOpts;
    posOpts.add("N", 1).add("stack", argc > 1 ? atoi(argv[1]) : 1).add("g_values", 1).add("g_directions", 1);

    // Define optional options
    options_description opts("Options");
    opts.add_options()
        ("target_stack", value<int>(&templateNumber), "Stack number of target for stack-to-stack registration.")
        ("dofin", value<vector<string>>(&dofinPaths)->multitoken(), "The transformations of the input stack to template in \'dof\' format used in IRTK/MIRTK. Use \'id\' for an identity transformation.")
        ("thickness", value<vector<double>>(&thickness)->multitoken(), "Give slice thickness. [Default: voxel size in z direction]")
        ("mask", value<string>(), "Binary mask to define the region of interest. [Default: whole image]")
        ("transformations", value<string>(&folder), "Use existing image-frame to volume transformations to initialize the reconstruction.")
        ("slice_transformations", value<string>(&sliceTransformationsFolder), "Use existing slice-location transformations to initialize the reconstruction.")
        ("rrintervals", value<vector<double>>(&rrLocs)->multitoken(), "R-R interval for slice-locations 1-L in input stacks. [Default: 1 s]")
        ("cardphase", value<vector<double>>(&cardPhases)->multitoken(), "Cardiac phase (0-2PI) for each image-frames 1-K. [Default: 0]")
        ("temporalpsfgauss", bool_switch(&isTemporalPSFGauss), "Use Gaussian temporal point spread function. [Default: temporal PSF = sinc()*Tukey_window()]")
        ("resolution", value<double>(&resolution), "Isotropic resolution of the volume [Default: 1.25mm]")
        ("numcardphase", value<int>(&numCardPhase), "Number of cardiac phases to reconstruct. [Default: 15]")
        ("rrinterval", value<double>(&rrInterval), "R-R interval of reconstructed cine volume. [Default: 1s]")
        ("rec_iterations", value<int>(&recIterations), "Number of super-resolution reconstruction iterations. [Default: 40]")
        ("alpha", value<double>(&alpha), "Alpha value for super-resolution loop. [Default: 3]")
        ("delta", value<double>(&delta), "Parameter to define what is an edge. [Default: 50]")
        ("lambda", value<double>(&lambda), "Smoothing parameter. [Default: 0.01]")
        ("smooth_mask", value<double>(&smoothMask), "Smooth the mask to reduce artefacts of manual segmentation. [Default: 4mm]")
        ("force_exclude", value<vector<int>>(&forceExcludedSlices)->multitoken(), "Force exclusion of image-frames with these indices.")
        ("force_exclude_sliceloc", value<vector<int>>(&forceExcludedLocs)->multitoken(), "Force exclusion of slice-locations with these indices.")
        ("force_exclude_stack", value<vector<int>>(&forceExcludedStacks)->multitoken(), "Force exclusion of stacks with these indices.")
        ("robust_statistics", bool_switch(&robustStatistics), "Switch on robust statistics. [Default: off]")
        ("no_regularisation", "Switch off adaptive regularisation.")
        ("limit_intensities", bool_switch(&limitIntensities), "Limit velocity magnitude according to the maximum/minimum values.")
        ("limit_time_window", "Threshold time window to 90%.")
        ("exclude_slices_only", bool_switch(&robustSlicesOnly), "Do not exclude individual voxels.")
        ("ref_vol", value<string>(&refVolName), "Reference volume for adjustment of spatial position of reconstructed volume.")
        ("reg_recon_to_ref", bool_switch(&regReconToRef), "Register reconstructed volume to reference volume. [Default: recon to ref]")
        ("ref_transformations", value<string>(&refTransformationsFolder), "Reference slice-to-volume transformation folder.")
        ("log_prefix", value<string>(&logID), "Prefix for the log file.")
        ("debug", bool_switch(&debug), "Debug mode - save intermediate results.")
        ("no_log", bool_switch(&noLog), "Do not redirect cout and cerr to log files.");

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

    cout << "Number of stacks : " << nStacks << endl;
    
    UniquePtr<BaseImage> tmp_image;

    // Read stacks
    for (int i = 0; i < nStacks; i++) {
        cout << "Reading stack " << stackFiles[i] << endl;
        RealImage stack;
        
        image_reader.reset(ImageReader::TryNew(stackFiles[i].c_str()));
        tmp_image.reset(image_reader->Run());
        stack = *tmp_image;
        
        stacks.push_back(move(stack));
    }

    //Read the gradient values from the text file
    ifstream gValFile(gValFileName);
    cout << " - Reading Gradient values from " << gValFileName << " : " << endl;
    if (gValFile.is_open()) {
        while (!gValFile.eof() && gValues.size() < stacks.size()) {
            double num;
            gValFile >> num;
            gValues.push_back(num);
        }
        gValFile.close();

        if (gValues.size() != stacks.size()) {
            cerr << "Not enough gradient values exist in the file!" << endl;
            return 1;
        }

        for (size_t i = 0; i < gValues.size(); i++)
            cout << gValues[i] << " ";
        cout << endl;
    } else {
        cerr << " - Unable to open file " << gValFileName << endl;
        return 1;
    }

    //Read the gradient directions from the text file
    ifstream gDirFile(gDirFileName);
    cout << " - Reading Gradient directions from " << gDirFileName << " : " << endl;
    if (gDirFile.is_open()) {
        Array<double> nums(3);
        size_t dir = 0;
        while (!gDirFile.eof() && gDirections.size() < stacks.size()) {
            gDirFile >> nums[dir];
            if (++dir == 3) {
                dir = 0;
                gDirections.push_back(nums);
            }
        }
        gDirFile.close();

        if (gDirections.size() != stacks.size()) {
            cerr << "Not enough gradient directions exist in the file!" << endl;
            return 1;
        }

        for (size_t i = 0; i < gDirections.size(); i++) {
            for (size_t j = 0; j < gDirections[i].size(); j++)
                cout << gDirections[i][j] << " ";
            cout << endl;
        }
    } else {
        cerr << " - Unable to open file " << gDirFileName << endl;
        return 1;
    }

    // Target stack
    if (vm.count("target_stack"))
        cout << "Target stack no. is " << templateNumber << " (zero-indexed stack no. " << --templateNumber << ")" << endl;

    //Read stack transformations
    if (!dofinPaths.empty()) {
        for (size_t i = 0; i < stacks.size(); i++) {
            cout << "Reading transformation " << dofinPaths[i];
            cout.flush();
            Transformation *transformation = dofinPaths[i] == "id" ? new RigidTransformation : Transformation::New(dofinPaths[i].c_str());
            unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(transformation));
            stackTransformations.push_back(*rigidTransf);
            cout << " done." << endl;
        }
        InvertStackTransformations(stackTransformations);
    }

    // Slice thickness per stack
    if (!thickness.empty()) {
        cout << "Slice thickness is ";
        for (size_t i = 0; i < stacks.size(); i++)
            cout << thickness[i] << " ";
        cout << endl;
    }

    // Stack location R-R Intervals
    if (!rrLocs.empty()) {
        cout << "R-R intervals are ";
        for (size_t i = 0; i < rrLocs.size(); i++)
            cout << i << ":" << rrLocs[i] << ", ";
        cout << "\b\b." << endl;
    } else {
        cerr << "5D cardiac velocity MRI reconstruction requires specification of R-R intervals." << endl;
        return 1;
    }

    // Cardiac phases
    if (!cardPhases.empty()) {
        cout << "Read cardiac phase for " << cardPhases.size() << " images" << endl;
    } else {
        cerr << "5D cardiac velocity MRI reconstruction requires specification of cardiac phases." << endl;
        return 1;
    }

    // R-R Interval of Reconstructed Volume
    if (vm.count("rrinterval")) {
        cout << "R-R interval of reconstructed volume is " << rrInterval << " s" << endl;
        reconstruction.SetReconstructedRRInterval(rrInterval);
    }

    // Binary mask for final volume
    if (vm.count("mask")) {
        cout << "Mask : " << vm["mask"].as<string>() << endl;
        mask = unique_ptr<RealImage>(new RealImage(vm["mask"].as<string>().c_str()));
    } else {
        cerr << "5D cardiac velocity MRI reconstruction requires mask to initilise reconstructed volume." << endl;
        return 1;
    }

    //No adaptive regularisation
    if (vm.count("no_regularisation")) {
        adaptiveRegularisation = false;
        cout << "No adaptive regularisation." << endl;
    }

    //Crop time window
    if (vm.count("limit_time_window"))
        reconstruction.LimitTimeWindow();

    // Force removal of certain slices
    if (!forceExcludedSlices.empty()) {
        cout << forceExcludedSlices.size() << " force excluded slices: ";
        for (size_t i = 0; i < forceExcludedSlices.size(); i++)
            cout << forceExcludedSlices[i] << " ";
        cout << endl;
    }

    // Force removal of certain stacks
    if (!forceExcludedStacks.empty()) {
        cout << forceExcludedStacks.size() << " force excluded stacks: ";
        for (size_t i = 0; i < forceExcludedStacks.size(); i++)
            cout << forceExcludedStacks[i] << " ";
        cout << endl;
    }

    // Force removal of certain slice-locations
    if (!forceExcludedLocs.empty()) {
        cout << forceExcludedLocs.size() << " force excluded slice-locations: ";
        for (size_t i = 0; i < forceExcludedLocs.size(); i++)
            cout << forceExcludedLocs[i] << " ";
        cout << endl;
    }

    // check that conflicting transformation folders haven't been given
    if (!folder.empty() && !sliceTransformationsFolder.empty()) {
        cerr << "Can not use both -transformations and -slice_transformations arguments." << endl;
        return 1;
    }

    //If transformations were not defined by user, set them to identity
    if (stackTransformations.empty())
        stackTransformations = Array<RigidTransformation>(stacks.size());

    //Initialise slice thickness if not given by user
    if (thickness.empty()) {
        cout << "Slice thickness is ";
        for (size_t i = 0; i < stacks.size(); i++) {
            double dx, dy, dz;
            stacks[i].GetPixelSize(&dx, &dy, &dz);
            thickness.push_back(dz);
            cout << thickness[i] << " ";
        }
        cout << "." << endl;
    }

    //Set temporal point spread function
    if (isTemporalPSFGauss)
        reconstruction.SetTemporalWeightGaussian();
    else
        reconstruction.SetTemporalWeightSinc();

    //Output volume
    Array<double> reconstructedCardPhase;
    cout << setprecision(3);
    cout << "Reconstructing " << numCardPhase << " cardiac phases: ";
    for (int i = 0; i < numCardPhase; i++) {
        reconstructedCardPhase.push_back(2 * PI * i / numCardPhase);
        cout << " " << reconstructedCardPhase[i] / PI << ",";
    }
    cout << "\b x PI." << endl;
    reconstruction.SetReconstructedCardiacPhase(reconstructedCardPhase);
    reconstruction.SetReconstructedTemporalResolution(rrInterval / numCardPhase);

    //Reference volume for adjustment of spatial position of reconstructed volume
    RealImage refVol;
    if (!refVolName.empty()) {
        cout << "Reading reference volume: " << refVolName << endl;
        refVol.Read(refVolName.c_str());
    }

    //Set debug mode
    if (debug)
        reconstruction.DebugOn();
    else
        reconstruction.DebugOff();

    //Set profiling mode
    if (profile)
        reconstruction.ProfileOn();
    else
        reconstruction.ProfileOff();
    cout << "Profiling: " << profile << endl;

    //Set adaptive regularisation option flag
    reconstruction.SetAdaptiveRegularisation(adaptiveRegularisation);

    //Set limit velocity magnitude intensities flag
    reconstruction.SetLimitIntensities(limitIntensities);

    //Set force excluded slices
    reconstruction.SetForceExcludedSlices(forceExcludedSlices);

    //Set force excluded stacks
    reconstruction.SetForceExcludedStacks(forceExcludedStacks);

    //Set force excluded stacks
    reconstruction.SetForceExcludedLocs(forceExcludedLocs);

    //Set low intensity cutoff for bias estimation
    //reconstruction.SetLowIntensityCutoff(lowIntensityCutoff);

    // Check whether the template stack can be identified
    if (templateNumber < 0) {
        cerr << "Please identify the template by assigning id transformation." << endl;
        return 1;
    }

    // Initialise Reconstructed Volume
    // Crop mask
    RealImage maskCropped = *mask;
    CropImage(maskCropped, *mask);  // TODO: TBD: use CropImage or CropImageIgnoreZ
    // Initialise reconstructed volume with isotropic resolution
    // if resolution==0 it will be determined from in-plane resolution of the image
    if (resolution <= 0)
        resolution = reconstruction.GetReconstructedResolutionFromTemplateStack(stacks[templateNumber]);
    if (debug)
        cout << "Initialising volume with isotropic voxel size " << resolution << "mm" << endl;

    // Create template 4D volume
    reconstruction.CreateTemplateCardiac4DFromStaticMask(maskCropped, resolution);

    // Set mask to reconstruction object
    reconstruction.SetMask(mask.get(), smoothMask);

    // Set verbose mode on with file
    if (!noLog)
        reconstruction.VerboseOn((logID + "log-registration.txt").c_str());
    else if (debug)
        reconstruction.VerboseOn();

    //set precision
    cout << setprecision(3);
    cerr << setprecision(3);

    RealImage average = reconstruction.CreateAverage(stacks, stackTransformations);
    if (debug)
        average.Write("average1.nii.gz");

    //Mask is transformed to the all stacks and they are cropped
    for (size_t i = 0; i < stacks.size(); i++) {
        //transform the mask
        RealImage m = reconstruction.GetMask();
        TransformMask(stacks[i], m, stackTransformations[i]);

        //Crop template stack
        CropImageIgnoreZ(stacks[i], m);
        if (debug) {
            m.Write((boost::format("mask%03i.nii.gz") % i).str().c_str());
            stacks[i].Write((boost::format("cropped%03i.nii.gz") % i).str().c_str());
        }
    }

    //Create slices and slice-dependent transformations
    reconstruction.CreateSlicesAndTransformationsCardiac4D(stacks, stackTransformations, thickness);
    // reconstruction.CreateSliceDirections(directions,bvalues);

    if (debug) {
        reconstruction.InitCorrectedSlices();
        reconstruction.InitError();
    }

    //if given, read transformations
    if (!folder.empty())
        reconstruction.ReadTransformations(folder.c_str());  // image-frame to volume registrations
    else if (!sliceTransformationsFolder.empty())     // slice-location to volume registrations
        reconstruction.ReadSliceTransformations(sliceTransformationsFolder.c_str());

    //if given, read reference transformations
    if (!refTransformationsFolder.empty())
        reconstruction.ReadRefTransformations(refTransformationsFolder.c_str());
    else
        reconstruction.InitTRE();

    //Mask all the phase slices
    reconstruction.MaskSlicesPhase();

    // Set R-R for each image
    reconstruction.SetLocRRInterval(rrLocs);

    //Set sigma for the bias field smoothing
    if (sigma > 0)
        reconstruction.SetSigma(sigma);
    else {
        //cerr<<"Please set sigma larger than zero. Current value: "<<sigma<<endl;
        //return 1;
        reconstruction.SetSigma(20);
    }

    //Set global bias correction flag
    if (globalBiasCorrection)
        reconstruction.GlobalBiasCorrectionOn();
    else
        reconstruction.GlobalBiasCorrectionOff();

    //Initialise data structures for EM
    reconstruction.InitializeEMVelocity4D();

    // Set all cardiac phases to given values
    reconstruction.SetSliceCardiacPhase(cardPhases);
    // Calculate Target Cardiac Phase in Reconstructed Volume for Slice-To-Volume Registration
    reconstruction.CalculateSliceToVolumeTargetCardiacPhase();
    // Calculate Temporal Weight for Each Slice
    reconstruction.CalculateSliceTemporalWeights();

    reconstruction.InitializeVelocityVolumes();
    reconstruction.InitializeGradientMoments(gDirections, gValues);

    reconstruction.SetSmoothingParameters(delta, lambda);
    reconstruction.SpeedupOff();

    if (robustSlicesOnly)
        reconstruction.ExcludeWholeSlicesOnly();

    reconstruction.InitializeEMValuesVelocity4D();

    // Generation of spacial coefficients
    reconstruction.CoeffInitCardiac4D();

    reconstruction.Set3DRecon();

    // Initialise velocity and phase limits
    reconstruction.InitialiseVelocityLimits();

    cout << "InitializeSliceGradients4D" << endl;
    reconstruction.InitializeSliceGradients4D();

    //-------------------------------------------------------------------------------------
    // Velocity reconstruction loop

    //Simulate slices
    reconstruction.SimulateSlicesCardiacVelocity4D();

    //Initialize robust statistics parameters
    reconstruction.InitializeRobustStatisticsVelocity4D();

    if (debug)
        reconstruction.SaveSliceInfo();

    for (int iteration = 0; iteration < recIterations; iteration++) {
        cout << "\n - Reconstruction iteration : " << iteration << "  " << endl;

        //Gradient descent step
        reconstruction.SetAlpha(alpha);
        reconstruction.SuperresolutionCardiacVelocity4D(iteration);
        reconstruction.SimulateSlicesCardiacVelocity4D();

        if (iteration + 1 < recIterations && robustStatistics) {
            reconstruction.MStepVelocity4D(iteration + 1);
            reconstruction.EStepVelocity4D();
        }

        if (debug) {
            reconstruction.SaveOuput(stacks);
            reconstruction.SaveReconstructedVelocity4D(iteration);
        }

    } // end of reconstruction loop

    if (debug)
        reconstruction.SaveSliceInfo();

    //-------------------------------------------------------------------------------------

    reconstruction.StaticMaskReconstructedVolume5D();
    reconstruction.SaveReconstructedVelocity4D(100);

    if (debug) {
        reconstruction.SaveSlices(stacks);
        reconstruction.SaveSimulatedSlices(stacks);
    }

    SVRTK_END_TIMING("all");
}
