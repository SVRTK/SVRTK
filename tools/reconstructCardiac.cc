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
#include "svrtk/ReconstructionCardiac4D.h"
#define SVRTK_TOOL
#include "svrtk/Profiling.h"

using namespace std;
using namespace mirtk;
using namespace svrtk;
using namespace boost::program_options;

// Application to perform reconstruction of volumetric cardiac cine MRI from thick-slice dynamic 2D MRI

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void PrintUsage(const options_description& opts) {
    // Print positional arguments
    cout << "Usage: reconstructCardiac [reconstructed] [N] [stack_1] .. [stack_N] <options>\n" << endl;
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
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();

    // Initialise profiling
    SVRTK_START_TIMING();

    //declare variables for input
    /// Name for output volume
    string outputName;
    /// Reference volume
    string refVolName;
    RigidTransformation transformationReconToRef;
    bool regReconToRef = false;

    /// Slice stacks
    Array<RealImage> stacks;
    Array<string> stackFiles;

    /// Stack transformation
    Array<RigidTransformation> stackTransformations;
    /// Stack heart masks
    Array<RealImage> masks;

    /// Stack thickness
    Array<double> thickness;
    ///number of stacks
    int nStacks;
    // Location R-R Intervals;
    Array<double> rrLocs;
    // Slice R-R Intervals
    Array<double> rr;
    // Slice cardiac phases
    Array<double> cardPhases;
    // Mean Displacement
    Array<double> meanDisplacement;
    Array<double> meanWeightedDisplacement;
    // Mean Target Registration Error
    Array<double> meanTRE;
    // Numbers of NMI bins for registration
    int nmiBins = 16;

    // Default values
    int templateNumber = 0;
    unique_ptr<RealImage> mask;
    int iterations = 4;
    bool debug = false;
    bool profile = false;
    bool outputTransformations = false;
    double sigma = 20;
    double motionSigma = 0;
    double resolution = 0.75;
    int numCardPhase = 15;
    double rrInterval = 1;
    bool isTemporalPSFGauss = false;
    double lambda = 0.02;
    double delta = 150;
    int levels = 3;
    double lastIterLambda = 0.01;
    int recIterations;
    int recIterationsFirst = 10;
    int recIterationsLast = -1;
    double averageValue = 700;
    double smoothMask = 4;

    //folder for slice-location registrations, if given
    string sliceTransformationsFolder;

    //folder for slice-to-volume registrations, if given
    string folder;

    //folder for reference slice-to-volume registrations, if given
    string refTransformationsFolder;

    //flag to swich the intensity matching on and off
    bool stackIntensityMatching = true;
    bool intensityMatching = true;

    bool rescaleStacks = false;
    bool stackRegistration = false;

    //flag to swich the robust statistics on and off
    bool robustStatistics = true;
    bool robustSlicesOnly = false;

    string infoFilename = "info.tsv";
    string logID;
    bool noLog = false;
    bool remoteFlag = false;

    //forced exclusion
    Array<int> forceExcludedSlices;
    Array<int> forceExcludedStacks;
    Array<int> forceExcludedLocs;

    //Create reconstruction object
    ReconstructionCardiac4D reconstruction;

    //Entropy of reconstructed volume
    Array<double> e;
    Array<Array<double>> entropy;

    // Paths of 'dofin' arguments
    vector<string> dofinPaths;

    // Paths of masks
    vector<string> maskFiles;

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
        ("stack_registration", bool_switch(&stackRegistration), "Perform stack-to-stack registration.")
        ("target_stack", value<int>(&templateNumber), "Stack number of target for stack-to-stack registration.")
        ("dofin", value<vector<string>>(&dofinPaths)->multitoken(), "The transformations of the input stack to template in \'dof\' format used in IRTK. Only rough alignment with correct orientation and some overlap is needed. Use \'id\' for an identity transformation for at leastone stack. The first stack with \'id\' transformation will be resampled as template.")
        ("thickness", value<vector<double>>(&thickness)->multitoken(), "Give slice thickness. [Default: voxel size in z direction]")
        ("mask", value<string>(), "Binary mask to define the region of interest. [Default: whole image]")
        ("transformations", value<string>(&folder), "Use existing image-frame to volume transformations to initialize the reconstruction.")
        ("slice_transformations", value<string>(&sliceTransformationsFolder), "Use existing slice-location transformations to initialize the reconstruction.")
        ("motion_sigma", value<double>(&motionSigma), "Stdev for smoothing transformations. [Default: 0s, no smoothing]")
        ("rrintervals", value<vector<double>>(&rrLocs)->multitoken(), "R-R interval for slice-locations 1-L in input stacks. [Default: 1 s]")
        ("cardphase", value<vector<double>>(&cardPhases)->multitoken(), "Cardiac phase (0-2PI) for each image-frames 1-K. [Default: 0]")
        ("temporalpsfgauss", bool_switch(&isTemporalPSFGauss), "Use Gaussian temporal point spread function. [Default: temporal PSF = sinc()*Tukey_window()]")
        ("resolution", value<double>(&resolution), "Isotropic resolution of the volume. [Default: 0.75mm]")
        ("numcardphase", value<int>(&numCardPhase), "Number of cardiac phases to reconstruct. [Default: 15]")
        ("rrinterval", value<double>(&rrInterval), "R-R interval of reconstructed cine volume. [Default: 1s]")
        ("iterations", value<int>(&iterations), "Number of registration-reconstruction iterations. [Default: 4]")
        ("rec_iterations", value<int>(&recIterationsFirst), "Number of super-resolution reconstruction iterations. [Default: 10]")
        ("rec_iterations_last", value<int>(&recIterationsLast), "Number of super-resolution reconstruction iterations for last iteration. [Default: 2 x rec_iterations]")
        ("sigma", value<double>(&sigma), "Stdev for bias field. [Default: 12mm]")
        ("average", value<double>(&averageValue), "Average intensity value for stacks. [Default: 700]")
        ("delta", value<double>(&delta), "Parameter to define what is an edge. [Default: 150]")
        ("lambda", value<double>(&lambda), "Smoothing parameter. [Default: 0.02]")
        ("lastIter", value<double>(&lastIterLambda), "Smoothing parameter for last iteration. [Default: 0.01]")
        ("multires", value<int>(&levels), "Multiresolution smoothing with given number of levels. [Default: 3]")
        ("smooth_mask", value<double>(&smoothMask), "Smooth the mask to reduce artefacts of manual segmentation. [Default: 4mm]")
        ("force_exclude", value<vector<int>>(&forceExcludedSlices)->multitoken(), "Force exclusion of image-frames with these indices.")
        ("force_exclude_sliceloc", value<vector<int>>(&forceExcludedLocs)->multitoken(), "Force exclusion of slice-locations with these indices.")
        ("force_exclude_stack", value<vector<int>>(&forceExcludedStacks)->multitoken(), "Force exclusion of stacks with these indices.")
        ("no_stack_intensity_matching", "Switch off stack intensity matching.")
        ("no_intensity_matching", "Switch off intensity matching.")
        ("no_robust_statistics", "Switch off robust statistics.")
        ("exclude_slices_only", bool_switch(&robustSlicesOnly), "Do not exclude individual voxels.")
        ("ref_vol", value<string>(&refVolName), "Reference volume for adjustment of spatial position of reconstructed volume.")
        ("reg_recon_to_ref", bool_switch(&regReconToRef), "Register reconstructed volume to reference volume. [Default: recon to ref]")
        ("ref_transformations", value<string>(&refTransformationsFolder), "Reference slice-to-volume transformation folder.")
        ("masks", value<vector<string>>(&maskFiles)->multitoken(), "Binary masks for all stacks.")
        ("rescale_stacks", bool_switch(&rescaleStacks), "Rescale stacks to avoid nan pixel errors. [Default: false]")
        ("nmi_bins", value<int>(&nmiBins), "Number of NMI bins for registration. [Default: 16]")
        ("log_prefix", value<string>(&logID), "Prefix for the log file.")
        ("info", value<string>(&infoFilename), "File name for slice information in tab-separated columns.")
        ("debug", bool_switch(&debug), "Debug mode - save intermediate results.")
        ("profile", bool_switch(&profile), "Profile - output profiling timings (also on in debug mode)")
        ("output_transformations", bool_switch(&outputTransformations), "Save transformation to file")
        ("remote", bool_switch(&remoteFlag), "Run SVR registration as remote functions in case of memory issues. [Default: false]")
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
        if (!maskFiles.empty() && maskFiles.size() < nStacks)
            throw error("Count of masks should equal to stack count!");
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

    UniquePtr<BaseImage> tmp_image;
    
    // Read stacks
    for (int i = 0; i < nStacks; i++) {
        if (debug) cout << "Reading stack " << stackFiles[i] << endl;
        RealImage stack;
        
        image_reader.reset(ImageReader::TryNew(stackFiles[i].c_str()));
        tmp_image.reset(image_reader->Run());
        stack = *tmp_image;
        
        stacks.push_back(move(stack));
    }

    // Target stack
    if (vm.count("target_stack") && debug)
        cout << "Target stack no. is " << templateNumber << " (zero-indexed stack no. " << --templateNumber << ")" << endl;

    //Read stack transformations
    if (!dofinPaths.empty()) {
        for (size_t i = 0; i < stacks.size(); i++) {
            if (debug) cout << "Reading transformation " << dofinPaths[i];
            cout.flush();
            Transformation *transformation = dofinPaths[i] == "id" ? new RigidTransformation : Transformation::New(dofinPaths[i].c_str());
            unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(transformation));
            stackTransformations.push_back(*rigidTransf);
            if (debug) cout << " done." << endl;
        }
        InvertStackTransformations(stackTransformations);
    }

    //Stack registration
    if (stackRegistration)
        cout << "Stack-stack registrations, if possible." << endl;

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
    }

    // Cardiac phases
    if (!cardPhases.empty())
        cout << "Read cardiac phase for " << cardPhases.size() << " images" << endl;

    // R-R Interval of Reconstructed Volume
    if (vm.count("rrinterval")) {
        cout << "R-R interval of reconstructed volume is " << rrInterval << " s." << endl;
        reconstruction.SetReconstructedRRInterval(rrInterval);
    }

    // Binary mask for final volume
    if (vm.count("mask")) {
        cout << "Mask : " << vm["mask"].as<string>() << endl;
        mask = unique_ptr<RealImage>(new RealImage(vm["mask"].as<string>().c_str()));
    } else {
        cerr << "Reconstruction of volumetric cardiac cine MRI from thick-slice dynamic 2D MRI requires mask to initilise reconstructed volume." << endl;
        return 1;
    }

    // Binary masks for all stacks
    if (!maskFiles.empty()) {
        if (debug) cout << "Reading stack masks ... ";
        for (size_t i = 0; i < stacks.size(); i++) {
            unique_ptr<RealImage> binaryMask(new RealImage(maskFiles[i].c_str()));
            masks.push_back(move(*binaryMask));
        }
        reconstruction.SetMaskedStacks();
        if (debug) cout << "done." << endl;
    }

    // Switch off stack intensity matching
    if (vm.count("no_intensity_matching")) {
        stackIntensityMatching = false;
        cout << "No stack intensity matching." << endl;
    }

    // Switch off intensity matching
    if (vm.count("no_intensity_matching"))
        intensityMatching = false;

    // Switch off robust statistics
    if (vm.count("no_robust_statistics"))
        robustStatistics = false;

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

    // -----------------------------------------------------------------------------

    // Read path to MIRTK executables for remote registration
    string strMirtkPath(argv[0]);
    strMirtkPath = strMirtkPath.substr(0, strMirtkPath.find_last_of("/"));
    const string strCurrentMainFilePath = boost::filesystem::current_path().string();

    // Create an empty file exchange directory
    const string strCurrentExchangeFilePath = strCurrentMainFilePath + "/tmp-file-exchange";
    boost::filesystem::remove_all(strCurrentExchangeFilePath.c_str());
    boost::filesystem::create_directory(strCurrentExchangeFilePath.c_str());

    //---------------------------------------------------------------------------------------------

    // check that conflicting transformation folders haven't been given
    if (!folder.empty() && !sliceTransformationsFolder.empty()) {
        cerr << "Can not use both -transformations and -slice_transformations arguments." << endl;
        return 1;
    }

    if (rescaleStacks) {
        for (size_t i = 0; i < stacks.size(); i++)
            Rescale(stacks[i], 1000);
    }

    //If transformations were not defined by user, set them to identity
    if (stackTransformations.empty())
        stackTransformations = Array<RigidTransformation>(stacks.size());

    //Initialise 2*slice thickness if not given by user
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

    Array<RealImage> maskedStacks = stacks;

    if (!masks.empty()) {
        const RigidTransformation rigidTransf;
        for (size_t i = 0; i < masks.size(); i++) {
            RealImage stackMask = masks[i];
            TransformMask(stacks[i], stackMask, rigidTransf);

            // ConnectivityType i_connectivity = CONNECTIVITY_26;
            // Dilate<RealPixel>(&stackMask, 7, i_connectivity);

            maskedStacks[i] *= stackMask;

            maskedStacks[i].Write((boost::format("masked-%1%.nii.gz") % i).str().c_str());
        }
        // reconstruction.CenterStacks(masks, stackTransformations, templateNumber);
    }

    //Set temporal point spread function
    if (isTemporalPSFGauss)
        reconstruction.SetTemporalWeightGaussian();
    else
        reconstruction.SetTemporalWeightSinc();

    //Output volume
    RealImage reconstructed;
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
    bool haveRefVol = false;
    if (!refVolName.empty()) {
        cout << "Reading reference volume: " << refVolName << endl;
        refVol.Read(refVolName.c_str());
        haveRefVol = true;
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

    //Set NMI bins for registration
    reconstruction.SetNMIBins(nmiBins);

    //Set force excluded slices
    reconstruction.SetForceExcludedSlices(forceExcludedSlices);

    //Set force excluded stacks
    reconstruction.SetForceExcludedStacks(forceExcludedStacks);

    //Set force excluded stacks
    reconstruction.SetForceExcludedLocs(forceExcludedLocs);

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

    //volumetric registration if input stacks are single time frame
    if (stackRegistration) {
        const ImageAttributes& attr = stacks[templateNumber].Attributes();
        if (attr._t > 1)
            reconstruction.GetVerboseLog() << "Skipping stack-to-stack registration; target stack has more than one time frame." << endl;
        else {
            reconstruction.StackRegistrations(maskedStacks, stackTransformations, templateNumber, NULL);
            if (debug || outputTransformations) {
                InvertStackTransformations(stackTransformations);
                for (size_t i = 0; i < stacks.size(); i++)
                    stackTransformations[i].Write((boost::format("stack-transformation%03i.dof") % i).str().c_str());
                InvertStackTransformations(stackTransformations);
            }
        }
    }

    RealImage average = reconstruction.CreateAverage(stacks, stackTransformations);
    if (debug)
        average.Write("average1.nii.gz");

    //Mask is transformed to the all stacks and they are cropped
    for (size_t i = 0; i < stacks.size(); i++) {
        //transform the mask
        RealImage m = reconstruction.GetMask();
        TransformMask(stacks[i], m, stackTransformations[i]);
        //Crop template stack

        // ConnectivityType connectivity2 = CONNECTIVITY_26;
        // Dilate<RealPixel>(&m, 5, connectivity2);

        CropImageIgnoreZ(stacks[i], m);
        if (debug) {
            m.Write((boost::format("mask%03i.nii.gz") % i).str().c_str());
            stacks[i].Write((boost::format("cropped%03i.nii.gz") % i).str().c_str());
        }
    }

    //Rescale intensities of the stacks to have the same average
    reconstruction.MatchStackIntensitiesWithMasking(stacks, stackTransformations, averageValue, !stackIntensityMatching);
    if (debug) {
        for (size_t i = 0; i < stacks.size(); i++)
            stacks[i].Write((boost::format("rescaledstack%03i.nii.gz") % i).str().c_str());
    }
    average = reconstruction.CreateAverage(stacks, stackTransformations);
    if (debug)
        average.Write("average2.nii.gz");

    //Create slices and slice-dependent transformations
    reconstruction.CreateSlicesAndTransformationsCardiac4D(stacks, stackTransformations, thickness);
    if (debug) {
        reconstruction.InitCorrectedSlices();
        reconstruction.InitError();
    }

    //if given, read transformations
    if (!folder.empty())
        reconstruction.ReadTransformations(folder.c_str());  // image-frame to volume registrations
    else if (!sliceTransformationsFolder.empty())
        reconstruction.ReadSliceTransformations(sliceTransformationsFolder.c_str()); // slice-location to volume registrations

    //if given, read reference transformations
    const bool haveRefTransformations = !refTransformationsFolder.empty();
    if (haveRefTransformations)
        reconstruction.ReadRefTransformations(refTransformationsFolder.c_str());
    else
        reconstruction.InitTRE();

    // // Mask all the slices
    // reconstruction.MaskSlices();

    // Set R-R for each image
    if (rrLocs.empty()) {
        reconstruction.SetSliceRRInterval(rrInterval);
        if (debug)
            cout << "No R-R intervals specified. All R-R intervals set to " << rrInterval << " s." << endl;
    } else
        reconstruction.SetLocRRInterval(rrLocs);

    //Set sigma for the bias field smoothing
    if (sigma > 0)
        reconstruction.SetSigma(sigma);
    else {
        //cerr<<"Please set sigma larger than zero. Current value: "<<sigma<<endl;
        //exit(1);
        reconstruction.SetSigma(20);
    }

    //Set global bias correction flag
    reconstruction.GlobalBiasCorrectionOff();

    //Initialise data structures for EM
    reconstruction.InitializeEM();

    // Calculate Cardiac Phase of Each Slice
    if (cardPhases.empty()) {  // no cardiac phases specified
        if (numCardPhase != 1) {    // reconstructing cine volume
            const ImageAttributes& attr = stacks[templateNumber].Attributes();
            if (attr._t > 1) {
                cerr << "Cardiac 4D reconstruction requires cardiac phase for each slice." << endl;
                return 1;
            }
        } else {    // reconstructing single cardiac phase volume
            reconstruction.SetSliceCardiacPhase();    // set all cardiac phases to zero
        }
    } else {
        reconstruction.SetSliceCardiacPhase(cardPhases);   // set all cardiac phases to given values
    }
    // Calculate Target Cardiac Phase in Reconstructed Volume for Slice-To-Volume Registration
    reconstruction.CalculateSliceToVolumeTargetCardiacPhase();
    // Calculate Temporal Weight for Each Slice
    reconstruction.CalculateSliceTemporalWeights();

    // File for evaluation output
    ofstream fileEv((logID + "log-evaluation.txt").c_str());

    //interleaved registration-reconstruction iterations
    cout << "Will run " << iterations << " main iterations:" << endl;

    for (int iter = 0; iter < iterations; iter++) {
        cout << "Iteration " << iter+1 << "... " << endl;

        //perform slice-to-volume registrations
        if (iter > 0) {
            reconstruction.GetVerboseLog() << "\n\nIteration " << iter << ":\n" << endl;
            if (remoteFlag) {
                reconstruction.RemoteSliceToVolumeRegistrationCardiac4D(iter, strMirtkPath, strCurrentExchangeFilePath);
            } else {
                reconstruction.SliceToVolumeRegistrationCardiac4D();
            }
            reconstruction.GetVerboseLog() << endl;

            // if (iter > 0 && debug)
            //     reconstruction.SaveRegistrationStep(stacks, iter);

            // process transformations
            if (motionSigma > 0)
                reconstruction.SmoothTransformations(motionSigma);
        }  // if ( iter > 0 )

        reconstruction.GetVerboseLog() << "\n\nIteration " << iter << ":\n" << endl;

        //Set smoothing parameters
        //amount of smoothing (given by lambda) is decreased with improving alignment
        //delta (to determine edges) stays constant throughout
        if (iter == iterations - 1)
            reconstruction.SetSmoothingParameters(delta, lastIterLambda);
        else {
            double l = lambda;
            for (int i = 0; i < levels; i++) {
                if (iter == iterations * (levels - i - 1) / levels)
                    reconstruction.SetSmoothingParameters(delta, l);
                l *= 2;
            }
        }

        //Use faster reconstruction during iterations and slower for final reconstruction
        if (iter < iterations - 1)
            reconstruction.SpeedupOn();
        else
            reconstruction.SpeedupOff();

        //Exclude whole slices only
        if (robustSlicesOnly)
            reconstruction.ExcludeWholeSlicesOnly();

        //Initialise values of weights, scales and bias fields
        reconstruction.InitializeEMValues();

        //Calculate matrix of transformation between voxels of slices and volume
        reconstruction.CoeffInitCardiac4D();

        //Initialize reconstructed image with Gaussian weighted reconstruction
        reconstruction.GaussianReconstructionCardiac4D();

        // Calculate Entropy
        e.clear();
        e.push_back(reconstruction.CalculateEntropy());

        // Save Initialised Volume to File
        if (debug) {
            reconstruction.GetReconstructedCardiac4D().Write((boost::format("init_mc%02i.nii.gz") % iter).str().c_str());
            reconstruction.GetVolumeWeights().Write((boost::format("volumeweights_mc%02i.nii.gz") % iter).str().c_str());
        }

        //Simulate slices (needs to be done after Gaussian reconstruction)
        reconstruction.SimulateSlicesCardiac4D();

        //Save intermediate simulated slices
        if (debug) {
            reconstruction.SaveSimulatedSlices(stacks, iter, 0);
            reconstruction.SaveSimulatedWeights(stacks, iter, 0);
            reconstruction.CalculateError();
            reconstruction.SaveError(stacks, iter, 0);
        }

        //Initialize robust statistics parameters
        reconstruction.InitializeRobustStatistics();

        //EStep
        if (robustStatistics)
            reconstruction.EStep();

        if (debug)
            reconstruction.SaveWeights(stacks, iter, 0);

        //number of reconstruction iterations
        if (iter == iterations - 1) {
            if (recIterationsLast < 0)
                recIterationsLast = 2 * recIterationsFirst;
            recIterations = recIterationsLast;
        } else {
            recIterations = recIterationsFirst;
        }
        if (debug)
            reconstruction.GetVerboseLog() << "rec_iterations = " << recIterations << endl;

        //reconstruction iterations
        for (int i = 0; i < recIterations; i++) {
            reconstruction.GetVerboseLog() << "\n  Reconstruction iteration " << i << endl;

            if (intensityMatching) {
                //calculate bias fields
                if (sigma > 0)
                    reconstruction.Bias();
                //calculate scales
                reconstruction.Scale();
            }

            //Update reconstructed volume
            reconstruction.SuperresolutionCardiac4D(i);

            if (intensityMatching && sigma > 0)
                reconstruction.NormaliseBiasCardiac4D(iter, i);

            //Save intermediate reconstructed volume
            if (debug) {
                reconstructed = reconstruction.GetReconstructedCardiac4D();
                StaticMaskVolume4D(reconstructed, reconstruction.GetMask(), -1);
                reconstructed.Write((boost::format("super_mc%02isr%02i.nii.gz") % iter % i).str().c_str());
            }

            // Calculate Entropy
            e.push_back(reconstruction.CalculateEntropy());

            // Simulate slices (needs to be done
            // after the update of the reconstructed volume)
            reconstruction.SimulateSlicesCardiac4D();

            if (i + 1 < recIterations) {
                //Save intermediate simulated slices
                if (debug) {
                    if (intensityMatching) {
                        reconstruction.CalculateCorrectedSlices();
                        reconstruction.SaveCorrectedSlices(stacks, iter, i + 1);
                        if (sigma > 0)
                            reconstruction.SaveBiasFields(stacks, iter, i + 1);
                    }
                    reconstruction.SaveSimulatedSlices(stacks, iter, i + 1);
                    reconstruction.CalculateError();
                    reconstruction.SaveError(stacks, iter, i + 1);
                }

                if (robustStatistics) {
                    reconstruction.MStep(i + 1);
                    reconstruction.EStep();
                }

                //Save intermediate weights
                if (debug)
                    reconstruction.SaveWeights(stacks, iter, i + 1);
            }

        }//end of reconstruction iterations

        //Mask reconstructed image to ROI given by the mask
        reconstruction.StaticMaskReconstructedVolume4D();

        //Save reconstructed image
        reconstructed = reconstruction.GetReconstructedCardiac4D();
        StaticMaskVolume4D(reconstructed, reconstruction.GetMask(), -1);
        if (debug) reconstructed.Write((boost::format("reconstructed_mc%02i.nii.gz") % iter).str().c_str());

        //Save Calculated Entropy
        entropy.push_back(e);

        //Evaluate - write number of included/excluded/outside/zero slices in each iteration in the file
        reconstruction.Evaluate(iter, fileEv);
        fileEv << endl;

        // Calculate Displacements
        if (haveRefVol) {
            // Get Current Reconstructed Volume
            reconstructed = reconstruction.GetReconstructedCardiac4D();
            StaticMaskVolume4D(reconstructed, reconstruction.GetMask(), -1);

            // Invert to get recon to ref transformation
            if (regReconToRef) {
                reconstruction.VolumeToVolumeRegistration(refVol, reconstructed, transformationReconToRef);
                // Invert to get recon to ref transformation
                transformationReconToRef.PutMatrix(transformationReconToRef.GetMatrix().Inverse());
            } else {
                reconstruction.VolumeToVolumeRegistration(reconstructed, refVol, transformationReconToRef);
            }

            // Save Transformation
            if (outputTransformations) transformationReconToRef.Write((boost::format("recon_to_ref_mc%02i.dof") % iter).str().c_str());

            // Calculate Displacements Relative to Alignment
            meanDisplacement.push_back(reconstruction.CalculateDisplacement(transformationReconToRef));
            meanWeightedDisplacement.push_back(reconstruction.CalculateWeightedDisplacement(transformationReconToRef));

            // Calculate TRE Relative to Alignment
            if (haveRefTransformations)
                meanTRE.push_back(reconstruction.CalculateTRE(transformationReconToRef));
        } else {
            // Calculate Displacement
            meanDisplacement.push_back(reconstruction.CalculateDisplacement());
            meanWeightedDisplacement.push_back(reconstruction.CalculateWeightedDisplacement());

            // Calculate TRE
            if (haveRefTransformations)
                meanTRE.push_back(reconstruction.CalculateTRE());
        }

        if (debug) {
            // Display Displacements and TRE
            cout << "Mean Displacement (iter " << iter << ") = " << meanDisplacement[iter] << " mm" << endl;
            cout << "Mean Weighted Displacement (iter " << iter << ") = " << meanWeightedDisplacement[iter] << " mm" << endl;
            if (haveRefTransformations)
                cout << "Mean TRE (iter " << iter << ") = " << meanTRE[iter] << " mm" << endl;

            // Save Info for Iteration
            cout << "SaveSliceInfoCardiac4D" << endl;
            reconstruction.SaveSliceInfoCardiac4D((boost::format("info_mc%02i.tsv") % iter).str().c_str(), stackFiles);
        }
    }// end of interleaved registration-reconstruction iterations

    if (debug) {
        //Display Entropy Values
        cout << setprecision(9);
        cout << "Calculated Entropy:" << endl;
        for (size_t iterMC = 0; iterMC < entropy.size(); iterMC++) {
            cout << iterMC << ":";
            for (size_t iterSR = 0; iterSR < entropy[iterMC].size(); iterSR++)
                cout << " " << entropy[iterMC][iterSR];
            cout << endl;
        }
        cout << setprecision(3);

        //Display Mean Displacements and TRE
        cout << "Mean Displacement:";
        for (size_t iterMC = 0; iterMC < meanDisplacement.size(); iterMC++)
            cout << " " << meanDisplacement[iterMC];
        cout << " mm" << endl;
        cout << "Mean Weighted Displacement:";
        for (size_t iterMC = 0; iterMC < meanWeightedDisplacement.size(); iterMC++)
            cout << " " << meanWeightedDisplacement[iterMC];
        cout << " mm" << endl;
        if (haveRefTransformations) {
            cout << "Mean TRE:";
            for (size_t iterMC = 0; iterMC < meanTRE.size(); iterMC++)
                cout << " " << meanTRE[iterMC];
            cout << " mm" << endl;
        }
    }

    // Remove the file exchange directory
    boost::filesystem::remove_all(strCurrentExchangeFilePath.c_str());

    //save final result
    if (debug)
        cout << "RestoreSliceIntensities" << endl;
    reconstruction.RestoreSliceIntensities();

    if (debug)
        cout << "ScaleVolumeCardiac4D" << endl;
    reconstruction.ScaleVolumeCardiac4D();

    if (debug)
        cout << "Saving Reconstructed Volume" << endl;
    reconstruction.GetReconstructedCardiac4D().Write(outputName.c_str());

    if (debug) {
        cout << "SaveSlices" << endl;
        reconstruction.SaveSlices(stacks);
    }

    if (debug || outputTransformations) {
        cout << "SaveTransformations" << endl;
        reconstruction.SaveTransformations();
    }

    //save final transformation to reference volume
    if (haveRefVol && outputTransformations)
        transformationReconToRef.Write("recon_to_ref.dof");

    if (!infoFilename.empty()) {
        if (debug)
            cout << "SaveSliceInfoCardiac4D" << endl;
        reconstruction.SaveSliceInfoCardiac4D(infoFilename.c_str(), stackFiles);
    }

    if (debug) {
        cout << "SaveWeights" << endl;
        reconstruction.SaveWeights(stacks);
        cout << "SaveBiasFields" << endl;
        reconstruction.SaveBiasFields(stacks);
        cout << "SaveSimulatedSlices" << endl;
        reconstruction.SaveSimulatedSlices(stacks);
        cout << "ReconstructionCardiac complete." << endl;
    }

    SVRTK_END_TIMING("all");
}
