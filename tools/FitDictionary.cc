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

// SVRTK Fit Dictionary code
#include "svrtk/Reconstruction.h"
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
    cout << "Usage: FitDictionary [MapName] [Dictionary] [Images] " << endl;
    cout << "  [MapName]            Name for the reconstructed map (Nifti format)" << endl;
    cout << "  [Dictionary]            The input Dictionary (.txt tab delimited format)" << endl;
    cout << "  [image_1] .. [image_N]            The input images of different TEs (Nifti format)" << endl<<endl;
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

    // Name for output volume
    string outputName;
    
    // Output Map
    RealImage OutputMap;

    // Debug boolean
    bool debug = false;
    
    // Number of input images
    int nImages = 3;

    // Array of images and image names
    Array<RealImage> images;
    vector<string> imageFiles;
    
    // Dictionary filename 
    string DictFile;
    
    // Array of T2 ranges for the dictionary
    vector<double> T2ValsStartStopEnd = {0,3000};
    
    // Input mask
    unique_ptr<RealImage> mask;
    bool masking = false;
       
    
    
    cout << "------------------------------------------------------" << endl;
    
    
    
    // -----------------------------------------------------------------------------
    // READ INPUT DATA AND OPTIONS
    // -----------------------------------------------------------------------------

    // Define required options
    options_description reqOpts;
    reqOpts.add_options()
    	("map", value<string>(&outputName)->required(), "Name for the T2 Map (Nifti format)")
        ("dictionary", value<string>(&DictFile)->required(), "The input Dictionary (.txt tab delimited format)")
        ("images", value<vector<string>>(&imageFiles)->multitoken()->required(), "The input images (Nifti format)");
        
    // Define positional options
    positional_options_description posOpts;
    posOpts.add("map", 1).add("dictionary", 1).add("images", -1);
    
    // Define optional options
    options_description opts("Options");
    opts.add_options()
        ("T2_vals", value<vector<double>>(&T2ValsStartStopEnd)->multitoken(), "Give the start and end values in the dictionaries (and optional step increment value). Default is integers in the range [0,3000]ms")
        ("mask", value<string>(), "Binary mask to define the region of interest. [Default: whole image]")
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
    
    
    // Read input images
    for (int i = 0; i < nImages; i++) {
        // Read Image of filename
        cout << "Image" << i << " : " << imageFiles[i];
            RealImage image(imageFiles[i].c_str());
        
        
        // Check if the intensity is not negative and correct if so
        double smin, smax;
        image.GetMinMax(&smin, &smax);

        // Print image info
        double dx = image.GetXSize(); double dy = image.GetYSize();
        double dz = image.GetZSize(); double dt = image.GetTSize();
        double sx = image.GetX(); double sy = image.GetY();
        double sz = image.GetZ(); double st = image.GetT();
        
        // Push individual image into images array
        images.push_back(move(image));
        
        
        // Display image info
        cout << "  ;  size : " << sx << " - " << sy << " - " << sz << " - " << st << "  ;";
        cout << "  voxel : " << dx << " - " << dy << " - " << dz << " - " << dt << "  ;";
        cout << "  range : [" << smin << "; " << smax << "]" << endl<<endl;
        }
        
    // Binary mask for reconstruction / final volume
    if (vm.count("mask")) {
        cout << "Mask : " << vm["mask"].as<string>() << endl;
        mask = unique_ptr<RealImage>(new RealImage(vm["mask"].as<string>().c_str()));
        masking = true;
    }

    
    Dictionary T2Dictionary(DictFile.c_str());
    T2Dictionary.SetImages(images);
    if (T2ValsStartStopEnd.size() == 2)
    	T2Dictionary.SetT2Vals(T2ValsStartStopEnd[0],T2ValsStartStopEnd[1]);
    if (T2ValsStartStopEnd.size() == 3)
    	T2Dictionary.SetT2Vals(T2ValsStartStopEnd[0],T2ValsStartStopEnd[1],T2ValsStartStopEnd[3]);
    if (masking) {
        cout << "We got here" << endl;
        OutputMap = T2Dictionary.FitDictionaryToMap(images, *mask);
    }
    else
        OutputMap = T2Dictionary.FitDictionaryToMap();

    OutputMap.Write(outputName.c_str());
    
    
    	    
    return 0;
}
