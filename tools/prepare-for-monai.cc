/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2018- King's College London
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
#include "mirtk/IOConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/ImageReader.h"
#include "mirtk/Resampling.h"
#include "mirtk/ResamplingWithPadding.h"
#include "mirtk/LinearInterpolateImageFunction.hxx"
#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageTransformation.h"
#include "mirtk/MultiLevelFreeFormTransformation.h"
#include "mirtk/FreeFormTransformation.h"
#include "mirtk/LinearFreeFormTransformation3D.h"

#include "svrtk/Utility.h"

using namespace std;
using namespace mirtk;
using namespace svrtk;
using namespace svrtk::Utility;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: mirtk prepare-for-monai [folder for the output resampled files: string (e.g., res-files)]" << endl;
    cout << "\t " << "[folder for the original renamed files: string (e.g., stack-files)]" << endl;
    cout << "\t " << "[output .json file for CNN segmentation: string (e.g., segmentation_run_files.json)]" << endl;
    cout << "\t " << "[output .csv file with the original file names: string (e.g., stack_info.csv)]" << endl;
    cout << "\t " << "[output resampling grid size: int (e.g., 128)]" << endl;
    cout << "\t " << "[number of input stacks followed by the file names: [N] [stack_1] ... [stack_N]]" << endl;
    cout << endl;
    cout << "Function for preparation a set of images for 3D UNet MONAI segmentation pipeline in https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py. " << endl;
    cout << endl;
    cout << "\t" << endl;
    cout << "\t" << endl;
    
    exit(0);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    
    
    char p='"';
//
//    cout << "["  << endl;
//    cout << " ? " << p << ".nii.gz" << endl;
//
//    exit(1);
    
    char buffer[256];
    
    if (argc < 3)
        usage();
    
    
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    
    const char *out_dir_fname;
    out_dir_fname = argv[1];
    argc--;
    argv++;
    
    cout << "folder for the output resampled files : " << out_dir_fname << endl;
    
    
    
    const char *out_org_dir_fname;
    out_org_dir_fname = argv[1];
    argc--;
    argv++;
    
    cout << "folder for the original renamed files : " << out_org_dir_fname << endl;
    
    
    
    
    string str_current_main_file_path;
    string str_current_exchange_file_path;
    
    system("pwd > pwd.txt ");
    ifstream pwd_file("pwd.txt");
    
    if (pwd_file.is_open()) {
        getline(pwd_file, str_current_main_file_path);
        pwd_file.close();
    } else {
        cout << "System error: no rights to write in the current folder" << endl;
        exit(1);
    }
    
    
    int tmp_log_mk;
    
    
    string str_out_dir_fname(out_dir_fname);
    
    string rm_folder_cmd = "if [ -d " + str_out_dir_fname + " ];then rm -r " + str_out_dir_fname + " ; fi ";
    tmp_log_mk = system(rm_folder_cmd.c_str());
    
    string create_folder_cmd = "mkdir " + str_out_dir_fname + " > tmp-log.txt ";
    tmp_log_mk = system(create_folder_cmd.c_str());
    
    string str_out_org_dir_fname(out_org_dir_fname);
    string rm_org_folder_cmd = "if [ -d " + str_out_org_dir_fname + " ];then rm -r " + str_out_org_dir_fname + " ; fi ";
    
    tmp_log_mk = system(rm_org_folder_cmd.c_str());
    string create_org_folder_cmd = "mkdir " + str_out_org_dir_fname + " > tmp-log.txt ";
    tmp_log_mk = system(create_org_folder_cmd.c_str());
    
    
    
    
    
    const char *out_json_fname;
    out_json_fname = argv[1];
    argc--;
    argv++;
    
    cout << "output .json file for CNN segmentation : " << out_json_fname << endl;
    
    
    const char *out_csv_info_ref_fname;
    out_csv_info_ref_fname = argv[1];
    argc--;
    argv++;
    
    cout << "output .csv file with original names : " << out_csv_info_ref_fname << endl;
    
    
    int grid_dim = 128;
    grid_dim = atoi(argv[1]);
    argc--;
    argv++;
    
    cout << "output resampling grid size : " << grid_dim << endl;
    
    
    
    int nStacks = 1;
    nStacks = atoi(argv[1]);
    argc--;
    argv++;
    
    
    Array<RealImage> stacks;
    
    cout << endl;
    cout << "input stacks (" << nStacks << ") : " << endl;
    
    
    sprintf(buffer, out_csv_info_ref_fname);
    ofstream info_ref;
    info_ref.open( buffer );
    info_ref << "org stack name" << "," << "dynamic" << "," << "output id" << endl;
    
    
    int mc = 1000;
    
    
    int c = mc;
    
    Array<string> stack_names;
    Array<Array<string>> mask_names;
    
    for (int i=0; i<nStacks; i++) {
        
        RealImage stack;
        
        tmp_fname = argv[1];
        image_reader.reset(ImageReader::TryNew(tmp_fname));
        
        stack.Read(tmp_fname);
        
        
        for (int t=0; t<stack.GetT(); t++) {
            
            RealImage tmp_stack;
            tmp_stack = stack.GetRegion(0,0,0,t,stack.GetX(),stack.GetY(),stack.GetZ(),(t+1));
            
            stacks.push_back(tmp_stack);
            
            string tmp_string_name(argv[1]);
            stack_names.push_back(tmp_string_name);
            
            cout << " - " << i << " (" << t << ") - " << tmp_fname << " -> " << "stack-" << c << ".nii.gz" << endl;
            
            string org_file_name = str_out_org_dir_fname + "/stack-" + to_string(c) + ".nii.gz";
            tmp_stack.Write(org_file_name.c_str());
            
            c = c + 1;
            
            info_ref << tmp_fname << "," << t << "," << c << endl;
            
        }
        
        argc--;
        argv++;
        
    }
    
    info_ref.close();
    
    cout << endl;
    

    
    sprintf(buffer, out_json_fname);
    ofstream info;
    info.open( buffer );
    
    int lin_interp_mode = 1;
    
    
    
    
    info << "{" << endl;
    info << p << "running" << p << ": [ " << endl;
    
    
    
    
    for (int i=0; i<stacks.size(); i++) {
        
        
        RealImage main_stack = stacks[i];
        
        //        main_stack = main_stack.GetRegion(0,0,0,0,main_stack.GetX(),main_stack.GetY(),main_stack.GetZ(),1);
        
        int x_max = main_stack.GetX();
        int y_max = main_stack.GetY();
        int z_max = main_stack.GetZ();
        
        int dim_max = x_max;
        if (y_max > dim_max)
            dim_max = y_max;
        if (z_max > dim_max)
            dim_max = z_max;
        
        double new_res = 1.01*(main_stack.GetXSize() * dim_max)/grid_dim;
        
        
        InterpolationMode interpolation = Interpolation_Linear;
        UniquePtr<InterpolateImageFunction> interpolator;
        interpolator.reset(InterpolateImageFunction::New(interpolation));
        
        
        RealImage res_stack;
        Resampling<RealPixel> resampler(new_res,new_res,new_res);
        resampler.Input(&main_stack);
        resampler.Output(&res_stack);
        resampler.Interpolator(interpolator.get());
        resampler.Run();
        
        ImageAttributes attr = res_stack.Attributes();
        attr._x = grid_dim;
        attr._y = grid_dim;
        attr._z = grid_dim;
        
        RealImage fin_stack(attr);
        
        double source_padding = 0;
        double target_padding = -inf;
        bool dofin_invert = false;
        bool twod = false;
        
        
        {
            MultiLevelFreeFormTransformation *mffd_init = new MultiLevelFreeFormTransformation;
            ImageTransformation *imagetransformation = new ImageTransformation;
            imagetransformation->Input(&main_stack);
            imagetransformation->Transformation(mffd_init);
            imagetransformation->Output(&fin_stack);
            imagetransformation->TargetPaddingValue(target_padding);
            imagetransformation->SourcePaddingValue(source_padding);
            imagetransformation->Interpolator(interpolator.get());  // &nn);
            imagetransformation->TwoD(twod);
            imagetransformation->Invert(dofin_invert);
            imagetransformation->Run();
        }
        
        
        fin_stack.PutMinMaxAsDouble(0,500);
        GreyImage grey_fin_stack = fin_stack;
        
        
        string res_file_name = str_out_dir_fname + "/in-res-stack-" + to_string(i+mc) + ".nii.gz";
        fin_stack.Write(res_file_name.c_str());
        
        
        info << "{" << endl;
        info << p << "image" << p << ": " << p << res_file_name << p << endl;
        
        if (i == stacks.size()-1)
            info << "}" << endl;
        else
            info << "}," << endl;
        

        
        
    }
    
    
    info << "]" << endl;
    info << "}" << endl;
    
    
    cout << endl;
    
    info.close();
    
    
    
    return 0;
}
