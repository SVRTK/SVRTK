/*
 * ....
 */

// MIRTK
#include "mirtk/Common.h"
#include "mirtk/Options.h"
#include "mirtk/IOConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/ImageReader.h"

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
    cout << "Usage: mirtk extract-packages [input_stack_name] [number_of_packages] " << endl;
    cout << endl;
    cout << "Function for extracting packages (along z) from an input stack of slice based on the given order." << endl;
    cout << "If the input [number_of_packages] value < 0 the number packages will be assigned to 4 if the slice spacing in z is < 2.25 the and 1 otherwise." << endl;
    cout << "The output package files have the same name as the original image + -p${i}.nii.gz: [input_stack_name]-p${i}.nii.gz" << endl;
    cout << "It also computes the least motion/artifact corrupted package based on NCC between sequential slices and it is saved as: package-template.nii.gz file." << endl;
    cout << endl;
    cout << "\t" << endl;
    cout << "\t" << endl;
    
    exit(0);
}


double SliceCC(RealImage slice_1, RealImage slice_2)
{
    
    int slice_1_N, slice_2_N;
    double *slice_1_ptr, *slice_2_ptr;
    
    int slice_1_n, slice_2_n;
    double slice_1_m, slice_2_m;
    double slice_1_sq, slice_2_sq;
    
    double tmp_val, diff_sum;
    double average_CC, CC_slice;
    
    int min_count = 20;
    
    
    slice_1_N = slice_1.NumberOfVoxels();
    slice_2_N = slice_2.NumberOfVoxels();
    
    slice_1_ptr = slice_1.Data();
    slice_2_ptr = slice_2.Data();
    
    slice_1_n = 0;
    slice_1_m = 0;
    for (int j = 0; j < slice_1_N; j++) {
        
        if (slice_1_ptr[j]>0.1 && slice_2_ptr[j]>0.1) {
            slice_1_m = slice_1_m + slice_1_ptr[j];
            slice_1_n = slice_1_n + 1;
        }
    }
    slice_1_m = slice_1_m / slice_1_n;
    
    slice_2_n = 0;
    slice_2_m = 0;
    for (int j = 0; j < slice_2_N; j++) {
        if (slice_1_ptr[j]>0.1 && slice_2_ptr[j]>0.1) {
            slice_2_m = slice_2_m + slice_2_ptr[j];
            slice_2_n = slice_2_n + 1;
        }
    }
    slice_2_m = slice_2_m / slice_2_n;
    
    if (slice_1_n<min_count || slice_2_n<min_count) {
        
        CC_slice = -1;
        
    }
    else {
        
        diff_sum = 0;
        slice_1_sq = 0;
        slice_2_sq = 0;
        
        for (int j = 0; j < slice_1_N; j++) {
            
            if (slice_1_ptr[j]>0.1 && slice_2_ptr[j]>0.1) {
                
                diff_sum = diff_sum + ((slice_1_ptr[j] - slice_1_m)*(slice_2_ptr[j] - slice_2_m));
                slice_1_sq = slice_1_sq + pow((slice_1_ptr[j] - slice_1_m), 2);
                slice_2_sq = slice_2_sq + pow((slice_2_ptr[j] - slice_2_m), 2);
            }
        }
        
        if ((slice_1_sq * slice_2_sq)>0)
            CC_slice = diff_sum / sqrt(slice_1_sq * slice_2_sq);
        else
            CC_slice = 0;
        
    }
    
    
    return CC_slice;
    
}


// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    
    
    if (argc != 3)
        usage();
    
    
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    
    //-------------------------------------------------------------------
    
    RealImage main_stack;
    
    tmp_fname = argv[1];
    image_reader.reset(ImageReader::TryNew(tmp_fname));
    tmp_image.reset(image_reader->Run());
    
    main_stack = *tmp_image;
    
    
    argc--;
    argv++;
    
    int number_of_packages = atoi(argv[1]);
    
    
    //-------------------------------------------------------------------
    
    if (number_of_packages < 0) {
        
        double dz = main_stack.GetZSize();
        
        if (dz > 2.25)
            number_of_packages = 1;
        else
            number_of_packages = 4;
        
    }
    
    
    if (number_of_packages > 1) {
        
        Array<RealImage> packages;
        SplitImage(main_stack, number_of_packages, packages);
        
        string org_name(tmp_fname);
        std::size_t pos = org_name.find(".nii");
        std::string main_name = org_name.substr (0, pos);
        std::string end_name = org_name.substr (pos, org_name.length());
        
        for (int t=0; t<number_of_packages; t++) {
            
            std::string stack_index = "-p" + to_string(t);
            string new_name = main_name + stack_index + end_name;
            char *c_new_name = &new_name[0];
            
            RealImage stack = packages[t];
            
            stack.Write(c_new_name);
            cout << c_new_name << endl;
            
        }
        
        
        
        
        RealImage selected_package_template = packages[0];
        
        double tmp_ncc = -1;
        int best_package_id = 0;
        
        
        for (int i=0; i<packages.size(); i++) {
            
            RealImage current_stack = packages[i];
            
            for (int z = 0; z < current_stack.GetZ(); z++) {
                for (int y = 0; y < current_stack.GetY(); y++) {
                    for (int x = 0; x < current_stack.GetX(); x++) {
                        if (current_stack(x,y,z)<1) {
                            current_stack(x,y,z) = 0;
                        }
                    }
                }
            }
            
            
            double ncc = 0;
            int count = 0;
            int sh=5;
            for (int z = 0; z < current_stack.GetZ()-1; z++) {
                RealImage slice_1 = current_stack.GetRegion(sh, sh, z, current_stack.GetX()-sh, current_stack.GetY()-sh, z+1);
                RealImage slice_2 = current_stack.GetRegion(sh, sh, z+1, current_stack.GetX()-sh, current_stack.GetY()-sh, z+2);
                double slice_ncc = SliceCC(slice_1, slice_2);
                if (slice_ncc>0) {
                    ncc = ncc + slice_ncc;
                    count += 1;
                }
            }
            if (count>0) {
                ncc /= count;
            }
            
            if (ncc > tmp_ncc) {
                tmp_ncc = ncc;
                best_package_id = i;
                selected_package_template = current_stack;
            }
            
        }
        
        selected_package_template.Write("package-template.nii.gz");
        
        
    } else {
        
        string org_name(tmp_fname);
        std::size_t pos = org_name.find(".nii");
        std::string main_name = org_name.substr (0, pos);
        std::string end_name = org_name.substr (pos, org_name.length());
        
        int t = 0;
        std::string stack_index = "-p" + to_string(t);
        string new_name = main_name + stack_index + end_name;
        char *c_new_name = &new_name[0];
        main_stack.Write(c_new_name);
        
        cout << c_new_name << endl;
        
        main_stack.Write("package-template.nii.gz");
        
    }
    
    
    
    //-------------------------------------------------------------------
    
    
    return 0;
}




