/*
 * ....
 */  

// MIRTK
#include "mirtk/Common.h"
#include "mirtk/Options.h" 
#include "mirtk/IOConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/ImageReader.h"

// SVRTK
#include "svrtk/ReconstructionFFD.h"

using namespace std;
using namespace mirtk;
using namespace svrtk;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: .... \n" << endl;
    exit(0);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------




int main(int argc, char **argv)
{
    
    
    if (argc != 5)
        usage();
    
    
    ReconstructionFFD reconstruction;
    
    const char *tmp_fname;
    const char *file_end_fname;
    const char *out_folder_fname;
    
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();


    //-------------------------------------------------------------------
   
    RealImage main_stack, main_mask;

    tmp_fname = argv[1];
    image_reader.reset(ImageReader::TryNew(tmp_fname));
    tmp_image.reset(image_reader->Run());
    main_stack = *tmp_image;
    
    string org_name_string(tmp_fname);
    
    argc--;
    argv++;

    
    tmp_fname = argv[1];
    main_mask.Read(tmp_fname);
    
    RigidTransformation *rigidTransf_mask = new RigidTransformation;
    reconstruction.TransformMask(main_stack, main_mask, *rigidTransf_mask);
    
    argc--;
    argv++;
    
    
    file_end_fname = argv[1];
    
    string file_end_fname_string(file_end_fname);
    
    argc--;
    argv++;
    
    
    out_folder_fname = argv[1];
    
    string out_folder_fname_string(out_folder_fname);
    
    
    //-------------------------------------------------------------------
    
    
    std::size_t pos = org_name_string.find(".nii");
    std::string main_name = org_name_string.substr (0, pos);
    std::string end_name = org_name_string.substr (pos, org_name_string.length());
    
    std::size_t pos2 = main_name.find_last_of("/");
    std::string begin_name = main_name.substr (pos2+1, main_name.length());
    
    string new_name = out_folder_fname_string + "/" + begin_name + file_end_fname_string + end_name;
    char *c_new_name = &new_name[0];
    
    main_mask.Write(c_new_name);
    
    cout << c_new_name << endl;
    

    //-------------------------------------------------------------------

    
    return 0;
}



