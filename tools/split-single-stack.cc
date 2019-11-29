/*
 * ....
 */  


#include "mirtk/Common.h"
#include "mirtk/Options.h" 

#include "mirtk/IOConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/ImageReader.h"


using namespace mirtk; 
using namespace std;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: split-single-stack [stack_name] \n" << endl;
    exit(0);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    
    
    if (argc != 2)
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


    if (main_stack.GetT() > 1) {
        
        string org_name(tmp_fname);
        std::size_t pos = org_name.find(".nii");
        std::string main_name = org_name.substr (0, pos);
        std::string end_name = org_name.substr (pos, org_name.length());

        for (int t=0; t<main_stack.GetT(); t++) {
        
            std::string stack_index = "-" + to_string(t);
            string new_name = main_name + stack_index + end_name;
            char *c_new_name = &new_name[0];
            
            RealImage stack = main_stack.GetRegion(0,0,0,t,main_stack.GetX(),main_stack.GetY(),main_stack.GetZ(),(t+1));

            stack.Write(c_new_name);
            cout << c_new_name << endl;
            
        }

    }
    

    //-------------------------------------------------------------------

    
    return 0;
}



