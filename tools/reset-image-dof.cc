/*
 * ....
 */  


#include "mirtk/Common.h"
#include "mirtk/Options.h" 

#include "mirtk/IOConfig.h"
#include "mirtk/GenericImage.h"
#include "mirtk/ImageReader.h"

#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"


using namespace mirtk; 
using namespace std;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: reset-image-dof [input_volume] [output_volume] [output_dof] [offset_dof] \n" << endl;
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
    
    const char *output_fname;
    output_fname = argv[1];
    
    argc--;
    argv++;
    
    const char *dof_fname;
    dof_fname = argv[1];
    
    argc--;
    argv++;

    const char *offset_fname;
    offset_fname = argv[1];
    
    argc--;
    argv++;
    
    
    //-------------------------------------------------------------------

    RigidTransformation *input_trasnformation = new RigidTransformation();
    RigidTransformation *offset = new RigidTransformation();
    
    double ox,oy,oz;
    main_stack.GetOrigin(ox,oy,oz);
    main_stack.PutOrigin(0,0,0);
    offset->PutTranslationX(ox);
    offset->PutTranslationY(oy);
    offset->PutTranslationZ(oz);
    offset->PutRotationX(0);
    offset->PutRotationY(0);
    offset->PutRotationZ(0);
    

    Matrix mo = offset->GetMatrix();
    Matrix m = input_trasnformation->GetMatrix();
    m=m*mo;
    input_trasnformation->PutMatrix(m);
    
    
    //-------------------------------------------------------------------
    
    mo.Invert();
    RigidTransformation *offset_output = new RigidTransformation();
    offset_output->PutMatrix(mo);
    
    
    //-------------------------------------------------------------------
    
    main_stack.Write(output_fname);
    input_trasnformation->Write(dof_fname);
    offset_output->Write(offset_fname);
    
    
    //-------------------------------------------------------------------
    
    
//    string org_name(tmp_fname);
//    std::size_t pos = org_name.find(".nii");
//    std::string main_name = org_name.substr (0, pos);
//
//
//    string reset_name = main_name + "-reset.nii";
//    char *c_reset_name = &reset_name[0];
//
//
//    string dof_name = main_name + "-reset-dof.dof";
//    char *c_dof_name = &dof_name[0];

    
    //-------------------------------------------------------------------

    
    return 0;
}



