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
    cout << "Usage: reset-dof-back [input_dof] [offset_dof] [output_dof] \n" << endl;
    exit(0);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    
    
    if (argc != 4)
        usage();
    
    
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();


    //-------------------------------------------------------------------

    const char *input_fname;
    input_fname = argv[1];
    
    Transformation *dof_input = Transformation::New(input_fname);
    RigidTransformation *input_trasnformation = dynamic_cast<RigidTransformation*> (dof_input);
    
    argc--;
    argv++;
    

    const char *offset_fname;
    offset_fname = argv[1];
    
    argc--;
    argv++;
    
    Transformation *dof_offset = Transformation::New(offset_fname);
    RigidTransformation *offset = dynamic_cast<RigidTransformation*> (dof_offset);
    

    const char *output_fname;
    output_fname = argv[1];
    
    argc--;
    argv++;

    Transformation *dof_output = Transformation::New(input_fname);
    RigidTransformation *output_trasnformation = dynamic_cast<RigidTransformation*> (dof_output);
    
    
    
    //-------------------------------------------------------------------
    

    Matrix mo = offset->GetMatrix();
    Matrix m = input_trasnformation->GetMatrix();
    m=m*mo;
    output_trasnformation->PutMatrix(m);
    
    
    output_trasnformation->Write(output_fname);

    
    //-------------------------------------------------------------------

    
    return 0;
}



