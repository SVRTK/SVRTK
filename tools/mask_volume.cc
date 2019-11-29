/*
 *
 */

#include "mirtk/Common.h"
#include "mirtk/Options.h"

#include "mirtk/NumericsConfig.h"
#include "mirtk/IOConfig.h"
#include "mirtk/TransformationConfig.h"
#include "mirtk/RegistrationConfig.h"

#include "mirtk/GenericImage.h" 
#include "mirtk/GenericRegistrationFilter.h"

#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"

#include "mirtk/ReconstructionFFD.h"
#include "mirtk/ImageReader.h"
#include "mirtk/Dilation.h"

using namespace mirtk;
using namespace std;
 
// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------
void usage()
{
    cout << "..." << endl;
    exit(1);
}
// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{

    char buffer[256];
    RealImage stack;

    char *output_name = NULL;

    ReconstructionFFD reconstruction;
    
    RealImage input_stack, input_mask, output_stack;
    

    char *tmp_fname = NULL;
    UniquePtr<BaseImage> tmp_image;
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();


    
    //read input name
    tmp_fname = argv[1];
    input_stack.Read(tmp_fname); 
    argc--;
    argv++;

    //read mask name
    tmp_fname = argv[1];
    input_mask.Read(tmp_fname); 
    argc--;
    argv++;

    //read output name
    output_name = argv[1];
    argc--;
    argv++;

    output_stack = input_stack;
    output_stack = 0;
    
    RealImage output_mask = output_stack;
    output_mask = 0;
    
    
    RigidTransformation *rigidTransf_mask = new RigidTransformation;
    reconstruction.TransformMask(input_stack, input_mask, *rigidTransf_mask);
    
    int sh = 0;
    
    for (int x = sh; x < input_stack.GetX()-sh; x++) {
       for (int y = sh; y < input_stack.GetY()-sh; y++) {
           for (int z = sh; z < input_stack.GetZ()-sh; z++) {

               if (input_mask(x,y,z)>0.1) {
                   output_stack(x,y,z) = input_stack(x,y,z);
                   output_mask(x,y,z) = 1;
               }
               else {
                   output_stack(x,y,z) = 0;
               }
           }
       }
   }

    output_stack.Write(output_name);
    
    output_mask.Write("new-mask.nii.gz");

    
    return 0;
}
