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
    cout << "Usage: split-stacks [N] [stack_1] .. [stack_N] \n" << endl;
    exit(0);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    
    
    if (argc < 2)
        usage();
    
    
    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();


    //-------------------------------------------------------------------
   
    
    int nStacks;

    //read number of stacks
    nStacks = atoi(argv[1]);
    cout<<"Number of input stacks : "<<nStacks<<endl;
    argc--;
    argv++;

    //-------------------------------------------------------------------
    
    // Read stacks
    Array<RealImage> stacks;
    RealImage stack;

    
    for (int i=0; i<nStacks; i++) {
        
        cout<<"- reading stack : "<<argv[1]<<endl;
        
        tmp_fname = argv[1];
        image_reader.reset(ImageReader::TryNew(tmp_fname));
        tmp_image.reset(image_reader->Run());
        
        stack = *tmp_image;
        
        argc--;
        argv++;
        stacks.push_back(stack);
    }

    //-------------------------------------------------------------------

    // Split stacks


    Array<RealImage> org_stacks = stacks;
    stacks.clear();

    char buffer[256];
    
    for (int i=0; i<org_stacks.size(); i++) {
        
        for (int t=0; t<org_stacks[i].GetT(); t++) {
            
            stack = org_stacks[i].GetRegion(0,0,0,t,org_stacks[i].GetX(),org_stacks[i].GetY(),org_stacks[i].GetZ(),(t+1));
            stacks.push_back(stack);
        }
    }
    
    org_stacks.clear();
    nStacks = stacks.size();

    
    for (int i=0; i<stacks.size(); i++) {

	sprintf(buffer,"stack-%i.nii.gz",i);
        stacks[i].Write(buffer);

    }

    cout << "Total number of stacks : " << nStacks << endl;

    //-------------------------------------------------------------------

    
    return 0;
}



