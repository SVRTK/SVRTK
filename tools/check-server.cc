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
    cout << "Usage: ... \n" << endl;
    exit(0);
}

// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
   
    system("echo ----------------------------------------------------- ");
    system("echo /projects/perinatal/peridata/fetalsvr/transfer/$(whoami) : ");
    system("ls /projects/perinatal/peridata/fetalsvr/transfer/$(whoami) ");
    system("echo ");
    system("echo ----------------------------------------------------- ");
    system("echo /projects/perinatal/peridata/fetalsvr/transfer/$(whoami)/Transfer : ");
    system("ls /projects/perinatal/peridata/fetalsvr/transfer/$(whoami)/Transfer ");
    system("echo ");
    system("echo ----------------------------------------------------- ");
    system("top -n 1 -b > /projects/perinatal/peridata/fetalsvr/transfer/$(whoami)/top.txt ");
    system("head -15 /projects/perinatal/peridata/fetalsvr/transfer/$(whoami)/top.txt ");
    system("echo ");
    system("echo ----------------------------------------------------- ");
    

    return 0;
}



