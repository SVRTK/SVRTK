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
    cout << "\t" << endl;
    cout << "\t" << endl;
    exit(0);
}
// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------


RealImage set_landmark(RealImage input_mask, int l1, int d)
{

    double x1, y1, z1, n1;
    x1 = 0; y1 = 0; z1 = 0; n1 = 0;
    
    int sh = 0;
    for (int x = sh; x < input_mask.GetX()-sh; x++) {
        for (int y = sh; y < input_mask.GetY()-sh; y++) {
            for (int z = sh; z < input_mask.GetZ()-sh; z++) {
                if (input_mask(x,y,z) == l1) {
                    x1 = x1 + x;
                    y1 = y1 + y;
                    z1 = z1 + z;
                    n1 = n1 + 1;
                    input_mask(x,y,z) = 0;
                }
            }
        }
    }
    
    if (n1 > 0) {
        x1 = x1 / n1; y1 = y1 / n1;  z1 = z1 / n1;
    }
    
    x1 = round(x1);
    y1 = round(y1);
    z1 = round(z1);
    
    
    sh = d;
    int rsquared = d * d;
    
    for (int x = x1 - sh; x < x1 + sh; x++) {
        for (int y = y1 - sh; y < y1 + sh; y++) {
            for (int z = z1 - sh; z < z1 + sh; z++) {
//                    input_mask(x,y,z) = l1;
                
                int dx = x - x1;
                int dy = y - y1;
                int dz = z - z1;
                if (dx * dx + dy * dy + dz * dz <= rsquared) {
                    input_mask(x,y,z) = l1;
                }
                
                
            }
        }
    }
    
    
    return input_mask;
    
}




int main(int argc, char **argv)
{

    char buffer[256];
    RealImage stack;

    char *output_name = NULL;

    
    RealImage input_mask, output_mask, reference_mask;
    

    char *tmp_fname = NULL;
    UniquePtr<BaseImage> tmp_image;
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();


    tmp_fname = argv[1];
    input_mask.Read(tmp_fname);
    argc--;
    argv++;
    
    
    output_name = argv[1];
    argc--;
    argv++;
    
    
    int num_l_ref, num_l_in, num_l_out;

    output_mask = input_mask;
    
    output_mask = 0;
 
    int sh = 1;
    
    int sh_x, sh_y, sh_z;

     for (int x = sh; x < input_mask.GetX()-sh; x++) {
        for (int y = sh; y < input_mask.GetY()-sh; y++) {
            for (int z = sh; z < input_mask.GetZ()-sh; z++) {
                
  
               num_l_in = 1;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 2;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }


               num_l_in = 5;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }


               num_l_in = 6;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }



               num_l_in = 3;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 4;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }




               num_l_in = 7;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 8;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }


               num_l_in = 32;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 33;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }


               num_l_in = 9;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;
                
               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 10;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 11;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 12;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 13;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 14;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }


               num_l_in = 15;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 16;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 17;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 18;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }


               num_l_in = 19;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 20;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }


               num_l_in = 21;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 22;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }


               num_l_in = 35;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;
               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 36;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }



               num_l_in = 27;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;
                
               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 28;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               num_l_in = 29;
                sh_x = 0;
                sh_y = 0;
                sh_z = 0;

               if ( input_mask(x,y,z) == num_l_in ) {
                   output_mask( (x+sh_x), (y+sh_y), (z+sh_z) ) = num_l_in;

               }

               
                
            }
        }
    }
    
    
    int d = 4;
    
    num_l_in = 1;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 2;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 3;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 4;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 5;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 6;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 7;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 8;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 9;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 10;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 11;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 12;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 13;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 14;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 15;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 16;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 17;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 18;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 19;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 20;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 21;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 22;
    output_mask = set_landmark(output_mask, num_l_in, d);

    
    num_l_in = 27;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 28;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 29;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 32;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 33;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 35;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    num_l_in = 36;
    output_mask = set_landmark(output_mask, num_l_in, d);
    
    
    
    output_mask.Write(output_name);

    
    return 0;
    
}


