/*
* SVRTK : SVR reconstruction based on MIRTK
*
* Copyright 2018-2021 King's College London
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

#include <set>

using namespace mirtk;
using namespace std;


struct Point3D {
    double x, y, z;
};


#include <cmath>
#include <iostream>
#include <cmath>


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



double func_1_landmark_distance(RealImage input_mask, int l1, int l2)
{
    double dist = 0;

    
    double x1, x2, y1, y2, z1, z2, n1, n2;
    x1 = 0; x2 = 0; y1 = 0; y2 = 0; z1 = 0; z2 = 0; n1 = 0; n2 = 0;
    
    int sh = 0;
    for (int x = sh; x < input_mask.GetX()-sh; x++) {
        for (int y = sh; y < input_mask.GetY()-sh; y++) {
            for (int z = sh; z < input_mask.GetZ()-sh; z++) {
                if (input_mask(x,y,z) == l1) {
                    x1 = x1 + x;
                    y1 = y1 + y;
                    z1 = z1 + z;
                    n1 = n1 + 1;
                }
                if (input_mask(x,y,z) == l2) {
                    x2 = x2 + x;
                    y2 = y2 + y;
                    z2 = z2 + z;
                    n2 = n2 + 1;
                }
            }
        }
    }
    
    if (n1 > 0) {
        x1 = x1 / n1; y1 = y1 / n1;  z1 = z1 / n1;
    } else {
        return 0;
    }
    
    if (n2 > 0) {
        x2 = x2 / n2; y2 = y2 / n2;  z2 = z2 / n2;
    } else {
        return 0;
    }
    
    input_mask.ImageToWorld(x1, y1, z1);
    input_mask.ImageToWorld(x2, y2, z2);
    
    dist = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
    
    return dist;
    
}


double func_2_landmark_angle(RealImage input_mask, int l1, int l2, int l3)
{
    double angle = 0;

    double x1, x2, x3, y1, y2, y3, z1, z2, z3, n1, n2, n3;
    x1 = 0; x2 = 0; x3 = 0; y1 = 0; y2 = 0; y3 = 0; z1 = 0; z2 = 0; z3 = 0; n1 = 0; n2 = 0; n3 = 0;

    int sh = 0;
    for (int x = sh; x < input_mask.GetX() - sh; x++) {
        for (int y = sh; y < input_mask.GetY() - sh; y++) {
            for (int z = sh; z < input_mask.GetZ() - sh; z++) {
                if (input_mask(x, y, z) == l1) {
                    x1 = x1 + x;
                    y1 = y1 + y;
                    z1 = z1 + z;
                    n1 = n1 + 1;
                }
                if (input_mask(x, y, z) == l2) {
                    x2 = x2 + x;
                    y2 = y2 + y;
                    z2 = z2 + z;
                    n2 = n2 + 1;
                }
                if (input_mask(x, y, z) == l3) {
                    x3 = x3 + x;
                    y3 = y3 + y;
                    z3 = z3 + z;
                    n3 = n3 + 1;
                }
            }
        }
    }

    if (n1 > 0) {
        x1 = x1 / n1; y1 = y1 / n1; z1 = z1 / n1;
    } else {
        return 0;
    }

    if (n2 > 0) {
        x2 = x2 / n2; y2 = y2 / n2; z2 = z2 / n2;
    } else {
        return 0;
    }

    if (n3 > 0) {
        x3 = x3 / n3; y3 = y3 / n3; z3 = z3 / n3;
    } else {
        return 0;
    }

    input_mask.ImageToWorld(x1, y1, z1);
    input_mask.ImageToWorld(x2, y2, z2);
    input_mask.ImageToWorld(x3, y3, z3);

    // Vectors representing the two lines
    double v1x = x2 - x1, v1y = y2 - y1, v1z = z2 - z1;
    double v2x = x3 - x2, v2y = y3 - y2, v2z = z3 - z2;

    // Dot product
    double dotProduct = v1x * v2x + v1y * v2y + v1z * v2z;

    // Magnitudes
    double magnitudeV1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    double magnitudeV2 = sqrt(v2x * v2x + v2y * v2y + v2z * v2z);

    // Calculate the angle in radians
    double angleRad = acos(dotProduct / (magnitudeV1 * magnitudeV2));

    // Convert angle from radians to degrees
    angle = 180.0 - (angleRad * 180.0 / M_PI);

    return angle;
}




//-------------measurements (bio) with labels----------------//

int main(int argc, char **argv)
{

    char buffer[256];
    RealImage stack;

    char *output_name = NULL;

    RealImage input_mask;
    
    char *tmp_fname = NULL;
    UniquePtr<BaseImage> tmp_image;
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    
    Array<int> l1_array;
    Array<int> l2_array;
    Array<int> l3_array;
    Array<int> l4_array;
    Array<int> bio_id_array;
    // 1 - line distance


    Array<int> bio_type_array;
    Array<string> bio_name_array;
    Array<double> bio_out_array;
    
    
//--1 - line distance- ASBL-PSBL-HPL-VPL-PaW-Rcho/Lcho-NPw-NB-PNTh-OFD-BPD-MCh-ROD/LOD-IOD-BOD-MxW-MdW---//

    l1_array.push_back(1);
    l2_array.push_back(2);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(1);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Skull biparietal diameter (1-2)");
    
    l1_array.push_back(5);
    l2_array.push_back(6);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(2);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Skull occipitofrontal diameter (5-6)");
    
    l1_array.push_back(3);
    l2_array.push_back(4);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(3);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Brain biparietal diameter (3-4)");
    
    l1_array.push_back(7);
    l2_array.push_back(8);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(4);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Brain occipitofrontal diameter Right (7-8)");
    
    l1_array.push_back(32);
    l2_array.push_back(33);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(5);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Brain occipitofrontal diameter Left (32-33)");
    
    l1_array.push_back(9);
    l2_array.push_back(10);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(6);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Transcerebellar diameter (9-10)");
    
    l1_array.push_back(11);
    l2_array.push_back(12);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(7);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Vermian height (11-12)");
    
    l1_array.push_back(13);
    l2_array.push_back(14);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(8);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Vermian width (13-14)");
    
    l1_array.push_back(15);
    l2_array.push_back(16);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(9);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Pons width (15-16)");
    
    l1_array.push_back(17);
    l2_array.push_back(18);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(10);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Corpus Callosum (17-18)");
    
    l1_array.push_back(19);
    l2_array.push_back(20);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(11);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Cavum Septum Pellucidum (19-20)");
    
    l1_array.push_back(21);
    l2_array.push_back(22);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(12);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Atrial diameter Right (21-22)");
    
    l1_array.push_back(35);
    l2_array.push_back(36);
    l3_array.push_back(-1);
    l4_array.push_back(-1);
    bio_id_array.push_back(13);
    bio_type_array.push_back(1);
    bio_name_array.push_back("Atrial diameter Left (35-36)");
    
    
    
    l1_array.push_back(27);
    l2_array.push_back(28);
    l3_array.push_back(29);
    l4_array.push_back(-1);
    bio_id_array.push_back(14);
    bio_type_array.push_back(2);
    bio_name_array.push_back("Tegmentovermian angle (27-28-29)");

    
    char *out_csv_name = NULL;
    
    out_csv_name = argv[1];
    argc--;
    argv++;

    
    int N = 0;
    N = atoi(argv[1]);
    argc--;
    argv++;
    
    sprintf(buffer, out_csv_name);
    ofstream info_bio;
    info_bio.open( buffer );
    
    
    info_bio << "lab_file" << ",";

    for (int i = 0; i < bio_id_array.size(); i++) {
        
        info_bio << bio_name_array[i] << "," ;
    }
    
    info_bio << endl;
    

    for (int n = 0; n < N; n++) {
        
        tmp_fname = argv[1];
        input_mask.Read(tmp_fname);
        argc--;
        argv++;
        
        cout << n << " : " << tmp_fname << endl; 
        
        info_bio << tmp_fname << ",";


        for (int i = 0; i < bio_id_array.size(); i++) {

            int l1 = l1_array[i];
            int l2 = l2_array[i];
            int l3 = l3_array[i];
            int l4 = l4_array[i];
            
            double bio = 0;
          
            if (bio_type_array[i] == 1) {
                bio = func_1_landmark_distance(input_mask, l1, l2);
            }
    
            if (bio_type_array[i] == 2) {
                bio = func_2_landmark_angle(input_mask, l1, l2, l3);
            }

            
            bio_out_array.push_back(bio);

            info_bio << bio << ",";
            
            cout << bio_name_array[i] << " " << bio_id_array[i] << " " <<  bio << endl;

            
        }
        
        info_bio << endl;
    }
    
    
    info_bio.close();


    
    return 0;
}


