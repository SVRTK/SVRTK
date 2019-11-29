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

//#include "mirtk/ReconstructionFFD.h"
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
    cout << "???" << endl;
    exit(1);
}

void RegisterLandmarks(Array<Point> source, Array<Point> target, RigidTransformation& output_transformation)
{
    
    RigidTransformation *tmp = new mirtk::RigidTransformation();
    output_transformation = *tmp;
    
    int i;
    double x, y, z;
    Vector w;
    Matrix u, v, r;
    
    // Calculate centroids
    Point target_centroid;
    Point source_centroid;
    
    for (i = 0; i < target.size (); i++) {
        target_centroid -= target[i] / target.size ();
        source_centroid -= source[i] / source.size ();
    }
    
    // Subtract centroids
    for (i = 0; i < target.size (); i++) {
        target[i] += target_centroid;
        source[i] += source_centroid;
    }
    
    // Calculate covariance matrix
    Matrix h (3, 3);
    Matrix a (3, 1);
    Matrix b (1, 3);
    Matrix m (4, 4);
    
    for (i = 0; i < target.size (); i++) {
        a (0, 0) = target[i]._x;
        a (1, 0) = target[i]._y;
        a (2, 0) = target[i]._z;
        b (0, 0) = source[i]._x;
        b (0, 1) = source[i]._y;
        b (0, 2) = source[i]._z;
        h += a * b;
    }
    
    // Calculate SVD
    h.SVD (u, w, v);
    
    // Calculate rotation matrix
    u.Transpose ();
    r = v * u;
    
    // Check whether matrix is a rotation
    if (r.Det () > 0.999) {
        
        x = target_centroid._x;
        y = target_centroid._y;
        z = target_centroid._z;
        
        // Calculate rotated centroid
        target_centroid._x = r (0, 0) * x + r (0, 1) * y + r (0, 2) * z;
        target_centroid._y = r (1, 0) * x + r (1, 1) * y + r (1, 2) * z;
        target_centroid._z = r (2, 0) * x + r (2, 1) * y + r (2, 2) * z;
        
        // Calculate transformation matrix
        m (0, 0) = r (0, 0);
        m (1, 0) = r (1, 0);
        m (2, 0) = r (2, 0);
        m (3, 0) = 0;
        m (0, 1) = r (0, 1);
        m (1, 1) = r (1, 1);
        m (2, 1) = r (2, 1);
        m (3, 1) = 0;
        m (0, 2) = r (0, 2);
        m (1, 2) = r (1, 2);
        m (2, 2) = r (2, 2);
        m (3, 2) = 0;
        m (0, 3) = target_centroid._x - source_centroid._x;
        m (1, 3) = target_centroid._y - source_centroid._y;
        m (2, 3) = target_centroid._z - source_centroid._z;
        m (3, 3) = 1;
        
        // Update transformation
        output_transformation.PutMatrix (m);
        output_transformation.UpdateParameter();
        
        
    } else {
        
        cout <<  "Landmark registration: Rotation involves reflection" << endl;
        
        // Search for most singular value
        i = 0;
        if ((w (0) < w (1)) && (w (0) < w (2)))
            i = 0;
        if ((w (1) < w (0)) && (w (1) < w (2)))
            i = 1;
        if ((w (2) < w (1)) && (w (2) < w (0)))
            i = 2;
        
        // Multiply column with most singular value by -1
        v (0, i) *= -1;
        v (1, i) *= -1;
        v (2, i) *= -1;
        
        // Recalculate rotation matrix
        r = v * u;
        
        x = target_centroid._x;
        y = target_centroid._y;
        z = target_centroid._z;
        
        // Calculate rotated centroid
        target_centroid._x = r (0, 0) * x + r (0, 1) * y + r (0, 2) * z;
        target_centroid._y = r (1, 0) * x + r (1, 1) * y + r (1, 2) * z;
        target_centroid._z = r (2, 0) * x + r (2, 1) * y + r (2, 2) * z;
        
        // Calculate transformation matrix
        m (0, 0) = r (0, 0);
        m (1, 0) = r (1, 0);
        m (2, 0) = r (2, 0);
        m (3, 0) = 0;
        m (0, 1) = r (0, 1);
        m (1, 1) = r (1, 1);
        m (2, 1) = r (2, 1);
        m (3, 1) = 0;
        m (0, 2) = r (0, 2);
        m (1, 2) = r (1, 2);
        m (2, 2) = r (2, 2);
        m (3, 2) = 0;
        m (0, 3) = target_centroid._x - source_centroid._x;
        m (1, 3) = target_centroid._y - source_centroid._y;
        m (2, 3) = target_centroid._z - source_centroid._z;
        m (3, 3) = 1;
        
        // Update transformation
        output_transformation.PutMatrix(m);
        output_transformation.UpdateParameter();
        
    }
    
    
}


// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{

    char buffer[256];
    
//    if (argc < 17)
//        usage();
    
    
    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();
    
    
    Array<Point> source_points;
    Array<Point> target_points;
    
    
    cout << "------------------------------------------------------" << endl;
    
    char *target_image_name = NULL;
    target_image_name = argv[1];
    argc--;
    argv++;
    
    cout << " - Target image name : " << target_image_name << endl;
    
    
    
    char *input_image_name = NULL;
    input_image_name = argv[1];
    argc--;
    argv++;
    
    cout << " - Input image name : " << input_image_name << endl;
    
    
    char *output_image_name = NULL;
    output_image_name = argv[1];
    argc--;
    argv++;
    
    cout << " - Output image name : " << output_image_name << endl;
    
    
    char *output_name = NULL;
    output_name = argv[1];
    argc--;
    argv++;
    
    cout << " - Output dof name : " << output_name << endl;
    
    
    int output_mode = 0;
    
    output_mode = atoi(argv[1]);
    argc--;
    argv++;
    
    cout << " - Output format mode : " << output_mode << endl;


    cout << "------------------------------------------------------" << endl;
    

    char *target_pointset_file = argv[1];
    argc--;
    argv++;
    
    ifstream in_target_pointset(target_pointset_file);
    int N_target_points;
    
    int coord = 0;
    int dir = 0;
    double num = 0;
    double dx, dy, dz;
    

    cout<<" - Reading target points from " << target_pointset_file << " : "<<endl;
    if (in_target_pointset.is_open()) {
        
        in_target_pointset >> N_target_points;
        
        cout << N_target_points << endl;
        
        while (!in_target_pointset.eof()) {
            in_target_pointset >> num;
            
            if (dir==0)
                dx = num;
            
            if (dir==1)
                dy = num;
            
            if (dir==2)
                dz = num;

            dir++;
            
            if (dir>=3) {
                dir=0;
                coord++;
                
                
                Point *p = new Point(dx, dy, dz);
                target_points.push_back(*p);
                
                cout << dx << " " << dy << " " << dz << endl;
                
                dx = 0;
                dy = 0;
                dz = 0;
                
            }
            
        }
        
        in_target_pointset.close();
        
    }
    else {
        cout << " - Unable to open file " << target_pointset_file << endl;
        exit(1);
    }
    
    
    cout << "------------------------------------------------------" << endl;
    
    
    
    char *source_pointset_file = argv[1];
    argc--;
    argv++;
    
    int N_source_points;
    ifstream in_source_pointset(source_pointset_file);
    
    coord = 0;
    dir = 0;
    num = 0;
    dx = 0;
    dy = 0;
    dz = 0;

    cout<<" - Reading target points from " << source_pointset_file << " : "<<endl;
    if (in_source_pointset.is_open()) {
        
        in_source_pointset >> N_source_points;
        
        cout << N_source_points << endl;
        
        while (!in_source_pointset.eof()) {
            in_source_pointset >> num;
            
            if (dir==0)
                dx = num;
            
            if (dir==1)
                dy = num;
            
            if (dir==2)
                dz = num;
            
            dir++;
            
            if (dir>=3) {
                dir=0;
                coord++;
                
                
                Point *p = new Point(dx, dy, dz);
                source_points.push_back(*p);
                
                cout << dx << " " << dy << " " << dz << endl;
                
                dx = 0;
                dy = 0;
                dz = 0;
                
            }
            
        }
        
        in_source_pointset.close();
        
    }
    else {
        cout << " - Unable to open file " << source_pointset_file << endl;
        exit(1);
    }
    
    cout << "------------------------------------------------------" << endl;
    
    
    if (N_source_points != N_target_points) {
        
        cout << "Error: number of source and target points should be the same." << endl;
        exit(1);
    }
    
    
    PointSet target_set, source_set;
    for (int i=0; i<N_target_points; i++) {
        
        target_set.Add(target_points[i]);
        source_set.Add(source_points[i]);

    }
    
    
    cout << "------------------------------------------------------" << endl;
    
    RigidTransformation output_transformation;
    RegisterLandmarks(source_points, target_points, output_transformation);
    
    cout << " - Output transformation " << output_name << " : ";
    cout << output_transformation.GetTranslationX() << " ";
    cout << output_transformation.GetTranslationY() << " ";
    cout << output_transformation.GetTranslationZ() << " | ";
    cout << output_transformation.GetRotationX() << " ";
    cout << output_transformation.GetRotationY() << " ";
    cout << output_transformation.GetRotationZ() << endl;
    
    
    output_transformation.Write(output_name);
    
    
    cout << "------------------------------------------------------" << endl;
    
    output_transformation.Transform(source_set);
    
    double error = 0;
    for (int i = 0; i < source_set.Size(); i++) {
        mirtk::Point p1 = target_set(i);
        mirtk::Point p2 = source_set(i);
        error += sqrt(pow(p1._x - p2._x, 2.0) + pow(p1._y - p2._y, 2.0) + pow(p1._z - p2._z, 2.0));
    }
    error /= target_set.Size();
    
    cout << " - Target registration error : " << error << endl;
    
    
    cout << "------------------------------------------------------" << endl;
    
  
    if (output_mode < 1) {
        
        output_transformation.Invert();
        
        RealImage processed_stack;
        processed_stack.Read(input_image_name);
    
        Matrix m = output_transformation.GetMatrix();
        processed_stack.PutAffineMatrix(m, true);
        
        processed_stack.Write(output_image_name);
        
    } else {
        
        
        RealImage source_stack;
        source_stack.Read(input_image_name);
        
        RealImage target_stack;
        target_stack.Read(target_image_name);
        
        RealImage output_volume = target_stack;
        
        
        double source_padding = 0;
        double target_padding = -inf;
        bool dofin_invert = false;
        bool twod = false;
        GenericBSplineInterpolateImageFunction<RealImage> interpolator;
        
        ImageTransformation *imagetransformation = new ImageTransformation;
        
        imagetransformation->Input(&source_stack);
        imagetransformation->Transformation(&output_transformation);
        imagetransformation->Output(&output_volume);
        imagetransformation->TargetPaddingValue(target_padding);
        imagetransformation->SourcePaddingValue(source_padding);
        imagetransformation->Interpolator(&interpolator);
        imagetransformation->TwoD(twod);
        imagetransformation->Invert(dofin_invert);
        imagetransformation->Run();

        output_volume.Write(output_image_name);
        
        delete imagetransformation;
        
        
    }
    
    
    
    cout << " - Input volume : " << input_image_name << endl;
    
    cout << " - Output volume : " << output_image_name << endl;
    
    
    cout << "------------------------------------------------------" << endl;
    
    
    return 0;
}



