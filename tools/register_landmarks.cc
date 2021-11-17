/*
 *
 */

// MIRTK
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

using namespace std;
using namespace mirtk;

// =============================================================================
// Auxiliary functions
// =============================================================================

// -----------------------------------------------------------------------------
void usage()
{
    cout << "???" << endl;
    exit(1);
}


void FindCentroid(RealImage input_mask, double& wx, double& wy, double& wz)
{

    double n = 0;
    double average_x = 0;
    double average_y = 0;
    double average_z = 0;

    for (int x=0; x<input_mask.GetX(); x++) {
        for (int y=0; y<input_mask.GetY(); y++) {
            for (int z=0; z<input_mask.GetZ(); z++) {

                if (input_mask(x,y,z)>0.5) {
                    average_x = average_x + x;
                    average_y = average_y + y;
                    average_z = average_z + z;
                    n = n + 1;
                }

            }
        }
    }

    if (n>0) {
        average_x = average_x / n;
        average_y = average_y / n;
        average_z = average_z / n;
    }

    average_x = round(average_x);
    average_y = round(average_y);
    average_z = round(average_z);

    input_mask.ImageToWorld(average_x, average_y, average_z);

    wx = average_x;
    wy = average_y;
    wz = average_z;

    return;
}






int RegisterLandmarks(Array<Point> source, Array<Point> target, RigidTransformation& output_transformation)
{

    output_transformation = RigidTransformation();

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

    int out_reflection = 0;

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

        out_reflection = 0;


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

        out_reflection = 1;

    }

    return out_reflection;
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

    RealImage target_image;
    RealImage source_image;


    cout << "------------------------------------------------------" << endl;

    char *target_image_name = NULL;
    target_image_name = argv[1];
    target_image.Read(target_image_name);

    argc--;
    argv++;

    cout << " - Target image : " << target_image_name << endl;



    char *source_image_name = NULL;
    source_image_name = argv[1];
    //source_image.Read(source_image_name);
    argc--;
    argv++;

    cout << " - Source image : " << source_image_name << endl;



    char *init_dof_name = NULL;
    init_dof_name = argv[1];
    argc--;
    argv++;

    Transformation *t = Transformation::New(init_dof_name);
    unique_ptr<RigidTransformation> init_transf(dynamic_cast<RigidTransformation*>(t));

    cout << " - Init dof : " << init_dof_name << endl;



    char *output_dof_name = NULL;
    output_dof_name = argv[1];
    argc--;
    argv++;

    cout << " - Output dof : " << output_dof_name << endl;


//    int output_mode = 0;
//
//    output_mode = atoi(argv[1]);
//    argc--;
//    argv++;
//
//    cout << " - Output format mode : " << output_mode << endl;


    cout << "------------------------------------------------------" << endl;

    int mask_number = 0;

    Array<RealImage> target_masks;
    Array<RealImage> source_masks;

    mask_number = atoi(argv[1]);
    argc--;
    argv++;


    int selected_point_number = atoi(argv[1]);
    argc--;
    argv++;


    Array<int> target_landmark_status;
    Array<int> source_landmark_status;


    int N_target_points = mask_number;
    int N_source_points = mask_number;

    int coord = 0;
    int dir = 0;
    double num = 0;
    double dx, dy, dz;


    cout << " - Number of masks : " << mask_number << endl;
    cout << endl;

    cout << " - Targets : " << endl;
    for (int i=0; i<mask_number; i++) {

        char *tmp_mask_name = NULL;
        tmp_mask_name = argv[1];
        cout << " - " << tmp_mask_name << " : ";

        RealImage tmp_mask;
        tmp_mask.Read(tmp_mask_name);
        target_masks.push_back(tmp_mask);

        double wx, wy, wz;
        wx = 0;
        wy = 0;
        wz = 0;
        FindCentroid(tmp_mask, wx, wy, wz);
        Point *p = new Point(wx, wy, wz);
        target_points.push_back(*p);

        RealPixel smin, smax;
        int st = 0;
        tmp_mask.GetMinMax(&smin, &smax);
        if (smax > 0.5) {
            target_landmark_status.push_back(1);
            st = 1;
        } else {
            target_landmark_status.push_back(-1);
            st = -1;
        }

        cout << " - " << st << " : " << wx << " " << wy << " " << wz << endl;

        argc--;
        argv++;

    }

    N_target_points = target_points.size();


    cout << endl;
    cout << " - Sources : " << endl;
    for (int i=0; i<mask_number; i++) {

        char *tmp_mask_name = NULL;
        tmp_mask_name = argv[1];
        cout << " - " << tmp_mask_name << " : ";

        RealImage tmp_mask;
        tmp_mask.Read(tmp_mask_name);
        source_masks.push_back(tmp_mask);


        RealPixel smin, smax;
        int st = 0;
        tmp_mask.GetMinMax(&smin, &smax);
        if (smax > 0.5) {
            source_landmark_status.push_back(1);
            st = 1;
        } else {
            source_landmark_status.push_back(-1);
            st = -1;
        }


        double wx, wy, wz;
        wx = 0;
        wy = 0;
        wz = 0;
        FindCentroid(tmp_mask, wx, wy, wz);
        Point *p = new Point(wx, wy, wz);
        source_points.push_back(*p);



        cout << " - " << st << " : " << wx << " " << wy << " " << wz << endl;


        argc--;
        argv++;

    }

    N_source_points = source_points.size();


    cout << "------------------------------------------------------" << endl;


    if (N_source_points != N_target_points) {

        cout << "Error: number of source and target points should be the same." << endl;
        exit(1);
    }


    Array<Point> selected_target_points;
    Array<Point> selected_source_points;


    Array<Point> selected_target_points_4;
    Array<Point> selected_source_points_4;


    int number_of_selected_points = 0;

    PointSet target_set, source_set;
    PointSet target_set_4, source_set_4;
    int detected_points_to_use = 0;
    for (int i=0; i<N_target_points; i++) {

        if (source_landmark_status[i] > 0 && target_landmark_status[i] > 0 && detected_points_to_use < selected_point_number) {

            target_set.Add(target_points[i]);
            source_set.Add(source_points[i]);

            selected_target_points.push_back(target_points[i]);
            selected_source_points.push_back(source_points[i]);

	    number_of_selected_points = number_of_selected_points + 1;

            detected_points_to_use = detected_points_to_use + 1;
            cout << i << " ";

            if (selected_target_points_4.size() < 4) {

                selected_target_points_4.push_back(target_points[i]);
                selected_source_points_4.push_back(source_points[i]);

            }

            target_set_4.Add(target_points[i]);
            source_set_4.Add(source_points[i]);


        }

    }

    cout << endl;





    cout << "------------------------------------------------------" << endl;

    RigidTransformation output_transformation;



    if (number_of_selected_points < 3) {

	cout << " - Not enough points - will use initial ... " << endl;
    	output_transformation = *init_transf;

	    cout << " - Output transformation (init) " << output_dof_name << " : ";
	    cout << output_transformation.GetTranslationX() << " ";
	    cout << output_transformation.GetTranslationY() << " ";
	    cout << output_transformation.GetTranslationZ() << " | ";
	    cout << output_transformation.GetRotationX() << " ";
	    cout << output_transformation.GetRotationY() << " ";
	    cout << output_transformation.GetRotationZ() << endl;

    }
    else {

	//    int out_reflection = RegisterLandmarks(source_points, target_points, output_transformation);
	    int out_reflection = RegisterLandmarks(selected_source_points, selected_target_points, output_transformation);

	    cout << " - Output transformation " << output_dof_name << " : ";
	    cout << output_transformation.GetTranslationX() << " ";
	    cout << output_transformation.GetTranslationY() << " ";
	    cout << output_transformation.GetTranslationZ() << " | ";
	    cout << output_transformation.GetRotationX() << " ";
	    cout << output_transformation.GetRotationY() << " ";
	    cout << output_transformation.GetRotationZ() << endl;


	    output_transformation.Transform(source_set);

	    double error = 0;
	    for (int i = 0; i < source_set.Size(); i++) {
		mirtk::Point p1 = target_set(i);
		mirtk::Point p2 = source_set(i);
		error += sqrt(pow(p1._x - p2._x, 2.0) + pow(p1._y - p2._y, 2.0) + pow(p1._z - p2._z, 2.0));
	    }
	    error /= target_set.Size();

	    cout << " - Target registration error (N) : " << error << endl;

	    if (out_reflection>0) {
		cout << "REFLECTION !!!!!" << endl;
	    }


    }


//    cout << "------------------------------------------------------" << endl;
//
//
//    RigidTransformation output_transformation_4;
//    //    int out_reflection = RegisterLandmarks(source_points, target_points, output_transformation);
//    int out_reflection_4 = RegisterLandmarks(selected_source_points_4, selected_target_points_4, output_transformation_4);
//
//    cout << " - Output transformation v2 (4) " << output_dof_name << " : ";
//    cout << output_transformation_4.GetTranslationX() << " ";
//    cout << output_transformation_4.GetTranslationY() << " ";
//    cout << output_transformation_4.GetTranslationZ() << " | ";
//    cout << output_transformation_4.GetRotationX() << " ";
//    cout << output_transformation_4.GetRotationY() << " ";
//    cout << output_transformation_4.GetRotationZ() << endl;
//
//
//    output_transformation_4.Transform(source_set_4);
//
//    double error_4 = 0;
//    for (int i = 0; i < source_set_4.Size(); i++) {
//        mirtk::Point p1 = target_set_4(i);
//        mirtk::Point p2 = source_set_4(i);
//        error_4 += sqrt(pow(p1._x - p2._x, 2.0) + pow(p1._y - p2._y, 2.0) + pow(p1._z - p2._z, 2.0));
//    }
//    error_4 /= target_set_4.Size();
//
//    cout << " - Target registration error v2 (4) : " << error_4 << endl;
//    if (out_reflection_4>0) {
//        cout << "REFLECTION !!!!!" << endl;
//    }


    cout << "------------------------------------------------------" << endl;

//    if (error_4 < error) {
//
//        output_transformation_4.Write(output_dof_name);
//        cout << "Selected : 4" << endl;
//
//    } else {

        output_transformation.Write(output_dof_name);
        cout << "Selected : " << number_of_selected_points << endl;

//    }


    cout << "------------------------------------------------------" << endl;




//    output_transformation.Invert();
//
//    RealImage processed_stack;
//    processed_stack.Read(input_image_name);
//
//    Matrix m = output_transformation.GetMatrix();
//    processed_stack.PutAffineMatrix(m, true);
//
//    processed_stack.Write(output_image_name);


//
//    if (out_reflection>0) {
//
//
//        cout << "REFLECTION !!!!!" << endl;
//
////        string org_name(output_image_name);
////        std::size_t pos = org_name.find(".nii");
////        std::string main_name = org_name.substr (0, pos);
////        string new_name = main_name + "-reflection.dof";
////        char *c_new_name = &new_name[0];
////        output_transformation.Write(c_new_name);
//
//
//
//    }
//
//
//
//    cout << "------------------------------------------------------" << endl;
//

    return 0;
}
