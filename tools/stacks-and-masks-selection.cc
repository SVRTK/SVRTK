
/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2008-2017 Imperial College London
 * Copyright 2018-2020 King's College London
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


// C++ Standard
#include <iostream>
#include <chrono>
#include <ctime>
#include <fstream>
#include <cmath>
#include <set>
#include <algorithm>
#include <thread>
#include <functional>
#include <vector>
#include <cstdlib>
#include <pthread.h>
#include <string>

#include "svrtk/Utility.h"

using namespace std;
using namespace mirtk;
using namespace svrtk;
using namespace svrtk::Utility;

// =============================================================================
//
// =============================================================================

// -----------------------------------------------------------------------------

void usage()
{
    cout << "Usage: mirtk stacks-and-masks-selection [number_of_input_stacks] [input_stack_1] ... [input_stack_n] " << endl;
    cout << "\t [input_mask_1] ... [input_mask_n] [output_folder_for_selected_stacks] [roi_dilation_degree_for_stack_cropping]" << endl;
    cout << "\t [include_volume_into_selection: 0/1 flag] [use_existing_template: 0/1 flag + [input_template_file]]" << endl;
    cout << endl;
    cout << "Function for automated preparation for an optimal input for reconstruct or reconstructBody 3D SVR SR functions. " << endl;
    cout << "The outputs include generated average template and mask and selected preregistered stacks." << endl;
    cout << "The stack selection is based on the median mark ROI volumes and cross-correlation between the stacks in the analysed ROI." << endl;
    cout << endl;
    cout << "\t" << endl;
    cout << "\t" << endl;

    exit(1);
}


double median_val(Array<double> in_vector)
{
  size_t size = in_vector.size();

  if (size == 0) {
    return 0;
  } else {
    sort(in_vector.begin(), in_vector.end());
    if (size % 2 == 0) {
      return (in_vector[size / 2 - 1] + in_vector[size / 2]) / 2;
    } else {
      return in_vector[size / 2];
    }
  }
}



// -----------------------------------------------------------------------------

// =============================================================================
// Main function
// =============================================================================

// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    const char *current_mirtk_path = argv[0];

    UniquePtr<ImageReader> image_reader;
    InitializeIOLibrary();

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();

    auto start_total = std::chrono::system_clock::now();
    auto end_total = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds;
    std::time_t end_time;

    //utility variables
    int i, j, x, y, z, ok;

    char buffer[256];

    RealImage stack;

    Array<RealImage> stacks;

    int nStacks;

    Array<RealImage> masks;

    string template_name;

    bool use_template = false;
    RealImage template_stack;


    bool include_volume = false;


    cout << "------------------------------------------------------" << endl;


    //read number of stacks
    nStacks = atoi(argv[1]);
    argc--;
    argv++;
    cout<<"Number of stacks : "<<nStacks<<endl;
    cout << endl;

    // Read stacks

    const char *tmp_fname;
    UniquePtr<BaseImage> tmp_image;



    for (i=0;i<nStacks;i++) {


        if (i == 0) {
            string tmp_str_name(argv[1]);
            template_name = tmp_str_name;
        }


        cout<<"Stack " << i << " : "<<argv[1]<<endl;

        tmp_fname = argv[1];
        image_reader.reset(ImageReader::TryNew(tmp_fname));
        tmp_image.reset(image_reader->Run());

        stack = *tmp_image;

        double smin, smax;
        stack.GetMinMax(&smin, &smax);

        if (smin < 0 || smax < 0) {
            template_stack.PutMinMaxAsDouble(0, 1000);
        }


        argc--;
        argv++;
        stacks.push_back(stack);
    }



    cout << endl;
    cout << "Masks : " << endl;
    for (i=0;i<nStacks;i++) {

        RealImage tmp_mask;
        tmp_mask.Read(argv[1]);

        tmp_mask = CreateMask(tmp_mask);

        if (tmp_mask.GetX() != stacks[i].GetX() || tmp_mask.GetY() != stacks[i].GetY() || tmp_mask.GetZ() != stacks[i].GetZ() || tmp_mask.GetT() != stacks[i].GetT()) {
            cout << "Error: the mask (" << argv[1] << ") dimensions are different from the corresponding stack (" << i << ") !" << endl;
            exit(1);
        }

        masks.push_back(tmp_mask);

        cout<< "Mask " << i << " : "<< argv[1] <<endl;

        argc--;
        argv++;
    }
    cout<<endl;


    const char *output_folder;
    output_folder = argv[1];
    argc--;
    argv++;


    int dilation_degree = 9;
    dilation_degree = atoi(argv[1]);
    argc--;
    argv++;


    int include_volume_option = atoi(argv[1]);
    argc--;
    argv++;


    if (include_volume_option > 0) {
        include_volume = true;
    }


    //whether we should use a template
    int template_selection_option = 0; //atoi(argv[1]);
    //argc--;
    //argv++;

    if (template_selection_option > 0) {

        template_stack.Read(argv[1]);
        use_template = true;

    }




    bool has_4D_stacks = false;

    for (i=0; i<stacks.size(); i++) {

        if (stacks[i].GetT()>1) {
            has_4D_stacks = true;
            break;
        }

    }


    if (has_4D_stacks) {

        cout << "Splitting stacks into dynamincs ... " << endl;

        Array<RealImage> new_stacks;
        Array<RealImage> new_masks;

        for (i=0; i<stacks.size(); i++) {

            if (stacks[i].GetT() == 1) {
                new_stacks.push_back(stacks[i]);
                new_masks.push_back(masks[i]);

            }
            else {
                for (int t=0; t<stacks[i].GetT(); t++) {

                    stack = stacks[i].GetRegion(0,0,0,t,stacks[i].GetX(),stacks[i].GetY(),stacks[i].GetZ(),(t+1));
                    new_stacks.push_back(stack);

                    RealImage tmp_mask = masks[i].GetRegion(0,0,0,t,masks[i].GetX(),masks[i].GetY(),masks[i].GetZ(),(t+1));
                    new_masks.push_back(tmp_mask);

                }

            }

        }


        nStacks = new_stacks.size();
        stacks.clear();
        masks.clear();

        cout << "New number of stacks : " << nStacks << endl;

        for (i=0; i<new_stacks.size(); i++) {

            stacks.push_back(new_stacks[i]);
            masks.push_back(new_masks[i]);
        }

        new_stacks.clear();
        new_masks.clear();

    }



    // -----------------------------------------------------------------------------


    cout << endl;
    cout << "Cropping stacks based on the input masks ... " << endl;


    Array<RealImage> new_stacks;
    Array<RealImage> new_masks;

    Array<double> volume_tmp_all;
    double volume_tmp_average = 0;
    double volume_tmp_stdev = 0;

    double num_tmp = 0;
    
    Array<double> volume_tmp_all_nonzero;

    cout << "Mask volumes : " << endl;

    for (i=0; i<stacks.size(); i++) {

        RealImage m_tmp = masks[i];
        double count = 0;
        for (int x = 0; x < m_tmp.GetX(); x++) {
            for (int y = 0; y < m_tmp.GetY(); y++) {
                for (int z = 0; z < m_tmp.GetZ(); z++) {
                    if (m_tmp(x,y,z) > 0.01) {
                        count = count + 1;
                    }
                }
            }
        }
        double volume_tmp = count*masks[i].GetXSize()*masks[i].GetYSize()*masks[i].GetZSize()/1000;

        double smin, smax;

        RealImage tmp_masked = stacks[i];
        tmp_masked *= masks[i];
        tmp_masked.GetMinMax(&smin, &smax);
        if (smax < 10) {
            volume_tmp = 0;
        }


        if (volume_tmp > 70) {
            volume_tmp_average = volume_tmp_average + volume_tmp;
            num_tmp = num_tmp + 1;
        }
        volume_tmp_all.push_back(volume_tmp);
        
        if (volume_tmp > 30) {
            volume_tmp_all_nonzero.push_back(volume_tmp);
        }
        
    }

    volume_tmp_average = volume_tmp_average / num_tmp;

    for (i=0; i<stacks.size(); i++) {
        double volume_diff = abs(volume_tmp_all[i] - volume_tmp_average);
        volume_tmp_stdev = volume_tmp_stdev + volume_diff*volume_diff;
        cout << " - " << i << " " << volume_tmp_all[i] << " cc" << endl;
    }

    volume_tmp_stdev = volume_tmp_stdev / stacks.size();
    volume_tmp_stdev = sqrt(volume_tmp_stdev);
    
    double median_volume = median_val(volume_tmp_all_nonzero); //volume_tmp_all);

    cout << " - average : " << volume_tmp_average << " +/- " << volume_tmp_stdev << " cc" <<  endl;
    cout << " - median : " << median_volume << " cc " << endl;

    double min_volume = 40;


    for (i=0; i<stacks.size(); i++) {

        RealImage m_tmp = masks[i];
        RealImage s_tmp = stacks[i];



        double volume_diff_test = abs(volume_tmp_all[i] - volume_tmp_average);

        double median_volume_selection_threshold = 0.5;
        double median_volume_diff_test = abs(volume_tmp_all[i] - median_volume);

//        if (volume_diff_test < 2.5*volume_tmp_stdev && volume_tmp_all[i] > volume_tmp_average*0.35) {
        if (median_volume_diff_test < median_volume_selection_threshold*median_volume) {

            RigidTransformation* tmp_r = new RigidTransformation();

            ConnectivityType i_connectivity = CONNECTIVITY_26;

            TransformMask(s_tmp, m_tmp, *tmp_r);

            RealImage dl_m_tmp = m_tmp;

            Dilate<RealPixel>(&dl_m_tmp, dilation_degree, i_connectivity);

            CropImage(s_tmp,dl_m_tmp);
            CropImage(m_tmp,dl_m_tmp);


            Dilate<RealPixel>(&m_tmp, 1, i_connectivity);

            masks[i] = m_tmp;
            stacks[i] = s_tmp;

            new_masks.push_back(m_tmp);
            new_stacks.push_back(s_tmp);


        } else {
            cout << " - stack " << i << " was excluded because of the small/large mask ROI " << endl;
        }

    }


    stacks.clear();
    masks.clear();



    for (i=0; i<new_stacks.size(); i++) {

        stacks.push_back(new_stacks[i]);
        masks.push_back(new_masks[i]);

    }


    new_stacks.clear();
    new_masks.clear();
    nStacks = stacks.size();


    cout << endl;
    if (!use_template) {
        cout << "Selection of the template and preliminary registrations based on the input masks ... " << endl;
    } else {
        cout << "Preliminary registration of all masked stacks to the template ... " << endl;
    }

    double norm_volume_local = 0;
    Array<double> all_volume_array;
    Array<double> all_slice_ncc_array;

    for (int i=0; i<stacks.size(); i++) {

        double mask_volume = 0;
        double slice_ncc = 0;

        cout << i << " ... " <<  endl;

        StackStats(stacks[i], masks[i], mask_volume, slice_ncc);

        all_volume_array.push_back(mask_volume);
        all_slice_ncc_array.push_back(slice_ncc);

        norm_volume_local = norm_volume_local + mask_volume;

    }

    norm_volume_local = norm_volume_local / stacks.size();


    for (int i=0; i<stacks.size(); i++) {
        all_volume_array[i] = all_volume_array[i] / norm_volume_local;
    }


    InterpolationMode interpolation = Interpolation_Linear;
    UniquePtr<InterpolateImageFunction> interpolator;
    interpolator.reset(InterpolateImageFunction::New(interpolation));
    //InterpolationMode interpolation_nn = Interpolation_NN;
    //interpolator.reset(InterpolateImageFunction::New(interpolation_nn));



    cout << "Stack metrics (normalised mask volume, slice NCC, average intersection volume, NCC with other stacks, combination of all metrics) : " << endl;

    Array<double> all_global_ncc_array;
    Array<double> all_global_volume_array;
    Array<double> all_global_stats_array;
    Array<Array<RigidTransformation>> prelim_stack_tranformations;

    double norm_volume = 0;


    for (int i=0; i<stacks.size(); i++) {

        double average_ncc = 0;
        double average_volume = 0;
        Array<RigidTransformation> current_stack_tranformations;

        cout << i << " ... " <<  endl;

        GlobalStackStats(stacks[i], masks[i], stacks, masks, average_ncc, average_volume, current_stack_tranformations);

        prelim_stack_tranformations.push_back(current_stack_tranformations);

        all_global_ncc_array.push_back(average_ncc);
        all_global_volume_array.push_back(average_volume);

        norm_volume = norm_volume + average_volume;
    }

    norm_volume = norm_volume / stacks.size();

    for (int i=0; i<stacks.size(); i++) {
        all_global_volume_array[i] = all_global_volume_array[i] / norm_volume;
    }

    int selected_template = 0;
    double max_global_stats = -1;

    double global_stats_average = 0;
    double count_stats = 0;


    for (int i=0; i<stacks.size(); i++) {

        double global_stats = all_global_volume_array[i] * all_global_ncc_array[i] * all_slice_ncc_array[i];
        all_global_stats_array.push_back(global_stats);


	if (include_volume) {

	   global_stats = global_stats*all_volume_array[i];

	}


        if (global_stats > 0) {
            global_stats_average = global_stats_average + global_stats;
            count_stats = count_stats + 1;
        }

        if (global_stats > max_global_stats && global_stats > 0) {
            max_global_stats = global_stats;
            selected_template = i;
        }

    }




    for (int i=0; i<stacks.size(); i++) {

        Matrix m = prelim_stack_tranformations[selected_template][i].GetMatrix();
        m.Invert();
        stacks[i].PutAffineMatrix(m, true);
        masks[i].PutAffineMatrix(m, true);

    }



    global_stats_average = global_stats_average / count_stats;

    new_stacks.clear();
    new_masks.clear();

    bool found_template = false;


    for (int i=0; i<stacks.size(); i++) {

        cout << " - " << i << " : " << all_volume_array[i] << " " << all_slice_ncc_array[i] << " " << all_global_volume_array[i] << " " << all_global_ncc_array[i] << " -> " << all_global_stats_array[i];

	//0.25
        if (all_global_stats_array[i] > global_stats_average*0.375 ) {

            new_masks.push_back(masks[i]);
            new_stacks.push_back(stacks[i]);

            if (i == selected_template && !found_template) {
                selected_template = new_stacks.size() - 1;
                found_template = true;
            }


        } else {
            cout << " - excluded ";
        }

        cout << endl;

    }




    stacks.clear();
    masks.clear();


    for (i=0; i<new_stacks.size(); i++) {

        stacks.push_back(new_stacks[i]);
        masks.push_back(new_masks[i]);

    }


    new_stacks.clear();
    new_masks.clear();
    nStacks = stacks.size();




    volume_tmp_all.clear();
    volume_tmp_average = 0;
    volume_tmp_stdev = 0;



    if (!use_template) {
        cout << " - selected template stack : " << selected_template << endl;
    } else {
        cout << " - identified best quality stack : " << selected_template << endl;
    }




    if (!use_template) {

        template_stack = stacks[selected_template];

    } else {

        RealImage masked_selected_template = stacks[selected_template];
        masked_selected_template *= masks[selected_template];

        RigidTransformation *r_init = new RigidTransformation();
        r_init->PutTranslationX(0.0001);
        r_init->PutTranslationY(0.0001);
        r_init->PutTranslationZ(-0.0001);

        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        Insert(params, "Background value for image 1", 0);

        GenericRegistrationFilter registration;
        registration.Parameter(params);
        registration.Input(&masked_selected_template, &template_stack);
        Transformation *dofout = nullptr;
        registration.Output(&dofout);
        registration.InitialGuess(r_init);
        registration.GuessParameter();
        registration.Run();
        unique_ptr<RigidTransformation> r_dofout(dynamic_cast<RigidTransformation*>(dofout));

        Matrix m = r_dofout->GetMatrix();

        for (int i=0; i<stacks.size(); i++) {
            stacks[i].PutAffineMatrix(m, true);
            masks[i].PutAffineMatrix(m, true);
        }
    }


    RealImage common_mask = masks[selected_template];

    if (use_template) {
        common_mask = template_stack;
    }


    common_mask *= 0;



    Resampling<RealPixel> resampling01(0.8,0.8,0.8);
    resampling01.Input(&common_mask);
    resampling01.Output(&common_mask);
    resampling01.Interpolator(interpolator.get());
    resampling01.Run();

    ImageAttributes attr = common_mask.Attributes();
    common_mask.PutPixelSize(attr._dx*1.25, attr._dy*1.25, attr._dz*1.25);



        double source_padding = 0;
        double target_padding = -inf;
        bool dofin_invert = false;
        bool twod = false;

    for (int i=0; i<stacks.size(); i++) {

        RealImage tr_mask = common_mask;

        MultiLevelFreeFormTransformation *mffd_init = new MultiLevelFreeFormTransformation;

        ImageTransformation *imagetransformation = new ImageTransformation;
        imagetransformation->Input(&masks[i]);
        imagetransformation->Transformation(mffd_init);
        imagetransformation->Output(&tr_mask);
        imagetransformation->TargetPaddingValue(target_padding);
        imagetransformation->SourcePaddingValue(source_padding);
        imagetransformation->Interpolator(interpolator.get());  // &nn);
        imagetransformation->TwoD(twod);
        imagetransformation->Invert(dofin_invert);
        imagetransformation->Run();

        common_mask += tr_mask;

    }

    common_mask /= stacks.size();



    RealImage common_mask2 = common_mask;

    Resampling<RealPixel> resampling0(0.8,0.8,0.8);
    resampling0.Input(&common_mask2);
    resampling0.Output(&common_mask);
    resampling0.Interpolator(interpolator.get());
    resampling0.Run();


    GaussianBlurring<RealPixel> gb2(1.0);
    gb2.Input(&common_mask);
    gb2.Output(&common_mask);
    gb2.Run();



    common_mask = ThresholdNormalisedMask(common_mask, 0.5);


    common_mask.Write("average_mask_cnn.nii.gz");


    cout << "Generated average mask for reconstruction ROI : average_mask_cnn.nii.gz " << endl;


    cout << endl;


    for (int i=0; i<stacks.size(); i++) {

        string output_path(output_folder);

        output_path = output_path + "/proc-stack-" + to_string(i) + ".nii.gz";

        cout << " - " << output_path << endl;

        stacks[i].Write(output_path.c_str());

    }


	RealImage final_template_stack = common_mask;


        MultiLevelFreeFormTransformation *mffd_init2 = new MultiLevelFreeFormTransformation;

        ImageTransformation *imagetransformation2 = new ImageTransformation;
        imagetransformation2->Input(&template_stack);
        imagetransformation2->Transformation(mffd_init2);
        imagetransformation2->Output(&final_template_stack);
        imagetransformation2->TargetPaddingValue(target_padding);
        imagetransformation2->SourcePaddingValue(source_padding);
        imagetransformation2->Interpolator(interpolator.get());
        imagetransformation2->TwoD(twod);
        imagetransformation2->Invert(dofin_invert);
        imagetransformation2->Run();



    final_template_stack.Write("selected_template.nii.gz");



    cout << "------------------------------------------------------" << endl;

    //The end of main()

    return 0;
}
