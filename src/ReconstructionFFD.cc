/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
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

#include "svrtk/ReconstructionFFD.h"
#include "svrtk/Profiling.h"
#include "svrtk/Parallel.h"

using namespace std;
using namespace mirtk;
using namespace svrtk::Utility;

namespace svrtk {

    //-------------------------------------------------------------------

    // run global ffd stack registration to the template

    void ReconstructionFFD::FFDStackRegistrations(Array<RealImage>& stacks, RealImage template_image, RealImage mask) {
        SVRTK_START_TIMING();
        
        GaussianBlurring<RealPixel> gb(1.2);
        ResamplingWithPadding<RealPixel> resampling(1.8, 1.8, 1.8, -1);
        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        resampling.Interpolator(&interpolator);
        resampling.Input(&template_image);
        resampling.Output(&template_image);
        resampling.Run();

        Array<RealImage> resampled_stacks;
        for (int i=0; i<stacks.size(); i++) {
            RealImage resampled_stack = stacks[i];
            gb.Input(&resampled_stack);
            gb.Output(&resampled_stack);
            gb.Run();
            RealImage tmp = resampled_stack;
            resampling.Input(&tmp);
            resampling.Output(&resampled_stack);
            resampling.Run();
            resampled_stacks.push_back(resampled_stack);
        }

        ParameterList params;
        Insert(params, "Transformation model", "FFD");
        Insert(params, "Control point spacing in X", 10);
        Insert(params, "Control point spacing in Y", 10);
        Insert(params, "Control point spacing in Z", 10);
        
        InterpolationMode interpolation_nn = Interpolation_NN;
        UniquePtr<InterpolateImageFunction> interpolator_nn;
        interpolator_nn.reset(InterpolateImageFunction::New(interpolation_nn));
        double source_padding = 0;
        double target_padding = -inf;
        bool dofin_invert = false;
        bool twod = false;
        
        for (int i=0; i<resampled_stacks.size(); i++) {
            
            GenericRegistrationFilter registration;
            registration.Parameter(params);
            registration.Input(&resampled_stacks[i], &template_image);
            Transformation *dofout = nullptr;
            registration.Output(&dofout);
    //            registration.InitialGuess(RigidTransformation());
            registration.GuessParameter();
            registration.Run();
            MultiLevelFreeFormTransformation* mffd_dofout;
            
            mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*>(dofout);
            mffd_dofout->Write((boost::format("ms-%1%.dof") % i).str().c_str());
    //            _global_mffd_transformations.push_back(mffd_dofout);

            RealImage transformed_main_mask = stacks[i];
            ImageTransformation *imagetransformation = new ImageTransformation;
            imagetransformation->Input(&mask);
            imagetransformation->Transformation(mffd_dofout);
            imagetransformation->Output(&transformed_main_mask);
            imagetransformation->TargetPaddingValue(target_padding);
            imagetransformation->SourcePaddingValue(source_padding);
            imagetransformation->Interpolator(interpolator_nn.get());
            imagetransformation->TwoD(twod);
            imagetransformation->Invert(dofin_invert);
            imagetransformation->Run();
            delete imagetransformation;
            
            CropImage(stacks[i], transformed_main_mask);
            stacks[i].Write((boost::format("fcropped-%1%.nii.gz") % i).str().c_str());
        }
        
        SVRTK_END_TIMING("FFDStackRegistrations");
        
    }

    // -----------------------------------------------------------------------------


} // namespace svrtk
