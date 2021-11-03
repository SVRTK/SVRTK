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

using namespace mirtk;

namespace svrtk {
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SetCP( int value, int step )
    {
        
        if (value > 4) {
    
            _cp_spacing.clear();
            
            for (int i=0; i<5; i++) {
                
                if (value > 4) {
                    _cp_spacing.push_back(value);
                    value -= step;
                }
                else { 
                    
                    _cp_spacing.push_back(5);
                }
                
            }

        }
        
        cout << "CP spacing : ";
        for (int i=0; i<_cp_spacing.size(); i++) {
            cout << _cp_spacing[i] << " ";
        }
        cout << endl;
    
    }
    
    //-------------------------------------------------------------------
    
    
    ReconstructionFFD::ReconstructionFFD()
    {
        _step = 0.0001;
        _debug = false;
        _quality_factor = 2;
        _sigma_bias = 12;
        _sigma_s = 0.025;
        _sigma_s2 = 0.025;
        _mix_s = 0.9;
        _mix = 0.9;
        _delta = 1;
        _lambda = 0.1;
        _alpha = (0.05 / _lambda) * _delta * _delta;
        _template_created = false;
        _have_mask = false;
        _low_intensity_cutoff = 0.01;
        _global_bias_correction = false;
        _adaptive = false;
        _robust_slices_only = false;

        
        _mffd_mode = true;
        
        _no_global_ffd_flag = false;
        
        _masked = false;

        
        _number_of_channels = 0;
        _multiple_channels_flag = false;
        
        
        _cp_spacing.push_back(15);
        _cp_spacing.push_back(10);
        _cp_spacing.push_back(5);
        _cp_spacing.push_back(5);

        _global_NCC_threshold = 0.7;
        _local_NCC_threshold = 0.5;
        _local_window_size = 20;

        _global_SSIM_threshold = -1;
        _local_SSIM_threshold = -1;

        _global_init_ffd = true;
        
        _excluded_stack = 99;
        
        _current_iteration = 0;
                
        int directions[13][3] = {
            { 1, 0, -1},
            { 0, 1, -1},
            { 1, 1, -1},
            { 1, -1, -1},
            { 1, 0, 0},
            { 0, 1, 0},
            { 1, 1, 0},
            { 1, -1, 0},
            { 1, 0, 1},
            { 0, 1, 1},
            { 1, 1, 1},
            { 1, -1, 1},
            { 0, 0, 1}
        };
        for (int i = 0; i < 13; i++)
            for (int j = 0; j < 3; j++)
                _directions[i][j] = directions[i][j];
        
    }
    
    //-------------------------------------------------------------------
    
    ReconstructionFFD::~ReconstructionFFD() { }


    //-------------------------------------------------------------------
    
    
    double ReconstructionFFD::CreateTemplate( RealImage* stack, double resolution )
    {
        double dx, dy, dz, d;
        
        ImageAttributes attr = stack->Attributes();
        
        attr._z += 2;
        
        RealImage enlarged;
        enlarged.Initialize(attr);
        
        if (resolution <= 0) {

            stack->GetPixelSize(&dx, &dy, &dz);
            if ((dx <= dy) && (dx <= dz))
                d = dx;
            else if (dy <= dz)
                d = dy;
            else
                d = dz;
        }
        else
            d = resolution;
        
        InterpolationMode interpolation = Interpolation_Linear;
        
        UniquePtr<InterpolateImageFunction> interpolator;
        interpolator.reset(InterpolateImageFunction::New(interpolation));
        
        Resampling<RealPixel> resampler(d, d, d);
        
        enlarged = *stack;
        
        resampler.Input(&enlarged);
        resampler.Output(&enlarged);
        resampler.Interpolator(interpolator.get());
        resampler.Run();
        
        _reconstructed = enlarged;
        _template_created = true;
        
        if (_debug)
            _reconstructed.Write("template.nii.gz");
        
        _reconstructed_attr = _reconstructed.Attributes();
        
        
        if (_multiple_channels_flag && _number_of_channels > 0) {
            
            enlarged = 0;
            for (int n=0; n<_number_of_channels; n++) {
                
                _mc_reconstructed.push_back(enlarged);
            }
            
        }
        
        
        return d;
    }


    //-------------------------------------------------------------------
    
    void ReconstructionFFD::NLMFiltering(Array<RealImage*> stacks)
    {
        char buffer[256];
        RealImage tmp_stack;

        NLDenoising *denoising = new NLDenoising;
        
        for (int i=0; i<stacks.size(); i++) {
            
            tmp_stack = *stacks[i];
            tmp_stack = denoising->Run(tmp_stack, 3, 1);
            
            RealImage *tmp_p = new RealImage(tmp_stack);
            stacks[i] = tmp_p;
            
            if (_debug) {
                sprintf(buffer, "denoised-%i.nii.gz", i);
                stacks[i]->Write(buffer);
            }
        }
        
    }
    
    //-------------------------------------------------------------------
    
    RealImage ReconstructionFFD::CreateMask( RealImage image )
    {
        //binarize mask
        RealPixel* ptr = image.Data();
        for (int i = 0; i < image.NumberOfVoxels(); i++) {
            *ptr = 1;
            ptr++;
        }
        return image;
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SetMask(RealImage *mask, double sigma, double threshold)
    {
        if (!_template_created) {
            cerr
            << "Please create the template before setting the mask, so that the mask can be resampled to the correct dimensions."
            << endl;
            exit(1);
        }
        
        _mask = _reconstructed;
        
        if (mask != NULL) {

            if (sigma > 0) {

                GaussianBlurring<RealPixel> gb(sigma);
                gb.Input(mask);
                gb.Output(mask);
                gb.Run();
                
                RealPixel* ptr = mask->Data();
                for (int i = 0; i < mask->NumberOfVoxels(); i++) {
                    if (*ptr > threshold)
                        *ptr = 1;
                    else
                        *ptr = 0;
                    ptr++;
                }
            }
            
            RigidTransformation transformation;
            ImageTransformation imagetransformation;

            InterpolationMode interpolation = Interpolation_NN;
            UniquePtr<InterpolateImageFunction> interpolator;
            interpolator.reset(InterpolateImageFunction::New(interpolation));
            
            imagetransformation.Input(mask);
            imagetransformation.Transformation(&transformation);
            imagetransformation.Output(&_mask);
            imagetransformation.TargetPaddingValue(-1);
            imagetransformation.SourcePaddingValue(0);
            imagetransformation.Interpolator(interpolator.get());
            imagetransformation.Run();
        }
        else {
            _mask = 1;
        }
        
        _have_mask = true;
        
        if (_debug)
            _mask.Write("transformed-mask.nii.gz");
        
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::TransformMask(RealImage& image, RealImage& mask, RigidTransformation& transformation)
    {
        ImageTransformation imagetransformation;
        
        InterpolationMode interpolation = Interpolation_NN;
        UniquePtr<InterpolateImageFunction> interpolator;
        interpolator.reset(InterpolateImageFunction::New(interpolation));
        
        imagetransformation.Input(&mask);
        imagetransformation.Transformation(&transformation);
        
        RealImage m = image;
        imagetransformation.Output(&m);
        imagetransformation.TargetPaddingValue(-1);
        imagetransformation.SourcePaddingValue(0);
        imagetransformation.Interpolator(interpolator.get());
        imagetransformation.Run();
        mask = m;
        
    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::TransformMaskP(RealImage* image, RealImage& mask)
    {
        
        ImageTransformation imagetransformation;
        RigidTransformation* rr = new RigidTransformation;
        InterpolationMode interpolation = Interpolation_NN;
        UniquePtr<InterpolateImageFunction> interpolator;
        interpolator.reset(InterpolateImageFunction::New(interpolation));
        
        imagetransformation.Input(&mask);
        imagetransformation.Transformation(rr);
        
        RealImage m = *image;
        imagetransformation.Output(&m);
        imagetransformation.TargetPaddingValue(-1);
        imagetransformation.SourcePaddingValue(0);
        imagetransformation.Interpolator(interpolator.get());
        imagetransformation.Run();
        mask = m;
        
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::ResetOrigin(GreyImage &image, RigidTransformation& transformation)
    {
        double ox,oy,oz;
        image.GetOrigin(ox,oy,oz);
        image.PutOrigin(0,0,0);
        transformation.PutTranslationX(ox);
        transformation.PutTranslationY(oy);
        transformation.PutTranslationZ(oz);
        transformation.PutRotationX(0);
        transformation.PutRotationY(0);
        transformation.PutRotationZ(0);
        
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::ResetOrigin(RealImage &image, RigidTransformation& transformation)
    {
        double ox,oy,oz;
        image.GetOrigin(ox,oy,oz);
        image.PutOrigin(0,0,0);
        transformation.PutTranslationX(ox);
        transformation.PutTranslationY(oy);
        transformation.PutTranslationZ(oz);
        transformation.PutRotationX(0);
        transformation.PutRotationY(0);
        transformation.PutRotationZ(0);
        
    }
    

    //-------------------------------------------------------------------
    
    void ReconstructionFFD::RestoreSliceIntensities()
    {
        int i;
        double factor;
        RealPixel *p;
        
        for (int inputIndex=0;inputIndex<_slices.size();inputIndex++) {
            factor = _stack_factor[_stack_index[inputIndex]];
            
            p = _slices[inputIndex]->Data();
            for(i=0; i<_slices[inputIndex]->NumberOfVoxels(); i++) {
                if(*p>0) *p = *p / factor;
                p++;
            }
        }
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::ScaleVolume()
    {
        double scalenum = 0, scaleden = 0;
        
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            RealImage slice = *_slices[inputIndex];
        
            RealImage w = *_weights[inputIndex];
            
            RealImage sim = *_simulated_slices[inputIndex];
            
            double sw = _slice_weight[inputIndex];
            
            if (_stack_index[inputIndex] == _excluded_stack)
                sw = 1;
            
            for (int i = 0; i < slice.GetX(); i++)
                for (int j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) > -1) {

                        if ( _simulated_weights[inputIndex]->GetAsDouble(i,j,0) > 0.99 ) {
                            scalenum += w(i, j, 0) * sw * slice(i, j, 0) * sim(i, j, 0);
                            scaleden += w(i, j, 0) * sw * sim(i, j, 0) * sim(i, j, 0);
                        }
                    }
        }
        
        double scale = scalenum / scaleden;
        
        if(_debug)
            cout<<"- scale = "<<scale;
        
        RealPixel *ptr = _reconstructed.Data();
        for(int i=0; i<_reconstructed.NumberOfVoxels(); i++) {
            if(*ptr>0) *ptr = *ptr * scale;
            ptr++;
        }
        cout<<endl;
    }
    
    //-------------------------------------------------------------------
    
    class ParallelSimulateSlicesFFD {
        ReconstructionFFD *reconstructor;
        
    public:
        ParallelSimulateSlicesFFD( ReconstructionFFD *_reconstructor ) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {

                reconstructor->_slice_inside[inputIndex] = false;

                POINT3D p;
                for ( int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++ )
                    for ( int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++ ) {
                    
                        reconstructor->_simulated_slices[inputIndex]->PutAsDouble(i, j, 0, 0);
                        reconstructor->_simulated_weights[inputIndex]->PutAsDouble(i, j, 0, 0);
                        reconstructor->_simulated_inside[inputIndex]->PutAsDouble(i, j, 0, 0);
                        
                        
                        if (reconstructor->_multiple_channels_flag) {
                            for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                reconstructor->_mc_simulated_slices[inputIndex][nc]->PutAsDouble(i, j, 0, 0);
                            }
                        }
                        

                        if ( reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) > -0.01 ) {
                            double weight = 0;
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            
                            double ss_sum = reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i, j, 0);
                            
                            Array<double> mc_ss_sum;
                            if (reconstructor->_multiple_channels_flag) {
                                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                    mc_ss_sum.push_back(reconstructor->_mc_simulated_slices[inputIndex][nc]->GetAsDouble(i, j, 0));
                                }
                            }
                            
                            
                            for ( int k = 0; k < n; k++ ) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                ss_sum = ss_sum + p.value * reconstructor->_reconstructed(p.x, p.y, p.z);
                                weight += p.value;
                                if ( reconstructor->_mask(p.x, p.y, p.z) > 0.1 ) {
                                    reconstructor->_simulated_inside[inputIndex]->PutAsDouble(i, j, 0, 1);
                                    reconstructor->_slice_inside[inputIndex] = true;
                                }
                                
                                if (reconstructor->_multiple_channels_flag) {
                                    for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                        mc_ss_sum[nc] = mc_ss_sum[nc] + p.value * reconstructor->_mc_reconstructed[nc](p.x, p.y, p.z);
                                    }
                                }
                                
                            }

                            if( weight > 0 ) {
                                double ss = ss_sum / weight;
                                
                                reconstructor->_simulated_slices[inputIndex]->PutAsDouble(i,j,0, ss);
                                reconstructor->_simulated_weights[inputIndex]->PutAsDouble(i,j,0, weight);
                                
                                if (reconstructor->_multiple_channels_flag) {
                                    for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                        double mc_ss = mc_ss_sum[nc] / weight;
                                        reconstructor->_mc_simulated_slices[inputIndex][nc]->PutAsDouble(i,j,0, mc_ss);
                                    }
                                }
                                
                            }
                        }
                
                    }
            }
        }
        
        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SimulateSlices()
    {
        ParallelSimulateSlicesFFD *p_sim = new ParallelSimulateSlicesFFD( this );
        (*p_sim)();
        
        delete p_sim;
        
    }
    
    
    
    //-------------------------------------------------------------------
    
    int DefaultNumberOfBinsS(const BaseImage *, double min_intensity, double max_intensity)
    {
        int nbins = iround(max_intensity - min_intensity) + 1;
        if (nbins > 256) nbins = 256;
        return nbins;
    }

    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SimulateStacks( Array<RealImage*>& stacks, bool simulate_excluded)
    {
        
        
        int n;
        RealImage sim;
        POINT3D p;
        double weight;
        
        int z, current_stack;
        z=-1;
        current_stack=-1;
        
        
        double slice_threshold = 0.6;
        
        if (simulate_excluded)
            slice_threshold = -10;
        
        if (_current_iteration < 2)
            slice_threshold = -10;

        
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            
            if (_stack_index[inputIndex] == _excluded_stack)
                _slice_weight[inputIndex] = 1;
            
            RealImage slice = *_slices[inputIndex];
            
            sim.Initialize( slice.Attributes() );
            sim = 0;
            
            if(_slice_weight[inputIndex]>slice_threshold)
            {
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -1) {
                            weight=0;
                            n = _volcoeffs[inputIndex][i][j].size();
                            for (int k = 0; k < n; k++) {
                                p = _volcoeffs[inputIndex][i][j][k];
                                sim(i, j, 0) += p.value * _reconstructed(p.x, p.y, p.z);
                                weight += p.value;
                            }
                            if(weight>0.99)
                                sim(i,j,0)/=weight;
                            
                            double f = exp(-(_bias[inputIndex])->GetAsDouble(i, j, 0)) * _scale[inputIndex];
                            
                            if (f>0)
                                sim(i,j,0) /= f;
                            
                            
                        }
            }
            
            if (_stack_index[inputIndex] == _excluded_stack)
                _slice_weight[inputIndex] = 0;
            
            
            if (_stack_index[inputIndex]==current_stack)
                z++;
            else {
                current_stack=_stack_index[inputIndex];
                z=0;
            }
            
            for(int i=0; i<sim.GetX(); i++)
                for(int j=0; j<sim.GetY(); j++) {
                    stacks[_stack_index[inputIndex]]->PutAsDouble(i,j,z,sim(i,j,0));
                }

        }
        
        
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

            if (_stack_index[inputIndex] == _excluded_stack)
                _slice_weight[inputIndex] = 0;
        }
         
         
        
        
    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::MatchStackIntensities( Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, double averageValue, bool together )
    {

        double sum, num;
        char buffer[256];
        unsigned int ind;
        int i, j, k;
        double x, y, z;
        Array<double> stack_average;
        
        _average_value = averageValue;
        
        for (ind = 0; ind < stacks.size(); ind++) {
            sum = 0;
            num = 0;
            for (i = 0; i < stacks[ind].GetX(); i++)
                for (j = 0; j < stacks[ind].GetY(); j++)
                    for (k = 0; k < stacks[ind].GetZ(); k++) {
  
                        x = i;
                        y = j;
                        z = k;

                        stacks[ind].ImageToWorld(x, y, z);
                        stack_transformations[ind].Transform(x, y, z);
                        _mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);

                        if ( stacks[ind](i, j, k) > 0 ) {
                            sum += stacks[ind](i, j, k);
                            num++;
                        }

                    }

            if (num > 0)
                stack_average.push_back(sum / num);
            else {
                cerr << "Stack " << ind << " has no overlap with ROI" << endl;
                exit(1);
            }
        }
        
        double global_average;
        if (together) {
            global_average = 0;
            for(i=0;i<stack_average.size();i++)
                global_average += stack_average[i];
            global_average/=stack_average.size();
        }
        
        if (_debug) {
            cout << "Stack average intensities are ";
            for (ind = 0; ind < stack_average.size(); ind++)
                cout << stack_average[ind] << " ";
            cout << endl;
            cout << "The new average value is " << averageValue << endl;
        }
        
        RealPixel *ptr;
        double factor;
        for (ind = 0; ind < stacks.size(); ind++) {
            if (together) {
                factor = averageValue / global_average;
                _stack_factor.push_back(factor);
            }
            else {
                factor = averageValue / stack_average[ind];
                _stack_factor.push_back(factor);
                
            }
            
            ptr = stacks[ind].Data();
            for (i = 0; i < stacks[ind].NumberOfVoxels(); i++) {
                if (*ptr > 0)
                    *ptr *= factor;
                ptr++;
            }
        }
        
        if (_debug) {
            for (ind = 0; ind < stacks.size(); ind++) {
                sprintf(buffer, "rescaled-stack%i.nii.gz", ind);
                stacks[ind].Write(buffer);
            }
            
            cout << "Slice intensity factors are ";
            for (ind = 0; ind < stack_average.size(); ind++)
                cout << _stack_factor[ind] << " ";
            cout << endl;
            cout << "The new average value is " << averageValue << endl;
        }
        
    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::MatchStackIntensitiesWithMasking( Array<RealImage*> stacks, Array<RigidTransformation>& stack_transformations, double averageValue, bool together )
    {

        double sum, num;
        char buffer[256];
        unsigned int ind;
        int i, j, k;
        double x, y, z;
        Array<double> stack_average;
        RealImage m;
        
        _average_value = averageValue;
        
        for (ind = 0; ind < stacks.size(); ind++) {
            m = *stacks[ind];
            m = 0;
            sum = 0;
            num = 0;
            for (i = 0; i < stacks[ind]->GetX(); i++)
                for (j = 0; j < stacks[ind]->GetY(); j++)
                    for (k = 0; k < stacks[ind]->GetZ(); k++) {

                        x = i;
                        y = j;
                        z = k;

                        stacks[ind]->ImageToWorld(x, y, z);
                        stack_transformations[ind].Transform(x, y, z);
                        _mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);

                        if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0)
                            && (z < _mask.GetZ()))
                        {
                            if (_mask(x, y, z) > 0 && _mask(x, y, z) < 2 )
                            {
                                m(i,j,k)=1;
                                sum += stacks[ind]->GetAsDouble(i, j, k);
                                num++;
                            }
                            else
                                m(i,j,k)=0;
                        }
                        
                    }
            if(_debug)
            {
                sprintf(buffer,"mask-for-matching%i.nii.gz",ind);
                m.Write(buffer);
            }

            if (num > 0)
                stack_average.push_back(sum / num);
            else {
                cerr << "Stack " << ind << " has no overlap with ROI" << endl;
                exit(1);
            }
        }
        
        double global_average;
        if (together) {
            global_average = 0;
            for(i=0;i<stack_average.size();i++)
                global_average += stack_average[i];
            global_average/=stack_average.size();
        }
        
        if (_debug) {
            cout << "Stack average intensities are ";
            for (ind = 0; ind < stack_average.size(); ind++)
                cout << stack_average[ind] << " ";
            cout << endl;
            cout << "The new average value is " << averageValue << endl;
        }
        
        RealPixel *ptr;
        double factor;
        for (ind = 0; ind < stacks.size(); ind++) {
            if (together) {
                factor = averageValue / global_average;
                _stack_factor.push_back(factor);
            }
            else {
                factor = averageValue / stack_average[ind];
                _stack_factor.push_back(factor);
                
            }
            
            ptr = stacks[ind]->Data();
            for (i = 0; i < stacks[ind]->NumberOfVoxels(); i++) {
                if (*ptr > 0)
                    *ptr *= factor;
                ptr++;
            }
        }
        
        if (_debug) {
            for (ind = 0; ind < stacks.size(); ind++) {
                sprintf(buffer, "rescaled-stack%i.nii.gz", ind);
                stacks[ind]->Write(buffer);
            }
            
            cout << "Slice intensity factors are ";
            for (ind = 0; ind < stack_average.size(); ind++)
                cout << _stack_factor[ind] << " ";
            cout << endl;
            cout << "The new average value is " << averageValue << endl;
        }
        
    }
    

    //-------------------------------------------------------------------

    
    void ReconstructionFFD::CreateSlicesAndTransformationsFFD( Array<RealImage*> stacks, Array<RigidTransformation> &stack_transformations, Array<double> thickness, int d_packages, Array<int> selected_slices, int template_number )
    {

        int start_stack_number = template_number;
        int stop_stack_number = start_stack_number+1;

        _template_slice_number = 0;
        
        bool use_selected = true;

        int Z_dim = stacks[template_number]->GetZ();
        int package_index = 0;
        

        for (unsigned int i = start_stack_number; i < stop_stack_number; i++) {
            
            ImageAttributes attr = stacks[i]->Attributes();
            
            int selected_slice_number;

            if (!use_selected)
                selected_slice_number = attr._z;
            else
                selected_slice_number = selected_slices.size();
            

            for (int j = 0; j < selected_slice_number; j++) {
                
                if (selected_slices[j]>Z_dim)
                    break;

                RealImage slice;

                if (!use_selected) {
                    slice = stacks[i]->GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                    
                }
                else {
                    slice = stacks[i]->GetRegion(0, 0, selected_slices[j], stacks[i]->GetX(), stacks[i]->GetY(), selected_slices[j] + 1);

                }

                if (package_index<(d_packages-1) && j!=0)
                    package_index++;
                else
                    package_index = 0; 

                
                RealPixel smin, smax;
                slice.GetMinMax(&smin, &smax);

                 _zero_slices.push_back(1);

                _min_val.push_back(smin);
                _max_val.push_back(smax);

                slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);

                GreyImage grey_slice = slice;
                
                GreyImage *p_grey_slice = new GreyImage(grey_slice);
                RealImage *p_slice = new RealImage(slice);
                
                
                _grey_slices.push_back(p_grey_slice);
                _slices.push_back(p_slice);
                
                slice = 1;
                RealImage *p_simulated_slice = new RealImage(slice);
                _simulated_slices.push_back(p_simulated_slice);
                
                RealImage *p_simulated_weight = new RealImage(slice);
                _simulated_weights.push_back(p_simulated_weight);
                
                RealImage *p_simulated_inside = new RealImage(slice);
                _simulated_inside.push_back(p_simulated_inside);
                
                _stack_index.push_back(i);
                
                _transformations.push_back(stack_transformations[i]);
                
                RigidTransformation *rigidTransf = new RigidTransformation;
                _package_transformations.push_back(*rigidTransf);
                delete rigidTransf;
                
                
                RealImage *p_weight = new RealImage(slice);
                _weights.push_back(p_weight);
                
                RealImage *p_bias = new RealImage(slice);
                _bias.push_back(p_bias);
                
                
                
                _scale.push_back(1);
                _slice_weight.push_back(1);
                
                
                ImageAttributes slice_attr = slice.Attributes();
                _slice_attr.push_back(slice_attr);
                
                _negative_J_values.push_back(0);
                _slice_cc.push_back(1);
                _reg_slice_weight.push_back(1);
                _stack_locs.push_back(j);

                slice = 1;
                RealImage *p_dif = new RealImage(slice);
                _slice_dif.push_back(p_dif);
                
                RealImage *p_2dncc = new RealImage(slice);
                _slice_2dncc.push_back(p_2dncc);
                
                GreyImage *p_mask = new GreyImage(grey_slice);
                _slice_masks.push_back(p_mask);
                
                _excluded_points.push_back(0);

                _structural_slice_weight.push_back(1);
                _slice_ssim.push_back(1);

                if (d_packages==1)
                    _package_index.push_back(i);
                else {

                    _package_index.push_back(i*d_packages + package_index);
                }

                MultiLevelFreeFormTransformation *mffd = new MultiLevelFreeFormTransformation();
                _mffd_transformations.push_back(mffd);

            }

            if (i == template_number)
                _template_slice_number = _slices.size();
        }
        
    }
    




    void ReconstructionFFD::CreateSlicesAndTransformationsFFDMC( Array<RealImage*> stacks, Array<Array<RealImage*>> mc_stacks, Array<RigidTransformation> &stack_transformations, Array<double> thickness, int d_packages, Array<int> selected_slices, int template_number )
    {
        
        int start_stack_number = template_number;
        int stop_stack_number = start_stack_number+1;
        
        _template_slice_number = 0;
        
        bool use_selected = true;
        
        
        
        int Z_dim = stacks[template_number]->GetZ();
        int package_index = 0;

        for (unsigned int i = start_stack_number; i < stop_stack_number; i++) {

            ImageAttributes attr = stacks[i]->Attributes();

            int selected_slice_number;
            
            
            if (!use_selected)
            selected_slice_number = attr._z;
            else
            selected_slice_number = selected_slices.size();
            
            
            for (int j = 0; j < selected_slice_number; j++) {
                
                if (selected_slices[j]>Z_dim)
                break;
                
                RealImage slice;

                if (!use_selected) {
                    slice = stacks[i]->GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                }
                else {
                    slice = stacks[i]->GetRegion(0, 0, selected_slices[j], stacks[i]->GetX(), stacks[i]->GetY(), selected_slices[j] + 1);
                }
                
                if (package_index<(d_packages-1) && j!=0)
                package_index++;
                else
                package_index = 0;
                
                
                RealPixel smin, smax;
                slice.GetMinMax(&smin, &smax);

                _zero_slices.push_back(1);
                
                _min_val.push_back(smin);
                _max_val.push_back(smax);
                
                slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                
                Array<RealImage*> tmp_mc_slices;
                Array<RealImage*> tmp_mc_simulated_slices;
                Array<RealImage*> tmp_mc_diff_slices;
                
                for (int n=0; n<_number_of_channels; n++) {
                    
                    RealImage slice_mc;
                    
                    slice_mc = mc_stacks[n][i]->GetRegion(0, 0, selected_slices[j], stacks[i]->GetX(), stacks[i]->GetY(), selected_slices[j] + 1);
                    
                    slice_mc.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                    
                    RealImage *p_mc_slice = new RealImage(slice_mc);
                    
                    slice_mc = 1;
                    RealImage *p_mc_sim_slice = new RealImage(slice_mc);
                    
                    slice_mc = 1;
                    RealImage *p_mc_diff_slice = new RealImage(slice_mc);
                    
                    tmp_mc_slices.push_back(p_mc_slice);
                    tmp_mc_simulated_slices.push_back(p_mc_sim_slice);
                    tmp_mc_diff_slices.push_back(p_mc_diff_slice);

                }

                _mc_slices.push_back(tmp_mc_slices);
                _mc_simulated_slices.push_back(tmp_mc_simulated_slices);
                _mc_slice_dif.push_back(tmp_mc_diff_slices);

                
                GreyImage grey_slice = slice;
                
                GreyImage *p_grey_slice = new GreyImage(grey_slice);
                RealImage *p_slice = new RealImage(slice);
                
                
                _grey_slices.push_back(p_grey_slice);
                _slices.push_back(p_slice);
                
                
                slice = 1;
                RealImage *p_simulated_slice = new RealImage(slice);
                _simulated_slices.push_back(p_simulated_slice);
                
                RealImage *p_simulated_weight = new RealImage(slice);
                _simulated_weights.push_back(p_simulated_weight);
                
                RealImage *p_simulated_inside = new RealImage(slice);
                _simulated_inside.push_back(p_simulated_inside);
                
                _stack_index.push_back(i);
                
                _transformations.push_back(stack_transformations[i]);
                
                RigidTransformation *rigidTransf = new RigidTransformation;
                _package_transformations.push_back(*rigidTransf);
                delete rigidTransf;
                
                
                RealImage *p_weight = new RealImage(slice);
                _weights.push_back(p_weight);
                
                RealImage *p_bias = new RealImage(slice);
                _bias.push_back(p_bias);
                
                
                
                _scale.push_back(1);
                _slice_weight.push_back(1);
                
                
                ImageAttributes slice_attr = slice.Attributes();
                _slice_attr.push_back(slice_attr);
                
                _negative_J_values.push_back(0);
                _slice_cc.push_back(1);
                _reg_slice_weight.push_back(1);
                _stack_locs.push_back(j);
                
                slice = 1;
                RealImage *p_dif = new RealImage(slice);
                _slice_dif.push_back(p_dif);
                
                RealImage *p_2dncc = new RealImage(slice);
                _slice_2dncc.push_back(p_2dncc);
                
                GreyImage *p_mask = new GreyImage(grey_slice);
                _slice_masks.push_back(p_mask);
                
                _excluded_points.push_back(0);
                
                _structural_slice_weight.push_back(1);
                _slice_ssim.push_back(1);
                
                if (d_packages==1)
                _package_index.push_back(i);
                else {
                    
                    _package_index.push_back(i*d_packages + package_index);
                }
                
                
                MultiLevelFreeFormTransformation *mffd = new MultiLevelFreeFormTransformation();
                _mffd_transformations.push_back(mffd);
                

            }
            
            if (i == template_number)
            _template_slice_number = _slices.size();
        }
        
    }
    
    


    
 
    void ReconstructionFFD::ResetSlicesAndTransformationsFFD()
    {

        _grey_slices.clear();
        _slices.clear();
        _simulated_slices.clear();
        _simulated_weights.clear();
        _simulated_inside.clear();
        _stack_index.clear();
        _transformations.clear();
        _mffd_transformations.clear();
        _probability_maps.clear();
        _negative_J_values.clear();
        _package_index.clear();
        _slice_cc.clear();
        _slice_dif.clear();
        _slice_2dncc.clear();

    }
    
    
    //-------------------------------------------------------------------


    void ReconstructionFFD::RigidStackRegistration( Array<GreyImage*> stacks, GreyImage* template_stack, Array<RigidTransformation>& stack_transformations, bool init_reset )
    {
        char buffer[256];

        double tx, ty, tz, rx, ry, rz;
        double x, y, z;

        x = 0;
        y = 0;
        z = 0;
        
        GreyImage tmp_template = *template_stack;
        
        RigidTransformation r_init = stack_transformations[0];
        
        if (init_reset) {
            tmp_template.GetOrigin(x, y, z);
            tmp_template.PutOrigin(0, 0, 0);

            r_init.PutTranslationX(-x);
            r_init.PutTranslationY(-y);
            r_init.PutTranslationZ(-z);
            r_init.PutRotationX(0);
            r_init.PutRotationY(0);
            r_init.PutRotationZ(0);
        }

        ParameterList params;
        Insert(params, "Transformation model", "Rigid"); 
        Insert(params, "Background value for image 2", 0);
//        Insert(params, "Background value for image 2", 0);

        for (int i=0; i<stacks.size(); i++) {

            GenericRegistrationFilter *registration = new GenericRegistrationFilter();
            registration->Parameter(params);

            registration->Input(stacks[i], &tmp_template);
            Transformation *dofout = nullptr;
            registration->Output(&dofout);
            registration->InitialGuess(&(r_init));
            registration->GuessParameter();
            registration->Run();

            RigidTransformation *r_dofout = dynamic_cast<RigidTransformation*> (dofout);
           
            stack_transformations[i] = *r_dofout;

            tx = stack_transformations[i].GetTranslationX();
            ty = stack_transformations[i].GetTranslationY();
            tz = stack_transformations[i].GetTranslationZ();
            rx = stack_transformations[i].GetRotationX();
            ry = stack_transformations[i].GetRotationY();
            rz = stack_transformations[i].GetRotationZ();

            if (init_reset){
                stack_transformations[i].PutTranslationX(tx+x);
                stack_transformations[i].PutTranslationY(ty+y);
                stack_transformations[i].PutTranslationZ(tz+z);
            }
   
        }

        
        for (int i=0; i<stack_transformations.size(); i++) {
            sprintf(buffer, "rs-%i.dof", i);
            stack_transformations[i].Write(buffer);
         
            double tx, ty, tz, rx, ry, rz;
            tx = stack_transformations[i].GetTranslationX();
            ty = stack_transformations[i].GetTranslationY();
            tz = stack_transformations[i].GetTranslationZ();
            rx = stack_transformations[i].GetRotationX();
            ry = stack_transformations[i].GetRotationY();
            rz = stack_transformations[i].GetRotationZ();
         
            cout << "- # " << i << " : " << tx << " " << ty << " " << tz << " | " << rx << " " << ry << " " << rz << endl;

        }

    }

    //-------------------------------------------------------------------------------
    

    void ReconstructionFFD::RigidPackageRegistration( Array<GreyImage*> stacks, GreyImage* template_stack, int nPackages, Array<RigidTransformation>& stack_transformations, bool init_reset )
    {

        char buffer[256];
        
        if (nPackages>1) {
            
            
            Array<GreyImage*> packages;
            Array<int> package_index;
            Array<RigidTransformation> package_transformations;
            

            for ( int stackIndex=0; stackIndex<stacks.size(); stackIndex++ ) {

                GreyImage org_stack = *stacks[stackIndex];
                
                packages.clear();
                package_index.clear();
                package_transformations.clear();

                // .....................................................................
                
                ImageAttributes attr = org_stack.Attributes();
                
                for (int i=0; i<attr._z; i++)
                    package_index.push_back(0);
                
                int pkg_z = attr._z / nPackages;
                double pkg_dz = attr._dz*nPackages;
                
                double x, y, z, sx, sy, sz, ox, oy, oz;
                
                for (int l = 0; l < nPackages; l++) {
                    attr = org_stack.Attributes();
                    if ((pkg_z * nPackages + l) < attr._z)
                        attr._z = pkg_z + 1;
                    else
                        attr._z = pkg_z;
                    attr._dz = pkg_dz;
                    
                    GreyImage stack(attr);
                    stack.GetOrigin(ox, oy, oz);
                    
                    for (int k = 0; k < stack.GetZ(); k++)
                        for (int j = 0; j < stack.GetY(); j++)
                            for (int i = 0; i < stack.GetX(); i++) {
                                stack.Put(i, j, k, org_stack(i, j, (k*nPackages+l)));
                                package_index[(k*nPackages+l)] = l;
                            }
                    
                    x = 0;
                    y = 0;
                    z = l;
                    org_stack.ImageToWorld(x, y, z);
                    sx = 0;
                    sy = 0;
                    sz = 0;
                    stack.PutOrigin(ox, oy, oz);
                    stack.ImageToWorld(sx, sy, sz);
                    stack.PutOrigin(ox + (x - sx), oy + (y - sy), oz + (z - sz));
                    sx = 0;
                    sy = 0;
                    sz = 0;
                    stack.ImageToWorld(sx, sy, sz);
                    
                    
                    GreyImage *tmp_grey_stack = new GreyImage(stack);
                    packages.push_back(tmp_grey_stack);
                    
                    package_transformations.push_back(stack_transformations[stackIndex]);
                    
                    double tx, ty, tz, rx, ry, rz;
                    
                    tx = stack_transformations[stackIndex].GetTranslationX();
                    ty = stack_transformations[stackIndex].GetTranslationY();
                    tz = stack_transformations[stackIndex].GetTranslationZ();
                    rx = stack_transformations[stackIndex].GetRotationX();
                    ry = stack_transformations[stackIndex].GetRotationY();
                    rz = stack_transformations[stackIndex].GetRotationZ();
                    
                }
                
                // .....................................................................
                
                
                RigidStackRegistration(packages, template_stack, package_transformations, false);
                
                for (int i=0; i<packages.size(); i++) {
                    
                    if (_debug) {
//                        sprintf(buffer, "rp-%i-%i.dof", stackIndex, i);
//                        package_transformations[i].Write(buffer);
                        
//                        sprintf(buffer, "p-%i-%i.nii.gz", stackIndex, i);
//                        packages[i]->Write(buffer);
                    }
                }
                
                packages.clear();
                
                
                for (int i=0; i<_slices.size(); i++) {
                    
                    if (_stack_index[i] == stackIndex) {
                        for (int j=0; j<package_index.size(); j++) {
                            
                            if (_stack_locs[i] == j) {
                                
                                double tx, ty, tz, rx, ry, rz;
                                tx = package_transformations[package_index[j]].GetTranslationX();
                                ty = package_transformations[package_index[j]].GetTranslationY();
                                tz = package_transformations[package_index[j]].GetTranslationZ();
                                rx = package_transformations[package_index[j]].GetRotationX();
                                ry = package_transformations[package_index[j]].GetRotationY();
                                rz = package_transformations[package_index[j]].GetRotationZ();
                                
                                _package_transformations[i].PutTranslationX(tx);
                                _package_transformations[i].PutTranslationY(ty);
                                _package_transformations[i].PutTranslationZ(tz);
                                _package_transformations[i].PutRotationX(rx);
                                _package_transformations[i].PutRotationY(ry);
                                _package_transformations[i].PutRotationZ(rz);
                                
                            }
                        }
                    }
                }
                
                // .....................................................................
                
            }
        
            
        }
        else{
            
            
            for (int i=0; i<_slices.size(); i++) {
                
                
                double tx, ty, tz, rx, ry, rz;
                tx = stack_transformations[_stack_index[i]].GetTranslationX();
                ty = stack_transformations[_stack_index[i]].GetTranslationY();
                tz = stack_transformations[_stack_index[i]].GetTranslationZ();
                rx = stack_transformations[_stack_index[i]].GetRotationX();
                ry = stack_transformations[_stack_index[i]].GetRotationY();
                rz = stack_transformations[_stack_index[i]].GetRotationZ();
                
                
                _package_transformations[i].PutTranslationX(tx);
                _package_transformations[i].PutTranslationY(ty);
                _package_transformations[i].PutTranslationZ(tz);
                _package_transformations[i].PutRotationX(rx);
                _package_transformations[i].PutRotationY(ry);
                _package_transformations[i].PutRotationZ(rz);
                
                
            }
            
            
        }

        
    }


    //-------------------------------------------------------------------

    void ReconstructionFFD::FFDStackRegistration( Array<GreyImage*> stacks, GreyImage* template_stack, Array<RigidTransformation> stack_transformations )
    {

        char buffer[256];
        
        _global_mffd_transformations.clear();
        
        
        if (!_no_global_ffd_flag) {


            GaussianBlurring<GreyPixel> gb(1.2);
            
            ResamplingWithPadding<GreyPixel> resampling(1.8, 1.8, 1.8, -1);
            GenericLinearInterpolateImageFunction<GreyImage> interpolator;
            resampling.Interpolator(&interpolator);

            
            Array<GreyImage*> resampled_stacks;
       
            for (int i=0; i<stacks.size(); i++) {
                
                GreyImage stack = *stacks[i];

                gb.Input(&stack);
                gb.Output(&stack);
                gb.Run();
                
                GreyImage tmp = stack;
                resampling.Input(&tmp);
                resampling.Output(&stack);
                resampling.Run();
                
                resampled_stacks.push_back(new GreyImage(stack));

            }

            GreyImage source = *template_stack;
            resampling.Input(&source);
            resampling.Output(&source);
            resampling.Run();

            
            ParameterList params;
            Insert(params, "Transformation model", "FFD"); // SVFFD
            //        Insert(params, "Optimisation method", "GradientDescent");
            
            //        Insert(params, "Image (dis-)similarity measure", "NMI");
            //        Insert(params, "No. of bins", 128);
            
            //        Insert(params, "Image (dis-)similarity measure", "NCC"); //NCC
            //        string type = "sigma";
            //        string units = "mm";
            //        double width = 30;
            //        Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);
            
            //        Insert(params, "Image interpolation mode", "Linear");
            
            //        Insert(params, "Background value for image 1", -1);
            //        Insert(params, "Background value for image 2", -1);
            
            Insert(params, "Control point spacing in X", 10);
            Insert(params, "Control point spacing in Y", 10);
            Insert(params, "Control point spacing in Z", 10);
            
            
            
            for (int i=0; i<resampled_stacks.size(); i++) {

                GenericRegistrationFilter *registration = new GenericRegistrationFilter();
                registration->Parameter(params);


                registration->Input( resampled_stacks[i], &source);
                Transformation *dofout = nullptr;
                registration->Output(&dofout);

                registration->InitialGuess(&(stack_transformations[i]));


                registration->GuessParameter();
                registration->Run();

                
                MultiLevelFreeFormTransformation *mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*> (dofout);
                

                sprintf(buffer, "ms-%i.dof", i);
                mffd_dofout->Write(buffer);

                // .............................................................

    //            RealImage d_stack = target; //source;
    //            d_stack = 0;
    //
    //            ImageAttributes attr = d_stack.Attributes();
    //            attr._t = 3;
    //
    //            RealImage d_xyz(attr);
    //
    //            mffd_dofout->InverseDisplacement(d_xyz);
    //
    //
    //            for (int j=0; j<3; j++) {
    //
    //                RealImage tmp = d_xyz.GetRegion(0,0,0,j,d_xyz.GetX(),d_xyz.GetY(),d_xyz.GetZ(),(j+1));
    //                d_stack += tmp*tmp;
    //            }
    //
    //            sprintf(buffer,"ssr-displacement-%i.nii.gz", i);
    //            d_stack.Write(buffer);
                

                delete registration;

            }

        } else {
            
            for (int i=0; i<stacks.size(); i++) {
            
                MultiLevelFreeFormTransformation *mffd_dofout = new MultiLevelFreeFormTransformation();
                
                sprintf(buffer, "ms-%i.dof", i);
                mffd_dofout->Write(buffer);
            }
            
            
        }
            
            
            
    }
    
    
//-------------------------------------------------------------------

    
    class ParallelCreateSliceMasks {
        
    public:
        
        ReconstructionFFD *reconstructor;
        GreyImage main_mask;
        
        ParallelCreateSliceMasks( ReconstructionFFD *in_reconstructon, GreyImage in_main_mask ) :
        reconstructor(in_reconstructon),
        main_mask(in_main_mask) { }
        
        
        void operator() (const blocked_range<size_t> &r) const {
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                char buffer[256];
                ConnectivityType i_connectivity = CONNECTIVITY_26;
                
                if (!reconstructor->_masked || reconstructor->_current_iteration > 0) {
                    
                    char buffer[256];
                    bool init = false;
                    int current_stack_index = 0;
                    
                    GaussianBlurring<GreyPixel> gb(1.7);
                    
                    MultiLevelFreeFormTransformation *mffd_init;
                    
                    if (reconstructor->_reg_slice_weight[inputIndex] > 0) {
                        
                        if (reconstructor->_current_iteration < 0) {
                            
                            mffd_init = new MultiLevelFreeFormTransformation(reconstructor->_package_transformations[inputIndex]);
                        }
                        
                        
                        InterpolationMode interpolation = Interpolation_NN;
                        UniquePtr<InterpolateImageFunction> interpolator;
                        interpolator.reset(InterpolateImageFunction::New(interpolation));
                        
                        
                        GreyImage input_volume = main_mask;
                        
                        Erode<GreyPixel>(&input_volume, 2, i_connectivity);
                        Dilate<GreyPixel>(&input_volume, 5, i_connectivity);
                        
                        GreyImage output_volume = *reconstructor->_grey_slices[inputIndex];
                        
                        double source_padding = 0;
                        double target_padding = -inf;
                        bool dofin_invert = false;
                        bool twod = false;
                        
                        if (reconstructor->_masked) {
                            target_padding = 0;
                        }
                        
                        ImageTransformation *imagetransformation = new ImageTransformation;
                        
                        imagetransformation->Input(&input_volume);
                        
                        if (reconstructor->_current_iteration < 0)
                            imagetransformation->Transformation(mffd_init);
                        else
                            imagetransformation->Transformation(reconstructor->_mffd_transformations[inputIndex]);

                        imagetransformation->Output(&output_volume);
                        imagetransformation->TargetPaddingValue(target_padding);
                        imagetransformation->SourcePaddingValue(source_padding);
                        imagetransformation->Interpolator(interpolator.get());
                        imagetransformation->TwoD(twod);
                        imagetransformation->Invert(dofin_invert);
                        imagetransformation->Run();
                        
                        delete imagetransformation;

                        gb.Input(&output_volume);
                        gb.Output(&output_volume);
                        gb.Run();

                        for (int x=0; x<output_volume.GetX(); x++) {
                            for (int y=0; y<output_volume.GetY(); y++) {
                                
                                if (output_volume(x,y,0) > 0.5) {
                                    output_volume(x,y,0) = 1;
                                    reconstructor->_slice_masks[inputIndex]->PutAsDouble(x,y,0,1);
                                }
                                else {
                                    output_volume(x,y,0) = 0;
                                    reconstructor->_slice_masks[inputIndex]->PutAsDouble(x,y,0,0);
                                }

                            }
                        }

                    }
                
                }
                else {
                    
                    for (int x=0; x<reconstructor->_slice_masks[inputIndex]->GetX(); x++) {
                        for (int y=0; y<reconstructor->_slice_masks[inputIndex]->GetY(); y++) {
                            
                            reconstructor->_slice_masks[inputIndex]->PutAsDouble(x,y,0,1);
                        }
                    }
                    
                    Erode<GreyPixel>(reconstructor->_slice_masks[inputIndex], 5, i_connectivity);
                    
                }

                
            }
            
        }
        
        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::FastCreateSliceMasks(GreyImage main_mask, int iter)
    {
    
        ParallelCreateSliceMasks *p_m = new ParallelCreateSliceMasks(this, main_mask);
        (*p_m)();
        delete p_m;

    }
    
    
    //-------------------------------------------------------------------

        
    void ReconstructionFFD::CreateSliceMasks(RealImage main_mask, int iter)
    {

        char buffer[256];
        bool init = false;
        int current_stack_index = 0;

        GaussianBlurring<RealPixel> gb(_slices[0]->GetXSize()*1.5);

        MultiLevelFreeFormTransformation *mffd_init;

        for ( int inputIndex = 0; inputIndex < _slices.size(); inputIndex++ ) {
            
            if (_reg_slice_weight[inputIndex] > 0) {

                if (iter < 0) {
                    if (current_stack_index == _stack_index[inputIndex]) {
                        init = true;
                    }

                    if ( init && _current_iteration == 0 && current_stack_index == _stack_index[inputIndex] && _global_init_ffd) {
                        
                        sprintf(buffer, "ms-%i.dof", _package_index[inputIndex]);
                        Transformation *tt = Transformation::New(buffer);
                        mffd_init = dynamic_cast<MultiLevelFreeFormTransformation*> (tt);
                        current_stack_index++;

                        init = false;
                    }
                }


                InterpolationMode interpolation = Interpolation_NN;
                UniquePtr<InterpolateImageFunction> interpolator;
                interpolator.reset(InterpolateImageFunction::New(interpolation));

                RealImage input_volume = main_mask;
                RealImage output_volume = *_slices[inputIndex];

                double source_padding = 0;
                double target_padding = -inf;
                bool dofin_invert = false;
                bool twod = false;

                ImageTransformation *imagetransformation = new ImageTransformation;

                imagetransformation->Input(&input_volume);
                
                if (iter < 0)
                    imagetransformation->Transformation(&(*mffd_init));
                else
                    imagetransformation->Transformation(_mffd_transformations[inputIndex]);
                
                
                imagetransformation->Output(&output_volume);
                imagetransformation->TargetPaddingValue(target_padding);
                imagetransformation->SourcePaddingValue(source_padding);
                imagetransformation->Interpolator(interpolator.get());
                imagetransformation->TwoD(twod);
                imagetransformation->Invert(dofin_invert);
                imagetransformation->Run();

                delete imagetransformation;

                gb.Input(&output_volume);
                gb.Output(&output_volume);
                gb.Run();

                for (int x=0; x<output_volume.GetX(); x++) {
                    for (int y=0; y<output_volume.GetY(); y++) {
                        
                        if (output_volume(x,y,0) > 0) {
                            _slice_masks[inputIndex]->PutAsDouble(x,y,0,1);
                        }
                        else {
                            _slice_masks[inputIndex]->PutAsDouble(x,y,0,0);
                        }
                            

                    }
                }

            }

        }


    }
    

    //-------------------------------------------------------------------


    void ReconstructionFFD::SVRStep( int inputIndex )
    {

        if (_max_val[inputIndex] > 1 && (_max_val[inputIndex] - _min_val[inputIndex]) > 1) {

            ParameterList params;
            Insert(params, "Transformation model", "FFD");
                    
            if (_masked) {
                Insert(params, "Background value for image 1", -1);
                Insert(params, "Background value for image 2", 0);
                        
            }

            int current_cp_spacing;
            
            if (_current_iteration <= 3)
                current_cp_spacing = _cp_spacing[_current_iteration];
            else
                current_cp_spacing = 5;
            
            if (_current_round == 2)
                current_cp_spacing -= 2;

 
            // if (reg_slice_weight[inputIndex] > 0) {
            Insert(params, "Control point spacing in X", current_cp_spacing);
            Insert(params, "Control point spacing in Y", current_cp_spacing);
            Insert(params, "Control point spacing in Z", current_cp_spacing);
            // }
                    
            GenericRegistrationFilter *registration = new GenericRegistrationFilter();
            registration->Parameter(params);
            
            registration->Input(_grey_slices[inputIndex], _p_grey_reconstructed);

            Transformation *dofout = nullptr;
            registration->Output(&dofout);
            
            
            if (_current_iteration == 0 && _current_round == 1) {

                registration->InitialGuess(_global_mffd_transformations[_stack_index[inputIndex]]);
            }
            else {
                if (_reg_slice_weight[inputIndex] > 0) {

                    registration->InitialGuess(_mffd_transformations[inputIndex]);
                }
                else {
                    MultiLevelFreeFormTransformation *r_init = new MultiLevelFreeFormTransformation(_package_transformations[inputIndex]);
                    registration->InitialGuess(&(*r_init));
                }
            }


            registration->GuessParameter();
            registration->Run();
            
            MultiLevelFreeFormTransformation *mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*> (dofout);
            _mffd_transformations[inputIndex] = mffd_dofout;

            registration->~GenericRegistrationFilter();

        }

    } 


    
    class ParallelSVRFFD {
        
    public:

        ReconstructionFFD *reconstructor;

        ParallelSVRFFD( ReconstructionFFD *in_reconstructor ) : 
        reconstructor(in_reconstructor) { }
        
        
        void operator() (const blocked_range<size_t> &r) const {

            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {

                reconstructor->SVRStep(inputIndex);
                
            }
            
        }

        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_grey_slices.size() ), *this );
        }
        
    };
    
 
    //-------------------------------------------------------------------
        
    
    void ReconstructionFFD::FastSliceToVolumeRegistration( int iter, int round, int number_of_stacks)
    {

        _current_iteration = iter;
        _current_round = round;

        _grey_reconstructed = _reconstructed;
        
        if (round == 2) {
            
            GaussianBlurring<GreyPixel> gb4(_grey_reconstructed.GetXSize()*1);
            gb4.Input(&_grey_reconstructed);
            gb4.Output(&_grey_reconstructed);
            gb4.Run();
        }

        if (_reconstructed.GetXSize() < 1.0) {
            GenericLinearInterpolateImageFunction<RealImage> interpolator_res;
            Resampling<RealPixel> resampling(0.95, 0.95, 0.95);
            resampling.Interpolator(&interpolator_res);

            RealImage reconstructed_resampled = _reconstructed;

            resampling.Input(&_reconstructed);
            resampling.Output(&reconstructed_resampled);
            resampling.Run();

            _grey_reconstructed = reconstructed_resampled;

        }
        
        _p_grey_reconstructed = new GreyImage(_grey_reconstructed);
        Array<MultiLevelFreeFormTransformation*> global_mffd_transformations;
        LoadGlobalTransformations(number_of_stacks, global_mffd_transformations );
        _global_mffd_transformations = global_mffd_transformations;

        
        ParallelSVRFFD  *p_svr = new ParallelSVRFFD(this);
        (*p_svr)();
        delete p_svr;
        
        
        if (_structural_exclusion) {
            cout << "EvaluateRegistration" << endl;
            EvaluateRegistration();
        }

        for ( int inputIndex=0; inputIndex<_slices.size(); inputIndex++ ) {
            for (int x=0; x<_slice_2dncc[inputIndex]->GetX(); x++)
                for (int y=0; y<_slice_2dncc[inputIndex]->GetY(); y++) {
                    _slice_2dncc[inputIndex]->PutAsDouble(x,y,0,1);
                }
        }
        
    }


    //-------------------------------------------------------------------

        
    class ParallelRemoteDSVR {
        
    public:
        
        ReconstructionFFD *reconstructor;
        int svr_range_start;
        int svr_range_stop;
        string str_mirtk_path;
        string str_current_main_file_path;
        string str_current_exchange_file_path;
        
        ParallelRemoteDSVR ( ReconstructionFFD *in_reconstructor, int in_svr_range_start, int in_svr_range_stop, string in_str_mirtk_path, string in_str_current_main_file_path, string in_str_current_exchange_file_path ) :
        reconstructor(in_reconstructor),
        svr_range_start(in_svr_range_start),
        svr_range_stop(in_svr_range_stop),
        str_mirtk_path(in_str_mirtk_path),
        str_current_main_file_path(in_str_current_main_file_path),
        str_current_exchange_file_path(in_str_current_exchange_file_path) { }
        
        
        void operator() (const blocked_range<size_t> &r) const {
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                if (reconstructor->_zero_slices[inputIndex] > 0) {
                    
                    std::thread::id this_id = std::this_thread::get_id();

                    string str_target, str_source, str_reg_model, str_ds_setting, str_dofin, str_dofout, str_bg1, str_bg2, str_log;
                    
                    
                    str_target = str_current_exchange_file_path + "/slice-" + to_string(inputIndex) + ".nii.gz";
                    str_source = str_current_exchange_file_path + "/current-source.nii.gz";
                    str_log = " > " + str_current_exchange_file_path + "/log-" + to_string(inputIndex) + ".txt";
                    
                    
                    str_reg_model = "-model FFD";
                    
                    str_dofout = "-dofout " + str_current_exchange_file_path + "/transformation-" + to_string(inputIndex) + ".dof ";
                    
                    if (reconstructor->_masked) {
                        
                        str_bg1 = "-bg1 " + to_string(-1);
                        str_bg2 = "-bg2 " + to_string(0);
                    }
                    
                    
                    int current_cp_spacing;
                    if (reconstructor->_current_iteration <= 3)
                        current_cp_spacing = reconstructor->_cp_spacing[reconstructor->_current_iteration];
                    else
                        current_cp_spacing = 5;
                    if (reconstructor->_current_round == 2)
                        current_cp_spacing -= 2;
                    
                    str_ds_setting = "-ds " + to_string(current_cp_spacing);
                    
                    if (reconstructor->_current_iteration <= 1) {
                        string r_name = str_current_exchange_file_path + "/r-init-" + to_string(inputIndex) + ".dof";
                        reconstructor->_package_transformations[inputIndex].Write(r_name.c_str());
                    }
                    
                    
                    if (reconstructor->_current_iteration == 0 && reconstructor->_current_round == 1) {
                        
                        str_dofin = "-dofin " + str_current_main_file_path + "/ms-" + to_string(reconstructor->_stack_index[inputIndex]) + ".dof ";
                    }
                    else {
                        if (reconstructor->_reg_slice_weight[inputIndex] > 0) {
                            
                            str_dofin = "-dofin " + str_current_exchange_file_path + "/transformation-" + to_string(inputIndex) + ".dof ";
                        }
                        else {
                            
                            str_dofin = "-dofin " + str_current_exchange_file_path + "/r-init-" + to_string(inputIndex) + ".dof ";

                        }
                    }
                    
                    string register_cmd = str_mirtk_path + "/register " + str_target + " " + str_source + " " + str_reg_model + " " + str_ds_setting + " " + str_bg1 + " " + str_bg2 + " " + str_dofin + " " + str_dofout + " " + str_log;
 
                    int tmp_log = system(register_cmd.c_str());
                    
                }
                
            }
            
        }
        
        void operator() () const {
            parallel_for( blocked_range<size_t>(svr_range_start, svr_range_stop), *this );
        }
        
    };
    
    
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::RemoteSliceToVolumeRegistration( int iter, int round, int number_of_stacks, string str_mirtk_path, string str_current_main_file_path, string str_current_exchange_file_path )
    {
        
        
        _current_iteration = iter;
        _current_round = round;
        _grey_reconstructed = _reconstructed;
        
        
        
        string str_source = str_current_exchange_file_path + "/current-source.nii.gz";
        
        if (round == 2) {
            
            RealImage tmp_source = _reconstructed;
            GaussianBlurring<RealPixel> gb4(tmp_source.GetXSize()*1);
            gb4.Input(&tmp_source);
            gb4.Output(&tmp_source);
            gb4.Run();
            
            tmp_source.Write(str_source.c_str());
            
        } else {
            
            _reconstructed.Write(str_source.c_str());
        }
        
        
        
        if (_current_iteration == 0 || _current_round == 1) {
            
            for (int inputIndex=0; inputIndex<_slices.size(); inputIndex++) {
                
                string str_target = str_current_exchange_file_path + "/slice-" + to_string(inputIndex) + ".nii.gz";
                _slices[inputIndex]->Write(str_target.c_str());
                
                RealPixel tmin, tmax;
                _slices[inputIndex]->GetMinMax(&tmin, &tmax);
                if (tmax > 1 && ((tmax-tmin) > 1) ) {
                    _zero_slices[inputIndex] = 1;
                } else {
                    _zero_slices[inputIndex] = -1;
                }
                
            }
        }
        
        
        int stride = 16;
        int svr_range_start = 0;
        int svr_range_stop = svr_range_start + stride;
        
        while ( svr_range_start < _slices.size() )
        {

            ParallelRemoteDSVR p_svr(this, svr_range_start, svr_range_stop, str_mirtk_path, str_current_main_file_path, str_current_exchange_file_path);
            p_svr();
            
            svr_range_start = svr_range_stop;
            svr_range_stop = svr_range_start + stride;
            
            if (svr_range_stop > _slices.size()) {
                svr_range_stop = _slices.size();
            }
            
        }
        
        
        
        for (int inputIndex=0; inputIndex<_slices.size(); inputIndex++) {
            
            string str_dofout = str_current_exchange_file_path + "/transformation-" + to_string(inputIndex) + ".dof";
            
            Transformation *tmp_transf = Transformation::New(str_dofout.c_str());
            MultiLevelFreeFormTransformation *tmp_mffd_transf = dynamic_cast<MultiLevelFreeFormTransformation*> (tmp_transf);
            _mffd_transformations[inputIndex] = tmp_mffd_transf;
            
        }
        

        if (_structural_exclusion) {
            cout << "EvaluateRegistration" << endl;
            EvaluateRegistration();
        }
        
        for ( int inputIndex=0; inputIndex<_slices.size(); inputIndex++ ) {
            for (int x=0; x<_slice_2dncc[inputIndex]->GetX(); x++) {
                for (int y=0; y<_slice_2dncc[inputIndex]->GetY(); y++) {
                    _slice_2dncc[inputIndex]->PutAsDouble(x,y,0,1);
                }
            }
        }
        
        
    }
    
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::LoadGlobalTransformations( int number_of_stacks, Array<MultiLevelFreeFormTransformation*>& global_mffd_transformations )
    {
    
        char buffer[256];

        global_mffd_transformations.clear();
        
        for (int i=0; i<number_of_stacks; i++) {
            
            
            MultiLevelFreeFormTransformation *mffd_init;
            sprintf(buffer, "ms-%i.dof", i);
            Transformation *tt = Transformation::New(buffer);
            mffd_init = dynamic_cast<MultiLevelFreeFormTransformation*> (tt);
            
            global_mffd_transformations.push_back(mffd_init);

        }
        
    }

    //-------------------------------------------------------------------
        
        
        
    RealImage ReconstructionFFD::StackIntersection( Array<RealImage*>& stacks, int templateNumber, RealImage template_mask )
    {
        
        char buffer[256];

        RealImage i_mask = *(stacks[templateNumber]);
        i_mask = 1;
        
        
        int ix, iy, iz, is;
        double sx, sy, sz;
        double wx, wy, wz;
        
        for (ix = 0; ix < stacks[templateNumber]->GetX(); ix++) {
            for (iy = 0; iy < stacks[templateNumber]->GetY(); iy++) {
                for (iz = 0; iz < stacks[templateNumber]->GetZ(); iz++) {
                    
                    i_mask(ix,iy,iz) = 1;
                    
                    wx = ix;
                    wy = iy;
                    wz = iz;
                    
                    i_mask.ImageToWorld(wx, wy, wz);
                    
                    if (stacks.size() > 1) {
                        
                        for (is = 1; is < stacks.size(); is++) {
                            
                            sx = wx;
                            sy = wy;
                            sz = wz;
                            
                            stacks[is]->WorldToImage(sx, sy, sz);

                            if (sx>0 && sx<stacks[is]->GetX() && sy>0 && sy<stacks[is]->GetY() && sz>0 && sz<stacks[is]->GetZ()) {
                                
                                i_mask(ix,iy,iz) = i_mask(ix,iy,iz) * 1;
                            }
                            else {
                                
                                i_mask(ix,iy,iz) = i_mask(ix,iy,iz) * 0;
                            }
                        }
                        
                        sx = wx;
                        sy = wy;
                        sz = wz;
                        
                        template_mask.WorldToImage(sx, sy, sz);
                        
                        int mx, my, mz;
                        mx = round(sx);
                        my = round(sy);
                        mz = round(sz);
                        
                        if (mx>0 && mx<template_mask.GetX() && my>0 && my<template_mask.GetY() && mz>0 && mz<template_mask.GetZ()) {

                            if (template_mask(mx,my,mz)>0)
                                i_mask(ix,iy,iz) = 1;
                        }
 
                        
                        
                    }
                }
            }
        }
        
        
        ConnectivityType i_connectivity = CONNECTIVITY_26;
        Dilate<RealPixel>(&i_mask, 5, i_connectivity);
        
        if (_debug) {
            i_mask.Write("intersection-mask.nii.gz");
        }
        
        return i_mask;

    }

    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::EvaluateRegistration()
    {
        
        char buffer[256];

        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        double source_padding = 0;
        double target_padding = -inf;
        bool dofin_invert = false;
        bool twod = false;
        
        if (_masked)
            target_padding = -1;
        
        
        Array<double> reg_ncc;
        double mean_ncc = 0;
        
        
        cout << " - excluded : ";
        
        for ( int inputIndex=0; inputIndex<_slices.size(); inputIndex++ ) {
            
            
            RealImage output =  *_slices[inputIndex];
            output = 0;
            
            ImageTransformation *imagetransformation = new ImageTransformation;
            
            imagetransformation->Input(&_reconstructed);
            imagetransformation->Transformation(_mffd_transformations[inputIndex]);
            imagetransformation->Output(&output);
            imagetransformation->TargetPaddingValue(target_padding);
            imagetransformation->SourcePaddingValue(source_padding);
            imagetransformation->Interpolator(&interpolator);
            imagetransformation->TwoD(twod);
            imagetransformation->Invert(dofin_invert);
            imagetransformation->Run();
            
            
            delete imagetransformation;
            
            RealImage target = *_slices[inputIndex];
            
            double output_ncc = 0;
            
            RealImage slice_mask = *_slice_masks[inputIndex];
            
            
            target *= slice_mask;
            output *= slice_mask;
            
            
            output_ncc = SliceCC(target, output);
            
            reg_ncc.push_back(output_ncc);
            mean_ncc += output_ncc;
            
            if (_stack_index[inputIndex] != _excluded_stack) {
                
                if (output_ncc > _global_NCC_threshold)
                    _reg_slice_weight[inputIndex] = 1;
                else {
                    _reg_slice_weight[inputIndex] = -1;
                    cout << inputIndex << " ";
                }
            }
            else {
                _reg_slice_weight[inputIndex] = 1;
            }
            
        }
        
        cout << endl;
        
        mean_ncc /= _slices.size();
        
        
        cout << " - mean registration ncc: " << mean_ncc << endl;
        
        
    }

    //-------------------------------------------------------------------
    
    
    double ReconstructionFFD::SliceCC(RealImage slice_1, RealImage slice_2)
    {
        int i, j;
        int slice_1_N, slice_2_N;
        double *slice_1_ptr, *slice_2_ptr;

        int slice_1_n, slice_2_n;
        double slice_1_m, slice_2_m;
        double slice_1_sq, slice_2_sq;

        double tmp_val, diff_sum;
        double average_CC, CC_slice;


        slice_1_N = slice_1.NumberOfVoxels();
        slice_2_N = slice_2.NumberOfVoxels();

        slice_1_ptr = slice_1.Data();
        slice_2_ptr = slice_2.Data();

        slice_1_n = 0;
        slice_1_m = 0;
        for (j = 0; j < slice_1_N; j++) {
            
            if (slice_1_ptr[j]>0 && slice_2_ptr[j]>0) {
                slice_1_m = slice_1_m + slice_1_ptr[j];
                slice_1_n = slice_1_n + 1;
            }
        }
        slice_1_m = slice_1_m / slice_1_n;

        slice_2_n = 0;
        slice_2_m = 0;
        for (j = 0; j < slice_2_N; j++) {
            if (slice_1_ptr[j]>0 && slice_2_ptr[j]>0) {
                slice_2_m = slice_2_m + slice_2_ptr[j];
                slice_2_n = slice_2_n + 1;
            }
        }
        slice_2_m = slice_2_m / slice_2_n;
        
        if (slice_2_n<2) {
            
            CC_slice = 1;
            
        }
        else {

            diff_sum = 0;
            slice_1_sq = 0;
            slice_2_sq = 0;

            for (j = 0; j < slice_1_N; j++) {
                if (slice_1_ptr[j]>0 && slice_2_ptr[j]>0) {
                    diff_sum = diff_sum + ((slice_1_ptr[j] - slice_1_m)*(slice_2_ptr[j] - slice_2_m));
                    slice_1_sq = slice_1_sq + pow((slice_1_ptr[j] - slice_1_m), 2);
                    slice_2_sq = slice_2_sq + pow((slice_2_ptr[j] - slice_2_m), 2);
                }
            }

            if ((slice_1_sq * slice_2_sq)>0)
                CC_slice = diff_sum / sqrt(slice_1_sq * slice_2_sq);
            else
                CC_slice = 0;
            
        }

        return CC_slice;
        
    }

    
    //-------------------------------------------------------------------
    
    class ParallelCCMap {
    
    public:
        ReconstructionFFD *reconstructor;
        
        ParallelCCMap(ReconstructionFFD *_reconstructor) :
        reconstructor(_reconstructor) { }
        
        
        void operator() (const blocked_range<size_t> &r) const {
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                int box_size = reconstructor->_local_window_size; // 20
                double threshold = reconstructor->_local_NCC_threshold;
                double mean_ncc = 0;

                if (!reconstructor->_structural_exclusion)
                    threshold = -1;
               
                mean_ncc = reconstructor->SliceCCMap2(inputIndex, box_size, threshold);
               
            }
            
        }
        
        void operator() () const {

            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::CStep2D()
    {

        char buffer[256];
        
//        ofstream N_csv_file1;
//        N_csv_file1.open("S-W.csv");
        
//         for (int i=0; i<_slices.size(); i++) {
//             N_csv_file1 << _slice_weight[i] << endl;
//         }
//         N_csv_file1.close();
        
        
        ParallelCCMap *p_cstep = new ParallelCCMap(this);
        (*p_cstep)();

        delete p_cstep;
        
//        ofstream N_csv_file;
//        N_csv_file.open("S-NCC.csv");
        
//         for (int i=0; i<_slices.size(); i++) {
//             N_csv_file << _structural_slice_weight[i] << endl;
//         }
//         N_csv_file.close();
        
    }

    //-------------------------------------------------------------------


    
    double ReconstructionFFD::LocalSSIM( RealImage slice, RealImage sim_slice )
    {
        
        double C1 = 6.5025, C2 = 58.5225;
        double SSIM = 0, DSSIM = 0;
        
        
        double mu1=0, mu2=0, var1=0, var2=0, covar=0, num=0;
        double x_sq=0, y_sq=0, xy=0;
        
        int box = round(slice.GetX()/2);
        
        int y = box;
        int x = box;
        
        for (int yy=0; yy<slice.GetY(); yy++) {
            for (int xx=0; xx<slice.GetX(); xx++) {
                
                if (slice(xx,yy,0)>0.01 && sim_slice(xx,yy,0)>0.01) {
                    
                    mu1 += slice(xx, yy, 0);
                    mu2 += sim_slice(xx,yy,0);
                    num += 1;
                    
                    x_sq += pow(slice(xx, yy, 0),2.0);
                    y_sq += pow(sim_slice(xx, yy, 0),2.0);
                    xy += slice(xx, yy, 0)*sim_slice(xx, yy, 0);
                }
                
            }
        }
        
        mu1 = mu1/num;
        mu2 = mu2/num;
        var1 = (x_sq/num)-pow(mu1,2.0);
        var2 = (y_sq/num)-pow(mu2,2.0);
        covar = (xy/num)-mu1*mu2;
        
        double curSSIM = ((2*mu1*mu2+C1)*(2*covar+C2)) / ( (pow(mu1,2.)+pow(mu2,2.)+C1) * (var1+var2+C2) );
        
        return curSSIM;

        
    }

    //-------------------------------------------------------------------


    double ReconstructionFFD::SliceCCMap2(int inputIndex, int box_size, double threshold)
    {

        RealImage slice_1 = *_slices[inputIndex];
        RealImage slice_2 = *_simulated_slices[inputIndex];
        
        RealImage slice_mask = *_slice_masks[inputIndex];
        

        for (int i = 0; i < slice_1.GetX(); i++) {
            for (int j = 0; j < slice_1.GetY(); j++) {
                if (slice_1(i, j, 0) > 0.01) {

                    slice_1(i, j, 0) *= exp(-(_bias[inputIndex])->GetAsDouble(i, j, 0)) * _scale[inputIndex];
                }
            }
        }

        GaussianBlurring<RealPixel> gb(slice_1.GetXSize()*0.65);
        gb.Input(&slice_1);
        gb.Output(&slice_1);
        gb.Run();
 
        
        
        RealImage m_slice_1 = slice_1 * slice_mask;
        RealImage m_slice_2 = slice_2 * slice_mask;

        RealImage CC_map = slice_1;
        double global_cc = SliceCC(m_slice_1, m_slice_2);

        CC_map = 1;


        double mean_ncc = 0.0; 
        int num = 0;
        
        for (int y = 0; y < slice_1.GetY(); y++) {
            for (int x = 0; x < slice_1.GetX(); x++) {
                if(slice_2(x,y,0)<0.01)
                    CC_map(x,y,0) = 1;

                if(slice_1(x,y,0)<0.01)
                    CC_map(x,y,0) = 1;

            }
        }

        
        int shift = round(box_size/2);
        
        RealImage region_1, region_2;
        
        for (int y = shift; y < (slice_1.GetY()-shift); y++) {
            for (int x = shift; x < (slice_1.GetX()-shift); x++) {
                
                if (slice_2(x,y,0) > 0.01 && slice_1(x,y,0) >0.01) {

                    region_1 = slice_1.GetRegion(x-shift, y-shift, 0, x+shift, y+shift, 1);
                    region_2 = slice_2.GetRegion(x-shift, y-shift, 0, x+shift, y+shift, 1);

                    RealImage region_mask1 = slice_1.GetRegion(x-shift, y-shift, 0, x+shift, y+shift, 1);
                    RealImage region_mask2 = slice_2.GetRegion(x-shift, y-shift, 0, x+shift, y+shift, 1);
                    region_mask1 = 0;
                    region_mask2 = 0;

                    for (int x=0; x<region_mask1.GetX(); x++) {
                        for (int y=0; y<region_mask1.GetY(); y++) {

                            int x2 = abs(shift-x)*abs(shift-x);
                            int y2 = abs(shift-y)*abs(shift-y);
                            int r = sqrt(x2+y2);

                            if (r < (shift+1)) {
                                region_mask1(x,y,0) = 1;
                                region_mask2(x,y,0) = 1;
                            }
 
                        }
                    }

                    region_1 *= region_mask1;
                    region_2 *= region_mask2;

                    CC_map(x,y,0) = LocalSSIM(region_1, region_2); //SliceCC(region_1, region_2);

                    mean_ncc += CC_map(x,y,0);
                    num++;
                    
                   if (threshold > -0.1) {
                        if (CC_map(x,y,0)<threshold)
                            CC_map(x,y,0) = 0.1;
                        else
                            CC_map(x,y,0) = 1;
                   }

                    
                   
                }
                else
                    CC_map(x,y,0) = 1;

            }
        }
 
        
        _slice_2dncc[inputIndex] = new RealImage(CC_map);

        if (num > 0)
            mean_ncc /= num;
        else 
            mean_ncc = 0;

               
        if (mean_ncc < 0)
            mean_ncc = 0;
                
        _structural_slice_weight[inputIndex] = global_cc;


        if (threshold > -0.1) {

            if (_structural_slice_weight[inputIndex]>_global_NCC_threshold) {

                _structural_slice_weight[inputIndex] = 1;

                if (_slice_weight[inputIndex] < 0.6) {
                    _structural_slice_weight[inputIndex] = _slice_weight[inputIndex]/10;
                }
            }
            else {
   
                _structural_slice_weight[inputIndex] = 0;
            }
        }

        return mean_ncc;
        
    }

    
    //-------------------------------------------------------------------
   
   struct JacobianImplFFD : public VoxelReduction {
       
       
       GreyImage *_image;
       BaseImage *_jacobian;
       BinaryImage *_mask;
       int _nvox;
       const Transformation *_dof;
       double _t, _t0;
       int _padding;
       double _outside;
       int _m, _n;
       int _noss;
       int _dtype;
       double _threshold;
       
       JacobianImplFFD() :
       _image(nullptr),
       _jacobian(nullptr),
       _mask(nullptr),
       _nvox(0),
       _dof(nullptr),
       _t(NaN),
       _t0(NaN),
       _padding(MIN_GREY),
       _outside(.0),
       _m(0),
       _n(0),
       _noss(-1),
       _dtype(MIRTK_VOXEL_GREY),
       _threshold(.0001) {
       }

       JacobianImplFFD(const JacobianImplFFD &other) :
       _image(other._image),
       _jacobian(other._jacobian),
       _mask(other._mask),
       _nvox(other._nvox),
       _dof(other._dof),
       _t(other._t),
       _t0(other._t0),
       _padding(other._padding),
       _outside(other._outside),
       _m(other._m),
       _n(other._n),
       _noss(other._noss),
       _dtype(other._dtype),
       _threshold(other._threshold) {
       }
       
       void split(const JacobianImplFFD &other) {
           _m = _n = 0;
       }
       
       void join(const JacobianImplFFD &other) {
           _m += other._m;
           _n += other._n;
       }
       
       double Jacobian(int i, int j, int k, const BinaryPixel *mask) {
           if (*mask == 0) {
               return _outside;
           }
           
           ++_m;
           
           double jac = numeric_limits<double>::quiet_NaN();
           
           double x = i, y = j, z = k;
           _image->ImageToWorld(x, y, z);
           
           jac = _dof->Jacobian(x, y, z, _t, _t0);
           if (jac < .0) ++_n;
           
           
           return IsNaN(jac) ? _outside : jac;
       }
       
       void operator()(int i, int j, int k, int, const BinaryPixel *mask, GreyPixel *out) {
           double jac = Jacobian(i, j, k, mask);
           *out = static_cast<GreyPixel> (100.0 * jac);
       }
       
       void operator()(int i, int j, int k, int, const BinaryPixel *mask, float *out) {
           double jac = Jacobian(i, j, k, mask);
           *out = static_cast<float> (100.0 * jac);
       }
       
       void operator()(int i, int j, int k, int, const BinaryPixel *mask, double *out) {
           double jac = Jacobian(i, j, k, mask);
           *out = 100.0 * jac;
       }
       
       void Run() {
           _nvox = _image->NumberOfSpatialVoxels();
           ImageAttributes attr = _image->Attributes();
           attr._t = 1, attr._dt = .0;
           if (IsNaN(_t0)) _t0 = _image->ImageToTime(0);
           if (IsNaN(_t)) _t = _t0 + _image->GetTSize();
           _jacobian->Initialize(attr, 1);
           
           const BSplineFreeFormTransformationSV *svffd = nullptr;
           if (_noss != 0) {
               svffd = dynamic_cast<const BSplineFreeFormTransformationSV *> (_dof);
               if (!svffd) {
                   const MultiLevelTransformation *mffd;
                   mffd = dynamic_cast<const MultiLevelTransformation *> (_dof);
                   if (mffd && mffd->NumberOfLevels() == 1 && mffd->GetGlobalTransformation()->IsIdentity()) {
                       svffd = dynamic_cast<const BSplineFreeFormTransformationSV *> (mffd->GetLocalTransformation(0));
                   }
               }
           }

           GreyImage *iout = nullptr;
           GenericImage<float> *fout = nullptr;
           GenericImage<double> *dout = nullptr;
           MIRTK_START_TIMING();
           if ((iout = dynamic_cast<GreyImage *> (_jacobian))) {
               ParallelForEachVoxel(attr, _mask, iout, *this);
           } else if ((fout = dynamic_cast<GenericImage<float> *> (_jacobian))) {
               ParallelForEachVoxel(attr, _mask, fout, *this);
           } else if ((dout = dynamic_cast<GenericImage<double> *> (_jacobian))) {
               ParallelForEachVoxel(attr, _mask, dout, *this);
           } else {
               cerr << "Output image data type must be either GreyPixel, float, or double" << endl;
               exit(1);
           }

       }
       

   };

    //-------------------------------------------------------------------
    
    
    
    //-------------------------------------------------------------------
    
    /*
    void ReconstructionFFD::JStep( int iteration ) {

        int negative_count = 0;
        
        for (int i=0; i<_slices.size(); i++) {
            
            if (iteration == 0) {
                negative_count = SliceJNegativeCount(i);
                _negative_J_values[i] = negative_count;
            }
            else {
                negative_count = _negative_J_values[i];
            }
            
            if (negative_count>0) {
                _slice_weight[i] = 0.0;
                // _slice_potential[i] = 0.0;
                
                cout << " - " << i << " : " << negative_count << endl;
            }
        }
        
    }

    */
    
    //-------------------------------------------------------------------
    
    /*
    
    int ReconstructionFFD::SliceJNegativeCount(int i)
    {
        
        
        int j=0;
        
        UniquePtr<GreyImage> target(new GreyImage(_slices[i]));
        
        JacobianImplFFD impl;
        bool asdisp = false;
        bool asvelo = false;
        
        impl._image = target.get();
        impl._jacobian = target.get();
        impl._dof = _mffd_transformations[i];  // &_mffd_transformations[i];
        
        const int nvox = target->NumberOfSpatialVoxels();

        // add mask - to evaluate the values only within the body ROI
        
        BinaryImage mask(target->Attributes(), 1);
        for (int idx = 0; idx < nvox; ++idx) {
            mask(idx) = (target->GetAsDouble(idx) > impl._padding ? 1 : 0);
        }
        impl._mask = &mask;
        
        
        UniquePtr<BaseImage> output;
        if (impl._dtype != target->GetDataType()) {
            output.reset(BaseImage::New(impl._dtype));
            impl._jacobian = output.get();
        }
        
        
        impl.Run();
        
        output.reset(target.release());
        
        int negative_count = 0;
        
        double jac;
        for (int n = 0; n < nvox; ++n) {
            if (mask(n) != 0) {
                jac = output->GetAsDouble(n);
                if (jac < 0)
                    negative_count++;
            }
        }
        
        return negative_count;
    }

     */
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SliceToReconstuced( int inputIndex, double& x, double& y, double& z, int& i, int& j, int& k, int mode )
    {

        _grey_slices[inputIndex]->ImageToWorld(x, y, z);
        _mffd_transformations[inputIndex]->Transform(-1, 1, x, y, z);
        _reconstructed.WorldToImage(x, y, z);
        
        if (mode == 0) {
            i = round(x);
            j = round(y);
            k = round(z);
            
        }
        else {
            i = (int) floor(x);
            j = (int) floor(y);
            k = (int) floor(z);
            
        }
        
    
    }
    
    //-------------------------------------------------------------------
    
    class ParallelCoeffInitFFD {
    public:
        ReconstructionFFD *reconstructor;
        
        ParallelCoeffInitFFD(ReconstructionFFD *_reconstructor) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                bool slice_inside;
                
                double vx, vy, vz;
                
                ImageAttributes slice_attr = reconstructor->_slice_attr[inputIndex];
                ImageAttributes recon_attr = reconstructor->_reconstructed_attr;
                
                vx = slice_attr._dx;
                vy = slice_attr._dy;
                vz = slice_attr._dz;
                
                double res = vx;
                
                POINT3D p;
                VOXELCOEFFS empty;
                SLICECOEFFS slicecoeffs(slice_attr._x, Array < VOXELCOEFFS > (slice_attr._y, empty));

                slice_inside = false;
                
                double dx, dy, dz;
                
                dx = slice_attr._dx;
                dy = slice_attr._dy;
                dz = slice_attr._dz;
                
                double sigmax = 1.2 * dx / 2.3548;
                double sigmay = 1.2 * dy / 2.3548;
                double sigmaz = dz / 2.3548;
                
                double size = res / reconstructor->_quality_factor;

                int xDim = round(2 * dx / size);
                int yDim = round(2 * dy / size);
                int zDim = round(2 * dz / size);
                
                int excluded = 0;

                ImageAttributes attr;
                attr._x = xDim;
                attr._y = yDim;
                attr._z = zDim;
                attr._dx = size;
                attr._dy = size;
                attr._dz = size;
                RealImage PSF(attr);
                
                double cx, cy, cz;
                cx = 0.5 * (xDim - 1);
                cy = 0.5 * (yDim - 1);
                cz = 0.5 * (zDim - 1);
                PSF.ImageToWorld(cx, cy, cz);
                
                double x, y, z;
                double sum = 0;
                int i, j, k;
                for (i = 0; i < xDim; i++)
                    for (j = 0; j < yDim; j++)
                        for (k = 0; k < zDim; k++) {
                            x = i;
                            y = j;
                            z = k;
                            PSF.ImageToWorld(x, y, z);
                            x -= cx;
                            y -= cy;
                            z -= cz;
                            PSF(i, j, k) = exp(-x * x / (2 * sigmax * sigmax) - y * y / (2 * sigmay * sigmay) - z * z / (2 * sigmaz * sigmaz));
                            sum += PSF(i, j, k);
                        }
                PSF /= sum;
                
                if (reconstructor->_debug)
                    if (inputIndex == 0)
                        PSF.Write("PSF.nii.gz");
                
                int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2)) * 2 + 1 + 2;
                attr._x = dim;
                attr._y = dim;
                attr._z = dim;
                attr._dx = res;
                attr._dy = res;
                attr._dz = res;
                RealImage tPSF(attr);
                int centre = (dim - 1) / 2;
                
                int ii, jj, kk;
                int tx, ty, tz;
                int nx, ny, nz;
                int l, m, n;
                double weight;
                
                bool excluded_slice = false;
                for (int ff=0; ff<reconstructor->_force_excluded.size(); ff++) {
                    
                    if (inputIndex == reconstructor->_force_excluded[ff])
                        excluded_slice = true;
                    
                }
                

                if (reconstructor->_reg_slice_weight[inputIndex] > 0 || reconstructor->_stack_index[inputIndex] == reconstructor->_excluded_stack && !excluded_slice) {

                for (i = 0; i < slice_attr._x; i++)
                    for (j = 0; j < slice_attr._y; j++)
                        
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) > -1 && reconstructor->_slice_2dncc[inputIndex]->GetAsDouble(i,j,0) > 0) {
                            x = i;
                            y = j;
                            z = 0;
                            
                            reconstructor->SliceToReconstuced(inputIndex, x, y, z, tx, ty, tz, 0);

                            for (ii = 0; ii < dim; ii++)
                                for (jj = 0; jj < dim; jj++)
                                    for (kk = 0; kk < dim; kk++)
                                        tPSF(ii, jj, kk) = 0;
                            
                            for (ii = 0; ii < xDim; ii++)
                                for (jj = 0; jj < yDim; jj++)
                                    for (kk = 0; kk < zDim; kk++) {

                                        x = ii;
                                        y = jj;
                                        z = kk;

                                        PSF.ImageToWorld(x, y, z);
                                        
                                        x -= cx;
                                        y -= cy;
                                        z -= cz;
                                        
                                        x /= dx;
                                        y /= dy;
                                        z /= dz;

                                        x += i;
                                        y += j;
                                        
                                        reconstructor->SliceToReconstuced(inputIndex, x, y, z, nx, ny, nz, -1);
                                        
                                        sum = 0;

                                        bool inside = false;
                                        for (l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < recon_attr._x))
                                                for (m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < recon_attr._y))
                                                        for (n = nz; n <= nz + 1; n++)
                                                            if ((n >= 0) && (n < recon_attr._z)) {
                                                                weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                                                sum += weight;
                                                                if (reconstructor->_mask(l, m, n) == 1) {
                                                                    inside = true;
                                                                    slice_inside = true;
                                                                }
                                                            }

                                        if ((sum <= 0) || (!inside))
                                            continue;

                                        for (l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < recon_attr._x))
                                                for (m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < recon_attr._y))
                                                        for (n = nz; n <= nz + 1; n++)
                                                            if ((n >= 0) && (n < recon_attr._z)) {
                                                                weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                                                
                                                                int aa, bb, cc;
                                                                aa = l - tx + centre;
                                                                bb = m - ty + centre;
                                                                cc = n - tz + centre;

                                                                double value = PSF(ii, jj, kk) * weight / sum;

                                                                if ((aa < 0) || (aa >= dim) || (bb < 0) || (bb >= dim) || (cc < 0) || (cc >= dim)) {
                                                                    
                                                                    char buffer[256];
                                                                     excluded++;
                                                                }
                                                                else {
                                                                    tPSF(aa, bb, cc) += value;
                                                                    
                                                                }
                                                            }
                                        
                                    }
                            
                            for (ii = 0; ii < dim; ii++)
                                for (jj = 0; jj < dim; jj++)
                                    for (kk = 0; kk < dim; kk++)
                                        if (tPSF(ii, jj, kk) > 0) {
                                            p.x = ii + tx - centre;
                                            p.y = jj + ty - centre;
                                            p.z = kk + tz - centre;
                                            p.value = tPSF(ii, jj, kk);
                                            slicecoeffs[i][j].push_back(p);
                                        }
                        }
                
                }

                reconstructor->_volcoeffs[inputIndex] = slicecoeffs;
                reconstructor->_slice_inside[inputIndex] = slice_inside;

            }
            
        }
        
        void operator() () const {

            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );

        }
        
    };
    
    
    
    void ReconstructionFFD::CoeffInit()
    {
    
        _volcoeffs.clear();
        _volcoeffs.resize(_slices.size());
        
        _slice_inside.clear();
        _slice_inside.resize(_slices.size());
        
        _reconstructed_attr = _reconstructed.Attributes();
        
        ParallelCoeffInitFFD *p_cinit = new ParallelCoeffInitFFD(this);
        (*p_cinit)();

        delete p_cinit;

        _volume_weights.Initialize( _reconstructed.Attributes() );
        _volume_weights = 0;
        
        
        int inputIndex, i, j, n, k;
        POINT3D p;
        for ( inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            for ( i = 0; i < _slices[inputIndex]->GetX(); i++)
                for ( j = 0; j < _slices[inputIndex]->GetY(); j++) {
                    
                    n = _volcoeffs[inputIndex][i][j].size();
                    for (k = 0; k < n; k++) {
                        p = _volcoeffs[inputIndex][i][j][k];
                        
                        double x = i, y = j, z = 0;
                        _slices[inputIndex]->ImageToWorld(x, y, z);
                        double jac = _mffd_transformations[inputIndex]->Jacobian(x, y, z, 0, 0);
                        
                        if ((100*jac) > 60) {
                            _volume_weights(p.x, p.y, p.z) += p.value;
                        }
                    }
                    
                }
        }
        
        for ( int z = 0; z < _volume_weights.GetZ(); z++) {
            for ( int y = 0; y < _volume_weights.GetY(); y++) {
                for ( int x = 0; x < _volume_weights.GetX(); x++) {
                    
                    if (_volume_weights(x,y,z) < 0.0001) {
                        _volume_weights(x,y,z) = 1;
                    }
                    
                }
            }
        }
        
        if (_debug)
            _volume_weights.Write("volume_weights.nii.gz");
        
        RealPixel *ptr = _volume_weights.Data();
        RealPixel *pm = _mask.Data();
        double sum = 0;
        int num=0;
        for (int i=0;i<_volume_weights.NumberOfVoxels();i++) {
            if (*pm==1) {
                sum+=*ptr;
                num++;
            }
            ptr++;
            pm++;
        }
        _average_volume_weight = sum/num;
        
        if(_debug) {
            cout<<"Average volume weight : "<<_average_volume_weight<<endl;
        }

        
    }
     

    //-------------------------------------------------------------------
    
    void ReconstructionFFD::GaussianReconstruction()
    {

        unsigned int inputIndex;
        int i, j, k, n;
        RealImage slice;
        double scale;
        POINT3D p;
        Array<int> voxel_num;
        int slice_vox_num;
        
        _reconstructed = 0;
        
        
        if (_multiple_channels_flag && (_number_of_channels > 0)) {
            for (int n=0; n<_number_of_channels; n++) {
                _mc_reconstructed[n] = 0;
            }
        }

        
        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {

            slice = *(_slices[inputIndex]);

            RealImage& b = *(_bias[inputIndex]);
  
            scale = _scale[inputIndex];
            
            slice_vox_num=0;
            
            
            Array<RealImage> mc_slices;
            if (_multiple_channels_flag && (_number_of_channels > 0)) {
                for (int n=0; n<_number_of_channels; n++) {
                    mc_slices.push_back(*_mc_slices[inputIndex][n]);
                }
            }
            

            
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {
                        
                        double x = i, y = j, z = 0;
                        _slices[inputIndex]->ImageToWorld(x, y, z);
                        double jac = _mffd_transformations[inputIndex]->Jacobian(x, y, z, 0, 0);
                        
                        if ((100*jac) > 60) {

                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                            
                            if (_multiple_channels_flag && (_number_of_channels > 0)) {
                                for (int n=0; n<_number_of_channels; n++) {
                                    mc_slices[n](i, j, 0) *= exp(-b(i, j, 0)) * scale;
                                }
                            }
                            

                            n = _volcoeffs[inputIndex][i][j].size();

                            if (n>0)
                                slice_vox_num++;

                            for (k = 0; k < n; k++) {
                                p = _volcoeffs[inputIndex][i][j][k];
                                _reconstructed(p.x, p.y, p.z) += p.value * slice(i, j, 0);
                                
                                if (_multiple_channels_flag && (_number_of_channels > 0)) {
                                    for (int n=0; n<_number_of_channels; n++) {
                                        _mc_reconstructed[n](p.x, p.y, p.z) += p.value * mc_slices[n](i, j, 0);
                                    }
                                }
                            }
                            
                        }
                    }
            voxel_num.push_back(slice_vox_num);

        }
        
        _reconstructed /= _volume_weights;
        
//         if (_debug)
            _reconstructed.Write("init-gaussian.nii.gz");

        
        if (_multiple_channels_flag && (_number_of_channels > 0)) {
            for (int n=0; n<_number_of_channels; n++) {
                
                _mc_reconstructed[n] /= _volume_weights;
                
//                sprintf(buffer,"mc-init-%i.nii.gz", n);
//                _mc_reconstructed[n].Write(buffer);
            }
        }
        
        
        Array<int> voxel_num_tmp;
        for (i=0;i<voxel_num.size();i++)
            voxel_num_tmp.push_back(voxel_num[i]);
        
        sort(voxel_num_tmp.begin(),voxel_num_tmp.end());
        int median = voxel_num_tmp[round(voxel_num_tmp.size()*0.5)];
        
        _small_slices.clear();
        for (i=0;i<voxel_num.size();i++)
            if (voxel_num[i]<0.1*median)
                _small_slices.push_back(i);
        
        if (_debug) {
            cout<<"Small slices:";
            for (i=0;i<_small_slices.size();i++)
                cout<<" "<<_small_slices[i];
            cout<<endl;
        }
    }
    
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::InitializeEM()
    {
        
        if (_debug)
            cout << "InitializeEM" << endl;
        
        
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

            _scale[inputIndex] = 1;
            _slice_weight[inputIndex] = 1;
            
        }

        _max_intensity = voxel_limits<RealPixel>::min();
        _min_intensity = voxel_limits<RealPixel>::max();

        for (int i = 0; i < _slices.size(); i++) {
            
            for (int y=0; y<_slices[i]->GetY(); y++) {
                for (int x=0; x<_slices[i]->GetX(); x++) {
                    
                    double val = _slices[i]->GetAsDouble(x,y,0);
                    _weights[i]->PutAsDouble(x,y,0,val);
                    _bias[i]->PutAsDouble(x,y,0,val);
                    
                    
                    if (_slices[i]->GetAsDouble(x,y,0) > 0) {
                        if (_slices[i]->GetAsDouble(x,y,0) > _max_intensity)
                            _max_intensity = _slices[i]->GetAsDouble(x,y,0);
                        
                        if (_slices[i]->GetAsDouble(x,y,0) < _min_intensity)
                            _min_intensity = _slices[i]->GetAsDouble(x,y,0);
                        
                    }

                }
            }
        }
        
        
        
        if (_multiple_channels_flag) {
                  
            for (int n=0; n<_number_of_channels; n++) {
                _max_intensity_mc.push_back(voxel_limits<RealPixel>::min());
                _min_intensity_mc.push_back(voxel_limits<RealPixel>::max());
            }
            
            for (int n=0; n<_number_of_channels; n++) {
                
                for (int i = 0; i < _slices.size(); i++) {
                    
                    for (int y=0; y<_slices[i]->GetY(); y++) {
                        for (int x=0; x<_slices[i]->GetX(); x++) {
                            
                            if (_slices[i]->GetAsDouble(x,y,0) > 0) {
                                
                                if (_mc_slices[i][n]->GetAsDouble(x,y,0) > _max_intensity_mc[n])
                                    _max_intensity_mc[n] = _mc_slices[i][n]->GetAsDouble(x,y,0);
                                
                                if (_mc_slices[i][n]->GetAsDouble(x,y,0) < _min_intensity_mc[n])
                                    _min_intensity_mc[n] = _mc_slices[i][n]->GetAsDouble(x,y,0);
                                
                            }
                            
                        }
                    }
                }

            }
        
        }
        
        
        
        
        for (int i = 0; i < _slices.size(); i++) {

            if (_stack_index[i] == _excluded_stack) {
                _slice_weight[i] = 0;
            }
        }

    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::InitializeEMValues()
    {
        
        for (unsigned int i = 0; i < _slices.size(); i++) {

            RealPixel *pw = _weights[i]->Data();
            RealPixel *pb = _bias[i]->Data();
            RealPixel *pi = _slices[i]->Data();
            for (int j = 0; j < _weights[i]->NumberOfVoxels(); j++) {
                if (*pi > -1) {
                    *pw = 1;
                    *pb = 0;
                }
                else {
                    *pw = 0;
                    *pb = 0;
                }
                pi++;
                pw++;
                pb++;
            }

            _slice_weight[i] = 1;
            
            _scale[i] = 1;
            
            if (_stack_index[i] == _excluded_stack)
                _slice_weight[i] = 0;
        }
        
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::InitializeRobustStatistics()
    {
        
        int i, j;
        RealImage slice, sim;
        double sigma = 0;
        int num = 0;
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            slice = *_slices[inputIndex];

            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) > -1) {

                        if ( (_simulated_inside[inputIndex]->GetAsDouble(i, j, 0)==1) &&(_simulated_weights[inputIndex]->GetAsDouble(i,j,0)>0.99) ) {
                            slice(i,j,0) -= _simulated_slices[inputIndex]->GetAsDouble(i,j,0);
                            sigma += slice(i, j, 0) * slice(i, j, 0);
                            num++;
                        }
                    }
            
            if (!_slice_inside[inputIndex])
                _slice_weight[inputIndex] = 0;
        }
        
        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            _slice_weight[_force_excluded[i]] = 0;
        
        bool done = false;

        for (unsigned int i = 0; i < _slices.size(); i++) {
            if (_stack_index[i] == _excluded_stack) {
                _slice_weight[i] = 0;
                if (!done) {
                    cout << "Excluded stack : " << _stack_index[i] << endl;
                    done = true;
                }
            }
        }
        
        _sigma = sigma / num;
        _sigma_s = 0.025;
        _mix = 0.9;
        _mix_s = 0.9;
        _m = 1 / (2.1 * _max_intensity - 1.9 * _min_intensity);
        
        if (_debug)
            cout << "Initializing robust statistics: " << "sigma=" << sqrt(_sigma) << " " << "m=" << _m
            << " " << "mix=" << _mix << " " << "mix_s=" << _mix_s << endl;
        
    }
    
    
    //-------------------------------------------------------------------
    
    class ParallelEStepFFD {
        ReconstructionFFD* reconstructor;
        Array<double> &slice_potential;
        
    public:
        
        void operator()( const blocked_range<size_t>& r ) const {
            

            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {

                double num = 0;
                for (int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++) {
                        
                        reconstructor->_weights[inputIndex]->PutAsDouble(i, j, 0, 0);
                        
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) != -1) {
                            
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();

                            if ( (n>0) && (reconstructor->_simulated_weights[inputIndex]->GetAsDouble(i,j,0) > 0) ) {

                                double g = reconstructor->G(reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0), reconstructor->_sigma);
                                double m = reconstructor->M(reconstructor->_m);
                                
                                double weight = g * reconstructor->_mix / (g *reconstructor->_mix + m * (1 - reconstructor->_mix));
                                reconstructor->_weights[inputIndex]->PutAsDouble(i, j, 0, weight);
                                
                                if(reconstructor->_simulated_weights[inputIndex]->GetAsDouble(i,j,0)>0.99) {
                                    slice_potential[inputIndex] += (1 - weight) * (1 - weight);
                                    num++;
                                }
                            }
                            else
                                reconstructor->_weights[inputIndex]->PutAsDouble(i, j, 0, 0);
                        }
                
                    }

                if (num > 0)
                    slice_potential[inputIndex] = sqrt(slice_potential[inputIndex] / num);
                else
                    slice_potential[inputIndex] = -1;
            }
             
             
            
            
        }
        
        ParallelEStepFFD( ReconstructionFFD *reconstructor,
                      Array<double> &slice_potential ) :
        reconstructor(reconstructor), slice_potential(slice_potential)
        { }
        
        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::EStep()
    {

        unsigned int inputIndex;
        RealImage slice, w, b, sim;
        int num = 0;
        Array<double> slice_potential(_slices.size(), 0);


        SliceDifference();
        
        ParallelEStepFFD *p_estep = new ParallelEStepFFD( this, slice_potential );
        (*p_estep)();

        delete p_estep;

        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            slice_potential[_force_excluded[i]] = -1;
        
        for (unsigned int i = 0; i < _small_slices.size(); i++)
            slice_potential[_small_slices[i]] = -1;

        for (inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
            if ((_scale[inputIndex]<0.2)||(_scale[inputIndex]>5)) {
                slice_potential[inputIndex] = -1;
            }
        
        for (unsigned int i = 0; i < _slices.size(); i++) {
            if (_stack_index[i] == _excluded_stack) {
                slice_potential[i] = -1;
            }
        }

        double sum = 0, den = 0, sum2 = 0, den2 = 0, maxs = 0, mins = 1;
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            if (slice_potential[inputIndex] >= 0) {

                sum += slice_potential[inputIndex] * _slice_weight[inputIndex];
                den += _slice_weight[inputIndex];
                sum2 += slice_potential[inputIndex] * (1 - _slice_weight[inputIndex]);
                den2 += (1 - _slice_weight[inputIndex]);

                if (slice_potential[inputIndex] > maxs)
                    maxs = slice_potential[inputIndex];
                if (slice_potential[inputIndex] < mins)
                    mins = slice_potential[inputIndex];
            }
        
        if (den > 0)
            _mean_s = sum / den;
        else
            _mean_s = mins;
        
        if (den2 > 0)
            _mean_s2 = sum2 / den2;
        else
            _mean_s2 = (maxs + _mean_s) / 2;
        
        sum = 0;
        den = 0;
        sum2 = 0;
        den2 = 0;
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            if (slice_potential[inputIndex] >= 0) {
                sum += (slice_potential[inputIndex] - _mean_s) * (slice_potential[inputIndex] - _mean_s)
                * _slice_weight[inputIndex];
                den += _slice_weight[inputIndex];
                
                sum2 += (slice_potential[inputIndex] - _mean_s2) * (slice_potential[inputIndex] - _mean_s2)
                * (1 - _slice_weight[inputIndex]);
                den2 += (1 - _slice_weight[inputIndex]);
                
            }
        
        if ((sum > 0) && (den > 0)) {
            _sigma_s = sum / den;
            if (_sigma_s < _step * _step / 6.28)
                _sigma_s = _step * _step / 6.28;
        }
        else {
            _sigma_s = 0.025;
            if (_debug) {
                if (sum <= 0)
                    cout << "All slices are equal. ";
                if (den < 0)
                    cout << "All slices are outliers. ";
                cout << "Setting sigma to " << sqrt(_sigma_s) << endl;
            }
        }
        
        if ((sum2 > 0) && (den2 > 0)) {
            _sigma_s2 = sum2 / den2;
            if (_sigma_s2 < _step * _step / 6.28)
                _sigma_s2 = _step * _step / 6.28;
        }
        else {
            _sigma_s2 = (_mean_s2 - _mean_s) * (_mean_s2 - _mean_s) / 4;
            if (_sigma_s2 < _step * _step / 6.28)
                _sigma_s2 = _step * _step / 6.28;
            
            if (_debug) {
                if (sum2 <= 0)
                    cout << "All slices are equal. ";
                if (den2 <= 0)
                    cout << "All slices inliers. ";
                cout << "Setting sigma_s2 to " << sqrt(_sigma_s2) << endl;
            }
        }
        
        double gs1, gs2;
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            if (slice_potential[inputIndex] == -1) {
                _slice_weight[inputIndex] = 0;
                continue;
            }
            
            if ((den <= 0) || (_mean_s2 <= _mean_s)) {
                _slice_weight[inputIndex] = 1;
                continue;
            }
            
            if (slice_potential[inputIndex] < _mean_s2)
                gs1 = G(slice_potential[inputIndex] - _mean_s, _sigma_s);
            else
                gs1 = 0;

            if (slice_potential[inputIndex] > _mean_s)
                gs2 = G(slice_potential[inputIndex] - _mean_s2, _sigma_s2);
            else
                gs2 = 0;

            double likelihood = gs1 * _mix_s + gs2 * (1 - _mix_s);
            if (likelihood > 0)
                _slice_weight[inputIndex] = gs1 * _mix_s / likelihood;
            else {
                if (slice_potential[inputIndex] <= _mean_s)
                    _slice_weight[inputIndex] = 1;
                if (slice_potential[inputIndex] >= _mean_s2)
                    _slice_weight[inputIndex] = 0;
                if ((slice_potential[inputIndex] < _mean_s2) && (slice_potential[inputIndex] > _mean_s)) //should not happen
                    _slice_weight[inputIndex] = 1;
            }
        }
        
        sum = 0;
        num = 0;
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            if (slice_potential[inputIndex] >= 0) {
                sum += _slice_weight[inputIndex];
                num++;
            }
        
        if (num > 0)
            _mix_s = sum / num;
        else {
            cout << "All slices are outliers. Setting _mix_s to 0.9." << endl;
            _mix_s = 0.9;
        }
        
        if (_debug) {
            cout << setprecision(3);
            cout << "Slice robust statistics parameters: ";
            cout << "means: " << _mean_s << " " << _mean_s2 << "  ";
            cout << "sigmas: " << sqrt(_sigma_s) << " " << sqrt(_sigma_s2) << "  ";
            cout << "proportions: " << _mix_s << " " << 1 - _mix_s << endl;
            cout << endl;
        }
        
    }
    
    
    //-------------------------------------------------------------------
    
    class ParallelScaleFFD {
        ReconstructionFFD *reconstructor;
        
    public:
        ParallelScaleFFD( ReconstructionFFD *_reconstructor ) :
        reconstructor(_reconstructor)
        { }
        
        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                double scalenum = 0;
                double scaleden = 0;
                
                for (int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++)
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) != -1) {
                            if(reconstructor->_simulated_weights[inputIndex]->GetAsDouble(i,j,0)>0.99) {
                                double eb = exp(-reconstructor->_bias[inputIndex]->GetAsDouble(i, j, 0));
                                scalenum += reconstructor->_weights[inputIndex]->GetAsDouble(i, j, 0) * reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) * eb * reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i, j, 0);
                                scaleden += reconstructor->_weights[inputIndex]->GetAsDouble(i, j, 0) * reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) * eb * reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) * eb;
                            }
                        }
                
                if (scaleden > 0)
                    reconstructor->_scale[inputIndex] = scalenum / scaleden;
                else
                    reconstructor->_scale[inputIndex] = 1;
                
            }
        }

        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::Scale()
    {
        
        ParallelScaleFFD *p_scale = new ParallelScaleFFD( this );
        (*p_scale)();

        delete p_scale;

        
        if (_debug) {
            cout << setprecision(3);
            cout << "Slice scale = ";
            for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex)
                cout << _scale[inputIndex] << " ";
            cout << endl;
        }
    }
    
    //-------------------------------------------------------------------
    
    class ParallelBiasFFD {
        ReconstructionFFD* reconstructor;
        
    public:
        
        void operator()( const blocked_range<size_t>& r ) const {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {

                RealImage slice = *reconstructor->_slices[inputIndex];
                RealImage& w = *reconstructor->_weights[inputIndex];
                RealImage b = *reconstructor->_bias[inputIndex];
                double scale = reconstructor->_scale[inputIndex];

                RealImage wb = w;

                RealImage wresidual( slice.Attributes() );
                wresidual = 0;
                
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -1) {
                            if( reconstructor->_simulated_weights[inputIndex]->GetAsDouble(i,j,0) > 0.99 ) {

                                double eb = exp(-b(i, j, 0));
                                slice(i, j, 0) *= (eb * scale);
                                wb(i, j, 0) = w(i, j, 0) * slice(i, j, 0);

                                if ( (reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i, j, 0) > 1) && (slice(i, j, 0) > 1)) {
                                    wresidual(i, j, 0) = log(slice(i, j, 0) / reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i, j, 0)) * wb(i, j, 0);
                                }
                            }
                            else {
                                wresidual(i, j, 0) = 0;
                                wb(i, j, 0) = 0;
                            }
                        }
                
                GaussianBlurring<RealPixel> gb(reconstructor->_sigma_bias);
                gb.Input(&wresidual);
                gb.Output(&wresidual);
                gb.Run();
                
                gb.Input(&wb);
                gb.Output(&wb);
                gb.Run();
                
                double sum = 0;
                double num = 0;
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -1) {
                            if (wb(i, j, 0) > 0)
                                b(i, j, 0) += wresidual(i, j, 0) / wb(i, j, 0);
                            sum += b(i, j, 0);
                            num++;
                        }
                
                if (!reconstructor->_global_bias_correction) {
                    double mean = 0;
                    if (num > 0)
                        mean = sum / num;
                    for (int i = 0; i < slice.GetX(); i++)
                        for (int j = 0; j < slice.GetY(); j++)
                            if ((slice(i, j, 0) > -1) && (num > 0)) {
                                b(i, j, 0) -= mean;
                            }
                }

                reconstructor->_bias[inputIndex] = new RealImage(b);
            }
        }
        
        ParallelBiasFFD( ReconstructionFFD *reconstructor ) :
        reconstructor(reconstructor)
        { }

        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::Bias()
    {
        ParallelBiasFFD *p_bias = new ParallelBiasFFD( this );
        (*p_bias)();

        delete p_bias;

    }
    

    //-------------------------------------------------------------------

    void ReconstructionFFD::SliceDifference() 
    {

        for (int inputIndex=0; inputIndex<_slices.size(); inputIndex++) {

            for (int i = 0; i < _slices[inputIndex]->GetX(); i++) {
                for (int j = 0; j < _slices[inputIndex]->GetY(); j++) {
                    
                    
                    double ds = _slices[inputIndex]->GetAsDouble(i, j, 0);
                    
                    
                    Array<double> mc_ds;
                    if (_multiple_channels_flag) {
                        for (int n=0; n<_number_of_channels; n++) {
                            mc_ds.push_back(_mc_slices[inputIndex][n]->GetAsDouble(i, j, 0));
                        }
                    }
                
                    
                    if (_slices[inputIndex]->GetAsDouble(i, j, 0) != -1) {

                        ds = ds * exp(-(_bias[inputIndex])->GetAsDouble(i, j, 0)) * _scale[inputIndex];
                        ds = ds - _simulated_slices[inputIndex]->GetAsDouble(i, j, 0);
                        
                        if (_multiple_channels_flag) {
                            for (int n=0; n<_number_of_channels; n++) {
                                mc_ds[n] = mc_ds[n] * exp(-(_bias[inputIndex])->GetAsDouble(i, j, 0)) * _scale[inputIndex];
                                mc_ds[n] = mc_ds[n] - _mc_simulated_slices[inputIndex][n]->GetAsDouble(i, j, 0);
                            }
                        }

                    }
                    else {
                        ds = 0;
                        
                        if (_multiple_channels_flag) {
                            for (int n=0; n<_number_of_channels; n++) {
                                mc_ds[n] = 0;
                            }
                        }
                    }
                    
                    _slice_dif[inputIndex]->PutAsDouble(i, j, 0, ds);
                    
                    if (_multiple_channels_flag) {
                        for (int n=0; n<_number_of_channels; n++) {
                            _mc_slice_dif[inputIndex][n]->PutAsDouble(i, j, 0, mc_ds[n]);
                        }
                    }

                }
            }
            
        }

    }

    //-------------------------------------------------------------------
    
    class ParallelSuperresolutionFFD {
        ReconstructionFFD* reconstructor;
    public:
        RealImage confidence_map;
        RealImage addon;
        Array<RealImage> mc_addons;

        void operator()( const blocked_range<size_t>& r ) {
            
            
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {

                double scale = reconstructor->_scale[inputIndex];

                POINT3D p;
                for (int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++)
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) != -1) {

                            if (reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i, j, 0) < 0.01) {
                                reconstructor->_slice_dif[inputIndex]->PutAsDouble(i, j, 0, 0);
                             
                                if (reconstructor->_multiple_channels_flag) {
                                    for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                        reconstructor->_mc_slice_dif[inputIndex][nc]->PutAsDouble(i, j, 0, 0);
                                    }
                                }
                                
                            }
                                

                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for (int k = 0; k < n; k++) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];

                                double jac = reconstructor->_mffd_transformations[inputIndex]->Jacobian(p.x, p.y, p.z, 0, 0);
                                
                                if ((100*jac) > 60) {
                                
                                    if (reconstructor->_structural_exclusion) {

                                        if (reconstructor->_stack_index[inputIndex] != reconstructor->_excluded_stack) {
                                            
                                            
                                            if (reconstructor->_slice_weight[inputIndex] < 0.6) {
                                                reconstructor->_structural_slice_weight[inputIndex] = reconstructor->_slice_weight[inputIndex]/10;
                                            }
                                            else {
                                                reconstructor->_structural_slice_weight[inputIndex] = reconstructor->_slice_weight[inputIndex];
                                            }

                                            double sw = reconstructor->_slice_2dncc[inputIndex]->GetAsDouble(i, j, 0);
                                            
                                            addon(p.x, p.y, p.z) += sw * p.value * reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0) * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_structural_slice_weight[inputIndex];
                                            confidence_map(p.x, p.y, p.z) += sw * p.value * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_structural_slice_weight[inputIndex];
                                            
                                            
                                            if (reconstructor->_multiple_channels_flag) {
                                                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                                    mc_addons[nc](p.x, p.y, p.z) += sw * p.value * reconstructor->_mc_slice_dif[inputIndex][nc]->GetAsDouble(i, j, 0) * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_structural_slice_weight[inputIndex];
                                                }
                                            }
                                        }

                                    }
                                    else {
                                        if (reconstructor->_robust_slices_only) {

                                            addon(p.x, p.y, p.z) += p.value * reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                            confidence_map(p.x, p.y, p.z) += p.value * reconstructor->_slice_weight[inputIndex];
                                            
                                            
                                            if (reconstructor->_multiple_channels_flag) {
                                                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                                    mc_addons[nc](p.x, p.y, p.z) += p.value * reconstructor->_mc_slice_dif[inputIndex][nc]->GetAsDouble(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                                }
                                            }

                                        } else {
                                            addon(p.x, p.y, p.z) += p.value * reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0) * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                            confidence_map(p.x, p.y, p.z) += p.value * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                            
                                            
                                            if (reconstructor->_multiple_channels_flag) {
                                                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                                                    mc_addons[nc](p.x, p.y, p.z) += p.value * reconstructor->_mc_slice_dif[inputIndex][nc]->GetAsDouble(i, j, 0) * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                                }
                                            }
                                        }
                                    }
                                
                                }

                            }
                        }
            }
            
            
            
        }
        
        ParallelSuperresolutionFFD( ParallelSuperresolutionFFD& x, split ) :
        reconstructor(x.reconstructor)
        {

            addon.Initialize( reconstructor->_reconstructed.Attributes() );
            addon = 0;
            
            confidence_map.Initialize( reconstructor->_reconstructed.Attributes() );
            confidence_map = 0;
            
            if (reconstructor->_multiple_channels_flag) {
                mc_addons.clear();
                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                    mc_addons.push_back(addon);
                }
            }
            
        }
        
        void join( const ParallelSuperresolutionFFD& y ) {
            addon += y.addon;
            confidence_map += y.confidence_map;
            
            if (reconstructor->_multiple_channels_flag) {
                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                    mc_addons[nc] += y.mc_addons[nc];
                }
            }
            
        }
        
        ParallelSuperresolutionFFD( ReconstructionFFD *reconstructor ) :
        reconstructor(reconstructor)
        {

            addon.Initialize( reconstructor->_reconstructed.Attributes() );
            addon = 0;

            confidence_map.Initialize( reconstructor->_reconstructed.Attributes() );
            confidence_map = 0;
            
            if (reconstructor->_multiple_channels_flag) {
                mc_addons.clear();
                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                    mc_addons.push_back(addon);
                }
            }
            
            
        }
        
        void operator() () {
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()), *this );
        }
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::Superresolution(int iter)
    {

        RealImage addon, original;

        original = _reconstructed;

        SliceDifference();
        
        ParallelSuperresolutionFFD *p_super = new  ParallelSuperresolutionFFD(this);
        (*p_super)();
        addon = p_super->addon;
        _confidence_map = p_super->confidence_map;
        
        
        Array<RealImage> mc_addons;
        Array<RealImage> mc_originals;
        
        if (_multiple_channels_flag) {
            mc_addons = p_super->mc_addons;
            mc_originals = _mc_reconstructed;
        }
        

        delete p_super;
        
        if(_debug) {
            char buffer[256];
            sprintf(buffer,"confidence-map-%i.nii.gz",_current_iteration);
            _confidence_map.Write(buffer);
            
            sprintf(buffer,"addon-%i.nii.gz",_current_iteration);
            addon.Write(buffer);
            
            if (_multiple_channels_flag) {
                   
                sprintf(buffer,"mc-addon-%i.nii.gz",_current_iteration);
                mc_addons[0].Write(buffer);
                   
            }
            
        }
        
        
        
        GaussianBlurring<RealPixel> gb_a(addon.GetXSize()*0.55);
        
        gb_a.Input(&addon);
        gb_a.Output(&addon);
        gb_a.Run();
        
        
        GaussianBlurring<RealPixel> gb_cf(_confidence_map.GetXSize()*0.55);
        
        gb_cf.Input(&_confidence_map);
        gb_cf.Output(&_confidence_map);
        gb_cf.Run();
        
        
        
        if (_multiple_channels_flag) {
            for (int nc=0; nc<_number_of_channels; nc++) {
                gb_a.Input(&mc_addons[nc]);
                gb_a.Output(&mc_addons[nc]);
                gb_a.Run();
            }
        }
        
        

        if(_debug) {
            char buffer[256];
            
            sprintf(buffer,"after-conf-%i.nii.gz",_current_iteration);
            _confidence_map.Write(buffer);
        }
        
        
        if (!_adaptive)
        for (int i = 0; i < addon.GetX(); i++) {
            for (int j = 0; j < addon.GetY(); j++) {
                for (int k = 0; k < addon.GetZ(); k++) {
                    if (_confidence_map(i, j, k) > 0) {

                        addon(i, j, k) /= _confidence_map(i, j, k);
                        
                        if (_multiple_channels_flag) {
                            for (int nc=0; nc<_number_of_channels; nc++) {
                                mc_addons[nc](i, j, k) /= _confidence_map(i, j, k);
                            }
                        }
                        
                        _confidence_map(i,j,k) = 1;
                        
                        
                    }
                    else  {
                        _confidence_map(i,j,k) = 1;
                        
                    }
                }
            }
        }
        

        if(_debug) {
            char buffer[256];
            
            sprintf(buffer,"after-addon-%i.nii.gz",_current_iteration);
            addon.Write(buffer);
        }
        
        _reconstructed += addon * _alpha;
        
        
        
        if (_multiple_channels_flag) {
            for (int nc=0; nc<_number_of_channels; nc++) {
            
                _mc_reconstructed[nc] += mc_addons[nc] * _alpha;
            }
        }
        
        

        for (int i = 0; i < _reconstructed.GetX(); i++)
            for (int j = 0; j < _reconstructed.GetY(); j++)
                for (int k = 0; k < _reconstructed.GetZ(); k++) {
                    if (_reconstructed(i, j, k) < _min_intensity * 0.9)
                        _reconstructed(i, j, k) = _min_intensity * 0.9;
                    if (_reconstructed(i, j, k) > _max_intensity * 1.1)
                        _reconstructed(i, j, k) = _max_intensity * 1.1;
                }
        
        
        if (_multiple_channels_flag) {
            for (int nc=0; nc<_number_of_channels; nc++) {
                for (int i = 0; i < _reconstructed.GetX(); i++)
                for (int j = 0; j < _reconstructed.GetY(); j++)
                for (int k = 0; k < _reconstructed.GetZ(); k++) {
                    if (_mc_reconstructed[nc](i, j, k) < _min_intensity_mc[nc] * 0.9)
                        _mc_reconstructed[nc](i, j, k) = _min_intensity_mc[nc] * 0.9;
                    if (_mc_reconstructed[nc](i, j, k) > _max_intensity_mc[nc] * 1.1)
                        _mc_reconstructed[nc](i, j, k) = _max_intensity_mc[nc] * 1.1;
                }
                
            }
        }
        
        
        
        AdaptiveRegularization(iter, original);
        if (_global_bias_correction)
            BiasCorrectVolume(original);
        
        
        if (_multiple_channels_flag) {
            AdaptiveRegularizationMC(iter, mc_originals);
        }
        

    }
    
    
    //-------------------------------------------------------------------
    
    class ParallelMStepFFD{
        ReconstructionFFD* reconstructor;
    public:
        double sigma;
        double mix;
        double num;
        double min;
        double max;
        
        void operator()( const blocked_range<size_t>& r ) {
            
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                
                for (int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++)
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) > -1) {

                            if ( reconstructor->_simulated_weights[inputIndex]->GetAsDouble(i,j,0) > 0.99 ) {
                                
                                double e = reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0);
                                sigma += e * e * reconstructor->_weights[inputIndex]->GetAsDouble(i, j, 0);
                                mix += reconstructor->_weights[inputIndex]->GetAsDouble(i, j, 0);

                                if (e < min)
                                    min = e;
                                if (e > max)
                                    max = e;
                                
                                num++;
                            }
                        }
            }

        }
        
        ParallelMStepFFD( ParallelMStepFFD& x, split ) :
        reconstructor(x.reconstructor)
        {
            sigma = 0;
            mix = 0;
            num = 0;
            min = 0;
            max = 0;
        }
        
        void join( const ParallelMStepFFD& y ) {
            if (y.min < min)
                min = y.min;
            if (y.max > max)
                max = y.max;
            
            sigma += y.sigma;
            mix += y.mix;
            num += y.num;
        }
        
        ParallelMStepFFD( ReconstructionFFD *reconstructor ) :
        reconstructor(reconstructor)
        {
            sigma = 0;
            mix = 0;
            num = 0;
            min = voxel_limits<RealPixel>::max();
            max = voxel_limits<RealPixel>::min();
        }
        
        void operator() () {
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()), *this );
        }
    };
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::MStep(int iter)
    {
        
        SliceDifference();

        ParallelMStepFFD *m_step = new ParallelMStepFFD(this);
        (*m_step)();


        double sigma = m_step->sigma;
        double mix = m_step->mix;
        double num = m_step->num;
        double min = m_step->min;
        double max = m_step->max;

        delete m_step;

        if (mix > 0) {
            _sigma = sigma / mix;
        }
        else {
            cerr << "Something went wrong: sigma=" << sigma << " mix=" << mix << endl;
            exit(1);
        }
        if (_sigma < _step * _step / 6.28)
            _sigma = _step * _step / 6.28;
        if (iter > 1)
            _mix = mix / num;
        
        _m = 1 / (max - min);
        
        if (_debug) {
            cout << "Voxel-wise robust statistics parameters: ";
            cout << "sigma = " << sqrt(_sigma) << " mix = " << _mix << " ";
            cout << " m = " << _m << endl;
        }
        
    }
    
    //-------------------------------------------------------------------
    
    class ParallelAdaptiveRegularization1FFD {
        ReconstructionFFD *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;
        
    public:
        ParallelAdaptiveRegularization1FFD( ReconstructionFFD *_reconstructor,
                                        Array<RealImage> &_b,
                                        Array<double> &_factor,
                                        RealImage &_original) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            int dx = reconstructor->_reconstructed.GetX();
            int dy = reconstructor->_reconstructed.GetY();
            int dz = reconstructor->_reconstructed.GetZ();
            for ( size_t i = r.begin(); i != r.end(); ++i ) {
                
                int x, y, z, xx, yy, zz;
                double diff;
                for (x = 0; x < dx; x++)
                    for (y = 0; y < dy; y++)
                        for (z = 0; z < dz; z++) {
                            xx = x + reconstructor->_directions[i][0];
                            yy = y + reconstructor->_directions[i][1];
                            zz = z + reconstructor->_directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                && (reconstructor->_confidence_map(x, y, z) > 0) && (reconstructor->_confidence_map(xx, yy, zz) > 0)) {
                                diff = (original(xx, yy, zz) - original(x, y, z)) * sqrt(factor[i]) / reconstructor->_delta;
                                b[i](x, y, z) = factor[i] / sqrt(1 + diff * diff);
                                
                            }
                            else
                                b[i](x, y, z) = 0;
                        }
            }
        }
        
        void operator() () const {
            parallel_for( blocked_range<size_t>(0, 13), *this );
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    class ParallelAdaptiveRegularization2FFD {
        ReconstructionFFD *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;
        
    public:
        ParallelAdaptiveRegularization2FFD( ReconstructionFFD *_reconstructor,
                                        Array<RealImage> &_b,
                                        Array<double> &_factor,
                                        RealImage &_original) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            int dx = reconstructor->_reconstructed.GetX();
            int dy = reconstructor->_reconstructed.GetY();
            int dz = reconstructor->_reconstructed.GetZ();
            for ( size_t x = r.begin(); x != r.end(); ++x ) {
                int xx, yy, zz;
                for (int y = 0; y < dy; y++)
                    for (int z = 0; z < dz; z++)
                        if(reconstructor->_confidence_map(x,y,z)>0)
                        {
                            double val = 0;
                            double sum = 0;
                            for (int i = 0; i < 13; i++) {
                                xx = x + reconstructor->_directions[i][0];
                                yy = y + reconstructor->_directions[i][1];
                                zz = z + reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                                    if(reconstructor->_confidence_map(xx,yy,zz)>0)
                                    {
                                        val += b[i](x, y, z) * original(xx, yy, zz);
                                        sum += b[i](x, y, z);
                                    }
                            }
                            
                            for (int i = 0; i < 13; i++) {
                                xx = x - reconstructor->_directions[i][0];
                                yy = y - reconstructor->_directions[i][1];
                                zz = z - reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                                    if(reconstructor->_confidence_map(xx,yy,zz)>0)
                                    {
                                        val += b[i](x, y, z) * original(xx, yy, zz);
                                        sum += b[i](x, y, z);
                                    }
                                
                            }
                            
                            val -= sum * original(x, y, z);
                            val = original(x, y, z) + reconstructor->_alpha * reconstructor->_lambda / (reconstructor->_delta * reconstructor->_delta) * val;
                            
                            reconstructor->_reconstructed(x, y, z) = val;
                        }
                
            }
        }
        
        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed.GetX()), *this );
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::AdaptiveRegularization(int iter, RealImage& original)
    {
        
        Array<double> factor(13,0);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }
        
        Array<RealImage> b;
        for (int i = 0; i < 13; i++)
            b.push_back( _reconstructed );
        
        ParallelAdaptiveRegularization1FFD *p_ad1 = new ParallelAdaptiveRegularization1FFD( this,
                                                                        b,
                                                                        factor,
                                                                        original );
        (*p_ad1)();

        delete p_ad1;
        
        RealImage original2 = _reconstructed;
        ParallelAdaptiveRegularization2FFD *p_ad2 = new  ParallelAdaptiveRegularization2FFD( this,
                                                                        b,
                                                                        factor,
                                                                        original2 );
        (*p_ad2)();
        
        delete p_ad2;

        if (_alpha * _lambda / (_delta * _delta) > 0.068) {
            cerr << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068." << endl;
        }
    }
    
    
    //-------------------------------------------------------------------
    
        
    void ReconstructionFFD::BiasCorrectVolume(RealImage& original)
    {

        RealImage residual = _reconstructed;
        RealImage weights = _mask;
        
        RealPixel *pr = residual.Data();
        RealPixel *po = original.Data();
        RealPixel *pw = weights.Data();
        
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {
            
            if ((*pw == 1) && (*po > _low_intensity_cutoff * _max_intensity) && (*pr > _low_intensity_cutoff * _max_intensity)) {
                *pr /= *po;
                *pr = log(*pr);
            }
            else {
                *pw = 0;
                *pr = 0;
            }
            pr++;
            po++;
            pw++;
        }

        GaussianBlurring<RealPixel> gb(_sigma_bias);
        gb.Input(&residual);
        gb.Output(&residual);
        gb.Run();

        gb.Input(&weights);
        gb.Output(&weights);
        gb.Run();
        
        pr = residual.Data();
        pw = weights.Data();
        RealPixel *pm = _mask.Data();
        RealPixel *pi = _reconstructed.Data();
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {
            
            if (*pm == 1) {
                *pr /= *pw;
                *pr = exp(*pr);
                *pi /= *pr;
                
                if (*pi < _min_intensity * 0.9)
                    *pi = _min_intensity * 0.9;
                if (*pi > _max_intensity * 1.1)
                    *pi = _max_intensity * 1.1;
            }
            else {
                *pr = 0;
            }
            pr++;
            pw++;
            pm++;
            pi++;
        }
        
    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::Evaluate(int iter)
    {
        cout << "Iteration " << iter << ": " << endl;
        
        cout << "Included slices: ";
        int sum = 0;
        unsigned int i;
        for (i = 0; i < _slices.size(); i++) {
            if ((_slice_weight[i] >= 0.5) && (_slice_inside[i])) {
                cout << i << " ";
                sum++;
            }
        }
        cout << endl << "Total: " << sum << endl;
        
        cout << "Excluded slices: ";
        sum = 0;
        for (i = 0; i < _slices.size(); i++) {
            if ((_slice_weight[i] < 0.5) && (_slice_inside[i])) {
                cout << i << " ";
                sum++;
            }
        }
        cout << endl << "Total: " << sum << endl;
        
        cout << "Outside slices: ";
        sum = 0;
        for (i = 0; i < _slices.size(); i++) {
            if (!(_slice_inside[i])) {
                cout << i << " ";
                sum++;
            }
        }
        cout << endl << "Total: " << sum << endl;
        

        cout << "Excluded structural slices: ";
        sum = 0;
        for (i = 0; i < _slices.size(); i++) {
            if (_reg_slice_weight[i] < 0.1) {
                cout << i << "(" << _stack_index[i] << ") ";
                sum++;
            }
        }
        cout << endl << "Total: " << sum << endl;
        
    }
    
    
    //-------------------------------------------------------------------
    
    class ParallelNormaliseBiasFFD{
        ReconstructionFFD* reconstructor;
    public:
        RealImage bias;
        
        void operator()( const blocked_range<size_t>& r ) {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                
                RealImage b = *reconstructor->_bias[inputIndex];

                double scale = reconstructor->_scale[inputIndex];
                
                RealPixel *pi = reconstructor->_slices[inputIndex]->Data();
                RealPixel *pb = b.Data();
                for(int i = 0; i<reconstructor->_slices[inputIndex]->NumberOfVoxels(); i++) {
                    if((*pi>-1)&&(scale>0))
                        *pb -= log(scale);
                    pb++;
                    pi++;
                }

                POINT3D p;
                for (int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++)
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) != -1) {

                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();

                            for (int k = 0; k < n; k++) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                bias(p.x, p.y, p.z) += p.value * b(i, j, 0);
                            }
                        }

            }
        }
        
        ParallelNormaliseBiasFFD( ParallelNormaliseBiasFFD& x, split ) :
        reconstructor(x.reconstructor)
        {
            bias.Initialize( reconstructor->_reconstructed.Attributes() );
            bias = 0;
        }
        
        void join( const ParallelNormaliseBiasFFD& y ) {
            bias += y.bias;
        }
        
        ParallelNormaliseBiasFFD( ReconstructionFFD *reconstructor ) :
        reconstructor(reconstructor)
        {
            bias.Initialize( reconstructor->_reconstructed.Attributes() );
            bias = 0;
        }
        
        void operator() () {
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()), *this );
        }
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::NormaliseBias( int iter )
    {
        
        ParallelNormaliseBiasFFD *p_bias = new ParallelNormaliseBiasFFD(this);
        (*p_bias)();
        RealImage bias = p_bias->bias;
        
        delete p_bias;

        bias /= _volume_weights;
        
        MaskImage(bias,0);
        RealImage m = _mask;
        GaussianBlurring<RealPixel> gb(_sigma_bias);
        
        gb.Input(&bias);
        gb.Output(&bias);
        gb.Run();
        
        gb.Input(&m);
        gb.Output(&m);
        gb.Run();
        bias/=m;
        
        RealPixel *pi, *pb;
        pi = _reconstructed.Data();
        pb = bias.Data();
        for (int i = 0; i<_reconstructed.NumberOfVoxels();i++) {
            if(*pi!=-1)
                *pi /=exp(-(*pb));
            pi++;
            pb++;
        }
        
    }

    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SaveBiasFields()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "bias%i.nii.gz", inputIndex);
            _bias[inputIndex]->Write(buffer);
        }
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SaveConfidenceMap()
    {
        _confidence_map.Write("confidence-map.nii.gz");
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SaveSlices()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            sprintf(buffer, "wt-%i-%i.nii.gz", _stack_index[inputIndex], inputIndex);
            _slices[inputIndex]->Write(buffer);
            
        }
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SaveWeights()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "weights%i.nii.gz", inputIndex);
            _weights[inputIndex]->Write(buffer);
        }
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SaveTransformations()
    {
        char buffer[256];
        
        if(_mffd_mode) {
            
            for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

                sprintf(buffer, "t-%i-%i.dof", _stack_index[inputIndex], inputIndex);
                _mffd_transformations[inputIndex]->Write(buffer);
            }
            
        }
        else {
            for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
                sprintf(buffer, "transformation%i.dof", inputIndex);
                _transformations[inputIndex].Write(buffer);
            }
        }
        
    }

    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SaveProbabilityMap( int i )
    {
        char buffer[256];
        sprintf( buffer, "probability_map%i.nii", i );
        _brain_probability.Write( buffer );
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SlicesInfo( const char* filename, Array<string> &stack_files )
    {
        ofstream info;
        info.open( filename );
        
        info << "stack_index" << "\t"
        << "stack_name" << "\t"
        << "included" << "\t"
        << "excluded" << "\t"
        << "outside" << "\t"
        << "weight" << "\t"
        << "scale" << "\t"
        << "TranslationX" << "\t"
        << "TranslationY" << "\t"
        << "TranslationZ" << "\t"
        << "RotationX" << "\t"
        << "RotationY" << "\t"
        << "RotationZ" << endl;
        
        for (int i = 0; i < _slices.size(); i++) {
            RigidTransformation& t = _transformations[i];
            info << _stack_index[i] << "\t"
            << stack_files[_stack_index[i]] << "\t"
            << (((_slice_weight[i] >= 0.5) && (_slice_inside[i]))?1:0) << "\t"
            << (((_slice_weight[i] < 0.5) && (_slice_inside[i]))?1:0) << "\t"
            << ((!(_slice_inside[i]))?1:0) << "\t"
            << _slice_weight[i] << "\t"
            << _scale[i] << "\t"
            << t.GetTranslationX() << "\t"
            << t.GetTranslationY() << "\t"
            << t.GetTranslationZ() << "\t"
            << t.GetRotationX() << "\t"
            << t.GetRotationY() << "\t"
            << t.GetRotationZ() << endl;
        }
        
        info.close();
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SplitImage( RealImage image, int packages, Array<RealImage>& stacks )
    {
        
        ImageAttributes attr = image.Attributes();
        
        int pkg_z = attr._z / packages;
        double pkg_dz = attr._dz*packages;

        char buffer[256];
        int i, j, k, l, t;
        double x, y, z, sx, sy, sz, ox, oy, oz;
        for (l = 0; l < packages; l++) {
            attr = image.Attributes();
            if ((pkg_z * packages + l) < attr._z)
                attr._z = pkg_z + 1;
            else
                attr._z = pkg_z;
            attr._dz = pkg_dz;
            
            RealImage stack(attr);
            stack.GetOrigin(ox, oy, oz);
            
            for (t = 0; t < stack.GetT(); t++) {
                for (k = 0; k < stack.GetZ(); k++)
                    for (j = 0; j < stack.GetY(); j++)
                        for (i = 0; i < stack.GetX(); i++)
                            stack(i, j, k, t) = image(i, j, k * packages + l, t);
            }
            
            x = 0;
            y = 0;
            z = l;
            image.ImageToWorld(x, y, z);
            sx = 0;
            sy = 0;
            sz = 0;
            stack.PutOrigin(ox, oy, oz);
            stack.ImageToWorld(sx, sy, sz);

            stack.PutOrigin(ox + (x - sx), oy + (y - sy), oz + (z - sz));
            sx = 0;
            sy = 0;
            sz = 0;
            stack.ImageToWorld(sx, sy, sz);

            stacks.push_back(stack);
        }

    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::CropImage(RealImage& image, RealImage& mask)
    {

        int i, j, k;
        int x1, x2, y1, y2, z1, z2;
        
        x1 = 0;
        y1 = 0;
        z1 = 0;
        x2 = image.GetX();
        y2 = image.GetY();
        z2 = image.GetZ();

        int sum = 0;
        for (k = image.GetZ() - 1; k >= 0; k--) {
            sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        z2 = k;
        
        sum = 0;
        for (k = 0; k <= image.GetZ() - 1; k++) {
            sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        z1 = k;
        
        sum = 0;
        for (j = image.GetY() - 1; j >= 0; j--) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        y2 = j;
        
        sum = 0;
        for (j = 0; j <= image.GetY() - 1; j++) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        y1 = j;
        
        sum = 0;
        for (i = image.GetX() - 1; i >= 0; i--) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (j = image.GetY() - 1; j >= 0; j--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        x2 = i;
        
        sum = 0;
        for (i = 0; i <= image.GetX() - 1; i++) {
            sum = 0;
            for (k = image.GetZ() - 1; k >= 0; k--)
                for (j = image.GetY() - 1; j >= 0; j--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        
        x1 = i;
        
        if (_debug) {
            if ((x2 < x1) || (y2 < y1) || (z2 < z1) ) {
                x1 = 0;
                y1 = 0;
                z1 = 0;
                x2 = 0;
                y2 = 0;
                z2 = 0;
            }
        }
        
        image = image.GetRegion(x1, y1, z1, x2+1, y2+1, z2+1);
        
    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::CropImageP(RealImage *image, RealImage* mask)
    {

        int i, j, k;
        int x1, x2, y1, y2, z1, z2;
        
        x1 = 0;
        y1 = 0;
        z1 = 0;
        x2 = image->GetX();
        y2 = image->GetY();
        z2 = image->GetZ();

        int sum = 0;
        for (k = image->GetZ() - 1; k >= 0; k--) {
            sum = 0;
            for (j = image->GetY() - 1; j >= 0; j--)
                for (i = image->GetX() - 1; i >= 0; i--)
                    if (mask->Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        z2 = k;
        
        sum = 0;
        for (k = 0; k <= image->GetZ() - 1; k++) {
            sum = 0;
            for (j = image->GetY() - 1; j >= 0; j--)
                for (i = image->GetX() - 1; i >= 0; i--)
                    if (mask->Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        z1 = k;
        
        sum = 0;
        for (j = image->GetY() - 1; j >= 0; j--) {
            sum = 0;
            for (k = image->GetZ() - 1; k >= 0; k--)
                for (i = image->GetX() - 1; i >= 0; i--)
                    if (mask->Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        y2 = j;

        sum = 0;
        for (j = 0; j <= image->GetY() - 1; j++) {
            sum = 0;
            for (k = image->GetZ() - 1; k >= 0; k--)
                for (i = image->GetX() - 1; i >= 0; i--)
                    if (mask->Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        y1 = j;
        
        sum = 0;
        for (i = image->GetX() - 1; i >= 0; i--) {
            sum = 0;
            for (k = image->GetZ() - 1; k >= 0; k--)
                for (j = image->GetY() - 1; j >= 0; j--)
                    if (mask->Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        x2 = i;

        sum = 0;
        for (i = 0; i <= image->GetX() - 1; i++) {
            sum = 0;
            for (k = image->GetZ() - 1; k >= 0; k--)
                for (j = image->GetY() - 1; j >= 0; j--)
                    if (mask->Get(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }
        
        x1 = i;
        
        if (_debug) {

            if ((x2 < x1) || (y2 < y1) || (z2 < z1) ) {
                x1 = 0;
                y1 = 0;
                z1 = 0;
                x2 = 0;
                y2 = 0;
                z2 = 0;
            }
        }

        RealImage tmp = image->GetRegion(x1, y1, z1, x2+1, y2+1, z2+1);
        RealImage *tmp_p = new RealImage(tmp);
        image = tmp_p;
        
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::InvertStackTransformations( Array<RigidTransformation>& stack_transformations )
    {

        for (unsigned int i = 0; i < stack_transformations.size(); i++) {
            stack_transformations[i].Invert();
            stack_transformations[i].UpdateParameter();
        }
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::MaskVolume()
    {
        RealPixel *pr = _reconstructed.Data();
        RealPixel *pm = _mask.Data();
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {
            if (*pm == 0)
                *pr = 0;
            pm++;
            pr++;
        }
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::MaskImage( RealImage& image, double padding )
    {
        if(image.NumberOfVoxels()!=_mask.NumberOfVoxels()) {
            cerr<<"Cannot mask the image - different dimensions"<<endl;
            exit(1);
        }
        RealPixel *pr = image.Data();
        RealPixel *pm = _mask.Data();
        
        for (int i = 0; i < image.NumberOfVoxels(); i++) {
            if (*pm == 0)
                *pr = padding;
            pm++;
            pr++;
        }
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::Rescale( RealImage* img, double max )
    {
        int i, n;
        RealPixel *ptr, min_val, max_val;

        img->GetMinMax(&min_val, &max_val);
        
        n   = img->NumberOfVoxels();
        ptr = img->Data();
        for (i = 0; i < n; i++)
            if ( ptr[i] > 0 )
                ptr[i] = double(ptr[i]) / double(max_val) * max;
    }
    

    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::FilterSlices( double alpha2D, double lambda2D, double delta2D )
    {
        
        for (int i=0; i<_slices.size(); i++) {
            AdaptiveRegularization2D( i, alpha2D, lambda2D, delta2D );
        }

    }
    
    
    //-------------------------------------------------------------------

    
    class Parallel2DAdaptiveRegularization1FFD {
        ReconstructionFFD *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;
        
    public:
        Parallel2DAdaptiveRegularization1FFD( ReconstructionFFD *_reconstructor,
                                             Array<RealImage> &_b,
                                             Array<double> &_factor,
                                             RealImage &_original ) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }
        
        
        void operator() (const blocked_range<size_t> &r) const {
            
            int dx = original.GetX();
            int dy = original.GetY();
            int dz = 0;
            
            for ( size_t i = r.begin(); i != r.end(); ++i ) {
                
                int z = 0;
                int zz = 0;
                
                int x, y, xx, yy;
                double diff;
                
                for (x = 0; x < dx; x++) {
                    for (y = 0; y < dy; y++) {
                        
                        xx = x + reconstructor->_directions2D[i][0];
                        yy = y + reconstructor->_directions2D[i][1];
                        
                        if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) ) {
                            diff = (original(xx, yy, zz) - original(x, y, z)) * sqrt(factor[i]) / reconstructor->_delta2D;
                            b[i](x, y, z) = factor[i] / sqrt(1 + diff * diff);
                        }
                        else {
                            b[i](x, y, z) = 0;
                        }
                    }
                }
                
            }
        }
        
        void operator() () const {
            parallel_for( blocked_range<size_t>(0, 5), *this );
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    class Parallel2DAdaptiveRegularization2FFD {
        ReconstructionFFD *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;
        RealImage &output;
        
    public:
        Parallel2DAdaptiveRegularization2FFD( ReconstructionFFD *_reconstructor,
                                             Array<RealImage> &_b,
                                             Array<double> &_factor,
                                             RealImage &_original,
                                             RealImage &_output ) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original),
        output(_output) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            
            int dx = original.GetX();
            int dy = original.GetY();
            int dz = 0;
            
            for ( size_t x = r.begin(); x != r.end(); ++x ) {
                
                int z = 0;
                int zz = 0;
                
                int xx, yy;
                
                for (int y = 0; y < dy; y++) {
                    
                    double val = 0;
                    double sum = 0;
                    
                    for (int i = 0; i < 5; i++) {
                        
                        xx = x + reconstructor->_directions2D[i][0];
                        yy = y + reconstructor->_directions2D[i][1];
                        
                        if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy)) {
                            
                            val += b[i](x, y, z) * original(xx, yy, zz);
                            sum += b[i](x, y, z);
                        }
                    }
                    
                    for (int i = 0; i < 5; i++) {
                        
                        xx = x - reconstructor->_directions2D[i][0];
                        yy = y - reconstructor->_directions2D[i][1];
                        
                        if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy)) {
                            
                            val += b[i](x, y, z) * original(xx, yy, zz);
                            sum += b[i](x, y, z);
                        }
                        
                    }
                    
                    val -= sum * original(x, y, z);
                    val = original(x, y, z) + reconstructor->_alpha2D * reconstructor->_lambda2D / (reconstructor->_delta2D * reconstructor->_delta2D) * val;
                    
                    output(x, y, z) = val;
                    
                }
                
            }
        }
        
        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_slice.GetX()), *this );
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::AdaptiveRegularization2D( int inputIndex, double alpha, double lambda, double delta )
    {
        
        RealImage main_original = *_slices[inputIndex];
        RealImage output = *_slices[inputIndex];

        _alpha2D = alpha;
        _lambda2D = lambda;
        _delta2D = delta;
        _slice = main_original;
        
        RealImage original;
        
        int directions2D[5][2] = {
            { 1, 0 },
            { 0, 1 },
            { 1, 1 },
            { 1, -1 },
            { 0, -1 }
        };
        
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 2; j++)
                _directions2D[i][j] = directions2D[i][j];
        

        Array<double> factor(5,0);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 2; j++)
                factor[i] += fabs(double(_directions2D[i][j]));
            factor[i] = 1 / factor[i];
        }
        
        Array<RealImage> b;
        for (int i = 0; i < 5; i++)
            b.push_back( original );
        
        original = main_original;
        
        Parallel2DAdaptiveRegularization1FFD *p_ad1 = new Parallel2DAdaptiveRegularization1FFD( this,
                                                                                               b,
                                                                                               factor,
                                                                                               original);
        (*p_ad1)();
        delete p_ad1;
        
        original = main_original;
        Parallel2DAdaptiveRegularization2FFD *p_ad2 = new  Parallel2DAdaptiveRegularization2FFD( this,
                                                                                                b,
                                                                                                factor,
                                                                                                original,
                                                                                                output);

        (*p_ad2)();

        original = output;
        
        
        

        delete p_ad2;
        
    }




    //-------------------------------------------------------------------

    class ParallelAdaptiveRegularization1FFDMC {
        ReconstructionFFD *reconstructor;
        Array<Array<RealImage>> &b;
        Array<double> &factor;
        Array<RealImage> &original;
        
        public:
        ParallelAdaptiveRegularization1FFDMC( ReconstructionFFD *_reconstructor,
                                           Array<Array<RealImage>> &_b,
                                           Array<double> &_factor,
                                           Array<RealImage> &_original) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            int dx = reconstructor->_reconstructed.GetX();
            int dy = reconstructor->_reconstructed.GetY();
            int dz = reconstructor->_reconstructed.GetZ();
            for ( size_t i = r.begin(); i != r.end(); ++i ) {
                
                for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                
                    int x, y, z, xx, yy, zz;
                    double diff;
                    for (x = 0; x < dx; x++)
                    for (y = 0; y < dy; y++)
                    for (z = 0; z < dz; z++) {
                        xx = x + reconstructor->_directions[i][0];
                        yy = y + reconstructor->_directions[i][1];
                        zz = z + reconstructor->_directions[i][2];
                        if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                            && (reconstructor->_confidence_map(x, y, z) > 0) && (reconstructor->_confidence_map(xx, yy, zz) > 0)) {
                            diff = (original[nc](xx, yy, zz) - original[nc](x, y, z)) * sqrt(factor[i]) / reconstructor->_delta;
                            b[i][nc](x, y, z) = factor[i] / sqrt(1 + diff * diff);
                            
                        }
                        else {
                            b[i][nc](x, y, z) = 0;
                        }
                    }
                    
                }
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, 13), *this );
            //init.terminate();
        }
        
    };


    //-------------------------------------------------------------------

    class ParallelAdaptiveRegularization2FFDMC {
        ReconstructionFFD *reconstructor;
        Array<Array<RealImage>> &b;
        Array<double> &factor;
        Array<RealImage> &original;
        
        public:
        ParallelAdaptiveRegularization2FFDMC( ReconstructionFFD *_reconstructor,
                                           Array<Array<RealImage>> &_b,
                                           Array<double> &_factor,
                                           Array<RealImage> &_original) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            int dx = reconstructor->_reconstructed.GetX();
            int dy = reconstructor->_reconstructed.GetY();
            int dz = reconstructor->_reconstructed.GetZ();
            for ( size_t x = r.begin(); x != r.end(); ++x ) {
                int xx, yy, zz;
                for (int y = 0; y < dy; y++)
                for (int z = 0; z < dz; z++)
                if(reconstructor->_confidence_map(x,y,z)>0)
                {
                    
                    for (int nc=0; nc<reconstructor->_number_of_channels; nc++) {
                    
                        double val = 0;
                        double sum = 0;
                        
                        for (int i = 0; i < 13; i++) {
                            xx = x + reconstructor->_directions[i][0];
                            yy = y + reconstructor->_directions[i][1];
                            zz = z + reconstructor->_directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                            if(reconstructor->_confidence_map(xx,yy,zz)>0)
                            {
                                val += b[i][nc](x, y, z) * original[nc](xx, yy, zz);
                                sum += b[i][nc](x, y, z);
                            }
                        }
                        
                        for (int i = 0; i < 13; i++) {
                            xx = x - reconstructor->_directions[i][0];
                            yy = y - reconstructor->_directions[i][1];
                            zz = z - reconstructor->_directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                            if(reconstructor->_confidence_map(xx,yy,zz)>0)
                            {
                                val += b[i][nc](x, y, z) * original[nc](xx, yy, zz);
                                sum += b[i][nc](x, y, z);
                            }
                            
                        }
                        
                        val -= sum * original[nc](x, y, z);
                        val = original[nc](x, y, z) + reconstructor->_alpha * reconstructor->_lambda / (reconstructor->_delta * reconstructor->_delta) * val;
                        
                        reconstructor->_mc_reconstructed[nc](x, y, z) = val;
                    }
                }
                
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed.GetX()), *this );
            //init.terminate();
        }
        
    };


    //-------------------------------------------------------------------


    void ReconstructionFFD::AdaptiveRegularizationMC( int iter, Array<RealImage>& mc_originals)
    {
        
        Array<double> factor(13,0);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
            factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }
        
        Array<Array<RealImage>> b;//(13);
        for (int i = 0; i < 13; i++) {
            b.push_back( _mc_reconstructed );
        }
        
        ParallelAdaptiveRegularization1FFDMC *p_ad1 = new ParallelAdaptiveRegularization1FFDMC( this,
                                                                                           b,
                                                                                           factor,
                                                                                           mc_originals );
        (*p_ad1)();
        
        delete p_ad1;
        
        Array<RealImage> mc_originals2 = _mc_reconstructed;
        ParallelAdaptiveRegularization2FFDMC *p_ad2 = new  ParallelAdaptiveRegularization2FFDMC( this,
                                                                                            b,
                                                                                            factor,
                                                                                            mc_originals2 );
        (*p_ad2)();
        
        delete p_ad2;
        
        if (_alpha * _lambda / (_delta * _delta) > 0.068) {
            cerr << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068." << endl;
        }
        
        
        
    }












    
    
    //-------------------------------------------------------------------
    
} // namespace svrtk
