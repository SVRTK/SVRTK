/*
 *
 */

#include "mirtk/ReconstructionFFD.h"


namespace mirtk {
    
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
        
        _masked = false;

        _cp_spacing.push_back(15);
        _cp_spacing.push_back(10);
        _cp_spacing.push_back(5);
        _cp_spacing.push_back(5);

        _global_NCC_threshold = 0.7; //65;
        _local_NCC_threshold = 0.45; //6; //5;
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
        
        //Get image attributes - image size and voxel size
        ImageAttributes attr = stack->GetImageAttributes();
        
        //enlarge stack in z-direction in case top of the head is cut off
        attr._z += 2;
        
        //create enlarged image
        RealImage enlarged;
        enlarged.Initialize(attr);
        
        //determine resolution of volume to reconstruct
        if (resolution <= 0) {
            //resolution was not given by user set it to min of res in x or y direction
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
        
        //initialize recontructed volume
        _reconstructed = enlarged;
        _template_created = true;
        
        if (_debug)
            _reconstructed.Write("template.nii.gz");
        
        _reconstructed_attr = _reconstructed.GetImageAttributes();
        
        //return resulting resolution of the template image
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
//            stacks[i] = &tmp_stack;
            
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
        RealPixel* ptr = image.GetPointerToVoxels();
        for (int i = 0; i < image.GetNumberOfVoxels(); i++) {
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
            //if sigma is nonzero first smooth the mask
            if (sigma > 0) {
                //blur mask
                GaussianBlurring<RealPixel> gb(sigma);
                
                gb.Input(mask);
                gb.Output(mask);
                gb.Run();
                
                //binarize mask
                RealPixel* ptr = mask->GetPointerToVoxels();
                for (int i = 0; i < mask->GetNumberOfVoxels(); i++) {
                    if (*ptr > threshold)
                        *ptr = 1;
                    else
                        *ptr = 0;
                    ptr++;
                }
            }
            
            //resample the mask according to the template volume using identity transformation
            RigidTransformation transformation;
            ImageTransformation imagetransformation;
            
            //GenericNearestNeighborExtrapolateImageFunction<RealImage> interpolator;
            InterpolationMode interpolation = Interpolation_NN;
            UniquePtr<InterpolateImageFunction> interpolator;
            interpolator.reset(InterpolateImageFunction::New(interpolation));
            
            imagetransformation.Input(mask);
            imagetransformation.Transformation(&transformation);
            imagetransformation.Output(&_mask);
            //target is zero image, need padding -1
            imagetransformation.TargetPaddingValue(-1);
            //need to fill voxels in target where there is no info from source with zeroes
            imagetransformation.SourcePaddingValue(0);
            imagetransformation.Interpolator(interpolator.get());
            imagetransformation.Run();
        }
        else {
            //fill the mask with ones
            _mask = 1;
        }
        //set flag that mask was created
        _have_mask = true;
        
        if (_debug)
            _mask.Write("transformed-mask.nii.gz");
        
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::TransformMask(RealImage& image, RealImage& mask, RigidTransformation& transformation)
    {
        //transform mask to the space of image
        ImageTransformation imagetransformation;
        
        InterpolationMode interpolation = Interpolation_NN;
        UniquePtr<InterpolateImageFunction> interpolator;
        interpolator.reset(InterpolateImageFunction::New(interpolation));
        
        imagetransformation.Input(&mask);
        imagetransformation.Transformation(&transformation);
        
        RealImage m = image;
        imagetransformation.Output(&m);
        //target contains zeros and ones image, need padding -1
        imagetransformation.TargetPaddingValue(-1);
        //need to fill voxels in target where there is no info from source with zeroes
        imagetransformation.SourcePaddingValue(0);
        imagetransformation.Interpolator(interpolator.get());
        imagetransformation.Run();
        mask = m;
        
    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::TransformMaskP(RealImage* image, RealImage& mask)
    {
        //transform mask to the space of image
        ImageTransformation imagetransformation;
        
        
        RigidTransformation* rr = new RigidTransformation;
        
        InterpolationMode interpolation = Interpolation_NN;
        UniquePtr<InterpolateImageFunction> interpolator;
        interpolator.reset(InterpolateImageFunction::New(interpolation));
        
        imagetransformation.Input(&mask);
        imagetransformation.Transformation(rr);
        
        RealImage m = *image;
        imagetransformation.Output(&m);
        //target contains zeros and ones image, need padding -1
        imagetransformation.TargetPaddingValue(-1);
        //need to fill voxels in target where there is no info from source with zeroes
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
            //calculate scaling factor
            factor = _stack_factor[_stack_index[inputIndex]];//_average_value;
            
            // read the pointer to current slice
            p = _slices[inputIndex]->GetPointerToVoxels();
            for(i=0; i<_slices[inputIndex]->GetNumberOfVoxels(); i++) {
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
            // alias for the current slice
            RealImage slice = *_slices[inputIndex];
            
            //alias for the current weight image
            RealImage w = *_weights[inputIndex];
            
            // alias for the current simulated slice
            RealImage sim = *_simulated_slices[inputIndex];
            
            double sw = _slice_weight[inputIndex];
            
            if (_stack_index[inputIndex] == _excluded_stack)
                sw = 1;
            
            for (int i = 0; i < slice.GetX(); i++)
                for (int j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) > -1) {
                        //scale - intensity matching
                        if ( _simulated_weights[inputIndex]->GetAsDouble(i,j,0) > 0.99 ) {
                            scalenum += w(i, j, 0) * sw * slice(i, j, 0) * sim(i, j, 0);
                            scaleden += w(i, j, 0) * sw * sim(i, j, 0) * sim(i, j, 0);
                        }
                    }
        } //end of loop for a slice inputIndex
        
        //calculate scale for the volume
        double scale = scalenum / scaleden;
        
        if(_debug)
            cout<<"- scale = "<<scale;
        
        RealPixel *ptr = _reconstructed.GetPointerToVoxels();
        for(int i=0; i<_reconstructed.GetNumberOfVoxels(); i++) {
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
                //Calculate simulated slice
//                reconstructor->_simulated_slices[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
//                reconstructor->_simulated_slices[inputIndex] = 0;
                
//                reconstructor->_simulated_weights[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
//                reconstructor->_simulated_weights[inputIndex] = 0;
                
//                reconstructor->_simulated_inside[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
//                reconstructor->_simulated_inside[inputIndex] = 0;
                
                reconstructor->_slice_inside[inputIndex] = false;

                
//                RealImage sim_slice = reconstructor->_simulated_slices[inputIndex];
//                RealImage sim_weight = reconstructor->_simulated_weights[inputIndex];
//                RealImage sim_inside = reconstructor->_simulated_inside[inputIndex];
                
                POINT3D p;
                for ( int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++ )
                    for ( int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++ ) {
                    
                        reconstructor->_simulated_slices[inputIndex]->PutAsDouble(i, j, 0, 0);
                        reconstructor->_simulated_weights[inputIndex]->PutAsDouble(i, j, 0, 0);
                        reconstructor->_simulated_inside[inputIndex]->PutAsDouble(i, j, 0, 0);

                        if ( reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) > -0.01 ) {
                            double weight = 0;
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            
                            double ss_sum = reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i, j, 0);
                            
                            for ( int k = 0; k < n; k++ ) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                ss_sum = ss_sum + p.value * reconstructor->_reconstructed(p.x, p.y, p.z);
                                weight += p.value;
                                if ( reconstructor->_mask(p.x, p.y, p.z) > 0.1 ) {
                                    reconstructor->_simulated_inside[inputIndex]->PutAsDouble(i, j, 0, 1);
                                    reconstructor->_slice_inside[inputIndex] = true;
                                }
                            }
                            
                            
                            
                            if( weight > 0 ) {
//                                double ss = reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i,j,0) / weight;
                                
                                double ss = ss_sum / weight;
                                
                                reconstructor->_simulated_slices[inputIndex]->PutAsDouble(i,j,0, ss);
                                reconstructor->_simulated_weights[inputIndex]->PutAsDouble(i,j,0, weight);
                            }
                        }
                
                    }
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
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
        z=-1; //this is the z coordinate of the stack
        current_stack=-1; //we need to know when to start a new stack
        
        
        double slice_threshold = 0.6;
        
        if (simulate_excluded)
            slice_threshold = -10;
        
        if (_current_iteration < 2)
            slice_threshold = -10;

        
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            
            if (_stack_index[inputIndex] == _excluded_stack)
                _slice_weight[inputIndex] = 1;
            
            
            // read the current slice
            RealImage slice = *_slices[inputIndex];
            
            //Calculate simulated slice
            sim.Initialize( slice.GetImageAttributes() );
            sim = 0;
            
            //do not simulate excluded slice
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
//                            else
//                                sim(i,j,0)=0;
                            
                            
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
            //end of loop for a slice inputIndex
        }
        
        
        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

            if (_stack_index[inputIndex] == _excluded_stack)
                _slice_weight[inputIndex] = 0;
        }
         
         
        
        
    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::MatchStackIntensities( Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, double averageValue, bool together )
    {
        //Calculate the averages of intensities for all stacks
        double sum, num;
        char buffer[256];
        unsigned int ind;
        int i, j, k;
        double x, y, z;
        Array<double> stack_average;
        
        //remember the set average value
        _average_value = averageValue;
        
        //averages need to be calculated only in ROI
        for (ind = 0; ind < stacks.size(); ind++) {
            sum = 0;
            num = 0;
            for (i = 0; i < stacks[ind].GetX(); i++)
                for (j = 0; j < stacks[ind].GetY(); j++)
                    for (k = 0; k < stacks[ind].GetZ(); k++) {
                        //image coordinates of the stack voxel
                        x = i;
                        y = j;
                        z = k;
                        //change to world coordinates
                        stacks[ind].ImageToWorld(x, y, z);
                        //transform to template (and also _mask) space
                        stack_transformations[ind].Transform(x, y, z);
                        //change to mask image coordinates - mask is aligned with template
                        _mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);
                        //if the voxel is inside mask ROI include it
                        // if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0)
                        //      && (z < _mask.GetZ()))
                        //      {
                        //if (_mask(x, y, z) == 1)
                        if ( stacks[ind](i, j, k) > 0 ) {
                            sum += stacks[ind](i, j, k);
                            num++;
                        }
                        //}
                    }
            //calculate average for the stack
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
        
        //Rescale stacks
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
            
            ptr = stacks[ind].GetPointerToVoxels();
            for (i = 0; i < stacks[ind].GetNumberOfVoxels(); i++) {
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
        //Calculate the averages of intensities for all stacks
        double sum, num;
        char buffer[256];
        unsigned int ind;
        int i, j, k;
        double x, y, z;
        Array<double> stack_average;
        RealImage m;
        
        //remember the set average value
        _average_value = averageValue;
        
        //averages need to be calculated only in ROI
        for (ind = 0; ind < stacks.size(); ind++) {
            m = *stacks[ind];
            m = 0;      // ????
            sum = 0;
            num = 0;
            for (i = 0; i < stacks[ind]->GetX(); i++)
                for (j = 0; j < stacks[ind]->GetY(); j++)
                    for (k = 0; k < stacks[ind]->GetZ(); k++) {
                        //image coordinates of the stack voxel
                        x = i;
                        y = j;
                        z = k;
                        //change to world coordinates
                        stacks[ind]->ImageToWorld(x, y, z);
                        //transform to template (and also _mask) space
                        stack_transformations[ind].Transform(x, y, z);
                        //change to mask image coordinates - mask is aligned with template
                        _mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);
                        //if the voxel is inside mask ROI include it
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
            //calculate average for the stack
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
        
        //Rescale stacks
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
            
            ptr = stacks[ind]->GetPointerToVoxels();
            for (i = 0; i < stacks[ind]->GetNumberOfVoxels(); i++) {
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
        
        
        
//        Array<RealImage> denoised_stacks = stacks;
//        NLMFiltering(denoised_stacks);
        
        //for each stack
        for (unsigned int i = start_stack_number; i < stop_stack_number; i++) {
            //image attributes contain image and voxel size
            
            ImageAttributes attr = stacks[i]->GetImageAttributes();
            
            
//            RealImage tmp_mask = mask;
//            TransformMaskP(stacks[i], tmp_mask);
            
            //attr._z is number of slices in the stack
            int selected_slice_number;
            
            
            if (!use_selected)
                selected_slice_number = attr._z;
            else
                selected_slice_number = selected_slices.size();
            

            for (int j = 0; j < selected_slice_number; j++) {
                
                if (selected_slices[j]>Z_dim)
                    break;

                //create slice by selecting the appropriate region of the stack
                RealImage slice;
                
//                RealImage denoised_slice;
                
                if (!use_selected) {
                    slice = stacks[i]->GetRegion(0, 0, j, attr._x, attr._y, j + 1);
//                    denoised_slice = denoised_stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                }
                else {
                    slice = stacks[i]->GetRegion(0, 0, selected_slices[j], stacks[i]->GetX(), stacks[i]->GetY(), selected_slices[j] + 1);
//                    denoised_slice = denoised_stacks[i].GetRegion(0, 0, selected_slices[j], stacks[i].GetX(), stacks[i].GetY(), selected_slices[j] + 1);
                }

                if (package_index<(d_packages-1) && j!=0)
                    package_index++;
                else
                    package_index = 0; 

                
                RealPixel smin, smax;
                slice.GetMinMax(&smin, &smax);
                
//                GaussianBlurring<RealPixel> gb(slice.GetXSize()*0.25);
//                
//                gb.Input(&slice);
//                gb.Output(&slice);
//                gb.Run();

                
                 _zero_slices.push_back(1);

                _min_val.push_back(smin);
                _max_val.push_back(smax);


                //set correct voxel size in the stack. Z size is equal to slice thickness.
                slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                
//                denoised_slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                
                
                //remember the slice
//                _slices.push_back(&slice);
                
                
                GreyImage grey_slice = slice;
//                _grey_slices.push_back(grey_slice); //denoised_slice);     // grey_slice);     //
                
                GreyImage *p_grey_slice = new GreyImage(grey_slice);
                RealImage *p_slice = new RealImage(slice);
                
                
                _grey_slices.push_back(p_grey_slice);
                _slices.push_back(p_slice);
                

//                _denoised_slices.push_back(denoised_slice);
//                _denoised_slices.push_back(slice);
                
                slice = 1;
                RealImage *p_simulated_slice = new RealImage(slice);
                _simulated_slices.push_back(p_simulated_slice);
                
                RealImage *p_simulated_weight = new RealImage(slice);
                _simulated_weights.push_back(p_simulated_weight);
                
                RealImage *p_simulated_inside = new RealImage(slice);
                _simulated_inside.push_back(p_simulated_inside);
                
                //remeber stack index for this slice
                _stack_index.push_back(i);
                
                //initialize slice transformation with the stack transformation
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
                
                
                ImageAttributes slice_attr = slice.GetImageAttributes();
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
                
//                RealImage *p_2dssim = new RealImage(slice);
//                _slice_2dssim.push_back(p_2dssim);

                
                _excluded_points.push_back(0);

                _structural_slice_weight.push_back(1);
                _slice_ssim.push_back(1);

                if (d_packages==1)
                    _package_index.push_back(i);
                else {

                    _package_index.push_back(i*d_packages + package_index);
                }
                
                
//                if (_mffd_mode) {
                
                    MultiLevelFreeFormTransformation *mffd = new MultiLevelFreeFormTransformation();
                    _mffd_transformations.push_back(mffd);
                    
//                }

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
        
//        if (_debug)
//            template_stack->Write("template-rigid.nii.gz");

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
        
//        stacks.clear();

    }

    //-------------------------------------------------------------------------------
    

    void ReconstructionFFD::RigidPackageRegistration( Array<GreyImage*> stacks, GreyImage* template_stack, int nPackages, Array<RigidTransformation>& stack_transformations, bool init_reset )
    {

        char buffer[256];

//        if (_debug)
//            template_stack->Write("r-template.nii.gz");
    
        
        if (nPackages>1) {
            
            
            Array<GreyImage*> packages;
            Array<int> package_index;
            Array<RigidTransformation> package_transformations;
            

            for ( int stackIndex=0; stackIndex<stacks.size(); stackIndex++ ) {
                
                
                
                GreyImage org_stack = *stacks[stackIndex];
                
                packages.clear();
                package_index.clear();
                package_transformations.clear();
                
//                cout << stackIndex << endl;
//
//                org_stack.Write("../y.nii.gz");
                
                
                // .....................................................................
                
                ImageAttributes attr = org_stack.GetImageAttributes();
                
                for (int i=0; i<attr._z; i++)
                    package_index.push_back(0);
                
                int pkg_z = attr._z / nPackages;
                double pkg_dz = attr._dz*nPackages;
                
                double x, y, z, sx, sy, sz, ox, oy, oz;
                
                for (int l = 0; l < nPackages; l++) {
                    attr = org_stack.GetImageAttributes();
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
                    stack.PutOrigin(ox, oy, oz); //adjust to original value
                    stack.ImageToWorld(sx, sy, sz);
                    stack.PutOrigin(ox + (x - sx), oy + (y - sy), oz + (z - sz));
                    sx = 0;
                    sy = 0;
                    sz = 0;
                    stack.ImageToWorld(sx, sy, sz);
                    
                    
                    GreyImage *tmp_grey_stack = new GreyImage(stack);
                    packages.push_back(tmp_grey_stack);
                    
//                    packages.push_back(&stack);
                    package_transformations.push_back(stack_transformations[stackIndex]);
                    
                    double tx, ty, tz, rx, ry, rz;
                    
                    tx = stack_transformations[stackIndex].GetTranslationX();
                    ty = stack_transformations[stackIndex].GetTranslationY();
                    tz = stack_transformations[stackIndex].GetTranslationZ();
                    rx = stack_transformations[stackIndex].GetRotationX();
                    ry = stack_transformations[stackIndex].GetRotationY();
                    rz = stack_transformations[stackIndex].GetRotationZ();
                    
                    
//                    if (_debug) {
//                        sprintf(buffer, "ri-%i-%i.dof", stackIndex, l);
//                        stack_transformations[stackIndex].Write(buffer);
//                    }
                    
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

//        stacks.clear();
        
    }


    //-------------------------------------------------------------------

    void ReconstructionFFD::FFDStackRegistration( Array<GreyImage*> stacks, GreyImage* template_stack, Array<RigidTransformation> stack_transformations )
    {

        char buffer[256];
        
        _global_mffd_transformations.clear();


        GaussianBlurring<GreyPixel> gb(1.2);
        
        ResamplingWithPadding<GreyPixel> resampling(1.8, 1.8, 1.8, -1); // 1.8
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

 
//        Insert(params, "Background value for image 1", -1);
//        Insert(params, "Background value for image 2", -1);


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
        
        Insert(params, "Control point spacing in X", 10);        // 10
        Insert(params, "Control point spacing in Y", 10);        // 10
        Insert(params, "Control point spacing in Z", 10);        // 10
        
        
        
        for (int i=0; i<resampled_stacks.size(); i++) {
            
            
//            GreyImage *target = resampled_stacks[i];

            GenericRegistrationFilter *registration = new GenericRegistrationFilter();
            registration->Parameter(params);


            registration->Input( resampled_stacks[i], &source);     // target
            Transformation *dofout = nullptr;
            registration->Output(&dofout);

            registration->InitialGuess(&(stack_transformations[i]));


            registration->GuessParameter();
            registration->Run();

            
            MultiLevelFreeFormTransformation *mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*> (dofout);
            
            

            //_global_mffd_transformations.push_back(mffd_dofout);


            // .............................................................

            sprintf(buffer, "ms-%i.dof", i);
            mffd_dofout->Write(buffer);

            // .............................................................

//            RealImage d_stack = target; //source;
//            d_stack = 0;
//
//            ImageAttributes attr = d_stack.GetImageAttributes();
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
            
    
            
//            delete mffd_dofout;
            delete registration;


        }

    }
    
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::GenerateGlobalDisplacement(Array<RealImage> stacks)
    {
        
        char buffer[256];
     
        for (int i=0; i<stacks.size(); i++) {
     
            // sprintf(buffer, "ms-%i.dof", i);  //_stack_index[inputIndex]);
            // Transformation *tt = Transformation::New(buffer);
            MultiLevelFreeFormTransformation *mffd = _global_mffd_transformations[i];
            //dynamic_cast<MultiLevelFreeFormTransformation*> (tt);
        

     
            RealImage d_stack = stacks[i];
            d_stack = 0;
            
            ImageAttributes attr = d_stack.GetImageAttributes();
            attr._t = 3;
            
            RealImage d_xyz(attr);
            
            mffd->InverseDisplacement(d_xyz);
            
            
            for (int j=0; j<3; j++) {
                
                RealImage tmp = d_xyz.GetRegion(0,0,0,j,d_xyz.GetX(),d_xyz.GetY(),d_xyz.GetZ(),(j+1));
                d_stack += tmp*tmp;
            }
            
            
            sprintf(buffer, "GD-%i.csv", i);
            ofstream GD_csv_file;
            GD_csv_file.open(buffer);
            for (int z=0; z<d_stack.GetZ(); z++)
                for (int y=0; y<d_stack.GetY(); y++)
                    for (int x=0; x<d_stack.GetX(); x++)
                        GD_csv_file << d_stack(x,y,z) << endl;
            
            GD_csv_file.close();
            
            
            // sprintf(buffer,"ssr-displacement-%i.nii.gz", i);
            // d_stack.Write(buffer);
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
                        imagetransformation->Interpolator(interpolator.get());  // &nn);
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
//                        reconstructor->_slice_masks[inputIndex] = output_volume;
                        
                    }
                
                }
                else {
                    
//                    reconstructor->_slice_masks[inputIndex] = 1;
                    for (int x=0; x<reconstructor->_slice_masks[inputIndex]->GetX(); x++) {
                        for (int y=0; y<reconstructor->_slice_masks[inputIndex]->GetY(); y++) {
                            
                            reconstructor->_slice_masks[inputIndex]->PutAsDouble(x,y,0,1);
                        }
                    }
                    
                    Erode<GreyPixel>(reconstructor->_slice_masks[inputIndex], 5, i_connectivity);
                    
                }

                
            } // end of inputIndex
            
        }
        
        
//        ~ParallelCreateSliceMasks() {
//
//
//        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
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

        // ConnectivityType i_connectivity = CONNECTIVITY_26;
        // Dilate<RealPixel>(&main_mask, 2, i_connectivity); 


        GaussianBlurring<RealPixel> gb(_slices[0]->GetXSize()*1.5);

        MultiLevelFreeFormTransformation *mffd_init;

        for ( int inputIndex = 0; inputIndex < _slices.size(); inputIndex++ ) {
            
            if (_reg_slice_weight[inputIndex] > 0) {

                if (iter < 0) {
                    if (current_stack_index == _stack_index[inputIndex]) {
                        init = true;
                    }

                    if ( init && _current_iteration == 0 && current_stack_index == _stack_index[inputIndex] && _global_init_ffd) {
                        
                        sprintf(buffer, "ms-%i.dof", _package_index[inputIndex]);  //_stack_index[inputIndex]);
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
                imagetransformation->Interpolator(interpolator.get());  // &nn);
                imagetransformation->TwoD(twod);
                imagetransformation->Invert(dofin_invert);
                imagetransformation->Run();

                delete imagetransformation;

                // Erode<RealPixel>(&output_volume, 8, i_connectivity);
                // Dilate<RealPixel>(&output_volume, 8, i_connectivity);
                
                gb.Input(&output_volume);
                gb.Output(&output_volume);
                gb.Run();

                for (int x=0; x<output_volume.GetX(); x++) {
                    for (int y=0; y<output_volume.GetY(); y++) {
                        
                        if (output_volume(x,y,0) > 0) {
//                            output_volume(x,y,0) = 1;
                            _slice_masks[inputIndex]->PutAsDouble(x,y,0,1);
                        }
                        else {
                            _slice_masks[inputIndex]->PutAsDouble(x,y,0,0);
                        }
                            

                    }
                }

//                _slice_masks[inputIndex] = &output_volume;
                
            }

        }



        /*
        
        ConnectivityType i_connectivity = CONNECTIVITY_26;
        

        if (!_masked || iter > 0) {
        
            char buffer[256];
            bool init = false;
            int current_stack_index = 0;

            // ConnectivityType i_connectivity = CONNECTIVITY_26;
            // Dilate<RealPixel>(&main_mask, 2, i_connectivity);


            GaussianBlurring<GreyPixel> gb(_slices[0].GetXSize()*1.5);

            MultiLevelFreeFormTransformation *mffd_init;
            
            

            for ( int inputIndex = 0; inputIndex < _slices.size(); inputIndex++ ) {
                
                
                if (_reg_slice_weight[inputIndex] > 0) {
                    
                    if (iter < 0) {
                        
                        mffd_init = new MultiLevelFreeFormTransformation(_package_transformations[inputIndex]);

                    }


                    InterpolationMode interpolation = Interpolation_NN;
                    UniquePtr<InterpolateImageFunction> interpolator;
                    interpolator.reset(InterpolateImageFunction::New(interpolation));

                    GreyImage input_volume = main_mask;
                    
                    
                    Dilate<GreyPixel>(&input_volume, 5, i_connectivity);
                    
                    GreyImage output_volume = _grey_slices[inputIndex];

                    double source_padding = 0;
                    double target_padding = -inf;
                    bool dofin_invert = false;
                    bool twod = false;
                    
                    if (_masked) {
                        target_padding = 0;
                    }

                    ImageTransformation *imagetransformation = new ImageTransformation;

                    imagetransformation->Input(&input_volume);
                    
                    if (iter < 0)
                        imagetransformation->Transformation(mffd_init);  //&(*mffd_init));
                    else
                        imagetransformation->Transformation(_mffd_transformations[inputIndex]); //&(_mffd_transformations[inputIndex]));
                    
                    
                    imagetransformation->Output(&output_volume);
                    imagetransformation->TargetPaddingValue(target_padding);
                    imagetransformation->SourcePaddingValue(source_padding);
                    imagetransformation->Interpolator(interpolator.get());  // &nn);
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
                                output_volume(x,y,0) = 1;
                                //if (_masked)
                                //    _grey_slices[inputIndex](x,y,0) = _slices[inputIndex](x,y,0);// _denoised_slices[inputIndex](x,y,0);
                            }
                            else {
                                output_volume(x,y,0) = 0;
                                //if (_masked)
                                //    _grey_slices[inputIndex](x,y,0) = -1;
                            }
                            
                            
                            
                        }
                    }
                    _slice_masks[inputIndex] = output_volume;
                    
//                    _grey_slices[inputIndex] = _slices[inputIndex] * _slice_masks[inputIndex];
                    
                }
                


            }
        }
        else {

            for ( int inputIndex = 0; inputIndex < _slices.size(); inputIndex++ ) {

                _slice_masks[inputIndex] = _slices[inputIndex];
  
                for (int x=0; x<_slice_masks[inputIndex].GetX(); x++) {
                    for (int y=0; y<_slice_masks[inputIndex].GetY(); y++) {

                        if (_slices[inputIndex](x,y,0)>-1)
                            _slice_masks[inputIndex](x,y,0) = 1;
                        else
                            _slice_masks[inputIndex](x,y,0) = 0;
                    }
                }
                
                
                Erode<GreyPixel>(&_slice_masks[inputIndex], 5, i_connectivity);
                
            }

        }
        
    */

    }
    
    
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::SliceDisplacement( Array<RealImage> stacks, int iter )
    {
        
        /*
        
        Array<RealImage> slice_displacements = _slices;
        
        for ( int inputIndex = 0; inputIndex < _slices.size(); inputIndex++ ) {
            
            for ( int x=0; x<_slices[inputIndex].GetX(); x++ ) {
                for ( int y=0; y<_slices[inputIndex].GetY(); y++ ) {
                    
                    double ix, iy, iz;
                    ix = x;
                    iy = y;
                    iz = 0;
                    
                    _slices[inputIndex].ImageToWorld(ix, iy, iz);
//                    _mffd_transformations[inputIndex].Displacement(ix, iy, iz);
                    _mffd_transformations[inputIndex]->Displacement(ix, iy, iz);
                    
                    double displ = sqrt(ix*ix + iy*iy + iz*iz);
                    
                    if (displ > 500)
                        displ = -1;
                    
                    slice_displacements[inputIndex](x,y,0) = displ;
                    
                }
            }
        }
        
        
        char buffer[256];
        
        RealImage stack;
        int z;
        
        for (int i = 0; i < stacks.size(); i++) {
            
            stack = stacks[i];
            stack = 0;
            z = 0;
            
            for (int j = 0; j < _slices.size(); j++) {
                
                if (_stack_index[j] == i) {
                    
                    for (int x = 0; x < stack.GetX(); x++) {
                        for (int y = 0; y < stack.GetY(); y++) {
                            
                            stack(x,y,z) = slice_displacements[j](x,y,0);
                        }
                    }
                    
                    z++;
                }
                
            }
            
            sprintf(buffer, "displacement-%i-%i.nii.gz", iter, i);
            stack.Write(buffer);
            
            Array<RealImage> packages;
            SplitImage(stack, 4, packages);
            for (int j=0; j<packages.size(); j++) {
                sprintf(buffer,"d-%i-%i-%i.nii.gz", iter, i, j);
                packages[j].Write(buffer);
            }
            
        }
        
        */
        
    }
    

    //-------------------------------------------------------------------

    /*
    void ReconstructionFFD::SlowSliceToVolumeRegistrationFFD(int iter, int round)
    {
        
        _grey_reconstructed = _reconstructed;

        char buffer[256];

        _current_iteration = iter;

        bool init = false;
        int current_stack_index = 0;

        
        if (round == 2) {
            
            GaussianBlurring<GreyPixel> gb4(_grey_reconstructed.GetXSize()*1);
            
            gb4.Input(&_grey_reconstructed);
            gb4.Output(&_grey_reconstructed);
            gb4.Run();
            
        }
        
        
        MultiLevelFreeFormTransformation *mffd_init;
        
        for ( int inputIndex = 0; inputIndex < _slices.size(); inputIndex++ ) {
            
            if (current_stack_index == _stack_index[inputIndex]) {
                init = true;
            }

            if ( init && _current_iteration == 0 && current_stack_index == _stack_index[inputIndex] && _global_init_ffd) {

                // sprintf(buffer, "ms-%i.dof", _package_index[inputIndex]);
                // Transformation *tt = Transformation::New(buffer);
                mffd_init = _global_mffd_transformations[inputIndex];
                //dynamic_cast<MultiLevelFreeFormTransformation*> (tt);
                current_stack_index++;

                init = false;
            }

            GreyPixel smin, smax;

            _grey_slices[inputIndex]->GetMinMax(&smin, &smax);

            
            if (smax > 1 && (smax - smin) > 1) {

                ParameterList params;
                Insert(params, "Transformation model", "FFD");        // SVFFD

                //                        Insert(params, "Optimisation method", "GradientDescent");
                //                        Insert(params, "Image (dis-)similarity measure", "NMI");
                //                        Insert(params, "No. of bins", 128);
                //                        Insert(params, "Image (dis-)similarity measure", "NCC");
                //                        string type = "sigma";
                //                        string units = "mm";
                //                        double width = 5;
                //                        Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);
                //                        Insert(params, "Background value", -1);
                
                if (_masked) {
                    
                    Insert(params, "Background value for image 1", -1);
                    Insert(params, "Background value for image 2", 0);
                    
                }
                

                if (_reg_slice_weight[inputIndex] > 0) {

                    int cp_spacing;
                    if (_current_iteration<=3)
                        cp_spacing = _cp_spacing[_current_iteration];
                    else
                        cp_spacing = 5;

                    if (round == 2)
                        cp_spacing -= 2;

                    Insert(params, "Control point spacing in X", cp_spacing);        // 10
                    Insert(params, "Control point spacing in Y", cp_spacing);        // 10
                    Insert(params, "Control point spacing in Z", cp_spacing);        // 10
                }

                //                        Insert(params, "Bending energy weight", 0.01);
                GenericRegistrationFilter *registration = new GenericRegistrationFilter();
                registration->Parameter(params);

                registration->Input(_grey_slices[inputIndex], &_grey_reconstructed);

                Transformation *dofout = nullptr;
                registration->Output(&dofout);

                
                if (_current_iteration == 0 && _global_init_ffd) {

                    registration->InitialGuess(&(*mffd_init));
                    
                }
                else {

                    if (_reg_slice_weight[inputIndex] > 0) {

                        registration->InitialGuess(_mffd_transformations[inputIndex]); //&(_mffd_transformations[inputIndex]));
                    }
                    else {
                        MultiLevelFreeFormTransformation *r_init = new MultiLevelFreeFormTransformation(_package_transformations[inputIndex]);
                        registration->InitialGuess(&(*r_init));
                        
                    }

                }
                

                
                registration->GuessParameter();
                registration->Run();

                MultiLevelFreeFormTransformation *mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*> (dofout);
                _mffd_transformations[inputIndex] = mffd_dofout; // *mffd_dofout;

                delete registration;
            } 
        }


        if (_current_iteration>0 || _structural_exclusion) {
            cout << "EvaluateRegistration" << endl;
            EvaluateRegistration();

            
        }
        
        for ( int inputIndex=0; inputIndex<_slices.size(); inputIndex++ ) {
            for (int x=0; x<_slice_2dncc[inputIndex]->GetX(); x++)
                for (int y=0; y<_slice_2dncc[inputIndex]->GetY(); y++) {
                    _slice_2dncc[inputIndex]->PutAsDouble(x,y,0,1);
                }
        }
        
//        for ( int inputIndex=0; inputIndex<_slices.size(); inputIndex++ ) {
//            (*_slice_2dncc[inputIndex]) = 1;
//        }

        _global_init_ffd = false;

    }

     
     */
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::SimulateMotion( RealImage input_volume, Array<RealImage*> stacks, int iter )
    {
        
        /*
        
        char buffer[256];
        
        Array<RealImage> new_stacks;

        for (int i = 0; i < stacks.size(); i++) {
            
            RealImage input_stack, output_stack;
            
            input_stack = stacks[i];
            output_stack = input_stack;
            output_stack = 0;
        
            double wx, wy, wz;
            int ix, iy, iz;
            
            for (int x = 0; x < input_stack.GetX(); x++) {
                for (int y = 0; y < input_stack.GetY(); y++) {
                    for (int z = 0; z < input_stack.GetZ(); z++) {
                        
                        wx = x;
                        wy = y;
                        wz = z;
                        
                        input_stack.ImageToWorld(wx, wy, wz);
                        input_volume.WorldToImage(wx, wy, wz);
                        
                        ix = round(wx);
                        iy = round(wy);
                        iz = round(wz);
                        
                        if (ix>-1 && iy>-1 && iz>-1 && ix<input_volume.GetX() && iy<input_volume.GetY() && iz<input_volume.GetZ()) {
                            output_stack(x,y,z) = input_volume(ix, iy, iz);
                        }
                    }
                }
            }
            
            new_stacks.push_back(output_stack);
            
            sprintf(buffer,"new-original-%i-%i.nii.gz", i, iter);
            output_stack.Write(buffer);
        }
        
        
        Array<RealImage> corrupted_stacks;
        
        double source_padding = 0;
        double target_padding = -inf;
        bool dofin_invert = false;
        bool twod = false;
        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        
        
        for (int i = 0; i < new_stacks.size(); i++) {

            RealImage output_stack = new_stacks[i];
            output_stack = 0;
            
            for (int j=0; j<_transformations.size(); j++) {
                
                if (_stack_index[j] == i) {
                    
                    cout << i << " : " << j << endl;
                    
                    RealImage transformed_slice = _slices[j];
                    RealImage input = input_volume;
                
                    ImageTransformation *imagetransformation = new ImageTransformation;
                    
                    imagetransformation->Input(&input);
                    imagetransformation->Transformation(_mffd_transformations[j]);  //&_mffd_transformations[j]);
                    imagetransformation->Output(&transformed_slice);
                    imagetransformation->TargetPaddingValue(target_padding);
                    imagetransformation->SourcePaddingValue(source_padding);
                    imagetransformation->Interpolator(&interpolator);
                    imagetransformation->TwoD(twod);
                    imagetransformation->Invert(dofin_invert);
                    imagetransformation->Run();
                    
                    
                    int ix, iy, iz;
                    double wx, wy, wz;
                    
                    double z = _stack_locs[j];
                    
                    for (int x=0; x<output_stack.GetX(); x++) {
                        for (int y=0; y<output_stack.GetY(); y++) {

//                            wx = x;
//                            wy = y;
//                            wz = z;
                            
                            output_stack(x, y, z) = transformed_slice(x, y, 0);
                            
//
//
//                            output_stack.ImageToWorld(wx, wy, wz);
//                            transformed_volume.WorldToImage(wx, wy, wz);
//
//                            ix = round(wx);
//                            iy = round(wy);
//                            iz = round(wz);
//
//                            if ( ix>-1 && iy>-1 && iz>-1 && ix<transformed_volume.GetX() && iy<transformed_volume.GetY() && iz<transformed_volume.GetZ() ) {
//
//                                output_stack(x, y, z) = transformed_volume(ix, iy, iz);
//                            }
                            
                        }
                    }
                
                    
                    
                }
                
            }
            
            corrupted_stacks.push_back(output_stack);
            
            sprintf(buffer,"new-corrupted-%i-%i.nii.gz", i, iter);
            output_stack.Write(buffer);
            
        }
        
        
        */
        
        
    }



    //-------------------------------------------------------------------

    /*
    
    class ParallelSliceToVolumeRegistrationFFD {
        
    public:

        ReconstructionFFD *reconstructor;
        GreyImage grey_reconstructed;
        Array<GreyImage> grey_slices;
//        Array<MultiLevelFreeFormTransformation*> mffd_transformations;
        Array<RigidTransformation> package_transformations;
        Array<int> stack_index;
        Array<double> reg_slice_weight;
        int current_iteration;
        int current_round;
        int cp_spacing;
        bool masked_flag;
        
        ParallelSliceToVolumeRegistrationFFD( ReconstructionFFD *in_reconstructor,
                                             GreyImage in_grey_reconstructed,
                                             Array<GreyImage> in_grey_slices,
//                                             Array<MultiLevelFreeFormTransformation*> in_mffd_transformations,
                                             Array<RigidTransformation> in_package_transformations,
                                             Array<int> in_stack_index,
                                             Array<double> in_reg_slice_weight,
                                             int in_current_iteration,
                                             int in_current_round,
                                             int in_cp_spacing,
                                             bool in_masked_flag ) :
        
        reconstructor(in_reconstructor),
        grey_reconstructed(in_grey_reconstructed),
        grey_slices(in_grey_slices),
//        mffd_transformations(in_mffd_transformations),
        package_transformations(in_package_transformations),
        stack_index(in_stack_index),
        reg_slice_weight(in_reg_slice_weight),
        current_iteration(in_current_iteration),
        current_round(in_current_round),
        cp_spacing(in_cp_spacing),
        masked_flag(in_masked_flag) {
            

        }
        
        
        void operator() (const blocked_range<size_t> &r) const {
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {

                char buffer[256];

                GreyPixel smin, smax;
                grey_slices[inputIndex].GetMinMax(&smin, &smax);

                if (smax > 1 && (smax - smin) > 1) {
                    
                    ParameterList params;
                    Insert(params, "Transformation model", "FFD");

                    if (masked_flag) {
                        Insert(params, "Background value for image 1", -1);
                        Insert(params, "Background value for image 2", 0);
                        
                    }
                    
//                    if (reg_slice_weight[inputIndex] > 0) {
                        Insert(params, "Control point spacing in X", cp_spacing);
                        Insert(params, "Control point spacing in Y", cp_spacing);
                        Insert(params, "Control point spacing in Z", cp_spacing);
//                    }
                

                    GenericRegistrationFilter *registration = new GenericRegistrationFilter();
                    registration->Parameter(params);
                    
                    registration->Input(&grey_slices[inputIndex], &grey_reconstructed);
                    
                    Transformation *dofout = nullptr;
                    registration->Output(&dofout);

                    
                    if (reg_slice_weight[inputIndex] > 0) {
                        registration->InitialGuess(reconstructor->_mffd_transformations[inputIndex]);
                        //mffd_transformations[inputIndex]);       //&(mffd_transformations[inputIndex]));
                        
                    }
                    else {
                        MultiLevelFreeFormTransformation *r_init = new MultiLevelFreeFormTransformation(package_transformations[inputIndex]);
                        registration->InitialGuess(&(*r_init));
                    }
                    
                    registration->GuessParameter();
                    registration->Run();
                    
                    MultiLevelFreeFormTransformation *mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*> (dofout);
                    
                    reconstructor->_mffd_transformations[inputIndex] = mffd_dofout;

//                    cout << inputIndex << endl;
                    
                    delete registration;
//                    delete mffd_dofout;
//                    delete dofout;
                    
                } // end of min/max

            } // end of inputIndex
            
        }
        
        ~ParallelSliceToVolumeRegistrationFFD() {
        
            
//            reconstructor->_mffd_transformations.clear();
            //            delete reconstructor;
            grey_slices.clear();
            //            mffd_transformations.clear();
            package_transformations.clear();
            stack_index.clear();
            reg_slice_weight.clear();
            
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, grey_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    */

        /*/
    class ParallelSliceToVolumeRegistrationInitFFD {
        
    public:
        
        ReconstructionFFD *reconstructor;
        GreyImage grey_reconstructed;
        Array<GreyImage> grey_slices;
//        Array<MultiLevelFreeFormTransformation*> mffd_transformations;
        Array<MultiLevelFreeFormTransformation*> global_mffd_transformations;
        Array<RigidTransformation> package_transformations;
        Array<int> stack_index;
        Array<double> reg_slice_weight;
        int current_iteration;
        int current_round;
        int cp_spacing;
        bool masked_flag;
        int number_of_stacks;
        
        
        ParallelSliceToVolumeRegistrationInitFFD( ReconstructionFFD *in_reconstructor,
                                                 GreyImage in_grey_reconstructed,
                                                 Array<GreyImage> in_grey_slices,
//                                                 Array<MultiLevelFreeFormTransformation*> in_mffd_transformations,
                                                 Array<RigidTransformation> in_package_transformations,
                                                 Array<MultiLevelFreeFormTransformation*> in_global_mffd_transformations,
                                                 Array<int> in_stack_index,
                                                 Array<double> in_reg_slice_weight,
                                                 int in_current_iteration,
                                                 int in_current_round,
                                                 int in_cp_spacing,
                                                 bool in_masked_flag,
                                                 int in_number_of_stacks ) :
        reconstructor(in_reconstructor),
        grey_reconstructed(in_grey_reconstructed),
        grey_slices(in_grey_slices),
//        mffd_transformations(in_mffd_transformations),
        package_transformations(in_package_transformations),
        global_mffd_transformations(in_global_mffd_transformations),
        stack_index(in_stack_index),
        reg_slice_weight(in_reg_slice_weight),
        current_iteration(in_current_iteration),
        current_round(in_current_round),
        cp_spacing(in_cp_spacing),
        masked_flag(in_masked_flag),
        number_of_stacks(in_number_of_stacks) {
            

        }
        
        
        void operator() (const blocked_range<size_t> &r) const {
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
//                 char buffer[256];
                
//                 GreyPixel smin, smax;
//                 grey_slices[inputIndex].GetMinMax(&smin, &smax);
                


//                 if (smax > 1 && (smax - smin) > 1) {
                    
//                     ParameterList params;
//                     Insert(params, "Transformation model", "FFD");
                    
//                     if (masked_flag) {
//                         Insert(params, "Background value for image 1", -1); //-1);
//                         Insert(params, "Background value for image 2", 0);
                        
//                     }
                    
//                     //                    if (reg_slice_weight[inputIndex] > 0) {
//                     Insert(params, "Control point spacing in X", cp_spacing);
//                     Insert(params, "Control point spacing in Y", cp_spacing);
//                     Insert(params, "Control point spacing in Z", cp_spacing);
//                     //                    }
                    
                    
//                     GenericRegistrationFilter *registration = new GenericRegistrationFilter();
//                     registration->Parameter(params);
                    
//                     registration->Input(&grey_slices[inputIndex], &grey_reconstructed);
                    
//                     Transformation *dofout = nullptr;
//                     registration->Output(&dofout);
                    
//                     registration->InitialGuess(global_mffd_transformations[stack_index[inputIndex]]);
//                     //&(global_mffd_transformations[stack_index[inputIndex]]));
                    
//                     registration->GuessParameter();
//                     registration->Run();
                    
// //                    cout << inputIndex << endl;
                    
//                     MultiLevelFreeFormTransformation *mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*> (dofout);
                    
//                     reconstructor->_mffd_transformations[inputIndex] = mffd_dofout; //*tmp_dof; //*mffd_dofout;

//                     delete registration;
//                    delete mffd_dofout;
//                    delete dofout;
                    
                } // end of min/max
                
            } // end of inputIndex
            
        }
        
        
        ~ParallelSliceToVolumeRegistrationInitFFD() {
            
            
//            delete reconstructor;
            grey_slices.clear();
//            mffd_transformations.clear();
//            global_mffd_transformations.clear();
            package_transformations.clear();
            stack_index.clear();
            reg_slice_weight.clear();
            
             

        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, grey_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    */
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SVRStepStack( int stackIndex )
    {
        
        ParameterList params;
        Insert(params, "Transformation model", "FFD");
        
        if (_masked) {
            Insert(params, "Background value for image 1", -1); //-1);
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
        
        
        
        
        bool init_flag = false;
        
        for (int inputIndex = 0; inputIndex<_grey_slices.size(); inputIndex++) {
            
            if (_stack_index[inputIndex] == stackIndex) {
                
                if (_max_val[inputIndex] > 1 && (_max_val[inputIndex] - _min_val[inputIndex]) > 1) {
                    
                    GenericRegistrationFilter *registration = new GenericRegistrationFilter();
                    registration->Parameter(params);
                
//                    if (!init_flag) {
                        registration->Input(_grey_slices[inputIndex], _p_grey_reconstructed);
                        init_flag = true;
//                    }
//                    else {
//                        registration->Reset();
//                        registration->ChangeTarget(_grey_slices[inputIndex]);
//                    }
                
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
            
        }
        
        
        
    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::SlowSliceToVolumeRegistrationFFD( int iter, int round, int number_of_stacks )
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
        
        
        //        delete _p_grey_reconstructed;
        _p_grey_reconstructed = new GreyImage(_grey_reconstructed); //&_grey_reconstructed;

        
        Array<MultiLevelFreeFormTransformation*> global_mffd_transformations;
        LoadGlobalTransformations(number_of_stacks, global_mffd_transformations );
        _global_mffd_transformations = global_mffd_transformations;
        
 
        for (int stackIndex=0; stackIndex<number_of_stacks; stackIndex++) {
            SVRStepStack(stackIndex);
        }


        
        /*
         
         
         int Nthreads = _grey_slices.size();
         
         rec_argument *ThreadArgs;
         
         // Reserve room for handles of threads in ThreadList
         ThreadList = (pthread_t *)calloc(Nthreads, sizeof(pthread_t));
         ThreadArgs = (rec_argument*)calloc(Nthreads, sizeof(rec_argument));
         
         for (int i = 0; i < Nthreads; i++) {
         ThreadArgs[i].inputIndex = i;
         ThreadArgs[i].recon_p = this;
         
         }
         
         
         
         for (int i = 0; i < Nthreads; i++) {
         
         if (pthread_create(&ThreadList[i], NULL, ThreadFunc, &ThreadArgs[i])) {
         cout << "Threads cannot be created" << endl;
         exit(0);
         }
         }
         
         for (int i=0; i < Nthreads; i++) {
         
         pthread_join(ThreadList[i], NULL);
         }
         
         */

        
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


    void ReconstructionFFD::SVRStep( int inputIndex )
    {

        if (_max_val[inputIndex] > 1 && (_max_val[inputIndex] - _min_val[inputIndex]) > 1) {

            

            ParameterList params;
            Insert(params, "Transformation model", "FFD");
                    
            if (_masked) {
                Insert(params, "Background value for image 1", -1); //-1);
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
                    
//            registration->Input(&_grey_slices[inputIndex], &_grey_reconstructed);
            registration->Input(_grey_slices[inputIndex], _p_grey_reconstructed);
//            registration->Input(_slices[inputIndex], &_reconstructed);


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
            
            
//            delete _mffd_transformations[inputIndex];
            
            MultiLevelFreeFormTransformation *mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*> (dofout);
            _mffd_transformations[inputIndex] = mffd_dofout;

            
            
            registration->~GenericRegistrationFilter();
            
//            delete registration;

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

//                pthread_exit(0);
                
            } // end of inputIndex
            
        }

        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_grey_slices.size() ), *this );
            //init.terminate();
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
        


//        delete _p_grey_reconstructed;
        _p_grey_reconstructed = new GreyImage(_grey_reconstructed);
        
            
//            _global_mffd_transformations.clear();
            
            Array<MultiLevelFreeFormTransformation*> global_mffd_transformations;
            LoadGlobalTransformations(number_of_stacks, global_mffd_transformations );
            _global_mffd_transformations = global_mffd_transformations;

            
//        }

        
        
        
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
    //                cout << inputIndex << " - " << this_id << " - " << endl;

                    
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
                    
                    
                    if (reconstructor->_current_iteration == 0 && reconstructor->_current_round == 1) {
                        
                        str_dofin = "-dofin " + str_current_main_file_path + "/ms-" + to_string(reconstructor->_stack_index[inputIndex]) + ".dof ";
                    }
                    else {
                        if (reconstructor->_reg_slice_weight[inputIndex] > 0) {
                            
                            str_dofin = "-dofin " + str_current_exchange_file_path + "/transformation-" + to_string(inputIndex) + ".dof ";
                        }
                        else {
                            
                            str_dofin = "-dofin " + str_current_exchange_file_path + "/r-init-" + to_string(inputIndex) + ".dof ";
                            MultiLevelFreeFormTransformation *r_init = new MultiLevelFreeFormTransformation(reconstructor->_package_transformations[inputIndex]);
                            r_init->Write(str_dofin.c_str());
                        }
                    }
                    
                    
                    
                    string register_cmd = str_mirtk_path + "/register " + str_target + " " + str_source + " " + str_reg_model + " " + str_ds_setting + " " + str_bg1 + " " + str_bg2 + " " + str_dofin + " " + str_dofout + " " + str_log;
                    
                    
    //                cout << register_cmd << endl;
                    
                    
                    int tmp_log = system(register_cmd.c_str());
                    
    //                cout << inputIndex << " - " << this_id << " + " << endl;
                    
                }
                
            } // end of inputIndex
            
        }
        
        // execute
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
//            cout << "Registering slices from " << svr_range_start << " to " << svr_range_stop << " : " << endl;

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
            
//            delete mffd_init;
            
        }
        
    }

    RealImage ReconstructionFFD::StackIntersection( Array<RealImage*>& stacks, int templateNumber, RealImage template_mask )
    {
        
        char buffer[256];
        
//        RealImage original_stack = stacks[templateNumber];
        
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
        Dilate<RealPixel>(&i_mask, 5, i_connectivity); // 5
        
        
//        i_mask.Write("../i-mask.nii.gz");
        
//        CropImage(stacks[templateNumber], i_mask);
        

//        if(_debug)
            i_mask.Write("i-mask.nii.gz");
//
//
//        stacks[1]->Write("../v01.nii.gz");
        
        return i_mask;

        
        /*
        char buffer[256];
        
        // RealImage original_stack = stacks[templateNumber];
        
        RealImage i_mask = *(p_stacks[templateNumber]);
        i_mask = 1;
        
        
        // int ix, iy, iz, is;
        double sx, sy, sz;
        double wx, wy, wz;
        
        

        if (p_stacks.size() > 1) {
            
            
            Array<ImageAttributes> array_attr;
            
            for (int i=0; i<p_stacks.size(); i++) {
                
                ImageAttributes attr = p_stacks[i]->GetImageAttributes();
                array_attr.push_back(attr);
            }
            
            double sx, sy, sz;
            double wx, wy, wz;
            int qx, qy, qz;
            
            for (int ix = 0; ix < i_mask.GetX(); ix++) {
                for (int iy = 0; iy < i_mask.GetY(); iy++) {
                    for (int iz = 0; iz < i_mask.GetZ(); iz++) {
            
                        i_mask(ix,iy,iz) = 1;
                        
                        wx = ix;
                        wy = iy;
                        wz = iz;
                        
                        array_attr[templateNumber].LatticeToWorld(wx, wy, wz);
                        
                        for (int is = 0; is < p_stacks.size(); is++) {
                        
                            if (is != templateNumber) {
                                
                                sx = wx;
                                sy = wy;
                                sz = wz;
                                
                                array_attr[is].WorldToLattice(sx, sy, sz);
                                
                                qx = round(sx);
                                qy = round(sy);
                                qz = round(sz);
                                
                                if (array_attr[is].IsInside(qx, qy, qz)) {
                                
                                    i_mask(ix,iy,iz) = i_mask(ix,iy,iz) * 1;
                                }
                                else {
                                    i_mask(ix,iy,iz) = i_mask(ix,iy,iz) * 0;
                                }
                                
                            }
                            
                            
                        }
                    }
                }
            }
            
            
            ConnectivityType i_connectivity = CONNECTIVITY_26;
            Dilate<RealPixel>(&i_mask, 15, i_connectivity); // 5
            
            
            for (int i=0; i<p_stacks.size(); i++) {
                
                RealImage i_tmp = i_mask;
                TransformMaskP(p_stacks[i], i_tmp);
                
                CropImageP(p_stacks[i], &i_tmp);
                
//                if (_debug) {
                    sprintf(buffer,"cropped-%i.nii.gz",i);
                    p_stacks[i]->Write(buffer);
//                }
                
            }
            
            
        }
        

//        if(_debug)
            i_mask.Write("i-mask.nii.gz");

        return i_mask;
         
         
         */


    }

    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::EvaluateRegistration()
    {
        
        char buffer[256];
        
//        RealImage source = _reconstructed;
        
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
            
            
            RealImage output =  *_slices[inputIndex];  // _denoised_slices[inputIndex]; //
            output = 0;
            
            ImageTransformation *imagetransformation = new ImageTransformation;
            
            imagetransformation->Input(&_reconstructed); //source);
            imagetransformation->Transformation(_mffd_transformations[inputIndex]);   //&(_mffd_transformations[inputIndex]));
            imagetransformation->Output(&output);
            imagetransformation->TargetPaddingValue(target_padding);
            imagetransformation->SourcePaddingValue(source_padding);
            imagetransformation->Interpolator(&interpolator);
            imagetransformation->TwoD(twod);
            imagetransformation->Invert(dofin_invert);
            imagetransformation->Run();
            
            
            delete imagetransformation;
            
            RealImage target = *_slices[inputIndex]; // _denoised_slices[inputIndex];    //
            
            //            if (_current_iteration < 1) {
            //                if (!_masked) {
            //                    GaussianBlurring<RealPixel> gb(target.GetXSize()*0.6);
            //                    gb.Input(&target);
            //                    gb.Output(&target);
            //                    gb.Run();
            //                }
            //                else {
            //                    GaussianBlurringWithPadding<RealPixel> gb(target.GetXSize()*0.6,-1);
            //                    gb.Input(&target);
            //                    gb.Output(&target);
            //                    gb.Run();
            //                }
            //            }
            
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

    GreyImage StackIntersection( Array<GreyImage> stacks, int templateNumber )
    {




    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::GenerateStackDisplacement(RealImage stack, int stack_number)
    {
        
        char buffer[256];
        
        RealImage d_stack = stack;
        RealImage d_slice;
        
        d_stack = 1;
        
        int z = 0;
        for (int i=0; i<_slices.size(); i++) {
            
            if (_stack_index[i]==stack_number) {

                d_slice = GenerateSliceDisplacement(i);
                
                for (int y=0; y<d_stack.GetY(); y++) {
                    for (int x=0; x<d_stack.GetX(); x++) {
                        
                        if (d_slice(x,y,0)>800)
                            d_stack(x,y,z) = -1;
                        else
                            d_stack(x,y,z) = d_slice(x,y,0);
                    }
                }
                z++;
            }
        }
        
        
        sprintf(buffer, "LD-%i.csv", stack_number);
        ofstream GD_csv_file;
        GD_csv_file.open(buffer);
        for (int z=0; z<d_stack.GetZ(); z++)
            for (int y=0; y<d_stack.GetY(); y++)
                for (int x=0; x<d_stack.GetX(); x++)
                    GD_csv_file << d_stack(x,y,z) << endl;
        
        GD_csv_file.close();
        
        
        sprintf(buffer,"svr-displacement-%i-%i.nii.gz", _current_iteration, stack_number);
        d_stack.Write(buffer);
        
        
    }
    
    
    //-------------------------------------------------------------------
    
    RealImage ReconstructionFFD::GenerateSliceDisplacement(int i)
    {
        
        RealImage d_slice = *_slices[i];
        d_slice = 0;
        
        ImageAttributes attr = d_slice.GetImageAttributes();
        attr._t = 3;
        
        RealImage d_xyz(attr);

//        _mffd_transformations[i].InverseDisplacement(d_xyz);
        _mffd_transformations[i]->InverseDisplacement(d_xyz);
        
        for (i=0; i<3; i++) {
            
            RealImage tmp = d_xyz.GetRegion(0,0,0,i,d_xyz.GetX(),d_xyz.GetY(),d_xyz.GetZ(),(i+1));
            d_slice += tmp*tmp;
        }
        
        // if (d_slice>10000)
        //     d_slice = -1;

        return d_slice;
        
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


        slice_1_N = slice_1.GetNumberOfVoxels();
        slice_2_N = slice_2.GetNumberOfVoxels();

        slice_1_ptr = slice_1.GetPointerToVoxels();
        slice_2_ptr = slice_2.GetPointerToVoxels();

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
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::CStep2D()
    {

        char buffer[256];
        
        ofstream N_csv_file1;
        N_csv_file1.open("S-W.csv");
        
        for (int i=0; i<_slices.size(); i++) {
            N_csv_file1 << _slice_weight[i] << endl;
        }
        N_csv_file1.close();
        
        
        ParallelCCMap *p_cstep = new ParallelCCMap(this);
        (*p_cstep)();

        delete p_cstep;
        
        ofstream N_csv_file;
        N_csv_file.open("S-NCC.csv");
        
        for (int i=0; i<_slices.size(); i++) {
            N_csv_file << _structural_slice_weight[i] << endl;
        }
        N_csv_file.close();
        
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
        
        
        /*
        double C1 = 6.5025, C2 = 58.5225;
        double SSIM = 0, DSSIM = 0;
        
        
        double mu1=0, mu2=0, var1=0, var2=0, covar=0, num=0;
        double x_sq=0, y_sq=0, xy=0;
        
        int box = round(slice->GetX()/2);
        
        int y = box;
        int x = box;
        
        int slice_N = slice.GetNumberOfVoxels();
        int sim_slice_N = sim_slice.GetNumberOfVoxels();
        
        double *slice_ptr = slice.GetPointerToVoxels();
        double *sim_slice_ptr = sim_slice.GetPointerToVoxels();
        
        
        for (int j = 0; j < slice_N; j++) {
            
            if (slice_ptr[j]>0.01 && sim_slice_ptr[j]>0.01) {
                
                mu1 += slice_ptr[j];
                mu2 += sim_slice_ptr[j];
                num += 1;
                
                x_sq += pow(slice_ptr[j], 2.0);
                y_sq += pow(sim_slice_ptr[j], 2.0);
                xy += slice_ptr[j]*sim_slice_ptr[j];
            }

        }

//        for (int yy=0; yy<slice.GetY(); yy++) {
//            for (int xx=0; xx<slice.GetX(); xx++) {
//
//                if (slice(xx,yy,0)>0.01 && sim_slice(xx,yy,0)>0.01) {
//
//                    mu1 += slice(xx, yy, 0);
//                    mu2 += sim_slice(xx,yy,0);
//                    num += 1;
//
//                    x_sq += pow(slice(xx, yy, 0),2.0);
//                    y_sq += pow(sim_slice(xx, yy, 0),2.0);
//                    xy += slice(xx, yy, 0)*sim_slice(xx, yy, 0);
//                }
//
//            }
//        }
        
        mu1 = mu1/num;
        mu2 = mu2/num;
        var1 = (x_sq/num)-pow(mu1,2.0);
        var2 = (y_sq/num)-pow(mu2,2.0);
        covar = (xy/num)-mu1*mu2;
        
        double curSSIM = ((2*mu1*mu2+C1)*(2*covar+C2)) / ( (pow(mu1,2.)+pow(mu2,2.)+C1) * (var1+var2+C2) );

        
        return curSSIM;
         
         
         */
    }

    //-------------------------------------------------------------------


    double ReconstructionFFD::SliceCCMap2(int inputIndex, int box_size, double threshold)
    {

        RealImage slice_1 = *_slices[inputIndex];   //_denoised_slices[inputIndex];
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


                    // .................................................
                    // circular region 
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
                    // .................................................


                    CC_map(x,y,0) = LocalSSIM(region_1, region_2); //SliceCC(region_1, region_2);

//                    _slice_2dssim[inputIndex]->PutAsDouble(x,y,0,CC_map(x,y,0));


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
 
        
        _slice_2dncc[inputIndex] = new RealImage(CC_map); //&CC_map;

        if (num > 0)
            mean_ncc /= num;
        else 
            mean_ncc = 0;

               
        if (mean_ncc < 0)
            mean_ncc = 0;
                
        _structural_slice_weight[inputIndex] = global_cc; //mean_ncc;


        if (threshold > -0.1) {
            // _weights[inputIndex] *= _slice_2dncc[inputIndex];

            if (_structural_slice_weight[inputIndex]>_global_NCC_threshold) {   // && _slice_weight[inputIndex]>0.4
                // _slice_weight[inputIndex] = 1; //_slice_weight[inputIndex];
                _structural_slice_weight[inputIndex] = 1;

                if (_slice_weight[inputIndex] < 0.6) {
                    _structural_slice_weight[inputIndex] = _slice_weight[inputIndex]/10;
                }
            }
            else {
                // _slice_weight[inputIndex] = 0;    
                _structural_slice_weight[inputIndex] = 0; //_slice_weight[inputIndex]/10;
            }
        }

        return mean_ncc;
        
    }

    
       //-------------------------------------------------------------------
    
    
    void ReconstructionFFD::Save3DNCC( Array<RealImage*> stacks, int iteration)
    {
        
        
        
//        unsigned int inputIndex;
//        int i, j, k, n;
//        RealImage sim;
//        POINT3D p;
//        double weight;
//
//
//        int z, current_stack;
//        z=-1;//this is the z coordinate of the stack
//        current_stack=-1; //we need to know when to start a new stack
//
//        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
//
//
//            if (_stack_index[inputIndex]==current_stack)
//                z++;
//            else {
//                current_stack=_stack_index[inputIndex];
//                z=0;
//            }
//
//            for(i=0;i<_slice_masks[inputIndex].GetX();i++)
//                for(j=0;j<_slice_masks[inputIndex].GetY();j++) {
//                    stacks[_stack_index[inputIndex]](i,j,z) = _slice_masks[inputIndex](i,j,0); //_slice_2dncc[inputIndex](i,j,0);
//                }
//            //end of loop for a slice inputIndex
//        }
//
//        char buffer[256];
//
//        for (i=0; i<stacks.size(); i++) {
//
//            sprintf(buffer, "s-mask-%i-%i.nii.gz", iteration, i);
//            stacks[i].Write(buffer);
//
//        }

        
        
//        int z, current_stack;
//        z=-1;//this is the z coordinate of the stack
//        current_stack=-1; //we need to know when to start a new stack
//
//        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
//
//
//            if (_stack_index[inputIndex]==current_stack)
//                z++;
//            else {
//                current_stack=_stack_index[inputIndex];
//                z=0;
//            }
//
//            for(i=0;i<_slice_2dncc[inputIndex].GetX();i++)
//                for(j=0;j<_slice_2dncc[inputIndex].GetY();j++) {
//                    stacks[_stack_index[inputIndex]](i,j,z) = _slice_2dssim[inputIndex](i,j,0); //_slice_2dncc[inputIndex](i,j,0);
//                }
//            //end of loop for a slice inputIndex
//        }
//
//        char buffer[256];
//
//        for (i=0; i<stacks.size(); i++) {
//
//            sprintf(buffer, "lncc-%i-%i.nii.gz", iteration, i);
//            sprintf(buffer, "o-ssim-%i-%i.nii.gz", iteration, i);
//            stacks[i].Write(buffer);
//
//        }
//
//
//        z=-1;//this is the z coordinate of the stack
//        current_stack=-1; //we need to know when to start a new stack
//
//        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
//
//
//            if (_stack_index[inputIndex]==current_stack)
//            z++;
//            else {
//                current_stack=_stack_index[inputIndex];
//                z=0;
//            }
//
//            for(i=0;i<_slice_2dncc[inputIndex].GetX();i++) {
//                for(j=0;j<_slice_2dncc[inputIndex].GetY();j++) {
//
//                    if (_slices[inputIndex](i,j,0) > 0 && _simulated_slices[inputIndex](i,j,0) > 0)
//                        stacks[_stack_index[inputIndex]](i,j,z) = _slices[inputIndex](i,j,0)*exp(-(_bias[inputIndex])(i,j,0))*_scale[inputIndex] - _simulated_slices[inputIndex](i,j,0);
//                    else
//                        stacks[_stack_index[inputIndex]](i,j,z) = -1;
//
//                    if (stacks[_stack_index[inputIndex]](i,j,z) > 10000 || stacks[_stack_index[inputIndex]](i,j,z) < -10000)
//                        stacks[_stack_index[inputIndex]](i,j,z) = -1;
//                }
//            }
//            //end of loop for a slice inputIndex
//        }
//
//
//        for (i=0; i<stacks.size(); i++) {
//
//            sprintf(buffer, "ss-error-%i-%i.nii.gz", iteration, i);
//            stacks[i].Write(buffer);
//
//        }
//
//
//        z=-1;//this is the z coordinate of the stack
//        current_stack=-1; //we need to know when to start a new stack
//
//        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
//
//
//            if (_stack_index[inputIndex]==current_stack)
//            z++;
//            else {
//                current_stack=_stack_index[inputIndex];
//                z=0;
//            }
//
//            for(i=0;i<_slice_2dncc[inputIndex].GetX();i++) {
//                for(j=0;j<_slice_2dncc[inputIndex].GetY();j++) {
//
//                    stacks[_stack_index[inputIndex]](i,j,z) = _simulated_slices[inputIndex](i,j,0);
//                }
//            }
//            //end of loop for a slice inputIndex
//        }
//
//
//        for (i=0; i<stacks.size(); i++) {
//
//            sprintf(buffer, "simulated-%i-%i.nii.gz", iteration, i);
//            stacks[i].Write(buffer);
//
//        }
        
    }
    
    
    

    //-------------------------------------------------------------------
    
    /*
    void ReconstructionFFD::SaveWeights3D(Array<RealImage> stacks, int iter)
    {
        
        char buffer[256];
        
        RealImage stack, slice;
        int z;
    
        for (int i = 0; i < stacks.size(); i++) {
            
            stack = stacks[i];
            stack = 0;
            z = 0;
            
            for (int j = 0; j < _slices.size(); j++) {
                
                if (_stack_index[j] == i) {
            
                    for (int x = 0; x < stack.GetX(); x++) {
                        for (int y = 0; y < stack.GetY(); y++) {

                            stack(x,y,z) = _weights[j](x,y,0);
                            
                        }
                    }
                    
                    z++;
                }
                
            }
            
            sprintf(buffer, "weight-%i-%i.nii.gz", iter, i);
            stack.Write(buffer);
        }

    }
     */
    
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
           // Default arguments
           if (IsNaN(_t0)) _t0 = _image->ImageToTime(0);
           if (IsNaN(_t)) _t = _t0 + _image->GetTSize();
           // Initialize output image
           _jacobian->Initialize(attr, 1);
           
           // Use scaling and squaring in case of SV FFD
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
    
    /*
    
    void ReconstructionFFD::GenerateMotionModel(RealImage stack, int stack_number) {
        
        RealImage j_stack = stack;
        RealImage j_slice;
        
        
        char buffer[256];
        
        Array<double> j_min_array;
        Array<double> j_max_array;
        Array<double> j_avg_array;
        
        double j_min, j_max, j_avg;
        
        Array<int> negative_J_values;
        int negative_count;
        
        int z = 0;
        
        
        for (int i=0; i<_slices.size(); i++) {
            
            if (_stack_index[i]==stack_number) {
                
                negative_count = 0;
                j_slice = GenerateSliceMotionModel(i, j_min, j_max, j_avg);
                
                for (int y=0; y<j_stack.GetY(); y++) {
                    for (int x = 0; x < j_stack.GetX(); x++) {
                        
                        if (j_slice(x,y,0)< -10 || j_slice(x,y,0)>120)
                            j_slice(x,y,0) = -10;

                        j_stack(x,y,z) = j_slice(x,y,0);


                        if (j_stack(x,y,z) < 0)
                            negative_count++;
                        
                    }
                }
                
                j_min_array.push_back(j_min);
                j_max_array.push_back(j_max);
                j_avg_array.push_back(j_avg);
                
                negative_J_values.push_back(negative_count);
                
                z++;
                
            }
            
        }
        
        
        sprintf(buffer,"gj-motion-model-%i-%i.nii.gz", _current_iteration, stack_number);
        j_stack.Write(buffer);
        

        ofstream N_csv_file;
        sprintf(buffer,"negative-%i-%i.csv", _current_iteration, stack_number);
        N_csv_file.open(buffer);
        
        for (int i=0; i<negative_J_values.size(); i++) {
            N_csv_file << negative_J_values[i] << endl;
        }
        N_csv_file.close();
        
        
        ofstream MIN_csv_file;
        sprintf(buffer,"m-min-%i-%i.csv", _current_iteration, stack_number);
        MIN_csv_file.open(buffer);
        
        for (int i=0; i<j_min_array.size(); i++) {
            MIN_csv_file << j_min_array[i] << endl;
        }
        MIN_csv_file.close();
        
        
        ofstream MAX_csv_file;
        sprintf(buffer,"m-max-%i-%i.csv", _current_iteration, stack_number);
        MAX_csv_file.open(buffer);
        
        for (int i=0; i<j_max_array.size(); i++) {
            MAX_csv_file << j_max_array[i] << endl;
        }
        MAX_csv_file.close();
        
        
        ofstream AVG_csv_file;
        sprintf(buffer,"m-avg-%i-%i.csv", _current_iteration, stack_number);
        AVG_csv_file.open(buffer);
        
        for (int i=0; i<j_avg_array.size(); i++) {
            AVG_csv_file << j_avg_array[i] << endl;
        }
        AVG_csv_file.close();
        
    }
     
     */

    
    //-------------------------------------------------------------------
    
    /*
    
    RealImage ReconstructionFFD::GenerateSliceMotionModel(int i, double& j_min, double& j_max, double& j_avg) {
        
        int j=0;

        UniquePtr<GreyImage> target(new GreyImage(_slices[i]));
        
        JacobianImplFFD impl;
        bool asdisp = false;
        bool asvelo = false;
 
        impl._image = target.get();
        impl._jacobian = target.get();
        impl._dof = _mffd_transformations[i];   //&_mffd_transformations[i];
        
        const int nvox = target->NumberOfSpatialVoxels();
        
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
        
        int num = 0;
        double jac, min = inf, max = -inf, avg = 0.;
        for (int n = 0; n < nvox; ++n) {
            if (mask(n) != 0) {
                jac = output->GetAsDouble(n);
                if (jac < min) min = jac;
                if (jac > max) max = jac;
                avg += jac;
                num += 1;
            }
        }
        min /= 100.0, max /= 100.0, avg /= 100.0;
        if (num > 0) avg /= num;
        
        
        j_min = min;
        j_max = max;
        j_avg = avg;
        
        RealImage tmp_slice = *_slices[i];
        
        double tt;
        
        for (int x=0; x < tmp_slice.GetX(); x++) {
            for (int y = 0; y < tmp_slice.GetY(); y++) {
                for (int z = 0; z < tmp_slice.GetZ(); z++) {
                    
                    tt = output->GetAsDouble(x,y,0);
                    tmp_slice.PutAsDouble(x,y,0,tt);
                }
            }
        }
        
        return tmp_slice;
    }

    */
    
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
        
//        _slice_attr[inputIndex].LatticeToWorld(&x, &y, &z);
        
//        ImageAttributes slice_attr = reconstructor->_slice_attr[inputIndex];
//        ImageAttributes recon_attr = reconstructor->_reconstructed_attr;
        
        
//        if(!_mffd_mode)
//            _transformations[inputIndex].Transform(x, y, z);
//        else
            _mffd_transformations[inputIndex]->Transform(-1, 1, x, y, z);
        
        _reconstructed.WorldToImage(x, y, z);
        
//        _reconstructed_attr.WorldToLattice(x, y, z);
        
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
    
    
    
    // new
    class ParallelCoeffInitFFD {
    public:
        ReconstructionFFD *reconstructor;
        
        ParallelCoeffInitFFD(ReconstructionFFD *_reconstructor) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                bool slice_inside;
                
                //get resolution of the volume
                double vx, vy, vz;
                
                ImageAttributes slice_attr = reconstructor->_slice_attr[inputIndex];
                ImageAttributes recon_attr = reconstructor->_reconstructed_attr;
                
                vx = slice_attr._dx;
                vy = slice_attr._dy;
                vz = slice_attr._dz;
                
                //volume is always isotropic
                double res = vx;
                
                //prepare structures for storage
                POINT3D p;
                VOXELCOEFFS empty;
                SLICECOEFFS slicecoeffs(slice_attr._x, Array < VOXELCOEFFS > (slice_attr._y, empty));
                
                
                //to check whether the slice has an overlap with mask ROI
                slice_inside = false;
                
                //PSF will be calculated in slice space in higher resolution
                
                //get slice voxel size to define PSF
                double dx, dy, dz;
                
                dx = slice_attr._dx;
                dy = slice_attr._dy;
                dz = slice_attr._dz;
                
                
                //sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM = dz through-plane)
                double sigmax = 1.2 * dx / 2.3548;
                double sigmay = 1.2 * dy / 2.3548;
                double sigmaz = dz / 2.3548;
                
                //calculate discretized PSF
                
                //isotropic voxel size of PSF - derived from resolution of reconstructed volume
                double size = res / reconstructor->_quality_factor;
                
                //number of voxels in each direction
                //the ROI is 2*voxel dimension
                
                int xDim = round(2 * dx / size);
                int yDim = round(2 * dy / size);
                int zDim = round(2 * dz / size);
                
                
                int excluded = 0;

                //image corresponding to PSF
                ImageAttributes attr;
                attr._x = xDim;
                attr._y = yDim;
                attr._z = zDim;
                attr._dx = size;
                attr._dy = size;
                attr._dz = size;
                RealImage PSF(attr);
                
                //centre of PSF
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
                            //continuous PSF does not need to be normalized as discrete will be
                            PSF(i, j, k) = exp(-x * x / (2 * sigmax * sigmax) - y * y / (2 * sigmay * sigmay) - z * z / (2 * sigmaz * sigmaz));
                            sum += PSF(i, j, k);
                        }
                PSF /= sum;
                
                if (reconstructor->_debug)
                    if (inputIndex == 0)
                        PSF.Write("PSF.nii.gz");
                
                //prepare storage for PSF transformed and resampled to the space of reconstructed volume
                //maximum dim of rotated kernel - the next higher odd integer plus two to account for rounding error of tx,ty,tz.
                //Note conversion from PSF image coordinates to tPSF image coordinates *size/res
                int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2)) * 2 + 1 + 2;
                //prepare image attributes. Voxel dimension will be taken from the reconstructed volume
                attr._x = dim;
                attr._y = dim;
                attr._z = dim;
                attr._dx = res;
                attr._dy = res;
                attr._dz = res;
                //create matrix from transformed PSF
                RealImage tPSF(attr);
                //calculate centre of tPSF in image coordinates
                int centre = (dim - 1) / 2;
                
                //for each voxel in current slice calculate matrix coefficients
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
                        
                        // if (reconstructor->_slices[inputIndex](i, j, 0) != -1) {
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) > -1 && reconstructor->_slice_2dncc[inputIndex]->GetAsDouble(i,j,0) > 0) {  //
                            //calculate centrepoint of slice voxel in volume space (tx,ty,tz)

                            x = i;
                            y = j;
                            z = 0;
                            
                            reconstructor->SliceToReconstuced(inputIndex, x, y, z, tx, ty, tz, 0);
                            
                            
                            //Clear the transformed PSF
                            for (ii = 0; ii < dim; ii++)
                                for (jj = 0; jj < dim; jj++)
                                    for (kk = 0; kk < dim; kk++)
                                        tPSF(ii, jj, kk) = 0;
                            
                            //for each POINT3D of the PSF
                            for (ii = 0; ii < xDim; ii++)
                                for (jj = 0; jj < yDim; jj++)
                                    for (kk = 0; kk < zDim; kk++) {
                                        //Calculate the position of the POINT3D of PSF centered over current slice voxel
                                        //This is a bit complicated because slices can be oriented in any direction
                                        
                                        //PSF image coordinates
                                        x = ii;
                                        y = jj;
                                        z = kk;
                                        //change to PSF world coordinates - now real sizes in mm
                                        PSF.ImageToWorld(x, y, z);
                                        //centre around the centrepoint of the PSF
                                        x -= cx;
                                        y -= cy;
                                        z -= cz;
                                        
                                        //Need to convert (x,y,z) to slice image coordinates because slices can have
                                        //transformations included in them (they are nifti)  and those are not reflected in
                                        //PSF. In slice image coordinates we are sure that z is through-plane
                                        
                                        //adjust according to voxel size
                                        x /= dx;
                                        y /= dy;
                                        z /= dz;
                                        //center over current voxel
                                        x += i;
                                        y += j;
                                        
                                        
                                        reconstructor->SliceToReconstuced(inputIndex, x, y, z, nx, ny, nz, -1);
                                        
                                        //not all neighbours might be in ROI, thus we need to normalize
                                        //(l,m,n) are image coordinates of 8 neighbours in volume space
                                        //for each we check whether it is in volume
                                        sum = 0;
                                        //to find wether the current slice voxel has overlap with ROI
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
                                        //if there were no voxels do nothing
                                        if ((sum <= 0) || (!inside))
                                            continue;
                                        //now calculate the transformed PSF
                                        for (l = nx; l <= nx + 1; l++)
                                            if ((l >= 0) && (l < recon_attr._x))
                                                for (m = ny; m <= ny + 1; m++)
                                                    if ((m >= 0) && (m < recon_attr._y))
                                                        for (n = nz; n <= nz + 1; n++)
                                                            if ((n >= 0) && (n < recon_attr._z)) {
                                                                weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                                                
                                                                //image coordinates in tPSF (centre,centre,centre) in tPSF is aligned with (tx,ty,tz)
                                                                int aa, bb, cc;
                                                                aa = l - tx + centre;
                                                                bb = m - ty + centre;
                                                                cc = n - tz + centre;
                                                                
                                                                //resulting value
                                                                double value = PSF(ii, jj, kk) * weight / sum;
                                                            
                                                                //Check that we are in tPSF
                                                                if ((aa < 0) || (aa >= dim) || (bb < 0) || (bb >= dim) || (cc < 0) || (cc >= dim)) {
                                                                    
                                                                    char buffer[256];
                                                                     excluded++;
                                                                }
                                                                else {
                                                                    //update transformed PSF
                                                                    tPSF(aa, bb, cc) += value;
                                                                    
                                                                }
                                                            }
                                        
                                    } //end of the loop for PSF points
                            
                            //store tPSF values
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
                        } //end of loop for slice voxels
                
                } // end of condition for preliminary exclusion of slices                

                reconstructor->_volcoeffs[inputIndex] = slicecoeffs;
                reconstructor->_slice_inside[inputIndex] = slice_inside;
                
                // reconstructor->_excluded_points[inputIndex] = excluded;

            }  //end of loop through the slices
            
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    

    //-------------------------------------------------------------------
    
    /*
    class ParallelCoeffInitFFD {
        public:
        ReconstructionFFD *reconstructor;
        
        ParallelCoeffInitFFD(ReconstructionFFD *_reconstructor) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                bool slice_inside;
                
                //current slice
                //irtkRealImage slice;
                
                //get resolution of the volume
                double vx, vy, vz;
                reconstructor->_reconstructed.GetPixelSize(&vx, &vy, &vz);
                //volume is always isotropic
                double res = vx;
                
                //start of a loop for a slice inputIndex
                
                //prepare structures for storage
                POINT3D p;
                VOXELCOEFFS empty;
                SLICECOEFFS slicecoeffs(reconstructor->_slices[inputIndex]->GetX(), Array < VOXELCOEFFS > (reconstructor->_slices[inputIndex]->GetY(), empty));
                
                //to check whether the slice has an overlap with mask ROI
                slice_inside = false;
                
                //PSF will be calculated in slice space in higher resolution
                
                //get slice voxel size to define PSF
                double dx, dy, dz;
                reconstructor->_slices[inputIndex]->GetPixelSize(&dx, &dy, &dz);
                
                //sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM = dz through-plane)
                double sigmax = 1.2 * dx / 2.3548;
                double sigmay = 1.2 * dy / 2.3548;
                double sigmaz = dz / 2.3548;
                
                //calculate discretized PSF
                
                //isotropic voxel size of PSF - derived from resolution of reconstructed volume
                double size = res / reconstructor->_quality_factor;
                
                //number of voxels in each direction
                //the ROI is 2*voxel dimension
                
                int xDim = round(2 * dx / size);
                int yDim = round(2 * dy / size);
                int zDim = round(2 * dz / size);

                int excluded = 0;
                
                //image corresponding to PSF
                ImageAttributes attr;
                attr._x = xDim;
                attr._y = yDim;
                attr._z = zDim;
                attr._dx = size;
                attr._dy = size;
                attr._dz = size;
                RealImage PSF(attr);
                
                //centre of PSF
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
                    //continuous PSF does not need to be normalized as discrete will be
                    PSF(i, j, k) = exp(
                                       -x * x / (2 * sigmax * sigmax) - y * y / (2 * sigmay * sigmay)
                                       - z * z / (2 * sigmaz * sigmaz));
                    sum += PSF(i, j, k);
                }
                PSF /= sum;
                
                if (reconstructor->_debug)
                if (inputIndex == 0)
                PSF.Write("PSF.nii.gz");
                
                //prepare storage for PSF transformed and resampled to the space of reconstructed volume
                //maximum dim of rotated kernel - the next higher odd integer plus two to account for rounding error of tx,ty,tz.
                //Note conversion from PSF image coordinates to tPSF image coordinates *size/res
                int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2)) * 2 + 1 + 2;
                //prepare image attributes. Voxel dimension will be taken from the reconstructed volume
                attr._x = dim;
                attr._y = dim;
                attr._z = dim;
                attr._dx = res;
                attr._dy = res;
                attr._dz = res;
                //create matrix from transformed PSF
                RealImage tPSF(attr);
                //calculate centre of tPSF in image coordinates
                int centre = (dim - 1) / 2;
                
                //for each voxel in current slice calculate matrix coefficients
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
                    
                    for (i = 0; i < reconstructor->_slices[inputIndex]->GetX(); i++)
                    for (j = 0; j < reconstructor->_slices[inputIndex]->GetY(); j++)
                    // if (reconstructor->_slices[inputIndex](i, j, 0) != -1) {
                    if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) != -1 && reconstructor->_slice_2dncc[inputIndex]->GetAsDouble(i,j,0) > 0) {
                        //calculate centrepoint of slice voxel in volume space (tx,ty,tz)
                        x = i;
                        y = j;
                        z = 0;
                        reconstructor->_slices[inputIndex]->ImageToWorld(x, y, z);
                        
                        
//                        // 10/2018
//                        if(!reconstructor->_mffd_mode)
//                        reconstructor->_transformations[inputIndex].Transform(x, y, z);
//                        else
//                        //                                reconstructor->_mffd_transformations[inputIndex].LocalTransform(-1, 1, x, y, z);
                        reconstructor->_mffd_transformations[inputIndex]->Transform(-1, 1, x, y, z);
                        
                        
                        
                        
                        reconstructor->_reconstructed.WorldToImage(x, y, z);
                        tx = round(x);
                        ty = round(y);
                        tz = round(z);
                        
                        //Clear the transformed PSF
                        for (ii = 0; ii < dim; ii++)
                        for (jj = 0; jj < dim; jj++)
                        for (kk = 0; kk < dim; kk++)
                        tPSF(ii, jj, kk) = 0;
                        
                        //for each POINT3D of the PSF
                        for (ii = 0; ii < xDim; ii++)
                        for (jj = 0; jj < yDim; jj++)
                        for (kk = 0; kk < zDim; kk++) {
                            //Calculate the position of the POINT3D of
                            //PSF centered over current slice voxel
                            //This is a bit complicated because slices
                            //can be oriented in any direction
                            
                            //PSF image coordinates
                            x = ii;
                            y = jj;
                            z = kk;
                            //change to PSF world coordinates - now real sizes in mm
                            PSF.ImageToWorld(x, y, z);
                            //centre around the centrepoint of the PSF
                            x -= cx;
                            y -= cy;
                            z -= cz;
                            
                            //Need to convert (x,y,z) to slice image
                            //coordinates because slices can have
                            //transformations included in them (they are
                            //nifti)  and those are not reflected in
                            //PSF. In slice image coordinates we are
                            //sure that z is through-plane
                            
                            //adjust according to voxel size
                            x /= dx;
                            y /= dy;
                            z /= dz;
                            //center over current voxel
                            x += i;
                            y += j;
                            
                            //convert from slice image coordinates to world coordinates
                            reconstructor->_slices[inputIndex]->ImageToWorld(x, y, z);
                            
                            //x+=(vx-cx); y+=(vy-cy); z+=(vz-cz);
                            //Transform to space of reconstructed volume
                            
                            reconstructor->_mffd_transformations[inputIndex]->Transform(-1, 1, x, y, z);
                            
                            
                            
                            //Change to image coordinates
                            reconstructor->_reconstructed.WorldToImage(x, y, z);
                            
                            //determine coefficients of volume voxels for position x,y,z
                            //using linear interpolation
                            
                            //Find the 8 closest volume voxels
                            
                            //lowest corner of the cube
                            nx = (int) floor(x);
                            ny = (int) floor(y);
                            nz = (int) floor(z);
                            
                            //not all neighbours might be in ROI, thus we need to normalize
                            //(l,m,n) are image coordinates of 8 neighbours in volume space
                            //for each we check whether it is in volume
                            sum = 0;
                            //to find wether the current slice voxel has overlap with ROI
                            bool inside = false;
                            for (l = nx; l <= nx + 1; l++)
                            if ((l >= 0) && (l < reconstructor->_reconstructed.GetX()))
                            for (m = ny; m <= ny + 1; m++)
                            if ((m >= 0) && (m < reconstructor->_reconstructed.GetY()))
                            for (n = nz; n <= nz + 1; n++)
                            if ((n >= 0) && (n < reconstructor->_reconstructed.GetZ())) {
                                weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                sum += weight;
                                if (reconstructor->_mask(l, m, n) == 1) {
                                    inside = true;
                                    slice_inside = true;
                                }
                            }
                            //if there were no voxels do nothing
                            if ((sum <= 0) || (!inside))
                            continue;
                            //now calculate the transformed PSF
                            for (l = nx; l <= nx + 1; l++)
                            if ((l >= 0) && (l < reconstructor->_reconstructed.GetX()))
                            for (m = ny; m <= ny + 1; m++)
                            if ((m >= 0) && (m < reconstructor->_reconstructed.GetY()))
                            for (n = nz; n <= nz + 1; n++)
                            if ((n >= 0) && (n < reconstructor->_reconstructed.GetZ())) {
                                weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                
                                //image coordinates in tPSF
                                //(centre,centre,centre) in tPSF is aligned with (tx,ty,tz)
                                int aa, bb, cc;
                                aa = l - tx + centre;
                                bb = m - ty + centre;
                                cc = n - tz + centre;
                                
                                //resulting value
                                double value = PSF(ii, jj, kk) * weight / sum;
                                
                                
                                
                                //Check that we are in tPSF
                                if ((aa < 0) || (aa >= dim) || (bb < 0) || (bb >= dim) || (cc < 0) || (cc >= dim)) {
                                    excluded++;
                                }
                                else {
                                    //update transformed PSF
                                    tPSF(aa, bb, cc) += value;
                                    
                                }
                            }
                            
                        } //end of the loop for PSF points
                        
                        //store tPSF values
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
                    } //end of loop for slice voxels
                    
                } // end of condition for preliminary exclusion of slices
                
                reconstructor->_volcoeffs[inputIndex] = slicecoeffs;
                reconstructor->_slice_inside[inputIndex] = slice_inside;
                
                
                // reconstructor->_excluded_points[inputIndex] = excluded;
                
            }  //end of loop through the slices
            
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    */
    
    void ReconstructionFFD::CoeffInit()
    {
    
        //clear slice-volume matrix from previous iteration
        _volcoeffs.clear();
        _volcoeffs.resize(_slices.size());
        
        //clear indicator of slice having and overlap with volumetric mask
        _slice_inside.clear();
        _slice_inside.resize(_slices.size());
        
        _reconstructed_attr = _reconstructed.GetImageAttributes();
        
        ParallelCoeffInitFFD *p_cinit = new ParallelCoeffInitFFD(this);
        (*p_cinit)();

        delete p_cinit;

        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        _volume_weights.Initialize( _reconstructed.GetImageAttributes() );
        _volume_weights = 0;
        
        cout << _reconstructed.GetXSize() << endl;

        cout << _slices.size() << endl;
        
        int inputIndex, i, j, n, k;
        POINT3D p;
        for ( inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            for ( i = 0; i < _slices[inputIndex]->GetX(); i++)
                for ( j = 0; j < _slices[inputIndex]->GetY(); j++) {
                    
                    n = _volcoeffs[inputIndex][i][j].size();
                    for (k = 0; k < n; k++) {
                        p = _volcoeffs[inputIndex][i][j][k];
                        _volume_weights(p.x, p.y, p.z) += p.value;
                    }
                    
                }
        }
        if (_debug)
            _volume_weights.Write("volume_weights.nii.gz");
        
        //find average volume weight to modify alpha parameters accordingly
        RealPixel *ptr = _volume_weights.GetPointerToVoxels();
        RealPixel *pm = _mask.GetPointerToVoxels();
        double sum = 0;
        int num=0;
        for (int i=0;i<_volume_weights.GetNumberOfVoxels();i++) {
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
        
        //clear _reconstructed image
        _reconstructed = 0;

        cout << _reconstructed.GetXSize() << endl;

        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            //copy the current slice
            slice = *(_slices[inputIndex]);
            //alias the current bias image
            RealImage& b = *(_bias[inputIndex]);
            //read current scale factor
            scale = _scale[inputIndex];
            
            slice_vox_num=0;
            
            //Distribute slice intensities to the volume
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {
                        //biascorrect and scale the slice
                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                        
                        //number of volume voxels with non-zero coefficients
                        //for current slice voxel
                        n = _volcoeffs[inputIndex][i][j].size();
                        
                        //if given voxel is not present in reconstructed volume at all,
                        //pad it
                        
                        //if (n == 0)
                        //_slices[inputIndex].PutAsDouble(i, j, 0, -1);
                        //calculate num of vox in a slice that have overlap with roi
                        if (n>0)
                            slice_vox_num++;
                        
                        //add contribution of current slice voxel to all voxel volumes
                        //to which it contributes
                        for (k = 0; k < n; k++) {
                            p = _volcoeffs[inputIndex][i][j][k];
                            _reconstructed(p.x, p.y, p.z) += p.value * slice(i, j, 0);

                            //displacement_map(p.x, p.y, p.z) += p.value * _slice_displacements[inputIndex](i, j, 0);
                        }
                    }
            voxel_num.push_back(slice_vox_num);
            //end of loop for a slice inputIndex
        }
        
        
//        GaussianBlurring<RealPixel> gb_cf(_reconstructed.GetXSize()*0.55);
//        
//        gb_cf.Input(&_reconstructed);
//        gb_cf.Output(&_reconstructed);
//        gb_cf.Run();
//        
//        
//        gb_cf.Input(&_volume_weights);
//        gb_cf.Output(&_volume_weights);
//        gb_cf.Run();
        
        
        
        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxe
        _reconstructed /= _volume_weights;
        
//         if (_debug)
            _reconstructed.Write("init.nii.gz");
        
        _volume_weights.Write("weights.nii.gz");
        
        //now find slices with small overlap with ROI and exclude them.
        
        Array<int> voxel_num_tmp;
        for (i=0;i<voxel_num.size();i++)
            voxel_num_tmp.push_back(voxel_num[i]);
        
        //find median
        sort(voxel_num_tmp.begin(),voxel_num_tmp.end());
        int median = voxel_num_tmp[round(voxel_num_tmp.size()*0.5)];
        
        //remember slices with small overlap with ROI
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

        //Find the range of intensities
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
        
        
        //Exclude the entire stack
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
            //Initialise voxel weights and bias values
            RealPixel *pw = _weights[i]->GetPointerToVoxels();
            RealPixel *pb = _bias[i]->GetPointerToVoxels();
            RealPixel *pi = _slices[i]->GetPointerToVoxels();
            for (int j = 0; j < _weights[i]->GetNumberOfVoxels(); j++) {
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
            
            //Initialise slice weights
            _slice_weight[i] = 1;
            
            //Initialise scaling factors for intensity matching
            _scale[i] = 1;
            
            if (_stack_index[i] == _excluded_stack)
                _slice_weight[i] = 0;
        }
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::InitializeRobustStatistics()
    {
        
        //Initialise parameter of EM robust statistics
        int i, j;
        RealImage slice, sim;
        double sigma = 0;
        int num = 0;
        
        //for each slice
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            slice = *_slices[inputIndex];
            
            //Voxel-wise sigma will be set to stdev of volumetric errors
            //For each slice voxel
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) > -1) {
                        //calculate stev of the errors
                        if ( (_simulated_inside[inputIndex]->GetAsDouble(i, j, 0)==1) &&(_simulated_weights[inputIndex]->GetAsDouble(i,j,0)>0.99) ) {
                            slice(i,j,0) -= _simulated_slices[inputIndex]->GetAsDouble(i,j,0);
                            sigma += slice(i, j, 0) * slice(i, j, 0);
                            num++;
                        }
                    }
            
            //if slice does not have an overlap with ROI, set its weight to zero
            if (!_slice_inside[inputIndex])
                _slice_weight[inputIndex] = 0;
        }
        
        //Force exclusion of slices predefined by user
        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            _slice_weight[_force_excluded[i]] = 0;
        
        
        
        bool done = false;
        //Exclude the entire stack
        for (unsigned int i = 0; i < _slices.size(); i++) {
            if (_stack_index[i] == _excluded_stack) {
                _slice_weight[i] = 0;
                if (!done) {
                    cout << "Excluded stack : " << _stack_index[i] << endl;
                    done = true;
                }
            }
        }
        
        //initialize sigma for voxelwise robust statistics
        _sigma = sigma / num;
        //initialize sigma for slice-wise robust statistics
        _sigma_s = 0.025;
        //initialize mixing proportion for inlier class in voxel-wise robust statistics
        _mix = 0.9;
        //initialize mixing proportion for outlier class in slice-wise robust statistics
        _mix_s = 0.9;
        //Initialise value for uniform distribution according to the range of intensities
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

//                reconstructor->_weights[inputIndex] = 0;
                
                double num = 0;
                //Calculate error, voxel weights, and slice potential
                for (int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++) {
                        
                        reconstructor->_weights[inputIndex]->PutAsDouble(i, j, 0, 0);
                        
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) != -1) {
                            //bias correct and scale the slice
                            // slice(i, j, 0) *= exp(-(reconstructor->_bias[inputIndex])(i, j, 0)) * scale;
                            
                            //number of volumetric voxels to which current slice voxel contributes
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            
                            // if n == 0, slice voxel has no overlap with volumetric ROI,
                            // do not process it
                            
                            if ( (n>0) && (reconstructor->_simulated_weights[inputIndex]->GetAsDouble(i,j,0) > 0) ) {
                                // slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                                
                                //calculate norm and voxel-wise weights
                                //Gaussian distribution for inliers (likelihood)
                                double g = reconstructor->G(reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0), reconstructor->_sigma);
                                //Uniform distribution for outliers (likelihood)
                                double m = reconstructor->M(reconstructor->_m);
                                
                                //voxel_wise posterior
                                double weight = g * reconstructor->_mix / (g *reconstructor->_mix + m * (1 - reconstructor->_mix));
                                reconstructor->_weights[inputIndex]->PutAsDouble(i, j, 0, weight);
                                
                                //calculate slice potentials
                                if(reconstructor->_simulated_weights[inputIndex]->GetAsDouble(i,j,0)>0.99) {
                                    slice_potential[inputIndex] += (1 - weight) * (1 - weight);
                                    num++;
                                }
                            }
                            else
                                reconstructor->_weights[inputIndex]->PutAsDouble(i, j, 0, 0);
                        }
                
                    }
                //evaluate slice potential
                if (num > 0)
                    slice_potential[inputIndex] = sqrt(slice_potential[inputIndex] / num);
                else
                    slice_potential[inputIndex] = -1; // slice has no unpadded voxels
            }
             
             
            
            
        }
        
        ParallelEStepFFD( ReconstructionFFD *reconstructor,
                      Array<double> &slice_potential ) :
        reconstructor(reconstructor), slice_potential(slice_potential)
        { }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::EStep()
    {
        //EStep performs calculation of voxel-wise and slice-wise posteriors (weights)
        
        unsigned int inputIndex;
        RealImage slice, w, b, sim;
        int num = 0;
        Array<double> slice_potential(_slices.size(), 0);


        SliceDifference();
        
        ParallelEStepFFD *p_estep = new ParallelEStepFFD( this, slice_potential );
        (*p_estep)();

        delete p_estep;

        //To force-exclude slices predefined by a user, set their potentials to -1
        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            slice_potential[_force_excluded[i]] = -1;
        
        //exclude slices identified as having small overlap with ROI, set their potentials to -1
        for (unsigned int i = 0; i < _small_slices.size(); i++)
            slice_potential[_small_slices[i]] = -1;
        
        //these are unrealistic scales pointing at misregistration - exclude the corresponding slices
        for (inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
            if ((_scale[inputIndex]<0.2)||(_scale[inputIndex]>5)) {
                slice_potential[inputIndex] = -1;
            }
        
        for (unsigned int i = 0; i < _slices.size(); i++) {
            if (_stack_index[i] == _excluded_stack) {
                slice_potential[i] = -1;
            }
        }
        
//        // exclude unrealistic transformations
//        if(_debug) {
//            cout<<endl<<"Slice potentials: ";
//            for (inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
//                cout<<slice_potential[inputIndex]<<" ";
//            cout<<endl;
//        }
        
        
        //Calulation of slice-wise robust statistics parameters.
        //This is theoretically M-step, but we want to use latest estimate of slice potentials to update the parameters
        
        //Calculate means of the inlier and outlier potentials
        double sum = 0, den = 0, sum2 = 0, den2 = 0, maxs = 0, mins = 1;
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            if (slice_potential[inputIndex] >= 0) {
                //calculate means
                sum += slice_potential[inputIndex] * _slice_weight[inputIndex];
                den += _slice_weight[inputIndex];
                sum2 += slice_potential[inputIndex] * (1 - _slice_weight[inputIndex]);
                den2 += (1 - _slice_weight[inputIndex]);
                
                //calculate min and max of potentials in case means need to be initalized
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
        
        //Calculate the variances of the potentials
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
        
        //_sigma_s
        if ((sum > 0) && (den > 0)) {
            _sigma_s = sum / den;
            //do not allow too small sigma
            if (_sigma_s < _step * _step / 6.28)
                _sigma_s = _step * _step / 6.28;
        }
        else {
            _sigma_s = 0.025;
            if (_debug) {
                if (sum <= 0)
                    cout << "All slices are equal. ";
                if (den < 0) //this should not happen
                    cout << "All slices are outliers. ";
                cout << "Setting sigma to " << sqrt(_sigma_s) << endl;
            }
        }
        
        //sigma_s2
        if ((sum2 > 0) && (den2 > 0)) {
            _sigma_s2 = sum2 / den2;
            //do not allow too small sigma
            if (_sigma_s2 < _step * _step / 6.28)
                _sigma_s2 = _step * _step / 6.28;
        }
        else {
            _sigma_s2 = (_mean_s2 - _mean_s) * (_mean_s2 - _mean_s) / 4;
            //do not allow too small sigma
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
        
        //Calculate slice weights
        double gs1, gs2;
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            //Slice does not have any voxels in volumetric ROI
            if (slice_potential[inputIndex] == -1) {
                _slice_weight[inputIndex] = 0;
                continue;
            }
            
            //All slices are outliers or the means are not valid
            if ((den <= 0) || (_mean_s2 <= _mean_s)) {
                _slice_weight[inputIndex] = 1;
                continue;
            }
            
            //likelihood for inliers
            if (slice_potential[inputIndex] < _mean_s2)
                gs1 = G(slice_potential[inputIndex] - _mean_s, _sigma_s);
            else
                gs1 = 0;
            
            //likelihood for outliers
            if (slice_potential[inputIndex] > _mean_s)
                gs2 = G(slice_potential[inputIndex] - _mean_s2, _sigma_s2);
            else
                gs2 = 0;
            
            //calculate slice weight
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
        
        //Update _mix_s this should also be part of MStep
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
//            cout << "Slice weights: ";
//            for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
//                cout << _slice_weight[inputIndex] << " ";
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
                
                // // alias the current slice
                // RealImage& slice = reconstructor->_slices[inputIndex];
                
                // //alias the current weight image
                // RealImage& w = reconstructor->_weights[inputIndex];
                
                // //alias the current bias image
                // RealImage& b = reconstructor->_bias[inputIndex];
                
                //initialise calculation of scale
                double scalenum = 0;
                double scaleden = 0;
                
                for (int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++)
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) != -1) {       // && (reconstructor->_slice_masks[inputIndex](i, j, 0) > 0
                            if(reconstructor->_simulated_weights[inputIndex]->GetAsDouble(i,j,0)>0.99) {
                                //scale - intensity matching
                                double eb = exp(-reconstructor->_bias[inputIndex]->GetAsDouble(i, j, 0));
                                scalenum += reconstructor->_weights[inputIndex]->GetAsDouble(i, j, 0) * reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) * eb * reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i, j, 0);
                                scaleden += reconstructor->_weights[inputIndex]->GetAsDouble(i, j, 0) * reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) * eb * reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) * eb;
                            }
                        }
                
                //calculate scale for this slice
                if (scaleden > 0)
                    reconstructor->_scale[inputIndex] = scalenum / scaleden;
                else
                    reconstructor->_scale[inputIndex] = 1;
                
            } //end of loop for a slice inputIndex
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
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
                // read the current slice
                RealImage slice = *reconstructor->_slices[inputIndex];
                
                //alias the current weight image
                RealImage& w = *reconstructor->_weights[inputIndex];
                
                //alias the current bias image
                RealImage b = *reconstructor->_bias[inputIndex];
                
                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];
                
                //prepare weight image for bias field
                RealImage wb = w;
                
                //simulated slice
                // irtkRealImage sim;
                // sim.Initialize( slice.GetImageAttributes() );
                // sim = 0;
                RealImage wresidual( slice.GetImageAttributes() );
                wresidual = 0;
                
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) > -1) {
                            if( reconstructor->_simulated_weights[inputIndex]->GetAsDouble(i,j,0) > 0.99 ) {
                                //bias-correct and scale current slice
                                double eb = exp(-b(i, j, 0));
                                slice(i, j, 0) *= (eb * scale);
                                
                                //calculate weight image
                                wb(i, j, 0) = w(i, j, 0) * slice(i, j, 0);
                                
                                //calculate weighted residual image
                                //make sure it is far from zero to avoid numerical instability
                                //if ((sim(i,j,0)>_low_intensity_cutoff*_max_intensity)&&(slice(i,j,0)>_low_intensity_cutoff*_max_intensity))
                                if ( (reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i, j, 0) > 1) && (slice(i, j, 0) > 1)) {
                                    wresidual(i, j, 0) = log(slice(i, j, 0) / reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i, j, 0)) * wb(i, j, 0);
                                }
                            }
                            else {
                                //do not take into account this voxel when calculating bias field
                                wresidual(i, j, 0) = 0;
                                wb(i, j, 0) = 0;
                            }
                        }
                
                //calculate bias field for this slice
                GaussianBlurring<RealPixel> gb(reconstructor->_sigma_bias);
                //smooth weighted residual
                gb.Input(&wresidual);
                gb.Output(&wresidual);
                gb.Run();
                
                //smooth weight image
                gb.Input(&wb);
                gb.Output(&wb);
                gb.Run();
                
                //update bias field
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
                
                //normalize bias field to have zero mean
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
                
//                RealImage *tmp_b = new RealImage(b);
                
                reconstructor->_bias[inputIndex] = new RealImage(b); //tmp_b;
            }
        }
        
        ParallelBiasFFD( ReconstructionFFD *reconstructor ) :
        reconstructor(reconstructor)
        { }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
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

//            _slice_dif[inputIndex] = _slices[inputIndex];
            // _slice_dif[inputIndex] = -1;
            

            for (int i = 0; i < _slices[inputIndex]->GetX(); i++) {
                for (int j = 0; j < _slices[inputIndex]->GetY(); j++) {
                    
                    
                    double ds = _slices[inputIndex]->GetAsDouble(i, j, 0);
                    
                    if (_slices[inputIndex]->GetAsDouble(i, j, 0) != -1) {

                        ds = ds * exp(-(_bias[inputIndex])->GetAsDouble(i, j, 0)) * _scale[inputIndex];
                        ds = ds - _simulated_slices[inputIndex]->GetAsDouble(i, j, 0);

                    }
                    else 
                        ds = 0;
                    
                    _slice_dif[inputIndex]->PutAsDouble(i, j, 0, ds);

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
//        RealImage all_addon;
//        RealImage all_confidence_map;
        
        void operator()( const blocked_range<size_t>& r ) {
            
            
            for (size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                // read the current slice
                // RealImage slice = reconstructor->_slices[inputIndex];

                //                //read the current weight image
                //                RealImage& w = reconstructor->_weights[inputIndex];
                //
                //                //read the current bias image
                //                RealImage& b = reconstructor->_bias[inputIndex];

                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];

                //Update reconstructed volume using current slice

                //Distribute error to the volume
                POINT3D p;
                for (int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++)
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) != -1) {
                            //bias correct and scale the slice
                            // slice(i, j, 0) *= exp(-(reconstructor->_bias[inputIndex])(i, j, 0)) * scale;

                            if (reconstructor->_simulated_slices[inputIndex]->GetAsDouble(i, j, 0) < 0.01)
                                reconstructor->_slice_dif[inputIndex]->PutAsDouble(i, j, 0, 0);

                            //     slice(i, j, 0) -= reconstructor->_simulated_slices[inputIndex](i, j, 0);
                            // else
                            //     slice(i, j, 0) = 0;

                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for (int k = 0; k < n; k++) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                
                                
//                                if (reconstructor->_stack_index[inputIndex] == reconstructor->_excluded_stack)
//                                    reconstructor->_slice_potential[inputIndex] = -1;
                                
                                if (reconstructor->_structural_exclusion) {

                                    if (reconstructor->_stack_index[inputIndex] != reconstructor->_excluded_stack) {
                                        
                                        
                                        if (reconstructor->_slice_weight[inputIndex] < 0.6) {
                                            reconstructor->_structural_slice_weight[inputIndex] = reconstructor->_slice_weight[inputIndex]/10;
                                        }
                                        else {
                                            reconstructor->_structural_slice_weight[inputIndex] = reconstructor->_slice_weight[inputIndex];
                                        }

                                        double sw = reconstructor->_slice_2dncc[inputIndex]->GetAsDouble(i, j, 0);
                                        
//                                        if (!reconstructor->_robust_slices_only) {

                                        
                                        // org
                                            addon(p.x, p.y, p.z) += sw * p.value * reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0) * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_structural_slice_weight[inputIndex];
                                            confidence_map(p.x, p.y, p.z) += sw * p.value * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_structural_slice_weight[inputIndex];
                                        
                                        
//                                        addon(p.x, p.y, p.z) += sw * p.value * reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0) * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_slice_weight[inputIndex];
//                                        confidence_map(p.x, p.y, p.z) += sw * p.value * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        
                                        
//                                        }
//                                        else {
//
//                                            addon(p.x, p.y, p.z) += sw * p.value * reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0) * reconstructor->_structural_slice_weight[inputIndex];
//                                            confidence_map(p.x, p.y, p.z) += sw * p.value * reconstructor->_structural_slice_weight[inputIndex];
//
//                                        }
                                        
//                                        all_addon(p.x, p.y, p.z) += p.value * reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0);
//                                        all_confidence_map(p.x, p.y, p.z) += p.value;
                                        
                                        
                                    }
                                    

                                }
                                else {
                                    if (reconstructor->_robust_slices_only) {

                                        addon(p.x, p.y, p.z) += p.value * reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        confidence_map(p.x, p.y, p.z) += p.value * reconstructor->_slice_weight[inputIndex];

                                        //                                    addon(p.x, p.y, p.z) += p.value * (reconstructor->_weights[inputIndex])(i, j, 0); // * reconstructor->_slice_weight[inputIndex];
                                        //                                    confidence_map(p.x, p.y, p.z) += p.value * (reconstructor->_weights[inputIndex])(i, j, 0); // * reconstructor->_slice_weight[inputIndex];

                                    } else {
                                        addon(p.x, p.y, p.z) += p.value * reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0) * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        confidence_map(p.x, p.y, p.z) += p.value * (reconstructor->_weights[inputIndex])->GetAsDouble(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                    }
                                }

                            }
                        }
            } //end of loop for a slice inputIndex
            
            
            
        }
        
        ParallelSuperresolutionFFD( ParallelSuperresolutionFFD& x, split ) :
        reconstructor(x.reconstructor)
        {
            //Clear addon
            addon.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            addon = 0;
            
            //Clear confidence map
            confidence_map.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            confidence_map = 0;
            
            
//            all_addon.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
//            all_addon = 0;
//
//            all_confidence_map.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
//            all_confidence_map = 0;

        }
        
        void join( const ParallelSuperresolutionFFD& y ) {
            addon += y.addon;
            confidence_map += y.confidence_map;
            
//            all_addon += y.all_addon;
//            all_confidence_map += y.all_confidence_map;
        }
        
        ParallelSuperresolutionFFD( ReconstructionFFD *reconstructor ) :
        reconstructor(reconstructor)
        {
            //Clear addon
            addon.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            addon = 0;
            
            //Clear confidence map
            confidence_map.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            confidence_map = 0;
            
            
//            all_addon.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
//            all_addon = 0;
//
//            all_confidence_map.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
//            all_confidence_map = 0;
            
            
        }
        
        // execute
        void operator() () {
            //task_scheduler_init init(8);
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()), *this );
            //init.terminate();
        }
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::Superresolution(int iter)
    {

        RealImage addon, original;

        //Remember current reconstruction for edge-preserving smoothing
        original = _reconstructed;

        SliceDifference();
        
        ParallelSuperresolutionFFD *p_super = new  ParallelSuperresolutionFFD(this);
        (*p_super)();
        addon = p_super->addon;
        _confidence_map = p_super->confidence_map;
        
        
//        RealImage all_confidence_map;
//        all_addon = p_super->all_addon;
//        all_confidence_map = p_super->all_confidence_map;
        
        
        delete p_super;
        


        if(_debug) {
            char buffer[256];
            sprintf(buffer,"confidence-map-%i.nii.gz",_current_iteration);
            _confidence_map.Write(buffer);
            
            sprintf(buffer,"addon-%i.nii.gz",_current_iteration);
            addon.Write(buffer);
            
            
//            sprintf(buffer,"all-confidence-map-%i.nii.gz",_current_iteration);
//            all_confidence_map.Write(buffer);
//
//            sprintf(buffer,"all-addon-%i.nii.gz",_current_iteration);
//            all_addon.Write(buffer);

        }
        
        
        
        GaussianBlurring<RealPixel> gb_a(addon.GetXSize()*0.55);
        
        gb_a.Input(&addon);
        gb_a.Output(&addon);
        gb_a.Run();
        
        
        GaussianBlurring<RealPixel> gb_cf(_confidence_map.GetXSize()*0.55);
        
        gb_cf.Input(&_confidence_map);
        gb_cf.Output(&_confidence_map);
        gb_cf.Run();
        
        

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
                        // ISSUES if _confidence_map(i, j, k) is too small leading
                        // to bright pixels
                        addon(i, j, k) /= _confidence_map(i, j, k);
                        //this is to revert to normal (non-adaptive) regularisation
                        _confidence_map(i,j,k) = 1;
                    }
                    else  {
                        // addon(i, j, k) /= 0.1;
                        _confidence_map(i,j,k) = 1;         // ??
                        
                    }
                }
            }
        }
        

        if(_debug) {
            char buffer[256];
            
            sprintf(buffer,"after-addon-%i.nii.gz",_current_iteration);
            addon.Write(buffer);
        }
        
        _reconstructed += addon * _alpha; //_average_volume_weight;
        
        //bound the intensities
        for (int i = 0; i < _reconstructed.GetX(); i++)
            for (int j = 0; j < _reconstructed.GetY(); j++)
                for (int k = 0; k < _reconstructed.GetZ(); k++) {
                    if (_reconstructed(i, j, k) < _min_intensity * 0.9)
                        _reconstructed(i, j, k) = _min_intensity * 0.9;
                    if (_reconstructed(i, j, k) > _max_intensity * 1.1)
                        _reconstructed(i, j, k) = _max_intensity * 1.1;
                }
        
        //Smooth the reconstructed image
        AdaptiveRegularization(iter, original);
        //Remove the bias in the reconstructed volume compared to previous iteration
        if (_global_bias_correction)
            BiasCorrectVolume(original);
        
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
                // read the current slice
                // RealImage slice = reconstructor->_slices[inputIndex];
                
//                //alias the current weight image
//                RealImage& w = reconstructor->_weights[inputIndex];
//
//                //alias the current bias image
//                RealImage& b = reconstructor->_bias[inputIndex];
                
                //identify scale factor
                // double scale = reconstructor->_scale[inputIndex];
                
                double LNCC = 0;
                int box_size = 14;
                
                //calculate error
                for (int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++)
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) > -1) {
                            //bias correct and scale the slice
                            // slice(i, j, 0) *= exp(-(reconstructor->_bias[inputIndex])(i, j, 0)) * scale;
                            
                            //otherwise the error has no meaning - it is equal to slice intensity
                            if ( reconstructor->_simulated_weights[inputIndex]->GetAsDouble(i,j,0) > 0.99 ) {
                                
                                // slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                                
                                //sigma and mix
                                double e = reconstructor->_slice_dif[inputIndex]->GetAsDouble(i, j, 0);
                                sigma += e * e * reconstructor->_weights[inputIndex]->GetAsDouble(i, j, 0);
                                mix += reconstructor->_weights[inputIndex]->GetAsDouble(i, j, 0);
                                
                                //_m
                                if (e < min)
                                    min = e;
                                if (e > max)
                                    max = e;
                                
                                num++;
                            }
                        }
            } //end of loop for a slice inputIndex
            
            
            
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
        
        // execute
        void operator() () {
            //task_scheduler_init init(8);
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()), *this );
            //init.terminate();
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
        
        //Calculate sigma and mix
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
        
        //Calculate m
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
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, 13), *this );
            //init.terminate();
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
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed.GetX()), *this );
            //init.terminate();
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
        
        Array<RealImage> b;//(13);
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
        //remove low-frequancy component in the reconstructed image which might have accured due to overfitting of the biasfield
        RealImage residual = _reconstructed;
        RealImage weights = _mask;
        
        //calculate weighted residual
        RealPixel *pr = residual.GetPointerToVoxels();
        RealPixel *po = original.GetPointerToVoxels();
        RealPixel *pw = weights.GetPointerToVoxels();
        for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
            //second and term to avoid numerical problems
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

        //blurring needs to be same as for slices
        GaussianBlurring<RealPixel> gb(_sigma_bias);
        //blur weigted residual
        gb.Input(&residual);
        gb.Output(&residual);
        gb.Run();
        //blur weight image
        gb.Input(&weights);
        gb.Output(&weights);
        gb.Run();
        
        //calculate the bias field
        pr = residual.GetPointerToVoxels();
        pw = weights.GetPointerToVoxels();
        RealPixel *pm = _mask.GetPointerToVoxels();
        RealPixel *pi = _reconstructed.GetPointerToVoxels();
        for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
            
            if (*pm == 1) {
                //weighted gaussian smoothing
                *pr /= *pw;
                //exponential to recover multiplicative bias field
                *pr = exp(*pr);
                //bias correct reconstructed
                *pi /= *pr;
                //clamp intensities to allowed range
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
                
//                if(reconstructor->_debug) {
//                    cout<<inputIndex<<" ";
//                }
                
                // alias the current slice
                // RealImage& slice = reconstructor->_slices[inputIndex];
                
                //read the current bias image
                RealImage b = *reconstructor->_bias[inputIndex];
                
                //read current scale factor
                double scale = reconstructor->_scale[inputIndex];
                
                RealPixel *pi = reconstructor->_slices[inputIndex]->GetPointerToVoxels();
                RealPixel *pb = b.GetPointerToVoxels();
                for(int i = 0; i<reconstructor->_slices[inputIndex]->GetNumberOfVoxels(); i++) {
                    if((*pi>-1)&&(scale>0))
                        *pb -= log(scale);
                    pb++;
                    pi++;
                }
                
                //Distribute slice intensities to the volume
                POINT3D p;
                for (int i = 0; i < reconstructor->_slice_attr[inputIndex]._x; i++)
                    for (int j = 0; j < reconstructor->_slice_attr[inputIndex]._y; j++)
                        if (reconstructor->_slices[inputIndex]->GetAsDouble(i, j, 0) != -1) {
                            //number of volume voxels with non-zero coefficients for current slice voxel
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            //add contribution of current slice voxel to all voxel volumes
                            //to which it contributes
                            for (int k = 0; k < n; k++) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                bias(p.x, p.y, p.z) += p.value * b(i, j, 0);
                            }
                        }
                //end of loop for a slice inputIndex
            }
        }
        
        ParallelNormaliseBiasFFD( ParallelNormaliseBiasFFD& x, split ) :
        reconstructor(x.reconstructor)
        {
            bias.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            bias = 0;
        }
        
        void join( const ParallelNormaliseBiasFFD& y ) {
            bias += y.bias;
        }
        
        ParallelNormaliseBiasFFD( ReconstructionFFD *reconstructor ) :
        reconstructor(reconstructor)
        {
            bias.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            bias = 0;
        }
        
        // execute
        void operator() () {
            //task_scheduler_init init(8);
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()), *this );
            //init.terminate();
        }
    };
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::NormaliseBias( int iter )
    {
        
        ParallelNormaliseBiasFFD *p_bias = new ParallelNormaliseBiasFFD(this);
        (*p_bias)();
        RealImage bias = p_bias->bias;
        
        delete p_bias;

        // normalize the volume by proportion of contributing slice voxels for each volume voxel
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
        pi = _reconstructed.GetPointerToVoxels();
        pb = bias.GetPointerToVoxels();
        for (int i = 0; i<_reconstructed.GetNumberOfVoxels();i++) {
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
        
        // header
        info << "stack_index" << "\t"
        << "stack_name" << "\t"
        << "included" << "\t" // Included slices
        << "excluded" << "\t"  // Excluded slices
        << "outside" << "\t"  // Outside slices
        << "weight" << "\t"
        << "scale" << "\t"
        // << _stack_factor[i] << "\t"
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
            << (((_slice_weight[i] >= 0.5) && (_slice_inside[i]))?1:0) << "\t" // Included slices
            << (((_slice_weight[i] < 0.5) && (_slice_inside[i]))?1:0) << "\t"  // Excluded slices
            << ((!(_slice_inside[i]))?1:0) << "\t"  // Outside slices
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
        
        ImageAttributes attr = image.GetImageAttributes();
        
        //slices in package
        int pkg_z = attr._z / packages;
        double pkg_dz = attr._dz*packages;

        char buffer[256];
        int i, j, k, l;
        double x, y, z, sx, sy, sz, ox, oy, oz;
        for (l = 0; l < packages; l++) {
            attr = image.GetImageAttributes();
            if ((pkg_z * packages + l) < attr._z)
                attr._z = pkg_z + 1;
            else
                attr._z = pkg_z;
            attr._dz = pkg_dz;
            
            //fill values in each stack
            RealImage stack(attr);
            stack.GetOrigin(ox, oy, oz);
            
            for (k = 0; k < stack.GetZ(); k++)
                for (j = 0; j < stack.GetY(); j++)
                    for (i = 0; i < stack.GetX(); i++)
                        stack.Put(i, j, k, image(i, j, k * packages + l));
            
            //adjust origin
            //original image coordinates
            x = 0;
            y = 0;
            z = l;
            image.ImageToWorld(x, y, z);
            //stack coordinates
            sx = 0;
            sy = 0;
            sz = 0;
            stack.PutOrigin(ox, oy, oz); //adjust to original value
            stack.ImageToWorld(sx, sy, sz);

            //adjust origin
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
        //Crops the image according to the mask
        int i, j, k;
        //ROI boundaries
        int x1, x2, y1, y2, z1, z2;
        
        //Original ROI
        x1 = 0;
        y1 = 0;
        z1 = 0;
        x2 = image.GetX();
        y2 = image.GetY();
        z2 = image.GetZ();
        
        //upper boundary for z coordinate
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
        
        //lower boundary for z coordinate
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
        
        //upper boundary for y coordinate
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
        
        //lower boundary for y coordinate
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
        
        //upper boundary for x coordinate
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
        
        //lower boundary for x coordinate
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
            // if no intersection with mask, force exclude
            if ((x2 < x1) || (y2 < y1) || (z2 < z1) ) {
                x1 = 0;
                y1 = 0;
                z1 = 0;
                x2 = 0;
                y2 = 0;
                z2 = 0;
            }
        }
        
        
        
        
        
        
        //Cut region of interest
        image = image.GetRegion(x1, y1, z1, x2+1, y2+1, z2+1);
        
    }
    
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::CropImageP(RealImage *image, RealImage* mask)
    {
        //Crops the image according to the mask
        int i, j, k;
        //ROI boundaries
        int x1, x2, y1, y2, z1, z2;
        
        //Original ROI
        x1 = 0;
        y1 = 0;
        z1 = 0;
        x2 = image->GetX();
        y2 = image->GetY();
        z2 = image->GetZ();
        
        //upper boundary for z coordinate
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
        
        //lower boundary for z coordinate
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
        
        //upper boundary for y coordinate
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
        
        //lower boundary for y coordinate
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
        
        //upper boundary for x coordinate
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
        
        //lower boundary for x coordinate
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
            // if no intersection with mask, force exclude
            if ((x2 < x1) || (y2 < y1) || (z2 < z1) ) {
                x1 = 0;
                y1 = 0;
                z1 = 0;
                x2 = 0;
                y2 = 0;
                z2 = 0;
            }
        }
        
        //Cut region of interest
        RealImage tmp = image->GetRegion(x1, y1, z1, x2+1, y2+1, z2+1);
        
        RealImage *tmp_p = new RealImage(tmp);
        
        image = tmp_p;
        
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::InvertStackTransformations( Array<RigidTransformation>& stack_transformations )
    {
        //for each stack
        for (unsigned int i = 0; i < stack_transformations.size(); i++) {
            //invert transformation for the stacks
            stack_transformations[i].Invert();
            stack_transformations[i].UpdateParameter();
        }
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::MaskVolume()
    {
        RealPixel *pr = _reconstructed.GetPointerToVoxels();
        RealPixel *pm = _mask.GetPointerToVoxels();
        for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
            if (*pm == 0)
                *pr = 0;
            pm++;
            pr++;
        }
    }
    
    //-------------------------------------------------------------------
    
    void ReconstructionFFD::MaskImage( RealImage& image, double padding )
    {
        if(image.GetNumberOfVoxels()!=_mask.GetNumberOfVoxels()) {
            cerr<<"Cannot mask the image - different dimensions"<<endl;
            exit(1);
        }
        RealPixel *pr = image.GetPointerToVoxels();
        RealPixel *pm = _mask.GetPointerToVoxels();
        
        for (int i = 0; i < image.GetNumberOfVoxels(); i++) {
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
        
        // Get lower and upper bound
        img->GetMinMax(&min_val, &max_val);
        
        n   = img->GetNumberOfVoxels();
        ptr = img->GetPointerToVoxels();
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
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, 5), *this );
            //init.terminate();
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
        
        // execute
        void operator() () const {
            //task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slice.GetX()), *this );
            //init.terminate();
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
    
} // namespace mirtk
