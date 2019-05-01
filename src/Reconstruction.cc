/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2008-2017 Imperial College London
 * Copyright 2018-2019 King's College London
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

#include "mirtk/Reconstruction.h"

using namespace std;

namespace mirtk {
    
    //-------------------------------------------------------------------
    
    void bbox( RealImage &stack, RigidTransformation &transformation, double &min_x, double &min_y, double &min_z, double &max_x, double &max_y, double &max_z ) {
        
        //cout << "bbox" << endl;
        
        min_x = DBL_MAX;
        min_y = DBL_MAX;
        min_z = DBL_MAX;
        max_x = -DBL_MAX;
        max_y = -DBL_MAX;
        max_z = -DBL_MAX;
        double x,y,z;
        // WARNING: do not search to increment by stack.GetZ()-1,
        // otherwise you would end up with a 0 increment for slices...
        for ( int i = 0; i <= stack.GetX(); i += stack.GetX() )
            for ( int j = 0; j <= stack.GetY(); j += stack.GetY() )
                for ( int k = 0; k <= stack.GetZ(); k += stack.GetZ() ) {
                    x = i;
                    y = j;
                    z = k;
                    //        cout << "line1 "
                    // << x << " "
                    // << y << " "
                    // << z << endl;
                    stack.ImageToWorld( x, y, z );
                    // FIXME!!!
                    transformation.Transform( x, y, z );
                    //transformation.Inverse( x, y, z );
                    //        cout << "line2 "
                    // << x << " "
                    // << y << " "
                    // << z << endl;
                    if ( x < min_x )
                        min_x = x;
                    if ( y < min_y )
                        min_y = y;
                    if ( z < min_z )
                        min_z = z;
                    if ( x > max_x )
                        max_x = x;
                    if ( y > max_y )
                        max_y = y;
                    if ( z > max_z )
                        max_z = z;
                }
        // cout << "bbox"
        //      << min_x << " "
        //      << min_y << " "
        //      << min_z << " "
        //      << max_x << " "
        //      << max_y << " "
        //      << max_z << endl;
    }
    
    void bboxCrop( RealImage &image ) {
        int min_x, min_y, min_z, max_x, max_y, max_z;
        min_x = image.GetX()-1;
        min_y = image.GetY()-1;
        min_z = image.GetZ()-1;
        max_x = 0;
        max_y = 0;
        max_z = 0;
        for ( int i = 0; i < image.GetX(); i++ )
            for ( int j = 0; j < image.GetY(); j++ )
                for ( int k = 0; k < image.GetZ(); k++ ) {
                    if ( image.Get(i, j, k) > 0 ) {
                        if ( i < min_x )
                            min_x = i;
                        if ( j < min_y )
                            min_y = j;
                        if ( k < min_z )
                            min_z = k;
                        if ( i > max_x )
                            max_x = i;
                        if ( j > max_y )
                            max_y = j;
                        if ( k > max_z )
                            max_z = k;
                    }
                }
        
        //Cut region of interest
        image = image.GetRegion( min_x, min_y, min_z,
                                max_x, max_y, max_z );
    }
    
    
    
    
    
    //-------------------------------------------------------------------
    
    void centroid(RealImage &image, double &x, double &y, double &z) {
        double sum_x = 0;
        double sum_y = 0;
        double sum_z = 0;
        double norm = 0;
        double v;
        for (int i = 0; i < image.GetX(); i++)
            for (int j = 0; j < image.GetY(); j++)
                for (int k = 0; k < image.GetZ(); k++) {
                    v = image.Get(i, j, k);
                    if (v <= 0)
                        continue;
                    sum_x += v*i;
                    sum_y += v*j;
                    sum_z += v*k;
                    norm += v;
                }
        
        x = sum_x / norm;
        y = sum_y / norm;
        z = sum_z / norm;
        
        image.ImageToWorld(x, y, z);
        
    }
    
    //-------------------------------------------------------------------
    
    Reconstruction::Reconstruction()
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
        _recon_type = _3D;
        
        _ffd = false;
        _blurring = false;
        
        _nmi_bins = -1; // 16
        
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
    
    Reconstruction::~Reconstruction() { }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::CenterStacks( Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber )
    {
        double x0, y0, z0;
        double x, y, z;
        int tx, ty, tz;
        RealImage mask;
        mask = stacks[templateNumber];
        
        
        for (tx=0; tx<mask.GetX(); tx++) {
            for (ty=0; ty<mask.GetY(); ty++) {
                for (tz=0; tz<mask.GetZ(); tz++) {
                    
                    if (mask(tx,ty,tz) < 0)
                        mask(tx,ty,tz) = 0;
                    
                }
            }
        }
        
        centroid(mask, x0, y0, z0);
        
        Matrix m1, m2;
        for (int i = 0; i < stacks.size(); i++) {
            if (i == templateNumber)
                continue;
            
            mask = stacks[i];
            
            for (tx=0; tx<mask.GetX(); tx++) {
                for (ty=0; ty<mask.GetY(); ty++) {
                    for (tz=0; tz<mask.GetZ(); tz++) {
                        
                        if (mask(tx,ty,tz) < 0)
                            mask(tx,ty,tz) = 0;
                        
                    }
                }
            }
            
            centroid(mask, x, y, z);
            
            RigidTransformation translation;
            translation.PutTranslationX(x0 - x);
            translation.PutTranslationY(y0 - y);
            translation.PutTranslationZ(z0 - z);
            
            m1 = stack_transformations[i].GetMatrix();
            m2 = translation.GetMatrix();
            stack_transformations[i].PutMatrix(m2 * m1);
        }
        
    }
    
    //-------------------------------------------------------------------
    
    class ParallelAverage {
        Reconstruction* reconstructor;
        Array<RealImage> &stacks;
        Array<RigidTransformation> &stack_transformations;
        
        /// Padding value in target (voxels in the target image with this
        /// value will be ignored)
        double targetPadding;
        
        /// Padding value in source (voxels outside the source image will
        /// be set to this value)
        double sourcePadding;
        
        double background;
        
        // Volumetric registrations are stack-to-template while slice-to-volume
        // registrations are actually performed as volume-to-slice
        // (reasons: technicalities of implementation)
        // so transformations need to be inverted beforehand.
        
        bool linear;
        
    public:
        RealImage average;
        RealImage weights;
        
        void operator()(const blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                
                GenericLinearInterpolateImageFunction<RealImage> interpolator;
                
                RealImage s = stacks[i];
                RigidTransformation t = stack_transformations[i];
                RealImage image;
                image.Initialize(reconstructor->_reconstructed.GetImageAttributes());
                image = image;
                
                int x, y, z;
                for (x=0; x<image.GetX(); x++) {
                    for (y=0; y<image.GetY(); y++) {
                        for (z=0; z<image.GetZ(); z++) {
                            image(x,y,z) = 0;
                        }
                    }
                }
                
                
                ImageTransformation imagetransformation;
                imagetransformation.Input(&s);
                imagetransformation.Transformation(&t);
                imagetransformation.Output(&image);
                imagetransformation.TargetPaddingValue(targetPadding);
                imagetransformation.SourcePaddingValue(sourcePadding);
                imagetransformation.Interpolator(&interpolator);
                imagetransformation.Run();
                
                RealPixel *pa = average.GetPointerToVoxels();
                RealPixel *pi = image.GetPointerToVoxels();
                RealPixel *pw = weights.GetPointerToVoxels();
                for (int p = 0; p < average.GetNumberOfVoxels(); p++) {
                    if (*pi != background) {
                        *pa += *pi;
                        *pw += 1;
                    }
                    pa++;
                    pi++;
                    pw++;
                }
            }
        }
        
        ParallelAverage(ParallelAverage& x, split) :
        reconstructor(x.reconstructor),
        stacks(x.stacks),
        stack_transformations(x.stack_transformations) {
            average.Initialize(reconstructor->_reconstructed.GetImageAttributes());
            average = 0;
            weights.Initialize(reconstructor->_reconstructed.GetImageAttributes());
            weights = 0;
            targetPadding = x.targetPadding;
            sourcePadding = x.sourcePadding;
            background = x.background;
            linear = x.linear;
        }
        
        void join(const ParallelAverage& y) {
            average += y.average;
            weights += y.weights;
        }
        
        ParallelAverage(Reconstruction *reconstructor,
                        Array<RealImage>& _stacks,
                        Array<RigidTransformation>& _stack_transformations,
                        double _targetPadding,
                        double _sourcePadding,
                        double _background,
                        bool _linear = false) :
        reconstructor(reconstructor),
        stacks(_stacks),
        stack_transformations(_stack_transformations) {
            average.Initialize(reconstructor->_reconstructed.GetImageAttributes());
            average = 0;
            weights.Initialize(reconstructor->_reconstructed.GetImageAttributes());
            weights = 0;
            targetPadding = _targetPadding;
            sourcePadding = _sourcePadding;
            background = _background;
            linear = _linear;
        }
        
        // execute
        
        void operator()() {
            //task_scheduler_init init(tbb_no_threads);
            parallel_reduce( blocked_range<size_t>(0, stacks.size()), *this );
            // init.terminate();
        }
    };
    
    
    //-------------------------------------------------------------------
    
    class ParallelSliceAverage{
        Reconstruction* reconstructor;
        Array<RealImage> &slices;
        Array<RigidTransformation> &slice_transformations;
        RealImage &average;
        RealImage &weights;
        
    public:
        
        void operator()( const blocked_range<size_t>& r ) const {
            for ( int k0 = r.begin(); k0 < r.end(); k0++) {
                for ( int j0 = 0; j0 < average.GetY(); j0++) {
                    for ( int i0 = 0; i0 < average.GetX(); i0++) {
                        double x0 = i0;
                        double y0 = j0;
                        double z0 = k0;
                        // Transform point into world coordinates
                        average.ImageToWorld(x0, y0, z0);
                        for (int inputIndex = 0; inputIndex < slices.size(); inputIndex++ ) {
                            double x = x0;
                            double y = y0;
                            double z = z0;
                            // Transform point
                            slice_transformations[inputIndex].Transform(x, y, z);
                            // Transform point into image coordinates
                            slices[inputIndex].WorldToImage(x, y, z);
                            int i = round(x);
                            int j = round(y);
                            int k = round(z);
                            // Check whether transformed point is in FOV of input
                            if ( (i >=  0) && (i < slices[inputIndex].GetX()) &&
                                (j >= 0) && (j <  slices[inputIndex].GetY()) &&
                                (k >= 0) && (k < slices[inputIndex].GetZ()) ) {
                                if (slices[inputIndex](i,j,k) > 0) {
                                    average.PutAsDouble( i0, j0, k0, 0, average( i0, j0, k0 ) + slices[inputIndex](i,j,k) );
                                    weights.PutAsDouble( i0, j0, k0, 0, weights( i0, j0, k0 ) + 1 );
                                }
                            }
                        }
                    }
                }
            }
        }
        
        ParallelSliceAverage( Reconstruction *reconstructor,
                             Array<RealImage>& _slices,
                             Array<RigidTransformation>& _slice_transformations,
                             RealImage &_average,
                             RealImage &_weights ) :
        reconstructor(reconstructor),
        slices(_slices),
        slice_transformations(_slice_transformations),
        average(_average),
        weights(_weights)
        {
            average.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            average = 0;
            weights.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            weights = 0;
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0,average.GetZ()), *this );
            //init.terminate();
        }
    };
    
    
    //-------------------------------------------------------------------
    
    RealImage Reconstruction::CreateAverage( Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations )
    {
        if (!_template_created) {
            cerr << "Please create the template before calculating the average of the stacks." << endl;
            exit(1);
        }
        
        InvertStackTransformations(stack_transformations);
        ParallelAverage parallelAverage( this,
                                        stacks,
                                        stack_transformations,
                                        -1,0,0, // target/source/background
                                        true );
        parallelAverage();
        RealImage average = parallelAverage.average;
        RealImage weights = parallelAverage.weights;
        
        average /= weights;
        InvertStackTransformations(stack_transformations);
        return average;
    }
    
    
    
    //-------------------------------------------------------------------
    
    
    double Reconstruction::CreateTemplate( RealImage stack, double resolution )
    {
        double dx, dy, dz, d;
        
        //Get image attributes - image size and voxel size
        ImageAttributes attr = stack.GetImageAttributes();
        
        //enlarge stack in z-direction in case top of the head is cut off
        attr._z += 2;
        
        //create enlarged image
        RealImage enlarged;
        enlarged.Initialize(attr);
        
        //determine resolution of volume to reconstruct
        if (resolution <= 0) {
            //resolution was not given by user
            // set it to min of res in x or y direction
            stack.GetPixelSize(&dx, &dy, &dz);
            if ((dx <= dy) && (dx <= dz))
                d = dx;
            else if (dy <= dz)
                d = dy;
            else
                d = dz;
        }
        else
            d = resolution;
        
        cout << "Constructing volume with isotropic voxel size " << d << endl;
        
        //resample "enlarged" to resolution "d"
        InterpolationMode interpolation = Interpolation_Linear;
        
        UniquePtr<InterpolateImageFunction> interpolator;
        interpolator.reset(InterpolateImageFunction::New(interpolation));
        
        Resampling<RealPixel> resampler(d, d, d);
        
        enlarged = stack;
        
        resampler.Input(&enlarged);
        resampler.Output(&enlarged);
        resampler.Interpolator(interpolator.get());
        resampler.Run();
        
        //initialize recontructed volume
        _reconstructed = enlarged;
        _template_created = true;
        
        // if (_debug)
        _reconstructed.Write("template.nii.gz");
        
        //return resulting resolution of the template image
        return d;
    }
    
    //-------------------------------------------------------------------
    
    double Reconstruction::CreateTemplateAniso(RealImage stack)
    {
        double dx, dy, dz, d;
        
        //Get image attributes - image size and voxel size
        ImageAttributes attr = stack.GetImageAttributes();
        
        //enlarge stack in z-direction in case top of the head is cut off
        attr._t = 1;
        
        //create enlarged image
        RealImage enlarged(attr);
        
        cout << "Constructing volume with anisotropic voxel size " << attr._x << " " <<attr._y<<" "<<attr._z << endl;
        
        //initialize recontructed volume
        _reconstructed = enlarged;
        _template_created = true;
        
        //return resulting resolution of the template image
        return attr._x;
    }
    
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SetTemplate( RealImage templateImage )
    {
        RealImage t2template = _reconstructed;
        
        int x, y, z;
        for (x=0; x<t2template.GetX(); x++) {
            for (y=0; y<t2template.GetY(); y++) {
                for (z=0; z<t2template.GetZ(); z++) {
                    
                    t2template(x,y,z) = 0;
                    
                }
            }
        }
        
        RigidTransformation tr;
        
        ImageTransformation imagetransformation;
        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        
        imagetransformation.Input(&templateImage);
        imagetransformation.Transformation(&tr);
        imagetransformation.Output(&t2template);
        //target contains zeros, need padding -1
        imagetransformation.TargetPaddingValue(-1);
        //need to fill voxels in target where there is no info from source with zeroes
        imagetransformation.SourcePaddingValue(0);
        imagetransformation.Interpolator(&interpolator);
        imagetransformation.Run();
        
        _reconstructed = t2template;
    }
    
    
    //-------------------------------------------------------------------
    
    RealImage Reconstruction::CreateMask( RealImage image )
    {
        //binarize mask
        RealPixel* ptr = image.GetPointerToVoxels();
        for (int i = 0; i < image.GetNumberOfVoxels(); i++) {
            
            if (*ptr > 0.5)
                *ptr = 1;
            else
                *ptr = 0;
            ptr++;
        }
        return image;
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::CreateMaskFromBlackBackground( Array<RealImage> stacks, Array<RigidTransformation> stack_transformations, double smooth_mask )
    {
        
        //Create average of the stack using currect stack transformations
        GreyImage average = CreateAverage(stacks, stack_transformations);
        
        GreyPixel* ptr = average.GetPointerToVoxels();
        for (int i = 0; i < average.GetNumberOfVoxels(); i++) {
            if (*ptr < 0)
                *ptr = 0;
            ptr++;
        }
        
        //Create mask of the average from the black background
        MeanShift msh(average, 0, 256);
        msh.GenerateDensity();
        msh.SetTreshold();
        msh.RemoveBackground();
        GreyImage mask = msh.ReturnMask();
        
        //Calculate LCC of the mask to remove disconnected structures
        MeanShift msh2(mask, 0, 256);
        msh2.SetOutput(&mask);
        msh2.Lcc(1);
        RealImage m = mask;
        SetMask(&m, smooth_mask);
        
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SetMask(RealImage *mask, double sigma, double threshold)
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
            _mask.Write("mask.nii.gz");
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::TransformMask(RealImage& image, RealImage& mask, RigidTransformation& transformation)
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
    
    void Reconstruction::ResetOrigin(GreyImage &image, RigidTransformation& transformation)
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
    
    void Reconstruction::ResetOrigin(RealImage &image, RigidTransformation& transformation)
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
    
    class ParallelStackRegistrations {
        Reconstruction *reconstructor;
        Array<RealImage>& stacks;
        Array<RigidTransformation>& stack_transformations;
        int templateNumber;
        RealImage& target;
        RigidTransformation& offset;
        
    public:
        ParallelStackRegistrations( Reconstruction *_reconstructor,
                                   Array<RealImage>& _stacks,
                                   Array<RigidTransformation>& _stack_transformations,
                                   int _templateNumber,
                                   RealImage& _target,
                                   RigidTransformation& _offset ) :
        reconstructor(_reconstructor),
        stacks(_stacks),
        stack_transformations(_stack_transformations),
        target(_target),
        offset(_offset) {
            templateNumber = _templateNumber;
        }
        
        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t i = r.begin(); i != r.end(); ++i ) {
                
                
                char buffer[256];
                
                //do not perform registration for template
                if (i == templateNumber)
                    continue;
                
                RealImage r_source = stacks[i];
                RealImage r_target = target;
                
                //                r_target = stacks[templateNumber];
                //                r_target.PutOrigin(0, 0, 0);
                
                RealImage tmp_stack;
                
                ResamplingWithPadding<RealPixel> resampling2(1.25, 1.25, 1.25,-1);
                GenericLinearInterpolateImageFunction<RealImage> interpolator2;
                resampling2.Interpolator(&interpolator2);
                
                tmp_stack = r_source;
                resampling2.Input(&tmp_stack);
                resampling2.Output(&r_source);
                resampling2.Run();
                
                // GenericNearestNeighborInterpolateImageFunction<RealImage> interpolator3;
                // resampling2.Interpolator(&interpolator3);
                
                tmp_stack = r_target;
                resampling2.Input(&tmp_stack);
                resampling2.Output(&r_target);
                resampling2.Run();
                
                // GaussianBlurring<RealPixel> gb2(0.7);
                // gb2.Input(&r_source);
                // gb2.Output(&r_source);
                // gb2.Run();
                
                Matrix mo = offset.GetMatrix();
                Matrix m = stack_transformations[i].GetMatrix();
                m=m*mo;
                stack_transformations[i].PutMatrix(m);
                
                ParameterList params;
                Insert(params, "Transformation model", "Rigid");
                Insert(params, "Image (dis-)similarity measure", "NMI");
                if (reconstructor->_nmi_bins>0)
                    Insert(params, "No. of bins", reconstructor->_nmi_bins);
                Insert(params, "Image interpolation mode", "Linear");
                Insert(params, "Background value", 0);
                Insert(params, "Optimisation method", "GradientDescent");
                
                //     Insert(params, "Image (dis-)similarity measure", "NCC");
                //     // Insert(params_ncc, "Image interpolation mode", "Linear");
                //    string type = "box";
                //    string units = "vox";
                //    double width = 0;
                //    Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);
                
                
                GenericRegistrationFilter registration;
                registration.Parameter(params);
                registration.Input(&r_target, &r_source);
                
                RigidTransformation dofin = stack_transformations[i];
                //                dofin.Invert();
                
                Transformation *dofout = nullptr;
                registration.Output(&dofout);
                
                registration.InitialGuess(&dofin);
                
                registration.GuessParameter();
                registration.Run();
                
                RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*> (dofout);
                stack_transformations[i] = *rigidTransf;
                
                
                
                
                
                
                //               ParameterList params_ncc;
                //                 Insert(params_ncc, "Optimisation method", "GradientDescent");
                //               Insert(params_ncc, "Transformation model", "Rigid");
                //               Insert(params_ncc, "Background value", 0);
                //               Insert(params_ncc, "Image (dis-)similarity measure", "NCC");
                //                Insert(params_ncc, "Image interpolation mode", "Linear");
                //               string type = "sigma";
                //               string units = "mm";
                //               double width = 5;
                //               Insert(params_ncc, string("Local window size [") + type + string("]"), ToString(width) + units);
                
                GenericRegistrationFilter rigidregistration_ncc;
                rigidregistration_ncc.Parameter(params);
                rigidregistration_ncc.Input(&r_target, &r_source);
                
                
                dofout = nullptr;
                rigidregistration_ncc.Output(&dofout);
                rigidregistration_ncc.InitialGuess(&stack_transformations[i]);
                rigidregistration_ncc.GuessParameter();
                rigidregistration_ncc.Run();
                
                RigidTransformation *rigidTransf_ncc = dynamic_cast<RigidTransformation*> (dofout);
                stack_transformations[i] = *rigidTransf_ncc;
                
                
                
                mo.Invert();
                m = stack_transformations[i].GetMatrix();
                m=m*mo;
                stack_transformations[i].PutMatrix(m);
                
                
                
                //save volumetric registrations
                if (reconstructor->_debug) {
                    //buffer to create the name
                    
                    sprintf(buffer, "stack-transformation%i.dof", i);
                    stack_transformations[i].Write(buffer);
                    sprintf(buffer, "stack%i.nii.gz", i);
                    stacks[i].Write(buffer);
                }
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, stacks.size() ), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    
    //    void Reconstruction::SequentialRegistrations(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber)
    //    {
    //        char buffer[256];
    //
    //    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::StackRegistrations(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, int templateNumber)
    {
        // if (_debug)
        // cout << "StackRegistrations" << endl;
        
        InvertStackTransformations(stack_transformations);
        
        
        RealImage target;
        target = stacks[templateNumber];
        RealImage m_tmp = _mask;
        RigidTransformation *rigidTransf_m = new RigidTransformation;
        TransformMask(target, m_tmp, *rigidTransf_m);
        target = target * m_tmp;
        target.Write("masked.nii.gz");
        
        
        if(_debug)
        {
            target.Write("target.nii.gz");
            stacks[0].Write("stack0.nii.gz");
        }
        
        RigidTransformation offset;
        ResetOrigin(target, offset);
        
        //register all stacks to the target
        ParallelStackRegistrations registration( this,
                                                stacks,
                                                stack_transformations,
                                                templateNumber,
                                                target,
                                                offset );
        registration();
        
        
        InvertStackTransformations(stack_transformations);
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::RestoreSliceIntensities()
    {
        if (_debug)
            cout << "Restoring the intensities of the slices. "<<endl;
        
        unsigned int inputIndex;
        int i;
        double factor;
        RealPixel *p;
        
        for (inputIndex=0;inputIndex<_slices.size();inputIndex++) {
            //calculate scaling factor
            factor = _stack_factor[_stack_index[inputIndex]];//_average_value;
            
            // read the pointer to current slice
            p = _slices[inputIndex].GetPointerToVoxels();
            for(i=0;i<_slices[inputIndex].GetNumberOfVoxels();i++) {
                if(*p>0) *p = *p / factor;
                p++;
            }
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::ScaleVolume()
    {
        // if (_debug)
        // cout << "Scaling volume: ";
        
        unsigned int inputIndex;
        int i, j;
        double scalenum = 0, scaleden = 0;
        
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            // alias for the current slice
            RealImage& slice = _slices[inputIndex];
            
            //alias for the current weight image
            RealImage& w = _weights[inputIndex];
            
            // alias for the current simulated slice
            RealImage& sim = _simulated_slices[inputIndex];
            
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {
                        //scale - intensity matching
                        if ( _simulated_weights[inputIndex](i,j,0) > 0.99 ) {
                            scalenum += w(i, j, 0) * _slice_weight[inputIndex] * slice(i, j, 0) * sim(i, j, 0);
                            scaleden += w(i, j, 0) * _slice_weight[inputIndex] * sim(i, j, 0) * sim(i, j, 0);
                        }
                    }
        } //end of loop for a slice inputIndex
        
        //calculate scale for the volume
        double scale = scalenum / scaleden;
        
        if(_debug)
            cout<<" scale = "<<scale;
        
        RealPixel *ptr = _reconstructed.GetPointerToVoxels();
        for(i=0;i<_reconstructed.GetNumberOfVoxels();i++) {
            if(*ptr>0) *ptr = *ptr * scale;
            ptr++;
        }
        cout<<endl;
    }
    
    //-------------------------------------------------------------------
    
    class ParallelSimulateSlices {
        Reconstruction *reconstructor;
        
    public:
        ParallelSimulateSlices( Reconstruction *_reconstructor ) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                //Calculate simulated slice
                reconstructor->_simulated_slices[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
                reconstructor->_simulated_slices[inputIndex] = 0;
                
                reconstructor->_simulated_weights[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
                reconstructor->_simulated_weights[inputIndex] = 0;
                
                reconstructor->_simulated_inside[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
                reconstructor->_simulated_inside[inputIndex] = 0;
                
                reconstructor->_slice_inside[inputIndex] = false;
                
                POINT3D p;
                for ( unsigned int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++ )
                    for ( unsigned int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++ )
                        if ( reconstructor->_slices[inputIndex](i, j, 0) != -1 ) {
                            double weight = 0;
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for ( unsigned int k = 0; k < n; k++ ) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                reconstructor->_simulated_slices[inputIndex](i, j, 0) += p.value * reconstructor->_reconstructed(p.x, p.y, p.z);
                                weight += p.value;
                                if (reconstructor->_mask(p.x, p.y, p.z) == 1) {
                                    reconstructor->_simulated_inside[inputIndex](i, j, 0) = 1;
                                    reconstructor->_slice_inside[inputIndex] = true;
                                }
                            }
                            if( weight > 0 ) {
                                reconstructor->_simulated_slices[inputIndex](i,j,0) /= weight;
                                reconstructor->_simulated_weights[inputIndex](i,j,0) = weight;
                            }
                        }
                
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SimulateSlices()
    {
        // if (_debug)
        // cout<<"Simulating slices."<<endl;
        
        ParallelSimulateSlices parallelSimulateSlices( this );
        parallelSimulateSlices();
        
        // if (_debug)
        // cout<<"done."<<endl;
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SimulateStacks( Array<RealImage>& stacks )
    {
        // if (_debug)
        // cout<<"Simulating stacks."<<endl;
        
        unsigned int inputIndex;
        int i, j, k, n;
        RealImage sim;
        POINT3D p;
        double weight;
        
        int z, current_stack;
        z=-1;//this is the z coordinate of the stack
        current_stack=-1; //we need to know when to start a new stack
        
        
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            
            // read the current slice
            RealImage& slice = _slices[inputIndex];
            
            //Calculate simulated slice
            sim.Initialize( slice.GetImageAttributes() );
            sim = 0;
            
            //do not simulate excluded slice
            if(_slice_weight[inputIndex]>0.5)
            {
                for (i = 0; i < slice.GetX(); i++)
                    for (j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            weight=0;
                            n = _volcoeffs[inputIndex][i][j].size();
                            for (k = 0; k < n; k++) {
                                p = _volcoeffs[inputIndex][i][j][k];
                                sim(i, j, 0) += p.value * _reconstructed(p.x, p.y, p.z);
                                weight += p.value;
                            }
                            if(weight>0.98)
                                sim(i,j,0)/=weight;
                            else
                                sim(i,j,0)=0;
                        }
            }
            
            if (_stack_index[inputIndex]==current_stack)
                z++;
            else {
                current_stack=_stack_index[inputIndex];
                z=0;
            }
            
            for(i=0;i<sim.GetX();i++)
                for(j=0;j<sim.GetY();j++) {
                    stacks[_stack_index[inputIndex]](i,j,z)=sim(i,j,0);
                }
            //end of loop for a slice inputIndex
        }
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::MatchStackIntensities( Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, double averageValue, bool together )
    {
        // if (_debug)
        // cout << "Matching intensities of stacks. ";
        
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
    
    
    void Reconstruction::MaskStacks(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations)
    {
        cout << "Masking stacks ... ";
        
        double x, y, z;
        int i, j, k;
        
        //Check whether we have a mask
        if (!_have_mask) {
            cout << "Could not mask slices because no mask has been set." << endl;
            return;
        }
        
        //mask slices
        for (int unsigned inputIndex = 0; inputIndex < stacks.size(); inputIndex++) {
            
            RealImage& stack = stacks[inputIndex];
            for (i = 0; i < stack.GetX(); i++)
                for (j = 0; j < stack.GetY(); j++) {
                    for (k = 0; k < stack.GetZ(); k++) {
                        //if the value is smaller than 1 assume it is padding
                        if (stack(i,j,k) < 0.01)
                            stack(i,j,k) = -1;
                        //image coordinates of a slice voxel
                        x = i;
                        y = j;
                        z = k;
                        //change to world coordinates in slice space
                        stack.ImageToWorld(x, y, z);
                        //world coordinates in volume space
                        stack_transformations[inputIndex].Transform(x, y, z);
                        //image coordinates in volume space
                        _mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);
                        //if the voxel is outside mask ROI set it to -1 (padding value)
                        if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0)
                            && (z < _mask.GetZ())) {
                            if (_mask(x, y, z) == 0)
                                stack(i, j, k) = -1;
                        }
                        else
                            stack(i, j, k) = -1;
                    }
                }
            //remember masked slice
            //_slices[inputIndex] = slice;
        }
        cout << "done." << endl;
    }
    
    
    
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::MatchStackIntensitiesWithMasking( Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations, double averageValue, bool together )
    {
        // if (_debug)
        // cout << "Matching intensities of stacks. ";
        
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
            m=stacks[ind];
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
                        if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0)
                            && (z < _mask.GetZ()))
                        {
                            if (_mask(x, y, z) == 1)
                            {
                                m(i,j,k)=1;
                                sum += stacks[ind](i, j, k);
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
    
    void Reconstruction::CreateSlicesAndTransformations( Array<RealImage> &stacks, Array<RigidTransformation> &stack_transformations, Array<double> &thickness, const Array<RealImage> &probability_maps )
    {
        // if (_debug)
        // cout << "CreateSlicesAndTransformations" << endl;
        
        //for each stack
        for (unsigned int i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            ImageAttributes attr = stacks[i].GetImageAttributes();
            
            //attr._z is number of slices in the stack
            for (int j = 0; j < attr._z; j++) {
                
                bool excluded = false;
                
                for (int fe=0; fe<_excluded_entirely.size(); fe++) {
                    
                    if (j == _excluded_entirely[fe])
                        excluded = true;
                    
                }
                
                if (!excluded) {
                    
                    
                    //create slice by selecting the appropreate region of the stack
                    RealImage slice = stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                    //set correct voxel size in the stack. Z size is equal to slice thickness.
                    slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                    //remember the slice
                    
                    
                    if (_blurring) {
                        
                        GaussianBlurring<RealPixel> gbt(0.75*slice.GetXSize());
                        gbt.Input(&slice);
                        gbt.Output(&slice);
                        gbt.Run();
                        
                        
                    }
                    
                    
                    _slices.push_back(slice);
                    _simulated_slices.push_back(slice);
                    _simulated_weights.push_back(slice);
                    _simulated_inside.push_back(slice);
                    //remeber stack index for this slice
                    _stack_index.push_back(i);
                    //initialize slice transformation with the stack transformation
                    _transformations.push_back(stack_transformations[i]);
                    
                    if (_ffd) {
                        
                        MultiLevelFreeFormTransformation *mffd = new MultiLevelFreeFormTransformation();
                        _mffd_transformations.push_back(*mffd);
                        
                    }
                    
                    if ( probability_maps.size() > 0 ) {
                        RealImage proba = probability_maps[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                        proba.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                        _probability_maps.push_back(proba);
                    }
                    
                }
                
            }
            
            
        }
        cout << "Number of slices: " << _slices.size() << endl;
        
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::ResetSlices( Array<RealImage>& stacks, Array<double>& thickness )
    {
        if (_debug)
            cout << "ResetSlices" << endl;
        
        _slices.clear();
        
        //for each stack
        for (unsigned int i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            ImageAttributes attr = stacks[i].GetImageAttributes();
            
            //attr._z is number of slices in the stack
            for (int j = 0; j < attr._z; j++) {
                //create slice by selecting the appropreate region of the stack
                RealImage slice = stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                //set correct voxel size in the stack. Z size is equal to slice thickness.
                slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                //remember the slice
                _slices.push_back(slice);
            }
        }
        cout << "Number of slices: " << _slices.size() << endl;
        
        for ( int i = 0; i < _slices.size(); i++ ) {
            _bias[i].Initialize( _slices[i].GetImageAttributes() );
            _weights[i].Initialize( _slices[i].GetImageAttributes() );
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SetSlicesAndTransformations( Array<RealImage>& slices, Array<RigidTransformation>& slice_transformations, Array<int>& stack_ids, Array<double>& thickness )
    {
        _slices.clear();
        _stack_index.clear();
        _transformations.clear();
        
        //for each slice
        for (unsigned int i = 0; i < slices.size(); i++) {
            //get slice
            RealImage slice = slices[i];
            std::cout << "setting slice " << i << "\n";
            slice.Print();
            //set correct voxel size in the stack. Z size is equal to slice thickness.
            slice.PutPixelSize(slice.GetXSize(), slice.GetYSize(), thickness[i]);
            //remember the slice
            _slices.push_back(slice);
            //remember stack index for this slice
            _stack_index.push_back(stack_ids[i]);
            //get slice transformation
            _transformations.push_back(slice_transformations[i]);
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::UpdateSlices(Array<RealImage>& stacks, Array<double>& thickness)
    {
        _slices.clear();
        //for each stack
        for (unsigned int i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            ImageAttributes attr = stacks[i].GetImageAttributes();
            
            //attr._z is number of slices in the stack
            for (int j = 0; j < attr._z; j++) {
                //create slice by selecting the appropreate region of the stack
                RealImage slice = stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                //set correct voxel size in the stack. Z size is equal to slice thickness.
                slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                //remember the slice
                _slices.push_back(slice);
            }
        }
        cout << "Number of slices: " << _slices.size() << endl;
        
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::MaskSlices()
    {
        cout << "Masking slices ... ";
        
        double x, y, z;
        int i, j;
        
        //Check whether we have a mask
        if (!_have_mask) {
            cout << "Could not mask slices because no mask has been set." << endl;
            return;
        }
        
        //mask slices
        for (int unsigned inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            RealImage& slice = _slices[inputIndex];
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++) {
                    //if the value is smaller than 1 assume it is padding
                    if (slice(i,j,0) < 0.01)
                        slice(i,j,0) = -1;
                    //image coordinates of a slice voxel
                    x = i;
                    y = j;
                    z = 0;
                    //change to world coordinates in slice space
                    slice.ImageToWorld(x, y, z);
                    //world coordinates in volume space
                    
                    if (!_ffd)
                        _transformations[inputIndex].Transform(x, y, z);
                    else
                        _mffd_transformations[inputIndex].Transform(-1, 1, x, y, z);
                    
                    
                    //image coordinates in volume space
                    _mask.WorldToImage(x, y, z);
                    x = round(x);
                    y = round(y);
                    z = round(z);
                    //if the voxel is outside mask ROI set it to -1 (padding value)
                    if ((x >= 0) && (x < _mask.GetX()) && (y >= 0) && (y < _mask.GetY()) && (z >= 0)
                        && (z < _mask.GetZ())) {
                        if (_mask(x, y, z) == 0)
                            slice(i, j, 0) = -1;
                    }
                    else
                        slice(i, j, 0) = -1;
                }
            //remember masked slice
            //_slices[inputIndex] = slice;
        }
        cout << "done." << endl;
    }
    
    //-------------------------------------------------------------------
    
    class ParallelSliceToVolumeRegistration {
    public:
        Reconstruction *reconstructor;
        
        ParallelSliceToVolumeRegistration(Reconstruction *_reconstructor) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            
            ImageAttributes attr = reconstructor->_reconstructed.GetImageAttributes();
            // GreyImage source = ;
            
            // source.Write("s.nii.gz");
            
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                //                cout << inputIndex << " ";
                
                char buffer[256];
                
                GreyPixel smin, smax;
                GreyImage target;
                GreyImage slice, w, b;
                RealImage t;
                
                Reconstruction dummy_reconstruction;
                
                ResamplingWithPadding<RealPixel> resampling(attr._dx,attr._dx,attr._dx,-1);
                
                
                GenericLinearInterpolateImageFunction<RealImage> interpolator;
                
                t = reconstructor->_slices[inputIndex];
                
                
                resampling.Input(&reconstructor->_slices[inputIndex]);
                resampling.Output(&t);
                resampling.Interpolator(&interpolator);
                resampling.Run();
                
                target=t;
                
                target.GetMinMax(&smin, &smax);
                
                
                if (smax > 1 && (smax - smin) > 1) {
                    
                    
                    ParameterList params;
                    Insert(params, "Transformation model", "Rigid");
                    // Insert(params, "Optimisation method", "GradientDescent");
                    Insert(params, "Image (dis-)similarity measure", "NMI");
                    if (reconstructor->_nmi_bins>0)
                        Insert(params, "No. of bins", reconstructor->_nmi_bins);
                    Insert(params, "Image interpolation mode", "Linear");
                    // Insert(params, "Background value", -1);
                    // Insert(params, "Background value for image 1", 0);
                    // Insert(params, "Background value for image 2", -1);
                    
                    GenericRegistrationFilter *registration = new GenericRegistrationFilter();
                    registration->Parameter(params);
                    
                    
                    //put origin to zero
                    RigidTransformation offset;
                    
                    dummy_reconstruction.ResetOrigin(target,offset);
                    Matrix mo = offset.GetMatrix();
                    Matrix m = reconstructor->_transformations[inputIndex].GetMatrix();
                    m=m*mo;
                    reconstructor->_transformations[inputIndex].PutMatrix(m);
                    
                    
                    registration->Input(&target, &reconstructor->_reconstructed);
                    Transformation *dofout = nullptr;
                    registration->Output(&dofout);
                    
                    
                    RigidTransformation tmp_dofin = reconstructor->_transformations[inputIndex];
                    
                    registration->InitialGuess(&tmp_dofin);
                    registration->GuessParameter();
                    
//                    cout << inputIndex << endl;
                    
                    registration->Run();
                    
                    // cout << inputIndex << " + " << endl;
                    
                    RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*> (dofout);
                    
                    reconstructor->_transformations[inputIndex] = *rigidTransf;
                    
                    //undo the offset
                    mo.Invert();
                    m = reconstructor->_transformations[inputIndex].GetMatrix();
                    m=m*mo;
                    reconstructor->_transformations[inputIndex].PutMatrix(m);
                    
                    delete registration;
                    
                }
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    
    class ParallelSliceToVolumeRegistrationFFD {
    public:
        Reconstruction *reconstructor;
        
        ParallelSliceToVolumeRegistrationFFD(Reconstruction *_reconstructor) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            
            ImageAttributes attr = reconstructor->_reconstructed.GetImageAttributes();
            // GreyImage source = ;
            
            // source.Write("s.nii.gz");
            
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                //                cout << inputIndex << " ";
                
                char buffer[256];
                
                GreyPixel smin, smax;
                GreyImage target;
                GreyImage slice, w, b;
                RealImage t;
                
                Reconstruction dummy_reconstruction;
                
                ResamplingWithPadding<RealPixel> resampling(attr._dx,attr._dx,attr._dx,-1);
                
                
                GenericLinearInterpolateImageFunction<RealImage> interpolator;
                
                t = reconstructor->_slices[inputIndex];
                
                
                resampling.Input(&reconstructor->_slices[inputIndex]);
                resampling.Output(&t);
                resampling.Interpolator(&interpolator);
                resampling.Run();
                
                target=t;
                
                target.GetMinMax(&smin, &smax);
                
                
                if (smax > 1 && (smax - smin) > 1) {
                    
                    
                    ParameterList params;
                    Insert(params, "Transformation model", "FFD");
                    // Insert(params, "Optimisation method", "GradientDescent");
                    Insert(params, "Image (dis-)similarity measure", "NMI");
                    if (reconstructor->_nmi_bins>0)
                        Insert(params, "No. of bins", reconstructor->_nmi_bins);
//                    Insert(params, "Image interpolation mode", "Linear");
                    // Insert(params, "Background value", -1);
                    // Insert(params, "Background value for image 1", 0);
                    // Insert(params, "Background value for image 2", -1);
//                    int cp_spacing = 5;
//                    Insert(params, "Control point spacing in X", cp_spacing);        // 10
//                    Insert(params, "Control point spacing in Y", cp_spacing);        // 10
//                    Insert(params, "Control point spacing in Z", cp_spacing);        // 10
//                    Insert(params, "Image (dis-)similarity measure", "NCC");
//                    string type = "sigma";
//                    string units = "mm";
//                    double width = 5;
//                    Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);

                    
                    GenericRegistrationFilter *registration = new GenericRegistrationFilter();
                    registration->Parameter(params);
                    
                    
//                    //put origin to zero
//                    RigidTransformation offset;
//
//                    dummy_reconstruction.ResetOrigin(target,offset);
//                    Matrix mo = offset.GetMatrix();
//                    Matrix m = reconstructor->_transformations[inputIndex].GetMatrix();
//                    m=m*mo;
//                    reconstructor->_transformations[inputIndex].PutMatrix(m);
                    
                    
                    registration->Input(&target, &reconstructor->_reconstructed);
                    Transformation *dofout = nullptr;
                    registration->Output(&dofout);
                    
                    
//                    RigidTransformation tmp_dofin = reconstructor->_transformations[inputIndex];
//                    registration->InitialGuess(&tmp_dofin);
                    
                    
                    registration->InitialGuess(&(reconstructor->_mffd_transformations[inputIndex]));
                    registration->GuessParameter();
                    
//                    cout << inputIndex << endl;
                    
                    registration->Run();
                    
                    // cout << inputIndex << " + " << endl;
                    
//                    RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*> (dofout);
                    
                    MultiLevelFreeFormTransformation *mffd_dofout = dynamic_cast<MultiLevelFreeFormTransformation*> (dofout);
                    reconstructor->_mffd_transformations[inputIndex] = *mffd_dofout;
                    
//                    reconstructor->_transformations[inputIndex] = *rigidTransf;
                    
//                    //undo the offset
//                    mo.Invert();
//                    m = reconstructor->_transformations[inputIndex].GetMatrix();
//                    m=m*mo;
//                    reconstructor->_transformations[inputIndex].PutMatrix(m);
                    
                    delete registration;
                    
                }
            }
        }
        
        // execute
        void operator() () const {
            task_scheduler_init init(8);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SliceToVolumeRegistration()
    {
        // if (_debug)
        //     cout << "SliceToVolumeRegistration" << endl;
        
        _reconstructed.Write("target.nii.gz");
        
        //        cout << "Slices : ";
        
        if (!_ffd) {
        
            ParallelSliceToVolumeRegistration registration(this);
            registration();
        }
        else {
            
            ParallelSliceToVolumeRegistrationFFD registration(this);
            registration();
        }
        
        //        cout << endl;
        
        
        
    }
    
    //-------------------------------------------------------------------
    
    
    void Reconstruction::NLMFiltering(Array<RealImage>& stacks)
    {
        
        char buffer[256];
        
        RealImage tmp_stack;
        
        
        
        
        NLDenoising *denoising = new NLDenoising;
        
        for (int i=0; i<stacks.size(); i++) {
            
            tmp_stack = stacks[i];
            tmp_stack = denoising->Run(tmp_stack, 3, 1);
            stacks[i] = tmp_stack;
            
            sprintf(buffer, "denoised-%i.nii.gz", i);
            stacks[i].Write(buffer);
            
        }
        
    }
    
    //-------------------------------------------------------------------
    
    class ParallelCoeffInit {
    public:
        Reconstruction *reconstructor;
        
        ParallelCoeffInit(Reconstruction *_reconstructor) :
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
//                cout << inputIndex << " ";
                
                //read the slice
                RealImage& slice = reconstructor->_slices[inputIndex];
                
                //prepare structures for storage
                POINT3D p;
                VOXELCOEFFS empty;
                SLICECOEFFS slicecoeffs(slice.GetX(), Array < VOXELCOEFFS > (slice.GetY(), empty));
                
                //to check whether the slice has an overlap with mask ROI
                slice_inside = false;
                
                //PSF will be calculated in slice space in higher resolution
                
                //get slice voxel size to define PSF
                double dx, dy, dz;
                slice.GetPixelSize(&dx, &dy, &dz);
                
                //sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM = dz through-plane)
                double sigmax, sigmay, sigmaz;
                if(reconstructor->_recon_type == _3D)
                {
                    sigmax = 1.2 * dx / 2.3548;
                    sigmay = 1.2 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                }
                
                if(reconstructor->_recon_type == _1D)
                {
                    sigmax = 0.5 * dx / 2.3548;
                    sigmay = 0.5 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                }
                
                if(reconstructor->_recon_type == _interpolate)
                {
                    sigmax = 0.5 * dx / 2.3548;
                    sigmay = 0.5 * dx / 2.3548;
                    sigmaz = 0.5 * dx / 2.3548;
                }
                
                
                //calculate discretized PSF
                
                //isotropic voxel size of PSF - derived from resolution of reconstructed volume
                double size = res / reconstructor->_quality_factor;
                
                //number of voxels in each direction
                //the ROI is 2*voxel dimension
                
                int xDim = round(2 * dx / size);
                int yDim = round(2 * dy / size);
                int zDim = round(2 * dz / size);
                
                ///test to make dimension always odd
                xDim = xDim/2*2+1;
                yDim = yDim/2*2+1;
                zDim = zDim/2*2+1;
                ///end test
                
                
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
                //maximum dim of rotated kernel - the next higher odd integer plus two to accound for rounding error of tx,ty,tz.
                //Note conversion from PSF image coordinates to tPSF image coordinates *size/res
                int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2))
                * 2 + 1 + 2;
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
                for (i = 0; i < slice.GetX(); i++)
                    for (j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //calculate centrepoint of slice voxel in volume space (tx,ty,tz)
                            x = i;
                            y = j;
                            z = 0;
                            slice.ImageToWorld(x, y, z);
                            
                            
//                            reconstructor->_transformations[inputIndex].Transform(x, y, z);
                            
                            if(!reconstructor->_ffd)
                                reconstructor->_transformations[inputIndex].Transform(x, y, z);
                            else
                                reconstructor->_mffd_transformations[inputIndex].Transform(-1, 1, x, y, z);
                            
                            
                            
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
                                        slice.ImageToWorld(x, y, z);
                                        
                                        //x+=(vx-cx); y+=(vy-cy); z+=(vz-cz);
                                        //Transform to space of reconstructed volume
//                                        reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                        
                                        if (!reconstructor->_ffd)
                                            reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                        else
                                            reconstructor->_mffd_transformations[inputIndex].Transform(-1, 1, x, y, z);
                                        
                                        
                                        
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
                                                                if ((aa < 0) || (aa >= dim) || (bb < 0) || (bb >= dim) || (cc < 0)
                                                                    || (cc >= dim)) {
//                                                                    cerr << "Error while trying to populate tPSF. " << aa << " " << bb
//                                                                    << " " << cc << endl;
//                                                                    cerr << l << " " << m << " " << n << endl;
//                                                                    cerr << tx << " " << ty << " " << tz << endl;
//                                                                    cerr << centre << endl;
//                                                                    tPSF.Write("tPSF.nii.gz");
//                                                                    exit(1);
                                                                }
                                                                else
                                                                    //update transformed PSF
                                                                    tPSF(aa, bb, cc) += value;
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
                
                reconstructor->_volcoeffs[inputIndex] = slicecoeffs;
                reconstructor->_slice_inside[inputIndex] = slice_inside;
                
            }  //end of loop through the slices
            
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::CoeffInit()
    {
        // if (_debug)
        // cout << "CoeffInit" << endl;
        
        //clear slice-volume matrix from previous iteration
        _volcoeffs.clear();
        _volcoeffs.resize(_slices.size());
        
        
        //clear indicator of slice having and overlap with volumetric mask
        _slice_inside.clear();
        _slice_inside.resize(_slices.size());
        
        //        cout << "Initialising matrix coefficients...";
        ParallelCoeffInit coeffinit(this);
        coeffinit();
        //        cout << " ... done." << endl;
        
        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        _volume_weights.Initialize( _reconstructed.GetImageAttributes() );
        _volume_weights = 0;
        
        int inputIndex, i, j, n, k;
        POINT3D p;
        for ( inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            
            bool excluded = false;
            
            for (int fe=0; fe<_force_excluded.size(); fe++) {
                
                if (inputIndex == _force_excluded[fe])
                    excluded = true;
                
            }
            
            if (!excluded) {
                
                for ( i = 0; i < _slices[inputIndex].GetX(); i++)
                    for ( j = 0; j < _slices[inputIndex].GetY(); j++) {
                        n = _volcoeffs[inputIndex][i][j].size();
                        for (k = 0; k < n; k++) {
                            p = _volcoeffs[inputIndex][i][j][k];
                            _volume_weights(p.x, p.y, p.z) += p.value;
                        }
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
            cout<<"Average volume weight is "<<_average_volume_weight<<endl;
        }
        
    }  //end of CoeffInit()
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::GaussianReconstruction()
    {
        //        cout << "Gaussian reconstruction ... ";
        unsigned int inputIndex;
        int i, j, k, n;
        RealImage slice;
        double scale;
        POINT3D p;
        Array<int> voxel_num;
        int slice_vox_num;
        
        //clear _reconstructed image
        _reconstructed = 0;
        
        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            //copy the current slice
            slice = _slices[inputIndex];
            //alias the current bias image
            RealImage& b = _bias[inputIndex];
            //read current scale factor
            scale = _scale[inputIndex];
            
            slice_vox_num=0;
            
            bool excluded = false;
            
            for (int fe=0; fe<_force_excluded.size(); fe++) {
                
                if (inputIndex == _force_excluded[fe])
                    excluded = true;
                
            }
            
            if (!excluded) {
                
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
                            }
                        }
                voxel_num.push_back(slice_vox_num);
                //end of loop for a slice inputIndex
            }
        }
        
        
        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxe
        _reconstructed /= _volume_weights;
        
        //        cout << "done." << endl;
        
        // if (_debug)
        _reconstructed.Write("init.nii.gz");
        
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
    
    
    class ParallelCoeffInitSF {
    public:
        
        Reconstruction *reconstructor;
        size_t _begin;
        size_t _end;
        
        ParallelCoeffInitSF(Reconstruction *_reconstructor, size_t begin, size_t end) :
        reconstructor(_reconstructor)
        {
            _begin = begin;
            _end = end;
        }
        
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
//                cout << inputIndex << " ";
//                cout.flush();
                //read the slice
                
                RealImage& slice = (reconstructor->_withMB == true) ? reconstructor->_slicesRwithMB[inputIndex] : reconstructor->_slices[inputIndex];
                
                //prepare structures for storage
                POINT3D p;
                VOXELCOEFFS empty;
                SLICECOEFFS slicecoeffs(slice.GetX(), Array < VOXELCOEFFS > (slice.GetY(), empty));
                
                //to check whether the slice has an overlap with mask ROI
                slice_inside = false;
                
                //PSF will be calculated in slice space in higher resolution
                
                //get slice voxel size to define PSF
                double dx, dy, dz;
                slice.GetPixelSize(&dx, &dy, &dz);
                
                //sigma of 3D Gaussian (sinc with FWHM=dx or dy in-plane, Gaussian with FWHM = dz through-plane)
                
                double sigmax, sigmay, sigmaz;
                if(reconstructor->_recon_type == _3D)
                {
                    sigmax = 1.2 * dx / 2.3548;
                    sigmay = 1.2 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                }
                
                if(reconstructor->_recon_type == _1D)
                {
                    sigmax = 0.5 * dx / 2.3548;
                    sigmay = 0.5 * dy / 2.3548;
                    sigmaz = dz / 2.3548;
                }
                
                if(reconstructor->_recon_type == _interpolate)
                {
                    sigmax = 0.5 * dx / 2.3548;
                    sigmay = 0.5 * dx / 2.3548;
                    sigmaz = 0.5 * dx / 2.3548;
                }
                /*
                 cout<<"Original sigma"<<sigmax<<" "<<sigmay<<" "<<sigmaz<<endl;
                 
                 //readjust for resolution of the volume
                 //double sigmax,sigmay,sigmaz;
                 double sigmamin = res/(3*2.3548);
                 
                 if((dx-res)>sigmamin)
                 sigmax = 1.2 * sqrt(dx*dx-res*res) / 2.3548;
                 else sigmax = sigmamin;
                 
                 if ((dy-res)>sigmamin)
                 sigmay = 1.2 * sqrt(dy*dy-res*res) / 2.3548;
                 else
                 sigmay=sigmamin;
                 if ((dz-1.2*res)>sigmamin)
                 sigmaz = sqrt(dz*dz-1.2*1.2*res*res) / 2.3548;
                 else sigmaz=sigmamin;
                 
                 cout<<"Adjusted sigma:"<<sigmax<<" "<<sigmay<<" "<<sigmaz<<endl;
                 */
                
                //calculate discretized PSF
                
                //isotropic voxel size of PSF - derived from resolution of reconstructed volume
                double size = res / reconstructor->_quality_factor;
                
                //number of voxels in each direction
                //the ROI is 2*voxel dimension
                
                int xDim = round(2 * dx / size);
                int yDim = round(2 * dy / size);
                int zDim = round(2 * dz / size);
                ///test to make dimension alwways odd
                xDim = xDim/2*2+1;
                yDim = yDim/2*2+1;
                zDim = zDim/2*2+1;
                ///end test
                
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
                //maximum dim of rotated kernel - the next higher odd integer plus two to accound for rounding error of tx,ty,tz.
                //Note conversion from PSF image coordinates to tPSF image coordinates *size/res
                int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2))
                * 2 + 1 + 2;
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
                for (i = 0; i < slice.GetX(); i++)
                    for (j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //calculate centrepoint of slice voxel in volume space (tx,ty,tz)
                            x = i;
                            y = j;
                            z = 0;
                            slice.ImageToWorld(x, y, z);
                            
                            if (reconstructor->_withMB)
                                reconstructor->_transformationsRwithMB[inputIndex].Transform(x, y, z);
                            else
                                reconstructor->_transformations[inputIndex].Transform(x, y, z);
                            
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
                                        slice.ImageToWorld(x, y, z);
                                        
                                        //x+=(vx-cx); y+=(vy-cy); z+=(vz-cz);
                                        //Transform to space of reconstructed volume
                                        if (reconstructor->_withMB)
                                            reconstructor->_transformationsRwithMB[inputIndex].Transform(x, y, z);
                                        else
                                            reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                        
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
                                                                if ((aa < 0) || (aa >= dim) || (bb < 0) || (bb >= dim) || (cc < 0)
                                                                    || (cc >= dim)) {
                                                                    cerr << "Error while trying to populate tPSF. " << aa << " " << bb
                                                                    << " " << cc << endl;
                                                                    cerr << l << " " << m << " " << n << endl;
                                                                    cerr << tx << " " << ty << " " << tz << endl;
                                                                    cerr << centre << endl;
                                                                    tPSF.Write("tPSF.nii.gz");
                                                                    exit(1);
                                                                }
                                                                else
                                                                    //update transformed PSF
                                                                    tPSF(aa, bb, cc) += value;
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
                
                reconstructor->_volcoeffsSF[inputIndex % (reconstructor->_slicePerDyn)] = slicecoeffs;
                reconstructor->_slice_insideSF[inputIndex % (reconstructor->_slicePerDyn)] = slice_inside;
                //cerr<<" Done "<<inputIndex % (reconstructor->_slicePerDyn)<<endl;
            }  //end of loop through the slices
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(_begin, _end),
                         *this );
            //init.terminate();
        }
    };
    
    //-------------------------------------------------------------------
    
    void Reconstruction::CoeffInitSF(int begin, int end)
    {
//        if (_debug)
//            cout << "CoeffInitSF" << endl;
        
        //clear slice-volume matrix from previous iteration
        _volcoeffsSF.clear();
        _volcoeffsSF.resize(_slicePerDyn);
        
        //clear indicator of slice having and overlap with volumetric mask
        _slice_insideSF.clear();
        _slice_insideSF.resize(_slicePerDyn);
        
//        cout << "Initialising matrix coefficients...";
//        cout.flush();
        
        ParallelCoeffInitSF coeffinit(this,begin,end);
        coeffinit();
        
        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        _volume_weightsSF.Initialize( _reconstructed.GetImageAttributes() );
        _volume_weightsSF = 0;
        
        int inputIndex, i, j, n, k;
        POINT3D p;
        if(_withMB)
            for ( inputIndex = begin; inputIndex < end; ++inputIndex) {
                for ( i = 0; i < _slicesRwithMB[inputIndex].GetX(); i++)
                    for ( j = 0; j < _slicesRwithMB[inputIndex].GetY(); j++) {
                        n = _volcoeffsSF[inputIndex % _slicePerDyn][i][j].size();
                        for (k = 0; k < n; k++) {
                            p = _volcoeffsSF[inputIndex % _slicePerDyn][i][j][k];
                            _volume_weightsSF(p.x, p.y, p.z) += p.value;
                        }
                    }
            }
        else {
            for ( inputIndex = begin; inputIndex < end; ++inputIndex) {
                for ( i = 0; i < _slices[inputIndex].GetX(); i++)
                    for ( j = 0; j < _slices[inputIndex].GetY(); j++) {
                        n = _volcoeffsSF[inputIndex % _slicePerDyn][i][j].size();
                        for (k = 0; k < n; k++) {
                            p = _volcoeffsSF[inputIndex % _slicePerDyn][i][j][k];
                            _volume_weightsSF(p.x, p.y, p.z) += p.value;
                        }
                    }
            }
        }
        
        if (_debug)
            _volume_weightsSF.Write("volume_weights.nii.gz");
        
        //find average volume weight to modify alpha parameters accordingly
        RealPixel *ptr = _volume_weightsSF.GetPointerToVoxels();
        RealPixel *pm = _mask.GetPointerToVoxels();
        double sum = 0;
        int num=0;
        for (int i=0;i<_volume_weightsSF.GetNumberOfVoxels();i++) {
            if (*pm==1) {
                sum+=*ptr;
                num++;
            }
            ptr++;
            pm++;
        }
        _average_volume_weightSF = sum/num;
        
        if(_debug) {
            cout<<"Average volume weight is "<<_average_volume_weightSF<<endl;
        }
        
    }  //end of CoeffInitSF()
    
    //-------------------------------------------------------------------
    
    void Reconstruction::GaussianReconstructionSF(Array<RealImage>& stacks)
    {
        Array<RigidTransformation> currentTransformations;
        Array<RealImage> currentSlices;
        Array<double> currentScales;
        Array<RealImage> currentBiases;
        
        RealImage interpolated;
        ImageAttributes attr, attr2;
        
        RealImage slice;
        double scale;
        int n;
        POINT3D p;
        Array<int> voxel_num;
        int slice_vox_num = 0;
        
        int counter = 0;
        interpolated  = _reconstructed;
        attr2 = interpolated.GetImageAttributes();
        
        // cleaning _interpolated
        for (int k = 0; k < attr2._z; k++) {
            for (int j = 0; j < attr2._y; j++) {
                for (int i = 0; i < attr2._x; i++) {
                    _reconstructed(i,j,k) = 0;
                }
            }
        }
        
        for (int dyn = 0; dyn < stacks.size(); dyn++)  {
            
            attr = stacks[dyn].GetImageAttributes();
            
            CoeffInitSF(counter,counter+attr._z);
            
            for (int s = 0; s < attr._z; s ++) {
                currentTransformations.push_back(_transformations[counter + s]);
                currentSlices.push_back(_slices[counter + s]);
                currentScales.push_back(_scale[counter + s]);
                currentBiases.push_back(_bias[counter + s]);
            }
            
            // cleaning interpolated
            for (int k = 0; k < attr2._z; k++) {
                for (int j = 0; j < attr2._y; j++) {
                    for (int i = 0; i < attr2._x; i++) {
                        interpolated(i,j,k) = 0;
                    }
                }
            }
            
            for (int s = 0; s < currentSlices.size(); s++) {
                
                //copy the current slice
                slice = currentSlices[s];
                //alias the current bias image
                RealImage& b = currentBiases[s];
                //read current scale factor
                scale = currentScales[s];
                
                slice_vox_num = 0;
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //biascorrect and scale the slice
                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                            
                            //number of volume voxels with non-zero coefficients
                            //for current slice voxel
                            n = _volcoeffsSF[s][i][j].size();
                            
                            //if given voxel is not present in reconstructed volume at all,
                            //pad it
                            
                            //if (n == 0)
                            //_slices[inputIndex].PutAsDouble(i, j, 0, -1);
                            //calculate num of vox in a slice that have overlap with roi
                            if (n>0)
                                slice_vox_num++;
                            
                            //add contribution of current slice voxel to all voxel volumes
                            //to which it contributes
                            for (int k = 0; k < n; k++) {
                                p = _volcoeffsSF[s][i][j][k];
                                interpolated(p.x, p.y, p.z) += p.value * slice(i, j, 0);
                            }
                        }
                voxel_num.push_back(slice_vox_num);
            }
            counter = counter + attr._z;
            currentSlices.clear();
            currentBiases.clear();
            currentTransformations.clear();
            currentScales.clear();
            interpolated /= _volume_weightsSF;
            _reconstructed += interpolated;
        }
        _reconstructed /= stacks.size();
        
        cout << "done." << endl;
        if (_debug)
            _reconstructed.Write("init.nii.gz");
        
        //now find slices with small overlap with ROI and exclude them.
        Array<int> voxel_num_tmp;
        for (int i=0;i<voxel_num.size();i++)
            voxel_num_tmp.push_back(voxel_num[i]);
        
        //find median
        sort(voxel_num_tmp.begin(),voxel_num_tmp.end());
        int median = voxel_num_tmp[round(voxel_num_tmp.size()*0.5)];
        
        //remember slices with small overlap with ROI
        _small_slices.clear();
        for (int i=0;i<voxel_num.size();i++)
            if (voxel_num[i]<0.1*median)
                _small_slices.push_back(i);
        
        if (_debug) {
            cout<<"Small slices:";
            for (int i=0;i<_small_slices.size();i++)
                cout<<" "<<_small_slices[i];
            cout<<endl;
        }
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::InitializeEM()
    {
        // if (_debug)
        // cout << "InitializeEM" << endl;
        
        _weights.clear();
        _bias.clear();
        _scale.clear();
        _slice_weight.clear();
        
        for (unsigned int i = 0; i < _slices.size(); i++) {
            //Create images for voxel weights and bias fields
            _weights.push_back(_slices[i]);
            _bias.push_back(_slices[i]);
            
            //Create and initialize scales
            _scale.push_back(1);
            
            //Create and initialize slice weights
            _slice_weight.push_back(1);
        }
        
        //Find the range of intensities
        _max_intensity = voxel_limits<RealPixel>::min();
        _min_intensity = voxel_limits<RealPixel>::max();
        for (unsigned int i = 0; i < _slices.size(); i++) {
            //to update minimum we need to exclude padding value
            RealPixel *ptr = _slices[i].GetPointerToVoxels();
            for (int ind = 0; ind < _slices[i].GetNumberOfVoxels(); ind++) {
                if (*ptr > 0) {
                    if (*ptr > _max_intensity)
                        _max_intensity = *ptr;
                    if (*ptr < _min_intensity)
                        _min_intensity = *ptr;
                }
                ptr++;
            }
        }
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::InitializeEMValues()
    {
        // if (_debug)
        // cout << "InitializeEMValues" << endl;
        
        for (unsigned int i = 0; i < _slices.size(); i++) {
            //Initialise voxel weights and bias values
            RealPixel *pw = _weights[i].GetPointerToVoxels();
            RealPixel *pb = _bias[i].GetPointerToVoxels();
            RealPixel *pi = _slices[i].GetPointerToVoxels();
            for (int j = 0; j < _weights[i].GetNumberOfVoxels(); j++) {
                if (*pi != -1) {
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
        }
        
        //Force exclusion of slices predefined by user
        for (unsigned int i = 0; i < _force_excluded.size(); i++){
            
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slices.size())
                _slice_weight[_force_excluded[i]] = 0;
        }
        
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::InitializeRobustStatistics()
    {
        // if (_debug)
        // cout << "InitializeRobustStatistics" << endl;
        
        //Initialise parameter of EM robust statistics
        int i, j;
        RealImage slice,sim;
        double sigma = 0;
        int num = 0;
        
        //for each slice
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            slice = _slices[inputIndex];
            
            //Voxel-wise sigma will be set to stdev of volumetric errors
            //For each slice voxel
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {
                        //calculate stev of the errors
                        if ( (_simulated_inside[inputIndex](i, j, 0)==1)
                            &&(_simulated_weights[inputIndex](i,j,0)>0.99) ) {
                            slice(i,j,0) -= _simulated_slices[inputIndex](i,j,0);
                            sigma += slice(i, j, 0) * slice(i, j, 0);
                            num++;
                        }
                    }
            
            //if slice does not have an overlap with ROI, set its weight to zero
            if (!_slice_inside[inputIndex])
                _slice_weight[inputIndex] = 0;
        }
        
        //Force exclusion of slices predefined by user
        for (unsigned int i = 0; i < _force_excluded.size(); i++) {
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slices.size())
                _slice_weight[_force_excluded[i]] = 0;
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
    
    class ParallelEStep {
        Reconstruction* reconstructor;
        Array<double> &slice_potential;
        
    public:
        
        void operator()( const blocked_range<size_t>& r ) const {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                //read current weight image
                reconstructor->_weights[inputIndex] = 0;
                
                //alias the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];
                
                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];
                
                double num = 0;
                //Calculate error, voxel weights, and slice potential
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                            
                            //number of volumetric voxels to which
                            // current slice voxel contributes
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            
                            // if n == 0, slice voxel has no overlap with volumetric ROI,
                            // do not process it
                            
                            if ( (n>0) &&
                                (reconstructor->_simulated_weights[inputIndex](i,j,0) > 0) ) {
                                slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                                
                                //calculate norm and voxel-wise weights
                                
                                //Gaussian distribution for inliers (likelihood)
                                double g = reconstructor->G(slice(i, j, 0), reconstructor->_sigma);
                                //Uniform distribution for outliers (likelihood)
                                double m = reconstructor->M(reconstructor->_m);
                                
                                //voxel_wise posterior
                                double weight = g * reconstructor->_mix / (g *reconstructor->_mix + m * (1 - reconstructor->_mix));
                                reconstructor->_weights[inputIndex].PutAsDouble(i, j, 0, weight);
                                
                                //calculate slice potentials
                                if(reconstructor->_simulated_weights[inputIndex](i,j,0)>0.99) {
                                    slice_potential[inputIndex] += (1 - weight) * (1 - weight);
                                    num++;
                                }
                            }
                            else
                                reconstructor->_weights[inputIndex].PutAsDouble(i, j, 0, 0);
                        }
                
                //evaluate slice potential
                if (num > 0)
                    slice_potential[inputIndex] = sqrt(slice_potential[inputIndex] / num);
                else
                    slice_potential[inputIndex] = -1; // slice has no unpadded voxels
            }
        }
        
        ParallelEStep( Reconstruction *reconstructor,
                      Array<double> &slice_potential ) :
        reconstructor(reconstructor), slice_potential(slice_potential)
        { }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::EStep()
    {
        //EStep performs calculation of voxel-wise and slice-wise posteriors (weights)
        // if (_debug)
        // cout << "EStep: " << endl;
        
        unsigned int inputIndex;
        RealImage slice, w, b, sim;
        int num = 0;
        Array<double> slice_potential(_slices.size(), 0);
        
        ParallelEStep parallelEStep( this, slice_potential );
        parallelEStep();
        
        //To force-exclude slices predefined by a user, set their potentials to -1
        for (unsigned int i = 0; i < _force_excluded.size(); i++) {
            
            if (_force_excluded[i] > 0 && _force_excluded[i] < _slices.size())
                slice_potential[_force_excluded[i]] = -1;
        }
        
        //exclude slices identified as having small overlap with ROI, set their potentials to -1
        for (unsigned int i = 0; i < _small_slices.size(); i++)
            slice_potential[_small_slices[i]] = -1;
        
        //these are unrealistic scales pointing at misregistration - exclude the corresponding slices
        for (inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
            if ((_scale[inputIndex]<0.2)||(_scale[inputIndex]>5)) {
                slice_potential[inputIndex] = -1;
            }
        
//        // exclude unrealistic transformations
//        if(_debug) {
//            cout<<endl<<"Slice potentials: ";
//            for (inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
//                cout<<slice_potential[inputIndex]<<" ";
//            cout<<endl;
//        }
//
        
        //Calulation of slice-wise robust statistics parameters.
        //This is theoretically M-step,
        //but we want to use latest estimate of slice potentials
        //to update the parameters
        
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
            cout << "Slice weights: ";
            for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                cout << _slice_weight[inputIndex] << " ";
            cout << endl;
        }
        
    }
    
    
    //-------------------------------------------------------------------
    
    class ParallelScale {
        Reconstruction *reconstructor;
        
    public:
        ParallelScale( Reconstruction *_reconstructor ) :
        reconstructor(_reconstructor)
        { }
        
        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                // alias the current slice
                RealImage& slice = reconstructor->_slices[inputIndex];
                
                //alias the current weight image
                RealImage& w = reconstructor->_weights[inputIndex];
                
                //alias the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];
                
                //initialise calculation of scale
                double scalenum = 0;
                double scaleden = 0;
                
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            if(reconstructor->_simulated_weights[inputIndex](i,j,0)>0.99) {
                                //scale - intensity matching
                                double eb = exp(-b(i, j, 0));
                                scalenum += w(i, j, 0) * slice(i, j, 0) * eb * reconstructor->_simulated_slices[inputIndex](i, j, 0);
                                scaleden += w(i, j, 0) * slice(i, j, 0) * eb * slice(i, j, 0) * eb;
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
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::Scale()
    {
        // if (_debug)
        // cout << "Scale" << endl;
        
        ParallelScale parallelScale( this );
        parallelScale();
        
        if (_debug) {
            cout << setprecision(3);
            cout << "Slice scale = ";
            for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex)
                cout << _scale[inputIndex] << " ";
            cout << endl;
        }
    }
    
    //-------------------------------------------------------------------
    
    class ParallelBias {
        Reconstruction* reconstructor;
        
    public:
        
        void operator()( const blocked_range<size_t>& r ) const {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                //alias the current weight image
                RealImage& w = reconstructor->_weights[inputIndex];
                
                //alias the current bias image
                RealImage b = reconstructor->_bias[inputIndex];
                
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
                        if (slice(i, j, 0) != -1) {
                            if( reconstructor->_simulated_weights[inputIndex](i,j,0) > 0.99 ) {
                                //bias-correct and scale current slice
                                double eb = exp(-b(i, j, 0));
                                slice(i, j, 0) *= (eb * scale);
                                
                                //calculate weight image
                                wb(i, j, 0) = w(i, j, 0) * slice(i, j, 0);
                                
                                //calculate weighted residual image
                                //make sure it is far from zero to avoid numerical instability
                                //if ((sim(i,j,0)>_low_intensity_cutoff*_max_intensity)&&(slice(i,j,0)>_low_intensity_cutoff*_max_intensity))
                                if ( (reconstructor->_simulated_slices[inputIndex](i, j, 0) > 1) && (slice(i, j, 0) > 1)) {
                                    wresidual(i, j, 0) = log(slice(i, j, 0) / reconstructor->_simulated_slices[inputIndex](i, j, 0)) * wb(i, j, 0);
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
                        if (slice(i, j, 0) != -1) {
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
                            if ((slice(i, j, 0) != -1) && (num > 0)) {
                                b(i, j, 0) -= mean;
                            }
                }
                
                reconstructor->_bias[inputIndex] = b;
            }
        }
        
        ParallelBias( Reconstruction *reconstructor ) :
        reconstructor(reconstructor)
        { }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::Bias()
    {
        // if (_debug)
        // cout << "Correcting bias ...";
        
        ParallelBias parallelBias( this );
        parallelBias();
        
        if (_debug)
            cout << "done. " << endl;
    }
    
    //-------------------------------------------------------------------
    
    class ParallelSuperresolution {
        Reconstruction* reconstructor;
    public:
        RealImage confidence_map;
        RealImage addon;
        
        void operator()( const blocked_range<size_t>& r ) {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                //read the current weight image
                RealImage& w = reconstructor->_weights[inputIndex];
                
                //read the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];
                
                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];
                
                //Update reconstructed volume using current slice
                
                //Distribute error to the volume
                POINT3D p;
                for ( int i = 0; i < slice.GetX(); i++)
                    for ( int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                            
                            if ( reconstructor->_simulated_slices[inputIndex](i,j,0) > 0 )
                                slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                            else
                                slice(i,j,0) = 0;
                            
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for (int k = 0; k < n; k++) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                
                                if (reconstructor->_robust_slices_only) {
                                    addon(p.x, p.y, p.z) += p.value * slice(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                    confidence_map(p.x, p.y, p.z) += p.value * reconstructor->_slice_weight[inputIndex];
                                }
                                else {
                                    addon(p.x, p.y, p.z) += p.value * slice(i, j, 0) * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                    confidence_map(p.x, p.y, p.z) += p.value * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                }
                            }
                        }
            } //end of loop for a slice inputIndex
        }
        
        ParallelSuperresolution( ParallelSuperresolution& x, split ) :
        reconstructor(x.reconstructor)
        {
            //Clear addon
            addon.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            addon = 0;
            
            //Clear confidence map
            confidence_map.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            confidence_map = 0;
        }
        
        void join( const ParallelSuperresolution& y ) {
            addon += y.addon;
            confidence_map += y.confidence_map;
        }
        
        ParallelSuperresolution( Reconstruction *reconstructor ) :
        reconstructor(reconstructor)
        {
            //Clear addon
            addon.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            addon = 0;
            
            //Clear confidence map
            confidence_map.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            confidence_map = 0;
        }
        
        // execute
        void operator() () {
            //task_scheduler_init init(tbb_no_threads);
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()), *this );
            //init.terminate();
        }
    };
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::Superresolution(int iter)
    {
        // if (_debug)
        // cout << "Superresolution " << iter << endl;
        
        int i, j, k;
        RealImage addon, original;
        
        
        //Remember current reconstruction for edge-preserving smoothing
        original = _reconstructed;
        
        ParallelSuperresolution parallelSuperresolution(this);
        parallelSuperresolution();
        addon = parallelSuperresolution.addon;
        _confidence_map = parallelSuperresolution.confidence_map;
        //_confidence4mask = _confidence_map;
        
        if(_debug) {
            char buffer[256];
            //sprintf(buffer,"confidence-map%i.nii.gz",iter);
            //_confidence_map.Write(buffer);
            _confidence_map.Write("confidence-map.nii.gz");
            //sprintf(buffer,"addon%i.nii.gz",iter);
            //addon.Write(buffer);
        }
        
        if (!_adaptive)
            for (i = 0; i < addon.GetX(); i++)
                for (j = 0; j < addon.GetY(); j++)
                    for (k = 0; k < addon.GetZ(); k++)
                        if (_confidence_map(i, j, k) > 0) {
                            // ISSUES if _confidence_map(i, j, k) is too small leading
                            // to bright pixels
                            addon(i, j, k) /= _confidence_map(i, j, k);
                            //this is to revert to normal (non-adaptive) regularisation
                            _confidence_map(i,j,k) = 1;
                        }
        
        _reconstructed += addon * _alpha; //_average_volume_weight;
        
        //bound the intensities
        for (i = 0; i < _reconstructed.GetX(); i++)
            for (j = 0; j < _reconstructed.GetY(); j++)
                for (k = 0; k < _reconstructed.GetZ(); k++) {
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
    
    class ParallelMStep{
        Reconstruction* reconstructor;
    public:
        double sigma;
        double mix;
        double num;
        double min;
        double max;
        
        void operator()( const blocked_range<size_t>& r ) {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                // read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                //alias the current weight image
                RealImage& w = reconstructor->_weights[inputIndex];
                
                //alias the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];
                
                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];
                
                //calculate error
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //bias correct and scale the slice
                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;
                            
                            //otherwise the error has no meaning - it is equal to slice intensity
                            if ( reconstructor->_simulated_weights[inputIndex](i,j,0) > 0.99 ) {
                                slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                                
                                //sigma and mix
                                double e = slice(i, j, 0);
                                sigma += e * e * w(i, j, 0);
                                mix += w(i, j, 0);
                                
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
        
        ParallelMStep( ParallelMStep& x, split ) :
        reconstructor(x.reconstructor)
        {
            sigma = 0;
            mix = 0;
            num = 0;
            min = 0;
            max = 0;
        }
        
        void join( const ParallelMStep& y ) {
            if (y.min < min)
                min = y.min;
            if (y.max > max)
                max = y.max;
            
            sigma += y.sigma;
            mix += y.mix;
            num += y.num;
        }
        
        ParallelMStep( Reconstruction *reconstructor ) :
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
            //task_scheduler_init init(tbb_no_threads);
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()), *this );
            //init.terminate();
        }
    };
    
    //-------------------------------------------------------------------
    
    void Reconstruction::MStep(int iter)
    {
        // if (_debug)
        // cout << "MStep" << endl;
        
        ParallelMStep parallelMStep(this);
        parallelMStep();
        double sigma = parallelMStep.sigma;
        double mix = parallelMStep.mix;
        double num = parallelMStep.num;
        double min = parallelMStep.min;
        double max = parallelMStep.max;
        
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
    
    class ParallelAdaptiveRegularization1 {
        Reconstruction *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;
        
    public:
        ParallelAdaptiveRegularization1( Reconstruction *_reconstructor,
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
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, 13), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    class ParallelAdaptiveRegularization2 {
        Reconstruction *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;
        
    public:
        ParallelAdaptiveRegularization2( Reconstruction *_reconstructor,
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
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed.GetX()), *this );
            //init.terminate();
        }
        
    };
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::AdaptiveRegularization(int iter, RealImage& original)
    {
        if (_debug)
            cout << "AdaptiveRegularization" << endl;
        
        Array<double> factor(13,0);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }
        
        Array<RealImage> b;//(13);
        for (int i = 0; i < 13; i++)
            b.push_back( _reconstructed );
        
        ParallelAdaptiveRegularization1 parallelAdaptiveRegularization1( this,
                                                                        b,
                                                                        factor,
                                                                        original );
        parallelAdaptiveRegularization1();
        
        RealImage original2 = _reconstructed;
        ParallelAdaptiveRegularization2 parallelAdaptiveRegularization2( this,
                                                                        b,
                                                                        factor,
                                                                        original2 );
        parallelAdaptiveRegularization2();
        
        if (_alpha * _lambda / (_delta * _delta) > 0.068) {
            cerr
            << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068."
            << endl;
        }
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::BiasCorrectVolume(RealImage& original)
    {
        //remove low-frequancy component in the reconstructed image which might have accured due to overfitting of the biasfield
        RealImage residual = _reconstructed;
        RealImage weights = _mask;
        
        //_reconstructed.Write("super-notbiascor.nii.gz");
        
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
        //residual.Write("residual.nii.gz");
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
    
    void Reconstruction::Evaluate(int iter)
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
        
    }
    
    
    //-------------------------------------------------------------------
    
    class ParallelNormaliseBias{
        Reconstruction* reconstructor;
    public:
        RealImage bias;
        
        void operator()( const blocked_range<size_t>& r ) {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                
                if(reconstructor->_debug) {
                    cout<<inputIndex<<" ";
                }
                
                // alias the current slice
                RealImage& slice = reconstructor->_slices[inputIndex];
                
                //read the current bias image
                RealImage b = reconstructor->_bias[inputIndex];
                
                //read current scale factor
                double scale = reconstructor->_scale[inputIndex];
                
                RealPixel *pi = slice.GetPointerToVoxels();
                RealPixel *pb = b.GetPointerToVoxels();
                for(int i = 0; i<slice.GetNumberOfVoxels(); i++) {
                    if((*pi>-1)&&(scale>0))
                        *pb -= log(scale);
                    pb++;
                    pi++;
                }
                
                //Distribute slice intensities to the volume
                POINT3D p;
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
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
        
        ParallelNormaliseBias( ParallelNormaliseBias& x, split ) :
        reconstructor(x.reconstructor)
        {
            bias.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            bias = 0;
        }
        
        void join( const ParallelNormaliseBias& y ) {
            bias += y.bias;
        }
        
        ParallelNormaliseBias( Reconstruction *reconstructor ) :
        reconstructor(reconstructor)
        {
            bias.Initialize( reconstructor->_reconstructed.GetImageAttributes() );
            bias = 0;
        }
        
        // execute
        void operator() () {
            //task_scheduler_init init(tbb_no_threads);
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()), *this );
            //init.terminate();
        }
    };
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::NormaliseBias( int iter )
    {
        if(_debug)
            cout << "Normalise Bias ... ";
        
        ParallelNormaliseBias parallelNormaliseBias(this);
        parallelNormaliseBias();
        RealImage bias = parallelNormaliseBias.bias;
        
        // normalize the volume by proportion of contributing slice voxels for each volume voxel
        bias /= _volume_weights;
        
        if(_debug)
            cout << "done." << endl;
        
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
        
        if (_debug) {
            char buffer[256];
            sprintf(buffer,"averagebias%i.nii.gz",iter);
            bias.Write(buffer);
        }
        
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
    
    void Reconstruction::ReadTransformation( char* folder )
    {
        int n = _slices.size();
        char name[256];
        char path[256];
        Transformation *transformation;
        RigidTransformation *rigidTransf;
        
        if (n == 0) {
            cerr << "Please create slices before reading transformations!" << endl;
            exit(1);
        }
        cout << "Reading transformations:" << endl;
        
        _transformations.clear();
        for (int i = 0; i < n; i++) {
            if (folder != NULL) {
                sprintf(name, "/transformation%i.dof", i);
                strcpy(path, folder);
                strcat(path, name);
            }
            else {
                sprintf(path, "transformation%i.dof", i);
            }
            transformation = Transformation::New(path);
            rigidTransf = dynamic_cast<RigidTransformation*>(transformation);
            _transformations.push_back(*rigidTransf);
            delete transformation;
            cout << path << endl;
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SetReconstructed( RealImage &reconstructed )
    {
        _reconstructed = reconstructed;
        _template_created = true;
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SetTransformations( Array<RigidTransformation>& transformations )
    {
        _transformations.clear();
        for ( int i = 0; i < transformations.size(); i++ )
            _transformations.push_back(transformations[i]);
        
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveBiasFields()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "bias%i.nii.gz", inputIndex);
            _bias[inputIndex].Write(buffer);
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveConfidenceMap()
    {
        _confidence_map.Write("confidence-map.nii.gz");
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveSlices()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            sprintf(buffer, "slice%i.nii.gz", inputIndex);
            _slices[inputIndex].Write(buffer);
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveSlicesWithTiming()
    {
        char buffer[256];
        cout<<"Saving slices with timing: ";
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            sprintf(buffer, "sliceTime%i.nii.gz", _slice_timing[inputIndex]);
            _slices[inputIndex].Write(buffer);
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveSimulatedSlices()
    {
        cout<<"Saving simulated slices ...";
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            sprintf(buffer, "simslice%i.nii.gz", inputIndex);
            _simulated_slices[inputIndex].Write(buffer);
        }
        cout<<"done."<<endl;
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveWeights()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "weights%i.nii.gz", inputIndex);
            _weights[inputIndex].Write(buffer);
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveRegistrationStep(Array<RealImage>& stacks,int step) {
        
        char buffer[256];
        ImageAttributes attr = stacks[0].GetImageAttributes();
        int stack;
        int slice;
        int threshold = attr._z;
        int counter = 0;
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            if (inputIndex >= threshold) {
                counter++;
                attr = stacks[counter].GetImageAttributes();
                threshold += attr._z;
            }
            stack = counter;
            slice = inputIndex - (threshold-attr._z);
            sprintf(buffer, "step%04i_travol%04islice%04i.dof", step, stack, slice);
            _transformations[inputIndex].Write(buffer);
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveTransformationsWithTiming()
    {
        char buffer[256];
        cout<<"Saving transformations with timing: ";
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            cout<<inputIndex<<" ";
            sprintf(buffer, "transformationTime%i.dof", _slice_timing[inputIndex]);
            _transformations[inputIndex].Write(buffer);
        }
        cout<<" done."<<endl;
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveTransformationsWithTiming(int iter)
    {
        char buffer[256];
        cout<<"Saving transformations with timing: ";
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            cout<<inputIndex<<" ";
            sprintf(buffer, "transformationTime%i-%i.dof", iter,_slice_timing[inputIndex]);
            _transformations[inputIndex].Write(buffer);
        }
        cout<<" done."<<endl;
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveTransformations()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "transformation%i.dof", inputIndex);
            _transformations[inputIndex].Write(buffer);
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::GetTransformations( Array<RigidTransformation> &transformations )
    {
        transformations.clear();
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            transformations.push_back( _transformations[inputIndex] );
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::GetSlices( Array<RealImage> &slices )
    {
        slices.clear();
        for (unsigned int i = 0; i < _slices.size(); i++)
            slices.push_back(_slices[i]);
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SetSlices( Array<RealImage> &slices )
    {
        _slices.clear();
        for (unsigned int i = 0; i < slices.size(); i++)
            _slices.push_back(slices[i]);
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SaveProbabilityMap( int i )
    {
        char buffer[256];
        sprintf( buffer, "probability_map%i.nii", i );
        _brain_probability.Write( buffer );
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SlicesInfo( const char* filename, Array<string> &stack_files )
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
            // << _stack_factor[i] << "\t"
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
    
    void Reconstruction::GetSliceAcquisitionOrder(Array<RealImage>& stacks, Array<int> &pack_num, Array<int> order, int step, int rewinder)
    {
        ImageAttributes attr;
        int slicesPerPackage;
        int slice_pos_counter, counter, temp, p, stepFactor, rewinderFactor;
        Array<int> fakeAscending;
        Array<int> realInterleaved;
        
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            
            attr = stacks[dyn].GetImageAttributes();
            slicesPerPackage = (attr._z/pack_num[dyn]);
            int z_slice_order[attr._z];
            int t_slice_order[attr._z];
            
            // ascending
            if (order[dyn] == 1) {
                
                counter = 0;
                slice_pos_counter = 0;
                p = 0;
                
                while(counter < attr._z)    {
                    
                    z_slice_order[counter] = slice_pos_counter;
                    t_slice_order[slice_pos_counter] = counter;
                    counter++;
                    slice_pos_counter = slice_pos_counter + pack_num[dyn];
                    
                    // start new package
                    if (slice_pos_counter >= attr._z) {
                        p++;
                        slice_pos_counter = p;
                    }
                }
                
                // copying
                for(temp = 0; temp < attr._z; temp++)     {
                    _z_slice_order.push_back(z_slice_order[temp]);
                    _t_slice_order.push_back(t_slice_order[temp]);
                }
            }
            
            // descending
            else if (order[dyn] == 2) {
                
                counter = 0;
                slice_pos_counter = attr._z - 1;
                int p = 0; // package counter
                
                while(counter < attr._z)    {
                    
                    z_slice_order[counter] = slice_pos_counter;
                    t_slice_order[slice_pos_counter] = counter;
                    counter++;
                    slice_pos_counter = slice_pos_counter - pack_num[dyn];
                    
                    // start new package
                    if (slice_pos_counter < 0) {
                        p++;
                        slice_pos_counter = attr._z - 1 - p;
                    }
                }
                
                // copying
                for(temp = 0; temp < attr._z; temp++)     {
                    _z_slice_order.push_back(z_slice_order[temp]);
                    _t_slice_order.push_back(t_slice_order[temp]);
                }
            }
            
            // default
            else if (order[dyn] == 3) {
                
                int index, restart;
                rewinderFactor = 1;
                stepFactor = 2;
                
                // pretending to do ascending within each package, and then shuffling according to default acquisition
                for(int p = 0; p < pack_num[dyn]; p++)
                {
                    // middle part of the stack
                    for(int s = 0; s < slicesPerPackage; s++)
                    {
                        slice_pos_counter = s*pack_num[dyn] + p;
                        fakeAscending.push_back(slice_pos_counter);
                    }
                    
                    // last slices for larger packages
                    if(attr._z > slicesPerPackage*pack_num[dyn]) {
                        slice_pos_counter = slicesPerPackage*pack_num[dyn] + p;
                        if (slice_pos_counter < attr._z) {
                            fakeAscending.push_back(slice_pos_counter);
                        }
                    }
                    
                    // shuffling
                    index = 0;
                    restart = 0;
                    for(int i = 0; i < fakeAscending.size(); i++)     {
                        if (index >= fakeAscending.size()) {
                            restart = restart + rewinderFactor;
                            index = restart;
                        }
                        realInterleaved.push_back(fakeAscending[index]);
                        index = index + stepFactor;
                    }
                    // clear fake ascending and start over
                    fakeAscending.clear();
                }
                
                // saving
                for(temp = 0; temp < attr._z; temp++)     {
                    z_slice_order[temp] = realInterleaved[temp];
                    t_slice_order[realInterleaved[temp]] = temp;
                }
                
                // copying
                for(temp = 0; temp < attr._z; temp++)     {
                    _z_slice_order.push_back(z_slice_order[temp]);
                    _t_slice_order.push_back(t_slice_order[temp]);
                }
                realInterleaved.clear();
            }
            
            // interleaved
            else {
                
                int index, restart;
                counter = 0;
                
                if (order[dyn] == 4)
                {
                    rewinderFactor = 1;
                }
                else
                {
                    stepFactor = step;
                    rewinderFactor = rewinder;
                }
                
                // pretending to do ascending within each package, and then shuffling according to interleaved acquisition
                for(int p = 0; p < pack_num[dyn]; p++)
                {
                    if (order[dyn] == 4)
                    {
                        // getting step size, from PPE
                        if((attr._z - counter) > slicesPerPackage*pack_num[dyn])        {
                            stepFactor = round(sqrt(double(slicesPerPackage + 1)));
                            counter++;
                        }
                        else
                            stepFactor = round(sqrt(double(slicesPerPackage)));
                    }
                    
                    // middle part of the stack
                    for(int s = 0; s < slicesPerPackage; s++)
                    {
                        slice_pos_counter = s*pack_num[dyn] + p;
                        fakeAscending.push_back(slice_pos_counter);
                    }
                    
                    // last slices for larger packages
                    if(attr._z > slicesPerPackage*pack_num[dyn]) {
                        slice_pos_counter = slicesPerPackage*pack_num[dyn] + p;
                        if (slice_pos_counter < attr._z) {
                            fakeAscending.push_back(slice_pos_counter);
                        }
                    }
                    
                    // shuffling
                    index = 0;
                    restart = 0;
                    for(int i = 0; i < fakeAscending.size(); i++)     {
                        if (index >= fakeAscending.size()) {
                            restart = restart + rewinderFactor;
                            index = restart;
                        }
                        realInterleaved.push_back(fakeAscending[index]);
                        index = index + stepFactor;
                    }
                    // clear fake ascending and start over
                    fakeAscending.clear();
                }
                
                // saving
                for(temp = 0; temp < attr._z; temp++)     {
                    z_slice_order[temp] = realInterleaved[temp];
                    t_slice_order[realInterleaved[temp]] = temp;
                }
                
                // copying
                for(temp = 0; temp < attr._z; temp++)     {
                    _z_slice_order.push_back(z_slice_order[temp]);
                    _t_slice_order.push_back(t_slice_order[temp]);
                }
                realInterleaved.clear();
            }
        }
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::flexibleSplitImage(Array<RealImage>& stacks, Array<RealImage>& sliceStacks, Array<int> &pack_num, Array<int> sliceNums, Array<int> order, int step, int rewinder)
    {
        RealImage image;
        ImageAttributes attr;
        int internalIterations, sliceNum;
        
        // location acquisition order
        Array<int> z_internal_slice_order;
        
        // calculate slice order
        char buffer[256];
        
        GetSliceAcquisitionOrder(stacks,pack_num,order,step,rewinder);
        
        // counters
        int counter1 = 0;
        int counter2 = 0;
        int counter3 = 0;
        
        int startIterations = 0;
        int endIterations = 0;
        int sum = 0;
        
        // dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            
            image = stacks[dyn];
            attr   = image.GetImageAttributes();
            
            // slice loop
            for (int sl = 0; sl < attr._z; sl++) {
                z_internal_slice_order.push_back(_z_slice_order[counter1 + sl]);
            }
            
            // fake packages
            while (sum < attr._z)    {
                sum = sum + sliceNums[counter2];
                counter2++;
            }
            endIterations = counter2;
            
            // fake package loop
            for (int iter = startIterations; iter < endIterations; iter++) {
                internalIterations = sliceNums[iter];
                RealImage stack(attr);
                // copying
                for(int sl = counter3; sl < internalIterations + counter3; sl++)     {
                    for(int j=0; j<stack.GetY();j++)
                        for(int i=0; i<stack.GetX();i++)
                        {
                            stack.Put(i,j,z_internal_slice_order[sl],image(i,j,z_internal_slice_order[sl]));
                        }
                }
                // pushing package
                sliceStacks.push_back(stack);
                counter3 = counter3 + internalIterations;
                
                
                for  (int i = 0; i < sliceStacks.size(); i++){
                    
                    //sprintf( buffer, "sliceStacks%i.nii", i );
                    //sliceStacks[i].Write( buffer );
                }
            }
            
            // updating varialbles for next dynamic
            z_internal_slice_order.clear();
            counter1 = counter1 + attr._z;
            counter3 = 0;
            sum = 0;
            startIterations = endIterations;
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::flexibleSplitImagewithMB(Array<RealImage>& stacks, Array<RealImage>& sliceStacks, Array<int> &pack_num, Array<int> sliceNums, Array<int> multiband_vector, Array<int> order, int step, int rewinder)
    {
        // initializing variables
        RealImage chunck;
        Array<RealImage> chuncks, chuncks_separated, chuncks_separated_reordered, chunksAll;
        Array<int> pack_num_chucks;
        RealImage image;
        RealImage multibanded, toAdd;
        ImageAttributes attr;
        int sliceMB;
        int stepFactor, startFactor, endFactor;
        int counter1, counter2, counter3, counter4, counter5;
        int multiband;
        
        int sum = 0;
        Array<int> sliceNumsChunks;
        
        startFactor = 0;
        endFactor = 0;
        // dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            
            image  = stacks[dyn];
            attr   = image.GetImageAttributes();
            RealImage multibanded(attr);
            multiband = multiband_vector[dyn];
            sliceMB = attr._z/multiband;
            
            for (int m = 0; m < multiband; m++) {
                chunck  = image.GetRegion(0,0,m*sliceMB,attr._x,attr._y,(m+1)*sliceMB);
                chuncks.push_back(chunck);
                pack_num_chucks.push_back(pack_num[dyn]);
            }
            
            while (sum < sliceMB)    {
                sum = sum + sliceNums[endFactor];
                endFactor++;
            }
            
            for (int m = 0; m < multiband; m++) {
                for (int iter = startFactor; iter < endFactor; iter++)    {
                    sliceNumsChunks.push_back(sliceNums[iter]);
                }
            }
            startFactor = endFactor;
            sum = 0;
        }
        
        // splitting each multiband subgroup
        flexibleSplitImage(chuncks, chunksAll, pack_num_chucks, sliceNumsChunks, order, step, rewinder);
        
        counter4 = 0;
        counter5 = 0;
        stepFactor = 0;
        // new dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            
            image  = stacks[dyn];
            attr   = image.GetImageAttributes();
            RealImage multibanded(attr);
            multiband = multiband_vector[dyn];
            sliceMB = attr._z/multiband;
            
            // stepping factor in vector
            while (sum < sliceMB)    {
                sum = sum + sliceNums[counter5 + stepFactor];
                stepFactor++;
            }
            
            // getting data from this dynamic
            for (int iter = 0; iter < multiband*stepFactor; iter++) {
                chuncks_separated.push_back(chunksAll[iter+counter4]);
            }
            counter4 = counter4 + multiband*stepFactor;
            
            // reordering chuncks_separated
            counter1 = 0;
            counter2 = 0;
            counter3 = 0;
            while (counter1 < chuncks_separated.size()) {
                
                chuncks_separated_reordered.push_back(chuncks_separated[counter2]);
                
                counter2 = counter2 + stepFactor;
                if (counter2 > (chuncks_separated.size() - 1))    {
                    counter3++;
                    counter2 = counter3;
                }
                counter1++;
            }
            
            // riassembling multiband packs
            counter1 = 0;
            counter2 = 0;
            while (counter1 < chuncks_separated_reordered.size())     {
                for (int m = 0; m < multiband; m++)    {
                    toAdd = chuncks_separated_reordered[counter1];
                    for (int k = 0; k < toAdd.GetZ(); k++)    {
                        
                        for(int j=0; j<toAdd.GetY();j++)
                            for(int i=0; i<toAdd.GetX();i++)
                                multibanded.Put(i,j,counter2,toAdd(i,j,k));
                        
                        counter2++;
                        
                    }
                    counter1++;
                }
                sliceStacks.push_back(multibanded);
                // clean multibanded for next iteration
                for(int k=0; k<multibanded.GetZ();k++)
                    for(int j=0; j<multibanded.GetY();j++)
                        for(int i=0; i<multibanded.GetX();i++)
                            multibanded.Put(i,j,k,0);
                counter2 = 0;
            }
            chuncks_separated.clear();
            chuncks_separated_reordered.clear();
            sum = 0;
            counter5 = counter5 + stepFactor;
            stepFactor = 0;
        }
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::splitPackages(Array<RealImage>& stacks, Array<int> &pack_num, Array<RealImage>& packageStacks, Array<int> order, int step, int rewinder)
    {
        RealImage image;
        ImageAttributes attr;
        int pkg_z, internalIterations;
        
        // location acquisition order
        Array<int> z_internal_slice_order;
        
        // counters
        int counter1 = 0;
        int counter2 = 0;
        int counter3 = 0;
        
        // calculate slice order
        GetSliceAcquisitionOrder(stacks,pack_num,order,step,rewinder);
        
        // dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            
            // current stack
            image  = stacks[dyn];
            attr   = image.GetImageAttributes();
            pkg_z  = attr._z/pack_num[dyn];
            
            // slice loop
            for (int sl = 0; sl < attr._z; sl++) {
                z_internal_slice_order.push_back(_z_slice_order[counter1 + sl]);
            }
            
            // package loop
            for(int p = 0; p < pack_num[dyn]; p++)
            {
                // slice excess for each package
                if((attr._z - counter2) > pkg_z*pack_num[dyn]) {
                    internalIterations = pkg_z + 1;
                    counter2++;
                }
                else{
                    internalIterations = pkg_z;
                }
                
                // copying
                RealImage stack(attr);
                for(int sl = counter3; sl < internalIterations + counter3; sl++) {
                    for(int j=0; j<stack.GetY();j++)
                        for(int i=0; i<stack.GetX();i++)
                        {
                            stack.Put(i,j,z_internal_slice_order[sl],image(i,j, z_internal_slice_order[sl]));
                        }
                    
                }
                // pushing package
                packageStacks.push_back(stack);
                // updating varialbles for next package
                counter3 = counter3 + internalIterations;
            }
            // updating varialbles for next dynamic
            z_internal_slice_order.clear();
            counter1 = counter1 + attr._z;
            counter2 = 0;
            counter3 = 0;
        }
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::splitPackageswithMB(Array<RealImage>& stacks, Array<int> &pack_num, Array<RealImage>& packageStacks, Array<int> multiband_vector, Array<int> order, int step, int rewinder)
    {
        // initializing variables
        RealImage chunck;
        Array<RealImage> chuncks, chuncks_separated, chuncks_separated_reordered, chunksAll;
        Array<int> pack_numAll;
        RealImage image;
        RealImage multibanded, toAdd;
        ImageAttributes attr;
        int sliceMB;
        int stepFactor;
        int counter1, counter2, counter3, counter4;
        int multiband;
        
        // dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            
            image  = stacks[dyn];
            attr   = image.GetImageAttributes();
            RealImage multibanded(attr);
            multiband = multiband_vector[dyn];
            sliceMB = attr._z/multiband;
            
            for (int m = 0; m < multiband; m++) {
                chunck  = image.GetRegion(0,0,m*sliceMB,attr._x,attr._y,(m+1)*sliceMB);
                chuncks.push_back(chunck);
                pack_numAll.push_back(pack_num[dyn]);
            }
        }
        
        // split package
        splitPackages(chuncks, pack_numAll, chunksAll, order, step, rewinder);
        
        counter4 = 0;
        // new dynamic loop
        for (int dyn = 0; dyn < stacks.size(); dyn++) {
            
            image  = stacks[dyn];
            attr   = image.GetImageAttributes();
            RealImage multibanded(attr);
            multiband = multiband_vector[dyn];
            sliceMB = attr._z/multiband;
            
            // getting data from this dynamic
            stepFactor = pack_num[dyn];
            for (int iter = 0; iter < multiband*stepFactor; iter++) {
                chuncks_separated.push_back(chunksAll[iter+counter4]);
            }
            counter4 = counter4 + multiband*stepFactor;
            
            // reordering chuncks_separated
            counter1 = 0;
            counter2 = 0;
            counter3 = 0;
            while (counter1 < chuncks_separated.size()) {
                
                chuncks_separated_reordered.push_back(chuncks_separated[counter2]);
                
                counter2 = counter2 + stepFactor;
                if (counter2 > (chuncks_separated.size() - 1))    {
                    counter3++;
                    counter2 = counter3;
                }
                counter1++;
            }
            
            // riassembling multiband slices
            counter1 = 0;
            counter2 = 0;
            while (counter1 < chuncks_separated_reordered.size())     {
                for (int m = 0; m < multiband; m++)    {
                    toAdd = chuncks_separated_reordered[counter1];
                    for (int k = 0; k < toAdd.GetZ(); k++)    {
                        
                        for(int j=0; j<toAdd.GetY();j++)
                            for(int i=0; i<toAdd.GetX();i++)
                                multibanded.Put(i,j,counter2,toAdd(i,j,k));
                        
                        counter2++;
                    }
                    counter1++;
                }
                packageStacks.push_back(multibanded);
                // clean multibanded
                for(int k=0; k<multibanded.GetZ();k++)
                    for(int j=0; j<multibanded.GetY();j++)
                        for(int i=0; i<multibanded.GetX();i++)
                            multibanded.Put(i,j,k,0);
                counter2 = 0;
            }
            chuncks_separated.clear();
            chuncks_separated_reordered.clear();
        }
    }
    
    //-------------------------------------------------------------------
    
    
    // 05/12
    
    void Reconstruction::newPackageToVolume(Array<RealImage>& stacks, Array<int> &pack_num, Array<int> multiband_vector, Array<int> order, int step, int rewinder, int iter, int steps)
    {
        char buffer[256];
        // copying transformations from previous iterations
        if (_previous_transformations.size() > 0)
            _previous_transformations.clear();
        for (int i = 0; i < _transformations.size(); i++)
            _previous_transformations.push_back(_transformations[i]);
        
        // initializing variable
        
        
        GreyImage t,s;
        RealImage target, firstPackage;
        Array<RealImage> packages;
        ImageAttributes attr;
        
        
        
        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        
        
        //irtkImageRigidRegistrationWithPadding rigidregistration;
        
        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        Insert(params, "Image (dis-)similarity measure", "NMI");
        if (_nmi_bins>0)
            Insert(params, "No. of bins", _nmi_bins);
        Insert(params, "Image interpolation mode", "Linear");
        //    Insert(params, "Background value", 0);
        Insert(params, "Background value for image 1", -1);
        Insert(params, "Background value for image 2", -1);
        
        GenericRegistrationFilter *rigidregistration = new GenericRegistrationFilter();
        rigidregistration->Parameter(params);
        
        
        
        Array<int> t_internal_slice_order;
        Array<RigidTransformation> internal_transformations;
        int multiband;
        
        Array<RealImage> sstacks;
        Array<int> spack_num;
        Array<int> smultiband_vector;
        Array<int> sorder;
        
        // other variables
        int counter1;
        int counter2 = 0;
        int counter3 = 0;
        int startIterations;
        int endIterations;
        int extra;
        int iterations;
        
        int wrapper = stacks.size()/steps;
        if ((stacks.size()%steps) > 0)
            wrapper++;
        
        int doffset;
        int count = 0;
        
        for (int w = 0; w < wrapper; w++)     {
            
            doffset = w*steps;
            // preparing input for this iterations
            for (int s = 0; s < steps; s++)        {
                if ((s+doffset) < stacks.size())
                {
                    sstacks.push_back(stacks[s+doffset]);
                    spack_num.push_back(pack_num[s+doffset]);
                    smultiband_vector.push_back(multiband_vector[s+doffset]);
                    sorder.push_back(order[s+doffset]);
                }
            }
            
            splitPackageswithMB(sstacks, spack_num, packages, smultiband_vector, sorder, step, rewinder);
            
            // other variables
            counter1 = 0;
            startIterations = 0;
            endIterations   = 0;
            
            for (int i = 0; i < sstacks.size(); i++) {
                
                firstPackage = packages[counter1];
                multiband = smultiband_vector[i];
                extra = (firstPackage.GetZ()/multiband)%(spack_num[i]);
                
                // slice loop
                for (int sl = 0; sl < firstPackage.GetZ(); sl++) {
                    t_internal_slice_order.push_back(_t_slice_order[counter3 + sl]);
                    internal_transformations.push_back(_transformations[counter2 + sl]);
                }
                
                // package look
                for (int j = 0; j < spack_num[i]; j++) {
                    
                    // performing registration
                    target = packages[counter1];
                    t = target;
                    s = _reconstructed;
                    
                    if (_debug) {
                        sprintf(buffer,"target%i-%i-%i.nii.gz",iter,i+doffset,j);
                        t.Write(buffer);
                        sprintf(buffer,"source%i-%i-%i.nii.gz",iter,i+doffset,j);
                        s.Write(buffer);
                    }
                    
                    //check whether package is empty (all zeros)
                    RealPixel tmin,tmax;
                    target.GetMinMax(&tmin,&tmax);
                    
                    if(tmax>0)
                    {
                        RigidTransformation offset;
                        ResetOrigin(t,offset);
                        Matrix mo = offset.GetMatrix();
                        Matrix m = internal_transformations[j].GetMatrix();
                        m=m*mo;
                        internal_transformations[j].PutMatrix(m);
                        
                        
                        rigidregistration->Input(&t, &s);
                        Transformation *dofout = nullptr;
                        rigidregistration->Output(&dofout);
                        
                        RigidTransformation tmp_dofin = internal_transformations[j];
                        
                        rigidregistration->InitialGuess(&tmp_dofin);
                        rigidregistration->GuessParameter();
                        rigidregistration->Run();
                        
                        RigidTransformation *rigid_dofout = dynamic_cast<RigidTransformation*> (dofout);
                        
                        internal_transformations[j] = *rigid_dofout;
                        
                        
                        //rigidregistration.SetInput(&t, &s);
                        //rigidregistration.SetOutput(&internal_transformations[j]);
                        //rigidregistration.GuessParameterPackageToVolume();
                        //rigidregistration.SetTargetPadding(0);
                        
                        
                        
                        /*
                         if (_debug) {
                         rigidregistration.Write("par-packages.rreg");
                         }
                         */
                        
                        rigidregistration->Run();
                        
                        mo.Invert();
                        m = internal_transformations[j].GetMatrix();
                        m=m*mo;
                        internal_transformations[j].PutMatrix(m);
                    }
                    
                    if (_debug) {
                        sprintf(buffer,"transformation%i-%i-%i.dof",iter,i+doffset,j);
                        internal_transformations[j].Write(buffer);
                    }
                    
                    // saving transformations
                    iterations = (firstPackage.GetZ()/multiband)/(spack_num[i]);
                    if (extra > 0) {
                        iterations++;
                        extra--;
                    }
                    endIterations = endIterations + iterations;
                    
                    for (int k = startIterations; k < endIterations; k++)     {
                        for (int l = 0; l < t_internal_slice_order.size(); l++) {
                            if (k == t_internal_slice_order[l]) {
                                _transformations[counter2+l].PutTranslationX(internal_transformations[j].GetTranslationX());
                                _transformations[counter2+l].PutTranslationY(internal_transformations[j].GetTranslationY());
                                _transformations[counter2+l].PutTranslationZ(internal_transformations[j].GetTranslationZ());
                                _transformations[counter2+l].PutRotationX(internal_transformations[j].GetRotationX());
                                _transformations[counter2+l].PutRotationY(internal_transformations[j].GetRotationY());
                                _transformations[counter2+l].PutRotationZ(internal_transformations[j].GetRotationZ());
                                _transformations[counter2+l].UpdateMatrix();
                            }
                        }
                    }
                    startIterations = endIterations;
                    counter1++;
                }
                // resetting variables for next dynamic
                startIterations = 0;
                endIterations = 0;
                t_internal_slice_order.clear();
                internal_transformations.clear();
                counter2 = counter2 + firstPackage.GetZ();
                counter3 = counter3 + firstPackage.GetZ();
            }
            
            
            //save overal slice order
            ImageAttributes attr = stacks[0].GetImageAttributes();
            int dyn, num;
            int slices_per_dyn = attr._z/multiband_vector[0];
            
            //slice order should repeat for each dynamic - only take first dynamic
            _slice_timing.clear();
            for (dyn=0;dyn<stacks.size();dyn++)
                for(int i = 0; i < attr._z; i++)
                {
                    _slice_timing.push_back( dyn*slices_per_dyn + _t_slice_order[i]);
                    //cout<<"index = "<<i<<endl;
                    //cout<<"attrz = "<<attr._z<<endl;
                    //cout<<"dyn = "<<dyn<<endl;
                    //cout<<"slices_per_dyn = "<<slices_per_dyn<<endl;
                    //cout<<"t = "<<_t_slice_order[i]<<endl;
                    cout<<"slice timing = "<<_slice_timing[i]<<endl;
                }
            
            
            
            for(int i = 0; i < _z_slice_order.size(); i++)
            {
                cout<<"z("<<i<<")="<<_z_slice_order[i]<<endl;
            }
            
            for(int i = 0; i < _t_slice_order.size(); i++)
            {
                cout<<"t("<<i<<")="<<_t_slice_order[i]<<endl;
            }
            
            // save transformations and clear
            _z_slice_order.clear();
            _t_slice_order.clear();
            
            sstacks.clear();
            spack_num.clear();
            smultiband_vector.clear();
            sorder.clear();
            packages.clear();
            counter3 = 0;
        }
    }
    
    
    //-------------------------------------------------------------------
    
    
    //-------------------------------------------------------------------
    
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SplitImage( RealImage image, int packages, Array<RealImage>& stacks )
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
    
    void Reconstruction::SplitImageEvenOdd( RealImage image, int packages, Array<RealImage>& stacks )
    {
        Array<RealImage> packs;
        Array<RealImage> packs2;
        cout<<"Split Image Even Odd: "<< packages <<" packages."<<endl;
        
        stacks.clear();
        SplitImage(image, packages, packs);
        
        for (int i=0; i<packs.size(); i++) {
            cout<<"Package "<< i <<": "<<endl;
            packs2.clear();
            SplitImage(packs[i], 2, packs2);
            stacks.push_back(packs2[0]);
            stacks.push_back(packs2[1]);
        }
        
        cout<<"done."<<endl;
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::SplitImageEvenOddHalf( RealImage image, int packages, Array<RealImage>& stacks, int iter )
    {
        Array<RealImage> packs;
        Array<RealImage> packs2;
        
        cout<<"Split Image Even Odd Half "<< iter <<endl;
        stacks.clear();
        if (iter>1)
            SplitImageEvenOddHalf(image, packages, packs, iter-1);
        else
            SplitImageEvenOdd(image, packages, packs);
        
        for (int i=0;i<packs.size();i++) {
            packs2.clear();
            HalfImage(packs[i],packs2);
            for (int j=0; j<packs2.size(); j++)
                stacks.push_back(packs2[j]);
        }
        
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::HalfImage( RealImage image, Array<RealImage>& stacks )
    {
        RealImage tmp;
        ImageAttributes attr = image.GetImageAttributes();
        stacks.clear();
        
        //We would not like single slices - that is reserved for slice-to-volume
        if(attr._z>=4) {
            tmp = image.GetRegion(0,0,0,attr._x,attr._y,attr._z/2);
            stacks.push_back(tmp);
            tmp = image.GetRegion(0,0,attr._z/2,attr._x,attr._y,attr._z);
            stacks.push_back(tmp);
        }
        else
            stacks.push_back(image);
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::PackageToVolume(Array<RealImage>& stacks, Array<int> &pack_num, int iter, bool evenodd, bool half, int half_iter)
    {
        //irtkImageRigidRegistrationWithPadding rigidregistration;
        
        GenericRegistrationFilter rigidregistration;
        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        Insert(params, "Image interpolation mode", "Linear");
        Insert(params, "Background value", -1);
        
        Insert(params, "Image (dis-)similarity measure", "NMI");
        if (_nmi_bins>0)
            Insert(params, "No. of bins", _nmi_bins);
        
        rigidregistration.Parameter(params);
        
        // _reconstructed.Write("s.nii.gz");
        
        
        RealImage t,s;
        RealImage target;
        Array<RealImage> packages;
        char buffer[256];
        ImageAttributes attr, attr2;
        
        GenericLinearInterpolateImageFunction<RealImage> interpolator;
        
        
        int firstSlice = 0;
        cout<<"Package to volume: "<<endl;
        for (unsigned int i = 0; i < stacks.size(); i++) {
            cout<<"Stack "<<i<<": First slice index is "<<firstSlice<<endl;
            
            packages.clear();
            
            if (evenodd) {
                if(half)
                    SplitImageEvenOddHalf(stacks[i],pack_num[i],packages,half_iter);
                else
                    SplitImageEvenOdd(stacks[i],pack_num[i],packages);
            }
            else
                SplitImage(stacks[i],pack_num[i],packages);
            
            
            
            for (unsigned int j = 0; j < packages.size(); j++) {
                
                
                cout<<"Package "<<j<<" of stack "<<i<<endl;
                if (_debug) {
                    sprintf(buffer,"package%i-%i-%i.nii.gz",iter,i,j);
                    packages[j].Write(buffer);
                }
                
                attr2=packages[j].GetImageAttributes();
                //packages are not masked at present
                
                Resampling<RealPixel> resampling(attr._dx,attr._dx, attr2._dz);
                
                target=packages[j];
                
                resampling.Input(&packages[j]);
                resampling.Output(&target);
                resampling.Interpolator(&interpolator);
                resampling.Run();
                
                t=target;
                s=_reconstructed;
                
                //find existing transformation
                double x,y,z;
                x=0;y=0;z=0;
                packages[j].ImageToWorld(x,y,z);
                stacks[i].WorldToImage(x,y,z);
                
                int firstSliceIndex = round(z)+firstSlice;
                cout<<"First slice index for package "<<j<<" of stack "<<i<<" is "<<firstSliceIndex<<endl;
                
                //put origin in target to zero
                RigidTransformation offset;
                ResetOrigin(t, offset);
                Matrix mo = offset.GetMatrix();
                Matrix m = _transformations[firstSliceIndex].GetMatrix();
                m=m*mo;
                _transformations[firstSliceIndex].PutMatrix(m);
                
                
                rigidregistration.Input(&t, &s);
                
                Transformation *dofout = nullptr;
                rigidregistration.Output(&dofout);
                
                
                // sprintf(buffer,"p-%i.nii.gz",j);
                // t.Write(buffer);
                
                // sprintf(buffer,"r-%i.dof",j);
                // _transformations[firstSliceIndex].Write(buffer);
                
                
                RigidTransformation tmp_dofin = _transformations[firstSliceIndex]; //offset;
                
                rigidregistration.InitialGuess(&tmp_dofin);
                rigidregistration.GuessParameter();
                rigidregistration.Run();
                
                RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*> (dofout);
                _transformations[firstSliceIndex] = *rigidTransf;
                
                //undo the offset
                mo.Invert();
                m = _transformations[firstSliceIndex].GetMatrix();
                m=m*mo;
                _transformations[firstSliceIndex].PutMatrix(m);
                
                
                if (_debug) {
                    sprintf(buffer,"transformation%i-%i-%i.dof",iter,i,j);
                    _transformations[firstSliceIndex].Write(buffer);
                }
                
                
                //set the transformation to all slices of the package
                cout<<"Slices of the package "<<j<<" of the stack "<<i<<" are: ";
                for (int k = 0; k < packages[j].GetZ(); k++) {
                    x=0;y=0;z=k;
                    packages[j].ImageToWorld(x,y,z);
                    stacks[i].WorldToImage(x,y,z);
                    int sliceIndex = round(z)+firstSlice;
                    cout<<sliceIndex<<" "<<endl;
                    
                    if(sliceIndex>=_transformations.size()) {
                        cerr<<"Reconstruction::PackageToVolume: sliceIndex out of range."<<endl;
                        cerr<<sliceIndex<<" "<<_transformations.size()<<endl;
                        exit(1);
                    }
                    
                    if(sliceIndex!=firstSliceIndex) {
                        _transformations[sliceIndex].PutTranslationX(_transformations[firstSliceIndex].GetTranslationX());
                        _transformations[sliceIndex].PutTranslationY(_transformations[firstSliceIndex].GetTranslationY());
                        _transformations[sliceIndex].PutTranslationZ(_transformations[firstSliceIndex].GetTranslationZ());
                        _transformations[sliceIndex].PutRotationX(_transformations[firstSliceIndex].GetRotationX());
                        _transformations[sliceIndex].PutRotationY(_transformations[firstSliceIndex].GetRotationY());
                        _transformations[sliceIndex].PutRotationZ(_transformations[firstSliceIndex].GetRotationZ());
                        _transformations[sliceIndex].UpdateMatrix();
                    }
                }
                
                
            }
            cout<<"End of stack "<<i<<endl<<endl;
            
            firstSlice += stacks[i].GetZ();
            
            
        }
        
    }
    
    
    //-------------------------------------------------------------------
    
    void Reconstruction::CropImage(RealImage& image, RealImage& mask)
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
        
        // if no intersection with mask, force exclude
        if ((x2 < x1) || (y2 < y1) || (z2 < z1) ) {
            x1 = 0;
            y1 = 0;
            z1 = 0;
            x2 = 0;
            y2 = 0;
            z2 = 0;
        }
        
        //Cut region of interest
        image = image.GetRegion(x1, y1, z1, x2+1, y2+1, z2+1);
    }
    
    //-------------------------------------------------------------------
    
    // GF 190416, useful for handling different slice orders
    void Reconstruction::CropImageIgnoreZ(RealImage& image, RealImage& mask)
    
    {
        //Crops the image according to the mask
        int i, j, k;
        //ROI boundaries
        int x1, x2, y1, y2, z1, z2;
        
        // Filling slices out of mask with zeors
        int sum = 0;
        for (k = image.GetZ() - 1; k >= 0; k--) {
            sum = 0;
            for (j = image.GetY() - 1; j >= 0; j--)
                for (i = image.GetX() - 1; i >= 0; i--)
                    if (mask.Get(i, j, k) > 0)
                        sum++;
            if (sum > 0) {
                k = k+1;
                break;
            }
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
        
        // Filling upper part
        for (k = z2; k < image.GetZ(); k++)    {
            for (j = 0; j < image.GetY(); j++)    {
                for (i = 0; i < image.GetX(); i++)    {
                    image.Put(i,j,k,0);
                }
            }
        }
        
        // Filling lower part
        for (k = 0; k < z1; k++)    {
            for (j = 0; j < image.GetY(); j++)    {
                for (i = 0; i < image.GetX(); i++)    {
                    image.Put(i,j,k,0);
                }
            }
        }
        
        //Original ROI
        x1 = 0;
        y1 = 0;
        z1 = 0;
        x2 = image.GetX();
        y2 = image.GetY();
        z2 = image.GetZ();
        
        z2 = z2-1;
        
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
        
        // if (_debug)
        //     cout << "Region of interest is " << x1 << " " << y1 << " " << z1 << " " << x2 << " " << y2
        //         << " " << z2 << endl;
        
        // if no intersection with mask, force exclude
        if ((x2 < x1) || (y2 < y1) || (z2 < z1) ) {
            x1 = 0;
            y1 = 0;
            z1 = 0;
            x2 = 0;
            y2 = 0;
            z2 = 0;
        }
        
        //Cut region of interest
        image = image.GetRegion(x1, y1, z1, x2+1, y2+1, z2+1);
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::InvertStackTransformations( Array<RigidTransformation>& stack_transformations )
    {
        //for each stack
        for (unsigned int i = 0; i < stack_transformations.size(); i++) {
            //invert transformation for the stacks
            stack_transformations[i].Invert();
            stack_transformations[i].UpdateParameter();
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::MaskVolume()
    {
        RealPixel *pr = _reconstructed.GetPointerToVoxels();
        RealPixel *pm = _mask.GetPointerToVoxels();
        for (int i = 0; i < _reconstructed.GetNumberOfVoxels(); i++) {
            if (*pm == 0)
                *pr = -1;
            pm++;
            pr++;
        }
    }
    
    //-------------------------------------------------------------------
    
    void Reconstruction::MaskImage( RealImage& image, double padding )
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
    
    void Reconstruction::Rescale( RealImage &img, double max )
    {
        int i, n;
        RealPixel *ptr, min_val, max_val;
        
        // Get lower and upper bound
        img.GetMinMax(&min_val, &max_val);
        
        n   = img.GetNumberOfVoxels();
        ptr = img.GetPointerToVoxels();
        for (i = 0; i < n; i++)
            if ( ptr[i] > 0 )
                ptr[i] = double(ptr[i]) / double(max_val) * max;
    }
    
    //-------------------------------------------------------------------
    
    
    
    void Reconstruction::BackgroundFiltering( Array<RealImage>& stacks, double fg_sigma , double bg_sigma )
    {
        
        // double fg_sigma = 0.3;
        // double bg_sigma = 10;
        
        char buffer[256];
        
        GaussianBlurring<RealPixel> gb3(stacks[0].GetXSize()*fg_sigma);     //0.85);
        
        GaussianBlurring<RealPixel> gb2(stacks[0].GetXSize()*bg_sigma);       //2);
        RealImage tmp_slice, tmp_slice_b;
        
        RealImage global_blurred, stack;
        
        
        for (int j = 0; j<stacks.size(); j++) {
            
            stack = stacks[j];
            
            sprintf(buffer,"original-%i.nii.gz",j);
            stack.Write(buffer);
            
            
            
            global_blurred = stacks[j];
            gb2.Input(&global_blurred);
            gb2.Output(&global_blurred);
            gb2.Run();
            
            
            for (int i = 0; i<stacks[j].GetZ(); i++) {
                
                
                tmp_slice = stacks[j].GetRegion(0,0,i,stacks[j].GetX(),stacks[j].GetY(),(i+1));
                tmp_slice_b = stacks[j].GetRegion(0,0,i,stacks[j].GetX(),stacks[j].GetY(),(i+1));
                
                gb3.Input(&tmp_slice_b);
                gb3.Output(&tmp_slice_b);
                gb3.Run();
                
                gb2.Input(&tmp_slice);
                gb2.Output(&tmp_slice);
                gb2.Run();
                
                
                for (int x = 0; x < stacks[j].GetX(); x++) {
                    for (int y = 0; y < stacks[j].GetY(); y++) {
                        
                        //                    stack(x,y,i) = tmp_slice_b(x,y,0) - 0.95*tmp_slice(x,y,0);
                        //                    if (stack(x, y, i) < 0)
                        //                        stack(x, y, i) = 1;
                        
                        
                        //                    dif_stack(x,y,i) = global_blurred(x,y,i) - tmp_slice(x,y,0);
                        
                        stack(x,y,i) = tmp_slice_b(x,y,0) + global_blurred(x,y,i) - tmp_slice(x,y,0);
                        if(stack(x,y,i) < 0)
                            stack(x,y,i) = 1;
                        
                    }
                }
            }
            
            stacks[j] = stack;
            
            
            sprintf(buffer,"filtered-%i.nii.gz",j);
            stack.Write(buffer);
            
            
        }
        
    }
    
    
    
    
    
} // namespace mirtk

