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

#include "svrtk/ReconstructionDWI.h"

using namespace std;
using namespace mirtk;

namespace svrtk {

    //-------------------------------------------------------------------


    ReconstructionDWI::ReconstructionDWI()
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
        _regul_steps = 1;
        _intensity_matching_GD = false;

        int directions[13][3] = {
            { 1, 0, -1 },
            { 0, 1, -1 },
            { 1, 1, -1 },
            { 1, -1, -1 },
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 1, 1, 0 },
            { 1, -1, 0 },
            { 1, 0, 1 },
            { 0, 1, 1 },
            { 1, 1, 1 },
            { 1, -1, 1 },
            { 0, 0, 1 }
        };
        for ( int i = 0; i < 13; i++ )
            for ( int j = 0; j < 3; j++ )
                _directions[i][j] = directions[i][j];

        _recon_type = _interpolate;

    }

    ReconstructionDWI::~ReconstructionDWI() {

        _lambdaLB = 0;
    }


    class ParallelAverage_DWI{
        ReconstructionDWI* reconstructor;
        Array<RealImage> &stacks;
        Array<RigidTransformation> &stack_transformations;

        double targetPadding;

        double sourcePadding;

        double background;

        bool linear;

    public:
        RealImage average;
        RealImage weights;

        void operator()( const blocked_range<size_t>& r ) {
            for ( size_t i = r.begin(); i < r.end(); ++i) {

                GenericLinearInterpolateImageFunction<RealImage> interpolator;

                RealImage s = stacks[i];
                RigidTransformation t = stack_transformations[i];
                RealImage image;
                image.Initialize(reconstructor->_reconstructed.Attributes());
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

                RealPixel *pa = average.Data();
                RealPixel *pi = image.Data();
                RealPixel *pw = weights.Data();
                for (int p = 0; p < average.NumberOfVoxels(); p++) {
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

        ParallelAverage_DWI( ParallelAverage_DWI& x, split ) :
        reconstructor(x.reconstructor),
        stacks(x.stacks),
        stack_transformations(x.stack_transformations)
        {
            average.Initialize( reconstructor->_reconstructed.Attributes() );
            average = 0;
            weights.Initialize( reconstructor->_reconstructed.Attributes() );
            weights = 0;
            targetPadding = x.targetPadding;
            sourcePadding = x.sourcePadding;
            background = x.background;
            linear = x.linear;
        }

        void join( const ParallelAverage_DWI& y ) {
            average += y.average;
            weights += y.weights;
        }

        ParallelAverage_DWI( ReconstructionDWI *reconstructor,
                        Array<RealImage>& _stacks,
                        Array<RigidTransformation>& _stack_transformations,
                        double _targetPadding,
                        double _sourcePadding,
                        double _background,
                        bool _linear=false ) :
        reconstructor(reconstructor),
        stacks(_stacks),
        stack_transformations(_stack_transformations)
        {
            average.Initialize( reconstructor->_reconstructed.Attributes() );
            average = 0;
            weights.Initialize( reconstructor->_reconstructed.Attributes() );
            weights = 0;
            targetPadding = _targetPadding;
            sourcePadding = _sourcePadding;
            background = _background;
            linear = _linear;
        }

        // execute
        void operator() () {

            parallel_reduce( blocked_range<size_t>(0,stacks.size()), *this );
        }
    };

    class ParallelSliceAverage_DWI{
        ReconstructionDWI* reconstructor;
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

                        average.ImageToWorld(x0, y0, z0);
                        for (int inputIndex = 0; inputIndex < slices.size(); inputIndex++ ) {
                            double x = x0;
                            double y = y0;
                            double z = z0;

                            slice_transformations[inputIndex].Transform(x, y, z);

                            slices[inputIndex].WorldToImage(x, y, z);
                            int i = round(x);
                            int j = round(y);
                            int k = round(z);

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

        ParallelSliceAverage_DWI( ReconstructionDWI *reconstructor,
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
            average.Initialize( reconstructor->_reconstructed.Attributes() );
            average = 0;
            weights.Initialize( reconstructor->_reconstructed.Attributes() );
            weights = 0;
        }

        // execute
        void operator() () const {
            parallel_for( blocked_range<size_t>(0,average.GetZ()), *this );
        }
    };

    RealImage ReconstructionDWI::CreateAverage(Array<RealImage>& stacks,
                                                    Array<RigidTransformation>& stack_transformations)
    {
        if (!_template_created)
            throw runtime_error("Please create the template before calculating the average of the stacks.");

        InvertStackTransformations(stack_transformations);
        ParallelAverage_DWI ParallelAverage_DWI( this,
                                        stacks,
                                        stack_transformations,
                                        -1,0,0,
                                        true );
        ParallelAverage_DWI();
        RealImage average = ParallelAverage_DWI.average;
        RealImage weights = ParallelAverage_DWI.weights;
        average /= weights;
        InvertStackTransformations(stack_transformations);
        return average;
    }

    double ReconstructionDWI::CreateTemplate(RealImage stack, double resolution)
    {
        double dx, dy, dz, d;

        ImageAttributes attr = stack.Attributes();

        attr._z += 2;
        attr._t = 1;

        RealImage enlarged(attr);

        if (resolution <= 0) {
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

        cout << "Reconstructed volume voxel size : " << d << endl;

        InterpolationMode interpolation = Interpolation_Linear;

        UniquePtr<InterpolateImageFunction> interpolator;
        interpolator.reset(InterpolateImageFunction::New(interpolation));


        RealPixel smin, smax;
        stack.GetMinMax(&smin, &smax);
        enlarged = stack;

        if (smin < -0.1) {
            ResamplingWithPadding<RealPixel> resampling(d, d, d, -1);
            GenericLinearInterpolateImageFunction<RealImage> interpolator;
            resampling.Input(&stack);
            resampling.Output(&enlarged);
            resampling.Interpolator(&interpolator);
            resampling.Run();

        }
        else {
            if (smin < 0.1) {
                ResamplingWithPadding<RealPixel> resampling(d, d, d, 0);
                GenericLinearInterpolateImageFunction<RealImage> interpolator;
                resampling.Input(&stack);
                resampling.Output(&enlarged);
                resampling.Interpolator(&interpolator);
                resampling.Run();

            }
            else {
                Resampling<RealPixel> resampler(d, d, d);
                resampler.Input(&enlarged);
                resampler.Output(&enlarged);
                resampler.Interpolator(interpolator.get());
                resampler.Run();

            }

        }

        _reconstructed = enlarged;
        _template_created = true;

        if (_debug)
            _reconstructed.Write("template.nii.gz");

        return d;

    }


    void ReconstructionDWI::SetTemplate(RealImage templateImage)
    {
        RealImage t2template = _reconstructed;

        t2template = 0;

        RigidTransformation tr;

        ImageTransformation imagetransformation;
        GenericLinearInterpolateImageFunction<RealImage> interpolator;

        imagetransformation.Input(&templateImage);
        imagetransformation.Transformation(&tr);
        imagetransformation.Output(&t2template);
        imagetransformation.TargetPaddingValue(-1);
        imagetransformation.SourcePaddingValue(0);
        imagetransformation.Interpolator(&interpolator);
        imagetransformation.Run();

        _reconstructed = t2template;
    }

    void ReconstructionDWI::SetMask(RealImage * mask, double sigma, double threshold)
    {

        if (!_template_created)
            throw runtime_error("Please create the template before setting the mask, so that the mask can be resampled to the correct dimensions.");

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
            _mask.Write("mask.nii.gz");

    }

    void ReconstructionDWI::SetMaskOrient(RealImage * mask, RigidTransformation transformation)
    {
        if (!_template_created)
            throw runtime_error("Please create the template before setting the mask, so that the mask can be resampled to the correct dimensions.");

        _mask = _reconstructed;

        if (mask != NULL) {

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
            _mask.Write("mask.nii.gz");
    }


    void ReconstructionDWI::SetTemplateImage(RealImage t, RigidTransformation tr)
    {

        RealImage t2template = _reconstructed;

        t2template = 0;

        ImageTransformation imagetransformation;
        GenericLinearInterpolateImageFunction<RealImage> interpolator;

        imagetransformation.Input(&t);
        imagetransformation.Transformation(&tr);
        imagetransformation.Output(&t2template);
        imagetransformation.TargetPaddingValue(-1);
        imagetransformation.SourcePaddingValue(0);
        imagetransformation.Interpolator(&interpolator);
        imagetransformation.Run();

        _reconstructed = t2template;
        _reconstructed.Write("templateimage.nii.gz");

    }


    double ReconstructionDWI::Consistency()
    {
        double sum = 0, num = 0, diff;
        for(int ind=0;ind<_slices.size();ind++)
        {
            for (int i=0; i<_slices[ind].GetX();i++)
                for (int j=0; j<_slices[ind].GetY();j++)
                    if(_slices[ind](i,j,0)>0)
                        if(_simulated_weights[ind](i,j,0)>0.99)
                        {
                            diff = _slices[ind](i,j,0)*exp(-_bias[ind](i, j, 0)) * _scale[ind] - _simulated_slices[ind](i,j,0);
                            sum+=diff*diff;
                            num++;
                        }
        }

        return sqrt(sum/num);
    }




    void ReconstructionDWI::TransformMask(RealImage& image, RealImage& mask,
                                           RigidTransformation& transformation)
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


    void ReconstructionDWI::ResetOrigin(GreyImage &image, RigidTransformation& transformation)
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



    void ReconstructionDWI::ResetOrigin(RealImage &image, RigidTransformation& transformation)
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




    class ParallelStackRegistrations_DWI {
        ReconstructionDWI *reconstructor;
        Array<RealImage>& stacks;
        Array<RigidTransformation>& stack_transformations;
        int templateNumber;
        GreyImage& target;
        RigidTransformation& offset;

    public:
        ParallelStackRegistrations_DWI( ReconstructionDWI *_reconstructor,
                                   Array<RealImage>& _stacks,
                                   Array<RigidTransformation>& _stack_transformations,
                                   int _templateNumber,
                                   GreyImage& _target,
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

                if (i == templateNumber)
                    continue;

                RealImage r_source = stacks[i];
                RealImage r_target = target;

                RealImage tmp_stack;

                ResamplingWithPadding<RealPixel> resampling2(1.5, 1.5, 1.5, 0);
                GenericLinearInterpolateImageFunction<RealImage> interpolator2;
                resampling2.Interpolator(&interpolator2);

                tmp_stack = r_source;
                resampling2.Input(&tmp_stack);
                resampling2.Output(&r_source);
                resampling2.Run();

                tmp_stack = r_target;
                resampling2.Input(&tmp_stack);
                resampling2.Output(&r_target);
                resampling2.Run();

                Matrix mo = offset.GetMatrix();
                Matrix m = stack_transformations[i].GetMatrix();
                m=m*mo;
                stack_transformations[i].PutMatrix(m);

                ParameterList params;
                Insert(params, "Transformation model", "Rigid");
//                if (reconstructor->_nmi_bins>0)
//                    Insert(params, "No. of bins", reconstructor->_nmi_bins);

//                if(reconstructor->_masked_stacks) {
//                    Insert(params, "Background value", 0);
//                }
//                else {
//                    Insert(params, "Background value for image 1", 0);
//                }

                Insert(params, "Background value for image 1", 0);

//                Insert(params, "Image (dis-)similarity measure", "NCC");
//                string type = "box";
//                string units = "vox";
//                double width = 0;
//                Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);


                GenericRegistrationFilter registration;
                registration.Parameter(params);

                registration.Input(&r_target, &r_source);

                Transformation *dofout = nullptr;
                registration.Output(&dofout);

                RigidTransformation dofin = stack_transformations[i];
                registration.InitialGuess(&dofin);

                registration.GuessParameter();
                registration.Run();

                unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(dofout));
                stack_transformations[i] = *rigidTransf;

                mo.Invert();
                m = stack_transformations[i].GetMatrix();
                m=m*mo;
                stack_transformations[i].PutMatrix(m);

                if (reconstructor->_debug) {

                    char buffer[256];
                    sprintf(buffer, "global-transformation%zu.dof", i);
                    stack_transformations[i].Write(buffer);
                    sprintf(buffer, "global-stack%zu.nii.gz", i);
                    stacks[i].Write(buffer);
                }


            }
        }

        void operator() () const {

            parallel_for( blocked_range<size_t>(0, stacks.size() ), *this );

        }

    };



    void ReconstructionDWI::StackRegistrations(Array<RealImage>& stacks,
                                                Array<RigidTransformation>& stack_transformations, int templateNumber)
    {
        if (_debug)
            cout << "StackRegistrations" << endl;

        InvertStackTransformations(stack_transformations);

        GreyImage target = stacks[templateNumber];

        if (_have_mask) {
            double x, y, z;
            for (int i = 0; i < target.GetX(); i++)
                for (int j = 0; j < target.GetY(); j++)
                    for (int k = 0; k < target.GetZ(); k++) {

                        x = i;
                        y = j;
                        z = k;

                        target.ImageToWorld(x, y, z);

                        _mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);

                        if (_mask.IsInside(x, y, z)) {
                            if (_mask(x, y, z) == 0)
                                target(i, j, k) = 0;
                        }
                        else
                            target(i, j, k) = 0;
                    }
        }

        if(_debug)
        {
            target.Write("target.nii.gz");
            stacks[0].Write("stack0.nii.gz");
        }
        RigidTransformation offset;
        ResetOrigin(target,offset);

        ParallelStackRegistrations_DWI registration( this,
                                                stacks,
                                                stack_transformations,
                                                templateNumber,
                                                target,
                                                offset );
        registration();

        InvertStackTransformations(stack_transformations);
    }



    void ReconstructionDWI::StackRegistrations(Array<RealImage>& stacks,
                                               Array<RigidTransformation>& stack_transformations)
    {
        InvertStackTransformations(stack_transformations);

        char buffer[256];

        GreyImage target = _reconstructed;

        if(_debug)
            target.Write("target-nomask.nii.gz");
        if (_have_mask)
        {
            double x, y, z;
            for (int i = 0; i < target.GetX(); i++)
                for (int j = 0; j < target.GetY(); j++)
                    for (int k = 0; k < target.GetZ(); k++)
                    {

                        x = i;
                        y = j;
                        z = k;

                        target.ImageToWorld(x, y, z);

                        _mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);

                        if (_mask.IsInside(x, y, z)) {
                            if (_mask(x, y, z) == 0)
                                target(i, j, k) = 0;
                        }
                        else
                            target(i, j, k) = 0;
                    }
        }



        if(_debug)
            target.Write("target.nii.gz");
        RigidTransformation offset;
        ResetOrigin(target,offset);

        for (int i = 0; i < (int)stacks.size(); i++)
        {

            GreyImage source = stacks[i];

            Matrix mo = offset.GetMatrix();
            Matrix m = stack_transformations[i].GetMatrix();
            m=m*mo;
            stack_transformations[i].PutMatrix(m);


            RealImage r_source = stacks[i];
            RealImage r_target = target;

            RealImage tmp_stack;

            ResamplingWithPadding<RealPixel> resampling2(1.5, 1.5, 1.5, 0);
            GenericLinearInterpolateImageFunction<RealImage> interpolator2;
            resampling2.Interpolator(&interpolator2);

            tmp_stack = r_source;
            resampling2.Input(&tmp_stack);
            resampling2.Output(&r_source);
            resampling2.Run();

            tmp_stack = r_target;
            resampling2.Input(&tmp_stack);
            resampling2.Output(&r_target);
            resampling2.Run();



            ParameterList params;
            Insert(params, "Transformation model", "Rigid");
            Insert(params, "Background value for image 1", 0);


//            Insert(params, "Image (dis-)similarity measure", "NCC");
//            string type = "box";
//            string units = "vox";
//            double width = 0;
//            Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);


            GenericRegistrationFilter registration;
            registration.Parameter(params);

            registration.Input(&r_target, &r_source);

            Transformation *dofout = nullptr;
            registration.Output(&dofout);

            RigidTransformation dofin = stack_transformations[i];
            registration.InitialGuess(&dofin);

            registration.GuessParameter();
            registration.Run();

            unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(dofout));
            stack_transformations[i] = *rigidTransf;


            mo.Invert();
            m = stack_transformations[i].GetMatrix();
            m=m*mo;
            stack_transformations[i].PutMatrix(m);


            if (_debug)
            {
                sprintf(buffer, "stack-transformation%i.dof.gz", i);
                stack_transformations[i].Write(buffer);
                sprintf(buffer, "stack%i.nii.gz", i);
                stacks[i].Write(buffer);
            }
        }

        InvertStackTransformations(stack_transformations);
    }





    void ReconstructionDWI::RestoreSliceIntensities()
    {
        if (_debug)
            cout << "Restoring the intensities of the slices. "<<endl;

        unsigned int inputIndex;
        int i;
        double factor;
        RealPixel *p;

        for (inputIndex=0;inputIndex<_slices.size();inputIndex++) {

            factor = _stack_factor[_stack_index[inputIndex]];

            p = _slices[inputIndex].Data();
            for(i=0;i<_slices[inputIndex].NumberOfVoxels();i++) {
                if(*p>0) *p = *p / factor;
                p++;
            }
        }
    }




    void ReconstructionDWI::ScaleVolume()
    {
        if (_debug)
            cout << "Scaling volume: ";

        unsigned int inputIndex;
        int i, j;
        double scalenum = 0, scaleden = 0;

        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

            RealImage& slice = _slices[inputIndex];

            RealImage& w = _weights[inputIndex];

            RealImage& sim = _simulated_slices[inputIndex];

            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {

                        if ( _simulated_weights[inputIndex](i,j,0) > 0.99 ) {
                            scalenum += w(i, j, 0) * _slice_weight[inputIndex] * slice(i, j, 0) * sim(i, j, 0);
                            scaleden += w(i, j, 0) * _slice_weight[inputIndex] * sim(i, j, 0) * sim(i, j, 0);
                        }
                    }
        }


        double scale = scalenum / scaleden;

        if(_debug)
            cout<<" scale = "<<scale;

        RealPixel *ptr = _reconstructed.Data();
        for(i=0;i<_reconstructed.NumberOfVoxels();i++) {
            if(*ptr>0) *ptr = *ptr * scale;
            ptr++;
        }
        cout<<endl;
    }

    class ParallelSimulateSlices_DWI {
        ReconstructionDWI *reconstructor;

    public:
        ParallelSimulateSlices_DWI( ReconstructionDWI *_reconstructor ) :
        reconstructor(_reconstructor) { }

        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {

                reconstructor->_simulated_slices[inputIndex].Initialize( reconstructor->_slices[inputIndex].Attributes() );
                reconstructor->_simulated_slices[inputIndex] = 0;

                reconstructor->_simulated_weights[inputIndex].Initialize( reconstructor->_slices[inputIndex].Attributes() );
                reconstructor->_simulated_weights[inputIndex] = 0;

                reconstructor->_simulated_inside[inputIndex].Initialize( reconstructor->_slices[inputIndex].Attributes() );
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

        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                         *this );
        }

    };

    void ReconstructionDWI::SimulateSlices()
    {
        if (_debug)
            cout<<"Simulating slices."<<endl;

        ParallelSimulateSlices_DWI ParallelSimulateSlices_DWI( this );
        ParallelSimulateSlices_DWI();

        if (_debug)
            cout<<"done."<<endl;
    }

    void ReconstructionDWI::SimulateStacks(Array<RealImage>& stacks, bool simulate_excluded)
    {
        if (_debug)
            cout<<"Simulating stacks."<<endl;

        unsigned int inputIndex;
        int i, j, k, n;
        RealImage sim;
        POINT3D p;
        double weight;

        int z, current_stack;
        z=-1;
        current_stack=-1;

        double threshold = 0.5;
        if (simulate_excluded)
            threshold = -1;

        _reconstructed.Write("reconstructed.nii.gz");
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

            cout<<inputIndex<<" ";

            RealImage& slice = _slices[inputIndex];

            sim.Initialize( slice.Attributes() );
            sim = 0;

            if(_slice_weight[inputIndex]>threshold)
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

        }
    }

    void ReconstructionDWI::MatchStackIntensities( Array<RealImage>& stacks,
                                                   Array<RigidTransformation>& stack_transformations,
                                                   double averageValue,
                                                   bool together )
    {
        if (_debug)
            cout << "Matching intensities of stacks. ";

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
                throw runtime_error("Stack " + to_string(ind) + " has no overlap with ROI");
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



    void ReconstructionDWI::MaskStacks(Array<RealImage>& stacks, Array<RigidTransformation>& stack_transformations)
    {
        cout << "Masking stacks ... ";

        double x, y, z;
        int i, j, k;

        if (!_have_mask) {
            cout << "Could not mask slices because no mask has been set." << endl;
            return;
        }

        for (int unsigned inputIndex = 0; inputIndex < stacks.size(); inputIndex++) {
            RealImage& stack = stacks[inputIndex];
            for (i = 0; i < stack.GetX(); i++)
                for (j = 0; j < stack.GetY(); j++) {
                    for (k = 0; k < stack.GetZ(); k++) {

                        if (stack(i,j,k) < 0.01)
                            stack(i,j,k) = -1;

                        x = i;
                        y = j;
                        z = k;

                        stack.ImageToWorld(x, y, z);

                        stack_transformations[inputIndex].Transform(x, y, z);

                        _mask.WorldToImage(x, y, z);
                        x = round(x);
                        y = round(y);
                        z = round(z);

                        if (_mask.IsInside(x, y, z)) {
                            if (_mask(x, y, z) == 0)
                                stack(i, j, k) = -1;
                        }
                        else
                            stack(i, j, k) = -1;
                    }
                }

        }
        cout << "done." << endl;
    }

    void ReconstructionDWI::MatchStackIntensitiesWithMasking(Array<RealImage>& stacks,
                                                              Array<RigidTransformation>& stack_transformations, double averageValue, bool together)
    {
        if (_debug)
            cout << "Matching intensities of stacks. ";

        double sum, num;
        char buffer[256];
        unsigned int ind;
        int i, j, k;
        double x, y, z;
        Array<double> stack_average;
        RealImage m;

        _average_value = averageValue;

        for (ind = 0; ind < stacks.size(); ind++) {
            m=stacks[ind];
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

                        if (_mask.IsInside(x, y, z)) {
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

            if (num > 0)
                stack_average.push_back(sum / num);
            else {
                throw runtime_error("Stack " + to_string(ind) + " has no overlap with ROI");
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


    void ReconstructionDWI::CreateSlicesAndTransformations( Array<RealImage> &stacks,
                                                            Array<RigidTransformation> &stack_transformations,
                                                            Array<double> &thickness)
    {
        if (_debug)
            cout << "CreateSlicesAndTransformations" << endl;

        for (unsigned int i = 0; i < stacks.size(); i++) {

            ImageAttributes attr = stacks[i].Attributes();

            for (int j = 0; j < attr._z; j++) {

                RealImage slice = stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);

                slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);

                _slices.push_back(slice);
                _original_slices.push_back(slice);
                _simulated_slices.push_back(slice);
                _simulated_weights.push_back(slice);
                _simulated_inside.push_back(slice);

                _intensity_weights.push_back(1);

                _stack_locs.push_back(j);

                _stack_index.push_back(i);

                _transformations.push_back(stack_transformations[i]);

            }
        }
        cout << "Number of slices: " << _slices.size() << endl;

    }

    void ReconstructionDWI::SetForceExcludedStacks(Array<int> &force_excluded_stacks)
    {
        for (uint s = 0; s<force_excluded_stacks.size(); s++)
            for (uint i=0; i<_slices.size(); i++)
                if (_stack_index[i]==force_excluded_stacks[s])
                    _force_excluded.push_back(i);
        int sum = 0;
        cout<<"SetForceExcludedStacks: force excluded slices: ";
        for (uint i=0; i<_force_excluded.size(); i++) {
            sum = sum+1;
        }

        cout << sum << endl;

    }


    void ReconstructionDWI::UpdateSlices(Array<RealImage>& stacks, Array<double>& thickness)
    {
        _slices.clear();

        for (unsigned int i = 0; i < stacks.size(); i++) {

            ImageAttributes attr = stacks[i].Attributes();

            for (int j = 0; j < attr._z; j++) {

                RealImage slice = stacks[i].GetRegion(0, 0, j, attr._x, attr._y, j + 1);
                slice.PutPixelSize(attr._dx, attr._dy, thickness[i]);
                _slices.push_back(slice);

            }
        }
        cout << "Number of slices: " << _slices.size() << endl;

    }

    void ReconstructionDWI::MaskSlices()
    {
        cout << "Masking slices ... ";

        double x, y, z;
        int i, j;

        if (!_have_mask) {
            cout << "Could not mask slices because no mask has been set." << endl;
            return;
        }

        for (int unsigned inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

            RealImage& slice = _slices[inputIndex];

            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++) {

                    if (slice(i,j,0) < 0.01)
                        slice(i,j,0) = -1;

                    x = i;
                    y = j;
                    z = 0;

                    slice.ImageToWorld(x, y, z);

                    _transformations[inputIndex].Transform(x, y, z);

                    _mask.WorldToImage(x, y, z);
                    x = round(x);
                    y = round(y);
                    z = round(z);

                    if (_mask.IsInside(x, y, z)) {
                        if (_mask(x, y, z) == 0)
                            slice(i, j, 0) = -1;
                    }
                    else
                        slice(i, j, 0) = -1;
                }

        }
        cout << "done." << endl;
    }


    int ReconstructionDWI::SliceCount(int inputIndex)
    {

        RealImage slice = _slices[inputIndex];

        int count = 0;


        for (int i = 0; i < slice.GetX(); i++) {
            for (int j = 0; j < slice.GetY(); j++) {

                if (slice(i,j,0) > 0.01) {
                    count = count + 1;
                }


            }
        }


        return count;

    }


    class ParallelSliceToVolumeRegistration_DWI {
    public:
        ReconstructionDWI *reconstructor;

        ParallelSliceToVolumeRegistration_DWI(ReconstructionDWI *_reconstructor) :
        reconstructor(_reconstructor) { }

        void operator() (const blocked_range<size_t> &r) const {

            ImageAttributes attr = reconstructor->_reconstructed.Attributes();

            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {

                char buffer[256];

                RealPixel smin, smax;

                RealImage target;
                target = reconstructor->_slices[inputIndex];
                target.GetMinMax(&smin, &smax);

                int count = reconstructor->SliceCount(inputIndex);


                if (count > 15 && smax > 0.1 && (smax - smin) > 0.01) {


                    ParameterList params;
                    Insert(params, "Transformation model", "Rigid");

                    RealPixel tmin, tmax;
                    reconstructor->_reconstructed.GetMinMax(&tmin, &tmax);


                    Insert(params, "Image (dis-)similarity measure", "NCC");
                    string type = "box";
                    string units = "vox";
                    double width = 0;
                    Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);

//                    if (smin < 1) {
//                        Insert(params, "Background value for image 1", -1);
//                    }
//
//                    if (tmin < 0) {
//                        Insert(params, "Background value for image 2", -1);
//                    }
//                    else {
//                        if (tmin < 1) {
//                            Insert(params, "Background value for image 2", 0);
//                        }
//                    }


//                    Insert(params, "Background value", -1);

                    Insert(params, "Background value for image 1", -1);
                    Insert(params, "Background value for image 2", -1);


                    GenericRegistrationFilter registration;
                    registration.Parameter(params);


                    RigidTransformation offset;

                    reconstructor->ResetOrigin(target,offset);
                    Matrix mo = offset.GetMatrix();
                    Matrix m = reconstructor->_transformations[inputIndex].GetMatrix();
                    m=m*mo;
                    reconstructor->_transformations[inputIndex].PutMatrix(m);


                    registration.Input(&target, &reconstructor->_reconstructed);

                    Transformation *dofout = nullptr;
                    registration.Output(&dofout);

                    registration.InitialGuess(&reconstructor->_transformations[inputIndex]);
                    registration.GuessParameter();
                    registration.Run();

                    unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(dofout));
                    reconstructor->_transformations[inputIndex] = *rigidTransf;

                    mo.Invert();
                    m = reconstructor->_transformations[inputIndex].GetMatrix();
                    m=m*mo;
                    reconstructor->_transformations[inputIndex].PutMatrix(m);
                }
            }
        }

        // execute
        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );
        }

    };

    void ReconstructionDWI::SliceToVolumeRegistration()
    {

        if (_debug)
            cout << "SliceToVolumeRegistration" << endl;
        ParallelSliceToVolumeRegistration_DWI registration(this);
        registration();
    }


    class ParallelCoeffInit_DWI {
    public:
        ReconstructionDWI *reconstructor;

        ParallelCoeffInit_DWI(ReconstructionDWI *_reconstructor) :
        reconstructor(_reconstructor) { }

        void operator() (const blocked_range<size_t> &r) const {

            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {

                bool slice_inside;

                double vx, vy, vz;
                reconstructor->_reconstructed.GetPixelSize(&vx, &vy, &vz);

                double res = vx;

                cout << inputIndex << " ";
                cout.flush();

                RealImage& slice = reconstructor->_slices[inputIndex];

                POINT3D p;
                VOXELCOEFFS empty;
                SLICECOEFFS slicecoeffs(slice.GetX(), Array < VOXELCOEFFS > (slice.GetY(), empty));

                slice_inside = false;

                double dx, dy, dz;
                slice.GetPixelSize(&dx, &dy, &dz);

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

                double size = res / reconstructor->_quality_factor;

                int xDim = round(2 * dx / size);
                int yDim = round(2 * dy / size);
                int zDim = round(2 * dz / size);

                xDim = xDim/2*2+1;
                yDim = yDim/2*2+1;
                zDim = zDim/2*2+1;

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

                            PSF(i, j, k) = exp(
                                               -x * x / (2 * sigmax * sigmax) - y * y / (2 * sigmay * sigmay)
                                               - z * z / (2 * sigmaz * sigmaz));
                            sum += PSF(i, j, k);
                        }
                PSF /= sum;

                if (reconstructor->_debug)
                    if (inputIndex == 0)
                        PSF.Write("PSF.nii.gz");

                int dim = (floor(ceil(sqrt(double(xDim * xDim + yDim * yDim + zDim * zDim)) * size / res) / 2))
                * 2 + 1 + 2;

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

                if (reconstructor->_intensity_weights[inputIndex] > 0) {

                    for (i = 0; i < slice.GetX(); i++)
                        for (j = 0; j < slice.GetY(); j++)
                            if (slice(i, j, 0) != -1) {

                                x = i;
                                y = j;
                                z = 0;
                                slice.ImageToWorld(x, y, z);
                                reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                reconstructor->_reconstructed.WorldToImage(x, y, z);
                                tx = round(x);
                                ty = round(y);
                                tz = round(z);

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

                                            slice.ImageToWorld(x, y, z);

                                            reconstructor->_transformations[inputIndex].Transform(x, y, z);

                                            reconstructor->_reconstructed.WorldToImage(x, y, z);

                                            nx = (int) floor(x);
                                            ny = (int) floor(y);
                                            nz = (int) floor(z);

                                            sum = 0;

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

                                            if ((sum <= 0) || (!inside))
                                                continue;

                                            for (l = nx; l <= nx + 1; l++)
                                                if ((l >= 0) && (l < reconstructor->_reconstructed.GetX()))
                                                    for (m = ny; m <= ny + 1; m++)
                                                        if ((m >= 0) && (m < reconstructor->_reconstructed.GetY()))
                                                            for (n = nz; n <= nz + 1; n++)
                                                                if ((n >= 0) && (n < reconstructor->_reconstructed.GetZ())) {
                                                                    weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));

                                                                    int aa, bb, cc;
                                                                    aa = l - tx + centre;
                                                                    bb = m - ty + centre;
                                                                    cc = n - tz + centre;

                                                                    double value = PSF(ii, jj, kk) * weight / sum;

                                                                    if ((aa < 0) || (aa >= dim) || (bb < 0) || (bb >= dim) || (cc < 0) || (cc >= dim)) {
                                                                        stringstream err;
                                                                        err << "Error while trying to populate tPSF. " << aa << " " << bb << " " << cc << endl;
                                                                        err << l << " " << m << " " << n << endl;
                                                                        err << tx << " " << ty << " " << tz << endl;
                                                                        err << centre << endl;
                                                                        tPSF.Write("tPSF.nii.gz");
                                                                        throw runtime_error(err.str());
                                                                    }
                                                                    else
                                                                        tPSF(aa, bb, cc) += value;
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

    void ReconstructionDWI::CoeffInit()
    {
        if (_debug)
            cout << "CoeffInit" << endl;

        _volcoeffs.clear();
        _volcoeffs.resize(_slices.size());

        _slice_inside.clear();
        _slice_inside.resize(_slices.size());

        cout << "Initialising matrix coefficients...";
        cout.flush();
        ParallelCoeffInit_DWI coeffinit(this);
        coeffinit();
        cout << " ... done." << endl;

        _volume_weights.Initialize( _reconstructed.Attributes() );
        _volume_weights = 0;

        int inputIndex, i, j, n, k;
        POINT3D p;
        for ( inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            for ( i = 0; i < _slices[inputIndex].GetX(); i++)
                for ( j = 0; j < _slices[inputIndex].GetY(); j++) {
                    n = _volcoeffs[inputIndex][i][j].size();
                    for (k = 0; k < n; k++) {
                        p = _volcoeffs[inputIndex][i][j][k];
                        _volume_weights(p.x, p.y, p.z) += p.value;
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
            cout<<"Average volume weight is "<<_average_volume_weight<<endl;
        }

    }


    void ReconstructionDWI::GaussianReconstruction(double small_slices_threshold)
    {
        cout << "Gaussian reconstruction ... ";
        unsigned int inputIndex;
        int i, j, k, n;
        RealImage slice;
        double scale;
        POINT3D p;
        Array<int> voxel_num;
        int slice_vox_num;

        _reconstructed = 0;

        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            slice = _slices[inputIndex];
            RealImage& b = _bias[inputIndex];
            scale = _scale[inputIndex];

            slice_vox_num=0;

            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {

                        if(_intensity_matching_GD)
                            slice(i, j, 0) *= b(i, j, 0) * scale;
                        else
                            slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                        n = _volcoeffs[inputIndex][i][j].size();
                        if (n>0)
                            slice_vox_num++;

                        for (k = 0; k < n; k++) {
                            p = _volcoeffs[inputIndex][i][j][k];
                            _reconstructed(p.x, p.y, p.z) += p.value * slice(i, j, 0);
                        }
                    }
            voxel_num.push_back(slice_vox_num);
        }

        _reconstructed /= _volume_weights;

        cout << "done." << endl;


            _reconstructed.Write("init.nii.gz");

        Array<int> voxel_num_tmp;
        for (i=0;i<voxel_num.size();i++)
            voxel_num_tmp.push_back(voxel_num[i]);

        sort(voxel_num_tmp.begin(),voxel_num_tmp.end());
        int median = voxel_num_tmp[round(voxel_num_tmp.size()*0.5)];

        _small_slices.clear();
        for (i=0;i<voxel_num.size();i++)
            if (voxel_num[i]<small_slices_threshold*median)
                _small_slices.push_back(i);

        if (_debug) {
            cout<<"Small slices:";
            for (i=0;i<_small_slices.size();i++)
                cout<<" "<<_small_slices[i];
            cout<<endl;
        }
    }




    void ReconstructionDWI::InitializeEM()
    {
        if (_debug)
            cout << "InitializeEM" << endl;

        _weights.clear();
        _bias.clear();
        _scale.clear();
        _slice_weight.clear();

        for (unsigned int i = 0; i < _slices.size(); i++) {
            _weights.push_back(_slices[i]);
            _bias.push_back(_slices[i]);
            _scale.push_back(1);
            _slice_weight.push_back(1);
        }

        _max_intensity = voxel_limits<RealPixel>::min();
        _min_intensity = voxel_limits<RealPixel>::max();
        for (unsigned int i = 0; i < _slices.size(); i++) {

            RealPixel *ptr = _slices[i].Data();
            for (int ind = 0; ind < _slices[i].NumberOfVoxels(); ind++) {
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



    void ReconstructionDWI::InitializeEMValues()
    {
        if (_debug)
            cout << "InitializeEMValues" << endl;

        for (unsigned int i = 0; i < _slices.size(); i++) {

            RealPixel *pw = _weights[i].Data();
            RealPixel *pb = _bias[i].Data();
            RealPixel *pi = _slices[i].Data();
            for (int j = 0; j < _weights[i].NumberOfVoxels(); j++) {
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

            _slice_weight[i] = 1;
            _scale[i] = 1;
        }

        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            _slice_weight[_force_excluded[i]] = 0;
    }

    void ReconstructionDWI::ResetBiasAndScale(Array<RealImage>& bias, Array<double>& scale)
    {
        if (_debug)
            cout << "ResetBiasAndScale" << endl;

        for (unsigned int i = 0; i < _slices.size(); i++) {

            bias.push_back(_bias[i]);
            scale.push_back(_scale[i]);

            RealPixel *pb = _bias[i].Data();
            RealPixel *pi = _slices[i].Data();
            for (int j = 0; j < _bias[i].NumberOfVoxels(); j++) {
                if (*pi != -1) {
                    *pb = 0;
                }
                else {
                    *pb = 0;
                }
                pi++;
                pb++;
            }

            _scale[i] = 1;
        }

    }

    void ReconstructionDWI::AddBiasAndScale(Array<RealImage>& bias, Array<double>& scale)
    {
        if (_debug)
            cout << "AddBiasAndScale" << endl;

        for (unsigned int i = 0; i < _slices.size(); i++) {
            _bias[i]+=bias[i];
            _scale[i]*=scale[i];

        }

    }



    void ReconstructionDWI::InitializeRobustStatistics()
    {
        if (_debug)
            cout << "InitializeRobustStatistics" << endl;

        int i, j;
        RealImage slice,sim;
        double sigma = 0;
        int num = 0;

        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            slice = _slices[inputIndex];

            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {

                        if ( (_simulated_inside[inputIndex](i, j, 0)==1)
                            &&(_simulated_weights[inputIndex](i,j,0)>0.99) ) {
                            slice(i,j,0) -= _simulated_slices[inputIndex](i,j,0);
                            sigma += slice(i, j, 0) * slice(i, j, 0);
                            num++;
                        }
                    }

            if (!_slice_inside[inputIndex])
                _slice_weight[inputIndex] = 0;
        }

        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            _slice_weight[_force_excluded[i]] = 0;

        _sigma = sigma / num;
        _sigma_s = 0.025;
        _mix = 0.9;
        _mix_s = 0.9;
        _m = 1 / (2.1 * _max_intensity - 1.9 * _min_intensity);

        if (_debug)
            cout << "Initializing robust statistics: " << "sigma=" << sqrt(_sigma) << " " << "m=" << _m
            << " " << "mix=" << _mix << " " << "mix_s=" << _mix_s << endl;

    }

    class ParallelEStep_DWI {
        ReconstructionDWI* reconstructor;
        Array<double> &slice_potential;

    public:

        void operator()( const blocked_range<size_t>& r ) const {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {

                RealImage slice = reconstructor->_slices[inputIndex];
                reconstructor->_weights[inputIndex] = 0;

                RealImage& b = reconstructor->_bias[inputIndex];

                double scale = reconstructor->_scale[inputIndex];

                double num = 0;

                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {

                            if(reconstructor->_intensity_matching_GD)
                                slice(i, j, 0) *= b(i, j, 0) * scale;
                            else
                                slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();

                            if ( (n>0) &&
                                (reconstructor->_simulated_weights[inputIndex](i,j,0) > 0) ) {
                                slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);

                                double g = reconstructor->G(slice(i, j, 0), reconstructor->_sigma);
                                double m = reconstructor->M(reconstructor->_m);

                                double weight = g * reconstructor->_mix / (g *reconstructor->_mix + m * (1 - reconstructor->_mix));
                                reconstructor->_weights[inputIndex].PutAsDouble(i, j, 0, weight);

                                if(reconstructor->_simulated_weights[inputIndex](i,j,0)>0.99) {
                                    slice_potential[inputIndex] += (1 - weight) * (1 - weight);
                                    num++;
                                }
                            }
                            else
                                reconstructor->_weights[inputIndex].PutAsDouble(i, j, 0, 0);
                        }

                if (num > 0)
                    slice_potential[inputIndex] = sqrt(slice_potential[inputIndex] / num);
                else
                    slice_potential[inputIndex] = -1;
            }
        }

        ParallelEStep_DWI( ReconstructionDWI *reconstructor,
                      Array<double> &slice_potential ) :
        reconstructor(reconstructor), slice_potential(slice_potential)
        { }

        void operator() () const {

            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ), *this );

        }

    };

    void ReconstructionDWI::EStep()
    {

        if (_debug)
            cout << "EStep: " << endl;

        unsigned int inputIndex;
        RealImage slice, w, b, sim;
        int num = 0;
        Array<double> slice_potential(_slices.size(), 0);

        ParallelEStep_DWI ParallelEStep_DWI( this, slice_potential );
        ParallelEStep_DWI();

        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            slice_potential[_force_excluded[i]] = -1;

        for (unsigned int i = 0; i < _small_slices.size(); i++)
            slice_potential[_small_slices[i]] = -1;

        for (inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
            if ((_scale[inputIndex]<0.2)||(_scale[inputIndex]>5)) {
                slice_potential[inputIndex] = -1;
            }

        if(_debug) {
            cout<<endl<<"Slice potentials: ";
            for (inputIndex = 0; inputIndex < slice_potential.size(); inputIndex++)
                cout<<slice_potential[inputIndex]<<" ";
            cout<<endl;
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
                if ((slice_potential[inputIndex] < _mean_s2) && (slice_potential[inputIndex] > _mean_s))
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
            cout << "Slice weights: ";
            for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                cout << _slice_weight[inputIndex] << " ";
            cout << endl;
        }

    }

    class ParallelScale_DWI {
        ReconstructionDWI *reconstructor;

    public:
        ParallelScale_DWI( ReconstructionDWI *_reconstructor ) :
        reconstructor(_reconstructor)
        { }

        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {

                RealImage& slice = reconstructor->_slices[inputIndex];
                RealImage& w = reconstructor->_weights[inputIndex];
                RealImage& b = reconstructor->_bias[inputIndex];

                double scalenum = 0;
                double scaleden = 0;

                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            if(reconstructor->_simulated_weights[inputIndex](i,j,0)>0.99) {

                                double eb;
                                if(reconstructor->_intensity_matching_GD)
                                    eb=b(i,j,0);
                                else
                                    eb = exp(-b(i, j, 0));
                                scalenum += w(i, j, 0) * slice(i, j, 0) * eb * reconstructor->_simulated_slices[inputIndex](i, j, 0);
                                scaleden += w(i, j, 0) * slice(i, j, 0) * eb * slice(i, j, 0) * eb;
                            }
                        }

                if (scaleden > 0)
                    reconstructor->_scale[inputIndex] = scalenum / scaleden;
                else
                    reconstructor->_scale[inputIndex] = 1;

            }
        }

        void operator() () const {

            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                         *this );

        }

    };

    void ReconstructionDWI::Scale()
    {
        if (_debug)
            cout << "Scale" << endl;

        ParallelScale_DWI ParallelScale_DWI( this );
        ParallelScale_DWI();


        if (_debug) {
            cout << setprecision(3);
            cout << "Slice scale = ";
            for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex)
                cout << _scale[inputIndex] << " ";
            cout << endl;
        }
    }

    class ParallelBias_DWI {
        ReconstructionDWI* reconstructor;

    public:

        void operator()( const blocked_range<size_t>& r ) const {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {

                RealImage slice = reconstructor->_slices[inputIndex];

                RealImage& w = reconstructor->_weights[inputIndex];

                RealImage b = reconstructor->_bias[inputIndex];

                double scale = reconstructor->_scale[inputIndex];

                RealImage wb = w;
                RealImage m = w;
                m=1;
                RealImage sim = reconstructor->_simulated_slices[inputIndex];

                char buffer[255];

                RealImage wresidual( slice.Attributes() );
                wresidual = 0;
                RealImage residual = wresidual;

                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            if( reconstructor->_simulated_weights[inputIndex](i,j,0) > 0.99 ) {

                                double eb = exp(-b(i, j, 0));
                                slice(i, j, 0) *= (eb * scale);

                                wb(i, j, 0) = w(i, j, 0) * slice(i, j, 0);

                                if ( (reconstructor->_simulated_slices[inputIndex](i, j, 0) > 1) && (slice(i, j, 0) > 1)) {
                                    wresidual(i, j, 0) = log(slice(i, j, 0) / reconstructor->_simulated_slices[inputIndex](i, j, 0)) * wb(i, j, 0);
                                    residual(i, j, 0) = log(slice(i, j, 0) / reconstructor->_simulated_slices[inputIndex](i, j, 0));

                                }
                            }
                            else {

                                wresidual(i, j, 0) = 0;
                                wb(i, j, 0) = 0;
                                residual(i,j,0)=0;
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
                        if (slice(i, j, 0) != -1) {
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
                            if ((slice(i, j, 0) != -1) && (num > 0)) {
                                b(i, j, 0) -= mean;
                            }
                }

                reconstructor->_bias[inputIndex] = b;

            }
        }

        ParallelBias_DWI( ReconstructionDWI *reconstructor ) :
        reconstructor(reconstructor)
        { }

        void operator() () const {

            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                         *this );
        }

    };

    void ReconstructionDWI::Bias()
    {
        if (_debug)
            cout << "Correcting bias ...";

        ParallelBias_DWI ParallelBias_DWI( this );
        ParallelBias_DWI();

        if (_debug)
            cout << "done. " << endl;
    }



    void ReconstructionDWI::SetBiasAlpha()
    {
        double average=0;
        int n=0;
        RealPixel *ps;
        for (int inputIndex=0; inputIndex<_slices.size(); inputIndex++)
        {
            ps = _slices[inputIndex].Data();
            for(int i=0; i<_slices[inputIndex].NumberOfVoxels();i++)
            {
                if (*ps!=-1)
                {
                    average+=*ps;
                    n++;
                }
                ps++;
            }
        }
        average/=n;
        _alpha_bias = 1/(2*average*average);
        cout<<"_alpha_bias = "<<_alpha_bias<<endl;

    }


    void ReconstructionDWI::ExcludeSlicesScale()
    {
        int inputIndex;
        for(inputIndex=0; inputIndex<_slices.size(); inputIndex++)
        {
            if(_scale[inputIndex]>1.25)
                _slice_weight[inputIndex]=0;
        }
    }


    class ParallelSuperresolution_DWI {
        ReconstructionDWI* reconstructor;
    public:
        RealImage confidence_map;
        RealImage addon;

        void operator()( const blocked_range<size_t>& r ) {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {

                RealImage slice = reconstructor->_slices[inputIndex];

                RealImage& w = reconstructor->_weights[inputIndex];

                RealImage& b = reconstructor->_bias[inputIndex];

                double scale = reconstructor->_scale[inputIndex];

                POINT3D p;
                for ( int i = 0; i < slice.GetX(); i++)
                    for ( int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {

                            if(reconstructor->_intensity_matching_GD)
                                slice(i, j, 0) *= b(i, j, 0) * scale;
                            else
                                slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                            if ( reconstructor->_simulated_slices[inputIndex](i,j,0) > 0 )
                                slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                            else
                                slice(i,j,0) = 0;

                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for (int k = 0; k < n; k++) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                if(reconstructor->_robust_slices_only)
                                {
                                    addon(p.x, p.y, p.z) += p.value * slice(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                    confidence_map(p.x, p.y, p.z) += p.value * reconstructor->_slice_weight[inputIndex];

                                }
                                else
                                {
                                    addon(p.x, p.y, p.z) += p.value * slice(i, j, 0) * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                    confidence_map(p.x, p.y, p.z) += p.value * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                }
                            }
                        }
            }
        }

        ParallelSuperresolution_DWI( ParallelSuperresolution_DWI& x, split ) :
        reconstructor(x.reconstructor)
        {

            addon.Initialize( reconstructor->_reconstructed.Attributes() );
            addon = 0;

            confidence_map.Initialize( reconstructor->_reconstructed.Attributes() );
            confidence_map = 0;
        }

        void join( const ParallelSuperresolution_DWI& y ) {
            addon += y.addon;
            confidence_map += y.confidence_map;
        }

        ParallelSuperresolution_DWI( ReconstructionDWI *reconstructor ) :
        reconstructor(reconstructor)
        {

            addon.Initialize( reconstructor->_reconstructed.Attributes() );
            addon = 0;

            confidence_map.Initialize( reconstructor->_reconstructed.Attributes() );
            confidence_map = 0;
        }


        void operator() () {

            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()),
                            *this );

        }
    };

    void ReconstructionDWI::Superresolution(int iter)
    {
        if (_debug)
            cout << "Superresolution " << iter << endl;

        int i, j, k;
        RealImage addon, original;

        original = _reconstructed;

        ParallelSuperresolution_DWI ParallelSuperresolution_DWI(this);
        ParallelSuperresolution_DWI();
        addon = ParallelSuperresolution_DWI.addon;
        _confidence_map = ParallelSuperresolution_DWI.confidence_map;


        if(_debug) {
            char buffer[256];

            _confidence_map.Write("confidence-map.nii.gz");

        }

        if (!_adaptive)
            for (i = 0; i < addon.GetX(); i++)
                for (j = 0; j < addon.GetY(); j++)
                    for (k = 0; k < addon.GetZ(); k++)
                        if (_confidence_map(i, j, k) > 0) {

                            addon(i, j, k) /= _confidence_map(i, j, k);

                            _confidence_map(i,j,k) = 1;
                        }

        _reconstructed += addon * _alpha;

        for (i = 0; i < _reconstructed.GetX(); i++)
            for (j = 0; j < _reconstructed.GetY(); j++)
                for (k = 0; k < _reconstructed.GetZ(); k++) {
                    if (_reconstructed(i, j, k) < _min_intensity * 0.9)
                        _reconstructed(i, j, k) = _min_intensity * 0.9;
                    if (_reconstructed(i, j, k) > _max_intensity * 1.1)
                        _reconstructed(i, j, k) = _max_intensity * 1.1;
                }

        for(i=0;i<_regul_steps;i++)
        {
            AdaptiveRegularization(iter, original);

        }

        if (_global_bias_correction)
            BiasCorrectVolume(original);

    }

    class ParallelMStep_DWI{
        ReconstructionDWI* reconstructor;
    public:
        double sigma;
        double mix;
        double num;
        double min;
        double max;

        void operator()( const blocked_range<size_t>& r ) {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {

                RealImage slice = reconstructor->_slices[inputIndex];

                RealImage& w = reconstructor->_weights[inputIndex];

                RealImage& b = reconstructor->_bias[inputIndex];

                double scale = reconstructor->_scale[inputIndex];

                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {

                            if(reconstructor->_intensity_matching_GD)
                                slice(i, j, 0) *= b(i, j, 0) * scale;
                            else
                                slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                            if ( reconstructor->_simulated_weights[inputIndex](i,j,0) > 0.99 ) {
                                slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);

                                double e = slice(i, j, 0);
                                sigma += e * e * w(i, j, 0);
                                mix += w(i, j, 0);

                                if (e < min)
                                    min = e;
                                if (e > max)
                                    max = e;

                                num++;
                            }
                        }
            }
        }

        ParallelMStep_DWI( ParallelMStep_DWI& x, split ) :
        reconstructor(x.reconstructor)
        {
            sigma = 0;
            mix = 0;
            num = 0;
            min = 0;
            max = 0;
        }

        void join( const ParallelMStep_DWI& y ) {
            if (y.min < min)
                min = y.min;
            if (y.max > max)
                max = y.max;

            sigma += y.sigma;
            mix += y.mix;
            num += y.num;
        }

        ParallelMStep_DWI( ReconstructionDWI *reconstructor ) :
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

    void ReconstructionDWI::MStep(int iter)
    {
        if (_debug)
            cout << "MStep" << endl;

        ParallelMStep_DWI ParallelMStep_DWI(this);
        ParallelMStep_DWI();
        double sigma = ParallelMStep_DWI.sigma;
        double mix = ParallelMStep_DWI.mix;
        double num = ParallelMStep_DWI.num;
        double min = ParallelMStep_DWI.min;
        double max = ParallelMStep_DWI.max;

        if (mix > 0) {
            _sigma = sigma / mix;
        }
        else {
            throw runtime_error("Something went wrong: sigma=" + to_string(sigma) + " mix=" + to_string(mix));
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

    class ParallelAdaptiveRegularization1_DWI {
        ReconstructionDWI *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;

    public:
        ParallelAdaptiveRegularization1_DWI( ReconstructionDWI *_reconstructor,
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
                                && (reconstructor->_mask(x, y, z) > 0) && (reconstructor->_mask(xx, yy, zz) > 0)) {
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

    class ParallelAdaptiveRegularization2_DWI {
        ReconstructionDWI *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;

    public:
        ParallelAdaptiveRegularization2_DWI( ReconstructionDWI *_reconstructor,
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
                        if(reconstructor->_mask(x,y,z)>0)
                        {
                            double val = 0;
                            double sum = 0;
                            for (int i = 0; i < 13; i++) {
                                xx = x + reconstructor->_directions[i][0];
                                yy = y + reconstructor->_directions[i][1];
                                zz = z + reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                                    if(reconstructor->_mask(xx,yy,zz)>0)
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
                                    if(reconstructor->_mask(xx,yy,zz)>0)
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

    void ReconstructionDWI::AdaptiveRegularization(int iter, RealImage& original)
    {
        if (_debug)
            cout << "AdaptiveRegularization: _delta = "<<_delta<<" _lambda = "<<_lambda <<" _alpha = "<<_alpha<< endl;

        Array<double> factor(13,0);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }

        Array<RealImage> b;//(13);
        for (int i = 0; i < 13; i++)
            b.push_back( _reconstructed );

        ParallelAdaptiveRegularization1_DWI ParallelAdaptiveRegularization1_DWI( this,
                                                                        b,
                                                                        factor,
                                                                        original );
        ParallelAdaptiveRegularization1_DWI();

        RealImage original2 = _reconstructed;
        ParallelAdaptiveRegularization2_DWI ParallelAdaptiveRegularization2_DWI( this,
                                                                        b,
                                                                        factor,
                                                                        original2 );
        ParallelAdaptiveRegularization2_DWI();

        if (_alpha * _lambda / (_delta * _delta) > 0.068) {
            cerr
            << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068."
            << endl;
        }
    }

    class ParallelLaplacianRegularization1_DWI {
        ReconstructionDWI *reconstructor;
        RealImage &laplacian;
        Array<double> &factor;
        RealImage &original;

    public:
        ParallelLaplacianRegularization1_DWI( ReconstructionDWI *_reconstructor,
                                         RealImage &_laplacian,
                                         Array<double> &_factor,
                                         RealImage &_original) :
        reconstructor(_reconstructor),
        laplacian(_laplacian),
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
                                && (reconstructor->_mask_internal(x, y, z) > 0)
                                && (reconstructor->_mask(xx, yy, zz) > 0))
                            {
                                diff = (original(xx, yy, zz) - original(x, y, z)) * factor[i];
                                laplacian(x, y, z) += diff;

                            }

                            xx = x - reconstructor->_directions[i][0];
                            yy = y - reconstructor->_directions[i][1];
                            zz = z - reconstructor->_directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                && (reconstructor->_mask_internal(x, y, z) > 0)
                                && (reconstructor->_mask(xx, yy, zz) > 0)) {
                                diff = (original(xx, yy, zz) - original(x, y, z)) * factor[i];
                                laplacian(x, y, z) += diff;

                            }
                        }
            }
        }

        // execute
        void operator() () const {
            parallel_for( blocked_range<size_t>(0, 13),
                         *this );
        }

    };

    class ParallelLaplacianRegularization2_DWI {
        ReconstructionDWI *reconstructor;
        RealImage &laplacian;
        Array<double> &factor;
        RealImage &original;

    public:
        ParallelLaplacianRegularization2_DWI( ReconstructionDWI *_reconstructor,
                                         RealImage &_laplacian,
                                         Array<double> &_factor,
                                         RealImage &_original) :
        reconstructor(_reconstructor),
        laplacian(_laplacian),
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
                        if(reconstructor->_mask(x,y,z)>0)
                        {
                            double val = 0;
                            double sum = 0;
                            for (int i = 0; i < 13; i++) {
                                xx = x + reconstructor->_directions[i][0];
                                yy = y + reconstructor->_directions[i][1];
                                zz = z + reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                                    if(reconstructor->_mask_internal(xx,yy,zz)>0)
                                    {
                                        val += factor[i] * laplacian(xx, yy, zz);
                                        sum += factor[i];
                                    }
                            }

                            for (int i = 0; i < 13; i++) {
                                xx = x - reconstructor->_directions[i][0];
                                yy = y - reconstructor->_directions[i][1];
                                zz = z - reconstructor->_directions[i][2];
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                                    if(reconstructor->_mask_internal(xx,yy,zz)>0)
                                    {
                                        val += factor[i] * laplacian(xx, yy, zz);
                                        sum += factor[i];
                                    }

                            }

                            if(reconstructor->_mask_internal(x,y,z)>0)
                                val -= sum * laplacian(x, y, z);

                            val = original(x, y, z) - reconstructor->_alpha * reconstructor->_lambda / (reconstructor->_delta * reconstructor->_delta) * val/0.068;

                            reconstructor->_reconstructed(x, y, z) = val;
                        }

            }
        }

        void operator() () const {
            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed.GetX()),
                         *this );
        }

    };

    void ReconstructionDWI::LaplacianRegularization(int iter, int t, RealImage& original)
    {
        if (_debug)
            cout << "LaplacianRegularization:  _lambda = "<<_lambda <<" _alpha = "<<_alpha<< endl;
        CreateInternalMask();

        Array<double> factor(13,0);
        double sum=0;
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / sqrt(factor[i]);
            sum+=factor[i];
        }
        for (int i = 0; i < 13; i++) {
            factor[i]/=2*sum;
        }

        RealImage laplacian(_reconstructed);
        laplacian=0;

        ParallelLaplacianRegularization1_DWI ParallelLaplacianRegularization1_DWI( this,
                                                                          laplacian,
                                                                          factor,
                                                                          original );
        ParallelLaplacianRegularization1_DWI();
        char buffer[256];
        sprintf(buffer,"laplacian%i-%i.nii.gz",iter,t);
        laplacian.Write(buffer);

        RealImage original2 = _reconstructed;
        ParallelLaplacianRegularization2_DWI ParallelLaplacianRegularization2_DWI( this,
                                                                          laplacian,
                                                                          factor,
                                                                          original2 );
        ParallelLaplacianRegularization2_DWI();

        if (_alpha * _lambda / (_delta * _delta) > 0.068) {
            cerr
            << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068."
            << endl;
        }
    }



    void ReconstructionDWI::CreateInternalMask()
    {

        _mask_internal = _mask;
        int directions[13][3] = {
            { 1, 0, -1 },
            { 0, 1, -1 },
            { 1, 1, -1 },
            { 1, -1, -1 },
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 1, 1, 0 },
            { 1, -1, 0 },
            { 1, 0, 1 },
            { 0, 1, 1 },
            { 1, 1, 1 },
            { 1, -1, 1 },
            { 0, 0, 1 }};


        int x, y, z, xx, yy, zz, i;
        int dx = _mask.GetX();
        int dy = _mask.GetY();
        int dz = _mask.GetZ();

        for (x = 0; x < dx; x++)
            for (y = 0; y < dy; y++)
                for (z = 0; z < dz; z++)
                    if(_mask_internal(x,y,z)==1)
                    {
                        for (i = 0; i < 13; i++) {
                            xx = x + directions[i][0];
                            yy = y + directions[i][1];
                            zz = z + directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                            {
                                if (_mask(xx,yy,zz)==0)
                                    _mask_internal(x,y,z)=0;
                            }
                            else
                                _mask_internal(x,y,z)=0;
                        }

                        for (i = 0; i < 13; i++) {
                            xx = x - directions[i][0];
                            yy = y - directions[i][1];
                            zz = z - directions[i][2];
                            if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                            {
                                if (_mask(xx,yy,zz)==0)
                                    _mask_internal(x,y,z)=0;
                            }
                            else
                                _mask_internal(x,y,z)=0;

                        }
                    }
        _mask_internal.Write("_mask_internal.nii.gz");
        _mask.Write("mask.nii.gz");
    }


    double ReconstructionDWI::LaplacianSmoothness(RealImage& original)
    {
        CreateInternalMask();

        Array<double> factor(13,0);
        double sum=0;
        int i,j;
        for (i = 0; i < 13; i++) {
            for (j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / sqrt(factor[i]);
            sum+=factor[i];
        }
        for (i = 0; i < 13; i++) {
            factor[i]/=2*sum;
        }

        RealImage laplacian(_reconstructed);
        laplacian=0;

        ParallelLaplacianRegularization1_DWI ParallelLaplacianRegularization1_DWI( this,
                                                                          laplacian,
                                                                          factor,
                                                                          original );
        ParallelLaplacianRegularization1_DWI();

        sum=0;
        int num=0;

        RealPixel* pl = laplacian.Data();
        RealPixel* pm = _mask_internal.Data();
        for(i=0;i<laplacian.NumberOfVoxels(); i++)
        {
            if(*pm==1)
            {
                sum+=*pl * *pl;
                num++;
            }
            pl++;
            pm++;
        }

        return sum;
    }




    class ParallelL22Regularization_DWI {
        ReconstructionDWI *reconstructor;
        Array<double> &factor;
        RealImage &original;

    public:
        ParallelL22Regularization_DWI( ReconstructionDWI *_reconstructor,
                                  Array<double> &_factor,
                                  RealImage &_original) :
        reconstructor(_reconstructor),
        factor(_factor),
        original(_original) { }

        void operator() (const blocked_range<size_t> &r) const {
            int dx = reconstructor->_reconstructed.GetX();
            int dy = reconstructor->_reconstructed.GetY();
            int dz = reconstructor->_reconstructed.GetZ();
            for ( size_t x = r.begin(); x != r.end(); ++x ) {
                int xx, yy, zz, xxx, yyy, zzz;
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
                                xxx = x + reconstructor->_directions[i][0]*2;
                                yyy = y + reconstructor->_directions[i][1]*2;
                                zzz = z + reconstructor->_directions[i][2]*2;
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                    &&(xxx >= 0) && (xxx < dx) && (yyy >= 0) && (yyy < dy) && (zzz >= 0) && (zzz < dz))
                                    if(reconstructor->_confidence_map(xx,yy,zz)>0)
                                    {
                                        val += factor[i] * (original(xxx, yyy, zzz)-4*original(xx, yy, zz))/3;
                                        sum += factor[i];
                                    }
                            }

                            for (int i = 0; i < 13; i++) {
                                xx = x - reconstructor->_directions[i][0];
                                yy = y - reconstructor->_directions[i][1];
                                zz = z - reconstructor->_directions[i][2];
                                xxx = x - reconstructor->_directions[i][0]*2;
                                yyy = y - reconstructor->_directions[i][1]*2;
                                zzz = z - reconstructor->_directions[i][2]*2;
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                    &&(xxx >= 0) && (xxx < dx) && (yyy >= 0) && (yyy < dy) && (zzz >= 0) && (zzz < dz))
                                    if(reconstructor->_confidence_map(xx,yy,zz)>0)
                                    {
                                        val += factor[i] * (original(xxx, yyy, zzz)-4*original(xx, yy, zz))/3;
                                        sum += factor[i];
                                    }

                            }

                            val += sum * original(x, y, z);
                            val = original(x, y, z)
                            - reconstructor->_alpha * reconstructor->_lambda / (reconstructor->_delta * reconstructor->_delta) * val;

                            reconstructor->_reconstructed(x, y, z) = val;
                        }

            }
        }

        // execute
        void operator() () const {

            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed.GetX()),
                         *this );

        }

    };

    void ReconstructionDWI::L22Regularization(int iter, RealImage& original)
    {
        if (_debug)
            //  cout << "AdaptiveRegularization."<< endl;
            cout << "L22Regularization: _delta = "<<_delta<<" _lambda = "<<_lambda <<" _alpha = "<<_alpha<< endl;

        Array<double> factor(13,0);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }

        RealImage original2 = _reconstructed;
        ParallelL22Regularization_DWI ParallelL22Regularization_DWI( this,
                                                            factor,
                                                            original2 );
        ParallelL22Regularization_DWI();

        if (_alpha * _lambda / (_delta * _delta) > 0.068) {
            cerr
            << "Warning: regularization might not have smoothing effect! Ensure that alpha*lambda/delta^2 is below 0.068."
            << endl;
        }
    }




    void ReconstructionDWI::BiasCorrectVolume(RealImage& original)
    {

        RealImage residual = _reconstructed;
        RealImage weights = _mask;

        RealPixel *pr = residual.Data();
        RealPixel *po = original.Data();
        RealPixel *pw = weights.Data();
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {

            if ((*pw == 1) && (*po > _low_intensity_cutoff * _max_intensity)
                && (*pr > _low_intensity_cutoff * _max_intensity)) {
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

    void ReconstructionDWI::Evaluate(int iter)
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


    class ParallelNormaliseBias_DWI{
        ReconstructionDWI* reconstructor;
    public:
        RealImage bias;

        void operator()( const blocked_range<size_t>& r ) {
            for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {

                if(reconstructor->_debug) {
                    cout<<inputIndex<<" ";
                }


                RealImage& slice = reconstructor->_slices[inputIndex];

                RealImage b = reconstructor->_bias[inputIndex];

                double scale = reconstructor->_scale[inputIndex];

                RealPixel *pi = slice.Data();
                RealPixel *pb = b.Data();
                for(int i = 0; i<slice.NumberOfVoxels(); i++) {
                    if((*pi>-1)&&(scale>0))
                        *pb -= log(scale);
                    pb++;
                    pi++;
                }

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

        ParallelNormaliseBias_DWI( ParallelNormaliseBias_DWI& x, split ) :
        reconstructor(x.reconstructor)
        {
            bias.Initialize( reconstructor->_reconstructed.Attributes() );
            bias = 0;
        }

        void join( const ParallelNormaliseBias_DWI& y ) {
            bias += y.bias;
        }

        ParallelNormaliseBias_DWI( ReconstructionDWI *reconstructor ) :
        reconstructor(reconstructor)
        {
            bias.Initialize( reconstructor->_reconstructed.Attributes() );
            bias = 0;
        }

        // execute
        void operator() () {

            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()),
                            *this );

        }
    };


    void ReconstructionDWI::NormaliseBias(int iter)
    {
        if(_debug)
            cout << "Normalise Bias ... ";

        ParallelNormaliseBias_DWI ParallelNormaliseBias_DWI(this);
        ParallelNormaliseBias_DWI();
        RealImage bias = ParallelNormaliseBias_DWI.bias;

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
        pi = _reconstructed.Data();
        pb = bias.Data();
        for (int i = 0; i<_reconstructed.NumberOfVoxels();i++) {
            if(*pi!=-1)
                *pi /=exp(-(*pb));
            pi++;
            pb++;
        }
    }

    void ReconstructionDWI::NormaliseBias(int iter, RealImage& image)
    {
        if(_debug)
            cout << "Normalise Bias ... ";

        ParallelNormaliseBias_DWI ParallelNormaliseBias_DWI(this);
        ParallelNormaliseBias_DWI();
        RealImage bias = ParallelNormaliseBias_DWI.bias;

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

        for(int t=0;t<image.GetT();t++)
        {
            RealImage stack = image.GetRegion(0,0,0,t,image.GetX(),image.GetY(),image.GetZ(),t+1);
            RealPixel *pi, *pb;
            pi = stack.Data();
            pb = bias.Data();
            for (int i = 0; i<stack.NumberOfVoxels();i++)
            {
                if(*pi!=-1)
                    *pi /=exp(-(*pb));
                pi++;
                pb++;
            }

            for (int i = 0; i < image.GetX(); i++)
                for (int j = 0; j < image.GetY(); j++)
                    for (int k = 0; k < image.GetZ(); k++)
                        image(i,j,k,t)=stack(i,j,k);

        }


    }


    void ReconstructionDWI::ReadTransformations(char* folder)
    {
        int n = _slices.size();
        char name[256];
        char path[256];
//        Transformation *transformation;
        RigidTransformation *rigidTransf;

        if (n == 0)
            throw runtime_error("Please create slices before reading transformations!");

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
//            transformation = irtkTransformation::New(path);
            // TODO: Fix this bug of rigidTransf being a NULL pointer, and called before initialised
            rigidTransf->Read(path);
            _transformations.push_back(*rigidTransf);

            cout << path << endl;
        }
    }

    void ReconstructionDWI::SetReconstructed( RealImage &reconstructed )
    {
        _reconstructed = reconstructed;
        _template_created = true;
    }

    void ReconstructionDWI::SetTransformations( Array<RigidTransformation>& transformations )
    {
        _transformations.clear();
        for ( int i = 0; i < transformations.size(); i++ )
            _transformations.push_back(transformations[i]);

    }

    void ReconstructionDWI::SaveBiasFields()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "bias%i.nii.gz", inputIndex);
            _bias[inputIndex].Write(buffer);
        }
    }

    void ReconstructionDWI::SaveConfidenceMap()
    {
        _confidence_map.Write("confidence-map.nii.gz");
    }

    void ReconstructionDWI::SaveSlices()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            sprintf(buffer, "slice-%i-%i.nii.gz", _stack_index[inputIndex], inputIndex);
            _slices[inputIndex].Write(buffer);
        }
    }

    void ReconstructionDWI::SaveSimulatedSlices()
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

    void ReconstructionDWI::SaveWeights()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "weights%i.nii.gz", inputIndex);
            _weights[inputIndex].Write(buffer);
        }
    }

    void ReconstructionDWI::SaveTransformations()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "transformation%i.dof", inputIndex);
            _transformations[inputIndex].Write(buffer);
        }
    }

    void ReconstructionDWI::GetTransformations( Array<RigidTransformation> &transformations )
    {
        transformations.clear();
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            transformations.push_back( _transformations[inputIndex] );
    }

    void ReconstructionDWI::GetSlices( Array<RealImage> &slices )
    {
        slices.clear();
        for (unsigned int i = 0; i < _slices.size(); i++)
            slices.push_back(_slices[i]);
    }

    void ReconstructionDWI::SetSlices( Array<RealImage> &slices )
    {
        _slices.clear();
        for (unsigned int i = 0; i < slices.size(); i++)
            _slices.push_back(slices[i]);
    }


    void ReconstructionDWI::SlicesInfo( const char* filename,
                                        Array<string> &stack_files )
    {
        ofstream info;
        info.open( filename );

        // header
        info << "slice_index" << "\t"
        << "stack_index" << "\t"
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
            << i << "\t"
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

    /* end Set/Get/Save operations */

    /* Package specific functions */
    void ReconstructionDWI::SplitImage(RealImage image, int packages, Array<RealImage>& stacks)
    {
        ImageAttributes attr = image.Attributes();

        //slices in package
        int pkg_z = attr._z/packages;
        double pkg_dz = attr._dz*packages;
        //cout<<"packages: "<<packages<<"; slices: "<<attr._z<<"; slices in package: "<<pkg_z<<endl;
        //cout<<"slice thickness "<<attr._dz<<"; slickess thickness in package: "<<pkg_dz<<endl;

        char buffer[256];
        int i,j,k,l;
        double x,y,z,sx,sy,sz,ox,oy,oz;
        for(l=0;l<packages;l++) {
            attr = image.Attributes();
            if((pkg_z*packages+l)<attr._z)
                attr._z = pkg_z+1;
            else
                attr._z = pkg_z;
            attr._dz = pkg_dz;

            cout<<"split image "<<l<<" has "<<attr._z<<" slices."<<endl;

            //fill values in each stack
            RealImage stack(attr);
            stack.GetOrigin(ox,oy,oz);

            cout<<"Stack "<<l<<":"<<endl;
            for(k=0; k<stack.GetZ();k++)
                for(j=0; j<stack.GetY();j++)
                    for(i=0; i<stack.GetX();i++)
                        stack.Put(i,j,k,image(i,j,k*packages+l));

            //adjust origin

            //original image coordinates
            x=0;y=0;z=l;
            image.ImageToWorld(x,y,z);
            cout<<"image: "<<x<<" "<<y<<" "<<z<<endl;
            //stack coordinates
            sx=0;sy=0;sz=0;
            stack.PutOrigin(ox,oy,oz); //adjust to original value
            stack.ImageToWorld(sx,sy,sz);
            cout<<"stack: "<<sx<<" "<<sy<<" "<<sz<<endl;
            //adjust origin
            cout<<"adjustment needed: "<<x-sx<<" "<<y-sy<<" "<<z-sz<<endl;
            stack.PutOrigin(ox + (x-sx), oy + (y-sy), oz + (z-sz));
            sx=0;sy=0;sz=0;
            stack.ImageToWorld(sx,sy,sz);
            cout<<"adjusted: "<<sx<<" "<<sy<<" "<<sz<<endl;

            //sprintf(buffer,"stack%i.nii.gz",l);
            //stack.Write(buffer);
            stacks.push_back(stack);
        }
        cout<<"done."<<endl;

    }


    void ReconstructionDWI::PackageToVolume( Array<RealImage>& stacks, Array<int> &pack_num, Array<RigidTransformation> stack_transformations )
    {


        int firstSlice = 0;
        Array<int> firstSlice_array;
        for (int i = 0; i < stacks.size(); i++) {
            firstSlice_array.push_back(firstSlice);
            firstSlice += stacks[i].GetZ();
        }

        //        cout<<"Package to volume: "<<endl;
        {

            for (int i = 0; i < stacks.size(); i++) {

                char buffer[256];

                Array<RealImage> packages;

                SplitImage(stacks[i],pack_num[i],packages);


                for (int j = 0; j < packages.size(); j++) {

                    GenericRegistrationFilter rigidregistration;
                    ParameterList params;
                    Insert(params, "Transformation model", "Rigid");
                    // Insert(params, "Image interpolation mode", "Linear");
                    Insert(params, "Background value for image 1", 0);
                    Insert(params, "Background value for image 2", -1);



                    Insert(params, "Image (dis-)similarity measure", "NCC");
                    string type = "box";
                    string units = "vox";
                    double width = 0;
                    Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);



//                    if (_nmi_bins>0)
//                        Insert(params, "No. of bins", _nmi_bins);

                    rigidregistration.Parameter(params);

                    RealImage target, source;

                    ImageAttributes attr, attr2;

                    if (_debug) {
                        sprintf(buffer,"package-%i-%i.nii.gz",i,j);
                        packages[j].Write(buffer);
                    }

                    attr2=packages[j].Attributes();
                    //packages are not masked at present

                    target=packages[j];

                    RealImage mask = _mask;
                    RigidTransformation mask_transform = stack_transformations[i]; //s[i];
                    TransformMask(target, mask, mask_transform);
                    target = target*mask;

                    source=_reconstructed;

                    //find existing transformation
                    double x,y,z;
                    x=0; y=0; z=0;
                    packages[j].ImageToWorld(x,y,z);
                    stacks[i].WorldToImage(x,y,z);

                    int firstSliceIndex = round(z)+firstSlice_array[i];
                    // cout<<"First slice index for package "<<j<<" of stack "<<i<<" is "<<firstSliceIndex<<endl;

                    //put origin in target to zero
                    RigidTransformation offset;
                    ResetOrigin(target, offset);
                    Matrix mo = offset.GetMatrix();
                    Matrix m = _transformations[firstSliceIndex].GetMatrix();
                    m=m*mo;
                    _transformations[firstSliceIndex].PutMatrix(m);


                    rigidregistration.Input(&target, &source);

                    Transformation *dofout = nullptr;
                    rigidregistration.Output(&dofout);


                    RigidTransformation tmp_dofin = _transformations[firstSliceIndex]; //offset;

                    rigidregistration.InitialGuess(&tmp_dofin);
                    rigidregistration.GuessParameter();
                    rigidregistration.Run();

                    unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(dofout));
                    _transformations[firstSliceIndex] = *rigidTransf;

                    //undo the offset
                    mo.Invert();
                    m = _transformations[firstSliceIndex].GetMatrix();
                    m=m*mo;
                    _transformations[firstSliceIndex].PutMatrix(m);

                    if (_debug) {
                        sprintf(buffer,"transformation-%i-%i.dof",i,j);
                        _transformations[firstSliceIndex].Write(buffer);
                    }

                    //set the transformation to all slices of the package
                    // cout<<"Slices of the package "<<j<<" of the stack "<<i<<" are: ";
                    for (int k = 0; k < packages[j].GetZ(); k++) {
                        x=0;y=0;z=k;
                        packages[j].ImageToWorld(x,y,z);
                        stacks[i].WorldToImage(x,y,z);
                        int sliceIndex = round(z)+firstSlice_array[i];
                        // cout<<sliceIndex<<" "<<endl;

                        if (sliceIndex >= _transformations.size())
                            throw runtime_error("Reconstruction::PackageToVolume: sliceIndex out of range.\n" + to_string(sliceIndex) + " " + to_string(_transformations.size()));

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
                // cout<<"End of stack "<<i<<endl<<endl;

            }

        }




    }





    void ReconstructionDWI::CropImage(RealImage& image, RealImage& mask, bool cropz)
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
                    if (mask(i, j, k) > 0)
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
                    if (mask(i, j, k) > 0)
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
                    if (mask(i, j, k) > 0)
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
                    if (mask(i, j, k) > 0)
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
                    if (mask(i, j, k) > 0)
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
                    if (mask(i, j, k) > 0)
                        sum++;
            if (sum > 0)
                break;
        }

        x1 = i;

        if (_debug)
            cout << "Region of interest is " << x1 << " " << y1 << " " << z1 << " " << x2 << " " << y2
            << " " << z2 << endl;

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
        if(cropz)
            image = image.GetRegion(x1, y1, z1, x2+1, y2+1, z2+1);
        else
        {
            image = image.GetRegion(x1, y1, 0, x2+1, y2+1, image.GetZ());
        }
    }

    void ReconstructionDWI::InvertStackTransformations( Array<RigidTransformation>& stack_transformations)
    {
        //for each stack
        for (unsigned int i = 0; i < stack_transformations.size(); i++) {
            //invert transformation for the stacks
            stack_transformations[i].Invert();
            stack_transformations[i].UpdateParameter();
            //stack_transformations[i].Print();
        }
    }

    void ReconstructionDWI::MaskVolume()
    {
        RealPixel *pr = _reconstructed.Data();
        RealPixel *pm = _mask.Data();
        for (int i = 0; i < _reconstructed.NumberOfVoxels(); i++) {
            if (*pm == 0)
                *pr = -1;
            pm++;
            pr++;
        }
    }

    void ReconstructionDWI::MaskImage(RealImage& image, double padding)
    {
        if (image.NumberOfVoxels() != _mask.NumberOfVoxels())
            throw runtime_error("Cannot mask the image - different dimensions");

        RealPixel *pr = image.Data();
        RealPixel *pm = _mask.Data();
        for (int i = 0; i < image.NumberOfVoxels(); i++) {
            if (*pm == 0)
                *pr = padding;
            pm++;
            pr++;
        }
    }

    /// Like PutMinMax but ignoring negative values (mask)
    void ReconstructionDWI::Rescale( RealImage &img, double max)
    {
        int i, n;
        RealPixel *ptr, min_val, max_val;

        // Get lower and upper bound
        img.GetMinMax(&min_val, &max_val);

        n   = img.NumberOfVoxels();
        ptr = img.Data();
        for (i = 0; i < n; i++)
            if ( ptr[i] > 0 )
                ptr[i] = double(ptr[i]) / double(max_val) * max;
    }








    void ReconstructionDWI::Init4D(int nStacks)
    {
        ImageAttributes attr = _reconstructed.Attributes();
        attr._t = nStacks;
        cout<<nStacks<<" stacks."<<endl;
        RealImage recon4D(attr);


        for(int t=0; t<recon4D.GetT();t++)
            for(int k=0; k<recon4D.GetZ();k++)
                for(int j=0; j<recon4D.GetY();j++)
                    for(int i=0; i<recon4D.GetX();i++)
                    {
                        recon4D(i,j,k,t)=_reconstructed(i,j,k);
                    }

        recon4D.Write("recon4D.nii.gz");
        _simulated_signal = recon4D;
    }

    void ReconstructionDWI::Init4DGauss(int nStacks, Array<RigidTransformation> &stack_transformations, int order)
    {
        unsigned int inputIndex;
        int i, j, k, t, n;
        RealImage slice;
        double scale;
        POINT3D p;

        ImageAttributes attr = _reconstructed.Attributes();
        attr._t = nStacks;
        cout<<nStacks<<" stacks."<<endl;
        RealImage recon4D(attr);
        RealImage weights(attr);

        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            //copy the current slice
            slice = _slices[inputIndex];
            //alias the current bias image
            RealImage& b = _bias[inputIndex];
            //read current scale factor
            scale = _scale[inputIndex];
            //cout<<scale<<" ";

            //Distribute slice intensities to the volume
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {

                        //biascorrect and scale the slice
                        //if(origDir==1)
                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                        //number of volume voxels with non-zero coefficients
                        //for current slice voxel
                        n = _volcoeffs[inputIndex][i][j].size();

                        //add contribution of current slice voxel to all voxel volumes
                        //to which it contributes
                        for (k = 0; k < n; k++) {
                            p = _volcoeffs[inputIndex][i][j][k];
                            recon4D(p.x, p.y, p.z, _stack_index[inputIndex]) += _slice_weight[inputIndex] * p.value * slice(i, j, 0);
                            weights(p.x, p.y, p.z, _stack_index[inputIndex]) += _slice_weight[inputIndex] * p.value;
                        }
                    }
            //} //end of loop for origDir

        }

        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxe
        recon4D.Write("recon4D.nii.gz");
        recon4D /= weights;
        recon4D.Write("recon4DGaussian.nii.gz");

        // rotate directions
        int dirIndex;
        double gx,gy,gz;
        Matrix dirs_xyz(nStacks,3);
        for (dirIndex = 0; dirIndex < nStacks; dirIndex++)
        {
            gx=_directionsDTI[0][dirIndex+1];
            gy=_directionsDTI[1][dirIndex+1];
            gz=_directionsDTI[2][dirIndex+1];
            RotateDirections(gx,gy,gz,stack_transformations[dirIndex]);
            dirs_xyz(dirIndex,0) = gx;
            dirs_xyz(dirIndex,1) = gy;
            dirs_xyz(dirIndex,2) = gz;
        }
        SphericalHarmonics sh;
        sh.InitSHT(dirs_xyz,order);
        _SH_coeffs = sh.Signal2Coeff(recon4D);
        _SH_coeffs.Write("_SH_coeffs.nii.gz");
        SimulateSignal();
    }


    void ReconstructionDWI::Init4DGaussWeighted(int nStacks, Array<RigidTransformation> &stack_transformations, int order)
    {
        unsigned int inputIndex;
        int i, j, k, t, n;
        RealImage slice;
        double scale;
        POINT3D p;

        ImageAttributes attr = _reconstructed.Attributes();
        attr._t = nStacks;
        cout<<nStacks<<" stacks."<<endl;
        RealImage recon4D(attr);
        RealImage weights(attr);

        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            //copy the current slice
            slice = _slices[inputIndex];
            //alias the current bias image
            RealImage& b = _bias[inputIndex];
            //read current scale factor
            scale = _scale[inputIndex];
            //cout<<scale<<" ";

            //Distribute slice intensities to the volume
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {

                        //biascorrect and scale the slice
                        //if(origDir==1)
                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                        //number of volume voxels with non-zero coefficients
                        //for current slice voxel
                        n = _volcoeffs[inputIndex][i][j].size();

                        //add contribution of current recon-test4D-gauss-weightedslice voxel to all voxel volumes
                        //to which it contributes
                        for (k = 0; k < n; k++) {
                            p = _volcoeffs[inputIndex][i][j][k];
                            recon4D(p.x, p.y, p.z, _stack_index[inputIndex]) += _slice_weight[inputIndex] * p.value * slice(i, j, 0);
                            weights(p.x, p.y, p.z, _stack_index[inputIndex]) += _slice_weight[inputIndex] * p.value;
                        }
                    }
            //} //end of loop for origDir

        }

        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxe
        recon4D.Write("recon4D.nii.gz");
        recon4D /= weights;
        recon4D.Write("recon4DGaussian.nii.gz");

        // rotate directions
        int dirIndex;
        double gx,gy,gz;
        Matrix dirs_xyz(nStacks,3);
        for (dirIndex = 0; dirIndex < nStacks; dirIndex++)
        {
            gx=_directionsDTI[0][dirIndex+1];
            gy=_directionsDTI[1][dirIndex+1];
            gz=_directionsDTI[2][dirIndex+1];
            RotateDirections(gx,gy,gz,stack_transformations[dirIndex]);
            dirs_xyz(dirIndex,0) = gx;
            dirs_xyz(dirIndex,1) = gy;
            dirs_xyz(dirIndex,2) = gz;
        }
        SphericalHarmonics sh;
        //sh.InitSHT(dirs_xyz,order);
        //_SH_coeffs = sh.Signal2Coeff(recon4D);
        //_SH_coeffs.Write("_SH_coeffs.nii.gz");
        //SimulateSignal();

        Matrix SHT = sh.SHbasis(dirs_xyz,order);
        Matrix tSHT = SHT;
        tSHT.Transpose();

        attr._t=SHT.Cols();
        RealImage shc(attr);
        Vector s(nStacks),c,ev;
        Matrix w(nStacks,nStacks);
        Matrix tmp;
        Matrix tmp1,tmp2;
        RealImage mask = _reconstructed;
        mask=0;

        double det;
        for(i=0;i<shc.GetX();i++)
            for(j=0;j<shc.GetY();j++)
                for(k=0;k<shc.GetZ();k++)
                {
                    for(t=0;t<nStacks;t++)
                    {
                        s(t)=recon4D(i,j,k,t);
                        w(t,t)=weights(i,j,k,t);
                    }

                    tmp = tSHT*w*SHT;
                    tmp.Eigenvalues(tmp1,ev,tmp2);

                    det=1;
                    for(t=0;t<ev.Rows();t++)
                        det*=ev(t);
                    //cerr<<det<<" ";
                    if(det>0.001)
                    {
                        mask(i,j,k)=1;
                        tmp.Invert();
                        c = tmp*tSHT*w*s;
                        for(t=0;t<SHT.Cols();t++)
                        {
                            shc(i,j,k,t)=c(t);
                        }
                    }
                }

        _SH_coeffs = shc;
        _SH_coeffs.Write("_SH_coeffs.nii.gz");
        mask.Write("shmask4D.nii.gz");
        SimulateSignal();

    }



    void ReconstructionDWI::SliceAverage(Array<RigidTransformation> stack_transformations, RealImage mask, Array<double>& average_values, Array<int>& areas, Array<double>& errors)
    {

        for (int i = 0; i < _slices.size(); i++) {
            average_values.push_back(0.0);
            areas.push_back(0);
            errors.push_back(0.0);
        }

        for (int i = 0; i < _slices.size(); i++) {

            RealImage m = mask;
            TransformMask(_slices[i], m, stack_transformations[_stack_index[i]]);
            RealImage slice = _slices[i]*m;

            double val = 0;
            int num = 0;
            double error = 0;

            for (int x=0; x<slice.GetX(); x++) {
                for (int y=0; y<slice.GetY(); y++) {

                    if (slice(x,y,0)>0) {
                        val = val + slice(x,y,0);
                        num = num + 1;

                        error = error + abs(_simulated_slices[i](x,y,0)-slice(x,y,0)*exp(-(_bias[i](x,y,0)))*_scale[i]);

                    }

                }
            }

            if (num>0) {
                val = val / num;
                error = error / num;
            }
            average_values[i] = val;
            areas[i] = num;
            errors[i] = error;
        }


    }


    void ReconstructionDWI::StructuralExclusion(Array<RigidTransformation> stack_transformations, RealImage mask, Array<RealImage> stacks, double intensity_threshold)
    {

        char buffer[255];

        for (int s = 0; s < stacks.size(); s++) {

            cout << s << " ....................................................... " << endl;


            for (int i = 0; i < _slices.size(); i++) {

                if (_stack_index[i] == s) {

                    RealImage m = mask;
                    TransformMask(_slices[i], m, stack_transformations[_stack_index[i]]);
                    RealImage slice = _slices[i]*m;

                    double val = 0;
                    int num = 0;

                    for (int x=0; x<slice.GetX(); x++) {
                        for (int y=0; y<slice.GetY(); y++) {

                            if (slice(x,y,0)>0) {
                                val = val + slice(x,y,0);
                                num = num + 1;

                            }

                            stacks[s](x,y,_stack_locs[i]) = slice(x,y,0);

                        }
                    }

                    if (num>0) {
                        val = val / num;
                    }

                    if (val < intensity_threshold && num > 100) {

                        _force_excluded.push_back(i);
                        _intensity_weights[i] = 1;

                        cout << i << " - " << _stack_locs[i] << " (" << _stack_index[i] << ") - " << val << " " << num << " -> " <<  "excluded" << endl;

                    } else {
                        _intensity_weights[i] = 1;
                    }



                }


            }

//            sprintf(buffer,"masked-tmp-%i.nii.gz", s);
//            stacks[s].Write(buffer);


        }

    }

    void ReconstructionDWI::SlicesInfo2(Array<RigidTransformation> stack_transformations, RealImage mask)
    {
        Array<double> errors;
        Array<double> average_values;
        Array<int> areas;
        SliceAverage(stack_transformations, mask, average_values, areas, errors);

        ofstream info;
        info.open( "info.csv" );

        // header
        info << "slice_index" << ","
        << "stack_index" << ","
        << "stack_loc" << ","
        << "included" << "," // Included slices
        << "excluded" << ","  // Excluded slices
        << "outside" << ","  // Outside slices
        << "weight" << ","
        << "scale" << ","
        // << _stack_factor[i] << ","
        << "TranslationX" << ","
        << "TranslationY" << ","
        << "TranslationZ" << ","
        << "RotationX" << ","
        << "RotationY" << ","
        << "RotationZ" << ","
        << "Error" << ","
        << "Area" << ","
        << "Min" << ","
        << "Mean" << ","
        << "Max" << endl;

        for (int i = 0; i < _slices.size(); i++) {

            double smin, smax;
            _slices[i].GetMinMax(&smin,&smax);

            double average = average_values[i];
            int area = areas[i];
            double error = errors[i];


            RigidTransformation& t = _transformations[i];
            info << _stack_index[i] << ","
            << i << ","
            << _stack_locs[i] << ","
            << (((_slice_weight[i] >= 0.5) && (_slice_inside[i]))?1:0) << "," // Included slices
            << (((_slice_weight[i] < 0.5) && (_slice_inside[i]))?1:0) << ","  // Excluded slices
            << ((!(_slice_inside[i]))?1:0) << ","  // Outside slices
            << _slice_weight[i] << ","
            << _scale[i] << ","
            // << _stack_factor[i] << ","
            << t.GetTranslationX() << ","
            << t.GetTranslationY() << ","
            << t.GetTranslationZ() << ","
            << t.GetRotationX() << ","
            << t.GetRotationY() << ","
            << t.GetRotationZ() << ","
            << error << ","
            << area << ","
            << smin << ","
            << average << ","
            << smax << endl;

        }

        info.close();
    }





    void ReconstructionDWI::GaussianReconstruction4D(int nStacks, Array<RigidTransformation> &stack_transformations, int order)
    {
        cout << "Gaussian reconstruction ... ";
        unsigned int inputIndex;
        int i, j, k, t, n;
        RealImage slice;
        double scale;
        POINT3D p;
        int dirIndex;
        double gx,gy,gz;
        ImageAttributes attr = _reconstructed.Attributes();


        //create SH matrix
        Matrix dirs_xyz(nStacks,3);
        Matrix SHT;
        for (dirIndex = 0; dirIndex < nStacks; dirIndex++)
        {
            gx=_directionsDTI[0][dirIndex+1];
            gy=_directionsDTI[1][dirIndex+1];
            gz=_directionsDTI[2][dirIndex+1];
            RotateDirections(gx,gy,gz,stack_transformations[dirIndex]);
            dirs_xyz(dirIndex,0) = gx;
            dirs_xyz(dirIndex,1) = gy;
            dirs_xyz(dirIndex,2) = gz;
        }
        SphericalHarmonics sh;
        sh.InitSHT(dirs_xyz,order);
        SHT = sh.SHbasis(dirs_xyz,order);
        Matrix tSHT = SHT;
        tSHT.Transpose();
        SHT.Print();
        tSHT.Print();
        attr._t=SHT.Cols();
        RealImage shc(attr);
        //shc.Write("shc.nii.gz");



        // Calculate estimate of the signal

        attr._t = nStacks;
        cout<<nStacks<<" stacks."<<endl;
        RealImage recon4D(attr);
        RealImage weights(attr);

        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            //copy the current slice
            slice = _slices[inputIndex];
            //alias the current bias image
            RealImage& b = _bias[inputIndex];
            //read current scale factor
            scale = _scale[inputIndex];
            //cout<<scale<<" ";

            //Distribute slice intensities to the volume
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {

                        //biascorrect and scale the slice
                        //if(origDir==1)
                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                        //number of volume voxels with non-zero coefficients
                        //for current slice voxel
                        n = _volcoeffs[inputIndex][i][j].size();

                        //add contribution of current slice voxel to all voxel volumes
                        //to which it contributes
                        for (k = 0; k < n; k++) {
                            p = _volcoeffs[inputIndex][i][j][k];
                            recon4D(p.x, p.y, p.z, _stack_index[inputIndex]) += _slice_weight[inputIndex] * p.value * slice(i, j, 0);
                            weights(p.x, p.y, p.z, _stack_index[inputIndex]) += _slice_weight[inputIndex] * p.value;
                        }
                    }
            //} //end of loop for origDir

        }

        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxe
        recon4D.Write("recon4D.nii.gz");
        recon4D /= weights;
        _simulated_signal = recon4D;
        RealImage mask = recon4D;
        mask=0;
        RealImage  _test_sh_coeffs = sh.Signal2Coeff(recon4D);
        _test_sh_coeffs.Write("_test_sh_coeffs.nii.gz");

        cout << "done." << endl;

        if (_debug)
        {
            recon4D.Write("recon4Dgaussian.nii.gz");
            weights.Write("weights.nii.gz");
        }

        //Calculate SH coeffs using the estimated signal and weights
        Vector s(nStacks),c,ev;
        Matrix w(nStacks,nStacks);
        Matrix tmp;
        Matrix tmp1,tmp2;

        double det;
        //cout<<"SHT:"<<endl;
        //SHT.Print();
        for(i=0;i<shc.GetX();i++)
            for(j=0;j<shc.GetY();j++)
                for(k=0;k<shc.GetZ();k++)
                {
                    for(t=0;t<nStacks;t++)
                    {
                        s(t)=recon4D(i,j,k,t);
                        w(t,t)=weights(i,j,k,t);
                    }

                    tmp = tSHT*w*SHT;
                    tmp.Eigenvalues(tmp1,ev,tmp2);

                    det=1;
                    for(t=0;t<ev.Rows();t++)
                        det*=ev(t);
                    //cerr<<det<<" ";
                    if(det>0.001)
                    {
                        mask(i,j,k)=1;
                        tmp.Invert();
                        c = tmp*tSHT*w*s;
                        for(t=0;t<SHT.Cols();t++)
                        {
                            shc(i,j,k,t)=c(t);
                        }
                    }
                }

        _SH_coeffs = shc;
        _SH_coeffs.Write("_SH_coeffs.nii.gz");
        mask.Write("shmask4D.nii.gz");
    }

    void ReconstructionDWI::GaussianReconstruction4D2(int nStacks)
    {
        cout << "Gaussian reconstruction ... ";
        unsigned int inputIndex;
        int i, j, k, n;
        RealImage slice;
        double scale;
        POINT3D p;
        int dirIndex, origDir;
        double bval,gx,gy,gz,dx,dy,dz,dotp,sigma=0.02,w,tw;

        ImageAttributes attr = _reconstructed.Attributes();
        attr._t = nStacks;
        attr._dt = 1;
        cout<<nStacks<<" stacks."<<endl;
        RealImage recon4D(attr);
        RealImage weights(attr);

        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            //copy the current slice
            slice = _slices[inputIndex];
            //alias the current bias image
            RealImage& b = _bias[inputIndex];
            //read current scale factor
            scale = _scale[inputIndex];
            cout<<scale<<" ";

            //direction for current slice
            dirIndex = _stack_index[inputIndex]+1;
            gx=_directionsDTI[0][dirIndex];
            gy=_directionsDTI[1][dirIndex];
            gz=_directionsDTI[2][dirIndex];
            RotateDirections(gx,gy,gz,inputIndex);
            bval=_bvalues[dirIndex];

            for (origDir=1; origDir<_bvalues.size();origDir++)
            {
                //cout<<origDir<<" ";

                dx=_directionsDTI[0][origDir];
                dy=_directionsDTI[1][origDir];
                dz=_directionsDTI[2][origDir];

                dotp = (dx*gx+dy*gy+dz*gz)/sqrt((dx*dx+dy*dy+dz*dz)*(gx*gx+gy*gy+gz*gz));
                //cout<<"origDir="<<origDir<<": "<<dotp<<"; ";
                tw = (fabs(dotp)-1)/sigma;
                w=exp(-tw*tw)/(6.28*sigma);
                //cout<<"weight = "<<w<<"; ";


                //Distribute slice intensities to the volume
                for (i = 0; i < slice.GetX(); i++)
                    for (j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {

                            //biascorrect and scale the slice
                            if(origDir==1)
                                slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                            //number of volume voxels with non-zero coefficients
                            //for current slice voxel
                            n = _volcoeffs[inputIndex][i][j].size();

                            //add contribution of current slice voxel to all voxel volumes
                            //to which it contributes
                            for (k = 0; k < n; k++) {
                                p = _volcoeffs[inputIndex][i][j][k];
                                recon4D(p.x, p.y, p.z,origDir-1) += _slice_weight[inputIndex] * w * p.value * slice(i, j, 0);
                                weights(p.x, p.y, p.z,origDir-1) += _slice_weight[inputIndex] * w * p.value;
                            }
                        }
            } //end of loop for origDir
            //cout<<endl;
            //end of loop for a slice inputIndex
            //sprintf(buffer, "corslice%i.nii.gz", inputIndex);
            //slice

        }

        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxe
        recon4D.Write("recon4D.nii.gz");
        recon4D /= weights;
        _simulated_signal = recon4D;

        cout << "done." << endl;

        if (_debug)
        {
            recon4D.Write("recon4Dgaussian.nii.gz");
            weights.Write("weights.nii.gz");
        }

    }

    void ReconstructionDWI::GaussianReconstruction4D3()
    {
        cout << "Gaussian reconstruction SH ... ";
        unsigned int inputIndex;
        int i, j, k, n;
        RealImage slice;
        double scale;
        POINT3D p;
        int dirIndex;
        double bval,gx,gy,gz;

        ImageAttributes attr = _reconstructed.Attributes();
        attr._t = _coeffNum;
        cout<<_coeffNum<<" SH coefficients."<<endl;
        RealImage recon4D(attr);
        RealImage weights(attr);

        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            //copy the current slice
            slice = _slices[inputIndex];
            //alias the current bias image
            RealImage& b = _bias[inputIndex];
            //read current scale factor
            scale = _scale[inputIndex];
            //cout<<scale<<" ";

            //direction for current slice
            dirIndex = _stack_index[inputIndex]+1;
            gx=_directionsDTI[0][dirIndex];
            gy=_directionsDTI[1][dirIndex];
            gz=_directionsDTI[2][dirIndex];
            RotateDirections(gx,gy,gz,inputIndex);
            bval=_bvalues[dirIndex];

            SphericalHarmonics sh;
            Matrix dir(1,3);
            dir(0,0)=gx;
            dir(0,1)=gy;
            dir(0,2)=gz;
            Matrix basis = sh.SHbasis(dir,_order);
            if (basis.Cols() != recon4D.GetT())
                throw runtime_error("GaussianReconstruction4D3:basis numbers does not match SH coefficients number.");

            //Distribute slice intensities to the volume
            for (i = 0; i < slice.GetX(); i++)
                for (j = 0; j < slice.GetY(); j++)
                    if (slice(i, j, 0) != -1) {

                        //biascorrect and scale the slice
                        slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                        //number of volume voxels with non-zero coefficients
                        //for current slice voxel
                        n = _volcoeffs[inputIndex][i][j].size();

                        //add contribution of current slice voxel to all voxel volumes
                        //to which it contributes
                        for (k = 0; k < n; k++) {
                            p = _volcoeffs[inputIndex][i][j][k];
                            for(unsigned int l = 0; l < basis.Cols(); l++ )
                            {
                                if(l==0)
                                {
                                    recon4D(p.x, p.y, p.z,l) += basis(0,l) *_slice_weight[inputIndex] * p.value * slice(i, j, 0);
                                    weights(p.x, p.y, p.z,l) += basis(0,l) * _slice_weight[inputIndex] * p.value;
                                }
                            }
                        }
                    }
            //end of loop for a slice inputIndex
        }

        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxe
        recon4D.Write("recon4D.nii.gz");
        recon4D /= weights;
        _SH_coeffs = recon4D;

        cout << "done." << endl;

        if (_debug)
        {
            recon4D.Write("recon4DgaussianSH.nii.gz");
            weights.Write("weights.nii.gz");
        }

    }

    void ReconstructionDWI::SimulateStacksDTI(Array<RealImage>& stacks, bool simulate_excluded)
    {
        if (_debug)
            cout<<"Simulating stacks."<<endl;
        cout.flush();

        unsigned int inputIndex;
        int i, j, k, n;
        RealImage sim;
        POINT3D p;
        double weight;

        int z, current_stack;
        z=-1;//this is the z coordinate of the stack
        current_stack=-1; //we need to know when to start a new stack

        double threshold = 0.5;
        if (simulate_excluded)
            threshold = -1;

        _reconstructed.Write("reconstructed.nii.gz");
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

            //    cout<<inputIndex<<" ";
            //    cout.flush();
            // read the current slice
            RealImage& slice = _slices[inputIndex];

            //Calculate simulated slice
            sim.Initialize( slice.Attributes() );
            sim = 0;

            //direction for current slice
            int dirIndex = _stack_index[inputIndex]+1;
            double gx=_directionsDTI[0][dirIndex];
            double gy=_directionsDTI[1][dirIndex];
            double gz=_directionsDTI[2][dirIndex];
            RotateDirections(gx,gy,gz,inputIndex);
            double bval=_bvalues[dirIndex];
            SphericalHarmonics sh;
            Matrix dir(1,3);
            dir(0,0)=gx;
            dir(0,1)=gy;
            dir(0,2)=gz;
            Matrix basis = sh.SHbasis(dir,_order);
            double sim_signal;

            //do not simulate excluded slice
            if(_slice_weight[inputIndex]>threshold)
            {
                for (i = 0; i < slice.GetX(); i++)
                    for (j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            weight=0;
                            n = _volcoeffs[inputIndex][i][j].size();
                            for (k = 0; k < n; k++) {
                                p = _volcoeffs[inputIndex][i][j][k];
                                //signal simulated from SH
                                sim_signal = 0;
                                for(unsigned int l = 0; l < basis.Cols(); l++ )
                                    sim_signal += _SH_coeffs(p.x, p.y, p.z,l)*basis(0,l);
                                //update slice
                                sim(i, j, 0) += p.value *sim_signal;
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

    void ReconstructionDWI::SimulateStacksDTIIntensityMatching(Array<RealImage>& stacks, bool simulate_excluded)
    {
        if (_debug)
            cout<<"Simulating stacks."<<endl;
        cout.flush();

        unsigned int inputIndex;
        int i, j, k, n;
        RealImage sim;
        POINT3D p;
        double weight;

        int z, current_stack;
        z=-1;//this is the z coordinate of the stack
        current_stack=-1; //we need to know when to start a new stack

        double threshold = 0.5;
        if (simulate_excluded)
            threshold = -1;

        _reconstructed.Write("reconstructed.nii.gz");
        for (inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {

            //    cout<<inputIndex<<" ";
            //    cout.flush();
            // read the current slice
            RealImage& slice = _slices[inputIndex];
            //read the current bias image
            RealImage& b = _bias[inputIndex];
            //identify scale factor
            double scale = _scale[inputIndex];


            //Calculate simulated slice
            sim.Initialize( slice.Attributes() );
            sim = 0;
            RealImage simulatedslice(sim), simulatedsliceint(sim), simulatedweights(sim);

            //direction for current slice
            int dirIndex = _stack_index[inputIndex]+1;
            double gx=_directionsDTI[0][dirIndex];
            double gy=_directionsDTI[1][dirIndex];
            double gz=_directionsDTI[2][dirIndex];
            RotateDirections(gx,gy,gz,inputIndex);
            double bval=_bvalues[dirIndex];
            SphericalHarmonics sh;
            Matrix dir(1,3);
            dir(0,0)=gx;
            dir(0,1)=gy;
            dir(0,2)=gz;
            Matrix basis = sh.SHbasis(dir,_order);
            double sim_signal;

            //do not simulate excluded slice
            if(_slice_weight[inputIndex]>threshold)
            {
                for (i = 0; i < slice.GetX(); i++)
                    for (j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            weight=0;
                            n = _volcoeffs[inputIndex][i][j].size();
                            for (k = 0; k < n; k++) {
                                p = _volcoeffs[inputIndex][i][j][k];
                                //signal simulated from SH
                                sim_signal = 0;
                                for(unsigned int l = 0; l < basis.Cols(); l++ )
                                    sim_signal += _SH_coeffs(p.x, p.y, p.z,l)*basis(0,l);
                                //update slice
                                sim(i, j, 0) += p.value *sim_signal;
                                weight += p.value;
                            }
                            simulatedweights(i,j,0)=weight;
                            if(weight>0.98)
                                sim(i,j,0)/=weight;
                            else
                                sim(i,j,0)=0;
                            simulatedslice(i,j,0)=sim(i,j,0);
                            //intensity parameters
                            if(_intensity_matching_GD)
                            {
                                double a=b(i, j, 0) * scale;
                                if(a>0)
                                    sim(i,j,0)/=a;
                                else
                                    sim(i,j,0)=0;
                            }
                            else
                                sim(i,j,0)/=(exp(-b(i, j, 0)) * scale);
                            simulatedsliceint(i,j,0)=sim(i,j,0);
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
            if(inputIndex == 528)
            {
                //          cout<<"InputIndex = "<<inputIndex<<endl;
                //          cout<<"stack="<<_stack_index[inputIndex]<<endl;
                sim.Write("sim.nii.gz");
            }

            //end of loop for a slice inputIndex

            if (_debug)
            {
                char buffer[256];
                sprintf(buffer,"simulatedslice%i.nii.gz",inputIndex);
                simulatedslice.Write(buffer);
                sprintf(buffer,"simulatedsliceint%i.nii.gz",inputIndex);
                simulatedsliceint.Write(buffer);
                sprintf(buffer,"simulatedbias%i.nii.gz",inputIndex);
                b.Write(buffer);
                sprintf(buffer,"simulatedweights%i.nii.gz",inputIndex);
                simulatedweights.Write(buffer);
            }
        }
    }



    void ReconstructionDWI::CorrectStackIntensities(Array<RealImage>& stacks)
    {
        int z, current_stack;
        z=-1;//this is the z coordinate of the stack
        current_stack=-1; //we need to know when to start a new stack

        for (int inputIndex = 0; inputIndex < _slices.size(); inputIndex ++)
        {
            if (_stack_index[inputIndex]==current_stack)
                z++;
            else
            {
                current_stack=_stack_index[inputIndex];
                z=0;
            }

            for(int i=0;i<stacks[_stack_index[inputIndex]].GetX();i++)
                for(int j=0;j<stacks[_stack_index[inputIndex]].GetY();j++)
                {
                    stacks[_stack_index[inputIndex]](i,j,z) *= (exp(-_bias[inputIndex](i, j, 0)) * _scale[inputIndex]);
                }
        }
    }

    class ParallelSimulateSlicesDTI {
        ReconstructionDWI *reconstructor;

    public:
        ParallelSimulateSlicesDTI( ReconstructionDWI *_reconstructor ) :
        reconstructor(_reconstructor) { }

        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                //Calculate simulated slice
                reconstructor->_simulated_slices[inputIndex].Initialize( reconstructor->_slices[inputIndex].Attributes() );
                reconstructor->_simulated_slices[inputIndex] = 0;

                reconstructor->_simulated_weights[inputIndex].Initialize( reconstructor->_slices[inputIndex].Attributes() );
                reconstructor->_simulated_weights[inputIndex] = 0;

                reconstructor->_simulated_inside[inputIndex].Initialize( reconstructor->_slices[inputIndex].Attributes() );
                reconstructor->_simulated_inside[inputIndex] = 0;

                reconstructor->_slice_inside[inputIndex] = false;


                //direction for current slice
                int dirIndex = reconstructor->_stack_index[inputIndex]+1;
                double gx=reconstructor->_directionsDTI[0][dirIndex];
                double gy=reconstructor->_directionsDTI[1][dirIndex];
                double gz=reconstructor->_directionsDTI[2][dirIndex];
                reconstructor->RotateDirections(gx,gy,gz,inputIndex);
                double bval=reconstructor->_bvalues[dirIndex];
                SphericalHarmonics sh;
                Matrix dir(1,3);
                dir(0,0)=gx;
                dir(0,1)=gy;
                dir(0,2)=gz;
                Matrix basis = sh.SHbasis(dir,reconstructor->_order);
                if (basis.Cols() != reconstructor->_SH_coeffs.GetT())
                    throw runtime_error("ParallelSimulateSlicesDTI:basis numbers does not match SH coefficients number.");

                double sim_signal;
                POINT3D p;
                for ( unsigned int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++ )
                    for ( unsigned int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++ )
                        if ( reconstructor->_slices[inputIndex](i, j, 0) != -1 ) {
                            double weight = 0;
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for ( unsigned int k = 0; k < n; k++ ) {
                                //PSF
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                //signal simulated from SH
                                sim_signal = 0;
                                for(unsigned int l = 0; l < basis.Cols(); l++ )
                                    sim_signal += reconstructor->_SH_coeffs(p.x, p.y, p.z,l)*basis(0,l);
                                //update slice
                                reconstructor->_simulated_slices[inputIndex](i, j, 0) += p.value * sim_signal;
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

            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                         *this );

        }

    };

    void ReconstructionDWI::SimulateSlicesDTI()
    {
        if (_debug)
            cout<<"Simulating slices DTI."<<endl;

        ParallelSimulateSlicesDTI parallelSimulateSlicesDTI( this );
        parallelSimulateSlicesDTI();

        if (_debug)
            cout<<"done."<<endl;
    }

    double ReconstructionDWI::ConsistencyDTI()
    {
        double ssd=0;
        int num = 0;
        double diff;
        for(int index = 0; index< _slices.size(); index++)
        {
            for(int i=0;i<_slices[index].GetX();i++)
                for(int j=0;j<_slices[index].GetY();j++)
                    if((_slices[index](i,j,0)>=0)&&(_simulated_inside[index](i,j,0)==1))
                    {
                        diff = _slices[index](i,j,0)-_simulated_slices[index](i,j,0);
                        ssd+=diff*diff;
                        num++;
                    }
        }
        cout<<"consistency: "<<ssd<<" "<<sqrt(ssd/num)<<endl;
        return ssd;
    }

    class ParallelSuperresolutionDTI {
        ReconstructionDWI* reconstructor;
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

                //direction for current slice
                int dirIndex = reconstructor->_stack_index[inputIndex]+1;
                double gx=reconstructor->_directionsDTI[0][dirIndex];
                double gy=reconstructor->_directionsDTI[1][dirIndex];
                double gz=reconstructor->_directionsDTI[2][dirIndex];
                reconstructor->RotateDirections(gx,gy,gz,inputIndex);
                double bval=reconstructor->_bvalues[dirIndex];
                SphericalHarmonics sh;
                Matrix dir(1,3);
                dir(0,0)=gx;
                dir(0,1)=gy;
                dir(0,2)=gz;
                Matrix basis = sh.SHbasis(dir,reconstructor->_order);
                if (basis.Cols() != reconstructor->_SH_coeffs.GetT())
                    throw runtime_error("ParallelSimulateSlicesDTI:basis numbers does not match SH coefficients number.");

                //Update reconstructed volume using current slice

                //Distribute error to the volume
                POINT3D p;
                for ( int i = 0; i < slice.GetX(); i++)
                    for ( int j = 0; j < slice.GetY(); j++)
                        if (slice(i, j, 0) != -1) {
                            //bias correct and scale the slice
                            if(reconstructor->_intensity_matching_GD)
                                slice(i, j, 0) *= b(i, j, 0) * scale;
                            else
                                slice(i, j, 0) *= exp(-b(i, j, 0)) * scale;

                            if ( reconstructor->_simulated_slices[inputIndex](i,j,0) > 0 )
                                slice(i,j,0) = slice(i,j,0) - reconstructor->_simulated_slices[inputIndex](i,j,0);
                            else
                                slice(i,j,0) = 0;

                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for (int k = 0; k < n; k++) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                if(reconstructor->_robust_slices_only)
                                {
                                    for(unsigned int l = 0; l < basis.Cols(); l++ )
                                    {
                                        addon(p.x, p.y, p.z,l) += p.value * basis(0,l) * slice(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        confidence_map(p.x, p.y, p.z,l) += p.value *reconstructor->_slice_weight[inputIndex];
                                    }

                                }
                                else
                                {
                                    for(unsigned int l = 0; l < basis.Cols(); l++ )
                                    {
                                        addon(p.x, p.y, p.z, l) += p.value * basis(0,l) * slice(i, j, 0) * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        confidence_map(p.x, p.y, p.z, l) += p.value * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        //p.value * basis(0,l) * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                    }
                                }
                            }
                        }
            } //end of loop for a slice inputIndex
        }

        ParallelSuperresolutionDTI( ParallelSuperresolutionDTI& x, split ) :
        reconstructor(x.reconstructor)
        {
            //Clear addon
            addon.Initialize( reconstructor->_SH_coeffs.Attributes() );
            addon = 0;

            //Clear confidence map
            confidence_map.Initialize( reconstructor->_SH_coeffs.Attributes() );
            confidence_map = 0;
        }

        void join( const ParallelSuperresolutionDTI& y ) {
            addon += y.addon;
            confidence_map += y.confidence_map;
        }

        ParallelSuperresolutionDTI( ReconstructionDWI *reconstructor ) :
        reconstructor(reconstructor)
        {
            //Clear addon
            addon.Initialize( reconstructor->_SH_coeffs.Attributes() );
            addon = 0;

            //Clear confidence map
            confidence_map.Initialize( reconstructor->_SH_coeffs.Attributes() );
            confidence_map = 0;
        }

        // execute
        void operator() () {
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()),
                            *this );
        }
    };

    void ReconstructionDWI::SuperresolutionDTI(int iter, bool tv, double sh_alpha)
    {
        if (_debug)
            cout << "SuperresolutionDTI " << iter << endl;

        int i, j, k, l,t;
        RealImage addon, original;


        //Remember current reconstruction for edge-preserving smoothing
        original = _SH_coeffs;

        ParallelSuperresolutionDTI parallelSuperresolutionDTI(this);
        parallelSuperresolutionDTI();
        addon = parallelSuperresolutionDTI.addon;
        _confidence_map = parallelSuperresolutionDTI.confidence_map;
        //_confidence4mask = _confidence_map;

        if(_debug) {
            char buffer[256];
            //sprintf(buffer,"confidence-map%i.nii.gz",iter);
            //_confidence_map.Write(buffer);
            _confidence_map.Write("confidence-map-superDTI.nii.gz");
            sprintf(buffer,"addon%i.nii.gz",iter);
            addon.Write(buffer);
        }

        double alpha;
        alpha=sh_alpha/_average_volume_weight;


        _SH_coeffs += addon * alpha;
        cout<<"alpha = "<<alpha<<endl;

        ////regularisation

        if (_lambdaLB > 0)
        {
            cout<<"_lambdaLB = "<<_lambdaLB<<endl;
            RealImage regul(addon);
            regul=0;
            double lb=0;
            double factor;
            for (t = 0; t < regul.GetT(); t++)
                for (i = 0; i < regul.GetX(); i++)
                    for (j = 0; j < regul.GetY(); j++)
                        for (k = 0; k < regul.GetZ(); k++)
                        {
                            if(t==0) lb=0;
                            if((t>=1)&&(t<=5)) lb=36.0/400.0;
                            if((t>=6)&&(t<=15)) lb=400.0/400.0;
                            factor = _lambdaLB * lb;
                            if(factor>1) factor=1;//this is to ensure that there cannot be negative effect
                            regul(i,j,k,t)=factor * _SH_coeffs(i,j,k,t);//alpha *
                        }
            regul.Write("regul.nii.gz");
            _SH_coeffs -= regul;

        }

        //TV on SH basis
        if(tv)
        {
            RealImage o,r;


            for (int steps = 0; steps < _regul_steps; steps ++)
            {
                r=_reconstructed;
                for(t=0;t<_SH_coeffs.GetT();t++)
                {
                    o=original.GetRegion(0,0,0,t,original.GetX(),original.GetY(),original.GetZ(),t+1);

                    _reconstructed=_SH_coeffs.GetRegion(0,0,0,t,original.GetX(),original.GetY(),original.GetZ(),t+1);

                    //L22Regularization(iter, o);
                    AdaptiveRegularization(iter, o);
                    //LaplacianRegularization(iter,t,o);


                    //put it back to _SH_coeffs
                    for (i = 0; i < _SH_coeffs.GetX(); i++)
                        for (j = 0; j < _SH_coeffs.GetY(); j++)
                            for (k = 0; k < _SH_coeffs.GetZ(); k++)
                            {
                                _SH_coeffs(i,j,k,t)=_reconstructed(i,j,k);
                            }
                }
                _reconstructed=r;
            }
        }


    }

    void ReconstructionDWI::NormaliseBiasSH(int iter)
    {
        NormaliseBias(iter, _SH_coeffs);

    }

    double ReconstructionDWI::LaplacianSmoothnessDTI()
    {
        if (_debug)
            cout << "LaplacianSmoothnessDTI "<< endl;

        double sum=0;
        RealImage o,r;
        r=_reconstructed;
        for(int t=0;t<_SH_coeffs.GetT();t++)
        {
            o=_SH_coeffs.GetRegion(0,0,0,t,_SH_coeffs.GetX(),_SH_coeffs.GetY(),_SH_coeffs.GetZ(),t+1);
            _reconstructed=_SH_coeffs.GetRegion(0,0,0,t,_SH_coeffs.GetX(),_SH_coeffs.GetY(),_SH_coeffs.GetZ(),t+1);
            sum+=LaplacianSmoothness(o);

        }
        _reconstructed=r;
        return sum;
    }





    void ReconstructionDWI::RotateDirections(double &dx, double &dy, double &dz, int i)
    {

        //Array end-point
        double x,y,z;
        //origin
        double ox,oy,oz;

        if (_debug)
        {
            //cout<<"Original direction "<<i<<"(dir"<<_stack_index[i]+1<<"): ";
            //cout<<dx<<", "<<dy<<", "<<dz<<". ";
            //cout<<endl;
        }

        //origin
        ox=0;oy=0;oz=0;
        _transformations[i].Transform(ox,oy,oz);

        //end-point
        x=dx;
        y=dy;
        z=dz;
        _transformations[i].Transform(x,y,z);

        dx=x-ox;
        dy=y-oy;
        dz=z-oz;

        if (_debug)
        {
            //cout<<"Rotated direction "<<i<<"(dir"<<_stack_index[i]+1<<"): ";
            //cout<<dx<<", "<<dy<<", "<<dz<<". ";
            //cout<<endl;
        }
    }

    void ReconstructionDWI::RotateDirections(double &dx, double &dy, double &dz, RigidTransformation &t)
    {

        //Array end-point
        double x,y,z;
        //origin
        double ox,oy,oz;

        if (_debug)
        {
            //cout<<"Original direction "<<i<<"(dir"<<_stack_index[i]+1<<"): ";
            //cout<<dx<<", "<<dy<<", "<<dz<<". ";
            //cout<<endl;
        }

        //origin
        ox=0;oy=0;oz=0;
        t.Transform(ox,oy,oz);

        //end-point
        x=dx;
        y=dy;
        z=dz;
        t.Transform(x,y,z);

        dx=x-ox;
        dy=y-oy;
        dz=z-oz;

        if (_debug)
        {
            //cout<<"Rotated direction "<<i<<"(dir"<<_stack_index[i]+1<<"): ";
            //cout<<dx<<", "<<dy<<", "<<dz<<". ";
            //cout<<endl;
        }
    }


    void ReconstructionDWI::CreateSliceDirections(Array< Array<double> >& directions, Array<double>& bvalues)
    {
        _directionsDTI = directions;
        _bvalues = bvalues;

        cout<<"B-values: ";
        for(uint i=0;i<_bvalues.size(); i++)
            cout<<_bvalues[i]<<" ";
        cout<<endl;

        cout<<"B-Arrays: ";
        for(uint j=0; j<_directionsDTI.size(); j++)
        {
            cout << " - " << j << " : ";
            for(uint i=0;i<_directionsDTI[j].size(); i++)
                cout<<_directionsDTI[j][i]<<" ";
            cout<<endl;
        }
        cout<<endl;
    }

    void ReconstructionDWI::InitSH(Matrix dirs, int order)
    {
        //SphericalHarmonics sh;
        if(_lambdaLB > 0)
            _sh.InitSHTRegul(dirs,_lambdaLB,order);
        else
            _sh.InitSHT(dirs,order);

        _SH_coeffs = _sh.Signal2Coeff(_simulated_signal);
//        _SH_coeffs.Write("_SH_coeffs-init.nii.gz");

//        _SH_coeffs = 0;

        _SH_coeffs.Write("init_SH_coeffs-init.nii.gz");

        _order = order;
        _dirs = dirs;
        _coeffNum = _sh.NforL(_order);
    }

    void ReconstructionDWI::InitSHT(Matrix dirs, int order)
    {
        SphericalHarmonics sh;
        if(_lambdaLB > 0)
            sh.InitSHTRegul(dirs,_lambdaLB,order);
        else
            sh.InitSHT(dirs,order);

        //_SH_coeffs = sh.Signal2Coeff(_simulated_signal);
        //_SH_coeffs.Write("_SH_coeffs.nii.gz");
        _order = order;
        _dirs = dirs;
        _coeffNum = sh.NforL(_order);
    }

    void ReconstructionDWI::SimulateSignal()//Matrix dirs, int order)
    {
        SphericalHarmonics sh;
        sh.InitSHT(_dirs,_order);//lambda,order);
        _simulated_signal=sh.Coeff2Signal(_SH_coeffs);
        _simulated_signal.Write("_simulated_signal.nii.gz");

    }


    RealImage ReconstructionDWI::ReturnSimulatedSignal()
    {
        return _simulated_signal;

    }


    void ReconstructionDWI::SaveSHcoeffs(int iteration)
    {
        char buffer[256];
        sprintf(buffer, "shCoeff%i.nii.gz", iteration);
        _SH_coeffs.Write(buffer);
    }

    void ReconstructionDWI::PostProcessMotionParameters(RealImage target, bool empty) //, int packages)
    {
        if (_slice_order.size() != _transformations.size())
            throw runtime_error("Slice order does not match the number of transformations.");

        //Reset origin for transformations
        int i;
        GreyImage t = target;
        RigidTransformation offset;
        ResetOrigin(t,offset);
        Matrix mo = offset.GetMatrix();
        Matrix imo = mo;
        imo.Invert();

        target.Write("target.nii.gz");
        t.Write("t.nii.gz");

        for(i=0;i<_transformations.size();i++)
        {
            Matrix m = _transformations[i].GetMatrix();
            m=imo*m*mo;
            _transformations[i].PutMatrix(m);
        }

        //standard deviation for gaussian kernel in number of slices
        double sigma = _motion_sigma;
        int j,iter,par;
        Array<double> kernel;
        for(i=-3*sigma; i<=3*sigma; i++)
        {
            kernel.push_back(exp(-(i/sigma)*(i/sigma)));
            //cout<<kernel[kernel.size()-1]<<" ";
        }

        Matrix parameters(6,_transformations.size());
        Matrix weights(6,_transformations.size());
        cout<<"Filling parameters and weights: "<<endl;
        //RealPixel smin, smax;

        ofstream fileOut("motion.txt", ofstream::out | ofstream::app);
        for(i=0;i<_transformations.size();i++)
        {
            parameters(0,i)=_transformations[_slice_order[i]].GetTranslationX();
            parameters(1,i)=_transformations[_slice_order[i]].GetTranslationY();
            parameters(2,i)=_transformations[_slice_order[i]].GetTranslationZ();
            parameters(3,i)=_transformations[_slice_order[i]].GetRotationX();
            parameters(4,i)=_transformations[_slice_order[i]].GetRotationY();
            parameters(5,i)=_transformations[_slice_order[i]].GetRotationZ();

            for(j=0;j<6;j++)
            {
                fileOut<<parameters(j,i);
                if(j<5)
                    fileOut<<",";
                else
                    fileOut<<endl;
            }

            if (empty)
            {

                RealPixel smin, smax;
                _slices[_slice_order[i]].GetMinMax(&smin,&smax);
                if(smax>-1)
                    for(j=0;j<6;j++)
                        weights(j,i)=1;
                else
                    for(j=0;j<6;j++)
                        weights(j,i)=0;

            }
            else
            {
                if(_slice_inside.size()>0)
                {
                    if(_slice_inside[_slice_order[i]])
                        for(j=0;j<6;j++)
                            weights(j,i)=1;
                    else
                        for(j=0;j<6;j++)
                            weights(j,i)=0;
                }
            }
        }


        Matrix den(6,_transformations.size());
        Matrix num(6,_transformations.size());
        Matrix kr(6,_transformations.size());

        Array<double> error,tmp;
        double median;
        int packages = 2;
        int dim = _transformations.size()/packages;
        int pack;

        cerr<<endl<<endl<<endl<<endl;

        for(iter = 0; iter<50;iter++)
        {
            //kernel regression
            for(pack=0;pack<packages;pack++)
            {

                for(par=0;par<6;par++)
                {

                    for(i=pack*dim;i<(pack+1)*dim;i++)
                    {

                        for(j=-3*sigma;j<=3*sigma;j++)
                            if(((i+j)>=pack*dim) && ((i+j)<(pack+1)*dim))
                            {

                                num(par,i)+=parameters(par,i+j)*kernel[j+3*sigma]*weights(par,i+j);
                                den(par,i)+=kernel[j+3*sigma]*weights(par,i+j);
                            }
                    }
                }
            }

            for(par=0;par<6;par++)
                for(i=0;i<_transformations.size();i++)
                    kr(par,i)=num(par,i)/den(par,i);

            //recalculate weights
            for(par=0;par<6;par++)
            {
                error.clear();
                tmp.clear();

                for(i=0;i<_transformations.size();i++)
                {
                    error.push_back(fabs(parameters(par,i)-kr(par,i)));
                    tmp.push_back(fabs(parameters(par,i)-kr(par,i)));
                }

                sort(tmp.begin(),tmp.end());
                median = tmp[round(tmp.size()*0.5)];

                for(i=0;i<_transformations.size();i++)
                {
                    if(weights(par,i)>0)
                    {
                        if(error[i]<=median*1.35)
                            weights(par,i)=1;
                        else
                            weights(par,i)=median*1.35/error[i];
                    }
                }
            }

        }

        ofstream fileOut2("motion-processed.txt", ofstream::out | ofstream::app);
        for(i=0;i<_transformations.size();i++)
        {
            _transformations[_slice_order[i]].PutTranslationX(kr(0,i));
            _transformations[_slice_order[i]].PutTranslationY(kr(1,i));
            _transformations[_slice_order[i]].PutTranslationZ(kr(2,i));
            _transformations[_slice_order[i]].PutRotationX(kr(3,i));
            _transformations[_slice_order[i]].PutRotationY(kr(4,i));
            _transformations[_slice_order[i]].PutRotationZ(kr(5,i));

            for(j=0;j<6;j++)
            {
                fileOut2<<kr(j,i);
                if(j<5)
                    fileOut2<<",";
                else
                    fileOut2<<endl;
            }
        }
        weights.Print();

        //Put origin back
        //mo.Invert();
        for(i=0;i<_transformations.size();i++)
        {
            Matrix m = _transformations[i].GetMatrix();
            m=mo*m*imo;
            _transformations[i].PutMatrix(m);
        }
    }

    void ReconstructionDWI::PostProcessMotionParametersHS(RealImage target, bool empty) //, int packages)
    {
        if (_slice_order.size() != _transformations.size())
            throw runtime_error("Slice order does not match the number of transformations.");

        //Reset origin for transformations
        int i;
        GreyImage t = target;
        RigidTransformation offset;
        ResetOrigin(t,offset);
        Matrix m;
        Matrix mo = offset.GetMatrix();
        Matrix imo = mo;
        imo.Invert();

        target.Write("target.nii.gz");
        t.Write("t.nii.gz");


        for(i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m=imo*m*mo;
            _transformations[i].PutMatrix(m);
        }

        //standard deviation for gaussian kernel in number of slices
        double sigma = _motion_sigma;
        int j,iter,par;
        Array<double> kernel;
        for(i=-3*sigma; i<=3*sigma; i++)
        {
            kernel.push_back(exp(-(i/sigma)*(i/sigma)));
            //cout<<kernel[kernel.size()-1]<<" ";
        }

        Matrix parameters(6,_transformations.size());
        Matrix weights(6,_transformations.size());
        cout<<"Filling parameters and weights: "<<endl;
        //RealPixel smin, smax;

        ofstream fileOut("motion.txt", ofstream::out | ofstream::app);
        for(i=0;i<_transformations.size();i++)
        {
            parameters(0,i)=_transformations[_slice_order[i]].GetTranslationX();
            parameters(1,i)=_transformations[_slice_order[i]].GetTranslationY();
            parameters(2,i)=_transformations[_slice_order[i]].GetTranslationZ();
            parameters(3,i)=_transformations[_slice_order[i]].GetRotationX();
            parameters(4,i)=_transformations[_slice_order[i]].GetRotationY();
            parameters(5,i)=_transformations[_slice_order[i]].GetRotationZ();

            for(j=0;j<6;j++)
            {
                fileOut<<parameters(j,i);
                if(j<5)
                    fileOut<<",";
                else
                    fileOut<<endl;
            }

            if (empty)
            {

                RealPixel smin, smax;
                _slices[_slice_order[i]].GetMinMax(&smin,&smax);
                if(smax>-1)
                    for(j=0;j<6;j++)
                        weights(j,i)=1;
                else
                    for(j=0;j<6;j++)
                        weights(j,i)=0;

            }
            else
            {
                if(_slice_inside.size()>0)
                {
                    if(_slice_inside[_slice_order[i]])
                        for(j=0;j<6;j++)
                            weights(j,i)=1;
                    else
                        for(j=0;j<6;j++)
                            weights(j,i)=0;
                }
            }
        }

        //parameters.Print();
        //weights.Print();

        Matrix den(6,_transformations.size());
        Matrix num(6,_transformations.size());
        Matrix kr(6,_transformations.size());
        //Matrix error(6,_transformations.size());
        Array<double> error,tmp;
        double median;
        int packages = 2;
        int dim = _transformations.size()/packages;
        int pack;

        cerr<<endl<<endl<<endl<<endl;

        for(iter = 0; iter<50;iter++)
        {
            //cerr<<"iter="<<iter<<endl;
            //kernel regression
            for(pack=0;pack<packages;pack++)
            {
                //cerr<<"pack="<<pack<<endl;

                for(par=0;par<6;par++)
                {
                    //cerr<<"par="<<par<<endl;
                    for(i=pack*dim;i<(pack+1)*dim;i++)
                    {
                        //cerr<<"i="<<i<<" pack*dim = "<<pack*dim<<" dim = "<<dim<<endl;
                        for(j=-3*sigma;j<=3*sigma;j++)
                            if(((i+j)>=pack*dim) && ((i+j)<(pack+1)*dim))
                            {
                                //cerr<<"j="<<j<<" i+j="<<i+j<<" par="<<par<<" j+3*sigma="<<j+3*sigma<<endl;
                                num(par,i)+=parameters(par,i+j)*kernel[j+3*sigma]*weights(par,i+j);
                                den(par,i)+=kernel[j+3*sigma]*weights(par,i+j);
                            }
                    }
                }
            }

            for(par=0;par<6;par++)
                for(i=0;i<_transformations.size();i++)
                    kr(par,i)=num(par,i)/den(par,i);

            cout<<"Iter "<<iter<<": median = ";

            //recalculate weights
            //for(par=0;par<6;par++)
            //{
            error.clear();
            tmp.clear();

            for(i=0;i<_transformations.size();i++)
            {

                ///CHange: TRE
                RigidTransformation processed;
                processed.PutTranslationX(kr(0,i));
                processed.PutTranslationY(kr(1,i));
                processed.PutTranslationZ(kr(2,i));
                processed.PutRotationX(kr(3,i));
                processed.PutRotationY(kr(4,i));
                processed.PutRotationZ(kr(5,i));

                RigidTransformation orig = _transformations[_slice_order[i]];

                //need to convert the transformations back to the original coordinate system
                m = orig.GetMatrix();
                m=mo*m*imo;
                orig.PutMatrix(m);

                m = processed.GetMatrix();
                m=mo*m*imo;
                processed.PutMatrix(m);

                RealImage slice = _slices[_slice_order[i]];

                double tre=0;
                int n=0;
                double x, y, z, xx, yy, zz, e = 0;
                for(int ii=0;ii<slice.GetX();ii++)
                    for(int jj=0;jj<slice.GetY();jj++)
                        if(slice(ii,jj,0)>-1)
                        {
                            x=ii; y=jj; z=0;
                            slice.ImageToWorld(x,y,z);
                            xx=x;yy=y;zz=z;
                            orig.Transform(x,y,z);
                            processed.Transform(xx,yy,zz);
                            x-=xx;
                            y-=yy;
                            z-=zz;
                            e += sqrt(x*x+y*y+z*z);
                            n++;
                        }
                if(n>0)
                {
                    e/=n;
                    error.push_back(e);
                    tmp.push_back(e);
                }
                else
                    error.push_back(-1);

                ///original
                //error.push_back(fabs(parameters(par,i)-kr(par,i)));
                //tmp.push_back(fabs(parameters(par,i)-kr(par,i)));
            }

            sort(tmp.begin(),tmp.end());
            median = tmp[round(tmp.size()*0.5)];
            cout<<median<<" ";

            for(i=0;i<_transformations.size();i++)
            {
                //if(weights(par,i)>0)
                if(error[i]>=0)
                {
                    if(error[i]<=median*1.35)
                        for(par=0;par<6;par++)
                            weights(par,i)=1;
                    else
                        for(par=0;par<6;par++)
                            weights(par,i)=median*1.35/error[i];
                }
                else
                    for(par=0;par<6;par++)
                        weights(par,i)=0;
            }
            //}
            cout<<endl;

        }

        ofstream fileOut2("motion-processed.txt", ofstream::out | ofstream::app);
        ofstream fileOut3("weights.txt", ofstream::out | ofstream::app);
        ofstream fileOut4("outliers.txt", ofstream::out | ofstream::app);
        ofstream fileOut5("empty.txt", ofstream::out | ofstream::app);
        for(i=0;i<_transformations.size();i++)
        {
            //decide whether outlier
            bool outlier  = false;
            for(j=0;j<6;j++)
            {
                if (weights(j,i)<0.5)
                    outlier = true;
                fileOut3<<weights(j,i);
                if(j<5)
                    fileOut3<<",";
                else
                    fileOut3<<endl;
            }

            if (outlier)
            {
                if(weights(0,i)>0)
                    fileOut4<<_slice_order[i]<<" ";
                else
                    fileOut5<<_slice_order[i]<<" ";

                _transformations[_slice_order[i]].PutTranslationX(kr(0,i));
                _transformations[_slice_order[i]].PutTranslationY(kr(1,i));
                _transformations[_slice_order[i]].PutTranslationZ(kr(2,i));
                _transformations[_slice_order[i]].PutRotationX(kr(3,i));
                _transformations[_slice_order[i]].PutRotationY(kr(4,i));
                _transformations[_slice_order[i]].PutRotationZ(kr(5,i));

                for(j=0;j<6;j++)
                {
                    fileOut2<<kr(j,i);
                    if(j<5)
                        fileOut2<<",";
                    else
                        fileOut2<<endl;
                }
            }
            else
            {
                for(j=0;j<6;j++)
                {
                    fileOut2<<parameters(j,i);
                    if(j<5)
                        fileOut2<<",";
                    else
                        fileOut2<<endl;
                }
            }

        }
        //weights.Print();

        //Put origin back
        //mo.Invert();
        for(i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m=mo*m*imo;
            _transformations[i].PutMatrix(m);
        }

    }


    void ReconstructionDWI::PostProcessMotionParametersHS2(RealImage target, bool empty) //, int packages)
    {
        if (_slice_order.size() != _transformations.size())
            throw runtime_error("Slice order does not match the number of transformations.");

        //Reset origin for transformations
        int i;
        GreyImage t = target;
        RigidTransformation offset;
        ResetOrigin(t,offset);
        Matrix m;
        Matrix mo = offset.GetMatrix();
        Matrix imo = mo;
        imo.Invert();

        target.Write("target.nii.gz");
        t.Write("t.nii.gz");


        for(i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m=imo*m*mo;
            _transformations[i].PutMatrix(m);
        }

        //standard deviation for gaussian kernel in number of slices
        double sigma = _motion_sigma;
        int j,iter,par;
        Array<double> kernel;
        for(i=-3*sigma; i<=3*sigma; i++)
        {
            kernel.push_back(exp(-(i/sigma)*(i/sigma)));
            //cout<<kernel[kernel.size()-1]<<" ";
        }

        Matrix parameters(6,_transformations.size());
        Matrix weights(6,_transformations.size());
        cout<<"Filling parameters and weights: "<<endl;
        //RealPixel smin, smax;

        ofstream fileOut("motion.txt", ofstream::out | ofstream::app);
        for(i=0;i<_transformations.size();i++)
        {
            parameters(0,i)=_transformations[_slice_order[i]].GetTranslationX();
            parameters(1,i)=_transformations[_slice_order[i]].GetTranslationY();
            parameters(2,i)=_transformations[_slice_order[i]].GetTranslationZ();
            parameters(3,i)=_transformations[_slice_order[i]].GetRotationX();
            parameters(4,i)=_transformations[_slice_order[i]].GetRotationY();
            parameters(5,i)=_transformations[_slice_order[i]].GetRotationZ();

            for(j=0;j<6;j++)
            {
                fileOut<<parameters(j,i);
                if(j<5)
                    fileOut<<",";
                else
                    fileOut<<endl;
            }

            if (empty)
            {

                RealPixel smin, smax;
                _slices[_slice_order[i]].GetMinMax(&smin,&smax);
                if(smax>-1)
                    for(j=0;j<6;j++)
                        weights(j,i)=1;
                else
                    for(j=0;j<6;j++)
                        weights(j,i)=0;

            }
            else
            {
                if(_slice_inside.size()>0)
                {
                    if(_slice_inside[_slice_order[i]])
                        for(j=0;j<6;j++)
                            weights(j,i)=1;
                    else
                        for(j=0;j<6;j++)
                            weights(j,i)=0;
                }
            }
        }

        //NEW - need to normalize rotation so that the first one is zero
        // this is to deal with cases when rotation is around 180

        double offsetrot[6];
        offsetrot[0] = 0;
        offsetrot[1] = 0;
        offsetrot[2] = 0;
        offsetrot[3] = parameters(3,0);
        offsetrot[4] = parameters(4,0);
        offsetrot[5] = parameters(5,0);


        Matrix den(6,_transformations.size());
        Matrix num(6,_transformations.size());
        Matrix kr(6,_transformations.size());
        //Matrix error(6,_transformations.size());
        Array<double> error,tmp;
        double median;
        int packages = 2;
        int dim = _transformations.size()/packages;
        int pack;

        cerr<<endl<<endl<<endl<<endl;

        for(iter = 0; iter<50;iter++)
        {

            //kernel regression
            for(pack=0;pack<packages;pack++)
            {

                for(par=0;par<6;par++)
                {

                    for(i=pack*dim;i<(pack+1)*dim;i++)
                    {

                        for(j=-3*sigma;j<=3*sigma;j++)
                            if(((i+j)>=pack*dim) && ((i+j)<(pack+1)*dim))
                            {

                                double value = parameters(par,i+j)-offsetrot[par];
                                if(value>180)
                                    value-=360;
                                if(value<-180)
                                    value+=360;
                                num(par,i)+=(value)*kernel[j+3*sigma]*weights(par,i+j);
                                den(par,i)+=kernel[j+3*sigma]*weights(par,i+j);
                            }
                    }
                }
            }

            for(par=0;par<6;par++)
                for(i=0;i<_transformations.size();i++)
                {
                    double value=num(par,i)/den(par,i)+offsetrot[par];
                    if(value<-180)
                        value+=360;
                    if(value>180)
                        value-=360;
                    kr(par,i)=value;
                }

            cout<<"Iter "<<iter<<": median = ";
            cout.flush();


            error.clear();
            tmp.clear();

            for(i=0;i<_transformations.size();i++)
            {

                ///CHange: TRE
                RigidTransformation processed;
                processed.PutTranslationX(kr(0,i));
                processed.PutTranslationY(kr(1,i));
                processed.PutTranslationZ(kr(2,i));
                processed.PutRotationX(kr(3,i));
                processed.PutRotationY(kr(4,i));
                processed.PutRotationZ(kr(5,i));

                RigidTransformation orig = _transformations[_slice_order[i]];

                //need to convert the transformations back to the original coordinate system
                m = orig.GetMatrix();
                m=mo*m*imo;
                orig.PutMatrix(m);

                m = processed.GetMatrix();
                m=mo*m*imo;
                processed.PutMatrix(m);

                RealImage slice = _slices[_slice_order[i]];

                double tre=0;
                int n=0;
                double x, y, z, xx, yy, zz, e = 0;
                for(int ii=0;ii<_reconstructed.GetX();ii=ii+3)
                    for(int jj=0;jj<_reconstructed.GetY();jj=jj+3)
                        for(int kk=0;kk<_reconstructed.GetZ();kk=kk+3)
                            if(_reconstructed(ii,jj,kk)>-1)
                            {

                                x=ii; y=jj; z=kk;
                                _reconstructed.ImageToWorld(x,y,z);
                                xx=x;yy=y;zz=z;
                                orig.Transform(x,y,z);
                                processed.Transform(xx,yy,zz);
                                x-=xx;
                                y-=yy;
                                z-=zz;
                                e += sqrt(x*x+y*y+z*z);
                                n++;
                            }
                if(n>0)
                {
                    e/=n;
                    error.push_back(e);
                    tmp.push_back(e);
                }
                else
                    error.push_back(-1);

            }

            sort(tmp.begin(),tmp.end());
            median = tmp[round(tmp.size()*0.5)];
            cout<<median<<" ";

            for(i=0;i<_transformations.size();i++)
            {
                //if(weights(par,i)>0)
                if(error[i]>=0)
                {
                    if(error[i]<=median*1.35)
                        for(par=0;par<6;par++)
                            weights(par,i)=1;
                    else
                        for(par=0;par<6;par++)
                            weights(par,i)=median*1.35/error[i];
                }
                else
                    for(par=0;par<6;par++)
                        weights(par,i)=0;
            }
            //}
            cout<<endl;

        }

        ofstream fileOut2("motion-processed.txt", ofstream::out | ofstream::app);
        ofstream fileOut3("weights.txt", ofstream::out | ofstream::app);
        ofstream fileOut4("outliers.txt", ofstream::out | ofstream::app);
        ofstream fileOut5("empty.txt", ofstream::out | ofstream::app);
        for(i=0;i<_transformations.size();i++)
        {
            //decide whether outlier
            bool outlier  = false;
            for(j=0;j<6;j++)
            {
                if (weights(j,i)<0.5)
                    outlier = true;
                fileOut3<<weights(j,i);
                if(j<5)
                    fileOut3<<",";
                else
                    fileOut3<<endl;
            }

            if (outlier)
            {
                if(weights(0,i)>0)
                    fileOut4<<_slice_order[i]<<" ";
                else
                    fileOut5<<_slice_order[i]<<" ";

                _transformations[_slice_order[i]].PutTranslationX(kr(0,i));
                _transformations[_slice_order[i]].PutTranslationY(kr(1,i));
                _transformations[_slice_order[i]].PutTranslationZ(kr(2,i));
                _transformations[_slice_order[i]].PutRotationX(kr(3,i));
                _transformations[_slice_order[i]].PutRotationY(kr(4,i));
                _transformations[_slice_order[i]].PutRotationZ(kr(5,i));

                for(j=0;j<6;j++)
                {
                    fileOut2<<kr(j,i);
                    if(j<5)
                        fileOut2<<",";
                    else
                        fileOut2<<endl;
                }
            }
            else
            {
                for(j=0;j<6;j++)
                {
                    fileOut2<<parameters(j,i);
                    if(j<5)
                        fileOut2<<",";
                    else
                        fileOut2<<endl;
                }
            }

        }


        for(i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m=mo*m*imo;
            _transformations[i].PutMatrix(m);
        }

    }


    void ReconstructionDWI::PostProcessMotionParameters2(RealImage target, bool empty) //, int packages)
    {
        if (_slice_order.size() != _transformations.size())
            throw runtime_error("Slice order does not match the number of transformations.");

        //Reset origin for transformations
        int i;
        GreyImage t = target;
        RigidTransformation offset;
        ResetOrigin(t,offset);
        Matrix m;
        Matrix mo = offset.GetMatrix();
        Matrix imo = mo;
        imo.Invert();

        target.Write("target.nii.gz");
        t.Write("t.nii.gz");


        for(i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m=imo*m*mo;
            _transformations[i].PutMatrix(m);
        }

        //standard deviation for gaussian kernel in number of slices
        double sigma = _motion_sigma;
        int j,iter,par;
        Array<double> kernel;
        for(i=-3*sigma; i<=3*sigma; i++)
        {
            kernel.push_back(exp(-(i/sigma)*(i/sigma)));
            //cout<<kernel[kernel.size()-1]<<" ";
        }

        Matrix parameters(6,_transformations.size());
        Matrix weights(6,_transformations.size());
        cout<<"Filling parameters and weights: "<<endl;
        //RealPixel smin, smax;

        ofstream fileOut("motion.txt", ofstream::out | ofstream::app);
        for(i=0;i<_transformations.size();i++)
        {
            parameters(0,i)=_transformations[_slice_order[i]].GetTranslationX();
            parameters(1,i)=_transformations[_slice_order[i]].GetTranslationY();
            parameters(2,i)=_transformations[_slice_order[i]].GetTranslationZ();
            parameters(3,i)=_transformations[_slice_order[i]].GetRotationX();
            parameters(4,i)=_transformations[_slice_order[i]].GetRotationY();
            parameters(5,i)=_transformations[_slice_order[i]].GetRotationZ();

            for(j=0;j<6;j++)
            {
                fileOut<<parameters(j,i);
                if(j<5)
                    fileOut<<",";
                else
                    fileOut<<endl;
            }

            if (empty)
            {

                RealPixel smin, smax;
                _slices[_slice_order[i]].GetMinMax(&smin,&smax);
                if(smax>-1)
                    for(j=0;j<6;j++)
                        weights(j,i)=1;
                else
                    for(j=0;j<6;j++)
                        weights(j,i)=0;

            }
            else
            {
                if(_slice_inside.size()>0)
                {
                    if(_slice_inside[_slice_order[i]])
                        for(j=0;j<6;j++)
                            weights(j,i)=1;
                    else
                        for(j=0;j<6;j++)
                            weights(j,i)=0;
                }
            }
        }

        //parameters.Print();
        //weights.Print();

        Matrix den(6,_transformations.size());
        Matrix num(6,_transformations.size());
        Matrix kr(6,_transformations.size());
        //Matrix error(6,_transformations.size());
        Array<double> error,tmp;
        double median;
        int packages = 2;
        int dim = _transformations.size()/packages;
        int pack;

        cerr<<endl<<endl<<endl<<endl;

        for(iter = 0; iter<50;iter++)
        {
            //cerr<<"iter="<<iter<<endl;
            //kernel regression
            for(pack=0;pack<packages;pack++)
            {
                //cerr<<"pack="<<pack<<endl;

                for(par=0;par<6;par++)
                {
                    //cerr<<"par="<<par<<endl;
                    for(i=pack*dim;i<(pack+1)*dim;i++)
                    {
                        //cerr<<"i="<<i<<" pack*dim = "<<pack*dim<<" dim = "<<dim<<endl;
                        for(j=-3*sigma;j<=3*sigma;j++)
                            if(((i+j)>=pack*dim) && ((i+j)<(pack+1)*dim))
                            {
                                //cerr<<"j="<<j<<" i+j="<<i+j<<" par="<<par<<" j+3*sigma="<<j+3*sigma<<endl;
                                num(par,i)+=parameters(par,i+j)*kernel[j+3*sigma]*weights(par,i+j);
                                den(par,i)+=kernel[j+3*sigma]*weights(par,i+j);
                            }
                    }
                }
            }

            for(par=0;par<6;par++)
                for(i=0;i<_transformations.size();i++)
                    kr(par,i)=num(par,i)/den(par,i);

            cout<<"Iter "<<iter<<": median = ";
            cout.flush();

            //recalculate weights
            //for(par=0;par<6;par++)
            //{
            error.clear();
            tmp.clear();

            for(i=0;i<_transformations.size();i++)
            {
                //cout<<i<<" ";
                //cout.flush();
                ///CHange: TRE
                RigidTransformation processed;
                processed.PutTranslationX(kr(0,i));
                processed.PutTranslationY(kr(1,i));
                processed.PutTranslationZ(kr(2,i));
                processed.PutRotationX(kr(3,i));
                processed.PutRotationY(kr(4,i));
                processed.PutRotationZ(kr(5,i));

                RigidTransformation orig = _transformations[_slice_order[i]];

                //need to convert the transformations back to the original coordinate system
                m = orig.GetMatrix();
                m=mo*m*imo;
                orig.PutMatrix(m);

                m = processed.GetMatrix();
                m=mo*m*imo;
                processed.PutMatrix(m);

                RealImage slice = _slices[_slice_order[i]];

                double tre=0;
                int n=0;
                double x, y, z, xx, yy, zz, e = 0;
                for(int ii=0;ii<_reconstructed.GetX();ii=ii+3)
                    for(int jj=0;jj<_reconstructed.GetY();jj=jj+3)
                        for(int kk=0;kk<_reconstructed.GetZ();kk=kk+3)
                            if(_reconstructed(ii,jj,kk)>-1)
                            {
                                //cout<<ii<<" "<<jj<<" "<<kk<<endl;
                                //cout.flush();
                                x=ii; y=jj; z=kk;
                                _reconstructed.ImageToWorld(x,y,z);
                                xx=x;yy=y;zz=z;
                                orig.Transform(x,y,z);
                                processed.Transform(xx,yy,zz);
                                x-=xx;
                                y-=yy;
                                z-=zz;
                                e += sqrt(x*x+y*y+z*z);
                                n++;
                            }
                if(n>0)
                {
                    e/=n;
                    error.push_back(e);
                    tmp.push_back(e);
                }
                else
                    error.push_back(-1);

            }

            sort(tmp.begin(),tmp.end());
            median = tmp[round(tmp.size()*0.5)];
            cout<<median<<" ";

            for(i=0;i<_transformations.size();i++)
            {
                //if(weights(par,i)>0)
                if(error[i]>=0)
                {
                    if(error[i]<=median*1.35)
                        for(par=0;par<6;par++)
                            weights(par,i)=1;
                    else
                        for(par=0;par<6;par++)
                            weights(par,i)=median*1.35/error[i];
                }
                else
                    for(par=0;par<6;par++)
                        weights(par,i)=0;
            }
            //}
            cout<<endl;

        }

        ofstream fileOut2("motion-processed.txt", ofstream::out | ofstream::app);
        ofstream fileOut3("weights.txt", ofstream::out | ofstream::app);
        ofstream fileOut4("outliers.txt", ofstream::out | ofstream::app);
        ofstream fileOut5("empty.txt", ofstream::out | ofstream::app);

        for(i=0;i<_transformations.size();i++)
        {
            fileOut3<<weights(j,i);

            if(weights(0,i)<=0)
                fileOut5<<_slice_order[i]<<" ";

            _transformations[_slice_order[i]].PutTranslationX(kr(0,i));
            _transformations[_slice_order[i]].PutTranslationY(kr(1,i));
            _transformations[_slice_order[i]].PutTranslationZ(kr(2,i));
            _transformations[_slice_order[i]].PutRotationX(kr(3,i));
            _transformations[_slice_order[i]].PutRotationY(kr(4,i));
            _transformations[_slice_order[i]].PutRotationZ(kr(5,i));

            for(j=0;j<6;j++)
            {
                fileOut2<<kr(j,i);
                if(j<5)
                    fileOut2<<",";
                else
                    fileOut2<<endl;
            }

        }

        //Put origin back
        for(i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m=mo*m*imo;
            _transformations[i].PutMatrix(m);
        }

    }





    class ParallelSliceToVolumeRegistrationSH {
    public:
        ReconstructionDWI *reconstructor;

        ParallelSliceToVolumeRegistrationSH(ReconstructionDWI *_reconstructor) :
        reconstructor(_reconstructor) { }

        void operator() (const blocked_range<size_t> &r) const {

            ImageAttributes attr = reconstructor->_reconstructed.Attributes();

            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {


                GreyPixel smin, smax;
                GreyImage target;
                RealImage slice, w, b, t;


                char buffer[256];

                //                t = reconstructor->_slices[inputIndex];
                //                resampling.SetInput(&reconstructor->_slices[inputIndex]);
                //                resampling.SetOutput(&t);
                //                resampling.Run();
                //                target=t;

                target = reconstructor->_slices[inputIndex];

                int ii = inputIndex;

                target.GetMinMax(&smin, &smax);

                if (smax > -1 && abs(smax - smin) > 1) {



                    //put origin to zero
                    RigidTransformation offset;
                    reconstructor->ResetOrigin(target,offset);
                    Matrix mo = offset.GetMatrix();
                    Matrix m = reconstructor->_transformations[inputIndex].GetMatrix();
                    m=m*mo;
                    reconstructor->_transformations[inputIndex].PutMatrix(m);

                    GreyImage source = reconstructor->_simulated_signal.GetRegion(0,0,0,reconstructor->_stack_index[inputIndex],reconstructor->_simulated_signal.GetX(),reconstructor->_simulated_signal.GetY(),reconstructor->_simulated_signal.GetZ(),reconstructor->_stack_index[inputIndex]+1);


                    GreyPixel tmin, tmax;
                    source.GetMinMax(&tmin, &tmax);


                    ParameterList params;
                    Insert(params, "Transformation model", "Rigid");

                    if (smin < 1) {
                        Insert(params, "Background value for image 1", -1);
                    }

                    if (tmin < 0) {
                        Insert(params, "Background value for image 2", -1);
                    }
                    else {
                        if (tmin < 1) {
                            Insert(params, "Background value for image 2", 0);
                        }
                    }

                    GenericRegistrationFilter registration;
                    registration.Parameter(params);

                    registration.Input(&target, &source);
                    Transformation *dofout = nullptr;
                    registration.Output(&dofout);

                    registration.InitialGuess(&reconstructor->_transformations[inputIndex]);
                    registration.GuessParameter();
                    registration.Run();

                    unique_ptr<RigidTransformation> rigidTransf(dynamic_cast<RigidTransformation*>(dofout));
                    reconstructor->_transformations[inputIndex] = *rigidTransf;

                    //undo the offset
                    mo.Invert();
                    m = reconstructor->_transformations[inputIndex].GetMatrix();
                    m=m*mo;
                    reconstructor->_transformations[inputIndex].PutMatrix(m);
                }
            }
        }

            // execute
            void operator() () const {

                parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                             *this );
            }

        };

        void ReconstructionDWI::SliceToVolumeRegistrationSH()
        {
            if (_debug)
                cout << "SliceToVolumeRegistration" << endl;
            ParallelSliceToVolumeRegistrationSH registration(this);
            registration();
        }


        class ParallelNormaliseBiasDTI{
            ReconstructionDWI* reconstructor;
        public:
            RealImage bias, weights;

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

                    RealPixel *pi = slice.Data();
                    RealPixel *pb = b.Data();
                    for(int i = 0; i<slice.NumberOfVoxels(); i++) {
                        if((*pi>-1)&&(scale>0))
                            *pb -= log(scale);
                        pb++;
                        pi++;
                    }

                    //Distribute slice intensities to the volume
                    POINT3D p;
                    int stackIndex = reconstructor->_stack_index[inputIndex];
                    double sliceWeight = reconstructor->_slice_weight[inputIndex];
                    for (int i = 0; i < slice.GetX(); i++)
                        for (int j = 0; j < slice.GetY(); j++)
                            if (slice(i, j, 0) != -1) {
                                //number of volume voxels with non-zero coefficients for current slice voxel
                                int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                                //add contribution of current slice voxel to all voxel volumes
                                //to which it contributes
                                for (int k = 0; k < n; k++) {
                                    p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                    bias(p.x, p.y, p.z,stackIndex) += sliceWeight * p.value * b(i, j, 0);
                                    weights(p.x, p.y, p.z,stackIndex) += sliceWeight * p.value;
                                }
                            }
                    //end of loop for a slice inputIndex
                }
            }

            ParallelNormaliseBiasDTI( ParallelNormaliseBiasDTI& x, split ) :
            reconstructor(x.reconstructor)
            {
                bias.Initialize( reconstructor->_simulated_signal.Attributes() );
                bias = 0;
                weights.Initialize( reconstructor->_simulated_signal.Attributes() );
                weights = 0;
            }

            void join( const ParallelNormaliseBiasDTI& y ) {
                bias += y.bias;
                weights += y.weights;
            }

            ParallelNormaliseBiasDTI( ReconstructionDWI *reconstructor ) :
            reconstructor(reconstructor)
            {
                bias.Initialize( reconstructor->_simulated_signal.Attributes() );
                bias = 0;
                weights.Initialize( reconstructor->_simulated_signal.Attributes() );
                weights = 0;
            }

            // execute
            void operator() () {

                parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()),
                                *this );
            }
        };

        class ParallelNormaliseBiasDTI2 {
            ReconstructionDWI *reconstructor;
            RealImage& bias;
        public:
            ParallelNormaliseBiasDTI2( ReconstructionDWI *_reconstructor, RealImage &_bias ) :
            reconstructor(_reconstructor), bias(_bias)
            {}

            void operator() (const blocked_range<size_t> &r) const {
                for ( size_t inputIndex = r.begin(); inputIndex < r.end(); ++inputIndex) {
                    if(reconstructor->_debug) {
                        cout<<inputIndex<<" ";
                    }
                    int ii = inputIndex;
                    char buffer[256];

                    // alias the current slice
                    RealImage& slice = reconstructor->_slices[inputIndex];

                    //read the current bias image
                    RealImage b(slice.Attributes());
                    //sprintf(buffer,"biaszero%i.nii.gz",ii);
                    //b.Write(buffer);

                    POINT3D p;
                    int stackIndex = reconstructor->_stack_index[inputIndex];
                    double sliceWeight = reconstructor->_slice_weight[inputIndex];
                    for (int i = 0; i < slice.GetX(); i++)
                        for (int j = 0; j < slice.GetY(); j++)
                            if (slice(i, j, 0) != -1) {
                                double weight = 0;
                                //number of volume voxels with non-zero coefficients for current slice voxel
                                int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                                //add contribution of current slice voxel to all voxel volumes
                                //to which it contributes
                                for (int k = 0; k < n; k++) {
                                    p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                    b(i,j,0) += p.value * bias(p.x, p.y, p.z,stackIndex);
                                    weight += p.value;
                                }
                                if( weight > 0 ) {
                                    b(i,j,0)/=weight;
                                }


                            }

                    reconstructor->_bias[inputIndex] -= b;


                    //end of loop for a slice inputIndex
                }
            }

            // execute
            void operator() () const {

                parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size()),
                             *this );
            }

        };

        void ReconstructionDWI::NormaliseBiasDTI(int iter, Array<RigidTransformation> &stack_transformations, int order)
        {
            if(_debug)
                cout << "Normalise Bias ... ";

            ParallelNormaliseBiasDTI parallelNormaliseBiasDTI(this);
            parallelNormaliseBiasDTI();
            RealImage bias = parallelNormaliseBiasDTI.bias;
            RealImage weights = parallelNormaliseBiasDTI.weights;


            bias /= weights;
            if (_debug) {
                char buffer[256];
                sprintf(buffer,"averagebiasorig%i.nii.gz",iter);
                bias.Write(buffer);
            }

            RealImage m = bias;
            m=0;
            RealPixel *pm, *pb;
            pm = m.Data();
            pb = bias.Data();
            for (int i = 0; i<m.NumberOfVoxels();i++) {
                if(*pb!=0)
                    *pm =1;
                pm++;
                pb++;
            }

            if (_debug) {
                char buffer[256];
                sprintf(buffer,"averagebiasmask%i.nii.gz",iter);
                m.Write(buffer);
            }



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
                sprintf(buffer,"averagebiasSH%i.nii.gz",iter);
                bias.Write(buffer);
            }

            ParallelNormaliseBiasDTI2 parallelNormaliseBiasDTI2(this,bias);
            parallelNormaliseBiasDTI2();


            if(_debug)
                cout << "done." << endl;
        }


        void ReconstructionDWI::NormaliseBiasSH2(int iter, Array<RigidTransformation> &stack_transformations, int order)
        {
            if(_debug)
                cout << "Normalise Bias ... ";

            // Calculate stack bias
            ParallelNormaliseBiasDTI parallelNormaliseBiasDTI(this);
            parallelNormaliseBiasDTI();
            RealImage bias = parallelNormaliseBiasDTI.bias;
            RealImage weights = parallelNormaliseBiasDTI.weights;

            bias /= weights;
            if (_debug) {
                char buffer[256];
                sprintf(buffer,"averagebiasorig%i.nii.gz",iter);
                bias.Write(buffer);
            }

            int dirIndex;
            double gx,gy,gz;
            int nStacks = bias.GetT();
            Matrix dirs_xyz(nStacks,3);
            for (dirIndex = 0; dirIndex < nStacks; dirIndex++)
            {
                gx=_directionsDTI[0][dirIndex+1];
                gy=_directionsDTI[1][dirIndex+1];
                gz=_directionsDTI[2][dirIndex+1];
                RotateDirections(gx,gy,gz,stack_transformations[dirIndex]);
                dirs_xyz(dirIndex,0) = gx;
                dirs_xyz(dirIndex,1) = gy;
                dirs_xyz(dirIndex,2) = gz;
            }

            SphericalHarmonics sh;
            Matrix SHT = sh.SHbasis(dirs_xyz,order);
            Matrix tSHT = SHT;
            tSHT.Transpose();

            RealImage biasSH(_SH_coeffs);
            biasSH=0;
            RealImage mask(_SH_coeffs);
            mask=0;

            int nCoeff = _SH_coeffs.GetT();
            Vector c(nCoeff),s,sb,cb,ev;
            Matrix w(nStacks,nStacks),tmp,tmp1,tmp2;
            double b,det;
            int i,j,k,t;
            for(i=0;i<biasSH.GetX();i++)
                for(j=0;j<biasSH.GetY();j++)
                    for(k=0;k<biasSH.GetZ();k++)
                        if(_SH_coeffs(i,j,k,0)>0)
                        {
                            //SH coeff
                            for(t=0; t<nCoeff;t++)
                                c(t)=_SH_coeffs(i,j,k,t);
                            //simulate HR signal
                            s = SHT*c;
                            //add bias to it
                            for(t=0; t<nStacks;t++)
                            {
                                b=bias(i,j,k,t);
                                s(t)/=exp(-b);
                                w(t,t)=weights(i,j,k,t);
                            }
                            // calculate SH coefficients with the effect of the bias
                            //and bias itself
                            tmp = tSHT*w*SHT;
                            tmp.Eigenvalues(tmp1,ev,tmp2);

                            det=1;
                            for(t=0;t<ev.Rows();t++)
                                det*=ev(t);
                            if(det>0.001)
                            {
                                tmp.Invert();
                                cb = tmp*tSHT*w*s;
                                for(t=0;t<SHT.Cols();t++)
                                    if((cb(t)!=0)&&(c(t)!=0))
                                    {
                                        biasSH(i,j,k,t)=log(cb(t)/c(t));
                                        mask(i,j,k,t)=1;
                                    }
                            }

                        }


            if (_debug) {
                char buffer[256];
                sprintf(buffer,"averagebiasmask%i.nii.gz",iter);
                mask.Write(buffer);
            }




            GaussianBlurring<RealPixel> gb(_sigma_bias);
            gb.Input(&biasSH);
            gb.Output(&biasSH);
            gb.Run();

            gb.Input(&mask);
            gb.Output(&mask);
            gb.Run();

            biasSH/=mask;



            if (_debug) {
                char buffer[256];
                sprintf(buffer,"averagebiasSH%i.nii.gz",iter);
                biasSH.Write(buffer);
            }

            //apply to stacks
            RealImage shCoeffs = _SH_coeffs/biasSH;

            for(i=0;i<biasSH.GetX();i++)
                for(j=0;j<biasSH.GetY();j++)
                    for(k=0;k<biasSH.GetZ();k++)
                        if(_SH_coeffs(i,j,k,0)>0)
                        {
                            //SH coeff
                            for(t=0; t<nCoeff;t++)
                            {
                                c(t)=_SH_coeffs(i,j,k,t);
                                cb(t)=shCoeffs(i,j,k,t);
                            }
                            //simulate HR signal
                            s = SHT*c;
                            sb = SHT*cb;
                            for(t=0; t<nCoeff;t++)
                            {
                                bias(i,j,k,t)=log(sb(t)/s(t));
                            }
                        }


            //redistribute to slices
            ParallelNormaliseBiasDTI2 parallelNormaliseBiasDTI2(this,bias);
            parallelNormaliseBiasDTI2();


            if(_debug)
                cout << "done." << endl;

        }





} // namespace svrtk
