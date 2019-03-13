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

#include "mirtk/ReconstructionCardiac4D.h"
#include "mirtk/Resampling.h"
#include "mirtk/GenericRegistrationFilter.h"
#include "mirtk/Transformation.h"
#include "mirtk/HomogeneousTransformation.h"
#include "mirtk/RigidTransformation.h"
#include "mirtk/ImageTransformation.h"
#include "mirtk/LinearInterpolateImageFunction.hxx"
#include <math.h>




using namespace std;

namespace mirtk {
    
    
    // -----------------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------------
    ReconstructionCardiac4D::ReconstructionCardiac4D():Reconstruction()
    {
        _recon_type = _3D;
        
        _no_sr = false;
    }
    
    // -----------------------------------------------------------------------------
    // Destructor
    // -----------------------------------------------------------------------------
    ReconstructionCardiac4D::~ReconstructionCardiac4D() { }
    
    // -----------------------------------------------------------------------------
    // Set Slice R-R Intervals
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SetSliceRRInterval( Array<double> rr )
    {
        _slice_rr = rr;
    }
    
    void ReconstructionCardiac4D::SetSliceRRInterval( double rr )
    {
        Array<double> slice_rr;
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            slice_rr.push_back(rr);
        }
        _slice_rr = slice_rr;
    }
    
    
    // -----------------------------------------------------------------------------
    // Set Slice R-R Intervals
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SetLocRRInterval( Array<double> rr )
    {
        Array<double> slice_rr;
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            slice_rr.push_back(rr[_loc_index[inputIndex]]);
        }
        _slice_rr = slice_rr;
    }
    
    
    // -----------------------------------------------------------------------------
    // Set Slice Cardiac Phases
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SetSliceCardiacPhase( Array<double> cardiacphases )
    {
        _slice_cardphase = cardiacphases;
    }
    
    void ReconstructionCardiac4D::SetSliceCardiacPhase()
    {
        _slice_cardphase.clear();
        for (unsigned int i=0; i<_slices.size(); i++)
            _slice_cardphase.push_back( 0 );
    }
    
    // -----------------------------------------------------------------------------
    // Determine Reconstructed Spatial Resolution
    // -----------------------------------------------------------------------------
    //determine resolution of volume to reconstruct
    double ReconstructionCardiac4D::GetReconstructedResolutionFromTemplateStack( RealImage stack )
    {
        double dx, dy, dz, d;
        stack.GetPixelSize(&dx, &dy, &dz);
        if ((dx <= dy) && (dx <= dz))
            d = dx;
        else if (dy <= dz)
            d = dy;
        else
            d = dz;
        return d;
    }
    
    
    // -----------------------------------------------------------------------------
    // Get Slice-Location Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::ReadSliceTransformation(char* slice_transformations_folder)
    {
        if (_slices.size()==0)
        {
            cerr << "Please create slices before reading transformations!" << endl;
            exit(1);
        }
        
        int nLoc = _loc_index.back() + 1;
        
        char name[256];
        char path[256];
        Array<RigidTransformation> loc_transformations;
        Transformation *transformation;
        RigidTransformation *rigidTransf;
        
        // Read transformations from file
        cout << "Reading transformations:" << endl;
        for (int iLoc = 0; iLoc < nLoc; iLoc++) {
            if (slice_transformations_folder != NULL) {
                sprintf(name, "/transformation%05i.dof", iLoc);
                strcpy(path, slice_transformations_folder);
                strcat(path, name);
            }
            else {
                sprintf(path, "transformation%03i.dof", iLoc);
            }
            transformation = Transformation::New(path);
            rigidTransf = dynamic_cast<RigidTransformation*>(transformation);
            loc_transformations.push_back(*rigidTransf);
            delete transformation;
            cout << path << endl;
        }
        
        // Assign transformations to single-frame images
        _transformations.clear();
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            _transformations.push_back(loc_transformations[_loc_index[inputIndex]]);
        }
        cout << "ReadSliceTransformations complete." << endl;
        
    }
    
    
    // -----------------------------------------------------------------------------
    // Set Reconstructed Cardiac Phases
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SetReconstructedCardiacPhase( Array<double> cardiacphases )
    {
        _reconstructed_cardiac_phases = cardiacphases;
    }
    
    
    // -----------------------------------------------------------------------------
    // Set Reconstructed R-R Interval
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SetReconstructedRRInterval( double rrinterval )
    {
        _reconstructed_rr_interval = rrinterval;
        if (_debug)
            cout<<"Reconstructed R-R interval = "<<_reconstructed_rr_interval<<" s."<<endl;
    }
    
    
    // -----------------------------------------------------------------------------
    // Set Reconstructed Cardiac Phases
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SetReconstructedTemporalResolution( double temporalresolution )
    {
        _reconstructed_temporal_resolution = temporalresolution;
        if (_debug)
            cout<<"Reconstructed temporal resolution = "<<_reconstructed_temporal_resolution<<" s."<<endl;
    }
    
    
    // -----------------------------------------------------------------------------
    // Initialise Reconstructed Volume from Static Mask
    // -----------------------------------------------------------------------------
    // Create zero image as a template for reconstructed volume
    void ReconstructionCardiac4D::CreateTemplateCardiac4DFromStaticMask( RealImage mask, double resolution )
    {
        // Get mask attributes - image size and voxel size
        ImageAttributes attr = mask.GetImageAttributes();
        
        // Set temporal dimension
        attr._t = _reconstructed_cardiac_phases.size();
        attr._dt = _reconstructed_temporal_resolution;
        
        // Create volume
        RealImage volume4D(attr);
        
        // Initialise volume values
        volume4D = 0;
        
        // Resample to specified resolution
        
        //NearestNeighborInterpolateImageFunction interpolator;
        
        // InterpolationMode interpolation = Interpolation_NN;
        // UniquePtr<InterpolateImageFunction> interpolator;
        // interpolator.reset(InterpolateImageFunction::New(interpolation));
        
        InterpolationMode interpolation = Interpolation_NN;
        UniquePtr<InterpolateImageFunction> interpolator;
        interpolator.reset(InterpolateImageFunction::New(interpolation));
        
        
        Resampling<RealPixel> resampling(resolution,resolution,resolution);
        resampling.Input(&volume4D);
        resampling.Output(&volume4D);
        resampling.Interpolator(interpolator.get());
        resampling.Run();
        
        // Set recontructed 4D volume
        _reconstructed4D = volume4D;
        _template_created = true;
        
//        //.........................................................................................
//
//        // 10-02
//
//        Resampling<RealPixel> resampling2(resolution,resolution,resolution);
//        GenericLinearInterpolateImageFunction<RealImage> interpolator2;
//
//        cout << "4" << endl;
//
//        RealImage template_stack = mask.GetRegion(0,0,0,0,mask.GetX(),mask.GetY(),mask.GetZ(),1);
//        resampling2.Input(&template_stack);
//        resampling2.Output(&template_stack);
//        resampling2.Interpolator(&interpolator2);
//        resampling2.Run();
//
//        cout << "5" << endl;
//
//        for (int t=0; t<volume4D.GetT(); t++)
//            for (int z=0; z<volume4D.GetZ(); z++)
//                for (int y=0; y<volume4D.GetY(); y++)
//                    for (int x=0; x<volume4D.GetX(); x++)
//                        volume4D(x,y,z,t) = template_stack(x,y,z);
//
//
//        cout << "6" << endl;
//
//        _reconstructed4D = volume4D;
//        _template_created = true;
//
//        //.........................................................................................
        
        
        // Set reconstructed 3D volume for reference by existing functions of irtkReconstruction class
        ImageAttributes attr4d = volume4D.GetImageAttributes();
        ImageAttributes attr3d = attr4d;
        attr3d._t = 1;
        RealImage volume3D(attr3d);
        _reconstructed = volume3D;
        
        // Debug
        if (_debug)
        {
            cout << "CreateTemplateCardiac4DFromStaticMask: created template 4D volume with "<<attr4d._t<<" time points and "<<attr4d._dt<<" s temporal resolution."<<endl;
        }
    }
    
    
    // -----------------------------------------------------------------------------
    // Match Stack Intensities With Masking
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::MatchStackIntensitiesWithMasking(Array<RealImage>& stacks,
                                                                   Array<RigidTransformation>& stack_transformations, double averageValue, bool together)
    {
        if (_debug)
            cout << "Matching intensities of stacks. ";
        
        cout<<setprecision(6);
        
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
            ImageAttributes attr = stacks[ind].GetImageAttributes();
            attr._t = 1;
            m.Initialize(attr);
            m = 0;
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
                                for ( int f = 0; f < stacks[ind].GetT(); f++)
                                {
                                    sum += stacks[ind](i, j, k, f);
                                    num++;
                                }
                            }
                        }
                    }
            if(_debug)
            {
                sprintf(buffer,"maskformatching%03i.nii.gz",ind);
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
            for(ind=0;ind<stack_average.size();ind++)
                global_average += stack_average[ind];
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
                sprintf(buffer, "rescaledstack%03i.nii.gz", ind);
                stacks[ind].Write(buffer);
            }
            
            cout << "Slice intensity factors are ";
            for (ind = 0; ind < stack_average.size(); ind++)
                cout << _stack_factor[ind] << " ";
            cout << endl;
            cout << "The new average value is " << averageValue << endl;
        }
        
        cout<<setprecision(3);
        
    }
    
    // -----------------------------------------------------------------------------
    // Initialise Stack Factor
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::InitStackFactor(Array<RealImage> &stacks)
    {
        _stack_factor.clear();
        for(unsigned int stackIndex=0; stackIndex<stacks.size(); stackIndex++)
            _stack_factor.push_back(1);
    }
    
    
    
    
    // -----------------------------------------------------------------------------
    // Create Slices and Associated Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CreateSlicesAndTransformationsCardiac4D( Array<RealImage> &stacks,
                                                                          Array<RigidTransformation> &stack_transformations,
                                                                          Array<double> &thickness,
                                                                          const Array<RealImage> &probability_maps )
    {
        
        double sliceAcqTime;
        int loc_index = 0;
        
        if (_debug)
            cout << "CreateSlicesAndTransformations" << endl;
        
        //for each stack
        for (unsigned int i = 0; i < stacks.size(); i++) {
            //image attributes contain image and voxel size
            ImageAttributes attr = stacks[i].GetImageAttributes();
            
            //attr._z is number of slices in the stack
            for (int j = 0; j < attr._z; j++) {
                
                //attr._t is number of frames in the stack
                for (int k = 0; k < attr._t; k++) {
                    
                    //create slice by selecting the appropreate region of the stack
                    RealImage slice = stacks[i].GetRegion(0, 0, j, k, attr._x, attr._y, j + 1, k + 1);
                    //set correct voxel size in the stack. Z size is equal to slice thickness.
                    slice.PutPixelSize(attr._dx, attr._dy, thickness[i], attr._dt);
                    //set slice acquisition time
                    sliceAcqTime = attr._torigin + k * attr._dt;
                    _slice_time.push_back(sliceAcqTime);
                    //set slice temporal resolution
                    _slice_dt.push_back(attr._dt);
                    //remember the slice
                    _slices.push_back(slice);
                    _simulated_slices.push_back(slice);
                    _simulated_weights.push_back(slice);
                    _simulated_inside.push_back(slice);
                    //remeber stack indices for this slice
                    _stack_index.push_back(i);
                    _loc_index.push_back(loc_index);
                    _stack_loc_index.push_back(j);
                    _stack_dyn_index.push_back(k);
                    //initialize slice transformation with the stack transformation
                    _transformations.push_back(stack_transformations[i]);
                    //initialise slice exclusion flags
                    _slice_excluded.push_back(0);
                    if ( probability_maps.size() > 0 ) {
                        RealImage proba = probability_maps[i].GetRegion(0, 0, j, k, attr._x, attr._y, j + 1, k + 1);
                        proba.PutPixelSize(attr._dx, attr._dy, thickness[i], attr._dt);
                        _probability_maps.push_back(proba);
                    }
                    //initialise cardiac phase to use for 2D-3D registration
                    _slice_svr_card_index.push_back(0);
                }
                loc_index++;
            }
        }
        cout << "Number of images: " << _slices.size() << endl;
        //set excluded slices
        for (unsigned int i = 0; i < _force_excluded.size(); i++)
            _slice_excluded[_force_excluded[i]] = 1;
        for (unsigned int i = 0; i < _force_excluded_stacks.size(); i++)
            for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                if (_force_excluded_stacks[i]==_stack_index[inputIndex])
                    _slice_excluded[inputIndex] = 1;
        for (unsigned int i = 0; i < _force_excluded_locs.size(); i++)
            for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
                if (_force_excluded_locs[i]==_loc_index[inputIndex])
                    _slice_excluded[inputIndex] = 1;
    }
    
    // -----------------------------------------------------------------------------
    // Reset Slices and Associated Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::ResetSlicesAndTransformationsCardiac4D()
    {
        _slice_time.clear();
        _slice_dt.clear();
        _slices.clear();
        _simulated_slices.clear();
        _simulated_weights.clear();
        _simulated_inside.clear();
        _stack_index.clear();
        _loc_index.clear();
        _stack_loc_index.clear();
        _stack_dyn_index.clear();
        _transformations.clear();
        _slice_excluded.clear();
        _probability_maps.clear();
        
    }
    
    
    // -----------------------------------------------------------------------------
    // InitCorrectedSlices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::InitCorrectedSlices()
    {
        _corrected_slices.clear();
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _corrected_slices.push_back(_slices[inputIndex]);
    }
    
    
    // -----------------------------------------------------------------------------
    // InitError
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::InitError()
    {
        _error.clear();
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            _error.push_back(_slices[inputIndex]);
    }
    
    
    // -----------------------------------------------------------------------------
    // Initialise Slice Temporal Weights
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::InitSliceTemporalWeights()
    {
        _slice_temporal_weight.clear();
        _slice_temporal_weight.resize(_reconstructed_cardiac_phases.size());
        for (unsigned int outputIndex = 0; outputIndex < _reconstructed_cardiac_phases.size(); outputIndex++)
        {
            _slice_temporal_weight[outputIndex].resize(_slices.size());
        }
    }
    
    
    // -----------------------------------------------------------------------------
    // Calculate Slice Temporal Weights
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateSliceTemporalWeights()
    {
        if (_debug)
            cout << "CalculateSliceTemporalWeights" << endl;
        InitSliceTemporalWeights();
        for (unsigned int outputIndex = 0; outputIndex < _reconstructed_cardiac_phases.size(); outputIndex++)
        {
            for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
            {
                _slice_temporal_weight[outputIndex][inputIndex] = CalculateTemporalWeight( _reconstructed_cardiac_phases[outputIndex], _slice_cardphase[inputIndex], _slice_dt[inputIndex], _slice_rr[inputIndex], _wintukeypct );
            }
        }
    }
    
    
    // -----------------------------------------------------------------------------
    // Calculate Angular Difference
    // -----------------------------------------------------------------------------
    // Angular difference between output cardiac phase (cardphase0) and slice cardiac phase (cardphase)
    double ReconstructionCardiac4D::CalculateAngularDifference( double cardphase0, double cardphase )
    {
        double angdiff;
        angdiff = ( cardphase - cardphase0 ) - ( 2 * PI ) * floor( ( cardphase - cardphase0 ) / ( 2 * PI ) );
        angdiff = ( angdiff <= PI ) ? angdiff : - ( 2 * PI - angdiff);
        return angdiff;
    }
    
    
    // -----------------------------------------------------------------------------
    // Calculate Temporal Weight
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateTemporalWeight( double cardphase0, double cardphase, double dt, double rr, double alpha )
    {
        double angdiff, dtrad, sigma, temporalweight;
        
        // Angular Difference
        angdiff = CalculateAngularDifference( cardphase0, cardphase );
        
        // Temporal Resolution in Radians
        dtrad = 2 * PI * dt / rr;
        
        // Temporal Weight
        if (_is_temporalpsf_gauss) {
            // Gaussian
            sigma = dtrad / 2.355;  // sigma = ~FWHM/2.355
            temporalweight = exp( -( angdiff * angdiff ) / (2 * sigma * sigma ) );
        }
        else {
            // Sinc
            temporalweight = sinc( PI * angdiff / dtrad ) * wintukey( angdiff, alpha );
        }
        
        return temporalweight;
    }
    
    
    // -----------------------------------------------------------------------------
    // Sinc Function
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::sinc(double x)
    {
        if (x == 0)
            return 1;
        return sin(x)/x;
    }
    
    // -----------------------------------------------------------------------------
    // Tukey Window Function
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::wintukey( double angdiff, double alpha )
    {
        // angdiff = angular difference (-PI to +PI)
        // alpha   = amount of window with tapered cosine edges (0 to 1)
        if ( fabs( angdiff ) > PI * ( 1 - alpha ) )
            return ( 1 + cos( ( fabs( angdiff ) - PI * ( 1 - alpha ) ) / alpha ) ) / 2;
        return 1;
    }
    
    // -----------------------------------------------------------------------------
    // ParallelCoeffInitCardiac4D
    // -----------------------------------------------------------------------------
    class ParallelCoeffInitCardiac4D {
    public:
        ReconstructionCardiac4D *reconstructor;
        
        ParallelCoeffInitCardiac4D(ReconstructionCardiac4D *_reconstructor) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                bool slice_inside;
                
                //current slice
                //RealImage slice;
                
                //get resolution of the volume
                double vx, vy, vz;
                reconstructor->_reconstructed4D.GetPixelSize(&vx, &vy, &vz);
                //volume is always isotropic
                double res = vx;
                //read the slice
                RealImage& slice = reconstructor->_slices[inputIndex];
                
                //prepare structures for storage
                POINT3D p;
                VOXELCOEFFS empty;
                SLICECOEFFS slicecoeffs(slice.GetX(), Array < VOXELCOEFFS > (slice.GetY(), empty));
                
                //to check whether the slice has an overlap with mask ROI
                slice_inside = false;
                
                if (reconstructor->_slice_excluded[inputIndex] == 0) {
                    
                    //start of a loop for a slice inputIndex
                    cout << inputIndex << " ";
                    cout.flush();
                    
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
                        sigmaz = 1 * dz / 2.3548;
                    }
                    
                    if(reconstructor->_no_sr)
                    {
                        sigmax = 0.5 * dx / 2.3548; // 1.2
                        sigmay = 0.5 * dy / 2.3548; // 1.2
                        sigmaz = 0.5 * dz / 2.3548;   // 1
                    }
                    
                    
                    if(reconstructor->_recon_type == _1D)
                    {
                        sigmax = 0.5 * dx / 2.3548;
                        sigmay = 0.5 * dy / 2.3548;
                        sigmaz = dz / 2.3548;
                    }
                    
                    if(reconstructor->_recon_type == _interpolate)
                    {
                        sigmax = 0.5 * dx / 2.3548; // dx
                        sigmay = 0.5 * dx / 2.3548; // dx
                        sigmaz = 0.5 * dx / 2.3548; // dx
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
                                reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                reconstructor->_reconstructed4D.WorldToImage(x, y, z);
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
                                            reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                            //Change to image coordinates
                                            reconstructor->_reconstructed4D.WorldToImage(x, y, z);
                                            
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
                                                if ((l >= 0) && (l < reconstructor->_reconstructed4D.GetX()))
                                                    for (m = ny; m <= ny + 1; m++)
                                                        if ((m >= 0) && (m < reconstructor->_reconstructed4D.GetY()))
                                                            for (n = nz; n <= nz + 1; n++)
                                                                if ((n >= 0) && (n < reconstructor->_reconstructed4D.GetZ())) {
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
                                                if ((l >= 0) && (l < reconstructor->_reconstructed4D.GetX()))
                                                    for (m = ny; m <= ny + 1; m++)
                                                        if ((m >= 0) && (m < reconstructor->_reconstructed4D.GetY()))
                                                            for (n = nz; n <= nz + 1; n++)
                                                                if ((n >= 0) && (n < reconstructor->_reconstructed4D.GetZ())) {
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
                    
                }  // if(_slice_excluded[inputIndex]==0)
                
                reconstructor->_volcoeffs[inputIndex] = slicecoeffs;
                reconstructor->_slice_inside[inputIndex] = slice_inside;
                
            }  //end of loop through the slices
            
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                         *this );
            //init.terminate();
        }
        
    };
    
    
    
    // -----------------------------------------------------------------------------
    // Calculate Transformation Matrix Between Slices and Voxels
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CoeffInitCardiac4D()
    {
        if (_debug)
            cout << "CoeffInit" << endl;
        
        
        //----------------------------------------------------
        
        ImageAttributes attr = _reconstructed4D.GetImageAttributes();
        _slice_contributions_volume.Initialize(attr);
        
        
//        CoordImage _slice_contributions_volume;
//        Array<Array<Point>> _slice_contributions_array;
        
        POINT3DS pi;
        pi.x = -1, pi.y = -1, pi.z = -1, pi.i = -1, pi.value = -1000, pi.w = 0;
        Array<POINT3DS> p_array;
        p_array.push_back(pi);
        
        int c_index = 0;
        
        for (int x=0; x<_reconstructed4D.GetX(); x++) {
            for (int y=0; y<_reconstructed4D.GetY(); y++) {
                for (int z=0; z<_reconstructed4D.GetZ(); z++) {
                    for (int t=0; t<_reconstructed4D.GetT(); t++) {

                        _slice_contributions_volume(x,y,z,t) = c_index;
                        _slice_contributions_array.push_back(p_array);
                        c_index++;
                        
                    }
                }
            }
        }
        
        
        //----------------------------------------------------
        
        
        //clear slice-volume matrix from previous iteration
        _volcoeffs.clear();
        _volcoeffs.resize(_slices.size());
        
        //clear indicator of slice having and overlap with volumetric mask
        _slice_inside.clear();
        _slice_inside.resize(_slices.size());
        
        cout << "Initialising matrix coefficients...";
        cout.flush();
        ParallelCoeffInitCardiac4D coeffinit(this);
        coeffinit();
        cout << " ... done." << endl;
        
        //prepare image for volume weights, will be needed for Gaussian Reconstruction
        cout << "Computing 4D volume weights..." << endl;
        ImageAttributes volAttr = _reconstructed4D.GetImageAttributes();
        _volume_weights.Initialize( volAttr );
        _volume_weights = 0;
        
        
        
        
        
        // TODO: investigate if this loop is taking a long time to compute, and consider parallelisation
        int i, j, n, k, outputIndex;
        unsigned int inputIndex;
        POINT3D p;
        cout << "    ... for input slice: ";
        
        double max_P = -1;
        
        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            for ( i = 0; i < _slices[inputIndex].GetX(); i++)
                for ( j = 0; j < _slices[inputIndex].GetY(); j++) {
                    n = _volcoeffs[inputIndex][i][j].size();
                    for (k = 0; k < n; k++) {
                        
                        p = _volcoeffs[inputIndex][i][j][k];
                        
                        if (p.value > max_P)
                            max_P = p.value;
                    }
                }
        }
        
        double p_value_limit = max_P*0.2;

        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            cout << inputIndex << ", ";
            cout.flush();
            for ( i = 0; i < _slices[inputIndex].GetX(); i++)
                for ( j = 0; j < _slices[inputIndex].GetY(); j++) {
                    n = _volcoeffs[inputIndex][i][j].size();
                    for (k = 0; k < n; k++) {
                        p = _volcoeffs[inputIndex][i][j][k];
                        
//                        _volcoeffs[inputIndex][i][j][k].value = max_P;
                        
                        for (outputIndex=0; outputIndex<_reconstructed4D.GetT(); outputIndex++)
                        {
                            _volume_weights(p.x, p.y, p.z, outputIndex) += _slice_temporal_weight[outputIndex][inputIndex] * p.value;
                            
                            if (_reconstructed4D.GetT() == 1)
                                _slice_temporal_weight[outputIndex][inputIndex] = 1;
                            
                            //----------------------------------------------------
                            if (p.value>p_value_limit && _slice_temporal_weight[outputIndex][inputIndex]>p_value_limit) {

                                int array_index = _slice_contributions_volume(p.x, p.y, p.z, outputIndex);
                                
                                POINT3DS ps;
                                ps.x = i, ps.y = j, ps.z = k, ps.i = inputIndex, ps.value = _slices[inputIndex](i,j,0,outputIndex), ps.w = p.value;
                                
                                _slice_contributions_array[array_index].push_back(ps);
                                
                            }
                            //----------------------------------------------------
                            
                        }
                    }
                }
        }
        cout << "\b\b." << endl;
        // if (_debug)
        //     _volume_weights.Write("volume_weights.nii.gz");
        
        
        
        //find average volume weight to modify alpha parameters accordingly
        double sum = 0;
        int num=0;
        for (i=0; i<_volume_weights.GetX(); i++)
            for (j=0; j<_volume_weights.GetY(); j++)
                for (k=0; k<_volume_weights.GetZ(); k++)
                    if (_mask(i,j,k)==1)
                        for (int f=0; f<_volume_weights.GetT(); f++) {
                            sum += _volume_weights(i,j,k,f);
                            num++;
                        }
        
        _average_volume_weight = sum/num;
        
        if(_debug) {
            cout<<"Average volume weight is "<<_average_volume_weight<<endl;
        }
        
    }
    
    
    // -----------------------------------------------------------------------------
    // PSF-Weighted Reconstruction
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::GaussianReconstructionCardiac4D()
    {
        if(_debug)
        {
            cout << "Gaussian reconstruction ... " << endl;
            cout << "\tinput slice:  ";
            cout.flush();
        }
        unsigned int inputIndex, outputIndex;
        int k, n;
        RealImage slice;
        double scale;
        POINT3D p;
        Array<int> voxel_num;
        int slice_vox_num;
        
        //clear _reconstructed image
        _reconstructed4D = 0;
        
        for (inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            
            if (_slice_excluded[inputIndex]==0) {
                
                if(_debug)
                {
                    cout << inputIndex << ", ";
                    cout.flush();
                }
                //copy the current slice
                slice = _slices[inputIndex];
                //alias the current bias image
                RealImage& b = _bias[inputIndex];
                //read current scale factor
                scale = _scale[inputIndex];
                
                slice_vox_num=0;
                
                //Distribute slice intensities to the volume
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
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
                                for (outputIndex=0; outputIndex<_reconstructed_cardiac_phases.size(); outputIndex++)
                                {
                                    _reconstructed4D(p.x, p.y, p.z, outputIndex) += _slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0);
                                }
                            }
                        }
                voxel_num.push_back(slice_vox_num);
            } //end of if (_slice_excluded[inputIndex]==0)
        } //end of loop for a slice inputIndex
        
        //normalize the volume by proportion of contributing slice voxels
        //for each volume voxe
        _reconstructed4D /= _volume_weights;
        
        if(_debug)
        {
            cout << inputIndex << "\b\b." << endl;
            cout << "... Gaussian reconstruction done." << endl << endl;
            cout.flush();
        }
        
        
        // if (_debug)
        //     _reconstructed4D.Write("init.nii.gz");
        
        //now find slices with small overlap with ROI and exclude them.
        
        Array<int> voxel_num_tmp;
        for (unsigned int i=0;i<voxel_num.size();i++)
            voxel_num_tmp.push_back(voxel_num[i]);
        
        //find median
        sort(voxel_num_tmp.begin(),voxel_num_tmp.end());
        int median = voxel_num_tmp[round(voxel_num_tmp.size()*0.5)];
        
        //remember slices with small overlap with ROI
        _small_slices.clear();
        for (unsigned int i=0;i<voxel_num.size();i++)
            if (voxel_num[i]<0.1*median)
                _small_slices.push_back(i);
        
        if (_debug) {
            cout<<"Small slices:";
            for (unsigned int i=0;i<_small_slices.size();i++)
                cout<<" "<<_small_slices[i];
            cout<<endl;
        }
    }
    
    
    // -----------------------------------------------------------------------------
    // Scale Volume
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::ScaleVolumeCardiac4D()
    {
        if (_debug)
            cout << "Scaling volume: ";
        
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
        
        RealPixel *ptr = _reconstructed4D.GetPointerToVoxels();
        for(i=0;i<_reconstructed4D.GetNumberOfVoxels();i++) {
            if(*ptr>0) *ptr = *ptr * scale;
            ptr++;
        }
        cout<<endl;
    }
    
    
    
    // -----------------------------------------------------------------------------
    // Parallel Class to Calculate Corrected Slices
    // -----------------------------------------------------------------------------
    class ParallelCalculateCorrectedSlices {
        ReconstructionCardiac4D *reconstructor;
        
    public:
        ParallelCalculateCorrectedSlices( ReconstructionCardiac4D *_reconstructor ) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                //read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                //alias the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];
                
                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];
                
                //calculate corrected voxels
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i,j,0) != -1) {
                            //bias correct and scale the voxel
                            slice(i,j,0) *= exp(-b(i, j, 0)) * scale;
                        }
                
                reconstructor->_corrected_slices[inputIndex] = slice;
                
            } //end of loop for a slice inputIndex
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                         *this );
            //init.terminate();
        }
        
    };
    
    
    // -----------------------------------------------------------------------------
    // Calculate Corrected Slices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateCorrectedSlices()
    {
        if (_debug)
            cout<<"Calculating corrected slices..."<<endl;
        
        ParallelCalculateCorrectedSlices parallelCalculateCorrectedSlices( this );
        parallelCalculateCorrectedSlices();
        
        if (_debug)
            cout<<"\t...calculating corrected slices done."<<endl;
    }
    
    
    // -----------------------------------------------------------------------------
    // Parallel Simulate Slices
    // -----------------------------------------------------------------------------
    class ParallelSimulateSlicesCardiac4D {
        ReconstructionCardiac4D *reconstructor;
        
    public:
        ParallelSimulateSlicesCardiac4D( ReconstructionCardiac4D *_reconstructor ) :
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
                for ( int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++ )
                    for ( int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++ )
                        if ( reconstructor->_slices[inputIndex](i, j, 0) != -1 ) {
                            double weight = 0;
                            int n = reconstructor->_volcoeffs[inputIndex][i][j].size();
                            for ( int k = 0; k < n; k++ ) {
                                p = reconstructor->_volcoeffs[inputIndex][i][j][k];
                                for ( int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++ ) {
                                    reconstructor->_simulated_slices[inputIndex](i, j, 0) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_reconstructed4D(p.x, p.y, p.z, outputIndex);
                                    weight += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                }
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
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                         *this );
            //init.terminate();
        }
        
    };
    
    
    // -----------------------------------------------------------------------------
    // Simulate Slices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SimulateSlicesCardiac4D()
    {
        if (_debug)
            cout<<"Simulating Slices..."<<endl;
        
        ParallelSimulateSlicesCardiac4D parallelSimulateSlices( this );
        parallelSimulateSlices();
        
        if (_debug)
            cout<<"\t...Simulating Slices done."<<endl;
    }
    
    
    // -----------------------------------------------------------------------------
    // ParallelSimulateStacksCardiac4D
    // -----------------------------------------------------------------------------
    class ParallelSimulateStacksCardiac4D {
    public:
        ReconstructionCardiac4D *reconstructor;
        
        ParallelSimulateStacksCardiac4D(ReconstructionCardiac4D *_reconstructor) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                if (reconstructor->_slice_excluded[inputIndex] != 1) {
                    
                    cout << inputIndex << " ";
                    cout.flush();
                    
                    //get resolution of the volume
                    double vx, vy, vz;
                    reconstructor->_reconstructed4D.GetPixelSize(&vx, &vy, &vz);
                    //volume is always isotropic
                    double res = vx;
                    //read the slice
                    RealImage& slice = reconstructor->_slices[inputIndex];
                    
                    //prepare structures for storage
                    POINT3D p;
                    VOXELCOEFFS empty;
                    SLICECOEFFS slicecoeffs(slice.GetX(), Array < VOXELCOEFFS > (slice.GetY(), empty));
                    
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
                                reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                reconstructor->_reconstructed4D.WorldToImage(x, y, z);
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
                                            reconstructor->_transformations[inputIndex].Transform(x, y, z);
                                            //Change to image coordinates
                                            reconstructor->_reconstructed4D.WorldToImage(x, y, z);
                                            
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
                                                if ((l >= 0) && (l < reconstructor->_reconstructed4D.GetX()))
                                                    for (m = ny; m <= ny + 1; m++)
                                                        if ((m >= 0) && (m < reconstructor->_reconstructed4D.GetY()))
                                                            for (n = nz; n <= nz + 1; n++)
                                                                if ((n >= 0) && (n < reconstructor->_reconstructed4D.GetZ())) {
                                                                    weight = (1 - fabs(l - x)) * (1 - fabs(m - y)) * (1 - fabs(n - z));
                                                                    sum += weight;
                                                                    if (reconstructor->_mask(l, m, n) == 1) {
                                                                        inside = true;
                                                                    }
                                                                }
                                            //if there were no voxels do nothing
                                            if ((sum <= 0) || (!inside))
                                                continue;
                                            //now calculate the transformed PSF
                                            for (l = nx; l <= nx + 1; l++)
                                                if ((l >= 0) && (l < reconstructor->_reconstructed4D.GetX()))
                                                    for (m = ny; m <= ny + 1; m++)
                                                        if ((m >= 0) && (m < reconstructor->_reconstructed4D.GetY()))
                                                            for (n = nz; n <= nz + 1; n++)
                                                                if ((n >= 0) && (n < reconstructor->_reconstructed4D.GetZ())) {
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
                    
                    //Calculate simulated slice
                    reconstructor->_simulated_slices[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
                    reconstructor->_simulated_slices[inputIndex] = 0;
                    
                    reconstructor->_simulated_weights[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
                    reconstructor->_simulated_weights[inputIndex] = 0;
                    
                    for ( int i = 0; i < reconstructor->_slices[inputIndex].GetX(); i++ )
                        for ( int j = 0; j < reconstructor->_slices[inputIndex].GetY(); j++ )
                            if ( reconstructor->_slices[inputIndex](i, j, 0) != -1 ) {
                                double weight = 0;
                                int n = slicecoeffs[i][j].size();
                                for ( int k = 0; k < n; k++ ) {
                                    p = slicecoeffs[i][j][k];
                                    for ( int outputIndex = 0; outputIndex < reconstructor->_reconstructed4D.GetT(); outputIndex++ ) {
                                        reconstructor->_simulated_slices[inputIndex](i, j, 0) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_reconstructed4D(p.x, p.y, p.z, outputIndex);
                                        weight += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value;
                                    }
                                }
                                if( weight > 0 ) {
                                    reconstructor->_simulated_slices[inputIndex](i,j,0) /= weight;
                                    reconstructor->_simulated_weights[inputIndex](i,j,0) = weight;
                                }
                            }
                    
                }  // if(_slice_excluded[inputIndex]==0)
                
            }  //end of loop through the slices
            
        }  //void operator() (const blocked_range<size_t> &r) const {
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                         *this );
            //init.terminate();
        }
        
    };
    
    
    // -----------------------------------------------------------------------------
    // Simulate Stacks
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SimulateStacksCardiac4D(Array<bool> stack_excluded)
    {
        
        cout << "SimulateStacksCardiac4D" << endl;
        
        //Initialise simlated images
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex)    _simulated_slices[inputIndex] = 0;
        
        //Initialise indicator of images having overlap with volume
        _slice_inside.clear();
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex)
            _slice_inside.push_back(true);
        
        //Simulate images
        cout << "Simulating...";
        cout.flush();
        ParallelSimulateStacksCardiac4D simulatestacks(this);
        simulatestacks();
        cout << " ... done." << endl;
        
    }
    
    
    
    // -----------------------------------------------------------------------------
    // Parallel Class to Calculate Error
    // -----------------------------------------------------------------------------
    class ParallelCalculateError {
        ReconstructionCardiac4D *reconstructor;
        
    public:
        ParallelCalculateError( ReconstructionCardiac4D *_reconstructor ) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                //initalize
                reconstructor->_error[inputIndex].Initialize( reconstructor->_slices[inputIndex].GetImageAttributes() );
                reconstructor->_error[inputIndex] = 0;
                
                //read the current slice
                RealImage slice = reconstructor->_slices[inputIndex];
                
                //alias the current bias image
                RealImage& b = reconstructor->_bias[inputIndex];
                
                //identify scale factor
                double scale = reconstructor->_scale[inputIndex];
                
                //calculate error
                for (int i = 0; i < slice.GetX(); i++)
                    for (int j = 0; j < slice.GetY(); j++)
                        if (slice(i,j,0) != -1)
                            if ( reconstructor->_simulated_weights[inputIndex](i,j,0) > 0 ) {
                                //bias correct and scale the voxel
                                slice(i,j,0) *= exp(-b(i, j, 0)) * scale;
                                //subtract simulated voxel
                                slice(i,j,0) -= reconstructor->_simulated_slices[inputIndex](i,j,0);
                                //assign as error
                                reconstructor->_error[inputIndex](i,j,0) = slice(i,j,0);
                            }
            } //end of loop for a slice inputIndex
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_slices.size() ),
                         *this );
            //init.terminate();
        }
        
    };
    
    
    // -----------------------------------------------------------------------------
    // Calculate Error
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateError()
    {
        if (_debug)
            cout<<"Calculating error..."<<endl;
        
        ParallelCalculateError parallelCalculateError( this );
        parallelCalculateError();
        
        if (_debug)
            cout<<"\t...calculating error done."<<endl;
    }
    
    
    // -----------------------------------------------------------------------------
    // Parallel Processing Class for Normalise Bias
    // -----------------------------------------------------------------------------
    class ParallelNormaliseBiasCardiac4D{
        ReconstructionCardiac4D* reconstructor;
        
    public:
        RealImage bias;
        RealImage volweight3d;
        
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
                                volweight3d(p.x,p.y,p.z) += p.value;
                            }
                        }
                //end of loop for a slice inputIndex
            }
        }
        
        ParallelNormaliseBiasCardiac4D( ParallelNormaliseBiasCardiac4D& x, split ) :
        reconstructor(x.reconstructor)
        {
            ImageAttributes attr = reconstructor->_reconstructed4D.GetImageAttributes();
            attr._t = 1;
            bias.Initialize( attr );
            bias = 0;
            volweight3d.Initialize( attr );
            volweight3d = 0;
        }
        
        void join( const ParallelNormaliseBiasCardiac4D& y ) {
            bias += y.bias;
            volweight3d += y.volweight3d;
        }
        
        ParallelNormaliseBiasCardiac4D( ReconstructionCardiac4D *reconstructor ) :
        reconstructor(reconstructor)
        {
            ImageAttributes attr = reconstructor->_reconstructed4D.GetImageAttributes();
            attr._t = 1;
            bias.Initialize( attr );
            bias = 0;
            volweight3d.Initialize( attr );
            volweight3d = 0;
        }
        
        // execute
        void operator() () {
            //task_scheduler_init init(tbb_no_threads);
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()),
                            *this );
            //init.terminate();
        }
    };
    
    
    // -----------------------------------------------------------------------------
    // Normalise Bias
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::NormaliseBiasCardiac4D(int iter, int rec_iter)
    {
        if(_debug)
            cout << "Normalise Bias ... ";
        
        ParallelNormaliseBiasCardiac4D parallelNormaliseBias(this);
        parallelNormaliseBias();
        RealImage bias = parallelNormaliseBias.bias;
        RealImage volweight3d = parallelNormaliseBias.volweight3d;
        
        // normalize the volume by proportion of contributing slice voxels for each volume voxel
        bias /= volweight3d;
        
        if(_debug)
            cout << "done." << endl;
        
        bias = StaticMaskVolume4D(bias,0);
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
            sprintf(buffer,"averagebias_mc%02isr%02i.nii.gz",iter,rec_iter);
            bias.Write(buffer);
        }
        
        for ( int i = 0; i < _reconstructed4D.GetX(); i++)
            for ( int j = 0; j < _reconstructed4D.GetY(); j++)
                for ( int k = 0; k < _reconstructed4D.GetZ(); k++)
                    for ( int f = 0; f < _reconstructed4D.GetT(); f++)
                        if(_reconstructed4D(i,j,k,f)!=-1)
                            _reconstructed4D(i,j,k,f) /=exp(-bias(i,j,k));
        
    }
    
    
    // -----------------------------------------------------------------------------
    // Calculate Target Cardiac Phase in Reconstructed Volume for Slice-To-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::CalculateSliceToVolumeTargetCardiacPhase()
    {
        int card_index;
        double angdiff;
        if (_debug)
            cout << "CalculateSliceToVolumeTargetCardiacPhase" << endl;
        _slice_svr_card_index.clear();
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            angdiff = PI + 0.001;  // NOTE: init angdiff larger than any possible calculated angular difference
            card_index = -1;
            for (unsigned int outputIndex = 0; outputIndex < _reconstructed_cardiac_phases.size(); outputIndex++)
            {
                if ( fabs( CalculateAngularDifference( _reconstructed_cardiac_phases[outputIndex], _slice_cardphase[inputIndex] ) ) < angdiff)
                {
                    angdiff = fabs( CalculateAngularDifference( _reconstructed_cardiac_phases[outputIndex], _slice_cardphase[inputIndex] ) );
                    card_index = outputIndex;
                }
            }
            _slice_svr_card_index.push_back(card_index);
            // if (_debug)
            //   cout << inputIndex << ":" << _slice_svr_card_index[inputIndex] << ", ";
        }
        // if (_debug)
        //   cout << "\b\b." << endl;
    }
    
    
    // -----------------------------------------------------------------------------
    // Parallel Slice-to-Volume Registration Class
    // -----------------------------------------------------------------------------
    class ParallelSliceToVolumeRegistrationCardiac4D {
    public:
        ReconstructionCardiac4D *reconstructor;
        
        ParallelSliceToVolumeRegistrationCardiac4D(ReconstructionCardiac4D *_reconstructor) :
        reconstructor(_reconstructor) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            
            ImageAttributes attr = reconstructor->_reconstructed4D.GetImageAttributes();
            
            
            
            
            for ( size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex ) {
                
                if (reconstructor->_slice_excluded[inputIndex] == 0) {
                    
                    //irtkImageRigidRegistrationWithPadding registration;
                    
                    GreyPixel smin, smax;
                    GreyImage target;
                    RealImage slice, w, b, t;
                    
                    
                    ResamplingWithPadding<RealPixel> resampling(attr._dx,attr._dx,attr._dx,-1);
                    Reconstruction dummy_reconstruction;
                    
                    GenericLinearInterpolateImageFunction<RealImage> interpolator;
                    
                    // TARGET
                    // get current slice
                    t = reconstructor->_slices[inputIndex];
                    // resample to spatial resolution of reconstructed volume
                    resampling.Input(&reconstructor->_slices[inputIndex]);
                    resampling.Output(&t);
                    resampling.Interpolator(&interpolator);
                    resampling.Run();
                    target=t;
                    // get pixel value min and max
                    target.GetMinMax(&smin, &smax);
                    
                    // SOURCE
                    if (smax > 0 && (smax-smin) > 1) {
                        
                        ParameterList params;
                        Insert(params, "Transformation model", "Rigid");
                        Insert(params, "Image (dis-)similarity measure", "NMI");
                        if (reconstructor->_nmi_bins>0)
                            Insert(params, "No. of bins", reconstructor->_nmi_bins);
                        Insert(params, "Image interpolation mode", "Linear");
                        // Insert(params, "Background value", -1);
                        Insert(params, "Background value for image 1", 0);
                        Insert(params, "Background value for image 2", -1);
                        
                        //                        Insert(params, "Image (dis-)similarity measure", "NCC");
                        //                        Insert(params, "Image interpolation mode", "Linear");
                        //                        string type = "sigma";
                        //                        string units = "mm";
                        //                        double width = 5;
                        //                        Insert(params, string("Local window size [") + type + string("]"), ToString(width) + units);
                        
                        GenericRegistrationFilter *registration = new GenericRegistrationFilter();
                        registration->Parameter(params);
                        
                        
                        
                        // put origin to zero
                        RigidTransformation offset;
                        dummy_reconstruction.ResetOrigin(target,offset);
                        Matrix mo = offset.GetMatrix();
                        Matrix m = reconstructor->_transformations[inputIndex].GetMatrix();
                        m=m*mo;
                        reconstructor->_transformations[inputIndex].PutMatrix(m);
                        
                        // TODO: extract nearest cardiac phase from reconstructed 4D to use as source
                        GreyImage source = reconstructor->_reconstructed4D.GetRegion( 0, 0, 0, reconstructor->_slice_svr_card_index[inputIndex], attr._x, attr._y, attr._z, reconstructor->_slice_svr_card_index[inputIndex]+1 );
                        
                        
                        target.Write("tt.nii.gz");
                        
                        
                        registration->Input(&target, &source);
                        Transformation *dofout = nullptr;
                        registration->Output(&dofout);
                        
                        RigidTransformation tmp_dofin = reconstructor->_transformations[inputIndex];
                        
                        
                        registration->InitialGuess(&tmp_dofin);
                        registration->GuessParameter();
                        registration->Run();
                        
                        
                        RigidTransformation *rigidTransf = dynamic_cast<RigidTransformation*> (dofout);
                        reconstructor->_transformations[inputIndex] = *rigidTransf;
                        
                        
                        //undo the offset
                        mo.Invert();
                        m = reconstructor->_transformations[inputIndex].GetMatrix();
                        m=m*mo;
                        reconstructor->_transformations[inputIndex].PutMatrix(m);
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
    
    
    // -----------------------------------------------------------------------------
    // Slice-to-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SliceToVolumeRegistrationCardiac4D()
    {
        
        _reconstructed4D.Write("zzz.nii.gz");
        
        if (_debug)
            cout << "SliceToVolumeRegistrationCardiac4D" << endl;
        ParallelSliceToVolumeRegistrationCardiac4D registration(this);
        registration();
    }
    
    
    // -----------------------------------------------------------------------------
    // Volume-to-Volume Registration
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::VolumeToVolumeRegistration(GreyImage target, GreyImage source, RigidTransformation& rigidTransf)
    {
        
        
        ParameterList params;
        Insert(params, "Transformation model", "Rigid");
        Insert(params, "Image (dis-)similarity measure", "NMI");
        if (_nmi_bins>0)
            Insert(params, "No. of bins", _nmi_bins);
        // Insert(params, "Image interpolation mode", "Linear");
        Insert(params, "Background value for image 1", -1);
        Insert(params, "Background value for image 2", -1);
        
        GenericRegistrationFilter registration;
        registration.Parameter(params);
        registration.Input(&target, &source);
        
        
        Transformation *dofout = nullptr;
        registration.Output(&dofout);
        
        RigidTransformation dofin = rigidTransf;
        registration.InitialGuess(&dofin);
        
        registration.GuessParameter();
        registration.Run();
        
        RigidTransformation *rigidTransf_dofout = dynamic_cast<RigidTransformation*> (dofout);
        rigidTransf = *rigidTransf_dofout;
        
        
        /*
         // Initialise registration
         ImageRigidRegistrationWithPadding registration;
         
         // Register source volume to target volume
         registration.SetInput(&target,&source);
         registration.SetOutput(&rigidTransf);
         registration.GuessParameter();
         registration.SetTargetPadding(-1);
         registration.Run();
         */
        
    }
    
    
    // -----------------------------------------------------------------------------
    // Calculate Displacement
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateDisplacement()
    {
        
        if (_debug)
            cout << "CalculateDisplacment" << endl;
        
        _slice_displacement.clear();
        _slice_tx.clear();
        _slice_ty.clear();
        _slice_tz.clear();
        double x,y,z,tx,ty,tz;
        double disp_sum_slice, disp_sum_total = 0;
        double tx_sum_slice, ty_sum_slice, tz_sum_slice = 0;
        int num_voxel_slice, num_voxel_total = 0;
        int k = 0;
        double slice_disp, mean_disp;
        double tx_slice, ty_slice, tz_slice;
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            disp_sum_slice = 0;
            num_voxel_slice = 0;
            slice_disp = -1;
            tx_slice = 0;
            ty_slice = 0;
            tz_slice = 0;
            tx_sum_slice = 0;
            ty_sum_slice = 0;
            tz_sum_slice = 0;
            
            if (_slice_excluded[inputIndex]==0) {
                for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                        if (_slices[inputIndex](i,j,k)!=-1) {
                            x = i;
                            y = j;
                            z = k;
                            _slices[inputIndex].ImageToWorld(x,y,z);
                            tx = x; ty = y; tz = z;
                            _transformations[inputIndex].Transform(tx,ty,tz);
                            disp_sum_slice += sqrt((tx-x)*(tx-x)+(ty-y)*(ty-y)+(tz-z)*(tz-z));
                            tx_sum_slice += tx-x;
                            ty_sum_slice += ty-y;
                            tz_sum_slice += tz-z;
                            num_voxel_slice += 1;
                        }
                    }
                }
                if ( num_voxel_slice>0 ) {
                    slice_disp = disp_sum_slice / num_voxel_slice;
                    tx_slice = tx_sum_slice / num_voxel_slice;
                    ty_slice = ty_sum_slice / num_voxel_slice;
                    tz_slice = tz_sum_slice / num_voxel_slice;
                    disp_sum_total += disp_sum_slice;
                    num_voxel_total += num_voxel_slice;
                }
            }
            
            _slice_displacement.push_back(slice_disp);
            _slice_tx.push_back(tx_slice);
            _slice_ty.push_back(ty_slice);
            _slice_tz.push_back(tz_slice);
            
        }
        
        if (num_voxel_total>0)
            mean_disp = disp_sum_total / num_voxel_total;
        else
            mean_disp = -1;
        
        return mean_disp;
        
    }
    
    
    double ReconstructionCardiac4D::CalculateDisplacement(RigidTransformation drift)
    {
        //Initialise
        double mean_disp;
        Matrix d = drift.GetMatrix();
        Matrix m;
        
        //Remove drift
        for(unsigned int i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m = d*m;
            _transformations[i].PutMatrix(m);
        }
        
        //Calculate displacements
        mean_disp = CalculateDisplacement();
        
        //Return drift
        d.Invert();
        for(unsigned int i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m = d*m;
            _transformations[i].PutMatrix(m);
        }
        
        //Output
        return mean_disp;
    }
    
    
    // -----------------------------------------------------------------------------
    // Calculate Weighted Displacement
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateWeightedDisplacement()
    {
        
        if (_debug)
            cout << "CalculateWeightedDisplacement" << endl;
        
        _slice_weighted_displacement.clear();
        
        double x,y,z,tx,ty,tz;
        double disp_sum_slice, disp_sum_total = 0;
        double weight_slice, weight_total = 0;
        int k = 0;
        double slice_disp, mean_disp = 0;
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            disp_sum_slice = 0;
            weight_slice = 0;
            slice_disp = -1;
            
            if (_slice_excluded[inputIndex]==0) {
                for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                        if (_slices[inputIndex](i,j,k)!=-1) {
                            x = i;
                            y = j;
                            z = k;
                            _slices[inputIndex].ImageToWorld(x,y,z);
                            tx = x; ty = y; tz = z;
                            _transformations[inputIndex].Transform(tx,ty,tz);
                            disp_sum_slice += _slice_weight[inputIndex]*_weights[inputIndex](i,j,k)*sqrt((tx-x)*(tx-x)+(ty-y)*(ty-y)+(tz-z)*(tz-z));
                            weight_slice += _slice_weight[inputIndex]*_weights[inputIndex](i,j,k);
                        }
                    }
                }
                if (weight_slice>0) {
                    slice_disp = disp_sum_slice / weight_slice;
                    disp_sum_total += disp_sum_slice;
                    weight_total += weight_slice;
                }
            }
            
            _slice_weighted_displacement.push_back(slice_disp);
            
        }
        
        if (weight_total>0)
            mean_disp = disp_sum_total / weight_total;
        else
            mean_disp = -1;
        
        return mean_disp;
        
    }
    
    
    double ReconstructionCardiac4D::CalculateWeightedDisplacement(RigidTransformation drift)
    {
        //Initialise
        double mean_disp;
        Matrix d = drift.GetMatrix();
        Matrix m;
        
        //Remove drift
        for(unsigned int i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m = d*m;
            _transformations[i].PutMatrix(m);
        }
        
        //Calculate displacements
        mean_disp = CalculateWeightedDisplacement();
        
        //Return drift
        d.Invert();
        for(unsigned int i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m = d*m;
            _transformations[i].PutMatrix(m);
        }
        
        //Output
        return mean_disp;
    }
    
    
    // -----------------------------------------------------------------------------
    // Calculate Target Registration Error
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::InitTRE()
    {
        _slice_tre.clear();
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            _slice_tre.push_back(-1);
        }
    }
    
    
    double ReconstructionCardiac4D::CalculateTRE()
    {
        
        if (_debug)
            cout << "CalculateTRE" << endl;
        
        _slice_tre.clear();
        
        double x,y,z,cx,cy,cz,px,py,pz;
        double tre_sum_slice, tre_sum_total = 0;
        int num_voxel_slice, num_voxel_total = 0;
        int k = 0;
        double slice_tre, mean_tre;
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            
            tre_sum_slice = 0;
            num_voxel_slice = 0;
            slice_tre = -1;
            
            if (_slice_excluded[inputIndex]==0) {
                for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                    for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                        if (_slices[inputIndex](i,j,k)!=-1) {
                            x = i; y = j; z = k;
                            _slices[inputIndex].ImageToWorld(x,y,z);
                            cx = x; cy = y; cz = z;
                            px = x; py = y; pz = z;
                            _transformations[inputIndex].Transform(cx,cy,cz);
                            _ref_transformations[inputIndex].Transform(px,py,pz);
                            tre_sum_slice += sqrt((cx-px)*(cx-px)+(cy-py)*(cy-py)+(cz-pz)*(cz-pz));
                            num_voxel_slice += 1;
                        }
                    }
                }
                if ( num_voxel_slice>0 ) {
                    slice_tre = tre_sum_slice / num_voxel_slice;
                    tre_sum_total += tre_sum_slice;
                    num_voxel_total += num_voxel_slice;
                }
            }
            
            _slice_tre.push_back(slice_tre);
            
        }
        
        if (num_voxel_total>0)
            mean_tre = tre_sum_total / num_voxel_total;
        else
            mean_tre = -1;
        
        return mean_tre;
        
    }
    
    
    double ReconstructionCardiac4D::CalculateTRE(RigidTransformation drift)
    {
        //Initialise
        double mean_tre;
        Matrix d = drift.GetMatrix();
        Matrix m;
        
        //Remove drift
        for(unsigned int i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m = d*m;
            _transformations[i].PutMatrix(m);
        }
        
        //Calculate displacements
        mean_tre = CalculateTRE();
        
        //Return drift
        d.Invert();
        for(unsigned int i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m = d*m;
            _transformations[i].PutMatrix(m);
        }
        
        //Output
        return mean_tre;
    }
    
    
    // -----------------------------------------------------------------------------
    // Smooth Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SmoothTransformations(double sigma_seconds, int niter, bool use_slice_inside)
    {
        
        if (_debug)
            cout<<"SmoothTransformations"<<endl<<"\tsigma = "<<sigma_seconds<<" s"<<endl;
        
        int i,j,iter,par;
        
        //Reset origin for transformations
        GreyImage t = _reconstructed4D;
        RigidTransformation offset;
        ResetOrigin(t,offset);
        Matrix m;
        Matrix mo = offset.GetMatrix();
        Matrix imo = mo;
        imo.Invert();
        if (_debug)
            offset.Write("reset_origin.dof");
        for(i=0;i<int(_transformations.size());i++)
        {
            m = _transformations[i].GetMatrix();
            m=imo*m*mo;
            _transformations[i].PutMatrix(m);
        }
        
        //initial weights
        Matrix parameters(6,_transformations.size());
        Matrix weights(6,_transformations.size());
        ofstream fileOut("motion.txt", ofstream::out | ofstream::app);
        
        for(i=0;i<int(_transformations.size());i++)
        {
            parameters(0,i)=_transformations[i].GetTranslationX();
            parameters(1,i)=_transformations[i].GetTranslationY();
            parameters(2,i)=_transformations[i].GetTranslationZ();
            parameters(3,i)=_transformations[i].GetRotationX();
            parameters(4,i)=_transformations[i].GetRotationY();
            parameters(5,i)=_transformations[i].GetRotationZ();
            
            //write unprocessed parameters to file
            for(j=0;j<6;j++)
            {
                fileOut<<parameters(j,i);
                if(j<5)
                    fileOut<<",";
                else
                    fileOut<<endl;
            }
            
            for(j=0;j<6;j++)
                weights(j,i)=0;             //initialise as zero
            if (!_slice_excluded[i])
            {
                if (!use_slice_inside)      //set weights based on image intensities
                {
                    RealPixel smin, smax;
                    _slices[i].GetMinMax(&smin,&smax);
                    if(smax>-1)
                        for(j=0;j<6;j++)
                            weights(j,i)=1;
                }
                else                        //set weights based on _slice_inside
                {
                    if(_slice_inside.size()>0)
                    {
                        if(_slice_inside[i])
                            for(j=0;j<6;j++)
                                weights(j,i)=1;
                    }
                }
            }
        }
        
        //initialise
        Matrix den(6,_transformations.size());
        Matrix num(6,_transformations.size());
        Matrix kr(6,_transformations.size());
        Array<double> error,tmp,kernel;
        double median;
        double sigma;
        int nloc = 0;
        for(i=0;i<int(_transformations.size());i++)
            if ((_loc_index[i]+1)>nloc)
                nloc = _loc_index[i] + 1;
        if (_debug)
            cout << "\tnumber of slice-locations = " << nloc << endl;
        int dim = _transformations.size()/nloc; // assuming equal number of dynamic images for every slice-location
        int loc;
        
        //step size for sampling volume in error calculation in kernel regression
        int step;
        double nstep=15;
        step = ceil(_reconstructed4D.GetX()/nstep);
        if(step>ceil(_reconstructed4D.GetY()/nstep))
            step = ceil(_reconstructed4D.GetY()/nstep);
        if(step>ceil(_reconstructed4D.GetZ()/nstep))
            step = ceil(_reconstructed4D.GetZ()/nstep);
        
        //kernel regression
        for(iter = 0; iter<niter;iter++)
        {
            
            for(loc=0;loc<nloc;loc++)
            {
                
                //gaussian kernel for current slice-location
                kernel.clear();
                sigma = ceil( sigma_seconds / _slice_dt[loc*dim] );
                for(j=-3*sigma; j<=3*sigma; j++)
                {
                    kernel.push_back(exp(-(j*_slice_dt[loc*dim]/sigma_seconds)*(j*_slice_dt[loc*dim]/sigma_seconds)));
                }
                
                //kernel-weighted summation
                for(par=0;par<6;par++)
                {
                    for(i=loc*dim;i<(loc+1)*dim;i++)
                    {
                        if(!_slice_excluded[i])
                        {
                            for(j=-3*sigma;j<=3*sigma;j++)
                                if(((i+j)>=loc*dim) && ((i+j)<(loc+1)*dim))
                                {
                                    num(par,i)+=parameters(par,i+j)*kernel[j+3*sigma]*weights(par,i+j);
                                    den(par,i)+=kernel[j+3*sigma]*weights(par,i+j);
                                }
                        }
                        else
                        {
                            num(par,i)=parameters(par,i);
                            den(par,i)=1;
                        }
                    }
                }
            }
            
            //kernel-weighted normalisation
            for(par=0;par<6;par++)
                for(i=0;i<int(_transformations.size());i++)
                    kr(par,i)=num(par,i)/den(par,i);
            
            //recalculate weights using target registration error with original transformations as targets
            error.clear();
            tmp.clear();
            for(i=0;i<int(_transformations.size());i++)
            {
                
                RigidTransformation processed;
                processed.PutTranslationX(kr(0,i));
                processed.PutTranslationY(kr(1,i));
                processed.PutTranslationZ(kr(2,i));
                processed.PutRotationX(kr(3,i));
                processed.PutRotationY(kr(4,i));
                processed.PutRotationZ(kr(5,i));
                
                RigidTransformation orig = _transformations[i];
                
                //need to convert the transformations back to the original coordinate system
                m = orig.GetMatrix();
                m=mo*m*imo;
                orig.PutMatrix(m);
                
                m = processed.GetMatrix();
                m=mo*m*imo;
                processed.PutMatrix(m);
                
                RealImage slice = _slices[i];
                
                int n=0;
                double x,y,z,xx,yy,zz,e;
                
                if(!_slice_excluded[i]) {
                    for(int ii=0;ii<_reconstructed4D.GetX();ii=ii+step)
                        for(int jj=0;jj<_reconstructed4D.GetY();jj=jj+step)
                            for(int kk=0;kk<_reconstructed4D.GetZ();kk=kk+step)
                                if(_reconstructed4D(ii,jj,kk,0)>-1)
                                {
                                    x=ii; y=jj; z=kk;
                                    _reconstructed4D.ImageToWorld(x,y,z);
                                    xx=x;yy=y;zz=z;
                                    orig.Transform(x,y,z);
                                    processed.Transform(xx,yy,zz);
                                    x-=xx;
                                    y-=yy;
                                    z-=zz;
                                    e += sqrt(x*x+y*y+z*z);
                                    n++;
                                }
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
            
            if ((_debug)&(iter==0))
                cout<<"\titeration:median_error(mm)...";
            if (_debug) {
                cout<<iter<<":"<<median<<", ";
                cout.flush();
            }
            
            for(i=0;i<int(_transformations.size());i++)
            {
                if((error[i]>=0)&(!_slice_excluded[i]))
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
            
        }
        
        if (_debug)
            cout<<"\b\b."<<endl;
        
        ofstream fileOut2("motion-processed.txt", ofstream::out | ofstream::app);
        ofstream fileOut3("weights.txt", ofstream::out | ofstream::app);
        ofstream fileOut4("outliers.txt", ofstream::out | ofstream::app);
        ofstream fileOut5("empty.txt", ofstream::out | ofstream::app);
        
        for(i=0;i<int(_transformations.size());i++)
        {
            fileOut3<<weights(j,i)<<" ";
            
            if(weights(0,i)<=0)
                fileOut5<<i<<" ";
            
            _transformations[i].PutTranslationX(kr(0,i));
            _transformations[i].PutTranslationY(kr(1,i));
            _transformations[i].PutTranslationZ(kr(2,i));
            _transformations[i].PutRotationX(kr(3,i));
            _transformations[i].PutRotationY(kr(4,i));
            _transformations[i].PutRotationZ(kr(5,i));
            
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
        for(i=0;i<int(_transformations.size());i++)
        {
            m = _transformations[i].GetMatrix();
            m=mo*m*imo;
            _transformations[i].PutMatrix(m);
        }
        
    }
    
    
    // -----------------------------------------------------------------------------
    // Scale Transformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::ScaleTransformations(double scale)
    {
        
        if (_debug)
            cout<<"Scaling transformations."<<endl<<"scale = "<<scale<<"."<<endl;
        
        if (scale==1)
            return;
        
        if (scale<0) {
            cerr<<"Scaling of transformations undefined for scale < 0.";
            exit(1);
        }
        
        unsigned int i;
        
        //Reset origin for transformations
        GreyImage t = _reconstructed4D;
        RigidTransformation offset;
        ResetOrigin(t,offset);
        Matrix m;
        Matrix mo = offset.GetMatrix();
        Matrix imo = mo;
        imo.Invert();
        if (_debug)
            offset.Write("reset_origin.dof");
        for(i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m = imo * m * mo;
            _transformations[i].PutMatrix(m);
        }
        
        //Scale transformations
        Matrix orig, scaled;
        int row, col;
        for(i=0;i<_transformations.size();i++)
        {
            orig = logm( _transformations[i].GetMatrix() );
            scaled = orig;
            for(row=0;row<4;row++)
                for(col=0;col<4;col++)
                    scaled(row,col) = scale * orig(row,col);
            _transformations[i].PutMatrix( expm( scaled ) );
        }
        
        //Put origin back
        for(i=0;i<_transformations.size();i++)
        {
            m = _transformations[i].GetMatrix();
            m=mo*m*imo;
            _transformations[i].PutMatrix(m);
        }
        
    }
    
    
    // -----------------------------------------------------------------------------
    // Mask Reconstructed Volume
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::StaticMaskReconstructedVolume4D()
    {
        for ( int i = 0; i < _mask.GetX(); i++) {
            for ( int j = 0; j < _mask.GetY(); j++) {
                for ( int k = 0; k < _mask.GetZ(); k++) {
                    if ( _mask(i,j,k) == 0 ) {
                        for ( int t = 0; t < _reconstructed4D.GetT(); t++) {
                            _reconstructed4D(i,j,k,t) = -1;
                        }
                    }
                }
            }
        }
    }
    
    
    // -----------------------------------------------------------------------------
    // Apply Static Mask to 4D Volume
    // -----------------------------------------------------------------------------
    RealImage ReconstructionCardiac4D::StaticMaskVolume4D(RealImage volume, double padding)
    {
        for ( int i = 0; i < volume.GetX(); i++) {
            for ( int j = 0; j < volume.GetY(); j++) {
                for ( int k = 0; k < volume.GetZ(); k++) {
                    if ( _mask(i,j,k) == 0 ) {
                        for ( int t = 0; t < volume.GetT(); t++) {
                            volume(i,j,k,t) = padding;
                        }
                    }
                }
            }
        }
        return volume;
    }
    
    
    // -----------------------------------------------------------------------------
    // Parallel Super-Resolution Class
    // -----------------------------------------------------------------------------
    class ParallelSuperresolutionCardiac4D {
        ReconstructionCardiac4D* reconstructor;
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
                                for (int outputIndex=0; outputIndex<reconstructor->_reconstructed4D.GetT(); outputIndex++) {
                                    if(reconstructor->_robust_slices_only)
                                    {
                                        addon(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        confidence_map(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * reconstructor->_slice_weight[inputIndex];
                                        
                                    }
                                    else
                                    {
                                        addon(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * slice(i, j, 0) * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                        confidence_map(p.x, p.y, p.z, outputIndex) += reconstructor->_slice_temporal_weight[outputIndex][inputIndex] * p.value * w(i, j, 0) * reconstructor->_slice_weight[inputIndex];
                                    }
                                }
                            }
                        }
            } //end of loop for a slice inputIndex
        }
        
        ParallelSuperresolutionCardiac4D( ParallelSuperresolutionCardiac4D& x, split ) :
        reconstructor(x.reconstructor)
        {
            //Clear addon
            addon.Initialize( reconstructor->_reconstructed4D.GetImageAttributes() );
            addon = 0;
            
            //Clear confidence map
            confidence_map.Initialize( reconstructor->_reconstructed4D.GetImageAttributes() );
            confidence_map = 0;
        }
        
        void join( const ParallelSuperresolutionCardiac4D& y ) {
            addon += y.addon;
            confidence_map += y.confidence_map;
        }
        
        ParallelSuperresolutionCardiac4D( ReconstructionCardiac4D *reconstructor ) :
        reconstructor(reconstructor)
        {
            //Clear addon
            addon.Initialize( reconstructor->_reconstructed4D.GetImageAttributes() );
            addon = 0;
            
            //Clear confidence map
            confidence_map.Initialize( reconstructor->_reconstructed4D.GetImageAttributes() );
            confidence_map = 0;
        }
        
        // execute
        void operator() () {
            //task_scheduler_init init(tbb_no_threads);
            parallel_reduce( blocked_range<size_t>(0,reconstructor->_slices.size()),
                            *this );
            //init.terminate();
        }
    };
    
    
    // -----------------------------------------------------------------------------
    // Super-Resolution of 4D Volume
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SuperresolutionCardiac4D( int iter )
    {
        if (_debug)
            cout << "Superresolution " << iter << endl;
        
        int i, j, k, t;
        RealImage addon, original;
        
        
        //Remember current reconstruction for edge-preserving smoothing
        original = _reconstructed4D;
        
        ParallelSuperresolutionCardiac4D parallelSuperresolution(this);
        parallelSuperresolution();
        
        addon = parallelSuperresolution.addon;
        _confidence_map = parallelSuperresolution.confidence_map;
        
        if(_debug) {
            // char buffer[256];
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
                        for (t = 0; t < addon.GetT(); t++)
                            if (_confidence_map(i, j, k, t) > 0) {
                                // ISSUES if _confidence_map(i, j, k, t) is too small leading
                                // to bright pixels
                                addon(i, j, k, t) /= _confidence_map(i, j, k, t);
                                //this is to revert to normal (non-adaptive) regularisation
                                _confidence_map(i,j,k,t) = 1;
                            }
        
        _reconstructed4D += addon * _alpha; //_average_volume_weight;
        
        //bound the intensities
        for (i = 0; i < _reconstructed4D.GetX(); i++)
            for (j = 0; j < _reconstructed4D.GetY(); j++)
                for (k = 0; k < _reconstructed4D.GetZ(); k++)
                    for (t = 0; t < _reconstructed4D.GetT(); t++)
                    {
                        if (_reconstructed4D(i, j, k, t) < _min_intensity * 0.9)
                            _reconstructed4D(i, j, k, t) = _min_intensity * 0.9;
                        if (_reconstructed4D(i, j, k, t) > _max_intensity * 1.1)
                            _reconstructed4D(i, j, k, t) = _max_intensity * 1.1;
                    }
        
        //Smooth the reconstructed image
        AdaptiveRegularizationCardiac4D(iter, original);
        
        //Remove the bias in the reconstructed volume compared to previous iteration
        /* TODO: update adaptive regularisation for 4d
         if (_global_bias_correction)
         BiasCorrectVolume(original);
         */
    }
    
    
    // -----------------------------------------------------------------------------
    // Parallel Adaptive Regularization Class 1: calculate smoothing factor, b
    // -----------------------------------------------------------------------------
    class ParallelAdaptiveRegularization1Cardiac4D {
        ReconstructionCardiac4D *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;
        
    public:
        ParallelAdaptiveRegularization1Cardiac4D( ReconstructionCardiac4D *_reconstructor,
                                                 Array<RealImage> &_b,
                                                 Array<double> &_factor,
                                                 RealImage &_original) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            int dx = reconstructor->_reconstructed4D.GetX();
            int dy = reconstructor->_reconstructed4D.GetY();
            int dz = reconstructor->_reconstructed4D.GetZ();
            int dt = reconstructor->_reconstructed4D.GetT();
            for ( size_t i = r.begin(); i != r.end(); ++i ) {
                //b[i] = reconstructor->_reconstructed;
                // b[i].Initialize( reconstructor->_reconstructed.GetImageAttributes() );
                
                int x, y, z, xx, yy, zz, t;
                double diff;
                for (x = 0; x < dx; x++)
                    for (y = 0; y < dy; y++)
                        for (z = 0; z < dz; z++) {
                            xx = x + reconstructor->_directions[i][0];
                            yy = y + reconstructor->_directions[i][1];
                            zz = z + reconstructor->_directions[i][2];
                            for (t = 0; t < dt; t++) {
                                if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz)
                                    && (reconstructor->_confidence_map(x, y, z, t) > 0) && (reconstructor->_confidence_map(xx, yy, zz, t) > 0)) {
                                    diff = (original(xx, yy, zz, t) - original(x, y, z, t)) * sqrt(factor[i]) / reconstructor->_delta;
                                    b[i](x, y, z, t) = factor[i] / sqrt(1 + diff * diff);
                                    
                                }
                                else
                                    b[i](x, y, z, t) = 0;
                            }
                        }
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, 13),
                         *this );
            //init.terminate();
        }
        
    };
    
    
    // -----------------------------------------------------------------------------
    // Parallel Adaptive Regularization Class 2: compute regularisation update
    // -----------------------------------------------------------------------------
    class ParallelAdaptiveRegularization2Cardiac4D {
        ReconstructionCardiac4D *reconstructor;
        Array<RealImage> &b;
        Array<double> &factor;
        RealImage &original;
        
    public:
        ParallelAdaptiveRegularization2Cardiac4D( ReconstructionCardiac4D *_reconstructor,
                                                 Array<RealImage> &_b,
                                                 Array<double> &_factor,
                                                 RealImage &_original) :
        reconstructor(_reconstructor),
        b(_b),
        factor(_factor),
        original(_original) { }
        
        void operator() (const blocked_range<size_t> &r) const {
            int dx = reconstructor->_reconstructed4D.GetX();
            int dy = reconstructor->_reconstructed4D.GetY();
            int dz = reconstructor->_reconstructed4D.GetZ();
            int dt = reconstructor->_reconstructed4D.GetT();
            for ( size_t x = r.begin(); x != r.end(); ++x ) {
                int xx, yy, zz;
                for (int y = 0; y < dy; y++)
                    for (int z = 0; z < dz; z++)
                        for (int t = 0; t < dt; t++) {
                            if(reconstructor->_confidence_map(x,y,z,t)>0)
                            {
                                double val = 0;
                                double sum = 0;
                                for (int i = 0; i < 13; i++)
                                {
                                    xx = x + reconstructor->_directions[i][0];
                                    yy = y + reconstructor->_directions[i][1];
                                    zz = z + reconstructor->_directions[i][2];
                                    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                                        if(reconstructor->_confidence_map(xx,yy,zz,t)>0)
                                        {
                                            val += b[i](x, y, z, t) * original(xx, yy, zz, t);
                                            sum += b[i](x, y, z, t);
                                        }
                                }
                                
                                for (int i = 0; i < 13; i++) {
                                    xx = x - reconstructor->_directions[i][0];
                                    yy = y - reconstructor->_directions[i][1];
                                    zz = z - reconstructor->_directions[i][2];
                                    if ((xx >= 0) && (xx < dx) && (yy >= 0) && (yy < dy) && (zz >= 0) && (zz < dz))
                                        if(reconstructor->_confidence_map(xx,yy,zz,t)>0)
                                        {
                                            val += b[i](x, y, z, t) * original(xx, yy, zz, t);
                                            sum += b[i](x, y, z, t);
                                        }
                                }
                                
                                val -= sum * original(x, y, z, t);
                                val = original(x, y, z, t)
                                + reconstructor->_alpha * reconstructor->_lambda / (reconstructor->_delta * reconstructor->_delta) * val;
                                reconstructor->_reconstructed4D(x, y, z, t) = val;
                            }
                        }
            }
        }
        
        // execute
        void operator() () const {
            //task_scheduler_init init(tbb_no_threads);
            parallel_for( blocked_range<size_t>(0, reconstructor->_reconstructed4D.GetX()),
                         *this );
            //init.terminate();
        }
        
    };
    
    
    // -----------------------------------------------------------------------------
    // Adaptive Regularization
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::AdaptiveRegularizationCardiac4D(int iter, RealImage& original)
    {
        if (_debug)
            cout << "AdaptiveRegularizationCardiac4D."<< endl;
        //cout << "AdaptiveRegularizationCardiac4D: _delta = "<<_delta<<" _lambda = "<<_lambda <<" _alpha = "<<_alpha<< endl;
        
        Array<double> factor(13,0);
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 3; j++)
                factor[i] += fabs(double(_directions[i][j]));
            factor[i] = 1 / factor[i];
        }
        
        Array<RealImage> b;//(13);
        for (int i = 0; i < 13; i++)
            b.push_back( _reconstructed4D );
        
        ParallelAdaptiveRegularization1Cardiac4D parallelAdaptiveRegularization1( this,
                                                                                 b,
                                                                                 factor,
                                                                                 original );
        parallelAdaptiveRegularization1();
        
        RealImage original2 = _reconstructed4D;
        ParallelAdaptiveRegularization2Cardiac4D parallelAdaptiveRegularization2( this,
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
    
    
    // -----------------------------------------------------------------------------
    // ReadTransformation
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::ReadTransformation(char* folder)
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
        cout << "Reading transformations from: " << folder << endl;
        
        _transformations.clear();
        for (int i = 0; i < n; i++) {
            if (folder != NULL) {
                sprintf(name, "/transformation%05i.dof", i);
                strcpy(path, folder);
                strcat(path, name);
            }
            else {
                sprintf(path, "transformation%05i.dof", i);
            }
            transformation = Transformation::New(path);
            rigidTransf = dynamic_cast<RigidTransformation*>(transformation);
            _transformations.push_back(*rigidTransf);
            delete transformation;
        }
    }
    
    
    // -----------------------------------------------------------------------------
    // ReadRefTransformation
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::ReadRefTransformation(char* folder)
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
        cout << "Reading reference transformations from: " << folder << endl;
        
        _ref_transformations.clear();
        for (int i = 0; i < n; i++) {
            if (folder != NULL) {
                sprintf(name, "/transformation%05i.dof", i);
                strcpy(path, folder);
                strcat(path, name);
            }
            else {
                sprintf(path, "transformation%05i.dof", i);
            }
            transformation = Transformation::New(path);
            rigidTransf = dynamic_cast<RigidTransformation*>(transformation);
            _ref_transformations.push_back(*rigidTransf);
            delete transformation;
        }
    }
    
    
    // -----------------------------------------------------------------------------
    // Calculate Entropy
    // -----------------------------------------------------------------------------
    double ReconstructionCardiac4D::CalculateEntropy()
    {
        
        double x;
        double sum_x_sq = 0;
        double x_max = 0;
        double entropy = 0;
        
        for ( int i = 0; i < _reconstructed4D.GetX(); i++)
            for ( int j = 0; j < _reconstructed4D.GetY(); j++)
                for ( int k = 0; k < _reconstructed4D.GetZ(); k++)
                    if ( _mask(i,j,k) == 1 )
                        for ( int f = 0; f < _reconstructed4D.GetT(); f++)
                        {
                            x = _reconstructed4D(i,j,k,f);
                            sum_x_sq += x*x;
                        }
        
        x_max = sqrt( sum_x_sq );
        
        for ( int i = 0; i < _reconstructed4D.GetX(); i++)
            for ( int j = 0; j < _reconstructed4D.GetY(); j++)
                for ( int k = 0; k < _reconstructed4D.GetZ(); k++)
                    if ( _mask(i,j,k) == 1 )
                        for ( int f = 0; f < _reconstructed4D.GetT(); f++)
                        {
                            x = _reconstructed4D(i,j,k,f);
                            if (x>0)
                                entropy += x/x_max * log( x/x_max );
                        }
        
        entropy = -entropy;
        
        return entropy;
        
    }
    
    
    // -----------------------------------------------------------------------------
    // SaveTransformations
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SaveTransformations()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "transformation%05i.dof", inputIndex);
            _transformations[inputIndex].Write(buffer);
        }
    }
    
    // -----------------------------------------------------------------------------
    // SaveBiasFields
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SaveBiasFields()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "bias%05i.nii.gz", inputIndex);
            _bias[inputIndex].Write(buffer);
        }
    }
    
    void ReconstructionCardiac4D::SaveBiasFields( Array<RealImage> &stacks )
    {
        
        if (_debug)
            cout << "SaveBiasFields as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> biasstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            biasstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                    biasstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _bias[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "bias%03i.nii.gz", i);
            biasstacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    void ReconstructionCardiac4D::SaveBiasFields( Array<RealImage> &stacks, int iter, int rec_iter )
    {
        
        if (_debug)
            cout << "SaveBiasFields as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> biasstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            biasstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                    biasstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _bias[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "bias%03i_mc%02isr%02i.nii.gz", i, iter, rec_iter);
            biasstacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    
    // -----------------------------------------------------------------------------
    // Save Corrected Slices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SaveCorrectedSlices()
    {
        if (_debug)
            cout<<"Saving corrected images ...";
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _corrected_slices.size(); inputIndex++)
        {
            sprintf(buffer, "correctedimage%05i.nii.gz", inputIndex);
            _corrected_slices[inputIndex].Write(buffer);
        }
        cout<<"done."<<endl;
    }
    
    void ReconstructionCardiac4D::SaveCorrectedSlices( Array<RealImage> &stacks )
    {
        
        if (_debug)
            cout << "Saving corrected images as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> imagestacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            imagestacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _corrected_slices.size(); ++inputIndex) {
            for (int i = 0; i < _corrected_slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _corrected_slices[inputIndex].GetY(); j++) {
                    imagestacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _corrected_slices[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "correctedstack%03i.nii.gz", i);
            imagestacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    void ReconstructionCardiac4D::SaveCorrectedSlices( Array<RealImage> &stacks, int iter, int rec_iter )
    {
        
        if (_debug)
            cout << "Saving error images as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> imagestacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            imagestacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _corrected_slices.size(); ++inputIndex) {
            for (int i = 0; i < _corrected_slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _corrected_slices[inputIndex].GetY(); j++) {
                    imagestacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _corrected_slices[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "correctedstack%03i_mc%02isr%02i.nii.gz", i, iter, rec_iter);
            imagestacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    
    
    // -----------------------------------------------------------------------------
    // SaveSimulatedSlices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SaveSimulatedSlices()
    {
        if (_debug)
            cout<<"Saving simulated images ...";
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            sprintf(buffer, "simimage%05i.nii.gz", inputIndex);
            _simulated_slices[inputIndex].Write(buffer);
        }
        cout<<"done."<<endl;
    }
    
    void ReconstructionCardiac4D::SaveSimulatedSlices( Array<RealImage> &stacks )
    {
        
        if (_debug)
            cout << "Saving simulated images as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> simstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            simstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _simulated_slices.size(); ++inputIndex) {
            for (int i = 0; i < _simulated_slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _simulated_slices[inputIndex].GetY(); j++) {
                    simstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _simulated_slices[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "simstack%03i.nii.gz", i);
            simstacks[i].Write(buffer);
        }
        
        
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    void ReconstructionCardiac4D::SaveSimulatedSlices( Array<RealImage> &stacks, int iter, int rec_iter )
    {
        
        if (_debug)
            cout << "Saving simulated images as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> simstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            simstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _simulated_slices.size(); ++inputIndex) {
            for (int i = 0; i < _simulated_slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _simulated_slices[inputIndex].GetY(); j++) {
                    simstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _simulated_slices[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "simstack%03i_mc%02isr%02i.nii.gz", i, iter, rec_iter);
            simstacks[i].Write(buffer);
        }
        
        for (int ii=0; ii<stacks.size(); ii++) {
            sprintf(buffer,"dif-%i-%i.nii.gz",ii,iter);
            simstacks[ii] = stacks[ii] - simstacks[ii];
            simstacks[ii].Write(buffer);
        }
        
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    void ReconstructionCardiac4D::SaveSimulatedSlices(Array<RealImage>& stacks, int stack_no)
    {
        
        char buffer[256];
        RealImage simstack;
        
        ImageAttributes attr = stacks[stack_no].GetImageAttributes();
        simstack.Initialize( attr );
        
        for (unsigned int inputIndex = 0; inputIndex < _simulated_slices.size(); ++inputIndex) {
            for (int i = 0; i < _simulated_slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _simulated_slices[inputIndex].GetY(); j++) {
                    if (stack_no==_stack_index[inputIndex]) {
                        simstack(i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _simulated_slices[inputIndex](i,j,0);
                    }
                }
            }
        }
        
        sprintf(buffer, "simstack%03i.nii.gz", stack_no);
        simstack.Write(buffer);
        
    }
    
    
    // -----------------------------------------------------------------------------
    // SaveSimulatedWeights
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SaveSimulatedWeights()
    {
        if (_debug)
            cout<<"Saving simulated weights ...";
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _simulated_weights.size(); inputIndex++)
        {
            sprintf(buffer, "simweight%05i.nii.gz", inputIndex);
            _simulated_weights[inputIndex].Write(buffer);
        }
        cout<<"done."<<endl;
    }
    
    void ReconstructionCardiac4D::SaveSimulatedWeights( Array<RealImage> &stacks )
    {
        
        if (_debug)
            cout << "Saving simulated weights as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> simstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            simstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _simulated_weights.size(); ++inputIndex) {
            for (int i = 0; i < _simulated_weights[inputIndex].GetX(); i++) {
                for (int j = 0; j < _simulated_weights[inputIndex].GetY(); j++) {
                    simstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _simulated_weights[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < simstacks.size(); i++) {
            sprintf(buffer, "simweightstack%03i.nii.gz", i);
            simstacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    void ReconstructionCardiac4D::SaveSimulatedWeights( Array<RealImage> &stacks, int iter, int rec_iter )
    {
        
        if (_debug)
            cout << "Saving simulated weights as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> simstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            simstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _simulated_weights.size(); ++inputIndex) {
            for (int i = 0; i < _simulated_weights[inputIndex].GetX(); i++) {
                for (int j = 0; j < _simulated_weights[inputIndex].GetY(); j++) {
                    simstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _simulated_weights[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < simstacks.size(); i++) {
            sprintf(buffer, "simweightstack%03i_mc%02isr%02i.nii.gz", i, iter, rec_iter);
            simstacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    
    // -----------------------------------------------------------------------------
    // SaveSlices
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SaveSlices()
    {
        
        if (_debug)
            cout << "SaveSlices" << endl;
        
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++)
        {
            sprintf(buffer, "image%05i.nii.gz", inputIndex);
            _slices[inputIndex].Write(buffer);
        }
    }
    
    void ReconstructionCardiac4D::SaveSlices( Array<RealImage> &stacks )
    {
        
        if (_debug)
            cout << "SaveSlices as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> imagestacks;
        Array<RealImage> wstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            imagestacks.push_back( stack );
            wstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                    imagestacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _slices[inputIndex](i,j,0);
                    wstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = 10*_weights[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "stack%03i.nii.gz", i);
            imagestacks[i].Write(buffer);
            
            sprintf(buffer, "w%03i.nii.gz", i);
            wstacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    
    // -----------------------------------------------------------------------------
    // Save Error
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SaveError()
    {
        if (_debug)
            cout<<"Saving error images ...";
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _error.size(); inputIndex++)
        {
            sprintf(buffer, "errorimage%05i.nii.gz", inputIndex);
            _error[inputIndex].Write(buffer);
        }
        cout<<"done."<<endl;
    }
    
    void ReconstructionCardiac4D::SaveError( Array<RealImage> &stacks )
    {
        
        if (_debug)
            cout << "Saving error images as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> errorstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            errorstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _error.size(); ++inputIndex) {
            for (int i = 0; i < _error[inputIndex].GetX(); i++) {
                for (int j = 0; j < _error[inputIndex].GetY(); j++) {
                    errorstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _error[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "errorstack%03i.nii.gz", i);
            errorstacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    void ReconstructionCardiac4D::SaveError( Array<RealImage> &stacks, int iter, int rec_iter )
    {
        
        if (_debug)
            cout << "Saving error images as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> errorstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            errorstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _error.size(); ++inputIndex) {
            for (int i = 0; i < _error[inputIndex].GetX(); i++) {
                for (int j = 0; j < _error[inputIndex].GetY(); j++) {
                    errorstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _error[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "errorstack%03i_mc%02isr%02i.nii.gz", i, iter, rec_iter);
            errorstacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    
    // -----------------------------------------------------------------------------
    // SaveWeights
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SaveWeights()
    {
        char buffer[256];
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); inputIndex++) {
            sprintf(buffer, "weights%05i.nii.gz", inputIndex);
            _weights[inputIndex].Write(buffer);
        }
    }
    
    void ReconstructionCardiac4D::SaveWeights( Array<RealImage> &stacks )
    {
        
        if (_debug)
            cout << "SaveWeights as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> weightstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            weightstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                    weightstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _weights[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "weight%03i.nii.gz", i);
            weightstacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    void ReconstructionCardiac4D::SaveWeights( Array<RealImage> &stacks, int iter, int rec_iter )
    {
        
        if (_debug)
            cout << "SaveWeights as stacks ...";
        
        char buffer[256];
        RealImage stack;
        Array<RealImage> weightstacks;
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            ImageAttributes attr = stacks[i].GetImageAttributes();
            stack.Initialize( attr );
            weightstacks.push_back( stack );
        }
        
        for (unsigned int inputIndex = 0; inputIndex < _slices.size(); ++inputIndex) {
            for (int i = 0; i < _slices[inputIndex].GetX(); i++) {
                for (int j = 0; j < _slices[inputIndex].GetY(); j++) {
                    weightstacks[_stack_index[inputIndex]](i,j,_stack_loc_index[inputIndex],_stack_dyn_index[inputIndex]) = _weights[inputIndex](i,j,0);
                }
            }
        }
        
        for (unsigned int i = 0; i < stacks.size(); i++) {
            sprintf(buffer, "weights%03i_mc%02isr%02i.nii.gz", i, iter, rec_iter);
            weightstacks[i].Write(buffer);
        }
        
        if (_debug)
            cout << " done." << endl;
        
    }
    
    
    
    
    // -----------------------------------------------------------------------------
    // SlicesInfo
    // -----------------------------------------------------------------------------
    void ReconstructionCardiac4D::SlicesInfoCardiac4D( const char* filename,
                                                      Array<string> &stack_files )
    {
        ofstream info;
        info.open( filename );
        
        info<<setprecision(3);
        
        // header
        info << "StackIndex" << "\t"
        << "StackLocIndex" << "\t"
        << "StackDynIndex" << "\t"
        << "LocIndex" << "\t"
        << "InputIndex" << "\t"
        << "File" << "\t"
        << "Scale" << "\t"
        << "StackFactor" << "\t"
        << "Time" << "\t"
        << "TemporalResolution" << "\t"
        << "CardiacPhase" << "\t"
        << "ReconCardPhaseIndex" << "\t"
        << "Included" << "\t" // Included slices
        << "Excluded" << "\t"  // Excluded slices
        << "Outside" << "\t"  // Outside slices
        << "Weight" << "\t"
        << "MeanDisplacement" << "\t"
        << "MeanDisplacementX" << "\t"
        << "MeanDisplacementY" << "\t"
        << "MeanDisplacementZ" << "\t"
        << "WeightedMeanDisplacement" << "\t"
        << "TRE" << "\t"
        << "TranslationX" << "\t"
        << "TranslationY" << "\t"
        << "TranslationZ" << "\t"
        << "RotationX" << "\t"
        << "RotationY" << "\t"
        << "RotationZ" << "\t";
        info << "\b" << endl;
        
        for (unsigned int i = 0; i < _slices.size(); i++) {
            RigidTransformation& t = _transformations[i];
            info << _stack_index[i] << "\t"
            << _stack_loc_index[i] << "\t"
            << _stack_dyn_index[i] << "\t"
            << _loc_index[i] << "\t"
            << i << "\t"
            << stack_files[_stack_index[i]] << "\t"
            << _scale[i] << "\t"
            << _stack_factor[_stack_index[i]] << "\t"
            << _slice_time[i] << "\t"
            << _slice_dt[i] << "\t"
            << _slice_cardphase[i] << "\t"
            << _slice_svr_card_index[i] << "\t"
            << (((_slice_weight[i] >= 0.5) && (_slice_inside[i]))?1:0) << "\t" // Included slices
            << (((_slice_weight[i] < 0.5) && (_slice_inside[i]))?1:0) << "\t"  // Excluded slices
            << ((!(_slice_inside[i]))?1:0) << "\t"  // Outside slices
            << _slice_weight[i] << "\t"
            << _slice_displacement[i] << "\t"
            << _slice_tx[i] << "\t"
            << _slice_ty[i] << "\t"
            << _slice_tz[i] << "\t"
            << _slice_weighted_displacement[i] << "\t"
            << _slice_tre[i] << "\t"
            << t.GetTranslationX() << "\t"
            << t.GetTranslationY() << "\t"
            << t.GetTranslationZ() << "\t"
            << t.GetRotationX() << "\t"
            << t.GetRotationY() << "\t"
            << t.GetRotationZ() << "\t";
            info << "\b" << endl;
        }
        
        info.close();
    }
    
    
    
    
    // -----------------------------------------------------------------------------
    
    
    
} // namespace mirtk



