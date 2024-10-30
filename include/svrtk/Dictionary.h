/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2008-2017 Imperial College London
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

#pragma once

// SVRTK
#include "svrtk/Common.h"
#include <Eigen/Dense>
#include<iostream>

using namespace std;
using namespace mirtk;
using namespace svrtk::Utility;

namespace svrtk {

    // Forward declarations
    namespace Parallel {
        class GlobalSimilarityStats;
        class QualityReport;
        class StackRegistrations;
        class SliceToVolumeRegistration;
        class SliceToVolumeRegistrationFFD;
        class RemoteSliceToVolumeRegistration;
        class CoeffInit;
        class CoeffInitSF;
        class Superresolution;
        class SStep;
        class MStep;
        class EStep;
        class Bias;
        class Scale;
        class NormaliseBias;
        class SimulateSlices;
        class SimulateMasks;
        class Average;
        class AdaptiveRegularization1;
        class AdaptiveRegularization2;
    }
    
    namespace ParallelqMRI {
    	class DictionaryFit;
    }

    /**
     * @brief Dictionary class used dictionary.
     */
    class Dictionary {
    protected:
    	// Eigen matrix of Dictionary values
    	Eigen::MatrixXd _DictionaryMatrix;
    	
    	// T2 Values
    	Eigen::VectorXd _T2Vals;
    	
    	// Array of images and image names
        Array<RealImage> _images;

        vector<string> _imageFiles;
    	
    	// Dictionary File name
    	const char * _DictFile;
    	
    public:

        // Output Map
        RealImage _OutputMap;

        // Dictionary constructor
        Dictionary();
        // Dictionary destructor
        ~Dictionary() {}
        
        // Copy constructor
        Dictionary(const Dictionary &D1){
        	_DictionaryMatrix = _DictionaryMatrix;
        	
        	_OutputMap = D1._OutputMap;
        	
        	_T2Vals = D1._T2Vals;
        	
        	_images = D1._images;
        	
        	_imageFiles = D1._imageFiles;
        	
        	_DictFile = D1._DictFile;
        }
        
        Dictionary(const char *filename){
            ifstream file;
            file.open(filename);
            if (!file)
                throw std::runtime_error("Could not find Dictionary text file");
        	_DictFile = filename;
        	readMatrix();
        }
        
        // ReadMatrix
        void readMatrix();

        void readMatrix(const char * );
        
        // Set Dictionaryfile name
        void SetFilename(const char *filename){
            ifstream file;
            file.open(filename);
            if (!file)
                throw std::runtime_error("Could not find Dictionary text file");
        	_DictFile = filename;
        }
        
        
        // Set T2 Values
        void SetT2Vals(Eigen::VectorXd T2Vals){
        	_T2Vals = T2Vals;
        }
        
        // Set T2 Values
        void SetT2Vals(int min, int max){
        	_T2Vals = Eigen::VectorXd::LinSpaced(max-min + 1,min,max);
        }

        // Set Images
        void SetImages(Array<RealImage> images){
            _images = images;
        }
        
        // Set T2 Values
        void SetT2Vals(double min, double max, double step){
        	double intrm = (max-min)/step;
        	intrm = intrm + 0.5 - (intrm<0);
        	int _sizeOfT2Vect = (int)intrm;
        	_T2Vals = Eigen::VectorXd::LinSpaced(_sizeOfT2Vect,min,max);
        }

        // Get T2 Values
        Eigen::VectorXd GetT2Vals(){
            return _T2Vals;
        }

        // Get T2 Values
        double GetT2Vals(int ind){
            return _T2Vals(ind);
        }
        
        // Get Dictionary Matrix
        Eigen::MatrixXd GetDictionaryMatrix(){
        	return _DictionaryMatrix;
        }
        
        // Unmasked Dictionary fitting:
        RealImage FitDictionaryToMap(const Array<RealImage> images);

        // Unmasked Dictionary fitting:
        RealImage FitDictionaryToMap();
        
        //Masked Dictionary fitting        
        RealImage FitDictionaryToMap(const Array<RealImage> images, RealImage mask);

        //Masked Dictionary fitting
        RealImage FitDictionaryToMap(RealImage mask);

        void PrintT2s(){
        	std::string sep = "\n----------------------------------------\n";
        	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        	std::cout << _T2Vals.format(CleanFmt) << sep;
        	}

        friend class ParallelqMRI::DictionaryFit;
        
        
   };  // end of Dictionary class definition

} // namespace svrtk
