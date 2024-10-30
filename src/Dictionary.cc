/*
 * SVRTK : SVR Dictionary based on MIRTK
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

// SVRTK
#include "svrtk/Dictionary.h"
#include "svrtk/Profiling.h"
#include "svrtk/Parallel.h"
#include "svrtk/ParallelqMRI.h"
#include <Eigen/Dense>
#define MAXBUFSIZE  ((int) 1e6)

namespace svrtk {

    Dictionary::Dictionary() {}
    
    
    
    void Dictionary::readMatrix()
    {
	    int cols = 0, rows = 0;
	    double buff[MAXBUFSIZE];

	    // Read numbers from file into buffer.
	    ifstream infile;
	    infile.open(_DictFile);
	    while (! infile.eof())
		{
		string line;
		getline(infile, line);

		int temp_cols = 0;
		stringstream stream(line);
		while(! stream.eof())
		    stream >> buff[cols*rows+temp_cols++];

		if (temp_cols == 0)
		    continue;

		if (cols == 0)
		    cols = temp_cols;

		rows++;
		}

	    infile.close();

	    rows--;

	    // Populate matrix with numbers.
	    Eigen::MatrixXd result(rows,cols);
	    for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		    result(i,j) = buff[ cols*i+j ];

	    _DictionaryMatrix = result;
    };

    void Dictionary::readMatrix(const char * DictFile)
    {
        ifstream file;
        file.open(DictFile);
        if (!file)
            throw std::runtime_error("Could not find Dictionary text file");

        int cols = 0, rows = 0;
        double buff[MAXBUFSIZE];

        // Read numbers from file into buffer.
        ifstream infile;
        infile.open(DictFile);
        while (! infile.eof())
        {
            string line;
            getline(infile, line);

            int temp_cols = 0;
            stringstream stream(line);
            while(! stream.eof())
                stream >> buff[cols*rows+temp_cols++];

            if (temp_cols == 0)
                continue;

            if (cols == 0)
                cols = temp_cols;

            rows++;
        }

        infile.close();

        rows--;

        // Populate matrix with numbers.
        Eigen::MatrixXd result(rows,cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result(i,j) = buff[ cols*i+j ];

        _DictionaryMatrix = result;
    };
    
    
    
    RealImage Dictionary::FitDictionaryToMap(const Array<RealImage> images){

        // Intialise Output map
        _OutputMap.Initialize(images[0].Attributes());
        _images = images;
        ParallelqMRI::DictionaryFit DictFit(this);
        DictFit();
        _OutputMap = DictFit.T2Map;
	    return _OutputMap;
    }

    RealImage Dictionary::FitDictionaryToMap(){
        // Intialise Output map
        _OutputMap.Initialize(_images[0].Attributes());
        ParallelqMRI::DictionaryFit DictFit(this);
        DictFit();
        RealImage& OutputMap = DictFit.T2Map;
        _OutputMap = OutputMap;
        return OutputMap;
    }
    
    RealImage Dictionary::FitDictionaryToMap(const Array<RealImage> images, RealImage mask){
    	int sh = 0;
    	
    	// Number of input images
    	int nImages = images.size();
    	
    	// Normalise Dictionary     
     	Eigen::MatrixXd NormDictMat = _DictionaryMatrix.rowwise().normalized();
     	
    	// Intialise Output map
    	_OutputMap = images[0];	
    	
    	// Loop through and assign T2 value to each voxel
    	for (int x = sh; x < mask.GetX()-sh; x++) {
            for (int y = sh; y < mask.GetY()-sh; y++) {
                for (int z = sh; z < mask.GetZ()-sh; z++) {

                    double maskval = mask(x,y,z);
                    if (maskval == 0)
                        _OutputMap(x,y,z) = 0;
                    else {

                    // Initialise Sample array for voxel
                    Eigen::VectorXd SampleIntensities(nImages);

                    // Fill sample array
                    //cout<<"Sample Intensities at [" << x << "," << y << "," << z << "]:" << endl;
                    for (int ii = 0; ii < nImages; ii++){
                        //cout << images[ii](x,y,z) << "   ";
                        SampleIntensities[ii] = images[ii](x,y,z);
                        }

                    //cout << endl;

                    // Normalise sample array:
                SampleIntensities.normalize();

                //cout<<"Normalised Sample Intensities at [" << x << "," << y << "," << z << "]:" << endl;
                    //cout << SampleIntensities << endl;

                // Perform scalar product and obtain T2 for max value
                Eigen::VectorXd NormScalarProducts;
                NormScalarProducts = NormDictMat*SampleIntensities;
                int indx;
                NormScalarProducts.maxCoeff(&indx);
                double T2_val = _T2Vals(indx);
                //cout << "Index at [" << x << "," << y << "," << z << "]:" << endl;
                //cout << indx << " and T2 value: " << T2_val << endl;

                _OutputMap(x,y,z) = T2_val;
                }
                }
            }
	    }
	    return _OutputMap;
    }
    
    
}
