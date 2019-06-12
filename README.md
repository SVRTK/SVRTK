MIRTK SVR Package
====================


SVR reconstruction package for MIRTK (https://biomedia.doc.ic.ac.uk/software/mirtk/) for fetal MRI motion correction. 

The general pipeline for fetal brain reconstruction is based on `reconstruction` IRTK application (https://biomedia.doc.ic.ac.uk/software/irtk/).



Installation 
------------

Please follow the installation instructions in InstallationInstructions.txt file. 


Run
---

Examples: 


brain reconstruction:

reconstruct ../outputSVR.nii.gz  4 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz ../stack4.nii.gz  -mask ../mask.nii.gz  -template_number 2  -thickness 2 2 2 2 -resolution 0.75 -iterations 3 
 
 ---
 
placenta reconstruction:
 
reconstructPlacenta ../outputDSVR.nii.gz  2 ../stack1.nii.gz ../stack2.nii.gz  -mask ../mask.nii.gz  -thickness 2 2 -resolution 1.0 -iterations 2 -template ../template.nii.gz -ffd -filter 3 
 
  ---
 


License
-------

The MIRTK SVRTK package is distributed under the terms of the
[Apache License Version 2](http://www.apache.org/licenses/LICENSE-2.0).



Citation and acknowledgements
-----------------------------

In case you found SVRTK useful please give appropriate credit to the software.

Publication:

Kuklisova-Murgasova, M., Quaghebeur, G., Rutherford, M. A., Hajnal, J. V., & Schnabel, J. A. (2012). Reconstruction of fetal brain MRI with intensity matching and complete outlier removal. Medical Image Analysis, 16(8), 1550–1564.


