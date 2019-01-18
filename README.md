MIRTK SVRTK Package
====================


SVR reconstruction package for MIRTK, originally known as `reconstruction` as part of IRTK.



Installation 
------------

SVRTK requires installation of MIRTK (https://github.com/BioMedIA/MIRTK) with TBB option.  



Run
---

Example: 

mkdir recon 

cd recon 

 reconstruct   ../outputSVR.nii.gz  4 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz ../stack4.nii.gz  -mask ../mask.nii.gz  -template_number 2  -thickness 2 2 2 2  -resolution 0.75 -iterations 3 



License
-------

The MIRTK SVRTK package is distributed under the terms of the
[Apache License Version 2](http://www.apache.org/licenses/LICENSE-2.0).



Citation and acknowledgements
-----------------------------

In case you found SVRTK useful please give appropriate credit to the software.

Publication:

Kuklisova-Murgasova, M., Quaghebeur, G., Rutherford, M. A., Hajnal, J. V., & Schnabel, J. A. (2012). Reconstruction of fetal brain MRI with intensity matching and complete outlier removal. Medical Image Analysis, 16(8), 1550–1564.


