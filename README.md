MIRTK ZSVR Package
====================


SVR reconstruction package for MIRTK, originally known as `reconstruction` as part of IRTK.


Example: 


mkdir recon

cd recon

reconstruct   ../outputSVR.nii.gz  6 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz ../stack4.nii.gz  -mask ../mask.nii.gz  -template_number 2  -thickness 2 2 2 2  -resolution 0.75 -iterations 3 



License
-------

The MIRTK ZSVR module is distributed under the terms of the
[Apache License Version 2](http://www.apache.org/licenses/LICENSE-2.0).
