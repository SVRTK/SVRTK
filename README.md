MIRTK SVR Package
====================


SVR reconstruction package for MIRTK (https://biomedia.doc.ic.ac.uk/software/mirtk/) for fetal MRI motion correction. 

The general pipeline for fetal brain reconstruction is based on `reconstruction` IRTK application : (https://biomedia.doc.ic.ac.uk/software/irtk/).

4D Cardiac reconstruction code was transferred to MIRTK from the original IRTK-based implementation by Joshua van Amerom : https://github.com/jfpva/irtk_cardiac4d .


Installation 
------------

Please follow installation instructions in InstallationInstructions.txt file. 


Run
---

Examples: 


brain reconstruction:

reconstruct ../outputSVR.nii.gz  4 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz ../stack4.nii.gz  -mask ../mask.nii.gz  -template_number 2  -thickness 2 2 2 2 -resolution 0.75 -iterations 3 
 
 ---
4D cardiac velocity reconstruction (it will require the input from 4D cardiac reconstruction using the full pipeline from: https://github.com/jfpva/fetal_cmr_4d ): 
 
reconstructionCardiacVelocity 4 ../phase_stack1.nii.gz ../phase_stack2.nii.gz ../phase_stack3.nii.gz ../phase_stack4.nii.gz ../phase_stack5.nii.gz ../g_values.txt ../g_directions.txt -thickness 6 6 6 6 6 -mask ../mask.nii.gz -rec_iterations 40 -transformations [folder with slice transformations from 4D cardiac reconstruction] -limit_intensities -rec_iterations 40 -resolution 1.25 -force_exclude [list of slices that should be excluded] -numcardphase 25 -rrinterval 0.407046 -rrintervals [number of rr_intervals] [list of rr_intervals] -cardphase [number of slices] [cardiac phases for each of the slices] -debug > log-main.txt


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

Publications:

 ---
original reconstruction pipeline:

Kuklisova-Murgasova, M., Quaghebeur, G., Rutherford, M. A., Hajnal, J. V., & Schnabel, J. A. (2012). Reconstruction of fetal brain MRI with intensity matching and complete outlier removal. Medical Image Analysis, 16(8), 1550–1564.

 ---
4D cardiac velocity reconstruction:

Roberts, T.A., van Amerom, J.F.P., Uus, A., Lloyd, D.F.A., Price, A.N., Tournier, J.-D., Jackson, L.H., Malik, S.J., van Poppel, M.P.M., Pushparajah, K., Rutherford, M.A., Rezavi, R., Deprez, M., Hajnal, J. V, 2019. Fetal whole-heart 4D flow cine MRI using multiple non-coplanar balanced SSFP stacks. bioRxiv 635797.

 ---
for the full 4D cardiac reconstruction pipeline - please use the original software https://github.com/jfpva/fetal_cmr_4d and cite:

van Amerom, J.F., Lloyd, D.F., Deprez, M., Price, A.N., Malik, S.J., Pushparajah, K., van Poppel, M.P., Rutherford, M.A., Razavi, R., Hajnal, J. V, 2019. Fetal whole-heart 4D imaging using motion-corrected multi-planar real-time MRI. Magn. Reson. Med. 82(3), 1055–1072.

 ---
 
