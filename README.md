
MIRTK SVR 4D Flow cine package
====================

SVR 4D flow cine reconstruction package for MIRTK.

SVR code originally based on `reconstruction` as part of IRTK: https://github.com/BioMedIA/IRTK.

Cardiac 4D reconstruction code was ported to MIRTK from the original IRTK implementation by Joshua van Amerom: https://github.com/jfpva/irtk_cardiac4d.



Installation
------------

For CardiacVelocity4D branch: `git clone -b CardiacVelocity4D https://github.com/SVRTK/SVRTK.git`

SVRTK requires installation of MIRTK (https://github.com/BioMedIA/MIRTK) with TBB option. 

(Note: the bin files are located in: ~/path/to/build/folder/lib/tools/)



Run
---

Pre-requisite: 4D magnitude cine generated using https://github.com/jfpva/fetal_cmr_4d.

Example for generating 4D flow cine, e.g., in shell: 
```shell
RECONDIR=~/path/to/recon/directory
cd $RECONDIR

mkdir vel_vol 
cd vel_vol

reconstructCardiacVelocity 5 ../data/phase_stack1.nii.gz ../data/phase_stack2.nii.gz ../data/phase_stack3.nii.gz ../data/phase_stack4.nii.gz ../data/phase_stack5.nii.gz ../data/g_values.txt ../data/g_directions.txt -thickness 6 6 6 6 6 -dofin [5 stack transformation00*.dof files in ../dc_vol/stack_transformations/] -transformations [folder with slice transformations from 4D cardiac reconstruction in ../cine_vol/transformations] -mask ../mask/mask.nii.gz -alpha 3 -limit_intensities -rec_iterations 40 -resolution 1.25 -force_exclude_stack 0 -force_exclude_sliceloc 0 -force_exclude [list of slices to exclude] -numcardphase 25 -rrinterval [R-R interval of reconstructed cine volume] -rrintervals [number of R-R intervals] [list of slice R-R intervals] -cardphase [number of slices] [cardiac phases for each of the slices] -debug > log-main.txt
```



License
-------

The MIRTK SVRTK package is distributed under the terms of the
[Apache License Version 2](http://www.apache.org/licenses/LICENSE-2.0).



Citation and acknowledgements
-----------------------------

In case you found SVRTK useful please give appropriate credit to the software.

Publications:

> Kuklisova-Murgasova, M., Quaghebeur, G., Rutherford, M. A., Hajnal, J. V., & Schnabel, J. A. (2012). Reconstruction of fetal brain MRI with intensity matching and complete outlier removal. Medical Image Analysis, 16(8), 1550–1564.: https://doi.org/10.1016/j.media.2012.07.004

> van Amerom, J. F. P., Lloyd, D. F. A., Deprez, M., Price, A. N., Malik, S. J., Pushparajah, K., van Poppel, M. P. M, Rutherford, M. A., Razavi, R., Hajnal, J. V. (2019). Fetal whole-heart 4D imaging using motion-corrected multi-planar real-time MRI. Magnetic Resonance in Medicine.: https://doi.org/10.1002/mrm.27858

> Roberts, T. A., van Amerom, J. F. P., Uus, A., Lloyd, D. F. A., Price, A. N., Tournier, J-D., Jackson, L. H., Malik, S. J., van Poppel, M. P. M, Pushparajah, K., Rutherford, M. A., Razavi, R., Deprez, M., Hajnal, J. V. (2019). Fetal whole-heart 4D flow cine MRI using multiple non-coplanar balanced SSFP stacks. bioRxiv.: https://doi.org/10.1101/635797


