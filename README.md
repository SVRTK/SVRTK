SVRTK - slice to volume reconstruction toolkit
====================

<p align="center">
  <img src="https://github.com/SVRTK/SVRTK/tree/master/additional_files/svr-logo.png" width="50%" alt='project-svrtk'>
</p>


SVR reconstruction package for MIRTK (https://biomedia.doc.ic.ac.uk/software/mirtk/) for fetal MRI motion correction including:
- 3D brain
- 3D body
- 4D whole fetal heart, including magnitude and blood flow reconstructions
- 3D placenta
- 3D and 4D multi-channel T2* 
- SH brain diffusion (HARDI) 


The general pipeline for fetal brain reconstruction is based on the  `reconstruction`  function in IRTK: https://biomedia.doc.ic.ac.uk/software/irtk/.

4D cardiac reconstruction code was ported from the original IRTK-based implementation by Joshua van Amerom: https://github.com/jfpva/irtk_cardiac4d.

SVRTK contributes to https://github.com/mriphysics/fetal_cmr_4d pipeline.



Docker 
---

Compiled SVR toolbox is available via [DockerHub](https://hub.docker.com/repository/docker/fetalsvrtk/svrtk):

```bash

docker pull fetalsvrtk/svrtk

docker run -it --rm --mount type=bind,source=location_on_your_machine,target=/home/data fetalsvrtk/svrtk /bin/bash

cd /home/data
mirtk reconstructBody ../outputDSVR.nii.gz 6 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz ../stack4.nii.gz ../stack5.nii.gz ../stack6.nii.gz -mask ../mask.nii.gz -template ../stack2.nii.gz -thickness 2.5 -default -remote -resolution 0.85

```

In order to make sure that reconstruction is fast enough - please select a sufficient number of CPUs (e.g., > 8) and amount of RAM (e.g., > 20 GB) in the Desktop 
Docker settings. You can increase RAM by using virtual RAM on Window and or swap on Ubuntu.  



Installation 
------------

Please follow installation instructions in InstallationInstructions.txt file.

Note, the software can be compiled on either Ubuntu or OS X. 
In order achieve optimal performance it is recommended to run reconstruction on machine  with minimum 6 CPU cores and 32 GB RAM. 
In case of "-9" memory errors - please run reconstruction with "-remote" option (and increase RAM/SWAP).



Run
---

Examples: 


3D brain reconstruction:

```bash
mirtk reconstruct ../outputSVR.nii.gz  5 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz ../stack4.nii.gz ../stack5.nii.gz -mask ../mask.nii.gz  -template ../stack3.nii.gz -thickness 2.5 2.5 2.5 2.5 2.5 -svr_only -resolution 0.75 -iterations 3 -remote
```
 
   ---
3D fetal body DSVR reconstruction:

```bash
mirtk reconstructBody ../outputDSVR.nii.gz 6 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz ../stack4.nii.gz ../stack5.nii.gz ../stack6.nii.gz -mask ../mask.nii.gz -template ../template-stack.nii.gz -thickness 2.5 -default -remote -resolution 0.85
```
  ---
3D placenta DSVR reconstruction:

```bash
mirtk reconstructPlacenta ../outputDSVR.nii.gz 3 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz -mask ../mask.nii.gz -template ../template-stack.nii.gz -thickness 2.5 -default -remote -resolution 1.25
```
  ---
3D multi-channel DSVR reconstruction for T2* (reconstruction of 3D T2* maps driven by one of the echoes, e.g., e2):

```bash
mirtk reconstructMC ../outputDSVR-E2.nii.gz 3 ../stack1-E2.nii.gz ../stack2-E2.nii.gz ../stack3-E2.nii.gz -channels 1 ../stack1-T2sMAP.nii.gz ../stack2-T2sMAP.nii.gz ../stack3-T2sMAP.nii.gz -mask ../mask.nii.gz -template ../template-stack-E2.nii.gz -thickness 2.5 -default -remote -no_intensity_matching -resolution 1.25
```  
(the output T2* map will be in mc-image-0.nii.gz file) 
  
 ---
4D cardiac velocity reconstruction:
- see fetal_cmr_4d git repository for full framework: https://github.com/tomaroberts/fetal_cmr_4d
 
```bash
mirtk reconstructCardiacVelocity 5 ../phase_stack1.nii.gz ../phase_stack2.nii.gz ../phase_stack3.nii.gz ../phase_stack4.nii.gz ../phase_stack5.nii.gz ../g_values.txt ../g_directions.txt -thickness 6 6 6 6 6 -mask ../mask.nii.gz -rec_iterations 40 -transformations [folder with slice transformations from 4D cardiac reconstruction] -limit_intensities -rec_iterations 40 -resolution 1.25 -force_exclude [list of slices that should be excluded] -numcardphase 25 -rrinterval 0.407046 -rrintervals [number of rr_intervals] [list of rr_intervals] -cardphase [number of slices] [cardiac phases for each of the slices] -debug > log-main.txt
```
  ---
 Higher order spherical harmonics (SH) reconstruction of fetal brain diffusion MRI:

```bash
mirtk reconstructDWI ../recon-3D-vol.nii.gz ../4D-DWI-stack.nii.gz ../gradient-directions.b ../target-atlas-space-T2-volume.nii.gz ../dof-to-atlas-space.dof -mask ../mask.nii.gz -order 4 -motion_sigma 15 -resolution 1.5 -thickness 2 -sigma 20 -iterations 5 -template [template_number] -motion_model_hs -sr_sh_iterations 10
```

The resulting reconstructed signal is in _simulated_signal.nii.gz and the SH coefficients are in shCoeff9.nii.gz files.
This should be followed by constrained spherical deconvolution for representation of the signal in SH basis based on the functions from MRtrix (https://github.com/MRtrix3/mrtrix3):

```bash
dwi2response tournier _simulated_signal.nii.gz response.txt -lmax 6 -grad ../gradient-directions.b  -force -mask ../mask-wm.nii.gz
dwi2fod csd signal.nii.gz response.txt csd-out.mif -lmax 6 -grad ../gradient-directions.b -force -mask ../mask-wm.nii.gz
```
 
 (please note that distorion and bias field correction prior should be applied prior to reconstruction)
 
  ---

License
-------

The MIRTK SVRTK package is distributed under the terms of the
[Apache License Version 2](http://www.apache.org/licenses/LICENSE-2.0). The license enables usage of SVRTK in both commercial and non-commercial applications, without restrictions on the licensing applied to the combined work.


Citation and acknowledgements
-----------------------------

Publications:

In case you found SVRTK useful please give appropriate credit to the software.

Original reconstruction pipeline for 3D fetal brain (original software using IRTK: https://gitlab.com/mariadeprez/irtk-simple):
> Kuklisova-Murgasova, M., Quaghebeur, G., Rutherford, M. A., Hajnal, J. V., & Schnabel, J. A. (2012). Reconstruction of fetal brain MRI with intensity matching and complete outlier removal. Medical Image Analysis, 16(8), 1550–1564.: https://doi.org/10.1016/j.media.2012.07.004

4D cardiac magnitude reconstruction (original software using IRTK: https://github.com/jfpva/fetal_cmr_4d):
> van Amerom, J. F. P., Lloyd, D. F. A., Deprez, M., Price, A. N., Malik, S. J., Pushparajah, K., van Poppel, M. P. M, Rutherford, M. A., Razavi, R., Hajnal, J. V. (2019). Fetal whole-heart 4D imaging using motion-corrected multi-planar real-time MRI. Magnetic Resonance in Medicine, 82(3): 1055-1072. : https://doi.org/10.1002/mrm.27858

4D cardiac velocity reconstruction:
> Roberts, T. A., van Amerom, J. F. P., Uus, A., Lloyd, D. F. A., Price, A. N., Tournier, J-D., Mohanadass, C. A., Jackson, L. H., Malik, S. J., van Poppel, M. P. M, Pushparajah, K., Rutherford, M. A., Razavi, R., Deprez, M., Hajnal, J. V. (2020).Fetal whole heart blood flow imaging using 4D cine MRI. Nat Commun 11, 4992: https://doi.org/10.1038/s41467-020-18790-1

3D DSVR fetal body / placenta reconstruction:
> Uus, A., Zhang, T., Jackson, L., Roberts, T., Rutherford, M., Hajnal, J.V., Deprez, M. (2020). Deformable Slice-to-Volume Registration for Motion Correction in Fetal Body MRI and Placenta. IEEE Transactions on Medical Imaging, 39(9), 2750-2759: http://dx.doi.org/10.1109/TMI.2020.2974844
 
SH brain diffusion reconstruction (HARDI): 
> Deprez, M., Price, A., Christiaens, D., Estrin, G.L., Cordero-Grande, L., Hutter, J., Daducci, A., Tournier, J-D., Rutherford, M., Counsell, S. J., Cuadra, M. B., Hajnal, J. V. (2020). Higher Order Spherical Harmonics Reconstruction of Fetal Diffusion MRI with Intensity Correction. IEEE Transactions on Medical Imaging, 39 (4), 1104–1113.: https://doi.org/10.1109/tmi.2019.2943565.

3D and 4D T2* placenta reconstruction: 
> Uus, A., Steinweg, J. K., Ho, A., Jackson, L. H., Hajnal, J. V., Rutherford, M. A., Deprez, M., Hutter, J. (2020) Deformable Slice-to-Volume Registration for Reconstruction of Quantitative T2* Placental and Fetal MRI. In: Hu Y. et al. (eds) Medical Ultrasound, and Preterm, Perinatal and Paediatric Image Analysis. ASMUS 2020, PIPPI 2020. Lecture Notes in Computer Science, vol 12437. Springer, Cham: https://doi.org/10.1007/978-3-030-60334-2_22


