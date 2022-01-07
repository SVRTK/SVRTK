Run
===

## 3D brain reconstruction

```bash
mirtk reconstruct ../outputSVR.nii.gz  5 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz ../stack4.nii.gz ../stack5.nii.gz -mask ../mask.nii.gz  -template ../stack3.nii.gz -thickness 2.5 2.5 2.5 2.5 2.5 -svr_only -resolution 0.75 -iterations 3 -remote
```

 _Notes: The template stack should be the least motion corrupted and the brain position should correspond to the average position between all stacks (e.g., in the middle of the acquisition). The mask should be created for the template stack and cover the brain/head only - without stationary maternal tissue._

   ---
## 3D fetal body DSVR reconstruction

```bash
mirtk reconstructBody ../outputDSVR.nii.gz 6 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz ../stack4.nii.gz ../stack5.nii.gz ../stack6.nii.gz -mask ../mask.nii.gz -template ../template-stack.nii.gz -thickness 2.5 -default -remote -resolution 0.85
```

 _Notes: The template stack should be the least motion corrupted and the body/thorax position should correspond to the average position between all stacks (e.g., in the middle of the acquisition). The mask should be created for the template stack and cover the investigated ROI._

  ---
## 3D placenta DSVR reconstruction

```bash
mirtk reconstructPlacenta ../outputDSVR.nii.gz 3 ../stack1.nii.gz ../stack2.nii.gz ../stack3.nii.gz -mask ../mask.nii.gz -template ../template-stack.nii.gz -thickness 2.5 -default -remote -resolution 1.25
```
  ---
## 3D multi-channel DSVR reconstruction for T2* (reconstruction of 3D T2* maps driven by one of the echoes, e.g., E2, and transformations are propagated to the T2* map channel)

```bash
mirtk reconstructMC ../outputDSVR-E2.nii.gz 3 ../stack1-E2.nii.gz ../stack2-E2.nii.gz ../stack3-E2.nii.gz -channels 1 ../stack1-T2sMAP.nii.gz ../stack2-T2sMAP.nii.gz ../stack3-T2sMAP.nii.gz -mask ../mask.nii.gz -template ../template-stack-E2.nii.gz -thickness 2.5 -default -remote -no_intensity_matching -resolution 1.25
```
 _Notes: The output reconstructed T2* map will be in mc-image-0.nii.gz file._

 ---
## 4D cardiac velocity reconstruction

```bash
mirtk reconstructCardiacVelocity 5 ../phase_stack1.nii.gz ../phase_stack2.nii.gz ../phase_stack3.nii.gz ../phase_stack4.nii.gz ../phase_stack5.nii.gz ../g_values.txt ../g_directions.txt -thickness 6 6 6 6 6 -mask ../mask.nii.gz -rec_iterations 40 -transformations [folder with slice transformations from 4D cardiac reconstruction] -limit_intensities -rec_iterations 40 -resolution 1.25 -force_exclude [list of slices that should be excluded] -numcardphase 25 -rrinterval 0.407046 -rrintervals [number of rr_intervals] [list of rr_intervals] -cardphase [number of slices] [cardiac phases for each of the slices] -debug > log-main.txt
```
 _See [fetal_cmr_4d](https://github.com/tomaroberts/fetal_cmr_4d) git repository for full framework._

  ---
## Higher order spherical harmonics (SH) reconstruction of fetal brain diffusion MRI

```bash
mirtk reconstructDWI ../recon-DWI-vol.nii.gz 2 ../4D-DWI-stack-1.nii.gz ../4D-DWI-stack-2.nii.gz ../gradient-directions-1.b ../gradient-directions-2.b 1000 ../target-atlas-space-T2-volume.nii.gz ../dof-to-atlas-space.dof -mask ../mask.nii.gz -order 4 -motion_sigma 15 -resolution 1.5 -thickness 2 -sigma 20 -iterations 5 -template [template_number, e.g., 10] -motion_model_hs -sr_sh_iterations 10 -resolution 1.75 -no_robust_statistics
```

 _Notes: The algorithm uses only 1 shell for reconstruction. You need to specify it after the .b files (e.g., B=1000). The rest of the shells will be excluded. The combined file with all gradient directions for all stacks will be in final-b-file.b._

The resulting reconstructed DWI signal will be in ../recon-DWI-vol.nii.gz and the SH coefficients are in shCoeff9.nii.gz files.
This should be followed by constrained spherical deconvolution for representation of the signal in SH basis based on the functions from MRtrix (https://github.com/MRtrix3/mrtrix3):

```bash v
dwi2response tournier ../recon-DWI-vol.nii.gz response.txt -lmax 6 -grad final-b-file.b  -force -mask ../mask-wm.nii.gz
dwi2fod csd signal.nii.gz response.txt csd-out.mif -lmax 6 -grad ../gradient-directions.b -force -mask ../mask-wm.nii.gz
```

 _Notes: Please note that distorion and bias field correction prior should be applied prior to reconstruction._
