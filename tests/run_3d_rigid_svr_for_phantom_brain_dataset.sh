#!/bin/bash -e
#
# SVRTK : SVR reconstruction based on MIRTK
#
# Copyright 2018-2021 King's College London
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#---------------------------------------------------------------------------------------------
# TEST FOR RUNNING 3D SVR ON A PHANTOM MOTION-CORRUPTED DATASET GENERATED FROM A TEMPLATE
# THE FILES FOR TESTING ARE IN: tests/testing_data/fetal_brain_mri_phantom_dataset
#---------------------------------------------------------------------------------------------

# update the path to compiled MIRTK
path_to_mirtk=PLEASE_UPDATE

# create a folder for processing files
mkdir out
cd out

# run 3D SVR reconstruction
${path_to_mirtk}/mirtk reconstruct ../SVR-output.nii.gz 6 ../simulated-stack* -template ../simulated-stack-d2.nii.gz -mask ../mask-for-stack-2.nii.gz -resolution 0.8 - thickness 2.5 2.5 2.5 2.5 2.5 2.5 -iterations 3 -svr_only



