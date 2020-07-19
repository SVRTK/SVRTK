#!/bin/bash -e


#/*
#* SVRTK : SVR reconstruction based on MIRTK
#*
#* Copyright 2018-2020 King's College London
#*
#* Licensed under the Apache License, Version 2.0 (the "License");
#* you may not use this file except in compliance with the License.
#* You may obtain a copy of the License at
#*
#*     http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing, software
#* distributed under the License is distributed on an "AS IS" BASIS,
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#* See the License for the specific language governing permissions and
#* limitations under the License.
#*/


if [ "$#" -ne 4 ]; then
    echo ""
    echo "# # # "
    echo "# # # "
    echo "# # # Illegal number of parameters !!! "
    echo "# # # "
    echo "# # # Usage : bash ./propagate-mask-3d-thorax-atlas.sh [input_3d_volume] [atlas_volume] [atlas_organ_mask] [atlas_thorax_mask]"
    echo "# # # "
    echo "# # # "
    echo ""
    exit 0
fi


# NOTE: in order to to use this script - update the path to compiled mirtk
mirtk_path=$(which mirtk)


if [[ ! -f "${mirtk_path}" ]];then
    echo "# # # "
    echo "# # # Error : Could not find MIRTK library in : " ${mirtk_path}
    echo ""
    exit 0
fi


echo
echo "----------------------------------------------------------------------------"
echo "----------------------------------------------------------------------------"
echo


if [[ ! -d "prop_proc" ]];then
    mkdir prop_proc
fi



if [[ ! -f "$1" ]];then
    echo
    echo " !!! no file found : " $1
    echo
    exit 0
fi

if [[ ! -f "$2" ]];then
    echo
    echo " !!! no file found : " $2
    echo
    exit 0
fi

if [[ ! -f "$3" ]];then
    echo
    echo " !!! no file found : " $3
    echo
    exit 0
fi

if [[ ! -f "$4" ]];then
    echo
    echo " !!! no file found : " $4
    echo
    exit 0
fi



cp $1 prop_proc/input_volume.nii.gz
cp $2 prop_proc/template_volume.nii.gz
cp $3 prop_proc/template_organ_mask.nii.gz
cp $4 prop_proc/template_thorax_mask.nii.gz


output_reset_volume=../main_volume_reset.nii.gz
output_organ_mask=../output_propagated_organ_mask_ZZ.nii.gz


echo " - input 3d volume : " $1
echo " - template : " $2
echo " - template organ mask : " $3
echo " - template thorax mask : " $4


cd prop_proc


input_volume=input_volume.nii.gz
template_volume=template_volume.nii.gz
template_organ_mask=template_organ_mask.nii.gz
template_thorax_mask=template_thorax_mask.nii.gz

cp input_volume.nii.gz org_volume.nii.gz

${mirtk_path} init-dof i.dof
${mirtk_path} edit-image ${input_volume} ${input_volume} -origin 0 0 0
${mirtk_path} edit-image ${template_volume} ${template_volume} -origin 0 0 0
${mirtk_path} edit-image ${template_thorax_mask} ${template_thorax_mask} -origin 0 0 0
${mirtk_path} edit-image ${template_organ_mask} ${template_organ_mask} -origin 0 0 0


echo
echo "----------------------------------------------------------------------------"
echo

echo " - running registration ... "
echo


${mirtk_path} register ${template_volume} ${input_volume} -model Affine+FFD -dofin i.dof -ds 15 -dofout d_global.dof -output test1.nii.gz -mask ${template_thorax_mask} -v 0

${mirtk_path} invert-dof d_global.dof d_global.dof
${mirtk_path} dilate-image ${template_thorax_mask}  dl-template_thorax_mask.nii.gz -iterations 4
${mirtk_path} transform-image dl-template_thorax_mask.nii.gz global_thorax_mask.nii.gz -dofin d_global.dof -interp NN -target ${input_volume}

${mirtk_path} register ${input_volume} ${template_volume} -model Affine+FFD -dofin i.dof -dofout d_main.dof -output test2.nii.gz -mask global_thorax_mask.nii.gz -v 0

${mirtk_path} transform-image ${template_organ_mask}  ${output_organ_mask}  -dofin d_main.dof -interp NN -target ${input_volume}
${mirtk_path} edit-image ${output_organ_mask} ${output_organ_mask} -copy-origin org_volume.nii.gz


echo
echo "----------------------------------------------------------------------------"
echo

echo " - output : "
echo


echo " - output propagated organ mask : " ${output_organ_mask}


echo
echo "----------------------------------------------------------------------------"
echo "----------------------------------------------------------------------------"
echo


