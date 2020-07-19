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


if [ "$#" -ne 3 ]; then
    echo ""
    echo "# # # "
    echo "# # # "
    echo "# # # Illegal number of parameters !!! "
    echo "# # # "
    echo "# # # Usage : bash ./propagate-mask-4d.sh [input_4d_stack] [template_stack] [template_mask]"
    echo "# # # "
    echo "# # # "
    echo ""
    exit 0
fi


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



if [[ ! -d "prop_proc" ]];then
    mkdir prop_proc
fi



input_stack=main_input_stack.nii.gz
template_stack=main_template_stack.nii.gz
template_mask=main_template_mask.nii.gz


cp $1 prop_proc/main_input_stack.nii.gz
cp $2 prop_proc/main_template_stack.nii.gz
cp $3 prop_proc/main_template_mask.nii.gz



cd prop_proc

output_mask=../output_propagated_mask.nii.gz
output_volume=../output_registered_volume.nii.gz
output_jac=../output_jacobian.nii.gz
output_original=../original_volume.nii.gz

output_jac_time=../time_output_jacobian.nii.gz
output_original_time=../time_original_volume.nii.gz


output_mask_average=../average_output_mask.nii.gz
output_original_average=../average_original_volume.nii.gz
output_jac_average=../average_output_jacobian.nii.gz



if [[ -f "tmp_stack_00.nii.gz" ]];then
    rm tmp_*.nii.gz
fi


${mirtk_path} extract-image-region ${input_stack} tmp_stack.nii.gz -split t

num_dyn=$(find . -maxdepth 1 -name "tmp_stack*" | wc -l)

mirtk init-dof i.dof


echo " - input 4d stack : " ${input_stack}
echo " - template stack : " ${template_stack}
echo " - template mask : " ${template_mask}
echo " - number of dynamics : " ${num_dyn}

echo
echo "----------------------------------------------------------------------------"
echo

echo " - running registration : "
echo


th=9

for ((j=0;j<${num_dyn};j++));
do
    
    if [[ "${j}" -gt "${th}" ]];then
        st_i=${j}
    else
        st_i=0${j}
    fi
    
    if [[ -f "tmp_stack_${st_i}.nii.gz" ]];then

        echo "  " ${st_i}

        ${mirtk_path} edit-image tmp_stack_${st_i}.nii.gz tmp_stack_${st_i}.nii.gz -torigin 0

        ${mirtk_path} register tmp_stack_${st_i}.nii.gz ${template_stack} -model FFD -bg -1 -dofout tmp_dout_${st_i}.dof -dofin i.dof -v 0

        ${mirtk_path} transform-image ${template_mask} tmp_mask_${st_i}.nii.gz -dofin tmp_dout_${st_i}.dof -target tmp_stack_${st_i}.nii.gz -interp NN
        ${mirtk_path} invert-dof tmp_dout_${st_i}.dof tmp_i_dout_${st_i}.dof
        ${mirtk_path} transform-image tmp_stack_${st_i}.nii.gz tmp_reg_${st_i}.nii.gz -dofin tmp_i_dout_${st_i}.dof -target tmp_stack_${st_i}.nii.gz -interp Linear

        ${mirtk_path} evaluate-jacobian tmp_stack_${st_i}.nii.gz tmp_jac_${st_i}.nii.gz tmp_dout_${st_i}.dof
    
    fi
    
done



echo
echo "----------------------------------------------------------------------------"
echo


echo " - combining images : "
echo

all_dynamics_files=" "

for ((i=0;i<10;i++));
do
    dynamics_file=tmp_mask_0${i}.nii.gz
    all_dynamics_files+=" ${dynamics_file}"
done

for ((i=10;i<${num_dyn};i++));
do
    dynamics_file=tmp_mask_${i}.nii.gz
    all_dynamics_files+=" ${dynamics_file}"
done

${mirtk_path} combine-images ${all_dynamics_files} ${output_mask}



all_dynamics_files=" "

for ((i=0;i<10;i++));
do
    dynamics_file=tmp_reg_0${i}.nii.gz
    all_dynamics_files+=" ${dynamics_file}"
done

for ((i=10;i<${num_dyn};i++));
do
    dynamics_file=tmp_reg_${i}.nii.gz
    all_dynamics_files+=" ${dynamics_file}"
done

${mirtk_path} combine-images ${all_dynamics_files} ${output_volume}



all_dynamics_files=" "

for ((i=0;i<10;i++));
do
    dynamics_file=tmp_jac_0${i}.nii.gz
    all_dynamics_files+=" ${dynamics_file}"
done

for ((i=10;i<${num_dyn};i++));
do
    dynamics_file=tmp_jac_${i}.nii.gz
    all_dynamics_files+=" ${dynamics_file}"
done

${mirtk_path} combine-images ${all_dynamics_files} ${output_jac}


${mirtk_path} edit-image ${output_mask} ${output_mask} -dt 0.01
${mirtk_path} edit-image ${output_volume} ${output_volume} -dt 0.01
${mirtk_path} edit-image ${output_jac} ${output_jac} -dt 0.01

cp ${input_stack} ${output_original}
${mirtk_path} edit-image ${output_original} ${output_original} -dt 0.01



${mirtk_path} flip-image ${output_original} ${output_original_time} -zt
${mirtk_path} flip-image ${output_jac} ${output_jac_time} -zt

${mirtk_path} edit-image ${output_original_time} ${output_original_time} -dz 1.5
${mirtk_path} edit-image ${output_jac_time} ${output_jac_time} -dz 1.5





${mirtk_path} average-images ${output_mask_average} -image ${output_mask}
${mirtk_path} average-images ${output_original_average} -image ${output_original}
${mirtk_path} average-images ${output_jac_average} -image ${output_jac}


echo " - output 4d mask : " ${output_mask}
echo " - output registered 4d dynamics : " ${output_volume}
echo " - output 4d jacobian from registration : " ${output_jac}

echo

echo " - original 4d volume (just for the reference / output original with 0.01 ms dt) : " ${output_original}
echo " - original 4d volume with flipped z-t (time profile) : " ${output_original_time}
echo " - original 4d jacobian with flipped z-t (time profile) : " ${output_jac_time}

echo

echo " - output average volume (3d) : " ${output_original_average}
echo " - output average mask (3d) : " ${output_mask_average}
echo " - output average jacobian (3d) : " ${output_jac_average}



echo
echo "----------------------------------------------------------------------------"
echo "----------------------------------------------------------------------------"
echo


