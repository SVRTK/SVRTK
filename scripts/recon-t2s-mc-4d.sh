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

echo
echo ".........................................................................." 
echo ".........................................................................."
echo




if [ "$#" -ne 4 ]; then
    echo ""
    echo "# # # "
    echo "# # # "
    echo "# # # Illegal number of parameters !!! "
    echo "# # # "
    echo "# # # Usage : bash ./recon-t2s-mc-4d.sh [input_4d_t2s-e2_stack] [input_4d_t2s-map_stack] [template_mask] [thickness] (please pass the full file paths)"
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




selected_roi=0
resolution=1.75



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



t2me_volume=$1
t2map_org_volume=$2
template_mask=$3
thickness=$4


dynamics_number=$(${mirtk_path} get-t ${t2map_org_volume})
template_number=$(${mirtk_path} get-middle-limits ${t2me_volume} 0 3)


main_path=$(pwd)


template_t2map_volume=${main_path}/recon_4d-t2s-e2-template.nii.gz
template_volume=${main_path}/recon_4d-t2s-map-template.nii.gz

t2map_volume_recon=${main_path}/recon_4d-t2s-map.nii.gz
t2me_volume_recon=${main_path}/recon_4d-t2s-e2.nii.gz




if [[ ! -d "tmp_proc_files" ]];then
    mkdir tmp_proc_files
fi

cd tmp_proc_files




echo
echo ".........................................................................."
echo ".........................................................................."
echo

echo
echo " - STEP 1: Reconstructing 3d template ... "
echo 



n1=${template_number}
n2=${n1}
#$(( ${n1} + 1 ))
${mirtk_path} extract-image-region ${t2me_volume} local_template-${template_number}.nii.gz -Rt1 ${n1} -Rt2 ${n2}
${mirtk_path} edit-image local_template-${template_number}.nii.gz local_template-${template_number}.nii.gz -torigin 0

cp ${template_mask} proc-mask-t2me4d-${template_number}_${stack_id}.nii.gz
mask_file=proc-mask-t2me4d-${template_number}_${stack_id}.nii.gz


tmp_dir_name=out-t2me4d-t2map
if [ ! -d ${tmp_dir_name} ]
then
	mkdir out-t2me4d-t2map
fi

cd out-t2me4d-t2map


n1=$(${mirtk_path} get-middle-limits ${t2me_volume} -1 2)
n2=$(${mirtk_path} get-middle-limits ${t2me_volume} 1 2)


${mirtk_path} extract-image-region  ${t2me_volume} ../local_t2me.nii.gz -Rt1 ${n1} -Rt2 ${n2} 
${mirtk_path} extract-image-region  ${t2map_volume} ../local_t2map.nii.gz -Rt1 ${n1} -Rt2 ${n2} 
${mirtk_path} edit-image ../local_t2me.nii.gz ../local_t2me.nii.gz -torigin 0 
${mirtk_path} edit-image ../local_t2map.nii.gz ../local_t2map.nii.gz -torigin 0 


${mirtk_path} reconstructMC ${template_volume} 1 ../local_t2me.nii.gz -channels 2 ../local_t2me.nii.gz ../local_t2map.nii.gz -template ../local_template-${template_number}.nii.gz -mask ../proc-mask-t2me4d-${template_number}_${stack_id}.nii.gz -iterations 2  -remote -dilation 7 -structural -no_intensity_matching -no_robust_statistics -cp 10 5 -delta 100 -lambda 0.015 -lastIter 0.010 -thickness ${thickness} -resolution ${resolution} -remote > tmp.txt


#  -no_global

cp mc-image-1.nii.gz ${template_t2map_volume}

echo " -outputs : " 
echo ${template_t2map_volume} 
echo ${template_volume}




echo
echo ".........................................................................."
echo


echo
echo " - STEP 2: Reconstructing all dynamics ... "
echo 


for ((i=0;i<${dynamics_number};i++));
do

	echo
	echo ".........................................................................."
	echo
	echo " - " ${i} " ... " 
	echo 

	i1=${i}
	i2=${i} #$(( ${i1} + 1 ))
 

	${mirtk_path} extract-image-region ${t2me_volume} current_t2me.nii.gz -Rt1 ${i1} -Rt2 ${i2} 
	${mirtk_path} extract-image-region ${t2map_volume} current_t2map.nii.gz -Rt1 ${i1} -Rt2 ${i2} 
	${mirtk_path} edit-image current_t2me.nii.gz current_t2me.nii.gz -torigin 0 
	${mirtk_path} edit-image current_t2map.nii.gz current_t2map.nii.gz -torigin 0 

	${mirtk_path} resample-image current_t2me.nii.gz tmp-res.nii.gz -size 1.7 1.7 1.7 -interp Linear
	${mirtk_path} register ${template_volume} tmp-res.nii.gz -model FFD -output tmp-template-t2me.nii.gz -v 0

	
	${mirtk_path} reconstructMC recon-${i}.nii.gz 1 current_t2me.nii.gz -channels 1 current_t2map.nii.gz -mask ${template_mask} -template tmp-template-t2me.nii.gz -iterations 2  -dilation 7 -cp 10 2 -no_robust_statistics -structural -no_intensity_matching -delta 100 -lambda 0.015 -lastIter 0.010 -thickness ${thickness} -resolution ${resolution}  -remote > tmp.txt

	cp mc-image-0.nii.gz map-recon-${i}.nii.gz
	

	${mirtk_path} register ${template_volume}  recon-${i}.nii.gz -model FFD -levels 8 -dofout z.dof > tmp-reg-${i}.txt -v 0
	${mirtk_path} register ${template_volume}  recon-${i}.nii.gz -model FFD -dofin z.dof -sim NCC -window 5 -dofout d-${i}.dof > tmp-reg-${i}.txt -v 0


	${mirtk_path} transform-image mc-image-0.nii.gz tr-map-recon-${i}.nii.gz -dofin d-${i}.dof -target ${template_volume} -interp BSpline
	${mirtk_path} convert-image tr-map-recon-${i}.nii.gz tr-map-recon-${i}.nii.gz -short 
	
	${mirtk_path} transform-image recon-${i}.nii.gz tr-recon-${i}.nii.gz -dofin d-${i}.dof -target ${template_volume} -interp BSpline
	${mirtk_path} convert-image tr-recon-${i}.nii.gz tr-recon-${i}.nii.gz -short 


	echo 

done 


echo
echo ".........................................................................."
echo


echo " - combining recons ("  ${dynamics_number} ") ... "

all_dynamics_files=" "
for ((i=0;i<${dynamics_number};i++));
do
	dynamics_file=tr-recon-${i}.nii.gz
	all_dynamics_files+=" ${dynamics_file}"
done 
${mirtk_path} combine-images ${all_dynamics_files} ${t2me_volume_recon}
${mirtk_path} edit-image ${t2me_volume_recon} ${t2me_volume_recon} -dt 1
${mirtk_path} convert-image ${t2me_volume_recon} ${t2me_volume_recon} -short 


all_dynamics_files=" "
for ((i=0;i<${dynamics_number};i++));
do
	dynamics_file=tr-map-recon-${i}.nii.gz
	all_dynamics_files+=" ${dynamics_file}" 
done 

${mirtk_path} combine-images ${all_dynamics_files} ${t2map_volume_recon}
${mirtk_path} edit-image ${t2map_volume_recon} ${t2map_volume_recon} -dt 1
${mirtk_path} convert-image ${t2map_volume_recon} ${t2map_volume_recon} -short 
	

echo " - output volumes : "

echo ${t2me_volume_recon}
echo ${t2map_volume_recon}



echo 
echo ".........................................................................."
echo ".........................................................................."
echo 









