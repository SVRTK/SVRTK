# ============================================================================
# SVRTK : SVR reconstruction based on MIRTK
#
# Copyright 2008-2017 Imperial College London
# Copyright 2018-2020 King's College London
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
# ============================================================================

##############################################################################
# @file  CMakeLists.txt 
# @brief Build configuration of MIRTK SVRTK tools.
##############################################################################

#set(USE_Python TRUE) 

mirtk_add_executable(
  reconstruct 
  SOURCES
    reconstruct.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)


mirtk_add_executable(
  reconstructDWI
  SOURCES
    reconstructDWI.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)


mirtk_add_executable(
  reconstructMC
  SOURCES
    reconstructMC.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)


mirtk_add_executable(
  reconstructCardiacVelocity
  SOURCES
    reconstructCardiacVelocity.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)


mirtk_add_executable(
  reconstructCardiac
  SOURCES
    reconstructCardiac.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)



mirtk_add_executable(
  convert
  SOURCES
    convert.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)


mirtk_add_executable(
  reconstructPlacenta
  SOURCES
    reconstructPlacenta.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)

mirtk_add_executable(
  reconstructAngio
  SOURCES
    reconstructAngio.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)

mirtk_add_executable(
  reconstructBody
  SOURCES
    reconstructBody.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)


mirtk_add_executable(
  combine_patches
  SOURCES
    combine_patches.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)

mirtk_add_executable(
  combine_masks
  SOURCES
    combine_masks.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)

mirtk_add_executable(
  filter_background
  SOURCES
    filter_background.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)

mirtk_add_executable(
  threshold_image
  SOURCES
    threshold_image.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)


mirtk_add_executable(
  pad_image
  SOURCES
    pad_image.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)

mirtk_add_executable(
  enlarge_image
  SOURCES
    enlarge_image.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)


mirtk_add_executable(
  crop_volume
  SOURCES
    crop_volume.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)

mirtk_add_executable(
  mask_volume
  SOURCES
    mask_volume.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)


mirtk_add_executable(
  extract-packages
  SOURCES
    extract-packages.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)



mirtk_add_executable(
  get-t
  SOURCES
    get-t.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)


mirtk_add_executable(
  get-dt
  SOURCES
    get-dt.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
)






#mirtk_add_executable(test DEPENDS ${BASIS_PYTHON_LIBRARY_TARGET})


