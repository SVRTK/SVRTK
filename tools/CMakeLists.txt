# ============================================================================
# SVRTK : SVR reconstruction based on MIRTK
#
# Copyright 2008-2017 Imperial College London
# Copyright 2018- King's College London
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
  guess-thickness
  SOURCES
    guess-thickness.cc
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
  guess-resolution
  SOURCES
    guess-resolution.cc
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
  extract-label-brain-lr
  SOURCES
    extract-label-brain-lr.cc
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
  label-biometry-brain
  SOURCES
    label-biometry-brain.cc
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
  replace-label-brain
  SOURCES
    replace-label-brain.cc
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
  resample-to-iso-grid
  SOURCES
    resample-to-iso-grid.cc
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
  centre-volume
  SOURCES
    centre-volume.cc
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
  pad-3d
  SOURCES
    pad-3d.cc
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
  register-landmarks
  SOURCES
    register-landmarks.cc
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
  stacks-and-masks-selection
  SOURCES
    stacks-and-masks-selection.cc
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
  stacks-selection
  SOURCES
    stacks-selection.cc
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
  transform-and-rename
  SOURCES
    transform-and-rename.cc
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
  median-average
  SOURCES
    median-average.cc
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
  extract-label
  SOURCES
    extract-label.cc
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
  FitDictionary
  SOURCES
    FitDictionary.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
    ${Eigen3_LIBRARIES}
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
  reconstructqMRI
  SOURCES
    reconstructqMRI.cc
  DEPENDS
    LibCommon
    LibNumerics
    LibImage
    LibIO
    LibRegistration
    LibTransformation
    LibSVRTK
    ${TBB}
    ${Eigen3_LIBRARIES}
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
  reconstructFFD
  SOURCES
    reconstructFFD.cc
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
  combine-patches
  SOURCES
    combine-patches.cc
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
  combine-masks
  SOURCES
    combine-masks.cc
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
  filter-image-background
  SOURCES
    filter-image-background.cc
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
  threshold-image
  SOURCES
    threshold-image.cc
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
  pad-image
  SOURCES
    pad-image.cc
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
  enlarge-image
  SOURCES
    enlarge-image.cc
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
  crop-image
  SOURCES
    crop-image.cc
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
  mask-image
  SOURCES
    mask-image.cc
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
  prepare-for-cnn
  SOURCES
    prepare-for-cnn.cc
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
  prepare-for-monai
  SOURCES
    prepare-for-monai.cc
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

mirtk_add_executable(
  image-label-statistics
  SOURCES
    image-label-statistics.cc
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
  nan
  SOURCES
    nan.cc
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
  replace-label
  SOURCES
    replace-label.cc
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
