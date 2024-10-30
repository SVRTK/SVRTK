# ============================================================================
# SVRTK : SVR reconstruction based on MIRTK
#
# Copyright Imperial College London
# Copyright 2018 Andreas Schuh
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
# ============================================================================

##############################################################################
# @file  CMakeLists.txt
# @brief Build configuration of MIRTK SVRTK library.
##############################################################################

set(HEADERS
  ${BINARY_INCLUDE_DIR}/mirtk/SVRTKExport.h
  ../svrtk/Reconstruction.h
  ../svrtk/ReconstructionDWI.h
  ../svrtk/ReconstructionCardiac4D.h
  ../svrtk/ReconstructionqMRI.h
  ../svrtk/ReconstructionCardiacVelocity4D.h
  ../svrtk/ReconstructionFFD.h
  ../svrtk/MeanShift.h
  ../svrtk/NLDenoising.h
  ../svrtk/SphericalHarmonics.h
  ../svrtk/Parallel.h
  ../svrtk/ParallelqMRI.h
  ../svrtk/Utility.h
  ../svrtk/Dictionary.h
)

set(SOURCES
  Reconstruction.cc
  ReconstructionDWI.cc
  ReconstructionCardiac4D.cc
  ReconstructionqMRI.cc
  ReconstructionCardiacVelocity4D.cc
  ReconstructionFFD.cc
  MeanShift.cc
  NLDenoising.cc
  SphericalHarmonics.cc
  Utility.cc
  Dictionary.cc
)

set(DEPENDS
  LibCommon
  LibNumerics
  LibImage
  LibRegistration
  LibIO
  LibTransformation
  ${Boost_LIBRARIES}
  ${TBB}
  ${Eigen3_LIBRARIES}
  
)

mirtk_add_library()
