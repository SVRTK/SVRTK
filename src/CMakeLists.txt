# ============================================================================
# SVRTK : SVR reconstruction based on MIRTK
#
# Copyright Imperial College London
# Copyright 2018 Andreas Schuh
# Copyright 2018-2019 King's College London
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
  Reconstruction.h
  ReconstructionCardiac4D.h
  MeanShift.h
)

set(SOURCES
  Reconstruction.cc
  ReconstructionCardiac4D.cc
  MeanShift.cc
)

set(DEPENDS
  LibCommon
  LibNumerics
  LibImage
  LibRegistration
  LibIO
  LibTransformation
  ${TBB}
)

list(APPEND DEPENDS tbb)

mirtk_add_library()



