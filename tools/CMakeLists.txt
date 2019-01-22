# ============================================================================
# SVRTK : SVR reconstruction based on MIRTK
#
# Copyright 2008-2017 Imperial College London
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

#mirtk_add_executable(test DEPENDS ${BASIS_PYTHON_LIBRARY_TARGET})



