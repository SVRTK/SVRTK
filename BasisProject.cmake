# ==============================================================================
# SVR : SVRTK reconstruction based on MIRTK
#
# Copyright 2013-2017 Imperial College London
# Copyright 2013-2017 Andreas Schuh
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
# ==============================================================================

################################################################################
# @file  BasisProject.cmake
# @brief Sets basic information about the MIRTK module and calls basis_project().
#
# This file defines basic information about a project by calling
# the basis_project() function. This basic information, also known as metadata,
# is used by CMake BASIS to setup the project. The dependencies to other modules
# have to be specified here such that the top-level IRTK project can analyze the
# inter-module dependencies, as well as dependencies on third-party libraries.
#
# @sa https://cmake-basis.github.io/standard/modules.html
#
# @ingroup BasisSettings
################################################################################

# Note: The #<*> dependency patterns are required by the basisproject tool and
#       should be kept on a separate line as last commented argument of the
#       corresponding options of the basis_project() command. The TEMPLATE
#       option and set argument are also required by this tool and should not
#       be changed manually. The argument is updated by basisproject --update.

basis_project (

  # ----------------------------------------------------------------------------
  # meta-data
  NAME        "SVRTK"
  VERSION     "1.0.0" # version of this module
  SOVERSION   "0"     # API yet unstable
  PACKAGE     "MIRTK"
  AUTHORS     "Maria Deprez, Alena Uus"
  DESCRIPTION "SVR Reconstruction for Medical Imaging."
  COPYRIGHT   "2013-2017 Imperial College London, 2013-2017 Andreas Schuh, 2018-2020 King's College London"
  LICENSE     "Apache License Version 2.0"
  CONTACT     "Maria Deprez <maria.murgasova@kcl.ac.uk>, Alena Uus <alena.uus@kcl.ac.uk>, Thomas Roberts <t.roberts@kcl.ac.uk>, Jo Hajnal <jo.hajnal@kcl.ac.uk>"
  TEMPLATE    "mirtk-module/1.0"

  # ----------------------------------------------------------------------------
  # dependencies
  DEPENDS
    MIRTK{Common,Numerics,Image,IO,Transformation,Registration}
    Boost{filesystem,program_options}
    Eigen3{Eigen}
  #<dependency>
  OPTIONAL_DEPENDS
    Python{Interp}
    TBB{tbb,malloc}
    #<optional-dependency>
  TOOLS_DEPENDS
    Python{Interp}
  TEST_DEPENDS
    Boost{unit_test_framework}
)
