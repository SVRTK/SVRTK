/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2021 King's College London
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "mirtk/Profiling.h"

/// Start measurement of execution time of current code block
#define SVRTK_START_TIMING    MIRTK_START_TIMING
/// Reset measurement of starting execution time of current code block
#define SVRTK_RESET_TIMING    MIRTK_RESET_TIMING

/// End measurement of execution time of current code block
#ifdef  SVRTK_TOOL
#define SVRTK_END_TIMING      if (debug || profile) MIRTK_END_TIMING
#else
#define SVRTK_END_TIMING      if (_debug || _profile) MIRTK_END_TIMING
#endif
