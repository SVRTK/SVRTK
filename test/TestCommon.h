/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2021- King's College London
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

// Boost
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/results_collector.hpp>

// Standard C++
#include <fstream>
#include <filesystem>

// MIRTK
#include "mirtk/IOConfig.h"

using namespace std;
using namespace std::filesystem;
using namespace boost::unit_test;

bool EqualFiles(const string& filename1, const string& filename2);

inline static void ExitOnFailure() {
    test_case::id_t id = framework::current_test_case().p_id;
    test_results result = results_collector.results(id);
    if (!result.passed())
        exit(1);
}
