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

#include "TestCommon.h"

bool EqualFiles(const string& filename1, const string& filename2) {
    // Open files at the end
    ifstream file1(filename1, ifstream::ate | ifstream::binary);
    ifstream file2(filename2, ifstream::ate | ifstream::binary);

    // Check if files are opened
    if (!file1.is_open() || !file2.is_open())
        return false;

    // Different file size
    if (file1.tellg() != file2.tellg())
        return false;

    // Rewind
    file1.seekg(0);
    file2.seekg(0);

    return equal(istreambuf_iterator<char>(file1), istreambuf_iterator<char>(), istreambuf_iterator<char>(file2));
}
