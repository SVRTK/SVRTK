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
#define BOOST_TEST_MODULE testReconstruction

// SVRTK
#include "TestCommon.h"
#include "svrtk/Reconstruction.h"

using namespace svrtk;

Reconstruction reconstructor;
string stacksFolder, masksFolder, expectedFolder;
Array<RealImage> stacks;
RealImage mask, templateStack, maskedTemplate;
Array<RigidTransformation> stackTransformations;

BOOST_AUTO_TEST_CASE(InitialiseTest) {
    InitializeIOLibrary();
    reconstructor.DebugOn();

    BOOST_CHECK_MESSAGE(framework::master_test_suite().argc == 2, "Please provide test data folder!");
    ExitOnFailure();

    const string dataFolder = framework::master_test_suite().argv[1];

    stacksFolder = dataFolder + "/stacks/";
    BOOST_CHECK_MESSAGE(is_directory(stacksFolder), "Stacks folder doesn't exist!");
    ExitOnFailure();

    masksFolder = dataFolder + "/masks/";
    BOOST_CHECK_MESSAGE(is_directory(masksFolder), "Masks folder doesn't exist!");
    ExitOnFailure();

    expectedFolder = dataFolder + "/expected/reconstruction/";
    BOOST_CHECK_MESSAGE(is_directory(expectedFolder), "Expected folder doesn't exist!");
    ExitOnFailure();
}

BOOST_AUTO_TEST_CASE(ReadStacks) {
    const array<string, 6> stackFiles {
        stacksFolder + "simulated-stack-d0.nii.gz",
        stacksFolder + "simulated-stack-d1.nii.gz",
        stacksFolder + "simulated-stack-d2.nii.gz",
        stacksFolder + "simulated-stack-d3.nii.gz",
        stacksFolder + "simulated-stack-d4.nii.gz",
        stacksFolder + "simulated-stack-d5.nii.gz"
    };

    // Read input stacks
    for (size_t i = 0; i < stackFiles.size(); i++) {
        RealImage stack(stackFiles[i].c_str());
        double smin, smax;
        stack.GetMinMax(&smin, &smax);
        if (smin < 0 || smax < 0)
            stack.PutMinMaxAsDouble(0, 1000);
        stacks.push_back(move(stack));
    }

    BOOST_CHECK_MESSAGE(stackFiles.size() == stacks.size(), "Stacks couldn't be read!");
    ExitOnFailure();

    templateStack = stacks[2];
    reconstructor.SetTemplateFlag(true);
}

BOOST_AUTO_TEST_CASE(ReadMask) {
    mask.Read((masksFolder + "mask-for-stack-2.nii.gz").c_str());

    RealImage m = mask;
    TransformMask(templateStack, m, RigidTransformation());

    // Crop template stack and prepare template for global volumetric registration
    maskedTemplate = templateStack * m;
    CropImage(maskedTemplate, m);
    CropImage(templateStack, m);

    // maskedTemplate.Write((expectedFolder + "maskedTemplate.nii.gz").c_str());

    RealImage maskExpected((expectedFolder + "maskedTemplate.nii.gz").c_str());
    BOOST_CHECK(maskedTemplate == maskExpected);
    ExitOnFailure();
}

BOOST_AUTO_TEST_CASE(CreateTemplate) {
    constexpr double resolutionExpected = 0.75;
    double resolution = reconstructor.CreateTemplate(maskedTemplate, 0.75);
    BOOST_CHECK(resolution == resolutionExpected);
    ExitOnFailure();

    RealImage reconstructedExpected((expectedFolder + "reconstructedTemplate.nii.gz").c_str());
    BOOST_CHECK(reconstructor.GetReconstructed() == reconstructedExpected);
    ExitOnFailure();
}

BOOST_AUTO_TEST_CASE(SetMask) {
    reconstructor.SetMask(&mask, 4);

    RealImage maskExpected((expectedFolder + "mask.nii.gz").c_str());
    BOOST_CHECK(reconstructor.GetMask() == maskExpected);
    ExitOnFailure();
}

BOOST_AUTO_TEST_CASE(StackRegistrations) {
    stackTransformations = Array<RigidTransformation>(stacks.size());
    reconstructor.StackRegistrations(stacks, stackTransformations, 0);

    auto formatter = boost::format("./global-transformation%1%.dof");
    auto formatterExpected = boost::format(expectedFolder + "global-transformation%1%.dof");
    for (size_t i = 0; i < stackTransformations.size(); i++) {
        BOOST_CHECK_MESSAGE(EqualFiles((formatter % i).str(), (formatterExpected % i).str()), "Stack transformation #" << i << " doesn't match!");
        ExitOnFailure();
    }
}
