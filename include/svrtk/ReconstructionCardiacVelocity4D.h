/*
 * SVRTK : SVR reconstruction based on MIRTK
 *
 * Copyright 2018-2021 King's College London
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

// SVRTK
#include "svrtk/ReconstructionCardiac4D.h"

using namespace mirtk;

namespace svrtk {

    // Forward declarations
    namespace Parallel {
        class EStepCardiacVelocity4D;
        class MStepCardiacVelocity4D;
        class SimulateSlicesCardiacVelocity4D;
        class SuperresolutionCardiacVelocity4D;
    }

    class ReconstructionCardiacVelocity4D: public ReconstructionCardiac4D {
    protected:
        // Arrays of gradient moment values
        Array<Array<double>> _g_directions;
        Array<double> _g_values;

        // Array of velocity directions
        Array<Array<double>> _v_directions;

        // Reconstructed 4D cardiac cine velocity images (for X, Y and Z components)
        Array<RealImage> _reconstructed5DVelocity;
        Array<RealImage> _confidence_maps_velocity;

        double _min_phase;
        double _max_phase;

        double _min_velocity;
        double _max_velocity;

        bool _adaptive_regularisation;
        bool _limit_intensities;

        static constexpr double gamma = 2 * PI * 0.042577;

        Array<RigidTransformation> _random_transformations;
        Array<Array<double>> _slice_g_directions;
        Array<Array<RealImage>> _simulated_velocities;

    public:
        /// ReconstructionCardiacVelocity4D constructor
        ReconstructionCardiacVelocity4D();
        /// ReconstructionCardiacVelocity4D destructor
        inline ~ReconstructionCardiacVelocity4D() {}

        /// Initialisation of slice gradients with respect to slice transformations
        void InitializeSliceGradients4D();

        /// Simulation of phase slices from velocity volumes
        void SimulateSlicesCardiacVelocity4D();

        /// Gradient descend step of velocity estimation
        void SuperresolutionCardiacVelocity4D(int iter);

        /// Adaptive regularization
        void AdaptiveRegularizationCardiacVelocity4D(int iter, Array<RealImage>& originals);

        /// Saving slice info int .csv
        void SaveSliceInfo();

        /// Save output files (simulated velocity and phase)
        void SaveOuput(Array<RealImage> stacks);

        /// Save reconstructed velocity volumes
        void SaveReconstructedVelocity4D(int iter);

        /// Mask phase slices
        void MaskSlicesPhase();

        /// Check reconstruction quality
        double Consistency();

        /// Initialisation of EM step
        void InitializeEMVelocity4D();

        /// Initialisation of EM values
        void InitializeEMValuesVelocity4D();

        /// Initialisation of robust statistics
        void InitializeRobustStatisticsVelocity4D();

        /// E-step (todo: optimise for velocity)
        void EStepVelocity4D();

        /// M-step (todo: optimise for velocity)
        void MStepVelocity4D(int iter);

        friend class Parallel::EStepCardiacVelocity4D;
        friend class Parallel::MStepCardiacVelocity4D;
        friend class Parallel::SimulateSlicesCardiacVelocity4D;
        friend class Parallel::SuperresolutionCardiacVelocity4D;

        ////////////////////////////////////////////////////////////////////////////////
        // Inline/template definitions
        ////////////////////////////////////////////////////////////////////////////////

        /// Initialisation of velocity volumes
        inline void InitializeVelocityVolumes() {
            _reconstructed4D = 0;
            _reconstructed5DVelocity = Array<RealImage>(_v_directions.size(), _reconstructed4D);
            _confidence_maps_velocity = Array<RealImage>(_v_directions.size(), _reconstructed4D);
        }

        /// Read gradient moment values
        inline void InitializeGradientMoments(const Array<Array<double>>& g_directions, const Array<double>& g_values) {
            _g_directions = g_directions;
            _g_values = g_values;
        }

        /// Masking reconstructed volume
        inline void StaticMaskReconstructedVolume5D() {
            for (size_t v = 0; v < _reconstructed5DVelocity.size(); v++)
                StaticMaskVolume4D(_reconstructed5DVelocity[v], _mask, 0);
        }

        /// Set flag for cropping temporal PSF
        inline void LimitTimeWindow() {
            _no_ts = true;
        }

        /// Set alpha for gradient descent
        inline void SetAlpha(double alpha) {
            _alpha = alpha;
        }

        /// Set adaptive regularisation flag
        inline void SetAdaptiveRegularisation(bool flag) {
            _adaptive_regularisation = flag;
        }

        /// Set flag for limiting intensities
        inline void SetLimitIntensities(bool flag) {
            _limit_intensities = flag;
        }

        /// Compute and set velocity limits
        inline void InitialiseVelocityLimits() {
            _min_phase = -3.14;
            _max_phase = 3.14;

            int templateNumber = 1;

            _min_velocity = _min_phase / (_g_values[templateNumber] * gamma);
            _max_velocity = _max_phase / (_g_values[templateNumber] * gamma);

            if (_debug)
                cout << " - velocity limits : [ " << _min_velocity << " ; " << _max_velocity << " ] " << endl;
        }

    };  // end of ReconstructionCardiacVelocity4D class definition

} // namespace svrtk
