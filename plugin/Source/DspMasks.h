#pragma once

#include "Utilities.h"

namespace inertia
{

class DspMasks
{
public:
    void reset(int numClusters, int numPerceptualBins, int fftBins);

    void compute(const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                 const std::array<std::array<float, kFeatureDim>, kMaxClusters>& centers,
                 const std::array<float, kFeatureDim>& featureWeights,
                 int numClusters,
                 int numPerceptualBins,
                 const std::array<int, kMaxFftBins>& fftToPerceptual,
                 int fftBins,
                 float sigma,
                 float distancePenalty,
                 float smoothCoeff,
                 float totalEnergy);

    const std::array<std::array<float, kMaxPerceptualBins>, kMaxClusters>& getPerceptualWeights() const noexcept { return smoothedPerceptualWeights_; }
    const std::array<std::array<float, kMaxFftBins>, kMaxClusters>& getFftWeights() const noexcept { return fftWeights_; }

private:
    std::array<std::array<float, kMaxPerceptualBins>, kMaxClusters> smoothedPerceptualWeights_ {};
    std::array<std::array<float, kMaxFftBins>, kMaxClusters> fftWeights_ {};
};

} // namespace inertia
