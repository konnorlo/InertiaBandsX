#pragma once

#include "Utilities.h"

namespace inertia
{

struct FeatureFrame
{
    int numBins = 0;
    int dim = kFeatureDim;
    float totalEnergy = 0.0f;

    std::array<float, kMaxPerceptualBins> logEnergy {};
    std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins> normalisedFeatures {};
};

class DspPerceptualBins
{
public:
    void prepare(double sampleRate, int fftSize, int perceptualBins);
    void reset();

    void computeFeatures(const std::array<float, kMaxFftBins>& magnitudes,
                         int fftBins,
                         FeatureFrame& outFeatures);

    int getNumPerceptualBins() const noexcept { return numPerceptualBins_; }
    const std::array<int, kMaxFftBins>& getFftToPerceptualMap() const noexcept { return fftToPerceptual_; }
    const std::array<float, kMaxPerceptualBins>& getPerceptualCenterHz() const noexcept { return binCenterHz_; }

private:
    void rebuildMapping();

    double sampleRate_ = 44100.0;
    int fftSize_ = 2048;
    int fftBins_ = 1025;
    int numPerceptualBins_ = kMaxPerceptualBins;

    std::array<int, kMaxFftBins> fftToPerceptual_ {};
    std::array<int, kMaxPerceptualBins> binStart_ {};
    std::array<int, kMaxPerceptualBins> binEnd_ {};
    std::array<float, kMaxPerceptualBins> binCenterHz_ {};
    std::array<float, kMaxPerceptualBins> logHzNorm_ {};

    std::array<float, kMaxPerceptualBins> prevLogEnergy_ {};
    std::array<float, kMaxPerceptualBins> emaShortEnergy_ {};
    std::array<float, kMaxPerceptualBins> emaLongEnergy_ {};

    std::array<float, kFeatureDim> featureMean_ {};
    std::array<float, kFeatureDim> featureVar_ {};

    float levelMean_ = 0.0f;
    float levelVar_ = 1.0f;
    bool statsInitialised_ = false;
};

} // namespace inertia
