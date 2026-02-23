#pragma once

#include "AutoHyperModelData.h"
#include "DspPerceptualBins.h"
#include "Utilities.h"

namespace inertia
{

struct AutoHyperPrediction
{
    float distancePenalty = 1.0f;
    float adaptRate = 0.25f;
    float lowPenalty = 0.35f;
    float highPenalty = 0.08f;
    float minimaxStrength = 0.35f;
    float confidence = 0.0f;
    bool valid = false;
};

class DspAutoHyperModel
{
public:
    void prepare(double sampleRate, int fftSize, int perceptualBins);
    void reset();

    void pushFrame(const FeatureFrame& frame);
    AutoHyperPrediction predict() noexcept;

private:
    static constexpr int kHistorySize = 256;
    static constexpr int kMinFramesForPrediction = 24;

    template <size_t N>
    float percentile(const std::array<float, N>& source,
                     int count,
                     float percentile,
                     std::array<float, N>& scratch) const noexcept;

    void pushHistorySample(float logEnergy,
                           float slope,
                           float spectralFlux,
                           float peakiness,
                           float brightness,
                           float flatness,
                           float transientness,
                           float lowHighRatio,
                           float linearEnergy) noexcept;

    AutoHyperPrediction evaluateModel(const std::array<float, kAutoHyperDescriptorCount>& descriptors) const noexcept;

    double sampleRate_ = 44100.0;
    int fftSize_ = kMaxFftSize;
    int numBins_ = kMaxPerceptualBins;

    std::array<float, kMaxPerceptualBins> prevLogEnergyBins_ {};
    float prevFrameLogEnergy_ = 0.0f;
    bool hasPreviousFrame_ = false;

    int writeIndex_ = 0;
    int historyCount_ = 0;

    std::array<float, kHistorySize> logEnergyHistory_ {};
    std::array<float, kHistorySize> slopeHistory_ {};
    std::array<float, kHistorySize> fluxHistory_ {};
    std::array<float, kHistorySize> peakinessHistory_ {};
    std::array<float, kHistorySize> brightnessHistory_ {};
    std::array<float, kHistorySize> flatnessHistory_ {};
    std::array<float, kHistorySize> transientHistory_ {};
    std::array<float, kHistorySize> lowHighRatioHistory_ {};
    std::array<float, kHistorySize> linearEnergyHistory_ {};

    std::array<float, kHistorySize> scratchA_ {};
    std::array<float, kHistorySize> scratchB_ {};

    std::array<float, kAutoHyperDescriptorCount> latestDescriptors_ {};
};

} // namespace inertia
