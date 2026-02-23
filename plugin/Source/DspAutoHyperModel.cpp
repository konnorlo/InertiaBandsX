#include "DspAutoHyperModel.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace inertia
{

void DspAutoHyperModel::prepare(double sampleRate, int fftSize, int perceptualBins)
{
    sampleRate_ = sampleRate;
    fftSize_ = clamp(fftSize, 512, kMaxFftSize);
    numBins_ = clamp(perceptualBins, 1, kMaxPerceptualBins);
    reset();
}

void DspAutoHyperModel::reset()
{
    prevLogEnergyBins_.fill(0.0f);
    prevFrameLogEnergy_ = 0.0f;
    hasPreviousFrame_ = false;

    writeIndex_ = 0;
    historyCount_ = 0;

    logEnergyHistory_.fill(0.0f);
    slopeHistory_.fill(0.0f);
    fluxHistory_.fill(0.0f);
    peakinessHistory_.fill(0.0f);
    brightnessHistory_.fill(0.0f);
    flatnessHistory_.fill(0.0f);
    transientHistory_.fill(0.0f);
    lowHighRatioHistory_.fill(0.0f);
    linearEnergyHistory_.fill(0.0f);

    latestDescriptors_.fill(0.0f);
}

template <size_t N>
float DspAutoHyperModel::percentile(const std::array<float, N>& source,
                                    int count,
                                    float percentileValue,
                                    std::array<float, N>& scratch) const noexcept
{
    const auto clampedCount = clamp(count, 1, static_cast<int>(N));

    for (int i = 0; i < clampedCount; ++i)
        scratch[static_cast<size_t>(i)] = source[static_cast<size_t>(i)];

    const auto p = clamp(percentileValue, 0.0f, 100.0f) * 0.01f;
    const auto index = clamp(static_cast<int>(std::floor((clampedCount - 1) * p)), 0, clampedCount - 1);

    auto begin = scratch.begin();
    auto nth = begin + index;
    auto end = begin + clampedCount;

    std::nth_element(begin, nth, end);
    return *nth;
}

void DspAutoHyperModel::pushHistorySample(float logEnergy,
                                          float slope,
                                          float spectralFlux,
                                          float peakiness,
                                          float brightness,
                                          float flatness,
                                          float transientness,
                                          float lowHighRatio,
                                          float linearEnergy) noexcept
{
    const auto i = static_cast<size_t>(writeIndex_);

    logEnergyHistory_[i] = logEnergy;
    slopeHistory_[i] = slope;
    fluxHistory_[i] = spectralFlux;
    peakinessHistory_[i] = peakiness;
    brightnessHistory_[i] = brightness;
    flatnessHistory_[i] = flatness;
    transientHistory_[i] = transientness;
    lowHighRatioHistory_[i] = lowHighRatio;
    linearEnergyHistory_[i] = linearEnergy;

    writeIndex_ = (writeIndex_ + 1) % kHistorySize;
    historyCount_ = std::min(historyCount_ + 1, kHistorySize);
}

void DspAutoHyperModel::pushFrame(const FeatureFrame& frame)
{
    const auto usedBins = clamp(frame.numBins, 1, numBins_);
    if (usedBins <= 0)
        return;

    std::array<float, kMaxPerceptualBins> slopeValues {};
    std::array<float, kMaxPerceptualBins> peakValues {};
    std::array<float, kMaxPerceptualBins> localScratchA {};
    std::array<float, kMaxPerceptualBins> localScratchB {};
    int slopeCount = 0;
    int peakCount = 0;

    float fluxSumSq = 0.0f;
    float totalLinear = 0.0f;
    float lowLinear = 0.0f;
    float highLinear = 0.0f;
    float sumLogLinear = 0.0f;

    const auto split = std::max(1, usedBins / 2);

    for (int i = 0; i < usedBins; ++i)
    {
        const auto logEnergy = frame.logEnergy[static_cast<size_t>(i)];

        const auto left = (i > 0) ? frame.logEnergy[static_cast<size_t>(i - 1)] : logEnergy;
        const auto right = (i < (usedBins - 1)) ? frame.logEnergy[static_cast<size_t>(i + 1)] : logEnergy;
        peakValues[static_cast<size_t>(peakCount++)] = logEnergy - (0.5f * (left + right));

        if (i > 0)
            slopeValues[static_cast<size_t>(slopeCount++)] = logEnergy - frame.logEnergy[static_cast<size_t>(i - 1)];

        const auto localFlux = hasPreviousFrame_ ? std::abs(logEnergy - prevLogEnergyBins_[static_cast<size_t>(i)]) : 0.0f;
        fluxSumSq += localFlux * localFlux;

        const auto clampedLog = clamp(logEnergy, -40.0f, 40.0f);
        const auto linear = std::exp(clampedLog);
        totalLinear += linear;
        sumLogLinear += clampedLog;

        if (i < split)
            lowLinear += linear;
        else
            highLinear += linear;

        prevLogEnergyBins_[static_cast<size_t>(i)] = logEnergy;
    }

    for (int i = usedBins; i < numBins_; ++i)
        prevLogEnergyBins_[static_cast<size_t>(i)] = 0.0f;

    const auto invBinCount = 1.0f / static_cast<float>(std::max(1, usedBins));

    const auto spectralFlux = std::sqrt(fluxSumSq * invBinCount);
    const auto slopeMedian = percentile(slopeValues, std::max(1, slopeCount), 50.0f, localScratchA);
    const auto peakMedian = percentile(peakValues, std::max(1, peakCount), 50.0f, localScratchB);

    const auto linearMean = totalLinear * invBinCount;
    const auto geometricMean = std::exp(sumLogLinear * invBinCount);
    const auto flatness = geometricMean / std::max(linearMean, kEpsilon);

    const auto frameLogEnergy = std::log(std::max(frame.totalEnergy, kEpsilon));
    const auto transientness = hasPreviousFrame_ ? std::max(frameLogEnergy - prevFrameLogEnergy_, 0.0f) : 0.0f;
    const auto brightness = highLinear / std::max(totalLinear, kEpsilon);
    const auto lowHighRatio = std::log1p(lowLinear / std::max(highLinear, kEpsilon));

    pushHistorySample(frameLogEnergy,
                      slopeMedian,
                      spectralFlux,
                      peakMedian,
                      brightness,
                      flatness,
                      transientness,
                      lowHighRatio,
                      std::max(frame.totalEnergy, 0.0f));

    prevFrameLogEnergy_ = frameLogEnergy;
    hasPreviousFrame_ = true;
}

AutoHyperPrediction DspAutoHyperModel::predict() noexcept
{
    AutoHyperPrediction fallback;

    if (historyCount_ < kMinFramesForPrediction)
        return fallback;

    const auto count = historyCount_;

    std::array<float, kAutoHyperDescriptorCount> descriptors {};
    descriptors[0] = percentile(logEnergyHistory_, count, 50.0f, scratchA_);
    descriptors[1] = percentile(logEnergyHistory_, count, 75.0f, scratchA_) - percentile(logEnergyHistory_, count, 25.0f, scratchB_);
    descriptors[2] = percentile(slopeHistory_, count, 50.0f, scratchA_);
    descriptors[3] = percentile(fluxHistory_, count, 50.0f, scratchA_);
    descriptors[4] = percentile(fluxHistory_, count, 75.0f, scratchA_) - percentile(fluxHistory_, count, 25.0f, scratchB_);
    descriptors[5] = percentile(peakinessHistory_, count, 50.0f, scratchA_);
    descriptors[6] = percentile(brightnessHistory_, count, 50.0f, scratchA_);
    descriptors[7] = percentile(flatnessHistory_, count, 50.0f, scratchA_);
    descriptors[8] = percentile(transientHistory_, count, 50.0f, scratchA_);
    descriptors[9] = percentile(lowHighRatioHistory_, count, 50.0f, scratchA_);
    descriptors[10] = percentile(logEnergyHistory_, count, 95.0f, scratchA_) - percentile(logEnergyHistory_, count, 10.0f, scratchB_);

    const auto silenceThreshold = 0.01f * percentile(linearEnergyHistory_, count, 50.0f, scratchA_);
    int silenceFrames = 0;
    for (int i = 0; i < count; ++i)
    {
        if (linearEnergyHistory_[static_cast<size_t>(i)] < silenceThreshold)
            ++silenceFrames;
    }

    descriptors[11] = static_cast<float>(silenceFrames) / static_cast<float>(std::max(1, count));

    for (auto& value : descriptors)
    {
        if (! std::isfinite(value))
            value = 0.0f;
    }

    latestDescriptors_ = descriptors;
    return evaluateModel(descriptors);
}

AutoHyperPrediction DspAutoHyperModel::evaluateModel(const std::array<float, kAutoHyperDescriptorCount>& descriptors) const noexcept
{
    AutoHyperPrediction prediction;

    std::array<float, kAutoHyperDescriptorCount> z {};
    float zNorm2 = 0.0f;
    for (int i = 0; i < kAutoHyperDescriptorCount; ++i)
    {
        const auto scale = std::max(kAutoHyperScales[static_cast<size_t>(i)], 1.0e-6f);
        z[static_cast<size_t>(i)] = (descriptors[static_cast<size_t>(i)] - kAutoHyperMeans[static_cast<size_t>(i)]) / scale;
        zNorm2 += z[static_cast<size_t>(i)] * z[static_cast<size_t>(i)];
    }

    std::array<double, kAutoHyperOutputCount> raw {};
    for (int o = 0; o < kAutoHyperOutputCount; ++o)
        raw[static_cast<size_t>(o)] = static_cast<double>(kAutoHyperIntercepts[static_cast<size_t>(o)]);

    auto accumulateTerm = [&](int termIndex, float termValue)
    {
        const auto t = static_cast<double>(termValue);
        for (int o = 0; o < kAutoHyperOutputCount; ++o)
            raw[static_cast<size_t>(o)] += static_cast<double>(kAutoHyperCoefficients[static_cast<size_t>(o)][static_cast<size_t>(termIndex)]) * t;
    };

    int termIndex = 0;

    for (int i = 0; i < kAutoHyperDescriptorCount; ++i)
        accumulateTerm(termIndex++, z[static_cast<size_t>(i)]);

    for (int i = 0; i < kAutoHyperDescriptorCount; ++i)
    {
        for (int j = i; j < kAutoHyperDescriptorCount; ++j)
            accumulateTerm(termIndex++, z[static_cast<size_t>(i)] * z[static_cast<size_t>(j)]);
    }

    for (int i = 0; i < kAutoHyperDescriptorCount; ++i)
    {
        for (int j = i; j < kAutoHyperDescriptorCount; ++j)
        {
            for (int k = j; k < kAutoHyperDescriptorCount; ++k)
                accumulateTerm(termIndex++, z[static_cast<size_t>(i)] * z[static_cast<size_t>(j)] * z[static_cast<size_t>(k)]);
        }
    }

    if (termIndex != kAutoHyperTermCount)
        return prediction;

    const auto distancePenalty = clamp(static_cast<float>(raw[0]), 0.0f, 2.5f);
    const auto normRate = clamp(static_cast<float>(raw[1]), 0.001f, 0.25f);
    const auto basePenaltyLog = clamp(static_cast<float>(raw[2]), -4.0f, 4.0f);
    const auto penaltySkew = clamp(static_cast<float>(raw[3]), -3.0f, 3.0f);
    const auto minimaxStrength = sigmoid(static_cast<float>(raw[4]));

    const auto base = std::exp(basePenaltyLog);
    const auto skew = std::tanh(penaltySkew);
    const auto lowPenalty = base * std::exp(skew);
    const auto highPenalty = 0.35f * base * std::exp(-0.5f * skew);

    prediction.distancePenalty = distancePenalty;
    prediction.adaptRate = normRate;
    prediction.lowPenalty = clamp(lowPenalty, 0.0f, 4.0f);
    prediction.highPenalty = clamp(highPenalty, 0.0f, 2.0f);
    prediction.minimaxStrength = clamp(minimaxStrength, 0.0f, 1.0f);

    const auto avgZNorm = std::sqrt(zNorm2 / static_cast<float>(kAutoHyperDescriptorCount));
    const auto ood = std::max(0.0f, avgZNorm - 1.0f);
    const auto confidence = std::exp(-0.8f * ood);
    prediction.confidence = clamp(confidence, 0.0f, 1.0f);
    prediction.valid = true;

    return prediction;
}

} // namespace inertia
