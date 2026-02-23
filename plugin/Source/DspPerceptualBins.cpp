#include "DspPerceptualBins.h"

namespace inertia
{

void DspPerceptualBins::prepare(double sampleRate, int fftSize, int perceptualBins)
{
    sampleRate_ = sampleRate;
    fftSize_ = clamp(fftSize, 512, kMaxFftSize);
    fftBins_ = (fftSize_ / 2) + 1;
    numPerceptualBins_ = clamp(perceptualBins, 16, kMaxPerceptualBins);

    rebuildMapping();
    reset();
}

void DspPerceptualBins::reset()
{
    prevLogEnergy_.fill(0.0f);
    emaShortEnergy_.fill(0.0f);
    emaLongEnergy_.fill(0.0f);
    featureMean_.fill(0.0f);
    featureVar_.fill(1.0f);
    levelMean_ = 0.0f;
    levelVar_ = 1.0f;
    statsInitialised_ = false;
}

void DspPerceptualBins::rebuildMapping()
{
    std::array<int, kMaxPerceptualBins + 1> edges {};

    const auto nyquist = static_cast<float>(sampleRate_ * 0.5);
    const float minHz = 40.0f;
    const float maxHz = std::max(minHz + 1.0f, nyquist);
    const float binHz = static_cast<float>(sampleRate_ / static_cast<double>(fftSize_));
    const auto minLogHz = std::log(minHz);
    const auto maxLogHz = std::log(maxHz);
    const auto invLogRange = 1.0f / std::max(maxLogHz - minLogHz, 1.0e-5f);

    edges[0] = 0;
    edges[numPerceptualBins_] = fftBins_;

    const auto ratio = maxHz / minHz;

    for (int i = 1; i < numPerceptualBins_; ++i)
    {
        const auto t = static_cast<float>(i) / static_cast<float>(numPerceptualBins_);
        const auto f = minHz * std::pow(ratio, t);
        const auto bin = static_cast<int>(std::floor(f / std::max(binHz, 1.0e-6f)));
        edges[i] = clamp(bin, 1, fftBins_ - 1);
    }

    for (int i = 1; i <= numPerceptualBins_; ++i)
        edges[i] = std::max(edges[i], edges[i - 1] + 1);

    for (int i = numPerceptualBins_ - 1; i >= 0; --i)
        edges[i] = std::min(edges[i], edges[i + 1] - 1);

    fftToPerceptual_.fill(numPerceptualBins_ - 1);

    for (int i = 0; i < numPerceptualBins_; ++i)
    {
        binStart_[i] = edges[i];
        binEnd_[i] = edges[i + 1];

        const auto start = clamp(binStart_[i], 0, fftBins_ - 1);
        const auto end = clamp(binEnd_[i], start + 1, fftBins_);
        const auto centerBin = 0.5f * static_cast<float>(start + end - 1);
        const auto centerHz = (centerBin + 0.5f) * binHz;
        binCenterHz_[i] = centerHz;
        const auto t = (std::log(std::max(centerHz, minHz)) - minLogHz) * invLogRange;
        logHzNorm_[i] = clamp((2.0f * t) - 1.0f, -1.0f, 1.0f);

        for (int f = start; f < end; ++f)
            fftToPerceptual_[f] = i;
    }

    for (int f = fftBins_; f < kMaxFftBins; ++f)
        fftToPerceptual_[f] = numPerceptualBins_ - 1;

    for (int i = numPerceptualBins_; i < kMaxPerceptualBins; ++i)
        binCenterHz_[i] = binCenterHz_[numPerceptualBins_ - 1];
}

void DspPerceptualBins::computeFeatures(const std::array<float, kMaxFftBins>& magnitudes,
                                        int fftBins,
                                        FeatureFrame& outFeatures)
{
    const auto usedFftBins = clamp(fftBins, 1, fftBins_);

    std::array<float, kMaxPerceptualBins> aggregated {};
    std::array<float, kMaxPerceptualBins> logEnergy {};
    std::array<float, kMaxPerceptualBins> localFlux {};
    std::array<float, kMaxPerceptualBins> flatnessLog {};
    std::array<float, kMaxPerceptualBins> transientness {};
    std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins> rawFeatures {};

    float totalEnergy = 0.0f;
    float lowEnergy = 0.0f;
    float highEnergy = 0.0f;
    float frameFlux = 0.0f;
    const auto lowHighSplit = std::max(1, numPerceptualBins_ / 2);

    for (int i = 0; i < numPerceptualBins_; ++i)
    {
        const auto start = clamp(binStart_[i], 0, usedFftBins - 1);
        const auto end = clamp(binEnd_[i], start + 1, usedFftBins);
        const auto count = std::max(1, end - start);

        float sum = 0.0f;
        float sumLogMag = 0.0f;
        for (int f = start; f < end; ++f)
        {
            const auto mag = std::max(0.0f, magnitudes[f]);
            sum += mag;
            sumLogMag += std::log(kEpsilon + mag);
        }

        aggregated[i] = sum;
        totalEnergy += sum;

        if (i < lowHighSplit)
            lowEnergy += sum;
        else
            highEnergy += sum;

        logEnergy[i] = std::log(kEpsilon + sum);
        outFeatures.logEnergy[i] = logEnergy[i];

        const auto meanMag = sum / static_cast<float>(count);
        const auto geometricMag = std::exp(sumLogMag / static_cast<float>(count));
        const auto flatness = geometricMag / std::max(meanMag, kEpsilon);
        flatnessLog[i] = clamp(std::log(kEpsilon + flatness), -10.0f, 0.0f);

        constexpr float shortAlpha = 0.35f;
        constexpr float longAlpha = 0.06f;
        emaShortEnergy_[i] += shortAlpha * (sum - emaShortEnergy_[i]);
        emaLongEnergy_[i] += longAlpha * (sum - emaLongEnergy_[i]);
        transientness[i] = clamp((emaShortEnergy_[i] - emaLongEnergy_[i]) / (emaLongEnergy_[i] + 1.0e-4f), -4.0f, 4.0f);
    }

    std::array<float, kFeatureDim> frameAverages {};

    for (int i = 0; i < numPerceptualBins_; ++i)
    {
        localFlux[i] = std::abs(logEnergy[i] - prevLogEnergy_[i]);
        frameFlux += localFlux[i];
    }

    frameFlux /= static_cast<float>(numPerceptualBins_);

    const auto globalLogEnergy = std::log(kEpsilon + totalEnergy);
    const auto brightness = std::log((highEnergy + kEpsilon) / (lowEnergy + kEpsilon));
    const auto levelStd = std::sqrt(levelVar_ + 1.0e-4f);
    const auto silenceConfidence = sigmoid(2.8f * (globalLogEnergy - (levelMean_ - (0.9f * levelStd))));
    const auto gateConfidence = sigmoid(6.0f * (frameFlux - 0.25f));
    const auto activityConfidence = clamp((2.0f * ((0.65f * silenceConfidence) + (0.35f * gateConfidence))) - 1.0f, -1.0f, 1.0f);

    for (int i = 0; i < numPerceptualBins_; ++i)
    {
        const auto left = (i > 0) ? logEnergy[i - 1] : logEnergy[i];
        const auto right = (i < numPerceptualBins_ - 1) ? logEnergy[i + 1] : logEnergy[i];

        rawFeatures[i][kFeatureLogEnergy] = logEnergy[i];
        rawFeatures[i][kFeatureSlope] = (i > 0) ? (logEnergy[i] - logEnergy[i - 1]) : 0.0f;
        rawFeatures[i][kFeatureLocalFlux] = localFlux[i];
        rawFeatures[i][kFeaturePeakiness] = logEnergy[i] - 0.5f * (left + right);
        rawFeatures[i][kFeatureLogHz] = logHzNorm_[i];
        rawFeatures[i][kFeatureFlatness] = flatnessLog[i];
        rawFeatures[i][kFeatureTransientness] = transientness[i];
        rawFeatures[i][kFeatureBrightness] = brightness;
        rawFeatures[i][kFeatureSpectralFlux] = frameFlux;
        rawFeatures[i][kFeatureActivityConfidence] = activityConfidence;

        for (int d = 0; d < kFeatureDim; ++d)
            frameAverages[d] += rawFeatures[i][d];
    }

    const auto invBins = 1.0f / static_cast<float>(numPerceptualBins_);

    for (int d = 0; d < kFeatureDim; ++d)
        frameAverages[d] *= invBins;

    if (! statsInitialised_)
    {
        featureMean_ = frameAverages;
        featureVar_.fill(1.0f);
        levelMean_ = globalLogEnergy;
        levelVar_ = 1.0f;
        statsInitialised_ = true;
    }
    else
    {
        constexpr float meanAlpha = 0.08f;
        constexpr float varAlpha = 0.08f;
        constexpr float levelAlpha = 0.06f;

        for (int d = 0; d < kFeatureDim; ++d)
        {
            const auto delta = frameAverages[d] - featureMean_[d];
            featureMean_[d] += meanAlpha * delta;
            const auto varianceEstimate = delta * delta;
            featureVar_[d] += varAlpha * (varianceEstimate - featureVar_[d]);
            featureVar_[d] = std::max(featureVar_[d], 1.0e-4f);
        }

        const auto levelDelta = globalLogEnergy - levelMean_;
        levelMean_ += levelAlpha * levelDelta;
        levelVar_ += levelAlpha * ((levelDelta * levelDelta) - levelVar_);
        levelVar_ = std::max(levelVar_, 1.0e-4f);
    }

    for (int i = 0; i < numPerceptualBins_; ++i)
    {
        for (int d = 0; d < kFeatureDim; ++d)
        {
            const auto stdDev = std::sqrt(featureVar_[d] + 1.0e-4f);
            outFeatures.normalisedFeatures[i][d] = (rawFeatures[i][d] - featureMean_[d]) / std::max(stdDev, 1.0e-4f);
        }

        prevLogEnergy_[i] = logEnergy[i];
    }

    for (int i = numPerceptualBins_; i < kMaxPerceptualBins; ++i)
    {
        outFeatures.logEnergy[i] = 0.0f;
        outFeatures.normalisedFeatures[i].fill(0.0f);
    }

    outFeatures.numBins = numPerceptualBins_;
    outFeatures.dim = kFeatureDim;
    outFeatures.totalEnergy = totalEnergy;
}

} // namespace inertia
