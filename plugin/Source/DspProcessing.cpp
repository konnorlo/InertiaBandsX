#include "DspProcessing.h"

namespace inertia
{

void DspProcessing::prepare(double sampleRate, int fftSize)
{
    fftSize_ = clamp(fftSize, 512, kMaxFftSize);
    fftBins_ = (fftSize_ / 2) + 1;

    const auto hzPerBin = static_cast<float>(sampleRate / static_cast<double>(fftSize_));

    for (int f = 0; f < fftBins_; ++f)
        binFrequencyHz_[f] = hzPerBin * static_cast<float>(f);

    for (int f = fftBins_; f < kMaxFftBins; ++f)
        binFrequencyHz_[f] = binFrequencyHz_[fftBins_ - 1];

    autoLevelCompensation_ = 1.0f;
}

void DspProcessing::computeBinMultipliers(const std::array<float, kMaxFftBins>& magnitudes,
                                          const std::array<std::array<float, kMaxFftBins>, kMaxClusters>& fftWeights,
                                          const ProcessingParams& params,
                                          std::array<float, kMaxFftBins>& outMultipliers,
                                          std::array<float, kMaxClusters>& outClusterLevels,
                                          std::array<std::array<float, kMaxFftBins>, kMaxClusters>* outClusterWetMultipliers)
{
    const auto bins = clamp(params.fftBins, 1, fftBins_);
    const auto k = clamp(params.numClusters, 2, kMaxClusters);
    const auto selectedCluster = (params.outputMode > 0) ? clamp(params.outputMode - 1, 0, k - 1) : -1;

    if (outClusterWetMultipliers != nullptr)
    {
        for (auto& cluster : *outClusterWetMultipliers)
            cluster.fill(0.0f);
    }

    std::array<float, kMaxClusters> gateByCluster {};
    std::array<float, kMaxClusters> clusterReferenceMag {};
    std::array<float, kMaxFftBins> wetMultipliers {};

    for (int cluster = 0; cluster < k; ++cluster)
    {
        float weightedLevel = 0.0f;
        float weightSum = 0.0f;

        for (int f = 0; f < bins; ++f)
        {
            const auto w = fftWeights[cluster][f];
            weightedLevel += w * std::log(kEpsilon + magnitudes[f]);
            weightSum += w;
        }

        const auto level = weightedLevel / std::max(weightSum, kEpsilon);
        outClusterLevels[cluster] = level;
        clusterReferenceMag[cluster] = std::max(std::exp(level), 1.0e-4f);

        if (params.gateEnabled)
        {
            const auto gateCore = sigmoid(params.gateSharpness * (level - params.gateThreshold));
            gateByCluster[cluster] = params.gateFloor + (1.0f - params.gateFloor) * gateCore;
        }
        else
        {
            gateByCluster[cluster] = 1.0f;
        }
    }

    for (int cluster = k; cluster < kMaxClusters; ++cluster)
    {
        outClusterLevels[cluster] = -12.0f;
        gateByCluster[cluster] = 0.0f;
        clusterReferenceMag[cluster] = 1.0e-4f;
    }

    const auto mix = clamp(params.globalMix, 0.0f, 1.0f);
    const auto autoCompEnabled = params.autoLevelCompensate && (! params.gateEnabled);

    float inputPower = 0.0f;
    float sumA2 = 0.0f;
    float sumAB = 0.0f;
    float sumB2 = 0.0f;

    for (int f = 0; f < bins; ++f)
    {
        const auto freq = std::max(binFrequencyHz_[f], 0.0f);
        const auto freq2 = freq * freq;
        const auto magnitude = std::max(0.0f, magnitudes[f]);

        float weightedMultiplier = 0.0f;
        float weightSum = 0.0f;

        for (int cluster = 0; cluster < k; ++cluster)
        {
            const auto& cp = params.clusterParams[cluster];
            if (cp.muted)
            {
                if (outClusterWetMultipliers != nullptr)
                    (*outClusterWetMultipliers)[cluster][f] = 0.0f;
                continue;
            }

            const auto hp2 = cp.hpHz * cp.hpHz;
            const auto lp2 = cp.lpHz * cp.lpHz;

            const auto hpBase = freq2 / (freq2 + hp2 + kEpsilon);
            const auto lpBase = lp2 / (freq2 + lp2 + kEpsilon);
            const auto hp = hpBase * hpBase;
            const auto lp = lpBase * lpBase;

            const auto levelNormalised = clamp((outClusterLevels[cluster] + 8.0f) * 0.15f, -1.0f, 1.0f);
            const auto levelDrive = params.driveFromLevel * std::max(levelNormalised, 0.0f);
            const auto drive = clamp((2.5f * params.saturationAmount) + (2.0f * levelDrive), 0.0f, 12.0f);

            const auto normalisedMag = magnitude / clusterReferenceMag[cluster];
            const auto compression = 1.0f / (1.0f + drive * normalisedMag + kEpsilon);
            const auto makeup = std::sqrt(1.0f + (0.3f * drive));
            const auto saturationCurve = clamp(makeup * compression, 0.2f, 3.0f);
            auto clusterMultiplier = cp.gainLinear * hp * lp * saturationCurve;

            if (params.gateEnabled)
                clusterMultiplier *= gateByCluster[cluster];

            clusterMultiplier = clamp(clusterMultiplier, 0.0f, 20.0f);

            if (outClusterWetMultipliers != nullptr)
                (*outClusterWetMultipliers)[cluster][f] = clusterMultiplier * params.outputGainLinear;

            const auto includeInMain = (selectedCluster >= 0)
                ? (cluster == selectedCluster)
                : (cp.outputRoute == 0);

            if (! includeInMain)
                continue;

            const auto w = fftWeights[cluster][f];
            if (w <= 0.0f)
                continue;

            weightedMultiplier += w * clusterMultiplier;
            weightSum += w;
        }

        if (weightSum <= kEpsilon)
        {
            wetMultipliers[f] = 0.0f;
            continue;
        }

        auto totalMultiplier = weightedMultiplier / weightSum;
        totalMultiplier = clamp(totalMultiplier, 0.0f, 20.0f);
        wetMultipliers[f] = totalMultiplier * params.outputGainLinear;

        if (autoCompEnabled)
        {
            const auto dryPart = magnitude * (1.0f - mix);
            const auto wetPart = magnitude * mix * wetMultipliers[f];

            inputPower += magnitude * magnitude;
            sumA2 += dryPart * dryPart;
            sumAB += dryPart * wetPart;
            sumB2 += wetPart * wetPart;
        }
    }

    float targetCompensation = 1.0f;
    if (autoCompEnabled && inputPower > kEpsilon && sumB2 > kEpsilon)
    {
        const auto A = sumB2;
        const auto B = 2.0f * sumAB;
        const auto C = sumA2 - inputPower;
        const auto discriminant = std::max((B * B) - (4.0f * A * C), 0.0f);
        const auto sqrtDisc = std::sqrt(discriminant);

        const auto r1 = (-B + sqrtDisc) / (2.0f * A);
        const auto r2 = (-B - sqrtDisc) / (2.0f * A);

        targetCompensation = (r1 > 0.0f) ? r1 : ((r2 > 0.0f) ? r2 : 1.0f);
        targetCompensation = clamp(targetCompensation, 0.25f, 4.0f);
    }

    const auto smoothing = autoCompEnabled ? 0.12f : 0.20f;
    const auto destination = autoCompEnabled ? targetCompensation : 1.0f;
    autoLevelCompensation_ += smoothing * (destination - autoLevelCompensation_);
    const auto compensation = clamp(autoLevelCompensation_, 0.25f, 4.0f);

    for (int f = 0; f < bins; ++f)
    {
        const auto wet = wetMultipliers[f] * compensation;
        outMultipliers[f] = clamp((1.0f - mix) + mix * wet, 0.0f, 20.0f);
    }

    for (int f = bins; f < kMaxFftBins; ++f)
        outMultipliers[f] = 1.0f;
}

} // namespace inertia
