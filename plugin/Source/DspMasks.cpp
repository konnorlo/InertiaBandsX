#include "DspMasks.h"

namespace inertia
{

void DspMasks::reset(int numClusters, int numPerceptualBins, int fftBins)
{
    const auto k = clamp(numClusters, 2, kMaxClusters);
    const auto m = clamp(numPerceptualBins, 1, kMaxPerceptualBins);
    const auto bins = clamp(fftBins, 1, kMaxFftBins);
    const auto uniform = 1.0f / static_cast<float>(k);

    for (int cluster = 0; cluster < kMaxClusters; ++cluster)
    {
        for (int i = 0; i < kMaxPerceptualBins; ++i)
            smoothedPerceptualWeights_[cluster][i] = (cluster < k && i < m) ? uniform : 0.0f;

        for (int f = 0; f < kMaxFftBins; ++f)
            fftWeights_[cluster][f] = (cluster < k && f < bins) ? uniform : 0.0f;
    }
}

void DspMasks::compute(const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                       const std::array<std::array<float, kFeatureDim>, kMaxClusters>& centers,
                       const std::array<float, kFeatureDim>& featureWeights,
                       int numClusters,
                       int numPerceptualBins,
                       const std::array<int, kMaxFftBins>& fftToPerceptual,
                       int fftBins,
                       float sigma,
                       float distancePenalty,
                       float smoothCoeff,
                       float totalEnergy)
{
    const auto k = clamp(numClusters, 2, kMaxClusters);
    const auto m = clamp(numPerceptualBins, 1, kMaxPerceptualBins);
    const auto bins = clamp(fftBins, 1, kMaxFftBins);

    const auto spread = std::max(0.05f, sigma);
    const auto invTwoSigmaSquared = 1.0f / (2.0f * spread * spread);
    const auto penaltyScale = std::max(0.0f, distancePenalty);

    const auto beta = clamp(smoothCoeff, 0.0f, 1.0f);
    const auto uniform = 1.0f / static_cast<float>(k);

    std::array<std::array<float, kMaxPerceptualBins>, kMaxClusters> newWeights {};

    if (totalEnergy < kSilenceEnergyThreshold)
    {
        for (int cluster = 0; cluster < k; ++cluster)
            for (int i = 0; i < m; ++i)
                newWeights[cluster][i] = uniform;
    }
    else
    {
        for (int i = 0; i < m; ++i)
        {
            float sum = 0.0f;

            for (int cluster = 0; cluster < k; ++cluster)
            {
                float d2 = 0.0f;
                for (int d = 0; d < kFeatureDim; ++d)
                {
                    const auto diff = (features[i][d] - centers[cluster][d]) * featureWeights[d];
                    d2 += diff * diff;
                }

                const auto distance = std::sqrt(d2 + kEpsilon);
                const auto penalty = 1.0f / (1.0f + (penaltyScale * distance));
                const auto w = std::exp(-d2 * invTwoSigmaSquared) * penalty;
                newWeights[cluster][i] = w;
                sum += w;
            }

            const auto inv = 1.0f / std::max(sum, kEpsilon);
            for (int cluster = 0; cluster < k; ++cluster)
                newWeights[cluster][i] *= inv;
        }
    }

    for (int cluster = 0; cluster < kMaxClusters; ++cluster)
    {
        for (int i = 0; i < kMaxPerceptualBins; ++i)
        {
            const auto target = (cluster < k && i < m) ? newWeights[cluster][i] : 0.0f;
            smoothedPerceptualWeights_[cluster][i] += beta * (target - smoothedPerceptualWeights_[cluster][i]);
        }
    }

    for (int f = 0; f < bins; ++f)
    {
        const auto i = clamp(fftToPerceptual[f], 0, m - 1);

        float sum = 0.0f;
        for (int cluster = 0; cluster < k; ++cluster)
        {
            auto w = clamp(smoothedPerceptualWeights_[cluster][i], 0.0f, 1.0f);
            fftWeights_[cluster][f] = w;
            sum += w;
        }

        if (sum > kEpsilon)
        {
            const auto inv = 1.0f / sum;
            for (int cluster = 0; cluster < k; ++cluster)
                fftWeights_[cluster][f] *= inv;
        }
        else
        {
            for (int cluster = 0; cluster < k; ++cluster)
                fftWeights_[cluster][f] = uniform;
        }

        for (int cluster = k; cluster < kMaxClusters; ++cluster)
            fftWeights_[cluster][f] = 0.0f;
    }

    for (int f = bins; f < kMaxFftBins; ++f)
    {
        for (int cluster = 0; cluster < kMaxClusters; ++cluster)
            fftWeights_[cluster][f] = 0.0f;
    }
}

} // namespace inertia
