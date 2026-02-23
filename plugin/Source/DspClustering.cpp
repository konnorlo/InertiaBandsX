#include "DspClustering.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace inertia
{
namespace
{
void normaliseFeatureWeights(std::array<float, kFeatureDim>& weights)
{
    float norm2 = 0.0f;

    for (auto& w : weights)
    {
        w = std::max(0.001f, w);
        norm2 += w * w;
    }

    const auto norm = std::sqrt(std::max(norm2, kEpsilon));

    if (norm <= kEpsilon)
    {
        const auto uniform = 1.0f / std::sqrt(static_cast<float>(kFeatureDim));
        for (auto& w : weights)
            w = uniform;

        return;
    }

    const auto invNorm = 1.0f / norm;
    for (auto& w : weights)
        w *= invNorm;
}
} // namespace

void DspClustering::reset(int numClusters, int numPerceptualBins)
{
    numClusters_ = clamp(numClusters, 2, kMaxClusters);
    numPerceptualBins_ = clamp(numPerceptualBins, 1, kMaxPerceptualBins);
    initialised_ = false;

    for (auto& center : currentCenters_)
        center.fill(0.0f);

    for (auto& center : targetCenters_)
        center.fill(0.0f);

    assignments_.fill(0);
    previousAssignments_.fill(0);

    const auto uniform = 1.0f / std::sqrt(static_cast<float>(kFeatureDim));
    featureWeights_.fill(uniform);
    manualWeights_.fill(uniform);

    adaptiveWeightsEnabled_ = false;
    adaptRate_ = 0.25f;
    lowPenalty_ = 0.35f;
    highPenalty_ = 0.08f;
    normPenalty_ = 0.4f;
    minimaxBlend_ = 0.35f;
    priorPenalty_ = 0.15f;

    semanticFrequencyPenalty_ = 0.08f;
    continuityPenalty_ = 0.04f;
    minClusterMassFraction_ = 0.05f;
}

void DspClustering::setFeatureWeightConfig(const std::array<float, kFeatureDim>& manualWeights,
                                           bool adaptiveEnabled,
                                           float adaptRate,
                                           float lowPenalty,
                                           float highPenalty,
                                           float normPenalty,
                                           float minimaxBlend,
                                           float priorPenalty)
{
    manualWeights_ = manualWeights;
    normaliseFeatureWeights(manualWeights_);

    adaptiveWeightsEnabled_ = adaptiveEnabled;
    adaptRate_ = clamp(adaptRate, 0.0f, 1.0f);
    lowPenalty_ = clamp(lowPenalty, 0.0f, 4.0f);
    highPenalty_ = clamp(highPenalty, 0.0f, 2.0f);
    normPenalty_ = clamp(normPenalty, 0.0f, 4.0f);
    minimaxBlend_ = clamp(minimaxBlend, 0.0f, 1.0f);
    priorPenalty_ = clamp(priorPenalty, 0.0f, 2.0f);

    semanticFrequencyPenalty_ = 0.03f + (0.18f * minimaxBlend_);
    continuityPenalty_ = 0.01f + (0.06f * priorPenalty_);
    minClusterMassFraction_ = clamp(0.03f + (0.04f * minimaxBlend_), 0.03f, 0.12f);

    if (! adaptiveWeightsEnabled_)
    {
        featureWeights_ = manualWeights_;
        return;
    }

    for (int d = 0; d < kFeatureDim; ++d)
        featureWeights_[d] += 0.05f * (manualWeights_[d] - featureWeights_[d]);

    normaliseFeatureWeights(featureWeights_);
}

float DspClustering::squaredDistance(const std::array<float, kFeatureDim>& a,
                                     const std::array<float, kFeatureDim>& b) const
{
    float distance = 0.0f;

    for (int d = 0; d < kFeatureDim; ++d)
    {
        const auto weightedDiff = featureWeights_[d] * (a[d] - b[d]);
        distance += weightedDiff * weightedDiff;
    }

    return distance;
}

void DspClustering::reorderCentersBySemantics()
{
    std::array<int, kMaxClusters> order {};
    std::array<int, kMaxClusters> inverseOrder {};
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.begin() + numClusters_, [&](int lhs, int rhs)
    {
        const auto lhsHz = targetCenters_[lhs][kFeatureLogHz];
        const auto rhsHz = targetCenters_[rhs][kFeatureLogHz];

        if (lhsHz != rhsHz)
            return lhsHz < rhsHz;

        const auto lhsTonal = targetCenters_[lhs][kFeaturePeakiness];
        const auto rhsTonal = targetCenters_[rhs][kFeaturePeakiness];
        if (lhsTonal != rhsTonal)
            return lhsTonal > rhsTonal;

        const auto lhsTransient = targetCenters_[lhs][kFeatureTransientness];
        const auto rhsTransient = targetCenters_[rhs][kFeatureTransientness];
        return lhsTransient > rhsTransient;
    });

    std::array<std::array<float, kFeatureDim>, kMaxClusters> reorderedCurrent {};
    std::array<std::array<float, kFeatureDim>, kMaxClusters> reorderedTarget {};

    for (int c = 0; c < numClusters_; ++c)
    {
        inverseOrder[order[c]] = c;
        reorderedCurrent[c] = currentCenters_[order[c]];
        reorderedTarget[c] = targetCenters_[order[c]];
    }

    for (int c = 0; c < numClusters_; ++c)
    {
        currentCenters_[c] = reorderedCurrent[c];
        targetCenters_[c] = reorderedTarget[c];
    }

    for (int i = 0; i < numPerceptualBins_; ++i)
    {
        assignments_[i] = inverseOrder[clamp(assignments_[i], 0, numClusters_ - 1)];
        previousAssignments_[i] = inverseOrder[clamp(previousAssignments_[i], 0, numClusters_ - 1)];
    }
}

void DspClustering::deterministicSeedFromFeatures(const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                                                  int numFeatureBins,
                                                  std::array<std::array<float, kFeatureDim>, kMaxClusters>& destination,
                                                  int startCluster,
                                                  int endCluster)
{
    const auto bins = std::max(1, numFeatureBins);

    for (int cluster = startCluster; cluster < endCluster; ++cluster)
    {
        const auto index = ((cluster + 1) * bins) / (numClusters_ + 1);
        const auto featureIndex = clamp(index, 0, bins - 1);
        destination[cluster] = features[featureIndex];
    }
}

void DspClustering::initialiseFromFeatures(const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                                           int numFeatureBins)
{
    deterministicSeedFromFeatures(features, numFeatureBins, targetCenters_, 0, numClusters_);
    currentCenters_ = targetCenters_;
    reorderCentersBySemantics();
    initialised_ = true;
}

void DspClustering::setClusterCount(int newClusterCount,
                                    const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                                    int numFeatureBins)
{
    const auto clampedCount = clamp(newClusterCount, 2, kMaxClusters);

    if (clampedCount == numClusters_)
        return;

    const auto previousCount = numClusters_;
    numClusters_ = clampedCount;

    if (! initialised_)
        return;

    if (numClusters_ > previousCount)
    {
        deterministicSeedFromFeatures(features, numFeatureBins, targetCenters_, previousCount, numClusters_);
        for (int i = previousCount; i < numClusters_; ++i)
            currentCenters_[i] = targetCenters_[i];

        reorderCentersBySemantics();
    }
}

void DspClustering::updateFeatureWeights(const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                                         int bins)
{
    if (! adaptiveWeightsEnabled_)
    {
        featureWeights_ = manualWeights_;
        return;
    }

    const auto usedBins = clamp(bins, 1, numPerceptualBins_);

    std::array<float, kFeatureDim> featureError {};
    std::array<std::array<float, kFeatureDim>, kMaxClusters> clusterFeatureError {};
    std::array<float, kMaxClusters> clusterError {};
    std::array<int, kMaxClusters> clusterCount {};

    for (int i = 0; i < usedBins; ++i)
    {
        const auto c = clamp(assignments_[i], 0, numClusters_ - 1);
        ++clusterCount[c];

        for (int d = 0; d < kFeatureDim; ++d)
        {
            const auto diff = features[i][d] - workingCenters_[c][d];
            const auto e = diff * diff;

            featureError[d] += e;
            clusterFeatureError[c][d] += e;
            clusterError[c] += e;
        }
    }

    auto worstCluster = 0;
    auto worstScore = -1.0f;

    for (int c = 0; c < numClusters_; ++c)
    {
        if (clusterCount[c] <= 0)
            continue;

        const auto score = clusterError[c] / static_cast<float>(clusterCount[c]);
        if (score > worstScore)
        {
            worstScore = score;
            worstCluster = c;
        }
    }

    const auto invBins = 1.0f / static_cast<float>(std::max(1, usedBins));
    const auto invWorstCount = 1.0f / static_cast<float>(std::max(1, clusterCount[worstCluster]));

    float norm2 = 0.0f;
    for (const auto w : featureWeights_)
        norm2 += w * w;

    const auto learningRate = adaptRate_ * 0.04f;
    if (learningRate <= 0.0f)
        return;

    auto candidate = featureWeights_;

    for (int d = 0; d < kFeatureDim; ++d)
    {
        const auto w = featureWeights_[d];

        const auto meanError = featureError[d] * invBins;
        const auto maxClusterError = clusterFeatureError[worstCluster][d] * invWorstCount;

        const auto gradData = 2.0f * w * (meanError + (minimaxBlend_ * maxClusterError));
        const auto gradLow = -lowPenalty_ / ((w + 0.02f) * (w + 0.02f));
        const auto gradHigh = 2.0f * highPenalty_ * w;
        const auto gradPrior = 2.0f * priorPenalty_ * (w - manualWeights_[d]);
        const auto gradNorm = 4.0f * normPenalty_ * (norm2 - 1.0f) * w;

        const auto gradient = gradData + gradLow + gradHigh + gradPrior + gradNorm;
        candidate[d] = clamp(w - (learningRate * gradient), 0.005f, 2.0f);
    }

    normaliseFeatureWeights(candidate);

    constexpr float blend = 0.2f;
    for (int d = 0; d < kFeatureDim; ++d)
        featureWeights_[d] += blend * (candidate[d] - featureWeights_[d]);

    normaliseFeatureWeights(featureWeights_);
}

void DspClustering::updateTargets(const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                                  int numFeatureBins,
                                  int maxIterations)
{
    const auto bins = clamp(numFeatureBins, 1, numPerceptualBins_);

    if (! initialised_)
        initialiseFromFeatures(features, bins);

    workingCenters_ = targetCenters_;

    std::array<std::array<float, kFeatureDim>, kMaxClusters> sum {};
    std::array<int, kMaxClusters> count {};

    const auto iterations = clamp(maxIterations, 1, 8);

    for (int iter = 0; iter < iterations; ++iter)
    {
        for (auto& s : sum)
            s.fill(0.0f);

        count.fill(0);

        for (int i = 0; i < bins; ++i)
        {
            float bestDistance = std::numeric_limits<float>::max();
            int bestCluster = 0;

            for (int c = 0; c < numClusters_; ++c)
            {
                const auto d2 = squaredDistance(features[i], workingCenters_[c]);

                const auto expectedBin = ((static_cast<float>(c) + 0.5f) / static_cast<float>(numClusters_)) * static_cast<float>(bins);
                const auto semanticDelta = (static_cast<float>(i) - expectedBin) / static_cast<float>(std::max(1, bins));
                const auto semanticPenalty = semanticFrequencyPenalty_ * semanticDelta * semanticDelta;

                const auto continuityPenalty = (previousAssignments_[i] == c) ? 0.0f : continuityPenalty_;
                const auto objective = d2 + semanticPenalty + continuityPenalty;

                if (objective < bestDistance)
                {
                    bestDistance = objective;
                    bestCluster = c;
                }
            }

            assignments_[i] = bestCluster;
            ++count[bestCluster];

            for (int d = 0; d < kFeatureDim; ++d)
                sum[bestCluster][d] += features[i][d];
        }

        for (int c = 0; c < numClusters_; ++c)
        {
            if (count[c] <= 0)
                continue;

            const auto invCount = 1.0f / static_cast<float>(count[c]);
            for (int d = 0; d < kFeatureDim; ++d)
                workingCenters_[c][d] = sum[c][d] * invCount;
        }

        const auto minClusterBins = std::max(1, static_cast<int>(std::floor(minClusterMassFraction_ * static_cast<float>(bins))));
        for (int c = 0; c < numClusters_; ++c)
        {
            if (count[c] >= minClusterBins)
                continue;

            float farthestDistance = -1.0f;
            int farthestIndex = 0;

            for (int i = 0; i < bins; ++i)
            {
                const auto assigned = assignments_[i];
                const auto d2 = squaredDistance(features[i], workingCenters_[assigned]);
                if (d2 > farthestDistance)
                {
                    farthestDistance = d2;
                    farthestIndex = i;
                }
            }

            workingCenters_[c] = features[farthestIndex];
        }
    }

    std::array<bool, kMaxClusters> newCenterUsed {};
    std::array<std::array<float, kFeatureDim>, kMaxClusters> matchedCenters {};

    for (int oldIndex = 0; oldIndex < numClusters_; ++oldIndex)
    {
        float bestDistance = std::numeric_limits<float>::max();
        int bestNewIndex = 0;

        for (int newIndex = 0; newIndex < numClusters_; ++newIndex)
        {
            if (newCenterUsed[newIndex])
                continue;

            const auto d2 = squaredDistance(targetCenters_[oldIndex], workingCenters_[newIndex]);
            if (d2 < bestDistance)
            {
                bestDistance = d2;
                bestNewIndex = newIndex;
            }
        }

        newCenterUsed[bestNewIndex] = true;
        matchedCenters[oldIndex] = workingCenters_[bestNewIndex];
    }

    targetCenters_ = matchedCenters;
    reorderCentersBySemantics();
    previousAssignments_ = assignments_;
    updateFeatureWeights(features, bins);
}

void DspClustering::advanceInertia(float alpha)
{
    if (! initialised_)
        return;

    const auto a = clamp(alpha, 0.0f, 1.0f);

    for (int c = 0; c < numClusters_; ++c)
    {
        for (int d = 0; d < kFeatureDim; ++d)
            currentCenters_[c][d] += a * (targetCenters_[c][d] - currentCenters_[c][d]);
    }
}

} // namespace inertia
