#include "DspPcaProjection.h"

#include <algorithm>
#include <cmath>

namespace inertia
{
namespace
{
constexpr int kPcaComponents = 3;

float dotProduct(const std::array<float, kMaxClusters>& a,
                 const std::array<float, kMaxClusters>& b,
                 int dims) noexcept
{
    float sum = 0.0f;
    for (int i = 0; i < dims; ++i)
        sum += a[i] * b[i];

    return sum;
}

void normalise(std::array<float, kMaxClusters>& v, int dims) noexcept
{
    const auto norm = std::sqrt(std::max(dotProduct(v, v, dims), kEpsilon));
    for (int i = 0; i < dims; ++i)
        v[i] /= norm;
}
} // namespace

void DspPcaProjection::computeProjection(const std::array<std::array<float, kMaxPerceptualBins>, kMaxClusters>& perceptualWeights,
                                         int numClusters,
                                         int numPerceptualBins,
                                         PcaProjectionFrame& outFrame) const noexcept
{
    outFrame = {};

    const auto dims = clamp(numClusters, 2, kMaxClusters);
    const auto points = clamp(numPerceptualBins, 1, kMaxPerceptualBins);

    outFrame.numPoints = points;
    outFrame.numDims = dims;
    outFrame.numClusters = dims;

    std::array<float, kMaxClusters> mean {};
    std::array<std::array<float, kMaxClusters>, kMaxPerceptualBins> centered {};

    for (int i = 0; i < points; ++i)
    {
        float dominantWeight = 0.0f;
        int dominantIndex = 0;

        for (int d = 0; d < dims; ++d)
        {
            const auto w = clamp(perceptualWeights[d][i], 0.0f, 1.0f);
            mean[d] += w;

            if (w >= dominantWeight)
            {
                dominantWeight = w;
                dominantIndex = d;
            }
        }

        outFrame.dominantCluster[i] = dominantIndex;
        outFrame.dominance[i] = dominantWeight;
    }

    const auto invPoints = 1.0f / static_cast<float>(points);
    for (int d = 0; d < dims; ++d)
        mean[d] *= invPoints;

    for (int i = 0; i < points; ++i)
    {
        for (int d = 0; d < dims; ++d)
            centered[i][d] = perceptualWeights[d][i] - mean[d];
    }

    std::array<std::array<float, kMaxClusters>, kMaxClusters> covariance {};

    const auto norm = 1.0f / static_cast<float>(std::max(1, points - 1));

    for (int r = 0; r < dims; ++r)
    {
        for (int c = 0; c < dims; ++c)
        {
            float value = 0.0f;
            for (int i = 0; i < points; ++i)
                value += centered[i][r] * centered[i][c];

            covariance[r][c] = value * norm;
        }
    }

    std::array<std::array<float, kMaxClusters>, kMaxClusters> deflated = covariance;
    std::array<std::array<float, kMaxClusters>, kPcaComponents> eigenVectors {};

    for (int component = 0; component < kPcaComponents; ++component)
    {
        if (component >= dims)
            break;

        std::array<float, kMaxClusters> v {};
        v[component] = 1.0f;

        for (int iter = 0; iter < 10; ++iter)
        {
            std::array<float, kMaxClusters> y {};

            for (int r = 0; r < dims; ++r)
            {
                for (int c = 0; c < dims; ++c)
                    y[r] += deflated[r][c] * v[c];
            }

            for (int prev = 0; prev < component; ++prev)
            {
                const auto projection = dotProduct(y, eigenVectors[prev], dims);
                for (int d = 0; d < dims; ++d)
                    y[d] -= projection * eigenVectors[prev][d];
            }

            const auto yNormSquared = dotProduct(y, y, dims);
            if (yNormSquared <= 1.0e-8f)
                break;

            v = y;
            normalise(v, dims);
        }

        eigenVectors[component] = v;

        std::array<float, kMaxClusters> cv {};

        for (int r = 0; r < dims; ++r)
        {
            for (int c = 0; c < dims; ++c)
                cv[r] += deflated[r][c] * v[c];
        }

        const auto eigenValue = dotProduct(v, cv, dims);

        for (int r = 0; r < dims; ++r)
        {
            for (int c = 0; c < dims; ++c)
                deflated[r][c] -= eigenValue * v[r] * v[c];
        }
    }

    for (int i = 0; i < points; ++i)
    {
        for (int component = 0; component < kPcaComponents; ++component)
        {
            float projected = 0.0f;

            if (component < dims)
            {
                for (int d = 0; d < dims; ++d)
                    projected += centered[i][d] * eigenVectors[component][d];
            }

            outFrame.points[i][component] = projected;
        }
    }

    std::array<float, kPcaComponents> maxAbs {};
    maxAbs.fill(1.0e-4f);

    for (int i = 0; i < points; ++i)
    {
        for (int component = 0; component < kPcaComponents; ++component)
            maxAbs[component] = std::max(maxAbs[component], std::abs(outFrame.points[i][component]));
    }

    for (int i = 0; i < points; ++i)
    {
        for (int component = 0; component < kPcaComponents; ++component)
            outFrame.points[i][component] = clamp(outFrame.points[i][component] / maxAbs[component], -1.0f, 1.0f);
    }

    for (int cluster = 0; cluster < dims; ++cluster)
    {
        std::array<float, 3> weightedSum {};
        float weightSum = 0.0f;

        for (int i = 0; i < points; ++i)
        {
            const auto w = clamp(perceptualWeights[cluster][i], 0.0f, 1.0f);
            weightSum += w;

            for (int component = 0; component < kPcaComponents; ++component)
                weightedSum[component] += w * outFrame.points[i][component];
        }

        outFrame.clusterActivity[cluster] = weightSum;

        if (weightSum > kEpsilon)
        {
            const auto invWeight = 1.0f / weightSum;
            for (int component = 0; component < kPcaComponents; ++component)
                outFrame.clusterCenters[cluster][component] = clamp(weightedSum[component] * invWeight, -1.0f, 1.0f);
        }
        else
        {
            outFrame.clusterCenters[cluster] = { 0.0f, 0.0f, 0.0f };
        }
    }
}

} // namespace inertia
