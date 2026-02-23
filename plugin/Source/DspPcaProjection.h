#pragma once

#include "Utilities.h"

namespace inertia
{

struct PcaProjectionFrame
{
    int numPoints = 0;
    int numDims = 0;
    int numClusters = 0;

    std::array<std::array<float, 3>, kMaxPerceptualBins> points {};
    std::array<int, kMaxPerceptualBins> dominantCluster {};
    std::array<float, kMaxPerceptualBins> dominance {};
    std::array<std::array<float, 3>, kMaxClusters> clusterCenters {};
    std::array<float, kMaxClusters> clusterActivity {};
    std::array<float, kMaxClusters> clusterCentroidHz {};
    std::array<float, kMaxClusters> clusterBandwidthHz {};
    std::array<float, kMaxClusters> clusterTonalness {};
    std::array<float, kMaxClusters> clusterTransientness {};
    std::array<float, kMaxClusters> clusterEnergyShare {};
    std::array<int, kMaxClusters> clusterSemanticRole {};
};

class DspPcaProjection
{
public:
    void computeProjection(const std::array<std::array<float, kMaxPerceptualBins>, kMaxClusters>& perceptualWeights,
                           int numClusters,
                           int numPerceptualBins,
                           PcaProjectionFrame& outFrame) const noexcept;
};

} // namespace inertia
