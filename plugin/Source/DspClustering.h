#pragma once

#include "Utilities.h"

namespace inertia
{

class DspClustering
{
public:
    void reset(int numClusters, int numPerceptualBins);

    void setFeatureWeightConfig(const std::array<float, kFeatureDim>& manualWeights,
                                bool adaptiveEnabled,
                                float adaptRate,
                                float lowPenalty,
                                float highPenalty,
                                float normPenalty,
                                float minimaxBlend,
                                float priorPenalty);

    void setClusterCount(int newClusterCount,
                         const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                         int numFeatureBins);

    void initialiseFromFeatures(const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                                int numFeatureBins);

    void updateTargets(const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                       int numFeatureBins,
                       int maxIterations);

    void advanceInertia(float alpha);

    int getClusterCount() const noexcept { return numClusters_; }
    bool isInitialised() const noexcept { return initialised_; }

    const std::array<std::array<float, kFeatureDim>, kMaxClusters>& getCurrentCenters() const noexcept { return currentCenters_; }
    const std::array<std::array<float, kFeatureDim>, kMaxClusters>& getTargetCenters() const noexcept { return targetCenters_; }
    const std::array<float, kFeatureDim>& getFeatureWeights() const noexcept { return featureWeights_; }

private:
    float squaredDistance(const std::array<float, kFeatureDim>& a,
                          const std::array<float, kFeatureDim>& b) const;

    void deterministicSeedFromFeatures(const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                                       int numFeatureBins,
                                       std::array<std::array<float, kFeatureDim>, kMaxClusters>& destination,
                                       int startCluster,
                                       int endCluster);

    void reorderCentersBySemantics();

    void updateFeatureWeights(const std::array<std::array<float, kFeatureDim>, kMaxPerceptualBins>& features,
                              int bins);

    int numClusters_ = 4;
    int numPerceptualBins_ = kMaxPerceptualBins;
    bool initialised_ = false;

    std::array<std::array<float, kFeatureDim>, kMaxClusters> currentCenters_ {};
    std::array<std::array<float, kFeatureDim>, kMaxClusters> targetCenters_ {};
    std::array<std::array<float, kFeatureDim>, kMaxClusters> workingCenters_ {};

    std::array<int, kMaxPerceptualBins> assignments_ {};
    std::array<int, kMaxPerceptualBins> previousAssignments_ {};

    std::array<float, kFeatureDim> featureWeights_ {};
    std::array<float, kFeatureDim> manualWeights_ {};

    bool adaptiveWeightsEnabled_ = false;
    float adaptRate_ = 0.25f;
    float lowPenalty_ = 0.35f;
    float highPenalty_ = 0.08f;
    float normPenalty_ = 0.4f;
    float minimaxBlend_ = 0.35f;
    float priorPenalty_ = 0.15f;

    float semanticFrequencyPenalty_ = 0.08f;
    float continuityPenalty_ = 0.04f;
    float minClusterMassFraction_ = 0.05f;
};

} // namespace inertia
