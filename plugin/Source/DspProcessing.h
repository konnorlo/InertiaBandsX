#pragma once

#include "Utilities.h"

namespace inertia
{

struct ClusterProcessingParams
{
    float gainLinear = 1.0f;
    float hpHz = 20.0f;
    float lpHz = 20000.0f;
    bool muted = false;
    int outputRoute = 0; // 0 = main, 1..kMaxAuxOutputs = aux output bus index
};

struct ProcessingParams
{
    double sampleRate = 44100.0;
    int fftSize = 2048;
    int fftBins = 1025;
    int numClusters = 4;

    float saturationAmount = 0.0f;
    float driveFromLevel = 0.0f;
    bool gateEnabled = false;
    float gateThreshold = -8.0f;
    float gateSharpness = 2.0f;
    float gateFloor = 0.0f;
    bool autoLevelCompensate = true;
    int outputMode = 0;

    float globalMix = 1.0f;
    float outputGainLinear = 1.0f;

    std::array<ClusterProcessingParams, kMaxClusters> clusterParams {};
};

class DspProcessing
{
public:
    void prepare(double sampleRate, int fftSize);

    void computeBinMultipliers(const std::array<float, kMaxFftBins>& magnitudes,
                               const std::array<std::array<float, kMaxFftBins>, kMaxClusters>& fftWeights,
                               const ProcessingParams& params,
                               std::array<float, kMaxFftBins>& outMultipliers,
                               std::array<float, kMaxClusters>& outClusterLevels,
                               std::array<std::array<float, kMaxFftBins>, kMaxClusters>* outClusterWetMultipliers = nullptr);

private:
    int fftSize_ = 2048;
    int fftBins_ = 1025;
    std::array<float, kMaxFftBins> binFrequencyHz_ {};
    float autoLevelCompensation_ = 1.0f;
};

} // namespace inertia
