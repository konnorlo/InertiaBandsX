#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace inertia
{
constexpr int kMaxChannels = 2;
constexpr int kMaxFftSize = 4096;
constexpr int kMaxFftBins = (kMaxFftSize / 2) + 1;
constexpr int kMaxPerceptualBins = 64;
constexpr int kMaxClusters = 8;
constexpr int kMaxAuxOutputs = kMaxClusters;
constexpr int kFeatureDim = 10;

enum FeatureIndex
{
    kFeatureLogEnergy = 0,
    kFeatureSlope = 1,
    kFeatureLocalFlux = 2,
    kFeaturePeakiness = 3,
    kFeatureLogHz = 4,
    kFeatureFlatness = 5,
    kFeatureTransientness = 6,
    kFeatureBrightness = 7,
    kFeatureSpectralFlux = 8,
    kFeatureActivityConfidence = 9
};

enum ClusterSemanticRole
{
    kRoleLowBody = 0,
    kRoleHarmonicMid = 1,
    kRoleTransient = 2,
    kRoleAirNoise = 3,
    kRolePresence = 4,
    kRoleTextureMid = 5
};

inline constexpr std::array<const char*, kFeatureDim> kFeatureNames {
    "LogEnergy",
    "Slope",
    "Flux",
    "Peakiness",
    "LogHz",
    "Flatness",
    "Transient",
    "Brightness",
    "SpecFlux",
    "Confidence"
};

inline constexpr std::array<const char*, 6> kClusterRoleNames {
    "Low Body",
    "Harmonic Mid",
    "Transient",
    "Air/Noise",
    "Presence",
    "Texture Mid"
};

constexpr float kEpsilon = 1.0e-8f;
constexpr float kSilenceEnergyThreshold = 1.0e-5f;

template <typename T>
inline T clamp(T value, T low, T high)
{
    return std::min(high, std::max(low, value));
}

inline float dbToLinear(float dB)
{
    return std::pow(10.0f, dB * 0.05f);
}

inline float msToSmoothingCoeff(float ms, float dtSeconds)
{
    if (ms <= 0.0f)
        return 1.0f;

    const auto tau = ms * 0.001f;
    const auto safeTau = std::max(tau, 1.0e-6f);
    return 1.0f - std::exp(-dtSeconds / safeTau);
}

inline float sigmoid(float x)
{
    x = clamp(x, -50.0f, 50.0f);
    return 1.0f / (1.0f + std::exp(-x));
}

inline int fftOrderFromSize(int fftSize)
{
    switch (fftSize)
    {
        case 512: return 9;
        case 1024: return 10;
        case 2048: return 11;
        case 4096: return 12;
        default: break;
    }

    return 11;
}

} // namespace inertia
