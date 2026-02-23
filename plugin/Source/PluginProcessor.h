#pragma once

#include <JuceHeader.h>
#include <array>
#include <atomic>

#include "DspClustering.h"
#include "DspAutoHyperModel.h"
#include "DspMasks.h"
#include "DspPcaProjection.h"
#include "DspPerceptualBins.h"
#include "DspProcessing.h"
#include "DspStft.h"

namespace inertia::ParamIDs
{
inline constexpr auto bypass = "bypass";
inline constexpr auto freeze = "freeze";
inline constexpr auto fftSize = "fftSize";
inline constexpr auto hopMode = "hopMode";
inline constexpr auto numClusters = "numClusters";
inline constexpr auto clusterUpdateHz = "clusterUpdateHz";
inline constexpr auto glideMs = "glideMs";
inline constexpr auto clusterSpread = "clusterSpread";
inline constexpr auto distancePenalty = "distancePenalty";
inline constexpr auto maskSmoothMs = "maskSmoothMs";
inline constexpr auto autoLevel = "autoLevel";
inline constexpr auto outputMode = "outputMode";
inline constexpr auto globalMix = "globalMix";
inline constexpr auto outputGainDb = "outputGainDb";

inline constexpr auto saturationAmount = "saturationAmount";
inline constexpr auto driveFromLevel = "driveFromLevel";
inline constexpr auto gateEnable = "gateEnable";
inline constexpr auto gateThreshold = "gateThreshold";
inline constexpr auto gateSharpness = "gateSharpness";
inline constexpr auto gateFloor = "gateFloor";
inline constexpr auto featureAdaptive = "featureAdaptive";
inline constexpr auto featureAdaptRate = "featureAdaptRate";
inline constexpr auto featureLowPenalty = "featureLowPenalty";
inline constexpr auto featureHighPenalty = "featureHighPenalty";
inline constexpr auto featureNormPenalty = "featureNormPenalty";
inline constexpr auto featureMinimax = "featureMinimax";
inline constexpr auto featurePriorPenalty = "featurePriorPenalty";

inline juce::String clusterGainDb(int index) { return "clusterGainDb" + juce::String(index); }
inline juce::String clusterHpHz(int index) { return "clusterHpHz" + juce::String(index); }
inline juce::String clusterLpHz(int index) { return "clusterLpHz" + juce::String(index); }
inline juce::String clusterMute(int index) { return "clusterMute" + juce::String(index); }
inline juce::String clusterRoute(int index) { return "clusterRoute" + juce::String(index); }
} // namespace inertia::ParamIDs

class InertiaBandsAudioProcessor final : public juce::AudioProcessor,
                                         private inertia::DspStft::FrameProcessor
{
public:
    static constexpr int kUiWaveformSamples = 512;

    struct ResponseFrame
    {
        int fftBins = 0;
        int fftSize = 0;
        float sampleRate = 44100.0f;
        int numClusters = 0;
        int waveformSamples = 0;
        std::array<float, kUiWaveformSamples> waveform {};
        std::array<float, inertia::kFeatureDim> featureWeights {};
        std::array<float, inertia::kMaxFftBins> mixedGain {};
        std::array<std::array<float, inertia::kMaxFftBins>, inertia::kMaxClusters> clusterGain {};
        std::array<int, inertia::kMaxClusters> clusterRoute {};
        std::array<bool, inertia::kMaxClusters> clusterMuted {};
    };

    InertiaBandsAudioProcessor();
    ~InertiaBandsAudioProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getValueTreeState() noexcept { return apvts_; }
    const juce::AudioProcessorValueTreeState& getValueTreeState() const noexcept { return apvts_; }

    bool copyPcaProjectionFrame(inertia::PcaProjectionFrame& destination) const noexcept;
    bool copyResponseFrame(ResponseFrame& destination) const noexcept;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

private:
    static juce::AudioProcessor::BusesProperties createBuses();

    void processSpectrum(
        std::array<std::array<juce::dsp::Complex<float>, inertia::kMaxFftSize>, inertia::kMaxChannels>& spectra,
        std::array<std::array<std::array<juce::dsp::Complex<float>, inertia::kMaxFftSize>, inertia::kMaxChannels>, inertia::kMaxAuxOutputs>& auxSpectra,
        int numAuxOutputs,
        int fftSize,
        int fftBins,
        int numChannels) override;

    void cacheParameterPointers();
    void refreshRuntimeConfig(bool forceReset);
    void updatePcaProjectionFrame() noexcept;

    float getParam(std::atomic<float>* parameter) const noexcept;

    inertia::ProcessingParams makeProcessingParams(int fftSize, int fftBins) const;

    juce::AudioProcessorValueTreeState apvts_;

    inertia::DspStft stft_;
    inertia::DspPerceptualBins perceptualBins_;
    inertia::DspClustering clustering_;
    inertia::DspAutoHyperModel autoHyperModel_;
    inertia::DspMasks masks_;
    inertia::DspProcessing processing_;
    inertia::DspPcaProjection pcaProjection_;

    inertia::FeatureFrame featureFrame_;

    std::array<float, inertia::kMaxFftBins> magnitudes_ {};
    std::array<float, inertia::kMaxFftBins> multipliers_ {};
    std::array<float, kUiWaveformSamples> latestWaveform_ {};
    int latestWaveformSamples_ = 0;
    std::array<float, inertia::kMaxClusters> clusterLevels_ {};
    std::array<std::array<float, inertia::kMaxFftBins>, inertia::kMaxClusters> clusterWetMultipliers_ {};
    std::array<int, inertia::kMaxClusters> clusterOutputRoutes_ {};

    std::array<inertia::PcaProjectionFrame, 2> pcaFrames_ {};
    std::atomic<int> pcaFrontFrameIndex_ { 0 };

    std::atomic<float>* bypassParam_ = nullptr;
    std::atomic<float>* freezeParam_ = nullptr;
    std::atomic<float>* fftSizeParam_ = nullptr;
    std::atomic<float>* hopModeParam_ = nullptr;
    std::atomic<float>* numClustersParam_ = nullptr;
    std::atomic<float>* clusterUpdateHzParam_ = nullptr;
    std::atomic<float>* glideMsParam_ = nullptr;
    std::atomic<float>* clusterSpreadParam_ = nullptr;
    std::atomic<float>* distancePenaltyParam_ = nullptr;
    std::atomic<float>* maskSmoothMsParam_ = nullptr;
    std::atomic<float>* autoLevelParam_ = nullptr;
    std::atomic<float>* outputModeParam_ = nullptr;
    std::atomic<float>* globalMixParam_ = nullptr;
    std::atomic<float>* outputGainDbParam_ = nullptr;

    std::atomic<float>* saturationAmountParam_ = nullptr;
    std::atomic<float>* driveFromLevelParam_ = nullptr;
    std::atomic<float>* gateEnableParam_ = nullptr;
    std::atomic<float>* gateThresholdParam_ = nullptr;
    std::atomic<float>* gateSharpnessParam_ = nullptr;
    std::atomic<float>* gateFloorParam_ = nullptr;
    std::atomic<float>* featureAdaptiveParam_ = nullptr;
    std::atomic<float>* featureAdaptRateParam_ = nullptr;
    std::atomic<float>* featureLowPenaltyParam_ = nullptr;
    std::atomic<float>* featureHighPenaltyParam_ = nullptr;
    std::atomic<float>* featureNormPenaltyParam_ = nullptr;
    std::atomic<float>* featureMinimaxParam_ = nullptr;
    std::atomic<float>* featurePriorPenaltyParam_ = nullptr;

    std::array<std::atomic<float>*, inertia::kMaxClusters> clusterGainDbParams_ {};
    std::array<std::atomic<float>*, inertia::kMaxClusters> clusterHpHzParams_ {};
    std::array<std::atomic<float>*, inertia::kMaxClusters> clusterLpHzParams_ {};
    std::array<std::atomic<float>*, inertia::kMaxClusters> clusterMuteParams_ {};
    std::array<std::atomic<float>*, inertia::kMaxClusters> clusterRouteParams_ {};

    std::array<ResponseFrame, 2> responseFrames_ {};
    std::atomic<int> responseFrontFrameIndex_ { 0 };

    double sampleRateHz_ = 44100.0;
    int currentFftSize_ = 2048;
    int currentHopSize_ = 1024;
    int activeClusters_ = 4;

    int hopsPerClusterUpdate_ = 1;
    int hopsUntilClusterUpdate_ = 1;

    int hopsPerPcaUpdate_ = 1;
    int hopsUntilPcaUpdate_ = 1;

    bool needsCenterInit_ = true;
    bool wasBypassedLastBlock_ = false;

    bool hasInferredHyperParams_ = false;
    float inferredDistancePenalty_ = 1.0f;
    float inferredAdaptRate_ = 0.25f;
    float inferredLowPenalty_ = 0.45f;
    float inferredHighPenalty_ = 0.08f;
    float inferredMinimax_ = 0.4f;
    float inferredConfidence_ = 0.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InertiaBandsAudioProcessor)
};
