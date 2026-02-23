#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <cmath>
#include <vector>

namespace
{
juce::NormalisableRange<float> makeSkewedFrequencyRange(float minHz, float maxHz)
{
    juce::NormalisableRange<float> range(minHz, maxHz, 1.0f);
    range.setSkewForCentre(std::sqrt(minHz * maxHz));
    return range;
}

juce::NormalisableRange<float> makeClusterGainRange()
{
    juce::NormalisableRange<float> range(-120.0f, 24.0f, 0.0f);
    range.setSkewForCentre(-18.0f);
    return range;
}

std::array<float, inertia::kFeatureDim> defaultFeatureWeights()
{
    std::array<float, inertia::kFeatureDim> weights {
        0.26f, // LogEnergy
        0.13f, // Slope
        0.18f, // Local flux
        0.10f, // Peakiness
        0.07f, // LogHz
        0.10f, // Flatness
        0.07f, // Transientness
        0.04f, // Brightness
        0.03f, // Spectral flux
        0.02f  // Confidence
    };

    float norm2 = 0.0f;
    for (const auto w : weights)
        norm2 += w * w;

    const auto invNorm = 1.0f / std::sqrt(std::max(norm2, inertia::kEpsilon));
    for (auto& w : weights)
        w *= invNorm;

    return weights;
}
} // namespace

juce::AudioProcessor::BusesProperties InertiaBandsAudioProcessor::createBuses()
{
    auto buses = juce::AudioProcessor::BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true);

    for (int bus = 0; bus < inertia::kMaxAuxOutputs; ++bus)
        buses = buses.withOutput("Aux " + juce::String(bus + 1), juce::AudioChannelSet::stereo(), false);

    return buses;
}

InertiaBandsAudioProcessor::InertiaBandsAudioProcessor()
    : AudioProcessor(createBuses()),
      apvts_(*this, nullptr, "PARAMETERS", createParameterLayout())
{
    cacheParameterPointers();
    setLatencySamples(currentFftSize_ - currentHopSize_);

    for (auto& frame : pcaFrames_)
        frame = {};

    for (auto& frame : responseFrames_)
        frame = {};
}

juce::AudioProcessorValueTreeState::ParameterLayout InertiaBandsAudioProcessor::createParameterLayout()
{
    using namespace inertia;

    std::vector<std::unique_ptr<juce::RangedAudioParameter>> parameters;
    parameters.reserve(28 + (kMaxClusters * 5));

    parameters.push_back(std::make_unique<juce::AudioParameterBool>(ParamIDs::bypass, "Bypass", false));
    parameters.push_back(std::make_unique<juce::AudioParameterBool>(ParamIDs::freeze, "Freeze", false));

    parameters.push_back(std::make_unique<juce::AudioParameterChoice>(
        ParamIDs::fftSize,
        "FFT Size",
        juce::StringArray { "1024", "2048", "512", "4096" },
        1));

    parameters.push_back(std::make_unique<juce::AudioParameterChoice>(
        ParamIDs::hopMode,
        "Hop",
        juce::StringArray { "N/2", "N/4" },
        0));

    parameters.push_back(std::make_unique<juce::AudioParameterInt>(ParamIDs::numClusters, "K Clusters", 2, 8, 4));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::clusterUpdateHz, "Cluster Update Hz", 5.0f, 40.0f, 20.0f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::glideMs, "Glide ms", 0.0f, 2000.0f, 150.0f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::clusterSpread, "Cluster Spread", 0.2f, 3.0f, 1.0f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::distancePenalty, "Distance Penalty", 0.0f, 6.0f, 1.0f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::maskSmoothMs, "Mask Smooth ms", 0.0f, 500.0f, 80.0f));
    parameters.push_back(std::make_unique<juce::AudioParameterBool>(ParamIDs::autoLevel, "Auto Level", true));

    parameters.push_back(std::make_unique<juce::AudioParameterChoice>(
        ParamIDs::outputMode,
        "Output Mode",
        juce::StringArray { "Mix", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7", "Cluster 8" },
        0));

    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::globalMix, "Global Mix", 0.0f, 1.0f, 1.0f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::outputGainDb, "Output Gain dB", -24.0f, 12.0f, 0.0f));

    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::saturationAmount, "Saturation Amount", 0.0f, 1.0f, 0.0f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::driveFromLevel, "Drive From Level", 0.0f, 2.0f, 0.0f));
    parameters.push_back(std::make_unique<juce::AudioParameterBool>(ParamIDs::gateEnable, "Gate Enable", false));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::gateThreshold, "Gate Threshold", -12.0f, 2.0f, -8.0f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::gateSharpness, "Gate Sharpness", 0.5f, 40.0f, 2.0f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::gateFloor, "Gate Floor", 0.0f, 0.2f, 0.02f));
    parameters.push_back(std::make_unique<juce::AudioParameterBool>(ParamIDs::featureAdaptive, "Adaptive Feature Weights", true));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::featureAdaptRate, "Feature Adapt Rate", 0.0f, 1.0f, 0.35f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::featureLowPenalty, "Feature Low Penalty", 0.0f, 4.0f, 0.45f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::featureHighPenalty, "Feature High Penalty", 0.0f, 2.0f, 0.08f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::featureNormPenalty, "Feature Norm Penalty", 0.0f, 4.0f, 0.5f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::featureMinimax, "Feature Minimax", 0.0f, 1.0f, 0.4f));
    parameters.push_back(std::make_unique<juce::AudioParameterFloat>(ParamIDs::featurePriorPenalty, "Feature Prior Penalty", 0.0f, 2.0f, 0.2f));

    const auto hpRange = makeSkewedFrequencyRange(20.0f, 2000.0f);
    const auto lpRange = makeSkewedFrequencyRange(200.0f, 20000.0f);

    const auto clusterGainRange = makeClusterGainRange();

    for (int cluster = 0; cluster < kMaxClusters; ++cluster)
    {
        const auto label = juce::String(cluster + 1);

        parameters.push_back(std::make_unique<juce::AudioParameterFloat>(
            ParamIDs::clusterGainDb(cluster),
            "Cluster " + label + " Gain dB",
            clusterGainRange,
            0.0f,
            juce::String(),
            juce::AudioProcessorParameter::genericParameter,
            [](float value, int)
            {
                if (value <= -119.5f)
                    return juce::String("-inf");

                return juce::String(value, 1);
            },
            [](const juce::String& text)
            {
                if (text.containsIgnoreCase("inf"))
                    return -120.0f;

                return text.getFloatValue();
            }));

        parameters.push_back(std::make_unique<juce::AudioParameterFloat>(
            ParamIDs::clusterHpHz(cluster),
            "Cluster " + label + " HP Hz",
            hpRange,
            20.0f));

        parameters.push_back(std::make_unique<juce::AudioParameterFloat>(
            ParamIDs::clusterLpHz(cluster),
            "Cluster " + label + " LP Hz",
            lpRange,
            20000.0f));

        parameters.push_back(std::make_unique<juce::AudioParameterBool>(
            ParamIDs::clusterMute(cluster),
            "Cluster " + label + " Mute",
            false));

        parameters.push_back(std::make_unique<juce::AudioParameterChoice>(
            ParamIDs::clusterRoute(cluster),
            "Cluster " + label + " Route",
            juce::StringArray { "Main", "Aux 1", "Aux 2", "Aux 3", "Aux 4", "Aux 5", "Aux 6", "Aux 7", "Aux 8" },
            0));

    }

    return { parameters.begin(), parameters.end() };
}

void InertiaBandsAudioProcessor::cacheParameterPointers()
{
    using namespace inertia;

    bypassParam_ = apvts_.getRawParameterValue(ParamIDs::bypass);
    freezeParam_ = apvts_.getRawParameterValue(ParamIDs::freeze);
    fftSizeParam_ = apvts_.getRawParameterValue(ParamIDs::fftSize);
    hopModeParam_ = apvts_.getRawParameterValue(ParamIDs::hopMode);
    numClustersParam_ = apvts_.getRawParameterValue(ParamIDs::numClusters);
    clusterUpdateHzParam_ = apvts_.getRawParameterValue(ParamIDs::clusterUpdateHz);
    glideMsParam_ = apvts_.getRawParameterValue(ParamIDs::glideMs);
    clusterSpreadParam_ = apvts_.getRawParameterValue(ParamIDs::clusterSpread);
    distancePenaltyParam_ = apvts_.getRawParameterValue(ParamIDs::distancePenalty);
    maskSmoothMsParam_ = apvts_.getRawParameterValue(ParamIDs::maskSmoothMs);
    autoLevelParam_ = apvts_.getRawParameterValue(ParamIDs::autoLevel);
    outputModeParam_ = apvts_.getRawParameterValue(ParamIDs::outputMode);
    globalMixParam_ = apvts_.getRawParameterValue(ParamIDs::globalMix);
    outputGainDbParam_ = apvts_.getRawParameterValue(ParamIDs::outputGainDb);

    saturationAmountParam_ = apvts_.getRawParameterValue(ParamIDs::saturationAmount);
    driveFromLevelParam_ = apvts_.getRawParameterValue(ParamIDs::driveFromLevel);
    gateEnableParam_ = apvts_.getRawParameterValue(ParamIDs::gateEnable);
    gateThresholdParam_ = apvts_.getRawParameterValue(ParamIDs::gateThreshold);
    gateSharpnessParam_ = apvts_.getRawParameterValue(ParamIDs::gateSharpness);
    gateFloorParam_ = apvts_.getRawParameterValue(ParamIDs::gateFloor);
    featureAdaptiveParam_ = apvts_.getRawParameterValue(ParamIDs::featureAdaptive);
    featureAdaptRateParam_ = apvts_.getRawParameterValue(ParamIDs::featureAdaptRate);
    featureLowPenaltyParam_ = apvts_.getRawParameterValue(ParamIDs::featureLowPenalty);
    featureHighPenaltyParam_ = apvts_.getRawParameterValue(ParamIDs::featureHighPenalty);
    featureNormPenaltyParam_ = apvts_.getRawParameterValue(ParamIDs::featureNormPenalty);
    featureMinimaxParam_ = apvts_.getRawParameterValue(ParamIDs::featureMinimax);
    featurePriorPenaltyParam_ = apvts_.getRawParameterValue(ParamIDs::featurePriorPenalty);

    for (int cluster = 0; cluster < inertia::kMaxClusters; ++cluster)
    {
        clusterGainDbParams_[cluster] = apvts_.getRawParameterValue(ParamIDs::clusterGainDb(cluster));
        clusterHpHzParams_[cluster] = apvts_.getRawParameterValue(ParamIDs::clusterHpHz(cluster));
        clusterLpHzParams_[cluster] = apvts_.getRawParameterValue(ParamIDs::clusterLpHz(cluster));
        clusterMuteParams_[cluster] = apvts_.getRawParameterValue(ParamIDs::clusterMute(cluster));
        clusterRouteParams_[cluster] = apvts_.getRawParameterValue(ParamIDs::clusterRoute(cluster));
    }
}

float InertiaBandsAudioProcessor::getParam(std::atomic<float>* parameter) const noexcept
{
    return (parameter != nullptr) ? parameter->load(std::memory_order_relaxed) : 0.0f;
}

bool InertiaBandsAudioProcessor::copyPcaProjectionFrame(inertia::PcaProjectionFrame& destination) const noexcept
{
    const auto front = pcaFrontFrameIndex_.load(std::memory_order_acquire);
    destination = pcaFrames_[front];
    return destination.numPoints > 0;
}

bool InertiaBandsAudioProcessor::copyResponseFrame(ResponseFrame& destination) const noexcept
{
    const auto front = responseFrontFrameIndex_.load(std::memory_order_acquire);
    destination = responseFrames_[front];
    return destination.fftBins > 0;
}

void InertiaBandsAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    sampleRateHz_ = sampleRate;

    stft_.prepare(sampleRate, samplesPerBlock, juce::jmin(getTotalNumInputChannels(), inertia::kMaxChannels));

    refreshRuntimeConfig(true);

    magnitudes_.fill(0.0f);
    multipliers_.fill(1.0f);
    latestWaveform_.fill(0.0f);
    latestWaveformSamples_ = 0;
    clusterLevels_.fill(-12.0f);
    for (auto& cluster : clusterWetMultipliers_)
        cluster.fill(0.0f);
    clusterOutputRoutes_.fill(0);

    for (auto& frame : pcaFrames_)
        frame = {};

    pcaFrontFrameIndex_.store(0, std::memory_order_relaxed);

    for (auto& frame : responseFrames_)
        frame = {};

    responseFrontFrameIndex_.store(0, std::memory_order_relaxed);
}

void InertiaBandsAudioProcessor::releaseResources()
{
}

bool InertiaBandsAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto in = layouts.getMainInputChannelSet();
    const auto outMain = layouts.getMainOutputChannelSet();

    if (in != outMain)
        return false;

    const auto mainSupported = (in == juce::AudioChannelSet::mono()) || (in == juce::AudioChannelSet::stereo());
    if (! mainSupported)
        return false;

    for (int bus = 1; bus < layouts.outputBuses.size(); ++bus)
    {
        const auto set = layouts.getChannelSet(false, bus);
        if (set.isDisabled())
            continue;

        if (set != outMain)
            return false;
    }

    return true;
}

void InertiaBandsAudioProcessor::refreshRuntimeConfig(bool forceReset)
{
    const auto fftChoice = static_cast<int>(std::round(getParam(fftSizeParam_)));
    const auto hopChoice = static_cast<int>(std::round(getParam(hopModeParam_)));

    auto requestedFft = 2048;
    switch (fftChoice)
    {
        case 0: requestedFft = 1024; break;
        case 1: requestedFft = 2048; break;
        case 2: requestedFft = 512; break;
        case 3: requestedFft = 4096; break;
        default: requestedFft = 2048; break;
    }

    const auto requestedHop = (hopChoice == 0) ? (requestedFft / 2) : (requestedFft / 4);

    const auto requestedClusters = inertia::clamp(static_cast<int>(std::round(getParam(numClustersParam_))), 2, inertia::kMaxClusters);

    const auto fftConfigChanged = forceReset || requestedFft != currentFftSize_ || requestedHop != currentHopSize_;
    const auto clustersChanged = forceReset || requestedClusters != activeClusters_;

    currentFftSize_ = requestedFft;
    currentHopSize_ = requestedHop;

    if (fftConfigChanged)
    {
        stft_.setConfig(currentFftSize_, currentHopSize_);
        setLatencySamples(stft_.getLatencySamples());
        perceptualBins_.prepare(sampleRateHz_, currentFftSize_, inertia::kMaxPerceptualBins);
        processing_.prepare(sampleRateHz_, currentFftSize_);
        autoHyperModel_.prepare(sampleRateHz_, currentFftSize_, perceptualBins_.getNumPerceptualBins());
        clustering_.reset(activeClusters_, perceptualBins_.getNumPerceptualBins());
        masks_.reset(activeClusters_, perceptualBins_.getNumPerceptualBins(), (currentFftSize_ / 2) + 1);
        needsCenterInit_ = true;
        hasInferredHyperParams_ = false;
        inferredConfidence_ = 0.0f;
    }

    if (clustersChanged)
    {
        activeClusters_ = requestedClusters;

        if (clustering_.isInitialised() && featureFrame_.numBins > 0)
            clustering_.setClusterCount(activeClusters_, featureFrame_.normalisedFeatures, featureFrame_.numBins);
        else
            clustering_.reset(activeClusters_, perceptualBins_.getNumPerceptualBins());

        masks_.reset(activeClusters_, perceptualBins_.getNumPerceptualBins(), (currentFftSize_ / 2) + 1);
        needsCenterInit_ = ! clustering_.isInitialised();
    }

    const auto requestedUpdateHz = inertia::clamp(getParam(clusterUpdateHzParam_), 5.0f, 40.0f);
    const auto minUpdateHz = 6.0f;
    const auto maxUpdateHz = (activeClusters_ <= 4) ? 14.0f : 10.0f;
    const auto updateHz = inertia::clamp(requestedUpdateHz, minUpdateHz, maxUpdateHz);
    const auto hopRate = static_cast<float>(sampleRateHz_ / static_cast<double>(std::max(currentHopSize_, 1)));

    hopsPerClusterUpdate_ = std::max(1, static_cast<int>(std::round(hopRate / std::max(updateHz, 0.1f))));
    hopsPerPcaUpdate_ = std::max(1, static_cast<int>(std::round(hopRate / 30.0f)));

    if (forceReset)
    {
        hopsUntilClusterUpdate_ = 1;
        hopsUntilPcaUpdate_ = 1;
        autoHyperModel_.reset();
        hasInferredHyperParams_ = false;
        inferredConfidence_ = 0.0f;
    }
    else
    {
        hopsUntilClusterUpdate_ = inertia::clamp(hopsUntilClusterUpdate_, 1, hopsPerClusterUpdate_);
        hopsUntilPcaUpdate_ = inertia::clamp(hopsUntilPcaUpdate_, 1, hopsPerPcaUpdate_);
    }
}

void InertiaBandsAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const auto totalInputChannels = getTotalNumInputChannels();
    const auto totalOutputChannels = getTotalNumOutputChannels();

    for (int channel = totalInputChannels; channel < totalOutputChannels; ++channel)
        buffer.clear(channel, 0, buffer.getNumSamples());

    auto mainBusBuffer = getBusBuffer(buffer, false, 0);
    {
        const auto blockSamples = mainBusBuffer.getNumSamples();
        const auto captureSamples = std::min(kUiWaveformSamples, std::max(0, blockSamples));
        latestWaveformSamples_ = captureSamples;
        latestWaveform_.fill(0.0f);

        if (captureSamples > 0 && mainBusBuffer.getNumChannels() > 0)
        {
            const auto* read = mainBusBuffer.getReadPointer(0);
            for (int i = 0; i < captureSamples; ++i)
            {
                const auto sourceIndex = (captureSamples > 1)
                    ? static_cast<int>((static_cast<long long>(i) * (blockSamples - 1)) / (captureSamples - 1))
                    : 0;
                latestWaveform_[static_cast<size_t>(i)] = read[sourceIndex];
            }
        }
    }

    std::array<std::array<float*, inertia::kMaxChannels>, inertia::kMaxAuxOutputs> auxWritePointers {};
    for (auto& bus : auxWritePointers)
        bus.fill(nullptr);

    const auto auxBusCount = std::min(inertia::kMaxAuxOutputs, std::max(0, getBusCount(false) - 1));
    for (int bus = 0; bus < auxBusCount; ++bus)
    {
        auto auxBusBuffer = getBusBuffer(buffer, false, bus + 1);
        auxBusBuffer.clear();

        const auto routeChannels = std::min(mainBusBuffer.getNumChannels(), std::min(auxBusBuffer.getNumChannels(), inertia::kMaxChannels));
        for (int channel = 0; channel < routeChannels; ++channel)
            auxWritePointers[bus][channel] = auxBusBuffer.getWritePointer(channel);
    }

    const auto bypassed = getParam(bypassParam_) > 0.5f;

    if (bypassed)
    {
        if (! wasBypassedLastBlock_)
            stft_.reset();

        wasBypassedLastBlock_ = true;
        return;
    }

    wasBypassedLastBlock_ = false;

    refreshRuntimeConfig(false);
    stft_.processBlock(mainBusBuffer, auxWritePointers, auxBusCount, *this);
}

inertia::ProcessingParams InertiaBandsAudioProcessor::makeProcessingParams(int fftSize, int fftBins) const
{
    inertia::ProcessingParams params;

    params.sampleRate = sampleRateHz_;
    params.fftSize = fftSize;
    params.fftBins = fftBins;
    params.numClusters = activeClusters_;

    params.saturationAmount = inertia::clamp(getParam(saturationAmountParam_), 0.0f, 1.0f);
    params.driveFromLevel = inertia::clamp(getParam(driveFromLevelParam_), 0.0f, 2.0f);

    params.gateEnabled = getParam(gateEnableParam_) > 0.5f;
    params.gateThreshold = getParam(gateThresholdParam_);
    params.gateSharpness = inertia::clamp(getParam(gateSharpnessParam_), 0.5f, 40.0f);
    params.gateFloor = inertia::clamp(getParam(gateFloorParam_), 0.0f, 0.2f);
    params.autoLevelCompensate = getParam(autoLevelParam_) > 0.5f;
    params.outputMode = inertia::clamp(static_cast<int>(std::round(getParam(outputModeParam_))), 0, inertia::kMaxClusters);

    params.globalMix = inertia::clamp(getParam(globalMixParam_), 0.0f, 1.0f);
    params.outputGainLinear = inertia::dbToLinear(getParam(outputGainDbParam_));

    for (int cluster = 0; cluster < inertia::kMaxClusters; ++cluster)
    {
        auto& cp = params.clusterParams[cluster];
        const auto gainDb = getParam(clusterGainDbParams_[cluster]);
        cp.gainLinear = (gainDb <= -119.5f) ? 0.0f : inertia::dbToLinear(gainDb);
        cp.hpHz = inertia::clamp(getParam(clusterHpHzParams_[cluster]), 20.0f, 2000.0f);
        cp.lpHz = inertia::clamp(getParam(clusterLpHzParams_[cluster]), 200.0f, 20000.0f);
        cp.muted = getParam(clusterMuteParams_[cluster]) > 0.5f;
        cp.outputRoute = inertia::clamp(static_cast<int>(std::round(getParam(clusterRouteParams_[cluster]))),
                                        0,
                                        inertia::kMaxAuxOutputs);
    }

    return params;
}

void InertiaBandsAudioProcessor::updatePcaProjectionFrame() noexcept
{
    const auto front = pcaFrontFrameIndex_.load(std::memory_order_relaxed);
    const auto back = 1 - front;
    auto& frame = pcaFrames_[back];
    pcaProjection_.computeProjection(masks_.getPerceptualWeights(), activeClusters_, featureFrame_.numBins, frame);

    const auto clusters = inertia::clamp(activeClusters_, 1, inertia::kMaxClusters);
    const auto bins = inertia::clamp(featureFrame_.numBins, 1, inertia::kMaxPerceptualBins);
    const auto& perceptualWeights = masks_.getPerceptualWeights();
    const auto& centerHz = perceptualBins_.getPerceptualCenterHz();

    std::array<float, inertia::kMaxClusters> clusterEnergy {};
    float totalEnergy = 0.0f;

    for (int c = 0; c < clusters; ++c)
    {
        float energySum = 0.0f;
        float freqSum = 0.0f;
        float freq2Sum = 0.0f;
        float tonalSum = 0.0f;
        float transientSum = 0.0f;

        for (int i = 0; i < bins; ++i)
        {
            const auto w = inertia::clamp(perceptualWeights[c][i], 0.0f, 1.0f);
            const auto binEnergy = std::exp(inertia::clamp(featureFrame_.logEnergy[i], -14.0f, 14.0f));
            const auto weightedEnergy = w * binEnergy;

            const auto hz = centerHz[i];
            const auto tonalness = inertia::clamp(0.5f + (0.18f * featureFrame_.normalisedFeatures[i][inertia::kFeaturePeakiness]), 0.0f, 1.0f);
            const auto transientness = inertia::clamp(0.5f + (0.18f * featureFrame_.normalisedFeatures[i][inertia::kFeatureTransientness]), 0.0f, 1.0f);

            energySum += weightedEnergy;
            freqSum += weightedEnergy * hz;
            freq2Sum += weightedEnergy * hz * hz;
            tonalSum += weightedEnergy * tonalness;
            transientSum += weightedEnergy * transientness;
        }

        clusterEnergy[c] = energySum;
        totalEnergy += energySum;

        if (energySum > inertia::kEpsilon)
        {
            const auto invEnergy = 1.0f / energySum;
            const auto centroidHz = freqSum * invEnergy;
            const auto variance = std::max(0.0f, (freq2Sum * invEnergy) - (centroidHz * centroidHz));
            const auto bandwidthHz = std::sqrt(variance);
            const auto tonalness = inertia::clamp(tonalSum * invEnergy, 0.0f, 1.0f);
            const auto transientness = inertia::clamp(transientSum * invEnergy, 0.0f, 1.0f);

            frame.clusterCentroidHz[c] = centroidHz;
            frame.clusterBandwidthHz[c] = bandwidthHz;
            frame.clusterTonalness[c] = tonalness;
            frame.clusterTransientness[c] = transientness;

            int role = inertia::kRoleTextureMid;
            if (transientness >= 0.66f)
                role = inertia::kRoleTransient;
            else if (centroidHz < 180.0f)
                role = inertia::kRoleLowBody;
            else if ((centroidHz >= 5500.0f) && (tonalness < 0.45f))
                role = inertia::kRoleAirNoise;
            else if ((tonalness >= 0.62f) && (centroidHz < 3800.0f))
                role = inertia::kRoleHarmonicMid;
            else if (centroidHz >= 3000.0f)
                role = inertia::kRolePresence;

            frame.clusterSemanticRole[c] = role;
        }
        else
        {
            frame.clusterCentroidHz[c] = 0.0f;
            frame.clusterBandwidthHz[c] = 0.0f;
            frame.clusterTonalness[c] = 0.0f;
            frame.clusterTransientness[c] = 0.0f;
            frame.clusterSemanticRole[c] = inertia::kRoleTextureMid;
        }
    }

    const auto invTotalEnergy = 1.0f / std::max(totalEnergy, inertia::kEpsilon);
    for (int c = 0; c < clusters; ++c)
        frame.clusterEnergyShare[c] = inertia::clamp(clusterEnergy[c] * invTotalEnergy, 0.0f, 1.0f);

    pcaFrontFrameIndex_.store(back, std::memory_order_release);
}

void InertiaBandsAudioProcessor::processSpectrum(
    std::array<std::array<juce::dsp::Complex<float>, inertia::kMaxFftSize>, inertia::kMaxChannels>& spectra,
    std::array<std::array<std::array<juce::dsp::Complex<float>, inertia::kMaxFftSize>, inertia::kMaxChannels>, inertia::kMaxAuxOutputs>& auxSpectra,
    int numAuxOutputs,
    int fftSize,
    int fftBins,
    int numChannels)
{
    const auto usedChannels = inertia::clamp(numChannels, 1, inertia::kMaxChannels);
    const auto auxOutputs = inertia::clamp(numAuxOutputs, 0, inertia::kMaxAuxOutputs);

    std::array<std::array<juce::dsp::Complex<float>, inertia::kMaxFftSize>, inertia::kMaxChannels> drySpectra {};
    for (int channel = 0; channel < usedChannels; ++channel)
    {
        for (int i = 0; i < fftSize; ++i)
            drySpectra[channel][i] = spectra[channel][i];
    }

    for (int bus = 0; bus < auxOutputs; ++bus)
    {
        for (int channel = 0; channel < usedChannels; ++channel)
        {
            for (int i = 0; i < fftSize; ++i)
                auxSpectra[bus][channel][i] = { 0.0f, 0.0f };
        }
    }

    for (int f = 0; f < fftBins; ++f)
    {
        float mag = 0.0f;

        for (int channel = 0; channel < usedChannels; ++channel)
        {
            const auto& c = drySpectra[channel][f];
            mag += std::sqrt((c.real() * c.real()) + (c.imag() * c.imag()));
        }

        magnitudes_[f] = mag / static_cast<float>(usedChannels);
    }

    for (int f = fftBins; f < inertia::kMaxFftBins; ++f)
        magnitudes_[f] = 0.0f;

    perceptualBins_.computeFeatures(magnitudes_, fftBins, featureFrame_);

    if (needsCenterInit_)
    {
        clustering_.initialiseFromFeatures(featureFrame_.normalisedFeatures, featureFrame_.numBins);
        masks_.reset(activeClusters_, featureFrame_.numBins, fftBins);
        needsCenterInit_ = false;
    }

    const auto manualFeatureWeights = defaultFeatureWeights();

    autoHyperModel_.pushFrame(featureFrame_);

    const auto adaptiveFeatureEnabled = true;
    auto featureAdaptRate = inertia::clamp(getParam(featureAdaptRateParam_), 0.0f, 1.0f);
    auto featureLowPenalty = inertia::clamp(getParam(featureLowPenaltyParam_), 0.0f, 4.0f);
    auto featureHighPenalty = inertia::clamp(getParam(featureHighPenaltyParam_), 0.0f, 2.0f);
    auto featureNormPenalty = inertia::clamp(getParam(featureNormPenaltyParam_), 0.0f, 4.0f);
    auto featureMinimax = inertia::clamp(getParam(featureMinimaxParam_), 0.0f, 1.0f);
    const auto featurePriorPenalty = inertia::clamp(getParam(featurePriorPenaltyParam_), 0.0f, 2.0f);

    auto distancePenalty = inertia::clamp(getParam(distancePenaltyParam_), 0.0f, 6.0f);

    --hopsUntilClusterUpdate_;
    const auto shouldRunClusterUpdate = (hopsUntilClusterUpdate_ <= 0);

    if (adaptiveFeatureEnabled && shouldRunClusterUpdate)
    {
        const auto prediction = autoHyperModel_.predict();
        if (prediction.valid)
        {
            const auto predictionConfidence = inertia::clamp(prediction.confidence, 0.0f, 1.0f);
            if (! hasInferredHyperParams_)
            {
                inferredDistancePenalty_ = prediction.distancePenalty;
                inferredAdaptRate_ = prediction.adaptRate;
                inferredLowPenalty_ = prediction.lowPenalty;
                inferredHighPenalty_ = prediction.highPenalty;
                inferredMinimax_ = prediction.minimaxStrength;
                inferredConfidence_ = predictionConfidence;
                hasInferredHyperParams_ = true;
            }
            else
            {
                const auto smoothing = 0.06f + (0.24f * predictionConfidence);
                inferredDistancePenalty_ += smoothing * (prediction.distancePenalty - inferredDistancePenalty_);
                inferredAdaptRate_ += smoothing * (prediction.adaptRate - inferredAdaptRate_);
                inferredLowPenalty_ += smoothing * (prediction.lowPenalty - inferredLowPenalty_);
                inferredHighPenalty_ += smoothing * (prediction.highPenalty - inferredHighPenalty_);
                inferredMinimax_ += smoothing * (prediction.minimaxStrength - inferredMinimax_);
                inferredConfidence_ += 0.2f * (predictionConfidence - inferredConfidence_);
            }
        }
        else
        {
            inferredConfidence_ *= 0.92f;
        }
    }

    if (adaptiveFeatureEnabled && hasInferredHyperParams_)
    {
        const auto confidence = inertia::clamp(inferredConfidence_, 0.0f, 1.0f);
        const auto blend = confidence * confidence;

        distancePenalty += blend * (inertia::clamp(inferredDistancePenalty_, 0.0f, 6.0f) - distancePenalty);
        featureAdaptRate += blend * (inertia::clamp(inferredAdaptRate_, 0.0f, 1.0f) - featureAdaptRate);
        featureLowPenalty += blend * (inertia::clamp(inferredLowPenalty_, 0.0f, 4.0f) - featureLowPenalty);
        featureHighPenalty += blend * (inertia::clamp(inferredHighPenalty_, 0.0f, 2.0f) - featureHighPenalty);
        featureMinimax += blend * (inertia::clamp(inferredMinimax_, 0.0f, 1.0f) - featureMinimax);
    }

    clustering_.setFeatureWeightConfig(manualFeatureWeights,
                                       adaptiveFeatureEnabled,
                                       featureAdaptRate,
                                       featureLowPenalty,
                                       featureHighPenalty,
                                       featureNormPenalty,
                                       featureMinimax,
                                       featurePriorPenalty);

    const auto freeze = getParam(freezeParam_) > 0.5f;

    if (shouldRunClusterUpdate)
    {
        if (! freeze)
            clustering_.updateTargets(featureFrame_.normalisedFeatures, featureFrame_.numBins, 3);

        hopsUntilClusterUpdate_ = hopsPerClusterUpdate_;
    }

    const auto hopSeconds = static_cast<float>(currentHopSize_ / sampleRateHz_);

    const auto requestedGlideMs = inertia::clamp(getParam(glideMsParam_), 0.0f, 2000.0f);
    const auto stabilityFloorMs = 140.0f + (35.0f * static_cast<float>(std::max(0, activeClusters_ - 2)));
    const auto constrainedGlideMs = (requestedGlideMs <= 0.0f) ? 0.0f : std::max(requestedGlideMs, stabilityFloorMs);
    const auto inertiaAlpha = freeze ? 0.0f : inertia::msToSmoothingCoeff(constrainedGlideMs, hopSeconds);
    clustering_.advanceInertia(inertiaAlpha);

    const auto spread = inertia::clamp(getParam(clusterSpreadParam_), 0.2f, 3.0f);
    const auto maskBeta = inertia::msToSmoothingCoeff(getParam(maskSmoothMsParam_), hopSeconds);

    masks_.compute(featureFrame_.normalisedFeatures,
                   clustering_.getCurrentCenters(),
                   clustering_.getFeatureWeights(),
                   activeClusters_,
                   featureFrame_.numBins,
                   perceptualBins_.getFftToPerceptualMap(),
                   fftBins,
                   spread,
                   distancePenalty,
                   maskBeta,
                   featureFrame_.totalEnergy);

    --hopsUntilPcaUpdate_;
    if (hopsUntilPcaUpdate_ <= 0)
    {
        updatePcaProjectionFrame();
        hopsUntilPcaUpdate_ = hopsPerPcaUpdate_;
    }

    const auto processingParams = makeProcessingParams(fftSize, fftBins);
    for (int cluster = 0; cluster < inertia::kMaxClusters; ++cluster)
        clusterOutputRoutes_[cluster] = processingParams.clusterParams[cluster].outputRoute;

    processing_.computeBinMultipliers(magnitudes_,
                                      masks_.getFftWeights(),
                                      processingParams,
                                      multipliers_,
                                      clusterLevels_,
                                      &clusterWetMultipliers_);

    {
        const auto front = responseFrontFrameIndex_.load(std::memory_order_relaxed);
        const auto back = 1 - front;
        auto& frame = responseFrames_[back];
        frame = {};
        frame.fftBins = fftBins;
        frame.fftSize = fftSize;
        frame.sampleRate = static_cast<float>(sampleRateHz_);
        frame.numClusters = activeClusters_;
        frame.waveformSamples = latestWaveformSamples_;
        frame.waveform = latestWaveform_;
        frame.featureWeights = clustering_.getFeatureWeights();

        for (int f = 0; f < fftBins; ++f)
            frame.mixedGain[f] = multipliers_[f];

        for (int cluster = 0; cluster < inertia::kMaxClusters; ++cluster)
        {
            frame.clusterRoute[cluster] = processingParams.clusterParams[cluster].outputRoute;
            frame.clusterMuted[cluster] = processingParams.clusterParams[cluster].muted;

            for (int f = 0; f < fftBins; ++f)
                frame.clusterGain[cluster][f] = clusterWetMultipliers_[cluster][f];
        }

        responseFrontFrameIndex_.store(back, std::memory_order_release);
    }

    for (int f = 0; f < fftBins; ++f)
    {
        const auto mainMultiplier = multipliers_[f];
        std::array<float, inertia::kMaxAuxOutputs> routeMultipliers {};

        if (auxOutputs > 0)
        {
            for (int cluster = 0; cluster < activeClusters_; ++cluster)
            {
                const auto route = clusterOutputRoutes_[cluster];
                if (route <= 0 || route > auxOutputs)
                    continue;

                const auto w = masks_.getFftWeights()[cluster][f];
                if (w <= 0.0f)
                    continue;

                routeMultipliers[route - 1] += w * clusterWetMultipliers_[cluster][f];
            }

            for (int route = 0; route < auxOutputs; ++route)
                routeMultipliers[route] = inertia::clamp(routeMultipliers[route], 0.0f, 20.0f);
        }

        for (int channel = 0; channel < usedChannels; ++channel)
        {
            spectra[channel][f] = drySpectra[channel][f] * mainMultiplier;

            for (int route = 0; route < auxOutputs; ++route)
                auxSpectra[route][channel][f] = drySpectra[channel][f] * routeMultipliers[route];
        }

        if (f > 0 && f < (fftSize / 2))
        {
            const auto mirror = fftSize - f;
            for (int channel = 0; channel < usedChannels; ++channel)
            {
                spectra[channel][mirror] = drySpectra[channel][mirror] * mainMultiplier;

                for (int route = 0; route < auxOutputs; ++route)
                    auxSpectra[route][channel][mirror] = drySpectra[channel][mirror] * routeMultipliers[route];
            }
        }
    }
}

juce::AudioProcessorEditor* InertiaBandsAudioProcessor::createEditor()
{
    return new InertiaBandsAudioProcessorEditor(*this);
}

bool InertiaBandsAudioProcessor::hasEditor() const
{
    return true;
}

const juce::String InertiaBandsAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool InertiaBandsAudioProcessor::acceptsMidi() const
{
    return false;
}

bool InertiaBandsAudioProcessor::producesMidi() const
{
    return false;
}

bool InertiaBandsAudioProcessor::isMidiEffect() const
{
    return false;
}

double InertiaBandsAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int InertiaBandsAudioProcessor::getNumPrograms()
{
    return 1;
}

int InertiaBandsAudioProcessor::getCurrentProgram()
{
    return 0;
}

void InertiaBandsAudioProcessor::setCurrentProgram(int)
{
}

const juce::String InertiaBandsAudioProcessor::getProgramName(int)
{
    return {};
}

void InertiaBandsAudioProcessor::changeProgramName(int, const juce::String&)
{
}

void InertiaBandsAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto xml = apvts_.copyState().createXml(); xml != nullptr)
        copyXmlToBinary(*xml, destData);
}

void InertiaBandsAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (auto xmlState = getXmlFromBinary(data, sizeInBytes))
    {
        if (xmlState->hasTagName(apvts_.state.getType()))
            apvts_.replaceState(juce::ValueTree::fromXml(*xmlState));
    }

    refreshRuntimeConfig(true);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new InertiaBandsAudioProcessor();
}
