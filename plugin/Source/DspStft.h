#pragma once

#include "Utilities.h"
#include <JuceHeader.h>

namespace inertia
{

class DspStft
{
public:
    class FrameProcessor
    {
    public:
        virtual ~FrameProcessor() = default;

        virtual void processSpectrum(
            std::array<std::array<juce::dsp::Complex<float>, kMaxFftSize>, kMaxChannels>& spectra,
            std::array<std::array<std::array<juce::dsp::Complex<float>, kMaxFftSize>, kMaxChannels>, kMaxAuxOutputs>& auxSpectra,
            int numAuxOutputs,
            int fftSize,
            int fftBins,
            int numChannels) = 0;
    };

    DspStft();

    void prepare(double sampleRate, int maxBlockSize, int numChannels);
    void reset();
    void setConfig(int fftSize, int hopSize);

    void processBlock(juce::AudioBuffer<float>& audio,
                      const std::array<std::array<float*, kMaxChannels>, kMaxAuxOutputs>& auxWritePointers,
                      int numAuxOutputs,
                      FrameProcessor& frameProcessor);

    int getFftSize() const noexcept { return fftSize_; }
    int getHopSize() const noexcept { return hopSize_; }
    int getLatencySamples() const noexcept { return fftSize_ - hopSize_; }

private:
    void rebuildWindowAndNormaliser();
    void processOneFrame(FrameProcessor& frameProcessor);

    juce::dsp::FFT fft512_;
    juce::dsp::FFT fft1024_;
    juce::dsp::FFT fft2048_;
    juce::dsp::FFT fft4096_;
    juce::dsp::FFT* activeFft_ = nullptr;

    double sampleRate_ = 44100.0;
    int fftSize_ = 2048;
    int fftBins_ = 1025;
    int hopSize_ = 1024;
    int numChannels_ = 2;

    int writePos_ = 0;
    int hopCounter_ = 0;
    int activeAuxOutputs_ = 0;

    float inverseScale512_ = 1.0f;
    float inverseScale1024_ = 1.0f;
    float inverseScale2048_ = 1.0f;
    float inverseScale4096_ = 1.0f;
    float inverseScale_ = 1.0f;

    std::array<float, kMaxFftSize> window_ {};
    std::array<float, kMaxFftSize> synthesisNormaliser_ {};

    std::array<std::array<float, kMaxFftSize>, kMaxChannels> inputRing_ {};
    std::array<std::array<float, kMaxFftSize>, kMaxChannels> outputRing_ {};
    std::array<std::array<std::array<float, kMaxFftSize>, kMaxChannels>, kMaxAuxOutputs> auxOutputRing_ {};

    std::array<std::array<juce::dsp::Complex<float>, kMaxFftSize>, kMaxChannels> timeDomain_ {};
    std::array<std::array<juce::dsp::Complex<float>, kMaxFftSize>, kMaxChannels> frequencyDomain_ {};
    std::array<std::array<std::array<juce::dsp::Complex<float>, kMaxFftSize>, kMaxChannels>, kMaxAuxOutputs> auxTimeDomain_ {};
    std::array<std::array<std::array<juce::dsp::Complex<float>, kMaxFftSize>, kMaxChannels>, kMaxAuxOutputs> auxFrequencyDomain_ {};
};

} // namespace inertia
