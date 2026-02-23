#include "DspStft.h"
#include <cmath>

namespace inertia
{
namespace
{
float estimateInverseScale(juce::dsp::FFT& fft, int fftSize)
{
    std::array<juce::dsp::Complex<float>, kMaxFftSize> input {};
    std::array<juce::dsp::Complex<float>, kMaxFftSize> frequency {};
    std::array<juce::dsp::Complex<float>, kMaxFftSize> output {};

    input[0] = { 1.0f, 0.0f };

    fft.perform(input.data(), frequency.data(), false);
    fft.perform(frequency.data(), output.data(), true);

    const auto recovered = std::abs(output[0].real());
    if (recovered < 1.0e-6f)
        return 1.0f / static_cast<float>(fftSize);

    return 1.0f / recovered;
}
} // namespace

DspStft::DspStft()
    : fft512_(9),
      fft1024_(10),
      fft2048_(11),
      fft4096_(12)
{
    inverseScale512_ = estimateInverseScale(fft512_, 512);
    inverseScale1024_ = estimateInverseScale(fft1024_, 1024);
    inverseScale2048_ = estimateInverseScale(fft2048_, 2048);
    inverseScale4096_ = estimateInverseScale(fft4096_, 4096);

    activeFft_ = &fft2048_;
    inverseScale_ = inverseScale2048_;

    rebuildWindowAndNormaliser();
    reset();
}

void DspStft::prepare(double sampleRate, int, int numChannels)
{
    sampleRate_ = sampleRate;
    numChannels_ = clamp(numChannels, 1, kMaxChannels);
    reset();
}

void DspStft::setConfig(int fftSize, int hopSize)
{
    int requestedFft = 2048;
    if (fftSize <= 512)
        requestedFft = 512;
    else if (fftSize <= 1024)
        requestedFft = 1024;
    else if (fftSize <= 2048)
        requestedFft = 2048;
    else
        requestedFft = 4096;

    const auto requestedHop = (hopSize <= requestedFft / 4) ? (requestedFft / 4) : (requestedFft / 2);

    const auto changed = (requestedFft != fftSize_) || (requestedHop != hopSize_);

    fftSize_ = requestedFft;
    fftBins_ = (fftSize_ / 2) + 1;
    hopSize_ = requestedHop;

    switch (fftSize_)
    {
        case 512:
            activeFft_ = &fft512_;
            inverseScale_ = inverseScale512_;
            break;
        case 1024:
            activeFft_ = &fft1024_;
            inverseScale_ = inverseScale1024_;
            break;
        case 2048:
            activeFft_ = &fft2048_;
            inverseScale_ = inverseScale2048_;
            break;
        case 4096:
            activeFft_ = &fft4096_;
            inverseScale_ = inverseScale4096_;
            break;
        default:
            activeFft_ = &fft2048_;
            inverseScale_ = inverseScale2048_;
            break;
    }

    if (changed)
    {
        rebuildWindowAndNormaliser();
        reset();
    }
}

void DspStft::reset()
{
    writePos_ = 0;
    hopCounter_ = 0;
    activeAuxOutputs_ = 0;

    for (auto& channel : inputRing_)
        channel.fill(0.0f);

    for (auto& channel : outputRing_)
        channel.fill(0.0f);

    for (auto& bus : auxOutputRing_)
        for (auto& channel : bus)
            channel.fill(0.0f);

    for (auto& channel : timeDomain_)
        for (auto& c : channel)
            c = { 0.0f, 0.0f };

    for (auto& channel : frequencyDomain_)
        for (auto& c : channel)
            c = { 0.0f, 0.0f };

    for (auto& bus : auxTimeDomain_)
        for (auto& channel : bus)
            for (auto& c : channel)
                c = { 0.0f, 0.0f };

    for (auto& bus : auxFrequencyDomain_)
        for (auto& channel : bus)
            for (auto& c : channel)
                c = { 0.0f, 0.0f };
}

void DspStft::rebuildWindowAndNormaliser()
{
    window_.fill(0.0f);
    synthesisNormaliser_.fill(1.0f);

    const auto denom = static_cast<float>(std::max(1, fftSize_ - 1));

    for (int i = 0; i < fftSize_; ++i)
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * juce::MathConstants<float>::pi * static_cast<float>(i) / denom));

    for (int phase = 0; phase < hopSize_; ++phase)
    {
        float sumSquares = 0.0f;
        for (int i = phase; i < fftSize_; i += hopSize_)
            sumSquares += window_[i] * window_[i];

        const auto gain = (sumSquares > kEpsilon) ? (1.0f / sumSquares) : 1.0f;

        for (int i = phase; i < fftSize_; i += hopSize_)
            synthesisNormaliser_[i] = gain;
    }
}

void DspStft::processBlock(juce::AudioBuffer<float>& audio,
                           const std::array<std::array<float*, kMaxChannels>, kMaxAuxOutputs>& auxWritePointers,
                           int numAuxOutputs,
                           FrameProcessor& frameProcessor)
{
    const auto numSamples = audio.getNumSamples();
    const auto channelsToProcess = std::min(numChannels_, audio.getNumChannels());
    activeAuxOutputs_ = clamp(numAuxOutputs, 0, kMaxAuxOutputs);

    for (int sample = 0; sample < numSamples; ++sample)
    {
        for (int channel = 0; channel < channelsToProcess; ++channel)
        {
            const auto input = audio.getSample(channel, sample);
            const auto output = outputRing_[channel][writePos_];

            audio.setSample(channel, sample, output);
            outputRing_[channel][writePos_] = 0.0f;
            inputRing_[channel][writePos_] = input;
        }

        for (int bus = 0; bus < activeAuxOutputs_; ++bus)
        {
            for (int channel = 0; channel < channelsToProcess; ++channel)
            {
                const auto output = auxOutputRing_[bus][channel][writePos_];
                auxOutputRing_[bus][channel][writePos_] = 0.0f;

                if (auto* destination = auxWritePointers[bus][channel]; destination != nullptr)
                    destination[sample] = output;
            }
        }

        ++hopCounter_;

        if (hopCounter_ >= hopSize_)
        {
            hopCounter_ -= hopSize_;
            processOneFrame(frameProcessor);
        }

        ++writePos_;
        if (writePos_ >= fftSize_)
            writePos_ = 0;
    }
}

void DspStft::processOneFrame(FrameProcessor& frameProcessor)
{
    const auto frameStart = (writePos_ + 1) % fftSize_;

    for (int channel = 0; channel < numChannels_; ++channel)
    {
        for (int i = 0; i < fftSize_; ++i)
        {
            auto sourceIndex = frameStart + i;
            if (sourceIndex >= fftSize_)
                sourceIndex -= fftSize_;

            timeDomain_[channel][i] = { inputRing_[channel][sourceIndex] * window_[i], 0.0f };
        }

        activeFft_->perform(timeDomain_[channel].data(), frequencyDomain_[channel].data(), false);
    }

    for (int bus = 0; bus < activeAuxOutputs_; ++bus)
    {
        for (int channel = 0; channel < numChannels_; ++channel)
        {
            for (int i = 0; i < fftSize_; ++i)
                auxFrequencyDomain_[bus][channel][i] = { 0.0f, 0.0f };
        }
    }

    frameProcessor.processSpectrum(frequencyDomain_, auxFrequencyDomain_, activeAuxOutputs_, fftSize_, fftBins_, numChannels_);

    for (int channel = 0; channel < numChannels_; ++channel)
    {
        activeFft_->perform(frequencyDomain_[channel].data(), timeDomain_[channel].data(), true);

        for (int i = 0; i < fftSize_; ++i)
        {
            auto destinationIndex = frameStart + i;
            if (destinationIndex >= fftSize_)
                destinationIndex -= fftSize_;

            auto value = timeDomain_[channel][i].real() * inverseScale_;
            value *= window_[i] * synthesisNormaliser_[i];
            outputRing_[channel][destinationIndex] += value;
        }
    }

    for (int bus = 0; bus < activeAuxOutputs_; ++bus)
    {
        for (int channel = 0; channel < numChannels_; ++channel)
        {
            activeFft_->perform(auxFrequencyDomain_[bus][channel].data(), auxTimeDomain_[bus][channel].data(), true);

            for (int i = 0; i < fftSize_; ++i)
            {
                auto destinationIndex = frameStart + i;
                if (destinationIndex >= fftSize_)
                    destinationIndex -= fftSize_;

                auto value = auxTimeDomain_[bus][channel][i].real() * inverseScale_;
                value *= window_[i] * synthesisNormaliser_[i];
                auxOutputRing_[bus][channel][destinationIndex] += value;
            }
        }
    }
}

} // namespace inertia
