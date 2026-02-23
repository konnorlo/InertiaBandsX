#pragma once

#include <JuceHeader.h>
#include <array>

#include "PluginProcessor.h"

class ChalkLookAndFeel final : public juce::LookAndFeel_V4
{
public:
    ChalkLookAndFeel() = default;
    void setChalkTypeface(juce::Typeface::Ptr typeface);

    juce::Font getLabelFont(juce::Label& label) override;
    juce::Typeface::Ptr getTypefaceForFont(const juce::Font& font) override;
    void drawLinearSlider(juce::Graphics& g,
                          int x,
                          int y,
                          int width,
                          int height,
                          float sliderPos,
                          float minSliderPos,
                          float maxSliderPos,
                          juce::Slider::SliderStyle style,
                          juce::Slider& slider) override;
    void drawToggleButton(juce::Graphics& g,
                          juce::ToggleButton& button,
                          bool shouldDrawButtonAsHighlighted,
                          bool shouldDrawButtonAsDown) override;
    void drawComboBox(juce::Graphics& g,
                      int width,
                      int height,
                      bool isButtonDown,
                      int buttonX,
                      int buttonY,
                      int buttonW,
                      int buttonH,
                      juce::ComboBox& box) override;
    void drawPopupMenuItem(juce::Graphics& g,
                           const juce::Rectangle<int>& area,
                           bool isSeparator,
                           bool isActive,
                           bool isHighlighted,
                           bool isTicked,
                           bool hasSubMenu,
                           const juce::String& text,
                           const juce::String& shortcutKeyText,
                           const juce::Drawable* icon,
                           const juce::Colour* textColourToUse) override;
    void positionComboBoxText(juce::ComboBox& box, juce::Label& label) override;
    void drawPopupMenuBackground(juce::Graphics& g, int width, int height) override;

private:
    juce::Typeface::Ptr chalkTypeface_;
};

class PcaProjectionVisualizer final : public juce::Component
{
public:
    void setFrame(const inertia::PcaProjectionFrame& frame);
    void setResponseFrame(const InertiaBandsAudioProcessor::ResponseFrame& frame) noexcept;
    void paint(juce::Graphics& g) override;
    void mouseDown(const juce::MouseEvent& event) override;
    void mouseDrag(const juce::MouseEvent& event) override;
    void mouseUp(const juce::MouseEvent& event) override;

    float getViewYaw() const noexcept;
    float getViewPitch() const noexcept;

private:
    inertia::PcaProjectionFrame frame_;
    inertia::PcaProjectionFrame smoothedFrame_;
    inertia::PcaProjectionFrame previousSmoothedFrame_;
    bool hasSmoothedFrame_ = false;
    bool hasPreviousFrame_ = false;
    bool dragging_ = false;
    juce::Point<int> lastMousePos_;
    float userYaw_ = 0.0f;
    float userPitch_ = -0.42f;
    float autoYaw_ = 0.72f;
    double lastUpdateSeconds_ = 0.0;
    InertiaBandsAudioProcessor::ResponseFrame responseFrame_ {};
    InertiaBandsAudioProcessor::ResponseFrame smoothedResponseFrame_ {};
    bool hasSmoothedResponse_ = false;
};

class InertiaBandsAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                               private juce::Timer
{
public:
    explicit InertiaBandsAudioProcessorEditor(InertiaBandsAudioProcessor& audioProcessor);
    ~InertiaBandsAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;
    using ComboBoxAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    void timerCallback() override;
    void configureSlider(juce::Slider& slider, const juce::String& suffix);
    void updateClusterVisibility();
    void updateClusterSemanticReadout();
    void updateFeatureWeightReadout(const InertiaBandsAudioProcessor::ResponseFrame& frame);
    void applyModeVisibility();
    void loadThemeAssets();
    void drawChalkboardBackground(juce::Graphics& g);
    void drawPanelChrome(juce::Graphics& g, juce::Rectangle<float> bounds, const juce::String& title, bool emphasise = false);
    void updateGlitchFromActivity();

    InertiaBandsAudioProcessor& processor_;
    juce::AudioProcessorValueTreeState& apvts_;
    ChalkLookAndFeel lookAndFeel_;

    juce::Image chalkboardTexture_;
    juce::Image chalkDustTexture_;
    juce::Image chalkSmudgeTexture_;
    juce::Image borderAtlasTexture_;
    juce::Image glitchNoiseTexture_;

    juce::Rectangle<int> topBarBounds_;
    juce::Rectangle<int> leftPanelBounds_;
    juce::Rectangle<int> centerPanelBounds_;
    juce::Rectangle<int> rightPanelBounds_;
    juce::Rectangle<int> clusterPanelBounds_;

    std::array<float, 28> lastMonitoredValues_ {};
    bool haveMonitoredValues_ = false;
    float glitchEnergy_ = 0.0f;
    float glitchAmount_ = 0.30f;
    int glitchFramesRemaining_ = 0;
    juce::Random glitchRandom_ { 0x51A3B8D };
    juce::Typeface::Ptr customTypeface_;

    juce::ToggleButton bypassButton_ { "Bypass" };
    juce::ToggleButton freezeButton_ { "Freeze" };
    juce::ToggleButton gateEnableButton_ { "Gate" };
    juce::ToggleButton autoLevelButton_ { "Auto Level" };
    juce::ToggleButton advancedModeButton_ { "Advanced" };

    juce::ComboBox fftSizeBox_;
    juce::ComboBox hopModeBox_;
    juce::ComboBox outputModeBox_;
    juce::Label fftLabel_ { {}, "FFT" };
    juce::Label hopLabel_ { {}, "Hop" };
    juce::Label outputModeLabel_ { {}, "Out" };

    juce::Slider clustersSlider_;
    juce::Slider updateHzSlider_;
    juce::Slider glideMsSlider_;
    juce::Slider spreadSlider_;
    juce::Slider distancePenaltySlider_;
    juce::Slider smoothMsSlider_;
    juce::Slider mixSlider_;
    juce::Slider outputGainSlider_;
    juce::Slider saturationSlider_;
    juce::Slider driveFromLevelSlider_;
    juce::Slider gateThresholdSlider_;
    juce::Slider gateSharpnessSlider_;
    juce::Slider gateFloorSlider_;
    juce::Slider featureAdaptRateSlider_;
    juce::Slider featureLowPenaltySlider_;
    juce::Slider featureHighPenaltySlider_;
    juce::Slider featureNormPenaltySlider_;
    juce::Slider featureMinimaxSlider_;
    juce::Slider featurePriorPenaltySlider_;

    juce::Label clustersLabel_ { {}, "K" };
    juce::Label updateHzLabel_ { {}, "Update Hz" };
    juce::Label glideMsLabel_ { {}, "Glide ms" };
    juce::Label spreadLabel_ { {}, "Spread" };
    juce::Label distancePenaltyLabel_ { {}, "Dist Penalty" };
    juce::Label smoothMsLabel_ { {}, "Mask Smooth" };
    juce::Label mixLabel_ { {}, "Mix" };
    juce::Label outputGainLabel_ { {}, "Output dB" };
    juce::Label saturationLabel_ { {}, "Saturation" };
    juce::Label driveFromLevelLabel_ { {}, "Drive From Level" };
    juce::Label gateThresholdLabel_ { {}, "Gate Thresh" };
    juce::Label gateSharpnessLabel_ { {}, "Gate Sharpness" };
    juce::Label gateFloorLabel_ { {}, "Gate Floor" };
    juce::Label featureAdaptRateLabel_ { {}, "Feat Adapt Rate" };
    juce::Label featureLowPenaltyLabel_ { {}, "Feat Low Pen" };
    juce::Label featureHighPenaltyLabel_ { {}, "Feat High Pen" };
    juce::Label featureNormPenaltyLabel_ { {}, "Feat Norm Pen" };
    juce::Label featureMinimaxLabel_ { {}, "Feat Minimax" };
    juce::Label featurePriorPenaltyLabel_ { {}, "Feat Prior Pen" };

    juce::Label pcaLabel_ { {}, "Cluster Weight PCA (R^n -> R^3)" };
    PcaProjectionVisualizer pcaVisualizer_;
    inertia::PcaProjectionFrame pcaFrameSnapshot_;

    std::array<std::array<juce::Label, 6>, inertia::kMaxClusters> clusterStatLabels_;
    std::array<juce::Slider, inertia::kMaxClusters> clusterGainSliders_;
    std::array<juce::Slider, inertia::kMaxClusters> clusterHpSliders_;
    std::array<juce::Slider, inertia::kMaxClusters> clusterLpSliders_;
    std::array<juce::ToggleButton, inertia::kMaxClusters> clusterMuteButtons_;
    std::array<juce::ComboBox, inertia::kMaxClusters> clusterRouteBoxes_;
    std::array<juce::Label, inertia::kFeatureDim> featureWeightLabels_;
    std::array<juce::Label, inertia::kFeatureDim> featureWeightValues_;
    std::unique_ptr<ButtonAttachment> bypassAttachment_;
    std::unique_ptr<ButtonAttachment> freezeAttachment_;
    std::unique_ptr<ButtonAttachment> gateEnableAttachment_;
    std::unique_ptr<ButtonAttachment> autoLevelAttachment_;

    std::unique_ptr<ComboBoxAttachment> fftAttachment_;
    std::unique_ptr<ComboBoxAttachment> hopAttachment_;
    std::unique_ptr<ComboBoxAttachment> outputModeAttachment_;

    std::unique_ptr<SliderAttachment> clustersAttachment_;
    std::unique_ptr<SliderAttachment> updateHzAttachment_;
    std::unique_ptr<SliderAttachment> glideMsAttachment_;
    std::unique_ptr<SliderAttachment> spreadAttachment_;
    std::unique_ptr<SliderAttachment> distancePenaltyAttachment_;
    std::unique_ptr<SliderAttachment> smoothMsAttachment_;
    std::unique_ptr<SliderAttachment> mixAttachment_;
    std::unique_ptr<SliderAttachment> outputGainAttachment_;
    std::unique_ptr<SliderAttachment> saturationAttachment_;
    std::unique_ptr<SliderAttachment> driveFromLevelAttachment_;
    std::unique_ptr<SliderAttachment> gateThresholdAttachment_;
    std::unique_ptr<SliderAttachment> gateSharpnessAttachment_;
    std::unique_ptr<SliderAttachment> gateFloorAttachment_;
    std::unique_ptr<SliderAttachment> featureAdaptRateAttachment_;
    std::unique_ptr<SliderAttachment> featureLowPenaltyAttachment_;
    std::unique_ptr<SliderAttachment> featureHighPenaltyAttachment_;
    std::unique_ptr<SliderAttachment> featureNormPenaltyAttachment_;
    std::unique_ptr<SliderAttachment> featureMinimaxAttachment_;
    std::unique_ptr<SliderAttachment> featurePriorPenaltyAttachment_;

    std::array<std::unique_ptr<SliderAttachment>, inertia::kMaxClusters> clusterGainAttachments_;
    std::array<std::unique_ptr<SliderAttachment>, inertia::kMaxClusters> clusterHpAttachments_;
    std::array<std::unique_ptr<SliderAttachment>, inertia::kMaxClusters> clusterLpAttachments_;
    std::array<std::unique_ptr<ButtonAttachment>, inertia::kMaxClusters> clusterMuteAttachments_;
    std::array<std::unique_ptr<ComboBoxAttachment>, inertia::kMaxClusters> clusterRouteAttachments_;

    int visibleClusters_ = 4;
    bool advancedMode_ = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InertiaBandsAudioProcessorEditor)
};
