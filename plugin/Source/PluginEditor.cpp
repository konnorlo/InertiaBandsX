#include "PluginEditor.h"
#include <BinaryData.h>

#include <algorithm>
#include <cmath>

namespace
{
const auto kChalkText = juce::Colour::fromRGB(232, 245, 238);
const auto kChalkTextDim = juce::Colour::fromRGBA(232, 245, 238, 186);
const auto kPanelFill = juce::Colour::fromRGBA(12, 28, 24, 154);
const auto kPanelFillEmphasis = juce::Colour::fromRGBA(18, 37, 31, 172);
const auto kBoardBase = juce::Colour::fromRGB(20, 57, 49);
juce::Typeface::Ptr gLoadedChalkTypeface;

juce::Image loadImageFromBinary(const void* data, int size)
{
    if (data == nullptr || size <= 0)
        return {};

    return juce::ImageFileFormat::loadFrom(data, static_cast<size_t>(size));
}

juce::Font chalkLabelFont(float height)
{
    if (gLoadedChalkTypeface != nullptr)
        return juce::Font(juce::FontOptions(gLoadedChalkTypeface).withHeight(height));

    return juce::Font(juce::FontOptions("Chalkboard SE", height, juce::Font::plain));
}

juce::Font valueFont(float height)
{
    if (gLoadedChalkTypeface != nullptr)
        return juce::Font(juce::FontOptions(gLoadedChalkTypeface).withHeight(height));

    return juce::Font(juce::FontOptions("Menlo", height, juce::Font::plain));
}

juce::Image makeProceduralNoiseTexture(int width, int height, juce::Colour base, juce::Colour tint, float tintScale)
{
    juce::Image img(juce::Image::ARGB, width, height, true);
    juce::Image::BitmapData pixels(img, juce::Image::BitmapData::writeOnly);
    juce::Random rng(0x7041B3D);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const auto n = rng.nextFloat();
            const auto grain = juce::jmap(n, 0.0f, 1.0f, -tintScale, tintScale);
            auto c = base.interpolatedWith(tint, 0.45f + grain);
            c = c.withAlpha(0.25f + (0.35f * n));
            pixels.setPixelColour(x, y, c);
        }
    }

    return img;
}

juce::Image makeProceduralSmudgeTexture(int width, int height, juce::Colour c0, juce::Colour c1)
{
    juce::Image img(juce::Image::ARGB, width, height, true);
    juce::Graphics g(img);
    g.fillAll(juce::Colours::transparentBlack);

    juce::Random rng(0x44A11C5);
    for (int i = 0; i < 120; ++i)
    {
        const auto x = rng.nextFloat() * static_cast<float>(width);
        const auto y = rng.nextFloat() * static_cast<float>(height);
        const auto w = 20.0f + (rng.nextFloat() * 120.0f);
        const auto h = 2.0f + (rng.nextFloat() * 10.0f);
        const auto a = 0.05f + (rng.nextFloat() * 0.16f);
        g.setColour(c0.interpolatedWith(c1, rng.nextFloat()).withAlpha(a));
        g.fillRoundedRectangle(x, y, w, h, h * 0.5f);
    }

    return img;
}

void makeLabel(juce::Label& label)
{
    label.setJustificationType(juce::Justification::centredLeft);
    label.setColour(juce::Label::textColourId, kChalkText);
    label.setFont(chalkLabelFont(13.0f));
}

void layoutControl(juce::Rectangle<int> row, juce::Label& label, juce::Slider& slider)
{
    label.setBounds(row.removeFromLeft(96));
    slider.setBounds(row);
}

juce::Colour clusterColour(int index)
{
    static constexpr std::array<juce::uint32, inertia::kMaxClusters> colours {
        0xff76e6cf, // mint
        0xff9fc8ff, // powder blue
        0xffffb06d, // amber coral
        0xffd7b7ff, // lavender
        0xffa8f0a0,
        0xffffd889,
        0xffff9fb2,
        0xffa6d8d8
    };

    return juce::Colour(colours[static_cast<size_t>(inertia::clamp(index, 0, inertia::kMaxClusters - 1))]);
}

const char* clusterRoleName(int role)
{
    const auto clamped = inertia::clamp(role, 0, static_cast<int>(inertia::kClusterRoleNames.size() - 1));
    return inertia::kClusterRoleNames[static_cast<size_t>(clamped)];
}

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 rotateForView(const Vec3& v, float yaw, float pitch)
{
    const auto cy = std::cos(yaw);
    const auto sy = std::sin(yaw);
    const auto cp = std::cos(pitch);
    const auto sp = std::sin(pitch);

    const auto x1 = (cy * v.x) + (sy * v.z);
    const auto z1 = (-sy * v.x) + (cy * v.z);

    const auto y2 = (cp * v.y) - (sp * z1);
    const auto z2 = (sp * v.y) + (cp * z1);

    return { x1, y2, z2 };
}

juce::Point<float> projectPoint(const Vec3& p, const juce::Rectangle<float>& plot)
{
    constexpr float cameraDistance = 3.4f;
    const auto z = std::max(0.15f, p.z + cameraDistance);
    const auto perspective = cameraDistance / z;
    const auto scale = 0.44f * std::min(plot.getWidth(), plot.getHeight());

    return {
        plot.getCentreX() + (p.x * scale * perspective),
        plot.getCentreY() - (p.y * scale * perspective)
    };
}

juce::Colour pc3Colour(float z)
{
    const auto t = inertia::clamp(0.5f * (z + 1.0f), 0.0f, 1.0f);
    const auto cool = juce::Colour::fromRGB(111, 165, 255);
    const auto warm = juce::Colour::fromRGB(255, 148, 91);
    return cool.interpolatedWith(warm, t);
}

void drawRoughBorder(juce::Graphics& g, juce::Rectangle<float> bounds, float jitter, juce::Colour c)
{
    juce::Path p;
    const auto x0 = bounds.getX();
    const auto y0 = bounds.getY();
    const auto x1 = bounds.getRight();
    const auto y1 = bounds.getBottom();
    const auto j = jitter;

    p.startNewSubPath(x0 + j, y0 + (0.7f * j));
    p.lineTo(x1 - (0.5f * j), y0 - (0.3f * j));
    p.lineTo(x1 + (0.4f * j), y1 - (0.6f * j));
    p.lineTo(x0 - (0.6f * j), y1 + (0.5f * j));
    p.closeSubPath();

    g.setColour(c);
    g.strokePath(p, juce::PathStrokeType(1.2f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
}

void drawStarPoint(juce::Graphics& g, juce::Point<float> point, float radius, juce::Colour base, float alpha, int rays)
{
    const auto haloRadius = radius * 2.4f;
    juce::ColourGradient gradient(base.withAlpha(0.36f * alpha),
                                  point.x,
                                  point.y,
                                  base.withAlpha(0.0f),
                                  point.x + haloRadius,
                                  point.y + haloRadius,
                                  true);
    gradient.addColour(0.45, base.withAlpha(0.18f * alpha));
    g.setGradientFill(gradient);
    g.fillEllipse(point.x - haloRadius, point.y - haloRadius, haloRadius * 2.0f, haloRadius * 2.0f);

    g.setColour(base.withAlpha(alpha));
    g.fillEllipse(point.x - radius, point.y - radius, radius * 2.0f, radius * 2.0f);

    g.setColour(juce::Colours::white.withAlpha(0.8f * alpha));
    g.fillEllipse(point.x - (radius * 0.35f), point.y - (radius * 0.35f), radius * 0.7f, radius * 0.7f);

    g.setColour(base.brighter(0.30f).withAlpha(0.34f * alpha));
    for (int r = 0; r < rays; ++r)
    {
        const auto a = (juce::MathConstants<float>::twoPi * static_cast<float>(r)) / static_cast<float>(std::max(1, rays));
        const auto inner = radius * 0.75f;
        const auto outer = radius * 1.95f;
        g.drawLine(point.x + std::cos(a) * inner,
                   point.y + std::sin(a) * inner,
                   point.x + std::cos(a) * outer,
                   point.y + std::sin(a) * outer,
                   1.0f);
    }
}
} // namespace

juce::Font ChalkLookAndFeel::getLabelFont(juce::Label& label)
{
    if (dynamic_cast<juce::Slider*> (label.getParentComponent()) != nullptr
        || dynamic_cast<juce::ComboBox*> (label.getParentComponent()) != nullptr)
        return valueFont(13.0f);

    return chalkLabelFont(12.5f);
}

void ChalkLookAndFeel::setChalkTypeface(juce::Typeface::Ptr typeface)
{
    chalkTypeface_ = std::move(typeface);
    gLoadedChalkTypeface = chalkTypeface_;
}

juce::Typeface::Ptr ChalkLookAndFeel::getTypefaceForFont(const juce::Font& font)
{
    if (chalkTypeface_ != nullptr)
        return chalkTypeface_;

    return juce::LookAndFeel_V4::getTypefaceForFont(font);
}

void ChalkLookAndFeel::drawLinearSlider(juce::Graphics& g,
                                        int x,
                                        int y,
                                        int width,
                                        int height,
                                        float sliderPos,
                                        float,
                                        float,
                                        juce::Slider::SliderStyle,
                                        juce::Slider& slider)
{
    const auto area = juce::Rectangle<float>(static_cast<float>(x),
                                             static_cast<float>(y),
                                             static_cast<float>(width),
                                             static_cast<float>(height));

    const auto rail = area.withCentre(area.getCentre()).withHeight(std::max(3.0f, area.getHeight() * 0.20f)).reduced(2.0f, 0.0f);
    g.setColour(juce::Colour::fromRGBA(222, 238, 229, 62));
    g.fillRoundedRectangle(rail, 2.0f);

    const auto enabled = slider.isEnabled();
    const auto thumbColour = enabled ? clusterColour(0) : juce::Colours::grey;
    const auto minX = rail.getX();
    const auto maxX = rail.getRight();
    const auto filled = rail.withRight(juce::jlimit(minX, maxX, sliderPos));

    g.setColour(thumbColour.withAlpha(0.70f));
    g.fillRoundedRectangle(filled, 2.2f);

    juce::Path nib;
    const auto r = std::max(4.0f, rail.getHeight() * 1.6f);
    const auto cx = juce::jlimit(minX, maxX, sliderPos);
    const auto cy = rail.getCentreY();
    nib.addEllipse(cx - r, cy - r, r * 2.0f, r * 2.0f);
    g.setColour(thumbColour.withAlpha(enabled ? 0.95f : 0.45f));
    g.fillPath(nib);
    g.setColour(kChalkText.withAlpha(0.65f));
    g.strokePath(nib, juce::PathStrokeType(1.0f));
}

void ChalkLookAndFeel::drawToggleButton(juce::Graphics& g,
                                        juce::ToggleButton& button,
                                        bool shouldDrawButtonAsHighlighted,
                                        bool)
{
    auto area = button.getLocalBounds().toFloat();
    const auto compactText = button.getButtonText().trim().length() <= 2;
    const auto box = compactText
        ? area.removeFromLeft(18.0f).withSizeKeepingCentre(14.0f, 14.0f)
        : area.removeFromRight(18.0f).withSizeKeepingCentre(14.0f, 14.0f);

    g.setColour(juce::Colour::fromRGBA(18, 38, 33, 180));
    g.fillRoundedRectangle(box, 2.0f);
    drawRoughBorder(g, box, shouldDrawButtonAsHighlighted ? 1.2f : 0.8f, kChalkText.withAlpha(0.65f));

    if (button.getToggleState())
    {
        juce::Path tick;
        tick.startNewSubPath(box.getX() + 2.0f, box.getCentreY());
        tick.lineTo(box.getX() + 5.0f, box.getBottom() - 2.0f);
        tick.lineTo(box.getRight() - 2.0f, box.getY() + 2.0f);
        g.setColour(juce::Colour::fromRGB(127, 232, 204));
        g.strokePath(tick, juce::PathStrokeType(1.6f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }

    g.setColour(kChalkText.withAlpha(button.isEnabled() ? 0.94f : 0.45f));
    g.setFont(chalkLabelFont(13.0f));
    auto textBounds = area.toNearestInt();
    if (compactText)
        textBounds = textBounds.withTrimmedLeft(20);
    else
        textBounds = textBounds.withTrimmedRight(4).translated(24, 0);

    g.drawFittedText(button.getButtonText(), textBounds, juce::Justification::centredLeft, 1);
}

void ChalkLookAndFeel::drawComboBox(juce::Graphics& g,
                                    int width,
                                    int height,
                                    bool isButtonDown,
                                    int buttonX,
                                    int buttonY,
                                    int buttonW,
                                    int buttonH,
                                    juce::ComboBox& box)
{
    juce::ignoreUnused(isButtonDown);

    const auto bounds = juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
    g.setColour(box.findColour(juce::ComboBox::backgroundColourId));
    g.fillRoundedRectangle(bounds, 3.0f);
    drawRoughBorder(g, bounds.reduced(0.5f), 0.8f, kChalkText.withAlpha(0.55f));

    auto arrow = juce::Path {};
    const auto arrowBounds = juce::Rectangle<float>(static_cast<float>(buttonX),
                                                    static_cast<float>(buttonY),
                                                    static_cast<float>(buttonW),
                                                    static_cast<float>(buttonH)).reduced(5.0f, 6.0f);
    arrow.startNewSubPath(arrowBounds.getX(), arrowBounds.getY());
    arrow.lineTo(arrowBounds.getCentreX(), arrowBounds.getBottom());
    arrow.lineTo(arrowBounds.getRight(), arrowBounds.getY());

    g.setColour(kChalkTextDim);
    g.strokePath(arrow, juce::PathStrokeType(1.3f));
}

void ChalkLookAndFeel::drawPopupMenuItem(juce::Graphics& g,
                                         const juce::Rectangle<int>& area,
                                         bool isSeparator,
                                         bool isActive,
                                         bool isHighlighted,
                                         bool isTicked,
                                         bool hasSubMenu,
                                         const juce::String& text,
                                         const juce::String& shortcutKeyText,
                                         const juce::Drawable* icon,
                                         const juce::Colour* textColourToUse)
{
    juce::ignoreUnused(isTicked, hasSubMenu, shortcutKeyText, icon);

    if (isSeparator)
    {
        g.setColour(juce::Colour::fromRGBA(30, 30, 30, 50));
        g.drawHorizontalLine(area.getCentreY(), static_cast<float>(area.getX() + 4), static_cast<float>(area.getRight() - 4));
        return;
    }

    const auto row = area.toFloat().reduced(2.0f, 1.0f);
    if (isHighlighted)
    {
        g.setColour(juce::Colour::fromRGBA(125, 177, 158, 145));
        g.fillRoundedRectangle(row, 3.0f);
    }

    auto textColour = juce::Colour::fromRGB(28, 24, 19);
    if (textColourToUse != nullptr)
        textColour = *textColourToUse;

    if (! isActive)
        textColour = textColour.withAlpha(0.45f);

    g.setColour(textColour);
    g.setFont(valueFont(13.0f));
    g.drawFittedText(text, area.reduced(9, 0), juce::Justification::centredLeft, 1);
}

void ChalkLookAndFeel::positionComboBoxText(juce::ComboBox& box, juce::Label& label)
{
    label.setBounds(1, 1, box.getWidth() - 22, box.getHeight() - 2);
    label.setFont(valueFont(13.0f));
    label.setJustificationType(juce::Justification::centredLeft);
    label.setColour(juce::Label::textColourId, box.findColour(juce::ComboBox::textColourId).withAlpha(1.0f));
}

void ChalkLookAndFeel::drawPopupMenuBackground(juce::Graphics& g, int width, int height)
{
    const auto b = juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
    g.setColour(juce::Colour::fromRGBA(239, 232, 210, 246));
    g.fillRoundedRectangle(b, 5.0f);
    g.setColour(juce::Colour::fromRGBA(87, 68, 49, 160));
    g.drawRoundedRectangle(b.reduced(0.8f), 5.0f, 1.0f);
}

void PcaProjectionVisualizer::setFrame(const inertia::PcaProjectionFrame& frame)
{
    const auto nowSeconds = juce::Time::getMillisecondCounterHiRes() * 0.001;
    if (lastUpdateSeconds_ <= 0.0)
        lastUpdateSeconds_ = nowSeconds;

    const auto dt = inertia::clamp(static_cast<float>(nowSeconds - lastUpdateSeconds_), 0.0f, 0.10f);
    lastUpdateSeconds_ = nowSeconds;

    if (! dragging_)
    {
        constexpr float kAutoRotateHz = 0.008f;
        constexpr float twoPi = juce::MathConstants<float>::twoPi;
        autoYaw_ += dt * twoPi * kAutoRotateHz;
    }

    frame_ = frame;
    if (hasSmoothedFrame_)
    {
        previousSmoothedFrame_ = smoothedFrame_;
        hasPreviousFrame_ = true;
    }

    if (! hasSmoothedFrame_
        || smoothedFrame_.numPoints != frame.numPoints
        || smoothedFrame_.numClusters != frame.numClusters)
    {
        smoothedFrame_ = frame;
        hasSmoothedFrame_ = true;
        repaint();
        return;
    }

    constexpr float alpha = 0.30f;

    smoothedFrame_.numPoints = frame.numPoints;
    smoothedFrame_.numDims = frame.numDims;
    smoothedFrame_.numClusters = frame.numClusters;

    const auto points = inertia::clamp(frame.numPoints, 0, inertia::kMaxPerceptualBins);
    const auto clusters = inertia::clamp(frame.numClusters, 0, inertia::kMaxClusters);

    for (int i = 0; i < points; ++i)
    {
        for (int d = 0; d < 3; ++d)
            smoothedFrame_.points[i][d] += alpha * (frame.points[i][d] - smoothedFrame_.points[i][d]);

        smoothedFrame_.dominance[i] += alpha * (frame.dominance[i] - smoothedFrame_.dominance[i]);
        smoothedFrame_.dominantCluster[i] = frame.dominantCluster[i];
    }

    for (int c = 0; c < clusters; ++c)
    {
        for (int d = 0; d < 3; ++d)
            smoothedFrame_.clusterCenters[c][d] += alpha * (frame.clusterCenters[c][d] - smoothedFrame_.clusterCenters[c][d]);

        smoothedFrame_.clusterActivity[c] += alpha * (frame.clusterActivity[c] - smoothedFrame_.clusterActivity[c]);
    }

    repaint();
}

void PcaProjectionVisualizer::setResponseFrame(const InertiaBandsAudioProcessor::ResponseFrame& frame) noexcept
{
    responseFrame_ = frame;

    if (! hasSmoothedResponse_
        || smoothedResponseFrame_.fftBins != frame.fftBins
        || smoothedResponseFrame_.numClusters != frame.numClusters)
    {
        smoothedResponseFrame_ = frame;
        hasSmoothedResponse_ = true;
        return;
    }

    constexpr float alpha = 0.45f;
    const auto bins = inertia::clamp(frame.fftBins, 0, inertia::kMaxFftBins);

    smoothedResponseFrame_.fftBins = frame.fftBins;
    smoothedResponseFrame_.fftSize = frame.fftSize;
    smoothedResponseFrame_.sampleRate = frame.sampleRate;
    smoothedResponseFrame_.numClusters = frame.numClusters;
    smoothedResponseFrame_.waveformSamples = frame.waveformSamples;
    smoothedResponseFrame_.waveform = frame.waveform;
    smoothedResponseFrame_.featureWeights = frame.featureWeights;

    for (int f = 0; f < bins; ++f)
        smoothedResponseFrame_.mixedGain[f] += alpha * (frame.mixedGain[f] - smoothedResponseFrame_.mixedGain[f]);

    repaint();
}

float PcaProjectionVisualizer::getViewYaw() const noexcept
{
    return autoYaw_ + userYaw_;
}

float PcaProjectionVisualizer::getViewPitch() const noexcept
{
    return inertia::clamp(userPitch_, -1.2f, 1.2f);
}

void PcaProjectionVisualizer::mouseDown(const juce::MouseEvent& event)
{
    dragging_ = true;
    lastMousePos_ = event.getPosition();
}

void PcaProjectionVisualizer::mouseDrag(const juce::MouseEvent& event)
{
    const auto pos = event.getPosition();
    const auto delta = pos - lastMousePos_;
    lastMousePos_ = pos;

    userYaw_ -= static_cast<float>(delta.x) * 0.010f;
    userPitch_ -= static_cast<float>(delta.y) * 0.008f;
    userPitch_ = inertia::clamp(userPitch_, -1.2f, 1.2f);
    repaint();
}

void PcaProjectionVisualizer::mouseUp(const juce::MouseEvent&)
{
    dragging_ = false;
}

void PcaProjectionVisualizer::paint(juce::Graphics& g)
{
    const auto bounds = getLocalBounds().toFloat();

    g.setColour(juce::Colour::fromRGBA(12, 32, 26, 148));
    g.fillRoundedRectangle(bounds, 6.0f);
    drawRoughBorder(g, bounds.reduced(0.5f), 0.9f, kChalkText.withAlpha(0.34f));

    auto plot = bounds.reduced(12.0f);

    const auto& eqFrame = hasSmoothedResponse_ ? smoothedResponseFrame_ : responseFrame_;
    if (eqFrame.waveformSamples > 1)
    {
        juce::Path waveformPath;
        const auto count = inertia::clamp(eqFrame.waveformSamples, 2, InertiaBandsAudioProcessor::kUiWaveformSamples);

        auto mapWaveToY = [&plot](float v)
        {
            return juce::jmap(inertia::clamp(v, -1.0f, 1.0f), 1.0f, -1.0f, plot.getY() + 5.0f, plot.getBottom() - 5.0f);
        };

        for (int i = 0; i < count; ++i)
        {
            const auto t = static_cast<float>(i) / static_cast<float>(std::max(1, count - 1));
            const auto x = juce::jmap(t, 0.0f, 1.0f, plot.getX(), plot.getRight());
            const auto y = mapWaveToY(eqFrame.waveform[static_cast<size_t>(i)]);
            if (i == 0)
                waveformPath.startNewSubPath(x, y);
            else
                waveformPath.lineTo(x, y);
        }

        g.setColour(kChalkText.withAlpha(0.08f));
        for (int i = 0; i < 7; ++i)
        {
            const auto y = juce::jmap(static_cast<float>(i), 0.0f, 6.0f, plot.getY(), plot.getBottom());
            g.drawHorizontalLine(static_cast<int>(y), plot.getX(), plot.getRight());
        }

        g.setColour(kChalkText.withAlpha(0.19f));
        g.strokePath(waveformPath, juce::PathStrokeType(1.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }

    g.setColour(kChalkText.withAlpha(0.10f));
    g.drawLine(plot.getX(), plot.getCentreY(), plot.getRight(), plot.getCentreY(), 1.0f);
    g.drawLine(plot.getCentreX(), plot.getY(), plot.getCentreX(), plot.getBottom(), 1.0f);

    const auto& drawFrame = hasSmoothedFrame_ ? smoothedFrame_ : frame_;
    const auto yaw = getViewYaw();
    const auto pitch = getViewPitch();

    if (drawFrame.numPoints <= 0)
    {
        g.setColour(kChalkText.withAlpha(0.62f));
        g.setFont(chalkLabelFont(13.0f));
        g.drawFittedText("Awaiting PCA data...", plot.toNearestInt(), juce::Justification::centred, 1);
        return;
    }

    const auto pointCount = inertia::clamp(drawFrame.numPoints, 1, inertia::kMaxPerceptualBins);
    const auto clusterCount = inertia::clamp(drawFrame.numClusters, 1, inertia::kMaxClusters);

    // Draw a projected 3D bounding cube.
    const std::array<Vec3, 8> corners {
        Vec3 { -1.0f, -1.0f, -1.0f },
        Vec3 {  1.0f, -1.0f, -1.0f },
        Vec3 {  1.0f,  1.0f, -1.0f },
        Vec3 { -1.0f,  1.0f, -1.0f },
        Vec3 { -1.0f, -1.0f,  1.0f },
        Vec3 {  1.0f, -1.0f,  1.0f },
        Vec3 {  1.0f,  1.0f,  1.0f },
        Vec3 { -1.0f,  1.0f,  1.0f }
    };

    const std::array<std::array<int, 2>, 12> edges {{
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
    }};

    std::array<juce::Point<float>, 8> projectedCorners {};
    for (int i = 0; i < 8; ++i)
        projectedCorners[i] = projectPoint(rotateForView(corners[i], yaw, pitch), plot);

    g.setColour(kChalkText.withAlpha(0.20f));
    for (const auto& edge : edges)
    {
        g.drawLine(projectedCorners[edge[0]].x,
                   projectedCorners[edge[0]].y,
                   projectedCorners[edge[1]].x,
                   projectedCorners[edge[1]].y,
                   1.0f + (0.25f * std::sin((0.3f * projectedCorners[edge[0]].x) + (0.2f * projectedCorners[edge[1]].y))));
    }

    std::array<int, inertia::kMaxPerceptualBins> drawOrder {};
    for (int i = 0; i < pointCount; ++i)
        drawOrder[i] = i;

    std::sort(drawOrder.begin(), drawOrder.begin() + pointCount, [&drawFrame](int a, int b)
    {
        return drawFrame.points[a][2] < drawFrame.points[b][2];
    });

    for (int order = 0; order < pointCount; ++order)
    {
        const auto i = drawOrder[order];
        const auto x = inertia::clamp(drawFrame.points[i][0], -1.0f, 1.0f);
        const auto y = inertia::clamp(drawFrame.points[i][1], -1.0f, 1.0f);
        const auto z = inertia::clamp(drawFrame.points[i][2], -1.0f, 1.0f);
        const auto projected = projectPoint(rotateForView({ x, y, z }, yaw, pitch), plot);

        const auto radius = 2.0f + (z + 1.0f) * 2.0f;
        const auto alpha = 0.28f + 0.72f * inertia::clamp(drawFrame.dominance[i], 0.0f, 1.0f);
        const auto clusterBase = clusterColour(drawFrame.dominantCluster[i]);
        const auto colour = clusterBase.interpolatedWith(pc3Colour(z), 0.45f).withAlpha(alpha);

        drawStarPoint(g, projected, radius, colour, alpha, 6);
    }

    float maxActivity = 0.0f;
    for (int c = 0; c < clusterCount; ++c)
        maxActivity = std::max(maxActivity, drawFrame.clusterActivity[c]);
    maxActivity = std::max(maxActivity, inertia::kEpsilon);

    for (int c = 0; c < clusterCount; ++c)
    {
        const auto center = drawFrame.clusterCenters[c];
        const auto z = inertia::clamp(center[2], -1.0f, 1.0f);
        const auto projected = projectPoint(rotateForView({ center[0], center[1], center[2] }, yaw, pitch), plot);
        const auto activityNorm = inertia::clamp(drawFrame.clusterActivity[c] / maxActivity, 0.0f, 1.0f);
        const auto radius = 5.5f + (3.0f * activityNorm);
        const auto color = clusterColour(c).interpolatedWith(pc3Colour(z), 0.35f).withAlpha(0.94f);

        if (hasPreviousFrame_ && c < previousSmoothedFrame_.numClusters)
        {
            const auto prev = previousSmoothedFrame_.clusterCenters[c];
            const auto prevProjected = projectPoint(rotateForView({ prev[0], prev[1], prev[2] }, yaw, pitch), plot);
            g.setColour(color.withAlpha(0.30f));
            g.drawLine(prevProjected.x, prevProjected.y, projected.x, projected.y, 1.3f);
        }

        drawStarPoint(g, projected, radius, color, 0.95f, 8);
        g.setColour(kChalkText.withAlpha(0.7f));
        g.drawEllipse(projected.x - radius, projected.y - radius, radius * 2.0f, radius * 2.0f, 1.1f);
    }

    auto keyRow = plot.removeFromBottom(16.0f);
    if (clusterCount > 0)
    {
        const auto cellWidth = keyRow.getWidth() / static_cast<float>(clusterCount);
        g.setFont(valueFont(9.0f));
        for (int c = 0; c < clusterCount; ++c)
        {
            auto cell = keyRow.withX(keyRow.getX() + (cellWidth * static_cast<float>(c))).withWidth(cellWidth);
            g.setColour(clusterColour(c).withAlpha(1.0f));
            g.drawFittedText("C" + juce::String(c + 1), cell.toNearestInt(), juce::Justification::centred, 1);
        }
    }

    g.setColour(kChalkTextDim.withAlpha(0.70f));
    g.setFont(chalkLabelFont(10.0f));
    g.drawText("Drag to orbit", plot.removeFromBottom(14.0f).toNearestInt(), juce::Justification::centredLeft);
}

InertiaBandsAudioProcessorEditor::InertiaBandsAudioProcessorEditor(InertiaBandsAudioProcessor& audioProcessor)
    : AudioProcessorEditor(&audioProcessor),
      processor_(audioProcessor),
      apvts_(audioProcessor.getValueTreeState())
{
    loadThemeAssets();
    setLookAndFeel(&lookAndFeel_);

    {
        customTypeface_ = juce::Typeface::createSystemTypefaceFor(BinaryData::StardosStencilBold_ttf,
                                                                   static_cast<size_t>(BinaryData::StardosStencilBold_ttfSize));
        if (customTypeface_ == nullptr)
        {
            customTypeface_ = juce::Typeface::createSystemTypefaceFor(BinaryData::StardosStencilRegular_ttf,
                                                                       static_cast<size_t>(BinaryData::StardosStencilRegular_ttfSize));
        }

        if (customTypeface_ == nullptr)
        {
            for (int i = 0; i < BinaryData::namedResourceListSize; ++i)
            {
                const juce::String name(BinaryData::namedResourceList[i]);
                if (! name.containsIgnoreCase("stardos"))
                    continue;

                int dataSize = 0;
                if (const auto* data = BinaryData::getNamedResource(name.toRawUTF8(), dataSize);
                    data != nullptr && dataSize > 0)
                {
                    customTypeface_ = juce::Typeface::createSystemTypefaceFor(data, static_cast<size_t>(dataSize));
                    if (customTypeface_ != nullptr)
                        break;
                }
            }
        }

        lookAndFeel_.setChalkTypeface(customTypeface_);
    }

    setSize(960, 720);

    fftSizeBox_.addItem("1024", 1);
    fftSizeBox_.addItem("2048", 2);
    fftSizeBox_.addItem("512", 3);
    fftSizeBox_.addItem("4096", 4);

    hopModeBox_.addItem("N/2", 1);
    hopModeBox_.addItem("N/4", 2);

    outputModeBox_.addItem("Mix", 1);
    for (int cluster = 0; cluster < inertia::kMaxClusters; ++cluster)
        outputModeBox_.addItem("C" + juce::String(cluster + 1), cluster + 2);

    for (auto* box : { &fftSizeBox_, &hopModeBox_, &outputModeBox_ })
    {
        box->setColour(juce::ComboBox::textColourId, juce::Colours::white);
        box->setColour(juce::ComboBox::outlineColourId, juce::Colours::transparentBlack);
        box->setColour(juce::ComboBox::backgroundColourId, juce::Colour::fromRGBA(15, 37, 31, 210));
        box->setColour(juce::PopupMenu::textColourId, juce::Colour::fromRGB(34, 31, 25));
        box->setColour(juce::PopupMenu::highlightedTextColourId, juce::Colour::fromRGB(18, 18, 18));
        box->setColour(juce::PopupMenu::highlightedBackgroundColourId, juce::Colour::fromRGBA(130, 180, 160, 110));
    }

    addAndMakeVisible(bypassButton_);
    addAndMakeVisible(freezeButton_);
    addAndMakeVisible(gateEnableButton_);
    addAndMakeVisible(autoLevelButton_);
    addAndMakeVisible(advancedModeButton_);
    addAndMakeVisible(fftSizeBox_);
    addAndMakeVisible(hopModeBox_);
    addAndMakeVisible(outputModeBox_);

    makeLabel(fftLabel_);
    makeLabel(hopLabel_);
    makeLabel(outputModeLabel_);
    addAndMakeVisible(fftLabel_);
    addAndMakeVisible(hopLabel_);
    addAndMakeVisible(outputModeLabel_);

    makeLabel(pcaLabel_);
    pcaLabel_.setJustificationType(juce::Justification::centredLeft);
    pcaLabel_.setVisible(false);
    addAndMakeVisible(pcaLabel_);
    addAndMakeVisible(pcaVisualizer_);
    pcaVisualizer_.setOpaque(false);

    advancedModeButton_.setToggleState(false, juce::dontSendNotification);
    advancedModeButton_.setClickingTogglesState(true);
    advancedModeButton_.onClick = [this]
    {
        advancedMode_ = advancedModeButton_.getToggleState();
        applyModeVisibility();
        resized();
        repaint();
    };

    bypassAttachment_ = std::make_unique<ButtonAttachment>(apvts_, inertia::ParamIDs::bypass, bypassButton_);
    freezeAttachment_ = std::make_unique<ButtonAttachment>(apvts_, inertia::ParamIDs::freeze, freezeButton_);
    gateEnableAttachment_ = std::make_unique<ButtonAttachment>(apvts_, inertia::ParamIDs::gateEnable, gateEnableButton_);
    autoLevelAttachment_ = std::make_unique<ButtonAttachment>(apvts_, inertia::ParamIDs::autoLevel, autoLevelButton_);

    fftAttachment_ = std::make_unique<ComboBoxAttachment>(apvts_, inertia::ParamIDs::fftSize, fftSizeBox_);
    hopAttachment_ = std::make_unique<ComboBoxAttachment>(apvts_, inertia::ParamIDs::hopMode, hopModeBox_);
    outputModeAttachment_ = std::make_unique<ComboBoxAttachment>(apvts_, inertia::ParamIDs::outputMode, outputModeBox_);

    const auto addControl = [this](juce::Label& label, juce::Slider& slider, const juce::String& suffix)
    {
        makeLabel(label);
        configureSlider(slider, suffix);
        addAndMakeVisible(label);
        addAndMakeVisible(slider);
    };

    addControl(clustersLabel_, clustersSlider_, "");
    addControl(updateHzLabel_, updateHzSlider_, " Hz");
    addControl(glideMsLabel_, glideMsSlider_, " ms");
    addControl(spreadLabel_, spreadSlider_, "");
    addControl(distancePenaltyLabel_, distancePenaltySlider_, "");
    addControl(smoothMsLabel_, smoothMsSlider_, " ms");
    addControl(mixLabel_, mixSlider_, "");
    addControl(outputGainLabel_, outputGainSlider_, " dB");
    addControl(saturationLabel_, saturationSlider_, "");
    addControl(driveFromLevelLabel_, driveFromLevelSlider_, "");
    addControl(gateThresholdLabel_, gateThresholdSlider_, "");
    addControl(gateSharpnessLabel_, gateSharpnessSlider_, "");
    addControl(gateFloorLabel_, gateFloorSlider_, "");
    addControl(featureAdaptRateLabel_, featureAdaptRateSlider_, "");
    addControl(featureLowPenaltyLabel_, featureLowPenaltySlider_, "");
    addControl(featureHighPenaltyLabel_, featureHighPenaltySlider_, "");
    addControl(featureNormPenaltyLabel_, featureNormPenaltySlider_, "");
    addControl(featureMinimaxLabel_, featureMinimaxSlider_, "");
    addControl(featurePriorPenaltyLabel_, featurePriorPenaltySlider_, "");

    clustersAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::numClusters, clustersSlider_);
    updateHzAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::clusterUpdateHz, updateHzSlider_);
    glideMsAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::glideMs, glideMsSlider_);
    spreadAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::clusterSpread, spreadSlider_);
    distancePenaltyAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::distancePenalty, distancePenaltySlider_);
    smoothMsAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::maskSmoothMs, smoothMsSlider_);
    mixAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::globalMix, mixSlider_);
    outputGainAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::outputGainDb, outputGainSlider_);
    saturationAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::saturationAmount, saturationSlider_);
    driveFromLevelAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::driveFromLevel, driveFromLevelSlider_);
    gateThresholdAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::gateThreshold, gateThresholdSlider_);
    gateSharpnessAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::gateSharpness, gateSharpnessSlider_);
    gateFloorAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::gateFloor, gateFloorSlider_);
    featureAdaptRateAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::featureAdaptRate, featureAdaptRateSlider_);
    featureLowPenaltyAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::featureLowPenalty, featureLowPenaltySlider_);
    featureHighPenaltyAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::featureHighPenalty, featureHighPenaltySlider_);
    featureNormPenaltyAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::featureNormPenalty, featureNormPenaltySlider_);
    featureMinimaxAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::featureMinimax, featureMinimaxSlider_);
    featurePriorPenaltyAttachment_ = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::featurePriorPenalty, featurePriorPenaltySlider_);

    for (int i = 0; i < inertia::kFeatureDim; ++i)
    {
        auto& name = featureWeightLabels_[i];
        auto& value = featureWeightValues_[i];

        name.setText("W " + juce::String(inertia::kFeatureNames[static_cast<size_t>(i)]), juce::dontSendNotification);
        makeLabel(name);

        value.setJustificationType(juce::Justification::centredRight);
        value.setFont(valueFont(12.5f));
        value.setEditable(false, false, false);
        value.setText("0.00", juce::dontSendNotification);
        value.setColour(juce::Label::textColourId, kChalkText.withAlpha(0.96f));
        value.setColour(juce::Label::backgroundColourId, juce::Colour::fromRGBA(12, 30, 25, 180));
        value.setColour(juce::Label::outlineColourId, juce::Colour::fromRGBA(230, 244, 236, 96));
        value.setBorderSize(juce::BorderSize<int>(1));

        addAndMakeVisible(name);
        addAndMakeVisible(value);
    }

    for (int cluster = 0; cluster < inertia::kMaxClusters; ++cluster)
    {
        auto& statLabels = clusterStatLabels_[cluster];
        static constexpr std::array<const char*, 6> defaults { "C1", "Texture", "0Hz", "Ton 0.00", "Tr 0.00", "Act 0%" };
        for (size_t i = 0; i < statLabels.size(); ++i)
        {
            auto& l = statLabels[i];
            l.setText(defaults[i], juce::dontSendNotification);
            makeLabel(l);
            l.setFont(i == 0 ? valueFont(13.0f) : valueFont(12.0f));
            addAndMakeVisible(l);
        }

        configureSlider(clusterGainSliders_[cluster], " dB");
        configureSlider(clusterHpSliders_[cluster], " Hz");
        configureSlider(clusterLpSliders_[cluster], " Hz");

        clusterGainSliders_[cluster].setTooltip("Cluster Gain");
        clusterHpSliders_[cluster].setTooltip("Cluster HP");
        clusterLpSliders_[cluster].setTooltip("Cluster LP");
        clusterMuteButtons_[cluster].setButtonText("M");
        clusterMuteButtons_[cluster].setTooltip("Cluster Mute");

        clusterRouteBoxes_[cluster].addItem("Main", 1);
        for (int bus = 0; bus < inertia::kMaxAuxOutputs; ++bus)
            clusterRouteBoxes_[cluster].addItem("Aux " + juce::String(bus + 1), bus + 2);
        clusterRouteBoxes_[cluster].setTooltip("Cluster Route");
        clusterRouteBoxes_[cluster].setColour(juce::ComboBox::textColourId, juce::Colours::white);
        clusterRouteBoxes_[cluster].setColour(juce::ComboBox::outlineColourId, juce::Colours::transparentBlack);
        clusterRouteBoxes_[cluster].setColour(juce::ComboBox::backgroundColourId, juce::Colour::fromRGBA(20, 45, 38, 224));
        clusterRouteBoxes_[cluster].setColour(juce::PopupMenu::textColourId, juce::Colour::fromRGB(34, 31, 25));
        clusterRouteBoxes_[cluster].setColour(juce::PopupMenu::highlightedTextColourId, juce::Colour::fromRGB(18, 18, 18));
        clusterRouteBoxes_[cluster].setColour(juce::PopupMenu::highlightedBackgroundColourId, juce::Colour::fromRGBA(130, 180, 160, 110));

        addAndMakeVisible(clusterGainSliders_[cluster]);
        addAndMakeVisible(clusterHpSliders_[cluster]);
        addAndMakeVisible(clusterLpSliders_[cluster]);
        addAndMakeVisible(clusterMuteButtons_[cluster]);
        addAndMakeVisible(clusterRouteBoxes_[cluster]);

        clusterGainAttachments_[cluster] = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::clusterGainDb(cluster), clusterGainSliders_[cluster]);
        clusterHpAttachments_[cluster] = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::clusterHpHz(cluster), clusterHpSliders_[cluster]);
        clusterLpAttachments_[cluster] = std::make_unique<SliderAttachment>(apvts_, inertia::ParamIDs::clusterLpHz(cluster), clusterLpSliders_[cluster]);
        clusterMuteAttachments_[cluster] = std::make_unique<ButtonAttachment>(apvts_, inertia::ParamIDs::clusterMute(cluster), clusterMuteButtons_[cluster]);
        clusterRouteAttachments_[cluster] = std::make_unique<ComboBoxAttachment>(apvts_, inertia::ParamIDs::clusterRoute(cluster), clusterRouteBoxes_[cluster]);
    }

    applyModeVisibility();
    startTimerHz(60);
    visibleClusters_ = -1;
    updateClusterVisibility();
}

InertiaBandsAudioProcessorEditor::~InertiaBandsAudioProcessorEditor()
{
    setLookAndFeel(nullptr);
}

void InertiaBandsAudioProcessorEditor::configureSlider(juce::Slider& slider, const juce::String& suffix)
{
    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 68, 16);
    slider.setTextValueSuffix(suffix);
    slider.setMouseDragSensitivity(200);
    slider.setVelocityBasedMode(true);
    slider.setVelocityModeParameters(1.0, 1, 0.09, true, juce::ModifierKeys::shiftModifier);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colour::fromRGBA(230, 244, 236, 96));
    slider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colour::fromRGBA(12, 30, 25, 178));
    slider.setColour(juce::Slider::textBoxTextColourId, kChalkText);
}

void InertiaBandsAudioProcessorEditor::paint(juce::Graphics& g)
{
    drawChalkboardBackground(g);

    drawPanelChrome(g, topBarBounds_.toFloat(), {}, true);
    drawPanelChrome(g, leftPanelBounds_.toFloat(), "Analysis & Behavior");
    drawPanelChrome(g, centerPanelBounds_.toFloat(), "Cluster Weight PCA (R^n -> R^3)");
    if (advancedMode_)
        drawPanelChrome(g, rightPanelBounds_.toFloat(), "Dynamics & Saturation");

    drawPanelChrome(g, clusterPanelBounds_.toFloat(), advancedMode_ ? "Cluster Cards" : "Weights, Cutoffs & Routing");

    g.setColour(kChalkText.withAlpha(0.95f));
    g.setFont(chalkLabelFont(18.0f));
    g.drawText("InertiaBands", topBarBounds_.withTrimmedLeft(12).removeFromTop(20), juce::Justification::centredLeft);

    g.setColour(kChalkTextDim.withAlpha(0.85f));
    g.setFont(chalkLabelFont(11.0f));
    g.drawText("Spectral Cluster Processor",
               topBarBounds_.withTrimmedLeft(14).withTrimmedTop(18).removeFromTop(14),
               juce::Justification::centredLeft);

    if (glitchEnergy_ > 0.01f)
    {
        const auto overlays = { centerPanelBounds_.toFloat(), leftPanelBounds_.toFloat() };
        for (const auto& zone : overlays)
        {
            if (glitchNoiseTexture_.isValid())
            {
                g.setTiledImageFill(glitchNoiseTexture_, zone.getX(), zone.getY(), 0.05f * glitchEnergy_ * glitchAmount_);
                g.fillRect(zone);
            }

            const auto lineCount = 1 + static_cast<int>(std::round(glitchEnergy_ * 4.0f));
            for (int i = 0; i < lineCount; ++i)
            {
                const auto y = zone.getY() + glitchRandom_.nextFloat() * zone.getHeight();
                const auto h = 1.0f + glitchRandom_.nextFloat() * (1.0f + (3.0f * glitchEnergy_));
                const auto a = (0.03f + (0.08f * glitchEnergy_)) * glitchAmount_;
                g.setColour(juce::Colour::fromRGBA(105, 178, 255, static_cast<juce::uint8>(255.0f * a)));
                g.fillRect(zone.withY(y).withHeight(h));

                g.setColour(juce::Colour::fromRGBA(255, 122, 74, static_cast<juce::uint8>(150.0f * a)));
                g.fillRect(zone.withY(y + 1.0f).withHeight(std::max(0.8f, h - 0.4f)).translated(1.5f, 0.0f));
            }
        }
    }
}

void InertiaBandsAudioProcessorEditor::resized()
{
    auto content = getLocalBounds().reduced(8, 24);
    topBarBounds_ = content.removeFromTop(36);
    content.removeFromTop(6);

    const auto mainHeight = advancedMode_
        ? std::max(260, content.getHeight() - 220)
        : std::max(220, content.getHeight() - 170);
    auto mainRow = content.removeFromTop(std::min(mainHeight, content.getHeight()));
    const auto leftW = static_cast<int>(mainRow.getWidth() * (advancedMode_ ? 0.32f : 0.38f));
    leftPanelBounds_ = mainRow.removeFromLeft(leftW).reduced(1);

    if (advancedMode_)
    {
        const auto centerW = static_cast<int>(mainRow.getWidth() * 0.66f);
        centerPanelBounds_ = mainRow.removeFromLeft(centerW).reduced(1);
        rightPanelBounds_ = mainRow.reduced(1);
    }
    else
    {
        centerPanelBounds_ = mainRow.reduced(1);
        rightPanelBounds_ = {};
    }

    content.removeFromTop(4);
    clusterPanelBounds_ = content.reduced(1);

    auto top = topBarBounds_.reduced(8, 4);
    auto controls = top;
    controls.removeFromLeft(188);
    const auto controlH = 22;
    const auto controlY = top.getY() + ((top.getHeight() - controlH) / 2);
    auto place = [controlH, controlY](juce::Rectangle<int>& row, juce::Component& c, int w)
    {
        c.setBounds(row.removeFromLeft(w).withY(controlY).withHeight(controlH));
        row.removeFromLeft(6);
    };

    place(controls, bypassButton_, 72);
    place(controls, freezeButton_, 72);
    place(controls, gateEnableButton_, 66);
    place(controls, autoLevelButton_, 96);

    place(controls, fftLabel_, 22);
    place(controls, fftSizeBox_, 90);
    place(controls, hopLabel_, 24);
    place(controls, hopModeBox_, 76);
    place(controls, outputModeLabel_, 24);
    place(controls, outputModeBox_, 94);
    place(controls, advancedModeButton_, 96);

    auto leftPanel = leftPanelBounds_.reduced(8, 22);
    auto centerPanel = centerPanelBounds_.reduced(8, 22);
    auto rightPanel = rightPanelBounds_.reduced(8, 22);
    leftPanel.removeFromTop(14);
    centerPanel.removeFromTop(16);
    rightPanel.removeFromTop(14);

    pcaLabel_.setBounds({});
    if (advancedMode_)
        pcaVisualizer_.setBounds(centerPanel);
    else
        pcaVisualizer_.setBounds(centerPanel.withSizeKeepingCentre(centerPanel.getWidth(), static_cast<int>(centerPanel.getHeight() * 0.88f)));

    const auto rowHeight = 16;
    const auto rowGap = 2;
    layoutControl(leftPanel.removeFromTop(rowHeight), clustersLabel_, clustersSlider_);
    leftPanel.removeFromTop(rowGap);
    layoutControl(leftPanel.removeFromTop(rowHeight), updateHzLabel_, updateHzSlider_);
    leftPanel.removeFromTop(rowGap);
    layoutControl(leftPanel.removeFromTop(rowHeight), glideMsLabel_, glideMsSlider_);
    leftPanel.removeFromTop(rowGap);
    layoutControl(leftPanel.removeFromTop(rowHeight), spreadLabel_, spreadSlider_);
    leftPanel.removeFromTop(rowGap);
    layoutControl(leftPanel.removeFromTop(rowHeight), distancePenaltyLabel_, distancePenaltySlider_);
    leftPanel.removeFromTop(rowGap);
    layoutControl(leftPanel.removeFromTop(rowHeight), smoothMsLabel_, smoothMsSlider_);
    leftPanel.removeFromTop(rowGap);
    layoutControl(leftPanel.removeFromTop(rowHeight), mixLabel_, mixSlider_);

    if (advancedMode_)
    {
        leftPanel.removeFromTop(rowGap);
        layoutControl(leftPanel.removeFromTop(rowHeight), featureAdaptRateLabel_, featureAdaptRateSlider_);
        leftPanel.removeFromTop(rowGap);
        layoutControl(leftPanel.removeFromTop(rowHeight), featureLowPenaltyLabel_, featureLowPenaltySlider_);
        leftPanel.removeFromTop(rowGap);
        layoutControl(leftPanel.removeFromTop(rowHeight), featureHighPenaltyLabel_, featureHighPenaltySlider_);
        leftPanel.removeFromTop(rowGap);
        layoutControl(leftPanel.removeFromTop(rowHeight), featureNormPenaltyLabel_, featureNormPenaltySlider_);
        leftPanel.removeFromTop(rowGap);
        layoutControl(leftPanel.removeFromTop(rowHeight), featureMinimaxLabel_, featureMinimaxSlider_);
        leftPanel.removeFromTop(rowGap);
        layoutControl(leftPanel.removeFromTop(rowHeight), featurePriorPenaltyLabel_, featurePriorPenaltySlider_);

        layoutControl(rightPanel.removeFromTop(rowHeight), outputGainLabel_, outputGainSlider_);
        rightPanel.removeFromTop(rowGap);
        layoutControl(rightPanel.removeFromTop(rowHeight), saturationLabel_, saturationSlider_);
        rightPanel.removeFromTop(rowGap);
        layoutControl(rightPanel.removeFromTop(rowHeight), driveFromLevelLabel_, driveFromLevelSlider_);
        rightPanel.removeFromTop(rowGap);
        layoutControl(rightPanel.removeFromTop(rowHeight), gateThresholdLabel_, gateThresholdSlider_);
        rightPanel.removeFromTop(rowGap);
        layoutControl(rightPanel.removeFromTop(rowHeight), gateSharpnessLabel_, gateSharpnessSlider_);
        rightPanel.removeFromTop(rowGap);
        layoutControl(rightPanel.removeFromTop(rowHeight), gateFloorLabel_, gateFloorSlider_);
    }
    else
    {
        for (int i = 0; i < inertia::kFeatureDim; ++i)
        {
            featureWeightLabels_[i].setBounds({});
            featureWeightValues_[i].setBounds({});
        }
    }

    auto clusterArea = clusterPanelBounds_.reduced(8, 14);
    const auto clusterRowHeight = std::max(14, (clusterArea.getHeight() / inertia::kMaxClusters) - 1);
    const auto labelWidth = std::max(260, static_cast<int>(clusterArea.getWidth() * 0.38f));
    const auto muteWidth = 34;
    const auto routeWidth = 106;

    for (int cluster = 0; cluster < inertia::kMaxClusters; ++cluster)
    {
        auto row = clusterArea.removeFromTop(clusterRowHeight).reduced(0, 1);
        auto statArea = row.removeFromLeft(labelWidth);
        auto cCell = statArea.removeFromLeft(26);
        auto roleCell = statArea.removeFromLeft(72);
        auto hzCell = statArea.removeFromLeft(64);
        auto tonCell = statArea.removeFromLeft(56);
        auto trCell = statArea.removeFromLeft(50);
        auto actCell = statArea;

        clusterStatLabels_[cluster][0].setBounds(cCell);
        clusterStatLabels_[cluster][1].setBounds(roleCell);
        clusterStatLabels_[cluster][2].setBounds(hzCell);
        clusterStatLabels_[cluster][3].setBounds(tonCell);
        clusterStatLabels_[cluster][4].setBounds(trCell);
        clusterStatLabels_[cluster][5].setBounds(actCell);

        const auto sliderWidth = std::max(58, (row.getWidth() - muteWidth - routeWidth) / 3);
        clusterGainSliders_[cluster].setBounds(row.removeFromLeft(sliderWidth));
        clusterHpSliders_[cluster].setBounds(row.removeFromLeft(sliderWidth));
        clusterLpSliders_[cluster].setBounds(row.removeFromLeft(sliderWidth));
        clusterMuteButtons_[cluster].setBounds(row.removeFromLeft(muteWidth));
        clusterRouteBoxes_[cluster].setBounds(row.removeFromLeft(routeWidth));
    }
}

void InertiaBandsAudioProcessorEditor::applyModeVisibility()
{
    const auto showAdvanced = advancedMode_;

    for (auto* c : { static_cast<juce::Component*>(&clustersLabel_),
                     static_cast<juce::Component*>(&clustersSlider_),
                     static_cast<juce::Component*>(&updateHzLabel_),
                     static_cast<juce::Component*>(&updateHzSlider_),
                     static_cast<juce::Component*>(&glideMsLabel_),
                     static_cast<juce::Component*>(&glideMsSlider_),
                     static_cast<juce::Component*>(&spreadLabel_),
                     static_cast<juce::Component*>(&spreadSlider_),
                     static_cast<juce::Component*>(&distancePenaltyLabel_),
                     static_cast<juce::Component*>(&distancePenaltySlider_),
                     static_cast<juce::Component*>(&smoothMsLabel_),
                     static_cast<juce::Component*>(&smoothMsSlider_),
                     static_cast<juce::Component*>(&mixLabel_),
                     static_cast<juce::Component*>(&mixSlider_) })
    {
        if (c != nullptr)
            c->setVisible(true);
    }

    for (auto* c : { static_cast<juce::Component*>(&outputGainLabel_),
                     static_cast<juce::Component*>(&outputGainSlider_),
                     static_cast<juce::Component*>(&saturationLabel_),
                     static_cast<juce::Component*>(&saturationSlider_),
                     static_cast<juce::Component*>(&driveFromLevelLabel_),
                     static_cast<juce::Component*>(&driveFromLevelSlider_),
                     static_cast<juce::Component*>(&gateThresholdLabel_),
                     static_cast<juce::Component*>(&gateThresholdSlider_),
                     static_cast<juce::Component*>(&gateSharpnessLabel_),
                     static_cast<juce::Component*>(&gateSharpnessSlider_),
                     static_cast<juce::Component*>(&gateFloorLabel_),
                     static_cast<juce::Component*>(&gateFloorSlider_) })
    {
        if (c != nullptr)
            c->setVisible(showAdvanced);
    }

    for (auto* c : { static_cast<juce::Component*>(&featureAdaptRateLabel_),
                     static_cast<juce::Component*>(&featureAdaptRateSlider_),
                     static_cast<juce::Component*>(&featureLowPenaltyLabel_),
                     static_cast<juce::Component*>(&featureLowPenaltySlider_),
                     static_cast<juce::Component*>(&featureHighPenaltyLabel_),
                     static_cast<juce::Component*>(&featureHighPenaltySlider_),
                     static_cast<juce::Component*>(&featureNormPenaltyLabel_),
                     static_cast<juce::Component*>(&featureNormPenaltySlider_),
                     static_cast<juce::Component*>(&featureMinimaxLabel_),
                     static_cast<juce::Component*>(&featureMinimaxSlider_),
                     static_cast<juce::Component*>(&featurePriorPenaltyLabel_),
                     static_cast<juce::Component*>(&featurePriorPenaltySlider_) })
    {
        if (c != nullptr)
            c->setVisible(showAdvanced);
    }

    for (int i = 0; i < inertia::kFeatureDim; ++i)
    {
        featureWeightLabels_[i].setVisible(false);
        featureWeightValues_[i].setVisible(false);
    }

    for (int cluster = 0; cluster < inertia::kMaxClusters; ++cluster)
    {
        const auto visible = (cluster < visibleClusters_);
        for (auto& l : clusterStatLabels_[cluster])
            l.setVisible(visible);
        clusterGainSliders_[cluster].setVisible(visible);
        clusterHpSliders_[cluster].setVisible(visible);
        clusterLpSliders_[cluster].setVisible(visible);
        clusterMuteButtons_[cluster].setVisible(visible);
        clusterRouteBoxes_[cluster].setVisible(visible);
    }
}

void InertiaBandsAudioProcessorEditor::loadThemeAssets()
{
    chalkboardTexture_ = loadImageFromBinary(BinaryData::chalkboard_base_tile_dark_verdant_1024_webp,
                                             BinaryData::chalkboard_base_tile_dark_verdant_1024_webpSize);
    chalkDustTexture_ = loadImageFromBinary(BinaryData::chalk_dust_tile_option_f_clean_1024_webp,
                                            BinaryData::chalk_dust_tile_option_f_clean_1024_webpSize);
    chalkSmudgeTexture_ = loadImageFromBinary(BinaryData::chalk_smudge_tile_option_h_ghost_1024_webp,
                                              BinaryData::chalk_smudge_tile_option_h_ghost_1024_webpSize);
    borderAtlasTexture_ = loadImageFromBinary(BinaryData::chalk_border_atlas_fleur_style_2048x512_webp,
                                              BinaryData::chalk_border_atlas_fleur_style_2048x512_webpSize);
    glitchNoiseTexture_ = loadImageFromBinary(BinaryData::glitch_noise_tile_option_d_digital_block_TRUE_512_webp,
                                              BinaryData::glitch_noise_tile_option_d_digital_block_TRUE_512_webpSize);

    // JUCE builds without WebP codec support will return invalid images: generate procedural fallbacks.
    if (! chalkboardTexture_.isValid())
        chalkboardTexture_ = makeProceduralNoiseTexture(512, 512, kBoardBase, juce::Colour::fromRGB(34, 86, 73), 0.16f);

    if (! chalkDustTexture_.isValid())
        chalkDustTexture_ = makeProceduralNoiseTexture(256, 256, juce::Colour::fromRGBA(231, 244, 236, 36), juce::Colours::white.withAlpha(0.30f), 0.45f);

    if (! chalkSmudgeTexture_.isValid())
        chalkSmudgeTexture_ = makeProceduralSmudgeTexture(512, 512, juce::Colour::fromRGBA(220, 238, 229, 44), juce::Colour::fromRGBA(150, 192, 177, 38));

    if (! borderAtlasTexture_.isValid())
        borderAtlasTexture_ = makeProceduralSmudgeTexture(512, 128, juce::Colour::fromRGBA(210, 232, 222, 28), juce::Colour::fromRGBA(162, 198, 186, 22));

    if (! glitchNoiseTexture_.isValid())
        glitchNoiseTexture_ = makeProceduralNoiseTexture(256, 256, juce::Colour::fromRGBA(110, 145, 170, 38), juce::Colour::fromRGBA(255, 120, 100, 22), 0.85f);
}

void InertiaBandsAudioProcessorEditor::drawChalkboardBackground(juce::Graphics& g)
{
    g.fillAll(kBoardBase);

    const auto full = getLocalBounds().toFloat();
    if (chalkboardTexture_.isValid())
    {
        g.setTiledImageFill(chalkboardTexture_, 0, 0, 0.70f);
        g.fillRect(full);
    }

    if (chalkSmudgeTexture_.isValid())
    {
        g.setTiledImageFill(chalkSmudgeTexture_, 0, 0, 0.34f);
        g.fillRect(full);
    }

    if (chalkDustTexture_.isValid())
    {
        g.setTiledImageFill(chalkDustTexture_, 0, 0, 0.22f);
        g.fillRect(full);
    }

    g.setColour(kChalkText.withAlpha(0.035f));
    for (int y = 26; y < getHeight(); y += 74)
        g.drawHorizontalLine(y, 8.0f, static_cast<float>(getWidth() - 8));

    for (int x = 32; x < getWidth(); x += 108)
        g.drawVerticalLine(x, 8.0f, static_cast<float>(getHeight() - 8));

    juce::ColourGradient vignette(juce::Colours::transparentBlack,
                                  full.getCentreX(),
                                  full.getCentreY(),
                                  juce::Colours::black.withAlpha(0.24f),
                                  full.getX(),
                                  full.getY(),
                                  true);
    g.setGradientFill(vignette);
    g.fillRect(full);
}

void InertiaBandsAudioProcessorEditor::drawPanelChrome(juce::Graphics& g, juce::Rectangle<float> bounds, const juce::String& title, bool emphasise)
{
    if (bounds.getWidth() < 8.0f || bounds.getHeight() < 8.0f)
        return;

    const auto base = emphasise ? kPanelFillEmphasis : kPanelFill;
    g.setColour(base);
    g.fillRoundedRectangle(bounds, 8.0f);

    if (chalkSmudgeTexture_.isValid())
    {
        g.setTiledImageFill(chalkSmudgeTexture_, bounds.getX(), bounds.getY(), emphasise ? 0.17f : 0.11f);
        g.fillRoundedRectangle(bounds.reduced(1.0f), 7.0f);
    }

    if (borderAtlasTexture_.isValid())
    {
        g.setTiledImageFill(borderAtlasTexture_, bounds.getX(), bounds.getY(), 0.10f);
        g.fillRoundedRectangle(bounds.reduced(2.0f), 6.5f);
    }

    const auto jitterBoost = (glitchFramesRemaining_ > 0) ? (1.1f + (1.6f * glitchEnergy_ * glitchAmount_)) : 0.85f;
    drawRoughBorder(g, bounds.reduced(0.8f), jitterBoost, kChalkText.withAlpha(0.42f));
    drawRoughBorder(g, bounds.reduced(2.0f), 0.5f * jitterBoost, kChalkText.withAlpha(0.22f));

    if (title.isNotEmpty())
    {
        auto header = bounds.toNearestInt().reduced(10, 6).removeFromTop(18);
        g.setColour(kChalkTextDim.withAlpha(0.95f));
        g.setFont(chalkLabelFont(13.0f));
        g.drawText(title, header, juce::Justification::centredLeft);
    }
}

void InertiaBandsAudioProcessorEditor::updateGlitchFromActivity()
{
    auto read = [this](const juce::String& id)
    {
        if (auto* p = apvts_.getRawParameterValue(id))
            return p->load();
        return 0.0f;
    };

    std::array<float, 28> now {};
    int i = 0;
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::numClusters);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::clusterUpdateHz);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::glideMs);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::clusterSpread);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::distancePenalty);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::maskSmoothMs);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::globalMix);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::outputGainDb);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::saturationAmount);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::driveFromLevel);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::gateThreshold);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::gateSharpness);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::gateFloor);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::featureAdaptRate);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::featureLowPenalty);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::featureHighPenalty);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::featureNormPenalty);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::featureMinimax);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::featurePriorPenalty);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::bypass);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::freeze);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::gateEnable);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::autoLevel);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::outputMode);
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::clusterGainDb(0));
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::clusterGainDb(1));
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::clusterHpHz(0));
    now[static_cast<size_t>(i++)] = read(inertia::ParamIDs::clusterLpHz(0));

    if (! haveMonitoredValues_)
    {
        lastMonitoredValues_ = now;
        haveMonitoredValues_ = true;
        return;
    }

    float maxDelta = 0.0f;
    float sumDelta = 0.0f;
    for (size_t n = 0; n < now.size(); ++n)
    {
        const auto d = std::abs(now[n] - lastMonitoredValues_[n]);
        maxDelta = std::max(maxDelta, d);
        sumDelta += d;
    }

    if (maxDelta > 0.008f)
    {
        const auto burst = inertia::clamp((2.8f * maxDelta) + (0.45f * sumDelta), 0.0f, 1.0f);
        glitchEnergy_ = inertia::clamp(glitchEnergy_ + burst, 0.0f, 1.0f);
        glitchFramesRemaining_ = std::max(glitchFramesRemaining_, 2 + static_cast<int>(std::round(12.0f * burst)));
    }

    if (glitchAmount_ > 0.0f && glitchRandom_.nextFloat() < (0.02f * glitchAmount_))
        glitchFramesRemaining_ = std::max(glitchFramesRemaining_, 1);

    if (glitchFramesRemaining_ > 0)
    {
        --glitchFramesRemaining_;
        glitchEnergy_ = inertia::clamp(glitchEnergy_ * 0.90f, 0.0f, 1.0f);
    }
    else
    {
        glitchEnergy_ = inertia::clamp(glitchEnergy_ * 0.84f, 0.0f, 1.0f);
    }

    lastMonitoredValues_ = now;
}

void InertiaBandsAudioProcessorEditor::timerCallback()
{
    updateClusterVisibility();

    if (processor_.copyPcaProjectionFrame(pcaFrameSnapshot_))
        pcaVisualizer_.setFrame(pcaFrameSnapshot_);

    InertiaBandsAudioProcessor::ResponseFrame responseFrame {};
    if (processor_.copyResponseFrame(responseFrame))
    {
        pcaVisualizer_.setResponseFrame(responseFrame);
        updateFeatureWeightReadout(responseFrame);
    }

    updateClusterSemanticReadout();
    updateGlitchFromActivity();

    if (glitchFramesRemaining_ > 0 || glitchEnergy_ > 0.01f || ! advancedMode_)
        repaint();
}

void InertiaBandsAudioProcessorEditor::updateClusterVisibility()
{
    auto* kParameter = apvts_.getRawParameterValue(inertia::ParamIDs::numClusters);
    const auto k = inertia::clamp(static_cast<int>(std::round((kParameter != nullptr) ? kParameter->load() : 4.0f)), 2, inertia::kMaxClusters);

    if (k == visibleClusters_)
        return;

    visibleClusters_ = k;
    applyModeVisibility();
}

void InertiaBandsAudioProcessorEditor::updateClusterSemanticReadout()
{
    const auto clusterCount = inertia::clamp(pcaFrameSnapshot_.numClusters, 0, inertia::kMaxClusters);

    for (int cluster = 0; cluster < inertia::kMaxClusters; ++cluster)
    {
        juce::String idText = "C" + juce::String(cluster + 1);
        juce::String roleText = "Texture";
        juce::String hzText = "0Hz";
        juce::String tonText = "Ton 0.00";
        juce::String trText = "Tr 0.00";
        juce::String actText = "Act 0%";
        auto textColour = juce::Colours::white.withAlpha(0.88f);

        if (cluster < clusterCount)
        {
            const auto role = clusterRoleName(pcaFrameSnapshot_.clusterSemanticRole[cluster]);
            const auto centroidHz = pcaFrameSnapshot_.clusterCentroidHz[cluster];
            const auto tonalness = pcaFrameSnapshot_.clusterTonalness[cluster];
            const auto transientness = pcaFrameSnapshot_.clusterTransientness[cluster];
            const auto activityPct = 100.0f * pcaFrameSnapshot_.clusterEnergyShare[cluster];

            roleText = juce::String(role);
            hzText = juce::String(centroidHz, 1) + "Hz";
            tonText = "Ton " + juce::String(tonalness, 2);
            trText = "Tr " + juce::String(transientness, 2);
            actText = "Act " + juce::String(activityPct, 1) + "%";

            textColour = clusterColour(cluster).interpolatedWith(juce::Colours::white, 0.45f).withAlpha(0.95f);
        }

        clusterStatLabels_[cluster][0].setText(idText, juce::dontSendNotification);
        clusterStatLabels_[cluster][1].setText(roleText, juce::dontSendNotification);
        clusterStatLabels_[cluster][2].setText(hzText, juce::dontSendNotification);
        clusterStatLabels_[cluster][3].setText(tonText, juce::dontSendNotification);
        clusterStatLabels_[cluster][4].setText(trText, juce::dontSendNotification);
        clusterStatLabels_[cluster][5].setText(actText, juce::dontSendNotification);

        clusterStatLabels_[cluster][0].setColour(juce::Label::textColourId, textColour);
        for (size_t i = 1; i < clusterStatLabels_[cluster].size(); ++i)
            clusterStatLabels_[cluster][i].setColour(juce::Label::textColourId, textColour.withAlpha(0.88f));
    }
}

void InertiaBandsAudioProcessorEditor::updateFeatureWeightReadout(const InertiaBandsAudioProcessor::ResponseFrame& frame)
{
    for (int i = 0; i < inertia::kFeatureDim; ++i)
    {
        const auto w = inertia::clamp(frame.featureWeights[static_cast<size_t>(i)], 0.0f, 3.0f);
        featureWeightValues_[i].setText(juce::String(w, 2), juce::dontSendNotification);
    }
}
