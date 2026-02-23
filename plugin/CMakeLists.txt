set(INERTIA_BANDS_FORMATS VST3)

if(APPLE AND BUILD_AU)
    list(APPEND INERTIA_BANDS_FORMATS AU)
endif()

juce_add_plugin(InertiaBands
    COMPANY_NAME "InertiaAudio"
    BUNDLE_ID "com.inertiaaudio.inertiabandsx"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    PLUGIN_MANUFACTURER_CODE Inrt
    PLUGIN_CODE Inb2
    FORMATS ${INERTIA_BANDS_FORMATS}
    PRODUCT_NAME "InertiaBandsX"
    VST3_CATEGORIES Fx
    AU_MAIN_TYPE kAudioUnitType_Effect
)

juce_generate_juce_header(InertiaBands)

juce_add_binary_data(InertiaBandsAssets
    SOURCES
        chalkboard_base_tile_dark_verdant_1024.webp
        chalk_dust_tile_option_f_clean_1024.webp
        chalk_smudge_tile_option_h_ghost_1024.webp
        chalk_border_atlas_fleur_style_2048x512.webp
        glitch_noise_tile_option_d_digital_block_TRUE_512.webp
        "paper note.png"
        assets/StardosStencil-Bold.ttf
        assets/StardosStencil-Regular.ttf
)

target_sources(InertiaBands
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginProcessor.h
        Source/PluginEditor.cpp
        Source/PluginEditor.h
        Source/DspStft.cpp
        Source/DspStft.h
        Source/DspClustering.cpp
        Source/DspClustering.h
        Source/DspAutoHyperModel.cpp
        Source/DspAutoHyperModel.h
        Source/AutoHyperModelData.h
        Source/DspPerceptualBins.cpp
        Source/DspPerceptualBins.h
        Source/DspMasks.cpp
        Source/DspMasks.h
        Source/DspProcessing.cpp
        Source/DspProcessing.h
        Source/DspPcaProjection.cpp
        Source/DspPcaProjection.h
        Source/Utilities.h
)

target_compile_definitions(InertiaBands
    PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(InertiaBands
    PRIVATE
        InertiaBandsAssets
        juce::juce_audio_utils
        juce::juce_dsp
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
