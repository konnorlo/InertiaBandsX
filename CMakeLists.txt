cmake_minimum_required(VERSION 3.22)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" INERTIA_BANDS_VERSION_RAW)
string(STRIP "${INERTIA_BANDS_VERSION_RAW}" INERTIA_BANDS_VERSION)
if(NOT INERTIA_BANDS_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "VERSION file must contain semantic version MAJOR.MINOR.PATCH, got: '${INERTIA_BANDS_VERSION}'")
endif()

project(InertiaBands VERSION ${INERTIA_BANDS_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_AU "Build AU format on macOS" ON)
option(BUILD_UNIVERSAL_MACOS_BINARY "Build universal macOS binary" ON)
set(JUCE_SOURCE_DIR "" CACHE PATH "Optional path to a local JUCE checkout")

if(APPLE AND BUILD_UNIVERSAL_MACOS_BINARY)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "macOS architectures" FORCE)
endif()

if(JUCE_SOURCE_DIR)
    message(STATUS "Using local JUCE from: ${JUCE_SOURCE_DIR}")
    add_subdirectory(${JUCE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/juce)
else()
    include(FetchContent)

    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Declare(
        juce
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 8.0.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(juce)
endif()

add_subdirectory(plugin)
