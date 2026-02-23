#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
VERSION="$(tr -d '[:space:]' < "${ROOT_DIR}/VERSION")"
ARTIFACT_DIR="${ROOT_DIR}/release"
PLUGIN_BUNDLE="${BUILD_DIR}/plugin/InertiaBands_artefacts/Release/VST3/InertiaBandsX.vst3"
OUT_ZIP="${ARTIFACT_DIR}/InertiaBandsX-macOS-v${VERSION}.vst3.zip"

mkdir -p "${ARTIFACT_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Xcode -DBUILD_AU=ON
cmake --build "${BUILD_DIR}" --config Release --target InertiaBands_VST3

if [[ ! -d "${PLUGIN_BUNDLE}" ]]; then
  echo "Missing built VST3 bundle at: ${PLUGIN_BUNDLE}" >&2
  exit 1
fi

ditto -c -k --sequesterRsrc --keepParent "${PLUGIN_BUNDLE}" "${OUT_ZIP}"
echo "Created: ${OUT_ZIP}"
