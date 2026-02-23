#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 scripts/bump_version.py
cmake -S . -B build -G Xcode -DBUILD_AU="${BUILD_AU:-ON}"
cmake --build build --config Release --target InertiaBands_VST3
