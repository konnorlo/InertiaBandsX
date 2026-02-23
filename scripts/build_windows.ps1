param(
    [string]$BuildAu = "OFF"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python scripts/bump_version.py
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_AU=$BuildAu
cmake --build build --config Release --target InertiaBands_VST3
