param(
    [string]$Generator = "Visual Studio 17 2022",
    [string]$Arch = "x64"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$build = Join-Path $root "build"
$version = (Get-Content (Join-Path $root "VERSION") -Raw).Trim()
$artifactDir = Join-Path $root "release"
$pluginBundle = Join-Path $build "plugin/InertiaBands_artefacts/Release/VST3/InertiaBandsX.vst3"
$zipOut = Join-Path $artifactDir ("InertiaBandsX-windows-v{0}.vst3.zip" -f $version)

New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null

cmake -S $root -B $build -G $Generator -A $Arch -DBUILD_AU=OFF
cmake --build $build --config Release --target InertiaBands_VST3

if (-not (Test-Path $pluginBundle)) {
    throw "Missing built VST3 bundle at: $pluginBundle"
}

if (Test-Path $zipOut) { Remove-Item -Force $zipOut }
Compress-Archive -Path $pluginBundle -DestinationPath $zipOut
Write-Host "Created: $zipOut"
