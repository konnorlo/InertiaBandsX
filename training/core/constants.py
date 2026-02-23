from __future__ import annotations

DESCRIPTOR_KEYS: tuple[str, ...] = (
    "log_energy_median",
    "log_energy_iqr",
    "spectral_slope_median",
    "spectral_flux_median",
    "spectral_flux_iqr",
    "peakiness_median",
    "brightness_median",
    "flatness_median",
    "transientness_median",
    "low_high_ratio",
    "dynamic_range_proxy",
    "silence_ratio",
)

MODEL_TARGET_KEYS: tuple[str, ...] = (
    "distance_penalty",
    "norm_rate",
    "base_penalty_log",
    "penalty_skew",
    "minimax_logit",
)

AUDIO_FILE_EXTENSIONS: tuple[str, ...] = (
    ".wav",
    ".flac",
    ".aif",
    ".aiff",
    ".ogg",
    ".mp3",
    ".m4a",
)

EPSILON = 1.0e-8
DEFAULT_FFT_SIZE = 2048
DEFAULT_HOP_SIZE = 1024

