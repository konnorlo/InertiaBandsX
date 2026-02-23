from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .config import TrainingConfig
from .constants import DESCRIPTOR_KEYS, EPSILON
from .types import DescriptorSummary


def _iqr(values: np.ndarray) -> float:
    return float(np.percentile(values, 75.0) - np.percentile(values, 25.0))


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / (denominator + EPSILON)


@dataclass
class StftFeatures:
    sample_rate: int
    duration_sec: float
    channels: int
    frame_count: int
    frequencies: np.ndarray
    magnitude: np.ndarray
    power: np.ndarray


class DescriptorExtractor:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def extract_file(self, audio_path: Path) -> DescriptorSummary:
        audio_path = audio_path.expanduser().resolve()
        features = self._compute_stft_features(audio_path)
        descriptors = self._compute_descriptors(features)

        return DescriptorSummary(
            audio_path=str(audio_path),
            sample_rate=features.sample_rate,
            duration_sec=features.duration_sec,
            channels=features.channels,
            frame_count=features.frame_count,
            descriptors=descriptors,
        )

    def _compute_stft_features(self, audio_path: Path) -> StftFeatures:
        audio, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
        if audio.size == 0:
            raise ValueError(f"empty audio file: {audio_path}")

        channels = int(audio.shape[1])
        mono = np.mean(audio, axis=1, dtype=np.float32)
        if mono.size < self.config.fft_size:
            pad = self.config.fft_size - mono.size
            mono = np.pad(mono, (0, pad))

        noverlap = self.config.fft_size - self.config.hop_size
        frequencies, _, spectrum = signal.stft(
            mono,
            fs=sample_rate,
            window="hann",
            nperseg=self.config.fft_size,
            noverlap=noverlap,
            nfft=self.config.fft_size,
            boundary=None,
            padded=False,
        )

        magnitude = np.abs(spectrum).astype(np.float64) + EPSILON
        power = magnitude * magnitude
        duration_sec = float(mono.size / max(sample_rate, 1))

        return StftFeatures(
            sample_rate=int(sample_rate),
            duration_sec=duration_sec,
            channels=channels,
            frame_count=int(magnitude.shape[1]),
            frequencies=frequencies.astype(np.float64),
            magnitude=magnitude,
            power=power,
        )

    def _compute_descriptors(self, features: StftFeatures) -> dict[str, float]:
        f = features.frequencies
        mag = features.magnitude
        power = features.power

        if mag.shape[1] == 0:
            raise ValueError("no STFT frames available for descriptor extraction")

        log_mag = np.log(mag + EPSILON)
        frame_energy = np.sum(power, axis=0)
        log_energy = np.log(frame_energy + EPSILON)

        log_freq = np.log(np.maximum(f, 1.0))
        x = log_freq[:, np.newaxis]
        x_centered = x - np.mean(x)
        y_centered = log_mag - np.mean(log_mag, axis=0, keepdims=True)
        slope = np.sum(x_centered * y_centered, axis=0) / (np.sum(x_centered * x_centered, axis=0) + EPSILON)

        mag_norm = mag / (np.mean(mag, axis=0, keepdims=True) + EPSILON)
        flux_delta = np.diff(mag_norm, axis=1, prepend=mag_norm[:, :1])
        spectral_flux = np.sqrt(np.mean(flux_delta * flux_delta, axis=0))

        peakiness = np.percentile(log_mag, 95.0, axis=0) - np.median(log_mag, axis=0)

        high_mask = f >= 2000.0
        low_mask = f <= 250.0
        high_power = np.sum(power[high_mask, :], axis=0) if np.any(high_mask) else np.zeros(power.shape[1], dtype=np.float64)
        low_power = np.sum(power[low_mask, :], axis=0) if np.any(low_mask) else np.zeros(power.shape[1], dtype=np.float64)
        total_power = np.sum(power, axis=0)

        brightness = _safe_ratio(high_power, total_power)
        flatness = np.exp(np.mean(np.log(mag + EPSILON), axis=0)) / (np.mean(mag, axis=0) + EPSILON)
        transientness = np.maximum(np.diff(log_energy, prepend=log_energy[0]), 0.0)
        low_high_ratio = np.log1p(_safe_ratio(low_power, high_power))

        dynamic_range_proxy = float(np.percentile(log_energy, 95.0) - np.percentile(log_energy, 10.0))
        silence_threshold = np.median(frame_energy) * 0.01
        silence_ratio = float(np.mean(frame_energy < silence_threshold))

        descriptor_values = {
            "log_energy_median": float(np.median(log_energy)),
            "log_energy_iqr": _iqr(log_energy),
            "spectral_slope_median": float(np.median(slope)),
            "spectral_flux_median": float(np.median(spectral_flux)),
            "spectral_flux_iqr": _iqr(spectral_flux),
            "peakiness_median": float(np.median(peakiness)),
            "brightness_median": float(np.median(brightness)),
            "flatness_median": float(np.median(flatness)),
            "transientness_median": float(np.median(transientness)),
            "low_high_ratio": float(np.median(low_high_ratio)),
            "dynamic_range_proxy": dynamic_range_proxy,
            "silence_ratio": silence_ratio,
        }

        # Ensure stable numeric values for the model pipeline.
        for key in DESCRIPTOR_KEYS:
            value = descriptor_values.get(key, 0.0)
            if not math.isfinite(value):
                descriptor_values[key] = 0.0

        return descriptor_values

