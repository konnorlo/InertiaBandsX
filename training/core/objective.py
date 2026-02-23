from __future__ import annotations

import math
from typing import Mapping

from .transforms import clamp, sigmoid
from .types import HyperParams, ObjectiveTerms


def _d(descriptors: Mapping[str, float], key: str, fallback: float = 0.0) -> float:
    return float(descriptors.get(key, fallback))


def score_hyperparams(descriptors: Mapping[str, float], params: HyperParams) -> ObjectiveTerms:
    flux_median = clamp(_d(descriptors, "spectral_flux_median"), 0.0, 3.0)
    flux_iqr = clamp(_d(descriptors, "spectral_flux_iqr"), 0.0, 3.0)
    transient = clamp(_d(descriptors, "transientness_median"), 0.0, 3.0)
    brightness = clamp(_d(descriptors, "brightness_median"), 0.0, 1.0)
    silence = clamp(_d(descriptors, "silence_ratio"), 0.0, 1.0)
    dyn_range = clamp(_d(descriptors, "dynamic_range_proxy"), 0.0, 20.0)
    low_high_ratio = clamp(_d(descriptors, "low_high_ratio"), 0.0, 6.0)

    target_distance = clamp(0.30 + (1.15 * flux_median) + (0.45 * transient) + (0.20 * brightness), 0.0, 2.5)
    target_norm = clamp(0.015 + (0.10 * flux_median) + (0.08 * transient), 0.001, 0.25)

    distance_error = (params.distance_penalty - target_distance) / 0.60
    norm_error = (params.norm_rate - target_norm) / 0.08

    sep = math.exp(-(distance_error * distance_error))
    sep *= 1.0 - (0.20 * clamp(abs(params.norm_rate - target_norm) / 0.25, 0.0, 1.0))
    sep = clamp(sep, 0.0, 1.0)

    intra = clamp((distance_error * distance_error) + (0.6 * norm_error * norm_error), 0.0, 4.0)
    jitter = clamp(((0.65 * flux_iqr) + (0.80 * transient)) * (0.45 + (2.0 * params.norm_rate)) + (0.20 * params.minimax_strength), 0.0, 4.0)

    loud_err = clamp(
        abs(math.log1p(params.low_penalty * low_high_ratio) - math.log1p(params.high_penalty * (0.25 + brightness))),
        0.0,
        4.0,
    )
    mask_flicker = clamp((flux_iqr + transient) * (1.0 - math.exp(-5.0 * params.norm_rate)), 0.0, 4.0)

    low_floor = 0.35 + (1.6 * silence) + (0.4 * (dyn_range / 10.0))
    low_distance = max(params.low_penalty - low_floor, 1.0e-3)
    low_pen_loss = clamp(1.0 / (low_distance * low_distance), 0.0, 50.0)

    high_ceiling = 0.90 + (1.8 * brightness) + (0.4 * flux_median)
    high_pen_loss = clamp(max(0.0, params.high_penalty - high_ceiling) ** 2, 0.0, 10.0)

    raw_score = (
        (0.35 * sep)
        - (0.30 * intra)
        - (0.15 * jitter)
        - (0.10 * loud_err)
        - (0.10 * mask_flicker)
        - (0.08 * low_pen_loss)
        - (0.04 * high_pen_loss)
    )
    score = sigmoid(raw_score)

    return ObjectiveTerms(
        score=score,
        sep=sep,
        intra=intra,
        jitter=jitter,
        loud_err=loud_err,
        mask_flicker=mask_flicker,
        low_pen_loss=low_pen_loss,
        high_pen_loss=high_pen_loss,
    )
