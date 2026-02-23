from __future__ import annotations

import math
from typing import Sequence

from .types import HyperParams

HIGH_PENALTY_SCALE = 0.35
LOW_SKEW_GAIN = 1.0
HIGH_SKEW_GAIN = 0.5


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def logit(value: float) -> float:
    p = clamp(value, 1.0e-6, 1.0 - 1.0e-6)
    return math.log(p / (1.0 - p))


def base_skew_to_penalties(base_penalty_log: float, penalty_skew: float) -> tuple[float, float]:
    base = math.exp(base_penalty_log)
    skew = math.tanh(penalty_skew)
    low_penalty = base * math.exp(LOW_SKEW_GAIN * skew)
    high_penalty = HIGH_PENALTY_SCALE * base * math.exp(-HIGH_SKEW_GAIN * skew)
    return low_penalty, high_penalty


def decode_model_outputs(raw_outputs: Sequence[float]) -> HyperParams:
    if len(raw_outputs) != 5:
        raise ValueError("expected 5 model outputs for hyperparameter decoding")

    distance_penalty = clamp(float(raw_outputs[0]), 0.0, 2.5)
    norm_rate = clamp(float(raw_outputs[1]), 0.001, 0.25)
    base_penalty_log = clamp(float(raw_outputs[2]), -4.0, 4.0)
    penalty_skew = clamp(float(raw_outputs[3]), -3.0, 3.0)
    minimax_strength = sigmoid(float(raw_outputs[4]))
    low_penalty, high_penalty = base_skew_to_penalties(base_penalty_log, penalty_skew)

    return HyperParams(
        distance_penalty=distance_penalty,
        norm_rate=norm_rate,
        base_penalty_log=base_penalty_log,
        penalty_skew=penalty_skew,
        minimax_strength=minimax_strength,
        low_penalty=low_penalty,
        high_penalty=high_penalty,
    )

