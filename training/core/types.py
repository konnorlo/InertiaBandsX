from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DescriptorSummary:
    audio_path: str
    sample_rate: int
    duration_sec: float
    channels: int
    frame_count: int
    descriptors: dict[str, float]


@dataclass(frozen=True)
class HyperParams:
    distance_penalty: float
    norm_rate: float
    base_penalty_log: float
    penalty_skew: float
    minimax_strength: float
    low_penalty: float
    high_penalty: float


@dataclass(frozen=True)
class ObjectiveTerms:
    score: float
    sep: float
    intra: float
    jitter: float
    loud_err: float
    mask_flicker: float
    low_pen_loss: float
    high_pen_loss: float


@dataclass(frozen=True)
class TrialResult:
    trial_index: int
    params: HyperParams
    terms: ObjectiveTerms


@dataclass(frozen=True)
class ModelMetrics:
    alpha: float
    degree: int
    k_folds: int
    cv_mse: float
    train_mse: float

