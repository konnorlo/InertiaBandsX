from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .types import HyperParams


def export_preset_json(
    output_path: Path,
    audio_path: Path,
    descriptors: Mapping[str, float],
    params: HyperParams,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    payload = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audio_path": str(audio_path),
        "descriptors": {key: float(value) for key, value in descriptors.items()},
        "hyperparameters": {
            "distance_penalty": float(params.distance_penalty),
            "norm_rate": float(params.norm_rate),
            "base_penalty_log": float(params.base_penalty_log),
            "penalty_skew": float(params.penalty_skew),
            "minimax_strength": float(params.minimax_strength),
            "low_penalty": float(params.low_penalty),
            "high_penalty": float(params.high_penalty),
        },
        "metadata": dict(metadata or {}),
    }

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path

