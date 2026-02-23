from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_FFT_SIZE, DEFAULT_HOP_SIZE


@dataclass(frozen=True)
class TrainingConfig:
    repo_root: Path
    training_root: Path
    data_dir: Path
    db_path: Path
    models_dir: Path
    presets_dir: Path
    fft_size: int = DEFAULT_FFT_SIZE
    hop_size: int = DEFAULT_HOP_SIZE
    polynomial_degree: int = 3
    k_folds: int = 10
    random_seed: int = 1337

    @classmethod
    def default(cls, repo_root: Path | None = None) -> "TrainingConfig":
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[2]
        repo_root = repo_root.resolve()
        training_root = repo_root / "training"
        data_dir = training_root / "data"
        models_dir = training_root / "models"
        presets_dir = training_root / "presets"
        config = cls(
            repo_root=repo_root,
            training_root=training_root,
            data_dir=data_dir,
            db_path=data_dir / "training.duckdb",
            models_dir=models_dir,
            presets_dir=presets_dir,
        )
        config.ensure_directories()
        return config

    def ensure_directories(self) -> None:
        self.training_root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

