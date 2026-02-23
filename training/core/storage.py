from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import soundfile as sf

from .config import TrainingConfig
from .constants import DESCRIPTOR_KEYS
from .types import DescriptorSummary, ModelMetrics, TrialResult


class DatasetStorage:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.ensure_directories()
        self._initialize_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.config.db_path))

    def _initialize_schema(self) -> None:
        descriptor_columns = ",\n".join([f"{name} DOUBLE" for name in DESCRIPTOR_KEYS])
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS audio_files (
                    audio_id VARCHAR PRIMARY KEY,
                    path VARCHAR UNIQUE NOT NULL,
                    source VARCHAR,
                    genre VARCHAR,
                    duration_sec DOUBLE,
                    sample_rate INTEGER,
                    channels INTEGER,
                    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS descriptors (
                    audio_id VARCHAR PRIMARY KEY,
                    {descriptor_columns},
                    frame_count INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bo_trials (
                    audio_id VARCHAR NOT NULL,
                    trial_index INTEGER NOT NULL,
                    score DOUBLE,
                    sep DOUBLE,
                    intra DOUBLE,
                    jitter DOUBLE,
                    loud_err DOUBLE,
                    mask_flicker DOUBLE,
                    low_pen_loss DOUBLE,
                    high_pen_loss DOUBLE,
                    distance_penalty DOUBLE,
                    norm_rate DOUBLE,
                    base_penalty_log DOUBLE,
                    penalty_skew DOUBLE,
                    minimax_strength DOUBLE,
                    low_penalty DOUBLE,
                    high_penalty DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS best_params (
                    audio_id VARCHAR PRIMARY KEY,
                    score DOUBLE,
                    distance_penalty DOUBLE,
                    norm_rate DOUBLE,
                    base_penalty_log DOUBLE,
                    penalty_skew DOUBLE,
                    minimax_strength DOUBLE,
                    low_penalty DOUBLE,
                    high_penalty DOUBLE,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id VARCHAR PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    degree INTEGER,
                    alpha DOUBLE,
                    k_folds INTEGER,
                    cv_mse DOUBLE,
                    train_mse DOUBLE,
                    artifact_path VARCHAR,
                    descriptor_keys_json VARCHAR,
                    notes VARCHAR
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_metadata (
                    audio_id VARCHAR PRIMARY KEY,
                    dataset_id VARCHAR,
                    dataset_config VARCHAR,
                    split VARCHAR,
                    source_url VARCHAR,
                    license VARCHAR,
                    creator VARCHAR,
                    tags_json VARCHAR,
                    raw_json VARCHAR,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def make_audio_id(self, audio_path: Path) -> str:
        normalized = str(audio_path.expanduser().resolve())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]

    def upsert_audio_file(self, audio_path: Path, source: str = "manual", genre: str = "") -> str:
        audio_path = audio_path.expanduser().resolve()
        info = sf.info(str(audio_path))
        audio_id = self.make_audio_id(audio_path)

        with self._connect() as conn:
            conn.execute("DELETE FROM audio_files WHERE audio_id = ?", [audio_id])
            conn.execute(
                """
                INSERT INTO audio_files (
                    audio_id, path, source, genre, duration_sec, sample_rate, channels
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    audio_id,
                    str(audio_path),
                    source,
                    genre,
                    float(info.duration),
                    int(info.samplerate),
                    int(info.channels),
                ],
            )
        return audio_id

    def list_audio_files(self) -> pd.DataFrame:
        query = """
            SELECT
                a.audio_id,
                a.path,
                a.source,
                a.genre,
                a.duration_sec,
                a.sample_rate,
                a.channels,
                d.audio_id IS NOT NULL AS has_descriptors
            FROM audio_files AS a
            LEFT JOIN descriptors AS d ON a.audio_id = d.audio_id
            ORDER BY a.imported_at DESC
        """
        with self._connect() as conn:
            return conn.execute(query).fetchdf()

    def upsert_audio_metadata(
        self,
        audio_id: str,
        dataset_id: str = "",
        dataset_config: str = "",
        split: str = "",
        source_url: str = "",
        license_text: str = "",
        creator: str = "",
        tags: list[str] | None = None,
        raw: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM audio_metadata WHERE audio_id = ?", [audio_id])
            conn.execute(
                """
                INSERT INTO audio_metadata (
                    audio_id, dataset_id, dataset_config, split, source_url, license,
                    creator, tags_json, raw_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    audio_id,
                    dataset_id,
                    dataset_config,
                    split,
                    source_url,
                    license_text,
                    creator,
                    json.dumps(tags or []),
                    json.dumps(raw or {}),
                ],
            )

    def upsert_descriptors(self, audio_id: str, summary: DescriptorSummary) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM descriptors WHERE audio_id = ?", [audio_id])
            columns = ", ".join(DESCRIPTOR_KEYS)
            placeholders = ", ".join(["?"] * len(DESCRIPTOR_KEYS))
            values = [float(summary.descriptors[key]) for key in DESCRIPTOR_KEYS]
            conn.execute(
                f"""
                INSERT INTO descriptors (audio_id, {columns}, frame_count, updated_at)
                VALUES (?, {placeholders}, ?, CURRENT_TIMESTAMP)
                """,
                [audio_id, *values, int(summary.frame_count)],
            )

    def get_descriptors(self, audio_id: str) -> dict[str, float] | None:
        with self._connect() as conn:
            frame = conn.execute(
                f"SELECT {', '.join(DESCRIPTOR_KEYS)} FROM descriptors WHERE audio_id = ?",
                [audio_id],
            ).fetchdf()
        if frame.empty:
            return None
        row = frame.iloc[0]
        return {key: float(row[key]) for key in DESCRIPTOR_KEYS}

    def get_audio_row(self, audio_id: str) -> pd.Series | None:
        with self._connect() as conn:
            frame = conn.execute("SELECT * FROM audio_files WHERE audio_id = ?", [audio_id]).fetchdf()
        if frame.empty:
            return None
        return frame.iloc[0]

    def insert_trials(self, audio_id: str, trials: Iterable[TrialResult]) -> None:
        with self._connect() as conn:
            for trial in trials:
                conn.execute(
                    """
                    INSERT INTO bo_trials (
                        audio_id, trial_index, score, sep, intra, jitter, loud_err, mask_flicker,
                        low_pen_loss, high_pen_loss, distance_penalty, norm_rate, base_penalty_log,
                        penalty_skew, minimax_strength, low_penalty, high_penalty, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    [
                        audio_id,
                        int(trial.trial_index),
                        float(trial.terms.score),
                        float(trial.terms.sep),
                        float(trial.terms.intra),
                        float(trial.terms.jitter),
                        float(trial.terms.loud_err),
                        float(trial.terms.mask_flicker),
                        float(trial.terms.low_pen_loss),
                        float(trial.terms.high_pen_loss),
                        float(trial.params.distance_penalty),
                        float(trial.params.norm_rate),
                        float(trial.params.base_penalty_log),
                        float(trial.params.penalty_skew),
                        float(trial.params.minimax_strength),
                        float(trial.params.low_penalty),
                        float(trial.params.high_penalty),
                    ],
                )

    def upsert_best_params(self, audio_id: str, trial: TrialResult) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM best_params WHERE audio_id = ?", [audio_id])
            conn.execute(
                """
                INSERT INTO best_params (
                    audio_id, score, distance_penalty, norm_rate, base_penalty_log, penalty_skew,
                    minimax_strength, low_penalty, high_penalty, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    audio_id,
                    float(trial.terms.score),
                    float(trial.params.distance_penalty),
                    float(trial.params.norm_rate),
                    float(trial.params.base_penalty_log),
                    float(trial.params.penalty_skew),
                    float(trial.params.minimax_strength),
                    float(trial.params.low_penalty),
                    float(trial.params.high_penalty),
                ],
            )

    def load_trials(self, audio_id: str) -> pd.DataFrame:
        query = """
            SELECT
                trial_index, score, distance_penalty, norm_rate, base_penalty_log,
                penalty_skew, minimax_strength, low_penalty, high_penalty, created_at
            FROM bo_trials
            WHERE audio_id = ?
            ORDER BY trial_index ASC
        """
        with self._connect() as conn:
            return conn.execute(query, [audio_id]).fetchdf()

    def load_training_frame(self) -> pd.DataFrame:
        descriptor_cols = ", ".join([f"d.{key}" for key in DESCRIPTOR_KEYS])
        query = f"""
            SELECT
                a.audio_id,
                a.path,
                {descriptor_cols},
                b.score,
                b.distance_penalty,
                b.norm_rate,
                b.base_penalty_log,
                b.penalty_skew,
                b.minimax_strength,
                b.low_penalty,
                b.high_penalty
            FROM best_params AS b
            JOIN descriptors AS d ON b.audio_id = d.audio_id
            JOIN audio_files AS a ON b.audio_id = a.audio_id
        """
        with self._connect() as conn:
            return conn.execute(query).fetchdf()

    def record_model(self, model_id: str, metrics: ModelMetrics, artifact_path: Path, notes: str = "") -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM models WHERE model_id = ?", [model_id])
            conn.execute(
                """
                INSERT INTO models (
                    model_id, degree, alpha, k_folds, cv_mse, train_mse, artifact_path, descriptor_keys_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    model_id,
                    int(metrics.degree),
                    float(metrics.alpha),
                    int(metrics.k_folds),
                    float(metrics.cv_mse),
                    float(metrics.train_mse),
                    str(artifact_path),
                    json.dumps(list(DESCRIPTOR_KEYS)),
                    notes,
                ],
            )

    def latest_model_path(self) -> Path | None:
        query = "SELECT artifact_path FROM models ORDER BY created_at DESC LIMIT 1"
        with self._connect() as conn:
            frame = conn.execute(query).fetchdf()
        if frame.empty:
            return None
        path = Path(str(frame.iloc[0]["artifact_path"]))
        return path if path.exists() else None

    def export_training_csv(self, output_path: Path) -> Path:
        frame = self.load_training_frame()
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        return output_path
