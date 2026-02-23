from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import TrainingConfig
from .constants import DESCRIPTOR_KEYS, MODEL_TARGET_KEYS
from .transforms import decode_model_outputs, logit
from .types import HyperParams, ModelMetrics


class HyperparameterModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self, frame: pd.DataFrame) -> tuple[dict, ModelMetrics]:
        min_rows = max(self.config.k_folds, 12)
        if frame.shape[0] < min_rows:
            raise ValueError(f"need at least {min_rows} rows in training set, found {frame.shape[0]}")

        x = frame.loc[:, list(DESCRIPTOR_KEYS)].to_numpy(dtype=np.float64)
        y = np.column_stack(
            [
                frame["distance_penalty"].to_numpy(dtype=np.float64),
                frame["norm_rate"].to_numpy(dtype=np.float64),
                frame["base_penalty_log"].to_numpy(dtype=np.float64),
                frame["penalty_skew"].to_numpy(dtype=np.float64),
                np.vectorize(logit)(frame["minimax_strength"].to_numpy(dtype=np.float64)),
            ]
        )

        alphas = np.logspace(-4.0, 2.0, 18)
        k_folds = min(self.config.k_folds, frame.shape[0])
        splitter = KFold(n_splits=k_folds, shuffle=True, random_state=self.config.random_seed)

        best_alpha = float(alphas[0])
        best_cv_mse = float("inf")

        for alpha in alphas:
            fold_errors: list[float] = []
            for train_idx, test_idx in splitter.split(x):
                model = self._build_pipeline(alpha=float(alpha))
                model.fit(x[train_idx], y[train_idx])
                pred = model.predict(x[test_idx])
                mse = float(np.mean((pred - y[test_idx]) ** 2))
                fold_errors.append(mse)
            cv_mse = float(np.mean(fold_errors))
            if cv_mse < best_cv_mse:
                best_cv_mse = cv_mse
                best_alpha = float(alpha)

        pipeline = self._build_pipeline(alpha=best_alpha)
        pipeline.fit(x, y)
        train_pred = pipeline.predict(x)
        train_mse = float(np.mean((train_pred - y) ** 2))

        metrics = ModelMetrics(
            alpha=best_alpha,
            degree=self.config.polynomial_degree,
            k_folds=k_folds,
            cv_mse=best_cv_mse,
            train_mse=train_mse,
        )

        bundle = {
            "pipeline": pipeline,
            "descriptor_keys": list(DESCRIPTOR_KEYS),
            "target_keys": list(MODEL_TARGET_KEYS),
            "metrics": asdict(metrics),
            "version": 1,
        }
        return bundle, metrics

    def save_model(self, bundle: dict, model_path: Path) -> Path:
        model_path = model_path.expanduser().resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path)
        return model_path

    @staticmethod
    def load_model(model_path: Path) -> dict:
        return joblib.load(str(model_path))

    @staticmethod
    def predict(bundle: Mapping, descriptors: Mapping[str, float]) -> HyperParams:
        descriptor_keys = bundle.get("descriptor_keys", list(DESCRIPTOR_KEYS))
        row = np.array([[float(descriptors.get(key, 0.0)) for key in descriptor_keys]], dtype=np.float64)
        raw = bundle["pipeline"].predict(row)[0]
        return decode_model_outputs(raw)

    @staticmethod
    def summarize_metrics(metrics: ModelMetrics) -> str:
        return (
            f"alpha={metrics.alpha:.6g}\n"
            f"degree={metrics.degree}\n"
            f"k_folds={metrics.k_folds}\n"
            f"cv_mse={metrics.cv_mse:.6f}\n"
            f"train_mse={metrics.train_mse:.6f}"
        )

    def _build_pipeline(self, alpha: float) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=self.config.polynomial_degree, include_bias=False)),
                ("ridge", Ridge(alpha=alpha, random_state=self.config.random_seed)),
            ]
        )
