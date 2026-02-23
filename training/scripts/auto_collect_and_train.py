from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from training.core.config import TrainingConfig
from training.core.descriptors import DescriptorExtractor
from training.core.modeling import HyperparameterModelTrainer
from training.core.optimizer import BayesianHyperOptimizer, OptimizationBudget
from training.core.storage import DatasetStorage
from training.scripts.fetch_datasets import fetch_dataset


@dataclass(frozen=True)
class DatasetTask:
    dataset_id: str
    split: str
    max_items: int
    trust_remote_code: bool = False
    dataset_config: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-collect data and retrain the hyperparameter model.")
    parser.add_argument("--bo-trials", type=int, default=10, help="BO trials per file")
    parser.add_argument("--bo-startup", type=int, default=4, help="BO startup trials per file")
    parser.add_argument("--bo-repeats", type=int, default=2, help="BO repeat evaluations per trial")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for BO")
    return parser.parse_args()


def missing_descriptor_ids(storage: DatasetStorage) -> list[str]:
    query = """
        SELECT a.audio_id
        FROM audio_files AS a
        LEFT JOIN descriptors AS d ON a.audio_id = d.audio_id
        WHERE d.audio_id IS NULL
        ORDER BY a.imported_at ASC
    """
    with duckdb.connect(str(storage.config.db_path)) as con:
        frame = con.execute(query).fetchdf()
    return [str(item) for item in frame["audio_id"].tolist()]


def missing_best_param_ids(storage: DatasetStorage) -> list[str]:
    query = """
        SELECT a.audio_id
        FROM audio_files AS a
        JOIN descriptors AS d ON a.audio_id = d.audio_id
        LEFT JOIN best_params AS b ON a.audio_id = b.audio_id
        WHERE b.audio_id IS NULL
        ORDER BY a.imported_at ASC
    """
    with duckdb.connect(str(storage.config.db_path)) as con:
        frame = con.execute(query).fetchdf()
    return [str(item) for item in frame["audio_id"].tolist()]


def maybe_update_best_equations(
    config: TrainingConfig,
    storage: DatasetStorage,
    model_id: str,
    bundle: dict,
    cv_mse: float,
) -> Path | None:
    best_equations_path = config.models_dir / "best_equations.json"
    with duckdb.connect(str(storage.config.db_path)) as con:
        best = con.execute(
            """
            SELECT model_id, cv_mse, artifact_path
            FROM models
            ORDER BY cv_mse ASC, created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if best is None:
        return None

    best_model_id = str(best[0])
    best_cv = float(best[1])
    if best_model_id != model_id and best_cv <= cv_mse:
        return None

    pipeline = bundle["pipeline"]
    poly = pipeline.named_steps["poly"]
    ridge = pipeline.named_steps["ridge"]
    feature_names = [str(x) for x in poly.get_feature_names_out(bundle["descriptor_keys"])]
    coefficients = {}
    for idx, target in enumerate(bundle["target_keys"]):
        row = ridge.coef_[idx]
        coefficients[target] = {
            feature_names[i]: float(row[i]) for i in range(len(feature_names))
        }

    payload = {
        "model_id": model_id,
        "cv_mse": cv_mse,
        "train_mse": float(bundle["metrics"]["train_mse"]),
        "degree": int(bundle["metrics"]["degree"]),
        "alpha": float(bundle["metrics"]["alpha"]),
        "descriptor_keys": list(bundle["descriptor_keys"]),
        "target_keys": list(bundle["target_keys"]),
        "intercept": [float(v) for v in ridge.intercept_],
        "coefficients": coefficients,
    }
    best_equations_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return best_equations_path


def run() -> None:
    args = parse_args()
    config = TrainingConfig.default()
    storage = DatasetStorage(config)
    extractor = DescriptorExtractor(config)
    optimizer = BayesianHyperOptimizer()
    trainer = HyperparameterModelTrainer(config)

    # Candidate datasets chosen for reliability in this environment.
    tasks = [
        DatasetTask(dataset_id="ashraq/esc50", split="train", max_items=400),
        DatasetTask(dataset_id="danavery/urbansound8K", split="train", max_items=400),
    ]

    print("== Dataset Import ==")
    for task in tasks:
        try:
            counters = fetch_dataset(
                dataset_id=task.dataset_id,
                dataset_config=task.dataset_config,
                split=task.split,
                max_items=task.max_items,
                output_audio_dir=(config.repo_root / "training/data/audio_cache").resolve(),
                allow_licenses=[""],
                audio_column=None,
                streaming=True,
                trust_remote_code=task.trust_remote_code,
            )
            print(f"{task.dataset_id}:{task.split} -> {counters}")
        except Exception as exc:
            print(f"{task.dataset_id}:{task.split} -> failed: {exc}")

    print("\n== Descriptor Extraction ==")
    to_extract = missing_descriptor_ids(storage)
    if not to_extract:
        print("no files missing descriptors")
    else:
        file_map = storage.list_audio_files().set_index("audio_id")
        for idx, audio_id in enumerate(to_extract, start=1):
            path = Path(str(file_map.loc[audio_id, "path"]))
            summary = extractor.extract_file(path)
            storage.upsert_descriptors(audio_id, summary)
            if idx % 25 == 0 or idx == len(to_extract):
                print(f"descriptors: {idx}/{len(to_extract)}")

    print("\n== Bayesian Optimization ==")
    to_optimize = missing_best_param_ids(storage)
    if not to_optimize:
        print("no files missing best_params")
    else:
        budget = OptimizationBudget(
            startup_trials=max(1, args.bo_startup),
            total_trials=max(1, args.bo_trials),
            repeats_per_trial=max(1, args.bo_repeats),
            seed=int(args.seed),
        )
        for idx, audio_id in enumerate(to_optimize, start=1):
            descriptors = storage.get_descriptors(audio_id)
            if descriptors is None:
                continue
            trials, best = optimizer.optimize(descriptors, budget)
            storage.insert_trials(audio_id, trials)
            storage.upsert_best_params(audio_id, best)
            if idx % 10 == 0 or idx == len(to_optimize):
                print(f"optimized: {idx}/{len(to_optimize)}")

    print("\n== Model Train ==")
    frame = storage.load_training_frame()
    if frame.empty:
        print("no training rows available")
        return
    bundle, metrics = trainer.train(frame)
    model_id = datetime.now().strftime("auto_model_%Y%m%d_%H%M%S")
    model_path = config.models_dir / f"{model_id}.joblib"
    trainer.save_model(bundle, model_path)
    storage.record_model(
        model_id=model_id,
        metrics=metrics,
        artifact_path=model_path,
        notes=f"auto_collect_and_train rows={frame.shape[0]}",
    )
    best_path = maybe_update_best_equations(
        config=config,
        storage=storage,
        model_id=model_id,
        bundle=bundle,
        cv_mse=float(metrics.cv_mse),
    )
    print(
        {
            "rows": int(frame.shape[0]),
            "model_path": str(model_path),
            "cv_mse": float(metrics.cv_mse),
            "train_mse": float(metrics.train_mse),
            "best_equations_path": str(best_path) if best_path else "",
        }
    )


if __name__ == "__main__":
    run()
