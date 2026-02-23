from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - UI fallback when pyqtgraph is missing
    pg = None

from training.app.workers import Worker
from training.core import (
    BayesianHyperOptimizer,
    DatasetStorage,
    DescriptorExtractor,
    HyperparameterModelTrainer,
    HyperparameterPredictor,
    OptimizationBudget,
    TrainingConfig,
)
from training.core.constants import AUDIO_FILE_EXTENSIONS
from training.core.inference import InferenceResult
from training.core.presets import export_preset_json


def _format_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


class TrainerMainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: TrainingConfig | None = None):
        super().__init__()
        self.config = config or TrainingConfig.default()
        self.storage = DatasetStorage(self.config)
        self.extractor = DescriptorExtractor(self.config)
        self.optimizer = BayesianHyperOptimizer()
        self.model_trainer = HyperparameterModelTrainer(self.config)
        self.predictor = HyperparameterPredictor(self.extractor, self.model_trainer, self.optimizer)
        self.thread_pool = QtCore.QThreadPool.globalInstance()

        self.current_model_path: Path | None = self.storage.latest_model_path()
        self.inference_audio_path: Path | None = None
        self.last_inference_result: InferenceResult | None = None
        self.bo_score_history: list[float] = []

        self.setWindowTitle("InertiaBands Trainer")
        self.resize(1320, 920)
        self._build_ui()
        self.refresh_audio_views()
        self._refresh_model_status()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self.setCentralWidget(central)

        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs, 1)

        tabs.addTab(self._build_import_tab(), "Import + Descriptors")
        tabs.addTab(self._build_bo_tab(), "Bayesian Opt")
        tabs.addTab(self._build_model_tab(), "Model")
        tabs.addTab(self._build_inference_tab(), "Inference")
        tabs.addTab(self._build_data_tab(), "Data")

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(140)
        root.addWidget(self.log_box)

    def _build_import_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        controls = QtWidgets.QHBoxLayout()
        self.add_files_button = QtWidgets.QPushButton("Add Audio Files")
        self.extract_selected_button = QtWidgets.QPushButton("Extract Selected")
        self.extract_all_button = QtWidgets.QPushButton("Extract All")
        self.refresh_audio_button = QtWidgets.QPushButton("Refresh")

        controls.addWidget(self.add_files_button)
        controls.addWidget(self.extract_selected_button)
        controls.addWidget(self.extract_all_button)
        controls.addWidget(self.refresh_audio_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.audio_table = QtWidgets.QTableWidget()
        self.audio_table.setColumnCount(8)
        self.audio_table.setHorizontalHeaderLabels(
            ["Audio ID", "Path", "Dur(s)", "SR", "Ch", "Genre", "Source", "Descriptors"]
        )
        self.audio_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.audio_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.audio_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header = self.audio_table.horizontalHeader()
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for col in (0, 2, 3, 4, 5, 6, 7):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        layout.addWidget(self.audio_table, 1)

        self.add_files_button.clicked.connect(self.on_add_files_clicked)
        self.extract_selected_button.clicked.connect(self.on_extract_selected_clicked)
        self.extract_all_button.clicked.connect(self.on_extract_all_clicked)
        self.refresh_audio_button.clicked.connect(self.refresh_audio_views)
        return tab

    def _build_bo_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        row = QtWidgets.QHBoxLayout()
        self.bo_audio_combo = QtWidgets.QComboBox()
        self.bo_startup_spin = QtWidgets.QSpinBox()
        self.bo_startup_spin.setRange(1, 200)
        self.bo_startup_spin.setValue(12)
        self.bo_trials_spin = QtWidgets.QSpinBox()
        self.bo_trials_spin.setRange(1, 400)
        self.bo_trials_spin.setValue(40)
        self.bo_repeats_spin = QtWidgets.QSpinBox()
        self.bo_repeats_spin.setRange(1, 8)
        self.bo_repeats_spin.setValue(3)
        self.bo_seed_spin = QtWidgets.QSpinBox()
        self.bo_seed_spin.setRange(0, 2_000_000_000)
        self.bo_seed_spin.setValue(self.config.random_seed)
        self.bo_run_button = QtWidgets.QPushButton("Run BO")

        row.addWidget(QtWidgets.QLabel("Audio"))
        row.addWidget(self.bo_audio_combo, 2)
        row.addWidget(QtWidgets.QLabel("Startup"))
        row.addWidget(self.bo_startup_spin)
        row.addWidget(QtWidgets.QLabel("Total"))
        row.addWidget(self.bo_trials_spin)
        row.addWidget(QtWidgets.QLabel("Repeats"))
        row.addWidget(self.bo_repeats_spin)
        row.addWidget(QtWidgets.QLabel("Seed"))
        row.addWidget(self.bo_seed_spin)
        row.addWidget(self.bo_run_button)
        layout.addLayout(row)

        self.bo_best_label = QtWidgets.QLabel("Best score: n/a")
        layout.addWidget(self.bo_best_label)

        if pg is not None:
            self.bo_plot = pg.PlotWidget()
            self.bo_plot.setBackground((15, 18, 24))
            self.bo_plot.showGrid(x=True, y=True, alpha=0.2)
            self.bo_curve = self.bo_plot.plot([], [], pen=pg.mkPen((85, 197, 255), width=2))
            layout.addWidget(self.bo_plot, 1)
        else:
            self.bo_plot = QtWidgets.QLabel("Install pyqtgraph to view BO convergence plot.")
            self.bo_plot.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.bo_plot, 1)
            self.bo_curve = None

        self.bo_trials_table = QtWidgets.QTableWidget()
        self.bo_trials_table.setColumnCount(9)
        self.bo_trials_table.setHorizontalHeaderLabels(
            ["Trial", "Score", "Dist", "Norm", "BaseLog", "Skew", "Minimax", "LowPen", "HighPen"]
        )
        self.bo_trials_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.bo_trials_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.bo_trials_table, 1)

        self.bo_run_button.clicked.connect(self.on_bo_run_clicked)
        return tab

    def _build_model_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        row = QtWidgets.QHBoxLayout()
        self.model_train_button = QtWidgets.QPushButton("Train Degree-3 Ridge")
        self.model_load_latest_button = QtWidgets.QPushButton("Load Latest Model")
        row.addWidget(self.model_train_button)
        row.addWidget(self.model_load_latest_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.model_path_label = QtWidgets.QLabel("Model: n/a")
        layout.addWidget(self.model_path_label)

        self.model_metrics_box = QtWidgets.QTextEdit()
        self.model_metrics_box.setReadOnly(True)
        layout.addWidget(self.model_metrics_box, 1)

        self.model_train_button.clicked.connect(self.on_model_train_clicked)
        self.model_load_latest_button.clicked.connect(self.on_model_load_latest_clicked)
        return tab

    def _build_inference_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        row = QtWidgets.QHBoxLayout()
        self.infer_audio_edit = QtWidgets.QLineEdit()
        self.infer_audio_edit.setReadOnly(True)
        self.infer_choose_button = QtWidgets.QPushButton("Choose Audio")
        self.infer_predict_button = QtWidgets.QPushButton("Predict Hyperparams")
        self.infer_export_button = QtWidgets.QPushButton("Export Preset JSON")

        row.addWidget(QtWidgets.QLabel("Audio"))
        row.addWidget(self.infer_audio_edit, 1)
        row.addWidget(self.infer_choose_button)
        row.addWidget(self.infer_predict_button)
        row.addWidget(self.infer_export_button)
        layout.addLayout(row)

        refine_row = QtWidgets.QHBoxLayout()
        self.infer_refine_check = QtWidgets.QCheckBox("Run local BO refinement")
        self.infer_refine_spin = QtWidgets.QSpinBox()
        self.infer_refine_spin.setRange(1, 60)
        self.infer_refine_spin.setValue(12)
        self.infer_refine_spin.setEnabled(False)
        self.infer_refine_check.toggled.connect(self.infer_refine_spin.setEnabled)
        refine_row.addWidget(self.infer_refine_check)
        refine_row.addWidget(QtWidgets.QLabel("Refine trials"))
        refine_row.addWidget(self.infer_refine_spin)
        refine_row.addStretch(1)
        layout.addLayout(refine_row)

        self.infer_output_box = QtWidgets.QTextEdit()
        self.infer_output_box.setReadOnly(True)
        layout.addWidget(self.infer_output_box, 1)

        self.infer_choose_button.clicked.connect(self.on_infer_choose_clicked)
        self.infer_predict_button.clicked.connect(self.on_infer_predict_clicked)
        self.infer_export_button.clicked.connect(self.on_infer_export_clicked)
        return tab

    def _build_data_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.data_paths_box = QtWidgets.QTextEdit()
        self.data_paths_box.setReadOnly(True)
        self.data_paths_box.setPlainText(
            "\n".join(
                [
                    f"DB: {self.config.db_path}",
                    f"Models: {self.config.models_dir}",
                    f"Presets: {self.config.presets_dir}",
                    f"Data root: {self.config.data_dir}",
                ]
            )
        )
        layout.addWidget(self.data_paths_box)

        row = QtWidgets.QHBoxLayout()
        self.data_export_csv_button = QtWidgets.QPushButton("Export Training CSV")
        row.addWidget(self.data_export_csv_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.data_export_csv_button.clicked.connect(self.on_data_export_csv_clicked)
        return tab

    def log(self, message: str) -> None:
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{stamp}] {message}")

    def refresh_audio_views(self) -> None:
        frame = self.storage.list_audio_files()
        self.audio_table.setRowCount(frame.shape[0])
        self.bo_audio_combo.blockSignals(True)
        self.bo_audio_combo.clear()

        for row_idx, row in frame.iterrows():
            audio_id = str(row["audio_id"])
            path = str(row["path"])

            items = [
                QtWidgets.QTableWidgetItem(audio_id),
                QtWidgets.QTableWidgetItem(path),
                QtWidgets.QTableWidgetItem(_format_float(row["duration_sec"], 2)),
                QtWidgets.QTableWidgetItem(str(int(row["sample_rate"]))),
                QtWidgets.QTableWidgetItem(str(int(row["channels"]))),
                QtWidgets.QTableWidgetItem(str(row["genre"] or "")),
                QtWidgets.QTableWidgetItem(str(row["source"] or "")),
                QtWidgets.QTableWidgetItem("yes" if bool(row["has_descriptors"]) else "no"),
            ]

            for col_idx, item in enumerate(items):
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, audio_id)
                if col_idx == 7:
                    color = QtGui.QColor("#6ee7b7" if bool(row["has_descriptors"]) else "#fca5a5")
                    item.setForeground(QtGui.QBrush(color))
                self.audio_table.setItem(row_idx, col_idx, item)

            self.bo_audio_combo.addItem(Path(path).name, userData=audio_id)

        self.bo_audio_combo.blockSignals(False)
        self.log(f"Loaded {frame.shape[0]} audio file(s) from dataset.")

    def _selected_audio_ids(self) -> list[str]:
        rows = self.audio_table.selectionModel().selectedRows()
        audio_ids: list[str] = []
        for model_index in rows:
            item = self.audio_table.item(model_index.row(), 0)
            if item is not None:
                audio_ids.append(str(item.data(QtCore.Qt.ItemDataRole.UserRole)))
        return audio_ids

    def on_add_files_clicked(self) -> None:
        pattern = " ".join([f"*{ext}" for ext in AUDIO_FILE_EXTENSIONS])
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select audio files",
            str(self.config.repo_root),
            f"Audio Files ({pattern});;All Files (*)",
        )
        if not paths:
            return

        imported = 0
        for path_str in paths:
            path = Path(path_str)
            if path.suffix.lower() not in AUDIO_FILE_EXTENSIONS:
                continue
            self.storage.upsert_audio_file(audio_path=path, source="manual", genre="")
            imported += 1
        self.log(f"Imported {imported} audio file(s).")
        self.refresh_audio_views()

    def on_extract_selected_clicked(self) -> None:
        selected = self._selected_audio_ids()
        if not selected:
            QtWidgets.QMessageBox.information(self, "Descriptor Extraction", "Select one or more rows first.")
            return
        self._start_descriptor_worker(selected)

    def on_extract_all_clicked(self) -> None:
        frame = self.storage.list_audio_files()
        if frame.empty:
            QtWidgets.QMessageBox.information(self, "Descriptor Extraction", "No audio files in dataset.")
            return
        self._start_descriptor_worker([str(value) for value in frame["audio_id"].tolist()])

    def _start_descriptor_worker(self, audio_ids: list[str]) -> None:
        self.extract_selected_button.setEnabled(False)
        self.extract_all_button.setEnabled(False)
        self.log(f"Starting descriptor extraction for {len(audio_ids)} file(s).")

        worker = Worker(self._descriptor_job, audio_ids)
        worker.signals.progress.connect(self._on_descriptor_progress)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_descriptor_finished)
        self.thread_pool.start(worker)

    def _descriptor_job(self, audio_ids: list[str], progress: Any) -> dict[str, int]:
        files = self.storage.list_audio_files().set_index("audio_id")
        processed = 0
        for idx, audio_id in enumerate(audio_ids):
            if audio_id not in files.index:
                continue
            path = Path(str(files.loc[audio_id, "path"]))
            summary = self.extractor.extract_file(path)
            self.storage.upsert_descriptors(audio_id, summary)
            processed += 1
            progress({"index": idx + 1, "total": len(audio_ids), "path": str(path)})
        return {"processed": processed}

    def _on_descriptor_progress(self, payload: dict) -> None:
        self.log(f"[{payload['index']}/{payload['total']}] descriptors -> {payload['path']}")

    def _on_descriptor_finished(self, result: dict) -> None:
        self.extract_selected_button.setEnabled(True)
        self.extract_all_button.setEnabled(True)
        self.log(f"Descriptor extraction finished. Processed {result.get('processed', 0)} file(s).")
        self.refresh_audio_views()

    def on_bo_run_clicked(self) -> None:
        audio_id = self.bo_audio_combo.currentData()
        if not audio_id:
            QtWidgets.QMessageBox.information(self, "Bayesian Optimization", "No audio selected.")
            return

        descriptors = self.storage.get_descriptors(str(audio_id))
        if descriptors is None:
            QtWidgets.QMessageBox.warning(self, "Bayesian Optimization", "Extract descriptors for this file first.")
            return

        budget = OptimizationBudget(
            startup_trials=int(self.bo_startup_spin.value()),
            total_trials=int(self.bo_trials_spin.value()),
            repeats_per_trial=int(self.bo_repeats_spin.value()),
            seed=int(self.bo_seed_spin.value()),
        )

        self.bo_run_button.setEnabled(False)
        self.bo_trials_table.setRowCount(0)
        self.bo_score_history.clear()
        self.bo_best_label.setText("Best score: running...")
        if self.bo_curve is not None:
            self.bo_curve.setData([], [])
        self.log(f"Running BO for audio_id={audio_id} with {budget.total_trials} trial(s).")

        worker = Worker(self._bo_job, str(audio_id), descriptors, budget)
        worker.signals.progress.connect(self._on_bo_progress)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_bo_finished)
        self.thread_pool.start(worker)

    def _bo_job(self, audio_id: str, descriptors: dict[str, float], budget: OptimizationBudget, progress: Any) -> dict[str, Any]:
        trials, best = self.optimizer.optimize(descriptors, budget, progress=progress)
        self.storage.insert_trials(audio_id, trials)
        self.storage.upsert_best_params(audio_id, best)
        return {"audio_id": audio_id, "trials": trials, "best": best}

    def _on_bo_progress(self, payload: dict) -> None:
        self.bo_score_history.append(float(payload["score"]))
        if self.bo_curve is not None:
            x = list(range(len(self.bo_score_history)))
            self.bo_curve.setData(x, self.bo_score_history)

        row = self.bo_trials_table.rowCount()
        self.bo_trials_table.insertRow(row)
        values = [
            str(int(payload["trial_index"])),
            _format_float(payload["score"], 5),
            _format_float(payload["distance_penalty"], 4),
            _format_float(payload["norm_rate"], 4),
            _format_float(payload["base_penalty_log"], 4),
            _format_float(payload["penalty_skew"], 4),
            _format_float(payload["minimax_strength"], 4),
            _format_float(payload["low_penalty"], 4),
            _format_float(payload["high_penalty"], 4),
        ]
        for col, text in enumerate(values):
            self.bo_trials_table.setItem(row, col, QtWidgets.QTableWidgetItem(text))

        best_score = max(self.bo_score_history) if self.bo_score_history else float("-inf")
        self.bo_best_label.setText(f"Best score: {_format_float(best_score, 5)}")

    def _on_bo_finished(self, result: dict[str, Any]) -> None:
        self.bo_run_button.setEnabled(True)
        best = result["best"]
        self.log(
            "BO finished for "
            f"{result['audio_id']}. Best score={best.terms.score:.5f}, "
            f"dist={best.params.distance_penalty:.4f}, norm={best.params.norm_rate:.4f}, "
            f"base={best.params.base_penalty_log:.4f}, skew={best.params.penalty_skew:.4f}, "
            f"minimax={best.params.minimax_strength:.4f}"
        )

    def on_model_train_clicked(self) -> None:
        self.model_train_button.setEnabled(False)
        self.model_metrics_box.setPlainText("Training model...")
        worker = Worker(self._model_train_job)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_model_train_finished)
        self.thread_pool.start(worker)

    def _model_train_job(self, progress: Any) -> dict[str, Any]:
        frame = self.storage.load_training_frame()
        bundle, metrics = self.model_trainer.train(frame)
        model_id = dt.datetime.now().strftime("model_%Y%m%d_%H%M%S")
        model_path = self.config.models_dir / f"{model_id}.joblib"
        self.model_trainer.save_model(bundle, model_path)
        self.storage.record_model(model_id=model_id, metrics=metrics, artifact_path=model_path, notes=f"rows={frame.shape[0]}")
        return {"model_path": model_path, "metrics": metrics, "rows": frame.shape[0]}

    def _on_model_train_finished(self, result: dict[str, Any]) -> None:
        self.model_train_button.setEnabled(True)
        self.current_model_path = Path(result["model_path"])
        metrics = result["metrics"]
        self.model_metrics_box.setPlainText(
            (
                f"Rows: {result['rows']}\n"
                f"alpha: {metrics.alpha:.6g}\n"
                f"degree: {metrics.degree}\n"
                f"k_folds: {metrics.k_folds}\n"
                f"cv_mse: {metrics.cv_mse:.6f}\n"
                f"train_mse: {metrics.train_mse:.6f}\n"
                f"saved: {self.current_model_path}"
            )
        )
        self._refresh_model_status()
        self.log(f"Model trained and saved: {self.current_model_path}")

    def on_model_load_latest_clicked(self) -> None:
        self.current_model_path = self.storage.latest_model_path()
        self._refresh_model_status()
        if self.current_model_path is None:
            self.log("No model artifact found yet.")
        else:
            self.log(f"Loaded latest model reference: {self.current_model_path}")

    def _refresh_model_status(self) -> None:
        if self.current_model_path is None:
            self.model_path_label.setText("Model: n/a")
        else:
            self.model_path_label.setText(f"Model: {self.current_model_path}")

    def on_infer_choose_clicked(self) -> None:
        pattern = " ".join([f"*{ext}" for ext in AUDIO_FILE_EXTENSIONS])
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose audio for inference",
            str(self.config.repo_root),
            f"Audio Files ({pattern});;All Files (*)",
        )
        if not path_str:
            return
        self.inference_audio_path = Path(path_str).expanduser().resolve()
        self.infer_audio_edit.setText(str(self.inference_audio_path))
        self.log(f"Selected inference audio: {self.inference_audio_path}")

    def on_infer_predict_clicked(self) -> None:
        if self.inference_audio_path is None:
            QtWidgets.QMessageBox.information(self, "Inference", "Choose an audio file first.")
            return
        if self.current_model_path is None:
            self.current_model_path = self.storage.latest_model_path()
            if self.current_model_path is None:
                QtWidgets.QMessageBox.warning(self, "Inference", "No trained model found. Train a model first.")
                return

        refine_trials = int(self.infer_refine_spin.value()) if self.infer_refine_check.isChecked() else 0
        self.infer_predict_button.setEnabled(False)
        self.infer_output_box.setPlainText("Running inference...")

        worker = Worker(self._inference_job, self.inference_audio_path, self.current_model_path, refine_trials)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_inference_finished)
        self.thread_pool.start(worker)

    def _inference_job(self, audio_path: Path, model_path: Path, refine_trials: int, progress: Any) -> dict[str, Any]:
        bundle = self.model_trainer.load_model(model_path)
        result = self.predictor.predict_from_audio(
            audio_path=audio_path,
            model_bundle=bundle,
            refine_trials=refine_trials,
            refine_seed=self.config.random_seed,
        )
        return {"result": result, "model_path": model_path}

    def _on_inference_finished(self, payload: dict[str, Any]) -> None:
        self.infer_predict_button.setEnabled(True)
        result: InferenceResult = payload["result"]
        self.last_inference_result = result

        chosen = result.refined if result.refined is not None else result.predicted
        descriptor_text = json.dumps(result.descriptor_summary.descriptors, indent=2)
        text = (
            f"Model: {payload['model_path']}\n"
            f"Audio: {result.descriptor_summary.audio_path}\n"
            f"Duration: {result.descriptor_summary.duration_sec:.2f}s\n"
            f"Sample rate: {result.descriptor_summary.sample_rate}\n"
            f"Frames: {result.descriptor_summary.frame_count}\n\n"
            f"Predicted:\n"
            f"  distance_penalty = {result.predicted.distance_penalty:.6f}\n"
            f"  norm_rate        = {result.predicted.norm_rate:.6f}\n"
            f"  base_penalty_log = {result.predicted.base_penalty_log:.6f}\n"
            f"  penalty_skew     = {result.predicted.penalty_skew:.6f}\n"
            f"  minimax_strength = {result.predicted.minimax_strength:.6f}\n"
            f"  low_penalty      = {result.predicted.low_penalty:.6f}\n"
            f"  high_penalty     = {result.predicted.high_penalty:.6f}\n"
        )
        if result.refined is not None:
            text += (
                "\nRefined:\n"
                f"  score            = {result.refinement_trial.terms.score:.6f}\n"
                f"  distance_penalty = {result.refined.distance_penalty:.6f}\n"
                f"  norm_rate        = {result.refined.norm_rate:.6f}\n"
                f"  base_penalty_log = {result.refined.base_penalty_log:.6f}\n"
                f"  penalty_skew     = {result.refined.penalty_skew:.6f}\n"
                f"  minimax_strength = {result.refined.minimax_strength:.6f}\n"
                f"  low_penalty      = {result.refined.low_penalty:.6f}\n"
                f"  high_penalty     = {result.refined.high_penalty:.6f}\n"
            )

        text += f"\nDescriptors:\n{descriptor_text}\n"
        self.infer_output_box.setPlainText(text)
        self.log(
            "Inference complete. "
            f"distance={chosen.distance_penalty:.4f}, norm={chosen.norm_rate:.4f}, "
            f"base={chosen.base_penalty_log:.4f}, skew={chosen.penalty_skew:.4f}, "
            f"minimax={chosen.minimax_strength:.4f}"
        )

    def on_infer_export_clicked(self) -> None:
        if self.last_inference_result is None:
            QtWidgets.QMessageBox.information(self, "Export Preset", "Run inference first.")
            return

        audio_path = Path(self.last_inference_result.descriptor_summary.audio_path)
        default_name = f"{audio_path.stem}_hyperparams.json"
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export preset JSON",
            str(self.config.presets_dir / default_name),
            "JSON Files (*.json)",
        )
        if not output_path:
            return

        params = self.last_inference_result.refined or self.last_inference_result.predicted
        exported = export_preset_json(
            output_path=Path(output_path),
            audio_path=audio_path,
            descriptors=self.last_inference_result.descriptor_summary.descriptors,
            params=params,
            metadata={"model_path": str(self.current_model_path or "")},
        )
        self.log(f"Preset exported: {exported}")

    def on_data_export_csv_clicked(self) -> None:
        default_output = self.config.data_dir / "training_dataset.csv"
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export training CSV",
            str(default_output),
            "CSV Files (*.csv)",
        )
        if not path_str:
            return
        exported = self.storage.export_training_csv(Path(path_str))
        self.log(f"Exported training CSV: {exported}")

    def _on_worker_error(self, traceback_text: str) -> None:
        self.extract_selected_button.setEnabled(True)
        self.extract_all_button.setEnabled(True)
        self.bo_run_button.setEnabled(True)
        self.model_train_button.setEnabled(True)
        self.infer_predict_button.setEnabled(True)
        self.log("Worker failed. See details below.")
        self.log_box.append(traceback_text)
        QtWidgets.QMessageBox.critical(self, "Worker Error", traceback_text)
