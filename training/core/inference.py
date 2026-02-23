from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .descriptors import DescriptorExtractor
from .modeling import HyperparameterModelTrainer
from .optimizer import BayesianHyperOptimizer
from .types import DescriptorSummary, HyperParams, TrialResult


@dataclass(frozen=True)
class InferenceResult:
    descriptor_summary: DescriptorSummary
    predicted: HyperParams
    refined: HyperParams | None
    refinement_trial: TrialResult | None


class HyperparameterPredictor:
    def __init__(
        self,
        extractor: DescriptorExtractor,
        model_trainer: HyperparameterModelTrainer,
        optimizer: BayesianHyperOptimizer,
    ):
        self.extractor = extractor
        self.model_trainer = model_trainer
        self.optimizer = optimizer

    def predict_from_audio(
        self,
        audio_path: Path,
        model_bundle: Mapping,
        refine_trials: int = 0,
        refine_seed: int = 1337,
    ) -> InferenceResult:
        summary = self.extractor.extract_file(audio_path)
        predicted = self.model_trainer.predict(model_bundle, summary.descriptors)

        if refine_trials <= 0:
            return InferenceResult(
                descriptor_summary=summary,
                predicted=predicted,
                refined=None,
                refinement_trial=None,
            )

        refinement = self.optimizer.optimize_local(
            descriptors=summary.descriptors,
            center=predicted,
            total_trials=refine_trials,
            seed=refine_seed,
        )
        return InferenceResult(
            descriptor_summary=summary,
            predicted=predicted,
            refined=refinement.params,
            refinement_trial=refinement,
        )

