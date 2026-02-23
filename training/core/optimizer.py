from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import optuna

from .objective import score_hyperparams
from .transforms import base_skew_to_penalties, clamp, logit, sigmoid
from .types import HyperParams, ObjectiveTerms, TrialResult


@dataclass(frozen=True)
class OptimizationBudget:
    startup_trials: int = 12
    total_trials: int = 40
    repeats_per_trial: int = 3
    seed: int = 1337


class BayesianHyperOptimizer:
    def optimize(
        self,
        descriptors: Mapping[str, float],
        budget: OptimizationBudget,
        progress: Callable[[dict], None] | None = None,
    ) -> tuple[list[TrialResult], TrialResult]:
        if budget.total_trials < 1:
            raise ValueError("total_trials must be >= 1")
        if budget.repeats_per_trial < 1:
            raise ValueError("repeats_per_trial must be >= 1")

        sampler = optuna.samplers.TPESampler(
            seed=budget.seed,
            n_startup_trials=min(budget.startup_trials, budget.total_trials),
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)

        trial_results: dict[int, TrialResult] = {}

        def objective(trial: optuna.trial.Trial) -> float:
            params = self._sample_trial_params(trial)
            terms = self._evaluate_trial(descriptors, params, budget, trial.number)
            result = TrialResult(trial_index=trial.number, params=params, terms=terms)
            trial_results[trial.number] = result

            if progress is not None:
                progress(
                    {
                        "type": "trial",
                        "trial_index": trial.number,
                        "score": terms.score,
                        "distance_penalty": params.distance_penalty,
                        "norm_rate": params.norm_rate,
                        "base_penalty_log": params.base_penalty_log,
                        "penalty_skew": params.penalty_skew,
                        "minimax_strength": params.minimax_strength,
                        "low_penalty": params.low_penalty,
                        "high_penalty": params.high_penalty,
                    }
                )

            return terms.score

        study.optimize(objective, n_trials=budget.total_trials, show_progress_bar=False)

        ordered = [trial_results[index] for index in sorted(trial_results)]
        best = max(ordered, key=lambda trial: trial.terms.score)
        return ordered, best

    def optimize_local(
        self,
        descriptors: Mapping[str, float],
        center: HyperParams,
        total_trials: int = 12,
        seed: int = 1337,
    ) -> TrialResult:
        local_budget = OptimizationBudget(startup_trials=4, total_trials=max(1, total_trials), repeats_per_trial=3, seed=seed)
        sampler = optuna.samplers.TPESampler(seed=local_budget.seed, n_startup_trials=local_budget.startup_trials)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        best_result: TrialResult | None = None

        distance_span = max(0.08, 0.20 * center.distance_penalty)
        norm_span = max(0.01, 0.30 * center.norm_rate)
        base_span = 0.60
        skew_span = 0.75
        minimax_span = 1.2
        center_logit = logit(center.minimax_strength)

        def objective(trial: optuna.trial.Trial) -> float:
            distance = trial.suggest_float(
                "distance_penalty",
                clamp(center.distance_penalty - distance_span, 0.0, 2.5),
                clamp(center.distance_penalty + distance_span, 0.0, 2.5),
            )
            norm_rate = trial.suggest_float(
                "norm_rate",
                clamp(center.norm_rate - norm_span, 0.001, 0.25),
                clamp(center.norm_rate + norm_span, 0.001, 0.25),
            )
            base_log = trial.suggest_float(
                "base_penalty_log",
                clamp(center.base_penalty_log - base_span, -4.0, 4.0),
                clamp(center.base_penalty_log + base_span, -4.0, 4.0),
            )
            skew = trial.suggest_float(
                "penalty_skew",
                clamp(center.penalty_skew - skew_span, -3.0, 3.0),
                clamp(center.penalty_skew + skew_span, -3.0, 3.0),
            )
            minimax_logit = trial.suggest_float(
                "minimax_logit",
                center_logit - minimax_span,
                center_logit + minimax_span,
            )
            minimax = sigmoid(minimax_logit)
            low_penalty, high_penalty = base_skew_to_penalties(base_log, skew)

            params = HyperParams(
                distance_penalty=distance,
                norm_rate=norm_rate,
                base_penalty_log=base_log,
                penalty_skew=skew,
                minimax_strength=minimax,
                low_penalty=low_penalty,
                high_penalty=high_penalty,
            )
            terms = self._evaluate_trial(descriptors, params, local_budget, trial.number)
            nonlocal best_result
            trial_result = TrialResult(trial_index=trial.number, params=params, terms=terms)
            if best_result is None or trial_result.terms.score > best_result.terms.score:
                best_result = trial_result
            return terms.score

        study.optimize(objective, n_trials=local_budget.total_trials, show_progress_bar=False)
        if best_result is None:
            raise RuntimeError("local optimization failed to evaluate any trials")
        return best_result

    def _sample_trial_params(self, trial: optuna.trial.Trial) -> HyperParams:
        distance_penalty = trial.suggest_float("distance_penalty", 0.0, 2.5)
        norm_rate = trial.suggest_float("norm_rate", 0.001, 0.25)
        base_penalty_log = trial.suggest_float("base_penalty_log", -4.0, 4.0)
        penalty_skew = trial.suggest_float("penalty_skew", -3.0, 3.0)
        minimax_logit = trial.suggest_float("minimax_logit", -4.0, 4.0)
        minimax_strength = sigmoid(minimax_logit)
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

    def _evaluate_trial(
        self,
        descriptors: Mapping[str, float],
        params: HyperParams,
        budget: OptimizationBudget,
        trial_index: int,
    ) -> ObjectiveTerms:
        accumulator = np.zeros(8, dtype=np.float64)

        for repeat in range(budget.repeats_per_trial):
            rng = np.random.default_rng((budget.seed + 31) * (trial_index + 1) * (repeat + 1))
            perturbed = self._perturb_descriptors(descriptors, rng)
            terms = score_hyperparams(perturbed, params)
            accumulator += np.array(
                [
                    terms.score,
                    terms.sep,
                    terms.intra,
                    terms.jitter,
                    terms.loud_err,
                    terms.mask_flicker,
                    terms.low_pen_loss,
                    terms.high_pen_loss,
                ],
                dtype=np.float64,
            )

        averaged = accumulator / float(budget.repeats_per_trial)
        return ObjectiveTerms(
            score=float(averaged[0]),
            sep=float(averaged[1]),
            intra=float(averaged[2]),
            jitter=float(averaged[3]),
            loud_err=float(averaged[4]),
            mask_flicker=float(averaged[5]),
            low_pen_loss=float(averaged[6]),
            high_pen_loss=float(averaged[7]),
        )

    @staticmethod
    def _perturb_descriptors(descriptors: Mapping[str, float], rng: np.random.Generator) -> dict[str, float]:
        perturbed: dict[str, float] = {}
        for key, value in descriptors.items():
            value = float(value)
            scale = 0.03 * max(abs(value), 0.1)
            noise = float(rng.normal(0.0, scale))
            if key == "silence_ratio":
                perturbed[key] = clamp(value + noise, 0.0, 1.0)
            elif key == "brightness_median":
                perturbed[key] = clamp(value + noise, 0.0, 1.0)
            else:
                perturbed[key] = value + noise
        return perturbed
