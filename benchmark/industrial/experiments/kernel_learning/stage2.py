from __future__ import annotations

from copy import deepcopy
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.industrial.classification import build_classification_dataset_adapter, compute_classification_metric
from benchmark.industrial.core import DatasetSpec, to_plain_data, write_json
from fedot_ind.core.kernel_learning.integration import KernelInitialPopulationError, KernelInitialPopulationBuilder
from fedot_ind.core.kernel_learning.selection import KernelImportanceItem, KernelImportanceReport
from .io import load_stage1_kernel_records
from .stage1 import DEFAULT_STAGE_METRICS

DEFAULT_STAGE2_OUTPUT_DIR = Path(
    "benchmark") / "results" / "kernel_learning" / "ucr_two_stage_optimization"

logger = logging.getLogger(__name__)


def importance_report_from_selection(selection: dict[str, Any]) -> KernelImportanceReport:
    importance = selection.get("kernel_importance") or {}
    raw_items = importance.get("items") or ()
    if raw_items:
        items = tuple(
            KernelImportanceItem(
                name=str(item["name"]),
                weight=float(item["weight"]),
                original_index=int(item.get("original_index", index)),
                rank=int(item.get("rank", index + 1)),
                selected_by=str(item.get("selected_by", "saved_importance")),
            )
            for index, item in enumerate(raw_items)
        )
        return KernelImportanceReport(
            items=items,
            selected_generators=tuple(item.name for item in items),
            selected_weights=tuple(item.weight for item in items),
            diagnostics=dict(importance.get("diagnostics", {})),
        )

    names = tuple(selection.get("important_generators")
                  or selection.get("selected_generators") or ())
    weights = tuple(float(weight) for weight in (
        selection.get("important_weights") or selection.get(
            "selected_weights") or ()
    ))
    if not weights:
        weights = tuple(1.0 / len(names) for _ in names) if names else ()
    items = tuple(
        KernelImportanceItem(
            name=name,
            weight=weights[index] if index < len(weights) else 0.0,
            original_index=index,
            rank=index + 1,
            selected_by="saved_selection",
        )
        for index, name in enumerate(names)
    )
    return KernelImportanceReport(
        items=items,
        selected_generators=tuple(item.name for item in items),
        selected_weights=tuple(item.weight for item in items),
        diagnostics={"source": "kernel_selection_artifact"},
    )


def build_stage2_initial_population(
        selection: dict[str, Any],
        *,
        build_pipelines: bool = True,
        lazy: bool = True,
        allow_empty_specs: bool = False,
):
    builder = KernelInitialPopulationBuilder(
        task_type="classification",
        head_model="rf",
        include_feature_union=True,
        max_union_size=3,
        allow_empty_specs=allow_empty_specs,
    )
    importance = importance_report_from_selection(selection)
    specs = builder.build_specs(importance)
    initial_population = []
    if build_pipelines:
        build_from_spec = builder.build_pipeline_builder_from_spec if lazy else builder.build_pipeline_from_spec
        initial_population = [build_from_spec(spec) for spec in specs]
    return builder, specs, initial_population


@dataclass
class KernelLearningStage2Runner:
    output_dir: str | Path = DEFAULT_STAGE2_OUTPUT_DIR
    metrics: tuple[str, ...] = DEFAULT_STAGE_METRICS
    timeout_minutes: int = 5
    pop_size: int = 5
    strict: bool = False

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

    def run(self, stage1_result, *, strict: bool | None = None) -> tuple[dict[str, Any], ...]:
        strict_mode = self.strict if strict is None else bool(strict)
        kernel_records = load_stage1_kernel_records(
            stage1_result.config.artifact_spec.output_dir, stage1_result.run_id)
        summaries = []
        for dataset_spec in stage1_result.config.datasets:
            kernel_record = kernel_records.get(dataset_spec.dataset_name)
            if kernel_record is None:
                summary = self._write_skipped_summary(
                    dataset_spec, reason="missing_stage1_kernel_record")
                summaries.append(summary)
                if strict_mode:
                    raise RuntimeError(
                        f"Stage2 skipped dataset {dataset_spec.dataset_name!r}: missing_stage1_kernel_record."
                    )
                continue
            summaries.append(self.iter_over_dataset(
                dataset_spec, kernel_record, strict=strict_mode))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / "stage2_summary.json", summaries)
        return tuple(summaries)

    def iter_over_dataset(
            self,
            dataset_spec: DatasetSpec,
            kernel_record: dict[str, Any],
            *,
            strict: bool | None = None,
    ) -> dict[str, Any]:
        output_dir = self._prepare_output_dir(dataset_spec)
        selection = kernel_record["kernel_selection"]
        builder = KernelInitialPopulationBuilder(
            task_type="classification", head_model="rf", allow_empty_specs=True)
        specs = ()
        summary = self._base_summary(dataset_spec, selection, specs, builder)
        try:
            builder, specs = self._build_initial_population(
                selection, output_dir)
            if not specs:
                reason = builder.diagnostics_.get(
                    "empty_specs_reason", "no_initial_population_specs")
                raise KernelInitialPopulationError(
                    f"Kernel initial population is empty: {reason}.")
            dataset = self._load_dataset(dataset_spec)
            fedot_config = self._build_fedot_config(output_dir, builder, specs)
            self._write_fedot_config(output_dir, fedot_config, specs)
            summary = self._base_summary(
                dataset_spec, selection, specs, builder)
            prediction = self._fit_predict(
                dataset, fedot_config, builder, specs)
            metrics = self._compute_metrics(dataset["test_y"], prediction)
            self._write_success_artifacts(
                output_dir, dataset["test_y"], prediction, metrics)
            summary.update({"status": "success", "metrics": metrics})
        except KernelInitialPopulationError as exc:
            self._write_failure_artifacts(output_dir)
            summary = self._base_summary(
                dataset_spec, selection, specs, builder)
            summary.update(self._error_payload(
                exc, reason="initial_population_error"))
            logger.exception(
                "Stage2 failed while building initial population for %s.", dataset_spec.dataset_name)
            write_json(output_dir / "optimizer_summary.json", summary)
            if self._strict_mode(strict):
                raise
            return summary
        except Exception as exc:
            self._write_failure_artifacts(output_dir)
            summary.update(self._error_payload(exc, reason="runtime_error"))
            logger.exception(
                "Stage2 failed while running dataset %s.", dataset_spec.dataset_name)
            write_json(output_dir / "optimizer_summary.json", summary)
            if self._strict_mode(strict):
                raise
            return summary
        write_json(output_dir / "optimizer_summary.json", summary)
        return summary

    def _strict_mode(self, strict: bool | None) -> bool:
        return self.strict if strict is None else bool(strict)

    def _write_skipped_summary(self, dataset_spec: DatasetSpec, *, reason: str) -> dict[str, Any]:
        output_dir = self._prepare_output_dir(dataset_spec)
        self._write_failure_artifacts(output_dir)
        summary = {
            "dataset_name": dataset_spec.dataset_name,
            "selected_generators": [],
            "initial_population_size": 0,
            "status": "skipped",
            "reason": reason,
            "builder_diagnostics": {},
        }
        write_json(output_dir / "optimizer_summary.json", summary)
        return summary

    def _error_payload(self, exc: Exception, *, reason: str) -> dict[str, Any]:
        return {
            "status": "failed",
            "reason": reason,
            "message": str(exc),
            "exception_type": exc.__class__.__name__,
            "traceback": traceback.format_exc(),
        }

    def _prepare_output_dir(self, dataset_spec: DatasetSpec) -> Path:
        output_dir = self.output_dir / dataset_spec.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _build_initial_population(self, selection: dict[str, Any], output_dir: Path):
        builder, specs, _ = build_stage2_initial_population(
            selection,
            build_pipelines=False,
            allow_empty_specs=True,
        )
        write_json(output_dir / "initial_population_specs.json",
                   [to_plain_data(spec) for spec in specs])
        return builder, specs

    def _load_dataset(self, dataset_spec: DatasetSpec) -> dict[str, np.ndarray]:
        record = build_classification_dataset_adapter(
            dataset_spec).load_dataset(dataset_spec)[0]
        return {
            "train_x": np.asarray(record.train_features, dtype=float),
            "train_y": np.asarray(record.train_target, dtype=object),
            "test_x": np.asarray(record.test_features, dtype=float),
            "test_y": np.asarray(record.test_target, dtype=object),
        }

    def _build_fedot_config(self, output_dir: Path, builder, specs) -> dict[str, Any]:
        from fedot_ind.core.repository.config_repository import (
            DEFAULT_CLF_AUTOML_CONFIG,
            DEFAULT_CLF_LEARNING_CONFIG,
            DEFAULT_COMPUTE_CONFIG,
        )

        automl_config = deepcopy(DEFAULT_CLF_AUTOML_CONFIG)
        automl_config["optimisation_strategy"] = {
            "optimisation_agent": "Industrial",
            "optimisation_strategy": {
                "mutation_agent": "random",
                "mutation_strategy": "growth_mutation_strategy",
            },
        }
        narrowed_operations = builder.restrict_available_operations(
            automl_config.get("available_operations"), specs)
        if narrowed_operations is not None:
            automl_config["available_operations"] = narrowed_operations

        learning_config = deepcopy(DEFAULT_CLF_LEARNING_CONFIG)
        learning_config["learning_strategy_params"] = {
            **learning_config.get("learning_strategy_params", {}),
            "timeout": self.timeout_minutes,
            "pop_size": self.pop_size,
            "with_tuning": False,
        }
        learning_config["optimisation_loss"] = {"quality_loss": "f1"}

        compute_config = deepcopy(DEFAULT_COMPUTE_CONFIG)
        compute_config["output_folder"] = str(output_dir)
        compute_config["automl_folder"] = {
            "optimisation_history": str(output_dir / "opt_hist"),
            "composition_results": str(output_dir / "comp_res"),
        }
        return {
            "industrial_config": {
                "problem": "classification",
                "optimizer": "IndustrialEvoOptimizer",
            },
            "automl_config": automl_config,
            "learning_config": learning_config,
            "compute_config": compute_config,
        }

    def _write_fedot_config(self, output_dir: Path, fedot_config: dict[str, Any], specs) -> None:
        serializable_config = deepcopy(fedot_config)
        serializable_config["automl_config"]["initial_assumption"] = [
            to_plain_data(spec) for spec in specs]
        write_json(output_dir / "fedot_config.json", serializable_config)

    def _fit_predict(self, dataset: dict[str, np.ndarray], fedot_config: dict[str, Any], builder, specs) -> np.ndarray:
        from fedot_ind.api.main import FedotIndustrial
        from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer

        runtime_config = deepcopy(fedot_config)
        runtime_config["industrial_config"]["optimizer"] = IndustrialEvoOptimizer
        runtime_config["automl_config"]["initial_assumption"] = [
            builder.build_pipeline_builder_from_spec(spec)
            for spec in specs
        ]
        model = FedotIndustrial(**runtime_config)
        model.fit((dataset["train_x"], dataset["train_y"]))
        return np.asarray(model.predict((dataset["test_x"], dataset["test_y"])), dtype=object).reshape(-1)

    def _compute_metrics(self, test_y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
        return {
            metric_name: compute_classification_metric(
                metric_name, test_y, prediction)
            for metric_name in self.metrics
        }

    def _write_success_artifacts(
            self,
            output_dir: Path,
            test_y: np.ndarray,
            prediction: np.ndarray,
            metrics: dict[str, float],
    ) -> None:
        pd.DataFrame(
            {
                "sample_index": np.arange(1, len(test_y) + 1),
                "y_true": test_y.astype(str),
                "y_pred": prediction.astype(str),
            }
        ).to_csv(output_dir / "predictions.csv", index=False)
        write_json(output_dir / "metrics.json", metrics)

    def _write_failure_artifacts(self, output_dir: Path) -> None:
        write_json(output_dir / "metrics.json", {})
        pd.DataFrame(columns=("sample_index", "y_true", "y_pred")).to_csv(
            output_dir / "predictions.csv",
            index=False,
        )

    def _base_summary(self, dataset_spec: DatasetSpec, selection: dict[str, Any], specs, builder) -> dict[str, Any]:
        return {
            "dataset_name": dataset_spec.dataset_name,
            "selected_generators": selection.get("important_generators") or selection.get("selected_generators"),
            "initial_population_size": len(specs),
            "status": "started",
            "builder_diagnostics": builder.diagnostics_,
        }


def run_stage2_for_dataset(dataset_spec: DatasetSpec, kernel_record: dict[str, Any]) -> dict[str, Any]:
    return KernelLearningStage2Runner().iter_over_dataset(dataset_spec, kernel_record)
