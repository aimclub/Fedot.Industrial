from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING, Protocol

from benchmark.experiments.kernel_learning.controls import apply_optional_limit, read_csv_env, read_positive_int_env
from benchmark.experiments.kernel_learning.datasets import (
    KernelLearningCustomDatasetPolicy,
    build_ucr_dataset_spec,
    normalize_custom_dataset_policy,
    resolve_ucr_dataset_plans,
)
from benchmark.industrial import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
)

if TYPE_CHECKING:
    from benchmark.industrial.experiments.registry import BenchmarkRunBundle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULTS_PATH = Path(__file__).with_name("defaults.json")
DEFAULTS_VERSION = "kernel_learning_benchmark_defaults@1"


@lru_cache(maxsize=1)
def load_kernel_learning_defaults(path: str | Path = DEFAULTS_PATH) -> dict[str, Any]:
    defaults_path = Path(path)
    payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Kernel Learning defaults root must be a mapping: {defaults_path}")
    version = str(payload.get("version", ""))
    if version != DEFAULTS_VERSION:
        raise ValueError(f"Unsupported Kernel Learning defaults version: {version}")
    return payload


def _experiment_payload(name: str) -> dict[str, Any]:
    experiments = load_kernel_learning_defaults().get("experiments", {})
    payload = experiments.get(name)
    if not isinstance(payload, dict):
        raise KeyError(f"Kernel Learning experiment defaults are missing: {name}")
    return payload


def _generator_set(name: str) -> tuple[str, ...]:
    generator_sets = load_kernel_learning_defaults().get("generator_sets", {})
    return tuple(str(item) for item in generator_sets.get(name, ()))


def _experiment_metrics(name: str) -> tuple[str, ...]:
    return tuple(str(item) for item in _experiment_payload(name).get("metrics", ()))


def _experiment_date(name: str) -> str:
    return str(_experiment_payload(name)["experiment_date"])


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _relative_project_path(path: str) -> Path:
    return PROJECT_ROOT / Path(path)


def _output_dir_from_template(name: str, experiment_date: str, key: str = "output_dir_template") -> str:
    template = str(_experiment_payload(name)[key])
    return template.format(experiment_date=experiment_date)


def _project_output_dir_from_template(name: str, experiment_date: str, key: str) -> Path:
    return _relative_project_path(_output_dir_from_template(name, experiment_date, key=key))


def _run_spec(name: str, run_name: str) -> RunSpec:
    payload = dict(_experiment_payload(name).get("run_spec", {}))
    payload["run_name"] = run_name
    return RunSpec(**payload)


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalize_payload(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _normalize_payload(item) for key, item in value.items()}
    return value


def _model_specs(group_name: str) -> tuple[ModelSpec, ...]:
    models = load_kernel_learning_defaults().get("models", {}).get(group_name, ())
    return tuple(ModelSpec(**_normalize_payload(model_payload)) for model_payload in models)


DEFAULT_UCR_EXPERIMENT_DATE = _experiment_date("ucr")
DEFAULT_TSER_EXPERIMENT_DATE = _experiment_date("tser")
DEFAULT_FORECASTING_EXPERIMENT_DATE = _experiment_date("forecasting")
DEFAULT_TWO_STAGE_EXPERIMENT_DATE = _experiment_date("two_stage_ucr")
DEFAULT_TSER_DATASETS = tuple(str(item) for item in _experiment_payload("tser").get("datasets", ()))
DEFAULT_M4_SUBSETS = tuple(str(item) for item in _experiment_payload("forecasting").get("subsets", ()))
DEFAULT_M4_SAMPLE_SIZE = int(_experiment_payload("forecasting").get("sample_size", 5))
NON_TOPOLOGICAL_GENERATORS = _generator_set("non_topological")
STAGE1_NON_TOPOLOGICAL_GENERATORS = _generator_set(
    str(_experiment_payload("two_stage_ucr").get("generator_set", "stage1_non_topological"))
)
STAGE_METRICS = _experiment_metrics("two_stage_ucr")
DEFAULT_UCR_CUSTOM_DATASET_POLICY = normalize_custom_dataset_policy(
    _experiment_payload("ucr").get("custom_dataset_policy", KernelLearningCustomDatasetPolicy.UCR_ONLY.value)
)
DEFAULT_TWO_STAGE_CUSTOM_DATASET_POLICY = normalize_custom_dataset_policy(
    _experiment_payload("two_stage_ucr").get("custom_dataset_policy", KernelLearningCustomDatasetPolicy.UCR_ONLY.value)
)
DEFAULT_STAGE1_RUN_ID = _optional_str(_experiment_payload("two_stage_ucr").get("stage1_run_id"))
DEFAULT_STAGE1_RUN_POLICY = str(_experiment_payload("two_stage_ucr").get("stage1_run_policy", "latest"))
DEFAULT_STAGE2_TIMEOUT_MINUTES = int(_experiment_payload("two_stage_ucr").get("timeout_minutes", 5))
DEFAULT_STAGE2_POP_SIZE = int(_experiment_payload("two_stage_ucr").get("pop_size", 5))


class KernelLearningSuiteConfig(Protocol):
    def build_suite_config(self) -> BenchmarkSuiteConfig:
        ...


@dataclass(frozen=True)
class KernelLearningUCRExperimentConfig:
    data_root: str | Path = _relative_project_path(str(_experiment_payload("ucr")["data_root"]))
    datasets: tuple[str, ...] = ()
    dataset_limit: int | None = None
    allowed_dataset_names: Sequence[str] | None = None
    custom_dataset_policy: KernelLearningCustomDatasetPolicy | str = DEFAULT_UCR_CUSTOM_DATASET_POLICY
    experiment_date: str = DEFAULT_UCR_EXPERIMENT_DATE
    output_dir: str | Path | None = None
    persist_on_run: bool = True
    run_name: str = "kernel_learning_ucr_suite"

    @classmethod
    def from_env(cls) -> "KernelLearningUCRExperimentConfig":
        return cls(
            datasets=read_csv_env("KERNEL_LEARNING_UCR_DATASETS"),
            dataset_limit=read_positive_int_env("KERNEL_LEARNING_UCR_LIMIT"),
        )

    def resolve_dataset_plans(self):
        allowed_names = self.allowed_dataset_names
        if allowed_names is None:
            allowed_names = _load_default_ucr_allowed_names()
        return resolve_ucr_dataset_plans(
            data_root=self.data_root,
            datasets=self.datasets,
            dataset_limit=self.dataset_limit,
            allowed_dataset_names=allowed_names,
            custom_dataset_policy=self.custom_dataset_policy,
        )

    def resolve_dataset_names(self) -> tuple[str, ...]:
        return tuple(plan.name for plan in self.resolve_dataset_plans())

    def build_suite_config(self) -> BenchmarkSuiteConfig:
        return BenchmarkSuiteConfig(
            task_type=TaskType.TS_CLASSIFICATION,
            datasets=tuple(
                build_ucr_dataset_spec(plan, data_root=self.data_root)
                for plan in self.resolve_dataset_plans()
            ),
            models=build_ucr_kernel_learning_models(),
            metrics=_experiment_metrics("ucr"),
            artifact_spec=ArtifactSpec(
                output_dir=str(self.output_dir or _output_dir_from_template("ucr", self.experiment_date)),
                persist_on_run=self.persist_on_run,
            ),
            run_spec=_run_spec("ucr", self.run_name),
        )


@dataclass(frozen=True)
class KernelLearningTSERExperimentConfig:
    data_root: str | Path = _relative_project_path(str(_experiment_payload("tser")["data_root"]))
    datasets: tuple[str, ...] = DEFAULT_TSER_DATASETS
    dataset_limit: int | None = None
    experiment_date: str = DEFAULT_TSER_EXPERIMENT_DATE
    output_dir: str | Path | None = None
    persist_on_run: bool = True
    run_name: str = "kernel_learning_tser_suite"

    @classmethod
    def from_env(cls) -> "KernelLearningTSERExperimentConfig":
        env_datasets = read_csv_env("KERNEL_LEARNING_TSER_DATASETS")
        return cls(
            datasets=env_datasets or cls.datasets,
            dataset_limit=read_positive_int_env("KERNEL_LEARNING_TSER_LIMIT"),
        )

    def resolve_dataset_names(self) -> tuple[str, ...]:
        return apply_optional_limit(self.datasets, self.dataset_limit)

    def build_suite_config(self) -> BenchmarkSuiteConfig:
        return BenchmarkSuiteConfig(
            task_type=TaskType.TS_REGRESSION,
            datasets=tuple(
                DatasetSpec(
                    benchmark="local_tser",
                    dataset_name=dataset_name,
                    adapter_options={
                        "local_data_root": str(self.data_root),
                        "download_if_missing": False,
                    },
                )
                for dataset_name in self.resolve_dataset_names()
            ),
            models=build_tser_kernel_learning_models(),
            metrics=_experiment_metrics("tser"),
            artifact_spec=ArtifactSpec(
                output_dir=str(self.output_dir or _output_dir_from_template("tser", self.experiment_date)),
                persist_on_run=self.persist_on_run,
            ),
            run_spec=_run_spec("tser", self.run_name),
        )


@dataclass(frozen=True)
class KernelLearningM4ExperimentConfig:
    subsets: tuple[str, ...] = DEFAULT_M4_SUBSETS
    sample_size: int = DEFAULT_M4_SAMPLE_SIZE
    experiment_date: str = DEFAULT_FORECASTING_EXPERIMENT_DATE
    output_dir: str | Path | None = None
    persist_on_run: bool = True
    run_name: str = "kernel_learning_forecasting_suite"

    @classmethod
    def from_env(cls) -> "KernelLearningM4ExperimentConfig":
        return cls(
            subsets=read_csv_env("KERNEL_LEARNING_M4_SUBSETS") or cls.subsets,
            sample_size=read_positive_int_env("KERNEL_LEARNING_M4_SAMPLE_SIZE", cls.sample_size) or cls.sample_size,
        )

    def build_suite_config(self) -> BenchmarkSuiteConfig:
        return BenchmarkSuiteConfig(
            task_type=TaskType.FORECASTING,
            datasets=tuple(
                DatasetSpec(
                    benchmark="m4",
                    dataset_name=f"m4_{subset.lower()}_kernel_learning",
                    subset=subset,
                    sample_size=self.sample_size,
                    adapter_options={"use_local_files": True},
                )
                for subset in self.subsets
            ),
            models=build_forecasting_kernel_learning_models(),
            metrics=_experiment_metrics("forecasting"),
            artifact_spec=ArtifactSpec(
                output_dir=str(
                    self.output_dir or _output_dir_from_template("forecasting", self.experiment_date)
                ),
                persist_on_run=self.persist_on_run,
            ),
            run_spec=_run_spec("forecasting", self.run_name),
        )


@dataclass(frozen=True)
class KernelLearningTwoStageUCRExperimentConfig:
    data_root: str | Path = _relative_project_path(str(_experiment_payload("two_stage_ucr")["data_root"]))
    datasets: tuple[str, ...] = ()
    stage1_output_dir: str | Path = _project_output_dir_from_template(
        "two_stage_ucr",
        DEFAULT_TWO_STAGE_EXPERIMENT_DATE,
        "stage1_output_dir_template",
    )
    stage2_output_dir: str | Path = _project_output_dir_from_template(
        "two_stage_ucr",
        DEFAULT_TWO_STAGE_EXPERIMENT_DATE,
        "stage2_output_dir_template",
    )
    stage1_run_id: str | None = DEFAULT_STAGE1_RUN_ID
    stage1_run_policy: str = DEFAULT_STAGE1_RUN_POLICY
    run_stage1: bool = False
    allowed_dataset_names: Sequence[str] | None = None
    custom_dataset_policy: KernelLearningCustomDatasetPolicy | str = DEFAULT_TWO_STAGE_CUSTOM_DATASET_POLICY
    generator_names: tuple[str, ...] = STAGE1_NON_TOPOLOGICAL_GENERATORS
    metrics: tuple[str, ...] = STAGE_METRICS
    timeout_minutes: int = DEFAULT_STAGE2_TIMEOUT_MINUTES
    pop_size: int = DEFAULT_STAGE2_POP_SIZE

    def resolve_stage1_dataset_plans(self):
        allowed_names = self.allowed_dataset_names
        if allowed_names is None:
            allowed_names = _load_default_ucr_allowed_names()
        return resolve_ucr_dataset_plans(
            data_root=self.data_root,
            datasets=self.datasets,
            allowed_dataset_names=allowed_names,
            custom_dataset_policy=self.custom_dataset_policy,
        )

    def resolve_stage1_dataset_names(self) -> tuple[str, ...]:
        return tuple(plan.name for plan in self.resolve_stage1_dataset_plans())

    def resolve_stage1_run_id(self) -> str | None:
        explicit_run_id = _optional_str(self.stage1_run_id)
        if explicit_run_id is not None:
            return explicit_run_id
        policy = str(self.stage1_run_policy).strip().lower()
        if policy == "latest":
            return None
        raise ValueError(f"Unsupported stage1_run_policy: {self.stage1_run_policy}")

    def load_or_run_stage1(self):
        from fedot_ind.core.kernel_learning.experiments_api import (
            KernelLearningStage1Runner,
            load_stage1_result_from_artifacts,
            resolve_existing_stage1_run_dir,
        )

        if self.run_stage1:
            return KernelLearningStage1Runner(
                data_root=self.data_root,
                output_dir=self.stage1_output_dir,
                datasets=self.resolve_stage1_dataset_names(),
                allowed_dataset_names=self.allowed_dataset_names or _load_default_ucr_allowed_names() or (),
                custom_dataset_policy=self.custom_dataset_policy,
                generator_names=self.generator_names,
                metrics=self.metrics,
            ).run()

        run_dir = resolve_existing_stage1_run_dir(
            stage1_output_dir=self.stage1_output_dir,
            run_id=self.resolve_stage1_run_id(),
        )
        return load_stage1_result_from_artifacts(
            run_dir,
            data_root=self.data_root,
            fallback_generators=self.generator_names,
            fallback_metrics=self.metrics,
        )

    def run_stage2(self, stage1_result):
        from fedot_ind.core.kernel_learning.experiments_api import KernelLearningStage2Runner

        return KernelLearningStage2Runner(
            output_dir=self.stage2_output_dir,
            metrics=self.metrics,
            timeout_minutes=self.timeout_minutes,
            pop_size=self.pop_size,
        ).run(stage1_result)


def build_ucr_kernel_learning_models() -> tuple[ModelSpec, ...]:
    return _model_specs("ucr")


def build_tser_kernel_learning_models() -> tuple[ModelSpec, ...]:
    return _model_specs("tser")


def build_forecasting_kernel_learning_models() -> tuple[ModelSpec, ...]:
    return _model_specs("forecasting")


def run_kernel_learning_suite(config: KernelLearningSuiteConfig) -> "BenchmarkRunBundle":
    from benchmark.industrial.experiments.registry import run_registered_suite

    return run_registered_suite(config.build_suite_config())


def print_benchmark_run_bundle(bundle: "BenchmarkRunBundle") -> None:
    print(f"Run ID: {bundle.result.run_id}")
    print(f"Output dir: {bundle.result.config.artifact_spec.output_dir}")
    print(f"Run dir: {bundle.run_dir}")
    print(f"Registry entry: {bundle.registry_entry_path}")


def _load_default_ucr_allowed_names() -> tuple[str, ...] | None:
    try:
        from fedot_ind.core.repository.constanst_repository import UNI_CLF_BENCH
    except ModuleNotFoundError:
        return None
    return tuple(UNI_CLF_BENCH)
