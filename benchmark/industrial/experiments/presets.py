from __future__ import annotations

import inspect
import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from benchmark.industrial.api import run_forecasting_benchmark_suite, run_tsc_benchmark_suite, run_tser_benchmark_suite
from benchmark.industrial.core import ArtifactSpec, BenchmarkSuiteConfig, DatasetSpec, ModelSpec, RunSpec, TaskType
from benchmark.industrial.experiments.fusion_over_raw.expand import build_fusion_over_raw_smoke_suite_config

PRESET_DEFAULTS_PATH = Path(__file__).with_name('preset_defaults.json')
PRESET_DEFAULTS_VERSION = 'benchmark_industrial_preset_defaults@1'

SUITE_RUNNERS_BY_TASK: dict[TaskType, Callable[[BenchmarkSuiteConfig], Any]] = {
    TaskType.FORECASTING: run_forecasting_benchmark_suite,
    TaskType.TS_CLASSIFICATION: run_tsc_benchmark_suite,
    TaskType.TS_REGRESSION: run_tser_benchmark_suite,
}


@lru_cache(maxsize=1)
def load_preset_defaults(path: str | Path = PRESET_DEFAULTS_PATH) -> dict[str, Any]:
    defaults_path = Path(path)
    payload = json.loads(defaults_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Benchmark preset defaults root must be a mapping: {defaults_path}')
    version = str(payload.get('version', ''))
    if version != PRESET_DEFAULTS_VERSION:
        raise ValueError(f'Unsupported benchmark preset defaults version: {version}')
    return payload


def _default_metrics(group_name: str) -> tuple[str, ...]:
    return tuple(str(item) for item in load_preset_defaults().get('metrics', {}).get(group_name, ()))


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalize_payload(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _normalize_payload(item) for key, item in value.items()}
    return value


def _default_model_specs(group_name: str) -> tuple[ModelSpec, ...]:
    models = load_preset_defaults().get('models', {}).get(group_name, ())
    return tuple(ModelSpec(**_normalize_payload(model_payload)) for model_payload in models)


DEFAULT_PRESET_OUTPUT_DIR = Path(str(load_preset_defaults()['output_dir_root']))
DEFAULT_OKHS_SMOOTHING_SERIES_IDS = tuple(
    str(item) for item in load_preset_defaults().get('okhs_smoothing_series_ids', ())
)


class BenchmarkPresetError(ValueError):
    pass


def build_local_m4_suite_config(
        *,
        subset: str = 'daily',
        sample_size: int | None = 3,
        random_seed: int = 42,
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        models: tuple[ModelSpec, ...] | None = None,
        include_optional_external: bool = False,
) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='m4',
                dataset_name=f'm4_{subset.lower()}_local',
                subset=subset,
                sample_size=sample_size,
                random_seed=random_seed,
                adapter_options={'use_local_files': True},
            ),
        ),
        models=models or _default_forecasting_models(include_optional_external=include_optional_external),
        metrics=_default_metrics('forecasting'),
        artifact_spec=_artifact_spec(output_dir, persist_on_run, 'm4'),
        run_spec=RunSpec(run_name=f'm4_{subset.lower()}_suite', primary_metric='mae'),
    )


def build_local_monash_suite_config(
        *,
        dataset_name: str = 'Bitcoin',
        subset: str = 'daily',
        sample_size: int | None = 3,
        random_seed: int = 42,
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        models: tuple[ModelSpec, ...] | None = None,
        include_optional_external: bool = False,
) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='monash',
                dataset_name=dataset_name,
                subset=subset,
                sample_size=sample_size,
                random_seed=random_seed,
                adapter_options={'use_local_files': True},
            ),
        ),
        models=models or _default_forecasting_models(include_optional_external=include_optional_external),
        metrics=_default_metrics('forecasting'),
        artifact_spec=_artifact_spec(output_dir, persist_on_run, 'monash'),
        run_spec=RunSpec(run_name=f'monash_{dataset_name.lower()}_suite', primary_metric='mae'),
    )


def build_local_ucr_suite_config(
        *,
        dataset_name: str = 'Lightning7',
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        models: tuple[ModelSpec, ...] | None = None,
) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=(DatasetSpec(benchmark='ucr', dataset_name=dataset_name),),
        models=models or _default_classification_models(),
        metrics=_default_metrics('classification'),
        artifact_spec=_artifact_spec(output_dir, persist_on_run, 'ucr'),
        run_spec=RunSpec(run_name=f'ucr_{dataset_name.lower()}_suite', primary_metric='accuracy'),
    )


def build_local_tser_suite_config(
        *,
        dataset_name: str = 'NaturalGasPricesSentiment',
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        models: tuple[ModelSpec, ...] | None = None,
) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        task_type=TaskType.TS_REGRESSION,
        datasets=(DatasetSpec(benchmark='local_tser', dataset_name=dataset_name),),
        models=models or _default_regression_models(),
        metrics=_default_metrics('regression'),
        artifact_spec=_artifact_spec(output_dir, persist_on_run, 'tser'),
        run_spec=RunSpec(run_name=f'tser_{dataset_name.lower()}_suite', primary_metric='rmse'),
    )


def build_local_okhs_smoothing_suite_config(
        *,
        subset: str = 'daily',
        series_ids: tuple[str, ...] = DEFAULT_OKHS_SMOOTHING_SERIES_IDS,
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        models: tuple[ModelSpec, ...] | None = None,
        anti_smoothing_policy: str = 'residual_bridge',
) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='m4',
                dataset_name=f'm4_{subset.lower()}_okhs_smoothing',
                subset=subset,
                series_ids=tuple(series_ids),
                adapter_options={'use_local_files': True},
            ),
        ),
        models=models or _default_okhs_smoothing_models(anti_smoothing_policy=anti_smoothing_policy),
        metrics=_default_metrics('forecasting'),
        artifact_spec=_artifact_spec(output_dir, persist_on_run, 'okhs_smoothing'),
        run_spec=RunSpec(run_name=f'm4_{subset.lower()}_okhs_smoothing', primary_metric='mae'),
    )


LOCAL_PRESET_BUILDERS: dict[str, Callable[..., BenchmarkSuiteConfig]] = {
    'm4': build_local_m4_suite_config,
    'monash': build_local_monash_suite_config,
    'okhs_smoothing': build_local_okhs_smoothing_suite_config,
    'ucr': build_local_ucr_suite_config,
    'tser': build_local_tser_suite_config,
    'fusion_over_raw_smoke': build_fusion_over_raw_smoke_suite_config,
}


def available_local_presets() -> tuple[str, ...]:
    return tuple(sorted(LOCAL_PRESET_BUILDERS))


def run_local_benchmark_preset(
        preset_name: str,
        *,
        dataset_name: str | None = None,
        subset: str | None = None,
        sample_size: int | None = None,
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        random_seed: int = 42,
        include_optional_external: bool = False,
        models: tuple[ModelSpec, ...] | None = None,
):
    normalized = preset_name.lower()
    builder = LOCAL_PRESET_BUILDERS.get(normalized)
    if builder is None:
        raise BenchmarkPresetError(
            f'Unsupported local benchmark preset: {preset_name}. '
            f'Available: {", ".join(available_local_presets())}'
        )

    candidate_kwargs = {
        'dataset_name': dataset_name,
        'subset': subset,
        'sample_size': sample_size,
        'output_dir': output_dir,
        'persist_on_run': persist_on_run,
        'random_seed': random_seed,
        'include_optional_external': include_optional_external,
        'models': models,
    }
    accepted = set(inspect.signature(builder).parameters)
    build_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in accepted and value is not None
    }
    # Booleans must remain explicit even when False.
    if 'persist_on_run' in accepted:
        build_kwargs['persist_on_run'] = persist_on_run
    if 'include_optional_external' in accepted:
        build_kwargs['include_optional_external'] = include_optional_external

    config = builder(**build_kwargs)
    if models is not None and 'models' not in accepted:
        config = replace(config, models=models)

    runner = SUITE_RUNNERS_BY_TASK.get(config.task_type)
    if runner is None:
        raise BenchmarkPresetError(f'No suite runner registered for task type: {config.task_type}')
    return runner(config)


def _artifact_spec(
        output_dir: str | Path | None,
        persist_on_run: bool,
        preset_name: str,
) -> ArtifactSpec:
    return ArtifactSpec(
        output_dir=str(Path(output_dir) if output_dir is not None else DEFAULT_PRESET_OUTPUT_DIR / preset_name),
        persist_on_run=persist_on_run,
    )


def _default_forecasting_models(*, include_optional_external: bool) -> tuple[ModelSpec, ...]:
    models = list(_default_model_specs('forecasting'))
    if include_optional_external:
        models.extend(_default_model_specs('forecasting_optional_external'))
    return tuple(models)


def _default_classification_models() -> tuple[ModelSpec, ...]:
    return _default_model_specs('classification')


def _default_regression_models() -> tuple[ModelSpec, ...]:
    return _default_model_specs('regression')


def _default_okhs_smoothing_models(*, anti_smoothing_policy: str) -> tuple[ModelSpec, ...]:
    models = []
    for spec in _default_model_specs('okhs_smoothing'):
        if spec.adapter_name != 'okhs':
            models.append(spec)
            continue
        params = dict(spec.params)
        params['anti_smoothing_policy'] = anti_smoothing_policy
        models.append(
            ModelSpec(
                adapter_name=spec.adapter_name,
                display_name=spec.display_name,
                tags=spec.tags,
                optional=spec.optional,
                params=params,
            )
        )
    return tuple(models)
