"""Expand fusion_over_raw defaults into Industrial BenchmarkSuiteConfig."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.industrial.core import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    TaskType,
)
from benchmark.industrial.experiments.fusion_over_raw.config import (
    BASELINE_PROFILES,
    BOTTLENECK_FUSIONS,
    MVP_FUSIONS,
    FusionOverRawConfigError,
    load_fusion_over_raw_defaults,
)

IMAGE_MODALITIES = frozenset({'gaf', 'stft', 'mtf'})


def build_fusion_over_raw_suite_config(
        *,
        defaults: Mapping[str, Any] | None = None,
        datasets: Sequence[str] | None = None,
        seeds: Sequence[int] | None = None,
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        include_bottleneck: bool = False,
        include_external_baseline: bool | None = None,
        local_data_root: str | Path | None = None,
        download_if_missing: bool = True,
        training_overrides: Mapping[str, Any] | None = None,
) -> BenchmarkSuiteConfig:
    payload = dict(defaults or load_fusion_over_raw_defaults())
    dataset_names = tuple(str(name) for name in (datasets if datasets is not None else payload['datasets']))
    seed_values = tuple(int(seed) for seed in (seeds if seeds is not None else payload['seeds']))
    if not dataset_names:
        raise FusionOverRawConfigError('At least one dataset is required.')
    if not seed_values:
        raise FusionOverRawConfigError('At least one seed is required.')

    adapter_options: dict[str, Any] = {'download_if_missing': download_if_missing}
    if local_data_root is not None:
        adapter_options['local_data_root'] = str(local_data_root)

    dataset_specs = tuple(
        DatasetSpec(
            benchmark='ucr',
            dataset_name=name,
            adapter_options=dict(adapter_options),
        )
        for name in dataset_names
    )
    model_specs = expand_fusion_over_raw_model_specs(
        payload,
        seeds=seed_values,
        include_bottleneck=include_bottleneck,
        include_external_baseline=include_external_baseline,
        training_overrides=training_overrides,
    )
    metrics = tuple(str(item) for item in payload.get('metrics', ('accuracy', 'balanced_accuracy', 'f1_macro')))
    primary_metric = str(payload.get('primary_metric', 'f1_macro'))
    resolved_output = Path(output_dir if output_dir is not None else payload.get(
        'output_dir', 'benchmark/results/industrial_presets/fusion_over_raw'))
    return BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=dataset_specs,
        models=model_specs,
        metrics=metrics,
        artifact_spec=ArtifactSpec(output_dir=str(resolved_output), persist_on_run=persist_on_run),
        run_spec=RunSpec(
            run_name=str(payload.get('experiment_name', 'fusion_over_raw')),
            primary_metric=primary_metric,
            random_seed=seed_values[0],
        ),
    )


def build_fusion_over_raw_smoke_suite_config(
        *,
        defaults: Mapping[str, Any] | None = None,
        output_dir: str | Path | None = None,
        persist_on_run: bool = True,
        train_features: np.ndarray | None = None,
        train_target: np.ndarray | None = None,
        test_features: np.ndarray | None = None,
        test_target: np.ndarray | None = None,
) -> BenchmarkSuiteConfig:
    payload = dict(defaults or load_fusion_over_raw_defaults())
    smoke = dict(payload.get('smoke') or {})
    seed = int(smoke.get('seed', 42))
    epochs = int(smoke.get('epochs', 1))
    patience = int(smoke.get('patience', 1))
    d_model = int(smoke.get('d_model', 16))
    batch_size = int(smoke.get('batch_size', 8))
    baseline_name = str(smoke.get('baseline', 'raw'))
    fusion_modalities = tuple(str(item) for item in smoke.get('fusion_modalities', ('raw', 'stats')))
    fusion_method = str(smoke.get('fusion_method', 'concat'))
    dataset_name = str(smoke.get('dataset_name', 'fusion_over_raw_smoke'))
    resolved_output = Path(
        output_dir
        if output_dir is not None
        else smoke.get('output_dir', 'benchmark/results/industrial_presets/fusion_over_raw_smoke')
    )

    rng = np.random.default_rng(seed)
    if train_features is None:
        train_features = rng.normal(size=(16, 24))
    if train_target is None:
        train_target = np.asarray(['a', 'b'] * 8, dtype=object)
    if test_features is None:
        test_features = rng.normal(size=(4, 24))
    if test_target is None:
        test_target = np.asarray(['a', 'b', 'a', 'b'], dtype=object)

    training_overrides = {
        'epochs': epochs,
        'patience': patience,
        'max_batch_size': batch_size,
        'd_model': d_model,
    }
    models = (
        _build_future_model_spec(
            payload,
            profile_or_modalities=(baseline_name,),
            fusion_method='concat',
            seed=seed,
            role='baseline',
            training_overrides=training_overrides,
        ),
        _build_future_model_spec(
            payload,
            profile_or_modalities=fusion_modalities,
            fusion_method=fusion_method,
            seed=seed,
            role='fusion',
            training_overrides=training_overrides,
        ),
    )
    metrics = tuple(str(item) for item in payload.get('metrics', ('accuracy', 'balanced_accuracy', 'f1_macro')))
    return BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=(
            DatasetSpec(
                benchmark='in_memory_tsc',
                dataset_name=dataset_name,
                adapter_options={
                    'record': {
                        'train_features': train_features,
                        'train_target': train_target,
                        'test_features': test_features,
                        'test_target': test_target,
                    }
                },
            ),
        ),
        models=models,
        metrics=metrics,
        artifact_spec=ArtifactSpec(output_dir=str(resolved_output), persist_on_run=persist_on_run),
        run_spec=RunSpec(
            run_name='fusion_over_raw_smoke',
            primary_metric=str(payload.get('primary_metric', 'f1_macro')),
            random_seed=seed,
        ),
    )


def expand_fusion_over_raw_model_specs(
        defaults: Mapping[str, Any] | None = None,
        *,
        seeds: Sequence[int] | None = None,
        include_bottleneck: bool = False,
        include_external_baseline: bool | None = None,
        training_overrides: Mapping[str, Any] | None = None,
) -> tuple[ModelSpec, ...]:
    payload = dict(defaults or load_fusion_over_raw_defaults())
    seed_values = tuple(int(seed) for seed in (seeds if seeds is not None else payload['seeds']))
    models: list[ModelSpec] = []

    for seed in seed_values:
        for baseline in payload.get('single_view_baselines', ()):
            profile = tuple(str(item) for item in baseline)
            models.append(
                _build_future_model_spec(
                    payload,
                    profile_or_modalities=profile,
                    fusion_method='concat',
                    seed=seed,
                    role='baseline',
                    training_overrides=training_overrides,
                )
            )
        for experiment in payload.get('fusion_experiments', ()):
            modalities = tuple(str(item) for item in experiment.get('modalities', ()))
            for fusion_method in experiment.get('fusions', ()):
                method = str(fusion_method)
                if method in BOTTLENECK_FUSIONS and not include_bottleneck:
                    continue
                if method not in MVP_FUSIONS and method not in BOTTLENECK_FUSIONS:
                    raise FusionOverRawConfigError(f'Unsupported fusion method: {method}')
                if method in MVP_FUSIONS or include_bottleneck:
                    models.append(
                        _build_future_model_spec(
                            payload,
                            profile_or_modalities=modalities,
                            fusion_method=method,
                            seed=seed,
                            role='fusion',
                            training_overrides=training_overrides,
                        )
                    )

        external = dict(payload.get('external_baseline') or {})
        enabled = bool(
            external.get(
                'enabled',
                True)) if include_external_baseline is None else include_external_baseline
        if enabled:
            models.append(_build_minirocket_model_spec(payload, seed=seed))

    return tuple(models)


def model_family_name(display_name: str) -> str:
    """Strip seed suffix used for unique Industrial display names."""
    name = str(display_name)
    marker = '__seed'
    if marker in name:
        return name.rsplit(marker, 1)[0]
    return name


def parse_model_tags(tags: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for tag in tags:
        text = str(tag)
        if ':' in text:
            key, value = text.split(':', 1)
            parsed[key] = value
        else:
            parsed[text] = text
    return parsed


def _build_future_model_spec(
        payload: Mapping[str, Any],
        *,
        profile_or_modalities: Sequence[str],
        fusion_method: str,
        seed: int,
        role: str,
        training_overrides: Mapping[str, Any] | None = None,
) -> ModelSpec:
    profiles = tuple(str(item) for item in profile_or_modalities)
    if not profiles:
        raise FusionOverRawConfigError('Modalities/profile list must be non-empty.')

    is_raw_larger = profiles == ('raw_larger',)
    if any(item not in BASELINE_PROFILES and item not in {'raw', 'stats', 'gaf', 'stft', 'mtf'} for item in profiles):
        unknown = [item for item in profiles if item not in BASELINE_PROFILES]
        raise FusionOverRawConfigError(f'Unsupported modalities/profile: {unknown}')

    modalities = ('raw',) if is_raw_larger else profiles
    family = '+'.join(profiles) if role == 'baseline' else f"{'+'.join(modalities)}__{fusion_method}"
    display_name = f'{family}__seed{seed}'
    tags = (
        role,
        'future',
        f'modalities:{"+".join(modalities)}',
        f'fusion:{fusion_method if role == "fusion" else "single"}',
        f'profile:{"+".join(profiles)}',
        f'family:{family}',
        f'seed:{seed}',
        'bottleneck' if fusion_method in BOTTLENECK_FUSIONS else 'mvp',
    )

    model_cfg = dict(payload.get('model') or {})
    overrides = dict(training_overrides or {})
    d_model = int(overrides.pop('d_model', model_cfg.get('d_model', 128)))
    resolved_fusion = fusion_method if role == 'fusion' else 'concat'
    params = {
        'fusion_method': resolved_fusion,
        'd_model': d_model,
        'modalities': list(modalities),
        'head_hidden_dim': model_cfg.get('head_hidden_dim'),
        'head_dropout': float(model_cfg.get('head_dropout', 0.2)),
        'encoder_kwargs': _encoder_kwargs_for_modalities(payload, modalities, raw_larger=is_raw_larger),
        'fusion_kwargs': _fusion_kwargs_for_method(payload, resolved_fusion),
        'preparation': {
            'transformation_config': _transformation_config(payload, modalities),
        },
        'training': _training_params(payload, seed=seed, overrides=overrides),
    }
    return ModelSpec(
        adapter_name='future_multimodal_classifier',
        display_name=display_name,
        tags=tags,
        optional=True,
        params=params,
    )


def _build_minirocket_model_spec(payload: Mapping[str, Any], *, seed: int) -> ModelSpec:
    external = dict(payload.get('external_baseline') or {})
    params = dict(external.get('params') or {})
    params['seed'] = seed
    display_name = f"{external.get('display_name', 'MiniRocketRidge')}__seed{seed}"
    return ModelSpec(
        adapter_name=str(external.get('adapter_name', 'minirocket_ridge_classifier')),
        display_name=display_name,
        tags=(
            'external',
            'baseline',
            'minirocket',
            'family:MiniRocketRidge',
            f'seed:{seed}',
        ),
        params=params,
    )


def _training_params(
        payload: Mapping[str, Any],
        *,
        seed: int,
        overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    training = dict(payload.get('training') or {})
    validation = dict(payload.get('validation') or {})
    merged = dict(training)
    if overrides:
        merged.update(overrides)
    return {
        'epochs': int(merged.get('epochs', 100)),
        'patience': int(merged.get('patience', 15)),
        'learning_rate': float(merged.get('lr', merged.get('learning_rate', 0.001))),
        'batch_size': int(merged.get('max_batch_size', merged.get('batch_size', 32))),
        'validation_fraction': float(validation.get('val_size', 0.2)),
        'seed': int(seed),
        'device': str(merged.get('device', 'cpu')),
    }


def _transformation_config(payload: Mapping[str, Any], modalities: Sequence[str]) -> dict[str, Any]:
    representations = dict(payload.get('representations') or {})
    config: dict[str, Any] = {
        'raw': {'per_sample_z_normalize': False},
    }
    if 'stats' in modalities:
        config['stats'] = dict(representations.get('stat_params') or {})
    if 'gaf' in modalities:
        config['gaf'] = dict(representations.get('gaf_params') or {})
    if 'stft' in modalities:
        config['stft'] = dict(representations.get('stft_params') or {})
    return config


def _encoder_kwargs_for_modalities(
        payload: Mapping[str, Any],
        modalities: Sequence[str],
        *,
        raw_larger: bool = False,
) -> dict[str, dict[str, Any]]:
    model_cfg = dict(payload.get('model') or {})
    kwargs: dict[str, dict[str, Any]] = {}
    for modality in modalities:
        if modality == 'raw':
            source = model_cfg.get('raw_larger_encoder' if raw_larger else 'raw_encoder') or {}
            kwargs[modality] = deepcopy(source)
        elif modality == 'stats':
            kwargs[modality] = deepcopy(model_cfg.get('stats_encoder') or {})
        elif modality in IMAGE_MODALITIES:
            kwargs[modality] = deepcopy(model_cfg.get('image_encoder') or {})
    return kwargs


def _fusion_kwargs_for_method(payload: Mapping[str, Any], fusion_method: str) -> dict[str, Any]:
    """Map shared prototype fusion hyperparams onto Industrial fusion constructors."""
    source = dict((payload.get('model') or {}).get('fusion') or {})
    if fusion_method in {'concat', 'gated'}:
        return {
            'hidden_dim': int(source.get('fusion_hidden_dim', 128)),
            'dropout': float(source.get('fusion_dropout', 0.2)),
        }
    if fusion_method == 'film':
        return {
            'context_hidden_dim': int(source.get('context_hidden_dim', 128)),
            'film_hidden_dim': int(source.get('film_hidden_dim', 128)),
            'dropout': float(source.get('fusion_dropout', 0.2)),
        }
    if fusion_method == 'raw_centered_residual':
        return {
            'context_hidden_dim': int(source.get('context_hidden_dim', 128)),
            'delta_hidden_dim': int(source.get('delta_hidden_dim', 128)),
            'dropout': float(source.get('residual_dropout', source.get('fusion_dropout', 0.2))),
        }
    if fusion_method in BOTTLENECK_FUSIONS:
        return {
            'num_latents': int(source.get('num_latents', 4)),
            'num_layers': int(source.get('num_bottleneck_layers', 1)),
            'num_heads': int(source.get('num_heads', 4)),
            'mlp_ratio': float(source.get('mlp_ratio', 4.0)),
            'dropout': float(source.get('bottleneck_dropout', 0.1)),
            'pooling': str(source.get('pooling', 'mean')),
        }
    raise FusionOverRawConfigError(f'Unsupported fusion method for kwargs mapping: {fusion_method}')
