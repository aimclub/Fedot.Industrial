# Forecasting Models Reference

## Назначение

Этот документ описывает текущие forecasting-модели Industrial после рефакторинга. Фокус: family, stage decomposition, гиперпараметры, tuning scope и diagnostics.

Главная идея ветки: модель должна быть понятна как stage graph, даже если наружу она экспортируется как удобный shell-класс.

## Model families

| Family | Модели | Назначение |
| --- | --- | --- |
| `lagged_linear` | `lagged_forecaster`, `lagged_ridge_forecaster`, `topo_forecaster` | Надёжные лаговые baseline-модели |
| `low_rank_linear` | `low_rank_lagged_ridge_forecaster`, `ssa_forecaster`, `mssa_forecaster` | Лаговые/страничные модели в низкоранговом пространстве |
| `operator_model` | `havok_forecaster`, `okhs_fdmd_forecaster`, `hybrid_ensemble_forecaster`, `classical_dmd` | Operator/DMD-style модели и гибриды |
| `neural_forecaster` | `patch_tst_model`, `tst_model`, `tcn_model`, `deepar_model`, `nbeats_model` | Neural forecasting heads |

## Lagged linear

### `lagged_forecaster`

Stage decomposition:

- `trajectory_transform`: `window_size`, `stride`
- `forecast_head`: `channel_model`, `alpha`

Что тюнится:

- `window_size`
- `stride`
- `alpha`

Что не тюнится:

- `channel_model` в текущем benchmark path фактически поддержан как `ridge`
- `device`

Diagnostics:

- stage tuning report в `BenchmarkRunRecord.metadata["stage_tuning_report"]`
- comparison в `metadata["stage_tuning_comparison"]`

### `lagged_ridge_forecaster`

Stage decomposition:

- `hankelisation -> ridge_head`

Основные параметры:

- `window_size`
- `stride`
- `alpha`

Тесты:

- `test_stage_tuning.py`
- `test_stage_tuning_runtime.py`
- `test_benchmark_v2.py`

### `topo_forecaster`

Stage decomposition:

- `hankelisation -> topological_features -> ridge_head`

Назначение:

- compatibility replacement для старого `topo_forecaster`, выровненный под стиль `lagged_ridge_forecaster`.

Тесты:

- `test_topo_forecaster.py`

## Low-rank linear

### `low_rank_lagged_ridge_forecaster`

Stage decomposition:

- `hankelisation -> svd_decomposition -> rank_truncation -> ridge_head`

Что тюнится:

- параметры окна;
- rank/decomposition policy;
- `alpha`.

Diagnostics:

- trajectory diagnostics;
- decomposition diagnostics;
- rank diagnostics;
- forecast head diagnostics.

### `ssa_forecaster` и `mssa_forecaster`

Stage decomposition:

- `page_embedding -> decomposition -> rank_truncation -> forecast_head`

Forecast head:

- `head_policy="mlp"` по умолчанию;
- `head_policy="linear"` как fallback.

MLP head tuning:

- `head_activation`
- `head_depth`

Не тюнится через search space:

- `head_epochs`
- `head_learning_rate`
- `device`

Причина: обучение регулируется early stopping, scheduler и runtime defaults.

Тесты:

- `test_mssa_forecaster.py`
- `test_forecasting_runtime.py`
- `test_stage_tuning.py`

## Operator models

### `havok_forecaster`

Stage decomposition:

- `hankelisation -> svd_decomposition -> state/forcing forecast heads`

Forecast head:

- MLP head по умолчанию;
- linear fallback для простых/коротких сценариев.

Что тюнится:

- trajectory/decomposition параметры;
- `head_activation`;
- `head_depth`.

Что не тюнится:

- `epochs`;
- `learning_rate`;
- `batch_size`;
- `device`.

Тесты:

- `test_havok_forecaster.py`
- `test_stage_tuning_runtime.py`

### `okhs_fdmd_forecaster`

Stage decomposition:

- `hankelisation -> decomposition -> rank_truncation -> okhs_fdmd_head`

Состояние:

- модель уже имеет typed spec/run-result layer;
- старый kernel-level backend остаётся compatibility backend;
- stage diagnostics вынесены в общий vocabulary.

### `hybrid_ensemble_forecaster`

Stage decomposition:

- branch A: lagged baseline;
- branch B: low-rank lagged baseline;
- branch C: operator/neural branch;
- ensemble head: weighted aggregation.

Diagnostics:

- branch forecasts;
- branch metrics;
- ensemble weights;
- branch diagnostics.

## Neural forecasters

Модели:

- `patch_tst_model`
- `tst_model`
- `tcn_model`
- `deepar_model`
- `nbeats_model`

Runtime defaults:

- `epochs=150`
- `batch_size=16`
- `learning_rate=0.001`
- `device="cuda"` с fallback через device policy

Что тюнится:

- архитектурные параметры модели;
- patch/context параметры там, где модель реально их использует;
- activation/depth для MLP-like heads.

Что не тюнится:

- `epochs`;
- `batch_size`;
- `learning_rate`;
- `device`.

Причина:

- эти параметры являются runtime policy, а не model search space;
- learning rate регулируется scheduler;
- длительность обучения регулируется early stopping.

Тесты:

- `test_neural_forecast_head_bridge.py`
- `test_forecasting_neural_common.py`
- `test_stage_tuning.py`
- `test_stage_tuning_runtime.py`

## Search space policy

Search space должен описывать только параметры, которые действительно меняют модельную гипотезу.

Не добавлять в search space:

- `device`
- `epochs`
- `batch_size`
- `learning_rate`
- progress/verbosity policy

Если параметр нужен для ускоренного benchmark example, его можно передать в `ModelSpec.params`, но это не означает, что он должен тюниться.

## Diagnostics policy

Каждая новая forecasting-модель должна отдавать:

- `model_family`;
- stage-level diagnostics;
- fit diagnostics;
- prediction diagnostics, если они есть;
- stage tuning hooks, если модель участвует в tuning.

Если модель не умеет выразиться через stages, она считается временным architectural outlier.
