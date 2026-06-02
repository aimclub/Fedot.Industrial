---
title: "PDL — постановка Issue по разработке"
aliases:
  - PDL Issue backlog
  - PDL development issues
tags:
  - pdl
  - issue
  - backlog
  - fedot-industrial
  - development
status: approved-for-planning
source:
  - [[pdl-theory-and-ideas|PDL — теория и идеи применения]]
created: 2026-06-01
---

# PDL — постановка Issue по разработке

Связанные заметки:

- [[pdl-theory-and-ideas|PDL — теория и идеи применения]]
- [[pdl-pr-plan|PDL — план PR]]

## Цель разработки

Развить модуль `fedot_ind/core/models/pdl` из прототипа pairwise difference wrapper в расширяемую библиотеку *
*relational target augmentation** для задач классификации, регрессии и последующего forecasting на временных рядах.

Текущая ветка: `refactor_monad_usage`.

Интересующий модуль:

```text
fedot_ind/core/models/pdl/
```

Ключевые направления:

- стабилизировать математические контракты PDL;
- развести pair construction, target construction, anchor selection, sampling и aggregation;
- добавить paper-correct posterior aggregation;
- добавить balanced pair sampling для small-N/high-K classification;
- добавить time-series-aware pair features;
- добавить расширяемые regression target modes;
- добавить memory-aware ограничения;
- закрепить всё тестами и benchmark-протоколом.

---

## Issue 1. Зафиксировать PDL contracts и семантику pair targets

### Проблема

В текущей реализации classification pair target использует внутреннюю семантику:

```text
0 = same
1 = different
```

При этом в теоретическом описании PDL часто используется обратная семантика:

```text
1 = same
0 = different
```

Для регрессии также нужно явно зафиксировать знак:

```text
delta = target_left - target_anchor
```

Сейчас эти соглашения существуют не как формальный контракт, а как implicit behavior внутри функций.

### Почему это важно

Одна ошибка знака или label convention делает PDL-модель внешне рабочей, но математически неверной. Особенно высок риск
при добавлении posterior aggregation, calibration, sample weights, class weights и новых target modes.

### Scope

- Описать target semantics в module-level документации.
- Добавить `pair_target_semantics` в diagnostics.
- Добавить явные имена переменных: `same_label`, `different_label`, `dissimilarity_target`, `delta_left_minus_anchor`.
- Зафиксировать regression sign convention.
- Добавить deterministic tests на tiny arrays.

### Acceptance criteria

- Контракт classification target явно описан.
- Контракт regression target явно описан.
- Diagnostics возвращает `pair_target_semantics`.
- Backward compatibility сохранена.
- Тесты проверяют both target values и interpretation.

### Тесты

- `test_classification_pair_target_semantics_same_is_zero_current_contract`
- `test_regression_pair_target_is_left_minus_anchor`
- `test_predict_same_probability_uses_same_label_column`
- `test_pair_target_semantics_is_reported_in_diagnostics`

### Non-goals

- Не менять публичное поведение classification target в этом Issue.
- Не внедрять posterior aggregation.

---

## Issue 2. Починить или удалить legacy preprocessing/transformer path

### Проблема

`PDCDataTransformer` выглядит недостроенным:

- объявлен `preprocessing_`, но в найденной реализации он не устанавливается перед использованием;
- `transform()` вызывает `self.preprocessing_.transform(X)`;
- для `y` должен использоваться `preprocessing_y_`, но есть риск вызова не того transformer;
- используется `warnings.catch_warnings()`, но `warnings` должен быть явно импортирован.

### Почему это важно

Это runtime-risk в AutoML pipeline. Ошибка проявится не всегда, а только при конкретной конфигурации preprocessing или
target encoding, поэтому её сложно диагностировать после релиза.

### Scope

- Принять одно из двух решений:
    - починить transformer и покрыть тестами;
    - вынести в `legacy.py` и удалить из production path.
- Если transformer остаётся:
    - добавить `self.preprocessing_`;
    - исправить y-transform;
    - добавить `import warnings`;
    - протестировать numeric, categorical и ordinal targets.

### Acceptance criteria

- `PDCDataTransformer.fit().transform()` не падает на базовых сценариях.
- `X` и `y` используют разные preprocessing pipelines.
- Нет обращения к неинициализированным атрибутам.
- Добавлены tests на fit/transform/inverse_transform, если inverse поддерживается.

### Тесты

- `test_pdc_data_transformer_initializes_x_preprocessor`
- `test_pdc_data_transformer_uses_y_preprocessor_for_target`
- `test_pdc_data_transformer_handles_numeric_features`
- `test_pdc_data_transformer_handles_categorical_target`

### Non-goals

- Не проектировать новый feature extraction layer для временных рядов.
- Не добавлять новые PDL target modes.

---

## Issue 3. Ввести strategy interfaces для PDL-модуля

### Проблема

Сейчас в `pairwise_core.py` и `pairwise_model.py` смешаны роли:

- normalization;
- anchor selection;
- pair feature construction;
- pair target construction;
- prediction chunking;
- aggregation;
- sklearn/FEDOT wrapper;
- diagnostics.

Такой дизайн будет быстро разрастаться при добавлении новых target modes, aggregators и time-series adapters.

### Почему это важно

PDL должен развиваться через независимые стратегии, а не через каскады `if/else`. Иначе каждое расширение будет повышать
риск регрессий в уже работающих режимах.

### Scope

Ввести интерфейсы:

```python
class PairFeatureBuilder(Protocol): ...
class PairTargetBuilder(Protocol): ...
class AnchorSelector(Protocol): ...
class PairSampler(Protocol): ...
class PairAggregator(Protocol): ...
class UncertaintyEstimator(Protocol): ...
```

Перенести текущую функциональность в default strategies.

### Acceptance criteria

- Публичные классы `PairwiseDifferenceClassifier` и `PairwiseDifferenceRegressor` сохраняют API.
- Старые параметры `pair_feature_mode`, `pairing_policy`, `max_pairs`, `anchors_per_class` работают.
- Current behavior сохраняется для default config.
- Новая стратегия может быть добавлена без изменения estimator façade.

### Тесты

- Backward compatibility tests для classifier/regressor.
- Unit tests на default strategy resolution.
- Unit tests на invalid strategy names.
- Test, что `random_state` прокидывается в стратегии, где нужен stochastic behavior.

### Non-goals

- Не менять алгоритмический behavior.
- Не добавлять posterior aggregation в этом Issue.

---

## Issue 4. Реализовать classification aggregators

### Проблема

Текущая classification aggregation использует среднюю similarity по anchors класса и нормировку по строке. Это простой
baseline, но не paper-correct posterior aggregation.

### Почему это важно

Именно aggregator превращает pairwise similarity в multiclass probability distribution. От него зависит calibration,
uncertainty и качество на imbalanced/high-cardinality задачах.

### Scope

Добавить aggregators:

- `mean_similarity` — текущий behavior;
- `paper_posterior` — posterior formula из PDL;
- `weighted_posterior` — posterior aggregation с весами anchors;
- optional `symmetric_similarity` на inference.

Добавить uncertainty outputs:

- `total_uncertainty`;
- `aleatoric_uncertainty`;
- `epistemic_uncertainty`.

### Acceptance criteria

- `predict_proba` возвращает матрицу `(n_samples, n_classes)`.
- Строки `predict_proba` суммируются в 1.
- `paper_posterior` проходит closed-form tests на toy examples.
- `weighted_posterior` совпадает с `paper_posterior` при uniform weights.
- Diagnostics содержит `aggregation_policy`.

### Тесты

- `test_mean_similarity_aggregator_matches_existing_behavior`
- `test_paper_posterior_closed_form_binary_case`
- `test_paper_posterior_closed_form_multiclass_case`
- `test_weighted_posterior_equals_posterior_with_uniform_weights`
- `test_uncertainty_shapes_and_non_negative_values`

### Non-goals

- Не менять default aggregator без отдельного migration decision.
- Не добавлять time-series-specific features.

---

## Issue 5. Добавить balanced pair sampler

### Проблема

Для classification with many classes отрицательных пар `different` обычно намного больше, чем положительных `same`. Без
balancing модель может выучить majority relation и деградировать как similarity model.

### Почему это важно

Это центральный риск пользовательской гипотезы small-N/high-K. Если его не закрыть, PDL может выглядеть сильным на
binary задачах и проваливаться на high-cardinality classification.

### Scope

Добавить pair sampling policies:

- `all`;
- `balanced`;
- `stratified_negative`;
- `class_weighted`;
- `anchor_balanced`.

Добавить поддержку `sample_weight`, если base estimator поддерживает `fit(..., sample_weight=...)`.

### Acceptance criteria

- Sampler контролирует pos/neg ratio.
- Diagnostics содержит `n_positive_pairs`, `n_negative_pairs`, `pair_balance_ratio`.
- Для estimator без `sample_weight` код не падает, а корректно fallback-ится.
- В high-K toy dataset sampler создаёт не пустой positive set для классов, где это возможно.

### Тесты

- `test_balanced_sampler_controls_positive_negative_ratio`
- `test_sampler_handles_singleton_classes`
- `test_sampler_reports_pair_balance_diagnostics`
- `test_sample_weight_is_passed_only_when_supported`

### Non-goals

- Не генерировать synthetic time series.
- Не внедрять contrastive learning.

---

## Issue 6. Расширить anchor selection

### Проблема

Текущий выбор anchors слишком простой:

- classification: evenly spaced anchors per class;
- regression: evenly spaced anchors по порядку выборки;
- `random_state` фактически не используется в deterministic selection.

### Почему это важно

PDL inference и aggregation полностью зависят от anchors. Плохие anchors увеличивают noise, latency и variance
prediction.

### Scope

Добавить anchor policies:

- `all`;
- `stratified_even`;
- `stratified_random`;
- `target_quantile`;
- `kmeans`;
- `prototype_per_class`;
- `validation_weighted`.

### Acceptance criteria

- `random_state` реально управляет stochastic policies.
- Anchors reproducible при одинаковом config.
- Для каждого представленного класса выбирается минимум один anchor, если это возможно.
- Diagnostics содержит `anchor_policy`, `n_anchors`, `anchors_per_class_effective`.
- Для regression target quantile policy покрывает диапазон target.

### Тесты

- `test_stratified_random_anchor_selector_is_reproducible`
- `test_anchor_selector_keeps_at_least_one_anchor_per_class`
- `test_target_quantile_anchor_selector_covers_target_range`
- `test_kmeans_anchor_selector_respects_anchor_budget`

### Non-goals

- Не обучать отдельную модель выбора anchors.
- Не добавлять ANN/top-k retrieval.

---

## Issue 7. Time-series pair feature adapters

### Проблема

Текущий path приводит входы размерности больше 2 к flatten-представлению. Для временных рядов это уничтожает явную
структуру времени и каналов.

### Почему это важно

Временные ряды могут быть похожи при phase shift, time warping, локальных motifs и spectral similarity. Flatten + diff
плохо выражает такие отношения.

### Scope

Добавить adapters:

- `FlattenTSAdapter` — baseline;
- `StatisticalTSAdapter`;
- `SpectralTSAdapter`;
- `DistanceTSAdapter`;
- optional `RocketTSAdapter`;
- support для форматов `(n, t)`, `(n, c, t)`, `(n, t, c)` через явный config.

Pair features:

```text
embedding + diff + absdiff + ts_distances
```

### Acceptance criteria

- 2D inputs продолжают работать.
- 3D inputs не теряют channel/time convention silently.
- Config явно задаёт или выводит axis convention.
- Tests покрывают univariate и multivariate series.
- Diagnostics содержит `ts_adapter`, `input_shape_original`, `feature_shape_after_adapter`.

### Тесты

- `test_flatten_ts_adapter_supports_2d_input`
- `test_ts_adapter_supports_n_c_t_input`
- `test_ts_adapter_supports_n_t_c_input_when_configured`
- `test_distance_ts_adapter_returns_expected_feature_shape`
- `test_shifted_sine_synthetic_dataset_smoke`

### Non-goals

- Не добавлять deep neural encoder.
- Не делать full benchmark на UCR/UEA в этом Issue.

---

## Issue 8. Regression target modes

### Проблема

Сейчас regression PDL поддерживает только absolute delta. Для временных рядов и industrial signals часто нужны
relative/log/rank/vector target relations.

### Почему это важно

Разные типы targets требуют разных инвариантностей. Absolute delta не подходит для всех задач: demand, energy,
degradation и forecasting horizons часто требуют scale-aware target transforms.

### Scope

Добавить target builders:

- `delta`;
- `log_delta`;
- `relative_delta`;
- `rank_sign`;
- `quantized_delta`;
- `multioutput_delta`.

Добавить aggregators:

- `mean_delta`;
- `weighted_delta`;
- `median_delta`;
- `trimmed_mean_delta`.

### Acceptance criteria

- `delta` сохраняет текущий behavior.
- `log_delta` имеет корректный inverse transform.
- `relative_delta` стабилен при малых denominators через epsilon.
- `multioutput_delta` поддерживает target shape `(n_samples, horizon)`.
- Diagnostics содержит `pair_target_mode`.

### Тесты

- `test_delta_target_builder_left_minus_anchor`
- `test_log_delta_roundtrip_positive_targets`
- `test_relative_delta_handles_zero_anchor_target_with_epsilon`
- `test_rank_sign_target_builder_outputs_negative_zero_positive`
- `test_multioutput_delta_preserves_horizon_dimension`

### Non-goals

- Не реализовывать probabilistic forecasting.
- Не менять FEDOT task API.

---

## Issue 9. Performance hardening и memory-aware pair generation

### Проблема

`max_pairs` ограничивает число пар, но не учитывает размерность pair features и dtype. Для временных рядов после flatten
или feature extraction размерность может быть большой.

### Почему это важно

Фактическая память:

$$
Memory \approx n_{pairs} \cdot d_{pair} \cdot bytes(dtype)
$$

Для `concat_diff`:

$$
d_{pair} = 3F
$$

Следовательно, одинаковый `max_pairs` может быть безопасным при малом `F` и опасным при большом `F`.

### Scope

Добавить:

- `max_pair_memory_mb`;
- memory estimation before allocation;
- error/warning policy;
- training chunk builder where compatible;
- diagnostics по памяти.

### Acceptance criteria

- Код не пытается создать pair matrix выше memory budget.
- Error message содержит estimated memory и config suggestions.
- Diagnostics содержит `estimated_pair_memory_mb`.
- Prediction chunking сохраняет текущий behavior.
- Для unsafe configuration есть deterministic test.

### Тесты

- `test_pair_memory_estimation_concat_diff`
- `test_pair_generation_refuses_allocation_above_budget`
- `test_pair_generation_allows_allocation_below_budget`
- `test_memory_diagnostics_are_reported`

### Non-goals

- Не внедрять distributed computation.
- Не переписывать sklearn training на streaming для всех estimators.

---

## Issue 10. Benchmark protocol for PDL

### Проблема

Нужно проверить исходную гипотезу: PDL должен быть полезен для маленьких датасетов с большим числом классов и для
временных рядов, где relational features дают больше информации, чем прямое обучение.

### Почему это важно

Без benchmark-протокола развитие PDL будет опираться на интуицию, а не на измеримые свойства.

### Scope

Сделать benchmark suite:

- synthetic classification: small-N/high-K;
- synthetic classification: phase shift / shape classes;
- synthetic regression: amplitude/phase/noise target;
- UCR/UEA subset;
- comparison baseline vs PDL variants.

Метрики:

- Macro-F1;
- balanced accuracy;
- calibration error;
- MAE/RMSE для regression;
- latency;
- memory;
- n_pairs;
- n_anchors.

### Acceptance criteria

- Benchmark script воспроизводим через `random_state`.
- Результаты сохраняются в machine-readable формате.
- Есть README с protocol description.
- Есть сравнение:
    - base model;
    - current PDL;
    - posterior PDL;
    - balanced/weighted PDL;
    - TS-adapter PDL.

### Тесты

- Smoke test benchmark на маленькой synthetic выборке.
- Test result schema.
- Test reproducibility при фиксированном seed.

### Non-goals

- Не объявлять SOTA claims без полноценного протокола.
- Не включать тяжёлые datasets в unit tests.

---

## Общий Definition of Done для всех Issue

- Есть unit tests на новую математику и edge cases.
- Есть diagnostics для новых режимов.
- Есть backward compatibility tests, если меняется internal path.
- Нет silent behavior для NaN/Inf, memory overflow и unsupported strategies.
- Публичный API estimator classes не ломается без отдельного migration decision.
- Документация содержит формулу или явный контракт для каждого нового режима.
