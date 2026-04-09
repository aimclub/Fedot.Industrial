# Forecasting Refactor Plan

Дата актуализации: `2026-04-09`

## Оглавление

1. [Контекст](#контекст)
2. [Цели рефакторинга](#цели-рефакторинга)
3. [Ключевые архитектурные решения](#ключевые-архитектурные-решения)
4. [Skill Routing](#skill-routing)
5. [Статус по этапам](#статус-по-этапам)
6. [PR-стек](#pr-стек)
7. [Стабилизируемые API и типы](#стабилизируемые-api-и-типы)
8. [Тестовая стратегия](#тестовая-стратегия)
9. [Текущий статус](#текущий-статус)
10. [Что дальше](#что-дальше)
11. [Assumptions](#assumptions)

## Контекст

Этот документ фиксирует целевой план рефакторинга forecasting track для `Fedot.Industrial` с учетом уже выполненных
изменений в кодовой базе:

- `projected OKHS / fDMD` уже реализованы как отдельный quality-path;
- `typed policies` и `benchmark/v2` уже существуют;
- часть baseline- и diagnostics-слоя уже перенесена на новый shared backend;
- дальнейшая работа должна не "переоткрывать" уже сделанное, а закрывать оставшиеся архитектурные и качественные gaps.

Документ опирается на:

- [Трек разработки по рефакторингу модуля прогнозирования для Fedot.Industrial](D:\data_old\WORK\Repo\Industiral\IndustrialTS\docs\pr_plans\Трек разработки по рефакторингу модуля прогнозирования для Fedot.Industrial.md)
- текущие изменения в `forecasting`, `OKHS/fDMD` и `benchmark/v2`
- skill-driven подход из `.codex`

## Цели рефакторинга

1. Стабилизировать forecasting substrate и убрать legacy-расхождения.
2. Вынести общее trajectory/embedding ядро для `SSA / mSSA / HAVOK / DMD / OKHS`.
3. Построить единый benchmark/evaluation слой вокруг `benchmark/v2`.
4. Добавить regime-aware baselines и diagnostics.
5. Довести `OKHS/fDMD` до устойчивого качества на реальных forecasting cohort-ах.
6. Подготовить почву для explainable routing и experimental deep OKHS bridge.

## Ключевые архитектурные решения

| Решение                      | Принцип                                                                 |
|------------------------------|-------------------------------------------------------------------------|
| Legacy `ssa_forecaster.py`   | Только `compatibility-wrapper`, не новая самостоятельная ветка развития |
| Canonical benchmark layer    | Только `benchmark/v2`, legacy benchmarking utilities не развиваются     |
| Shared forecasting substrate | Один trajectory/embedding backend для `SSA / mSSA / HAVOK / DMD / OKHS` |
| OKHS quality work            | Отдельный quality track, не смешивать с deep OKHS                       |
| Deep OKHS                    | Только experimental path после стабилизации analytical projected OKHS   |
| PR delivery model            | Последовательный PR-стек с mergeable slices                             |

## Skill Routing

| Зона работы                         | Lead skill                     | Companion / supporting skills  |
|-------------------------------------|--------------------------------|--------------------------------|
| Core refactor и extraction          | `fedot-pure-core-shell`        | `fedot-invariant-tests-review` |
| Tests и regression harness          | `fedot-invariant-tests-review` | `fedot-pure-core-shell`        |
| Typed diagnostics / routing outputs | `fedot-typed-domain-errors`    | `fedot-safe-configs`           |
| Config, manifests, routing policy   | `fedot-safe-configs`           | `fedot-typed-domain-errors`    |
| Adapter / benchmark integration     | `fedot-extension-contract`     | `fedot-pure-core-shell`        |

## Статус по этапам

| PR / Issue | Название                                    | Статус        | Краткий итог                                                                                                                                                    |
|------------|---------------------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PR-1       | Forecasting Core Stabilization              | `done`        | `ssa_forecaster` переведен в compatibility-wrapper, baseline path стабилизирован                                                                                |
| PR-2       | Shared Trajectory Embedding Backend         | `done`        | Добавлен общий trajectory backend для Hankel/Page/stack/decode/truncate                                                                                         |
| PR-3       | mSSA Forecaster v1                          | `done`        | `mSSA` добавлен в forecasting registry и benchmark/v2                                                                                                           |
| PR-4       | Regime Diagnostics Layer                    | `done`        | Добавлены typed regime diagnostics и сериализация в benchmark                                                                                                   |
| PR-5       | HAVOK Forecaster + Event-Aware Artifacts    | `done`        | HAVOK встроен в benchmark/v2, добавлены forcing artifacts и calm/active metrics                                                                                 |
| PR-6       | OKHS/fDMD Anti-Smoothing Refactor           | `in_progress` | Добавлены anti-smoothing diagnostics и bounded residual-bridge correction, но нет полноценной real-M4 acceptance cohort в CI                                    |
| PR-7       | Regime-Aware Routing Policy                 | `in_progress` | Added typed routing decision, deterministic rule-set, fallback path and benchmark metadata integration; orchestration-level adoption is the remaining follow-up |
| PR-8       | Deep OKHS Two-Phase Forecasting Bridge      | `pending`     | Не начат                                                                                                                                                        |
| PR-9       | Decomposition / Selective-Compute Follow-up | `pending`     | Не начат                                                                                                                                                        |

## PR-стек

### PR-1. Forecasting Core Stabilization

Lead skills: `fedot-pure-core-shell` + `fedot-invariant-tests-review`

Изменения:

- перевести `ssa_forecaster.py` в `compatibility-wrapper` над shared backend;
- зафиксировать legacy SSA как `compatibility-only`;
- закрепить единый путь интеграции forecasting baselines через `model_repository.py` и `benchmark/v2`;
- добавить минимальный forecasting test scaffold:
    - unit на shape/policy contracts;
    - integration на honest rolling-origin;
    - regression harness для OKHS smoothing cohort.

Публичный эффект:

- `ssa_forecaster` остается доступным по старому имени;
- docs/examples больше не позиционируют SSA как отдельную независимую реализацию.

Статус:

- `done`

### PR-2. Shared Trajectory Embedding Backend

Lead skills: `fedot-pure-core-shell` + `fedot-typed-domain-errors`

Изменения:

- общий backend в `fedot_ind/core/operation/transformation/data/`:
    - `build_hankel(...)`
    - `build_page(...)`
    - `stack_multivariate(...)`
    - `decode_diagonal_average(...)`
    - `decode_page(...)`
    - `estimate_window(...)`
    - `truncate_rank(...)`
- typed diagnostics для window/stride/rank;
- перевод OKHS trajectory preprocessing на shared backend там, где это не ломает projected path.

Публичный эффект:

- единый trajectory/embedding vocabulary;
- ad hoc Hankel/Page construction в forecasters больше не допускается как source-of-truth.

Статус:

- `done`

### PR-3. mSSA Forecaster v1

Lead skills: `fedot-pure-core-shell` + `fedot-invariant-tests-review`

Изменения:

- новый `mssa_forecaster.py`;
- `Page-matrix` вариант:
    - stacked Page embedding;
    - HSVT denoising;
    - vector forecast head;
    - rolling-origin support;
- интеграция в forecasting registry и `benchmark/v2`.

Публичный эффект:

- новый forecasting model key для `mSSA`;
- diagnostics: `window`, `rank`, `threshold`, `horizon`, `coupling mode`.

Статус:

- `done`

### PR-4. Regime Diagnostics Layer

Lead skills: `fedot-typed-domain-errors` + `fedot-safe-configs`

Изменения:

- новый `regime_diagnostics.py`;
- structured outputs:
    - `ACF decay`
    - `dominant period`
    - `spectral concentration / flatness`
    - `local linearity proxy`
    - `switching score`
- benchmark-serializable typed payload.

Публичный эффект:

- forecasting orchestration и `benchmark/v2` получают structured diagnostics payload;
- документируется база для будущего deterministic routing.

Статус:

- `done`

### PR-5. HAVOK Forecaster + Event-Aware Benchmark Artifacts

Lead skills: `fedot-pure-core-shell` + `fedot-extension-contract` + `fedot-invariant-tests-review`

Изменения:

- интегрирован `havok_forecaster.py`;
- v1 path:
    - Hankel embedding
    - SVD to delay coordinates
    - `(A, B)` regression
    - short-horizon forecast
    - forcing diagnostics
- расширен `benchmark/v2`:
    - forcing timeline
    - `mae_active` / `mae_calm`
    - event overlay artifacts

Публичный эффект:

- новый forecasting model key для `HAVOK`;
- новые event-aware benchmark artifacts.

Статус:

- `done`

### PR-6. OKHS/fDMD Quality Track: Anti-Smoothing Refactor

Lead skills: `fedot-pure-core-shell` + `fedot-invariant-tests-review`

Цель:

- зафиксировать и уменьшить `smoothing collapse` в projected OKHS DMD path.

Что уже сделано:

- structured anti-smoothing diagnostics:
    - `collapse_detected`
    - `train_tail_amplitude`
    - `forecast_amplitude_before/after`
    - `forecast_monotone_ratio_before/after`
    - `train_tail_oscillation_score`
    - `envelope_ratio_before`
- bounded correction path:
    - `anti_smoothing_policy="residual_bridge"`
- проброс diagnostics в benchmark metadata и analytics.

Что еще нужно:

- зафиксировать реальный failure cohort из M4-like рядов;
- встроить acceptance check не только на синтетике и mocked benchmark path, но и на реальном benchmark cohort;
- определить, снижает ли correction collapse-rate и не ухудшает ли MASE/sMAPE на стабильной части набора.

Публичный эффект:

- расширены OKHS diagnostics в `okhs_forecasting.py` и `benchmark/v2`;
- пока acceptance еще не считается закрытым на уровне всего quality track.

Статус:

- `in_progress`

### PR-7. Regime-Aware Routing Policy

Lead skills: `fedot-typed-domain-errors` + `fedot-safe-configs`

Status: `in_progress`

Already implemented:

- typed `RegimeRoutingDecision` and `RegimeRoutingPolicy`;
- deterministic `recommend_forecasting_model(...)`;
- fallback path for weak / insufficient structure;
- benchmark metadata integration via `routing_recommendation`;
- routing determinism tests and benchmark regression checks.

Remaining follow-up:

- connect routing to user-facing orchestration, not only benchmark metadata;
- define adoption path for automatic model recommendation / selection.

Планируемые изменения:

- deterministic recommender поверх diagnostics:
    - periodic -> `SSA / mSSA`
    - switching / bursty -> `HAVOK`
    - locally linear latent -> `DMD / OKHS`
    - weak structure -> `AR / lagged fallback`
- typed routing result object;
- explainable routing output для API и benchmark artifacts.

Статус:

- `pending`

### PR-8. Deep OKHS Two-Phase Forecasting Bridge

Lead skills: `fedot-extension-contract` + `fedot-pure-core-shell`

Планируемые изменения:

- experimental path `method="deep_projected_fdmd"`;
- phase 1:
    - encoder / decoder
    - surrogate `W`
- phase 2:
    - frozen latent trajectories
    - analytical `fDMD`
- без backprop через spectral stage в `v1`.

Статус:

- `pending`

### PR-9. Decomposition and Selective-Compute Follow-up

Lead skills: `fedot-pure-core-shell`

Планируемые изменения:

- randomized SVD;
- leverage sampling;
- reconstruction diagnostics;
- затем reusable ROM utilities:
    - `gappy POD`
    - `DEIM`

Статус:

- `pending`

## Стабилизируемые API и типы

### Shared forecasting substrate

- shared trajectory embedding API для:
    - `SSA`
    - `mSSA`
    - `HAVOK`
    - `DMD`
    - `OKHS`

### Forecasting models

- `ssa_forecaster` as compatibility-only wrapper
- `mssa_forecaster`
- `havok_forecaster`
- `okhs_forecasting`

### Typed outputs

- `RegimeDiagnosticsResult`
- benchmark-v2 run metadata for:
    - regime diagnostics
    - HAVOK forcing analysis
    - OKHS anti-smoothing diagnostics
    - future routing explanation

## Тестовая стратегия

| Этап | Минимальный test scope                                                                         |
|------|------------------------------------------------------------------------------------------------|
| PR-1 | unit + integration scaffold for forecasting, SSA compatibility tests                           |
| PR-2 | embedding shape contracts, decode round-trips, deterministic window/stride/rank tests          |
| PR-3 | multivariate mSSA tests, noisy/missing-value sanity, benchmark registration                    |
| PR-4 | deterministic diagnostics tests and JSON serialization checks                                  |
| PR-5 | synthetic switching integration tests, HAVOK artifact generation, calm-vs-active metric splits |
| PR-6 | fixed regression suite for M4-like smoothing cohort                                            |
| PR-7 | routing determinism and explainability tests                                                   |
| PR-8 | two-phase deep OKHS smoke path, artifact separation tests                                      |
| PR-9 | decomposition utility invariants and selective-compute regression tests                        |

Общее правило:

- каждый новый public forecasting model должен иметь:
    - unit tests;
    - хотя бы один integration benchmark scenario.

## Текущий статус

### Что уже закрыто

Фактически закрыты следующие блоки:

1. `Forecasting Core Stabilization`
2. `Shared Trajectory Embedding Backend`
3. `mSSA Forecaster v1`
4. `Regime Diagnostics Layer`
5. `HAVOK Forecaster + Event-Aware Benchmark Artifacts`

### Что находится в работе

Сейчас активная стадия:

- `PR-6. OKHS/fDMD Quality Track: Anti-Smoothing Refactor`

Прогресс по этому этапу:

- baseline diagnostics уже есть;
- bounded correction path уже реализован;
- benchmark metadata уже прокидывается;
- tests на synthetic / mocked benchmark path уже есть.

Незакрытые части этого этапа:

- реальный `M4-like failure cohort`;
- измеримое снижение `smoothing collapse` на фиксированном наборе серий;
- acceptance на реальных benchmark runs, а не только на synthetic regression.

### Общая оценка прогресса

По крупному forecasting refactor track текущая готовность оценивается примерно как:

- `65-75%` по architectural foundation
- `50-60%` по quality/selection/orchestration vision

Причина разброса:

- базовый substrate уже в хорошем состоянии;
- но routing policy, deep bridge и часть OKHS quality acceptance еще впереди.

## Что дальше

### Ближайший следующий шаг

Следующий рациональный шаг:

1. собрать фиксированный `OKHS smoothing failure cohort` из M4-like серий;
2. встроить его в benchmark regression harness;
3. измерить effect size для `anti_smoothing_policy="residual_bridge"`.

Это позволит:

- честно закрыть `PR-6`;
- отделить реальные улучшения от synthetic-only успехов;
- подготовить понятную базу для `PR-7`.

### После этого

Дальнейший порядок работы:

1. Закрыть `PR-6` на реальном cohort-е.
2. Перейти к `PR-7` и сделать `regime-aware routing policy`.
3. После стабилизации routing и OKHS quality перейти к `PR-8`:
   experimental `deep OKHS two-phase bridge`.
4. И только затем идти в `PR-9`:
   selective-compute / ROM follow-up.

## Assumptions

- `benchmark/v2` — единственный целевой benchmark layer.
- Legacy benchmarking utilities поддерживаются только ради совместимости.
- Legacy SSA не развивается как отдельный standalone baseline.
- `mSSA v1` — это `Page-matrix baseline`, а не полный SSA framework.
- `HAVOK v1` — сначала `univariate-first`, затем расширение на multivariate.
- Deep OKHS остается `experimental`, пока projected analytical OKHS не проходит dedicated anti-smoothing regression
  suite.
- План сознательно не переоткрывает уже выполненные `projected OKHS / benchmark-v2` refactor slices и использует их как
  текущий фундамент.
