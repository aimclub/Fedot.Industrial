# Onboarding Guide И Руководство По Разработке Для `Fedot.Industrial`

## Оглавление

- [1. Зачем нужен этот документ](#1-зачем-нужен-этот-документ)
- [2. На каких принципах он основан](#2-на-каких-принципах-он-основан)
- [3. Быстрый вход в репозиторий](#3-быстрый-вход-в-репозиторий)
- [4. Карта репозитория](#4-карта-репозитория)
- [5. Как выбирать ведущий architectural skill](#5-как-выбирать-ведущий-architectural-skill)
- [6. Базовый workflow для задач и PR](#6-базовый-workflow-для-задач-и-pr)
- [7. Onboarding для forecasting stack](#7-onboarding-для-forecasting-stack)
- [8. Onboarding для `benchmark/v2`](#8-onboarding-для-benchmarkv2)
- [9. Архитектурные правила Industrial](#9-архитектурные-правила-industrial)
- [10. Антипаттерны, которых стоит избегать](#10-антипаттерны-которых-стоит-избегать)
- [11. Чек-лист перед PR](#11-чек-лист-перед-pr)
- [12. Полезные примеры из текущей кодовой базы](#12-полезные-примеры-из-текущей-кодовой-базы)
- [13. Рекомендуемые первые задачи для нового разработчика](#13-рекомендуемые-первые-задачи-для-нового-разработчика)

## 1. Зачем нужен этот документ

Этот документ выполняет сразу две роли:

- **onboarding guide** для нового разработчика в `Fedot.Industrial`;
- **единое engineering-руководство** для проектирования, рефакторинга, тестирования и code review.

Главная цель: сделать так, чтобы новый участник команды мог быстро понять:

- как устроен репозиторий;
- где искать ключевые модули;
- как здесь принято рефакторить код;
- как устроен текущий forecasting stack;
- как работает `benchmark/v2`;
- какие правила считаются “хорошим тоном” в Industrial.

Если свести идею документа к одной фразе:

**`Fedot.Industrial` развивается как typed, testable, composable codebase, где orchestration остаётся в shell-слое, а
доменная логика выносится в явные и проверяемые contracts.**

## 2. На каких принципах он основан

Документ собран на основе FEDOT skills из локальной папки `.codex`:

- `fedot-skillset-guide`
- `fedot-refactor-router`
- `fedot-pure-core-shell`
- `fedot-invariant-tests-review`
- `fedot-safe-configs`
- `fedot-typed-domain-errors`
- `fedot-extension-contract`

Важно понимать, что это не “пересказ skill docs”, а их адаптация под реальный Industrial-код.

Сейчас в репозитории уже хорошо выражены следующие подходы:

- `pure-core-shell` refactor;
- typed runtime/state objects;
- invariant-oriented tests;
- safe normalization для registry, routing, verbosity, split specs;
- stage-aware forecasting runtime и benchmark layer.

А вот `extension-contract` в Industrial пока выражен слабее, чем в полном FEDOT extension flow. Поэтому его здесь лучше
воспринимать как **направление развития для новых integrations**, а не как уже полностью реализованный слой.

## 3. Быстрый вход в репозиторий

Если вы впервые открыли Industrial, рекомендуемый порядок чтения такой:

1. [industrial_development_guide.md](./industrial_development_guide.md)
2. [radical_forecasting_refactor_plan.md](./radical_forecasting_refactor_plan.md)
3. [forecasting_phase_2_roadmap.md](./forecasting_phase_2_roadmap.md)
4. [forecasting_suite_workflow.md](../benchmark_v2/forecasting_suite_workflow.md)
5. [benchmark_v2_overview.md](../benchmark_v2/benchmark_v2_overview.md)

Если вы идёте именно в forecasting-разработку, затем лучше переходить к:

1. [forecasting_runtime.py](../../fedot_ind/core/models/ts_forecasting/forecasting_runtime.py)
2. [stage_tuning_execution.py](../../fedot_ind/core/models/ts_forecasting/forecast_tuning/stage_tuning_execution.py)
3. [stage_tuning_runtime.py](../../fedot_ind/core/models/ts_forecasting/forecast_tuning/stage_tuning_runtime.py)
4. [regime_routing.py](../../fedot_ind/core/models/ts_forecasting/regime_utils/regime_routing.py)
5. [forecasting.py](../../benchmark/v2/forecasting.py)

## 4. Карта репозитория

Наиболее важные каталоги для ежедневной разработки:

| Путь                        | Назначение                                                         |
|-----------------------------|--------------------------------------------------------------------|
| `fedot_ind/core/models`     | модели, runtime-объекты и domain-specific implementations          |
| `fedot_ind/core/operation`  | data transformations, decomposition, industrial operation layer    |
| `fedot_ind/core/repository` | registries, defaults, repository metadata, aliases                 |
| `fedot_ind/core/tuning`     | search spaces и tuning-related infrastructure                      |
| `benchmark/v2`              | canonical benchmark/evaluation/publication layer                   |
| `tests/unit/core`           | unit tests для core/runtime/model logic                            |
| `tests/unit/models`         | benchmark/runtime integration tests и model-level regression tests |
| `docs/dev`                  | roadmap, refactor plans, engineering documentation                 |
| `docs/benchmark_v2`         | benchmark-specific guides, quickstart и workflow docs              |

### Forecasting-пакеты после реорганизации

Внутри [`ts_forecasting`](../../fedot_ind/core/models/ts_forecasting/__init__.py) модельный слой теперь разделён
тематически:

| Пакет             | Содержимое                                               |
|-------------------|----------------------------------------------------------|
| `lagged_model`    | `lagged_ridge`, `low_rank_lagged`, `ssa`, `mssa`, `topo` |
| `dmd_models`      | `havok`, `okhs_fdmd` и близкие operator-model paths      |
| `ensemble_models` | hybrid/composite forecasting models                      |
| `forecast_tuning` | stage tuning plan, execution и runtime bridge            |
| `neural_models`   | neural forecast heads и bridge/runtime logic             |
| `regime_utils`    | diagnostics и routing                                    |
| `lagged_strategy` | legacy/compatibility слой для старых entrypoints         |

Это разделение важно для onboarding: если задача касается не всего forecasting stack, а только одного среза, лучше сразу
идти в соответствующий тематический пакет.

## 5. Как выбирать ведущий architectural skill

В Industrial почти каждая нетривиальная задача имеет доминирующий architectural concern. Обычно нужен **один lead skill
** и максимум 1-2 companion skills.

| Ведущий skill                  | Когда он главный                                                    | Типичные companions                                         |
|--------------------------------|---------------------------------------------------------------------|-------------------------------------------------------------|
| `fedot-pure-core-shell`        | Большой метод смешивает orchestration и domain-логику               | `fedot-invariant-tests-review`, `fedot-typed-domain-errors` |
| `fedot-invariant-tests-review` | Нужно стабилизировать refactor, сделать review или усилить coverage | `fedot-pure-core-shell`                                     |
| `fedot-safe-configs`           | В центре задачи parsing/defaulting/normalization                    | `fedot-typed-domain-errors`                                 |
| `fedot-typed-domain-errors`    | Состояния и решения скрыты в строках, флагах, `None` или dict blobs | `fedot-pure-core-shell`                                     |
| `fedot-extension-contract`     | Добавляется новая integration/registry/runtime contract             | `fedot-safe-configs`, `fedot-invariant-tests-review`        |
| `fedot-refactor-router`        | Непонятно, какой concern главный                                    | любой из перечисленных                                      |
| `fedot-skillset-guide`         | Нужно собрать стратегию работы из нескольких skills                 | routing зависит от задачи                                   |

Практическое правило:

- если основная проблема в структуре кода, lead почти всегда `fedot-pure-core-shell`;
- если главная боль в raw params и неоднозначной normalization semantics, нужен `fedot-safe-configs`;
- если у задачи много скрытых состояний, добавляется `fedot-typed-domain-errors`;
- если refactor уже почти сделан и нужно “застолбить” поведение, в центр выходит `fedot-invariant-tests-review`.

## 6. Базовый workflow для задач и PR

### Шаг 1. Сначала определить dominant change shape

Перед началом работы полезно ответить на вопрос:

`Что здесь главное: структура кода, типизация домена, конфигурация, extension-contract или тесты?`

Если ответ неочевиден, задача сначала маршрутизируется как design problem, а не как немедленная coding task.

### Шаг 2. Зафиксировать boundary

Нужно заранее понять:

- какой public API или shell остаётся стабильным;
- где refactor должен быть backward-compatible;
- где допустим redesign.

### Шаг 3. Выделить pure logic

Почти всегда хорошие кандидаты на extraction:

- normalization;
- default resolution;
- parameter planning;
- split construction;
- stage routing;
- metric mapping;
- artifact pruning;
- model-family routing;
- diagnostics aggregation.

### Шаг 4. Назвать доменные сущности

Если функция возвращает большой устойчивый `dict`, это сильный сигнал, что пора вводить:

- `dataclass`;
- `Enum`;
- typed result object;
- structured error/result record.

### Шаг 5. Подтянуть shell

Shell должен:

- собрать raw input;
- вызвать helper/runtime;
- записать side effects;
- сохранить metadata и diagnostics;
- не дублировать domain-логику.

### Шаг 6. Добавить mirrored tests

Минимум после каждого содержательного refactor:

- boundary test на публичный путь;
- unit tests на extracted helper/coordinator;
- invariants на детерминизм, idempotence, shape preservation, routing consistency и т.д.

## 7. Onboarding для forecasting stack

### 7.1. Ключевая архитектурная идея

Forecasting stack в Industrial больше не должен мыслиться как набор монолитных forecaster-классов.

Текущая целевая модель такая:

1. `trajectory_transform`
2. `decomposition`
3. `rank_truncation`
4. `forecast_head`

Даже если конкретная модель упакована как shell-класс, хороший инженерный вопрос всегда звучит так:

- как она строит trajectory;
- как формирует latent representation;
- как выбирает rank;
- какой head реально делает forecast.

### 7.2. Ключевые runtime-типы

Базовый forecasting runtime уже типизирован. Самые важные типы лежат
в [forecasting_runtime.py](../../fedot_ind/core/models/ts_forecasting/forecasting_runtime.py):

- `ForecastTensorBatch`
- `TensorDevicePolicy`
- `ForecastingSplitSpec`
- `ForecastingFoldSplit`
- `ForecastingEvaluationResult`
- `TrajectoryTransformResult`
- `DecompositionResult`
- `RankTruncationResult`

Новый разработчик должен начать знакомство именно с ними, потому что они формируют общий vocabulary forecasting runtime.

### 7.3. Как читать forecasting stack по слоям

#### 1. Runtime substrate

Файл:

- [forecasting_runtime.py](../../fedot_ind/core/models/ts_forecasting/forecasting_runtime.py)

Здесь находятся:

- typed contracts;
- split semantics;
- batch conversion;
- low-level forecasting heads и shared runtime helpers.

#### 2. Tuning layer

Файлы:

- [stage_tuning.py](../../fedot_ind/core/models/ts_forecasting/forecast_tuning/stage_tuning.py)
- [stage_tuning_execution.py](../../fedot_ind/core/models/ts_forecasting/forecast_tuning/stage_tuning_execution.py)
- [stage_tuning_runtime.py](../../fedot_ind/core/models/ts_forecasting/forecast_tuning/stage_tuning_runtime.py)

Здесь сосредоточены:

- stage groups;
- search-space slicing;
- sequential tuning orchestration;
- runtime bridge от tuning к реальной оценке модели на ряде.

#### 3. Regime-aware layer

Файлы:

- [regime_diagnostics.py](../../fedot_ind/core/models/ts_forecasting/regime_utils/regime_diagnostics.py)
- [regime_routing.py](../../fedot_ind/core/models/ts_forecasting/regime_utils/regime_routing.py)

Это слой, который отвечает не за обучение модели, а за:

- структурную диагностику ряда;
- typed routing recommendation;
- family-level interpretation benchmark results.

#### 4. Model families

Сейчас forecasting-модели читаются удобнее не “по алфавиту”, а по семействам:

- `lagged_model`
- `dmd_models`
- `ensemble_models`
- `neural_models`

### 7.4. Как ориентироваться в model families

| Семейство         | Типичный смысл                                             | Куда идти первым                                                                                                                                                                                                   |
|-------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `lagged_model`    | lagged/page/low-rank linear baselines и совместимые модели | [lagged_ridge_forecaster.py](../../fedot_ind/core/models/ts_forecasting/lagged_model/lagged_ridge_forecaster.py), [mssa_forecaster.py](../../fedot_ind/core/models/ts_forecasting/lagged_model/mssa_forecaster.py) |
| `dmd_models`      | operator-style and DMD-like forecasting                    | [havok_forecaster.py](../../fedot_ind/core/models/ts_forecasting/dmd_models/havok_forecaster.py), [okhs_fdmd_forecaster.py](../../fedot_ind/core/models/ts_forecasting/dmd_models/okhs_fdmd_forecaster.py)         |
| `ensemble_models` | composite/hybrid models                                    | [hybrid_ensemble_forecaster.py](../../fedot_ind/core/models/ts_forecasting/ensemble_models/hybrid_ensemble_forecaster.py)                                                                                          |
| `neural_models`   | neural forecast heads и bridge-слой                        | [neural_forecast_head.py](../../fedot_ind/core/models/ts_forecasting/neural_models/neural_forecast_head.py)                                                                                                        |
| `lagged_strategy` | legacy compatibility path                                  | использовать только если задача касается backward-compatibility                                                                                                                                                    |

### 7.5. Как дебажить forecasting-модель

Полезный порядок:

1. посмотреть runtime/type contract;
2. проверить stage tuning plan;
3. проверить diagnostics через `get_diagnostics()` или benchmark metadata;
4. только потом идти в конкретный model shell.

Практически это значит:

1. открыть [forecasting_runtime.py](../../fedot_ind/core/models/ts_forecasting/forecasting_runtime.py);
2.
открыть [stage_tuning_runtime.py](../../fedot_ind/core/models/ts_forecasting/forecast_tuning/stage_tuning_runtime.py);
3. открыть нужный shell,
   например [mssa_forecaster.py](../../fedot_ind/core/models/ts_forecasting/lagged_model/mssa_forecaster.py)
   или [havok_forecaster.py](../../fedot_ind/core/models/ts_forecasting/dmd_models/havok_forecaster.py);
4. проверить benchmark adapter в [forecasting.py](../../benchmark/v2/forecasting.py), если баг проявляется через suite.

### 7.6. Чего ждать от “хорошей” forecasting-модели в Industrial

У зрелой модели должны быть:

- явные параметры и defaults;
- diagnostics в serializable виде;
- stage tuning contract;
- benchmark-compatible adapter path;
- понятное family mapping;
- mirrored tests.

Если хотя бы половины этого нет, значит модель ещё живёт скорее как legacy implementation, а не как stage-aware citizen.

## 8. Onboarding для `benchmark/v2`

### 8.1. Роль `benchmark/v2`

`benchmark/v2` — это **canonical benchmark/evaluation/publication layer** для Industrial.

Он нужен не только для запуска моделей, но и для того, чтобы:

- сравнивать model families;
- сохранять diagnostics;
- делать routing evaluation;
- публиковать artifacts;
- прогонять stage tuning и сравнение `baseline vs tuned`.

### 8.2. С чего начать чтение

Лучший порядок:

1. [core.py](../../benchmark/v2/core.py)
2. [forecasting.py](../../benchmark/v2/forecasting.py)
3. [analytics.py](../../benchmark/v2/analytics.py)
4. [presets.py](../../benchmark/v2/presets.py)
5. [api.py](../../benchmark/v2/api.py)
6. [verbosity.py](../../benchmark/v2/verbosity.py)
7. [progress.py](../../benchmark/v2/progress.py)

Для нового разработчика самый важный момент такой: `benchmark/v2` — это не “просто скрипт запуска”, а отдельный
orchestration/framework layer со своими contracts.

### 8.3. Главные сущности `benchmark/v2`

Смотреть в [core.py](../../benchmark/v2/core.py):

- `BenchmarkSuiteConfig`
- `DatasetSpec`
- `ModelSpec`
- `ForecastingSeriesRecord`
- `BenchmarkRunRecord`
- `MetricRecord`
- `PredictionRecord`
- `ForecastingBenchmarkResult`

Именно через эти объекты проходит почти весь benchmark flow.

### 8.4. Как устроен forecasting flow внутри benchmark

Хорошая входная точка:

- [forecasting_suite_workflow.md](../benchmark_v2/forecasting_suite_workflow.md)

В актуальном коде orchestration уже разнесён на coordinator-объекты
внутри [forecasting.py](../../benchmark/v2/forecasting.py):

- `ForecastingSuiteRunner`
- `ForecastingSeriesArtifactsRecorder`
- `ForecastingPostFitTuningCoordinator`

Это важный паттерн Industrial:

- большой pipeline не должен жить как одна гигантская функция;
- orchestration лучше выражать coordinator-shell классами;
- heavy logic нужно делить по этапам выполнения.

### 8.5. Что делает benchmark runner на практике

На высоком уровне runner:

1. валидирует config;
2. итерируется по datasets;
3. итерируется по models;
4. итерируется по series;
5. собирает regime diagnostics;
6. строит routing recommendation;
7. выполняет forecast;
8. считает baseline metrics;
9. запускает post-fit stage tuning;
10. считает tuned metrics;
11. сохраняет records и publication artifacts.

### 8.6. Progress и verbosity policies

В `benchmark/v2` уже есть централизованные policy layers:

- [progress_policy.py](../../fedot_ind/core/models/ts_forecasting/progress_policy.py)
- [verbosity.py](../../benchmark/v2/verbosity.py)

Это значит:

- новые логирующие или pruning-решения не стоит добавлять ad hoc внутри runner-а;
- лучше встраивать их в policy layer и пробрасывать дальше как явный contract.

### 8.7. Где обычно появляются ошибки при работе с benchmark

Типичные зоны:

- `ModelSpec.params` и adapter construction;
- несогласованность между registry/defaults/search space;
- длина forecast horizon;
- stage tuning runtime config;
- family/adapter alias normalization;
- serialization of diagnostics and metadata.

Практический совет:

если ошибка появляется “во время suite”, сначала проверьте:

1. как строится adapter в [forecasting.py](../../benchmark/v2/forecasting.py);
2. как нормализуется model name в [forecasting_registry.py](../../fedot_ind/core/repository/forecasting_registry.py);
3. есть ли у модели корректные defaults
   в [default_operation_params.json](../../fedot_ind/core/repository/data/default_operation_params.json);
4. соответствует ли search space реальному constructor contract.

## 9. Архитектурные правила Industrial

### 9.1. Thin shell, rich pure core

Что должно оставаться в shell:

- lifecycle;
- orchestration;
- I/O;
- runtime wiring;
- logging/progress;
- registry integration;
- serialization and artifact recording.

Что должно уходить в pure core:

- decisions;
- normalization;
- validation;
- routing;
- parameter resolution;
- split planning;
- metric mapping;
- stage decomposition;
- deterministic transformations.

Хорошие текущие примеры:

- [ForecastingSuiteRunner](../../benchmark/v2/forecasting.py)
- [SequentialStageTuningRunner](../../fedot_ind/core/models/ts_forecasting/forecast_tuning/stage_tuning_execution.py)
- [ForecastingSeriesEvaluator](../../fedot_ind/core/models/ts_forecasting/forecast_tuning/stage_tuning_runtime.py)

### 9.2. Domain states должны быть first-class objects

Хорошие примеры:

- [ForecastTensorBatch](../../fedot_ind/core/models/ts_forecasting/forecasting_runtime.py)
- [ForecastingSplitSpec](../../fedot_ind/core/models/ts_forecasting/forecasting_runtime.py)
- [ForecastingEvaluationResult](../../fedot_ind/core/models/ts_forecasting/forecasting_runtime.py)
- [RegimeRoutingDecision](../../fedot_ind/core/models/ts_forecasting/regime_utils/regime_routing.py)
- [ForecastingVerbosityPolicy](../../benchmark/v2/verbosity.py)

### 9.3. Config flow должен быть явным

Любая config-like сущность должна проходить путь:

`raw -> parsed -> validated -> normalized -> defaulted -> typed`

Хорошие примеры:

- [canonical_forecasting_model_name](../../fedot_ind/core/repository/forecasting_registry.py)
- [resolve_forecasting_verbosity_policy](../../benchmark/v2/verbosity.py)
- split normalization в [forecasting_runtime.py](../../fedot_ind/core/models/ts_forecasting/forecasting_runtime.py)

### 9.4. Один canonical route на одно domain action

Новая модель или integration path не должна строиться тремя параллельными маршрутами.

Желаемое состояние:

- один registry vocabulary;
- один alias normalization path;
- один default/search-space contract;
- один benchmark adapter path;
- один preferred runtime entrypoint.

### 9.5. Benchmark и runtime должны говорить на одном языке

Если runtime оперирует stages, families и diagnostics, benchmark должен видеть их в тех же терминах.

Иначе возникает drift между:

- тем, как модель устроена;
- тем, как она тюнится;
- тем, как она публикуется;
- тем, как она объясняется пользователю или разработчику.

## 10. Антипаттерны, которых стоит избегать

### 10.1. Большой метод, который делает всё сразу

Сигналы:

- итерирует dataset/model/series;
- обучает модель;
- считает метрики;
- пишет artifacts;
- логирует прогресс;
- нормализует конфиг;
- ещё и принимает routing decisions.

Это почти всегда кандидат на coordinator refactor.

### 10.2. Stringly typed branching

Плохо:

- хаотичные строковые режимы;
- alias handling в нескольких местах;
- routing, построенный на наборе частично совпадающих строк.

### 10.3. Внутренний `dict[str, Any]` как основной протокол

`dict` на boundary допустим. Внутри нескольких слоёв лучше переходить к typed records.

### 10.4. Hidden error semantics

Плохо, когда:

- `None` означает и отсутствие, и ошибку, и fallback;
- expected failures превращаются в generic exception;
- fallback случается молча и не виден в metadata.

### 10.5. Тесты только через толстый integration path

После refactor должны появляться не только integration tests, но и unit tests на extracted collaborators.

### 10.6. Несколько путей интеграции одной модели

Если constructor contract, defaults, search space и benchmark adapter живут по разным правилам, это почти всегда
приводит к drift и трудноуловимым runtime bugs.

## 11. Чек-лист перед PR

### Архитектура

- public boundary сохранён или изменение объявлено явно;
- orchestration и pure logic разделены;
- новые domain states названы явно;
- нет дублирующей normalization logic в нескольких местах;
- у модели или integration path есть один canonical route.

### Конфигурация

- raw payload нормализуется один раз;
- defaults заданы явно;
- unsafe parsing не используется;
- aliases и registry names сведены к canonical form.

### Тесты

- есть хотя бы один boundary test;
- pure helpers покрыты unit tests;
- есть invariants на важное поведение;
- regression-case, ради которого делался refactor, закреплён тестом.

### Benchmark/diagnostics

- metadata и artifacts не скрывают важные decision paths;
- stage/routing diagnostics сериализуются;
- progress/verbosity logic централизована.

## 12. Полезные примеры из текущей кодовой базы

### 12.1. Typed runtime substrate

Файл:

- [forecasting_runtime.py](../../fedot_ind/core/models/ts_forecasting/forecasting_runtime.py)

Почему полезно читать:

- показывает, как в Industrial моделируются runtime state и split contracts;
- задаёт vocabulary для forecasting runtime.

### 12.2. Разделение orchestration на coordinators

Файлы:

- [stage_tuning_execution.py](../../fedot_ind/core/models/ts_forecasting/forecast_tuning/stage_tuning_execution.py)
- [stage_tuning_runtime.py](../../fedot_ind/core/models/ts_forecasting/forecast_tuning/stage_tuning_runtime.py)
- [forecasting.py](../../benchmark/v2/forecasting.py)

Почему полезно читать:

- это живой пример `pure-core-shell` refactor на больших функциях.

### 12.3. Typed routing и family mapping

Файл:

- [regime_routing.py](../../fedot_ind/core/models/ts_forecasting/regime_utils/regime_routing.py)

Почему полезно читать:

- показывает, как domain decision можно оформить как стабильный typed contract.

### 12.4. Safe policy normalization

Файлы:

- [forecasting_registry.py](../../fedot_ind/core/repository/forecasting_registry.py)
- [verbosity.py](../../benchmark/v2/verbosity.py)

Почему полезно читать:

- это хорошие примеры centralized normalization and policy resolution.

### 12.5. Forecasting benchmark orchestration

Файл:

- [forecasting.py](../../benchmark/v2/forecasting.py)

Почему полезно читать:

- это один из главных примеров того, как в Industrial должны выглядеть coordinator-shell классы.

## 13. Рекомендуемые первые задачи для нового разработчика

### Вариант 1. Безопасный onboarding

- поправить docs/links/examples;
- усилить unit tests для уже выделенных helpers;
- добавить coverage на boundary cases и normalization rules.

### Вариант 2. Forecasting-oriented onboarding

- взять одну модель из `lagged_model`;
- проверить её constructor/default/search space/benchmark adapter consistency;
- добавить missing diagnostics или missing tests.

### Вариант 3. Benchmark-oriented onboarding

- пройти цепочку `ModelSpec -> adapter -> run_record -> analytics artifact`;
- проверить, что metadata и publication pack согласованы;
- устранить один drift между runtime и benchmark vocabulary.

### Вариант 4. Refactor-oriented onboarding

- найти крупную orchestration function;
- сначала выделить typed helper или coordinator;
- затем добавить mirrored tests;
- только после этого трогать outer shell.

---

## Краткая итоговая формула

Если запомнить только одно правило, пусть это будет оно:

**В `Fedot.Industrial` мы стараемся превращать неявную procedural-логику в typed, testable и composable contracts,
сохраняя public shells тонкими, а domain-логику — явной, локализованной и проверяемой.**

## 14. Release-ready forecasting documentation

Перед публикацией forecasting-ветки и при дальнейшем onboarding стоит читать общий guide вместе с отдельными forecasting-документами:

- [forecasting_runtime_api_reference.md](./forecasting_runtime_api_reference.md) — текущие runtime contracts, stage tuning, policies, persistence/resume и visualizer API.
- [forecasting_models_reference.md](./forecasting_models_reference.md) — model families, stage decomposition, tuning/non-tuning параметры и diagnostics.
- [benchmark_v2_forecasting_guide.md](./benchmark_v2_forecasting_guide.md) — `ForecastingSuiteRunner`, item-level persistence, resume mode, post-fit tuning comparison и публикационные artifacts.
- [forecasting_branch_development_history.md](./forecasting_branch_development_history.md) — история решений ветки и сознательные ограничения перед merge.
- [forecasting_merge_artifact_policy.md](./forecasting_merge_artifact_policy.md) — что нельзя тащить в PR: benchmark outputs, `progress/items`, datasets, checkpoints, archives и визуализации.
