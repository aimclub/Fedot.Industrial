# OKHS Product-Layer Epic

## Purpose

Этот epic фиксирует задачи, которые уже выходят за пределы рефакторинга `example + core + wrapper API`.
Они нужны для выполнения постановки OKHS-DMD, но их не стоит смешивать с PR-1..PR-3

## Epic Goals

1. Benchmark runner для сопоставления OKHS-DMD с базовыми forecasting-подходами.
2. Uncertainty layer для доверительных интервалов и confidence-aware forecast diagnostics.
3. Rolling forecast orchestration с политикой переобучения и контролем drift.
4. Явная orchestration-логика выбора `q` на основе fixed/data-driven/search policy.
5. Единая product-level оболочка для forecasting, regression и classification сценариев.

## Non-Goals

- Переписывание `okhs.py` заново.
- Замена PR-1..PR-3 большими архитектурными изменениями.
- Встраивание AutoML-уровня без стабильного typed API снизу.

## Workstreams

### 1. Benchmark Harness

- Сделать воспроизводимый runner для синтетических и прикладных наборов.
- Добавить baseline-модели: naive, AR-like, classical DMD, существующие forecasting wrappers.
- Зафиксировать единую схему train/validation/test split и набор метрик.

Acceptance:

- один entrypoint;
- воспроизводимые seed-aware запуски;
- сохраняемый markdown/json summary.

### 2. Uncertainty and Diagnostics

- Выделить интерфейс для interval forecast и uncertainty metadata.
- Добавить минимум один практический путь: ensemble/bootstrap или residual-based interval estimation.
- Подготовить diagnostics-слой, который не смешан с `predict`.

Acceptance:

- `predict` остаётся pure inference boundary;
- uncertainty считается отдельным вызовом;
- в отчётах доступны интервалы и quality flags.

### 3. Rolling Forecast and Refit Policy

- Поддержать rolling window и expanding window сценарии.
- Ввести `refit_policy`, `update_frequency`, `drift_threshold`.
- Зафиксировать интерфейс online/offline evaluation.

Acceptance:

- rolling forecast не требует ручной склейки циклов в examples;
- политика переобучения конфигурируется явно;
- есть regression tests на shape и длину горизонта.

### 4. Q Orchestration

- Поднять `q_policy` выше уровня отдельных wrapper-ов.
- Поддержать fixed, data-driven и search-based стратегии.
- Добавить traceable metadata: почему выбран конкретный `q`, на каких данных и с какими ограничениями.

Acceptance:

- выбор `q` не скрыт в `OccupationKernel`;
- decision trace сериализуется;
- search policy можно отключить без изменения модели.

### 5. Multi-Task Product Surface

- Выравнять API для forecasting, regression и classification.
- Уточнить, какие части shared, а какие task-specific.
- Свести AutoML-слой к orchestration поверх typed kernels/forecasters, а не к скрытым эвристикам внутри них.

Acceptance:

- task-specific конфиг не ломает общий OKHS vocabulary;
- product-layer сценарии используют один словарь методов и policy types;
- документация отражает реальные, а не предполагаемые возможности.

## Proposed Sequence

1. Benchmark harness.
2. Q orchestration metadata.
3. Rolling forecast runner.
4. Uncertainty layer.
5. Multi-task product surface and AutoML integration.

## Dependencies

- Merge PR-1: honest example runner.
- Merge PR-2: typed core policies и отделение plotting.
- Merge PR-3: unified `OKHSMethod` и явный `q_policy`.

## Risks

- Смешивание benchmark/orchestration/uncertainty в одном PR быстро размоет границы ответственности.
- Без typed API снизу uncertainty и AutoML будут повторять те же скрытые эвристики
- Без честного benchmark runner нельзя качественно сравнить benefit от выбора `q` и rolling refit.
