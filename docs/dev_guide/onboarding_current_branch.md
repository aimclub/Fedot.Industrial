# Onboarding по текущей ветке Fedot.Industrial

Цель документа:

- быстро ввести нового разработчика в реальную архитектуру текущей ветки;
- показать, какие модули являются опорными, а какие относятся к легаси;
- зафиксировать текущие инженерные принципы разработки;
- объяснить, почему forecasting сейчас является эталонным рефакторинговым контуром;
- встроить беклог по `anomaly_detection` в формате issue/PR.

## 1. Что изменилось по сравнению с исходным onboarding-документом

Старый документ полезен как историческая карта репозитория, но он уже не отражает архитектурный центр тяжести текущей
ветки.

Что в нем полезно:

- репозиторий исторически рос вокруг `api`, `core`, `repository`, `examples`, `tools`;
- в проекте много task-specific логики и много legacy-интеграции с FEDOT;
- понимание того, что `models`, `operation` и `repository` остаются самыми важными зонами кода.

Что устарело:

- репозиторий больше нельзя понимать только через старые модули вроде `core/architecture/pipelines` или через
  статический список моделей;
- центр новых изменений сместился в сторону runtime-стратегий, typed contracts, thin shells и `benchmark/v2`;
- forecasting уже не просто набор forecaster-классов, а архитектурный шаблон для дальнейших рефакторингов;
- anomaly detection больше нельзя развивать как побочный случай классификации;
- новый код должен ориентироваться не только на старую структуру папок, но и на принципы
  из `docs/dev_guide/fp_informed_style.md`,
  `docs/dev/*` и локального knowledge-base в `.codex`.

## 2. Что читать в первые 1-2 дня

Рекомендуемый порядок входа:

1. `README.rst`
2. `docs/dev_guide/fp_informed_style.md`
3. `docs/dev/radical_forecasting_refactor_plan.md`
4. `docs/dev/forecasting_phase_3_roadmap.md`
5. `docs/dev/anomaly_detection_phase_1_roadmap.md`
6. `docs/benchmark_v2/benchmark_v2_overview.md`
7. `docs/benchmark_v2/quickstart.md`
8. `.codex/skills-src/fedot-pure-core-shell/SKILL.md`
9. `.codex/skills-src/fedot-invariant-tests-review/SKILL.md`

Если задача затрагивает рефакторинг или проектирование нового runtime, сначала смотри `docs/dev/*`, потом код.

## 3. Ментальная модель репозитория в текущей ветке

### 3.1. Публичный shell

Основной пользовательский вход по-прежнему находится в:

- `fedot_ind/api/main.py`
- `fedot_ind/api/utils/*`

`FedotIndustrial` остается главным facade-классом. Это публичный shell, который:

- собирает конфиги;
- инициализирует repository и solver;
- нормализует входные данные;
- координирует fit/predict/evaluate.

Это не лучшее место для бизнес-логики новых алгоритмов. Новая логика должна опускаться ниже, в runtime-слой или в pure
helpers.

### 3.2. Runtime-стратегии и task routing

Ключевой слой интеграции между FEDOT и Industrial сейчас находится в:

- `fedot_ind/core/operation/interfaces/industrial_model_strategy.py`
- `fedot_ind/core/operation/interfaces/industrial_preprocessing_strategy.py`
- `fedot_ind/core/operation/interfaces/forecasting_runtime_strategy.py`

Это одна из самых важных архитектурных абстракций.

Почему:

- здесь до сих пор живет значительная часть легаси;
- здесь видно, какие task-specific runtime path уже выделены;
- forecasting уже получил свою runtime стратегию;
- anomaly detection пока еще не получил.

Практический вывод:

- если задача связана с новыми task-specific сценариями запуска - почти всегда нужно смотреть именно сюда;
- если задача касается forecasting, используем `forecasting_runtime_strategy.py` как референс;
- если задача касается anomaly detection, нужно уходить от текущей зависимости на classification-style dispatch.

### 3.3. Repository как реестр возможностей и обертка на легаси

Смотри в:

- `fedot_ind/core/repository/model_repository.py`
- `fedot_ind/core/repository/data/*`
- `fedot_ind/core/repository/*registry*.py`

Роль этого слоя:

- "регистрация" моделей и preprocessing-операций;
- отоброжение имени модели -> в его практическую реализацию;
- дефолтные параметры;
- compatibility wiring для FEDOT.

Важно понимать:

- это один из самых "типизированных" (с помощью строк) слоев репозитория;
- здесь высокая концентрация легаси-решений;
- именно сюда в итоге должен приходить любой новый runtime path через явные canonical names и alias policy;
- forecasting уже частично прошел этот путь;
- anomaly detection находится в процессе перехода.

### 3.4. `core/models` и `core/operation`

Условно:

- `core/models/*` содержит model implementations;
- `core/operation/*` содержит transformation/decomposition/runtime-friendly operations.

Текущий архитектурный тренд:

- тяжелую вычислительную и decision-логику выносить в deterministic helpers и typed runtime objects;
- сами model classes и strategy classes превращать в thin shells;
- новые сущности проектировать не через mode flags, а через explicit dataclasses/enums/contracts.

### 3.5. `benchmark/v2` как каноничный "слой валидации"

Это один из важнейших модулей текущей ветки:

- `benchmark/v2/core.py`
- `benchmark/v2/api.py`
- `benchmark/v2/presets.py`
- `benchmark/v2/manifests.py`
- `benchmark/v2/registry.py`
- `benchmark/v2/forecasting.py`
- `benchmark/v2/classification.py`
- `benchmark/v2/regression.py`

Сегодня `benchmark/v2` уже является валидацией для таких стратегий как:

- forecasting;
- ts classification;
- ts regression.

Именно сюда должен прийти `anomaly_detection`.

Если новый код нельзя нормально прогнать и сравнить через `benchmark/v2`, то он почти наверняка
еще не встроен в актуальную архитектуру ветки.

### 3.6. `docs`, `examples`, `tests`

Это не просто “приложение”, а как часть архитектурного контура:

- `docs/dev/*` — roadmap и source-of-truth для крупных рефакторингов;
- `docs/benchmark_v2/*` — документация по актуальному benchmark layer;
- `examples/benchmark_v2/*` — примеры prefered execution path;
- `tests/unit/*` — зеркальный unit-level слой;
- `tests/integration/*` — integration/smoke coverage.

## 4. Где в репозитории живет легаси

Ниже список зон, которые новый разработчик должен сразу воспринимать как “архитектурно чувствительные”.

### 4.1. `fedot_ind/api/main.py`

Проблемы:

- много orchestration logic в одном facade;
- конфигурирование, routing, preprocessing и solver wiring живут близко друг к другу;
- часть flow исторически рассчитана на generic FEDOT semantics, а не на task-specific runtime.

Как относиться:

- публичный API сохраняем;
- новые архитектурные решения стараемся реализовывать ниже этого слоя;
- сюда поднимаем только стабильные shell-level entrypoints.

### 4.2. `industrial_model_strategy.py`

Это текущий legacy hub для task/model dispatch.

Главная проблема для anomaly detection:

- `IndustrialAnomalyDetectionStrategy` сейчас наследуется от `IndustrialSkLearnClassificationStrategy`;
- это означает classification-by-accident архитектуру;
- detection runtime semantics, scoring, calibration и event aggregation здесь не являются first-class.

### 4.3. `industrial_preprocessing_strategy.py`

Модуль очень важный, но тяжелый:

- много совместимости;
- много ветвления;
- много implicit assumptions про shape/format/output mode.

Это типичное место для рефакторинга по схеме pure core + thin shell.

### 4.4. `model_repository.py`

Проблемы:

- giant registry;
- stringly-typed model naming;
- смешение historical names, canonical names и compatibility aliases;
- сложно понять, какие модели являются первичными, а какие legacy.

Практическое правило:

- новые модели не добавляем как еще один случай “просто словаря” без явной canonical naming policy;
- любые новые family-level изменения должны иметь companion registry/metadata layer.

### 4.5. `core/models/detection/*`

Это один из главных легаси-контуров в текущей ветке.

Прямые сигналы легаси:

- `anomaly_detector.py` опирается на mode flags (`lagged`, `full`, `batch`);
- detection task завернут в `TaskTypesEnum.classification`;
- thresholding и score-to-label логика частично скрыты внутри model shell;
- старые detector families живут как heterogeneous set без единого runtime contract;
- benchmark/v2 detection пока отсутствует.

### 4.6. `core/architecture/pipelines/*`

Исторически важный слой, но не architectural center текущей ветки.

Используй его как compatibility/history layer, а не как место, откуда нужно копировать новую архитектуру.

### 4.7. `ensemble/*` и часть `examples/*`

Это полезные артефакты, но:

- они не определяют target architecture;
- часть примеров устарела;
- папка `examples/outdated_examples` — явно не source-of-truth.

## 5. Текущие инженерные принципы ветки

Этот репозиторий сейчас движется не к “еще большему числу умных классов”, а к более явной архитектуре.

### 5.1. Pure core + thin effect shell

Новая логика должна стремиться к разделению:

- pure core:
    - validate
    - normalize
    - transform
    - route
    - plan
    - aggregate
    - score
- effectful shell:
    - file IO
    - network IO
    - runtime/backend calls
    - logging
    - solver wiring

### 5.2. Typed contracts вместо mode flags

Предпочитаем:

- `dataclass`
- `Enum`
- explicit result objects
- stable runtime vocabulary

Избегаем:

- raw `dict[str, Any]` как основной transport layer;
- stringly-typed `mode/status/type`;
- implicit state transitions;
- “магических” branch-политик внутри больших методов.

### 5.3. Benchmark-first мышление

Если в forecasting архитектурный центр уже смещен в `benchmark/v2`, то новые большие task stacks тоже должны двигаться
туда.

### 5.4. Тесты как архитектурный контракт

После рефакторинга ожидается не только smoke coverage, но и:

- unit tests для pure logic;
- facade/boundary tests для публичного shell;
- invariant-style tests:
    - determinism
    - idempotence
    - monotonicity
    - round-trip
    - conservation/count preservation
    - no future leakage

## 6. Forecasting как референс для новых разработчиков

Если нужно понять, как в этой ветке должен выглядеть “правильный” task-specific refactor, смотри на forecasting.

Почему forecasting сейчас является референсом:

- у него уже есть выделенный runtime strategy;
- есть roadmap в `docs/dev/*`;
- stage vocabulary уже сделана явной;
- benchmark/v2 уже умеет запускать и публиковать результаты;
- часть legacy-path уже редиректится в dedicated runtime;
- идет движение к graph/primitive-first architecture.

Практический вывод:

- новый detection runtime лучше проектировать по аналогии с forecasting refactor, а не по аналогии со
  старым `AnomalyDetector`;
- если есть выбор между “дописать еще один частный case в legacy strategy” и “сделать typed runtime + benchmark
  integration”, выбираем второе.

## 7. Текущее состояние `anomaly_detection`

### 7.1. Что является legacy source-of-truth

Сейчас legacy contour detection сосредоточен в:

- `fedot_ind/core/models/detection/anomaly_detector.py`
- `fedot_ind/core/models/detection/anomaly/algorithms/*`
- `fedot_ind/core/models/detection/custom/stat_detector.py`
- `fedot_ind/core/models/detection/probalistic/*`
- `fedot_ind/core/models/detection/subspaces/*`
- `fedot_ind/core/operation/interfaces/industrial_model_strategy.py`
- `fedot_ind/core/repository/model_repository.py`

### 7.2. Что уже намечено в новом контуре

В текущем рабочем контуре уже появились заготовки под новый detection stack:

- `fedot_ind/core/models/detection/runtime.py`
- `fedot_ind/core/models/detection/stage_tuning.py`
- `fedot_ind/core/models/detection/modern_detectors.py`
- `fedot_ind/core/repository/detection_registry.py`
- `docs/dev/anomaly_detection_phase_1_roadmap.md`

Целевая идея:

- detection-first runtime;
- typed contracts;
- explicit stage vocabulary;
- canonical model families;
- явная calibration layer;
- event aggregation как first-class stage;
- benchmark/v2 detection suite;
- risk-ready export для следующего failure-modeling слоя.

### 7.3. Что еще не завершено

На момент этой версии документа не завершены:

- полноценный detection runtime strategy;
- repository wiring и default params для новых detector families;
- `benchmark/v2` suite для anomaly detection;
- public preset + MPSI local preset;
- migration path для legacy detector names;
- системный набор unit/integration tests;
- обновление examples и migration docs.

## 8. Как безопасно брать задачи в этой ветке

### 8.1. Сначала найди shell boundary

Перед изменениями ответь себе:

- это facade?
- это runtime strategy?
- это registry/config boundary?
- это benchmark runner?
- это pure helper candidate?

### 8.2. Затем выдели логику, которую можно сделать pure

Особенно ищи:

- normalization;
- validation;
- split logic;
- calibration rules;
- routing rules;
- aggregation logic;
- alias/canonical mapping;
- artifact planning.

### 8.3. Потом проверь benchmark path

Если задача затрагивает новую task family, новый runtime или новый family-level evaluation path, скорее всего нужно:

- обновить `benchmark/v2`;
- добавить preset или manifest path;
- зафиксировать артефакты и метрики.

### 8.4. Обнови docs/examples/tests вместе с кодом

Для крупных изменений ожидается:

- код;
- roadmap/doc update;
- tests;
- пример запуска или manifest/preset.

## 9. Backlog по `anomaly_detection`: issue-level декомпозиция

Ниже backlog в формате, пригодном для GitHub Issues.

### Issue 1 - Detection runtime contracts + stage vocabulary

Почему:

сейчас anomaly detection опирается на legacy base class с mode flags (`lagged` / `full` / `batch`) и implicit state. Это
мешает сделать detection first-class задачей и не позволяет стабильно строить calibration, events, transfer и benchmark
artifacts.

Задачи:

- зафиксировать canonical runtime objects для detection:
    - `DetectionWindowBatch`
    - `DetectionSplitSpec`
    - `RegimeSegment`
    - `AnomalyScoreSeries`
    - `DetectionEvent`
    - `TransferAlignmentReport`
    - `DetectionSeriesEvaluation`
    - `DetectionStageTuningPlan`
    - `RiskFeatureFrame`
- ввести стабильную stage vocabulary:
    - `data_quality`
    - `regime_segmentation`
    - `representation`
    - `anomaly_scoring`
    - `calibration`
    - `event_aggregation`
    - `transfer_alignment`
    - `interpretation`
- вынести split/windowing/calibration/event aggregation в reusable helpers
- зафиксировать no-future-leakage semantics для split и calibration

DoD:

- typed runtime contracts существуют как source-of-truth для нового detection stack
- mode flags перестают быть единственным механизмом моделирования detection flow
- stage vocabulary описана и используется в коде
- есть unit tests на windowing, calibration, aggregation и split invariants

Labels: architecture, detection, runtime, blocker

### Issue 2 - Canonical detection registry + alias policy

Почему:

сейчас detection naming живет в giant legacy registry и не разделяет canonical models, aliases и legacy families. Это
усложняет migration path и мешает benchmark/v2 агрегировать результаты по model family.

Задачи:

- ввести canonical detection naming policy
- явно разделить:
    - first-class detector families
    - legacy detector families
    - compatibility aliases
- оформить family mapping для benchmark/reporting/tuning
- подключить новый detection registry к repository metadata

DoD:

- для detection есть canonical names и aliases policy
- legacy detector names больше не являются source-of-truth
- family-level grouping можно использовать в benchmark/v2 и tuning
- migration path документирован

Labels: architecture, repository, detection, parallelizable

### Issue 3 - Detection runtime strategy вместо classification-by-accident

Почему:

сейчас `IndustrialAnomalyDetectionStrategy` наследуется от classification strategy. Это архитектурно неверно: у
detection другие semantics вывода, калибровки, scoring и event aggregation.

Задачи:

- выделить dedicated detection runtime strategy
- перестать использовать classification task semantics как внутренний detection substrate
- определить boundary adapter между `InputData` и новым detection runtime
- минимизировать зависимость detection path от `MultiDimPreprocessingStrategy`

DoD:

- detection runtime strategy существует как отдельный путь
- classification inheritance больше не является архитектурным центром detection task
- shell boundary остается совместимой с текущими public entrypoints
- есть boundary/smoke tests через public strategy path

Labels: architecture, detection, runtime, blocker

### Issue 4 - First-class detector families + explicit calibration layer

Почему:

сейчас detection families гетерогенны, thresholding частично скрыт в model shells, а новые detectors не оформлены как
единый runtime-first стек.

Задачи:

- сделать canonical wave-1 families:
    - `feature_iforest_detector`
    - `feature_oneclass_detector`
    - `conv_autoencoder_detector`
    - `tcn_autoencoder_detector`
- вынести thresholding в explicit calibration layer
- поддержать calibration strategies:
    - `mad`
    - `quantile`
    - `regime_conditional`
    - `domain_calibrated`
- перевести legacy families в non-canonical namespace:
    - `legacy_lstm_autoencoder_detector`
    - `legacy_arima_detector`
    - `legacy_sst_detector`
    - `legacy_kalman_detector`
    - `legacy_functional_pca_detector`

DoD:

- first-class detectors работают через единый runtime contract
- calibration strategy задается явно и попадает в diagnostics
- old hidden threshold heuristics больше не являются основным поведением
- есть tests на monotonic threshold behavior и score-to-event conversion

Labels: detection, models, calibration, blocker

### Issue 5 - Benchmark V2 detection suite + public local benchmark

Почему:

сейчас `benchmark/v2` покрывает forecasting, classification и regression, но не anomaly detection. Пока detection не
встроен сюда, рефакторинг остается частично “невидимым” и трудно сравнимым.

Задачи:

- добавить `TaskType.ANOMALY_DETECTION`
- реализовать detection suite runner в `benchmark/v2`
- добавить dataset adapters под локальные public detection CSV
- определить detection metrics:
    - point-level precision/recall/f1
    - event-level precision/recall/f1
    - false positives per series
    - delay-aware metric при наличии разметки
- подключить manifests, presets, registry и publication artifacts

DoD:

- `benchmark/v2` умеет запускать anomaly detection suite
- есть хотя бы один public local preset
- detection runs попадают в registry и manifest flow
- benchmark сохраняет structured diagnostics и summary artifacts

Labels: benchmark-v2, detection, evaluation, blocker

### Issue 6 - MPSI local preset + data policy + manifests

Почему:

industrial use case по МПСИ является одним из главных драйверов рефакторинга. Без отдельного локального preset и явной
data policy этот контур останется документом, а не инженерным path.

Задачи:

- зафиксировать MPSI data policy:
    - analog telemetry и vibration как first-class signals
    - discrete PIMS tags как context/sync/regime signals
    - raw archives не коммитим
    - храним manifests, mappings, lightweight fixtures, derived metadata
- добавить lightweight local preset для MPSI
- оформить manifest-driven запуск
- определить базовый набор acceptance metrics для industrial contour

DoD:

- есть MPSI local preset
- есть manifest example
- data policy описана в docs
- raw industrial archives не нужны для smoke path

Labels: detection, industrial, data, benchmark-v2

### Issue 7 - Transfer alignment + risk scaffold

Почему:

refactor detection нужен не только ради anomaly scores, но и как база для следующего слоя failure/risk forecasting. Это
требует явного scaffold, а не ad-hoc feature export из моделей.

Задачи:

- добавить baseline transfer path:
    - `domain_invariant_scaling`
    - `CORAL/feature_alignment`
    - `few_shot_calibration`
- добавить `RiskFeatureFrame` export
- связать events, regimes, domain/node context и maintenance-ready joins
- сделать diagnostics и export пригодными для следующей волны failure modeling

DoD:

- transfer alignment report формируется явно
- risk-ready feature export существует как typed artifact
- detection runtime может работать на unseen domain/node сценариях
- есть tests на basic alignment semantics и export shape

Labels: detection, transfer-learning, risk, parallelizable

### Issue 8 - Examples, docs и migration guide для нового detection stack

Почему:

без обновления docs/examples новый runtime останется “скрытым знанием”, а новые разработчики продолжат копировать legacy
path.

Задачи:

- обновить docs по anomaly detection
- добавить benchmark/v2 quick path для detection
- обновить или заменить legacy examples
- оформить migration guide:
    - old detector names -> canonical names
    - classification-style path -> detection runtime path
    - benchmark old flow -> benchmark/v2 flow

DoD:

- есть onboarding-совместимая документация по detection stack
- есть example/preset/manifest path
- legacy и canonical naming различаются явно
- новый developer может воспроизвести public detection smoke без чтения исходников

Labels: docs, examples, detection, onboarding

## 10. Предлагаемый PR stack по `anomaly_detection`

Ниже предлагаемый стек PR, который можно вести параллельно командой из 2-3 разработчиков.

### PR-1 - Detection runtime foundation

Цель:

сделать typed runtime и stage vocabulary архитектурным source-of-truth для detection.

Scope:

- runtime contracts
- split/window helpers
- calibration/event aggregation helpers
- stage tuning contracts
- initial tests

Связанные issue:

- Issue 1
- Issue 2

DoD:

- typed runtime contracts и stage vocabulary лежат в репозитории
- detection registry и canonical naming policy существуют
- есть unit coverage на core runtime helpers

### PR-2 - Detection runtime strategy + repository wiring

Цель:

вывести detection из classification-by-accident path и подключить новый runtime к repository/runtime layer.

Scope:

- dedicated detection runtime strategy
- boundary adapter для `InputData`
- repository wiring
- alias policy integration
- default params update

Связанные issue:

- Issue 2
- Issue 3

DoD:

- detection больше не опирается архитектурно на classification strategy
- new canonical detector names подключены к repository
- public shell path продолжает работать

### PR-3 - Canonical detectors + explicit calibration

Цель:

сделать first-class detector families рабочим вертикальным срезом нового detection runtime.

Scope:

- `feature_iforest_detector`
- `feature_oneclass_detector`
- `conv_autoencoder_detector`
- `tcn_autoencoder_detector`
- calibration layer
- event diagnostics

Связанные issue:

- Issue 4

DoD:

- canonical detectors работают через единый runtime contract
- calibration strategies попадают в diagnostics и tuning
- legacy detectors остаются как compatibility/legacy namespace

### PR-4 - Benchmark V2 detection + public preset

Цель:

сделать detection наблюдаемым, сравнимым и запускаемым через актуальный benchmark layer.

Scope:

- `TaskType.ANOMALY_DETECTION`
- detection suite runner
- public local dataset adapter
- manifests
- presets
- registry artifacts

Связанные issue:

- Issue 5

DoD:

- detection запускается через `benchmark/v2`
- public local preset работает
- результат попадает в registry и publication artifacts

### PR-5 - MPSI preset + transfer/risk scaffold

Цель:

замкнуть рефакторинг на реальный industrial contour, не перетаскивая в git сырые архивы.

Scope:

- MPSI local preset
- data policy docs
- transfer alignment baseline
- `RiskFeatureFrame`
- smoke validation на industrial contour

Связанные issue:

- Issue 6
- Issue 7

DoD:

- есть lightweight MPSI path
- transfer/risk artifacts существуют
- public и industrial contour проходят через один detection runtime

### PR-6 - Docs, examples, migration

Цель:

сделать новый detection stack понятным для следующих разработчиков и убрать ambiguity между old path и new path.

Scope:

- docs update
- onboarding update
- examples/presets update
- migration guide
- final hardening tests

Связанные issue:

- Issue 8

DoD:

- docs и examples соответствуют новому detection path
- onboarding ведет новичка в актуальный архитектурный контур
- migration path описан явно

## 11. Как распределять задачи по команде 2-3 разработчиков

### Вариант для 3 разработчиков

Dev A:

- runtime contracts
- detection strategy
- split/calibration/event aggregation

Dev B:

- detector families
- transfer alignment
- risk scaffold

Dev C:

- benchmark/v2 detection
- presets/manifests
- docs/examples/migration

### Вариант для 2 разработчиков

Dev A:

- runtime + strategy + repository wiring

Dev B:

- detectors + benchmark/v2 + docs/presets

В этом варианте PR-6 почти наверняка придется делать отдельной стабилизационной итерацией.

## 12. Короткая памятка новому разработчику

- Не копируй архитектуру по старым примерам автоматически.
- Если не уверен, ориентируйся на forecasting refactor, а не на старый detection path.
- Любой новый runtime сначала проектируй как typed core, потом как shell integration.
- Если логика важна для оценки, думай сразу о `benchmark/v2`.
- Если рефакторишь legacy, ищи mode flags, giant methods, implicit state и stringly-typed routing.
- После кода почти всегда нужны docs и tests, а не только “рабочий класс”.

