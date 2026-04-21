# Backlog по `anomaly_detection`: issue-level декомпозиция

Ниже backlog в формате, пригодном для GitHub Issues.

## Issue 1 - Detection runtime contracts + stage vocabulary

Сейчас anomaly detection опирается на легаси с mode flags (`lagged` / `full` / `batch`) . Это
мешает сделать detection first-class задачей и не позволяет стабильно строить calibration, events, transfer и benchmark
artifacts. Задачи:

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

Почему: Сейчас detection naming живет в огромном легаси репозитории и не разделяет canonical models, aliases и legacy
families.
Это усложняет миграцию моделей и мешает benchmark/v2 агрегировать результаты по model family.

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

Почему - сейчас `IndustrialAnomalyDetectionStrategy` наследуется от classification strategy. Это архитектурно неверно: у
detection другие механизмы вывода, калибровки, scoring и event aggregation.

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

Почему - сейчас модели детектирования гетерогенны, thresholding частично скрыт внутри реализации моделей,
а новые модели не оформлены как единый runtime-first стек.

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

Почему - сейчас `benchmark/v2` покрывает forecasting, classification и regression, но не anomaly detection. Пока
detection не
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

Почему - текущий рефакто нужен не только для получения anomaly scores,
но и как база для следующего слоя failure/risk forecasting.

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

Почему- без обновления docs/examples новый runtime останется “скрытым”, а новые разработчики продолжат копировать легаси

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

## Предлагаемый PR stack по `anomaly_detection`

Ниже предлагаемый стек PR на команду из 2-3 разработчиков.

### PR-1 - Detection runtime foundation

Цель - сделать typed runtime и stage vocabulary архитектурным source-of-truth для detection.

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