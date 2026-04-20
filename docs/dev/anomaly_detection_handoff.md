# Anomaly Detection Refactor Handoff

Ниже сообщение, которое можно целиком вставить в новый чат с чистым контекстом.

```md
Продолжаем реализацию Phase 1 рефакторинга anomaly detection в репозитории `D:\data_old\WORK\Repo\Industiral\IndustrialTS`.

Текущая цель:
- довести рефакторинг `fedot_ind/core/models/detection` до рабочего vertical slice;
- сделать detection first-class задачей, а не оберткой над classification;
- подключить новый detection runtime к benchmark/v2;
- сохранить совместимость на уровне разумных alias, но не держаться за старую архитектуру.

Что уже сделано в рабочем дереве:
- создан новый typed runtime: [runtime.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/core/models/detection/runtime.py)
- создан реестр canonical detection names и alias: [detection_registry.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/core/repository/detection_registry.py)
- создан stage-aware tuning contract: [stage_tuning.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/core/models/detection/stage_tuning.py)
- добавлены thin-shell modern detectors: [modern_detectors.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/core/models/detection/modern_detectors.py)
- сохранен roadmap Phase 1: [anomaly_detection_phase_1_roadmap.md](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/docs/dev/anomaly_detection_phase_1_roadmap.md)

Что содержат новые файлы:
- `runtime.py`:
  - dataclass/runtime контракты `DetectionWindowBatch`, `DetectionSplitSpec`, `RegimeSegment`, `AnomalyScoreSeries`, `DetectionEvent`, `TransferAlignmentReport`, `DetectionSeriesEvaluation`, `RiskFeatureFrame`
  - helpers для windowing, statistical features, regime inference, calibration, event aggregation, transfer alignment, risk feature export
- `stage_tuning.py`:
  - stage vocabulary: `data_quality`, `regime_segmentation`, `representation`, `anomaly_scoring`, `calibration`, `event_aggregation`, `transfer_alignment`, `interpretation`
  - `DetectionStageTuningPlan` и fallback-группы параметров для canonical detectors
- `detection_registry.py`:
  - canonical names: `feature_iforest_detector`, `feature_oneclass_detector`, `conv_autoencoder_detector`, `tcn_autoencoder_detector`
  - legacy aliases: `iforest_detector`, `stat_detector`, `conv_ae_detector`, `lstm_ae_detector`, `sst`, `arima_detector`, `unscented_kalman_filter`, `functional_pca`
- `modern_detectors.py`:
  - `BaseRuntimeAnomalyDetector`
  - `FeatureIsolationForestDetector`
  - `FeatureOneClassDetector`
  - `ConvAutoencoderDetector`
  - `TCNAutoencoderDetector`
  - fit/predict flow уже строится через новый runtime, stage diagnostics, score series, events и risk frame

Что еще не доведено:
- новые detectors пока не подключены в основной model repository и default params
- `fedot_ind/core/models/detection/__init__.py` пока пустой
- detection еще не встроен в `benchmark/v2`
- `problem='anomaly_detection'` все еще живет в classification-style wiring и требует перевода на first-class dispatch
- не добавлены полноценные unit/integration tests под новый runtime
- не выполнена syntax/pytest проверка новых файлов

Ключевые места, которые нужно смотреть следующими:
- [fedot_ind/core/repository/model_repository.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/core/repository/model_repository.py)
- [fedot_ind/core/repository/data/industrial_model_repository.json](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/core/repository/data/industrial_model_repository.json)
- [fedot_ind/core/repository/data/default_operation_params.json](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/core/repository/data/default_operation_params.json)
- [fedot_ind/core/operation/interfaces/industrial_model_strategy.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/fedot_ind/core/operation/interfaces/industrial_model_strategy.py)
- [benchmark/v2/core.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/core.py)
- [benchmark/v2/api.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/api.py)
- [benchmark/v2/presets.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/presets.py)
- [benchmark/v2/manifests.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/manifests.py)
- [benchmark/v2/registry.py](/D:/data_old/WORK/Repo/Industiral/IndustrialTS/benchmark/v2/registry.py)

Полезный контекст по данным:
- в репозитории уже есть публичные detection CSV для benchmark: `examples/data/benchmark/detection/data/...`
- формат включает normal и anomaly series; в anomaly файлах есть как минимум `anomaly` и `changepoint`
- для industrial-кейса есть материалы в `docs/pr_plans/anomaly_detection_refactoring_plan/` и `examples/real_world_examples/industrial_examples/anomaly_detection/`

Рекомендуемый следующий план действий:
1. Подключить new runtime detectors в модельный реестр и aliases, не ломая legacy names.
2. Экспортировать новые сущности из `fedot_ind/core/models/detection/__init__.py`.
3. Добавить `TaskType.ANOMALY_DETECTION` и отдельный suite в `benchmark/v2`.
4. Реализовать detection benchmark runner на существующих CSV из `examples/data/benchmark/detection/data`.
5. Добавить lightweight local MPSI preset или derived fixture без сырых архивов.
6. Добавить unit tests для runtime/stage tuning/detectors и smoke integration tests для benchmark/v2 detection.
7. Прогнать хотя бы `py_compile` и таргетные `pytest`.

Ожидаемые ограничения и риски:
- в рабочем дереве много посторонних untracked файлов и артефактов; их не нужно трогать без необходимости
- работай только точечно и не откатывай чужие изменения
- `benchmark/v2` уже используется для forecasting/classification/regression, поэтому detection лучше добавлять аддитивно
- желательно не тащить сырые MPSI архивы в git; только manifests, lightweight fixtures и derived metadata

Что важно проверить в первую очередь:
- синтаксис `runtime.py`, `modern_detectors.py`, `stage_tuning.py`, `detection_registry.py`
- нет ли конфликтов с текущими FEDOT `InputData`/`ModelImplementation` контрактами
- как лучше встроить detection в operation repository без сохранения classification-by-accident архитектуры

Если делаешь следующую итерацию, начни с быстрого аудита текущих новых файлов и потом переходи к repository wiring + benchmark/v2 detection + tests.
```

