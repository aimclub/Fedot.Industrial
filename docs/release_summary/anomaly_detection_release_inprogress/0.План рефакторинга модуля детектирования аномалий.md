# Detection Refactor Phase 1

## Purpose

This document is the source-of-truth for the first radical refactor wave of the anomaly-detection stack in
`Fedot.Industrial`.

The target architecture is:

- detection-first rather than classification-by-accident;
- typed runtime contracts instead of mode flags and implicit state;
- stage-aware diagnostics and benchmark artifacts;
- equal emphasis on public detection benchmarks and the MPSI industrial use case;
- explicit scaffolding for later failure-risk modeling without blocking the anomaly-detection wave on that work.

## Architectural Defaults

- Scope of wave 1: anomaly detection and supporting runtime only.
- Canonical evaluation layer: `benchmark/v2`.
- Canonical internal data flow: typed detection runtime objects, not raw `InputData` beyond the boundary.
- Public FEDOT operation names may remain as aliases, but aliases are not the architectural source-of-truth.
- Failure forecasting is deferred; this wave exports risk-ready features and joins only.

## Target Runtime Vocabulary

Wave 1 introduces explicit detection runtime objects:

- `DetectionWindowBatch`
- `DetectionSplitSpec`
- `RegimeSegment`
- `AnomalyScoreSeries`
- `DetectionEvent`
- `TransferAlignmentReport`
- `DetectionSeriesEvaluation`
- `DetectionStageTuningPlan`
- `RiskFeatureFrame`

The canonical stage vocabulary is:

1. `data_quality`
2. `regime_segmentation`
3. `representation`
4. `anomaly_scoring`
5. `calibration`
6. `event_aggregation`
7. `transfer_alignment`
8. `interpretation`

## Model Families

Wave 1 first-class detection families:

- `feature_iforest_detector`
- `feature_oneclass_detector`
- `conv_autoencoder_detector`
- `tcn_autoencoder_detector`

Legacy and experimental families are preserved behind non-canonical names:

- `legacy_lstm_autoencoder_detector`
- `legacy_arima_detector`
- `legacy_sst_detector`
- `legacy_kalman_detector`
- `legacy_functional_pca_detector`

## Delivery Slices

### Slice 1. Runtime Foundation

- introduce typed runtime contracts;
- introduce calibration, event aggregation and transfer-alignment helpers;
- publish canonical detection naming and stage-tuning contracts.

Acceptance:

- new detection runtime exists;
- thin-shell detectors can consume it;
- roadmap and tests are checked in together.

### Slice 2. First-Class Detectors

- move classical baselines to runtime-first detectors;
- rewrite reconstruction detectors on the new runtime;
- expose stage diagnostics and risk-ready exports.

Acceptance:

- all first-class detectors emit typed scores, events and stage diagnostics;
- hidden threshold heuristics are replaced by named calibration strategies.

### Slice 3. Benchmark V2 Detection

- add a dedicated `benchmark/v2` detection suite;
- support one public local benchmark and one lightweight MPSI local preset;
- persist structured diagnostics and family summaries.

Acceptance:

- one public preset and one MPSI preset run through the same detection benchmark suite;
- results are persisted as structured benchmark artifacts.

## MPSI Data Policy

- continuous analog telemetry and vibration are first-class signals;
- discrete PIMS tags are used mainly for synchronization, regime context and interpretation;
- raw MPSI archives are not committed as benchmark fixtures;
- the repository stores lightweight derived fixtures, manifests and metadata only.

## Exit Criteria

Wave 1 is complete when:

- the detection runtime is the source-of-truth for first-class detectors;
- benchmark-v2 detection exists;
- public and MPSI local presets both run successfully;
- risk-ready feature exports are available for the next failure-modeling wave.
