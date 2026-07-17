# Detection Benchmark Examples

This package keeps the SKAB anomaly-detection wrapper and local historical
result metadata. The current smoke entrypoint reads task-based fixtures from
`examples/utils/data/anomaly_detection/skab`; full historical outputs stay external.

## Files

| File | Use case |
| --- | --- |
| `ts_anomaly_detection_skab_bench.py` | Current SKAB context builder. |
| `results_manifest.json` | Expected local result directories and metric semantics. |
