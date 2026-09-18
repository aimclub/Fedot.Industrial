# SKAB Fixture

Small SKAB CSV fixture used by anomaly-detection examples and loader tests.

## Layout

| Path | Purpose |
| --- | --- |
| `anomaly-free/anomaly-free.csv` | Normal-mode training slice. |
| `valve1/*.csv` | Test cases for outlet valve closing scenarios. |
| `valve2/*.csv` | Test cases for inlet valve closing scenarios. |
| `other/*.csv` | Additional leak, rotor imbalance, water-level, and cavitation scenarios. |

Consumers should resolve this folder through
`fedot_ind.tools.loader.resolve_skab_data_root()` instead of hard-coding the
historical `examples/utils/data/benchmark/detection/data` path.
