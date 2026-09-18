# Anomaly Detection Data

Contains anomaly-detection fixtures grouped by dataset family.

## Fixtures

- `skab/`: SKAB valve/anomaly-free CSV files consumed by `DataLoader` and
  SKAB benchmark contexts.

Local LIMAN or other vibration datasets should live under this task folder, for
example `liman/vibro/without_failure/...`, but are not committed as repository
fixtures.

When migrating local data from legacy pre-`utils` real-world paths,
preserve the regime folders (`without_failure`, `bearing_failure`,
`rotor_failure`, `stator_failure`) under `liman/vibro/`.
