# Contract Map

## Canonical Entities

IndustrialTS does not currently center around one `fedot/extensions` package. In this repository, the closest contract surfaces live in benchmark and model-adapter layers.

Use these entities as the stable vocabulary:

- `BenchmarkSuiteConfig`
- `DatasetSpec`
- `ModelSpec`
- `RunSpec`
- adapter-specific metadata records
- typed benchmark and product-layer results

## Current Responsibility Split

### `benchmark/v2/core.py`

Defines the typed data contract for benchmark configs, run records, metric records, prediction records, and aggregate reports.

### `benchmark/v2/forecasting.py`

Owns:

- dataset adapters
- model adapters
- optional external baseline handling
- runtime forecasting execution and record creation

### `benchmark/v2/manifests.py` and `benchmark/v2/presets.py`

Own typed manifest and preset loading, normalization, and config materialization.

### `benchmark/v2/registry.py`

Owns registered run persistence, artifact manifests, and run-level registry state.

### `fedot_ind/core/models/kernel` and `fedot_ind/core/models/automl`

Act as runtime adapters over validated configs, policies, and benchmark-facing model wrappers.

## Canonical Integration Flow

1. Author a typed preset, manifest, or model spec.
2. Load and validate it into canonical config objects.
3. Instantiate dataset and model adapters from that validated config.
4. Smoke test or skip unavailable external integrations explicitly.
5. Persist runtime behavior and artifacts from the validated spec.

## Practical Principle

The more work that can happen from a validated typed spec, the less downstream code needs to care about raw adapter metadata or optional dependency quirks.
