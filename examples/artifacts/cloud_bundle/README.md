# IndustrialTS Artifact Cloud Bundle

This folder is the handoff point for publishing examples artifacts outside git.
It contains a machine-readable manifest and keeps raw-data policy explicit.

## Ownership And Size Policy

Owner: IndustrialTS benchmark maintainers
Last refreshed: 2026-07-14
Refresh command: `python -m examples.artifacts`
External archive: https://disk.yandex.ru/d/Ch_7K26rukpAWw
Max committed file size: 5 MB
Max committed bundle size: 100 MB

## Rules

- Upload large raw datasets, checkpoints, archives, and full benchmark runs through DVC or a manual cloud folder.
- Keep credentials in `.dvc/config.local` or environment variables only.
- Lightweight summaries, plots, notebooks, manifests, and CSV/Markdown tables can be mirrored for review.

## Groups

| Key | Category | Inventory | Storage | Local path | Cloud path | Files | Size bytes | Local files |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `benchmark_analysis_publication_packs` | `viz_artifacts` | `pack` | `canonical_local` | `examples/artifacts/cloud_bundle/benchmark_showcase/analysis_of_results` | `benchmark_showcase/analysis_of_results` | 93 | 4065006 | 93 |
| `forecasting_benchmark_publication_packs` | `viz_artifacts` | `pack` | `canonical_local` | `examples/artifacts/cloud_bundle/benchmark_showcase/forecasting` | `benchmark_showcase/forecasting` | 40 | 2671651 | 40 |
| `industrial_domain_publication_packs` | `viz_artifacts` | `pack` | `canonical_local` | `examples/artifacts/cloud_bundle/domain_showcase/industrial_examples` | `domain_showcase/industrial_examples` | 147 | 780029 | 147 |
| `benchmark_history_archive` | `archive_data` | `shallow` | `manifest_only` | `examples/utils/data/benchmark_history` | `raw_inputs/benchmark_history` | 5 | 2319279 | 0 |
| `examples_utils_local_data` | `raw_and_fixture_data` | `shallow` | `manifest_only` | `examples/utils/data` | `raw_inputs/examples_utils_data` | 44 | 1723253411 | 0 |
| `current_kernel_learning_full_runs` | `run_artifacts` | `manifest` | `manifest_only` | `benchmark/results/kernel_learning` | `benchmark_runs/kernel_learning` | 3 | 67914 | 0 |

See `cloud_bundle_manifest.json` for the full catalog.
