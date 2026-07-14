# Forecasting Benchmark Examples

Forecasting benchmark examples now use current `benchmark.industrial` config
previews and publication-pack entrypoints. Local Kaggle CSV files are expected
under `examples/utils/data/forecasting/kaggle_inventory` or another explicit
`data_dir` passed to the entrypoint; they are not committed.

Historical M4 result directories are external inputs described in
`results_manifest.json`. Use the root `external_data_manifest.json` for the public Yandex Disk archive and optional DVC metadata.
