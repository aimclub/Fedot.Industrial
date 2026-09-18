# OKHS Forecasting

This package keeps the applied OKHS forecasting scenario and a small
`MonashBitcoin_30.csv` sample. The current entrypoint builds typed benchmark
configs first; execution requires the optional forecasting runtime.

Use `vis_utils.render_okhs_forecasting_progress` and
`vis_utils.render_okhs_acceptance_pack` for visualization/report artifacts after
a real run. Both functions delegate to shared `benchmark.industrial`
visualization and evaluation helpers.

`example_common.py`, `okhs_experiment_utils.py`, and `okhs_advanced.py` contain
the optional research helpers for pycaputo/deep-OKHS experiments. They are kept
inside this forecasting package so imports and implementations stay near the
scenario entrypoint.
