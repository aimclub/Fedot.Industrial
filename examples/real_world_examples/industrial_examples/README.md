# Industrial Domain Examples

This package contains applied domain scenarios. The notebooks are preserved as
domain references, while `current_api.py` and `scenario_defaults.json` define the
current task type, model set, metric direction, expected local data path, and
artifact location for each scenario.

## Contract

- New scenario logic lives in the domain package next to its public entrypoint.
- Constants and model defaults live in `scenario_defaults.json`.
- Shared aggregation and visualization should use `benchmark.industrial`.
- Raw data, checkpoints, and high-churn outputs stay outside git and are
  described by `../external_data_manifest.json`.

Run a lightweight preflight without loading local data:

```powershell
python -m examples.real_world_examples.industrial_examples.current_api
```
