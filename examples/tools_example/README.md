# Tools Examples

Status: MCP-ready Industrial service layer.

This package exposes JSON-friendly actions that can be mapped to MCP tools by
an external server or agent runtime. The implementation is intentionally a pure
Python service layer: `registry.py` owns tool specs and dispatch,
`contracts.py` owns request/response records, and `services/` owns the actual
Industrial actions.

```powershell
python -m examples.tools_example
```

## Tool Catalog

| Tool | Capability | Default behavior |
| --- | --- | --- |
| `industrial_load_data` | `data.load` | Lists task data or previews a path. |
| `industrial_train_model` | `model.train` | Builds a training config unless `execute=true`. |
| `industrial_run_evolution` | `optimization.evolution` | Builds Kernel Learning two-stage optimization config unless `execute=true`. |
| `industrial_run_pdl_training` | `model.train.pdl` | Builds a PDL classifier/regressor config unless `execute=true`. |
| `industrial_detect_anomalies` | `anomaly.detect` | Resolves anomaly context unless `execute=true`. |

Compatibility aliases such as `industrial_tsc_smoke` and
`industrial_forecasting_config_preview` are still exported, but new agent work
should call the action-oriented tools above.

## Execution Policy

All heavy tools are dry-run by default. Pass `execute=true` to train models,
run evolutionary optimization, or score anomaly data. Every response is a
structured `ToolResponse` with `status`, `dry_run`, `data`, optional
`artifacts`, and optional `error`.

Example payload:

```json
{
  "task_type": "ts_classification",
  "dataset_name": "Lightning7",
  "output_dir": "benchmark/results/tools_example/ucr_lightning7",
  "execute": false
}
```

## Local Inputs

Optional datasets such as LIMAN vibration files and big-data folds are local
inputs. Keep them untracked and use paths from `tool_defaults.json`, for
example `examples/utils/data/anomaly_detection/liman`.

New tool implementations should live in a thematic `services/<area>.py` module
next to the public registry. `__init__.py` should remain a short public index;
large constants belong in JSON defaults next to the entrypoint.
