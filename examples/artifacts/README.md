# Examples Artifact Hub

This package is the single handoff point for IndustrialTS example artifacts.
It does not move large local data into git. Instead it keeps a catalog,
inventory, cloud-bundle manifest, and a lightweight static showcase that point
to the current benchmark and real-world report packs.

## What Lives Here

| Path | Purpose |
| --- | --- |
| `artifact_catalog.json` | Declarative map of raw inputs, benchmark history, notebooks, and canonical report assets under `cloud_bundle/`. |
| `current_api.py` | Python API for loading the catalog, building an inventory, writing cloud manifests, and rendering the showcase. |
| `showcase/` | Generated local HTML showcase plus CSV/JSON inventory. |
| `cloud_bundle/` | Canonical local folder for lightweight report assets plus generated manifest/README for Google Drive or DVC storage. |

## Render

```bash
python -m examples.artifacts
```

The command writes:

- `examples/artifacts/showcase/index.html`
- `examples/artifacts/showcase/artifact_inventory.csv`
- `examples/artifacts/showcase/artifact_inventory.json`
- `examples/artifacts/showcase/cloud_bundle/cloud_bundle_manifest.json`
- `examples/artifacts/cloud_bundle/cloud_bundle_manifest.json`
- `examples/artifacts/cloud_bundle/local_artifacts.json`
- `examples/artifacts/cloud_bundle/benchmark_showcase/...`
- `examples/artifacts/cloud_bundle/domain_showcase/...`

For cloud handoff without rendering HTML:

```bash
python -m examples.artifacts --manifest-only --output-dir examples/artifacts/cloud_bundle
```

`--manifest-only` intentionally writes only metadata. Use the default command
to refresh the local artifact inventory for metrics tables, Markdown summaries,
source metadata, and plots that already live in `cloud_bundle/`.

## Cloud Policy

Use the generated `cloud_bundle_manifest.json` as the upload checklist.
Lightweight artifacts such as summaries, plots, notebooks, manifests, and CSV
or Markdown tables live under `cloud_bundle/` as the canonical local source.
Large raw datasets, checkpoints, archives, and full benchmark run directories
should be uploaded through DVC or a manual cloud folder and referenced through
`examples/real_world_examples/external_data_manifest.json`.
