# Examples Artifact Hub

This package is the single handoff point for IndustrialTS example artifacts.
It does not move large local data into git. Instead it keeps a catalog,
inventory, cloud-bundle manifest, and optional local static showcase that point
to the current benchmark and real-world report packs in external storage.

## What Lives Here

| Path | Purpose |
| --- | --- |
| `artifact_catalog.json` | Declarative map of raw inputs, benchmark history, notebooks, and externally published report assets. |
| `current_api.py` | Python API for loading the catalog, building an inventory, writing cloud manifests, and rendering the showcase. |
| `showcase/` | Ignored local HTML showcase plus CSV/JSON inventory. |
| `cloud_bundle/` | Ignored local handoff folder for generated manifest/README and optional upload payloads for Yandex Disk, DVC, or another external storage. |

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

`showcase/` and `cloud_bundle/` are generated local outputs and are ignored by git.

For cloud handoff without rendering HTML:

```bash
python -m examples.artifacts --manifest-only --output-dir examples/artifacts/cloud_bundle
```

`--manifest-only` intentionally writes only metadata. Use the default command
to refresh the local artifact inventory for metrics tables, Markdown summaries,
source metadata, and plots that already live in the ignored local `cloud_bundle/`.

## Cloud Policy

Use the generated `cloud_bundle_manifest.json` as the upload checklist.
Generated artifacts such as summaries, plots, notebooks, manifests, and CSV
or Markdown tables may be staged under the local ignored `cloud_bundle/` before
upload, but the GitHub repository stores only the catalog, source notebooks,
README files, and code that regenerates the bundle.
Large raw datasets, checkpoints, archives, and full benchmark run directories
should be uploaded through DVC or a manual cloud folder and referenced through
`examples/real_world_examples/external_data_manifest.json`.


## Size Policy

`cloud_bundle/` is a local generated handoff directory. Do not commit generated
report packs from it.

- Single committed generated artifact: 0 MB by default.
- Whole committed `cloud_bundle/`: 0 MB by default.
- Raw datasets, checkpoints, archives, and full benchmark run folders stay outside git and are referenced through `examples/real_world_examples/external_data_manifest.json`.
- The current public archive for external users is `https://disk.yandex.ru/d/Ch_7K26rukpAWw`.
