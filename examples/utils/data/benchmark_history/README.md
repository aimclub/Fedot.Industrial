# Benchmark History Data

This directory is the local/DVC boundary for historical Industrial benchmark
results and source archives used by `examples.real_world_examples`.

Tracked files here should be manifests, README files, and lightweight report
fixtures. Raw benchmark runs, checkpoints, compressed archives, and local source
datasets are treated as DVC or manually downloaded inputs. External delivery is
described in `examples/real_world_examples/external_data_manifest.json`.

## Layout

| Path | Purpose |
| --- | --- |
| `real_world_archive/` | Evolution history and M3/M4 local archive migrated from the benchmark example area. |
