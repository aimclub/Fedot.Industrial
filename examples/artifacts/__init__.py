"""Central artifact catalog and showcase helpers for IndustrialTS examples."""

from .current_api import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_SHOWCASE_DIR,
    build_artifact_inventory,
    index_local_artifacts,
    load_artifact_catalog,
    render_artifact_showcase,
    write_cloud_bundle_manifest,
)

__all__ = [
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_SHOWCASE_DIR",
    "build_artifact_inventory",
    "index_local_artifacts",
    "load_artifact_catalog",
    "render_artifact_showcase",
    "write_cloud_bundle_manifest",
]
