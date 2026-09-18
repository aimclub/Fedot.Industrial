from __future__ import annotations

import argparse
from pathlib import Path

from .current_api import (
    DEFAULT_CLOUD_BUNDLE_DIR,
    DEFAULT_SHOWCASE_DIR,
    render_artifact_showcase,
    write_cloud_bundle_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render IndustrialTS examples artifact showcase.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SHOWCASE_DIR),
        help="Directory for index.html, inventory tables, and cloud bundle manifest.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write the cloud bundle manifest and README into the output directory.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.manifest_only:
        manifest_path = write_cloud_bundle_manifest(output_dir, include_local_index=False)
        print(manifest_path)
        return
    index_path = render_artifact_showcase(output_dir)
    bundle_manifest_path = write_cloud_bundle_manifest(DEFAULT_CLOUD_BUNDLE_DIR, include_local_index=True)
    print(index_path)
    print(bundle_manifest_path)


if __name__ == "__main__":
    main()
