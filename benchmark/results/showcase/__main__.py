from __future__ import annotations

import argparse
from pathlib import Path

from .current_api import DEFAULT_SHOWCASE_DIR, render_results_showcase


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the benchmark/results Industrial showcase.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SHOWCASE_DIR),
        help="Directory where showcase tables, plots, and README are written.",
    )
    args = parser.parse_args()
    manifest_path = render_results_showcase(Path(args.output_dir))
    print(manifest_path)


if __name__ == "__main__":
    main()

