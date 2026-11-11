import argparse
import json
from pathlib import Path

from .runner import compare_mutations, write_comparison


def main():
    parser = argparse.ArgumentParser(description="Paired bounded structural mutation study (no training)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    print(write_comparison(compare_mutations(config), args.output_dir))


if __name__ == "__main__":
    main()
