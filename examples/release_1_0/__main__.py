"""python -m examples.release_1_0 [--execute]."""

import argparse
import json

from .config import SCENARIOS, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS,
                        default="motions_graph")
    parser.add_argument(
        "--config", help="Adjacent defaults are used unless a JSON file is supplied")
    parser.add_argument("--backend", choices=("sequential", "process"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", default="results/release_1_0")
    args = parser.parse_args(argv)
    config = load_config(args.scenario, args.config,
                         backend=args.backend, device=args.device)
    print(json.dumps({"mode": "execute" if args.execute else "preview", "config": config.to_record(),
                      "output_dir": str(args.output_dir)}, indent=2))
    if args.execute:
        from .runner import run
        report = run(config, args.output_dir)
        print(json.dumps({key: report[key] for key in (
            "status", "train_count", "test_count", "metrics")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
