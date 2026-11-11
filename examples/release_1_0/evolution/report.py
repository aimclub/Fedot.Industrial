"""Render inspectable comparison tables and figures from completed searches."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from benchmark.industrial.evaluation.evolution_parallelism import build_search_comparison
from benchmark.industrial.visualization.evolution_parallelism import render_evolution_parallelism_pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if any(args.output_dir.resolve().is_relative_to(directory.resolve()) for directory in args.runs):
        parser.error('Report output must be outside its source runs')
    frames = build_search_comparison(args.runs)
    render_evolution_parallelism_pack(frames, args.output_dir)
    sources = []
    for directory in args.runs:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.html', '.log'):
                sources.append({'path': str(path.resolve()), 'bytes': path.stat().st_size,
                                'sha256': sha256(path.read_bytes()).hexdigest()})
    (args.output_dir / 'report_manifest.json').write_text(json.dumps({
        'schema_version': 'industrial_evolution_parallelism_report@1', 'sources': sources,
        'limitations': ['Single-seed host measurements; background load not controlled.',
                        'Sum of RSS is not unique physical memory.',
                        'Same-workload speedup requires equal configuration multisets and fitness records.'],
    }, indent=2) + '\n', encoding='utf-8')
    print(frames['comparison'].to_string(index=False))


if __name__ == '__main__':
    main()
