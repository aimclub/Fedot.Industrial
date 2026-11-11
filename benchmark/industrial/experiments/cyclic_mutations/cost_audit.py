"""Read-only cost attribution from operation traces; never load model data or fit."""
import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def summarize_operations(rows):
    operations, candidates = defaultdict(lambda: [0, 0.]), {}
    for row in rows:
        seconds = row['exclusive_seconds']
        if not isinstance(seconds, (float, int)) or not isfinite(seconds) or seconds < 0:
            raise ValueError('Operation duration must be finite and nonnegative')
        total = operations[row['operation']]
        total[0] += 1
        total[1] += seconds
        candidate = candidates.setdefault(row['evaluation_id'], dict(
            evaluation_id=row['evaluation_id'], configuration_id=row['configuration_id'],
            configuration=row['configuration'], operation_seconds=defaultdict(float)))
        candidate['operation_seconds'][row['operation']] += seconds
    total_seconds = sum(item[1] for item in operations.values())
    summary = [dict(operation=name, calls=count, worker_seconds=seconds,
                    fraction_of_traced_work=seconds / total_seconds if total_seconds else 0.)
               for name, (count, seconds) in operations.items()]
    return sorted(summary, key=lambda r: -r['worker_seconds']), candidates


def audit_run(directory):
    root = Path(directory)
    paths = sorted((root / 'traces').glob('operations_*.jsonl'))
    operations, candidates = summarize_operations(row for p in paths for row in read_jsonl(p))
    evaluations = [row for p in sorted((root / 'traces').glob('evaluations_*.jsonl')) for row in read_jsonl(p)]
    for row in evaluations:
        if row['evaluation_id'] in candidates:
            candidates[row['evaluation_id']].update(evaluation_seconds=row['seconds'], valid=row['valid'])
    phases_path, batches_path = root / 'phase_timings.json', root / 'batch_timings.json'
    phases = json.loads(phases_path.read_text('utf-8'))
    batches = json.loads(batches_path.read_text('utf-8'))
    traced_seconds = sum(row['worker_seconds'] for row in operations)
    consumed_seconds = sum(row['evaluation_seconds'] for row in candidates.values() if 'evaluation_seconds' in row)
    source_paths = [*paths, *sorted((root / 'traces').glob('evaluations_*.jsonl')), phases_path, batches_path]
    return dict(schema_version='industrial_search_cost_audit@1', source=str(root.resolve()), fit_calls=0,
                attribution='exclusive worker-seconds, not additive wall time; interrupted operations may be absent',
                phases=phases, operations=operations,
                traced_worker_seconds=traced_seconds, completed_evaluation_worker_seconds=consumed_seconds,
                batch_capacity_seconds=sum(b['candidates'] * b['execution_seconds'] for b in batches),
                candidates=sorted(candidates.values(), key=lambda r: -r.get('evaluation_seconds', 0.)),
                sources={str(p.resolve()): sha256(p.read_bytes()).hexdigest() for p in source_paths})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'costs.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (args.output_dir / 'operations.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['operation', 'calls', 'worker_seconds', 'fraction_of_traced_work'])
        writer.writeheader()
        writer.writerows(result['operations'])
    print(json.dumps({k: result[k] for k in ('fit_calls', 'operations', 'completed_evaluation_worker_seconds',
                                             'batch_capacity_seconds')}))


if __name__ == '__main__':
    main()
