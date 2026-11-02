"""Collect measured evidence without importing or fitting model implementations."""

import json
from pathlib import Path

from benchmark.industrial.evaluation.staged_search import staged_trial_row, paired_differences
from benchmark.industrial.visualization.mutation_diversity import render_mutation_diversity_report


def build_report(root, output_dir):
    import pandas as pd
    root, output_dir = Path(root), Path(output_dir)
    jobs = json.loads((root / 'jobs.json').read_text('utf-8'))
    rows = [staged_trial_row(root / f"{job['task']}_{job['seed']}_{job['mode']}", job) for job in jobs]
    render_mutation_diversity_report(rows, output_dir, arm_order=('fixed', 'staged'),
                                     title='Fixed versus staged mutations: TRAIN-only comparison')
    pairs = paired_differences(rows)
    (output_dir / 'paired_differences.json').write_text(json.dumps(pairs, indent=2) + '\n', encoding='utf-8')
    pd.DataFrame(pairs).to_csv(output_dir / 'paired_differences.csv', index=False)
    return rows, pairs
