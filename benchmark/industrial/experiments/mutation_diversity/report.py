"""Report entrypoint: ingestion and rendering live with their owning modules."""
import json
from pathlib import Path
from benchmark.industrial.evaluation.mutation_diversity import trial_row
from benchmark.industrial.visualization.mutation_diversity import render_mutation_diversity_report


def build_report(root, output_dir):
    root = Path(root)
    jobs = json.loads((root / 'jobs.json').read_text('utf-8'))
    rows = [trial_row(root / f"{r['task']}_{r['seed']}_{r['arm']}", r) for r in jobs]
    return render_mutation_diversity_report(rows, output_dir)
