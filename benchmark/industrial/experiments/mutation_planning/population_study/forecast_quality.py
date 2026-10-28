"""Exploratory SSA diagnosis with TRAIN-only selection and a sealed test stage."""

import argparse
import csv
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import file_hash, loss_value
from .forecast_quality_protocol import load_config, select_candidates, temporal_cuts
from .forecast_recheck import _diagnostic_snapshot, require_cpu_build
from .protocol import canonical
from .runner import write_json


def load_histories(path, names):
    with Path(path).open(newline='', encoding='utf-8') as source:
        rows = [row for row in csv.reader(source) if row and row[0] in names]
    if sorted(row[0] for row in rows) != sorted(names):
        raise ValueError('Missing or duplicate requested series')
    result = {}
    for row in rows:
        cells = row[1:]
        while cells and not cells[-1].strip():
            cells.pop()
        if any(not value.strip() for value in cells):
            raise ValueError('Interior missing values cannot be removed from a time series')
        result[row[0]] = np.asarray([float(value) for value in cells])
    if any(not len(values) or not np.isfinite(values).all() for values in result.values()):
        raise ValueError('Series must be nonempty and finite')
    return result


class _ReferenceHead:
    """Independent CPU float64 least-squares probe, not a production replacement."""

    def __init__(self, features, target):
        x, y = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
        self.x_mean, self.y_mean = x.mean(axis=0), y.mean(axis=0)
        self.weights_ = np.linalg.lstsq(x - self.x_mean, y - self.y_mean, rcond=None)[0]
        self.intercept_ = self.y_mean - self.x_mean @ self.weights_

    def predict(self, features):
        return (np.asarray(features) - self.x_mean) @ self.weights_ + self.y_mean


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _linear_diagnostics(core):
    design, target = core._build_head_training_matrices(core.denoised_series_)
    # Compare the same quantized problem that the current float32 head sees.
    design, target = design.astype(np.float32).astype(float), target.astype(np.float32).astype(float)
    reference = _ReferenceHead(design, target)
    head = core.head_
    condition = float(np.linalg.cond(np.column_stack([np.ones(len(design)), design])))
    diagnostics = {'condition_number': condition if np.isfinite(condition) else None,
                   'design_rank': int(np.linalg.matrix_rank(design - design.mean(axis=0))),
                   'design_columns': design.shape[1], 'rows': len(design),
                   'reference_rmse': float(np.sqrt(np.mean((reference.predict(design) - target) ** 2)))}
    if hasattr(head, 'weights_') and head.weights_ is not None:
        weights = _array(head.weights_).reshape(-1)
        companion = np.zeros((len(weights), len(weights)))
        companion[0] = weights[::-1]
        companion[1:, :-1] = np.eye(len(weights) - 1)
        diagnostics.update(coefficient_norm=float(np.linalg.norm(weights)),
                           spectral_radius=float(np.max(np.abs(np.linalg.eigvals(companion)))),
                           actual_rmse=float(np.sqrt(np.mean((_array(head.predict(design)) - target) ** 2))))
    return diagnostics, reference


def forecast(candidate, history, horizon, seed):
    """Fit each candidate afresh; references never alter registered operations."""
    if candidate.kind == 'naive':
        if len(history) < candidate.period:
            raise ValueError('History is shorter than the seasonal period')
        return np.resize(history[-candidate.period:], horizon), {'period': candidate.period}
    if candidate.kind == 'drift':
        return (history[-1] + np.arange(1, horizon + 1) * (history[-1] - history[0]) / (len(history) - 1), {})
    import torch
    from fedot import create_data
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope

    with industrial_extension_scope(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        builder = PipelineBuilder()
        parameters = {**json.loads(candidate.parameters_json), 'head_random_seed': seed}
        for name in candidate.chain:
            builder.add_node(name, params=parameters if name == candidate.chain[-1] else None)
        pipeline = builder.build()
        verifier = IndustrialGraphVerifier(adapter=None, task_type='ts_forecasting')
        if not verifier.verify(pipeline):
            raise ValueError(f'Invalid diagnostic graph: {verifier.last_report.error_codes}')
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(horizon))
        data = create_data(history, target=history, task=task,
                           ts_forecast_horizon=horizon, without_target=True, use_cache=False)
        try:
            pipeline.fit(data)
            core = pipeline.root_node.fitted_operation.implementation.model_
            diagnostics = core.get_diagnostics()
            diagnostics['history_input_length'] = len(history)
            diagnostics['history_used_length'] = core.series_length_
            if core.head_policy_ == 'linear':
                numeric, reference = _linear_diagnostics(core)
                diagnostics['linear_system'] = numeric
                if candidate.reference_solver:
                    core.head_ = reference
                    diagnostics['original_linear_system'] = numeric
                    diagnostics['linear_system'], _ = _linear_diagnostics(core)
                    diagnostics['forecast_head'] = {**diagnostics['forecast_head'],
                                                    'solver': 'numpy_centered_lstsq_reference'}
            if candidate.denoised_state:
                prediction = core.predict()
            else:
                prediction = _array(pipeline.predict(replace(data, target=None, state='predict')).predict)
            diagnostics['reference_solver'] = candidate.reference_solver
            diagnostics['denoised_state'] = candidate.denoised_state
            return np.asarray(prediction).reshape(-1), diagnostics
        finally:
            pipeline.unfit()


def evaluate(candidate, history, actual, seed, *, scale, dataset, fold, stage):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training-only MASE scale must be positive')
    row = {'candidate': candidate.name, 'dataset': dataset, 'fold': fold, 'seed': seed, 'stage': stage,
           'train_length': len(history), 'horizon': len(actual), 'normalization_scale': scale}
    started = perf_counter()
    try:
        prediction, diagnostics = forecast(candidate, history, len(actual), seed)
        loss = loss_value('ts_forecasting', actual, prediction, scale)
        row.update(status='succeeded', mase=loss, prediction=prediction.tolist(), target=actual.tolist(),
                   diagnostics=diagnostics)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        row.update(status='failed', error_type=type(error).__name__, error_message=str(error))
    return {**row, 'seconds': perf_counter() - started}


def _admit(worktree):
    if worktree is None:
        raise ValueError('Execution requires an audited FEDOT worktree')
    import fedot
    import torch
    from fedot_ind.integration.fedot.compat import audit_compatibility_worktree
    require_cpu_build(torch)
    audit = audit_compatibility_worktree(worktree)
    if not Path(fedot.__file__).resolve().is_relative_to(Path(audit['worktree']) / 'fedot'):
        raise ValueError('Active FEDOT must belong to the audited worktree')
    torch.set_num_threads(1)
    return audit


def run_validation(config, candidates, output, audit):
    histories = load_histories(config['train'], config['datasets'])
    signature = _diagnostic_snapshot()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'purpose': 'exploratory_diagnostics_not_EVO04_acceptance', 'config': config,
                'implementation_sha256': signature, 'train_sha256': file_hash(config['train']),
                'compatibility': audit, 'test_used_for_selection': False,
                'versions': {name: version(name) for name in ('numpy', 'torch', 'scikit-learn', 'fedot')}}
    write_json(output / 'manifest.json', manifest)
    records = []
    for dataset, values in histories.items():
        for fold, cut in enumerate(temporal_cuts(len(values), config['horizon'], config['folds'])):
            history, actual = values[:cut], values[cut:cut + config['horizon']]
            scale = float(np.mean(np.abs(np.diff(history))))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Training-only MASE scale must be positive')
            for candidate in candidates:
                for seed in config['seeds']:
                    records.append(evaluate(candidate, history, actual, seed, scale=scale,
                                            dataset=dataset, fold=fold, stage='validation'))
            print(f'VALIDATION {dataset} fold={fold} train_end={cut}', flush=True)
    summaries, selections = select_candidates(records, config, candidates)
    if (signature != _diagnostic_snapshot() or file_hash(config['train']) != manifest['train_sha256']
            or _admit(audit['worktree']) != audit):
        raise ValueError('Code or input data changed during validation')
    write_json(output / 'validation.json', {'records': records, 'summaries': summaries, 'selections': selections})
    write_json(output / 'seal.json', {name: file_hash(output / name) for name in ('manifest.json', 'validation.json')})
    render_reports(output, 'validation')
    return selections


def run_test(config, candidates, output, audit):
    seal = json.loads((output / 'seal.json').read_text(encoding='utf-8'))
    if (type(seal) is not dict or set(seal) != {'manifest.json', 'validation.json'}
            or any(type(digest) is not str or len(digest) != 64
                   or any(char not in '0123456789abcdef' for char in digest) for digest in seal.values())):
        raise ValueError('Validation seal must contain exactly the two expected SHA256 records')
    if any(file_hash(output / name) != digest for name, digest in seal.items()):
        raise ValueError('Frozen validation evidence changed')
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    validation = json.loads((output / 'validation.json').read_text(encoding='utf-8'))
    signature = _diagnostic_snapshot()
    if (canonical(manifest['config']) != canonical(config) or manifest['implementation_sha256'] != signature
            or manifest['train_sha256'] != file_hash(config['train']) or manifest['compatibility'] != audit):
        raise ValueError('Code, configuration, data or FEDOT differs from frozen validation')
    _, selections = select_candidates(validation['records'], config, candidates)
    if selections != validation['selections'] or (output / 'test.json').exists():
        raise ValueError('Selection differs or test stage already exists')
    histories = load_histories(config['train'], config['datasets'])
    futures = load_histories(config['test'], config['datasets'])
    test_hash = file_hash(config['test'])
    records = []
    for dataset, history in histories.items():
        if len(futures[dataset]) != config['horizon']:
            raise ValueError('Test horizon differs from frozen protocol')
        names = set(config['test_comparators']) | {name for name in selections[dataset].values() if name is not None}
        scale = float(np.mean(np.abs(np.diff(history[:-config['horizon']]))))
        for candidate in candidates:
            if candidate.name in names:
                for seed in config['seeds']:
                    records.append(evaluate(candidate, history, futures[dataset], seed, scale=scale,
                                            dataset=dataset, fold=None, stage='test'))
    if (signature != _diagnostic_snapshot() or test_hash != file_hash(config['test'])
            or manifest['train_sha256'] != file_hash(config['train']) or _admit(audit['worktree']) != audit):
        raise ValueError('Code or test data changed during evaluation')
    write_json(output / 'test.json', {'test_sha256': test_hash, 'records': records,
                                      'selections': selections, 'full_study_accepted': False})
    render_reports(output, 'test')
    return selections


def render_reports(output, stage):
    """Reuse the shared renderer; plot identities include actual dataset and fold."""
    import pandas as pd
    from benchmark.industrial.visualization.forecast_comparison import render_forecast_comparison_pack
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    config = manifest['config']
    report = json.loads((output / f'{stage}.json').read_text(encoding='utf-8'))
    histories = load_histories(config['train'], config['datasets'])
    rows = report['records']
    table_rows = [{key: value for key, value in row.items() if key not in ('prediction', 'target', 'diagnostics')}
                  | row.get('diagnostics', {}).get('linear_system', {}) for row in rows]
    pd.DataFrame(table_rows).to_csv(output / f'{stage}_metrics.csv', index=False)
    if stage == 'validation':
        pd.DataFrame(report['summaries']).to_csv(output / 'validation_summary.csv', index=False)
    for dataset in config['datasets']:
        selected = report['selections'][dataset]
        names = {'naive', 'seasonal7', selected['best'], selected['best_industrial']}
        fold = config['folds'] - 1 if stage == 'validation' else None
        observed = [row for row in rows if row['dataset'] == dataset and row['fold'] == fold
                    and row['seed'] == config['seeds'][0] and row['status'] == 'succeeded'
                    and row['candidate'] in names]
        if not observed:
            continue
        history = histories[dataset][:-config['horizon']] if stage == 'validation' else histories[dataset]
        render_forecast_comparison_pack(
            history=history[-6 * config['horizon']:], actual=observed[0]['target'],
            forecasts={row['candidate']: row['prediction'] for row in observed},
            output_dir=output / 'figures' / stage / dataset, series_id=f'{dataset}_{stage}',
            title=f'{dataset}: {stage}, horizon={config["horizon"]}',
            source_metadata={'stage': stage, 'seed': config['seeds'][0], 'fold': fold,
                             'history_end': len(history), 'full_history_length': len(history),
                             'train_sha256': manifest['train_sha256'],
                             'record_source_sha256': file_hash(output / f'{stage}.json')})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compatibility-worktree', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args(argv)
    config, candidates = load_config(args.config)
    if not args.execute:
        print(json.dumps({'status': 'planned', 'config': config, 'test_stage': args.evaluate_test}))
        return 0
    audit = _admit(args.compatibility_worktree)
    method = run_test if args.evaluate_test else run_validation
    print(json.dumps(method(config, candidates, args.output.resolve(), audit)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
