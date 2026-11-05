"""Fixed-graph TRAIN replay and repeated transform timing; no quality tuning."""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

from .cost_audit import audit_run, read_jsonl


@contextmanager
def reference_windows():
    from fedot_ind.core.operation.transformation.torch_backend.statistical import quantile_extractor as module
    from fedot_ind.core.operation.transformation.torch_backend.statistical.window_batch import _reference_windows
    original = module.TorchQuantileExtractor._can_batch_numpy
    with patch.object(module.TorchQuantileExtractor, '_can_batch_numpy',
                      lambda self, values: self.window_size == 0 and original(self, values)), patch.object(
            module, 'window_statistics', lambda values, plan, limit: _reference_windows(values, plan)):
        yield


def flags():
    import torch
    return dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled())


def fixed_cases(source):
    candidates = []
    failed = [r for r in read_jsonl(source / 'coordinator_events.jsonl')
              if r['event_type'] == 'evaluation' and r['status'] == 'failed']
    for row in failed:
        raw = json.loads((source / 'history/candidates' / f"{row['individual_id']}.json").read_text('utf-8'))
        candidates.append(raw['metadata']['evaluation_context'])
    slowest = audit_run(source)['candidates'][0]
    candidates.append(slowest)
    return candidates


def replay(root, data, cases):
    import numpy as np
    import dask
    from benchmark.industrial.experiments.mutation_planning.population_study.winner_ranking.core import FrozenCandidate
    from benchmark.industrial.experiments.mutation_planning.population_study.winner_ranking.runner import build_pipeline
    from benchmark.industrial.experiments.evolution_parallelism.search import loss_value, to_numpy
    from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json
    from fedot_ind.core.optimizer.randomness import operator_random_scope
    from fedot_ind.core.operation.transformation.representation.statistical.feature_reuse import run_with_feature_reuse
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    results = []
    for index, case in enumerate(cases):
        configuration = canonical_json(case['configuration'])
        identity = PreparedEvaluationGraph(None, configuration).configuration_id
        if identity != case['configuration_id']:
            raise ValueError('Recorded candidate identity differs')
        candidate = FrozenCandidate(str(index), identity, configuration, 'cost-only replay')
        predictions = {}
        for mode in ('reference', 'optimized'):
            with ExitStack() as stack:
                stack.enter_context(industrial_extension_scope())
                stack.enter_context(dask.config.set(scheduler='synchronous'))
                stack.enter_context(operator_random_scope(case.get('seed', 42)))
                if mode == 'reference':
                    stack.enter_context(reference_windows())
                for fold, (train, valid) in enumerate(data.folds):
                    model = build_pipeline(candidate)
                    try:
                        start = perf_counter()
                        run_with_feature_reuse(model.fit, train, n_jobs=1)
                        output = run_with_feature_reuse(model.predict, replace(valid, target=None, state='predict'))
                        seconds = perf_counter() - start
                        prediction = to_numpy(output.predict).copy()
                        if mode == 'reference':
                            predictions[fold] = prediction
                        else:
                            np.testing.assert_array_equal(prediction, predictions[fold])
                        np.save(root / f'case{index}_fold{fold}_{mode}.npy', prediction, allow_pickle=False)
                        row = dict(case=index, configuration_id=identity, configuration=case['configuration'],
                                   fold=fold, mode=mode, seconds=seconds,
                                   loss=loss_value('regression', valid.target, prediction),
                                   prediction_sha256=sha256(prediction.tobytes()).hexdigest())
                        results.append(row)
                        (root / 'replay.json').write_text(json.dumps(results,
                                                                     indent=2, allow_nan=False) + '\n', encoding='utf-8')
                        print(json.dumps({k: row[k] for k in ('case', 'fold', 'mode', 'seconds', 'loss')}), flush=True)
                    finally:
                        model.unfit()
    return results


def transform_timings(root, data):
    import dask
    import numpy as np
    from fedot.core.operations.operation_parameters import OperationParameters
    from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
    from fedot_ind.integration.fedot.extensions.factories import make_deferred_factory
    from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor
    from benchmark.industrial.experiments.evolution_parallelism.monitoring import ProcessTreeMonitor
    values = data.folds[0][0].features[:32].numpy().copy()
    basis = make_deferred_factory(load_industrial_extension_catalog().operation('wavelet_basis'))(dict(
        wavelet='mexh', n_components=2, low_freq=False, task_type='regression', use_cache=False))
    with dask.config.set(scheduler='synchronous'):
        transformed = basis.transform(values)
    expected = None
    result = {}
    for mode in ('reference', 'optimized'):
        measured = []
        with ExitStack() as stack:
            memory = stack.enter_context(ProcessTreeMonitor(root / f'memory_{mode}', interval=.02))
            if mode == 'reference':
                stack.enter_context(reference_windows())
            extractor = TorchQuantileExtractor(OperationParameters(window_size=10, stride=1, use_sliding_window=True,
                                                                   add_global_features=True, use_cache=False))
            start = perf_counter()
            actual = extractor._transform(SimpleNamespace(features=transformed))
            cold_seconds = perf_counter() - start
            if expected is None:
                expected = actual.copy()
            np.testing.assert_array_equal(actual, expected)
            for _ in range(3):
                start = perf_counter()
                actual = extractor._transform(SimpleNamespace(features=transformed))
                measured.append(perf_counter() - start)
                np.testing.assert_array_equal(actual, expected)
        result[mode] = dict(cold_seconds=cold_seconds, repeats=measured, median_seconds=median(measured),
                            shape=list(actual.shape), sha256=sha256(actual.tobytes()).hexdigest(),
                            memory=memory.summary)
    (root / 'transforms.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(transform_timings=result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--transforms-only', action='store_true')
    args = parser.parse_args()
    cases = fixed_cases(args.source_run)
    if not args.execute:
        print(json.dumps(dict(execute=False, fixed_cases=len(cases), test_used=False)))
        return
    from benchmark.industrial.experiments.mutation_quality.execution.frozen_data import load_frozen_data
    lock = json.loads((args.bundle / 'lock.json').read_text('utf-8'))
    data = load_frozen_data(lock['base']['directory'], lock, 'Tecator')
    if data.test is not None or flags() != lock['execution']['torch_flags']:
        raise ValueError('Unexpected control access or Torch settings')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    transform_timings(args.output_dir, data)
    results = [] if args.transforms_only else replay(args.output_dir, data, cases)
    if flags() != lock['execution']['torch_flags']:
        raise ValueError('Torch settings changed')
    (args.output_dir / 'verified.json').write_text(json.dumps(dict(
        cases=0 if args.transforms_only else len(cases), fits=len(results), feature_matrices_equal=True,
        predictions_equal=None if args.transforms_only else True, test_used=False,
        outer_train_control_used=False, torch_flags=flags()), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
