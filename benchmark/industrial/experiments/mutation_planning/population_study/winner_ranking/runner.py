"""Run 100 fresh native fits on TRAIN, with fixed graphs and locked selections."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from importlib import resources
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .core import admit_candidates, choose_inner, summarize
from .. import nested_validation
from ..data import file_hash, loss_value
from ..forecast_recheck import _diagnostic_snapshot, require_cpu_build
from ..nested_validation.core import make_plan
from ..nested_validation.runner import load_settings as nested_settings, positive_scale
from ..preparation import fitted_model_parameters, pipeline_configuration
from ..protocol import load_protocol
from ..ridge_diagnostics.runner import _numpy, torch_flags, train_only_guard
from ..runner import write_json
from ..supervised_diagnostics import _runtime, _tensor, _train_arrays
from fedot_ind.core.optimizer.evaluation.preparation import canonical_json


@contextmanager
def ranking_data_guard(dataset):
    """Reject raw held-out data and historical mixed TRAIN/TEST record containers."""
    active = [True]

    def audit(event, args):
        if active[0] and event == 'open' and isinstance(args[0], (str, bytes)):
            name = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).name.lower()
            if name == 'records.json' or ('_test.' in name and Path(name).suffix in ('.ts', '.csv', '.tsv', '.arff')):
                raise RuntimeError('Mixed historical records and official TEST access are forbidden')
    sys.addaudithook(audit)
    try:
        with train_only_guard(dataset) as accesses:
            yield accesses
    finally:
        active[0] = False


def load_settings(audit_path):
    raw = json.loads(resources.files(__package__).joinpath('defaults.json').read_text(encoding='utf-8'))
    if raw['schema_version'] != 'industrial_winner_ranking@1' or raw['run_seed'] != 42:
        raise ValueError('Frozen ranking settings differ')
    nested_path = resources.files(nested_validation).joinpath('defaults.json')
    if file_hash(Path(str(nested_path))) != raw['nested_defaults_sha256']:
        raise ValueError('Previously approved split settings changed')
    if file_hash(Path(audit_path)) != raw['audit_sha256']:
        raise ValueError('Require the frozen TRAIN-only candidate-pool audit')
    source = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    choices = admit_candidates(source['catalogue'], raw['candidates'])
    observed_winners = {row['candidate'] for row in source['winners']}
    if observed_winners != set(raw['candidates']):
        raise ValueError('Frozen candidate set must equal the actual observed winners')
    return raw, nested_settings()[0], choices


def build_pipeline(candidate):
    """Reconstruct node references, not independent branch copies or new defaults."""
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    raw = candidate.configuration
    built = {}

    def build(index):
        if index not in built:
            spec = raw['nodes'][index]
            parents = [build(parent) for parent in spec['parents']]
            node = PipelineNode(spec['operation'], nodes_from=parents or None)
            node.parameters = spec['parameters']
            built[index] = node
        return built[index]
    pipeline = Pipeline(build(0))
    if canonical_json(pipeline_configuration(pipeline, 'regression')) != candidate.configuration_json:
        raise ValueError('Native reconstruction changed canonical parameters, parent order or sharing')
    return pipeline


def evaluate_native(candidate, seed, training_features, training_target, validation_features):
    """Neither validation labels nor outer arrays enter the training boundary."""
    from fedot_ind.core.operation.transformation.representation.statistical.feature_reuse import run_with_feature_reuse
    from fedot_ind.core.optimizer.randomness import CandidateRandomness, operator_random_scope
    pipeline = build_pipeline(candidate)
    evaluation_seed = CandidateRandomness(seed, 'evaluation:' + candidate.configuration_id, 0).seed
    try:
        import torch
        # Same fixed run-level Torch seed and configuration-addressed Python/NumPy scope.
        torch.manual_seed(seed)
        with operator_random_scope(evaluation_seed):
            started = perf_counter()
            fitted = run_with_feature_reuse(pipeline.fit, _tensor(training_features, training_target, 'regression'))
            fit_seconds = perf_counter() - started
            started = perf_counter()
            output = run_with_feature_reuse(pipeline.predict, _tensor(validation_features, None, 'regression'))
            prediction = _numpy(output.predict).reshape(-1)
            predict_seconds = perf_counter() - started
        if prediction.shape != (len(validation_features),) or not np.isfinite(prediction).all():
            raise ValueError('Require finite predictions aligned to validation samples')
        np.testing.assert_array_equal(_numpy(fitted.target).reshape(-1), training_target)
        np.testing.assert_array_equal(_numpy(fitted.idx), np.arange(len(training_features)))
        np.testing.assert_array_equal(_numpy(output.idx), np.arange(len(validation_features)))
        if canonical_json(pipeline_configuration(pipeline, 'regression')) != candidate.configuration_json:
            raise ValueError('Native fit/predict changed the frozen configuration')
        diagnostics = {'configuration_id': candidate.configuration_id, 'evaluation_seed': evaluation_seed,
                       'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
                       'training_samples': len(training_target), 'fitted_models': fitted_model_parameters(pipeline),
                       'root_training_features_shape': list(fitted.features.shape),
                       'root_training_features_dtype': str(fitted.features.dtype)}
        if pipeline.root_node.operation.operation_type == 'pdl_reg':
            facade = pipeline.root_node.fitted_operation
            implementation = getattr(facade, 'implementation', facade)
            np.testing.assert_array_equal(implementation.target_, training_target)
            if (len(implementation.train_features_) != len(training_target)
                    or implementation.base_model.get_params()['random_state'] != 42):
                raise ValueError('PDL training subset or frozen base seed changed')
            diagnostics.update(pdl_diagnostics=implementation.get_diagnostics(),
                               anchor_indices=implementation.anchor_indices_.tolist())
        return prediction, diagnostics
    finally:
        pipeline.unfit()


def evaluate_split(choices, seed, features, target, split, context, directory):
    training, validation = np.asarray(split.training), np.asarray(split.validation)
    scale = positive_scale(target[training])
    rows = []
    for choice in choices:
        try:
            prediction, diagnostics = evaluate_native(choice, seed, features[training].copy(), target[training].copy(),
                                                      features[validation].copy())
            row = {**context, 'candidate': choice.name, 'status': 'succeeded', **asdict(split),
                   'normalization_scale': scale, 'training_target': target[training].tolist(),
                   'target': target[validation].tolist(), 'prediction': prediction.tolist(),
                   'loss': loss_value('regression', target[validation], prediction, scale), **diagnostics}
            rows.append(row)
            write_json(directory / f"{context['stage']}_{context.get('inner_fold', 'outer')}_{choice.name}.json", row)
        except Exception as error:
            write_json(directory / 'failure.json', {**context, 'candidate': choice.name, 'status': 'failed',
                                                    'error_type': type(error).__name__, 'message': str(error)})
            raise
    return rows


def run_fold(config, choices, features, target, fold, directory):
    started = perf_counter()
    outer_training = np.asarray(fold.split.training)
    inner_rows = []
    # Only the outer training subset is available to inner training and ranking.
    x, y = features[outer_training].copy(), target[outer_training].copy()
    for ordinal, split in enumerate(fold.inner):
        context = {'outer_fold': fold.ordinal, 'inner_fold': ordinal, 'stage': 'inner'}
        inner_rows.extend(evaluate_split(choices, config['run_seed'], x, y, split, context, directory))
        print(f'  outer={fold.ordinal} inner={ordinal}: 5/5 complete', flush=True)
    names = tuple(choice.name for choice in choices)
    selection = choose_inner(inner_rows, names, len(fold.inner))
    lock = directory / 'selection.json'
    write_json(lock, {'outer_fold': fold.ordinal, 'outer_evaluated': False, 'selection': selection})
    locked_hash = file_hash(lock)
    inner_seconds = perf_counter() - started
    print(f"LOCKED outer={fold.ordinal} winner={selection['candidate']} inner={selection['mean_inner_loss']:.6f}",
          flush=True)
    started = perf_counter()
    outer_rows = evaluate_split(choices, config['run_seed'], features, target, fold.split,
                                {'outer_fold': fold.ordinal, 'stage': 'outer'}, directory)
    if file_hash(lock) != locked_hash:
        raise ValueError('Selection lock changed during outer evaluation')
    median = np.full(len(fold.split.validation), np.median(y))
    baseline = {'outer_fold': fold.ordinal, 'prediction': median.tolist(),
                'loss': loss_value('regression', target[list(fold.split.validation)], median, positive_scale(y))}
    stage = {'outer_fold': fold.ordinal, 'selection_sha256': locked_hash, 'selection': selection,
             'inner_seconds': inner_seconds, 'outer_seconds': perf_counter() - started, 'median': baseline}
    return inner_rows, outer_rows, stage


def validate_rows(inner, outer, features, target, plan, choices):
    by_name = {choice.name: choice for choice in choices}
    from fedot_ind.core.optimizer.randomness import CandidateRandomness
    for row in [*inner, *outer]:
        fold = plan[row['outer_fold']]
        split = fold.inner[row['inner_fold']] if row['stage'] == 'inner' else fold.split
        mapping = np.asarray(fold.split.training) if row['stage'] == 'inner' else np.arange(len(target))
        if tuple(row['training']) != split.training or tuple(row['validation']) != split.validation:
            raise ValueError('Recorded indices differ from the approved split')
        tr, val = mapping[list(split.training)], mapping[list(split.validation)]
        np.testing.assert_array_equal(row['training_target'], target[tr])
        np.testing.assert_array_equal(row['target'], target[val])
        scale = positive_scale(target[tr])
        loss = loss_value('regression', target[val], row['prediction'], scale)
        choice = by_name[row['candidate']]
        if (row['status'] != 'succeeded' or row['configuration_id'] != choice.configuration_id
                or row['training_samples'] != len(tr)
                or row['evaluation_seed'] != CandidateRandomness(42, 'evaluation:' + choice.configuration_id, 0).seed
                or scale != row['normalization_scale']
                or not np.isclose(loss, row['loss'], rtol=1e-12, atol=1e-12)
                or row['fit_seconds'] < 0 or row['predict_seconds'] < 0):
            raise ValueError('Identity, seed, timing, or TRAIN-only loss replay failed')
        if not np.isfinite([row['fit_seconds'], row['predict_seconds']]).all():
            raise ValueError('Timings must be finite')


def run(protocol_path, audit_path, output, *, worktree=None, execute=False):
    config, nested, choices = load_settings(audit_path)
    protocol = load_protocol(protocol_path)
    if protocol.digest != nested.population_protocol_sha256:
        raise ValueError('Population protocol differs from the approved snapshot')
    study = protocol.to_record()
    scenario = {'name': config['dataset'], 'task': 'regression'}
    if scenario not in study['scenarios']:
        raise ValueError('Dataset must belong to the frozen protocol')
    planned = {'purpose': 'conditional_fixed_pool_TRAIN_rank_stability', 'config': config,
               'candidates': [asdict(choice) for choice in choices], 'inner_fits': 75, 'outer_fits': 25,
               'total_native_pipeline_fits': 100, 'official_test_read_or_hashed': False,
               'full_study_accepted': False, 'protocol_sha256': protocol.digest}
    if not execute:
        return planned
    import fedot
    import torch
    from fedot_ind.integration.fedot.compat import audit_compatibility_worktree
    require_cpu_build(torch)
    if worktree is None:
        raise ValueError('Require the explicitly audited FEDOT worktree')
    audit = audit_compatibility_worktree(worktree)
    if not Path(fedot.__file__).resolve().is_relative_to(Path(audit['worktree']) / 'fedot'):
        raise ValueError('Active FEDOT must belong to the audited worktree')
    output = Path(output)
    if output.exists():
        raise ValueError('Require fresh output; no overwrite or silent partial resume')

    def signature():
        paths = [*Path(__file__).parent.glob('*.py'), Path(__file__).with_name('defaults.json'),
                 *Path(nested_validation.__file__).parent.glob('*.*'), Path(protocol_path), Path(audit_path),
                 Path(__file__).parents[1] / 'ridge_diagnostics/runner.py']
        return {**_diagnostic_snapshot(), **{str(p.resolve()): file_hash(p) for p in paths if p.is_file()}}

    implementation = signature()
    inner, outer, stages = [], [], []
    with ranking_data_guard(config['dataset']) as accesses, _runtime():
        if torch_flags() != config['torch_flags']:
            raise ValueError('Frozen Torch flags must be preserved')
        features, target, path = _train_arrays(study, scenario)
        if len(target) != config['samples'] or file_hash(path) != config['train_sha256']:
            raise ValueError('Require the original unchanged NaturalGas TRAIN')
        folds = make_plan(len(target), nested)
        # Check all native graph round trips before the first fit.
        for choice in choices:
            build_pipeline(choice).unfit()
        output.mkdir(parents=True)
        manifest = {
            **planned, 'status': 'frozen', 'source_train': str(path.resolve()),
            'split_plan': [asdict(fold) for fold in folds],
            'implementation_sha256': implementation, 'fedot_compatibility': audit,
            'python_executable': sys.executable,
            'packages': {name: version(name) for name in
                         ('torch', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'dask')},
            'randomness': 'run_seed42_Torch_and_configuration_addressed_evaluation_scope',
            'cost_scope': 'fresh_native_fit_predict_no_search_uncontrolled_host_load',
            'interpretation': 'previously_inspected_TRAIN_fixed_pool_conditional_diagnostic',
        }
        write_json(output / 'manifest.json', manifest)
        for fold in folds:
            if signature() != implementation or torch_flags() != config['torch_flags']:
                raise ValueError('Source or Torch drift before a fold')
            directory = output / f'outer_{fold.ordinal}'
            directory.mkdir()
            print(f'OUTER {fold.ordinal + 1}/5 START', flush=True)
            inside, outside, stage = run_fold(config, choices, features, target, fold, directory)
            inner.extend(inside)
            outer.extend(outside)
            stages.append(stage)
            write_json(output / 'inner_records.json', inner)
            write_json(output / 'outer_records.json', outer)
            write_json(output / 'stages.json', stages)
            print(f'OUTER {fold.ordinal + 1}/5 COMPLETE native_fits={len(inner) + len(outer)}/100', flush=True)
        validate_rows(inner, outer, features, target, folds, choices)
        selections = {row['outer_fold']: row['selection'] for row in stages}
        result = summarize(inner, outer, selections, tuple(choice.name for choice in choices))
        if (signature() != implementation or file_hash(path) != config['train_sha256']
                or torch_flags() != config['torch_flags'] or audit_compatibility_worktree(worktree) != audit):
            raise ValueError('Source, data, dependency or Torch drift during the study')
        for stage in stages:
            if file_hash(output / f"outer_{stage['outer_fold']}/selection.json") != stage['selection_sha256']:
                raise ValueError('Selection lock integrity failed')
        write_json(output / 'summary.json', result)
        completion = {'status': 'complete', 'successful_native_fits': len(inner) + len(outer),
                      'failed_native_fits': 0, 'dataset_accesses': accesses,
                      'torch_flags': torch_flags(), 'official_test_read_or_hashed': False,
                      'source_data_dependency_integrity_verified': True}
        write_json(output / 'completion.json', completion)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compatibility-worktree', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.audit, args.output,
                         worktree=args.compatibility_worktree, execute=args.execute), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
