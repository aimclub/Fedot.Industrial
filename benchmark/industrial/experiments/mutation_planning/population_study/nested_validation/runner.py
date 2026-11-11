"""Execute nested TRAIN selection, lock decisions, then evaluate outer predictions."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib import resources
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .core import InnerScore, candidates, make_plan, parse_config, select_candidate, summarize_outer
from .. import ridge_diagnostics
from ..data import file_hash, loss_value
from ..forecast_recheck import _diagnostic_snapshot, require_cpu_build
from ..preparation import configure_pipeline, fitted_model_parameters, pipeline_configuration
from ..protocol import canonical, load_protocol
from ..ridge_diagnostics.analysis import fit_head, parse_config as parse_grid
from ..ridge_diagnostics.runner import _numpy, extract_features, torch_flags, train_only_guard
from ..runner import write_json
from ..supervised_diagnostics import _runtime, _tensor, _train_arrays


def load_settings(config_path=None):
    approved = json.loads(resources.files(__package__).joinpath('defaults.json').read_text(encoding='utf-8'))
    raw = approved if config_path is None else json.loads(Path(config_path).read_text(encoding='utf-8'))
    config = parse_config(raw)
    if canonical(raw) != canonical(approved):
        raise ValueError('Nested configuration differs from the approved bundled protocol')
    grid_raw = json.loads(resources.files(ridge_diagnostics).joinpath('defaults.json').read_text(encoding='utf-8'))
    if sha256(canonical(grid_raw).encode()).hexdigest() != config.ridge_grid_sha256:
        raise ValueError('Ridge grid differs from the approved protocol')
    return config, parse_grid(grid_raw), raw, grid_raw


def positive_scale(target):
    value = float(np.std(target))
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Require a positive standard deviation of training targets only')
    return value


def tune_inner(study, config, grid, training_features, training_target, splits, directory):
    """The interface receives only outer-training arrays and local inner indices."""
    rows, evidence, scores = [], [], []
    for ordinal, split in enumerate(splits):
        tr, val = np.asarray(split.training), np.asarray(split.validation)
        scale = positive_scale(training_target[tr])
        for structure in grid.structures:
            x, xv, y, native = extract_features(study, structure, config.feature_seed,
                                                training_features, training_target, tr, val, grid)
            np.savez_compressed(directory / f'inner_{ordinal}_{structure}.npz', training=x, validation=xv, target=y)
            evidence.append({'inner_fold': ordinal, 'structure': structure, **asdict(split),
                             'normalization_scale': scale, 'training_target': y.tolist(),
                             'validation_target': training_target[val].tolist(), **native})
            for choice in candidates(grid):
                if choice.structure != structure:
                    continue
                started = perf_counter()
                result = fit_head(x, y, xv, scaling=choice.scaling, alpha=choice.alpha)
                loss = loss_value('regression', training_target[val], result['validation_prediction'], scale)
                rows.append({'inner_fold': ordinal, **asdict(choice), 'validation_loss': loss,
                             'head_fit_predict_diagnostics_seconds': perf_counter() - started, **result})
                scores.append(InnerScore(ordinal, choice, loss))
            write_json(directory / 'inner_records.json', rows)
            write_json(directory / 'inner_features.json', evidence)
        print(f'  inner {ordinal + 1}/{len(splits)} complete: {len(rows)} configurations', flush=True)
    return select_candidate(tuple(scores), grid, len(splits))


def predict_pdl(study, config, features, target, training, validation):
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot_ind.core.operation.transformation.representation.statistical.feature_reuse import run_with_feature_reuse
    from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json
    from fedot_ind.core.optimizer.randomness import CandidateRandomness, operator_random_scope
    builder = PipelineBuilder()
    for operation in ('quantile_extractor', 'pdl_reg'):
        builder.add_node(operation, params=study['operation_parameters'][operation])
    pipeline = configure_pipeline(builder.build(), study['operation_parameters'])
    identity = PreparedEvaluationGraph(None, canonical_json(pipeline_configuration(pipeline, 'regression')))
    seed = CandidateRandomness(config.feature_seed, 'evaluation:' + identity.configuration_id, 0).seed
    started = perf_counter()
    try:
        with operator_random_scope(seed):
            run_with_feature_reuse(pipeline.fit, _tensor(features[training], target[training], 'regression'))
            prediction = _numpy(run_with_feature_reuse(
                pipeline.predict, _tensor(features[validation], None, 'regression')).predict).reshape(-1)
        facade = pipeline.root_node.fitted_operation
        implementation = getattr(facade, 'implementation', facade)
        np.testing.assert_array_equal(implementation.target_, target[training])
        expected_seed = study['operation_parameters']['pdl_reg']['random_state']
        if implementation.base_model.get_params()['random_state'] != expected_seed:
            raise ValueError('PDL base-model seed differs from the frozen protocol')
        return prediction, {'configuration_id': identity.configuration_id, 'evaluation_seed': seed,
                            'configuration': json.loads(identity.configuration_json),
                            'fitted_models': fitted_model_parameters(pipeline),
                            'pdl_diagnostics': implementation.get_diagnostics(),
                            'training_sample_count': len(implementation.train_features_),
                            'anchor_indices': implementation.anchor_indices_.tolist(),
                            'fit_predict_seconds': perf_counter() - started}
    finally:
        pipeline.unfit()


def evaluate_outer(study, config, grid, features, target, fold, selection, directory):
    """Called only after selection.json is written; outer targets cannot tune it."""
    training, validation = np.asarray(fold.split.training), np.asarray(fold.split.validation)
    scale = positive_scale(target[training])
    rows, evidence = [], []

    def add(method, prediction, **diagnostics):
        prediction = np.asarray(prediction).reshape(-1)
        rows.append({'outer_fold': fold.ordinal, 'method': method,
                     'training_indices': training.tolist(), 'validation_indices': validation.tolist(),
                     'normalization_scale': scale, 'target': target[validation].tolist(),
                     'prediction': prediction.tolist(),
                     'outer_loss': loss_value('regression', target[validation], prediction, scale), **diagnostics})

    add('median', np.full(len(validation), np.median(target[training])))
    for structure in grid.structures:
        x, xv, y, native = extract_features(study, structure, config.feature_seed,
                                            features, target, training, validation, grid)
        np.savez_compressed(directory / f'outer_{structure}.npz', training=x, validation=xv, target=y)
        evidence.append({'structure': structure, **native})
        started = perf_counter()
        fixed = fit_head(x, y, xv, scaling='none', alpha=grid.reference_alpha)
        add(f'{structure}_alpha1', fixed['validation_prediction'], fit_diagnostics=fixed,
            head_fit_predict_diagnostics_seconds=perf_counter() - started)
        if structure == selection.candidate.structure:
            started = perf_counter()
            result = fit_head(x, y, xv, scaling=selection.candidate.scaling, alpha=selection.candidate.alpha)
            add('selected', result['validation_prediction'], selected_configuration=asdict(selection.candidate),
                mean_inner_loss=selection.mean_inner_loss, fit_diagnostics=result,
                head_fit_predict_diagnostics_seconds=perf_counter() - started)
    prediction, pdl = predict_pdl(study, config, features, target, training, validation)
    add('q_pdl', prediction, **pdl)
    write_json(directory / 'outer_features.json', evidence)
    write_json(directory / 'outer_records.json', rows)
    return rows


def run_fold(study, config, grid, features, target, fold, directory):
    started = perf_counter()
    tr = np.asarray(fold.split.training)
    selection = tune_inner(study, config, grid, features[tr].copy(), target[tr].copy(), fold.inner, directory)
    inner_seconds = perf_counter() - started
    locked = {'outer_fold': fold.ordinal, **asdict(selection), 'inner_selection_seconds': inner_seconds,
              'outer_evaluated': False}
    path = directory / 'selection.json'
    write_json(path, locked)
    locked_hash = file_hash(path)
    print(f'LOCKED outer={fold.ordinal} {asdict(selection.candidate)} '
          f'inner={selection.mean_inner_loss:.6f}', flush=True)
    started = perf_counter()
    outer = evaluate_outer(study, config, grid, features, target, fold, selection, directory)
    if file_hash(path) != locked_hash:
        raise ValueError('The selected configuration changed after outer evaluation')
    return outer, {'outer_fold': fold.ordinal, 'selection_sha256': locked_hash,
                   'inner_selection_seconds': inner_seconds, 'outer_stage_seconds': perf_counter() - started,
                   'selected': asdict(selection.candidate), 'mean_inner_loss': selection.mean_inner_loss}


def run(protocol_path, output, *, config_path=None, worktree=None, execute=False):
    config, grid, raw, grid_raw = load_settings(config_path)
    protocol = load_protocol(protocol_path)
    if protocol.digest != config.population_protocol_sha256:
        raise ValueError('Population protocol differs from the approved snapshot')
    study = protocol.to_record()
    scenario = {'name': grid.dataset, 'task': 'regression'}
    if scenario not in study['scenarios']:
        raise ValueError('NaturalGas must belong to the frozen full-study protocol')
    pairs = config.outer_folds * (config.inner_folds + 1) * len(grid.structures)
    grid_fits = config.outer_folds * config.inner_folds * len(candidates(grid))
    plan = {'purpose': 'nested_TRAIN_procedure_evaluation_not_EVO04_acceptance',
            'config': raw, 'grid': grid_raw, 'protocol_sha256': protocol.digest,
            'inner_grid_fits': grid_fits, 'native_reference_fits': pairs, 'standalone_replay_fits': pairs,
            'outer_ridge_fits': config.outer_folds * (len(grid.structures) + 1),
            'total_ridge_fits': grid_fits + 2 * pairs + config.outer_folds * (len(grid.structures) + 1),
            'pdl_fits': config.outer_folds, 'official_test_read_or_hashed': False, 'full_study_accepted': False}
    if not execute:
        return plan
    import fedot
    import torch
    from fedot_ind.integration.fedot.compat import audit_compatibility_worktree
    require_cpu_build(torch)
    if worktree is None:
        raise ValueError('Require an explicitly audited FEDOT worktree')
    audit = audit_compatibility_worktree(worktree)
    if not Path(fedot.__file__).resolve().is_relative_to(Path(audit['worktree']) / 'fedot'):
        raise ValueError('Active FEDOT must belong to the audited worktree')
    output = Path(output)
    if output.exists():
        raise ValueError('Require a fresh output directory')

    def signature():
        paths = [*Path(__file__).parent.glob('*.py'), Path(__file__).with_name('defaults.json'),
                 *Path(ridge_diagnostics.__file__).parent.glob('*.py'),
                 Path(ridge_diagnostics.__file__).with_name('defaults.json'), Path(protocol_path)]
        if config_path is not None:
            paths.append(Path(config_path))
        return {**_diagnostic_snapshot(), **{str(path.resolve()): file_hash(path) for path in paths}}

    implementation = signature()
    rows, stages = [], []
    with train_only_guard(grid.dataset) as accesses, _runtime():
        flags = torch_flags()
        if flags != {'deterministic_algorithms': False, 'deterministic_warn_only': False,
                     'float32_matmul_precision': 'highest', 'cudnn_deterministic': False,
                     'cudnn_benchmark': False, 'threads': 1}:
            raise ValueError('Frozen full-series Torch flags must be preserved')
        features, target, path = _train_arrays(study, scenario)
        digest = file_hash(path)
        folds = make_plan(len(target), config)
        output.mkdir(parents=True)
        manifest = {**plan, 'status': 'frozen', 'source_train': str(path.resolve()), 'source_train_sha256': digest,
                    'source_samples': len(target), 'split_plan': [asdict(fold) for fold in folds],
                    'implementation_sha256': implementation, 'fedot_compatibility': audit,
                    'python_executable': sys.executable, 'torch_flags': flags,
                    'packages': {name: version(name) for name in
                                 ('torch', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'dask')},
                    'inner_scale': 'std_inner_training_target_ddof0', 'outer_scale': 'std_outer_training_target_ddof0',
                    'inner_aggregation': 'equal_weight_mean_of_three_normalized_RMSEs',
                    'selection_lock': 'persist_before_outer_evaluation_no_feedback',
                    'cost_scope': 'diagnostic_wall_time_including_replay_not_production_search_cost'}
        write_json(output / 'manifest.json', manifest)
        for fold in folds:
            if signature() != implementation or file_hash(path) != digest or torch_flags() != flags:
                raise ValueError('Code, data or Torch flags changed during the nested study')
            torch.manual_seed(config.feature_seed)
            directory = output / f'outer_{fold.ordinal}'
            directory.mkdir()
            print(f'OUTER {fold.ordinal + 1}/{len(folds)}: inner selection begins', flush=True)
            predictions, timing = run_fold(study, config, grid, features, target, fold, directory)
            rows.extend(predictions)
            stages.append(timing)
            write_json(output / 'outer_records.json', rows)
            write_json(output / 'stages.json', stages)
            print(f'OUTER {fold.ordinal + 1}/{len(folds)} complete', flush=True)
        if (signature() != implementation or file_hash(path) != digest or torch_flags() != flags
                or audit_compatibility_worktree(worktree) != audit):
            raise ValueError('Code, dependency, data or Torch drift during the nested study')
        summary = summarize_outer(rows, folds)
        write_json(output / 'summary.json', summary)
        write_json(output / 'completion.json', {'status': 'complete', 'outer_folds': len(folds),
                                                'outer_predictions': len(rows), 'torch_flags': flags,
                                                'dataset_accesses': accesses,
                                                **{name: plan[name] for name in ('inner_grid_fits', 'total_ridge_fits',
                                                                                 'pdl_fits', 'native_reference_fits',
                                                                                 'standalone_replay_fits',
                                                                                 'outer_ridge_fits')},
                                                'official_test_read_or_hashed': False})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--compatibility-worktree', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.output, config_path=args.config,
                         worktree=args.compatibility_worktree, execute=args.execute), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
