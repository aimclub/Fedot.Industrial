"""Execute the frozen Ridge grid using original TRAIN only, with native replay."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .analysis import fit_head, finite_matrix, parse_config, summarize
from ..data import file_hash, loss_value
from ..forecast_recheck import _diagnostic_snapshot, require_cpu_build
from ..preparation import configure_pipeline, pipeline_configuration
from ..protocol import load_protocol
from ..runner import write_json
from ..supervised_diagnostics import _pipeline, _runtime, _tensor, _train_arrays


def torch_flags():
    import torch
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'threads': torch.get_num_threads()}


@contextmanager
def train_only_guard(dataset):
    """Audit hooks cannot be removed; the inactive hook is inert after this scope."""
    active, reads = [True], []

    def audit(event, args):
        if not active[0] or event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        if path.suffix.lower() in ('.ts', '.csv', '.tsv') and path.name.lower().startswith(dataset.lower() + '_'):
            if '_test.' in path.name.lower():
                raise RuntimeError('Official TEST access is forbidden in TRAIN diagnostics')
            reads.append(str(path.resolve()))

    sys.addaudithook(audit)
    try:
        yield reads
    finally:
        active[0] = False


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def extract_features(study, structure, seed, features, target, training, validation, config):
    from fedot_ind.core.operation.transformation.representation.statistical.feature_reuse import run_with_feature_reuse
    from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json
    from fedot_ind.core.optimizer.randomness import CandidateRandomness, operator_random_scope
    train = _tensor(features[training], target[training], 'regression')
    predict_train = replace(train, target=None, state='predict')
    predict_validation = _tensor(features[validation], None, 'regression')
    pipeline = configure_pipeline(_pipeline(structure, 'regression', study['operation_parameters']),
                                  study['operation_parameters'])
    identity = PreparedEvaluationGraph(None, canonical_json(pipeline_configuration(pipeline, 'regression')))
    evaluation_seed = CandidateRandomness(seed, 'evaluation:' + identity.configuration_id, 0).seed
    started = perf_counter()
    try:
        with operator_random_scope(evaluation_seed):
            # Version-pinned boundary: fit parents on inner TRAIN and capture the
            # root's training input, then require replay through the public API.
            fit_data = run_with_feature_reuse(pipeline.root_node._get_tensor_data,
                                              tensor_data=train, parent_operation='fit')
            train_features = finite_matrix(_numpy(fit_data.features))
            fit_target = _numpy(fit_data.target).reshape(-1).copy()
            np.testing.assert_array_equal(fit_target, target[training])
            native_fit = run_with_feature_reuse(pipeline.fit, train)
            native_validation = run_with_feature_reuse(pipeline.predict, predict_validation)
            native_prediction = _numpy(native_validation.predict).reshape(-1)
            predicted_train_features = finite_matrix(_numpy(run_with_feature_reuse(
                pipeline.root_node._get_tensor_data, tensor_data=predict_train, parent_operation='predict').features))
            validation_features = finite_matrix(_numpy(run_with_feature_reuse(
                pipeline.root_node._get_tensor_data, tensor_data=predict_validation,
                parent_operation='predict').features))
        actual_training = finite_matrix(_numpy(native_fit.features))
        actual_validation = finite_matrix(_numpy(native_validation.features))
        for observed, expected in ((actual_training, train_features),
                                   (actual_validation, validation_features)):
            if observed.shape != expected.shape or observed.dtype != expected.dtype:
                raise ValueError('Native Ridge input shape or dtype differs from the extracted matrix')
            np.testing.assert_allclose(observed, expected, rtol=config.replay_rtol, atol=config.replay_atol)
        np.testing.assert_array_equal(_numpy(native_fit.idx), _numpy(fit_data.idx))
        np.testing.assert_array_equal(_numpy(native_validation.idx), np.arange(len(validation)))
        np.testing.assert_array_equal(_numpy(native_fit.target).reshape(-1), fit_target)
        np.testing.assert_allclose(predicted_train_features, train_features,
                                   rtol=config.replay_rtol, atol=config.replay_atol)
        reference = fit_head(actual_training, fit_target, actual_validation,
                             scaling='none', alpha=config.reference_alpha)
        np.testing.assert_allclose(reference['validation_prediction'], native_prediction,
                                   rtol=config.replay_rtol, atol=config.replay_atol)
        np.testing.assert_allclose(reference['train_prediction'], _numpy(native_fit.predict).reshape(-1),
                                   rtol=config.replay_rtol, atol=config.replay_atol)
        facade = pipeline.root_node.fitted_operation
        model = getattr(facade, 'implementation', facade)
        if model.get_params(deep=False) != reference['model_parameters']:
            raise ValueError('Native and standalone Ridge parameters differ')
        return actual_training, actual_validation, fit_target, {
            'configuration_id': identity.configuration_id, 'configuration': json.loads(identity.configuration_json),
            'evaluation_seed': evaluation_seed, 'native_model_parameters': model.get_params(deep=False),
            'native_feature_dtype': str(train_features.dtype), 'native_target_dtype': str(fit_target.dtype),
            'native_training_input_max_abs_difference': float(np.abs(actual_training - train_features).max()),
            'native_validation_input_max_abs_difference': float(np.abs(actual_validation - validation_features).max()),
            'native_validation_prediction': native_prediction.tolist(),
            'native_train_prediction': _numpy(native_fit.predict).reshape(-1).tolist(),
            'replay_max_abs_error': float(np.max(np.abs(np.asarray(reference['validation_prediction'])
                                                        - native_prediction))),
            'fit_predict_feature_max_abs_difference': float(np.abs(predicted_train_features - train_features).max()),
            'extraction_and_replay_seconds': perf_counter() - started}
    finally:
        pipeline.unfit()


def run(protocol_path, output, *, config_path=None, worktree=None, execute=False):
    config_path = Path(config_path) if config_path else Path(__file__).with_name('defaults.json')
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    config = parse_config(raw)
    protocol = load_protocol(protocol_path)
    study = protocol.to_record()
    scenario = {'name': config.dataset, 'task': 'regression'}
    if scenario not in study['scenarios']:
        raise ValueError('Dataset must belong to the frozen study')
    seeds = config.primary_seeds + config.additional_seeds
    plan = {'purpose': 'TRAIN_only_fixed_structure_grid_not_full_acceptance', 'config': raw,
            'protocol_sha256': protocol.digest, 'ridge_fits': len(seeds) * len(config.structures)
            * len(config.scalings) * len(config.alphas), 'native_reference_fits': len(seeds) * len(config.structures),
            'standalone_replay_fits': len(seeds) * len(config.structures),
            'total_ridge_fits': len(seeds) * len(config.structures) *
            (len(config.scalings) * len(config.alphas) + 2),
            'official_test_read_or_hashed': False, 'full_study_accepted': False}
    if not execute:
        return plan
    import fedot
    import torch
    from sklearn.model_selection import train_test_split
    from fedot_ind.integration.fedot.compat import audit_compatibility_worktree
    require_cpu_build(torch)
    if worktree is None:
        raise ValueError('Require the explicitly audited FEDOT worktree')
    audit = audit_compatibility_worktree(worktree)
    if not Path(fedot.__file__).resolve().is_relative_to(Path(audit['worktree']) / 'fedot'):
        raise ValueError('Active FEDOT must belong to the audited worktree')
    output = Path(output)
    if output.exists():
        raise ValueError('Require a fresh output directory')

    def signature():
        return {**_diagnostic_snapshot(), **{str(path.resolve()): file_hash(path) for path in
                (*Path(__file__).parent.glob('*.py'), config_path, Path(protocol_path))}}

    implementation = signature()
    packages = {name: version(name) for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'dask')}
    rows, folds = [], []
    with train_only_guard(config.dataset) as accesses, _runtime():
        flags = torch_flags()
        expected_flags = {'deterministic_algorithms': False, 'deterministic_warn_only': False,
                          'float32_matmul_precision': 'highest', 'cudnn_deterministic': False,
                          'cudnn_benchmark': False, 'threads': 1}
        if flags != expected_flags:
            raise ValueError(f'Frozen full-series Torch flags differ: {flags}')
        features, target, path = _train_arrays(study, scenario)
        digest = file_hash(path)
        output.mkdir(parents=True)
        manifest = {**plan, 'implementation_sha256': implementation, 'fedot_compatibility': audit,
                    'source_train': str(path.resolve()), 'source_train_sha256': digest,
                    'source_samples': len(target), 'python_executable': sys.executable,
                    'packages': packages, 'torch_flags': flags,
                    'reference_dtype': 'preserve_actual_native_feature_and_target_dtypes_per_fold',
                    'scaler_fit_scope': 'inner_TRAIN_only', 'status': 'frozen',
                    'selection_scope': 'fixed_two_Ridge_structures_not_PDL_or_full_search'}
        write_json(output / 'manifest.json', manifest)
        for seed in seeds:
            if signature() != implementation or file_hash(path) != digest or torch_flags() != flags:
                raise ValueError('Sources or Torch flags changed during diagnostics')
            # Match the full worker's job seed; no Torch backend flag is changed.
            torch.manual_seed(seed)
            training, validation = train_test_split(np.arange(len(target)), test_size=config.validation_fraction,
                                                    random_state=seed)
            scale = float(np.std(target[training]))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Require positive inner-TRAIN target standard deviation')
            median_prediction = np.full(len(validation), np.median(target[training]))
            for structure in config.structures:
                print(f'TRAIN seed={seed} structure={structure}', flush=True)
                x_train, x_validation, y_train, evidence = extract_features(
                    study, structure, seed, features, target, training, validation, config)
                # Local replay matrices are generated artifacts, never new raw inputs.
                np.savez_compressed(output / f'features_{seed}_{structure}.npz',
                                    training=x_train, validation=x_validation, target=y_train)
                fold = {'seed': seed, 'structure': structure, 'training_indices': training.tolist(),
                        'validation_indices': validation.tolist(), 'normalization_scale': scale,
                        'training_target': y_train.tolist(), 'validation_target': target[validation].tolist(),
                        'median_loss': loss_value('regression', target[validation], median_prediction, scale),
                        **evidence}
                folds.append(fold)
                for scaling in config.scalings:
                    for alpha in config.alphas:
                        started = perf_counter()
                        record = fit_head(x_train, y_train, x_validation, scaling=scaling, alpha=alpha)
                        rows.append({'seed': seed, 'structure': structure, 'scaling': scaling, 'alpha': alpha,
                                     'fit_and_diagnostic_seconds': perf_counter() - started,
                                     'train_loss': loss_value('regression', y_train,
                                                              record['train_prediction'], scale),
                                     'validation_loss': loss_value('regression', target[validation],
                                                                   record['validation_prediction'], scale), **record})
                write_json(output / 'records.json', rows)
                write_json(output / 'folds.json', folds)
                print(f'COMPLETED {len(rows)}/{plan["ridge_fits"]} Ridge fits', flush=True)
        if (signature() != implementation or file_hash(path) != digest or torch_flags() != flags
                or audit_compatibility_worktree(worktree) != audit):
            raise ValueError('Source, dependency or Torch drift during the study')
        summary = summarize(rows, config)
        write_json(output / 'summary.json', summary)
        write_json(output / 'completion.json', {'status': 'complete', 'ridge_fits': len(rows),
                                                'native_reference_fits': len(folds), 'torch_flags': torch_flags(),
                                                'standalone_replay_fits': len(folds),
                                                'total_ridge_fits': len(rows) + 2 * len(folds),
                                                'dataset_accesses': accesses, 'official_test_read_or_hashed': False})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--compatibility-worktree', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = run(args.protocol, args.output, config_path=args.config, worktree=args.compatibility_worktree,
                 execute=args.execute)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
