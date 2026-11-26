"""Evaluate one admitted graph against one or more sealed development folds."""

from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .data import Panel, canonical_bytes, digest
from .recurrence import recurrence_manifest
from .space import NEURAL, Graph
from .training import LabelCodec, Recipe, neural_model_specs


def _forest_factory(params):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=params['n_estimators'], n_jobs=1, random_state=params['random_state'])


def _regression_forest(params):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=params['n_estimators'], n_jobs=1, random_state=params['random_state'])


def _ridge_factory(params):
    from sklearn.linear_model import Ridge

    return Ridge(alpha=params['alpha'])


_ROCKET_MODEL_LIMIT = 6
_ROCKET_FEATURE_LIMIT = 12
_ROCKET_FEATURE_BYTES = 64 * 1024 ** 2
_ROCKET_MODELS = OrderedDict()
_ROCKET_FEATURES = OrderedDict()
_ROCKET_HITS = Counter()


def _rocket_input_key(values):
    return tuple(values.shape), digest(memoryview(values.numpy()))


def _remember_rocket_features(key, features):
    if features.nbytes == 0 or features.nbytes > _ROCKET_FEATURE_BYTES:
        return
    while (len(_ROCKET_FEATURES) >= _ROCKET_FEATURE_LIMIT or features.nbytes
           + sum(value.nbytes for value in _ROCKET_FEATURES.values()) > _ROCKET_FEATURE_BYTES):
        _ROCKET_FEATURES.popitem(last=False)
    _ROCKET_FEATURES[key] = features.copy()


class PilotMiniRocket:
    """Retain one CPU feature map fitted only on the admitted training panel."""

    def __init__(self, params):
        self.seed = params['seed']
        self.scope = params.get('development_sha256')
        self.fit_key = None
        self.model = None

    def fit(self, features):
        import torch
        from fedot_ind.core.models.nn.network_impl.feature_extraction.mini_rocket import MiniRocketFeatures

        features = torch.as_tensor(features, dtype=torch.float32, device='cpu').detach().contiguous()
        if self.scope is not None:
            self.fit_key = (self.scope, self.seed, _rocket_input_key(features))
            previous = _ROCKET_MODELS.get(self.fit_key)
            if previous is not None:
                self.model, random_state = previous
                # The existing feature map fit leaves NumPy in a seeded state.
                np.random.set_state(random_state)
                _ROCKET_MODELS.move_to_end(self.fit_key)
                _ROCKET_HITS['fit'] += 1
                return self
        self.model = MiniRocketFeatures(features.shape[1], features.shape[2], random_state=self.seed).cpu()
        self.model.fit(features)
        if self.fit_key is not None:
            _ROCKET_MODELS[self.fit_key] = (self.model, np.random.get_state())
            if len(_ROCKET_MODELS) > _ROCKET_MODEL_LIMIT:
                _ROCKET_MODELS.popitem(last=False)
        return self

    def transform(self, features):
        import torch

        if self.model is None or not self.model.prefit:
            raise RuntimeError('MiniRocket has no fitted feature map')
        features = torch.as_tensor(features, dtype=torch.float32, device='cpu').detach().contiguous()
        if self.fit_key is not None:
            key = (self.fit_key, _rocket_input_key(features))
            previous = _ROCKET_FEATURES.get(key)
            if previous is not None:
                _ROCKET_FEATURES.move_to_end(key)
                _ROCKET_HITS['transform'] += 1
                return previous.copy()
        with torch.inference_mode():
            values = self.model(features).numpy()
        if self.fit_key is not None:
            _remember_rocket_features(key, values)
        return values


def _manifest(*, regression=False):
    from fedot.core.repository.dataset_types import DataTypesEnum as D
    from fedot.core.repository.tasks import TaskTypesEnum as T
    from fedot.extensions import (
        ExternalModelSpec, ExternalTransformSpec, ModelCapabilities, ModelHyperparamsSchema, TransformCapabilities,
    )

    forest = ExternalModelSpec('pilot_rf', _forest_factory, ModelCapabilities((T.classification,), (D.tabular,)),
                               ModelHyperparamsSchema(required=('n_estimators', 'random_state')))
    rocket = ExternalTransformSpec('pilot_minirocket', PilotMiniRocket,
                                   TransformCapabilities((T.classification, T.regression), (D.ts,), D.tabular),
                                   ModelHyperparamsSchema(required=('seed',), optional=('development_sha256',)))
    if not regression:
        return recurrence_manifest((*neural_model_specs(), forest), (rocket,))
    from .regression import regression_model_specs
    cap = ModelCapabilities((T.regression,), (D.tabular,))
    ridge = ExternalModelSpec('pilot_ridge', _ridge_factory, cap, ModelHyperparamsSchema(required=('alpha',)))
    forest = ExternalModelSpec('pilot_regression_forest', _regression_forest, cap,
                               ModelHyperparamsSchema(required=('n_estimators', 'random_state')))
    return recurrence_manifest((*regression_model_specs(), ridge, forest), (rocket,))


@dataclass(frozen=True)
class ValidationFold:
    fit: Panel
    score: Panel
    epoch_validation: tuple[int, ...]


@dataclass(frozen=True)
class Job:
    graph: Graph
    folds: tuple[ValidationFold, ...]
    classes: tuple[str, ...]
    recipes: dict[str, Recipe]
    device: str
    rocket_scope: str | None = None


def _branch_predictions(branch, fold, codec, recipes, seed, device, rocket_scope=None):
    import torch
    from fedot import create_data
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder

    channels, length = fold.fit.features.shape[1:]
    regression = codec is None
    inputs = partial(create_data, task='regression' if regression else 'classification', data_type='ts',
                     use_cache=False)
    training = inputs(fold.fit.features, target=fold.fit.target if regression else codec.encode(fold.fit.target))
    evaluation = inputs(fold.score.features)
    training = replace(training, idx=torch.as_tensor(fold.fit.rows))
    evaluation = replace(evaluation, idx=torch.as_tensor(fold.score.rows), target=None, state='predict')

    builder = PipelineBuilder()
    for operation in branch.operations:
        params = None
        if operation == 'quantile_extractor_torch':
            params = {'use_cache': False}
        elif operation == 'minirocket_extractor':
            operation, params = 'pilot_minirocket', {'seed': seed}
            if regression and rocket_scope is not None:
                params['development_sha256'] = rocket_scope
        elif operation == 'recurrence_image':
            params = dict(channels=channels, length=length)
        elif operation == 'logreg':
            params = {'C': (1., .1, 10., 100.)[branch.parameter]}
        elif operation in ('pilot_rf', 'pilot_regression_forest'):
            params = {'n_estimators': (100, 32, 64, 128)[branch.parameter], 'random_state': seed}
        elif operation == 'pilot_ridge':
            params = {'alpha': (1., .1, 10., 100.)[branch.parameter]}
        elif operation in NEURAL:
            recipe = recipes[operation]
            params = dict(learning_rate=recipe.learning_rate, batch_size=recipe.batch_size,
                          channels=channels, length=length, device=device,
                          fit_rows=fold.fit.rows.tolist(), epoch_validation=list(fold.epoch_validation), seed=seed,
                          partition_sha256=digest(canonical_bytes(
                              [fold.fit.dataset, 'regression' if regression else codec.classes])),
                          representation_sha256=digest(canonical_bytes(branch.operations)))
            if not regression:
                params['class_count'] = len(codec.classes)
        builder.add_node(operation, params=params)
    pipeline = builder.build()
    try:
        pipeline.fit(training)
        output = pipeline.predict(evaluation, output_mode='default' if regression else 'full_probs')
        predictions = np.asarray(output.predict)
        n = len(fold.score.rows)
        if not np.array_equal(np.asarray(output.idx).reshape(-1), fold.score.rows) or not np.isfinite(predictions).all():
            raise ValueError('Predictions did not preserve finite values and the evaluation object keys')
        if regression:
            if predictions.shape not in ((n,), (n, 1)):
                raise ValueError('Expected one scalar prediction per object')
            predictions = predictions.reshape(-1)
        elif (predictions.shape != (n, len(codec.classes)) or np.any(predictions < 0)
              or not np.allclose(predictions.sum(axis=1), 1, atol=1e-6)):
            raise ValueError('Class probabilities or evaluation object keys changed in the pipeline')
        trace = None
        if branch.head in NEURAL:
            trace = pipeline.root_node.fitted_operation.trace
            if (trace['device'] != device or (trace['optimizer_steps'] == 0 and trace['route'] != 'exact')):
                raise RuntimeError('No evidence of training or exact reuse on the requested device')
        return predictions, trace
    finally:
        pipeline.unfit()


def evaluate(task, job: Job):
    from fedot.extensions import extension_scope
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope

    regression = job.graph.regression
    rocket_hits_before = _ROCKET_HITS.copy() if job.rocket_scope is not None else None
    codec = None if regression else LabelCodec(job.classes)
    scores = []
    with industrial_extension_scope(), extension_scope(_manifest(regression=regression)):
        for fold in job.folds:
            outputs = [_branch_predictions(branch, fold, codec, job.recipes, task.seed, job.device, job.rocket_scope)
                       for branch in job.graph.branches]
            probs = np.mean([prediction for prediction, _ in outputs], axis=0)
            if regression:
                from .regression import score_regression
                scores.append(dict(**score_regression(fold.score, fold.score.rows, probs),
                                   keys=fold.score.rows.tolist(), predictions=probs.tolist(),
                                   training=[trace for _, trace in outputs if trace is not None]))
                continue
            actual, predicted = codec.encode(fold.score.target), probs.argmax(axis=1)
            scores.append(dict(
                balanced_accuracy=float(balanced_accuracy_score(actual, predicted)),
                accuracy=float(accuracy_score(actual, predicted)), f1_macro=float(f1_score(actual, predicted, average='macro')),
                keys=fold.score.rows.tolist(), probabilities=probs.tolist(), labels=np.asarray(job.classes)[predicted].tolist(),
                training=[trace for _, trace in outputs if trace is not None],
            ))
    loss = (np.mean([row['rmse'] for row in scores]) if regression
            else 1 - np.mean([row['balanced_accuracy'] for row in scores]))
    result = dict(loss=loss.item(), folds=scores)
    if rocket_hits_before is not None:
        result['minirocket_reuse'] = dict(_ROCKET_HITS - rocket_hits_before)
    return result
