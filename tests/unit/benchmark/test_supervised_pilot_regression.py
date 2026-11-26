from dataclasses import replace
import json

import numpy as np
import pytest

from benchmark.industrial.experiments.supervised_pilot.data import Panel, digest
from benchmark.industrial.experiments.supervised_pilot.protocol import TSER_VERSION
from benchmark.industrial.experiments.supervised_pilot.regression import (
    Sampling, SamplingEvidence, exact_series_evidence, observed_temporal_panel, regression_model_specs,
    regression_splits, score_regression,
)
from benchmark.industrial.experiments.supervised_pilot.recurrence import recurrence_manifest
from benchmark.industrial.protocols.schema import ProtocolError
from benchmark.industrial.protocols.assets import LocalAsset


def test_group_members_never_cross_a_regression_split():
    rows = np.arange(60)
    panel = Panel('Synthetic', rows, np.zeros((60, 1, 8)), np.arange(60, dtype=float))
    groups = tuple(str(i // 3) for i in rows)
    evidence = SamplingEvidence(Sampling.GROUPED, 'synthetic subjects', tuple(rows), groups)
    split = regression_splits(panel, evidence)
    pairs = [(split.development, split.assessment), (split.final_epoch_fit, split.final_epoch_validation)]
    pairs.extend((fold.fit, fold.score) for fold in split.inner)
    pairs.extend((fold.epoch_fit, fold.epoch_validation) for fold in split.inner)
    for fit, score in pairs:
        assert not {groups[i] for i in fit} & {groups[i] for i in score}


def test_exact_groups_ignore_target_and_normalize_signed_zero():
    rows = np.arange(60)
    features = np.stack((rows // 2, np.zeros(60)), axis=1)[:, None, :]
    features[::2, 0, 1] = -0.0
    panel = Panel('Synthetic', rows, features, rows.astype(float))
    evidence = exact_series_evidence(panel, 'published sample design')
    assert evidence == exact_series_evidence(replace(panel, target=panel.target[::-1]), evidence.source)
    assert len(set(evidence.groups)) == 30
    assert evidence.groups[::2] == evidence.groups[1::2]
    split = regression_splits(panel, evidence)
    assert not {evidence.groups[i] for i in split.development} & {evidence.groups[i] for i in split.assessment}


def test_temporal_splits_skip_the_overlap_and_reject_a_reordered_panel():
    rows = tuple(range(160))
    panel = Panel('Synthetic', np.asarray(rows), np.zeros((160, 1, 10)), np.arange(160, dtype=float))
    evidence = SamplingEvidence(Sampling.TEMPORAL, 'dated samples', rows, times=rows, gap=2)
    split = regression_splits(panel, evidence)
    assert min(split.assessment) - max(split.development) == 3
    for fold in split.inner:
        assert min(fold.score) - max(fold.fit) >= 3
        assert min(fold.epoch_validation) - max(fold.epoch_fit) >= 3
    with pytest.raises(ProtocolError, match='does not match'):
        regression_splits(replace(panel, rows=panel.rows[::-1]), evidence)


def test_calendar_preflight_restores_source_time_without_sorting_object_keys(tmp_path):
    header = '@problemName Sensor\n@timestamps true\n@univariate true\n@targetlabel true\n@data\n'
    rows = ('(2020-01-03 08:00:00,1),(2020-01-03 08:10:00,2):2',
            '(2020-01-02 08:00:00,1),(2020-01-02 08:10:00,2):1')
    name = 'Sensor_TRAIN.ts'

    def load(cases):
        payload = (header + '\n'.join(cases)).encode()
        (tmp_path / name).write_bytes(payload)
        return observed_temporal_panel(tmp_path, 'Sensor', LocalAsset(name, digest(payload)))

    panel, evidence = load(rows)
    assert tuple(panel.rows) == (0, 1) and evidence.times[0] > evidence.times[1]
    with pytest.raises(ProtocolError, match='Overlapping'):
        load((rows[0], rows[0]))


def test_seal_requires_all_sampling_evidence_before_opening_train(tmp_path, monkeypatch):
    from benchmark.industrial.experiments.supervised_pilot import regression_inputs

    settings = tmp_path / 'evidence.json'
    settings.write_text(json.dumps({'version': TSER_VERSION, 'datasets': {}}), encoding='utf-8')

    def forbidden(*_args):
        pytest.fail('A partial data catalog must be rejected before reading TRAIN sources')

    monkeypatch.setattr(regression_inputs, 'load_regression_train', forbidden)
    with pytest.raises(ProtocolError, match='partial protocol'):
        regression_inputs.seal_regression(tmp_path, tmp_path / 'sealed', settings)
    assert not (tmp_path / 'sealed').exists()


def test_seal_rejects_changed_frozen_split(tmp_path, monkeypatch):
    from benchmark.industrial.experiments.supervised_pilot import regression_inputs

    profile = regression_inputs.tser_profile()
    settings = tmp_path / 'evidence.json'
    paths = {row['name']: {'path': f"{row['name']}_TRAIN.ts"} for row in profile['datasets']}
    settings.write_text(json.dumps({'version': TSER_VERSION, 'datasets': paths}), encoding='utf-8')
    rows = np.arange(60)
    synthetic = Panel('Tecator', rows, rows[:, None, None], rows.astype(float))
    monkeypatch.setattr(regression_inputs, 'version', lambda _: profile['split_profile']['sklearn'])
    monkeypatch.setattr(regression_inputs, 'load_regression_train', lambda *_: synthetic)
    with pytest.raises(ProtocolError, match='split does not match'):
        regression_inputs.seal_regression(tmp_path, tmp_path / 'sealed', settings)
    assert not (tmp_path / 'sealed').exists()


def test_tser_admission_requires_every_dataset_and_cuda_trace(tmp_path):
    from benchmark.industrial.experiments.supervised_pilot.__main__ import _require_admission
    from benchmark.industrial.experiments.supervised_pilot.data import canonical_bytes

    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes(payload))

    bundle = tmp_path / 'bundle'
    admission = tmp_path / 'admission'
    seal = dict(protocol=TSER_VERSION, datasets=[dict(dataset=name) for name in ('First', 'Second')])
    write(bundle / 'seal.json', seal)
    write(admission / 'run.json', dict(seal_sha256=digest(canonical_bytes(seal))))
    status = admission / 'admit_status.json'
    write(status, [dict(dataset='First', exit_code=0)])
    with pytest.raises(ValueError, match='every required dataset'):
        _require_admission(bundle, admission)
    write(status, [dict(dataset=name, exit_code=0) for name in ('First', 'Second')])
    for name, device in [('First', 'cuda:0'), ('Second', 'cpu')]:
        result = dict(status='succeeded', value=dict(folds=[dict(training=[dict(device=device)])]))
        write(admission / name / 'admit' / 'cuda_admission.json', dict(results=[result] * 3))
    with pytest.raises(ValueError, match='Second'):
        _require_admission(bundle, admission)


def test_calendar_recovery_does_not_guess_unmatched_or_overlapping_objects():
    from benchmark.industrial.experiments.supervised_pilot.calendar_evidence import recover_calendar, require_unique_calendar

    index = np.arange(100)
    prices = index.astype(float) ** 2
    cases = np.stack((prices[50:70], prices[5:25]))[:, None, :]
    assert require_unique_calendar(recover_calendar(cases, index, prices)) == (50, 5)
    wrong = cases.copy()
    wrong[0, 0, -1] += .5
    with pytest.raises(ProtocolError, match='one dated window exactly'):
        require_unique_calendar(recover_calendar(wrong, index, prices))
    overlap = np.stack((prices[50:70], prices[55:75]))[:, None, :]
    with pytest.raises(ProtocolError, match='overlap'):
        require_unique_calendar(recover_calendar(overlap, index, prices))


def test_regression_score_rejects_broadcast_and_permutation():
    panel = Panel('Synthetic', np.array([3, 5, 9]), np.zeros((3, 1, 8)), np.full(3, 4.))
    assert score_regression(panel, [3, 5, 9], [3, 4, 5]) == dict(rmse=np.sqrt(2 / 3), mae=2 / 3, r2=None)
    with pytest.raises(ProtocolError, match='finite vector'):
        score_regression(panel, [3, 5, 9], [[3], [4], [5]])
    with pytest.raises(ProtocolError, match='object order'):
        score_regression(panel, [3, 9, 5], [3, 4, 5])


def test_regression_minirocket_reuses_only_exact_fit_and_transform_inputs(monkeypatch):
    from collections import Counter, OrderedDict
    import torch
    from benchmark.industrial.experiments.supervised_pilot import runtime
    from fedot_ind.core.models.nn.network_impl.feature_extraction.mini_rocket import MiniRocketFeatures

    monkeypatch.setattr(runtime, '_ROCKET_MODELS', OrderedDict())
    monkeypatch.setattr(runtime, '_ROCKET_FEATURES', OrderedDict())
    monkeypatch.setattr(runtime, '_ROCKET_HITS', Counter())
    features = np.random.default_rng(3).normal(size=(18, 1, 32)).astype(np.float32)
    train, score = features[:12], features[12:]
    calls = Counter()
    old_fit, old_forward = MiniRocketFeatures.fit, MiniRocketFeatures.forward

    def counted_fit(model, *args):
        calls['fit'] += 1
        return old_fit(model, *args)

    def counted_forward(model, *args):
        calls['forward'] += 1
        return old_forward(model, *args)

    monkeypatch.setattr(MiniRocketFeatures, 'fit', counted_fit)
    monkeypatch.setattr(MiniRocketFeatures, 'forward', counted_forward)
    params = dict(seed=42, development_sha256='one admitted development receipt')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first = runtime.PilotMiniRocket(params).fit(train)
        cold = first.transform(score)
        expected = cold.copy()
        cold[:] = -1
        second = runtime.PilotMiniRocket(params).fit(train.copy())
        np.testing.assert_array_equal(second.transform(score.copy()), expected)
        assert calls == {'fit': 1, 'forward': 2}
        assert runtime._ROCKET_HITS == {'fit': 1, 'transform': 1}

        uncached = runtime.PilotMiniRocket({'seed': 42}).fit(train)
        np.testing.assert_array_equal(uncached.transform(score), expected)
        second.transform(score[::-1].copy())
        runtime.PilotMiniRocket(params).fit(train[::-1].copy())
        runtime.PilotMiniRocket({**params, 'seed': 123}).fit(train)
        runtime.PilotMiniRocket({**params, 'development_sha256': 'different receipt'}).fit(train)
        assert calls == {'fit': 5, 'forward': 8}
        assert runtime._ROCKET_HITS == {'fit': 1, 'transform': 1}
    finally:
        torch.set_num_threads(threads)


def test_regression_minirocket_reuse_does_not_change_ridge_predictions():
    import torch
    from benchmark.industrial.experiments.supervised_pilot.runtime import Job, ValidationFold, evaluate
    from benchmark.industrial.experiments.supervised_pilot.space import initial
    from fedot_ind.core.optimizer.evaluation.contracts import EvaluationTask

    features = np.random.default_rng(6).normal(size=(20, 1, 32))
    panel = Panel('Synthetic', np.arange(len(features)), features, features.mean(axis=(1, 2)))
    fold = ValidationFold(panel.take(tuple(range(16))), panel.take((16, 17, 18, 19)), (12, 13, 14, 15))
    mini_rocket = initial(regression=True)[3]
    job = Job(mini_rocket, (fold,), (), {}, 'cpu')
    task = EvaluationTask(mini_rocket.key, 11)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reference = evaluate(task, job)
        scoped_job = replace(job, reuse_scope='unique testing receipt')
        evaluate(task, scoped_job)
        repeated = evaluate(task, scoped_job)
    finally:
        torch.set_num_threads(threads)
    assert repeated['folds'] == reference['folds']
    assert repeated['loss'] == reference['loss']
    assert repeated['minirocket_reuse'] == {'fit': 1, 'transform': 2}


def test_classical_prediction_reuse_is_bound_to_the_fit_and_score_inputs(monkeypatch):
    from collections import OrderedDict
    import torch
    from benchmark.industrial.experiments.supervised_pilot import runtime
    from benchmark.industrial.experiments.supervised_pilot.space import initial
    from fedot_ind.core.optimizer.evaluation.contracts import EvaluationTask

    monkeypatch.setattr(runtime, '_CLASSICAL_PREDICTIONS', OrderedDict())
    features = np.random.default_rng(17).normal(size=(24, 1, 48))
    panel = Panel('Synthetic', np.arange(len(features)), features, features.mean(axis=(1, 2)))
    fold = runtime.ValidationFold(panel.take(tuple(range(20))), panel.take((20, 21, 22, 23)), (16, 17, 18, 19))
    graph = initial(regression=True)[0]
    job = runtime.Job(graph, (fold,), (), {}, 'cpu', 'sealed development role')
    task = EvaluationTask(graph.key, 13)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reference = runtime.evaluate(task, replace(job, reuse_scope=None))
        first = runtime.evaluate(task, job)
        again = runtime.evaluate(task, job)
        assert reference['folds'] == first['folds'] == again['folds']
        assert first['classical_prediction_reuse'] == 0
        assert again['classical_prediction_reuse'] == 1
        changed_fold = replace(fold, score=replace(fold.score, target=fold.score.target + 1))
        rescored = runtime.evaluate(task, replace(job, folds=(changed_fold,)))
        assert rescored['classical_prediction_reuse'] == 1
        assert rescored['folds'][0]['predictions'] == again['folds'][0]['predictions']
        assert rescored['loss'] != again['loss']
        for changed_fold in (replace(fold, fit=replace(fold.fit, target=fold.fit.target + 1)),
                             replace(fold, score=replace(fold.score, features=fold.score.features + 1))):
            assert runtime.evaluate(task, replace(job, folds=(changed_fold,)))['classical_prediction_reuse'] == 0
        assert runtime.evaluate(EvaluationTask(graph.key, 14), job)['classical_prediction_reuse'] == 0
        assert runtime.evaluate(task, replace(job, reuse_scope='another role'))['classical_prediction_reuse'] == 0
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('graph_index', (0, 3, 4, 5, 6, 7, 8))
def test_common_graph_runner_accepts_regression_targets(graph_index, monkeypatch):
    import torch
    from benchmark.industrial.experiments.supervised_pilot import training
    from benchmark.industrial.experiments.supervised_pilot.runtime import Job, ValidationFold, evaluate
    from benchmark.industrial.experiments.supervised_pilot.space import initial
    from fedot_ind.core.optimizer.evaluation.contracts import EvaluationTask

    policy = replace(training.training_policy(), min_epochs=1, max_epochs=1, width=4, depth=1)
    monkeypatch.setattr(training, 'training_policy', lambda: policy)
    x = np.random.default_rng(8).normal(size=(20, 1, 64))
    y = x.mean(axis=(1, 2)) * 4
    panel = Panel('Synthetic', np.arange(len(y)), x, y)
    fold = ValidationFold(panel.take(tuple(range(16))), panel.take((16, 17, 18, 19)), (12, 13, 14, 15))
    graph = initial(regression=True)[graph_index]
    recipes = {branch.head: training.Recipe(.001, 8) for branch in graph.branches}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = evaluate(EvaluationTask(graph.key, 42), Job(graph, (fold,), (), recipes, 'cpu'))
    finally:
        torch.set_num_threads(old_threads)
    assert result['loss'] >= 0 and result['folds'][0]['keys'] == [16, 17, 18, 19]
    assert len(result['folds'][0]['predictions']) == 4


@pytest.mark.parametrize('family', ('tser_inception', 'tser_resnet', 'cnn2d_regressor'))
def test_regression_network_outputs_a_single_real_value_in_fedot(family, monkeypatch):
    import torch
    from fedot import create_data
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.extensions import extension_scope
    from benchmark.industrial.experiments.supervised_pilot import training

    policy = replace(training.training_policy(), min_epochs=1, max_epochs=1, width=4, depth=1)
    monkeypatch.setattr(training, 'training_policy', lambda: policy)
    x = np.random.default_rng(4).normal(size=(12, 1, 20))
    data = create_data(x, target=np.linspace(-1, 2, len(x)), task='regression', data_type='ts', use_cache=False)
    rows = np.asarray(data.idx).reshape(-1).tolist()
    params = dict(channels=1, length=20, batch_size=4, device='cpu', learning_rate=.001, seed=17,
                  fit_rows=rows, epoch_validation=rows[-3:], partition_sha256='same', representation_sha256=family)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with extension_scope(recurrence_manifest(regression_model_specs())):
            builder = PipelineBuilder()
            if family == 'cnn2d_regressor':
                builder.add_node('recurrence_image', params={'channels': 1, 'length': 20})
            pipeline = builder.add_node(family, params=params).build()
            try:
                pipeline.fit(data)
                output = pipeline.predict(replace(data, target=None, state='predict'))
                assert len(output.predict) == len(x) and output.predict.numel() == len(x)
            finally:
                pipeline.unfit()
    finally:
        torch.set_num_threads(old_threads)
