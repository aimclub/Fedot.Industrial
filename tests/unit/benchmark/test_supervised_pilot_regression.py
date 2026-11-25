from dataclasses import replace

import numpy as np
import pytest

from benchmark.industrial.experiments.supervised_pilot.data import Panel, digest
from benchmark.industrial.experiments.supervised_pilot.regression import (
    Sampling, SamplingEvidence, observed_temporal_panel, regression_model_specs, regression_splits, score_regression,
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


def test_regression_score_rejects_broadcast_and_permutation():
    panel = Panel('Synthetic', np.array([3, 5, 9]), np.zeros((3, 1, 8)), np.full(3, 4.))
    assert score_regression(panel, [3, 5, 9], [3, 4, 5]) == dict(rmse=np.sqrt(2 / 3), mae=2 / 3, r2=None)
    with pytest.raises(ProtocolError, match='finite vector'):
        score_regression(panel, [3, 5, 9], [[3], [4], [5]])
    with pytest.raises(ProtocolError, match='object order'):
        score_regression(panel, [3, 9, 5], [3, 4, 5])


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
