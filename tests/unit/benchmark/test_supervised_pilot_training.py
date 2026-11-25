from dataclasses import replace

import numpy as np
import pytest
import torch

from benchmark.industrial.experiments.supervised_pilot.recurrence import RecurrenceTransform, recurrence_manifest
from benchmark.industrial.experiments.supervised_pilot.training import (
    LabelCodec, NeuralClassifier, neural_model_specs, recipe_grid, training_policy,
)


@pytest.fixture(autouse=True)
def small_torch_pool():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


@pytest.fixture
def panel():
    x = np.random.default_rng(17).normal(size=(14, 1, 20))
    y = np.arange(len(x)) % 2
    keys = tuple(range(100, 100 + len(x)))
    params = dict(learning_rate=0.001, batch_size=4, device='cpu', seed=42, channels=1, length=20, class_count=2,
                  fit_rows=keys, epoch_validation=keys[-4:], partition_sha256='split', representation_sha256='raw')
    return keys, x, y, params


@pytest.mark.parametrize('architecture', ('cnn2d_classifier', 'tsc_inception', 'tsc_resnet'))
def test_selection_and_refit_on_the_declared_rows(architecture, panel):
    keys, x, y, params = panel
    if architecture == 'cnn2d_classifier':
        x = RecurrenceTransform(True, 1, 20).transform(x)
    policy = replace(training_policy(), width=4, depth=1, min_epochs=2, max_epochs=2)
    model = NeuralClassifier(architecture, params, policy).fit(keys, x, y, {})
    columns = model.predict_proba(x)
    assert model.trace['searched_epochs'] == 2
    assert model.trace['fit_rows'] == keys
    assert model.trace['optimizer_steps'] == model.trace['selected_epoch'] * 4
    assert columns.shape == (len(y), 2)
    np.testing.assert_allclose(columns.sum(axis=1), 1, atol=1e-6)
    with pytest.raises(ValueError, match='object keys'):
        model.fit(keys[::-1], x, y, {})


def test_exact_reuse_keeps_outputs_but_rejects_changed_recipe_or_masks(panel):
    keys, x, y, params = panel
    policy = replace(training_policy(), width=4, depth=1, min_epochs=1, max_epochs=1)

    def fit(**extra):
        return NeuralClassifier('tsc_resnet', {**params, **extra}, policy).fit(keys, x, y, {})

    initial, again = fit(), fit()
    assert again.trace['route'] == 'exact' and again.trace['optimizer_steps'] == 0
    np.testing.assert_array_equal(initial.predict_proba(x), again.predict_proba(x))
    for override in ({'batch_size': 5}, {'learning_rate': 0.003}, {'seed': 123},
                     {'epoch_validation': keys[:4]}, {'representation_sha256': 'elsewhere'}):
        assert fit(**override).trace['route'] == 'refit'


def test_labels_and_batch_aliases():
    codec = LabelCodec(('false', 'true'))
    np.testing.assert_array_equal(codec.encode(np.array(['true', 'false'])), [1, 0])
    with pytest.raises(ValueError, match='Unknown'):
        codec.encode(np.array(['not in TRAIN']))
    nominal = list(recipe_grid(training_policy(), (15, 16)))
    assert len(nominal) == 6
    assert nominal[2][0].batch_size == 32 and nominal[2][1].batch_size == 16


def test_real_pipeline_decodes_image_before_classification(panel, monkeypatch):
    from fedot import create_data
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.extensions import extension_scope
    from benchmark.industrial.experiments.supervised_pilot import training

    keys, x, y, params = panel
    policy = replace(training_policy(), width=4, depth=1, min_epochs=1, max_epochs=1)
    monkeypatch.setattr(training, 'training_policy', lambda: policy)
    data = create_data(x, target=y, task='classification', data_type='ts', use_cache=False)
    data = replace(data, idx=torch.as_tensor(keys))
    with extension_scope(recurrence_manifest(neural_model_specs())):
        pipeline = (PipelineBuilder().add_node('recurrence_image', params={'channels': 1, 'length': 20})
                    .add_node('cnn2d_classifier', params=params).build())
        try:
            pipeline.fit(data)
            fitted = pipeline.predict(replace(data, target=None, state='predict'), output_mode='full_probs')
            assert tuple(fitted.predict.shape) == (len(x), 2)
        finally:
            pipeline.unfit()
