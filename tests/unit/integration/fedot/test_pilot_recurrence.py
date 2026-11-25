from dataclasses import replace

import numpy as np
import pytest

from benchmark.industrial.experiments.supervised_pilot.recurrence import (
    RecurrenceImageLayout, RecurrenceTransform, recurrence_manifest,
)


@pytest.mark.parametrize('channels', (1, 6))
def test_pack_preserves_metric_and_channel_order(channels):
    layout = RecurrenceImageLayout.for_panel(channels, 12)
    panel = np.stack([np.arange(12) ** (1 + c % 2) + c for c in range(channels)])[None, :, :]
    packed = RecurrenceTransform(True, channels, 12).transform(panel)
    planes = layout.unpack(packed)
    assert planes.shape == (1, 3 * channels, 5, 5)
    for c in range(channels):
        solo = RecurrenceImageLayout.for_panel(1, 12).unpack(
            RecurrenceTransform(True, 1, 12).transform(panel[:, c:c + 1]))
        np.testing.assert_array_equal(planes[:, c * 3:(c + 1) * 3], solo)
    np.testing.assert_array_equal(layout.unpack(packed[::-1].astype(float)), planes)
    with pytest.raises(ValueError, match='layout'):
        layout.unpack(packed[:, :, :-1])


def test_zero_window_distances_use_the_explicit_contract():
    signal = np.r_[np.zeros(8), 1.].reshape(1, 1, -1)
    layout = RecurrenceImageLayout.for_panel(1, 9)
    image = layout.unpack(RecurrenceTransform(True, 1, 9).transform(signal))
    expected = np.array([[0, 1], [1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(image[0], np.stack([expected] * 3))
    zeros = np.zeros((2, 1, 12))
    assert not RecurrenceTransform(True, 1, 12).transform(zeros).any()
    assert np.isfinite(RecurrenceTransform().transform(zeros)).all()


def test_recurrence_matches_existing_numpy_implementation_on_nonzero_windows():
    from fedot.core.operations.operation_parameters import OperationParameters
    from fedot_ind.core.operation.transformation.representation.recurrence.recurrence_extractor import (
        RecurrenceExtractor,
    )

    panel = np.random.default_rng(32).normal(size=(2, 1, 20))
    with np.errstate(divide='ignore', invalid='ignore'):
        for image in (False, True):
            original = RecurrenceExtractor(OperationParameters(window_size=8, rec_metric='euclidean',
                                                               image_mode=image))
            emitted = RecurrenceTransform(image, 1, 20).transform(panel)
            expected = np.stack([original.generate_recurrence_features(row[0]) for row in panel])
            if image:
                expected = RecurrenceImageLayout.for_panel(1, 20).pack(expected)
            np.testing.assert_array_equal(emitted, expected)


@pytest.mark.parametrize(('operation', 'axis'), [('fourier_basis', 2), ('wavelet_basis', 6)])
def test_spectral_channels_anchor_a_tabular_route(operation, axis):
    from fedot import create_data
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.extensions import extension_scope
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope

    panel = np.random.default_rng(13).normal(size=(8, 2, 32))
    data = create_data(panel, target=np.arange(len(panel)) % 2, task='classification', data_type='ts', use_cache=False)
    predict = replace(data, target=None, state='predict')
    with industrial_extension_scope(), extension_scope(recurrence_manifest()):
        spectral = PipelineBuilder().add_node(operation).build()
        table = PipelineBuilder().add_node(operation).add_node('quantile_extractor_torch').build()
        try:
            spectral.fit(data)
            coefficients = spectral.predict(predict).features
            assert tuple(coefficients.shape) == (8, axis, 32)
            table.fit(data)
            features = table.predict(predict).features
            assert features.ndim == 2 and len(features) == len(panel)
            assert np.isfinite(features.numpy()).all()
        finally:
            spectral.unfit()
            table.unfit()


class _ImageReader:
    def fit(self, features, target):
        self.layout = RecurrenceImageLayout.for_panel(1, 16)
        self.layout.unpack(features)
        return self

    def predict(self, features):
        return self.layout.unpack(features)[:, 1, 0, -1]


@pytest.mark.parametrize('operation', ('recurrence_rqa', 'recurrence_image'))
def test_recurrence_runs_through_fedot_fit_and_predict(operation):
    from fedot import create_data
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.repository.dataset_types import DataTypesEnum as D
    from fedot.core.repository.tasks import TaskTypesEnum as T
    from fedot.extensions import ExternalModelSpec, ModelCapabilities, extension_scope

    rng = np.random.default_rng(17)
    panel = rng.normal(size=(10, 1, 16))
    y = np.arange(len(panel)) % 2
    data = create_data(panel, target=y, task='classification', data_type=D.ts, use_cache=False)
    output_spec = ExternalModelSpec('recurrence_image_probe', _ImageReader,
                                    ModelCapabilities((T.classification,), (D.image,)))
    image = operation == 'recurrence_image'
    params = dict(channels=1, length=16) if image else None
    ending = 'recurrence_image_probe' if image else 'logreg'
    with extension_scope(recurrence_manifest((output_spec,))):
        pipeline = PipelineBuilder().add_node(operation, params=params).add_node(ending).build()
        try:
            pipeline.fit(data)
            result = pipeline.predict(replace(data, target=None, state='predict'))
            assert len(result.predict) == len(panel)
            if image:
                expected = _ImageReader().fit(RecurrenceTransform(True, 1, 16).transform(panel), y)
                np.testing.assert_allclose(
                    result.predict.reshape(-1), expected.predict(RecurrenceTransform(True, 1, 16).transform(panel)))
        finally:
            pipeline.unfit()
