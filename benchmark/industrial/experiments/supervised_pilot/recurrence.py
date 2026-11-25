"""Unambiguous recurrence representations for the supervised pilot."""

from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.spatial.distance import pdist, squareform


METRICS = ('cosine', 'euclidean', 'canberra')
MAX_POINTS = 128
WINDOW = 8


@dataclass(frozen=True)
class RecurrenceImageLayout:
    channels: int
    pixels: int
    version: str = 'recurrence_image@1'
    metrics: tuple[str, ...] = METRICS
    window: int = WINDOW
    stride: int = 1
    storage: str = 'uint8'

    def __post_init__(self):
        if (type(self.channels) is not int or self.channels < 1 or type(self.pixels) is not int
                or not 1 <= self.pixels <= MAX_POINTS - WINDOW + 1
                or (self.version, self.metrics, self.window, self.stride, self.storage)
                != ('recurrence_image@1', METRICS, WINDOW, 1, 'uint8')):
            raise ValueError('Unsupported recurrence image layout')

    @classmethod
    def for_panel(cls, channels, length):
        return cls(channels, min(length, MAX_POINTS) - WINDOW + 1)

    @property
    def image_shape(self):
        return self.channels * len(self.metrics), self.pixels, self.pixels

    def pack(self, image):
        image = np.asarray(image)
        if image.ndim != 4 or image.shape[1:] != self.image_shape or image.dtype != np.uint8:
            raise ValueError('Image does not match the declared recurrence layout')
        return image.reshape(len(image), -1, self.pixels)

    def unpack(self, packed):
        values = np.asarray(packed)
        planes, height, width = self.image_shape
        if (values.ndim != 3 or values.shape[1:] != (planes * height, width)
                or values.dtype.kind not in 'uif' or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > 255) or np.any(values % 1)):
            raise ValueError('Features do not match the declared packed recurrence layout')
        # FEDOT can promote the transport dtype, but all stored pixels remain exact integers.
        return values.reshape(len(values), *self.image_shape).astype(np.float32) / 255


def _panel(features):
    values = np.asarray(features, dtype=float)
    if values.ndim != 3 or not values.shape[0] or not values.shape[1] or values.shape[2] < WINDOW:
        raise ValueError('Recurrence input must be an (objects, channels, time) panel with at least 8 points')
    if not np.isfinite(values).all():
        raise ValueError('Recurrence input must be finite')
    return values


def _resample(channel):
    if len(channel) <= MAX_POINTS:
        return channel
    return np.interp(np.linspace(0, len(channel) - 1, MAX_POINTS), np.arange(len(channel)), channel)


def _windows(channel):
    return np.lib.stride_tricks.sliding_window_view(channel, WINDOW)


def _cosine_windows(windows):
    distances = pdist(windows, 'cosine')
    zero = np.all(windows == 0, axis=1)
    left, right = np.triu_indices(len(windows), k=1)
    with_zero = zero[left] | zero[right]
    distances[with_zero] = np.logical_xor(zero[left], zero[right])[with_zero]
    return distances


def _image_channel(channel):
    from fedot_ind.core.operation.transformation.data.kernel_matrix import colorise

    windows = _windows(channel)
    distances = (_cosine_windows(windows), pdist(windows, 'euclidean'), pdist(windows, 'canberra'))
    if not all(np.isfinite(values).all() for values in distances):
        raise ValueError('Non-finite pairwise distance')
    return np.stack([colorise(squareform(values)) for values in distances])


def _rqa_channel(channel):
    from fedot_ind.core.operation.transformation.data.kernel_matrix import TSTransformer
    from fedot_ind.core.operation.transformation.representation.recurrence.recurrence_extractor import (
        RecurrenceExtractor,
    )
    from fedot_ind.core.operation.transformation.representation.recurrence.sequences import (
        RecurrenceFeatureExtractor,
    )

    windows = _windows(channel)
    with np.errstate(divide='ignore', invalid='ignore'):
        if not np.all(windows == windows[0]):
            from fedot.core.operations.operation_parameters import OperationParameters

            extractor = RecurrenceExtractor(OperationParameters(window_size=0, rec_metric='euclidean'))
            return extractor.generate_recurrence_features(windows.T)
        transform = TSTransformer(windows.T, rec_metric='euclidean')
        similarity = np.ones((len(windows), len(windows)))
        matrix = transform.binarization(similarity, threshold=None)
        stats = RecurrenceFeatureExtractor(matrix).quantification_analysis()
        return np.nan_to_num(np.array(tuple(stats.values())), posinf=0, neginf=0)


class RecurrenceTransform:
    def __init__(self, image=False, channels=None, length=None):
        self.image = image
        self.layout = RecurrenceImageLayout.for_panel(channels, length) if image else None

    def transform(self, features):
        panel = _panel(features)
        if not self.image:
            return np.stack([np.concatenate([_rqa_channel(_resample(row)) for row in sample]) for sample in panel])
        if self.layout != RecurrenceImageLayout.for_panel(*panel.shape[1:]):
            raise ValueError('Input panel does not match the recurrence edge declaration')
        images = np.stack([np.concatenate([_image_channel(_resample(row)) for row in sample]) for sample in panel])
        return self.layout.pack(images)


def _transform_factory(image, params):
    return RecurrenceTransform(image, params.get('channels'), params.get('length'))


def _logreg_factory(params):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=params['C'], max_iter=1000)


def recurrence_manifest(models=(), transforms=()):
    from fedot.core.repository.dataset_types import DataTypesEnum as D
    from fedot.core.repository.tasks import TaskTypesEnum as T
    from fedot.extensions import (
        ExtensionManifest, ExternalModelSpec, ExternalTransformSpec, ModelCapabilities,
        ModelHyperparamsSchema, TransformCapabilities,
    )

    def spec(name, *, image, output):
        capability = TransformCapabilities((T.classification, T.regression), (D.ts,), output, requires_fit=False)
        parameters = ModelHyperparamsSchema(required=('channels', 'length') if image else ())
        return ExternalTransformSpec(name, partial(_transform_factory, image), capability, parameters)

    logreg = ExternalModelSpec('logreg', _logreg_factory, ModelCapabilities((T.classification,), (D.tabular,)),
                               ModelHyperparamsSchema(optional=('C',), defaults={'C': 1.}))
    return ExtensionManifest(
        'industrial_supervised_pilot', '1.0.0', module=__name__, models=(logreg, *models),
        transforms=(spec('recurrence_rqa', image=False, output=D.tabular),
                    spec('recurrence_image', image=True, output=D.image), *transforms),
    )
