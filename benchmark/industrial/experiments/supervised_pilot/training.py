"""Object-level selection and refit on the common Industrial training loop."""

from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import lru_cache, partial
from importlib.resources import files
from math import isfinite

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedot_ind.neural.training.engine import SupervisedBatch, TrainingPlan, TrainingSession
from ...protocols.json_io import parse_protocol_json
from .data import canonical_bytes, digest
from .recurrence import RecurrenceImageLayout


@dataclass(frozen=True)
class TrainingPolicy:
    min_epochs: int
    max_epochs: int
    stop_patience: int
    plateau_patience: int
    plateau_factor: float
    lr_min: float
    learning_rates: tuple[float, ...]
    batches: tuple[int, ...]
    width: int
    depth: int
    cnn_blocks: tuple[int, ...]
    cnn_projection: int

    def __post_init__(self):
        ints = (self.min_epochs, self.max_epochs, self.stop_patience, self.plateau_patience,
                self.width, self.depth, self.cnn_projection, *self.batches, *self.cnn_blocks)
        if (any(type(value) is not int or value < 1 for value in ints)
                or self.min_epochs > self.max_epochs or not self.batches or not self.learning_rates
                or len(self.cnn_blocks) != 2 or not 0 < self.plateau_factor < 1 or not 0 < self.lr_min
                or any(not isfinite(lr) or lr < self.lr_min for lr in self.learning_rates)):
            raise ValueError('Invalid supervised training policy')


@lru_cache(maxsize=1)
def training_policy():
    payload = parse_protocol_json(files(__package__).joinpath('neural_defaults.json').read_text(encoding='utf-8'))
    for name in ('learning_rates', 'batches', 'cnn_blocks'):
        payload[name] = tuple(payload[name])
    return TrainingPolicy(**payload)


@dataclass(frozen=True)
class Recipe:
    learning_rate: float
    batch_size: int

    def __post_init__(self):
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('Recipe learning rate must be positive and finite')
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('Recipe batch size must be a positive integer')


def recipe_grid(policy: TrainingPolicy, training_sizes):
    """Keep every nominal recipe and link it to its identical feasible batch schedule."""
    training_sizes = tuple(training_sizes)
    representatives = {}
    for lr in policy.learning_rates:
        for batch in policy.batches:
            signature = lr, tuple(min(batch, size) for size in training_sizes)
            recipe = Recipe(lr, batch)
            yield recipe, representatives.setdefault(signature, recipe)


class _Residual1D(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv1d(width, width, 3, padding=1, bias=False),
                                    nn.BatchNorm1d(width), nn.ReLU(),
                                    nn.Conv1d(width, width, 3, padding=1, bias=False), nn.BatchNorm1d(width))

    def forward(self, features):
        return torch.relu(features + self.branch(features))


def _network(family, channels, outputs, policy):
    if family == 'cnn2d_classifier':
        from fedot_ind.neural.encoders.config import ConvBlockConfig, EncoderConfig
        from fedot_ind.neural.encoders.encoders import CNNEncoder

        config = EncoderConfig('cnn', policy.cnn_projection, 4, in_channels=channels, dropout=0.,
                               conv_blocks=tuple(ConvBlockConfig(w, dropout=0.) for w in policy.cnn_blocks))
        return nn.Sequential(CNNEncoder(config), nn.Linear(policy.cnn_projection, outputs))
    if family == 'tsc_inception':
        from fedot_ind.core.models.nn.network_modules.layers.special import InceptionBlock

        stem = InceptionBlock(input_dim=channels, number_of_filters=policy.width, depth=policy.depth, ks=16)
        width = 4 * policy.width
    elif family == 'tsc_resnet':
        width = policy.width
        stem = nn.Sequential(nn.Conv1d(channels, width, 5, padding=2),
                             *(_Residual1D(width) for _ in range(policy.depth)))
    else:
        raise ValueError('Unknown pilot neural architecture')
    return nn.Sequential(stem, nn.AdaptiveAvgPool1d(1), nn.Flatten(1), nn.Linear(width, outputs))


class LabelCodec:
    def __init__(self, classes):
        self.classes = tuple(classes)
        if len(self.classes) < 2 or tuple(sorted(set(self.classes))) != self.classes:
            raise ValueError('Class names must be unique and sorted')

    def encode(self, labels):
        values = np.asarray(labels)
        if values.ndim != 1:
            raise ValueError('Class labels must be a vector')
        mapping = {label: i for i, label in enumerate(self.classes)}
        if any(value not in mapping for value in values):
            raise ValueError('Unknown evaluation label')
        return np.array([mapping[value] for value in values], dtype=np.int64)


_TRAINED = OrderedDict()


class NeuralClassifier:
    def __init__(self, family, params, policy=None):
        self.family = family
        self.params = dict(params)
        self.policy = policy or training_policy()
        self.recipe = Recipe(params['learning_rate'], params['batch_size'])
        self.device = torch.device(params['device'])
        if self.device.type not in ('cpu', 'cuda') or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise RuntimeError('The explicitly requested training device is unavailable')
        self.model = None
        self.trace = None

    def _features(self, features):
        channels, length = self.params['channels'], self.params['length']
        if self.family == 'cnn2d_classifier':
            array = RecurrenceImageLayout.for_panel(channels, length).unpack(features)
        else:
            array = np.asarray(features, dtype=np.float32)
            if array.ndim != 3 or array.shape[1:] != (channels, length) or not np.isfinite(array).all():
                raise ValueError('A one-dimensional encoder requires the declared time panel')
        return np.ascontiguousarray(array)

    def _init(self, channels):
        return _network(self.family, channels, self.params['class_count'], self.policy).to(self.device)

    def _batch_adapter(self, batch):
        return SupervisedBatch(*(value.to(self.device) for value in batch))

    def _loader(self, x, y, *, training):
        return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                          batch_size=self.recipe.batch_size, shuffle=training, drop_last=False,
                          generator=torch.Generator().manual_seed(self.params['seed']))

    def _session(self, x, y, *, epochs, validation=None):
        model = self._init(x.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.recipe.learning_rate)
        session = TrainingSession(TrainingPlan(epochs), model, optimizer, self._criterion(),
                                  self._loader(x, y, training=True), validation, batch_adapter=self._batch_adapter)
        return model, optimizer, session

    def _criterion(self):
        return nn.CrossEntropyLoss()

    def _target(self, indices):
        if (not np.isfinite(indices).all() or np.any(indices % 1)
                or set(indices) != set(range(self.params['class_count']))):
            raise ValueError('Training targets must cover the declared class map')
        return indices.astype(np.int64)

    def _select(self, x, y, fit, score):
        policy = self.policy
        valid = self._loader(x[score], y[score], training=False)
        _, optimizer, session = self._session(x[fit], y[fit], epochs=policy.max_epochs, validation=valid)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=policy.plateau_patience, factor=policy.plateau_factor, min_lr=policy.lr_min)
        schedule, best_epoch = [], 0
        for epoch in range(1, policy.max_epochs + 1):
            schedule.append(float(optimizer.param_groups[0]['lr']))
            progress = session.run(until_epoch=epoch)
            plateau.step(progress.losses[-1].validation)
            best_epoch = progress.best_epoch
            if epoch >= policy.min_epochs and epoch - best_epoch >= policy.stop_patience:
                break
        return tuple(schedule[:best_epoch]), len(schedule)

    def fit(self, idx, features, target, params):
        x = self._features(features)
        raw = np.asarray(target)
        if raw.shape not in ((len(x),), (len(x), 1)):
            raise ValueError('Expected exactly one target per object')
        y = self._target(raw.reshape(-1))
        rows = tuple(np.asarray(idx).reshape(-1).tolist())
        if rows != tuple(self.params['fit_rows']) or len(set(rows)) != len(x):
            raise ValueError('FEDOT changed the admitted training object keys')
        selected = tuple(self.params['epoch_validation'])
        if not selected or len(set(selected)) != len(selected) or not set(selected) < set(rows):
            raise ValueError('Epoch validation must be a proper subset of the admitted training rows')
        score = np.isin(rows, selected)
        identity = digest(canonical_bytes([type(self).__name__, self.family, self.params, asdict(self.policy),
                                           str(torch.__version__),
                                           digest(x.tobytes()), digest(y.tobytes())]))
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.params['seed'])
            if identity in _TRAINED:
                state, previous = _TRAINED[identity]
                _TRAINED.move_to_end(identity)
                self.model = self._init(x.shape[1])
                self.model.load_state_dict(state)
                self.trace = {**previous, 'route': 'exact', 'optimizer_steps': 0}
                return self
            schedule, tried = self._select(x, y, ~score, score)
            torch.manual_seed(self.params['seed'])
            self.model, optimizer, session = self._session(x, y, epochs=len(schedule))
            for epoch, lr in enumerate(schedule, 1):
                optimizer.param_groups[0]['lr'] = lr
                session.run(until_epoch=epoch)
            self.trace = dict(device=str(next(self.model.parameters()).device), route='refit',
                              selected_epoch=len(schedule), searched_epochs=tried, schedule=schedule,
                              fit_rows=rows, epoch_validation=selected,
                              optimizer_steps=len(schedule) * len(self._loader(x, y, training=True)))
            _TRAINED[identity] = ({k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}, self.trace)
            if len(_TRAINED) > 12:
                _TRAINED.popitem(last=False)
        return self

    def _outputs(self, features):
        if self.model is None:
            raise RuntimeError('Model is not fitted')
        x = torch.from_numpy(self._features(features))
        self.model.eval()
        with torch.inference_mode():
            return torch.cat([self.model(batch.to(self.device)).cpu() for batch in x.split(self.recipe.batch_size)])

    def predict_proba(self, features):
        return self._outputs(features).softmax(dim=1).numpy()

    def predict(self, features):
        return self.predict_proba(features).argmax(axis=1)


def neural_model_specs():
    from fedot.core.repository.dataset_types import DataTypesEnum as D
    from fedot.core.repository.tasks import TaskTypesEnum as T
    from fedot.extensions import ExternalModelSpec, ModelCapabilities, ModelHyperparamsSchema

    fields = ('batch_size', 'channels', 'class_count', 'device', 'epoch_validation', 'fit_rows', 'learning_rate',
              'length', 'partition_sha256', 'representation_sha256', 'seed')
    return tuple(
        ExternalModelSpec(
            name, partial(NeuralClassifier, name),
            ModelCapabilities((T.classification,), (kind,), tags=('neural',), output_data_type=D.tabular),
            ModelHyperparamsSchema(required=fields),
        )
        for name, kind in (('cnn2d_classifier', D.image), ('tsc_inception', D.ts), ('tsc_resnet', D.ts))
    )
