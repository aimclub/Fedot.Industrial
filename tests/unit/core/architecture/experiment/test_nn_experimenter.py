import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters, \
    ObjectDetectionExperimenter, SegmentationExperimenter

NUM_SAMPLES = 100
INPUT_SIZE = 10
OUTPUT_SIZE = 5
BATCH_SIZE = 32


class DummyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class DummyModelSegmentation(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return {'out': self.linear(x)}


class SimpleDataset(Dataset):
    def __init__(self, num_samples, input_size, output_size):
        self.inputs = torch.rand((num_samples, input_size))
        self.targets = torch.randint(0, output_size, (num_samples,))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


@pytest.fixture
def dummy_data_loader():
    dataset = SimpleDataset(NUM_SAMPLES, INPUT_SIZE, OUTPUT_SIZE)
    shuffle = True
    return DataLoader(dataset,
                      batch_size=BATCH_SIZE,
                      shuffle=shuffle)


def test_classification_experimenter(dummy_data_loader):
    model = DummyModel(INPUT_SIZE, OUTPUT_SIZE)
    experimenter = ClassificationExperimenter(model=model,
                                              metric='accuracy',
                                              device='cpu')

    fit_params = FitParameters(dataset_name='test',
                               train_dl=dummy_data_loader,
                               val_dl=dummy_data_loader,
                               num_epochs=1)

    experimenter.fit(p=fit_params, phase='train',
                     model_losses=None,
                     filter_pruning=None,
                     start_epoch=0,
                     initial_validation=False)

    labels = experimenter.predict(dataloader=dummy_data_loader, proba=False)
    probs = experimenter.predict(dataloader=dummy_data_loader, proba=True)
    probs_ = experimenter.predict(dataloader=dummy_data_loader)

    assert labels is not None
    assert probs is not None
    assert probs_ is not None
    for obj in (labels, probs_, probs):
        assert isinstance(obj, dict)
    assert experimenter.best_score != 0
    assert experimenter.model.training is False


def test_object_detection_experimenter(dummy_data_loader):
    model = DummyModel(INPUT_SIZE, OUTPUT_SIZE)
    experimenter = ClassificationExperimenter(model=model,
                                              metric='accuracy',
                                              device='cpu')

    fit_params = FitParameters(dataset_name='test',
                               train_dl=dummy_data_loader,
                               val_dl=dummy_data_loader,
                               num_epochs=1)

    experimenter.fit(p=fit_params, phase='train',
                     model_losses=None,
                     filter_pruning=None,
                     start_epoch=0,
                     initial_validation=False)

    labels = experimenter.predict(dataloader=dummy_data_loader, proba=False)
    probs = experimenter.predict(dataloader=dummy_data_loader, proba=True)
    probs_ = experimenter.predict(dataloader=dummy_data_loader)
    assert labels is not None
    assert probs is not None
    assert probs_ is not None
    for obj in (labels, probs_, probs):
        assert isinstance(obj, dict)
    assert experimenter.best_score != 0
    assert experimenter.model.training is False
