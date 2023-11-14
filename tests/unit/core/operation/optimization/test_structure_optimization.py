import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters
from fedot_ind.core.operation.optimization.structure_optimization import SVDOptimization, SFPOptimization

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


@pytest.fixture()
def solver():
    model = DummyModel(INPUT_SIZE, OUTPUT_SIZE)
    experimenter = ClassificationExperimenter(model=model,
                                              metric='accuracy',
                                              device='cpu')
    return experimenter
