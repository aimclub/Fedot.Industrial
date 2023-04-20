import os

import pytest
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, ToTensor

from fedot_ind.core.architecture.datasets.object_detection_datasets import COCODataset
from fedot_ind.core.architecture.datasets.prediction_datasets import PredictionFolderDataset
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FasterRCNNExperimenter, \
    FitParameters
from fedot_ind.core.architecture.utils.utils import PROJECT_PATH

DATASETS_PATH = os.path.abspath(PROJECT_PATH + '/../tests/data/datasets')


@pytest.fixture()
def prepare_classification(tmp_path: str = './tmp'):
    from pathlib import Path
    tmp_path = Path(os.path.abspath(tmp_path))

    transform = Compose([ToTensor(), Resize((256, 256))])
    root_path_train = os.path.join(DATASETS_PATH, 'Agricultural/train')
    root_path_val = os.path.join(DATASETS_PATH, 'Agricultural/val')
    train_ds = ImageFolder(root=root_path_train, transform=transform)
    val_ds = ImageFolder(root=root_path_val, transform=transform)
    exp_params = {
        'model': resnet18(num_classes=3),
        'device': 'cpu'
    }
    fit_params = FitParameters(
        dataset_name='Agricultural',
        train_dl=DataLoader(
            dataset=train_ds,
            batch_size=16,
            shuffle=True,
            num_workers=4
        ),
        val_dl=DataLoader(
            dataset=val_ds,
            batch_size=16,
            num_workers=4
        ),
        num_epochs=1,
        models_path=tmp_path.joinpath('models'),
        summary_path=tmp_path.joinpath('summary')
    )
    yield exp_params, fit_params, tmp_path


def classification_predict(experimenter):
    test_predict_path = os.path.join(DATASETS_PATH, 'Agricultural/val')
    image_files = []
    for address, dirs, files in os.walk(test_predict_path):
        for name in files:
            image_files.append(os.path.join(address, name))
    dataset = PredictionFolderDataset(
        image_folder=test_predict_path,
        transform=Compose([ToTensor(), Resize((256, 256))])
    )
    dataloader = DataLoader(dataset)
    preds = experimenter.predict(dataloader)
    assert set(preds.keys()) == set(image_files)
    for k, v in preds.items():
        assert v in [0, 1, 2]
    proba_preds = experimenter.predict_proba(dataloader)
    assert set(proba_preds.keys()) == set(image_files)
    for k, v in proba_preds.items():
        assert len(v) == 3


def test_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    experimenter.fit(p=fit_params)
    assert os.path.exists(tmp_path.joinpath('models/Agricultural/ResNet/train.sd.pt'))
    assert os.path.exists(tmp_path.joinpath('summary/Agricultural/ResNet/train/train.csv'))
    assert os.path.exists(tmp_path.joinpath('summary/Agricultural/ResNet/train/val.csv'))
    classification_predict(experimenter)


@pytest.fixture()
def prepare_detection(tmp_path):
    dataset = COCODataset(
        images_path=os.path.join(DATASETS_PATH, 'ALET10/test'),
        json_path=os.path.join(DATASETS_PATH, 'ALET10/test.json'),
        transform=ToTensor()
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )
    exp_params = {
        'num_classes': len(dataset.classes) + 1,
        'model_params': {'weights': 'DEFAULT'},
        'device': 'cpu'
    }
    fit_params = FitParameters(
        dataset_name='ALET10',
        train_dl=dataloader,
        val_dl=dataloader,
        num_epochs=1,
        optimizer_params={'lr': 0.0001},
        models_path=tmp_path.joinpath('models'),
        summary_path=tmp_path.joinpath('summary')
    )
    yield exp_params, fit_params, tmp_path


def detection_predict(experimenter):
    test_predict_path = os.path.join(DATASETS_PATH, 'ALET10/test')
    image_files = []
    for address, dirs, files in os.walk(test_predict_path):
        for name in files:
            image_files.append(os.path.join(address, name))
    dataset = PredictionFolderDataset(
        image_folder=test_predict_path,
        transform=ToTensor()
    )
    dataloader = DataLoader(dataset, collate_fn=lambda x: tuple(zip(*x)))
    preds = experimenter.predict(dataloader)
    assert set(preds.keys()) == set(image_files)
    for k, v in preds.items():
        assert set(v.keys()) == {'labels', 'boxes'}
    proba_preds = experimenter.predict_proba(dataloader)
    assert set(proba_preds.keys()) == set(image_files)
    for k, v in proba_preds.items():
        assert set(v.keys()) == {'labels', 'boxes', 'scores'}


def test_fasterrcnn_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = FasterRCNNExperimenter(**exp_params)
    experimenter.fit(p=fit_params)
    assert os.path.exists(tmp_path.joinpath('models/ALET10/FasterRCNN/train.sd.pt'))
    assert os.path.exists(tmp_path.joinpath('summary/ALET10/FasterRCNN/train/train.csv'))
    assert os.path.exists(tmp_path.joinpath('summary/ALET10/FasterRCNN/train/val.csv'))
    detection_predict(experimenter)
