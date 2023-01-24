import os

import pytest
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Resize

from core.architecture.datasets.object_detection_datasets import COCODataset
from core.architecture.datasets.prediction_datasets import PredictionFolderDataset
from core.architecture.experiment.nn_experimenter import ClassificationExperimenter, \
    FasterRCNNExperimenter
from core.architecture.utils.utils import PROJECT_PATH
from core.operation.optimization.structure_optimization import SVDOptimization, \
    SFPOptimization

SVD_PARAMS = {'energy_thresholds': [0.9]}
SFP_PARAMS = {'pruning_ratio': 0.5}
DATASETS_PATH = os.path.join(PROJECT_PATH, 'tests/data/datasets/')


@pytest.fixture()
def prepare_classification(tmp_path):
    transform = Compose([ToTensor(), Resize((256, 256))])
    train_ds = ImageFolder(root=DATASETS_PATH + 'Agricultural/train', transform=transform)
    val_ds = ImageFolder(root=DATASETS_PATH + 'Agricultural/val', transform=transform)
    exp_params = {
        'model': resnet18(num_classes=3),
        'gpu': False
    }
    fit_params = {
        'dataset_name': 'Agricultural',
        'train_dataset': train_ds,
        'val_dataset': val_ds,
        'num_epochs': 1,
        'dataloader_params': {'batch_size': 16, 'num_workers': 4},
        'models_path': tmp_path.joinpath('models'),
        'summary_path': tmp_path.joinpath('summary')
    }
    yield exp_params, fit_params, tmp_path


def classification_predict(experimenter):
    test_predict_path = os.path.join(DATASETS_PATH, 'Agricultural/val/banana')
    dataset = PredictionFolderDataset(
        image_folder=test_predict_path,
        transform=Compose([ToTensor(), Resize((256, 256))])
    )
    preds = experimenter.predict(dataset)
    assert set(preds.keys()) == set(os.listdir(test_predict_path))
    for k, v in preds.items():
        assert v in [0, 1, 2]
    proba_preds = experimenter.predict_proba(dataset)
    assert set(proba_preds.keys()) == set(os.listdir(test_predict_path))
    for k, v in proba_preds.items():
        assert len(v) == 3


def test_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    experimenter.fit(**fit_params)
    assert os.path.exists(tmp_path.joinpath('models/Agricultural/ResNet/trained.sd.pt'))
    classification_predict(experimenter)


def test_sfp_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SFPOptimization(**SFP_PARAMS)
    experimenter.fit(structure_optimization=optimization, **fit_params)
    root = tmp_path.joinpath('models/Agricultural/ResNet_SFP_P-0.50/')
    assert os.path.exists(root.joinpath('trained.sd.pt'))
    # assert os.path.exists(root.joinpath('fine-tuning.sd.pt'))
    classification_predict(experimenter)


def test_svd_channel_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='channel', **SVD_PARAMS)
    experimenter.fit(structure_optimization=optimization, **fit_params)
    root = tmp_path.joinpath('models/Agricultural/ResNet_SVD_channel_O-100.0_H-0.001000/')
    assert os.path.exists(root.joinpath('trained.sd.pt'))
    assert os.path.exists(root.joinpath('trained.model.pt'))
    assert os.path.exists(root.joinpath('e_0.9.sd.pt'))
    classification_predict(experimenter)


def test_svd_spatial_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='spatial', **SVD_PARAMS)
    experimenter.fit(structure_optimization=optimization, **fit_params)
    root = tmp_path.joinpath('models/Agricultural/ResNet_SVD_spatial_O-100.0_H-0.001000/')
    assert os.path.exists(root.joinpath('trained.sd.pt'))
    assert os.path.exists(root.joinpath('trained.model.pt'))
    assert os.path.exists(root.joinpath('e_0.9.sd.pt'))
    classification_predict(experimenter)


@pytest.fixture()
def prepare_detection(tmp_path):
    transform = Compose([ToTensor()])
    dataset = COCODataset(
        images_path=DATASETS_PATH + 'ALET10/test',
        json_path=DATASETS_PATH + 'ALET10/test.json',
        transform=transform)
    exp_params = {
        'num_classes': len(dataset.classes) + 1,
        'model_params': {'weights': 'DEFAULT'},
        'gpu': False
    }
    fit_params = {
        'dataset_name': 'ALET10',
        'train_dataset': dataset,
        'val_dataset': dataset,
        'num_epochs': 1,
        'dataloader_params': {
            'batch_size': 2,
            'num_workers': 2
        },
        'models_path': tmp_path.joinpath('models'),
        'summary_path': tmp_path.joinpath('summary')
    }
    yield exp_params, fit_params, tmp_path


def detection_predict(experimenter):
    test_predict_path = os.path.join(DATASETS_PATH, 'ALET10/test')
    dataset = PredictionFolderDataset(
        image_folder=test_predict_path,
        transform=ToTensor()
    )
    preds = experimenter.predict(dataset)
    assert set(preds.keys()) == set(os.listdir(test_predict_path))
    for k, v in preds.items():
        assert set(v.keys()) == {'labels', 'boxes'}
    proba_preds = experimenter.predict_proba(dataset)
    assert set(proba_preds.keys()) == set(os.listdir(test_predict_path))
    for k, v in proba_preds.items():
        assert set(v.keys()) == {'labels', 'boxes', 'scores'}


def test_fasterrcnn_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = FasterRCNNExperimenter(**exp_params)
    experimenter.fit(**fit_params)
    assert os.path.exists(tmp_path.joinpath('models/ALET10/FasterRCNN/trained.sd.pt'))
    detection_predict(experimenter)


def test_sfp_fasterrcnn_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = FasterRCNNExperimenter(**exp_params)
    optimization = SFPOptimization(**SFP_PARAMS)
    experimenter.fit(structure_optimization=optimization, **fit_params)
    root = tmp_path.joinpath('models/ALET10/FasterRCNN_SFP_P-0.50/')
    assert os.path.exists(root.joinpath('trained.sd.pt'))
    detection_predict(experimenter)


def test_svd_channel_fasterrcnn_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = FasterRCNNExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='channel', **SVD_PARAMS)
    experimenter.fit(structure_optimization=optimization, **fit_params)
    root = tmp_path.joinpath('models/ALET10/FasterRCNN_SVD_channel_O-100.0_H-0.001000/')
    assert os.path.exists(root.joinpath('trained.sd.pt'))
    assert os.path.exists(root.joinpath('trained.model.pt'))
    assert os.path.exists(root.joinpath('e_0.9.sd.pt'))
    detection_predict(experimenter)


def test_svd_spatial_fasterrcnn_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = FasterRCNNExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='spatial', **SVD_PARAMS)
    experimenter.fit(structure_optimization=optimization, **fit_params)
    root = tmp_path.joinpath('models/ALET10/FasterRCNN_SVD_spatial_O-100.0_H-0.001000/')
    assert os.path.exists(root.joinpath('trained.sd.pt'))
    assert os.path.exists(root.joinpath('trained.model.pt'))
    assert os.path.exists(root.joinpath('e_0.9.sd.pt'))
    detection_predict(experimenter)
