import os

import pytest
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize

from core.architecture.datasets.object_detection_datasets import COCODataset, collate_fn
from core.architecture.experiment.CVModule import ClassificationExperimenter, \
    FasterRCNNExperimenter
from core.architecture.utils.utils import PROJECT_PATH

SVD_PARAMS = {
    'orthogonal_loss_factor': 10,
    'hoer_loss_factor': 0.001,
    'energy_thresholds': [0.9],
    'finetuning_epochs': 1
}

SFP_PARAMS = {
    'pruning_ratio': 0.5,
    'finetuning_epochs': 1
}

DATASETS_PATH = os.path.join(PROJECT_PATH, 'tests/data/datasets/')


@pytest.fixture()
def get_test_classification_dataset(tmp_path):
    transform = Compose([ToTensor(), Resize((256, 256))])
    train_ds = ImageFolder(root=DATASETS_PATH + 'Agricultural/train', transform=transform)
    val_ds = ImageFolder(root=DATASETS_PATH + 'Agricultural/val', transform=transform)
    exp_params = {
        'dataset_name': 'Agricultural',
        'train_dataset': train_ds,
        'val_dataset': val_ds,
        'num_classes': 3,
        'dataloader_params': {'batch_size': 16, 'num_workers': 4},
        'model': 'ResNet18',
        'model_params': {},
        'models_saving_path': tmp_path.joinpath('models'),
        'optimizer': torch.optim.Adam,
        'optimizer_params': {},
        'target_loss': torch.nn.CrossEntropyLoss,
        'loss_params': {},
        'metric': 'f1',
        'summary_path': tmp_path.joinpath('runs'),
        'summary_per_class': True,
        'gpu': False
    }
    yield exp_params, tmp_path


def classification_predict(experimenter):
    test_predict_path = DATASETS_PATH + 'Agricultural/val/banana'
    preds = experimenter.predict(test_predict_path)
    assert set(preds.keys()) == set(os.listdir(test_predict_path))
    for k, v in preds.items():
        assert v in [0, 1, 2]
    proba_preds = experimenter.predict_proba(test_predict_path)
    assert set(proba_preds.keys()) == set(os.listdir(test_predict_path))
    for k, v in proba_preds.items():
        assert len(v) == 3


def test_wrong_structure_optimization(get_test_classification_dataset):
    exp_params, tmp_path = get_test_classification_dataset
    with pytest.raises(ValueError):
        experimenter = ClassificationExperimenter(
            **exp_params,
            structure_optimization='no',
            structure_optimization_params={}
        )


def test_classification_experimenter(get_test_classification_dataset):
    exp_params, tmp_path = get_test_classification_dataset
    experimenter = ClassificationExperimenter(
        **exp_params,
        structure_optimization='none',
        structure_optimization_params={}
    )
    experimenter.fit(1)
    assert os.path.exists(tmp_path.joinpath('models/Agricultural/ResNet18/trained.sd.pt'))
    classification_predict(experimenter)


def test_sfp_classification_experimenter(get_test_classification_dataset):
    exp_params, tmp_path = get_test_classification_dataset
    experimenter = ClassificationExperimenter(
        **exp_params,
        structure_optimization='SFP',
        structure_optimization_params=SFP_PARAMS
    )
    experimenter.fit(1)
    root = tmp_path.joinpath('models/Agricultural/ResNet18_SFP_P-0.50/')
    assert os.path.exists(root.joinpath('trained.sd.pt'))
    assert os.path.exists(root.joinpath('fine-tuning.sd.pt'))
    assert os.path.exists(root.joinpath('pruned.model.pt'))
    assert os.path.exists(root.joinpath('fine-tuned.model.pt'))
    classification_predict(experimenter)


def test_svd_channel_classification_experimenter(get_test_classification_dataset):
    exp_params, tmp_path = get_test_classification_dataset
    experimenter = ClassificationExperimenter(
        **exp_params,
        structure_optimization='SVD',
        structure_optimization_params={
            'decomposing_mode': 'channel',
            **SVD_PARAMS
        }
    )
    experimenter.fit(1)
    root = tmp_path.joinpath('models/Agricultural/ResNet18_SVD_channel_O-10.0_H-0.001000/')
    assert os.path.exists(root.joinpath('trained.sd.pt'))
    assert os.path.exists(root.joinpath('fine-tuning_e_0.9.sd.pt'))
    assert os.path.exists(root.joinpath('fine-tuned_e_0.9.model.pt'))
    classification_predict(experimenter)


def test_svd_spatial_classification_experimenter(get_test_classification_dataset):
    exp_params, tmp_path = get_test_classification_dataset
    experimenter = ClassificationExperimenter(
        **exp_params,
        structure_optimization='SVD',
        structure_optimization_params={
            'decomposing_mode': 'spatial',
            **SVD_PARAMS
        }
    )
    experimenter.fit(1)
    root = tmp_path.joinpath('models/Agricultural/ResNet18_SVD_spatial_O-10.0_H-0.001000/')
    assert os.path.exists(root.joinpath('trained.sd.pt'))
    assert os.path.exists(root.joinpath('fine-tuning_e_0.9.sd.pt'))
    assert os.path.exists(root.joinpath('fine-tuned_e_0.9.model.pt'))
    classification_predict(experimenter)


@pytest.fixture()
def get_test_detection_dataset(tmp_path):
    transform = Compose([ToTensor()])
    dataset = COCODataset(
        images_path=DATASETS_PATH + 'ALET10/test',
        json_path=DATASETS_PATH + 'ALET10/test.json',
        transform=transform)
    exp_params = {
        'dataset_name': 'ALET10',
        'train_dataset': dataset,
        'val_dataset': dataset,
        'num_classes': len(dataset.classes) + 1,
        'dataloader_params': {'batch_size': 1, 'num_workers': 2,
                              'collate_fn': collate_fn},
        'model_params': {'weights_backbone': None},
        # 'model_params': {'weights': 'DEFAULT'},
        'models_saving_path': tmp_path.joinpath('models'),
        'optimizer': torch.optim.SGD,
        'optimizer_params': {'lr': 0.0001, 'momentum': 0.9, 'weight_decay': 0.0005},
        'scheduler_params': {'step_size': 5, 'gamma': 0.5},
        'metric': 'map',
        'summary_path': tmp_path.joinpath('runs'),
        'summary_per_class': False,
        'gpu': False
    }
    yield exp_params, tmp_path


def detection_predict(experimenter):
    test_predict_path = DATASETS_PATH + 'ALET10/test'
    preds = experimenter.predict(test_predict_path)
    assert set(preds.keys()) == set(os.listdir(test_predict_path))
    for k, v in preds.items():
        assert set(v.keys()) == {'labels', 'boxes'}
    proba_preds = experimenter.predict_proba(test_predict_path)
    assert set(proba_preds.keys()) == set(os.listdir(test_predict_path))
    for k, v in proba_preds.items():
        assert set(v.keys()) == {'labels', 'boxes', 'scores'}


def test_fasterrcnn_experimenter(get_test_detection_dataset):
    exp_params, tmp_path = get_test_detection_dataset
    experimenter = FasterRCNNExperimenter(
        **exp_params,
        structure_optimization='none',
        structure_optimization_params={}
    )
    experimenter.fit(1)
    assert os.path.exists(tmp_path.joinpath('runs/ALET10/FasterR-CNN/ResNet50'))
    detection_predict(experimenter)


@pytest.mark.skip(reason='still in development')
def test_sfp_fasterrcnn_experimenter(get_test_detection_dataset):
    exp_params, tmp_path = get_test_detection_dataset
    experimenter = FasterRCNNExperimenter(
        **exp_params,
        structure_optimization='SFP',
        structure_optimization_params=SFP_PARAMS
    )
    experimenter.fit(1)
    root = tmp_path.joinpath('models/ALET10/FasterR-CNN/ResNet50_SFP_P-0.50/')
    assert os.path.exists(root.joinpath('pruned.model.pt'))
    assert os.path.exists(root.joinpath('fine-tuned.model.pt'))
    detection_predict(experimenter)


def test_svd_channel_fasterrcnn_experimenter(get_test_detection_dataset):
    exp_params, tmp_path = get_test_detection_dataset
    experimenter = FasterRCNNExperimenter(
        **exp_params,
        structure_optimization='SVD',
        structure_optimization_params={
            'decomposing_mode': 'channel',
            **SVD_PARAMS
        }
    )
    experimenter.fit(1)
    root = tmp_path.joinpath('models/ALET10/FasterR-CNN/ResNet50_SVD_channel_O-10.0_H-0.001000/')
    assert os.path.exists(root.joinpath('fine-tuned_e_0.9.model.pt'))
    detection_predict(experimenter)


def test_svd_spatial_fasterrcnn_experimenter(get_test_detection_dataset):
    exp_params, tmp_path = get_test_detection_dataset
    experimenter = FasterRCNNExperimenter(
        **exp_params,
        structure_optimization='SVD',
        structure_optimization_params={
            'decomposing_mode': 'spatial',
            **SVD_PARAMS
        }
    )
    experimenter.fit(1)
    root = tmp_path.joinpath(
        'models/ALET10/FasterR-CNN/ResNet50_SVD_spatial_O-10.0_H-0.001000/')
    assert os.path.exists(root.joinpath('fine-tuned_e_0.9.model.pt'))
    detection_predict(experimenter)
