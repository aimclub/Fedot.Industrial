import os

from torchvision.transforms import ToTensor, Resize, Compose

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.architecture.utils.utils import PROJECT_PATH


DATASETS_PATH = os.path.abspath(PROJECT_PATH + '/../tests/data/datasets')


def test_image_classification():
    fed = FedotIndustrial(task='image_classification', num_classes=3)
    fed.fit(
        dataset_path=os.path.join(DATASETS_PATH, 'Agricultural/train'),
        transform=Compose([ToTensor(), Resize((256, 256))]),
        num_epochs=2
    )
    fed.predict(
        data_path=os.path.join(DATASETS_PATH, 'Agricultural/val'),
        transform=Compose([ToTensor(), Resize((256, 256))]),
    )
    fed.predict_proba(
        data_path=os.path.join(DATASETS_PATH, 'Agricultural/val'),
        transform=Compose([ToTensor(), Resize((256, 256))]),
    )
    fed.get_metrics()
    fed.save_metrics()


def test_image_classification_svd():
    fed = FedotIndustrial(
        task='image_classification',
        num_classes=3,
        optimization='svd',
        optimization_params={'energy_thresholds': [0.9]}
    )
    fed.fit(
        dataset_path=os.path.join(DATASETS_PATH, 'Agricultural/train'),
        transform=Compose([ToTensor(), Resize((256, 256))]),
        num_epochs=2,
        finetuning_params={'num_epochs': 1}
    )


def test_object_detection():
    fed = FedotIndustrial(task='object_detection', num_classes=3)
    fed.fit(
        dataset_path=os.path.join(DATASETS_PATH, 'minerals'),
        num_epochs=2
    )
    fed.predict(data_path=os.path.join(DATASETS_PATH, 'minerals'))
    fed.predict_proba(data_path=os.path.join(DATASETS_PATH, 'minerals'))
    fed.get_metrics()

