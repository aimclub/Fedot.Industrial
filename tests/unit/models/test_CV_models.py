from core.architecture.datasets.classification_datasets import CustomClassificationDataset
from core.architecture.experiment.CVModule import ClassificationExperimenter
from core.architecture.utils.Testing import *
from core.models.cnn.sfp_models import *
from core.models.cnn.classification_models import *
import pytest
import os

from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum

GENERAL_CNN_MODEL_PARAMS = {'dataloader_params': {'batch_size': 1},
                            'model_params': {},
                            'optimizer_params': {},
                            'loss_params': {},
                            'metric': 'f1',
                            'structure_optimization_params': {},
                            'summary_per_class': True}


def get_image_classification_data():
    """ Method for loading data with images in .npy format (training_data.npy, training_labels.npy,
    test_data.npy, test_labels.npy) that are used in tests.This npy files are a truncated version
    of the MNIST dataset, that contains only 10 first images.
    """
    test_data_path = '../../data/test_data.npy'
    test_labels_path = '../../data/test_labels.npy'
    train_data_path = '../../data/training_data.npy'
    train_labels_path = '../../data/training_labels.npy'

    test_file_path = str(os.path.dirname(__file__))
    training_path_features = os.path.join(test_file_path, train_data_path)
    training_path_labels = os.path.join(test_file_path, train_labels_path)
    test_path_features = os.path.join(test_file_path, test_data_path)
    test_path_labels = os.path.join(test_file_path, test_labels_path)
    task = Task(TaskTypesEnum.classification)

    x_train, y_train = training_path_features, training_path_labels
    x_test, y_test = test_path_features, test_path_labels

    dataset_to_train = InputData.from_image(images=x_train,
                                            labels=y_train,
                                            task=task)
    dataset_to_validate = InputData.from_image(images=x_test,
                                               labels=y_test,
                                               task=task)

    return dataset_to_train, dataset_to_validate


@pytest.fixture()
def clf_models_list():
    return CLF_MODELS_ONE_CHANNEL


@pytest.fixture()
def sfp_models_list():
    return SFP_MODELS


def test_clf_models(clf_model_list):
    for clf_model in clf_model_list.keys():
        dataset_to_train, dataset_to_validate = get_image_classification_data()
        train_dataset = CustomClassificationDataset(images=dataset_to_train.features, targets=dataset_to_train.target)
        model = ClassificationExperimenter(train_dataset=train_dataset,
                                           val_dataset=train_dataset,
                                           num_classes=10,
                                           model=clf_model,
                                           **GENERAL_CNN_MODEL_PARAMS)
        TestModule = ModelTestingModule(model=model)
        train_feats_MNIST = TestModule.fit(timeout=2)
        test_feats_MNIST = TestModule.predict(dataset_to_validate)
        assert train_feats_MNIST is not None
        assert test_feats_MNIST is not None


def test_sfp_models(window_feature_generators_list):
    for sfp_model in window_feature_generators_list.values():
        model = sfp_model()
        TestModule = ModelTestingModule(model=model)
        train_feats_earthquakes, test_feats_earthquakes = TestModule.extract_from_binary(dataset_name='Earthquakes')
        train_feats_lightning7, test_feats_lightning7 = TestModule.extract_from_multi_class(dataset_name='Lightning7')
        assert train_feats_lightning7 is not None
        assert test_feats_lightning7 is not None

        assert train_feats_earthquakes is not None
        assert test_feats_earthquakes is not None


# if __name__ == '__main__':
#     test_clf_models(CLF_MODELS_ONE_CHANNEL)
#     test_sfp_models(SFP_MODELS)
