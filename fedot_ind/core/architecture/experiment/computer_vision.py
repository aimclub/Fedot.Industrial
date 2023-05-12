"""This module contains the class and functions for integrating the computer vision module into the framework API."""
from typing import Optional, Callable, Tuple, Dict
import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from fedot_ind.core.architecture.abstraction.сheckers import parameter_value_check
from fedot_ind.core.architecture.experiment.nn_experimenter import NNExperimenter, ClassificationExperimenter, \
    ObjectDetectionExperimenter, SegmentationExperimenter, FitParameters
from fedot_ind.core.operation.optimization.structure_optimization import StructureOptimization, SVDOptimization, \
    SFPOptimization
from fedot_ind.core.architecture.datasets.splitters import train_test_split
from fedot_ind.core.architecture.datasets.prediction_datasets import PredictionFolderDataset
from fedot_ind.core.architecture.datasets.object_detection_datasets import YOLODataset


def get_classification_dataloaders(
        dataset_path: str,
        dataloader_params: Dict,
        transform: Callable
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns the training and validation data loaders for image classification (`torchvision.datasets.ImageFolder`).

    Args:
        dataset_path: Image folder path.
        dataloader_params: Parameter dictionary passed to `torch.utils.data.DataLoader`
        transform: The image transformation function passed to (`torchvision.datasets.ImageFolder`).

    Returns:
        `(train_dataloader, validation_dataloader)`
    """
    ds = ImageFolder(root=dataset_path, transform=transform)
    train_ds, val_ds = train_test_split(ds)
    train_dl = DataLoader(train_ds, shuffle=True, **dataloader_params)
    val_dl = DataLoader(val_ds, **dataloader_params)
    return train_dl, val_dl


def get_object_detection_dataloaders(
        dataset_path: str,
        dataloader_params: Dict,
        transform: Callable
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns the training and validation data loaders for object detection (YOLO style).

    Args:
        dataset_path: Path to folder with images and their annotations.
        dataloader_params: Parameter dictionary passed to `torch.utils.data.DataLoader`
        transform: The image transformation function passed to dataset initialization.

    Returns:
        `(train_dataloader, validation_dataloader)`
    """
    ds = YOLODataset(image_folder=dataset_path, transform=transform)
    n = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [n, len(ds) - n], generator=torch.Generator().manual_seed(31))
    train_dl = DataLoader(train_ds, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), **dataloader_params)
    val_dl = DataLoader(val_ds, collate_fn=lambda x: tuple(zip(*x)), **dataloader_params)
    return train_dl, val_dl


def get_segmentation_dataloaders(
        dataset_path: str,
        dataloader_params: Dict,
        transform: Callable
) -> Tuple[DataLoader, DataLoader]:
    return NotImplementedError


CV_TASKS = {
    'image_classification': {
        'experimenter': ClassificationExperimenter,
        'model': resnet18,
        'data': get_classification_dataloaders,
    },
    'object_detection': {
        'experimenter': ObjectDetectionExperimenter,
        'model': ssdlite320_mobilenet_v3_large,
        'data': get_object_detection_dataloaders
    },
    'semantic_segmentation': {
        'experimenter': SegmentationExperimenter,
        'model': deeplabv3_resnet50,
        'data': get_segmentation_dataloaders
    }
}

OPTIMIZATIONS = {
    'svd': SVDOptimization,
    'sfp': SFPOptimization,
}

class CVExperimenter:
    def __init__(self, kwargs):
        """
        Initializes the experimenter object according to the given computer vision task.
        """
        self.task: str = kwargs.pop('task')
        self.path: str = kwargs.pop('output_folder')
        self.optim: Optional[StructureOptimization] = None
        if 'model' in kwargs.keys():
            assert isinstance(kwargs['model'], torch.nn.Module), 'Model must be an instance of torch.nn.Module'
        else:
            assert 'num_classes' in kwargs.keys(), 'It is necessary to pass the number of classes or the model object'
            kwargs['model'] = CV_TASKS[self.task]['model'](num_classes=kwargs.pop('num_classes'))

        if 'optimization' in kwargs.keys():
            optimization = kwargs.pop('optimization')
            parameter_value_check('optimization', optimization, set(OPTIMIZATIONS.keys()))
            opt_params = kwargs.pop('optimization_params', {})
            self.optim = OPTIMIZATIONS[optimization](**opt_params)

        self.exp: NNExperimenter = CV_TASKS[self.task]['experimenter'](**kwargs)
        self.val_dl: Optional[DataLoader] = None

    def fit(self, dataset_path: str, **kwargs):
        """
        Starts training the model.

        Args:
            dataset_path: Path to dataset.

        Returns:
            Trained model (`torch.nn.Module`).
        """

        train_dl, val_dl = CV_TASKS[self.task]['data'](
            dataset_path=dataset_path,
            dataloader_params=kwargs.pop('dataloader_params', {'batch_size': 8, 'num_workers': 4}),
            transform=kwargs.pop('transform', ToTensor())
        )
        self.val_dl = val_dl

        ds_name = kwargs.pop('dataset_name', dataset_path.split('/')[-1])
        ft_params = kwargs.pop('finetuning_params', {})
        fit_parameters = FitParameters(
            dataset_name=ds_name,
            train_dl=train_dl,
            val_dl=val_dl,
            num_epochs=kwargs.pop('num_epochs', 50),
            models_path=os.path.join(self.path, 'models'),
            summary_path=os.path.join(self.path, 'summary'),
            **kwargs
        )
        if self.optim is None:
            self.exp.fit(p=fit_parameters)
        else:
            ft_parameters = FitParameters(
                dataset_name=ds_name,
                train_dl=train_dl,
                val_dl=val_dl,
                num_epochs=ft_params.pop('num_epochs', 10),
                models_path=os.path.join(self.path, 'models'),
                summary_path=os.path.join(self.path, 'summary'),
                **ft_params
            )
            self.optim.fit(exp=self.exp, params=fit_parameters, ft_params=ft_parameters)

        return self.exp.model

    def predict(self, data_path, **kwargs):
        """
        Computes predictions for data in folder.

        Args:
            data_path: Path to image folder.

        Returns:
            Predictions dictionary.
        """
        dataset = PredictionFolderDataset(
            image_folder=data_path,
            transform=kwargs.pop('transform', ToTensor())
        )
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=(lambda x: tuple(zip(*x))) if self.task == 'object_detection' else None,
            **kwargs
        )
        return self.exp.predict(dataloader=dataloader)

    def predict_proba(self, data_path, **kwargs):
        """
        Computes probability predictions for data in folder.

        Args:
            data_path: Path to image folder.

        Returns:
            Predictions dictionary.
        """
        dataset = PredictionFolderDataset(
            image_folder=data_path,
            transform=kwargs.pop('transform', ToTensor())
        )
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=(lambda x: tuple(zip(*x))) if self.task == 'object_detection' else None,
            **kwargs
        )
        return self.exp.predict_proba(dataloader=dataloader)

    def get_metrics(self, **kwargs):
        """
        Runs model validation and returns metric values.

        Returns:
            Metrics dictionary
        """
        if self.val_dl is not None:
            return self.exp.val_loop(dataloader=self.val_dl, **kwargs)
        else:
            raise AttributeError('No validation data. Call the fit method before.')

    def load(self, path) -> None:
        """
        Load the model state dict.

        Args:
            path: Path to the model state dict.
        """
        if self.optim is None:
            self.exp.load_model(path)
        else:
            self.optim.load_model(path)

    def save_metrics(self, **kwargs) -> None:
        """Displays an informational message with the metrics save path"""
        print(f'All metrics were saved during training in {self.path}')

    def save_predict(self, **kwargs) -> None:
        raise NotImplementedError
