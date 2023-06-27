"""This module contains the class and functions for integrating the computer vision module into the framework API."""
import os
from typing import Callable, Dict, Optional, Tuple
from functools import partial
import logging
from urllib.error import URLError

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import ToTensor

from fedot_ind.core.architecture.abstraction.сheckers import parameter_value_check
from fedot_ind.core.architecture.datasets.object_detection_datasets import YOLODataset
from fedot_ind.core.architecture.datasets.prediction_datasets import PredictionFolderDataset
from fedot_ind.core.architecture.datasets.splitters import train_test_split
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters, \
    NNExperimenter, ObjectDetectionExperimenter, SegmentationExperimenter
from fedot_ind.core.operation.optimization.structure_optimization import SFPOptimization, StructureOptimization, \
    SVDOptimization


def get_classification_dataloaders(
        dataset_path: str,
        dataloader_params: Dict,
        transform: Callable
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Returns the training and validation data loaders for image classification (`torchvision.datasets.ImageFolder`).

    Args:
        dataset_path: Image folder path.
        dataloader_params: Parameter dictionary passed to `torch.utils.data.DataLoader`
        transform: The image transformation function passed to (`torchvision.datasets.ImageFolder`).

    Returns:
        `(train_dataloader, validation_dataloader, idx_to_class)`
    """
    ds = ImageFolder(root=dataset_path, transform=transform)
    train_ds, val_ds = train_test_split(ds)
    train_dl = DataLoader(train_ds, shuffle=True, **dataloader_params)
    val_dl = DataLoader(val_ds, **dataloader_params)
    idx_to_class = {idx: cls for cls, idx in ds.class_to_idx.items()}
    return train_dl, val_dl, idx_to_class


def get_object_detection_dataloaders(
        dataset_path: str,
        dataloader_params: Dict,
        transform: Callable
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Returns the training and validation data loaders for object detection (YOLO style).

    Args:
        dataset_path: Path to folder with images and their annotations.
        dataloader_params: Parameter dictionary passed to `torch.utils.data.DataLoader`
        transform: The image transformation function passed to dataset initialization.

    Returns:
        `(train_dataloader, validation_dataloader, idx_to_class)`
    """
    train_ds = YOLODataset(path=dataset_path, transform=transform)
    val_ds = YOLODataset(path=dataset_path, transform=transform, train=False)
    train_dl = DataLoader(train_ds, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), **dataloader_params)
    val_dl = DataLoader(val_ds, collate_fn=lambda x: tuple(zip(*x)), **dataloader_params)
    idx_to_class = {idx: cls for idx, cls in enumerate(train_ds.classes)}
    return train_dl, val_dl, idx_to_class


def get_segmentation_dataloaders(
        dataset_path: str,
        dataloader_params: Dict,
        transform: Callable
) -> Tuple[DataLoader, DataLoader]:
    return NotImplementedError


def classification_idx_to_class(
        preds: Dict[str, int],
        idx_to_class: Dict[int, str],
        proba: bool = False
) -> Dict[str, str]:
    return preds if proba else {img: idx_to_class[idx] for img, idx in preds.items()}


def detection_idx_to_class(
        preds: Dict[str, Dict],
        idx_to_class: Dict[int, str],
        proba: bool = False,
) -> Dict[str, Dict]:
    return {
        img: {
            k: (v if k != 'labels' else [idx_to_class[idx] for idx in v]) for k, v in pred.items()
        } for img, pred in preds.items()
    }


CV_TASKS = {
    'image_classification': {
        'experimenter': ClassificationExperimenter,
        'model': resnet18,
        'data': get_classification_dataloaders,
        'idx_to_class': classification_idx_to_class,
    },
    'object_detection': {
        'experimenter': ObjectDetectionExperimenter,
        'model': ssdlite320_mobilenet_v3_large,
        'data': get_object_detection_dataloaders,
        'idx_to_class': detection_idx_to_class,
    },
    'semantic_segmentation': {
        'experimenter': SegmentationExperimenter,
        'model': deeplabv3_resnet50,
        'data': get_segmentation_dataloaders,
        'idx_to_class': lambda x, y, z: x
    }
}

OPTIMIZATIONS = {
    'svd': SVDOptimization,
    'sfp': SFPOptimization,
}


class CVExperimenter:
    """
    This class is used to integrate the computer vision module into the framework API.

    Args:
        kwargs: Keyword arguments for the experimenter object.

    Attributes:
        task: Computer vision task name.
        path: Path to the folder where the results of the experiment will be saved.
        optim: Optimization object.
        exp: Experimenter object.
        val_dl: Validation data loader.

    """
    def __init__(self, kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.task: str = kwargs.pop('task')
        self.path: str = kwargs.pop('output_folder')
        self.optim: Optional[StructureOptimization] = None
        if 'model' in kwargs.keys():
            assert isinstance(kwargs['model'], torch.nn.Module), 'Model must be an instance of torch.nn.Module'
        else:
            assert 'num_classes' in kwargs.keys(), 'It is necessary to pass the number of classes or the model object'

            num_classes = kwargs.pop('num_classes')

            try:
                kwargs['model'] = CV_TASKS[self.task]['model'](num_classes=num_classes)
            except URLError:
                # Fix for possible SSL error
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                kwargs['model'] = CV_TASKS[self.task]['model'](num_classes=num_classes)

        if 'optimization' in kwargs.keys():
            optimization = kwargs.pop('optimization')
            parameter_value_check('optimization', optimization, set(OPTIMIZATIONS.keys()))
            opt_params = kwargs.pop('optimization_params', {})
            self.optim = OPTIMIZATIONS[optimization](**opt_params)

        if 'device' not in kwargs.keys():
            kwargs['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.exp: NNExperimenter = CV_TASKS[self.task]['experimenter'](**kwargs)
        self.val_dl: Optional[DataLoader] = None
        self.idx_to_class: Optional[Dict[int: str]] = None
        self.logger.info(f'{type(self.exp).__name__} initialised')

    def fit(self, dataset_path: str, **kwargs):
        """
        Starts training the model.

        Args:
            dataset_path: Path to dataset.

        Returns:
            Trained model (`torch.nn.Module`).
        """
        self.logger.info('Dataset preparing')
        train_dl, val_dl, idx_to_class = CV_TASKS[self.task]['data'](
            dataset_path=dataset_path,
            dataloader_params=kwargs.pop('dataloader_params', {'batch_size': 8, 'num_workers': 4}),
            transform=kwargs.pop('transform', ToTensor())
        )
        self.val_dl = val_dl
        self.idx_to_class = idx_to_class

        ds_name = kwargs.pop('dataset_name', dataset_path.split('/')[-1])
        ft_params = kwargs.pop('finetuning_params', {})
        num_epoch = kwargs.pop('num_epochs', 50)
        if 'lr_scheduler' not in kwargs.keys():
            kwargs['lr_scheduler'] = partial(
                ReduceLROnPlateau,
                factor=0.2,
                mode='max',
                patience=int(num_epoch / 10),
                verbose=True
            )

        fit_parameters = FitParameters(
            dataset_name=ds_name,
            train_dl=train_dl,
            val_dl=val_dl,
            num_epochs=num_epoch,
            models_path=os.path.join(self.path, 'models'),
            summary_path=os.path.join(self.path, 'summary'),
            **kwargs
        )
        if self.optim is None:
            self.exp.fit(p=fit_parameters)
        else:
            ft_num_epoch = ft_params.pop('num_epochs', 10)
            if 'lr_scheduler' not in ft_params.keys():
                ft_params['lr_scheduler'] = partial(
                    ReduceLROnPlateau,
                    factor=0.2,
                    mode='max',
                    patience=int(ft_num_epoch / 10),
                    verbose=True
                )
            ft_parameters = FitParameters(
                dataset_name=ds_name,
                train_dl=train_dl,
                val_dl=val_dl,
                num_epochs=ft_num_epoch,
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
        predictions = self.exp.predict(dataloader=dataloader)
        return CV_TASKS[self.task]['idx_to_class'](predictions, self.idx_to_class)

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
        predictions = self.exp.predict_proba(dataloader=dataloader)
        return CV_TASKS[self.task]['idx_to_class'](predictions, self.idx_to_class, True)

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
            self.optim.load_model(self.exp, path)

    def save(self, path) -> None:
        """
        Save the model state dict.

        Args:
            path: Path to the model state dict.
        """
        self.exp.save_model(file_path=path)

    def save_metrics(self, **kwargs) -> None:
        """Displays an informational message with the metrics save path"""
        print(f'All metrics were saved during training in {self.path}')

    def save_predict(self, **kwargs) -> None:
        raise NotImplementedError
