"""
This module contains classes for CNN structure optimization.
"""

import logging
import os
from typing import Callable, Dict, List, Optional, Type
from abc import ABC, abstractmethod
from functools import partial

import torch
from torchvision.models import ResNet

from fedot_ind.core.architecture.abstraction.writers import WriterComposer, TFWriter, CSVWriter, Writer
from fedot_ind.core.architecture.experiment.nn_experimenter import NNExperimenter, FitParameters
from fedot_ind.core.metrics.loss.svd_loss import OrthogonalLoss, HoyerLoss
from fedot_ind.core.operation.decomposition.decomposed_conv import DecomposedConv2d
from fedot_ind.core.operation.optimization.sfp_tools import percentage_filter_zeroing, energy_filter_zeroing, prune_resnet, load_sfp_resnet_model
from fedot_ind.core.operation.optimization.svd_tools import energy_svd_pruning, decompose_module, load_svd_state_dict


class StructureOptimization(ABC):
    """Generalized class for model structure optimization."""

    def __init__(
            self,
            description,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.description = description

    @abstractmethod
    def fit(
            self,
            exp: NNExperimenter,
            params: FitParameters,
            ft_params: Optional[FitParameters] = None
    ) -> None:
        """Run model training with optimization.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, exp: NNExperimenter, state_dict_path: str) -> None:
        """Loads the optimized model into the experimenter.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            state_dict_path: Path to state_dict file.
        """
        raise NotImplementedError


class SVDOptimization(StructureOptimization):
    """Singular value decomposition for model structure optimization.

    Args:
        energy_thresholds: List of pruning hyperparameters.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        forward_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
        hoer_loss_factor: The hyperparameter by which the Hoyer loss function is
            multiplied.
        orthogonal_loss_factor: The hyperparameter by which the orthogonal loss
            function is multiplied.
    """

    def __init__(
            self,
            energy_thresholds: List[float] = [0.9, 0.95, 0.99, 0.999],
            decomposing_mode: str = 'channel',
            forward_mode: str = 'one_layer',
            hoer_loss_factor: float = 0.1,
            orthogonal_loss_factor: float = 10,
    ) -> None:
        super().__init__(
            description=(
                f"_SVD_{decomposing_mode}_O-{orthogonal_loss_factor}"
                f"_H-{hoer_loss_factor}"
            )
        )
        self.energy_thresholds = energy_thresholds
        self.decomposing_mode = decomposing_mode
        self.forward_mode = forward_mode
        self.hoer_loss = HoyerLoss(hoer_loss_factor)
        self.orthogonal_loss = OrthogonalLoss(orthogonal_loss_factor)
        self.finetuning = False

    def fit(
            self,
            exp: NNExperimenter,
            params: FitParameters,
            ft_params: Optional[FitParameters] = None
    ) -> None:
        """Run model training with optimization.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        exp.name += self.description
        self.finetuning = False
        writer = CSVWriter(os.path.join(params.summary_path, params.dataset_name, exp.name, params.description))
        size = {'size': exp.size_of_model(), 'params': exp.number_of_model_params()}
        writer.write_scores(phase='size', scores=size, x='default')
        self.logger.info(f"Default size: {size['size']:.2f} Mb")
        decompose_module(exp.model, self.decomposing_mode, forward_mode=self.forward_mode)
        exp.model.to(exp.device)
        size = {'size': exp.size_of_model(), 'params': exp.number_of_model_params()}
        writer.write_scores(phase='size', scores=size, x='decomposed')
        self.logger.info(f"SVD decomposed size: {size['size']:.2f} Mb")

        exp.fit(p=params, model_losses=self.__loss)
        self.optimize(exp=exp, params=params, ft_params=ft_params, writer=writer)

    def optimize(
            self,
            exp: NNExperimenter,
            params: FitParameters,
            ft_params: Optional[FitParameters],
            writer: Writer
    ) -> None:
        """Prunes the trained model at the given thresholds.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
            writer: Object for recording metrics.
        """
        self.finetuning = True
        models_path = os.path.join(params.models_path, params.dataset_name, exp.name, params.description)
        exp.save_model(os.path.join(models_path, 'trained'), state_dict=False)
        for e in self.energy_thresholds:
            str_e = f'e_{e}'
            exp.load_model(os.path.join(models_path, 'trained'), state_dict=False)
            exp._apply_function(
                func=partial(energy_svd_pruning, energy_threshold=e),
                condition=lambda x: isinstance(x, DecomposedConv2d)
            )
            exp.save_model(os.path.join(models_path, str_e))
            scores = {'size': exp.size_of_model(), 'params': exp.number_of_model_params()}
            writer.write_scores(phase='size', scores=scores, x=str_e)
            scores.update(exp.val_loop(params.val_dl, params.class_metrics))
            writer.write_scores(phase='pruning', scores=scores, x=str_e)
            self.logger.info((f"pruning with {e=}, size: {scores['size']:.2f} Mb, "
                             f"{exp.metric}: {scores[exp.metric]:.4f}"))
            if ft_params is not None:
                exp.best_score = scores[exp.metric]
                exp.fit(
                    p=ft_params,
                    phase=str_e,
                    model_losses=self.__loss,
                    start_epoch=params.num_epochs,
                    initial_validation=True
                )

    def __loss(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Computes the orthogonal and the Hoer loss for the model.

        Args:
            model: CNN model with DecomposedConv layers.

        Returns:
            A dict ``{loss_name: loss_value}``
                where loss_name is a string and loss_value is a floating point tensor with a single value.
        """
        losses = {'orthogonal_loss': self.orthogonal_loss(model)}
        if not self.finetuning:
            losses['hoer_loss'] = self.hoer_loss(model)
        return losses

    def load_model(self, exp: NNExperimenter, state_dict_path: str) -> None:
        """Loads the optimized model into the experimenter.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            state_dict_path: Path to state_dict file.
        """
        load_svd_state_dict(
            model=exp.model,
            state_dict_path=state_dict_path,
            decomposing_mode=self.decomposing_mode,
            forward_mode=self.forward_mode
        )
        exp.model.to(exp.device)
        self.logger.info("Model state dict loaded.")


class SFPOptimization(StructureOptimization):
    """Soft filter pruning for model structure optimization.

    Args:
        zeroing_fn: Partially initialized filter zeroing function.
        model_class: The class of models to which the final pruning function is applicable.
        final_pruning_fn: Function implementing the model final pruning of the ``model_class``.
        load_model_fn: Function implementing the model loading of the ``model_class``.
    """

    def __init__(
            self,
            zeroing_fn: partial = partial(percentage_filter_zeroing, pruning_ratio=0.2),
            model_class: Type = ResNet,
            final_pruning_fn: Callable = prune_resnet,
            load_model_fn: Callable = load_sfp_resnet_model
    ) -> None:
        description = f"_SFP"
        for k, v in zeroing_fn.keywords.items():
            description += f"_{k}-{v}"
        super().__init__(
            description=description,
        )
        self.zeroing_fn = zeroing_fn
        self.pruning_fn = final_pruning_fn
        self.model_class = model_class
        self.load_model_fn = load_model_fn

    def fit(
            self,
            exp: NNExperimenter,
            params: FitParameters,
            ft_params: Optional[FitParameters] = None
    ) -> None:
        """Run model training with optimization.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        exp.name += self.description
        exp.fit(p=params, filter_pruning=dict(func=self.zeroing_fn, condition=lambda x: isinstance(x, torch.nn.Conv2d)))
        if isinstance(exp.model, self.model_class):
            self.final_pruning(exp=exp, params=params, ft_params=ft_params)
        else:
            self.logger.warning(f'Final pruning function "{self.pruning_fn.__name__}"' +
                                f'supports only {self.model_class.__name__} models.')

    def final_pruning(
            self,
            exp: NNExperimenter,
            params: FitParameters,
            ft_params: Optional[FitParameters],
    ) -> None:
        """Final model pruning after training.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        writer = CSVWriter(os.path.join(params.summary_path, params.dataset_name, exp.name, params.description))
        size = {'size': exp.size_of_model(), 'params': exp.number_of_model_params()}
        writer.write_scores(phase='size', scores=size, x='default')
        self.logger.info(f"Default size: {size['size']:.2f} Mb")

        exp.model = self.pruning_fn(model=exp.model)
        exp.model.to(exp.device)
        exp.save_model(
            os.path.join(params.models_path, params.dataset_name, exp.name, params.description, 'pruned')
        )
        size = {'size': exp.size_of_model(), 'params': exp.number_of_model_params()}
        writer.write_scores(phase='size', scores=size, x='pruned')
        self.logger.info(f"Pruned size: {size['size']:.2f} Mb")

        exp.best_score = 0
        if ft_params is not None:
            exp.fit(p=ft_params, phase='pruned', start_epoch=params.num_epochs)

    def load_model(self, exp: NNExperimenter, state_dict_path: str) -> None:
        """Loads the optimized model into the experimenter.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            state_dict_path: Path to state_dict file.
        """
        try:
            exp.model = self.load_model_fn(model=exp.model, state_dict_path=state_dict_path)
            self.logger.info("Model state dict loaded.")
        except Exception:
            self.logger.error(f'Loading function "{self.load_model_fn.__name__}"' +
                              f"can't load {self.model_class.__name__} model.")
