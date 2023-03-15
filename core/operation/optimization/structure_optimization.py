"""
This module contains classes for CNN structure optimization.
"""

import os
from typing import Dict, List, Optional, Callable, Type

import torch
from fedot.core.log import default_log as Logger
from torchvision.models import ResNet

from core.architecture.abstraction.writers import WriterComposer, TFWriter, CSVWriter, \
    Writer
from core.architecture.experiment.nn_experimenter import NNExperimenter, FitParameters
from core.metrics.loss.svd_loss import OrthogonalLoss, HoyerLoss
from core.operation.decomposition.decomposed_conv import DecomposedConv2d
from core.operation.optimization.sfp_tools import create_percentage_filter_zeroing_fn, \
    create_energy_filter_zeroing_fn, prune_resnet, load_sfp_resnet_model
from core.operation.optimization.svd_tools import create_energy_svd_pruning, \
    decompose_module, load_svd_state_dict


class StructureOptimization:
    """Generalized class for model structure optimization."""

    def __init__(
            self,
            description,
    ) -> None:
        self.logger = Logger(self.__class__.__name__)
        self.description = description

    def fit(
            self,
            exp: NNExperimenter,
            params: FitParameters,
            ft_params: Optional[FitParameters] = None
    ) -> None:
        """Run model training with optimization.

        Args:
            exp: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        raise NotImplementedError

    def load_model(self, exp: NNExperimenter, state_dict_path: str) -> None:
        """Loads the optimized model into the experimenter.

        Args:
            exp: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            state_dict_path: Path to state_dict file.
        """
        raise NotImplementedError


class SVDOptimization(StructureOptimization):
    """Singular value decomposition for model structure optimization.

    Args:
        energy_thresholds: List of pruning hyperparameters.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        hoer_loss_factor: The hyperparameter by which the Hoyer loss function is
            multiplied.
        orthogonal_loss_factor: The hyperparameter by which the orthogonal loss
            function is multiplied.
    """

    def __init__(
            self,
            energy_thresholds: List[float],
            decomposing_mode: str = 'channel',
            hoer_loss_factor: float = 0.001,
            orthogonal_loss_factor: float = 100,
    ) -> None:
        super().__init__(
            description=(
                f"_SVD_{decomposing_mode}_O-{orthogonal_loss_factor}"
                f"_H-{hoer_loss_factor}"
            )
        )
        self.energy_thresholds = energy_thresholds
        self.decomposing_mode = decomposing_mode
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
            exp: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        exp.name += self.description
        self.finetuning = False
        writer = CSVWriter(os.path.join(params.summary_path, params.dataset_name, exp.name))
        size = {'size': exp.size_of_model(), 'params': exp.number_of_model_params()}
        writer.write_scores(phase='size', scores=size, x='default')
        self.logger.info(f"Default size: {size['size']:.2f} Mb")
        decompose_module(exp.model, self.decomposing_mode)
        exp.model.to(exp.device)
        size = {'size': exp.size_of_model(), 'params': exp.number_of_model_params()}
        writer.write_scores(phase='size', scores=size, x='decomposed')
        self.logger.info(f"SVD decomposed size: {size['size']:.2f} Mb")

        exp.fit(p=params, model_losses=self.loss)
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
            exp: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        self.finetuning = True
        models_path = os.path.join(params.models_path, params.dataset_name, exp.name)
        exp.save_model(os.path.join(models_path, 'trained'), state_dict=False)
        for e in self.energy_thresholds:
            str_e = f'e_{e}'
            exp.load_model(os.path.join(models_path, 'trained'), state_dict=False)
            exp.apply_func(
                func=create_energy_svd_pruning(energy_threshold=e),
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
                    model_losses=self.loss,
                    start_epoch=params.num_epochs
                )

    def loss(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Computes the orthogonal and the Hoer loss for the model.

        Args:
            model: CNN model with DecomposedConv layers.

        Returns:
            A dict ``{loss_name: loss_value}`` where loss_name is a string
                and loss_value is a floating point tensor with a single value.
        """
        losses = {'orthogonal_loss': self.orthogonal_loss(model)}
        if not self.finetuning:
            losses['hoer_loss'] = self.hoer_loss(model)
        return losses

    def load_model(self, exp: NNExperimenter, state_dict_path: str) -> None:
        """Loads the optimized model into the experimenter.

        Args:
            exp: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            state_dict_path: Path to state_dict file.
        """
        load_svd_state_dict(
            model=exp.model,
            state_dict_path=state_dict_path,
            decomposing_mode=self.decomposing_mode
        )
        exp.model.to(exp.device)
        self.logger.info("Model state dict loaded.")


class SFPOptimization(StructureOptimization):
    """Soft filter pruning for model structure optimization.

    Args:
        zeroing_mode: ``'percentage'`` or ``'energy'`` zeroing strategy of convolutional layer filters.
        zeroing_mode_params: Parameter dictionary passed to zeroing function.
        final_pruning_fn: Function implementing the final pruning of the model.
        model_class: The class of models to which the final pruning function is applicable.
    """

    def __init__(
            self,
            zeroing_mode: str,
            zeroing_mode_params: Dict,
            final_pruning_fn: Callable = prune_resnet,
            model_class: Type = ResNet
    ) -> None:
        description = f"_SFP"
        for k, v in zeroing_mode_params.items():
            description+=f"_{k}-{v}"
        super().__init__(
            description=description,
        )
        self.modes = {
            'percentage': create_percentage_filter_zeroing_fn,
            'energy': create_energy_filter_zeroing_fn
        }
        self.zeroing_fn = self.modes[zeroing_mode](**zeroing_mode_params)
        self.pruning_fn = final_pruning_fn
        self.model_class = model_class

    def fit(
            self,
            exp: NNExperimenter,
            params: FitParameters,
            ft_params: Optional[FitParameters] = None
    ) -> None:
        """Run model training with optimization.

        Args:
            exp: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        exp.name += self.description
        path = os.path.join(params.dataset_name, exp.name, 'train')
        model_path = os.path.join(params.models_path, path)
        writer = WriterComposer(
            path=os.path.join(params.summary_path, path),
            writers=[TFWriter, CSVWriter]
        )

        optimizer = params.optimizer(exp.model.parameters(), **params.optimizer_params)
        self.logger.info(f"{exp.name}, using device: {exp.device}")
        for epoch in range(1, params.num_epochs + 1):
            self.logger.info(f"Epoch {epoch}")
            train_scores = exp.train_loop(
                dataloader=params.train_dl,
                optimizer=optimizer
            )
            writer.write_scores('train', train_scores, epoch)
            exp.apply_func(
                func=self.zeroing_fn,
                condition=lambda x: isinstance(x, torch.nn.Conv2d)
            )
            val_scores = exp.val_loop(
                dataloader=params.val_dl,
                class_metrics=params.class_metrics
            )
            writer.write_scores('val', val_scores, epoch)
            exp.save_model_sd_if_best(val_scores=val_scores, file_path=model_path)
        exp.load_model(model_path)
        writer.close()

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
            exp: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            params: An object containing training parameters.
            ft_params: An object containing fine-tuning parameters for optimized model.
        """
        writer = CSVWriter(os.path.join(params.summary_path, params.dataset_name, exp.name))
        size = {'size': exp.size_of_model(), 'params': exp.number_of_model_params()}
        writer.write_scores(phase='size', scores=size, x='default')
        self.logger.info(f"Default size: {size['size']:.2f} Mb")

        exp.model = self.pruning_fn(model=exp.model)
        exp.model.to(exp.device)
        exp.save_model(
            os.path.join(params.models_path, params.dataset_name, exp.name, 'pruned')
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
            exp: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            state_dict_path: Path to state_dict file.
        """
        exp.model = load_sfp_resnet_model(
            model=exp.model,
            state_dict_path=state_dict_path
        )
        exp.model.to(exp.device)
        self.logger.info("Model loaded.")
