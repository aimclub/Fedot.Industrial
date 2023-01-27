import os
from typing import Dict, List

import torch
from fedot.core.log import default_log as Logger
from torch.utils.tensorboard import SummaryWriter

from core.architecture.experiment.nn_experimenter import NNExperimenter, FitParameters
from core.metrics.loss.svd_loss import OrthogonalLoss, HoyerLoss
from core.operation.decomposition.decomposed_conv import DecomposedConv2d
from core.operation.optimization.sfp_tools import zerolize_filters
from core.operation.optimization.svd_tools import energy_threshold_pruning, \
    decompose_module


def write_scores(
        writer: SummaryWriter,
        phase: str,
        scores: Dict[str, float],
        x: int,
):
    """Write scores from dictionary by SummaryWriter.

    Args:
        writer: SummaryWriter object for writing scores.
        phase: Experiment phase for grouping records, e.g. 'train'.
        scores: Dictionary {metric_name: value}.
        x: The independent variable.
    """
    for key, score in scores.items():
        writer.add_scalar(f"{phase}/{key}", score, x)


class StructureOptimization:
    """Generalized class for model structure optimization."""

    def __init__(
            self,
            description,
            finetuning_epochs: int
    ) -> None:
        self.logger = Logger(self.__class__.__name__)
        self.description = description
        self.finetuning_epochs = finetuning_epochs
        self.finetuning = False

    def fit(
            self,
            exp: NNExperimenter,
            params: FitParameters

    ) -> None:
        """Run experiment.

        Args:
            experimenter: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            num_epochs: Number of epochs.
        """
        pass


class SVDOptimization(StructureOptimization):
    """Singular value decomposition for model structure optimization.

    Args:
        energy_thresholds: List of pruning hyperparameters.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method
            (default: ``'channel'``).
        hoer_loss_factor: The hyperparameter by which the Hoyer loss function is
            multiplied (default: ``0.001``).
        orthogonal_loss_factor: The hyperparameter by which the orthogonal loss
            function is multiplied (default: ``100``).
        finetuning_epochs: Number of fine-tuning epochs (default: ``1``).
    """

    def __init__(
            self,
            energy_thresholds: List[float],
            decomposing_mode: str = 'channel',
            hoer_loss_factor: float = 0.001,
            orthogonal_loss_factor: float = 100,
            finetuning_epochs: int = 1,
    ) -> None:
        super().__init__(
            description=(
                f"_SVD_{decomposing_mode}_O-{orthogonal_loss_factor:.1f}"
                f"_H-{hoer_loss_factor:.6f}"
            ),
            finetuning_epochs=finetuning_epochs
        )
        self.energy_thresholds = energy_thresholds
        self.decomposing_mode = decomposing_mode
        self.hoer_loss = HoyerLoss(hoer_loss_factor)
        self.orthogonal_loss = OrthogonalLoss(orthogonal_loss_factor)

    def fit(self, exp: NNExperimenter, params: FitParameters) -> None:
        """Run experiment.

        Args:
            exp: An instance of the experimenter class, e.g. ``ClassificationExperimenter``.
            num_epochs: Number of epochs.
        """

        exp.name += self.description
        decompose_module(exp.model, self.decomposing_mode)
        exp.model.to(exp.device)
        self.logger.info(f"SVD decomposed size: {exp.size_of_model():.2f} Mb")

        exp.fit(p=params, model_losses=self.loss)
        self.optimize(exp=exp, params=params)

    def optimize(self, exp: NNExperimenter, params: FitParameters) -> None:
        self.finetuning = True
        epoch = params.num_epochs
        params.num_epochs = self.finetuning_epochs
        model_path = os.path.join(
            params.models_path, params.dataset_name, exp.name, 'trained'
        )
        exp.save_model(model_path, state_dict=False)
        for e in self.energy_thresholds:
            str_e = f'e_{e}'
            exp.load_model(model_path, state_dict=False)
            exp.apply_func(
                func=energy_threshold_pruning,
                func_params={'energy_threshold': e},
                condition=lambda x: isinstance(x, DecomposedConv2d)
            )
            self.logger.info(f"pruning with e={e}, size: {exp.size_of_model():.2f} Mb")
            exp.save_model(os.path.join(params.models_path, exp.name, str_e))
            exp.best_score = 0
            exp.fit(p=params, phase=str_e, model_losses=self.loss, start_epoch=epoch)

    def loss(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        losses = {'orthogonal_loss': self.orthogonal_loss(model)}
        if not self.finetuning:
            losses['hoer_loss'] = self.hoer_loss(model)
        return losses


class SFPOptimization(StructureOptimization):
    """Soft filter pruning for model structure optimization.

    Args:
        pruning_ratio: Pruning hyperparameter, percentage of pruned filters.
        finetuning_epochs: Number of fine-tuning epochs (default: ``1``).
    """

    def __init__(
            self,
            pruning_ratio: float,
            finetuning_epochs: int = 1,
    ) -> None:
        super().__init__(
            description=f"_SFP_P-{pruning_ratio:.2f}",
            finetuning_epochs=finetuning_epochs
        )
        self.pruning_ratio = pruning_ratio

    def fit(self, exp: NNExperimenter, params: FitParameters) -> None:
        """Run experiment.

        Args:
            exp: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            num_epochs: Number of epochs.
        """
        exp.name += self.description
        path = os.path.join(params.dataset_name, exp.name, 'train')
        model_path = os.path.join(params.models_path, path)
        writer = SummaryWriter(os.path.join(params.summary_path, path))

        optimizer = params.optimizer(exp.model.parameters(), **params.optimizer_params)
        self.logger.info(f"{exp.name}, using device: {exp.device}")
        for epoch in range(1, params.num_epochs + 1):
            self.logger.info(f"Epoch {epoch}")
            train_scores = exp.train_loop(
                dataloader=params.train_dl,
                optimizer=optimizer
            )
            write_scores(writer, 'train', train_scores, epoch)
            exp.apply_func(
                func=zerolize_filters,
                func_params={'pruning_ratio': self.pruning_ratio},
                condition=lambda x: isinstance(x, torch.nn.Conv2d)
            )
            val_scores = exp.val_loop(
                dataloader=params.val_dl,
                class_metrics=params.class_metrics
            )
            write_scores(writer, 'val', val_scores, epoch)
            exp.save_model_sd_if_best(val_scores=val_scores, file_path=model_path)
        exp.load_model(model_path)
        writer.close()

