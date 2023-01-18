import os
from typing import Type, Dict, List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.metrics.loss.svd_loss import OrthogonalLoss, HoyerLoss
from core.operation.optimization.svd_tools import prune_sdv_model, \
    energy_threshold_pruning, decompose_module


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
        self.description = description
        self.finetuning_epochs = finetuning_epochs
        self.finetuning = False

    def fit(
            self,
            experimenter,
            description: str,
            train_dl: DataLoader,
            val_dl: DataLoader,
            num_epochs: int,
            optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
            models_path: str = 'models',
            summary_path: str = 'summary',
            class_metrics: bool = False,
            optimizer_params: Dict = {},
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

    def fit(
            self,
            experimenter,
            description: str,
            train_dl: DataLoader,
            val_dl: DataLoader,
            num_epochs: int,
            optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
            models_path: str = 'models',
            summary_path: str = 'summary',
            class_metrics: bool = False,
            optimizer_params: Dict = {},
    ) -> None:
        """Run experiment.

        Args:
            experimenter: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
            num_epochs: Number of epochs.
        """

        description += self.description
        decompose_module(experimenter.model, self.decomposing_mode)
        print(f"SVD decomposed size: {experimenter.size_of_model():.2f} Mb")
        print(f"{description}, using device: {experimenter.device}")

        experimenter.run_epochs(
            writer=SummaryWriter(os.path.join(summary_path, description, 'train')),
            train_dl=train_dl,
            val_dl=val_dl,
            num_epochs=num_epochs,
            optimizer=optimizer(experimenter.model.parameters(), **optimizer_params),
            model_path=os.path.join(models_path, description, 'trained'),
            class_metrics=class_metrics,
            model_losses=self.loss
        )
        self.optimize(
            experimenter=experimenter,
            train_dl=train_dl,
            val_dl=val_dl,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            models_path=os.path.join(models_path, description),
            summary_path=os.path.join(summary_path, description),
            class_metrics=class_metrics
        )

    def optimize(
            self,
            experimenter,
            train_dl: DataLoader,
            val_dl: DataLoader,
            optimizer: Type[torch.optim.Optimizer],
            optimizer_params: Dict,
            models_path: str,
            summary_path: str,
            class_metrics: bool,
    ) -> None:
        self.finetuning = True
        experimenter.save_model(
            file_path=os.path.join(models_path, 'trained'),
            state_dict=False
        )
        for e in self.energy_thresholds:
            str_e = f'e_{e}'
            experimenter.load_model(
                file_path=os.path.join(models_path, 'trained'),
                state_dict=False
            )
            prune_sdv_model(
                model=experimenter.model,
                pruning_fn=energy_threshold_pruning,
                pruning_params={'energy_threshold': e}
            )

            experimenter.save_model(os.path.join(models_path, str_e))
            val_scores = experimenter.val_loop(
                dataloader=val_dl,
                class_metrics=class_metrics
            )
            writer = SummaryWriter(os.path.join(summary_path, str_e))
            experimenter.best_score = val_scores[experimenter.metric]
            write_scores(writer, 'fine-tuning', val_scores, 0)

            print(f"fine-tuning e={e}, using device: {experimenter.device}")
            experimenter.run_epochs(
                writer=writer,
                train_dl=train_dl,
                val_dl=val_dl,
                num_epochs=self.finetuning_epochs,
                optimizer=optimizer(experimenter.model.parameters(), **optimizer_params),
                model_path=os.path.join(models_path, str_e),
                class_metrics=class_metrics,
                model_losses=self.loss
            )

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

    # def fit(
    #         self,
    #         experimenter,
    #         description: str,
    #         train_dl: DataLoader,
    #         val_dl: DataLoader,
    #         num_epochs: int,
    #         optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
    #         models_path: str = 'models',
    #         summary_path: str = 'summary',
    #         class_metrics: bool = False,
    #         optimizer_params: Dict = {},
    # ) -> None:
    #     """Run experiment.
    #
    #     Args:
    #         experimenter: An instance of the experimenter class, e.g.
    #         ``ClassificationExperimenter``.
    #         num_epochs: Number of epochs.
    #     """
    #     pass
    #
    # def optimize_during_training(self) -> None:
    #     """ Apply optimization after train loop every epoch."""
    #     for module in self.exp.get_optimizable_module().modules():
    #         if isinstance(module, torch.nn.Conv2d):
    #             zerolize_filters(conv=module, pruning_ratio=self.pruning_ratio)
    #
    # def final_optimize(self) -> None:
    #     """Apply optimization after training."""
    #     self.exp.load_model_state_dict()
    #     self.exp.summary_per_class = False
    #     results = {'default_scores': self.optimization_summary()}
    #
    #     pruned_sd, input_size, output_size = prune_resnet_state_dict(
    #         self.exp.model.state_dict()
    #     )
    #     self.exp.model = SFP_MODELS[self.exp.optimizable_module_name](
    #         num_classes=self.exp.num_classes,
    #         input_size=input_size,
    #         output_size=output_size,
    #         pruning_ratio=self.pruning_ratio
    #     )
    #     self.exp.model.load_state_dict(pruned_sd)
    #     self.exp.model.to(self.exp.device)
    #     results['pruned_scores'] = self.optimization_summary()
    #     self.exp.save_model(name='pruned')
    #
    #     self.exp.finetune(num_epochs=self.finetuning_epochs)
    #     results['finetuned_scores'] = self.optimization_summary()
    #     self.exp.save_model(name='fine-tuned')
    #
    #     for k in results['default_scores']:
    #         print(
    #             f"{k}: {results['default_scores'][k]:.3f}, "
    #             f"{results['pruned_scores'][k]:.3f}, "
    #             f"{results['finetuned_scores'][k]:.3f}"
    #         )
    #     torch.save(results, os.path.join(
    #         self.exp.summary_path, self.exp.name, 'results.pt'
    #     ))
    #
    # def optimization_summary(self) -> Dict:
    #     """Validate model and return scores."""
    #     scores = self.exp.val_loop()
    #     scores['size'] = self.exp.size_of_model()
    #     scores['n_params'] = self.exp.number_of_params()
    #     return scores
