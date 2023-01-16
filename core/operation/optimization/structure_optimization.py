import os
import time
from typing import Type, Dict, List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.metrics.loss.svd_loss import OrthogonalLoss, HoyerLoss
from core.architecture.experiment.nn_experimenter import NNExperimenter, write_scores
from core.operation.decomposition.decomposed_conv import DecomposedConv2d
from core.operation.optimization.svd_tools import energy_threshold_pruning, decompose_module


class StructureOptimization:
    """Generalized class for model structure optimization."""

    def __init__(
            self,
            description,
            finetuning_epochs: int
    ) -> None:
        self.description = description
        self.finetuning_epochs= finetuning_epochs

    def fit(
            self,
            experimenter: NNExperimenter,
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
        self.hoer_loss_factor = hoer_loss_factor
        self.orthogonal_loss_factor = orthogonal_loss_factor

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
        losses = {
            'hoer_loss': HoyerLoss(self.hoer_loss_factor),
            'orthogonal_loss': OrthogonalLoss(
                device=experimenter.device,
                factor=self.orthogonal_loss_factor
            )
        }
        decompose_module(experimenter.model, self.decomposing_mode)
        print(f"SVD decomposed size: {experimenter.size_of_model():.2f} Mb")
        description += self.description
        models_path = os.path.join(models_path, description)
        writer = SummaryWriter(os.path.join(summary_path, description, 'train'))
        optimizer = optimizer(experimenter.model.parameters(), **optimizer_params)
        best_score = 0

        print(f"{description}, using device: {experimenter.device}")
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}")
            train_scores = experimenter.train_loop(
                dataloader=train_dl,
                optimizer=optimizer,
                model_losses=losses
            )
            write_scores(writer, 'train', train_scores, epoch)
            val_scores = experimenter.val_loop(
                dataloader=val_dl,
                class_metrics=class_metrics
            )
            write_scores(writer, 'val', val_scores, epoch)
            if val_scores[experimenter.metric] > best_score:
                best_score = val_scores[experimenter.metric]
                print(f'Best {experimenter.metric} score: {best_score}')
                experimenter.save_model(file_name='train', dir_path=models_path)
        writer.close()
        p_writer = SummaryWriter(os.path.join(summary_path, description, 'pruned'))
        ft_writer = SummaryWriter(os.path.join(summary_path, description, 'fine-tuned'))
        losses['hoer_loss'] = HoyerLoss(factor=0)
        experimenter.load_model(file_name='train', dir_path=models_path)
        experimenter.save_model(
            file_name='train',
            dir_path=models_path,
            state_dict=False
        )
        for e in self.energy_thresholds:
            int_e = int(e * 100000)
            experimenter.load_model(
                file_name='train',
                dir_path=models_path,
                state_dict=False
            )
            for module in experimenter.model.modules():
                if isinstance(module, DecomposedConv2d):
                    energy_threshold_pruning(conv=module, energy_threshold=e)

            val_scores = experimenter.val_loop(
                dataloader=val_dl,
                class_metrics=class_metrics
            )
            val_scores['size'] = experimenter.size_of_model()
            val_scores['n_params'] = experimenter.number_of_model_params()
            write_scores(p_writer, 'optimization', val_scores, int_e)

            ft_name = f'fine-tuning_e_{e}'
            best_score = val_scores[experimenter.metric]
            writer = SummaryWriter(os.path.join(summary_path, description, ft_name))
            print(f"fine-tuning e={e}, using device: {experimenter.device}")
            for epoch in range(1, self.finetuning_epochs + 1):
                print(f"Fine-tuning epoch {epoch}")
                train_scores = experimenter.train_loop(
                    dataloader=train_dl,
                    optimizer=optimizer,
                    model_losses=losses
                )
                write_scores(writer, 'fine-tuning_train', train_scores, epoch)
                val_scores = experimenter.val_loop(
                    dataloader=val_dl,
                    class_metrics=class_metrics
                )
                write_scores(writer, 'fine-tuning_val', val_scores, epoch)
                if val_scores[experimenter.metric] > best_score:
                    best_score = val_scores[experimenter.metric]
                    experimenter.save_model(file_name=ft_name, dir_path=models_path)
            experimenter.load_model(file_name=ft_name, dir_path=models_path)
            writer.close()

            val_scores = experimenter.val_loop(
                dataloader=val_dl,
                class_metrics=class_metrics
            )
            val_scores['size'] = experimenter.size_of_model()
            val_scores['n_params'] = experimenter.number_of_model_params()
            write_scores(ft_writer, 'optimization', val_scores, int_e)
            experimenter.save_model(
                file_name=ft_name,
                dir_path=models_path,
                state_dict=False
            )
        p_writer.close()
        ft_writer.close()


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
