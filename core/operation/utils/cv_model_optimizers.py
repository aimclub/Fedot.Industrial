import copy
import os
import time
from typing import List

import torch
from torch.utils.tensorboard import SummaryWriter

from core.metrics.svd_loss import OrthogonalLoss, HoyerLoss
from core.models.cnn.decomposed_conv import DecomposedConv2d
from core.models.cnn.sfp_models import SFP_MODELS
from core.operation.utils.sfp_tools import zerolize_filters, prune_resnet_state_dict
from core.operation.utils.svd_tools import energy_threshold_pruning, decompose_module


class GeneralizedStructureOptimization:
    """Generalized class for model structure optimization.

    Args:
        experimenter: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
    """

    def __init__(
        self,
        experimenter,
    ):
        self.exp = experimenter
        self.losses = {}

    def optimize_during_training(self) -> None:
        """Have to implement optimization method applied after train loop every epoch."""
        pass

    def final_optimize(self) -> None:
        """Have to implement optimization method applied once after training."""
        pass


class SVDOptimization(GeneralizedStructureOptimization):
    """Singular value decomposition for model structure optimization.

    Args:
        experimenter: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        hoer_loss_factor: The hyperparameter by which the Hoyer loss function is
            multiplied.
        orthogonal_loss_factor: The hyperparameter by which the orthogonal loss
            function is multiplied.
        energy_thresholds: List of pruning hyperparameters.
        finetuning_epochs: Number of fine-tuning epochs.
    """

    def __init__(
        self,
        experimenter,
        decomposing_mode: str,
        hoer_loss_factor: float,
        orthogonal_loss_factor: float,
        energy_thresholds: List[float],
        finetuning_epochs: int,
    ) -> None:
        super().__init__(
            experimenter=experimenter,
        )
        decompose_module(experimenter.get_optimizable_module(), decomposing_mode)
        print(f"SVD decomposed size: {self.exp.size_of_model():.2f} MB")
        self.losses['hoer_loss'] = HoyerLoss(hoer_loss_factor)
        self.losses['orthogonal_loss'] = OrthogonalLoss(
            self.exp.device, orthogonal_loss_factor
        )
        self.energy_thresholds = energy_thresholds
        self.finetuning_epochs = finetuning_epochs
        self.exp.name += (
            f"_SVD_{decomposing_mode}_O-{orthogonal_loss_factor:.1f}"
            f"_H-{hoer_loss_factor:.6f}"
        )

    def final_optimize(self) -> None:
        """Apply optimization after training."""
        p_writer = SummaryWriter(
            os.path.join(self.exp.summary_path, self.exp.name) + '_pruned'
        )
        ft_writer = SummaryWriter(
            os.path.join(self.exp.summary_path, self.exp.name) + '_fine-tuned'
        )
        self.losses['hoer_loss'] = HoyerLoss(factor=0)
        self.exp.summary_per_class = False
        self.exp.load_model_state_dict()
        default_model = copy.deepcopy(self.exp.model)
        self.exp.default_scores.update(self.exp.val_loop())

        for e in self.energy_thresholds:
            int_e = int(e * 100000)
            self.exp.model = copy.deepcopy(default_model)
            start = time.time()
            self.prune_model(e)
            pruning_time = time.time() - start
            p_writer.add_scalar('abs(e)/pruning_time', pruning_time, int_e)
            self.optimization_summary(e=int_e, writer=p_writer)
            self.exp.finetune(num_epochs=self.finetuning_epochs, postfix=f"_e-{e}")
            self.optimization_summary(e=int_e, writer=ft_writer)
            self.exp.save_model(postfix=f"_e-{e}")

    def prune_model(self, energy_threshold) -> None:
        """Prune the model weights to the energy_threshold.

        Args:
            energy_threshold: pruning hyperparameter, the lower the threshold, the more
        singular values will be pruned.
        """
        for module in self.exp.get_optimizable_module().modules():
            if isinstance(module, DecomposedConv2d):
                energy_threshold_pruning(conv=module, energy_threshold=energy_threshold)

    def optimization_summary(self, e: int, writer: SummaryWriter) -> None:
        """Validate model and write scores.

        Args:
            e: ``energy_threshold`` as integer for writing scores.
            writer: SummaryWriter object for writing scores.
        """
        val_scores = self.exp.val_loop()
        val_scores['size'] = self.exp.size_of_model()
        val_scores['n_params'] = self.exp.number_of_params()
        for key, score in val_scores.items():
            score_p = score / self.exp.default_scores[key] * 100
            delta_score = score - self.exp.default_scores[key]
            key = key.split('/')[-1]
            writer.add_scalar(f'abs(e)/{key}', score, e)
            writer.add_scalar(f'percentage(e)/{key}', score_p, e)
            writer.add_scalar(f'delta(e)/{key}', delta_score, e)


class SFPOptimization(GeneralizedStructureOptimization):
    """Soft filter pruning for model structure optimization.

    Args:
        experimenter: An instance of the experimenter class, e.g.
            ``ClassificationExperimenter``.
        pruning_ratio: Pruning hyperparameter, percentage of pruned filters.
    """

    def __init__(
            self,
            experimenter,
            pruning_ratio: float,
    ) -> None:
        super().__init__(
            experimenter=experimenter,
        )
        self.pruning_ratio = pruning_ratio
        self.exp.name += f"_SFP_P-{pruning_ratio:.2f}"

    def optimize_during_training(self) -> None:
        """ Apply optimization after train loop every epoch."""
        for module in self.exp.get_optimizable_module().modules():
            if isinstance(module, torch.nn.Conv2d):
                zerolize_filters(conv=module, pruning_ratio=self.pruning_ratio)

    def final_optimize(self) -> None:
        """Apply optimization after training."""
        self.exp.load_model_state_dict()
        self.exp.summary_per_class = False
        default_scores = self.exp.val_loop()
        pruned_sd, input_size, output_size = prune_resnet_state_dict(
            self.exp.model.state_dict()
        )
        self.exp.model = SFP_MODELS[self.exp.optimizable_module_name](
            num_classes=self.exp.num_classes,
            input_size=input_size,
            output_size=output_size,
            pruning_ratio=self.pruning_ratio
        )
        self.exp.model.load_state_dict(pruned_sd)
        self.exp.model.to(self.exp.device)
        val_scores = self.exp.val_loop()
        for key, score in val_scores.items():
            print(f"{key}: {default_scores[key]:.3f}, {score:.3f}")
        print(f"SFP pruned size: {self.exp.size_of_model():.2f} MB")
        self.exp.save_model()


OPTIMIZATIONS = {
    'none': GeneralizedStructureOptimization,
    'SVD': SVDOptimization,
    'SFP': SFPOptimization,
}
