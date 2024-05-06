import torch.nn as nn
from typing import List
import torch.distributions as distributions
import torch


class DistributionLoss(nn.Module):
    """
    DistributionLoss base class.

    Class should be inherited for all distribution losses, i.e. if a network predicts
    the parameters of a probability distribution, DistributionLoss can be used to
    score those parameters and calculate loss for given true values.

    Define two class attributes in a child class:

    Attributes:
        distribution_class (distributions.Distribution): torch probability distribution
        distribution_arguments (List[str]): list of parameter names for the distribution

    Further, implement the methods :py:meth:`~map_x_to_distribution` and :py:meth:`~rescale_parameters`.
    """

    distribution_class: distributions.Distribution
    distribution_arguments: List[str]

    def __init__(
        self, quantiles: List[float] = [.05, .25, .5, .75, .95], reduction="mean"
    ):
        """
        Initialize loss

        Args:
            quantiles (List[float], optional): quantiles for probability range.
                Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
        """
        super().__init__()
        self.quantiles = quantiles
        self.reduction = getattr(torch, reduction)

    # def rescale_parameters(
    #     self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: RevIN
    # ) -> torch.Tensor:
    #     """
    #     Rescale normalized parameters into the scale required for the output.

    #     Args:
    #         parameters (torch.Tensor): normalized parameters (indexed by last dimension)
    #         target_scale (torch.Tensor): scale of parameters (n_batch_samples x (center, scale))
    #         encoder (BaseEstimator): original encoder that normalized the target in the first place

    #     Returns:
    #         torch.Tensor: parameters in real/not normalized space
    #     """
    #     return encoder(dict(prediction=parameters, target_scale=target_scale))


    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Distribution:
        """
        Map the a tensor of parameters to a probability distribution.

        Args:
            x (torch.Tensor): parameters for probability distribution. Last dimension will index the parameters

        Returns:
            distributions.Distribution: torch probability distribution as defined in the
                class attribute ``distribution_class``
        """
        raise NotImplementedError("implement this method")

    def forward(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        distribution = self.map_x_to_distribution(y_pred)
        loss = -distribution.log_prob(y_actual)
        loss = self.reduction(loss)
        return loss

 
class NormalDistributionLoss(DistributionLoss):
    """
    Normal distribution loss.
    """

    distribution_class = distributions.Normal
    distribution_arguments = ["affine_loc", "affine_scale", "loc", "scale"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        loc = x[..., 2]
        scale = x[..., 3]
        distr = self.distribution_class(loc=loc, scale=scale)
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        return distributions.TransformedDistribution(distr, [scaler])
        

    # def rescale_parameters(
    #     self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    # ) -> torch.Tensor:
    #     self._transformation = encoder.transformation
    #     loc = parameters[..., 0]
    #     scale = F.softplus(parameters[..., 1])
    #     return torch.concat(
    #         [target_scale.unsqueeze(1).expand(-1, loc.size(1), -1), loc.unsqueeze(-1), scale.unsqueeze(-1)], dim=-1
    #     )
