import torch
import numpy as np
from scipy.special import gamma

try:
    from pymittagleffler import mittag_leffler
except ImportError:
    from fedot_ind.core.operation.transformation.representation.kernel.utils import mittag_leffler

class MittagLefflerAutograd(torch.autograd.Function):
    '''Дифференцируемая функция Миттаг-Леффлера для PyTorch. Поддерживает backpropagation.'''
    @staticmethod
    def forward(ctx, z_tensor, q):
        device = z_tensor.device
        z_np = z_tensor.detach().cpu().numpy()

        res_np = mittag_leffler(z_np, q, 1.0)
        res_tensor = torch.tensor(res_np, dtype=torch.complex128, device=device)

        ctx.save_for_backward(z_tensor)
        ctx.q = q
        return res_tensor

    @staticmethod
    def backward(ctx, grad_output):
        z_tensor, = ctx.saved_tensors
        q = ctx.q
        device = z_tensor.device
        z_np = z_tensor.detach().cpu().numpy()

        grad_z_np = np.zeros_like(z_np, dtype=np.complex128)
        mask_zero = np.abs(z_np) < 1e-6

        if not np.all(mask_zero):
            z_nonzero = z_np[~mask_zero]
            ml_deriv_part = mittag_leffler(z_nonzero, q, 1.0 - q)
            grad_z_np[~mask_zero] = ml_deriv_part / (q * z_nonzero)

        if np.any(mask_zero):
            grad_z_np[mask_zero] = 1.0 / gamma(q + 1.0)

        grad_z_tensor = torch.tensor(grad_z_np, dtype=torch.complex128, device=device)
        grad_input = grad_output * grad_z_tensor

        #Для q градиент не поддерживается, возвращаем None
        return grad_input, None

def _mittag_leffler_tensor(z_tensor, q):
    return MittagLefflerAutograd.apply(z_tensor, q)