import torch
import torch.nn as nn

class SqrtScaledMSELoss(nn.Module):
    """
    Pérdida L = MSE(sqrt(pred / scale), sqrt(target / scale)).
    La raíz cuadrada amplifica el error en tiempos pequeños (fase inicial de enfriamiento rápido).
    Si scale es None, no realiza reescalado de las variables (adecuado cuando el dataset ya devuelve datos escalados).
    """
    def __init__(self, scale: float | None = None):
        super().__init__()
        self.scale = scale

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.scale is not None:
            p_s = p / self.scale
            t_s = t / self.scale
        else:
            p_s = p
            t_s = t
        return nn.functional.mse_loss(torch.sqrt(p_s.clamp_min(1e-6)), torch.sqrt(t_s.clamp_min(1e-6)))
