"""PyTorch Lightning module for PINN training."""

from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as L
import torch
import torch.nn as nn

from .config import PINNConfig, RuntimeConfig
from .physics import composite_loss
from .population_risk_optimizer import PopulationRiskAdamW


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "swish": nn.SiLU,
}


class PINN(nn.Module):
    """Neural network for u(x, t, nu) with tanh/swish activation.

    Inputs [x, t, ln(nu)] are concatenated and processed through a tanh MLP.
    """

    def __init__(self, hidden_units: int, hidden_layers: int, activation: str = "tanh"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS)}")
        act = _ACTIVATIONS[activation]

        input_dim = 3
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_units), act()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), act()]
        layers += [nn.Linear(hidden_units, 1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        psi = torch.log(nu.clamp(min=torch.finfo(self.net[0].weight.dtype).tiny))
        return self.net(torch.cat([x, t, psi], dim=-1))


class PINNLightning(L.LightningModule):
    """PyTorch Lightning module for PINN training."""

    def __init__(self, cfg: PINNConfig, runtime_cfg=None):
        super().__init__()
        self.cfg = cfg
        self.runtime_cfg = runtime_cfg

        self.pinn = PINN(
            hidden_units=cfg.hidden_units,
            hidden_layers=cfg.hidden_layers,
            activation=cfg.activation,
        )

        self.training_data = None
        self.val_data = None

    def forward(self, x: torch.Tensor, t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        return self.pinn(x, t, nu)

    def set_training_data(self, data):
        """Set training data for the loss computation."""
        self.training_data = data

    def set_validation_data(self, val_x, val_t, val_u, nu_val=None):
        """Set validation grid data."""
        self.val_data = {
            "x": val_x,
            "t": val_t,
            "u": val_u,
            "nu": nu_val if nu_val is not None else self.cfg.nu,
        }

    def training_step(self, batch, batch_idx):
        """Compute loss on training batch."""
        if self.training_data is None:
            raise RuntimeError("Training data not set")

        # Ensure model is on the same device as training data
        device = self.training_data.x_ic.device
        self.pinn = self.pinn.to(device)

        loss_breakdown = composite_loss(self.pinn, self.training_data, self.cfg)

        self.log("train/loss_total", loss_breakdown.total, prog_bar=True)
        self.log("train/loss_ic", loss_breakdown.ic)
        self.log("train/loss_bc", loss_breakdown.bc)
        self.log("train/loss_f", loss_breakdown.f)

        return loss_breakdown.total

    def configure_optimizers(self):
        """Configure optimizer based on training stage."""
        if self.cfg.use_population_risk:
            return PopulationRiskAdamW(
                self.parameters(),
                lr=self.cfg.adam_lr,
                batch_size=self.cfg.population_risk_batch_size,
            )
        else:
            return torch.optim.Adam(self.parameters(), lr=self.cfg.adam_lr)