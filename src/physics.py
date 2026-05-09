"""Physics-informed loss and residual computation for Burgers equation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import RuntimeConfig
from .data import TrainingTensors


@dataclass
class LossBreakdown:
    """Breakdown of loss components for logging and analysis."""

    total: torch.Tensor
    ic: torch.Tensor
    bc: torch.Tensor
    f: torch.Tensor


def physics_residual(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    """Compute the physics residual f = u_t + u*u_x - nu*u_xx.

    Uses autograd to compute derivatives of the network output with respect to
    spatial (x) and temporal (t) coordinates.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model that computes u(x, t, nu).
    x : torch.Tensor
        Spatial coordinate, shape (n, 1), requires_grad=True.
    t : torch.Tensor
        Temporal coordinate, shape (n, 1), requires_grad=True.
    nu : torch.Tensor
        Viscosity parameter, shape (n, 1).

    Returns
    -------
    torch.Tensor
        Residual f, same shape as x.
    """
    u = model(x, t, nu)
    grads = torch.autograd.grad

    u_x = grads(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grads(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = grads(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return u_t + u * u_x - nu * u_xx


def composite_loss(
    model: torch.nn.Module, data: TrainingTensors, cfg
) -> LossBreakdown:
    """Compute composite weighted loss: IC + BC + PDE residual.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.
    data : TrainingTensors
        Training data with IC, BC, and collocation points.
    cfg : PINNConfig
        Configuration with loss weights (w_ic, w_bc, w_f).

    Returns
    -------
    LossBreakdown
        Breakdown of loss components.
    """
    u_ic_hat = model(data.x_ic, data.t_ic, data.nu_ic)
    loss_ic = torch.mean((u_ic_hat - data.u_ic) ** 2)

    u_bc_hat = model(data.x_bc, data.t_bc, data.nu_bc)
    loss_bc = torch.mean((u_bc_hat - data.u_bc) ** 2)

    f = physics_residual(model, data.x_f, data.t_f, data.nu_f)
    loss_f = torch.mean(f**2)

    total = cfg.w_ic * loss_ic + cfg.w_bc * loss_bc + cfg.w_f * loss_f
    return LossBreakdown(
        total=total, ic=loss_ic.detach(), bc=loss_bc.detach(), f=loss_f.detach()
    )
