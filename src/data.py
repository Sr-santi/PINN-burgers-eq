"""Data generation and sampling utilities for PINN training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import qmc

from .config import PINNConfig, RuntimeConfig


@dataclass
class TrainingTensors:
    """Container for training data tensors (IC, BC, collocation)."""

    x_ic: torch.Tensor
    t_ic: torch.Tensor
    u_ic: torch.Tensor
    nu_ic: torch.Tensor
    x_bc: torch.Tensor
    t_bc: torch.Tensor
    u_bc: torch.Tensor
    nu_bc: torch.Tensor
    x_f: torch.Tensor
    t_f: torch.Tensor
    nu_f: torch.Tensor


def _lhs(d: int, n: int, seed: int) -> np.ndarray:
    """Return (n, d) Latin-Hypercube samples in the unit cube [0, 1)^d."""
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    return sampler.random(n=n)


def _to_tensor(
    arr: np.ndarray, requires_grad: bool = False, runtime_cfg: RuntimeConfig | None = None
) -> torch.Tensor:
    """Convert numpy array to torch tensor with specified dtype and device."""
    if runtime_cfg is None:
        runtime_cfg = RuntimeConfig()
    return torch.tensor(
        arr, dtype=runtime_cfg.dtype, device=runtime_cfg.device, requires_grad=requires_grad
    )


def _log_uniform_from_lhs01(
    u01: np.ndarray, nu_min: float, nu_max: float
) -> np.ndarray:
    """Map entries in u01 in [0,1] to log-uniform samples in [nu_min, nu_max]."""
    log_min = math.log(nu_min)
    log_max = math.log(nu_max)
    return np.exp(log_min + u01 * (log_max - log_min))


def build_dataset(
    cfg: PINNConfig, seed: int = 1234, runtime_cfg: RuntimeConfig | None = None
) -> TrainingTensors:
    """Build Phase 1 training dataset with fixed nu.

    Parameters
    ----------
    cfg : PINNConfig
        Configuration with domain bounds, sampling counts, and nu value.
    seed : int
        Random seed for reproducibility.
    runtime_cfg : RuntimeConfig, optional
        Runtime configuration for device/dtype. If None, uses defaults.

    Returns
    -------
    TrainingTensors
        Training tensors for IC, BC, and collocation points.
    """
    if runtime_cfg is None:
        runtime_cfg = RuntimeConfig()

    u_ic_x = cfg.x_min + (cfg.x_max - cfg.x_min) * _lhs(1, cfg.n_ic, seed)
    x_ic = u_ic_x.reshape(-1, 1)
    t_ic = np.zeros_like(x_ic)
    u_ic = -np.sin(math.pi * x_ic)
    nu_ic = np.full_like(x_ic, cfg.nu, dtype=np.float64)

    half = cfg.n_bc // 2
    bc_t_left = cfg.t_min + (cfg.t_max - cfg.t_min) * _lhs(1, half, seed + 1)
    bc_t_right = cfg.t_min + (cfg.t_max - cfg.t_min) * _lhs(1, cfg.n_bc - half, seed + 2)
    x_bc = np.concatenate(
        [np.full_like(bc_t_left, cfg.x_min), np.full_like(bc_t_right, cfg.x_max)], axis=0
    )
    t_bc = np.concatenate([bc_t_left, bc_t_right], axis=0)
    u_bc = np.zeros_like(x_bc)
    nu_bc = np.full_like(x_bc, cfg.nu, dtype=np.float64)

    lhs_f = _lhs(2, cfg.n_f, seed + 3)
    x_f = cfg.x_min + (cfg.x_max - cfg.x_min) * lhs_f[:, [0]]
    t_f = cfg.t_min + (cfg.t_max - cfg.t_min) * lhs_f[:, [1]]
    nu_f = np.full_like(x_f, cfg.nu, dtype=np.float64)

    return TrainingTensors(
        x_ic=_to_tensor(x_ic, runtime_cfg=runtime_cfg),
        t_ic=_to_tensor(t_ic, runtime_cfg=runtime_cfg),
        u_ic=_to_tensor(u_ic, runtime_cfg=runtime_cfg),
        nu_ic=_to_tensor(nu_ic, runtime_cfg=runtime_cfg),
        x_bc=_to_tensor(x_bc, runtime_cfg=runtime_cfg),
        t_bc=_to_tensor(t_bc, runtime_cfg=runtime_cfg),
        u_bc=_to_tensor(u_bc, runtime_cfg=runtime_cfg),
        nu_bc=_to_tensor(nu_bc, runtime_cfg=runtime_cfg),
        x_f=_to_tensor(x_f, requires_grad=True, runtime_cfg=runtime_cfg),
        t_f=_to_tensor(t_f, requires_grad=True, runtime_cfg=runtime_cfg),
        nu_f=_to_tensor(nu_f, runtime_cfg=runtime_cfg),
    )


def build_parametric_dataset(
    cfg: PINNConfig, seed: int = 1234, runtime_cfg: RuntimeConfig | None = None
) -> TrainingTensors:
    """Build Phase 2 training dataset with log-uniform nu distribution.

    Parameters
    ----------
    cfg : PINNConfig
        Configuration with domain bounds, sampling counts, and nu range.
    seed : int
        Random seed for reproducibility.
    runtime_cfg : RuntimeConfig, optional
        Runtime configuration for device/dtype. If None, uses defaults.

    Returns
    -------
    TrainingTensors
        Training tensors for IC, BC, and collocation points with sampled nu.
    """
    if runtime_cfg is None:
        runtime_cfg = RuntimeConfig()

    lo, hi = cfg.nu_min_param, cfg.nu_max_param

    lhs_ic = _lhs(2, cfg.n_ic, seed + 11)
    x_ic = cfg.x_min + (cfg.x_max - cfg.x_min) * lhs_ic[:, [0]]
    t_ic = np.zeros_like(x_ic)
    u_ic = -np.sin(math.pi * x_ic)
    nu_ic = _log_uniform_from_lhs01(lhs_ic[:, 1], lo, hi).reshape(-1, 1)

    half = cfg.n_bc // 2
    lhs_bl = _lhs(2, half, seed + 12)
    lhs_br = _lhs(2, cfg.n_bc - half, seed + 13)
    t_left = cfg.t_min + (cfg.t_max - cfg.t_min) * lhs_bl[:, [1]]
    x_left = np.full_like(t_left, cfg.x_min)
    nu_left = _log_uniform_from_lhs01(lhs_bl[:, 0], lo, hi).reshape(-1, 1)

    t_right = cfg.t_min + (cfg.t_max - cfg.t_min) * lhs_br[:, [1]]
    x_right = np.full_like(t_right, cfg.x_max)
    nu_right = _log_uniform_from_lhs01(lhs_br[:, 0], lo, hi).reshape(-1, 1)

    x_bc = np.concatenate([x_left, x_right], axis=0)
    t_bc = np.concatenate([t_left, t_right], axis=0)
    u_bc = np.zeros_like(x_bc)
    nu_bc = np.concatenate([nu_left, nu_right], axis=0)

    lhs_f = _lhs(3, cfg.n_f, seed + 14)
    x_f = cfg.x_min + (cfg.x_max - cfg.x_min) * lhs_f[:, [0]]
    t_f = cfg.t_min + (cfg.t_max - cfg.t_min) * lhs_f[:, [1]]
    nu_f = _log_uniform_from_lhs01(lhs_f[:, 2], lo, hi).reshape(-1, 1)

    return TrainingTensors(
        x_ic=_to_tensor(x_ic, runtime_cfg=runtime_cfg),
        t_ic=_to_tensor(t_ic, runtime_cfg=runtime_cfg),
        u_ic=_to_tensor(u_ic, runtime_cfg=runtime_cfg),
        nu_ic=_to_tensor(nu_ic, runtime_cfg=runtime_cfg),
        x_bc=_to_tensor(x_bc, runtime_cfg=runtime_cfg),
        t_bc=_to_tensor(t_bc, runtime_cfg=runtime_cfg),
        u_bc=_to_tensor(u_bc, runtime_cfg=runtime_cfg),
        nu_bc=_to_tensor(nu_bc, runtime_cfg=runtime_cfg),
        x_f=_to_tensor(x_f, requires_grad=True, runtime_cfg=runtime_cfg),
        t_f=_to_tensor(t_f, requires_grad=True, runtime_cfg=runtime_cfg),
        nu_f=_to_tensor(nu_f, runtime_cfg=runtime_cfg),
    )


def plot_training_points(data: TrainingTensors, cfg: PINNConfig) -> plt.Figure:
    """Visualize the spatial distribution of training points.

    Parameters
    ----------
    data : TrainingTensors
        Training data with IC, BC, and collocation points.
    cfg : PINNConfig
        Configuration for axis limits.

    Returns
    -------
    plt.Figure
        Figure showing training point distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(
        data.t_f.detach().cpu().numpy(),
        data.x_f.detach().cpu().numpy(),
        s=2,
        alpha=0.35,
        c="#444",
        label=f"collocation ({data.x_f.shape[0]})",
    )
    ax.scatter(
        data.t_ic.cpu().numpy(),
        data.x_ic.cpu().numpy(),
        s=14,
        c="#1f77b4",
        label=f"IC ({data.x_ic.shape[0]})",
    )
    ax.scatter(
        data.t_bc.cpu().numpy(),
        data.x_bc.cpu().numpy(),
        s=14,
        c="#d62728",
        label=f"BC ({data.x_bc.shape[0]})",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_xlim(cfg.t_min, cfg.t_max)
    ax.set_ylim(cfg.x_min, cfg.x_max)
    ax.set_title("LHS training point distribution")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    return fig
