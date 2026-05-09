"""Utilities for exact solutions, metrics, and validation."""

from __future__ import annotations

import math

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss

from .config import PINNConfig, RuntimeConfig


def _gh_nodes(n_nodes: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Probabilist Gauss-Hermite nodes/weights for weight exp(-s^2/2).

    E[f(s)] = (1/sqrt(2*pi)) * sum_i w_i f(s_i) with s ~ N(0, 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Nodes and weights arrays, shape (n_nodes,) each.
    """
    s, w = hermegauss(n_nodes)
    return s.astype(np.float64), w.astype(np.float64)


_GH_NODES_DEFAULT = _gh_nodes(200)


def burgers_exact(
    x: np.ndarray, t: np.ndarray, nu: float, n_nodes: int = 200
) -> np.ndarray:
    """Analytical solution of the bounded viscous Burgers equation.

    Evaluated via the Hopf integral with Gauss-Hermite quadrature in the rescaled
    variable s = (y - x) / sqrt(2*nu*t). Returns a matrix of shape (len(x), len(t)).
    Points at t=0 use the initial condition directly.

    Parameters
    ----------
    x : np.ndarray
        Spatial points, shape (n_x,).
    t : np.ndarray
        Temporal points, shape (n_t,).
    nu : float
        Viscosity parameter.
    n_nodes : int, optional
        Number of Gauss-Hermite quadrature nodes (default 200).

    Returns
    -------
    np.ndarray
        Solution matrix u(x, t), shape (n_x, n_t).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)

    if n_nodes == _GH_NODES_DEFAULT[0].size:
        s_nodes, s_weights = _GH_NODES_DEFAULT
    else:
        s_nodes, s_weights = _gh_nodes(n_nodes)

    u = np.zeros((x.size, t.size), dtype=np.float64)
    t_zero = t == 0.0

    if np.any(t_zero):
        u[:, t_zero] = (-np.sin(math.pi * x))[:, None]

    t_pos = ~t_zero
    if not np.any(t_pos):
        return u

    t_vals = t[t_pos]
    sqrt_factor = np.sqrt(2.0 * nu * t_vals)

    y = x[:, None, None] + sqrt_factor[None, :, None] * s_nodes[None, None, :]
    h = (np.cos(math.pi * y) - 1.0) / (2.0 * nu * math.pi)
    h_min = h.min(axis=-1, keepdims=True)
    weights = s_weights[None, None, :] * np.exp(-(h - h_min))

    denom = weights.sum(axis=-1)
    numer = (s_nodes[None, None, :] * weights).sum(axis=-1)

    u[:, t_pos] = -np.sqrt(2.0 * nu / t_vals)[None, :] * (numer / denom)
    return u


def exact_on_grid(cfg: PINNConfig, nu: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build reference solution on a regular grid.

    Parameters
    ----------
    cfg : PINNConfig
        Configuration with domain bounds and grid resolution.
    nu : float, optional
        Viscosity parameter. If None, uses cfg.nu.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple (x_grid, t_grid, u_grid) where u_grid has shape (len(x_grid), len(t_grid)).
    """
    if nu is None:
        nu = cfg.nu

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.val_grid_nx)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.val_grid_nt)
    u = burgers_exact(x, t, nu=nu, n_nodes=cfg.gh_quadrature_nodes)
    return x, t, u


@torch.no_grad()
def predict_on_grid(
    model: torch.nn.Module,
    x: np.ndarray,
    t: np.ndarray,
    nu_scalar: float,
    runtime_cfg: RuntimeConfig | None = None,
) -> np.ndarray:
    """Evaluate network on a (x, t) grid at fixed nu.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.
    x : np.ndarray
        Spatial points, shape (n_x,).
    t : np.ndarray
        Temporal points, shape (n_t,).
    nu_scalar : float
        Viscosity value.
    runtime_cfg : RuntimeConfig, optional
        Runtime configuration. If None, uses defaults.

    Returns
    -------
    np.ndarray
        Predicted solution on grid, shape (n_x, n_t).
    """
    model.eval()
    
    # Determine device and dtype from model parameters
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    X, T = np.meshgrid(x, t, indexing="ij")
    pts_x = torch.tensor(X.reshape(-1, 1), dtype=dtype, device=device)
    pts_t = torch.tensor(T.reshape(-1, 1), dtype=dtype, device=device)
    pts_n = torch.full_like(pts_x, nu_scalar, dtype=dtype, device=device)
    u_pred = model(pts_x, pts_t, pts_n).cpu().numpy().reshape(X.shape)
    model.train()
    return u_pred


def l2_relative_error(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    """Compute relative L2 error ||u_pred - u_true||_2 / ||u_true||_2.

    Parameters
    ----------
    u_pred : np.ndarray
        Predicted solution.
    u_true : np.ndarray
        Ground truth solution.

    Returns
    -------
    float
        Relative L2 error.
    """
    return float(np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true))


def mean_l2_relative_over_nu(
    model: torch.nn.Module,
    x: np.ndarray,
    t: np.ndarray,
    nu_list: np.ndarray,
    runtime_cfg: RuntimeConfig | None = None,
) -> float:
    """Compute mean relative L2 error over a list of nu values.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.
    x : np.ndarray
        Spatial points for evaluation.
    t : np.ndarray
        Temporal points for evaluation.
    nu_list : np.ndarray
        List of viscosity values to evaluate over.
    runtime_cfg : RuntimeConfig, optional
        Runtime configuration. If None, uses defaults.

    Returns
    -------
    float
        Mean relative L2 error across all nu values.
    """
    acc = 0.0
    for nu_val in np.atleast_1d(nu_list):
        nu_f = float(nu_val)
        u_star = burgers_exact(x, t, nu=nu_f)
        acc += l2_relative_error(predict_on_grid(model, x, t, nu_f), u_star)
    return acc / max(len(np.atleast_1d(nu_list)), 1)
