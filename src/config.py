"""Configuration classes for PINN training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class PINNConfig:
    """Physics-Informed Neural Network configuration."""

    # Domain configuration
    nu: float = 0.01 / math.pi
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # Training data sampling
    n_ic: int = 100
    n_bc: int = 100
    n_f: int = 10_000

    # Network architecture
    hidden_layers: int = 6
    hidden_units: int = 40
    activation: str = "tanh"

    # Loss weights
    w_ic: float = 1.0
    w_bc: float = 1.0
    w_f: float = 1.0

    # Adam optimizer (Stage 1)
    adam_epochs: int = 8_000
    adam_lr: float = 1e-3

    # L-BFGS optimizer (Stage 2)
    lbfgs_max_iter: int = 5_000
    lbfgs_tol_grad: float = 1e-9
    lbfgs_tol_change: float = 1e-12
    lbfgs_history: int = 50

    # Logging and validation
    log_every: int = 500
    val_target_l2: float = 1e-3

    # Validation grid configuration
    val_grid_nx: int = 256
    val_grid_nt: int = 100
    gh_quadrature_nodes: int = 200

    # Phase 2 parametric nu
    nu_min_param: float = 0.001 / math.pi
    nu_max_param: float = 0.1 / math.pi
    n_val_nu: int = 5

    # Phase 3: Population-Risk optimizer settings
    use_population_risk: bool = False
    population_risk_batch_size: int = 10_000

    # Mode configuration
    fast_mode: bool = False


@dataclass
class RuntimeConfig:
    """Runtime configuration for device, dtype, and seeding."""

    seed: int = 1234
    device: torch.device | None = None
    dtype: torch.dtype = torch.float32
    checkpoint_dir: Path | None = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def setup(self):
        """Apply seed and dtype to torch."""
        import numpy as np

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_default_dtype(self.dtype)

    def get_device_info(self) -> str:
        """Return device information string."""
        info = f"PyTorch {torch.__version__} | device = {self.device}"
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            info += f" | GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB"
        return info


def get_default_config(fast_mode: bool = False) -> PINNConfig:
    """Get default PINN configuration with optional fast-mode overrides.

    Parameters
    ----------
    fast_mode : bool
        If True, reduce epochs and iterations for quick smoke tests.

    Returns
    -------
    PINNConfig
        Configuration with fast-mode adjustments applied if requested.
    """
    cfg = PINNConfig(fast_mode=fast_mode)
    if fast_mode:
        cfg.adam_epochs = 600
        cfg.lbfgs_max_iter = 120
        cfg.log_every = 200
    return cfg
