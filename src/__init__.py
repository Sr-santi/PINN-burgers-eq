"""PINN Burgers equation solver package."""

from .config import PINNConfig, RuntimeConfig, get_default_config
from .data import TrainingTensors, build_dataset, build_parametric_dataset, plot_training_points
from .model import PINN, PINNLightning
from .physics import LossBreakdown, composite_loss, physics_residual
from .population_risk_optimizer import PopulationRiskAdamW
from .utils import (
    burgers_exact,
    exact_on_grid,
    l2_relative_error,
    mean_l2_relative_over_nu,
    predict_on_grid,
)

__all__ = [
    "PINNConfig",
    "RuntimeConfig",
    "get_default_config",
    "TrainingTensors",
    "build_dataset",
    "build_parametric_dataset",
    "plot_training_points",
    "PINN",
    "PINNLightning",
    "LossBreakdown",
    "composite_loss",
    "physics_residual",
    "PopulationRiskAdamW",
    "burgers_exact",
    "exact_on_grid",
    "l2_relative_error",
    "mean_l2_relative_over_nu",
    "predict_on_grid",
]
