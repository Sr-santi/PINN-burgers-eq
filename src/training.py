from .physics import composite_loss
from .utils import predict_on_grid, l2_relative_error, mean_l2_relative_over_nu
import numpy as np
import time
import torch
from .model import PINNLightning
import pytorch_lightning as L
from typing import Any, Dict, List

def _compute_l2_error(model, val_data):
    """Compute L2 relative error from validation data.
    
    Handles both scalar nu and list-of-nu validation.
    """
    if val_data is None:
        return float('nan')
    nu_val = val_data["nu"]
    try:
        if isinstance(nu_val, (list, np.ndarray)):
            return mean_l2_relative_over_nu(
                model, val_data["x"], val_data["t"], np.asarray(nu_val)
            )
        else:
            u_pred = predict_on_grid(
                model, val_data["x"], val_data["t"], float(nu_val)
            )
            return l2_relative_error(u_pred, val_data["u"])
    except Exception:
        return float('nan')

class LossHistoryCallback(L.Callback):
    """Callback to track losses during Lightning training."""
    
    def __init__(self, log_every: int = 500):
        super().__init__()
        self.log_every = log_every
        self.losses = {
            'epoch': [],
            'total': [],
            'ic': [],
            'bc': [],
            'f': [],
            'l2_error': [],
        }
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called at the end of each training epoch."""
        epoch = trainer.current_epoch + 1
        
        if not (epoch % self.log_every == 0 or epoch == 1):
            return
        
        metrics = trainer.callback_metrics
        
        if 'train/loss_total' not in metrics:
            return
        
        self.losses['epoch'].append(epoch)
        self.losses['total'].append(metrics['train/loss_total'].item())
        self.losses['ic'].append(metrics['train/loss_ic'].item())
        self.losses['bc'].append(metrics['train/loss_bc'].item())
        self.losses['f'].append(metrics['train/loss_f'].item())
        
        self.losses['l2_error'].append(_compute_l2_error(pl_module.pinn, pl_module.val_data))
    
    def get_losses(self) -> Dict[str, List]:
        """Return collected loss history."""
        return self.losses

def train_lbfgs_manual(
    model_lightning,
    training_data,
    cfg,
    phase_name="Phase",
    log_interval=500,
    validation_data=None,
):
    """
    Run manual L-BFGS optimization (single C++ call, 5000 internal iterations).
    
    Parameters
    ----------
    model_lightning : PINNLightning
        Lightning module with trained .pinn (from Adam stage)
    training_data : TrainingTensors
        Training data (IC, BC, collocation)
    cfg : PINNConfig
        Configuration with lbfgs_* settings
    phase_name : str
        Name for logging (e.g., "Phase 1", "Phase 2")
    log_interval : int
        Log loss every N iterations (set to None to disable)
    validation_data : dict, optional
        Dict with keys 'x', 't', 'u', 'nu' for validation (optional)
    
    Returns
    -------
    PINNLightning
        Model wrapped in Lightning module after L-BFGS training
    """
    
    # Extract and prepare PINN model
    model_pinn = model_lightning.pinn
    model_pinn.eval()
    
    device = training_data.x_ic.device
    dtype = training_data.x_ic.dtype
    model_pinn = model_pinn.to(device=device, dtype=dtype)
    
    # Configure L-BFGS optimizer
    lbfgs = torch.optim.LBFGS(
        model_pinn.parameters(),
        max_iter=cfg.lbfgs_max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=cfg.lbfgs_tol_grad,
        tolerance_change=cfg.lbfgs_tol_change,
        history_size=cfg.lbfgs_history,
    )
    
    print(f"[{phase_name} Stage 2] L-BFGS: max_iter={cfg.lbfgs_max_iter} (single C++ call)")
    
    # Setup iteration counter for optional logging
    iter_counter = [0]
    loss_history = {
        "iterations": [],
        "total_loss": [],
        "ic_loss": [],
        "bc_loss": [],
        "f_loss": [],
        "l2_error": [],
    }
    
    def closure():
        """L-BFGS closure: compute loss and gradients."""
        lbfgs.zero_grad(set_to_none=True)
        loss_breakdown = composite_loss(model_pinn, training_data, cfg)
        loss_breakdown.total.backward()
        
        # Optional periodic logging
        if log_interval is not None:
            iter_counter[0] += 1
            if iter_counter[0] % log_interval == 0:
                # Store loss values
                loss_history["iterations"].append(iter_counter[0])
                loss_history["total_loss"].append(loss_breakdown.total.item())
                loss_history["ic_loss"].append(loss_breakdown.ic.item())
                loss_history["bc_loss"].append(loss_breakdown.bc.item())
                loss_history["f_loss"].append(loss_breakdown.f.item())
                
                # Compute L2 relative error if validation data is provided
                l2_val = _compute_l2_error(model_pinn, validation_data)
                loss_history["l2_error"].append(l2_val)

                print(
                    f"  it {iter_counter[0]:>6d} | tot={loss_breakdown.total.item():.3e} | "
                    f"ic={loss_breakdown.ic.item():.2e} bc={loss_breakdown.bc.item():.2e} "
                    f"f={loss_breakdown.f.item():.2e}"
                    + (f" | L2={l2_val:.3e}" if l2_val == l2_val else "")
                )
        
        return loss_breakdown.total
    
    # Run L-BFGS: single Python call, 5000 internal C++ iterations
    t0 = time.perf_counter()
    lbfgs.step(closure)
    elapsed = time.perf_counter() - t0
    
    print(f"[{phase_name} Stage 2] L-BFGS complete in {elapsed:.1f}s")
    
    # Wrap result back into Lightning module
    result_model = PINNLightning(cfg)
    result_model.pinn = model_pinn
    result_model.set_training_data(training_data)
    
    # Optionally set validation data
    if validation_data is not None:
        result_model.set_validation_data(
            validation_data["x"],
            validation_data["t"],
            validation_data["u"],
            validation_data.get("nu"),
        )
    
    return result_model, loss_history