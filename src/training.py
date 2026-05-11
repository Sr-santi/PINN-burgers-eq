from .physics import composite_loss
import time
import torch
from .model import PINNLightning

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
    
    def closure():
        """L-BFGS closure: compute loss and gradients."""
        lbfgs.zero_grad(set_to_none=True)
        loss_breakdown = composite_loss(model_pinn, training_data, cfg)
        loss_breakdown.total.backward()
        
        # Optional periodic logging
        if log_interval is not None:
            iter_counter[0] += 1
            if iter_counter[0] % log_interval == 0:
                print(
                    f"  it {iter_counter[0]:>6d} | tot={loss_breakdown.total.item():.3e} | "
                    f"ic={loss_breakdown.ic.item():.2e} bc={loss_breakdown.bc.item():.2e} "
                    f"f={loss_breakdown.f.item():.2e}"
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
    
    return result_model