"""Population-Risk gated optimizer for improved generalization in PINNs.

This module implements PopulationRiskAdamW, a variant of AdamW that applies
a Signal-to-Noise Ratio (SNR) gate to suppress updates on noisy parameters.

Mathematical Basis:
    Paper: "A Theory of Generalization in Deep Learning" (Section 6)
    
    Update gate condition: b * (m_t)^2 > v_t
    where:
      - b: batch size (effective sample size)
      - m_t: first moment (EMA of gradients, estimates μ)
      - v_t: second moment (EMA of squared gradients, estimates E[g²])
      - Variance estimate: σ² ≈ v_t - m_t²
    
    Only apply parameter updates where signal (m_t²) exceeds noise threshold.
    This prevents the model from fitting random fluctuations in noisy data.
"""

from __future__ import annotations

import torch
import torch.optim
from typing import Any


class PopulationRiskAdamW(torch.optim.AdamW):
    """AdamW optimizer with population-risk SNR gating for robustness.
    
    Extends torch.optim.AdamW by applying a per-parameter gate during the
    update step. The gate suppresses parameter updates when the signal-to-noise
    ratio (SNR) is low, preventing overfitting to noise in the data.
    
    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, optional
        Learning rate (default: 1e-3).
    betas : tuple, optional
        Coefficients for computing running averages of gradients (default: (0.9, 0.999)).
    eps : float, optional
        Term added for numerical stability (default: 1e-8).
    weight_decay : float, optional
        Weight decay coefficient (default: 0.01).
    batch_size : int, optional
        Effective batch/dataset size for SNR gate (default: 10000).
        Should match the size of the effective training set used in loss computation.
    amsgrad : bool, optional
        Whether to use AMSGrad variant (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        batch_size: int = 10000,
        amsgrad: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self.batch_size = batch_size

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step with SNR gating.
        
        The population-risk gate is computed and applied to the parameter update.
        
        Process:
        1. Accumulate gradients and compute loss (standard Adam).
        2. Update first moment (m) and second moment (v) with EMA.
        3. Apply bias correction to m and v.
        4. Compute SNR gate: gate = (batch_size * m²) > v
           This compares signal power (m²) to total second moment (v).
        5. Mask the parameter delta by the gate (multiply by float(gate)).
        6. Apply weight decay and learning rate as in standard AdamW.
        
        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.
        
        Returns
        -------
        torch.Tensor
            The loss value (if closure is provided).
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Handle sparse gradients (skip population-risk gating)
                if grad.is_sparse:
                    raise RuntimeError(
                        "PopulationRiskAdamW does not support sparse gradients"
                    )

                amsgrad = group["amsgrad"]

                # Get or initialize state for this parameter
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1 - group["betas"][0] ** state["step"]
                bias_correction2 = 1 - group["betas"][1] ** state["step"]

                # Decay the first and second moment running averages
                exp_avg.mul_(group["betas"][0]).add_(grad, alpha=1 - group["betas"][0])
                exp_avg_sq.mul_(group["betas"][1]).addcmul_(
                    grad, grad, value=1 - group["betas"][1]
                )

                # Apply bias correction
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2

                # ============================================================
                # POPULATION-RISK SNR GATE COMPUTATION
                # ============================================================
                # Gate condition: b * (m_t)^2 > v_t
                # where m_t is the bias-corrected first moment (estimating signal μ)
                #       v_t is the bias-corrected second moment (estimating E[g²])
                #
                # Interpretation:
                #   - Signal power = (m_t)^2 (mean gradient magnitude squared)
                #   - Noise level ~ v_t - (m_t)^2 (variance of gradients)
                #   - Gate threshold = v_t / b (normalized noise level)
                #   - Update only when signal >> noise
                # ============================================================
                gate = (self.batch_size * (exp_avg_corrected ** 2)) > exp_avg_sq_corrected

                # Compute denominator (standard Adam)
                denom = exp_avg_sq_corrected.sqrt() + group["eps"]

                # Compute parameter delta (standard AdamW step)
                step_size = group["lr"] / bias_correction1
                delta = step_size * exp_avg_corrected / denom

                # Apply weight decay (AdamW style)
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["weight_decay"] * group["lr"])

                # ============================================================
                # GATED UPDATE: Multiply delta by gate mask
                # ============================================================
                # Convert boolean gate to float (True=1.0, False=0.0)
                # Then element-wise multiply with parameter delta
                # This zeros out updates for parameters with low SNR
                # ============================================================
                gated_delta = delta * gate.float()
                p.add_(gated_delta, alpha=-1)

        return loss
