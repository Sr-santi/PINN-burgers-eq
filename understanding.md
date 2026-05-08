Here is a concise **“what happened”** plus the **step-by-step process** implemented in `PINN_model.ipynb`, aligned with the notebook sections.

---

### What happened (context)

Two practical issues showed up during development:

1. **Exact reference solution** — Your specification used a truncated **Fourier/Bessel-series** Cole–Hopf form. At \(\nu = 0.01/\pi\) that series is **numerically nasty in float64** (huge cancellations near \(x=\pm 1\) and \(t=0\)), so the notebook **does not** use those coefficients directly. Instead it uses the **same underlying Cole–Hopf solution**, but evaluates it via a **Hopf integral** rewritten as **Gauss–Hermite expectations** (Section 2 + `burgers_exact`). That gives a stable reference on a \(256\times 100\) grid for validation.

2. **Notebook execution** — At one point the **title cell was a code cell**, so automated execution tried to run Markdown as Python and failed with a **syntax error**. That was fixed by making the first cell **markdown** again.

There is also an optional **fast path**: if the environment variable `PINN_FAST_NOTEBOOK` is `1` / `true` / `yes`, `CFG` shortens Adam and L-BFGS so a full `nbconvert` run finishes quickly (smoke test), while the default `CFG` keeps the long training schedule from your spec.

---

### Step-by-step process (as implemented)

1. **Problem setup**  
   Solve viscous Burgers  
   \(u*t + u u_x = \nu u*{xx}\) on \(x\in[-1,1]\), \(t\in[0,1]\), with  
   \(u(x,0)=-\sin(\pi x)\) and \(u(\pm 1,t)=0\), with **Phase 1** fixed \(\nu = 0.01/\pi\).

2. **Imports & device**  
   NumPy, SciPy `qmc`, PyTorch, Matplotlib; fixed seeds; `float32` default; train on **CUDA** if available.

3. **Configuration (`PINNConfig`)**  
   Hyperparameters: domain bounds, \(N*{ic}\), \(N*{bc}\), \(N*f=10{,}000\) collocation points, MLP depth/width, `tanh` activation, loss weights \(w*{ic},w\_{bc},w_f\), Adam/L-BFGS settings, log interval, plus optional fast mode above.

4. **Analytical ground truth**  
   Build `burgers_exact(x, t, nu)` with **Gauss–Hermite** quadrature (variable \(s\) after \(y = x + \sqrt{2\nu t}\,s\)), shift `h` by its min for numerical range, and at **\(t=0\)** set \(u\) to \(-\sin(\pi x)\) exactly.  
   Then `exact_on_grid` fills `X_GRID`, `T_GRID`, `U_EXACT` for plots and error metrics.

5. **Training data (LHS)**  
   `scipy.stats.qmc.LatinHypercube` produces:
   - **IC points**: random \(x\) at \(t=0\), targets \(-\sin(\pi x)\).
   - **BC points**: \(x=\pm 1\), random \(t\), targets \(0\).
   - **Collocation**: \((x,t)\) in the interior domain, used only for the **PDE residual** (target \(0\)).  
     Tensors are moved to the device; collocation \((x_f,t_f)\) has **`requires_grad=True`** for derivatives.

6. **Network (`PINN`)**  
   MLP: input \((x,t)\) → scalar \(\hat u\); **no ReLU** (per spec: need smooth second derivatives); **tanh** (or swish via SiLU in code comments).

7. **Physics residual**  
   `physics_residual` uses `torch.autograd.grad` with `create_graph=True` to form  
   \(f = \hat u*t + \hat u \hat u_x - \nu \hat u*{xx}\) at collocation points.

8. **Loss**  
   Weighted sum of MSEs: IC fit, BC fit, and **mean \(f^2\)** (physics).

9. **Optimization (two stages)**
   - **Adam** for many steps (exploration).
   - **L-BFGS** with a **closure** (exploitation / sharper residual reduction).  
     Periodically, **relative \(L_2\) error** on the reference grid is computed by `predict_on_grid` vs `U_EXACT`.

10. **Validation & plots**  
    Final \(\hat u\) on the grid, **relative \(L_2\)**, max error, heatmaps (exact / PINN / absolute error), and time snapshots.

11. **Phase 2 (documented only)**  
    The last markdown section describes extending inputs to \((x,t,\nu)\) and sampling \(\nu\) on a log scale—**not trained in this notebook**, just the hand-off.

---

If you want this tied to **exact function names** in the file, the main anchors are: `PINNConfig`, `burgers_exact` / `exact_on_grid`, `build_dataset` / `TrainingTensors`, `PINN`, `physics_residual`, `composite_loss`, and `train_pinn`.
