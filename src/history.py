import numpy as np
import matplotlib.pyplot as plt

def plot_lightning_history(
    adam_epochs,
    adam_losses_dict,  # keys: 'total', 'ic', 'bc', 'f'
    lbfgs_iterations,
    lbfgs_losses_dict,  # keys: 'total_loss', 'ic_loss', 'bc_loss', 'f_loss'
    cfg,
    adam_l2=None,       # L2 relative error per logged Adam epoch
    lbfgs_l2=None,      # L2 relative error per logged L-BFGS iteration
):
    """
    Plot Adam + L-BFGS training history
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Combine iterations: Adam epochs + L-BFGS iterations (offset)
    adam_x = np.array(adam_epochs)
    lbfgs_x = np.array(lbfgs_iterations) + (adam_x[-1] if len(adam_x) > 0 else 0)
    
    # Concatenate Adam and L-BFGS data into unified curves
    total_loss = np.concatenate([adam_losses_dict['total'], lbfgs_losses_dict['total_loss']])
    ic_loss = np.concatenate([adam_losses_dict['ic'], lbfgs_losses_dict['ic_loss']])
    bc_loss = np.concatenate([adam_losses_dict['bc'], lbfgs_losses_dict['bc_loss']])
    f_loss = np.concatenate([adam_losses_dict['f'], lbfgs_losses_dict['f_loss']])
    all_x = np.concatenate([adam_x, lbfgs_x])
    
    # Unified loss components
    axes[0].semilogy(all_x, total_loss, label="total", c="k")
    axes[0].semilogy(all_x, ic_loss, label="IC", c="#1f77b4")
    axes[0].semilogy(all_x, bc_loss, label="BC", c="#d62728")
    axes[0].semilogy(all_x, f_loss, label="physics", c="#2ca02c")
    
    # Mark Adam/L-BFGS transition
    if len(adam_x) > 0 and len(lbfgs_x) > 0:
        switch = lbfgs_x[0]
        axes[0].axvline(switch, ls="--", c="grey", alpha=0.6)
        axes[0].text(switch, axes[0].get_ylim()[1], " L-BFGS", ha="left", va="top", color="grey", fontsize=9)
    
    axes[0].set_xlabel("epoch / iteration")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Composite loss components")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc='best')
    
    # L2 relative validation error — unified curve
    if adam_l2 is not None and lbfgs_l2 is not None:
        all_l2 = np.concatenate([np.asarray(adam_l2), np.asarray(lbfgs_l2)])
        axes[1].semilogy(all_x, all_l2, c="C3")
    elif adam_l2 is not None and len(adam_l2) > 0:
        axes[1].semilogy(adam_x, adam_l2, c="C3")
    
    # Mark transition on L2 plot and target line
    if len(adam_x) > 0 and len(lbfgs_x) > 0:
        axes[1].axvline(lbfgs_x[0], ls="--", c="grey", alpha=0.6)
    
    axes[1].axhline(cfg.val_target_l2, ls="--", c="grey", label=f"target $10^{{{int(np.log10(cfg.val_target_l2))}}}$")
    axes[1].set_xlabel("epoch / iteration")
    axes[1].set_ylabel(r"$\|u_\theta - u_*\|_2 / \|u_*\|_2$")
    axes[1].set_title("L2 relative validation error")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    fig.tight_layout()