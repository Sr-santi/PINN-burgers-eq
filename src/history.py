import numpy as np
import matplotlib.pyplot as plt

def plot_lightning_history(
    adam_epochs,
    adam_losses_dict,  # keys: 'total', 'ic', 'bc', 'f'
    lbfgs_iterations,
    lbfgs_losses_dict,  # keys: 'total_loss', 'ic_loss', 'bc_loss', 'f_loss'
    cfg
):
    """
    Plot combined Adam + L-BFGS training history (mimics PINN_model.ipynb style).
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Combine iterations: Adam epochs + L-BFGS iterations (offset)
    adam_x = np.array(adam_epochs)
    lbfgs_x = np.array(lbfgs_iterations) + adam_x[-1]  # Offset by final Adam epoch
    
    # Combine losses
    total_loss = np.concatenate([adam_losses_dict['total'], lbfgs_losses_dict['total_loss']])
    ic_loss = np.concatenate([adam_losses_dict['ic'], lbfgs_losses_dict['ic_loss']])
    bc_loss = np.concatenate([adam_losses_dict['bc'], lbfgs_losses_dict['bc_loss']])
    f_loss = np.concatenate([adam_losses_dict['f'], lbfgs_losses_dict['f_loss']])
    
    # Plot combined epochs
    all_x = np.concatenate([adam_x, lbfgs_x])
    
    # Loss components
    axes[0].semilogy(adam_x, adam_losses_dict['total'], label="total (Adam)", c="k", alpha=0.7)
    axes[0].semilogy(lbfgs_x, lbfgs_losses_dict['total_loss'], label="total (L-BFGS)", c="k")
    axes[0].semilogy(adam_x, adam_losses_dict['ic'], label="IC (Adam)", c="#1f77b4", alpha=0.5)
    axes[0].semilogy(lbfgs_x, lbfgs_losses_dict['ic_loss'], label="IC (L-BFGS)", c="#1f77b4")
    axes[0].semilogy(adam_x, adam_losses_dict['bc'], label="BC (Adam)", c="#d62728", alpha=0.5)
    axes[0].semilogy(lbfgs_x, lbfgs_losses_dict['bc_loss'], label="BC (L-BFGS)", c="#d62728")
    axes[0].semilogy(adam_x, adam_losses_dict['f'], label="physics (Adam)", c="#2ca02c", alpha=0.5)
    axes[0].semilogy(lbfgs_x, lbfgs_losses_dict['f_loss'], label="physics (L-BFGS)", c="#2ca02c")
    
    # Mark Adam/L-BFGS transition
    switch = adam_x[-1]
    axes[0].axvline(switch, ls="--", c="grey", alpha=0.6)
    axes[0].text(switch, axes[0].get_ylim()[1], " L-BFGS", ha="left", va="top", color="grey", fontsize=9)
    
    axes[0].set_xlabel("epoch / iteration")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Composite loss components")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc='best')
    
    # Placeholder for L2 validation error (requires tracking during training)
    # For now, just show message
    axes[1].text(0.5, 0.5, "L2 validation error\n(requires callback tracking)", 
                 ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_xlabel("epoch / iteration")
    axes[1].set_ylabel(r"$\|u_\theta - u_*\|_2 / \|u_*\|_2$")
    axes[1].set_title("L2 relative validation error")
    axes[1].grid(alpha=0.3)
    
    fig.tight_layout()
    return fig