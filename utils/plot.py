"""Plotting helpers.

    plot_training_curves   train.py: loss / accuracy per epoch (saved to file)
    two_laws               notebook 6: relationship (1) and (2) side by side

No backend is set here, so notebooks still show figures inline. Headless scripts
(train.py) select the "Agg" backend themselves before importing this module.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path="training_curves.png"):
    """Plots train/val loss and accuracy across epochs side by side and saves to save_path."""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o")
    axes[0].plot(epochs, val_losses, label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, label="Train Accuracy", marker="o")
    axes[1].plot(epochs, val_accs, label="Val Accuracy", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def two_laws(l1, l2, cross, k_ref, out_png=None):
    """l1: laws.law1(...) result.  l2: laws.law2(...)[(dataset, k_ref)]."""
    t1, t2 = l1["table"], l2["table"]
    xs = np.linspace(t1.log_f.min() - 0.1, t1.log_f.max() + 0.1, 50)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.scatter(t1.log_f, t1.auc, s=70, color="#065A82", zorder=3)
    ax.plot(xs, l1["slope"] * xs + l1["intercept"], color="#065A82", lw=1.5)
    for l, r in t1.iterrows():
        ax.annotate(l[:11], (r.log_f, r.auc), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("log10 f  (% of training image-text pairs)")
    ax.set_ylabel("zero-shot AUC")
    ax.set_title(f"Law 1: slope {l1['slope']:+.3f}  (r={l1['r']:+.2f}, p={l1['p']:.3f})")
    ax.grid(alpha=.3)

    ax = axes[1]
    ax.scatter(t2.log_f, t2.delta, s=70, color="#B85042", zorder=3)
    ax.plot(xs, l2["slope"] * xs + l2["intercept"], color="#B85042", lw=1.5)
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(np.log10(cross), color="#02A37A", ls="--", lw=1.4, label=f"crossing  f={cross:.1f}%")
    for l, r in t2.iterrows():
        ax.annotate(l[:11], (r.log_f, r.delta), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("log10 f  (% of training image-text pairs)")
    ax.set_ylabel(f"AUC change at K={k_ref}")
    ax.set_title(f"Law 2: slope {l2['slope']:+.3f}  (rho={l2['rho']:+.2f}, p={l2['ps']:.3f})")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    plt.show()
