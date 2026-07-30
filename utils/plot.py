import matplotlib
matplotlib.use("Agg")  # avoids needing a display, safe for remote/headless training runs
import matplotlib.pyplot as plt


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
