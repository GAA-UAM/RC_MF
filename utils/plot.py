import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("magma")


def plot_training_history(history, metric="rmse", name="title"):
    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    train_key = f"train_{metric}"
    val_key = f"val_{metric}"

    train_values = np.array(history.get(train_key, []))
    val_values = np.array(history.get(val_key, []))

    epochs = np.arange(1, len(train_values) + 1)

    axes[0].plot(
        epochs,
        train_values,
        marker="o",
        markersize=6,
        markeredgewidth=1.5,
        linewidth=2.5,
        label=f"Train {metric.upper()}",
        alpha=0.9,
    )

    if len(val_values) > 0:
        axes[0].plot(
            epochs[: len(val_values)],
            val_values,
            marker="s",
            markersize=6,
            markeredgewidth=1.5,
            linewidth=2.5,
            label=f"Val {metric.upper()}",
            alpha=0.9,
        )

    if history.get("best_epoch") is not None:
        best_epoch = history["best_epoch"] + 1
        best_val = history.get("best_val_mse")

        if metric == "rmse" and best_val is not None:
            best_val = np.sqrt(best_val)

        best_color = plt.cm.magma(0.85)

        axes[0].axvline(
            best_epoch,
            linestyle="--",
            linewidth=3,
            color=best_color,
            alpha=0.9,
            label="Best Epoch",
        )

        if best_val is not None:
            axes[0].scatter(
                best_epoch,
                best_val,
                s=150,
                color=best_color,
                edgecolors="black",
                zorder=10,
                label="Best Value",
            )

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel(metric.upper(), fontsize=12)
    axes[0].set_title(f"Training & Validation {metric.upper()}", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=True)

    train_loss = np.array(history.get("train_loss", []))
    val_loss = np.array(history.get("val_loss", []))
    epochs_loss = np.arange(1, len(train_loss) + 1)

    axes[1].plot(
        epochs_loss,
        train_loss,
        marker="o",
        markersize=6,
        markeredgewidth=1.5,
        linewidth=2.5,
        label="Train Loss",
        alpha=0.9,
    )

    if len(val_loss) > 0:
        axes[1].plot(
            epochs_loss[: len(val_loss)],
            val_loss,
            marker="s",
            markersize=6,
            markeredgewidth=1.5,
            linewidth=2.5,
            label="Val Loss",
            alpha=0.9,
        )

    if history.get("best_epoch") is not None:
        best_epoch = history["best_epoch"] + 1

        best_color = plt.cm.magma(0.85)

        axes[1].axvline(
            best_epoch,
            linestyle="--",
            linewidth=3,
            color=best_color,
            alpha=0.9,
            label="Best Epoch",
        )

        if best_epoch - 1 < len(val_loss):
            axes[1].scatter(
                best_epoch,
                val_loss[best_epoch - 1],
                s=150,
                color=best_color,
                edgecolors="black",
                zorder=10,
                label="Best Value",
            )

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_title("Training & Validation Loss", fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=True)

    plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
