"""
Model evaluation and visualization utilities.

Provides functions to:
    - Evaluate model on test set with classification report
    - Plot confusion matrix
    - Plot training history across both training phases
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm
import tensorflow as tf

from config import (
    CLASS_NAMES,
    CLASSIFICATION_THRESHOLD,
    RESULTS_DIR,
)
from src.data import prepare_test_dataset


def evaluate_model(model: tf.keras.Model) -> dict:
    """
    Run full evaluation pipeline on test set.

    Generates:
        - Classification report (printed + saved to file)
        - Confusion matrix plot (saved to results/)

    Args:
        model: Trained Keras model.

    Returns:
        Dictionary with y_true, y_pred, and classification report string.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = _predict_on_test(model)
    y_pred_binary = (y_pred > CLASSIFICATION_THRESHOLD).astype(int)

    # Classification report
    report = classification_report(
        y_true, y_pred_binary, target_names=CLASS_NAMES
    )
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(report)

    # Save report to file
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred_binary)
    _plot_confusion_matrix(cm, save_path=RESULTS_DIR / "confusion_matrix.png")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "report": report,
    }


def plot_training_history(
    history_phase1: tf.keras.callbacks.History,
    history_phase2: tf.keras.callbacks.History,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot training and validation loss/accuracy across both phases.

    The vertical dashed line marks the transition from Phase 1
    (frozen base) to Phase 2 (fine-tuning).
    """
    p1_epochs = len(history_phase1.history["loss"])
    total_epochs_p1 = list(range(1, p1_epochs + 1))
    total_epochs_p2 = list(
        range(
            p1_epochs + 1,
            p1_epochs + len(history_phase2.history["loss"]) + 1,
        )
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(total_epochs_p1, history_phase1.history["loss"], "b-", label="Train (P1)")
    ax1.plot(
        total_epochs_p1, history_phase1.history["val_loss"], "b--", label="Val (P1)"
    )
    ax1.plot(total_epochs_p2, history_phase2.history["loss"], "r-", label="Train (P2)")
    ax1.plot(
        total_epochs_p2, history_phase2.history["val_loss"], "r--", label="Val (P2)"
    )
    ax1.axvline(
        x=p1_epochs + 0.5, color="gray", linestyle=":", label="Fine-tuning start"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(
        total_epochs_p1, history_phase1.history["accuracy"], "b-", label="Train (P1)"
    )
    ax2.plot(
        total_epochs_p1,
        history_phase1.history["val_accuracy"],
        "b--",
        label="Val (P1)",
    )
    ax2.plot(
        total_epochs_p2, history_phase2.history["accuracy"], "r-", label="Train (P2)"
    )
    ax2.plot(
        total_epochs_p2,
        history_phase2.history["val_accuracy"],
        "r--",
        label="Val (P2)",
    )
    ax2.axvline(
        x=p1_epochs + 0.5, color="gray", linestyle=":", label="Fine-tuning start"
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")
    plt.show()


def _predict_on_test(model: tf.keras.Model) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions on the full test set."""
    test_ds = prepare_test_dataset()

    test_labels = []
    predictions = []

    for imgs, labels in tqdm(test_ds, desc="Predicting on test set"):
        batch_preds = model.predict(imgs, verbose=0)
        predictions.extend(batch_preds)
        test_labels.extend(labels)

    return np.array(test_labels), np.array(predictions).ravel()


def _plot_confusion_matrix(
    cm: np.ndarray, save_path: str | Path | None = None
) -> None:
    """Plot and optionally save a confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=CLASS_NAMES
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Test Set")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
