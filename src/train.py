"""
Two-phase training pipeline for hurricane damage detector.

Phase 1: Train classification head with frozen ResNet50 base.
Phase 2: Fine-tune conv5 block with low learning rate.

Usage:
    python -m src.train
"""

import tensorflow as tf

from config import (
    PHASE1_EPOCHS,
    PHASE2_EPOCHS,
    PHASE2_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MODEL_DIR,
    RESULTS_DIR,
)
from src.data import download_and_extract_data, prepare_train_and_val_datasets
from src.model import build_model, unfreeze_top_layers
from src.evaluate import plot_training_history


def train() -> tf.keras.Model:
    """
    Execute the full training pipeline.

    Returns:
        The trained and fine-tuned model.
    """
    # Setup
    download_and_extract_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = prepare_train_and_val_datasets(augment=True)
    model = build_model()

    # ── Phase 1: Train classification head ──────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Training classification head (base frozen)")
    print("=" * 60)

    history_phase1 = model.fit(
        train_ds,
        epochs=PHASE1_EPOCHS,
        validation_data=val_ds,
    )

    # ── Phase 2: Fine-tune conv5 block ──────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning conv5 block")
    print("=" * 60)

    model = unfreeze_top_layers(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
        ),
    ]

    history_phase2 = model.fit(
        train_ds,
        epochs=PHASE2_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    # ── Save model and training plots ───────────────────────
    save_path = MODEL_DIR / "hurricane_detector.keras"
    model.save(str(save_path))
    print(f"\nModel saved to {save_path}")

    plot_training_history(
        history_phase1,
        history_phase2,
        save_path=RESULTS_DIR / "training_history.png",
    )

    return model


if __name__ == "__main__":
    from src.evaluate import evaluate_model

    trained_model = train()
    evaluate_model(trained_model)
