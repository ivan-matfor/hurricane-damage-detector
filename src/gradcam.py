"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.

Generates heatmaps showing which regions of an image the model focuses on
when making damage/no_damage predictions. This is critical for:
    - Validating that the model looks at meaningful features (debris, water, structural damage)
    - Detecting spurious correlations (e.g., always focusing on image corners)
    - Building trust with stakeholders by explaining model decisions

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
    https://arxiv.org/abs/1610.02391
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import (
    CLASS_NAMES,
    GRADCAM_LAYER_NAME,
    GRADCAM_NUM_SAMPLES,
    RESULTS_DIR,
    IMG_DIMS,
)
from src.data import prepare_test_dataset


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    image: tf.Tensor,
    target_layer_name: str = GRADCAM_LAYER_NAME,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single image.

    How it works:
        1. Forward pass: get the target layer's output and the model's prediction.
        2. Backward pass: compute gradients of the prediction w.r.t. the target
           layer's activations.
        3. Weight each activation channel by the mean gradient (global average pooling
           of gradients = importance weight per channel).
        4. Weighted sum of activation channels → raw heatmap.
        5. ReLU (keep only features that positively influence the prediction)
           and normalize to [0, 1].

    Args:
        model: Trained Keras model.
        image: Single image tensor of shape (H, W, 3).
        target_layer_name: Name of the convolutional layer to visualize.
            Default is the last conv layer in ResNet50's conv5 block.

    Returns:
        Heatmap as a 2D numpy array normalized to [0, 1].
    """
    # Build two sub-models to avoid cross-graph tensor issues with nested models:
    # inner_model: resnet50_input -> [target_conv_output, resnet50_output]
    # The classification head layers are applied manually afterwards.
    base_model = model.get_layer("resnet50")
    inner_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(target_layer_name).output, base_model.output],
    )

    # Collect layers after ResNet50 (GlobalAveragePooling2D + Dense)
    post_resnet_layers = []
    resnet_found = False
    for layer in model.layers:
        if resnet_found:
            post_resnet_layers.append(layer)
        if layer.name == "resnet50":
            resnet_found = True

    image_batch = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        # Apply ResNet50 preprocessing (same op used during training)
        preprocessed = tf.keras.applications.resnet50.preprocess_input(
            tf.cast(image_batch, tf.float32)
        )
        # Get conv activations and ResNet50 output in one forward pass
        conv_outputs, resnet_out = inner_model(preprocessed, training=False)
        # Apply classification head
        x = resnet_out
        for layer in post_resnet_layers:
            x = layer(x)
        pred_score = x[:, 0]

    # Gradients of the prediction w.r.t. the conv layer output
    grads = tape.gradient(pred_score, conv_outputs)

    # Global average pooling of gradients → importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of activation channels
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: keep only positive contributions
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1]
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        image: Original image as uint8 array (H, W, 3).
        heatmap: Grad-CAM heatmap (H_map, W_map), values in [0, 1].
        alpha: Blending factor (0 = only image, 1 = only heatmap).

    Returns:
        Blended image as uint8 array (H, W, 3).
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis],
        (image.shape[0], image.shape[1]),
    ).numpy()[:, :, 0]

    # Apply colormap (jet)
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    image_uint8 = image.astype(np.uint8)
    overlay = (
        (1 - alpha) * image_uint8 + alpha * heatmap_colored
    ).astype(np.uint8)

    return overlay


def visualize_gradcam(
    model: tf.keras.Model,
    num_samples: int = GRADCAM_NUM_SAMPLES,
    save_path: str | Path | None = None,
) -> None:
    """
    Generate Grad-CAM visualizations for sample test images.

    Creates a grid showing (3 columns per row):
        - Original image with true label
        - Grad-CAM heatmap
        - Heatmap overlay with predicted label

    Args:
        model: Trained Keras model.
        num_samples: Number of sample images to visualize.
        save_path: Path to save the figure. If None, only displays.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_ds = prepare_test_dataset()

    # Collect sample images and labels
    images, labels, predictions = [], [], []
    for img_batch, lbl_batch in test_ds.take(1):
        batch_preds = model.predict(img_batch, verbose=0).ravel()
        for i in range(min(num_samples, len(img_batch))):
            images.append(img_batch[i].numpy())
            labels.append(int(lbl_batch[i].numpy()))
            predictions.append(float(batch_preds[i]))

    # Create visualization grid
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]  # Ensure 2D indexing

    for i in range(len(images)):
        img = images[i]
        true_label = CLASS_NAMES[labels[i]]
        pred_score = predictions[i]
        pred_label = CLASS_NAMES[int(pred_score > 0.5)]

        # Compute heatmap
        heatmap = compute_gradcam_heatmap(model, tf.constant(img))
        overlay = overlay_heatmap(img, heatmap)

        # Resize heatmap for standalone display
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis],
            IMG_DIMS,
        ).numpy()[:, :, 0]

        # Original image
        axes[i, 0].imshow(img.astype(np.uint8))
        axes[i, 0].set_title(f"True: {true_label}")
        axes[i, 0].axis("off")

        # Heatmap only
        axes[i, 1].imshow(heatmap_resized, cmap="jet")
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis("off")

        # Overlay
        correct = "✓" if true_label == pred_label else "✗"
        color = "green" if true_label == pred_label else "red"
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            f"Pred: {pred_label} ({pred_score:.2f}) {correct}",
            color=color,
        )
        axes[i, 2].axis("off")

    plt.suptitle(
        "Grad-CAM: What the Model Sees",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM visualization saved to {save_path}")
    plt.show()


def visualize_errors_with_gradcam(
    model: tf.keras.Model,
    max_errors: int = 8,
    save_path: str | Path | None = None,
) -> None:
    """
    Visualize Grad-CAM specifically on misclassified images.

    This is crucial for error analysis: understanding WHY the model
    gets certain images wrong (ambiguous damage, vegetation confusion,
    cloud cover, etc.).

    Args:
        model: Trained Keras model.
        max_errors: Maximum number of misclassified images to show.
        save_path: Path to save the figure.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_ds = prepare_test_dataset()

    # Collect misclassified images
    errors = []
    for img_batch, lbl_batch in test_ds:
        preds = model.predict(img_batch, verbose=0).ravel()
        pred_binary = (preds > 0.5).astype(int)

        for i in range(len(img_batch)):
            true_lbl = int(lbl_batch[i].numpy())
            if pred_binary[i] != true_lbl:
                errors.append(
                    (img_batch[i].numpy(), true_lbl, float(preds[i]))
                )
            if len(errors) >= max_errors:
                break
        if len(errors) >= max_errors:
            break

    if not errors:
        print("No misclassified images found!")
        return

    n = len(errors)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (img, true_lbl, pred_score) in enumerate(errors):
        true_name = CLASS_NAMES[true_lbl]
        pred_name = CLASS_NAMES[int(pred_score > 0.5)]

        heatmap = compute_gradcam_heatmap(model, tf.constant(img))
        overlay = overlay_heatmap(img, heatmap)
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis], IMG_DIMS
        ).numpy()[:, :, 0]

        axes[i, 0].imshow(img.astype(np.uint8))
        axes[i, 0].set_title(f"True: {true_name}", color="green")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap_resized, cmap="jet")
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            f"Pred: {pred_name} ({pred_score:.2f})", color="red"
        )
        axes[i, 2].axis("off")

    plt.suptitle(
        "Grad-CAM: Error Analysis (Misclassified Images)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Error analysis plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    model = tf.keras.models.load_model("saved_model/hurricane_detector.keras")
    visualize_gradcam(model, save_path=RESULTS_DIR / "gradcam_samples.png")
    visualize_errors_with_gradcam(model, save_path=RESULTS_DIR / "gradcam_errors.png")


