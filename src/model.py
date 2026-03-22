"""
Model architecture: ResNet50-based binary classifier for hurricane damage detection.
"""

import tensorflow as tf
from config import IMG_SHAPE, PHASE1_LEARNING_RATE


def build_model() -> tf.keras.Model:
    """
    Build a ResNet50-based binary classifier using transfer learning.

    Architecture:
        - ResNet50 pretrained on ImageNet (frozen initially)
        - GlobalAveragePooling2D to reduce spatial dimensions
        - Dense(1, sigmoid) for binary classification

    Why ResNet50:
        - Strong feature extraction for image classification tasks
        - Skip connections help with gradient flow during fine-tuning
        - Well-documented baseline for transfer learning in CV
        - ResNet50 is a good balance between depth and compute cost

    Returns:
        Compiled Keras model ready for Phase 1 training.
    """
    base_model = tf.keras.applications.ResNet50(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, x, name="hurricane_damage_detector")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_top_layers(model: tf.keras.Model) -> tf.keras.Model:
    """
    Unfreeze the last convolutional block (conv5) for fine-tuning.

    Why only conv5:
        - Lower layers capture generic features (edges, textures)
          that transfer well across domains.
        - Higher layers capture task-specific features that benefit
          from adaptation to hurricane damage patterns.
        - Unfreezing too many layers risks catastrophic forgetting
          of useful pretrained representations.

    Why skip BatchNormalization:
        - BN layers have running mean/variance from ImageNet.
        - Unfreezing them with small fine-tuning data causes
          unstable statistics and degrades performance.

    Args:
        model: The model after Phase 1 training.

    Returns:
        The same model (modified in place) with conv5 layers unfrozen.
    """
    
    base_model = model.get_layer("resnet50")

    for layer in base_model.layers:
        if layer.name.startswith("conv5") and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        ):
            layer.trainable = True

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    total_count = len(base_model.layers)
    print(f"Trainable layers in ResNet50: {trainable_count}/{total_count}")

    return model
