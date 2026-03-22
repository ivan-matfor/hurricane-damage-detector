"""
Data loading, augmentation, and dataset preparation.

Downloads the Hurricane Harvey satellite imagery dataset from Kaggle
and prepares TensorFlow datasets for training, validation, and testing.

Dataset structure (from Kaggle):
    train_another/       -> 10,000 images (5,000 per class)
    validation_another/  -> 2,000 images (1,000 per class)
    test_another/        -> 9,000 images (unbalanced: ~8,000 damage / ~1,000 no_damage)
"""

import subprocess
import zipfile
import tensorflow as tf

from config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    DATA_DIR,
    IMG_DIMS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    CLASS_NAMES,
    SEED,
    TEST_BATCH_SIZE,
    KAGGLE_DATASET,
)


def download_and_extract_data() -> None:
    """
    Download dataset from Kaggle and extract if not already present.

    Requires:
        - kaggle package installed (`pip install kaggle`)
        - Kaggle API token at ~/.kaggle/kaggle.json
          (download from https://www.kaggle.com/settings → API → Create New Token)
    """
    if TRAIN_DIR.exists() and TEST_DIR.exists():
        print("Dataset already exists, skipping download.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from Kaggle: {KAGGLE_DATASET}")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(DATA_DIR),
        ],
        check=True,
    )

    # Find and extract the zip file
    zip_files = list(DATA_DIR.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError("No zip file found after Kaggle download.")

    for zip_path in zip_files:
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()

    print("Dataset ready.")
    _print_dataset_summary()


def _print_dataset_summary() -> None:
    """Print a summary of the downloaded dataset."""
    for split_dir, label in [
        (TRAIN_DIR, "Train"),
        (VALIDATION_DIR, "Validation"),
        (TEST_DIR, "Test"),
    ]:
        if split_dir.exists():
            counts = {
                cls: len(list((split_dir / cls).glob("*")))
                for cls in CLASS_NAMES
                if (split_dir / cls).exists()
            }
            total = sum(counts.values())
            print(f"  {label}: {total} images {counts}")


def prepare_train_and_val_datasets(
    augment: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load training and validation datasets.

    If a dedicated validation directory exists (Kaggle structure),
    uses it directly. Otherwise, splits training data using
    VALIDATION_SPLIT ratio.

    Args:
        augment: Whether to apply data augmentation to the training set.

    Returns:
        Tuple of (train_ds, val_ds) as tf.data.Dataset objects.
    """
    if VALIDATION_DIR.exists():
        # Use the dedicated validation split from the dataset
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            str(TRAIN_DIR),
            class_names=CLASS_NAMES,
            seed=SEED,
            image_size=IMG_DIMS,
            batch_size=BATCH_SIZE,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            str(VALIDATION_DIR),
            class_names=CLASS_NAMES,
            seed=SEED,
            image_size=IMG_DIMS,
            batch_size=BATCH_SIZE,
        )
    else:
        # Fallback: split training data
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            str(TRAIN_DIR),
            validation_split=VALIDATION_SPLIT,
            subset="training",
            class_names=CLASS_NAMES,
            seed=SEED,
            image_size=IMG_DIMS,
            batch_size=BATCH_SIZE,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            str(TRAIN_DIR),
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            class_names=CLASS_NAMES,
            seed=SEED,
            image_size=IMG_DIMS,
            batch_size=BATCH_SIZE,
        )

    if augment:
        train_ds = _apply_augmentation(train_ds)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


def prepare_test_dataset() -> tf.data.Dataset:
    """Load the test dataset for evaluation."""
    return tf.keras.preprocessing.image_dataset_from_directory(
        str(TEST_DIR),
        class_names=CLASS_NAMES,
        seed=SEED,
        image_size=IMG_DIMS,
        batch_size=TEST_BATCH_SIZE,
    )


def _apply_augmentation(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Apply data augmentation transforms to a dataset.

    Transforms applied:
        - Random horizontal flip (buildings can face either direction)
        - Random vertical flip (satellite imagery has no fixed orientation)
        - Random contrast adjustment (simulates varying lighting/weather)

    These transforms are safe for satellite/aerial imagery because
    orientation is arbitrary and lighting varies across captures.
    Rotations and crops are intentionally avoided — at 128x128 px,
    aggressive spatial transforms risk destroying small damage features.
    """
    dataset = dataset.map(
        lambda img, lbl: (tf.image.random_flip_left_right(img), lbl),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda img, lbl: (tf.image.random_flip_up_down(img), lbl),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda img, lbl: (
            tf.image.random_contrast(img, lower=0.2, upper=1.5),
            lbl,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.shuffle(2000)
    return dataset
