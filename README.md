# Hurricane Damage Detector

Binary image classifier that detects hurricane damage in satellite
imagery using transfer learning with ResNet50 and Grad-CAM
for model interpretability.

## Results

> **Note:** Results will be updated after training. Values below are placeholders.

| Metric    | No Damage | Damage | Overall |
|-----------|-----------|--------|---------|
| Precision | —         | —      | —       |
| Recall    | —         | —      | —       |
| F1-Score  | —         | —      | —       |

<!-- TODO: Insert training_history.png, confusion_matrix.png, gradcam.png -->

## Approach

### Why Transfer Learning with ResNet50

With ~10,000 training images at 128×128 px, training a deep CNN from
scratch would likely overfit. ResNet50 pretrained on ImageNet provides
strong general-purpose feature extraction (edges, textures, shapes),
and its skip connections help maintain gradient flow during fine-tuning.

### Two-Phase Training Strategy

- **Phase 1:** Frozen ResNet50 base — train only the classification head
  (GlobalAveragePooling2D → Dense sigmoid). This lets the head learn to
  map pretrained features to damage/no_damage decisions.
- **Phase 2:** Unfreeze the conv5 block and fine-tune with a low learning
  rate (1e-5). This adapts high-level features to hurricane damage patterns
  while preserving useful low-level representations. BatchNormalization
  layers stay frozen to avoid unstable running statistics.

### Data Augmentation

- **Horizontal + vertical flips:** Satellite images have no fixed orientation
- **Random contrast:** Simulates varying lighting and weather conditions

Aggressive augmentations (rotation, crops) are avoided — at 128×128 px,
they risk destroying small damage features.

### Model Interpretability (Grad-CAM)

Grad-CAM heatmaps show which image regions drive the model's predictions.
This validates that the model focuses on meaningful features (debris, water,
structural collapse) rather than spurious correlations.

An error analysis module visualizes misclassified images with Grad-CAM
to understand failure modes.

## Project Structure

```
hurricane-damage-detector/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters and paths centralized
├── src/
│   ├── __init__.py
│   ├── data.py                # Kaggle download, augmentation, dataset prep
│   ├── model.py               # ResNet50 architecture (build + compile)
│   ├── train.py               # Two-phase training pipeline
│   ├── evaluate.py            # Metrics, confusion matrix, training plots
│   └── gradcam.py             # Grad-CAM visualizations + error analysis
├── notebooks/
│   └── exploration.ipynb      # EDA, sample images, augmentation demos
├── results/                   # Auto-generated during training
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── gradcam_samples.png
│   └── gradcam_errors.png
└── .gitignore
```

## Usage

### Setup

```bash
git clone https://github.com/ivan-matfor/hurricane-damage-detector.git
cd hurricane-damage-detector
pip install -r requirements.txt
```

### Kaggle API Setup

The dataset is downloaded automatically from Kaggle. You need a Kaggle
API token:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token
2. Place the downloaded `kaggle.json` in `~/.kaggle/`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Train

```bash
python -m src.train
```

This downloads the dataset (if needed), runs both training phases,
saves the model to `saved_model/`, and generates training plots in `results/`.

### Evaluate

```python
from src.evaluate import evaluate_model
import tensorflow as tf

model = tf.keras.models.load_model("saved_model/hurricane_detector.keras")
results = evaluate_model(model)
```

### Grad-CAM Visualizations

```python
from src.gradcam import visualize_gradcam, visualize_errors_with_gradcam
import tensorflow as tf

model = tf.keras.models.load_model("saved_model/hurricane_detector.keras")

# Sample predictions with heatmaps
visualize_gradcam(model, save_path="results/gradcam_samples.png")

# Misclassified images with heatmaps
visualize_errors_with_gradcam(model, save_path="results/gradcam_errors.png")
```

## Dataset

* **Source:** [IEEE DataPort — Detecting Damaged Buildings on Post-Hurricane
  Satellite Imagery](https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized)
  (Cao & Choe, 2018)
* **Mirror:** [Kaggle — Satellite Images of Hurricane Damage](https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage)
* **Images:** ~23,000 satellite patches (128×128 px, RGB) from Hurricane Harvey (2017)
* **Splits:**
  - Train: 10,000 images (5,000 per class)
  - Validation: 2,000 images (1,000 per class)
  - Test: 9,000 images (unbalanced)
* **Classes:** `damage`, `no_damage`
* **Citation:** Cao, Q.D. & Choe, Y. (2018). DOI: [10.21227/sdad-1e56](https://dx.doi.org/10.21227/sdad-1e56)

## Key Decisions & Tradeoffs

- **128×128 resolution:** Matches the original dataset. Larger resolutions
  would require upscaling (no real information gain) and more compute.
- **Binary crossentropy over categorical:** Single sigmoid output is more
  efficient for binary problems and avoids the overhead of one-hot encoding.
- **EarlyStopping patience=3:** Prevents overfitting without stopping too
  early during fine-tuning, where loss can be noisy.
- **Only conv5 unfrozen:** Conservative fine-tuning approach. Lower blocks
  capture universal features; only the highest block adapts to domain.

## Future Improvements

- [ ] Experiment with EfficientNetB0 and compare metrics
- [ ] Increase Grad-CAM samples and add systematic error analysis
- [ ] Multi-class severity levels (minor / moderate / severe)
- [ ] Deploy as a simple web app with Streamlit or Gradio

## Author

**[Ivan Mateo Forcen](https://github.com/ivan-matfor)**
