# Hurricane Damage Detector

#### -- Project Status: [In Process]

## Project Intro/Objective

Binary image classifier that detects hurricane damage in satellite imagery using transfer learning with ResNet50. The model classifies 128x128 satellite image patches as `damage` or `no_damage`, achieving ~94% test accuracy on 12,228 images. Built as a portfolio project to demonstrate applied deep learning for computer vision in the context of real-world disaster response.

### Methods Used

* Transfer Learning (ResNet50, ImageNet weights)
* Two-Phase Training (frozen base → fine-tuning conv5 block)
* Data Augmentation (flips, random contrast)
* Binary Image Classification
* Model Interpretability (Grad-CAM) *(planned)*

### Technologies

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Scikit-Learn
* Matplotlib
* gdown

## Project Description

Satellite and aerial imagery are increasingly used by agencies like FEMA and organizations such as Maxar and Planet Labs to assess damage after hurricanes and other natural disasters. Automated classification of damage from imagery enables faster triage of affected areas, prioritization of rescue operations, and more efficient allocation of emergency resources.

This project uses a ResNet50 backbone pretrained on ImageNet, fine-tuned on a dataset of ~23,000 satellite image patches (128x128 px) from areas affected by Hurricane Harvey. Training follows a two-phase strategy: first training only the classification head with the base frozen, then fine-tuning the top convolutional block (conv5) with a low learning rate (1e-5) to adapt high-level features to hurricane damage patterns.

**Planned improvements:**

* Grad-CAM visualizations to show which image regions drive the model's predictions
* Error analysis on misclassified samples
* Model comparison (ResNet50 vs EfficientNet vs VGG16)
* Multi-class severity levels (minor / major / destroyed)

## Project Structure

```
hurricane-damage-detector/
├── README.md
├── requirements.txt
├── config.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── notebooks/
│   └── exploration.ipynb
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
└── .gitignore
```

## Getting Started

1. Clone this repo:
    ```bash
    git clone https://github.com/ivan-matfor/hurricane-damage-detector.git
    cd hurricane-damage-detector
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the model (downloads dataset automatically):
    ```bash
    python -m src.train
    ```

4. Evaluate on test set:
    ```python
    from src.evaluate import evaluate_model
    import tensorflow as tf

    model = tf.keras.models.load_model("saved_model/hurricane_detector.keras")
    results = evaluate_model(model)
    ```

## Dataset

* **Source:** [IEEE DataPort — Detecting Damaged Buildings on Post-Hurricane Satellite Imagery](https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized) (Cao & Choe, 2018)
* **Mirror:** [Kaggle — Satellite Images of Hurricane Damage](https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage)
* **Images:** ~23,000 satellite patches (128×128 px, RGB) from Hurricane Harvey (2017)
* **Test set:** 12,228 images
* **Classes:** `damage`, `no_damage`
* **Split:** 80/20 train/validation from train set; separate test set
* **Citation:** Cao, Q.D. & Choe, Y. (2018). *Detecting Damaged Buildings on Post-Hurricane Satellite Imagery Based on Customized Convolutional Neural Networks.* DOI: [10.21227/sdad-1e56](https://dx.doi.org/10.21227/sdad-1e56)

## Featured Notebooks/Deliverables

* [Exploration Notebook](notebooks/exploration.ipynb) — EDA, sample images, augmentation demos
* [Results](results/) — Training curves, confusion matrix, classification report

## Contributing Members

**Author: [Ivan Mateo Forcen](https://github.com/ivan-matfor)**
