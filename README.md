# Chest X-Ray Pneumonia Classification

A deep-learning project that classifies chest X-ray images into
**NORMAL** and **PNEUMONIA** categories. The notebook explores the
dataset, addresses class imbalance with augmentation-based oversampling,
compares a transfer-learning model with a custom CNN, evaluates
predictions, and includes a function for predicting the class of an
individual X-ray image.

> **Important:** This project is an educational machine-learning
> demonstration, not a medical diagnostic tool. Predictions must not be
> used to make healthcare decisions.

## Project Overview

The project uses TensorFlow/Keras to build and compare two
image-classification models:

1.  **Transfer learning with EfficientNetB0** pretrained on ImageNet.
2.  **A custom CNN** built with convolutional and max-pooling layers.

The notebook reports that the pretrained model performed better in its
comparison and uses it for the example prediction workflow. No fixed
accuracy figure is stated here because the notebook's final metrics
depend on the run and environment.

## Dataset

The notebook downloads the Chest X-Ray dataset from the following
archive:

`https://d3ilbtxij3aepc.cloudfront.net/projects/CNN-PROJECT-7-11/Chest-Xray-2.zip`

The dataset contains two classes:

-   `NORMAL` --- X-ray images labelled normal.
-   `PNEUMONIA` --- X-ray images labelled as pneumonia.

The expected directory structure is:

``` text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The notebook uses **224 × 224** image dimensions and a **batch size of
32**. It also counts the images per class, visualizes class
distribution, and displays sample X-ray images.

### Check the data split

In the notebook, `val_dataset` is loaded from `TEST_DIR`, while
`test_dataset` is loaded from `VAL_DIR`. This swaps the usual validation
and test assignments. Confirm that this is intentional before training
or evaluating the model. Keep the test set separate from model selection
and validation.

## Workflow

### 1. Data loading and exploration

-   Download and extract the dataset.
-   Load images with `tf.keras.utils.image_dataset_from_directory`.
-   Inspect image and label batch shapes.
-   Count class samples and visualize the class distribution.
-   Display example X-ray images.

### 2. Class-imbalance handling

The notebook identifies an imbalance between the NORMAL and PNEUMONIA
classes. It separates NORMAL examples from the training dataset and
applies augmentation to create **700 additional augmented NORMAL
samples**.

Augmentation operations: - Random rotation - Random translation - Random
zoom

The generated examples are concatenated with the original training
examples and shuffled to form the balanced training dataset.
Augmentation is applied to training data only; validation and test data
should remain unchanged.

### 3. Model A --- EfficientNetB0 transfer learning

The pretrained EfficientNetB0 base uses ImageNet weights and has its
layers frozen during the notebook's initial training. The classification
head contains: - Global average pooling - Dense layer with 128 units and
ReLU activation - Dropout with a rate of 0.3 - Softmax output layer for
the two classes

Training configuration in the notebook: - Optimizer: Adam - Learning
rate: `0.001` - Loss: Sparse categorical cross-entropy - Metric:
Accuracy - Maximum epochs: `15` - Early stopping: monitors validation
loss, patience `3`, restores best weights

### 4. Model B --- custom CNN

The notebook also defines a CNN from scratch with: - Three convolutional
blocks using 32, 64, and 128 filters - ReLU activations and max-pooling
layers - Flatten layer - Dense layer with 128 units and ReLU
activation - Softmax output layer

This model is trained for up to 15 epochs with Adam and sparse
categorical cross-entropy. The notebook plots the training and
validation loss and accuracy of both models for comparison.

### 5. Evaluation

The notebook evaluates the selected pretrained model using predictions
on the dataset assigned to `test_dataset`. It produces: - A
classification report - A confusion matrix heatmap - Training/validation
loss plots - Training/validation accuracy plots

For a trustworthy final evaluation, verify the dataset split assignment
and report metrics from a held-out test set that was not used for model
selection.

### 6. Single-image prediction

The `prediction_system(image_path)` function: 1. Reads an image with
OpenCV. 2. Converts the image from BGR to RGB. 3. Resizes it to 224 ×
224. 4. Adds a batch dimension. 5. Runs inference with the trained
model. 6. Selects the class with the highest softmax score. 7. Displays
the image, predicted class, and confidence percentage.

The notebook's example calls use image paths from the extracted dataset.
Update the image path to match your environment.

## Tech Stack

-   Python
-   TensorFlow / Keras
-   EfficientNetB0
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   OpenCV
-   scikit-learn
-   Jupyter Notebook / Kaggle environment

## Getting Started

### Requirements

Install the main dependencies in your Python environment:

``` bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn pillow
```

The notebook also uses `requests` to download the dataset.

### Run the notebook

1.  Open `main.ipynb` in Jupyter Notebook or upload it to a compatible
    notebook environment such as Kaggle.
2.  Ensure the environment has enough memory for image loading and model
    training. GPU acceleration may help when available.
3.  Run the dataset download and extraction cells, and check that the
    directory paths match your environment.
4.  Verify the validation/test assignments before training.
5.  Run the cells in order to explore the data, build the datasets,
    train both models, compare results, and evaluate predictions.
6.  Call `prediction_system("/path/to/xray_image.jpeg")` with the path
    to an image you want to inspect.

## Repository Structure

``` text
.
├── main.ipynb       # Data exploration, preprocessing, training, evaluation, prediction
└── README.md        # Project documentation
```

The dataset is downloaded separately by the notebook and is not assumed
to be included in the repository.

## Notes and Limitations

-   This is a binary image-classification project for learning and
    experimentation.
-   A model's confidence score is not the same as clinical certainty.
-   The notebook's preprocessing includes a scaled dataset
    (`image / 255.0`) but the transfer-learning model is trained on
    `balanced_training_ds`, not `scaled_train_ds`. Review preprocessing
    consistency when reproducing or improving the experiment.
-   The notebook sets dataset cardinality to 185 batches. Confirm that
    the actual number of batches matches this assertion if dataset size
    or batch size changes.
-   Validate the data split, class-wise precision/recall, confusion
    matrix, and generalization before drawing conclusions.
-   This project is not intended for clinical use.

## Future Improvements

-   Correct and document the validation/test split.
-   Save the trained model and add a reproducible inference script.
-   Add precision, recall, F1-score, and ROC-AUC tracking.
-   Tune the decision threshold and examine class-wise errors.
-   Add a simple web interface for image upload and prediction.
-   Evaluate on an independent dataset and obtain appropriate clinical
    validation before any real-world medical use.

## Author

Add your name, GitHub profile, and project links here.
