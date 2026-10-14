# Chest X-Ray Pneumonia Detection Using Deep Learning

A deep learning project that uses **Convolutional Neural Networks (CNNs) and transfer learning** to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

The project uses **TensorFlow, Keras, and EfficientNetB0** to build an image classification pipeline, evaluate model performance, visualize predictions, and provide an interactive interface using Streamlit.

> **Disclaimer:** This project is intended for educational and research purposes only. Its predictions are not a substitute for professional medical diagnosis.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Project Objectives](#-project-objectives)
* [Dataset](#-dataset)
* [Technologies Used](#-technologies-used)
* [Project Workflow](#-project-workflow)
* [Model Architecture](#-model-architecture)
* [Handling Class Imbalance](#-handling-class-imbalance)
* [Model Evaluation](#-model-evaluation)
* [Prediction Visualization](#-prediction-visualization)
* [Streamlit Web Application](#-streamlit-web-application)
* [Installation](#-installation)
* [Usage](#-usage)
* [Project Structure](#-project-structure)
* [Future Improvements](#-future-improvements)
* [Author](#-author)

---

## 🔍 Overview

Pneumonia is a respiratory infection that can be associated with visible abnormalities in chest X-ray images.

This project explores how deep learning and transfer learning can be applied to medical image classification.

The model analyzes an input chest X-ray image and predicts whether it belongs to one of the following classes:

* **NORMAL:** The image belongs to the normal class in the training dataset.
* **PNEUMONIA:** The image belongs to the pneumonia class in the training dataset.

The project covers the complete machine learning workflow, from dataset exploration and preprocessing to model training, evaluation, prediction visualization, and web application development.

## 🎯 Project Objectives

* Develop a deep learning model for binary chest X-ray image classification.
* Apply transfer learning using the EfficientNetB0 architecture.
* Preprocess and resize images to a consistent input shape.
* Explore class imbalance and investigate strategies to improve minority-class representation.
* Evaluate model performance using multiple classification metrics.
* Visualize individual predictions and model evaluation results.
* Build an interactive Streamlit application for image uploads and predictions.
* Gain practical experience in computer vision and deep learning deployment.

---

## 📂 Dataset

The dataset contains chest X-ray images organized into two classes:

| Class     | Description                     |
| --------- | ------------------------------- |
| NORMAL    | Images categorized as normal    |
| PNEUMONIA | Images categorized as pneumonia |

The dataset is organized into three subsets:

* **Training set:** Used to train the model.
* **Validation set:** Used to monitor training and support model selection.
* **Test set:** Used for final evaluation on unseen images.

### Dataset Distribution

The training dataset initially contains the following class distribution:

| Class     | Number of Images | Percentage |
| --------- | ---------------: | ---------: |
| NORMAL    |            1,341 |     25.71% |
| PNEUMONIA |            3,875 |     74.29% |
| **Total** |        **5,216** |   **100%** |

The class distribution indicates a significant imbalance between the two categories.

To address this issue, the project explores data augmentation and oversampling strategies to increase the representation of the minority class during training.

**Note:** These figures describe the training dataset and should not be interpreted as the sizes of the validation or test sets.

---

## 🛠️ Technologies Used

| Technology     | Purpose                                     |
| -------------- | ------------------------------------------- |
| Python         | Core programming language                   |
| TensorFlow     | Deep learning framework                     |
| Keras          | Model development and training              |
| EfficientNetB0 | Transfer learning architecture              |
| NumPy          | Numerical operations                        |
| Pandas         | Data analysis                               |
| Matplotlib     | Data and prediction visualization           |
| Seaborn        | Statistical visualization                   |
| Scikit-learn   | Model evaluation metrics                    |
| OpenCV         | Image processing                            |
| Streamlit      | Interactive web application                 |
| Google Colab   | Model development and GPU-assisted training |
| Git and GitHub | Version control and project hosting         |

---

## 🔄 Project Workflow

The project follows an end-to-end computer vision pipeline.

1. **Data Collection:** Load the chest X-ray image dataset.
2. **Exploratory Data Analysis:** Examine class distributions and sample images.
3. **Data Preprocessing:** Resize images to `224 × 224` pixels and prepare them for model input.
4. **Data Augmentation:** Apply suitable transformations to training images.
5. **Class Imbalance Handling:** Explore oversampling and augmentation for the minority class.
6. **Transfer Learning:** Use EfficientNetB0 with pretrained ImageNet weights.
7. **Model Training:** Train the classification layers and evaluate validation performance.
8. **Model Evaluation:** Analyze classification metrics and prediction errors.
9. **Prediction Visualization:** Display input images alongside their predicted classes.
10. **Deployment:** Build an interactive Streamlit application.

---

## 🧠 Model Architecture

The project uses **EfficientNetB0**, a convolutional neural network architecture pretrained on ImageNet.

Transfer learning allows the model to reuse previously learned visual features rather than learning every feature from scratch.

### Model Pipeline

```text
Chest X-ray Image
        |
        v
Image Loading and Preprocessing
        |
        v
Resize to 224 × 224 × 3
        |
        v
EfficientNetB0
(Pretrained ImageNet Weights)
        |
        v
Global Average Pooling
        |
        v
Dense Classification Layers
        |
        v
Binary Classification
        |
        v
NORMAL / PNEUMONIA
```

### Training Configuration

| Parameter             | Configuration                    |
| --------------------- | -------------------------------- |
| Model                 | EfficientNetB0                   |
| Pretrained Weights    | ImageNet                         |
| Input Image Size      | 224 × 224                        |
| Input Channels        | 3 (RGB)                          |
| Batch Size            | 32                               |
| Optimizer             | Adam                             |
| Initial Learning Rate | 0.001                            |
| Training Strategy     | Transfer Learning                |
| Early Stopping        | Used to help prevent overfitting |

The pretrained feature extractor is initially frozen so that the newly added classification layers can learn the target task.

Fine-tuning can be explored later by selectively unfreezing some pretrained layers and training with a smaller learning rate.

*The configuration above describes the intended training setup. The final architecture and hyperparameters should match the model used in the completed training run.*

---

## ⚖️ Handling Class Imbalance

The original training dataset contains substantially more PNEUMONIA images than NORMAL images.

This imbalance can affect the model's learning behavior and may cause it to favor the majority class.

The project explores the following approaches:

### 1. Data Augmentation

Apply suitable transformations to training images, such as:

* Small rotations
* Horizontal translations
* Zoom transformations
* Contrast adjustments

Augmentation should be applied only to training data and should preserve medically meaningful image characteristics.

### 2. Oversampling

Increase the representation of the NORMAL class during training by sampling additional examples from the available minority-class images.

Augmented versions of existing images can help introduce variation without requiring an equal number of original images in each class.

### 3. Evaluation Beyond Accuracy

Accuracy alone may hide poor performance on the minority class.

Precision, recall, F1-score, and the confusion matrix provide additional information about classification errors.

**Important:** Augmentation and oversampling should be performed only on the training set. Validation and test datasets must remain independent and unaltered for reliable evaluation.

---

## 📊 Model Evaluation

The model should be evaluated using multiple metrics to understand its classification performance.

| Metric           | Purpose                                                   |
| ---------------- | --------------------------------------------------------- |
| Accuracy         | Measures the proportion of correct predictions            |
| Precision        | Measures how many positive predictions are correct        |
| Recall           | Measures how many actual positive examples are identified |
| F1-score         | Combines precision and recall                             |
| Confusion Matrix | Shows correct and incorrect predictions for each class    |
| ROC-AUC          | Measures discrimination across classification thresholds  |

### Evaluation Visualizations

The project can include:

* Training and validation accuracy curves
* Training and validation loss curves
* Confusion matrix
* Classification report
* Individual prediction visualizations
* ROC curve

Actual test results should be added after completing the final evaluation.

| Metric    | Test Result    |
| --------- | -------------- |
| Accuracy  | To be measured |
| Precision | To be measured |
| Recall    | To be measured |
| F1-score  | To be measured |
| ROC-AUC   | To be measured |

No performance values are reported here because verified final test results have not yet been established.

---

## 🖼️ Prediction Visualization

The prediction pipeline loads a chest X-ray image, preprocesses it, passes it through the trained model, and displays the predicted class.

The visualization can include:

* Original chest X-ray image
* Actual class label, when available
* Predicted class
* Prediction confidence, when supported by the model output
* Correct or incorrect prediction status

For test-set evaluation, predictions should be generated from the same preprocessing pipeline used during model development.

Displaying multiple predictions together helps identify patterns in classification errors and understand where the model may struggle.

---

## 🌐 Streamlit Web Application

The project includes a planned interactive dashboard built with Streamlit.

### Features

* Upload a chest X-ray image.
* Preview the uploaded image.
* Run the trained model on the image.
* Display the predicted class.
* Display the model's output score when available.
* Present a simple, user-friendly interface.

### Application Workflow

```text
User Uploads an X-ray Image
            |
            v
Image Preprocessing
            |
            v
Load Trained Model
            |
            v
Generate Prediction
            |
            v
Display Predicted Class
```

The Streamlit application provides a convenient way to interact with the trained model without manually executing notebook cells.

---

## 💻 Installation

### Prerequisites

* Python 3.10 or another Python version compatible with your TensorFlow installation
* pip
* Git
* A trained model file for running predictions
* The chest X-ray dataset for training and evaluation

### 1. Clone the Repository

```bash
git clone https://github.com/shiv-1024/chest-xray-pneumonia-detection.git
```

Replace the repository name in the command if your actual GitHub repository uses a different name.

### 2. Navigate to the Project Directory

```bash
cd chest-xray-pneumonia-detection
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

Create a `requirements.txt` file containing the packages required by your implementation.

For example:

```text
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
streamlit
Pillow
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For reproducible installations, use the exact package versions tested with your final project.

---

## 🚀 Usage

### Training the Model

1. Download and extract the dataset.
2. Organize the images into training, validation, and test directories.
3. Update the dataset paths in the notebook or Python scripts.
4. Run the preprocessing and exploratory data analysis steps.
5. Train the model using the selected transfer learning architecture.
6. Evaluate the model on the validation set.
7. Save the trained model for later inference.

### Making Predictions

Load the trained model and pass a preprocessed chest X-ray image to the prediction function.

The preprocessing used during inference must be consistent with the model's training pipeline.

### Launching the Streamlit Application

If your Streamlit application is saved as `app.py`, run:

```bash
streamlit run app.py
```

Streamlit will start a local web server and provide an address where you can open the application.

**Note:** The training notebook, trained model, and Streamlit application must be included in or correctly referenced by the repository before these instructions will work as written.

---

## 📁 Project Structure

A recommended repository structure is:

```text
chest-xray-pneumonia-detection/
│
├── notebooks/
│   └── chest_xray_pneumonia_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The exact structure can be adjusted to match the files in your repository.

**Dataset and model files:** Large image datasets and trained model files do not necessarily belong in Git. Consider using a dataset download link and Git LFS or an appropriate model-hosting service when needed.

---

## 🔮 Future Improvements

* Fine-tune pretrained EfficientNetB0 layers.
* Compare EfficientNetB0 with ResNet50, DenseNet121, and other CNN architectures.
* Experiment with class weights and alternative sampling strategies.
* Perform systematic hyperparameter tuning.
* Evaluate generalization on an independent test dataset.
* Add Grad-CAM visualizations to explore which image regions influence predictions.
* Improve the Streamlit dashboard with model evaluation charts.
* Package the inference pipeline for deployment.
* Add automated testing and reproducible training configurations.

---

## 👨‍💻 Author

**P Sivarajadurai**

B.Tech Graduate in Artificial Intelligence and Data Science

* GitHub: [@shiv-1024](https://github.com/shiv-1024)
* LinkedIn: [Sivarajadurai P](https://www.linkedin.com/in/sivarajadurai-p-33904b252/)

### Skills Demonstrated

`Python` · `Deep Learning` · `Computer Vision` · `TensorFlow` · `Keras` · `Transfer Learning` · `EfficientNetB0` · `Image Classification` · `Data Augmentation` · `Model Evaluation` · `Streamlit`

---

⭐ If you find this project interesting, consider giving the repository a star!
