# CIFAR-10 Image Classification Using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) from scratch using **TensorFlow and Keras** to perform multi-class image classification on the CIFAR-10 dataset.

The model learns spatial features from RGB images through multiple convolutional and max-pooling layers and classifies images into one of ten categories using a Softmax output layer.

The project covers the complete deep learning workflow, including dataset preparation, CNN architecture development, model training, performance evaluation, visualization, and image prediction.

> **Test accuracy: 71.28 %** after 10 epochs, from a model with only ~357 K parameters.

## Project Objectives

- Understand and implement CNN architecture for image classification.
- Perform image preprocessing and pixel normalization.
- Extract image features using convolutional and max-pooling layers.
- Train a CNN model using TensorFlow/Keras.
- Evaluate model performance using accuracy and loss metrics.
- Visualize training progress and model predictions.
- Analyze classification performance using a confusion matrix.

## Dataset – CIFAR-10

The project uses the CIFAR-10 dataset, which contains 60,000 RGB images distributed across 10 different classes.

| Dataset | Number of Images |
|---|---:|
| Training dataset | 50,000 |
| Validation dataset | 10,000 |
| Test dataset | 10,000 |
| Image dimensions | 32 × 32 × 3 |
| Number of classes | 10 |

### Classification Categories

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

Dataset source: [CIFAR-10 – University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)

## CNN Architecture

The CNN architecture is implemented from scratch without using pretrained models.

| Layer | Configuration |
|---|---|
| Input | 32 × 32 × 3 |
| Convolutional Block 1 | Conv2D (32 filters, 3 × 3) + ReLU |
| Max Pooling 1 | 2 × 2 |
| Convolutional Block 2 | Conv2D (64 filters, 3 × 3) + ReLU |
| Max Pooling 2 | 2 × 2 |
| Convolutional Block 3 | Conv2D (128 filters, 3 × 3) + ReLU |
| Flatten | 2,048 features |
| Fully Connected | Dense (128) + ReLU |
| Regularization | Dropout (0.5) |
| Output | Dense (10) + Softmax |

**Total parameters:** 356,810

**Trainable parameters:** 356,810

## Technologies and Libraries

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

## Model Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |
| Epochs | 10 |
| Batch Size | 64 |
| Validation Split | 20% |
| Dropout Rate | 0.5 |
| Input Normalization | Pixel values divided by 255 |

## Experimental Results

The model was trained for 10 epochs and evaluated on the independent CIFAR-10 test dataset.

| Metric | Result |
|---|---:|
| Training Accuracy | 75.36% |
| Validation Accuracy | 71.61% |
| Test Accuracy | 71.28% |
| Test Loss | 0.8358 |

### Model Evaluation

The following evaluation visualizations are included in the notebook:

- Training and validation accuracy curves.
- Training and validation loss curves.
- Sample image predictions with actual and predicted labels.
- Confusion matrix covering all ten CIFAR-10 classes.

The model demonstrates its ability to learn image features and distinguish between multiple object categories.

## Project Structure

```text
cifar10-cnn-image-classification/
│
├── CIFAR10_CNN_Image_Classification.ipynb
├── README.md
├── requirements.txt
```

## How to Run the Project

### Option 1: Google Colab

1. Clone or download this GitHub repository.
2. Open the `.ipynb` notebook in Google Colab.
3. Execute the notebook cells sequentially.
4. The CIFAR-10 dataset will be downloaded automatically.
5. Train the CNN model and evaluate its performance.

### Option 2: Local Environment

Install the required dependencies:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `CIFAR10_CNN_Image_Classification.ipynb` and execute the cells.

## Future Improvements

- Experiment with deeper CNN architectures.
- Implement Batch Normalization and additional regularization.
- Experiment with different optimizers and learning-rate schedules.
- Improve classification accuracy through hyperparameter tuning.
- Compare the CNN implementation with ResNet and other architectures.

## Key Learnings

- CNN architecture design and implementation from scratch.
- Image preprocessing and feature extraction.
- Model training, validation, and testing.
- Overfitting analysis and regularization using Dropout.
- Performance evaluation using confusion matrices.
- Practical implementation of deep learning using TensorFlow/Keras.
