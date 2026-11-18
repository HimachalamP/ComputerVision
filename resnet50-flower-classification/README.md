# ResNet-50 Flower Image Classification with PyTorch

A computer vision project that classifies flower images into five
categories using a pretrained ResNet-50 model and transfer learning.

## Project Overview

This project uses **ResNet-50**, pretrained on ImageNet, as a feature
extractor for a five-class flower image classification task. The
original final fully connected layer is replaced with a new layer
containing five output units. The pretrained backbone parameters are
frozen, and the new classification layer is trained on the flower
dataset.

### Classes

-   Daisy
-   Dandelion
-   Roses
-   Sunflowers
-   Tulips

## Technologies

-   Python
-   PyTorch
-   Torchvision
-   NumPy
-   Matplotlib
-   Seaborn
-   scikit-learn
-   Google Colab

## Dataset and Preprocessing

The notebook loads the images with `torchvision.datasets.ImageFolder`.
The dataset directory is configured as `/content/flower_photos` in the
Colab notebook.

Expected directory structure:

``` text
flower_photos/
├── daisy/
├── dandelion/
├── roses/
├── sunflowers/
└── tulips/
```

The notebook applies the following preprocessing:

1.  Resize each image to **224 × 224** pixels.
2.  Convert the image to a PyTorch tensor.
3.  Normalize RGB channels using ImageNet statistics:
    -   Mean: `[0.485, 0.456, 0.406]`
    -   Standard deviation: `[0.229, 0.224, 0.225]`
4.  Split the dataset into 80% training and 20% validation subsets.

The DataLoader uses a batch size of 16. The notebook's sample batch has
shape `[16, 3, 224, 224]`.

## Model Architecture and Transfer Learning

The model is based on `torchvision.models.resnet50` with the default
pretrained ImageNet weights.

The final fully connected layer is replaced with:

``` python
model.fc = nn.Linear(model.fc.in_features, 5)
```

All model parameters are frozen initially, then the parameters of
`model.fc` are enabled for training. This makes the experiment a
**feature-extraction transfer-learning setup**, rather than full-network
fine-tuning.

## Training Configuration

  Parameter                                                 Value
  -------------------------- ------------------------------------
  Model                                                 ResNet-50
  Pretrained weights               ImageNet (Torchvision default)
  Epochs                                                       10
  Batch size                                                   16
  Learning rate                                             0.001
  Optimizer                                                  Adam
  Loss function                                  CrossEntropyLoss
  Input size                                        224 × 224 × 3
  Train / validation split                              80% / 20%
  Device selection             CUDA when available; otherwise CPU

## Results

The following values are copied from the notebook's recorded training
output.

    Epoch   Training accuracy   Validation accuracy
  ------- ------------------- ---------------------
        1              78.44%                89.24%
        2              89.92%                91.69%
        3              90.97%                91.83%
        4              92.78%                91.69%
        5              93.90%                92.10%
        6              94.79%                92.37%
        7              95.44%                91.96%
        8              95.61%                93.05%
        9              96.15%                92.23%
       10              96.80%            **93.32%**

**Recorded final-epoch accuracy:** - Training accuracy: **96.80%** -
Validation accuracy: **93.32%**

These are the notebook's recorded results for its 80/20 random split.
The split is not seeded in the current notebook, so a new run may
produce different results.

## Evaluation

The notebook calculates a confusion matrix using scikit-learn and
visualizes it with Seaborn. The intended evaluation compares true and
predicted labels for the five flower classes.

**Important evaluation note:** In the current notebook, `all_preds` and
`all_labels` are extended during validation on every epoch and are not
cleared between epochs. Consequently, the displayed confusion matrix
combines predictions from all 10 epochs; it does not represent the final
model's predictions from a single validation pass. Recompute the
confusion matrix after training using one validation pass with freshly
initialized prediction and label lists before treating it as the final
evaluation result.

## Repository Structure

A suggested repository layout:

``` text
resnet50-flower-classification/
├── notebooks/
│   └── resnet50-transfer-learning-5-flower-classification.ipynb
├── results/
│   └── README.md
└── README.md
```

Keep the dataset out of Git unless its license and size make
redistribution appropriate. The notebook currently expects the dataset
at `/content/flower_photos`; update that path or add a dataset setup
cell when preparing the notebook for reuse.

## How to Run

1.  Open the notebook in Google Colab or a local Jupyter environment.
2.  Make the `flower_photos` dataset available and ensure it follows the
    class-folder structure shown above.
3.  If necessary, update `data_dir` in the data-preparation cell.
4.  Install the notebook dependencies if they are not already available:
    -   `torch`
    -   `torchvision`
    -   `torchsummary`
    -   `matplotlib`
    -   `numpy`
    -   `seaborn`
    -   `scikit-learn`
5.  Run the notebook cells in order.

The notebook automatically selects CUDA when available and otherwise
uses the CPU.

## Current Scope and Future Improvements

The current notebook demonstrates transfer learning and five-class image
classification. Possible next steps:

-   Add training-only data augmentation.
-   Track and plot average training loss and validation loss per epoch.
-   Save the trained model weights and add a separate inference example
    for unseen images.
-   Compare frozen-feature extraction with selective or full
    fine-tuning.
