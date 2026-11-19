# Flower Classification with ResNet-50 (PyTorch → ONNX)

Classifying 5 flower species with a **ResNet-50 pretrained on ImageNet**. The backbone is frozen and only a new final layer is trained, in PyTorch on Google Colab (GPU). The trained model is then **exported to ONNX and checked with ONNX Runtime**.


| Metric | Result |
|---|---|
| **Validation accuracy (PyTorch)** | **92.51%** |
| **Validation accuracy (ONNX Runtime)** | **92.51%** |
| **PyTorch ↔ ONNX prediction agreement** | **100%** (734 / 734) |
| **Max logit difference** | 2.61 × 10⁻⁵ |
| **Single-image latency (GPU)** | 21.32 ms (PyTorch) → **5.95 ms (ONNX)**, **3.6× faster** |


##  Project Overview

This project uses **ResNet-50**, pretrained on ImageNet, as a feature extractor for a five-class flower image classification task. The original final fully connected layer is replaced with a new layer containing five output units. The pretrained backbone parameters are frozen, and the new classification layer is trained on the flower dataset.

| Item | Details |
|---|---|
| **Task** | Multi-class image classification (5 classes) |
| **Model** | ResNet-50 (`ResNet50_Weights.DEFAULT`, ImageNet) |
| **Approach** | Transfer learning: frozen backbone, new `fc` layer (2048 → 5, 10,245 trainable params) |
| **Framework** | PyTorch, torchvision |
| **Deployment format** | ONNX (opset 20, dynamic batch), ONNX Runtime 1.32 (CUDA EP) |
| **Dataset** | TensorFlow *flower_photos* (3,670 images) |
| **Split** | 80% train (2,936) / 20% validation (734), `random_split` |
| **Environment** | Google Colab, GPU runtime |

**Classes:** `daisy` · `dandelion` · `roses` · `sunflowers` · `tulips`


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


##  Pipeline

```
Images → Preprocess → ResNet-50 (frozen) + new FC → Train → Evaluate
                                                          │
                                   Save .pth ← ───────────┘
                                       │
                                   Export .onnx → Validate with ONNX Runtime → Benchmark
```

1. **Preprocessing**: resize to 224×224, convert to a tensor, and normalize with the ImageNet mean and std.
2. **Model**: load the pretrained ResNet-50 and replace `model.fc` with `nn.Linear(2048, 5)`.
3. **Freezing**: `requires_grad = False` on every layer except `fc`.
4. **Training**: `CrossEntropyLoss` with the `Adam` optimizer, 10 epochs.
5. **PyTorch testing**: save the weights, reload them into a fresh model, and evaluate on the validation set.
6. **ONNX export**: `torch.onnx.export(..., dynamo=True)` with a dynamic batch dimension, then check the file with `onnx.checker`.
7. **ONNX testing**: run the validation set through ONNX Runtime, compare its outputs with PyTorch, and benchmark speed.


## Training Configuration

| Parameter | Value |
|---|---|
| Model | ResNet-50 |
| Pretrained weights | ImageNet (Torchvision default) |
| Epochs | 10 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Input size | 224 × 224 × 3 |
| Train / validation split | 80% / 20% |
| Device selection | CUDA when available; otherwise CPU |

## Training Results

| Epoch | Train Loss* | Train Acc | Val Acc |
|:---:|:---:|:---:|:---:|
| 1 | 138.01 | 77.96% | 91.14% |
| 2 | 72.59 | 88.59% | 91.69% |
| 3 | 57.06 | 91.25% | 92.64% |
| 4 | 47.56 | 92.37% | 92.92% |
| 5 | 41.43 | 94.07% | 92.78% |
| 6 | 34.78 | 94.96% | **93.19%** |
| 7 | 33.07 | 95.16% | **93.19%** |
| 8 | 30.45 | 95.64% | **93.19%** |
| 9 | 30.15 | 95.44% | 92.37% |
| 10 | 28.43 | 96.01% | 92.51% |

<sub>*Loss is summed over the 184 training batches in each epoch, not averaged.</sub>

Validation accuracy passes **91% after the first epoch**, because ImageNet features already transfer well to flowers. It peaks at 93.19% (epochs 6–8) and then levels off around 92.5%, while training accuracy keeps rising to 96%. That gap is a mild sign of overfitting, which data augmentation or early stopping would address.


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

##  ONNX Export & Validation

### Export details

| Property | Value |
|---|---|
| Exporter | `torch.onnx.export(..., dynamo=True)` (torch.export-based) |
| Opset | 20 |
| Input | `input`: `[batch, 3, 224, 224]` (dynamic batch) |
| Output | `logits`: `[batch, 5]` |
| File size | 94.2 MB (single file, no external data) |
| `onnx.checker` | ✅ Passed |
| Runtime | ONNX Runtime 1.32.0, `CUDAExecutionProvider` |

### Accuracy: PyTorch vs ONNX

| Check | Result |
|---|---|
| Max absolute logit difference (batch of 16) | **2.61 × 10⁻⁵** ✅ (`rtol=1e-3, atol=1e-4`) |
| PyTorch validation accuracy | 92.51% |
| ONNX validation accuracy | 92.51% |
| Prediction agreement | **100.00%** (0 of 734 differ) |

### Inference speed (Colab GPU)

| Batch size | PyTorch (ms) | ONNX Runtime (ms) | Speed-up | ONNX throughput |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 21.32 | **5.95** | **3.58×** | 168 img/s |
| 16 | 50.93 | 53.41 | 0.95× | 300 img/s |

<sub>Mean of 50 runs after 10 warm-up runs. Both timings include copying data from CPU to GPU and back.</sub>

- **For a single image, ONNX Runtime is 3.6× faster.** PyTorch's eager mode has a fixed cost on every call (Python dispatch and launching each layer separately). ONNX Runtime runs one pre-optimized graph with fused operators, so that cost mostly disappears. This is the case that matters for real-time inference on one frame at a time.
- **At batch size 16 both are about the same**, because the GPU's computation now dominates the time and both backends use cuDNN kernels for it.


## Repository Structure

A suggested repository layout:

``` text
resnet50-flower-classification/
├── resnet50-flower-classification-pytorch.ipynb
├── README.md
└── requirements.txt
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
```
torch
torchvision
torchsummary
numpy
matplotlib
seaborn
scikit-learn
pillow
onnx
onnxscript
onnxruntime-gpu   # or onnxruntime for CPU-only
```
5.  Run the notebook cells in order.

The notebook automatically selects CUDA when available and otherwise
uses the CPU.

## Current Scope and Future Improvements

The current notebook demonstrates transfer learning and five-class image classification. 

### Possible next steps:
-   **Data augmentation** for training only (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`) to reduce overfitting
-   Save the trained model weights and add a separate inference example for unseen images.
-   Compare frozen-feature extraction with selective or full fine-tuning.
-   **TensorRT**: build an FP16/INT8 engine from the ONNX model and compare its latency.
-   **Edge deployment** on NVIDIA Jetson

##  What I Learned

- How transfer learning reaches over 90% accuracy within an epoch on a small dataset
- Freezing a pretrained backbone and replacing the classification head
- Building a PyTorch training and validation loop with `DataLoader` and `random_split`
- Reading a confusion matrix to find which classes the model confuses
- Exporting a PyTorch model to **ONNX** with the dynamo exporter and a dynamic batch size
- Checking an exported model against the original: logit tolerance, prediction agreement, and accuracy on the whole validation set
- Where ONNX Runtime is faster (small-batch latency) and where it isn't (large batches limited by GPU computation)
