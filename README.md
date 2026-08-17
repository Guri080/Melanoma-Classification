# Melanoma Classification

Deep learning framework for skin lesion classification using the **ISIC 2018** and **ISIC 2020** datasets. The project explores convolutional neural networks and vision transformers for melanoma detection, with particular attention to the severe class imbalance present in medical imaging datasets.

The repository supports training and evaluation of several pretrained architectures, including **ResNet, EfficientNet, and Swin Transformer**, as well as custom CNN baselines.

## Overview

Automated melanoma detection is challenging because skin lesion datasets are often highly imbalanced, with malignant samples representing only a small fraction of the available images.

This project investigates different model architectures and training strategies for improving classification performance under this imbalance.

The pipeline supports:

* **Binary melanoma classification** using ISIC 2020
* **7-class skin lesion classification** using ISIC 2018
* ImageNet pretrained models and transfer learning
* Class-weighted loss
* Weighted random sampling
* Focal loss
* Data augmentation
* Layer freezing and progressive unfreezing
* Multi-GPU training
* Checkpointing and experiment logging

## Datasets

### ISIC 2020

The ISIC 2020 dataset is used for binary classification:

| Label | Class    |
| ----- | -------- |
| `0`   | Benign   |
| `1`   | Melanoma |

The dataset is highly imbalanced, making it useful for evaluating techniques designed to improve minority-class detection.

### ISIC 2018

ISIC 2018 Task 3 is used for multi-class skin lesion classification across seven diagnostic categories:

| Label | Diagnosis                                     |
| ----- | --------------------------------------------- |
| MEL   | Melanoma                                      |
| NV    | Melanocytic Nevus                             |
| BCC   | Basal Cell Carcinoma                          |
| AKIEC | Actinic Keratosis / Intraepithelial Carcinoma |
| BKL   | Benign Keratosis                              |
| DF    | Dermatofibroma                                |
| VASC  | Vascular Lesion                               |

## Models

The training pipeline provides a model factory for experimenting with several architectures.

### Pretrained Models

* **ResNet-18**
* **ResNet-50**
* **EfficientNet-B4**
* **Swin Transformer Base**

The classification heads of the pretrained ImageNet models are replaced to match the number of target classes.

### Custom Models

The repository also includes custom CNN architectures:

* `conv_B`
* `conv_T`

These provide smaller baselines for comparison with larger pretrained architectures.

## Data Preprocessing

Images are converted to RGB and transformed before being passed to the model.

Training augmentation includes:

* Square padding
* Resize to `224 × 224`
* Random horizontal flipping
* Random vertical flipping
* Random rotation
* ImageNet normalization

Validation images use the same resizing and normalization pipeline without random augmentation.

## Handling Class Imbalance

A major focus of this project is mitigating class imbalance.

### Weighted Random Sampling

Samples can be drawn according to inverse class frequency using PyTorch's `WeightedRandomSampler`.

This increases the probability that underrepresented classes appear during training.

```bash
--strategy sampler
```

### Weighted Loss

Class weights can instead be incorporated directly into the loss function.

```bash
--strategy weighted_loss
```

### Focal Loss

The repository implements focal loss to reduce the contribution of easily classified samples and place greater emphasis on difficult examples.

```bash
--loss focal
```

Cross-entropy loss is also supported:

```bash
--loss CE
```

## Transfer Learning

Models can be initialized using ImageNet pretrained weights:

```bash
--PT
```

The framework also supports freezing the pretrained backbone and initially training only the classification head:

```bash
--freeze
```

Progressive layer unfreezing is implemented for supported architectures, allowing deeper sections of the network to become trainable during later stages of training.

## Evaluation Metrics

Because standard accuracy can be misleading for highly imbalanced medical datasets, the training pipeline tracks several metrics:

* **Balanced Accuracy**
* **F1 Score**
* **ROC-AUC**
* Training Loss
* Validation Loss
* Per-class performance

For binary classification, binary F1 and ROC-AUC are computed.

For multi-class classification, macro F1 and one-vs-rest ROC-AUC are used.

Experiment results are written to CSV files for later analysis.

## Results

Models were evaluated on **ISIC 2020** for binary melanoma classification and **ISIC 2018** for seven-class skin lesion classification using ROC-AUC and F1 score.

| Model                | ISIC 2020 AUC |         F1 | ISIC 2018 AUC |         F1 |
| -------------------- | ------------: | ---------: | ------------: | ---------: |
| **Swin Transformer** |    **0.8498** |     0.5759 |    **0.9829** |     0.7315 |
| **ResNet-50**        |        0.8488 | **0.5957** |        0.9769 |     0.8226 |
| **EfficientNet**     |        0.6897 |     0.5529 |        0.9823 | **0.8324** |

### Class Imbalance Experiments

ResNet-50 was also used to compare loss functions and class-balancing strategies on the imbalanced dataset.

| Loss              | Strategy      | Val. Accuracy |         F1 |        AUC |
| ----------------- | ------------- | ------------: | ---------: | ---------: |
| **Cross-Entropy** | Weighted Loss |        61.06% | **0.2621** | **0.8498** |
| Cross-Entropy     | Sampler       |    **62.53%** |     0.2573 |     0.7679 |
| Focal Loss        | Weighted Loss |        53.98% |     0.1212 |     0.7461 |
| Focal Loss        | Sampler       |        60.12% |     0.2326 |     0.8214 |

Overall, **Swin Transformer achieved the highest AUC on both datasets**, while ResNet-50 and EfficientNet achieved the highest F1 scores on ISIC 2020 and ISIC 2018, respectively.


## Project Structure

```text
Melanoma-Classification/
│
├── dataloader.py       # ISIC 2018 and ISIC 2020 dataset loaders
├── loss.py             # Custom focal loss implementation
├── main_script.py      # Training, validation, and experiment pipeline
├── models.py           # CNN and transformer architectures
├── script.sh           # Example training command
└── test.py             # Model parameter inspection
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Guri080/Melanoma-Classification.git
cd Melanoma-Classification
```

Create a Python environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the primary dependencies:

```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install pillow opencv-python
pip install tqdm timm transformers
```

## Dataset Setup

Download the desired dataset from the ISIC Archive:

* ISIC 2018 Task 3
* ISIC 2020 Challenge Dataset

The current implementation expects dataset locations to be specified inside `main_script.py`.

Update the dataset paths to point to your local dataset directories before training.

For example:

```python
root_2020 = "/path/to/isic2020/images"
```

> **Note:** The repository currently imports a custom `PadSquare` transformation from `custom_transformations.py`. Ensure this transformation is available before running the training pipeline.

## Training

The main training interface is `main_script.py`.

A basic experiment can be launched with:

```bash
python main_script.py \
    --data_flag isic2020 \
    --model_flag resnet50_224 \
    --batch_size 128 \
    --epochs 100 \
    --run resnet50_baseline
```

### Training With a Pretrained Model

```bash
python main_script.py \
    --data_flag isic2020 \
    --model_flag resnet50_224 \
    --PT \
    --batch_size 128 \
    --epochs 100 \
    --run pretrained_resnet50
```

### Training With Weighted Sampling

```bash
python main_script.py \
    --data_flag isic2020 \
    --model_flag resnet50_224 \
    --PT \
    --strategy sampler \
    --run resnet50_sampler
```

### Training With Focal Loss

```bash
python main_script.py \
    --data_flag isic2020 \
    --model_flag efficientnet \
    --PT \
    --loss focal \
    --run efficientnet_focal
```

### Training a Swin Transformer

```bash
python main_script.py \
    --data_flag isic2018 \
    --model_flag swin \
    --PT \
    --loss CE \
    --run swin_isic2018
```

## Command-Line Arguments

| Argument        | Description                       |
| --------------- | --------------------------------- |
| `--data_flag`   | Dataset: `isic2018` or `isic2020` |
| `--model_flag`  | Model architecture                |
| `--PT`          | Use pretrained ImageNet weights   |
| `--resume`      | Resume training from a checkpoint |
| `--freeze`      | Freeze backbone layers            |
| `--batch_size`  | Training batch size               |
| `--epochs`      | Number of training epochs         |
| `--loss`        | `CE` or `focal`                   |
| `--strategy`    | `sampler` or `weighted_loss`      |
| `--run`         | Experiment name                   |
| `--accelerator` | Accelerator configuration         |

Supported model identifiers include:

```text
resnet18_224
resnet50_224
efficientnet
swin
conv_B
conv_T
```

## Multi-GPU Training

When multiple CUDA GPUs are available, the training pipeline automatically wraps the model using PyTorch `DataParallel`.

This allows experiments to scale across multiple GPUs without changing the training command.

## Experiment Logging

Training statistics are stored in CSV files for each experiment.

Logged values include:

```text
epoch
train_loss
train_acc
val_loss
val_acc
f1_score
auc
```

Additional class-specific statistics are also recorded to better understand performance on melanoma and benign samples independently.

## Technologies

* Python
* PyTorch
* Torchvision
* Scikit-learn
* NumPy
* Pandas
* OpenCV
* PIL
* timm
* Hugging Face Transformers
* CUDA

## Motivation

In medical image classification, high overall accuracy does not necessarily imply strong clinical performance. A model that predicts the majority class for nearly every image can achieve high accuracy while failing to identify malignant lesions.

This project therefore focuses not only on model architecture, but also on **class imbalance, sampling strategies, loss functions, transfer learning, and clinically more informative evaluation metrics**.

The goal is to better understand how these choices affect melanoma detection and generalization across skin lesion classification datasets.

## Disclaimer

This repository is intended for **research and educational purposes only**. The models produced by this project are not validated medical devices and should not be used for clinical diagnosis or medical decision-making.
