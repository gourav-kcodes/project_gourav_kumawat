# Skin Lesion Classification Using Convolutional Neural Networks

**Student:** Gourav Kumawat
**Roll Number:** 20231094
**Course:** Image and Video Processing with Deep Learning (DS3273)

---

## Kaggle Notebook

The full training notebook with all outputs, plots, and results is available here:

[View Kaggle Notebook](https://www.kaggle.com/code/gkkumar7383fuck/project)

---

## Project Overview

Skin cancer is one of the most common cancers worldwide, and early detection significantly improves survival rates. Expert dermatologists are scarce in many regions, and manual examination is slow and prone to variability between clinicians. This project builds an automated deep learning classifier that, given a dermoscopic photograph of a skin lesion, predicts which of seven diagnostic categories it belongs to — with confidence scores for each class.

The system is trained on the HAM10000 dataset using a pretrained ResNet-50 backbone via transfer learning, with weighted cross-entropy loss to handle severe class imbalance, and a two-phase training strategy that first warms up the classification head before fine-tuning the entire network.

---

## Problem Statement

### Task
Given a dermoscopic JPEG image of a skin lesion (resized to 224×224 for model input), predict one of seven diagnostic categories and output softmax confidence scores for all classes.

| Abbreviation | Full Name | Type |
|---|---|---|
| `akiec` | Actinic Keratosis / Intraepithelial Carcinoma | Pre-cancerous |
| `bcc` | Basal Cell Carcinoma | Malignant |
| `bkl` | Benign Keratosis | Benign |
| `df` | Dermatofibroma | Benign |
| `mel` | Melanoma | Most dangerous |
| `nv` | Melanocytic Nevi (Moles) | Benign, most common |
| `vasc` | Vascular Lesions | Benign |

### Why This Is Hard
The core challenge is distinguishing subtle visual differences in texture, colour distribution, border geometry, and structural asymmetry across seven visually similar lesion types. The dataset is severely imbalanced — melanocytic nevi (`nv`) account for 66.9% of all images, while rare classes like dermatofibroma (`df`, 1.1%) and vascular lesions (`vasc`, 1.4%) have very few training examples. Standard cross-entropy loss would cause the model to simply predict `nv` for most inputs. This is a classification problem that demands both hierarchical feature learning and deliberate handling of class imbalance.

---

## Dataset

**Name:** HAM10000 (Human Against Machine with 10,000 training images)
**Author:** Philipp Tschandl, Cliff Rosendahl, Harald Kittler
**Source:** [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and Harvard Dataverse

| Class | Full Name | Count | Percentage |
|-------|-----------|-------|-----------|
| akiec | Actinic Keratosis | 327 | 3.3% |
| bcc | Basal Cell Carcinoma | 514 | 5.1% |
| bkl | Benign Keratosis | 1,099 | 11.0% |
| df | Dermatofibroma | 115 | 1.1% |
| mel | Melanoma | 1,113 | 11.1% |
| nv | Melanocytic Nevi | 6,705 | 66.9% |
| vasc | Vascular Lesions | 142 | 1.4% |
| **Total** | | **10,015** | **100%** |

Images are fully annotated dermoscopic photographs, with ground truth confirmed by histopathology or expert consensus. The dataset is split across two image folders (`HAM10000_images_part_1` and `HAM10000_images_part_2`) with a companion CSV (`HAM10000_metadata.csv`) containing image IDs, class labels, patient age, sex, and lesion localisation.

---

## Data Pipeline

### Lesion-Level Deduplication
Some lesions in HAM10000 have multiple images taken from different angles or zoom levels. A naive image-level split would allow the same lesion to appear in both training and test sets, causing data leakage and inflated test accuracy. To prevent this, the split is performed at the **lesion level** — all images of a given lesion go to the same split.

```
Unique lesions: 7,470
Total images:   10,015
```

### Stratified 70 / 15 / 15 Split

| Split | Images |
|-------|--------|
| Train | 7,002 |
| Val | 1,532 |
| Test | 1,481 |

Stratification ensures class proportions are preserved in each split, which is critical given the severe imbalance.

### Data Augmentation

**Training transforms:**
```
Resize to 256×256
RandomCrop to 224×224
RandomHorizontalFlip
RandomVerticalFlip
RandomRotation(±30°)
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
ToTensor
Normalize(ImageNet mean/std)
```

Unlike brain MRI scans, dermoscopic images have no anatomical orientation — the camera can be placed at any angle. Both horizontal and vertical flips are therefore valid augmentations here. Colour jitter simulates variation in lighting conditions and dermoscope hardware across clinical settings.

**Evaluation transforms (no augmentation):**
```
Resize to 224×224
ToTensor
Normalize(ImageNet mean/std)
```

---

## Handling Class Imbalance: Weighted Cross-Entropy Loss

Standard cross-entropy loss treats all classes equally, which causes the model to overfit to `nv` (67% of data) and ignore rare classes. Inverse-frequency class weights are computed from the training set and passed to the loss function, so misclassifying a rare class is penalised much more heavily than misclassifying a common one.

| Class | Train Count | Weight |
|-------|------------|--------|
| akiec | 230 | 0.1312 |
| bcc | 366 | 0.0825 |
| bkl | 774 | 0.0390 |
| df | 76 | **0.3972** |
| mel | 778 | 0.0388 |
| nv | 4,679 | 0.0065 |
| vasc | 99 | **0.3049** |

`df` and `vasc` receive the highest weights because they are the rarest classes. `nv` receives the lowest weight because it is by far the most common.

---

## Model Architecture

### Base: ResNet-50

ResNet-50 is a 50-layer deep residual network pretrained on ImageNet (1.2 million images, 1000 classes). Its key innovation is skip connections (residual connections) that add the input of each block directly to its output, allowing gradients to flow unchanged through very deep networks and preventing the vanishing gradient problem.

The architecture consists of:
- Initial 7×7 convolution with 64 filters and stride 2
- Max pooling layer
- Four residual layer groups (`layer1` through `layer4`), each with multiple bottleneck blocks
- Each bottleneck block: 1×1 conv → 3×3 conv → 1×1 conv + skip connection

### Custom Classification Head

The original ResNet-50 `fc` layer (1000 ImageNet classes) is replaced with:

```
Linear(2048 → 512)
ReLU
Dropout(p=0.5)
Linear(512 → 7)
```

The 50% dropout rate is higher than typical because the dataset is small relative to the model capacity, and the class imbalance makes overfitting a significant risk.

### Full Architecture Summary

```
Input: (batch, 3, 224, 224)
    ↓
Conv1: 7×7, stride 2, 64 filters      [Phase 1: frozen | Phase 2: trained]
    ↓
MaxPool: 3×3, stride 2
    ↓
Layer1: 3× Bottleneck, 256ch           [Phase 1: frozen | Phase 2: trained]
    ↓
Layer2: 4× Bottleneck, 512ch           [Phase 1: frozen | Phase 2: trained]
    ↓
Layer3: 6× Bottleneck, 1024ch          [Phase 1: frozen | Phase 2: trained]
    ↓
Layer4: 3× Bottleneck, 2048ch          [Phase 1: frozen | Phase 2: trained]
    ↓
AdaptiveAvgPool → 2048-dim vector
    ↓
Linear(2048 → 512) + ReLU + Dropout(0.5)  [always trained]
    ↓
Linear(512 → 7)                             [always trained]
    ↓
Output: (batch, 7) logits → Softmax → class probabilities
```

**Phase 1 trainable parameters:** 1,052,679 (head only)
**Total parameters:** 24,560,711

---

## Two-Phase Training Strategy

### Why Two Phases?
When you unfreeze a pretrained backbone immediately, the random weights in the new classification head produce large, noisy gradients that can destroy the carefully learned ImageNet features in the backbone. The two-phase approach solves this by first stabilising the head before exposing the backbone to gradient updates.

### Phase 1 — Head Warmup (10 epochs)
- Backbone fully frozen (23.5M parameters fixed)
- Only the 1M-parameter classification head is trained
- Optimizer: Adam, lr = 1e-3, weight_decay = 1e-4
- Loss: Weighted CrossEntropyLoss
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)

The head learns to map ImageNet features to skin lesion categories. By epoch 10 the head is stable enough that unfreezing the backbone won't cause catastrophic interference.

### Phase 2 — Full Fine-Tuning (15 epochs)
- All layers unfrozen (24.5M parameters trainable)
- Lower learning rate: Adam, lr = 1e-4 to preserve pretrained features
- Same loss function and scheduler

The backbone can now adapt its higher-level feature detectors to the specific visual characteristics of dermoscopic images — texture patterns, colour gradients, and border irregularities that differ from the natural images it was trained on.

---

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Phase 1 Epochs | 10 |
| Phase 2 Epochs | 15 |
| Total Epochs | 25 |
| Batch Size | 32 |
| Phase 1 LR | 1e-3 |
| Phase 2 LR | 1e-4 |
| Weight Decay | 1e-4 |
| Dropout | 0.5 |
| Loss Function | Weighted CrossEntropyLoss |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| GPU | Tesla T4 (Kaggle) |

---

## Results

### Overall Metrics

| Metric | Score |
|--------|-------|
| Test Accuracy | **75.29%** |
| Macro F1-Score | **0.6201** |
| Weighted F1-Score | **0.7705** |
| Total Test Images | 1,481 |

### Training Progression

**Phase 1 (frozen backbone):**
- Started at val accuracy 0.428 (epoch 1)
- Best val accuracy by end of Phase 1: **0.547** (epoch 3)
- Val macro F1 peaked at 0.350 — the head learned basic discrimination but struggled with rare classes

**Phase 2 (full fine-tuning):**
- Immediately improved: val accuracy jumped to 0.590 by epoch 2
- Continued improving to **0.728** at epoch 11
- Train accuracy reached 0.715 by final epoch
- The gap between train (0.715) and best val (0.728) is healthy — no severe overfitting

### Per-Class Results

| Class | Precision | Recall | F1-Score | Support | Analysis |
|-------|-----------|--------|----------|---------|---------|
| akiec | 0.32 | 0.72 | **0.44** | 46 | Low precision — model over-predicts this class. Visually similar to bkl and mel |
| bcc | 0.57 | 0.72 | **0.63** | 71 | Reasonable performance. Distinct vascular patterns help discrimination |
| bkl | 0.66 | 0.52 | **0.58** | 168 | Good precision but misses ~half the true cases. Visually overlaps with nv and mel |
| df | 0.83 | 0.50 | **0.63** | 20 | High precision but low recall — model is conservative about predicting df |
| mel | 0.41 | 0.61 | **0.49** | 165 | Most clinically important class. Low precision means many false alarms |
| nv | 0.95 | 0.82 | **0.88** | 992 | Strongest class — dominant training signal despite weighting |
| vasc | 0.53 | 1.00 | **0.69** | 19 | Perfect recall — model catches every vasc case, but with false positives |

### Key Observations

**`nv` dominates performance.** Despite class weighting, `nv` is so overrepresented (66.9% of data, 992 of 1481 test images) that the model has learned its features extremely well — F1 of 0.88 vs 0.49 for `mel`.

**`vasc` achieves perfect recall (1.00)** — every vascular lesion in the test set was correctly identified. This is partly because vascular lesions have highly distinctive red/pink colour patterns that stand out from other lesion types even to an untrained eye. The weighted loss gives `vasc` a high penalty weight (0.3049) which pushes the model to be aggressive about predicting it.

**`mel` (melanoma) is the most clinically critical class and also one of the weakest** — F1 of 0.49, precision of 0.41. This means the model produces many false positives for melanoma (other lesions incorrectly flagged as melanoma) and misses ~39% of true melanoma cases. In a clinical context this would need to be addressed — high recall for melanoma is essential even at the cost of precision.

**`akiec` has the lowest F1 (0.44)** with precision of only 0.32. The model over-predicts akiec — many non-akiec lesions get classified as akiec. This is expected: actinic keratosis is a pre-cancerous condition that visually overlaps significantly with both benign keratosis (`bkl`) and early melanoma (`mel`).

**`df` has the highest precision (0.83)** but low recall (0.50) — the model is very confident when it predicts dermatofibroma but misses half the true cases, likely classifying them as `nv` or `bkl`. With only 76 training examples, `df` is data-limited.

### Why Macro F1 (0.62) Is More Meaningful Than Accuracy (75.29%)
A naive classifier that always predicts `nv` would achieve ~67% accuracy while being completely useless for clinical purposes. The macro F1 of 0.62, which averages F1 equally across all 7 classes regardless of frequency, is a more honest measure of the model's ability to handle the full diagnostic range.

---

## Repository Structure

```
project_gourav_kumawat/
│
├── checkpoints/
│   └── final_weights.pth        # Best model weights (val acc = 72.8%)
│
├── data/
│   ├── akiec_01.jpg             # 10 sample images per class
│   ├── akiec_02.jpg
│   ├── ...
│   ├── bcc_01.jpg ... bcc_10.jpg
│   ├── bkl_01.jpg ... bkl_10.jpg
│   ├── df_01.jpg  ... df_10.jpg
│   ├── mel_01.jpg ... mel_10.jpg
│   ├── nv_01.jpg  ... nv_10.jpg
│   └── vasc_01.jpg ... vasc_10.jpg  # 70 images total
│
├── config.py                    # All hyperparameters and paths
├── dataset.py                   # SkinLesionDataset class and dataloaders
├── model.py                     # SkinLesionClassifier (ResNet50 + custom head)
├── train.py                     # Training loop with two-phase support
├── predict.py                   # Inference function for image paths
└── interface.py                 # Standardised exports for grader
```

---

## File Descriptions

**`config.py`** — Single source of truth for all hyperparameters: dataset paths, class names, image dimensions (224×224×3), batch size (32), learning rates for both phases (1e-3 and 1e-4), dropout (0.5), hidden units (512), split ratios, random seed, and ImageNet normalisation statistics. All other files import from here.

**`dataset.py`** — Contains the `SkinLesionDataset` PyTorch Dataset class, the `get_image_path()` function that searches both image part folders, lesion-level deduplication logic, stratified 70/15/15 splitting, train and eval transform pipelines, and `build_dataloaders()` which constructs all three DataLoaders. `the_dataloader` is aliased at the bottom for the interface.

**`model.py`** — Defines `SkinLesionClassifier`, which loads the pretrained ResNet50 backbone, freezes all parameters, and replaces the final FC layer with a custom 2-layer head (Linear → ReLU → Dropout → Linear) outputting 7 class logits. `get_model()` instantiates and moves to device.

**`train.py`** — Contains `train_skin_lesion_model()` which runs a generic training loop for any number of epochs, saving a checkpoint every epoch and the best model by validation accuracy. Also contains `run_two_phase_training()` which orchestrates the full Phase 1 + Phase 2 sequence with correct optimizer setup for each phase, and `compute_class_weights()` for inverse-frequency weighting.

**`predict.py`** — Contains `classify_skin_lesions(list_of_img_paths)` which loads the saved weights and returns predicted class, confidence percentage, and full 7-class probability breakdown for each image path. Also contains `batch_inference_to_csv()` which saves predictions to a CSV file matching the batch inference output described in the proposal.

**`interface.py`** — Re-exports all key components under standardised names for programmatic evaluation by the grader: `TheModel`, `the_trainer`, `the_predictor`, `TheDataset`, `the_dataloader`, `the_batch_size`, `total_epochs`.

---

## How to Run Inference

```python
from predict import classify_skin_lesions

results = classify_skin_lesions([
    'data/mel_01.jpg',
    'data/nv_03.jpg',
    'data/vasc_07.jpg'
])

for r in results:
    print(r['predicted_class'], r['confidence'], r['probabilities'])
```

---

## Potential Improvements

1. **EfficientNet-B4 or Vision Transformer** — more powerful backbones designed for fine-grained visual classification
2. **Oversampling rare classes** — augment `df` and `vasc` with synthetic examples using SMOTE or GAN-based generation
3. **Test-Time Augmentation (TTA)** — average predictions across multiple augmented views of each test image, typically adding 1-3% accuracy
4. **Grad-CAM visualisations** — highlight which regions of the dermoscopic image drive the prediction, making the model interpretable for clinical use
5. **Ensemble of multiple models** — combining ResNet50, EfficientNet, and DenseNet predictions
6. **Higher recall for melanoma** — use a lower classification threshold for `mel` to prioritise catching all melanoma cases even at the cost of false positives
