<div align="center">

# 🐾 Animals-10 Image Classification

**Custom CNNs compared with transfer learning on 10 animal classes. Best result: 97.5% test accuracy.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/W%26B-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)

</div>

---

## 📌 Overview

A neural networks course project. We classify images from the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset (dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant). We built our own configurable CNN and compared it with fine-tuned pretrained backbones.

## 🏆 Results

| Experiment | Type | Test accuracy | Test F1 (macro) |
|---|---|:---:|:---:|
| `exp1_baseline`: 4 conv blocks, BatchNorm | Custom CNN | 82.33% | 82.35% |
| `exp2_deeper`: up to 512 channels, dropout 0.4 | Custom CNN | 64.14% | 63.97% |
| `exp3_skip_dropout`: skip connections + bottlenecks | Custom CNN | 76.41% | 76.37% |
| `exp4`: ResNet18 | Transfer learning | 95.42% | 95.46% |
| **`exp5`: ResNet50** | **Transfer learning** | **97.47%** | **97.52%** |
| `exp6`: MobileNetV3-Large | Transfer learning | 94.98% | 94.99% |

Full training curves and confusion matrices are in the notebook. The W&B report is in [`wandb_report.pdf`](wandb_report.pdf).

## 🔬 Pipeline

1. **EDA.** Class distribution, image sizes, color channels, and checks for grayscale images stored as RGB
2. **Balancing.** Downsampling of majority classes and removal of images smaller than 112×112
3. **Split.** Stratified 70 / 15 / 15 train / val / test
4. **Preprocessing.** Resize to 224×224, horizontal flip, brightness/contrast jitter, and normalization with train-only mean and std
5. **Training.** Adam, weight decay, ReduceLROnPlateau, early stopping, macro precision/recall/F1
6. **Transfer learning.** Frozen pretrained backbones with a new 10-class head

## 📁 Structure

```
zneus_animal.ipynb        # full pipeline: EDA → training → evaluation
best_model_exp*.pth       # best checkpoint of every experiment
changelog.md              # weekly progress log
wandb_report.pdf          # experiment tracking report
requirements.txt
```

## 🚀 Run it

```bash
pip install -r requirements.txt
jupyter notebook zneus_animal.ipynb
```

The dataset is downloaded automatically with `kagglehub`. To skip W&B logging, set `USE_WANDB = False` in the first cell.

## 👥 Authors

Matej Herzog · Mykyta Nosenko
