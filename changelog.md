# Changelog

## Week 1 – Exploratory Data Analysis
- In the first week we performed general EDA on the provided dataset of animal images categorized into 10 classes. [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- We inspected the folder structure, counted images per class, and visualized sample images to understand class distribution and image quality.
- We then counted number of different channel counts across images. We found that almost all images were RGB and only a few were of different channel counts (grayscale, RGBA). 
- We then analyzed image dimensions (width and height) and found a wide variety of sizes.
- We also tried to identify wheter images differed across RGB channels and inferred wheter they appear to be grayscale images stored as RGB. We found that some images did appear to be grayscale. But the number was insignificant. So we decided to keep all RGB images as is.

## Week 2 – Preprocessing and Data Splitting
- In the second week we focused on preparing the data pipeline: we created a stratified 70/15/15 train/validation/test split on the balanced file list to avoid leakage and preserve class proportions.
- We computed per-channel mean and standard deviation only on the training split and used those statistics for normalization across all splits.
- Our training transforms converted images to RGB, resized them to 224x224, applied random horizontal flips with p=0.5 and mild brightness/contrast jitter, then converted to tensors and normalized with the computed stats.
- Validation and test transforms mirrored training but without augmentation. We implemented the `AnimalDataset` and `DataLoader` with batch size 64 (workers set conservatively for stability) and visually checked denormalized samples to confirm correctness.

## Week 3 – Experiments and CNN Architectures
- In the third week we implemented a configurable `CustomCNN` that supports batch normalization, dropout, bottlenecks, skip connections, and adaptive pooling, trained with Adam, weight decay, ReduceLROnPlateau scheduling, early stopping, and macro precision/recall/F1 tracking.
- We ran three custom experiments: `exp1_baseline` (4 conv blocks up to 256 channels, BN on, no dropout, FC [256, 128]); `exp2_deeper` (deeper stack to 512 channels, BN with dropout 0.4, FC [512, 512, 256, 256]); and `exp3_skip_dropout` (adds skip connections and bottlenecks, BN with dropout 0.2, FC [512, 512, 256]).
- For transfer learning we fine-tuned frozen backbones with pretrained weight ResNet18, ResNet50, and MobileNetV3-Large by replacing the classifier head with a 10-class output and training for 10 epochs.
- We logged training and validation histories, produced confusion matrices, and saved best checkpoints for each run, then summarized results with metric tables and plots for both custom and pretrained models.
- Our CustomCNN experiments were logged into Weights & Biases. And Classification report is included in the project repository.

## Authors
- Matej Herzog, Nosenko Mykyta
