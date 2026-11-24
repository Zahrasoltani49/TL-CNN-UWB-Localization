# Transfer Learning-Based UWB Localisation

## 1. Overview

This repository contains the source code associated with the fine-tuning of a pre-trained Convolutional Neural Network (CNN) using transfer learning and UWB data collected at Tyndall National Institute, including the library and an adjacent meeting room.

The objective is to investigate the effectiveness of transfer learning in adapting a pre-trained CNN model to a new indoor environment with a limited amount of training data. The performance of the transfer learning approach is compared against a CNN trained from scratch using the same target-environment dataset.

## 2. Dataset

The dataset used for fine-tuning the pre-trained model was collected at Tyndall National Institute, in the library and an adjacent meeting room.

The training data are provided in:

`Dataset/training_dataset/`

The dataset is also publicly available on Zenodo:

https://doi.org/10.5281/zenodo.21218079

Please refer to the Zenodo record for further details about the dataset.

## 3. Model Training

The 'separableCNN' script is used to retrain the sequential baseline CNN model using depthwise separable convolutional layers to further reduce the model size.

This code is based on the implementation provided in the following work:

Klemen Bregar. (2023). UWB Positioning and Tracking Data Set (Version 2.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.8280736

Additionally, a Gradient Boosting Machine (GBM) model is implemented in 'gbm.py' to compare the performance of the proposed model against a traditional machine learning approach.

Two Python scripts are provided to train the model in 2 different approaches.

### 3.1. `train_tl`

This script fine-tunes the pre-trained CNN model using transfer learning and the training dataset collected in the target environment.

### 3.2. `train_from_scratch`

This script trains a CNN model from scratch using the limited training data collected in the target environment.

These two scripts allow the further analysis to compare and demonstrate the effectiveness of transfer learning in adapting a pre-trained model to a new indoor environment.

## 4. Model Inference and Localisation Evaluation

### `test_model_library`

This script performs inference using the trained models and evaluates their performance at three different discrete waypoints in the target environment.

The measurements collected at these waypoints are used to evaluate the model's performance during the localisation phase.

The script estimates the target position using:

- **Least Squares (LS):** Conventional position estimation using the measured UWB ranges.
- **Proposed Model:** Position estimation using the CNN-based ranging-error correction approach.

The results are saved in a `.csv` file for further analysis.

The output CSV file includes:

- Ground-truth X and Y coordinates.
- Estimated X and Y coordinates obtained using the conventional LS method.
- Estimated X and Y coordinates obtained using the proposed CNN-based approach.

These results can be used to compare the localisation performance of the conventional LS method and the proposed approach.


## 5. Repository Structure

```text
.
├── Dataset/
│   └── training_dataset/
├── train_tl
├── train_from_scratch
├── test_model_library
└── README.md
```

