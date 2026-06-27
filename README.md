# Multimodal_mixed_strata_model

This repository provides the implementation code, processed numerical input data, LOF-cleaned tunnelling-parameter data, representative raw vibration samples, and result-analysis scripts for multimodal mixed strata identification in shield tunnelling.

The complete processed spectrogram image dataset used as the image-modality input is available on Zenodo:

```text
https://doi.org/10.5281/zenodo.20645879
```

## 1. Overview

This repository contains code and data for a deep multimodal fusion model for real-time identification of mixed strata in shield tunnelling. The model uses two types of inputs:

Numerical input: LOF-cleaned tunnelling parameters and vibration RMS features.

Image input: processed frequency-domain spectrogram images generated from three-dimensional vibration signals.

The numerical branch is based on an MLP, and the image branch is based on CNNs. The extracted features are concatenated for final stratum classification.

The stratum labels are:

```text
0 = HSS: Homogeneous Stratum
1 = DMS: Dual-layer Mixed Stratum
2 = TMS: Triple-layer Mixed Stratum
```

## 2. Repository Contents

```text
.
├── Data_cleaning_LOF.py
├── Spectrogram_generation.py
├── Multimodal_model_main_program.py
├── Result_Analysis.py
├── requirements.txt
├── README.md
│
├── 1HSS_original.xlsx
├── 1HSS_clean.xlsx
├── 2DMS_original.xlsx
├── 2DMS_clean.xlsx
├── 3TMS_original.xlsx
├── 3TMS_clean.xlsx
│
├── tunnelling_parameter_1.xlsx
│
├── 6_1_1.mat
├── 6_1_2.mat
├── 6_1_3.mat
│
├── 8_2_1.zip
├── 8_2_2.zip
├── 8_2_3.zip
│
├── 15_3_1.zip
├── 15_3_2.zip
└── 15_3_3.zip
```

## 3. Data Description

### 3.1 LOF-Cleaned Tunnelling Parameters

The following files provide tunnelling parameters before and after LOF-based cleaning for the three stratum types:

```text
1HSS_original.xlsx
1HSS_clean.xlsx
2DMS_original.xlsx
2DMS_clean.xlsx
3TMS_original.xlsx
3TMS_clean.xlsx
```

Before LOF processing, tunnelling-parameter samples containing zero values were removed. The LOF-cleaned data were then obtained using `Data_cleaning_LOF.py`.

The LOF settings are:

```text
n_neighbors = 5
contamination = 0.02
```

LOF-identified samples were processed by linear interpolation.

### 3.2 Numerical Input for the Multimodal Model

The file below is the numerical input used by the multimodal model:

```text
tunnelling_parameter_1.xlsx
```

This file was constructed by combining:

- LOF-cleaned tunnelling parameters;
- corresponding vibration RMS values in three directions;
- stratum labels.

The vibration RMS samples corresponding to removed zero-value tunnelling-parameter records were also deleted, so that the tunnelling parameters and vibration features remained aligned.

The numerical input columns include:

```text
推进速度
扭矩
推力
竖向振动加速度RMS
横向振动加速度RMS
轴向振动加速度RMS
地层分类
```

### 3.3 Processed Spectrogram Image Dataset

The complete processed spectrogram image dataset used as the image-modality input of the multimodal model is deposited on Zenodo:

```text
https://doi.org/10.5281/zenodo.20645879
```

The Zenodo dataset contains three compressed files corresponding to the three stratum types:

```text
11.zip = HSS: Homogeneous Stratum
22.zip = DMS: Dual-layer Mixed Stratum
33.zip = TMS: Triple-layer Mixed Stratum
```

After downloading and decompressing the Zenodo files, the folders `11`, `22`, and `33` should be placed in the same working directory level as `tunnelling_parameter_1.xlsx`, or in the directory expected by the scripts.

## 4. Raw Vibration Data and Spectrogram Generation

Representative raw vibration files are provided to demonstrate the preprocessing workflow from raw vibration signals to frequency-domain spectrogram images.

Each sample `.mat` file contains several variables. In the spectrogram generation script:

```text
a2 = time sequence
a3 = vibration acceleration sequence
```

The time interval of `a2` is approximately 0.000390625 s, corresponding to a sampling frequency of 2560 Hz.

Some representative raw vibration files are provided directly in `.mat` format, while others are compressed as `.zip` files due to file-size limitations. Please unzip the `.zip` files before running `Spectrogram_generation.py`.

After decompression, the `.mat` files should be placed in the same working directory as the spectrogram generation script or in the directory specified by the user.

The sample raw vibration files are named as:

```text
a_b_c
```

where:

```text
a = randomly selected sample group index
b = stratum type
c = vibration direction
```

The vibration direction is defined as:

```text
1 = vertical direction
2 = horizontal direction
3 = axial direction
```

For example:

```text
6_1_1.mat
6_1_2.mat
6_1_3.mat
```

represent one randomly selected group of raw vibration signals from stratum type 1, including vertical, horizontal, and axial directions.

The provided raw vibration samples were anonymized and self-numbered to remove project-identifiable information while preserving the correspondence among sample group, stratum type, and vibration direction.

## 5. How to Run

### 5.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 5.2 Run LOF-Based Data Cleaning

```bash
python Data_cleaning_LOF.py
```

This script performs LOF-based cleaning for tunnelling parameters and outputs cleaned data and comparison figures.

### 5.3 Generate Spectrogram Images from Sample Raw Vibration Data

```bash
python Spectrogram_generation.py
```

The script reads `.mat` files and generates FFT-based spectrogram images.

Default settings:

```text
Sampling frequency = 2560 Hz
Segment duration = 5 s
Image size = 224 × 224 pixels
```

### 5.4 Train the Multimodal Model

```bash
python Multimodal_model_main_program.py
```

This script performs chronological model training and saves the trained models and scaler, including:

```text
final_model.pth
final_scaler.pkl
logs/best_model_fold_1.pth
...
logs/best_model_fold_8.pth
```

Full model training requires the processed spectrogram image dataset aligned with `tunnelling_parameter_1.xlsx`.

The complete processed spectrogram image dataset is available on Zenodo:

```text
https://doi.org/10.5281/zenodo.20645879
```

The representative `.mat` and `.zip` files in this GitHub repository are mainly used to demonstrate the raw vibration data structure and the spectrogram generation procedure.

### 5.5 Hyperparameter Settings

To improve reproducibility, the main implementation, training, preprocessing, and evaluation hyperparameters are summarized below.

#### Model Configuration

The multimodal model consists of one numerical branch and three image branches. The numerical branch uses a single-hidden-layer MLP to encode the numerical features. The image branch uses CNNs to extract spatial features from vertical, horizontal, and axial vibration spectrograms. The extracted numerical and image features are concatenated for final stratum classification.

```text
Numerical input features = 6
Numerical feature dimension = 128
Image input size = 1 × 224 × 224
Image feature dimension per spectrogram branch = 128
Fusion feature dimension = 512
Number of classes = 3
Dropout rate = 0.5
Activation function = ReLU
```

The class labels are:

```text
0 = HSS: Homogeneous Stratum
1 = DMS: Dual-layer Mixed Stratum
2 = TMS: Triple-layer Mixed Stratum
```

#### Training Hyperparameters

```text
Random seed = 42
Batch size = 32
Maximum epochs = 100
Optimizer = AdamW
Learning rate = 1e-4
Weight decay = 1e-4
Loss function = class-weighted CrossEntropyLoss
Gradient clipping norm = 3.0
Early stopping patience = 12
Minimum validation-loss improvement = 0.001
Learning-rate scheduler = ReduceLROnPlateau
Scheduler patience = 4
Scheduler factor = 0.5
```

Class weights in the cross-entropy loss are calculated from the class distribution of the training subset:

```text
class_weight = 1 / sqrt(class_count)
```

The class weights are then normalized so that their sum equals the number of classes.

#### Data Loading Settings

```text
Training batch size = 32
Validation batch size = 32
Test batch size = 32
Training shuffle = True
Validation shuffle = False
Test shuffle = False
num_workers = 0
```

#### Image Preprocessing and Augmentation

For training images, light data augmentation is applied:

```text
Resize = 224 × 224
RandomAffine degrees = 3
RandomAffine translate = (0.02, 0.02)
RandomAffine scale = (0.98, 1.02)
ColorJitter brightness = 0.08
ColorJitter contrast = 0.08
Normalize mean = [0.5]
Normalize std = [0.5]
```

For validation, testing, and importance analysis, deterministic preprocessing is used:

```text
Resize = 224 × 224
Normalize mean = [0.5]
Normalize std = [0.5]
```

#### Numerical-Feature Preprocessing

The numerical features are standardized using `StandardScaler`. To avoid information leakage, the scaler is fitted only on the training subset and then applied to the validation and test subsets.

The numerical input features are:

```text
推进速度
扭矩
推力
竖向振动加速度RMS
横向振动加速度RMS
轴向振动加速度RMS
```

#### LOF-Based Data-Cleaning Parameters

```text
n_neighbors = 5
contamination = 0.02
```

Before LOF processing, tunnelling-parameter samples containing zero values are removed. LOF-identified abnormal samples are processed by linear interpolation.

#### Evaluation Settings

For final model evaluation, the latest 20% of samples in each class are retained as the independent later-stage test set. The earlier 80% of samples are further divided chronologically into training and validation subsets at a ratio of 9:1.

```text
Training set = 72%
Validation set = 8%
Independent test set = 20%
```

The final trained model and scaler are saved as:

```text
final_model.pth
final_scaler.pkl
```

The fold models are saved in:

```text
logs/best_model_fold_1.pth
...
logs/best_model_fold_8.pth
```

### 5.6 Run Result Analysis

After model training is completed, run:

```bash
python Result_Analysis.py
```

This script loads the saved models and reproduces the main evaluation and analysis results in the manuscript order, including:

```text
Blocked temporal forward-chaining evaluation
Final validation and independent later-stage test evaluation
Cross-modality ablation analysis
Directional vibration spectrogram contribution analysis
Numerical feature sensitivity analysis
```

The output files are saved to:

```text
results/chapter5_results/
```

If the saved fold models are not needed, the fold-evaluation part can be skipped by running:

```bash
python Result_Analysis.py --skip-kfold
```

## 6. Reproducibility Workflow

The main data-processing and model-evaluation workflow is:

```text
1. Remove tunnelling-parameter samples containing zero values.
2. Apply LOF-based outlier detection to the remaining tunnelling parameters.
3. Replace LOF-identified samples by interpolation.
4. Extract vibration RMS features corresponding to the retained samples.
5. Delete the vibration samples corresponding to removed zero-value tunnelling-parameter records.
6. Combine the LOF-cleaned tunnelling parameters and vibration RMS features to form the numerical input.
7. Generate vertical, horizontal, and axial spectrograms from raw vibration data.
8. Use the processed spectrogram image dataset deposited on Zenodo as the image-modality input.
9. Synchronize the numerical and image inputs.
10. Train the multimodal fusion model.
11. Run the result-analysis script to reproduce the reported evaluation and analysis outputs.
```

## 7. Data Availability

The implementation code, processed numerical input data, LOF-cleaned tunnelling-parameter data, representative raw vibration files, and result-analysis scripts are available in this GitHub repository.

The complete processed spectrogram image dataset used as the image-modality input of the multimodal model is available on Zenodo:

```text
https://doi.org/10.5281/zenodo.20645879
```

The processed spectrogram image dataset is deposited on Zenodo because of its large file size. Zenodo is used as the long-term public data repository for the large-size processed image dataset.

Representative anonymized raw vibration files are provided in this GitHub repository to demonstrate the data structure and the spectrogram generation workflow.

The anonymized numbering rule preserves the correspondence among the randomly selected sample group, stratum type, and vibration direction, while excluding project-identifiable information such as exact project location, exact chainage, original ring numbers, and detailed construction logs.

The released files are provided for academic research and reproducibility purposes.

## 8. Citation

If you use this repository, the provided code, or the dataset, please cite the corresponding manuscript and the Zenodo dataset.

Processed spectrogram image dataset:

```text
https://doi.org/10.5281/zenodo.20645879
```
