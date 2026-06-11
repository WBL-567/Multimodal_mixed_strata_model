# ata_model

# A Deep Multimodal Fusion Model for Real-Time Identification of Mixed Strata in Shield Tunnelling

This repository provides the implementation code, processed numerical input data, LOF-cleaned tunnelling-parameter data, representative sample raw vibration data, and scripts used for the manuscript:

**A deep multimodal fusion model for real-time identification of mixed strata in shield tunnelling**

The complete processed spectrogram image dataset used as the image-modality input of the multimodal model is available on Zenodo:

https://doi.org/10.5281/zenodo.20645879

## 1. Overview

This study proposes a deep multimodal fusion model for real-time identification of mixed strata in shield tunnelling. The model uses two types of inputs:

* **Numerical input**: LOF-cleaned tunnelling parameters and vibration RMS features.
* **Image input**: processed frequency-domain spectrogram images generated from three-dimensional vibration signals.

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

The files below provide tunnelling parameters before and after LOF-based cleaning for the three stratum types:

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

LOF-detected outliers were replaced by linear interpolation.

### 3.2 Numerical Input for the Multimodal Model

The file below is the numerical input used by the multimodal model:

```text
tunnelling_parameter_1.xlsx
```

This file was constructed by combining:

* LOF-cleaned tunnelling parameters;
* corresponding vibration RMS values in three directions;
* stratum labels.

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

These spectrogram images are the processed image input data used by the multimodal fusion model.

## 4. Raw Vibration Data and Spectrogram Generation

The complete raw three-dimensional vibration dataset is not fully included in this GitHub repository because of its large file size and engineering data management requirements. Instead, this repository provides representative sample raw vibration files to demonstrate the preprocessing workflow from raw vibration signals to frequency-domain spectrogram images.

Each sample `.mat` file contains several variables. In the spectrogram generation script:

```text
a2 = time sequence
a3 = vibration acceleration sequence
```

The time interval of `a2` is approximately 0.000390625 s, corresponding to a sampling frequency of 2560 Hz.

Some representative raw vibration files are provided directly in `.mat` format, while others are compressed as `.zip` files due to file-size limitations. Please unzip the `.zip` files before running `Spectrogram_generation.py`. After decompression, the `.mat` files should be placed in the same working directory as the spectrogram generation script or in the directory specified by the user.

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

The script automatically reads `.mat` files in the current folder and generates FFT-based spectrogram images.

Default settings:

```text
Sampling frequency = 2560 Hz
Segment duration = 5 s
Image size = 224 × 224 pixels
```

### 5.4 Train and Evaluate the Multimodal Model

```bash
python Multimodal_model_main_program.py
```

This script includes model training, validation, testing, and ablation analysis.

Please note that full model training requires the processed spectrogram image dataset aligned with `tunnelling_parameter_1.xlsx`. The complete processed spectrogram image dataset is available on Zenodo:

```text
https://doi.org/10.5281/zenodo.20645879
```

The representative `.mat` and `.zip` files in this GitHub repository are mainly used to demonstrate the raw vibration data structure and the spectrogram generation procedure.

## 6. Reproducibility Workflow

The main data-processing workflow is:

```text
1. Remove tunnelling-parameter samples containing zero values.
2. Apply LOF-based outlier detection to the remaining tunnelling parameters.
3. Replace LOF-detected outliers by interpolation.
4. Extract vibration RMS features corresponding to the retained samples.
5. Delete the vibration samples corresponding to removed zero-value tunnelling-parameter records.
6. Combine the LOF-cleaned tunnelling parameters and vibration RMS features to form the numerical input.
7. Generate vertical, horizontal, and axial spectrograms from raw vibration data.
8. Use the processed spectrogram image dataset deposited on Zenodo as the image-modality input.
9. Synchronize the numerical and image inputs.
10. Train and evaluate the multimodal fusion model.
```

## 7. Data Availability

The code, processed numerical input data, LOF-cleaned tunnelling-parameter data, and representative sample raw vibration data are available in this GitHub repository.

The complete processed spectrogram image dataset used as the image-modality input of the multimodal model is available on Zenodo:

```text
https://doi.org/10.5281/zenodo.20645879
```

The complete processed spectrogram image dataset is deposited on Zenodo rather than directly included in this GitHub repository because of its large file size and the file-size limitations of GitHub. Zenodo is used as the long-term public data repository for the large-size processed image dataset.

The complete raw three-dimensional vibration dataset is not fully uploaded because of its large file size and engineering data management requirements. Representative raw vibration files are provided in this GitHub repository to demonstrate the data structure and the spectrogram generation workflow.

To remove project-identifiable information, the provided raw vibration samples were anonymized and self-numbered. The numbering rule preserves the correspondence among the randomly selected sample group, stratum type, and vibration direction, while excluding sensitive engineering information such as exact project location, exact chainage, original ring numbers, detailed construction logs, and project-identifiable information.

The released files are provided for academic research and reproducibility purposes.


## 8. Citation

If you use this repository, the provided code, or the dataset, please cite the corresponding manuscript and the Zenodo dataset:

**A deep multimodal fusion model for real-time identification of mixed strata in shield tunnelling**

Processed spectrogram image dataset:

```text
https://doi.org/10.5281/zenodo.20645879
```
