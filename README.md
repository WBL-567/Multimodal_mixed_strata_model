# Multimodal_mixed_strata_model

# A Deep Multimodal Fusion Model for Real-Time Identification of Mixed Strata in Shield Tunnelling

This repository provides the code, processed numerical data, LOF-cleaned tunnelling-parameter data, and representative sample raw vibration data for demonstrating the preprocessing workflow of the manuscript:

**A deep multimodal fusion model for real-time identification of mixed strata in shield tunnelling**

## 1. Overview

This study proposes a deep multimodal fusion model for real-time identification of mixed strata in shield tunnelling. The model uses two types of inputs:

* **Numerical input**: LOF-cleaned tunnelling parameters and vibration RMS features.
* **Image input**: frequency-domain spectrograms generated from three-dimensional vibration signals.

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

## 4. Raw Vibration Data and Spectrogram Generation

Due to the large file size of the complete raw three-dimensional vibration dataset and the generated spectrogram image dataset, they are not fully included in this GitHub repository. In addition, the original monitoring data are associated with an engineering project and therefore require data management and anonymization.

To support reproducibility, this repository provides processed numerical input data, LOF-cleaned tunnelling-parameter data, representative sample raw vibration files, and the spectrogram generation script. The sample raw vibration files can be used to demonstrate the conversion process from raw vibration signals to frequency-domain spectrogram images.

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

### 5.3 Generate Spectrogram Images

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

Please note that full model training requires the complete spectrogram image dataset aligned with `tunnelling_parameter_1.xlsx`. Since the complete image dataset is too large to be uploaded to this repository, the provided sample `.mat` and `.zip` files are mainly used to demonstrate the spectrogram generation procedure.

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
8. Synchronize the numerical and image inputs.
9. Train and evaluate the multimodal fusion model.
```

## 7. Data Availability

The complete raw vibration dataset and the complete generated spectrogram image dataset are not fully uploaded because of their large file size and engineering data management requirements.

Sensitive engineering information has been removed or anonymized, including exact project location, exact chainage, original ring numbers, detailed construction logs, and project-identifiable information.

The released files are provided for academic research and reproducibility purposes.

## 8. Citation

If you use this repository or the provided code, please cite the corresponding manuscript:

**A deep multimodal fusion model for real-time identification of mixed strata in shield tunnelling**
