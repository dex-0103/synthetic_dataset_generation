# Synthetic Data Generation for Healthcare using CTGAN

A end-to-end pipeline for generating realistic synthetic healthcare data using the **CTGAN** model from the [SDV (Synthetic Data Vault)](https://sdv.dev/) library. The project uses the **PIMA Indians Diabetes Dataset** as the source, generates privacy-safe synthetic patient records, and validates the quality of the generated data through statistical analysis and machine learning evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)

---

## Overview

Real-world healthcare data is sensitive, limited, and often restricted due to privacy regulations. Synthetic data generation offers a solution — artificial records that statistically mirror real patient data without exposing any personal information.

This project demonstrates a complete synthetic data generation workflow:

1. Explore and understand the raw dataset
2. Clean and preprocess the data
3. Train a CTGAN model to learn data patterns
4. Generate synthetic patient records
5. Validate the synthetic data through statistical comparison and ML evaluation

---

## Dataset

**PIMA Indians Diabetes Dataset**

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function score |
| `Age` | Age in years |
| `Outcome` | Target variable — 1 (Diabetic), 0 (Non-diabetic) |

- **Total records:** 768
- **Class distribution:** 500 Non-diabetic, 268 Diabetic

---

## Project Structure

```
synthetic-data-generation/
│
├── diabetes.csv                        # Raw dataset
├── cleaned_diabetes_v2.csv             # Cleaned and preprocessed dataset
├── synthetic_diabetes_v2.csv           # Generated synthetic dataset
├── metadata_v2.json                    # CTGAN metadata file
│
├── dataset_loading_and_preprocessing.ipynb   # EDA and preprocessing
├── model_training.ipynb                      # CTGAN training and generation
├── data_validation.ipynb                     # Statistical comparison
├── model_evaluation.ipynb                    # ML-based evaluation
│
└── README.md
```

---

## Pipeline

### Dataset Exploration

- Loaded the PIMA Indians Diabetes dataset using Pandas
- Reviewed dataset structure, shape, and data types
- Identified data quality issues: zero values in `Glucose`, `BloodPressure`, `BMI`, `SkinThickness`, and `Insulin` that are medically unrealistic
- Performed column-level analysis to understand feature distributions

### Data Cleaning and Preprocessing

- Replaced unrealistic zero values in `Glucose`, `BloodPressure`, `BMI`, `SkinThickness`, and `Insulin` with `NaN`
- Imputed missing values using column-wise **median** (robust to outliers in healthcare data)
- Verified no missing values remained after imputation
- Identified numerical vs. categorical columns
- Saved the cleaned dataset as `cleaned_diabetes_v2.csv`

> Note: Standard scaling was explored but intentionally omitted from the final pipeline to preserve original value ranges for CTGAN training.

### Synthetic Data Generation with CTGAN

- Used the **SDV** library's `CTGANSynthesizer` model
- Detected dataset metadata automatically using `SingleTableMetadata`
- Trained the CTGAN model with:
  - `epochs = 1000`
  - `batch_size = 100`
- Generated **768 synthetic patient records** (matching the original dataset size)
- Saved the synthetic dataset as `synthetic_diabetes_v2.csv`

```python
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

synthesizer = CTGANSynthesizer(metadata, epochs=1000, batch_size=100)
synthesizer.fit(df)

synthetic_data = synthesizer.sample(num_rows=len(df))
```

### Data Validation (Statistical Comparison)

- Compared **mean** and **standard deviation** values for key features (`Glucose`, `BMI`, `Age`) between real and synthetic datasets
- Plotted distribution histograms for `Glucose`, `BMI`, and `Age` to visually compare real vs. synthetic patterns
- Generated **correlation heatmaps** for both datasets to verify that CTGAN preserved inter-feature relationships
- Compared feature correlations with the `Outcome` variable to validate that predictive patterns were retained

### Model Evaluation

- Trained a **Random Forest Classifier** separately on real data and synthetic data (80/20 train-test split)
- Evaluated both models using Accuracy, Precision, Recall, and F1 Score
- Compared results to determine whether synthetic data preserves predictive utility

---

## Results

### Model Performance Comparison

| Metric | Real Data | Synthetic Data |
|---|---|---|
| **Accuracy** | 74.03% | **79.87%** |
| **Precision** | 63.64% | **80.88%** |
| **Recall** | 63.64% | **75.34%** |
| **F1 Score** | 63.64% | **78.01%** |

The model trained on synthetic data achieved **higher scores across all metrics**, demonstrating that CTGAN successfully captured and preserved the underlying statistical patterns from the original healthcare dataset. This is a strong indicator that the synthetic data is realistic and suitable for downstream machine learning tasks.

### Synthetic Data Classification Report

```
              precision    recall  f1-score   support

           0       0.79      0.84      0.81        81
           1       0.81      0.75      0.78        73

    accuracy                           0.80       154
   macro avg       0.80      0.80      0.80       154
weighted avg       0.80      0.80      0.80       154
```

---

## Requirements

```
python >= 3.12
pandas
numpy
scikit-learn
sdv
matplotlib
seaborn
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/synthetic-data-generation.git
cd synthetic-data-generation

# Create and activate a virtual environment
python -m venv env
source env/bin/activate        # On Windows: env\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn sdv matplotlib seaborn
```

---

## Usage

Run the notebooks in order:

```bash
# Load and preprocess the dataset
jupyter notebook dataset_loading_and_preprocessing.ipynb

# Train CTGAN and generate synthetic data
jupyter notebook model_training.ipynb

# Validate synthetic data statistically
jupyter notebook data_validation.ipynb

# Evaluate using machine learning
jupyter notebook model_evaluation.ipynb
```

---

## Technologies Used

| Tool / Library | Purpose |
|---|---|
| **Python 3.12** | Core programming language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations |
| **SDV (CTGAN)** | Synthetic data generation |
| **Scikit-learn** | Preprocessing and ML evaluation |
| **Matplotlib / Seaborn** | Data visualization |
| **Jupyter Notebook** | Interactive development environment |

---

## Why Synthetic Healthcare Data?

- **Privacy protection** — No real patient information is exposed
- **Data augmentation** — Increases dataset size for better ML training
- **Research accessibility** — Synthetic datasets can be shared freely
- **Bias mitigation** — Can help balance class distributions
- **Compliance** — Reduces regulatory risk when sharing medical data
