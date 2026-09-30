# 🚢 Titanic Survival Prediction

[![Kaggle Public Score](https://img.shields.io/badge/Kaggle%20Public%20Score-0.74641-blue.svg)](https://www.kaggle.com/competitions/titanic)
[![Python](https://img.shields.io/badge/Python-3.x-brightgreen.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning project based on the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic).

The goal of this project is to predict whether a passenger survived the Titanic disaster using demographic, passenger, and travel information.

---

## 📌 Executive Summary

This project follows an end-to-end machine learning workflow, starting from raw data inspection and exploratory data analysis to feature engineering, hyperparameter experimentation, cross-validation, and submission to Kaggle.

* **Best Local Model**: Random Forest + `Title` Feature
* **Validation Accuracy**: **82.12%**
* **5-Fold Cross-Validation Accuracy**: **80.13% (±2.55%)**
* **Final Kaggle Public Score**: **0.74641** 🏆

---

## 📌 Project Overview & Workflow

1. **🔍 Data Inspection**: Audit the raw training (891 rows) and test sets (418 rows).
2. **📊 Exploratory Data Analysis (EDA)**: Uncover demographic and socio-economic survival drivers.
3. **🧹 Missing Value Treatment**: Apply median imputation for `Age` and mode imputation for `Embarked`.
4. **🛠️ Feature Engineering**: Construct high-signal features (`FamilySize`, `Title`, `CabinDeck`).
5. **⚙️ Preprocessing Pipeline**: Scale numerical attributes and one-hot encode categorical factors using Scikit-Learn pipelines.
6. **🤖 Benchmark Model Selection**: Compare Logistic Regression, $k$-NN, Decision Tree, and Random Forest models.
7. **📈 Model Evaluation & Tuning**: Evaluate precision, recall, F1-score, and perform hyperparameter tuning using Grid Search.
8. **🔄 Cross-Validation**: Measure cross-fold variance using 5-fold CV.
9. **🌲 Final Pipeline Selection**: Train the final model combining engineered features and optimized parameters.
10. **🧪 Inference & Submission**: Generate predictions on test data and submit to Kaggle.

---

## 📂 Dataset Overview

The dataset comes from the official [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic).

* **Training Set**: 891 passengers (12 columns, ground truth target: `Survived`)
* **Test Set**: 418 passengers (11 columns, target hidden)
* **Target Distribution**: Non-survivors = 549 (61.62%), Survivors = 342 (38.38%)

### Feature Reference & Missing Value Audit

| Variable | Description | Data Type | Missing Count (Train) | Imputation Strategy |
|---|---|---|---|---|
| `PassengerId` | Unique ID | Integer | 0 | None |
| `Survived` | Target Variable (0 = No, 1 = Yes) | Binary | 0 | None (Target) |
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | Categorical | 0 | None |
| `Name` | Passenger name | String | 0 | Feature Extracted (`Title`) |
| `Sex` | Gender (`male`, `female`) | Categorical | 0 | One-Hot Encoded |
| `Age` | Passenger age in years | Continuous | 177 | Median Imputation |
| `SibSp` | # of siblings / spouses aboard | Integer | 0 | Included in `FamilySize` |
| `Parch` | # of parents / children aboard | Integer | 0 | Included in `FamilySize` |
| `Ticket` | Ticket number | String | 0 | Excluded |
| `Fare` | Passenger fare | Continuous | 0 | Scaled |
| `Cabin` | Cabin number | String | 687 | Extracted Deck / Dropped |
| `Embarked` | Port of Embarkation (C, Q, S) | Categorical | 2 | Mode Imputation |

> **Note**: Raw dataset files (`train.csv`, `test.csv`) are excluded from this repository in compliance with competition guidelines.

---

## 🔎 Exploratory Data Analysis (EDA)

Key demographic findings from the training dataset:

### 1. Survival by Sex
Female passengers had a significantly higher survival rate than male passengers.
| Sex | Survival Rate |
|---|---:|
| **Female** | **74.20%** |
| **Male** | **18.89%** |

### 2. Survival by Passenger Class
First-class passengers survived at double the rate of third-class passengers.
| Class | Survival Rate |
|---|---:|
| **1st Class** | **62.96%** |
| **2nd Class** | **47.28%** |
| **3rd Class** | **24.24%** |

### 3. Combined Sex & Class Breakdown
The intersection of gender and class highlights extreme survival disparity:
| Sex | Class | Survival Rate |
|---|---|---:|
| Female | 1st Class | **96.81%** |
| Female | 2nd Class | **92.11%** |
| Female | 3rd Class | **50.00%** |
| Male | 1st Class | **36.89%** |
| Male | 2nd Class | **15.74%** |
| Male | 3rd Class | **13.54%** |

---

## 🛠️ Feature Engineering

Two core engineered features provided significant lift in predictive accuracy:

### 1. `FamilySize`
Aggregates immediate family members aboard:
$$\text{FamilySize} = \text{SibSp} + \text{Parch} + 1$$

* **Single Travelers (`FamilySize = 1`)**: 30.35% survival rate
* **Small Families (`FamilySize = 2-4`)**: Peak survival rate of 55.28% – 72.41%
* **Large Families (`FamilySize ≥ 5`)**: Sharp decline to <20.00% survival rate

### 2. `Title` Extraction
Extracted titles from passenger names to capture socio-economic status, gender, and age:
* **Normalizations**:
  * `Mlle` $\rightarrow$ `Miss`, `Ms` $\rightarrow$ `Miss`, `Mme` $\rightarrow$ `Mrs`
* **Rare Title Aggregation**: Titles with low counts (*Dr, Rev, Col, Major, Capt, Lady, Sir*) were aggregated into a unified `Rare` category.
* **Counts**: `Mr` (517), `Miss` (185), `Mrs` (126), `Master` (40), `Rare` (23)

---

## 🤖 Model Experimentation & Comparisons

Models were evaluated using a stratified 80/20 train/validation split (712 train samples, 179 validation samples).

### Model Performance Comparison

| Model Architecture | Hyperparameters / Notes | Validation Accuracy |
|---|---|---:|
| **Decision Tree** | Default (Unconstrained, Overfit on train: ~98.31%) | 78.21% |
| **Decision Tree** | Grid-Searched (`max_depth=6`) | 77.09% |
| **Expanded Random Forest** | Included `CabinDeck` Feature | 78.21% |
| **$k$-Nearest Neighbors ($k$-NN)** | Grid-Searched Best CV ($k=20$) | 79.33% |
| **Logistic Regression** | Baseline Pipeline | 80.45% |
| **Random Forest** | Baseline (No `Title`) | 81.56% |
| **Random Forest (Tuned)** | `max_depth=5`, `min_samples_leaf=1`, `n_estimators=100` | 80.45% |
| 🏆 **Random Forest + Title** | **Engineered `Title` Feature Included** | **82.12%** |

### Detailed Classification Report (Final Model: Random Forest + Title)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| **Did Not Survive (0)** | 0.84 | 0.87 | 0.86 |
| **Survived (1)** | 0.78 | 0.74 | 0.76 |
| **Overall / Weighted Average** | **0.82** | **0.82** | **0.82** |

---

## 📊 Feature Importance

Relative Gini importances computed from the tuned Random Forest model:

| Rank | Feature | Importance Score |
|---|---|---:|
| 1 | `Fare` | **0.2456** |
| 2 | `Age` | **0.2389** |
| 3 | `Sex - Female` | **0.1462** |
| 4 | `Sex - Male` | **0.1318** |
| 5 | `Pclass - 3` | **0.0499** |
| 6 | `FamilySize` | **0.0495** |
| 7 | `Pclass - 1` | **0.0303** |
| 8 | `SibSp` | **0.0290** |
| 9 | `Parch` | **0.0255** |
| 10 | `Embarked - S` | **0.0164** |
| 11 | `Pclass - 2` | **0.0160** |

---

## 🔄 Cross-Validation Results

A 5-fold cross-validation was run on the winning **Random Forest + Title** pipeline to ensure stability across dataset splits:

| Fold | Cross-Validation Accuracy |
|---|---:|
| **Fold 1** | 79.89% |
| **Fold 2** | 79.21% |
| **Fold 3** | **84.27%** |
| **Fold 4** | 76.40% |
| **Fold 5** | 80.90% |
| **Mean Accuracy** | **80.13%** |
| **Standard Deviation** | **±2.55%** |

---

## 🏆 Kaggle Competition Submission

The final model pipeline was retrained and used to generate predictions for the 418 unseen passengers in `test.csv`.

* **Output File**: `submission.csv` (`PassengerId`, `Survived`)
* **Kaggle Leaderboard Public Score**: **0.74641**

---

## 📂 Repository Structure

```text
titanic-survival-ml/
│
├── .gitignore          # Ignores raw dataset and submission files
├── 01_titanic.ipynb    # Complete Analysis, EDA, Modeling & CV Notebook
└── README.md           # Project Documentation
