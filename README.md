# 🚢 Titanic Survival Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Titanic-20BEFF?logo=kaggle&logoColor=white)

An end-to-end machine learning classification project based on the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic). The goal is to predict whether a passenger survived the Titanic disaster using demographic, passenger, and travel information.

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Feature Engineering](#️-feature-engineering)
- [Models and Results](#-models-and-results)
- [Model Evaluation](#-model-evaluation)
- [Cross-Validation](#-5-fold-cross-validation)
- [Kaggle Submission](#-kaggle-submission)
- [Key Takeaways](#-key-takeaways)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)

## 📌 Project Overview

The project follows a complete supervised learning workflow:

```
Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation
     → Hyperparameter Tuning → Cross-Validation → Kaggle Submission
```

The main objective was to build and compare multiple classification models while understanding each stage of the machine learning pipeline.

**Covered in this project:**

- Exploratory Data Analysis (EDA)
- Handling missing values
- Categorical feature encoding
- Feature engineering and scaling
- Stratified train/validation splitting
- Training and comparing classification models
- Evaluation with multiple metrics
- Hyperparameter tuning with Grid Search
- 5-fold cross-validation
- Generating a Kaggle submission

## 🛠️ Feature Engineering

| Feature | Description |
|---|---|
| `FamilySize` | Combines `SibSp` and `Parch` |
| `Title` | Extracted from passenger names |
| `CabinDeck` | Extracted from cabin information |

The `Title` feature was particularly useful: the final **Random Forest + Title** model reached the best validation accuracy.

## 🤖 Models and Results

All models were evaluated on an 80/20 stratified validation split.

| Model | Validation Accuracy |
|---|---|
| Decision Tree | 78.21% |
| K-Nearest Neighbors | 79.33% |
| Logistic Regression | 80.45% |
| Random Forest | 81.56% |
| Tuned Random Forest (Grid Search) | 80.45% |
| **Random Forest + Title** | **82.12%** |

## 📊 Model Evaluation

The final model was evaluated using accuracy, precision, recall, F1-score, a confusion matrix, and 5-fold cross-validation.

**Classification report (final model):**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Did Not Survive | 84% | 87% | 86% |
| Survived | 78% | 74% | 76% |
| **Weighted Avg** | **82%** | **82%** | **82%** |

## 🔄 5-Fold Cross-Validation

The final Random Forest + Title pipeline was evaluated across five train/validation splits for a broader view of performance.

| Fold | Accuracy |
|---|---|
| Fold 1 | 79.89% |
| Fold 2 | 79.21% |
| Fold 3 | 84.27% |
| Fold 4 | 76.40% |
| Fold 5 | 80.90% |
| **Mean ± Std** | **80.13% ± 2.55%** |

## 🏆 Kaggle Submission

The final model was retrained on the available training data and used to predict survival for the 418 unseen passengers in the Kaggle test set.

**Public Leaderboard Score: `0.74641`**

## 💡 Key Takeaways

- Adding the `Title` feature gave the largest single improvement over the baseline Random Forest.
- Grid Search did not beat the default Random Forest on this validation split, so tuning does not always help on a small dataset.
- The ±2.55% spread across folds shows that a single validation split can be noisy. The cross-validation mean (80.13%) is a more reliable estimate than the 82.12% single-split score.

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| Jupyter Notebook | Development and experimentation |
| Kaggle | Dataset and submission |

## 📁 Project Structure

```
titanic-survival-ml/
├── .gitignore
├── 01_titanic.ipynb
└── README.md
```

> Raw Kaggle dataset files are excluded from the repository.

## 🚀 Getting Started

1. Clone the repository:
```bash
   git clone https://github.com/<your-username>/titanic-survival-ml.git
   cd titanic-survival-ml
```
2. Install dependencies:
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```
3. Download `train.csv` and `test.csv` from the [Kaggle Titanic page](https://www.kaggle.com/competitions/titanic/data) and place them in the project folder.
4. Open and run the notebook:
```bash
   jupyter notebook 01_titanic.ipynb
```
