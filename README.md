🚢 Titanic Survival Prediction

An end-to-end Machine Learning classification project based on the Kaggle Titanic Competition.

The goal of this project is to predict whether a passenger survived the Titanic disaster using demographic, passenger, and travel information.

⸻

📌 Project Overview

The project follows a complete supervised machine learning workflow:

Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Hyperparameter Tuning → Cross-Validation → Kaggle Submission

The main objective was to build and compare multiple classification models while understanding each stage of the machine learning workflow.

⸻

🔍 What I Worked On

* Exploratory Data Analysis (EDA)
* Handling missing values
* Categorical feature encoding
* Feature engineering
* Feature scaling
* Train/validation splitting
* Classification models
* Model evaluation
* Hyperparameter tuning with Grid Search
* 5-fold cross-validation
* Kaggle prediction and submission

⸻

🛠️ Feature Engineering

Several features were engineered to improve the model:

* FamilySize — combines SibSp and Parch
* Title — extracted from passenger names
* CabinDeck — extracted from cabin information

The Title feature was particularly useful, with the final Random Forest + Title model achieving 82.12% validation accuracy.

⸻

🤖 Machine Learning Models

The following classification algorithms were trained and evaluated:

Model	Validation Accuracy
Decision Tree	78.21%
K-Nearest Neighbors (KNN)	79.33%
Logistic Regression	80.45%
Random Forest	81.56%
Tuned Random Forest	80.45%
Random Forest + Title	82.12%

The final Random Forest + Title model achieved 82.12% validation accuracy on the 80/20 stratified validation split.

⸻

📊 Model Evaluation

The final model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* 5-fold Cross-Validation

Final Model Classification Report

Class	Precision	Recall	F1-Score
Did Not Survive	84%	87%	86%
Survived	78%	74%	76%
Weighted Average	82%	82%	82%

⸻

🔄 5-Fold Cross-Validation

The final Random Forest + Title pipeline was evaluated using 5-fold cross-validation:

Fold	Accuracy
Fold 1	79.89%
Fold 2	79.21%
Fold 3	84.27%
Fold 4	76.40%
Fold 5	80.90%
Mean	80.13%
Standard Deviation	±2.55%

The average cross-validation accuracy was 80.13% ± 2.55%, providing a broader evaluation across multiple training/validation splits.

⸻

🏆 Kaggle Competition

The final model was retrained on the available training data and used to generate predictions for the 418 unseen passengers in the Kaggle test dataset.

Kaggle Public Leaderboard Score: 0.74641 🏆

View the Kaggle Titanic Competition

⸻

🛠️ Technologies Used

Technology	Purpose
Python	Programming language
Pandas	Data manipulation
NumPy	Numerical computation
Matplotlib	Data visualization
Seaborn	Statistical visualization
Scikit-learn	Machine Learning
Jupyter Notebook	Development & experimentation
Kaggle	Dataset & model submission

⸻

📁 Project Structure

titanic-survival-ml/
│
├── .gitignore
├── 01_titanic.ipynb
└── README.md

Raw Kaggle dataset files are excluded from the repository.
