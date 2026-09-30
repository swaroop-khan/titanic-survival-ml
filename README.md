# Titanic Survival Prediction

A machine learning project based on the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

The goal is to predict whether a passenger survived the Titanic disaster using passenger and ticket information.

## Dataset

The dataset contains information about passengers aboard the Titanic, including:

- Passenger class
- Sex
- Age
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Fare
- Port of embarkation
- Cabin information
- Passenger name and title

The Kaggle dataset files are not included in this repository.

## Exploratory Data Analysis

Some of the main observations from the training data:

- Female passengers had a substantially higher survival rate than male passengers.
- First-class passengers had a higher survival rate than second- and third-class passengers.
- Passenger class and sex showed strong differences in survival rates.
- Family size showed different survival patterns depending on the size of the family.

## Feature Engineering

The project includes the following engineered features:

### FamilySize

Calculated as:

```text
FamilySize = SibSp + Parch + 1
