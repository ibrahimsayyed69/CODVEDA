# Medical Insurance Cost Prediction

This repository contains a Machine Learning project designed to analyze and predict medical insurance costs using the classic Medical Cost Personal Dataset. The project implements, evaluates, and compares three different regression models to determine which algorithm performs best at predicting medical charges.

---

## Project Overview

Medical insurance pricing is influenced by various demographic and lifestyle factors. This project aims to:

1. Analyze the impact of features like smoking status, BMI, and age on medical charges.
2. Preprocess categorical variables into numerical formats suitable for machine learning models.
3. Train and Compare three regression algorithms:
* Linear Regression (Baseline)
* Decision Tree Regressor
* Random Forest Regressor (Ensemble)


4. Evaluate performance using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) Score.

---

## Dataset Description

The dataset (insurance.csv) consists of 1,338 records with the following features:

* age: Age of primary beneficiary (Numerical)
* sex: Insurance contractor gender (Categorical: female, male)
* bmi: Body mass index (Numerical: kg/m^2)
* children: Number of children covered by health insurance (Numerical)
* smoker: Smoking status (Categorical: yes, no)
* region: The beneficiary's residential area in the US (Categorical)
* charges: Individual medical costs billed by health insurance (Numerical - Target Variable)

---

## Getting Started

### Prerequisites

Make sure you have Python installed, along with the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

### Installation & Execution

1. Clone this repository to your local machine.
2. Place your insurance.csv file in the same directory as the script.
3. Run the script:

python insurance_prediction.py

---

## Data Preprocessing

The pipeline handles categorical data transformation prior to model training:

* Binary Mapping: sex (female: 0, male: 1) and smoker (no: 0, yes: 1).
* One-Hot Encoding: region is converted into dummy variables using pd.get_dummies() with drop_first=True to avoid the dummy variable trap.
* Train-Test Split: The dataset is split into 80% training and 20% testing sets.

---

## Model Performance & Comparison

After training the models on the dataset, the evaluation metrics on the test set yield the following results:

* Random Forest:
* MSE: ~20,942,520.26
* RMSE: ~4,576.30
* R2 Score: ~0.8651


* Linear Regression:
* MSE: ~33,596,915.02
* RMSE: ~5,796.28
* R2 Score: ~0.7833


* Decision Tree:
* MSE: ~42,446,908.01
* RMSE: ~6,515.13
* R2 Score: ~0.7266



### Key Insights:

* Random Forest outperforms the other models significantly, capturing roughly 86.5% of the variance in medical charges.
* Smoking status is the most critical predictor of high medical costs. The Linear Regression model indicates that being a smoker adds an average of $23,651.13 to medical charges, holding all other features constant.

---