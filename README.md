# Daily-Electricity-Consumption-prediction

This repository contains a machine learning pipeline for forecasting daily energy consumption using a stacked ensemble approach. The goal is to minimize the prediction error by combining multiple regression models and optimizing preprocessing, feature engineering, and hyperparameters.

## 📌 Project Overview

This project was developed as part of the **Data Science Academy** selection process. It aims to build a robust and accurate regression model that predicts energy consumption using historical data and engineered features. The final solution integrates multiple models using the **Stacking Regressor** technique, resulting in a significant reduction in RMSE from 46 to **18**.

## 🧰 Technologies & Libraries

- **Python**
- `pandas`, `numpy` – data handling
- `matplotlib`, `seaborn` – data visualization
- `scikit-learn` – model building, evaluation, preprocessing
- `CatBoost`, `LightGBM` – gradient boosting models
- `Ridge` – used as meta-learner in stacking
- Visual Studio Code – development environment

## 📊 Dataset

The dataset includes various numerical and categorical features (not included in this repository due to privacy). The task is to predict energy consumption based on these features.

## 🔁 Pipeline Steps

1. **Data Exploration**
   - Inspect structure and summary statistics
   - Visualize distributions and correlations

2. **Preprocessing**
   - Label encoding and OneHot encoding for categorical features
   - Feature scaling and normalization

3. **Feature Engineering**
   - Creating new informative features
   - Handling missing values and outliers

4. **Modeling**
   - Base models: `CatBoostRegressor`, `LightGBMRegressor`, `RandomForestRegressor`
   - Meta-model: `Ridge Regression`
   - Combined using `StackingRegressor` from `sklearn`

5. **Evaluation**
   - Metric used: **Root Mean Squared Error (RMSE)**
   - Before optimization: RMSE ≈ 46
   - After optimization: RMSE ≈ **18**

6. **Visualization**
   - Feature distributions
   - Prediction vs actual plots
   - Correlation heatmaps

## 🧠 Key Learnings

- Ensemble learning (stacking) significantly improves performance by leveraging the strengths of multiple models.
- Feature engineering and careful preprocessing are critical to reducing error.
- Hyperparameter tuning and encoding strategies affect both base and meta models in stacking.

## 🐞 Challenges

- Handling categorical features with imbalanced classes
- Selecting models that are diverse yet complementary for stacking
- Balancing the complexity and generalization of the model

## 🚀 Getting Started

Clone this repository and install the required libraries:

```bash
git clone https://github.com/your-username/energy-forecasting-stacking.git
cd energy-forecasting-stacking
pip install -r requirements.txt
