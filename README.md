# Data Crunch - CSE, UoM

# Harveston Climate Prediction 🌾🌦️

## Overview
Harveston's climate is shifting unpredictably, affecting agriculture and food security. This project aims to develop time series forecasting models to predict five critical environmental variables:
- Average Temperature (°C)
- Radiation (W/m²)
- Rain Amount (mm)
- Wind Speed (km/h)
- Wind Direction (°)

## Models Used
### 1. RandomForestRegressor ([Code 1](code%201.py))
- Uses **Random Forest**, an **ensemble learning model** that builds multiple decision trees and averages their predictions.
- Handles feature engineering, categorical encoding, and missing data preprocessing.

### 2. LightGBM ([Code 2](code%202.py))
- Implements **LightGBM**, an optimized **ensemble gradient boosting model** that builds trees sequentially to improve predictions.
- Uses boosting techniques for efficient learning and high performance.
- Performs hyperparameter tuning and feature extraction.

### 3. XGBoost ([Code 3](code%203.py))
- Uses **XGBoost**, another powerful **ensemble gradient boosting model** known for efficiency and regularization.
- Applies gradient boosting with optimized tree-building techniques.
- Handles categorical encoding, missing values, and feature engineering.

### 4. LSTM ([Code 4](code%204.py))
- Uses an LSTM (Long Short-Term Memory) model, a type of **recurrent neural network** (RNN) designed for sequential data processing.
- Employs memory cells to capture long-term dependencies and temporal patterns in data.
- Trains the model with early stopping, batch processing, and validation to optimize performance.

### 5. Gradient Boosting ([Code 5](code%205.py))
- Uses Gradient Boosting, an **ensemble learning technique** that builds trees sequentially to minimize prediction errors.
- Applies boosting by adjusting model weights iteratively to improve accuracy.
- Optimizes tree depth, learning rate, and subsampling for enhanced performance and generalization.

### 6. BaggingRegressor ([Code 6](code%206.py))
- Implements Bagging, an ensemble learning method that trains multiple base regressors (Decision Trees) on random subsets of the data and averages their predictions.
- Enhances model stability, reduces variance, and improves generalization by combining multiple weak learners.
- Uses bootstrap sampling and feature selection to improve robustness and mitigate overfitting.

### 7. Stack ([Code 7](code%207.py))
- Implements a stacked ensemble model that combines multiple base regressors to enhance prediction accuracy.
- Uses diverse base models, including Random Forest, Gradient Boosting, XGBoost, Ridge Regression, and Lasso Regression, to capture different aspects of the data.
- Employs k-fold cross-validation to generate out-of-fold predictions for training a meta-model (XGBoost) that learns from the base models' outputs.

## Dataset
- The dataset contains historical environmental records from different kingdoms in Harveston.
- The test dataset includes `ID`, `Year`, `Month`, `Day`, and `kingdom`, requiring predictions for the five target variables.

## Evaluation Metric
Predictions are evaluated using **Symmetric Mean Absolute Percentage Error (sMAPE)**:

$$sMAPE = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_{true,i} - y_{pred,i}|}{(|y_{true,i}| + |y_{pred,i}|)/2}$$

The final score is the average sMAPE across all target columns.
