# Data-Crunch

# Harveston Climate Prediction 🌾🌦️

## Overview
Harveston's climate is shifting unpredictably, affecting agriculture and food security. This project aims to develop time series forecasting models to predict five critical environmental variables:
- Average Temperature (°C)
- Radiation (W/m²)
- Rain Amount (mm)
- Wind Speed (km/h)
- Wind Direction (°)

## Models Used
### 1. RandomForestRegressor (Code 1)
- Uses **Random Forest**, an ensemble learning model that builds multiple decision trees and averages their predictions.
- Handles feature engineering, categorical encoding, and missing data preprocessing.

### 2. LightGBM (Code 2)
- Implements **LightGBM**, an optimized **ensemble gradient boosting model** that builds trees sequentially to improve predictions.
- Uses boosting techniques for efficient learning and high performance.
- Performs hyperparameter tuning and feature extraction.

### 3. XGBoost (Code 3)
- Uses **XGBoost**, another powerful **ensemble gradient boosting model** known for efficiency and regularization.
- Applies gradient boosting with optimized tree-building techniques.
- Handles categorical encoding, missing values, and feature engineering.

## Dataset
- The dataset contains historical environmental records from different kingdoms in Harveston.
- The test dataset includes `ID`, `Year`, `Month`, `Day`, and `kingdom`, requiring predictions for the five target variables.
