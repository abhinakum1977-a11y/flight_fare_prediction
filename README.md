# Flight Fare Prediction using Machine Learning

A Machine Learning regression project that predicts airline ticket prices based on airline, journey date, source, destination, duration, and number of stops.

## Project Overview

Flight ticket prices vary depending on multiple factors such as airline, travel route, departure time, journey duration, and total stops. This project analyzes historical flight data, performs feature engineering, and trains regression models to accurately predict flight fares.

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering from date & time
- Categorical feature encoding
- Multiple regression model training
- Hyperparameter tuning
- Flight fare prediction

## Dataset

Historical domestic flight dataset containing airline, source, destination, route, departure/arrival time, duration, total stops, and ticket price.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- RandomizedSearchCV

## Evaluation Metrics

- Mean Absolute Error (MAE)
- R² Score
- Prediction Error Analysis

## Installation

```bash
git clone https://github.com/yourusername/flight-fare-prediction.git

cd flight-fare-prediction

pip install -r requirements.txt
```

## Future Improvements

- Deploy using Streamlit
- Real-time flight fare prediction
- XGBoost & CatBoost implementation
- Integration with live flight APIs
