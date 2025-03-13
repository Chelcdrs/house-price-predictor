import numpy as np
import pandas as pd
import os
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model path
model_path = "house_price_model.pkl"
scaler_path = "scaler.pkl"

# 1. Data Preprocessing
# Load the dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target  # Target variable

# Split the data into features (X) and target (y)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']
scaler = StandardScaler()  # Scale the numerical features to standardize them
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Model Training & Evaluation
# Hyperparameter Optimization
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
logger.info(f"Best Model Parameters: {grid_search.best_params_}")

# Evaluate model and calculate performance metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error (MAE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root Mean Squared Error (RMSE)
r2 = r2_score(y_test, y_pred)  # R² score

# Print model performance
logger.info(f"Model Performance: MAE={mae}, RMSE={rmse}, R²={r2}")

# Save model with compression to reduce size
joblib.dump(model, model_path, compress=3)  # Compress the model with a compression level (3 is a good balance)
joblib.dump(scaler, scaler_path, compress=3)  # Compress the scaler with compression level