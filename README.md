# House Price Prediction - Machine Learning Model

## Table of Contents
- [How to Run the Project Locally](#how-to-run-the-project-locally)
- [How to Use the API](#how-to-use-the-api)
- [Example Prediction Request](#example-prediction-request)
- [GitHub Repository](#github-repository)

## 1. Introduction
This project predicts house prices using machine learning. The steps include loading a dataset (California Housing Dataset), cleaning the data, training a model, improving it, and creating an API for predictions. The model is deployed using FastAPI, so people can make predictions using a web request.

## 2. Data Preprocessing & Feature Engineering

### a. Dataset Overview:
We used the California Housing Dataset from Scikit-learn. It has details about houses like location, income, and house age.
The target (what we're predicting) is the house price (MedHouseVal).

### b. Data Preprocessing:
- **Missing Values**: The dataset didn't have any missing data, so we didn’t need to fix that.
- **Feature Scaling**: We used StandardScaler() to scale the numbers, so that all features are treated equally.
- **Feature Selection**: We used all features in the dataset because they are important for predicting the house price.

### c. Feature Engineering:
- We scaled the data using StandardScaler().
- We split the data into features (X) and target (y).
- The data was then split into training and testing sets (80% for training, 20% for testing).

## 3. Model Selection & Optimization

### a. Model Selection:
We chose a Random Forest Regressor model because it's good at making predictions from complex data and doesn’t overfit easily.

### b. Model Optimization:
We improved the model by tuning its settings. We tried different values for:
- n_estimators: Number of trees (tried 50, 100, 200).
- max_depth: Maximum depth of trees (tried None, 10, 20).
- min_samples_split: Minimum number of samples to split (tried 2, 5, 10).

### c. Model Evaluation:
After improving the model, we tested it and calculated:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

### d. Model Performance:
The final model performed well with:
- MAE: 0.530
- RMSE: 0.670
- R² score: 0.835

### e. Saving the Model:
The trained model was saved using Joblib as `house_price_model.pkl` and the scaler as `scaler.pkl`.

## 4. Model Deployment

### a. API Deployment Strategy:
The model is deployed using FastAPI, which allows others to get predictions through a web request.

### b. Deployment Steps:
- The app.py file contains the FastAPI app. It has an endpoint `/predict`, which takes the input data in JSON format and returns the predicted house price.
- **Docker**: The app is containerized using Docker. This makes it easy to run anywhere without worrying about setup.

### c. API Usage Guide:
- **Endpoint**: `/predict`
- **Request Format**: JSON with features like MedInc, HouseAge, AveRooms, etc. Example request:
  {
    "MedInc": 8.0,
    "HouseAge": 30.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1200.0,
    "AveOccup": 3.5,
    "Latitude": 34.0,
    "Longitude": -118.0
  }
- **Response**: The API will return the predicted house price. Example response:
  {
      "predicted_price": 3.75
  }

### d. Testing the API:
You can test the API with tools like Postman or cURL by sending a POST request to the `/predict` endpoint with your input data.

### e. Docker Setup:
- **Dockerfile**: The project is packaged using Docker. The Dockerfile installs the necessary dependencies and runs the FastAPI app.
- **Build the Docker image**: docker build -t house-price-prediction .
- **Run the Docker container**: docker run -p 8000:8000 house-price-prediction

## 5. GitHub Repository
The project is stored in a GitHub repository that includes:
- Python scripts (main.py, app.py).
- Model file (`house_price_model.pkl`, `scaler.pkl`).
- Dockerfile.
- requirements.txt for dependencies.

[GitHub Repository](Link to GitHub Repository)

## 6. Conclusion
This project shows how to clean data, train a machine learning model, optimize it, deploy it as an API using FastAPI, and package the app using Docker. The model provides accurate house price predictions and the API makes it easy for others to use the model.

## How to Run the Project Locally
1. Clone the repository:
   git clone https://github.com/yourusername/house-price-prediction.git
2. Install the required dependencies:
   pip install -r requirements.txt
3. Run the FastAPI app:
   python app.py
4. Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## How to Use the API
- Send a POST request to the `/predict` endpoint with the necessary JSON data (house features) and receive the predicted house price as a response.

## Example Prediction Request
POST request to [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) with the following JSON data:

{
  "MedInc": 8.0,
  "HouseAge": 30.0,
  "AveRooms": 6.0,
  "AveBedrms": 1.0,
  "Population": 1200.0,
  "AveOccup": 3.5,
  "Latitude": 34.0,
  "Longitude": -118.0
}

Response:
{
    "predicted_price": 3.75
}
