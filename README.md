# House Price Prediction – MLOps Project

## Project Overview

This project demonstrates a simple end-to-end Machine Learning and MLOps workflow for predicting house prices.

The model estimates the price of a house using four input features:

- Area of the house in square feet
- Number of bedrooms
- Number of bathrooms
- Age of the house

A **Linear Regression** model is used for prediction.

The project also demonstrates basic MLOps concepts such as experiment tracking using MLflow, model persistence using Joblib, source-code versioning using Git, and model deployment using FastAPI.

---

## Project Workflow

The complete workflow is:

```text
Generate Dataset
      ↓
Load and Validate Data
      ↓
Select Features and Target
      ↓
80% Training / 20% Testing
      ↓
Train Linear Regression Model
      ↓
Evaluate the Model
      ↓
Track Experiment using MLflow
      ↓
Save Model using Joblib
      ↓
Deploy Model using FastAPI
      ↓
Predict House Price
```

---

## Project Structure

```text
house-price-prediction-mlops/
│
├── data/
│   └── house_prices.csv
│
├── models/
│   └── house_model.joblib
│
├── generate_data.py
├── train.py
├── api.py
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

**generate_data.py**

Creates the house-price dataset and stores it in:

```text
data/house_prices.csv
```

**train.py**

Performs data loading, validation, train-test splitting, Linear Regression training, model evaluation, MLflow tracking, and Joblib model saving.

**api.py**

Loads the trained model and provides a FastAPI endpoint for house-price prediction.

**requirements.txt**

Contains the Python packages required to run the project.

---

## Dataset

The dataset contains the following columns:

| Column | Description |
|---|---|
| `area_sqft` | Area of the house in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `house_age` | Age of the house in years |
| `price` | House price in INR |

The first five records are the illustrative examples provided for the practical assignment.

The program generates another **995 synthetic records**, resulting in a total of:

```text
1000 records
```

The additional records are generated with variations in area, bedrooms, bathrooms, house age and price.

The synthetic data is intended only for demonstrating the Machine Learning and MLOps workflow.

---

## Machine Learning Model

The project uses:

```text
Linear Regression
```

The independent variables are:

```text
X = area_sqft, bedrooms, bathrooms, house_age
```

The dependent variable is:

```text
y = price
```

The dataset is divided into:

```text
Training Data : 80%
Testing Data  : 20%
```

For 1,000 records, this gives approximately:

```text
Training records : 800
Testing records  : 200
```

---

## Model Evaluation

After training, predictions are generated for the test data.

The model is evaluated using three regression metrics:

### MAE – Mean Absolute Error

Measures the average absolute difference between the actual house price and predicted house price.

Lower MAE indicates better prediction accuracy.

### RMSE – Root Mean Squared Error

Measures prediction error while giving greater importance to larger errors.

Lower RMSE is better.

### R² – R-Squared Score

Shows how much variation in house prices is explained by the model.

An R² value closer to `1` indicates a better fit.

---

## MLflow Experiment Tracking

MLflow is used to keep track of the model training experiment.

The experiment name is:

```text
House Price Prediction
```

For each training run, the project records information such as:

```text
Model Type
Test Size
Random State
Number of Records
Training Records
Testing Records
Feature Count
```

The following evaluation metrics are also recorded:

```text
MAE
RMSE
R²
```

The trained Scikit-learn model is also logged as an MLflow model artifact.

To start the MLflow interface:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

Select the **House Price Prediction** experiment to inspect the run parameters, metrics and model artifact.

---

## Model Saving with Joblib

Apart from MLflow tracking, the trained Linear Regression model is saved locally using Joblib.

The model is stored at:

```text
models/house_model.joblib
```

This allows the trained model to be loaded later for prediction without training it again.

---

## FastAPI Deployment

FastAPI is used to expose the trained model as a prediction API.

The API loads:

```text
models/house_model.joblib
```

and accepts house information from the user.

The model then returns the estimated house price.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tejadande123-cyber/house-price-prediction-mlops.git
```

Move into the project directory:

```bash
cd house-price-prediction-mlops
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Required Packages

```bash
python -m pip install -r requirements.txt
```

---

## Running the Project

### Step 1 – Generate the Dataset

```bash
python generate_data.py
```

This creates:

```text
data/house_prices.csv
```

The program also displays the first five records, missing-value information and basic dataset statistics.

### Step 2 – Train the Model

```bash
python train.py
```

This performs the complete training workflow:

```text
Load Dataset
      ↓
Validate Dataset
      ↓
Split Training and Testing Data
      ↓
Train Linear Regression
      ↓
Generate Predictions
      ↓
Calculate MAE, RMSE and R²
      ↓
Record Experiment in MLflow
      ↓
Save Model using Joblib
```

After successful training, the model will be available at:

```text
models/house_model.joblib
```

### Step 3 – View MLflow Experiments

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

### Step 4 – Start the FastAPI Application

Open another terminal and run:

```bash
uvicorn api:app --reload
```

The application will run locally at:

```text
http://127.0.0.1:8000
```

### Step 5 – Open Swagger UI

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger UI can be used to test the prediction API directly from the browser.

---

## Prediction Example

A prediction request contains the house details.

Example:

```json
{
  "area_sqft": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "house_age": 5
}
```

The API processes these values using the trained Linear Regression model and returns a response in the following form:

```json
{
  "predicted_price": 0000000.00,
  "currency": "INR"
}
```

The actual predicted value depends on the trained model.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations and synthetic data generation |
| Pandas | Dataset creation and processing |
| Scikit-learn | Linear Regression and model evaluation |
| MLflow | Experiment tracking and model artifact logging |
| Joblib | Saving and loading the trained model |
| FastAPI | Model deployment as an API |
| Uvicorn | Running the FastAPI application |
| Git | Source-code version control |
| GitHub | Repository hosting |

---

## Learning Outcomes

This practical project demonstrates how the different stages of a Machine Learning application can be connected into a basic MLOps workflow.

The project covers:

- Dataset preparation
- Data validation
- Feature and target selection
- Train-test splitting
- Linear Regression
- Model evaluation
- Experiment tracking
- Model persistence
- Version control
- REST API deployment
- Swagger-based API testing

---

## Note

The dataset used in this project contains synthetically generated records and is intended for learning and demonstration purposes.

The predicted prices should therefore not be considered real-world property valuations.
