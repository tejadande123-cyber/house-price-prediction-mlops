from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/house_prices.csv")
MODEL_PATH = Path("models/house_model.joblib")

FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "house_age",
]

TARGET = "price"
EXPERIMENT_NAME = "House Price Prediction"
TEST_SIZE = 0.20
RANDOM_STATE = 10

print("\nLoading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. Run python generate_data.py first."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

required_columns = FEATURES + [TARGET]
missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")

if df.empty:
    raise ValueError("Dataset is empty!")

if df[required_columns].isnull().any().any():
    raise ValueError("Dataset contains missing values!")

if not all(pd.api.types.is_numeric_dtype(df[column]) for column in required_columns):
    raise ValueError("All model columns must be numeric!")

print("\nData validation successful!")

# Independent variables and dependent variable
X = df[FEATURES]
y = df[TARGET]

print("\nIndependent variables (X):", FEATURES)
print("Dependent variable (y):", TARGET)

# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print("\nData splitting completed!")
print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)
print("\nMLflow experiment initialized:", EXPERIMENT_NAME)

with mlflow.start_run(run_name="Linear_Regression_Run"):

    model = LinearRegression()

    print("\nTraining Linear Regression model...")
    model.fit(X_train, y_train)
    print("Model training completed!")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation Results")
    print(f"MAE  : ₹{mae:,.2f}")
    print(f"RMSE : ₹{rmse:,.2f}")
    print(f"R2   : {r2:.4f}")

    # Record model parameters
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("total_records", len(df))
    mlflow.log_param("training_records", len(X_train))
    mlflow.log_param("testing_records", len(X_test))
    mlflow.log_param("feature_count", len(FEATURES))

    # Record evaluation metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Save trained model as an MLflow artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        name="house_price_model",
    )

    print("\nModel and metrics logged in MLflow!")

    # Save model using Joblib
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model saved successfully!")
    print("Model location:", MODEL_PATH)

print("\nLearned model coefficients:")
for feature, coefficient in zip(FEATURES, model.coef_):
    print(f"{feature}: {coefficient:,.2f}")

print(f"Intercept: {model.intercept_:,.2f}")

print("\n" + "=" * 50)
print("HOUSE PRICE MLOPS PIPELINE COMPLETED!")
print("=" * 50)
print("Dataset loading       : SUCCESS")
print("Data validation       : SUCCESS")
print("Train-test splitting  : SUCCESS")
print("Linear Regression     : SUCCESS")
print("Model evaluation      : SUCCESS")
print("MLflow tracking       : SUCCESS")
print("Joblib model saving   : SUCCESS")
