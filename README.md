# House Price Prediction Using MLOps

## Project Information

-   **Name:** Dande Teja
-   **Employee Code:** 8276
-   **District:** Namakkal
-   **Project:** House Price Prediction Using MLOps
-   **GitHub Repository:**
    https://github.com/tejadande123-cyber/house-price-prediction-mlops

## 1. Project Overview

This project implements a simple end-to-end Machine Learning and MLOps
workflow for house price prediction.

The application predicts the price of a house using:

-   House area in square feet
-   Number of bedrooms
-   Number of bathrooms
-   Age of the house

A **Linear Regression** model is trained using Scikit-learn. The
experiment is tracked with **MLflow**, the trained model is saved using
**Joblib**, the source code is versioned using **Git/GitHub**, and the
model is deployed as a REST API using **FastAPI**.

## 2. Problem Statement

A real estate company wants to develop a Machine Learning application
that predicts the price of a house based on its size, number of
bedrooms, number of bathrooms and age of the property.

The objective of this practical is to build a simple Machine Learning
model and implement an MLOps workflow to:

1.  Prepare the dataset
2.  Preprocess the data
3.  Train a Linear Regression model
4.  Evaluate the model
5.  Track experiments using MLflow
6.  Save the trained model
7.  Maintain the source code using Git
8.  Deploy the model using FastAPI
9.  Test the prediction API using Swagger UI

## 3. Project Structure

``` text
house-price-prediction-mlops/
|
+-- data/
|   +-- house_prices.csv
|
+-- models/
|   +-- house_model.joblib
|
+-- generate_data.py
+-- train.py
+-- api.py
+-- requirements.txt
+-- .gitignore
+-- README.md
```

## 4. Dataset

The dataset is stored in:

``` text
data/house_prices.csv
```

It contains the following columns:

  Column        Description
  ------------- --------------------------------------
  `area_sqft`   House area in square feet
  `bedrooms`    Number of bedrooms
  `bathrooms`   Number of bathrooms
  `house_age`   Age of the house in years
  `price`       House price in INR - target variable

The five illustrative records given in the assignment are retained:

    Area   Bedrooms   Bathrooms   Age   Price (INR)
  ------ ---------- ----------- ----- -------------
    1000          2           1     5     3,500,000
    1500          3           2     3     5,500,000
    2000          3           2     8     6,500,000
    2500          4           3     2     9,500,000
    1200          2           2    10     3,800,000

The program generates **995 additional synthetic records**, resulting in
a total of **1,000 records**.

The synthetic data is used only for this learning exercise.

### Synthetic Price Logic

The generated price follows a simple assumed relationship:

``` python
price = (
    500000
    + 2800 * area_sqft
    + 250000 * bedrooms
    + 300000 * bathrooms
    - 40000 * house_age
    + noise
)
```

Random noise is added to create variation in the synthetic dataset.

## 5. Task 1 - Dataset Preparation

The dataset is created in `generate_data.py`.

Important operations:

``` python
OUTPUT = Path("data/house_prices.csv")
df.to_csv(OUTPUT, index=False)

print(df.head())
print(df.isnull().sum())
```

This completes the required activities:

-   Create `house_prices.csv`
-   Load/process data using Pandas
-   Display the first five records
-   Check missing values

## 6. Task 2 - Data Preprocessing

The independent variables are:

``` python
FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "house_age"
]
```

The target variable is:

``` python
TARGET = "price"
```

The data is separated into `X` and `y`:

``` python
X = df[FEATURES]
y = df[TARGET]
```

The dataset is split into **80% training** and **20% testing**:

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=10
)
```

For 1,000 records:

-   Training records: **800**
-   Testing records: **200**

## 7. Task 3 - Model Development

A Linear Regression model is used:

``` python
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The model is evaluated using:

-   MAE - Mean Absolute Error
-   RMSE - Root Mean Squared Error
-   R2 - R-Squared Score

``` python
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
```

### Obtained Results

  Metric             Result
  -------- ----------------
  MAE        INR 316,895.84
  RMSE       INR 402,549.35
  R2                 0.9805

The R2 value is approximately **0.98** for this synthetic exercise
dataset.

## 8. Task 4 - Experiment Tracking with MLflow

The MLflow experiment is created with the name:

``` text
House Price Prediction
```

The experiment is initialized using:

``` python
mlflow.set_experiment(EXPERIMENT_NAME)
```

The run name is:

``` text
Linear_Regression_Run
```

### Parameters Logged

The following parameters are recorded:

-   Model type
-   Test size
-   Random state
-   Total records
-   Training records
-   Testing records
-   Feature count

``` python
mlflow.log_param("model_type", "Linear Regression")
mlflow.log_param("test_size", TEST_SIZE)
mlflow.log_param("random_state", RANDOM_STATE)
mlflow.log_param("total_records", len(df))
mlflow.log_param("training_records", len(X_train))
mlflow.log_param("testing_records", len(X_test))
mlflow.log_param("feature_count", len(FEATURES))
```

### Metrics Logged

``` python
mlflow.log_metric("mae", mae)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
```

### Model Artifact

The trained Scikit-learn model is also logged to MLflow:

``` python
mlflow.sklearn.log_model(
    sk_model=model,
    name="house_price_model"
)
```

To view the MLflow interface:

``` bash
mlflow ui
```

Open:

``` text
http://127.0.0.1:5000
```

## 9. Task 5 - Model Versioning

### Joblib Model Saving

The trained model is saved as:

``` text
models/house_model.joblib
```

using:

``` python
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
```

This saved model can be reused by the API without retraining the model
every time.

### Git and GitHub

Git is used for source-code version control.

GitHub repository:

**https://github.com/tejadande123-cyber/house-price-prediction-mlops**

Typical Git workflow:

``` bash
git add .
git commit -m "Update House Price Prediction MLOps project"
git push
```

## 10. Task 6 - Model Deployment with FastAPI

The trained Joblib model is loaded by `api.py`:

``` python
model = joblib.load(MODEL_PATH)
```

The input structure is:

``` python
class HouseInput(BaseModel):
    area_sqft: float
    bedrooms: int
    bathrooms: int
    house_age: float
```

The prediction endpoint is:

``` text
POST /predict
```

Main prediction logic:

``` python
@app.post("/predict")
def predict(house: HouseInput):
    row = pd.DataFrame(
        [house.model_dump()],
        columns=FEATURES
    )

    prediction = float(model.predict(row)[0])

    return {
        "predicted_price": round(prediction, 2),
        "currency": "INR"
    }
```

## 11. API Testing with Swagger UI

Start FastAPI using:

``` bash
uvicorn api:app --reload
```

Open Swagger UI:

``` text
http://127.0.0.1:8000/docs
```

Example request:

``` json
{
  "area_sqft": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "house_age": 5
}
```

The tested API returned a successful HTTP **200** response:

``` json
{
  "predicted_price": 5823005.79,
  "currency": "INR"
}
```

## 12. Dependencies

The project uses:

``` text
numpy
pandas
scikit-learn
mlflow
joblib
fastapi
uvicorn
```

Install them using:

``` bash
python -m pip install -r requirements.txt
```

## 13. How to Run the Complete Project

### Step 1 - Install dependencies

``` bash
python -m pip install -r requirements.txt
```

### Step 2 - Generate dataset

``` bash
python generate_data.py
```

### Step 3 - Train and evaluate the model

``` bash
python train.py
```

### Step 4 - View MLflow experiment

``` bash
mlflow ui
```

Open:

``` text
http://127.0.0.1:5000
```

### Step 5 - Start FastAPI

``` bash
uvicorn api:app --reload
```

### Step 6 - Test using Swagger UI

Open:

``` text
http://127.0.0.1:8000/docs
```

Use `POST /predict`, enter the house details and execute the request.

## 14. MLOps Workflow

``` text
Dataset Generation
        |
        v
Data Validation
        |
        v
Feature / Target Selection
        |
        v
80% Training / 20% Testing
        |
        v
Linear Regression Training
        |
        v
Model Prediction
        |
        v
MAE / RMSE / R2 Evaluation
        |
        +----------------------+
        |                      |
        v                      v
MLflow Tracking          Joblib Model Saving
        |                      |
        +-----------+----------+
                    |
                    v
              FastAPI Deployment
                    |
                    v
              Swagger UI Testing
```

## 15. Conclusion

This project demonstrates a complete basic MLOps workflow for a house
price prediction application.

The practical covers:

-   Dataset creation
-   Pandas-based data handling
-   Missing-value checking
-   Feature and target selection
-   Train-test splitting
-   Linear Regression
-   Model evaluation
-   MLflow experiment tracking
-   Model artifact logging
-   Joblib model persistence
-   Git/GitHub version control
-   FastAPI deployment
-   Swagger UI testing

The model achieved an R2 score of approximately **0.98** on the
synthetic exercise dataset. Since the dataset and price relationship are
synthetic, the result is intended to demonstrate the Machine Learning
and MLOps workflow rather than provide real-world property valuations.

------------------------------------------------------------------------

**Submitted by:** Dande Teja\
**Employee Code:** 8276\
**District:** Namakkal\
**GitHub:**
https://github.com/tejadande123-cyber/house-price-prediction-mlops
