from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path("data/house_prices.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(10)

n = 995

# Five illustrative records given in the assignment
example_data = pd.DataFrame({
    "area_sqft": [1000, 1500, 2000, 2500, 1200],
    "bedrooms": [2, 3, 3, 4, 2],
    "bathrooms": [1, 2, 2, 3, 2],
    "house_age": [5, 3, 8, 2, 10],
    "price": [3500000, 5500000, 6500000, 9500000, 3800000],
})

# Generate 995 synthetic records
area_sqft = rng.integers(600, 3501, n)

bedrooms = np.clip(
    np.round(area_sqft / 900 + rng.normal(0, 0.6, n)),
    1, 5
).astype(int)

bathrooms = np.clip(
    np.round(bedrooms * 0.7 + rng.normal(0, 0.5, n)),
    1, 4
).astype(int)

house_age = rng.integers(0, 31, n)

# Synthetic price relationship plus random variation
noise = rng.normal(0, 400000, n)

price = (
    500000
    + 2800 * area_sqft
    + 250000 * bedrooms
    + 300000 * bathrooms
    - 40000 * house_age
    + noise
)

price = np.round(price, -3).astype(int)

synthetic_data = pd.DataFrame({
    "area_sqft": area_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "house_age": house_age,
    "price": price,
})

# 5 example rows + 995 synthetic rows = 1000
df = pd.concat([example_data, synthetic_data], ignore_index=True)

df.to_csv(OUTPUT, index=False)

print("\nDataset created successfully!")
print("Dataset location:", OUTPUT)
print("Total records:", len(df))

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset statistics:")
print(df.describe())
