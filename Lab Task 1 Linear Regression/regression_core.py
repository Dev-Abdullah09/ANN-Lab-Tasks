"""
regression_core.py
------------------
Shared code used by the training script, the Gradio app and the Flask app.

Two different regression datasets are used:
  1. Diabetes     -> predict disease progression after one year (scikit-learn)
  2. Auto MPG     -> predict a car's fuel efficiency in miles per gallon (vega_datasets)

For each dataset we fit a plain Linear Regression model and expose a
predict() helper so that every interface uses exactly the same model.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from vega_datasets import data as vega_data

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_diabetes_df():
    """Diabetes dataset: 442 patients, 10 baseline features, 1 target."""
    raw = load_diabetes(as_frame=True)
    df = raw.frame.rename(columns={"target": "progression"})
    return df


def load_cars_df():
    """Auto MPG dataset: cleaned and reduced to numeric columns only."""
    df = vega_data.cars()
    df = df.rename(
        columns={
            "Miles_per_Gallon": "MPG",
            "Weight_in_lbs": "Weight",
        }
    )
    df["ModelYear"] = pd.to_datetime(df["Year"]).dt.year
    df = df[["Cylinders", "Displacement", "Horsepower", "Weight",
             "Acceleration", "ModelYear", "MPG"]]
    df = df.dropna().reset_index(drop=True)  # remove rows with missing values
    return df


# ---------------------------------------------------------------------------
# 2. Dataset registry (name, target, friendly description)
# ---------------------------------------------------------------------------
DATASETS = {
    "diabetes": {
        "title": "Diabetes Progression",
        "loader": load_diabetes_df,
        "target": "progression",
        "target_label": "Disease progression score (higher = worse)",
        "description": (
            "442 diabetes patients. Ten baseline measurements (age, sex, BMI, "
            "blood pressure and six blood-serum values, all mean-centred and "
            "scaled by scikit-learn) are used to predict a quantitative "
            "measure of disease progression one year later."
        ),
    },
    "cars": {
        "title": "Auto MPG (Fuel Efficiency)",
        "loader": load_cars_df,
        "target": "MPG",
        "target_label": "Fuel efficiency (miles per gallon)",
        "description": (
            "Specifications of cars built between 1970 and 1982. Cylinders, "
            "engine displacement, horsepower, weight, acceleration and model "
            "year are used to predict fuel efficiency in miles per gallon."
        ),
    },
}


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
def train(name):
    """Train a LinearRegression model for the dataset `name` and return a
    dictionary holding the model, the split data, metrics and feature ranges."""
    cfg = DATASETS[name]
    df = cfg["loader"]()
    target = cfg["target"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "train_r2": r2_score(y_train, train_pred),
        "test_r2": r2_score(y_test, test_pred),
        "mae": mean_absolute_error(y_test, test_pred),
        "mse": mean_squared_error(y_test, test_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
    }

    # min / max / mean of every feature (used to build input sliders)
    ranges = {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
        }
        for col in X.columns
    }

    return {
        "name": name,
        "config": cfg,
        "df": df,
        "features": list(X.columns),
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "test_pred": test_pred,
        "metrics": metrics,
        "ranges": ranges,
    }


_CACHE = {}


def get_bundle(name):
    """Train once and reuse (so the web apps start quickly)."""
    if name not in _CACHE:
        _CACHE[name] = train(name)
    return _CACHE[name]


def predict(name, values):
    """Predict the target for one sample.

    `values` is a list or dict holding one number per feature, in the same
    order as get_bundle(name)["features"].
    """
    bundle = get_bundle(name)
    feats = bundle["features"]
    if isinstance(values, dict):
        values = [values[f] for f in feats]
    row = pd.DataFrame([values], columns=feats)
    return float(bundle["model"].predict(row)[0])
