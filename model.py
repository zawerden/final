from __future__ import annotations
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from numpy.typing import NDArray
from joblib import dump

def train_regression_model(x_train: NDArray[np.float64], y_train: NDArray[np.float64]) -> LinearRegression:
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_regression_model(
    model: LinearRegression, 
    x_test: NDArray[np.float64], 
    y_test: NDArray[np.float64]
) -> tuple[float, float]:
    predictions = model.predict(x_test)
    mse = float(mean_squared_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    return mse, r2

def save_artifacts(model: LinearRegression, x_data: NDArray[np.float64], y_data: NDArray[np.float64]) -> None:
    dump(model, "linear_regression_model.joblib")
    dump(x_data, "X.joblib")
    dump(y_data, "y.joblib")

def main() -> None:
    x_raw, y_raw = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    x_scaled = np.interp(x_raw, (x_raw.min(), x_raw.max()), (-3, 3))
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_raw, test_size=0.2, random_state=42)
    
    model = train_regression_model(x_train, y_train)
    _ = evaluate_regression_model(model, x_test, y_test)
    save_artifacts(model, x_scaled, y_raw)

if __name__ == '__main__':
    main()