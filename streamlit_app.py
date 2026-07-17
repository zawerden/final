from __future__ import annotations
import os
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike, NDArray
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression

@st.cache_resource
def load_serialized_model(filename: str = "linear_regression_model.joblib") -> LinearRegression:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл моделі '{filename}' не знайдено.")
    return load(filename)

@st.cache_data
def load_historical_datasets(x_path: str = "X.joblib", y_path: str = "y.joblib") -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Файли базових датасетів X або y відсутні.")
    return load(x_path), load(y_path)

def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> NDArray[np.float64]:
    model = load_serialized_model(filename)
    y = model.predict(np.asarray(X))
    return np.asarray(y)

def visualize_difference(input_feature: float, prediction: float) -> None:
    try:
        X_data, y_data = load_historical_datasets("X.joblib", "y.joblib")
    except FileNotFoundError as e:
        st.error(f"Помилка візуалізації: {e}")
        return

    closest_idx = _index_of_closest(X_data, input_feature)
    actual_x = float(np.ravel(X_data)[closest_idx])
    actual_target = float(np.ravel(y_data)[closest_idx])
    difference = actual_target - prediction

    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()

    ax.scatter(X_data, y_data, color="lightgray", alpha=0.7, label="Dataset (Distribution)")
    ax.scatter(actual_x, actual_target, color="blue", s=130, zorder=4, label="Actual Target (Closest)")
    ax.scatter(input_feature, prediction, color="red", s=130, zorder=5, label="Predicted Target")
    ax.plot([input_feature, actual_x], [prediction, actual_target], "k--", lw=1.5, label="Difference Delta")

    mid_x = (input_feature + actual_x) / 2
    mid_y = (prediction + actual_target) / 2
    ax.annotate(
        f"Delta: {difference:.2f}",
        xy=(mid_x, mid_y),
        xytext=(mid_x + 0.2, mid_y + 5),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=10,
        fontweight="bold"
    )

    ax.set_title("Аналіз відхилень: Фактичні значення vs Прогноз")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.legend(loc="upper left")
    ax.grid(True)

    st.pyplot(fig)

def _index_of_closest(X: ArrayLike, k: float) -> int:
    X_arr = np.asarray(X)
    idx = (np.abs(X_arr - k)).argmin()
    return int(idx)

def create_streamlit_app() -> None:
    st.title("Прогноз лінійної регресії")
    st.write("Вебдодаток для інтерактивного інференсу моделі та розрахунку відхилень точок.")

    input_feature = st.slider("Оберіть значення ознаки X:", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)

    if st.button("Predict value", type="primary"):
        try:
            x_matrix = np.array([[input_feature]], dtype=np.float64)
            prediction_raw = load_and_predict(x_matrix)
            prediction_scalar = float(np.ravel(prediction_raw)[0])
            
            st.success(f"🎯 Прогнозоване значення Target (y): **{prediction_scalar:.4f}**")
            visualize_difference(input_feature, prediction_scalar)
            
        except FileNotFoundError:
            st.error("🚨 Помилка: Файл моделі `linear_regression_model.joblib` не знайдено.")
        except Exception as e:
            st.error(f"Внутрішня помилка виконання інференсу: {e}")

if __name__ == '__main__':
    create_streamlit_app()