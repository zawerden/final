from __future__ import annotations
import os
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression

@st.cache_resource
def load_serialized_model(path: str) -> LinearRegression:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл моделі не знайдено: {path}")
    return load(path)

@st.cache_data
def load_historical_datasets(x_path: str, y_path: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Файли даних відсутні.")
    return load(x_path), load(y_path)

def _index_of_closest(x_data: NDArray[np.float64], target_value: float) -> int:
    return int(np.argmin(np.abs(np.ravel(x_data) - target_value)))

def render_analysis_plot(
    x_data: NDArray[np.float64], 
    y_data: NDArray[np.float64], 
    input_feature: float, 
    prediction: float, 
    actual_x: float, 
    actual_y: float
) -> Figure:
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    
    ax.scatter(x_data, y_data, color="lightblue", label="Фактичні значення", alpha=0.6)
    ax.scatter(input_feature, prediction, color="red", s=120, label="Прогноз", zorder=5)
    ax.scatter(actual_x, actual_y, color="green", s=120, label="Найближче фактичне значення", zorder=5)
    ax.plot([input_feature, actual_x], [prediction, actual_y], color="gray", linestyle="--")
    
    ax.set_title("Фактичні значення та прогнози")
    ax.set_xlabel("Ознака X")
    ax.set_ylabel("Ціль y")
    ax.legend()
    ax.grid(True)
    return fig

def run_streamlit_app() -> None:
    st.title("Прогноз лінійної регресії")
    st.write("Цей додаток використовує навчену модель для прогнозування та візуалізації.")
    
    try:
        x_data, y_data = load_historical_datasets("X.joblib", "y.joblib")
        model = load_serialized_model("linear_regression_model.joblib")
        min_val, max_val = float(x_data.min()), float(x_data.max())
    except FileNotFoundError as err:
        st.error(f"🚨 Помилка ініціалізації: {err}")
        st.info("Будь ласка, спочатку запустіть скрипт навчання: `python model.py`")
        return

    input_value = st.slider("Оберіть значення ознаки X:", min_value=min_val, max_value=max_val, value=0.0)

    if st.button("Зробити прогноз", type="primary"):
        try:
            x_input = np.array([[input_value]], dtype=np.float64)
            prediction_array = model.predict(x_input)
            pred_value = float(np.ravel(prediction_array)[0])
            
            st.success(f"Прогнозоване значення y: **{pred_value:.2f}**")
            
            closest_idx = _index_of_closest(x_data, input_value)
            actual_x = float(x_data[closest_idx])
            actual_y = float(y_data[closest_idx])
            
            fig = render_analysis_plot(x_data, y_data, input_value, pred_value, actual_x, actual_y)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Помилка обчислення: {e}")

if __name__ == '__main__':
    run_streamlit_app()