import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_learning_curves(history: dict[str, list[float]]):
    """
    Genera gráficos académicos de las curvas de aprendizaje:
    1. Pérdida (Loss) de entrenamiento vs validación.
    2. MAE (Error Absoluto Medio) en segundos de entrenamiento vs validación.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Configurar estilo estético académico premium
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Gráfico de Pérdida (Loss)
    ax1.plot(epochs, history["train_loss"], label="Entrenamiento (Loss)", color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
    ax1.plot(epochs, history["val_loss"], label="Validación (Loss)", color="#ff7f0e", linewidth=2.5, marker="s", markersize=4)
    ax1.set_title("Curva de Pérdida de Aprendizaje", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Épocas", fontsize=11, labelpad=8)
    ax1.set_ylabel("Pérdida (Loss)", fontsize=11, labelpad=8)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # 2. Gráfico de MAE (segundos)
    ax2.plot(epochs, history["train_mae"], label="Entrenamiento (MAE)", color="#2ca02c", linewidth=2.5, marker="o", markersize=4)
    ax2.plot(epochs, history["val_mae"], label="Validación (MAE)", color="#d62728", linewidth=2.5, marker="s", markersize=4)
    ax2.set_title("Evolución del Error Absoluto Medio (MAE)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Épocas", fontsize=11, labelpad=8)
    ax2.set_ylabel("MAE (segundos)", fontsize=11, labelpad=8)
    ax2.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_prediction_calibration(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Modelo"):
    """
    Genera un diagrama de dispersión de predicción vs valor real (calibración).
    Incluye la línea diagonal de predicción perfecta (y = x) y un cuadro de métricas del modelo.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calcular métricas básicas para mostrar en el gráfico
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.0
    
    # Estilo del gráfico
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.figure(figsize=(7, 6))
    
    # Gráfico de dispersión
    plt.scatter(y_true, y_pred, color="#17becf", alpha=0.55, edgecolors="none", s=35, label="Predicciones")
    
    # Línea diagonal de referencia
    max_val = max(y_true.max(), y_pred.max()) + 10
    min_val = min(y_true.min(), y_pred.min()) - 10
    plt.plot([min_val, max_val], [min_val, max_val], color="#7f7f7f", linestyle="--", linewidth=2.0, label="Predicción Perfecta ($y = x$)")
    
    plt.title(f"Calibración de Predicción - {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Tiempo Real ($t\\_seconds$)", fontsize=11, labelpad=8)
    plt.ylabel("Tiempo Predicho (segundos)", fontsize=11, labelpad=8)
    
    # Cuadro de texto de métricas
    textstr = "\n".join((
        f"MAE: {mae:.2f} s",
        f"RMSE: {rmse:.2f} s",
        f"$R^2$: {r2:.4f}"
    ))
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#d3d3d3")
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment="top", bbox=props)
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc="lower right", frameon=True, facecolor="white")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_error_by_time_window(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Modelo"):
    """
    Muestra la distribución del error absoluto medio (MAE) agrupado por ventanas de tiempo reales.
    Esto permite identificar la ventana de confiabilidad del modelo termográfico.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    abs_err = np.abs(y_true - y_pred)
    
    # Definir ventanas de tiempo reales (segundos)
    bins = [0, 60, 120, 180, float("inf")]
    labels = ["0-60s\n(Inicial)", "60-120s\n(Medio)", "120-180s\n(Tardío)", "180s+\n(Límite Físico)"]
    
    # Agrupar datos en contenedores
    indices = np.digitize(y_true, bins) - 1
    
    window_maes = []
    window_counts = []
    
    for i in range(len(labels)):
        mask = (indices == i)
        if mask.any():
            window_maes.append(abs_err[mask].mean())
            window_counts.append(mask.sum())
        else:
            window_maes.append(0.0)
            window_counts.append(0)
            
    # Estilo del gráfico
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.figure(figsize=(8, 5))
    
    colors = ["#2ca02c", "#bcbd22", "#ff7f0e", "#d62728"]
    bars = plt.bar(labels, window_maes, color=colors, alpha=0.8, edgecolor="none", width=0.55)
    
    plt.title(f"Error MAE por Ventana Temporal - {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("MAE (segundos)", fontsize=11, labelpad=8)
    plt.xlabel("Ventana Cronológica Real", fontsize=11, labelpad=8)
    
    # Colocar valores numéricos sobre cada barra
    for bar, count in zip(bars, window_counts):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 1.5, f"{height:.2f}s\n(n={count})", 
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
            
    plt.ylim(0, max(window_maes) + 12)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()
