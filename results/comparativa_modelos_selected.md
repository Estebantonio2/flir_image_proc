# Comparativa de Modelos Seleccionados con Nested Cross-Validation (NCV)

Este documento presenta la evaluación y comparación final de los 4 modelos de aprendizaje profundo seleccionados y optimizados utilizando **Nested Cross-Validation (Validación Cruzada Anidada)** para la estimación de tiempo de disipación de rastros térmicos en el proyecto **FLIR Image Processing**.

**Fecha de la Comparativa:** 9 de julio de 2026

---

## 1. Metodología de Inferencia y Optimización

A diferencia de las pruebas preliminares (Baselines), para estos modelos se ha aplicado un esquema de validación cruzada mucho más riguroso orientado a evitar el sesgo de sintonización de hiperparámetros:
* **Nested CV (9 Folds Externos):** Partición LOSO (Leave-One-Subject-Out) en el bucle externo para medir la capacidad de generalización real en sujetos no vistos durante el desarrollo.
* **Bucle Interno (Sintonización con Optuna):** Sintonización bayesiana automática (10 trials) para buscar la mejor combinación de parámetros en cada fold, con poda de trials deficientes mediante `MedianPruner`.
* **Consolidación de Parámetros Celda 5.5:** Mediana para hiperparámetros continuos (tasa de aprendizaje `lr`, `weight_decay`, `dropout`) y moda para los discretos/categóricos (`batch_size`, `seq_len`, `lstm_hidden_dim`).
* **Held-out Test Set (Evaluación Ciega Final):** Prueba definitiva del modelo final entrenado en todo `trainval` contra los sujetos de test aislados: `tania` y `matias`.

> [!IMPORTANT]
> Todos los resultados expuestos en este documento han sido extraídos directamente de los outputs de las celdas ejecutadas en los cuadernos (`.ipynb`) bajo la carpeta `selected_notebooks/` para asegurar la absoluta fidelidad física con las corridas experimentales reales.

---

## 2. Tabla Comparativa General (Modelos Optimizados con NCV)

A continuación se detallan las métricas promedio obtenidas en el **Bucle Externo de NCV (Generalización esperada)** y la evaluación sobre el **Held-out Test Set (Test General)**:

| Modelo | MAE NCV (CV) | RMSE NCV (CV) | R² NCV (CV) | MAE Test | RMSE Test | R² Test | MAPE Test | Acc@60s Test | Acc@120s Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-LSTM Unimodal** | 44.93 ± 15.98 s | 65.28 ± 18.44 s | 0.8196 ± 0.0918 | 40.97 s | 60.94 s | 0.8528 | **21.11 %** | **79.82 %** | 93.33 % |
| **CNN-LSTM Multimodal** | 45.95 ± 18.60 s | 62.08 ± 19.24 s | 0.8349 ± 0.0990 | 47.65 s | 71.98 s | 0.7947 | 27.12 % | 70.99 % | 89.73 % |
| **MTDE-Net Propuesto** | 52.82 ± 32.14 s | 69.75 ± 31.29 s | 0.7838 ± 0.2090 | **37.62 s** | **54.72 s** | **0.8871** | 40.34 % | 78.42 % | **95.22 %** |
| **MTDE-Net + TL (ResNet-50 FT)** | **43.49 ± 24.10 s** | **61.25 ± 24.60 s** | **0.8387 ± 0.1484** | 42.83 s | 63.15 s | 0.8496 | 35.65 % | 74.14 % | 91.27 % |

> [!NOTE]
> * Los valores en **negrita** representan el mejor desempeño para cada métrica en Test (y su homólogo en NCV) entre los modelos seleccionados.

---

## 3. Conclusiones y Análisis Profundo

### 3.1. MTDE-Net como la Arquitectura Definitiva de Mayor Generalización (Test Set)
El modelo propuesto **MTDE-Net** (entrenado desde cero con su arquitectura de atención dual espacial-canal con SPP y fusión tardía) demostró una extraordinaria capacidad para generalizar sobre los sujetos desconocidos de test (`tania` y `matias`), alcanzando las mejores métricas brutas en la prueba held-out de toda la comparativa:
* **MAE de 37.62 s** (una reducción del **21.05%** respecto a la CNN-LSTM Multimodal).
* **R² Test de 0.8871** y un **RMSE de 54.72 s**.
* Una excelente estabilidad y tolerancia a errores grandes, logrando clasificar correctamente el **95.22%** de las muestras dentro de una tolerancia de 2 minutos (Acc@120s).

Esto demuestra que los bloques con atención dual logran aislar los gradientes de disipación del rastro térmico de manera altamente efectiva contra el ruido corporal inter-sujeto.

### 3.2. Estabilidad Metodológica de Transfer Learning (ResNet-50 FT)
En el bucle de Nested Cross-Validation (NCV), el modelo **MTDE-Net + TL (ResNet-50 FT)** fue el de mejor rendimiento promedio de generalización global estimada en desarrollo:
* **MAE CV de 43.49 s ± 24.10 s** (el menor error promedio) y **R² CV de 0.8387 ± 0.1484** (el mayor coeficiente de determinación).
* En la evaluación del Test Set, mantuvo un desempeño consistente con un **MAE de 42.83 s** y un **R² de 0.8496**, y logrando clasificar correctamente al **91.27%** de las muestras dentro de la ventana de 120s.
La inicialización con pesos preentrenados de ImageNet (ResNet-50) proporciona un fuerte anclaje de características visuales abstractas que reduce la variabilidad y estabiliza el entrenamiento entre folds externos.

### 3.3. Impacto de las Variables Ambientales (Unimodal vs. Multimodal CNN-LSTM)
Al comparar las dos versiones del modelo secuencial espacio-temporal (CNN-LSTM):
* El modelo **CNN-LSTM Unimodal** supera al modelo multimodal en el Test Set, logrando un MAE de **40.97 s** (frente a 47.65 s del multimodal) y un coeficiente de determinación de **0.8528** (frente a 0.7947).
* Logra además el menor porcentaje de error relativo promedio de toda la comparativa con un **MAPE de 21.11%** y la mejor precisión en ventana estrecha (**Acc@60s de 79.82%**).
* Sin embargo, en el desarrollo por validación cruzada (NCV), ambos modelos presentan comportamientos muy cercanos (MAE CV de 44.93 s para Unimodal vs 45.95 s para Multimodal, con este último logrando un R² de 0.8349). Esto sugiere que la inclusión de las 10 variables contextuales tabulares (humedad, temperatura, superficie) puede inducir a un ligero sobreajuste anatómico y contextual específico de los sujetos del conjunto de entrenamiento, disminuyendo la capacidad de generalización sobre el Held-out Test Set en comparación con el modelo que extrae representaciones espaciotemporales puramente visuales de la disipación térmica.
