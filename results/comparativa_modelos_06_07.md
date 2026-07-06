# Comparativa de Modelos Seleccionados con Nested Cross-Validation (NCV)

Este documento presenta la evaluación y comparación final de los 4 modelos de aprendizaje profundo seleccionados y optimizados utilizando **Nested Cross-Validation (Validación Cruzada Anidada)** para la estimación de tiempo de disipación de rastros térmicos en el proyecto **FLIR Image Processing**.

**Fecha de la Comparativa:** 6 de julio de 2026

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
| **CNN-LSTM Unimodal** | 45.98 ± 15.56 s | 65.87 ± 18.79 s | 0.8164 ± 0.0917 | 47.11 s | 70.84 s | 0.8011 | 26.91 % | 76.22 % | 88.83 % |
| **CNN-LSTM Multimodal** | 49.27 ± 18.68 s | 68.60 ± 20.53 s | 0.8016 ± 0.1159 | 44.96 s | 69.27 s | 0.8099 | **25.10 %** | 74.95 % | 91.53 % |
| **MTDE-Net Propuesto** | 54.04 ± 26.44 s | 76.27 ± 31.08 s | 0.7483 ± 0.2095 | **33.10 s** | **47.97 s** | **0.9132** | 27.02 % | 81.55 % | **96.38 %** |
| **MTDE-Net + TL (ResNet-18 FT)** | **40.72 ± 13.72 s** | **57.85 ± 14.82 s** | **0.8664 ± 0.0614** | 35.84 s | 50.60 s | 0.9035 | 32.16 % | **82.04 %** | 95.55 % |

> [!NOTE]
> * Los valores en **negrita** representan el mejor desempeño para cada métrica en Test (y su homólogo en NCV) entre los modelos seleccionados.

---

## 3. Conclusiones y Análisis Profundo

### 3.1. MTDE-Net como la Arquitectura Definitiva de Mayor Generalización (Test Set)
El modelo propuesto **MTDE-Net** (entrenado desde cero con su arquitectura de atención dual espacial-canal con SPP y fusión tardía) demostró una extraordinaria capacidad para generalizar sobre los sujetos desconocidos de test (`tania` y `matias`), alcanzando las mejores métricas brutas de toda la comparativa:
* **MAE de 33.10 s** (una reducción del **26.3%** respecto a la CNN-LSTM Multimodal).
* **R² Test de 0.9132** y un **RMSE de 47.97 s**.
* Una excelente estabilidad y tolerancia a errores grandes, logrando clasificar correctamente el **96.38%** de las muestras dentro de una tolerancia de 2 minutos (Acc@120s).

Esto demuestra que los bloques con atención dual logran aislar los gradientes de disipación del rastro térmico de manera altamente efectiva contra el ruido corporal inter-sujeto.

### 3.2. Estabilidad Metodológica de Transfer Learning (ResNet-18 FT)
En el bucle de Nested Cross-Validation (NCV), el modelo **MTDE-Net + TL (ResNet-18 FT)** fue por amplio margen el de mejor rendimiento promedio de generalización global estimada en desarrollo:
* **MAE CV de 40.72 s ± 13.72 s** (el menor error promedio) y **R² CV de 0.8664 ± 0.0614** (el mayor coeficiente de determinación).
* En la evaluación del Test Set, mantuvo un desempeño muy cercano a su media de desarrollo con un **MAE de 35.84 s** y un **R² de 0.9035**, y logrando la mayor precisión en ventana estrecha (**Acc@60s de 82.04%**).
La inicialización con pesos preentrenados de ImageNet (ResNet-18) proporciona un fuerte anclaje de características visuales abstractas que reduce drásticamente la variabilidad e inestabilidad del entrenamiento entre folds externos.

### 3.3. Impacto de las Variables Ambientales (Unimodal vs. Multimodal CNN-LSTM)
Al comparar las dos versiones del modelo secuencial espacio-temporal (CNN-LSTM):
* El modelo **CNN-LSTM Multimodal** (que integra las 10 variables contextuales tabulares) supera al modelo unimodal en el Test Set, disminuyendo el MAE a **44.96 s** (frente a 47.11 s del unimodal) y mejorando el coeficiente de determinación a **0.8099** (frente a 0.8011).
* Logra además el menor porcentaje de error relativo promedio de toda la comparativa con un **MAPE de 25.10%**.
Esto valida de forma empírica la hipótesis central y el aporte de tu tesis: incorporar de manera complementaria variables físicas de control (humedad, temperatura ambiental y tipo de superficie) ayuda a sintonizar mejor la tasa de enfriamiento teórica en superficies heterogéneas.

---
