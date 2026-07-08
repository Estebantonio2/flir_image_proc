# Comparativa de Modelos de Aprendizaje Profundo (Resultados Actuales)

Este documento presenta una evaluación y comparación detallada de los modelos de aprendizaje profundo desarrollados para la estimación de tiempo en secuencias de imágenes térmicas del proyecto **FLIR Image Processing**, utilizando únicamente los resultados actuales en los notebooks de la carpeta `notebooks/`.

**Fecha de la Comparativa:** 8 de julio de 2026

---

## 1. Metodología de Evaluación

Para asegurar la robustez y capacidad de generalización de los modelos, se ha seguido una estrategia de partición estricta basada en sujetos:
* **Held-out Test Set (Test de Generalización):** Reservado de manera fija con los sujetos `tania` (mujer), `matias` (hombre) y el sujeto incompleto `eduardo`.
* **Entrenamiento y Validación (Cross-Validation):** Validación cruzada de 9 folds sobre los 9 sujetos completos restantes.
* **Métrica de Pérdida principal:** Sqrt-Scaled MSE Loss (`SqrtScaledMSELoss`) para estabilizar el gradiente en las predicciones temporales.

> [!IMPORTANT]
> Todos los resultados expuestos en este documento han sido extraídos directamente de los outputs de las celdas ejecutadas en los cuadernos (`.ipynb`) del proyecto para asegurar la fidelidad con los experimentos reales.

---

## 2. Tabla Comparativa General (Modelos de los Notebooks)

A continuación se detallan los resultados obtenidos por los 4 modelos principales entrenados en sus fases de **Validación Cruzada (CV)** y **Test de Generalización**:

| Modelo | MAE CV | RMSE CV | R² CV | MAE Test | RMSE Test | R² Test | MAPE Test | Acc@60s Test | Acc@120s Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D-CNN (Baseline)** | 48.09 ± 11.88 s | 67.11 ± 16.35 s | 0.8104 ± 0.1015 | 58.37 s | 81.54 s | 0.7365 | 26.68 % | 62.70 % | 85.41 % |
| **CNN-LSTM (Baseline)** | 31.79 ± 13.41 s | 45.75 ± 16.28 s | 0.9077 ± 0.0737 | **44.46 s** | **66.01 s** | **0.8273** | **24.72 %** | **77.48 %** | **90.27 %** |
| **DSTFS Adapted (Baseline)** | 38.00 ± 16.28 s | 53.86 ± 18.15 s | 0.8793 ± 0.0822 | 63.42 s | 87.04 s | 0.7144 | 63.37 % | 59.80 % | 82.04 % |
| **MTDE-Net (Baseline)** | **27.48 ± 9.54 s** | **40.69 ± 10.02 s** | **0.9340 ± 0.0322** | 56.02 s | 87.14 s | 0.7137 | 35.13 % | 67.55 % | 87.31 % |

> [!NOTE]
> * Los valores en **negrita** representan el mejor desempeño para cada métrica en Test (y su homólogo en CV) entre los modelos baseline evaluados.

---

## 3. Resultados de Transfer Learning (Modelos Multimodales 2D)

Métricas obtenidas mediante transferencia de conocimiento de backbones pre-entrenados en ImageNet sobre la primera imagen térmica combinada con metadatos del paciente (Cuaderno `05_MTDE_Net_transfer_learning.ipynb`):

| Configuración | MAE CV | RMSE CV | R² CV | MAE Test | RMSE Test | R² Test | MAPE Test | Acc@60s Test | Acc@120s Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 Fine-tuned** | **29.13 ± 9.86 s** | **44.37 ± 17.26 s** | **0.9160 ± 0.0758** | 41.77 s | 62.35 s | 0.8534 | 37.54 % | 75.29 % | 93.08 % |
| **ResNet-18 Frozen (FE)** | 59.80 ± 19.15 s | 85.45 ± 17.29 s | 0.7143 ± 0.1223 | 67.13 s | 93.82 s | 0.6682 | 82.70 % | 57.99 % | 81.55 % |
| **ResNet-50 Fine-tuned** | 31.67 ± 10.34 s | 47.16 ± 14.64 s | 0.9090 ± 0.0630 | **36.60 s** | **54.80 s** | **0.8868** | **32.34 %** | **81.71 %** | **94.07 %** |

---

## 4. Conclusiones y Análisis Profundo

### 4.1. Generalización del Modelo en Sujetos Desconocidos (Test de Generalización)
Una de las observaciones más críticas es la discrepancia de rendimiento entre la Validación Cruzada y el conjunto de prueba aislado (Test Set). Todos los modelos experimentan una degradación de sus métricas al evaluar en sujetos cuyos patrones térmicos nunca fueron expuestos durante el entrenamiento.
* El modelo **1D-CNN** incrementa su MAE de 48.09 s (CV) a 58.37 s (Test), mientras que el **CNN-LSTM Baseline** sube de 31.79 s a 44.46 s.
* El modelo propuesto **MTDE-Net (Baseline)** pasa de un MAE de 27.48 s en CV a 56.02 s en Test. Esta degradación del desempeño resalta el desafío de generalizar a nuevas características anatómicas y térmicas de sujetos desconocidos en datasets de tamaño limitado, especialmente ahora que se han removido variables tabulares que servían como "atajos" artificiales para identificar a los sujetos.

### 4.2. Ventaja del Modelamiento Temporal (CNN-LSTM vs. 1D-CNN)
La evolución térmica de la piel es una señal dinámica intrínsecamente ligada al tiempo. Al contrastar **1D-CNN** (que procesa las curvas térmicas de forma estática o con convoluciones unidimensionales) con la arquitectura híbrida **CNN-LSTM**, se observa el gran impacto del modelamiento recurrente:
* El MAE en el Test Set disminuye un **23.8%** (de 58.37 s a 44.46 s).
* Además, **CNN-LSTM** logra una excelente tolerancia a errores cortos (**Acc@60s: 77.48%**) y el menor error relativo porcentual (**MAPE: 24.72%**). La memoria de largo plazo (LSTM) ayuda a capturar la inercia térmica de los sujetos de forma significativamente más efectiva.

### 4.3. El Aporte de la Arquitectura Multimodal (MTDE-Net)
El modelo propuesto **MTDE-Net (Baseline)** destaca por lograr un excelente desempeño en la Validación Cruzada, con un coeficiente de determinación sumamente alto (**R² CV: 0.9340**) y el menor error cuadrático medio (**RMSE CV: 40.69 s**), superando a todos los demás baselines.
Al reducir las variables ambientales a solo temperatura y humedad (dejando un vector tabular final de 4 dimensiones), hemos evitado que el modelo se apoye en exceso en "atajos" no generalizables. Sin embargo, en el Test Set sufre una degradación (MAE Test de 56.02 s frente a los 44.46 s de CNN-LSTM), lo que demuestra que para generalizar a sujetos no vistos sin depender de atajos tabulares se requiere combinar la fusión multimodal con técnicas temporales (como en CNN-LSTM) o regularizadores más fuertes.

### 4.4. Transfer Learning (Modelos Multimodales 2D)
El enfoque de **Transfer Learning con ResNet-50 (Fine-Tuned)** obtuvo el mejor desempeño absoluto entre los modelos de una sola imagen con metadatos, alcanzando un **MAE Test de 36.60 s** y un **R² Test de 0.8868**.
El fine-tuning completo del extractor de características convolucional 2D permite proyectar la evolución del mapa térmico en el tiempo y espacio simultáneamente de forma mucho más expresiva que las arquitecturas entrenadas desde cero. Por otro lado, la configuración con backbone congelado (**ResNet-18 Frozen**) muestra un rendimiento drásticamente inferior (MAE Test: 67.13 s), lo que confirma que las representaciones de ImageNet necesitan adaptarse (fine-tuning) a las firmas térmicas específicas de la piel para poder generalizar correctamente en la tarea de estimación de tiempo.
