# Comparativa de Modelos de Aprendizaje Profundo (Resultados Actuales)

Este documento presenta una evaluación y comparación detallada de los modelos de aprendizaje profundo desarrollados para la estimación de tiempo en secuencias de imágenes térmicas del proyecto **FLIR Image Processing**, utilizando únicamente los resultados actuales en los notebooks de la carpeta `notebooks/`.

**Fecha de la Comparativa:** 5 de julio de 2026

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
| **CNN-LSTM (Baseline)** | **31.79 ± 13.41 s** | 45.75 ± 16.28 s | 0.9077 ± 0.0737 | **44.46 s** | 66.01 s | 0.8273 | **24.72 %** | **77.48 %** | 90.27 % |
| **DSTFS Adapted (Baseline)** | 38.00 ± 16.28 s | 53.86 ± 18.15 s | 0.8793 ± 0.0822 | 63.42 s | 87.04 s | 0.7144 | 63.37 % | 59.80 % | 82.04 % |
| **MTDE-Net (Baseline)** | 31.80 ± 14.91 s | **44.46 ± 14.99 s** | **0.9181 ± 0.0659** | 48.94 s | **65.13 s** | **0.8401** | 37.86 % | 63.92 % | **93.41 %** |

> [!NOTE]
> * Los valores en **negrita** representan el mejor desempeño para cada métrica en Test (y su homólogo en CV) entre los modelos baseline evaluados.

---

## 3. Resultados de Transfer Learning (Modelos Multimodales 2D)

Métricas obtenidas mediante transferencia de conocimiento de backbones pre-entrenados en ImageNet sobre la primera imagen térmica combinada con metadatos del paciente (Cuaderno `05_MTDE_Net_transfer_learning.ipynb`):

| Configuración | MAE CV | RMSE CV | R² CV | MAE Test | RMSE Test | R² Test | MAPE Test | Acc@60s Test | Acc@120s Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 Fine-tuned** | **29.56 ± 13.00 s** | **44.84 ± 22.04 s** | 0.9084 ± 0.1095 | **30.88 s** | **44.97 s** | **0.9238** | 32.29 % | **86.49 %** | **97.03 %** |
| **ResNet-18 Frozen (FE)** | 56.19 ± 20.42 s | 81.44 ± 20.58 s | 0.7358 ± 0.1382 | 56.34 s | 80.62 s | 0.7549 | 76.74 % | 68.04 % | 87.81 % |
| **ResNet-50 Fine-tuned** | 30.69 ± 6.26 s | 46.14 ± 8.36 s | **0.9170 ± 0.0312** | 37.09 s | 55.57 s | 0.8836 | **29.22 %** | 78.75 % | 94.56 % |

---

## 4. Conclusiones y Análisis Profundo

### 4.1. Generalización del Modelo en Sujetos Desconocidos (Test de Generalización)
Una de las observaciones más críticas es la discrepancia de rendimiento entre la Validación Cruzada y el conjunto de prueba aislado (Test Set). Todos los modelos experimentan una degradación de sus métricas al evaluar en sujetos cuyos patrones térmicos nunca fueron expuestos durante el entrenamiento.
* El modelo **1D-CNN** incrementa su MAE de 48.09 s (CV) a 58.37 s (Test), mientras que el **CNN-LSTM Baseline** sube de 31.79 s a 44.46 s.
* El modelo propuesto **MTDE-Net (Baseline)** pasa de un MAE de 31.80 s en CV a 48.94 s en Test. Esta degradación del desempeño resalta el desafío de generalizar a nuevas características anatómicas y térmicas de sujetos desconocidos en datasets de tamaño limitado.

### 4.2. Ventaja del Modelamiento Temporal (CNN-LSTM vs. 1D-CNN)
La evolución térmica de la piel es una señal dinámica intrínsecamente ligada al tiempo. Al contrastar **1D-CNN** (que procesa las curvas térmicas de forma estática o con convoluciones unidimensionales) con la arquitectura híbrida **CNN-LSTM**, se observa el gran impacto del modelamiento recurrente:
* El MAE en el Test Set disminuye un **23.8%** (de 58.37 s a 44.46 s).
* Además, **CNN-LSTM** logra una excelente tolerancia a errores cortos (**Acc@60s: 77.48%**) y el menor error relativo porcentual (**MAPE: 24.72%**). La memoria de largo plazo (LSTM) ayuda a capturar la inercia térmica de los sujetos de forma significativamente más efectiva.

### 4.3. El Aporte de la Arquitectura Multimodal (MTDE-Net)
El modelo propuesto **MTDE-Net (Baseline)** destaca por lograr un excelente desempeño en la Validación Cruzada, con un coeficiente de determinación alto (**R² CV: 0.9181**) y el menor error cuadrático medio (**RMSE CV: 44.46 s**).
La incorporación de metadatos tabulares (edad, género, etc.) a través de su arquitectura multimodal actúa como un fuerte anclaje de regularización, permitiendo al modelo aprender relaciones fisiológicas útiles a pesar de la variabilidad inter-sujeto. Sin embargo, en el Test Set, sufre una ligera desviación frente a CNN-LSTM, lo que sugiere que para generalizar de forma óptima a sujetos no vistos, el modelo requiere una regularización temporal más explícita o una optimización fina de hiperparámetros.

### 4.4. Transfer Learning (Modelos Multimodales 2D)
El enfoque de **Transfer Learning con ResNet-18 (Fine-Tuned)** obtuvo el mejor desempeño sobresaliente absoluto en toda la comparativa, alcanzando un **MAE Test de 30.88 s** y un **R² Test de 0.9238**.
El fine-tuning completo del extractor de características convolucional 2D permite proyectar la evolución del mapa térmico en el tiempo y espacio simultáneamente de forma mucho más expresiva que las arquitecturas entrenadas desde cero. Por otro lado, la configuración con backbone congelado (**ResNet-18 Frozen**) muestra un rendimiento drásticamente inferior (MAE Test: 56.34 s), lo que confirma que las representaciones de ImageNet necesitan adaptarse (fine-tuning) a las firmas térmicas específicas de la piel para poder generalizar correctamente en la tarea de estimación de tiempo.
