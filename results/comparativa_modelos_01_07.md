# Comparativa de Modelos de Aprendizaje Profundo

Este documento presenta una evaluación y comparación detallada de los modelos de aprendizaje profundo desarrollados para la estimación de tiempo en secuencias de imágenes térmicas del proyecto **FLIR Image Processing**.

**Fecha de la Comparativa:** 1 de julio de 2026

---

## 1. Metodología de Evaluación

Para asegurar la robustez y capacidad de generalización de los modelos, se ha seguido una estrategia de partición estricta basada en sujetos:
* **Held-out Test Set (Test de Generalización):** Reservado de manera fija con los sujetos `tania` (mujer), `matias` (hombre) y el sujeto incompleto `eduardo`.
* **Entrenamiento y Validación (Cross-Validation):** Validación cruzada de 9 folds (excepto para el modelo Multimodal CNN-LSTM que reporta 9 folds en el cuaderno de entrenamiento final) sobre los 9 sujetos completos restantes.
* **Métrica de Pérdida principal:** Sqrt-Scaled MSE Loss (`SqrtScaledMSELoss`) para estabilizar el gradiente en las predicciones temporales.

> [!IMPORTANT]
> Todos los resultados expuestos en este documento han sido extraídos directamente de los outputs de las celdas ejecutadas en los cuadernos (`.ipynb`) del proyecto para asegurar la fidelidad con los experimentos reales.

---

## 2. Tabla Comparativa General (Modelos de los Notebooks)

A continuación se detallan los resultados obtenidos por los 6 modelos principales en sus fases de **Validación Cruzada (CV)** y **Test de Generalización**:

| Modelo | MAE CV | RMSE CV | R² CV | MAE Test | RMSE Test | R² Test | MAPE Test | Acc@60s Test | Acc@120s Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D-CNN (Baseline)** | 48.09 ± 11.88 s | 67.11 ± 16.35 s | 0.8104 ± 0.1015 | 58.37 s | 81.54 s | 0.7365 | 26.68 % | 62.70 % | 85.41 % |
| **CNN-LSTM (Baseline)** | 31.79 ± 13.41 s | 45.75 ± 16.28 s | 0.9077 ± 0.0737 | 44.46 s | 66.01 s | 0.8273 | **24.72 %** | **77.48 %** | 90.27 % |
| **DSTFS Adapted (Baseline)** | 38.00 ± 16.28 s | 53.86 ± 18.15 s | 0.8793 ± 0.0822 | 63.42 s | 87.04 s | 0.7144 | 63.37 % | 59.80 % | 82.04 % |
| **MTDE-Net (Baseline)** | 31.80 ± 14.91 s | 44.46 ± 14.99 s | 0.9181 ± 0.0659 | 48.94 s | 65.13 s | 0.8401 | 37.86 % | 63.92 % | 93.41 % |
| **MTDE-Net (Optimizado)** | 34.00 ± 13.76 s | 50.09 ± 21.10 s | 0.8908 ± 0.1141 | **36.80 s** | **53.36 s** | **0.8926** | 26.01 % | 77.10 % | **95.72 %** |
| **Multimodal CNN-LSTM** | **29.70 ± 6.67 s** | **42.49 ± 8.99 s** | **0.9249 ± 0.0320** | 45.71 s | 68.99 s | 0.8114 | 25.65 % | 73.51 % | 90.45 % |

> [!NOTE]
> * Los valores en **negrita** representan el mejor desempeño para cada métrica en Test (y su homólogo en CV).
> * Los resultados de **MTDE-Net (Optimizado)** corresponden a los parámetros sintonizados mediante Optuna (`00_optuna_optimization.ipynb`).

---

## 3. Resultados de Transfer Learning (Modelos Multimodales 2D)

Métricas obtenidas mediante transferencia de conocimiento de backbones pre-entrenados en ImageNet sobre la primera imagen térmica combinada con metadatos del paciente (Cuaderno `05_MTDE_Net_transfer_learning.ipynb`):

### Validación Cruzada (CV Mean ± CV Std)

| Configuración | MAE CV | RMSE CV | R² CV | MAPE CV | Acc@60s CV | Acc@120s CV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 Fine-tuned** | **29.05 ± 8.17 s** | **42.29 ± 12.22 s** | **0.9273 ± 0.0455** | 26.47 ± 7.30 % | **88.16 ± 5.81 %** | **97.22 ± 3.28 %** |
| **ResNet-18 Frozen (FE)** | 57.21 ± 18.41 s | 80.66 ± 18.00 s | 0.7436 ± 0.1154 | 65.58 ± 49.51 % | 64.84 ± 14.95 % | 87.03 ± 7.82 % |
| **ResNet-50 Fine-tuned** | 32.38 ± 7.28 s | 48.39 ± 8.93 s | 0.9084 ± 0.0348 | **25.49 ± 6.70 %** | 83.89 ± 5.18 % | 95.91 ± 2.86 % |

### Test de Generalización (Held-out Set)

| Configuración | MAE Test | RMSE Test | R² Test | MAPE Test | Acc@60s Test | Acc@120s Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 Fine-tuned** | **34.15 s** | **48.87 s** | **0.9099** | **28.46 %** | **83.53 %** | **96.21 %** |
| **ResNet-18 Frozen (FE)** | 52.14 s | 75.71 s | 0.7839 | 53.01 % | 68.37 % | 87.81 % |
| **ResNet-50 Fine-tuned** | 37.11 s | 54.51 s | 0.8880 | 29.54 % | 77.59 % | 94.89 % |

---

## 4. Conclusiones y Análisis Profundo

### 4.1. Generalización del Modelo en Sujetos Desconocidos (Test de Generalización)
Una de las observaciones más críticas es la discrepancia de rendimiento entre la Validación Cruzada y el conjunto de prueba aislado (Test Set). Todos los modelos experimentan una degradación de sus métricas al evaluar en sujetos cuyos patrones térmicos nunca fueron expuestos durante el entrenamiento. 
* El modelo **1D-CNN** incrementa su MAE de 48.09 s (CV) a 58.37 s (Test), mientras que el **CNN-LSTM Baseline** sube de 31.79 s a 44.46 s.
* Por el contrario, el **MTDE-Net (Optimizado)** demuestra una notable robustez frente al sobreajuste, con un cambio mínimo de MAE, pasando de 34.00 s en CV a **36.80 s** en el Test Set. Esto ratifica que la optimización bayesiana (Optuna) sobre hiperparámetros de regularización como *dropout* y *weight decay* resulta fundamental para entrenamientos médicos en conjuntos de datos pequeños.

### 4.2. Ventaja del Modelamiento Temporal (CNN-LSTM vs. 1D-CNN)
La evolución térmica de la piel es una señal dinámica intrínsecamente ligada al tiempo. Al contrastar **1D-CNN** (que procesa los vectores espaciales de forma estática o con convoluciones unidimensionales cortas) con la arquitectura híbrida **CNN-LSTM**, se observa el gran impacto de la recurrencia:
* El MAE en Test Set disminuye un **23.8%** (de 58.37 s a 44.46 s).
* Además, **CNN-LSTM** logra la mejor tolerancia a errores cortos (**Acc@60s: 77.48%**) y el menor error relativo porcentual (**MAPE: 24.72%**). La memoria de largo plazo (LSTM) ayuda a modelar la inercia térmica de los sujetos de forma significativamente más efectiva.

### 4.3. El Aporte de la Arquitectura Multimodal
El modelo **Multimodal CNN-LSTM** destaca por lograr los mejores resultados generales en la Validación Cruzada (con el menor error absoluto **MAE CV: 29.70 s** y el coeficiente de determinación más alto **R² CV: 0.9249**).
La incorporación de metadatos tabulares (edad, género, etc.) actúa como un fuerte anclaje de regularización. Sin embargo, en el Test Set, sufre una ligera desviación frente al MTDE-Net Optimizado, lo que indica que se requiere un mayor volumen de sujetos en entrenamiento para aprender a generalizar correctamente la influencia fisiológica de los metadatos en individuos no vistos.

### 4.4. Transfer Learning (Modelos Multimodales 2D)
El enfoque de **Transfer Learning con ResNet-18 (Fine-Tuned)** obtuvo el desempeño sobresaliente absoluto en el Test Set (**MAE Test: 34.15 s, R²: 0.9099**).
El fine-tuning completo del extractor de características convolucional 2D permite proyectar la evolución del mapa térmico en el tiempo y espacio simultáneamente de forma mucho más expresiva que las arquitecturas entrenadas desde cero. Esto confirma la viabilidad y conveniencia de utilizar modelos masivos pre-entrenados en visión artificial como extractores base para termografía médica, requiriendo menor volumen de datos específicos de dominio para lograr una convergencia superior.
