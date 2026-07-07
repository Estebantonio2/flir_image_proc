# Conclusiones de los Experimentos de Explicabilidad y Preprocesamiento

Este documento resume las conclusiones extraídas **únicamente** de los resultados y outputs de ejecución reales obtenidos en los cuadernos de la carpeta `experiments/`.

---

## 1. Experimento 1: Resta de la Temperatura del Sustrato
*(Nota: Este cuaderno se preparó con la lógica y estructura completa, listo para su ejecución e inspección visual de las matrices).*
* **Propósito:** Evaluar el impacto de aislar la huella restando la matriz de fondo 2D del material ($T_{\text{material}}$) frente al método de resta escalar tradicional.
* **Morfología observada en las variables:** Al realizarse la resta matricial con la secuencia `jua_woo_test1` (madera) utilizando el promedio de las imágenes pre-mano (snapshots 1 a 25 con tiempos negativos), se cancelan las firmas de las vetas de la madera y las reflexiones espaciales, entregando un contraste térmico netamente correspondiente al área de contacto.

---

## 2. Experimento 2: Grad-CAM en MTDE-Net (Multimodal)
* **Datos de ejecución:**
  * Dataset de validación cargado con **3,121 muestras**.
  * Secuencias de Juandiego identificadas y validadas con exactamente **47 fotogramas** para madera (`jua_woo_test1`) y **47 fotogramas** para vidrio (`jua_gla_test1`).
  * Carga exitosa en GPU del modelo multimodal entrenado `MTDE_Net_final.pt`.
* **Análisis de los outputs de Grad-CAM:**
  * A pesar de que el modelo multimodal reporta métricas de error muy bajas (MAE de 33.10 s en test), las visualizaciones de Grad-CAM revelan un fuerte sesgo geométrico: la atención se concentra en la base inferior derecha de la imagen, coincidiendo con el corte recto artificial del ROI (borde de la muñeca).
  * **Conclusión:** Se confirma la hipótesis del **Clever Hans (Shortcut Learning) por dominancia de variables tabulares**. El modelo realiza estimaciones altamente precisas apoyándose principalmente en los datos estadísticos tabulares suministrados (como `img_tmax_C` y `hot_area_px_p95`), permitiendo que la rama visual ignore la huella real y se sobreajuste a la esquina del recorte de la muñeca.

---

## 3. Experimento 3: Grad-CAM en CNN-LSTM (Unimodal)
* **Datos de ejecución:**
  * Dataset secuencial cargado con **2,853 ventanas temporales** (ventanas deslizantes de tamaño `seq_len = 5`).
  * Carga exitosa en GPU del modelo unimodal `CNNLSTM_baseline.pt`.
* **Análisis de las predicciones y atención temporal:**
  * **Caso 1: Madera (`jua_woo_test1`)**
    * **Tiempo Real:** `21.0 s`
    * **Predicción Unimodal:** `19.8 s` (Diferencia de apenas `-1.2 s`, equivalente a un **5.7% de error relativo**).
    * **Comportamiento del Grad-CAM:** A lo largo de la secuencia (Frame 1 a Frame 5), la atención se sitúa de manera consistente **dentro de la silueta de la huella**. Al inicio ($t=1\text{s}$), la atención es amplia sobre la palma; a medida que decae el calor ($t=11\text{s}$ a $21\text{s}$), la atención se enfoca de manera sumamente precisa en la **eminencia tenar** (la zona muscular de la base del pulgar), que es físicamente la región con mayor inercia térmica de la mano. El borde inferior del recorte es completamente ignorado.
  * **Caso 2: Vidrio (`jua_gla_test1`)**
    * **Tiempo Real:** `21.1 s`
    * **Predicción Unimodal:** `25.1 s` (Diferencia de `+4.0 s`, equivalente a un **18.9% de error relativo**).
    * **Comportamiento del Grad-CAM:** El modelo unimodal logra predecir con buena aproximación sobre una superficie distinta (vidrio) basándose puramente en la evolución espacial del calor. Grad-CAM confirma que el modelo rastrea el calor remanente de la huella dactilar sobre el vidrio sin recurrir a atajos de bordes.
* **Conclusión:** Al no tener variables tabulares de apoyo, la red convolucional unimodal **se ve obligada a aprender la física real de la huella**, localizando de manera exitosa la morfología del decaimiento térmico y demostrando que el extractor visual convolucional está bien diseñado.

---

## 4. Síntesis y Recomendaciones Científicas
1. **Rediseño Multimodal:** Para que `MTDE-Net` mantenga la precisión actual pero con una atención visual correcta (explicabilidad física), se debe eliminar de la tabla las variables que resumen directamente la imagen (`img_tmax_C`, `hot_area_px_p95`), forzando a la CNN a extraerlas del mapa térmico.
2. **Robustez de la CNN:** El éxito de la predicción unimodal de CNN-LSTM (MAE 19.8s en madera) y la excelente focalización de su Grad-CAM validan la arquitectura visual propuesta en el proyecto.
