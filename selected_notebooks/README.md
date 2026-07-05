# Modelos Seleccionados y Validación Rigurosa (Nested CV)

Esta carpeta contiene los cuadernos y scripts definitivos utilizados para la optimización de hiperparámetros y la evaluación final mediante **Nested Cross-Validation (Validación Cruzada Anidada)** sobre los 4 modelos seleccionados:

1. **CNN-LSTM (Baseline)**: Arquitectura base híbrida de la literatura para procesar secuencias temporales térmicas.
2. **Multimodal CNN-LSTM (Baseline Multimodal)**: Extensión del baseline que integra características temporales térmicas con metadatos clínicos del paciente.
3. **MTDE-Net (Optimizado - Propuesta Base)**: Arquitectura propuesta para la estimación de tiempo de rastro térmico, entrenada desde cero y optimizada con Optuna.
4. **MTDE-Net + Transfer Learning (ResNet-18 Fine-tuned - Propuesta + TL)**: Arquitectura propuesta combinada con transferencia de aprendizaje (ResNet-18 con fine-tuning completo).

## Estructura del Proceso de Validación
* **Bucle Externo:** 9-Fold Cross-Validation (Leave-One-Subject-Out) para evaluar el desempeño de generalización sin sesgo sobre los 9 sujetos completos.
* **Bucle Interno (Optimización):** División simple de validación (Hold-out: 7 sujetos para entrenamiento, 1 sujeto para validación) para evaluar la calidad de los hiperparámetros propuestos por `Optuna` en cada fold externo.
* **Held-out Test Set (Test Fijo):** Evaluación final del modelo óptimo entrenado sobre los sujetos reservados (`tania`, `matias`, `eduardo`).
