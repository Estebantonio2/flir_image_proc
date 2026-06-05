# Plan de evaluacion experimental

## Objetivo

Alinear el entrenamiento y la evaluacion de los modelos con la recomendacion del asesor:

- Reservar sujetos para test antes de cualquier optimizacion.
- Hacer validacion cruzada por sujeto sobre el resto del dataset.
- Ajustar hiperparametros con Optuna sin usar el test.
- Reportar resultados por fold y resultados finales en test.

## 1. Preparar la metadata

Ejecutar el preprocesamiento para regenerar:

```text
processed_data/metadata_full.csv
processed_data/metadata_train.csv
```

La metadata debe contener como minimo:

```text
name
gender
surface
is_complete
sequence_id
t_seconds
```

Interpretacion:

```text
gender:
  male   -> carpeta raw_data/0_nombre/...
  female -> carpeta raw_data/1_nombre/...

is_complete:
  True  -> material sin prefijo x_, por ejemplo madera o vidrio
  False -> material con prefijo x_, por ejemplo x_madera o x_vidrio
```

Los sujetos/materiales con `is_complete=False` no significan secuencias malas. Significan que ese sujeto/material no tiene el set completo esperado de pruebas. Se usaran como test.

## 2. Revisar sujetos disponibles

Antes de entrenar, revisar cuantos sujetos completos e incompletos hay.

```python
import pandas as pd
from src.utils import subject_summary

df = pd.read_csv("../processed_data/metadata_train.csv")

summary = subject_summary(df)
print(summary)
```

Verificar:

- Que haya al menos un hombre y una mujer si se quiere test balanceado por genero.
- Que los sujetos marcados como incompletos sean realmente los que se quieren reservar para test.
- Que los sujetos completos sean suficientes para validacion cruzada.

## 3. Definir test fijo

Crear el plan de particion:

```python
import pandas as pd
from src.utils import build_subject_split_plan

df = pd.read_csv("../processed_data/metadata_train.csv")

plan = build_subject_split_plan(
    df,
    n_splits=5,
    seed=42,
    reserve_incomplete_for_test=True,
)

print("Sujetos test:", plan.test_subjects)
print("Sujetos train/val:", plan.trainval_subjects)
```

Regla:

```text
test        -> sujetos con is_complete=False
train/val   -> sujetos con is_complete=True
```

Importante: ningun sujeto de `plan.test_subjects` debe usarse en Optuna, entrenamiento de folds, seleccion de arquitectura ni decision de hiperparametros.

## 4. Validacion cruzada por sujeto

Usar los folds del plan:

```python
for fold in plan.folds:
    print("Fold:", fold.fold)
    print("Train subjects:", fold.train_subjects)
    print("Val subjects:", fold.val_subjects)
```

Cada fold debe cumplir:

```text
train_subjects ∩ val_subjects = vacio
train_subjects ∩ test_subjects = vacio
val_subjects   ∩ test_subjects = vacio
```

Si hay 5 o mas sujetos completos, se hacen 5 folds.

Si hay menos de 5 sujetos completos, el codigo usa automaticamente el numero posible de folds, equivalente a leave-one-subject-out cuando corresponde.

## 5. Adaptar loaders por fold

Para loaders que aceptan `sequence_ids`, convertir sujetos a secuencias:

```python
from src.utils import sequence_ids_for_subjects

train_seq_ids = sequence_ids_for_subjects(df, fold.train_subjects)
val_seq_ids = sequence_ids_for_subjects(df, fold.val_subjects)
test_seq_ids = sequence_ids_for_subjects(df, plan.test_subjects)
```

Ejemplo con loaders secuenciales:

```python
train_ds = ThermalSequenceDataset(
    metadata_csv="../processed_data/metadata_train.csv",
    is_train=True,
    sequence_ids=train_seq_ids,
)

val_ds = ThermalSequenceDataset(
    metadata_csv="../processed_data/metadata_train.csv",
    is_train=False,
    sequence_ids=val_seq_ids,
)
```

Para loaders que todavia no aceptan `sequence_ids`, agregar soporte equivalente o usar indices/sujetos para filtrar antes de construir el dataset.

## 6. Adaptar Optuna

La funcion objetivo de Optuna debe evaluar cada trial sobre todos los folds.

Estructura recomendada:

```python
def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }

    fold_scores = []

    for fold in plan.folds:
        train_seq_ids = sequence_ids_for_subjects(df, fold.train_subjects)
        val_seq_ids = sequence_ids_for_subjects(df, fold.val_subjects)

        train model using train_seq_ids
        evaluate model using val_seq_ids

        fold_scores.append(val_metric)

    return mean(fold_scores)
```

El test no aparece en esta funcion.

Optuna debe escoger hiperparametros usando solo los folds de validacion.

## 7. Entrenamiento final del modelo campeon

Despues de Optuna:

1. Tomar los mejores hiperparametros.
2. Entrenar un modelo nuevo usando todos los sujetos de `plan.trainval_subjects`.
3. Evaluar una sola vez en `plan.test_subjects`.

Ejemplo conceptual:

```python
best_params = study.best_params

final_train_seq_ids = sequence_ids_for_subjects(df, plan.trainval_subjects)
final_test_seq_ids = sequence_ids_for_subjects(df, plan.test_subjects)

train final model using final_train_seq_ids
evaluate final model using final_test_seq_ids
```

## 8. Reporte de resultados

Reportar por fold:

```text
fold
train_subjects
val_subjects
MAE
RMSE
R2
otras metricas relevantes
```

Reportar resumen:

```text
MAE medio +/- std
RMSE medio +/- std
R2 medio +/- std
```

Reportar test final:

```text
test_subjects
MAE test
RMSE test
R2 test
```

Separar claramente:

```text
Validacion cruzada:
  sirve para seleccion de hiperparametros y estimacion de estabilidad.

Test final:
  sirve como evaluacion final independiente.
```

## 9. Materiales

Por ahora se entrenara con los materiales juntos.

Cuando se integren ambos materiales en un solo modelo, incluir el material como variable de entrada:

```text
glass -> 1
wood  -> 2
```

En el codigo actual `surface` ya existe, por lo que el mapeo puede hacerse asi:

```python
df["material_id"] = df["surface"].map({
    "glass": 1,
    "wood": 2,
})
```

Mas adelante se pueden agregar tres experimentos:

```text
1. Solo vidrio
2. Solo madera
3. Vidrio + madera con material_id como variable
```

## 10. Checklist

- [ ] Regenerar metadata con `gender` e `is_complete`.
- [ ] Revisar `subject_summary(df)`.
- [ ] Confirmar sujetos reservados para test.
- [ ] Cambiar notebooks para usar `build_subject_split_plan`.
- [ ] Reemplazar `split_by_sequence` por folds por sujeto.
- [ ] Adaptar Optuna para promediar resultados por fold.
- [ ] Guardar metricas por fold.
- [ ] Entrenar modelo final con todos los sujetos completos.
- [ ] Evaluar una sola vez en sujetos incompletos reservados para test.
- [ ] Reportar media/std de CV y metricas de test final.
