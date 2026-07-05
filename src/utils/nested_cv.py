import pandas as pd

def get_inner_split(train_subjects: list[str], fold_externo_idx: int) -> tuple[list[str], list[str]]:
    """
    Divide de forma determinista y reproducible la lista de sujetos de entrenamiento de un fold externo
    en: 7 sujetos para entrenamiento interno y 1 sujeto para validación interna (Hold-out).
    
    Args:
        train_subjects: Lista de sujetos del fold externo.
        fold_externo_idx: Índice del fold externo actual (0-indexed).
        
    Returns:
        tuple: (lista_train_interno, lista_val_interno)
    """
    sorted_subs = sorted(train_subjects)
    val_idx = fold_externo_idx % len(sorted_subs)
    val_subject = sorted_subs[val_idx]
    train_subs = [s for s in sorted_subs if s != val_subject]
    return train_subs, [val_subject]


class NCVTracker:
    """
    Clase para registrar, promediar y dar formato a los resultados de cada fold
    en la validación cruzada anidada (Nested CV).
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.folds_results = []
        
    def add_fold_result(self, fold_idx: int, best_params: dict, metrics: dict):
        """
        Registra las métricas obtenidas en el conjunto de prueba (sujeto excluido)
        para un fold externo determinado.
        
        Args:
            fold_idx: Índice del fold (0-indexed).
            best_params: Diccionario con los mejores parámetros encontrados en la búsqueda.
            metrics: Diccionario con llaves 'mae', 'rmse', 'r2', 'mape', 'acc_60', 'acc_120'
        """
        result = {
            "Fold": fold_idx + 1,
            **metrics,
            "Mejores Parámetros": str(best_params)
        }
        self.folds_results.append(result)
        
    def get_summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.folds_results)
        
    def print_summary(self):
        df = self.get_summary_dataframe()
        print(f"\n=================== RESULTADOS DE NESTED CV PARA {self.model_name.upper()} ===================")
        print(df.to_markdown(index=False))
        
        # Columnas de métricas a promediar
        metric_cols = ["mae", "rmse", "r2", "mape", "acc_60", "acc_120"]
        # Filtrar las columnas que estén disponibles en el dataframe
        available_cols = [col for col in df.columns if col.lower() in metric_cols]
        
        summary_rows = []
        for col in available_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # Formatear la salida según el tipo de métrica
            if col.lower() == "r2":
                expr = f"{mean_val:.4f} ± {std_val:.4f}"
            elif col.lower() in ["mape", "acc_60", "acc_120"]:
                # Si las métricas están en porcentaje, agregar '%'
                if mean_val <= 1.0: # asumimos formato ratio, convertir a porcentaje
                    expr = f"{mean_val*100:.2f}% ± {std_val*100:.2f}%"
                else:
                    expr = f"{mean_val:.2f}% ± {std_val:.2f}%"
            else:
                expr = f"{mean_val:.2f} s ± {std_val:.2f} s"
                
            summary_rows.append({
                "Métrica": col.upper(),
                "Media ± Desviación Estándar (Outer CV)": expr
            })
            
        summary_df = pd.DataFrame(summary_rows)
        print("\n=== RESUMEN GLOBAL DE GENERALIZACIÓN (Media ± Std) ===")
        print(summary_df.to_markdown(index=False))
        print("=========================================================================\n")
        return summary_df
