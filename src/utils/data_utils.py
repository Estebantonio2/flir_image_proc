import pandas as pd
import numpy as np

def split_by_sequence(df: pd.DataFrame, fraction: float = 0.8, seed: int = 42) -> tuple[list[int], list[int]]:
    """
    Garantiza una separación estricta de secuencias físicas de rastro térmico
    para evitar fuga de información (data leakage) entre entrenamiento y validación.
    """
    seqs = df["sequence_id"].drop_duplicates().to_numpy()
    np.random.default_rng(seed).shuffle(seqs)
    train_seqs = set(seqs[:max(1, int(len(seqs) * fraction))])
    
    train_indices = df.index[df["sequence_id"].isin(train_seqs)].tolist()
    val_indices = df.index[~df["sequence_id"].isin(train_seqs)].tolist()
    
    return train_indices, val_indices
