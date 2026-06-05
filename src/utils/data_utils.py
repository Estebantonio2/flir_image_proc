import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectCvFold:
    fold: int
    train_subjects: list[str]
    val_subjects: list[str]
    train_indices: list[int]
    val_indices: list[int]


@dataclass(frozen=True)
class SubjectSplitPlan:
    test_subjects: list[str]
    trainval_subjects: list[str]
    test_indices: list[int]
    trainval_indices: list[int]
    folds: list[SubjectCvFold]

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


def build_subject_split_plan(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    test_subjects: list[str] | None = None,
    reserve_incomplete_for_test: bool = True,
) -> SubjectSplitPlan:
    """
    Reserva sujetos para test y crea folds de validación cruzada por sujeto.

    Regla recomendada para este proyecto:
    - Sujetos con is_complete=False se reservan para test.
    - Los sujetos restantes se usan para CV, sin mezclar un mismo sujeto entre
      train y validación.
    """
    _require_columns(df, ["name"])

    normalized = df.copy()
    normalized["name"] = normalized["name"].astype(str).str.strip().str.lower()

    explicit_test_subjects = {
        subject.strip().lower()
        for subject in (test_subjects or [])
        if subject.strip()
    }

    incomplete_subjects: set[str] = set()
    if reserve_incomplete_for_test and "is_complete" in normalized.columns:
        complete_mask = _as_bool_series(normalized["is_complete"])
        incomplete_subjects = set(normalized.loc[~complete_mask, "name"].dropna().unique())

    selected_test_subjects = explicit_test_subjects | incomplete_subjects

    all_subjects = sorted(normalized["name"].dropna().unique())
    trainval_subjects = [
        subject for subject in all_subjects
        if subject not in selected_test_subjects
    ]

    if not trainval_subjects:
        raise ValueError("No quedaron sujetos para train/validación después de reservar test.")

    test_indices = normalized.index[normalized["name"].isin(selected_test_subjects)].tolist()
    trainval_indices = normalized.index[normalized["name"].isin(trainval_subjects)].tolist()
    folds = make_subject_cv_folds(
        normalized.loc[trainval_indices],
        n_splits=n_splits,
        seed=seed,
    )

    return SubjectSplitPlan(
        test_subjects=sorted(selected_test_subjects),
        trainval_subjects=trainval_subjects,
        test_indices=test_indices,
        trainval_indices=trainval_indices,
        folds=folds,
    )


def make_subject_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[SubjectCvFold]:
    """
    Crea K folds por sujeto. Si hay menos sujetos que n_splits, usa leave-one-subject-out.
    """
    _require_columns(df, ["name"])

    normalized = df.copy()
    normalized["name"] = normalized["name"].astype(str).str.strip().str.lower()

    subjects = normalized["name"].dropna().drop_duplicates().to_numpy()
    if len(subjects) < 2:
        raise ValueError("Se necesitan al menos 2 sujetos para crear folds de CV.")

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    effective_splits = min(max(2, n_splits), len(subjects))
    subject_folds = [
        list(map(str, fold_subjects))
        for fold_subjects in np.array_split(subjects, effective_splits)
        if len(fold_subjects) > 0
    ]

    folds: list[SubjectCvFold] = []
    for fold_idx, val_subjects in enumerate(subject_folds, start=1):
        val_subject_set = set(val_subjects)
        train_subjects = [
            str(subject) for subject in subjects
            if str(subject) not in val_subject_set
        ]

        train_indices = normalized.index[normalized["name"].isin(train_subjects)].tolist()
        val_indices = normalized.index[normalized["name"].isin(val_subject_set)].tolist()

        folds.append(
            SubjectCvFold(
                fold=fold_idx,
                train_subjects=sorted(train_subjects),
                val_subjects=sorted(val_subjects),
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    return folds


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resume género, completitud y materiales disponibles por sujeto."""
    _require_columns(df, ["name"])

    normalized = df.copy()
    normalized["name"] = normalized["name"].astype(str).str.strip().str.lower()

    aggregations = {
        "sequence_id": "nunique",
        "surface": lambda values: ", ".join(sorted(map(str, pd.Series(values).dropna().unique()))),
    }

    if "gender" in normalized.columns:
        aggregations["gender"] = lambda values: _first_non_null(values, default="unknown")

    if "is_complete" in normalized.columns:
        normalized["is_complete"] = _as_bool_series(normalized["is_complete"])
        aggregations["is_complete"] = "all"

    summary = normalized.groupby("name").agg(aggregations).reset_index()
    return summary.rename(columns={"sequence_id": "n_sequences"})


def sequence_ids_for_subjects(df: pd.DataFrame, subjects: list[str]) -> list[str]:
    """Devuelve los sequence_id pertenecientes a una lista de sujetos."""
    _require_columns(df, ["name", "sequence_id"])

    normalized = df.copy()
    normalized["name"] = normalized["name"].astype(str).str.strip().str.lower()
    subject_set = {
        subject.strip().lower()
        for subject in subjects
        if subject.strip()
    }

    sequence_ids = (
        normalized.loc[normalized["name"].isin(subject_set), "sequence_id"]
        .dropna()
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    return sequence_ids


def _as_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)

    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "si", "sí"})


def _first_non_null(values, default: str = "") -> str:
    series = pd.Series(values).dropna()
    if series.empty:
        return default

    return str(series.iloc[0])


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
