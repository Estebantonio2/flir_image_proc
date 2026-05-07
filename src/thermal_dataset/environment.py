from pathlib import Path

import numpy as np
import pandas as pd


def find_environment_excels(test_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    excel_files: list[Path] = []

    for ext in extensions:
        excel_files.extend(test_dir.glob(f"*{ext}"))

    excel_files = [
        file for file in excel_files
        if not file.name.startswith("~$")
    ]

    return sorted(excel_files)


def read_environment_excel(
    excel_path: Path,
    sheet_name: str = "List",
) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    df.columns = [str(col).strip() for col in df.columns]

    time_col = _find_column(df, ["Time", "time", "Date/Time", "Datetime"])
    temp_col = _find_column(
        df,
        ["Temperature°C", "Temperature °C", "Temperature", "Temp°C", "Temp"],
    )
    rh_col = _find_column(
        df,
        ["Humidity%RH", "Humidity %RH", "Humidity", "RH", "RH%"],
    )

    if time_col is None or temp_col is None or rh_col is None:
        raise ValueError(
            "No se pudieron identificar las columnas requeridas del Excel. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    out = df[[time_col, temp_col, rh_col]].copy()
    out = out.rename(
        columns={
            time_col: "env_datetime",
            temp_col: "ambient_temp_C",
            rh_col: "ambient_rh_pct",
        }
    )

    out["env_datetime"] = pd.to_datetime(out["env_datetime"])
    out["ambient_temp_C"] = pd.to_numeric(out["ambient_temp_C"], errors="coerce")
    out["ambient_rh_pct"] = pd.to_numeric(out["ambient_rh_pct"], errors="coerce")

    out = out.dropna(
        subset=["env_datetime", "ambient_temp_C", "ambient_rh_pct"]
    )

    out = out.sort_values("env_datetime").reset_index(drop=True)

    return out


def get_environment_at_time(
    env_df: pd.DataFrame | None,
    capture_datetime,
    use_interpolation: bool = True,
) -> dict:
    if env_df is None or env_df.empty:
        return {
            "ambient_temp_C": np.nan,
            "ambient_rh_pct": np.nan,
            "env_match_method": "missing_excel",
            "env_time_diff_s": np.nan,
        }

    capture_ts = pd.Timestamp(capture_datetime)
    env_times = env_df["env_datetime"]

    if use_interpolation:
        return _interpolate_environment(env_df, capture_ts)

    return _nearest_environment(env_df, capture_ts)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def _nearest_environment(
    env_df: pd.DataFrame,
    capture_ts: pd.Timestamp,
) -> dict:
    env_times = env_df["env_datetime"]
    idx = (env_times - capture_ts).abs().idxmin()
    row = env_df.loc[idx]

    diff_s = abs((row["env_datetime"] - capture_ts).total_seconds()) # type: ignore

    return {
        "ambient_temp_C": float(row["ambient_temp_C"]), # type: ignore
        "ambient_rh_pct": float(row["ambient_rh_pct"]), # type: ignore
        "env_match_method": "nearest",
        "env_time_diff_s": float(diff_s),
    }


def _interpolate_environment(
    env_df: pd.DataFrame,
    capture_ts: pd.Timestamp,
) -> dict:
    env_times = env_df["env_datetime"]

    # Evitamos extrapolación fuera del rango medido.
    if capture_ts <= env_times.iloc[0] or capture_ts >= env_times.iloc[-1]:
        nearest = _nearest_environment(env_df, capture_ts)
        nearest["env_match_method"] = "nearest_outside_range"
        return nearest

    temp_series = env_df.set_index("env_datetime")["ambient_temp_C"]
    rh_series = env_df.set_index("env_datetime")["ambient_rh_pct"]

    combined_index = temp_series.index.union(pd.DatetimeIndex([capture_ts]))

    temp_interp = (
        temp_series
        .reindex(combined_index)
        .sort_index()
        .interpolate(method="time")
        .loc[capture_ts]
    )

    rh_interp = (
        rh_series
        .reindex(combined_index)
        .sort_index()
        .interpolate(method="time")
        .loc[capture_ts]
    )

    nearest_idx = (env_times - capture_ts).abs().idxmin()
    nearest_time = env_df.loc[nearest_idx, "env_datetime"]
    diff_s = abs((nearest_time - capture_ts).total_seconds()) # type: ignore

    return {
        "ambient_temp_C": float(temp_interp),
        "ambient_rh_pct": float(rh_interp),
        "env_match_method": "linear_interpolation",
        "env_time_diff_s": float(diff_s),
    }