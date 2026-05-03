import numpy as np


def compute_delta_t(
    thermal: np.ndarray,
    ambient_temp_C: float,
) -> np.ndarray:
    if np.isnan(ambient_temp_C):
        return np.full_like(thermal, np.nan, dtype=np.float32)

    return (thermal - np.float32(ambient_temp_C)).astype(np.float32)


def compute_basic_thermal_features(
    thermal: np.ndarray,
    delta_t: np.ndarray,
) -> dict:
    features = {
        "img_tmin_C": _safe_nanmin(thermal),
        "img_tmean_C": _safe_nanmean(thermal),
        "img_tmax_C": _safe_nanmax(thermal),
        "img_tstd_C": _safe_nanstd(thermal),
    }

    features.update(
        {
            "delta_tmin_C": _safe_nanmin(delta_t),
            "delta_tmean_C": _safe_nanmean(delta_t),
            "delta_tmax_C": _safe_nanmax(delta_t),
            "delta_tstd_C": _safe_nanstd(delta_t),
        }
    )

    hot_features = compute_hot_region_features(thermal, delta_t)
    features.update(hot_features)

    return features


def compute_hot_region_features(
    thermal: np.ndarray,
    delta_t: np.ndarray,
    percentile: float = 95,
) -> dict:
    if np.all(np.isnan(delta_t)):
        return {
            "hot_threshold_deltaT_C": np.nan,
            "hot_area_px_p95": np.nan,
            "hot_tmean_C_p95": np.nan,
            "hot_delta_tmean_C_p95": np.nan,
        }

    threshold = float(np.nanpercentile(delta_t, percentile))
    hot_mask = delta_t >= threshold

    return {
        "hot_threshold_deltaT_C": threshold,
        "hot_area_px_p95": int(np.sum(hot_mask)),
        "hot_tmean_C_p95": _safe_nanmean(thermal[hot_mask]),
        "hot_delta_tmean_C_p95": _safe_nanmean(delta_t[hot_mask]),
    }


def _safe_nanmin(array: np.ndarray) -> float:
    if array.size == 0 or np.all(np.isnan(array)):
        return np.nan

    return float(np.nanmin(array))


def _safe_nanmean(array: np.ndarray) -> float:
    if array.size == 0 or np.all(np.isnan(array)):
        return np.nan

    return float(np.nanmean(array))


def _safe_nanmax(array: np.ndarray) -> float:
    if array.size == 0 or np.all(np.isnan(array)):
        return np.nan

    return float(np.nanmax(array))


def _safe_nanstd(array: np.ndarray) -> float:
    if array.size == 0 or np.all(np.isnan(array)):
        return np.nan

    return float(np.nanstd(array))