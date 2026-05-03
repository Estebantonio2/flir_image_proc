from pathlib import Path

import numpy as np
from flirimageextractor import FlirImageExtractor


def extract_thermal_array(jpg_path: Path) -> np.ndarray:
    """
    Extrae la matriz térmica radiométrica desde un JPG FLIR.

    Retorna:
        np.ndarray float32 con shape H x W
    """
    flir = FlirImageExtractor()
    flir.process_image(str(jpg_path))

    thermal = flir.get_thermal_np()

    if thermal is None:
        raise ValueError(f"No se pudo extraer matriz térmica desde {jpg_path}")

    thermal = np.asarray(thermal, dtype=np.float32)

    if thermal.ndim != 2:
        raise ValueError(
            f"La matriz térmica debe ser 2D. "
            f"Archivo: {jpg_path.name}. Shape obtenido: {thermal.shape}"
        )

    return thermal