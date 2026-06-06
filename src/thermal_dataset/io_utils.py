from pathlib import Path
import shutil

import numpy as np


def create_output_dirs(output_root: Path, save_delta_t_npy: bool = True) -> None:
    (output_root / "raw_jpg").mkdir(parents=True, exist_ok=True)
    (output_root / "thermal_npy").mkdir(parents=True, exist_ok=True)
    if save_delta_t_npy:
        (output_root / "deltaT_npy").mkdir(parents=True, exist_ok=True)


def list_test_dirs(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(f"No existe la carpeta raw_root: {raw_root}")

    return sorted([path for path in raw_root.iterdir() if path.is_dir()])


def list_images(test_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    images: set[Path] = set()

    for ext in extensions:
        images.update(path.resolve() for path in test_dir.glob(f"*{ext}"))

    return sorted(images)


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def copy_jpg(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
