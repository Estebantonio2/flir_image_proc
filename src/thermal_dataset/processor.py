from pathlib import Path

import pandas as pd

from .config import DatasetConfig
from .environment import (
    find_environment_excel,
    read_environment_excel,
    get_environment_at_time,
)
from .features import (
    compute_delta_t,
    compute_basic_thermal_features,
)
from .filename_parser import (
    parse_capture_datetime,
    parse_snapshot_number,
    compute_elapsed_seconds,
)
from .io_utils import (
    create_output_dirs,
    list_test_dirs,
    list_images,
    save_npy,
    copy_jpg,
)
from .thermal_extractor import extract_thermal_array


def build_clean_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    create_output_dirs(config.output_root)

    all_rows: list[dict] = []
    all_warnings: list[dict] = []

    test_dirs = list_test_dirs(config.raw_root)

    for test_dir in test_dirs:
        rows, warnings = process_sequence_folder(test_dir, config)
        all_rows.extend(rows)
        all_warnings.extend(warnings)

    metadata = pd.DataFrame(all_rows)
    warnings_df = pd.DataFrame(all_warnings)

    if not metadata.empty:
        metadata = metadata.sort_values(
            ["sequence_id", "snapshot_number"]
        ).reset_index(drop=True)

    metadata_path = config.output_root / config.metadata_filename
    warnings_path = config.output_root / config.warnings_filename

    metadata.to_csv(metadata_path, index=False)
    warnings_df.to_csv(warnings_path, index=False)

    return metadata, warnings_df


def process_sequence_folder(
    sequence_dir: Path,
    config: DatasetConfig,
) -> tuple[list[dict], list[dict]]:
    sequence_id = sequence_dir.name

    rows: list[dict] = []
    warnings: list[dict] = []

    image_paths = list_images(sequence_dir, config.image_extensions)

    if not image_paths:
        warnings.append(
            {
                "sequence_id": sequence_id,
                "file": "",
                "warning": "No se encontraron imágenes JPG.",
            }
        )
        return rows, warnings

    image_paths = sorted(image_paths, key=parse_snapshot_number)

    env_df = _load_environment_if_available(sequence_dir, config, warnings)

    start_datetime = parse_capture_datetime(image_paths[0])

    for image_path in image_paths:
        try:
            row = process_single_image(
                image_path=image_path,
                sequence_id=sequence_id,
                start_datetime=start_datetime,
                env_df=env_df,
                config=config,
            )
            rows.append(row)

        except Exception as exc:
            warnings.append(
                {
                    "sequence_id": sequence_id,
                    "file": str(image_path),
                    "warning": str(exc),
                }
            )

    return rows, warnings


def process_single_image(
    image_path: Path,
    sequence_id: str,
    start_datetime,
    env_df: pd.DataFrame | None,
    config: DatasetConfig,
) -> dict:
    snapshot_number = parse_snapshot_number(image_path)
    capture_datetime = parse_capture_datetime(image_path)
    elapsed_seconds = compute_elapsed_seconds(capture_datetime, start_datetime)

    environment = get_environment_at_time(
        env_df=env_df,
        capture_datetime=capture_datetime,
        use_interpolation=config.use_interpolation,
    )

    thermal = extract_thermal_array(image_path)
    delta_t = compute_delta_t(
        thermal=thermal,
        ambient_temp_C=environment["ambient_temp_C"],
    )

    features = compute_basic_thermal_features(
        thermal=thermal,
        delta_t=delta_t,
    )

    sample_id = make_sample_id(
        sequence_id=sequence_id,
        snapshot_number=snapshot_number,
        elapsed_seconds=elapsed_seconds,
    )

    image_relpath = Path(config.raw_jpg_dirname) / f"{sample_id}.jpg"
    thermal_relpath = Path(config.thermal_dirname) / f"{sample_id}_thermal.npy"
    delta_t_relpath = Path(config.delta_t_dirname) / f"{sample_id}_deltaT.npy"

    thermal_abspath = config.output_root / thermal_relpath
    delta_t_abspath = config.output_root / delta_t_relpath

    save_npy(thermal_abspath, thermal)
    save_npy(delta_t_abspath, delta_t)

    if config.copy_raw_jpg:
        copy_jpg(image_path, config.output_root / image_relpath)

    row = {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "snapshot_number": snapshot_number,

        "source_image_path": str(image_path),
        "image_path": str(image_relpath) if config.copy_raw_jpg else str(image_path),
        "thermal_path": str(thermal_relpath),
        "deltaT_path": str(delta_t_relpath),

        "capture_datetime": capture_datetime.isoformat(sep=" "),
        "t_seconds": elapsed_seconds,
        "label_time_s": elapsed_seconds,

        "ambient_temp_C": environment["ambient_temp_C"],
        "ambient_rh_pct": environment["ambient_rh_pct"],
        "env_match_method": environment["env_match_method"],
        "env_time_diff_s": environment["env_time_diff_s"],

        "thermal_height": thermal.shape[0],
        "thermal_width": thermal.shape[1],
    }

    row.update(features)

    return row


def make_sample_id(
    sequence_id: str,
    snapshot_number: int,
    elapsed_seconds: float,
) -> str:
    elapsed_int = int(round(elapsed_seconds))

    clean_sequence = (
        sequence_id
        .replace(" ", "_")
        .replace("-", "_")
    )

    return f"{clean_sequence}_snap{snapshot_number:04d}_{elapsed_int:04d}s"


def _load_environment_if_available(
    sequence_dir: Path,
    config: DatasetConfig,
    warnings: list[dict],
) -> pd.DataFrame | None:
    sequence_id = sequence_dir.name

    excel_path = find_environment_excel(
        test_dir=sequence_dir,
        extensions=config.excel_extensions,
    )

    if excel_path is None:
        warnings.append(
            {
                "sequence_id": sequence_id,
                "file": "",
                "warning": "No se encontró Excel ambiental.",
            }
        )
        return None

    try:
        return read_environment_excel(
            excel_path=excel_path,
            sheet_name=config.excel_sheet_name,
        )

    except Exception as exc:
        warnings.append(
            {
                "sequence_id": sequence_id,
                "file": str(excel_path),
                "warning": f"No se pudo leer el Excel ambiental: {exc}",
            }
        )
        return None