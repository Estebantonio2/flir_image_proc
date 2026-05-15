from pathlib import Path

import cv2
import pandas as pd
from pandas.errors import EmptyDataError

from .config import DatasetConfig
from .environment import (
    find_environment_excels,
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
from .roi import RoiBox, crop_array_to_roi, crop_image_to_roi, select_or_confirm_roi
from .thermal_extractor import extract_thermal_array


def build_clean_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict] = []
    all_warnings: list[dict] = []

    metadata_path = config.output_root / config.metadata_filename
    warnings_path = config.output_root / config.warnings_filename

    if metadata_path.exists() and not config.overwrite_existing:
        print(f"Cargando metadata existente desde {metadata_path}...")
        all_rows = _read_csv_records_if_not_empty(metadata_path)

    if warnings_path.exists() and not config.overwrite_existing:
        all_warnings = _read_csv_records_if_not_empty(warnings_path)

    person_dirs = list_test_dirs(config.raw_root)
    if config.target_person_surface is not None:
        person_dirs = [
            person_dir for person_dir in person_dirs
            if person_dir.name == config.target_person_surface
        ]
        if not person_dirs:
            raise FileNotFoundError(
                f"No existe la carpeta solicitada en raw_data: {config.target_person_surface}"
            )

    for person_dir in person_dirs:
        person_name_raw = person_dir.name
        parts = person_name_raw.split("_")
        person = parts[0]
        surface = parts[1] if len(parts) > 1 else "unknown"

        person_surface_dirname = f"{person}_{surface}"
        output_dir = config.output_root / person_surface_dirname
        is_target_person = config.target_person_surface == person_surface_dirname
        
        if output_dir.exists() and not config.overwrite_existing and not is_target_person:
            print(f"La carpeta '{person_surface_dirname}' ya existe en processed_data. Omitiendo procesamiento...")
            continue
            
        print(f"Procesando carpeta: '{person_surface_dirname}'...")
        create_output_dirs(output_dir)
        
        test_dirs = list_test_dirs(person_dir)
        if config.target_test_num is not None:
            if config.target_test_num < 1 or config.target_test_num > len(test_dirs):
                raise ValueError(
                    f"La secuencia {config.target_test_num} no existe en {person_surface_dirname}. "
                    f"Secuencias disponibles: 1-{len(test_dirs)}"
                )
            test_dirs = [test_dirs[config.target_test_num - 1]]

        for i, test_dir in enumerate(test_dirs, 1):
            test_num = config.target_test_num if config.target_test_num is not None else i
            rows, warnings = process_sequence_folder(
                test_dir, person_dir, person, surface, test_num, config
            )
            all_rows.extend(rows)
            all_warnings.extend(warnings)

    # Eliminar posibles filas duplicadas antes de crear el CSV
    if config.target_person_surface is not None and config.target_test_num is not None:
        parts = config.target_person_surface.split("_")
        person = parts[0]
        surface = parts[1] if len(parts) > 1 else "unknown"
        target_sequence_id = f"{person[:3]}_{surface[:3]}_test{config.target_test_num}"
        all_rows = [
            row for row in all_rows
            if row.get("sequence_id") != target_sequence_id
            or row.get("_newly_processed") is True
        ]
        all_warnings = [
            warning for warning in all_warnings
            if warning.get("sequence_id") != target_sequence_id
        ]

    for row in all_rows:
        row.pop("_newly_processed", None)

    metadata = pd.DataFrame(all_rows).drop_duplicates(subset=["name", "surface", "sample_id"], keep="last").reset_index(drop=True)
    warnings_df = pd.DataFrame(all_warnings).drop_duplicates().reset_index(drop=True)

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
    person_dir: Path,
    person: str,
    surface: str,
    test_num: int,
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

    env_df = _load_environment_if_available(person_dir, sequence_id, config, warnings)

    start_datetime = parse_capture_datetime(image_paths[0])
    person_surface_dirname = f"{person}_{surface}"
    new_sequence_id = f"{person[:3]}_{surface[:3]}_test{test_num}"
    roi = None

    if config.use_roi:
        try:
            roi, _ = select_or_confirm_roi(
                image_path=image_paths[0],
                manual=config.manual_roi,
                center_ratio=config.roi_detection_center_ratio,
            )
            print(
                f"ROI {new_sequence_id}: "
                f"x1={roi.x1}, y1={roi.y1}, x2={roi.x2}, y2={roi.y2}"
            )
        except Exception as exc:
            warnings.append(
                {
                    "sequence_id": sequence_id,
                    "file": str(image_paths[0]),
                    "warning": f"No se pudo definir ROI: {exc}",
                }
            )
            return rows, warnings

    for image_path in image_paths:
        try:
            row = process_single_image(
                image_path=image_path,
                sequence_id=new_sequence_id,
                person_surface_dirname=person_surface_dirname,
                person=person,
                surface=surface,
                start_datetime=start_datetime,
                env_df=env_df,
                config=config,
                roi=roi,
            )
            row["_newly_processed"] = True
            rows.append(row)
            _append_environment_timing_warning_if_needed(
                warnings=warnings,
                row=row,
                image_path=image_path,
                config=config,
            )

        except Exception as exc:
            warnings.append(
                {
                    "sequence_id": sequence_id,
                    "file": str(image_path),
                    "warning": str(exc),
                }
            )

    return rows, warnings


def _append_environment_timing_warning_if_needed(
    warnings: list[dict],
    row: dict,
    image_path: Path,
    config: DatasetConfig,
) -> None:
    env_time_diff_s = row.get("env_time_diff_s")

    if pd.isna(env_time_diff_s):
        return

    env_time_diff_s = float(env_time_diff_s)
    if env_time_diff_s < config.max_environment_time_diff_s:
        return

    warnings.append(
        {
            "sequence_id": row["sequence_id"],
            "file": str(image_path),
            "warning": (
                "No se encontró una medición ambiental a menos de "
                f"{config.max_environment_time_diff_s:g} segundos. "
                f"Fila ambiental más cercana: {env_time_diff_s:.1f} segundos."
            ),
        }
    )


def process_single_image(
    image_path: Path,
    sequence_id: str,
    person_surface_dirname: str,
    person: str,
    surface: str,
    start_datetime,
    env_df: pd.DataFrame | None,
    config: DatasetConfig,
    roi: RoiBox | None = None,
) -> dict:
    snapshot_number = parse_snapshot_number(image_path)
    capture_datetime = parse_capture_datetime(image_path)
    elapsed_seconds = compute_elapsed_seconds(capture_datetime, start_datetime)

    environment = get_environment_at_time(
        env_df=env_df,
        capture_datetime=capture_datetime,
        use_interpolation=config.use_interpolation,
    )

    sample_id = make_sample_id(
        sequence_id=sequence_id,
        snapshot_number=snapshot_number,
        elapsed_seconds=elapsed_seconds,
    )

    image_relpath = Path(person_surface_dirname) / config.raw_jpg_dirname / f"{sample_id}.jpg"
    thermal_relpath = Path(person_surface_dirname) / config.thermal_dirname / f"{sample_id}_thermal.npy"
    delta_t_relpath = Path(person_surface_dirname) / config.delta_t_dirname / f"{sample_id}_deltaT.npy"

    if config.copy_raw_jpg:
        copy_jpg(image_path, config.output_root / image_relpath)

    thermal = extract_thermal_array(image_path)
    delta_t = compute_delta_t(
        thermal=thermal,
        ambient_temp_C=environment["ambient_temp_C"],
    )

    if config.use_roi:
        if roi is None:
            raise ValueError("config.use_roi=True, pero no se recibió ROI.")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo leer imagen para aplicar ROI: {image_path}")

        source_image_shape = image.shape[:2]
        thermal = crop_array_to_roi(
            array=thermal,
            roi=roi,
            source_image_shape=source_image_shape,
        )
        delta_t = crop_array_to_roi(
            array=delta_t,
            roi=roi,
            source_image_shape=source_image_shape,
        )

        if config.copy_raw_jpg:
            image_roi = crop_image_to_roi(
                image=image,
                roi=roi,
            )
            cv2.imwrite(str(config.output_root / image_relpath), image_roi)

    features = compute_basic_thermal_features(
        thermal=thermal,
        delta_t=delta_t,
    )

    thermal_abspath = config.output_root / thermal_relpath
    delta_t_abspath = config.output_root / delta_t_relpath

    save_npy(thermal_abspath, thermal)
    save_npy(delta_t_abspath, delta_t)

    row = {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "snapshot_number": snapshot_number,
        "name": person,
        "surface": surface,
        "hand": "derecha",

        "source_image_path": str(image_path),
        "image_path": image_relpath.as_posix() if config.copy_raw_jpg else str(image_path),
        "thermal_path": thermal_relpath.as_posix(),
        "deltaT_path": delta_t_relpath.as_posix(),

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

    if roi is not None:
        row.update(
            {
                "roi_x1": roi.x1,
                "roi_y1": roi.y1,
                "roi_x2": roi.x2,
                "roi_y2": roi.y2,
            }
        )

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
    search_dir: Path,
    sequence_id: str,
    config: DatasetConfig,
    warnings: list[dict],
) -> pd.DataFrame | None:
    excel_paths = find_environment_excels(
        test_dir=search_dir,
        extensions=config.excel_extensions,
    )

    if not excel_paths:
        warnings.append(
            {
                "sequence_id": sequence_id,
                "file": "",
                "warning": "No se encontró Excel ambiental.",
            }
        )
        return None

    dfs = []
    for excel_path in excel_paths:
        try:
            df = read_environment_excel(
                excel_path=excel_path,
                sheet_name=config.excel_sheet_name,
            )
            dfs.append(df)
        except Exception as exc:
            warnings.append(
                {
                    "sequence_id": sequence_id,
                    "file": str(excel_path),
                    "warning": f"No se pudo leer el Excel ambiental: {exc}",
                }
            )

    if not dfs:
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["env_datetime"]).sort_values("env_datetime").reset_index(drop=True)
    
    return combined_df


def _read_csv_records_if_not_empty(path: Path) -> list[dict]:
    try:
        return pd.read_csv(path).to_dict("records")
    except EmptyDataError:
        return []
