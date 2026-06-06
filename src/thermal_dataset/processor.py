from pathlib import Path
from dataclasses import dataclass

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
from .naming import translate_token
from .roi import RoiBox, crop_array_to_roi, crop_image_to_roi, select_or_confirm_roi
from .thermal_extractor import extract_thermal_array


PRE_HAND_SEQUENCE_PREFIX = "test_pre_mano"
PRE_HAND_BASELINE_TIMES_S = list(range(-130, -9, 5))
TEN_MINUTE_CAPTURE_TIMES_S = (
    list(range(1, 62, 5))
    + list(range(71, 302, 10))
    + list(range(331, 602, 30))
)
PRE_HAND_CAPTURE_TIMES_S = PRE_HAND_BASELINE_TIMES_S + TEN_MINUTE_CAPTURE_TIMES_S


@dataclass(frozen=True)
class RawPersonSurface:
    person: str
    raw_person: str
    gender: str
    raw_surface: str
    surface: str
    is_complete: bool
    path: Path

    @property
    def dirname(self) -> str:
        return f"{self.person}_{self.surface}"

    @property
    def raw_dirname(self) -> str:
        return f"{self.raw_person}_{self.raw_surface}"


def discover_person_surface_dirs(raw_root: Path) -> list[RawPersonSurface]:
    """Discover raw data in persona/superficie and legacy persona_superficie layouts."""
    sessions: list[RawPersonSurface] = []

    for first_level_dir in list_test_dirs(raw_root):
        nested_sessions = _discover_nested_person_surface_dirs(first_level_dir)
        if nested_sessions:
            sessions.extend(nested_sessions)
            continue

        legacy_session = _parse_legacy_person_surface_dir(first_level_dir)
        if legacy_session is not None:
            sessions.append(legacy_session)

    return sorted(
        sessions,
        key=lambda session: (
            session.person,
            session.surface,
            session.raw_surface,
            session.path.as_posix(),
        ),
    )


def _discover_nested_person_surface_dirs(person_dir: Path) -> list[RawPersonSurface]:
    raw_person = person_dir.name.strip().lower()
    person, gender = _parse_person_and_gender(raw_person)
    sessions: list[RawPersonSurface] = []

    for surface_dir in list_test_dirs(person_dir):
        raw_surface = surface_dir.name.strip().lower()
        surface_name, is_complete = _parse_surface_and_completeness(raw_surface)
        if not _looks_like_person_surface_data_dir(surface_dir):
            continue

        sessions.append(
            RawPersonSurface(
                person=person,
                raw_person=raw_person,
                gender=gender,
                raw_surface=raw_surface,
                surface=translate_token(surface_name),
                is_complete=is_complete,
                path=surface_dir,
            )
        )

    return sessions


def _parse_legacy_person_surface_dir(person_surface_dir: Path) -> RawPersonSurface | None:
    dirname = person_surface_dir.name.strip().lower()
    parts = dirname.split("_")
    if len(parts) < 2:
        return None

    if not _looks_like_person_surface_data_dir(person_surface_dir):
        return None

    if parts[0] in {"0", "1"} and len(parts) >= 3:
        raw_person = "_".join(parts[:2])
        raw_surface = "_".join(parts[2:])
    else:
        raw_person = parts[0]
        raw_surface = "_".join(parts[1:])

    person, gender = _parse_person_and_gender(raw_person)
    surface_name, is_complete = _parse_surface_and_completeness(raw_surface)
    return RawPersonSurface(
        person=person,
        raw_person=raw_person,
        gender=gender,
        raw_surface=raw_surface,
        surface=translate_token(surface_name),
        is_complete=is_complete,
        path=person_surface_dir,
    )


def _parse_person_and_gender(raw_person: str) -> tuple[str, str]:
    parts = raw_person.split("_", maxsplit=1)
    if len(parts) == 2 and parts[0] == "0":
        return parts[1], "male"

    if len(parts) == 2 and parts[0] == "1":
        return parts[1], "female"

    return raw_person, "unknown"


def _parse_surface_and_completeness(raw_surface: str) -> tuple[str, bool]:
    if raw_surface.startswith("x_"):
        return raw_surface[2:], False

    return raw_surface, True


def _looks_like_person_surface_data_dir(path: Path) -> bool:
    return any(child.is_dir() and child.name.startswith("test_") for child in path.iterdir())


def _list_sequence_dirs(person_surface_dir: Path) -> list[Path]:
    return sorted(
        path for path in list_test_dirs(person_surface_dir)
        if path.name.startswith("test_")
    )


def _matches_target_person_surface(
    person_surface: RawPersonSurface,
    target: str,
) -> bool:
    normalized_target = target.strip().lower().replace("\\", "/")
    if not normalized_target:
        return False

    target_parts = [part for part in normalized_target.split("/") if part]

    if len(target_parts) == 2:
        target_person, target_surface = target_parts
        target_surface, _ = _parse_surface_and_completeness(target_surface)
        translated_surface = translate_token(target_surface)
        return (
            target_person in {person_surface.person, person_surface.raw_person}
            and person_surface.surface == translated_surface
        )

    aliases = {
        person_surface.person,
        person_surface.raw_person,
        person_surface.path.name.lower(),
        person_surface.dirname.lower(),
        person_surface.raw_dirname.lower(),
        f"{person_surface.person}/{person_surface.raw_surface}",
        f"{person_surface.person}/{person_surface.surface}",
        f"{person_surface.raw_person}/{person_surface.raw_surface}",
        f"{person_surface.raw_person}/{person_surface.surface}",
    }

    return normalized_target in aliases


def build_clean_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict] = []
    all_warnings: list[dict] = []

    metadata_full_path = config.output_root / config.metadata_full_filename
    metadata_path = config.output_root / config.metadata_filename
    warnings_path = config.output_root / config.warnings_filename

    if not config.overwrite_existing:
        if metadata_full_path.exists():
            print(f"Cargando metadata existente desde {metadata_full_path}...")
            all_rows = _read_csv_records_if_not_empty(metadata_full_path)
        elif metadata_path.exists():
            print(f"Cargando metadata existente desde {metadata_path}...")
            all_rows = _read_csv_records_if_not_empty(metadata_path)

    if warnings_path.exists() and not config.overwrite_existing:
        all_warnings = _read_csv_records_if_not_empty(warnings_path)

    person_surface_dirs = discover_person_surface_dirs(config.raw_root)
    if config.target_person_surface is not None:
        person_surface_dirs = [
            person_surface for person_surface in person_surface_dirs
            if _matches_target_person_surface(
                person_surface=person_surface,
                target=config.target_person_surface,
            )
        ]
        if not person_surface_dirs:
            raise FileNotFoundError(
                f"No existe la carpeta solicitada en raw_data: {config.target_person_surface}"
            )

    for person_surface in person_surface_dirs:
        person = person_surface.person
        gender = person_surface.gender
        surface = person_surface.surface
        is_complete = person_surface.is_complete
        person_surface_dirname = person_surface.dirname
        output_dir = config.output_root / person_surface_dirname
        is_target_person = (
            config.target_person_surface is not None
            and _matches_target_person_surface(person_surface, config.target_person_surface)
        )
        
        if output_dir.exists() and not config.overwrite_existing and not is_target_person:
            print(f"La carpeta '{person_surface_dirname}' ya existe en processed_data. Omitiendo procesamiento...")
            continue
            
        print(f"Procesando carpeta: '{person_surface_dirname}'...")
        create_output_dirs(output_dir, save_delta_t_npy=config.save_delta_t_npy)
        
        test_dirs = _list_sequence_dirs(person_surface.path)
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
                test_dir, person_surface.path, person, gender, surface, is_complete, test_num, config
            )
            all_rows.extend(rows)
            all_warnings.extend(warnings)

    # Eliminar posibles filas duplicadas antes de crear el CSV
    processed_seq_ids = {row.get("sequence_id") for row in all_rows if row.get("_newly_processed")}
    
    clean_rows = []
    for row in all_rows:
        seq_id = row.get("sequence_id")
        if seq_id in processed_seq_ids and not row.get("_newly_processed"):
            continue
        row.pop("_newly_processed", None)
        clean_rows.append(row)
        
    all_rows = clean_rows

    metadata = pd.DataFrame(all_rows)
    if not metadata.empty:
        metadata = metadata.drop_duplicates(subset=["sequence_id", "snapshot_number"], keep="last").reset_index(drop=True)
        
    warnings_df = pd.DataFrame(all_warnings)
    if not warnings_df.empty:
        warnings_df = warnings_df.drop_duplicates().reset_index(drop=True)

    if not metadata.empty:
        metadata = metadata.sort_values(
            ["sequence_id", "snapshot_number"]
        ).reset_index(drop=True)

    metadata_full_path = config.output_root / config.metadata_full_filename
    metadata_train_path = config.output_root / config.metadata_train_filename
    warnings_path = config.output_root / config.warnings_filename

    # Guardar metadata_full (todas las columnas de trazabilidad)
    try:
        metadata.to_csv(metadata_full_path, index=False)
    except PermissionError as e:
        print(f"Error de permisos: No se pudo guardar '{metadata_full_path}'. ¿Está abierto en otra aplicación (ej. Excel)? {e}")
        raise

    # Crear y guardar versión reducida para entrenamiento (metadata_train)
    train_columns = [
        "sequence_id",
        "name",
        "gender",
        "surface",
        "is_complete",
        "thermal_path",
        "t_seconds",
        "ambient_temp_C",
        "ambient_rh_pct",
        "img_tmax_C",
        "img_tstd_C",
        "delta_tmean_C",
        "delta_tstd_C",
        "hot_area_px_p95",
        "hot_delta_tmean_C_p95",
    ]
    metadata_train = metadata.reindex(columns=train_columns)
    try:
        metadata_train.to_csv(metadata_train_path, index=False)
    except PermissionError as e:
        print(f"Error de permisos: No se pudo guardar '{metadata_train_path}'. ¿Está abierto en otra aplicación (ej. Excel)? {e}")
        raise

    try:
        warnings_df.to_csv(warnings_path, index=False)
    except PermissionError as e:
        print(f"Advertencia: No se pudo escribir en '{warnings_path}' porque está bloqueado por otro proceso.")

    return metadata, warnings_df


def process_sequence_folder(
    sequence_dir: Path,
    person_dir: Path,
    person: str,
    gender: str,
    surface: str,
    is_complete: bool,
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
            roi_image_path = _select_roi_reference_image(image_paths, sequence_dir)
            roi, _ = select_or_confirm_roi(
                image_path=roi_image_path,
                manual=config.manual_roi,
                center_ratio=config.roi_detection_center_ratio,
            )
            print(
                f"ROI {new_sequence_id}: "
                f"ref={roi_image_path.name}, "
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

    is_pre_hand = sequence_dir.name.startswith(PRE_HAND_SEQUENCE_PREFIX)

    for image_path in image_paths:
        try:
            snapshot_number = parse_snapshot_number(image_path)
            
            # Determinar tiempo programado basado en el número de snapshot, no en el índice de archivo
            scheduled_elapsed_seconds = None
            if is_pre_hand:
                idx = snapshot_number - 1
                if 0 <= idx < len(PRE_HAND_CAPTURE_TIMES_S):
                    scheduled_elapsed_seconds = float(PRE_HAND_CAPTURE_TIMES_S[idx])
                else:
                    # Si el snapshot excede el cronograma, process_single_image usará compute_elapsed_seconds
                    pass

            row = process_single_image(
                image_path=image_path,
                sequence_id=new_sequence_id,
                person_surface_dirname=person_surface_dirname,
                person=person,
                gender=gender,
                surface=surface,
                is_complete=is_complete,
                start_datetime=start_datetime,
                env_df=env_df,
                config=config,
                roi=roi,
                scheduled_elapsed_seconds=scheduled_elapsed_seconds,
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


def _select_roi_reference_image(
    image_paths: list[Path],
    sequence_dir: Path,
) -> Path:
    if not sequence_dir.name.startswith(PRE_HAND_SEQUENCE_PREFIX):
        return image_paths[0]

    for image_path in image_paths:
        if parse_snapshot_number(image_path) == 27:
            return image_path

    if len(image_paths) >= 27:
        return image_paths[26]

    return image_paths[-1]


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
    gender: str,
    surface: str,
    is_complete: bool,
    start_datetime,
    env_df: pd.DataFrame | None,
    config: DatasetConfig,
    roi: RoiBox | None = None,
    scheduled_elapsed_seconds: float | None = None,
) -> dict:
    snapshot_number = parse_snapshot_number(image_path)
    capture_datetime = parse_capture_datetime(image_path)
    elapsed_seconds = (
        float(scheduled_elapsed_seconds)
        if scheduled_elapsed_seconds is not None
        else compute_elapsed_seconds(capture_datetime, start_datetime)
    )

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
    save_npy(thermal_abspath, thermal)

    if config.save_delta_t_npy:
        delta_t_abspath = config.output_root / delta_t_relpath
        save_npy(delta_t_abspath, delta_t)

    row = {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "snapshot_number": snapshot_number,
        "name": person,
        "gender": gender,
        "surface": surface,
        "is_complete": is_complete,
        "hand": "right",

        "source_image_path": str(image_path),
        "image_path": image_relpath.as_posix() if config.copy_raw_jpg else str(image_path),
        "thermal_path": thermal_relpath.as_posix(),

        "capture_datetime": capture_datetime.isoformat(sep=" "),
        "t_seconds": elapsed_seconds,

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
