from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    raw_root: Path = Path("raw_data")
    output_root: Path = Path("processed_data")

    copy_raw_jpg: bool = True
    use_interpolation: bool = True
    max_environment_time_diff_s: float = 60.0
    overwrite_existing: bool = True

    use_roi: bool = True
    manual_roi: bool = True
    roi_detection_center_ratio: float = 0.8
    target_person_surface: str | None = None
    target_test_num: int | None = None

    raw_jpg_dirname: str = "raw_jpg"
    thermal_dirname: str = "thermal_npy"
    delta_t_dirname: str = "deltaT_npy"

    metadata_filename: str = "metadata.csv"
    metadata_full_filename: str = "metadata_full.csv"
    metadata_train_filename: str = "metadata_train.csv"
    warnings_filename: str = "processing_warnings.csv"

    excel_sheet_name: str = "List"

    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".JPG", ".JPEG")
    excel_extensions: tuple[str, ...] = (".xls", ".xlsx")
