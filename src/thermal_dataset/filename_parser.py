from datetime import datetime
from pathlib import Path
import re


_FILENAME_PATTERN = re.compile(
    r"^(?P<snapshot>\d+)__"
    r"(?P<date>\d{2}-\d{2}-\d{4})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})-(?P<millis>\d+)$"
)


def parse_snapshot_number(path: Path) -> int:
    stem = path.stem
    match = _FILENAME_PATTERN.match(stem)

    if not match:
        raise ValueError(f"Nombre de archivo no válido: {path.name}")

    return int(match.group("snapshot"))


def parse_capture_datetime(path: Path) -> datetime:
    stem = path.stem
    match = _FILENAME_PATTERN.match(stem)

    if not match:
        raise ValueError(f"Nombre de archivo no válido: {path.name}")

    date_str = match.group("date")
    hour = match.group("hour")
    minute = match.group("minute")
    second = match.group("second")
    millis = match.group("millis")

    microseconds = int(millis.ljust(6, "0")[:6])

    base_dt = datetime.strptime(
        f"{date_str} {hour}:{minute}:{second}",
        "%d-%m-%Y %H:%M:%S",
    )

    return base_dt.replace(microsecond=microseconds)


def compute_elapsed_seconds(capture_dt: datetime, start_dt: datetime) -> float:
    return float((capture_dt - start_dt).total_seconds())