from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RoiBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return asdict(self)


def suggest_roi_from_image(
    image: np.ndarray,
    padding_ratio: float = 0.1,
    thresh_percent: float = 0.6,
    min_area: float = 50,
    center_ratio: float = 1.0,
) -> RoiBox:
    if image.ndim != 3:
        raise ValueError(f"Se esperaba imagen BGR/RGB con 3 canales. Shape: {image.shape}")

    image_height, image_width = image.shape[:2]

    if not 0 < center_ratio <= 1:
        raise ValueError(f"center_ratio debe estar en (0, 1]. Valor recibido: {center_ratio}")

    crop_x1, crop_y1, crop_x2, crop_y2 = _center_crop_bounds(
        image_width=image_width,
        image_height=image_height,
        center_ratio=center_ratio,
    )

    detection_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    max_val = np.max(gray)
    thresh_val = int(max_val * thresh_percent)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    if not valid_contours:
        raise ValueError("No se detectó una ROI válida en la imagen.")

    all_points = np.vstack(valid_contours)
    x, y, width, height = cv2.boundingRect(all_points)
    x += crop_x1
    y += crop_y1

    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)

    return RoiBox(
        x1=max(x - pad_x, 0),
        y1=max(y - pad_y, 0),
        x2=min(x + width + pad_x, image_width),
        y2=min(y + height + pad_y, image_height),
    )


def select_or_confirm_roi(
    image_path: Path,
    output_size: int = 224,
    manual: bool = True,
    padding_ratio: float = 0.1,
    thresh_percent: float = 0.6,
    center_ratio: float = 1.0,
) -> tuple[RoiBox, np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    suggested_roi = suggest_roi_from_image(
        image=image,
        padding_ratio=padding_ratio,
        thresh_percent=thresh_percent,
        center_ratio=center_ratio,
    )

    final_roi = suggested_roi
    if manual:
        preview = image.copy()
        cv2.rectangle(
            preview,
            (suggested_roi.x1, suggested_roi.y1),
            (suggested_roi.x2, suggested_roi.y2),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            preview,
            "Enter/Space: aceptar | arrastra: ajustar | c: cancelar",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
        )

        selected = cv2.selectROI(
            f"ROI - {image_path.name}",
            preview,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyAllWindows()

        if selected != (0, 0, 0, 0):
            x, y, width, height = [int(value) for value in selected]
            final_roi = RoiBox(x1=x, y1=y, x2=x + width, y2=y + height)

    roi_224 = crop_square_resize_image(image, final_roi, output_size=output_size)
    return final_roi, roi_224


def crop_square_resize_image(
    image: np.ndarray,
    roi: RoiBox,
    output_size: int = 224,
) -> np.ndarray:
    crop = image[roi.y1:roi.y2, roi.x1:roi.x2]
    if crop.size == 0:
        raise ValueError(f"La ROI no contiene pixeles: {roi}")

    height, width = crop.shape[:2]
    size = max(height, width)
    square = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    square[y_offset:y_offset + height, x_offset:x_offset + width] = crop

    return cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def crop_square_resize_array(
    array: np.ndarray,
    roi: RoiBox,
    source_image_shape: tuple[int, int],
    output_size: int = 224,
) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Se esperaba una matriz térmica 2D. Shape: {array.shape}")

    array_roi = scale_roi_to_array(
        roi=roi,
        source_image_shape=source_image_shape,
        array_shape=array.shape,
    )

    crop = array[array_roi.y1:array_roi.y2, array_roi.x1:array_roi.x2]
    if crop.size == 0:
        raise ValueError(f"La ROI escalada no contiene pixeles: {array_roi}")

    height, width = crop.shape
    size = max(height, width)
    pad_top = (size - height) // 2
    pad_bottom = size - height - pad_top
    pad_left = (size - width) // 2
    pad_right = size - width - pad_left

    square = np.pad(
        crop,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="edge",
    )

    resized = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def scale_roi_to_array(
    roi: RoiBox,
    source_image_shape: tuple[int, int],
    array_shape: tuple[int, int],
) -> RoiBox:
    image_height, image_width = source_image_shape
    array_height, array_width = array_shape

    scale_x = array_width / image_width
    scale_y = array_height / image_height

    x1 = int(round(roi.x1 * scale_x))
    y1 = int(round(roi.y1 * scale_y))
    x2 = int(round(roi.x2 * scale_x))
    y2 = int(round(roi.y2 * scale_y))

    return RoiBox(
        x1=max(min(x1, array_width - 1), 0),
        y1=max(min(y1, array_height - 1), 0),
        x2=max(min(x2, array_width), 1),
        y2=max(min(y2, array_height), 1),
    )


def _center_crop_bounds(
    image_width: int,
    image_height: int,
    center_ratio: float,
) -> tuple[int, int, int, int]:
    crop_width = int(round(image_width * center_ratio))
    crop_height = int(round(image_height * center_ratio))
    x1 = (image_width - crop_width) // 2
    y1 = (image_height - crop_height) // 2
    return x1, y1, x1 + crop_width, y1 + crop_height
