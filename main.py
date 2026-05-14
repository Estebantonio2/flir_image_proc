import argparse
from pathlib import Path

from src.thermal_dataset import DatasetConfig, build_clean_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Procesa imágenes FLIR y deja thermal/deltaT recortados al ROI.",
    )
    parser.add_argument(
        "person_surface",
        nargs="?",
        help="Carpeta a reprocesar, por ejemplo: esteban_madera.",
    )
    parser.add_argument(
        "test_num",
        nargs="?",
        type=int,
        help="Número de secuencia dentro de esa carpeta, por ejemplo: 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        raw_root=Path("raw_data"),
        output_root=Path("processed_data"),
        copy_raw_jpg=True,
        use_interpolation=True,
        overwrite_existing=False,
        use_roi=True,
        manual_roi=True,
        target_person_surface=args.person_surface,
        target_test_num=args.test_num,
    )

    if args.test_num is not None and args.person_surface is None:
        raise ValueError("Para indicar una secuencia también debes indicar la carpeta, por ejemplo: esteban_madera 1")

    metadata, warnings_df = build_clean_dataset(config)

    print("Preprocesamiento terminado.")
    print(f"Muestras procesadas: {len(metadata)}")
    print(f"Advertencias: {len(warnings_df)}")
    print("Salida térmica/deltaT: ROI con tamaño original del recorte")
    print(f"Metadata: {config.output_root / config.metadata_filename}")
    print(f"Warnings: {config.output_root / config.warnings_filename}")


if __name__ == "__main__":
    main()
