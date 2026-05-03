from pathlib import Path

from src.thermal_dataset import DatasetConfig, build_clean_dataset


def main() -> None:
    config = DatasetConfig(
        raw_root=Path("raw_data"),
        output_root=Path("processed_data"),
        copy_raw_jpg=True,
        use_interpolation=True,
    )

    metadata, warnings_df = build_clean_dataset(config)

    print("Preprocesamiento terminado.")
    print(f"Muestras procesadas: {len(metadata)}")
    print(f"Advertencias: {len(warnings_df)}")
    print(f"Metadata: {config.output_root / config.metadata_filename}")
    print(f"Warnings: {config.output_root / config.warnings_filename}")


if __name__ == "__main__":
    main()