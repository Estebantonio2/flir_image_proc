import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.thermal_dataset.naming import translate_compound_name


DEFAULT_SEQUENCE_PREFIX = "test_pre_mano"
DEFAULT_PERSON = "fabricio"
DEFAULT_SURFACE = "glass"
PROCESSED_ROOT = Path("processed_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grafica la curva termica baseline-contacto-disipacion desde processed_data.",
    )
    parser.add_argument(
        "--source-sequence-prefix",
        default=DEFAULT_SEQUENCE_PREFIX,
        help="Prefijo de la subcarpeta raw original a buscar dentro de source_image_path. Default: test_pre_mano.",
    )
    parser.add_argument(
        "--person",
        default=DEFAULT_PERSON,
        help="Nombre de la persona en metadata. Default: fabricio.",
    )
    parser.add_argument(
        "--surface",
        default=DEFAULT_SURFACE,
        help="Superficie en metadata. Default: glass.",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=120.0,
        help="Segundos iniciales sin mano. Default: 120.",
    )
    parser.add_argument(
        "--contact-seconds",
        type=float,
        default=10.0,
        help="Duracion aproximada de contacto de la mano antes de disipar. Default: 10.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta PNG de salida.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Ruta CSV con la serie usada para la grafica.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra la ventana interactiva de la grafica.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curve = build_curve_from_processed(
        processed_root=PROCESSED_ROOT,
        person=args.person,
        surface=args.surface,
        source_sequence_prefix=args.source_sequence_prefix,
    )

    output_path = args.output or PROCESSED_ROOT / f"{args.person}_{args.surface}_pre_hand_curve.png"
    csv_output_path = args.csv_output or PROCESSED_ROOT / f"{args.person}_{args.surface}_pre_hand_curve.csv"
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(csv_output_path, index=False)

    plot_curve(
        curve=curve,
        source_sequence_prefix=args.source_sequence_prefix,
        baseline_seconds=args.baseline_seconds,
        contact_seconds=args.contact_seconds,
        output_path=output_path,
        show=args.show,
    )

    print(f"Processed root: {PROCESSED_ROOT}")
    print(f"Secuencia fuente: {args.source_sequence_prefix}")
    print(f"Muestras: {len(curve)}")
    print(f"CSV: {csv_output_path}")
    print(f"Grafica: {output_path}")


def build_curve_from_processed(
    processed_root: Path,
    person: str,
    surface: str,
    source_sequence_prefix: str,
) -> pd.DataFrame:
    metadata_path = processed_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata procesada: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    rows = metadata[
        (metadata["name"].astype(str).str.lower() == person.lower())
        & (metadata["surface"].astype(str).str.lower() == surface.lower())
        & (
            metadata["source_image_path"]
            .astype(str)
            .str.replace("\\", "/", regex=False)
            .str.contains(f"/{source_sequence_prefix}", case=False, regex=False)
        )
    ].copy()

    if rows.empty:
        raise ValueError(
            "No se encontraron muestras preprocesadas para "
            f"name={person}, surface={surface}, source_sequence_prefix={source_sequence_prefix} "
            f"en {metadata_path}."
        )

    rows = rows.sort_values("snapshot_number").reset_index(drop=True)

    thermal_arrays = [
        np.load(processed_root / str(row["thermal_path"]))
        for _, row in rows.iterrows()
    ]
    global_tmin = min(float(np.nanmin(thermal)) for thermal in thermal_arrays)
    global_tmax = max(float(np.nanmax(thermal)) for thermal in thermal_arrays)

    if np.isclose(global_tmax, global_tmin):
        raise ValueError(
            "No se puede normalizar la secuencia porque todas las temperaturas tienen el mismo valor."
        )

    curve_rows = []
    for (_, row), thermal in zip(rows.iterrows(), thermal_arrays):
        normalized_thermal = (thermal - global_tmin) / (global_tmax - global_tmin)
        curve_row = {
            "file": Path(str(row["source_image_path"])).name,
            "sample_id": row["sample_id"],
            "sequence_id": row["sequence_id"],
            "snapshot_number": int(row["snapshot_number"]),
            "capture_datetime": row["capture_datetime"],
            "t_seconds": float(row["t_seconds"]),
            "img_tmean_C": float(np.nanmean(thermal)),
            "img_tmin_sequence_C": global_tmin,
            "img_tmax_sequence_C": global_tmax,
            "normalized_tmean_0_1": float(np.nanmean(normalized_thermal)),
        }
        curve_rows.append(curve_row)

    return pd.DataFrame(curve_rows)


def plot_curve(
    curve: pd.DataFrame,
    source_sequence_prefix: str,
    baseline_seconds: float,
    contact_seconds: float,
    output_path: Path,
    show: bool,
) -> None:
    contact_end = baseline_seconds + contact_seconds

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        curve["t_seconds"],
        curve["normalized_tmean_0_1"],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        label="Normalized mean temperature",
    )
    ax.axvspan(0, baseline_seconds, color="#4c78a8", alpha=0.12, label="No hand")
    ax.axvspan(baseline_seconds, contact_end, color="#f58518", alpha=0.16, label="Hand contact")
    ax.axvspan(contact_end, curve["t_seconds"].max(), color="#54a24b", alpha=0.10, label="Dissipation")
    ax.axvline(baseline_seconds, color="#f58518", linestyle="--", linewidth=1)
    ax.axvline(contact_end, color="#54a24b", linestyle="--", linewidth=1)

    ax.set_title(f"Dissipation curve - {translate_compound_name(source_sequence_prefix)}")
    ax.set_xlabel("Time since first frame (s)")
    ax.set_ylabel("Normalized mean temperature (0-1)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
