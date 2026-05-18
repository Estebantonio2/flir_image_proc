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
DEFAULT_METRIC = "footprint_mean_delta_C"


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
        "--baseline-end",
        type=float,
        default=-10.0,
        help="Ultima captura sin mano. Default: -10.",
    )
    parser.add_argument(
        "--dissipation-start",
        type=float,
        default=5.0,
        help="Primera captura de disipacion luego de retirar la mano. Default: 5.",
    )
    parser.add_argument(
        "--contact-frame-time",
        type=float,
        default=0.0,
        help="Captura con la mano aun puesta, justo antes de retirarla. Default: 0.",
    )
    parser.add_argument(
        "--include-contact-frame",
        action="store_true",
        help="Incluye el frame de contacto en la linea principal de la grafica.",
    )
    parser.add_argument(
        "--metric",
        choices=(
            "footprint_mean_delta_C",
            "footprint_positive_mean_delta_C",
            "footprint_energy_normalized_0_1",
            "mean_delta_baseline_C",
            "mean_deltaT_baseline_C",
            "hot_p95_delta_baseline_C",
            "hot_p95_deltaT_baseline_C",
            "normalized_mean_0_1",
        ),
        default=DEFAULT_METRIC,
        help=(
            "Metrica a graficar. Default: footprint_mean_delta_C "
            "(misma zona de huella, definida en t=0, contra baseline pixel-a-pixel)."
        ),
    )
    parser.add_argument(
        "--footprint-percentile",
        type=float,
        default=85.0,
        help=(
            "Percentil del residual en t=0 usado para fijar la mascara de huella. "
            "Default: 85."
        ),
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
        baseline_end=args.baseline_end,
        dissipation_start=args.dissipation_start,
        contact_frame_time=args.contact_frame_time,
        footprint_percentile=args.footprint_percentile,
    )

    output_path = args.output or PROCESSED_ROOT / f"{args.person}_{args.surface}_pre_hand_curve.png"
    csv_output_path = args.csv_output or PROCESSED_ROOT / f"{args.person}_{args.surface}_pre_hand_curve.csv"
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(csv_output_path, index=False)

    plot_curve(
        curve=curve,
        source_sequence_prefix=args.source_sequence_prefix,
        baseline_end=args.baseline_end,
        dissipation_start=args.dissipation_start,
        contact_frame_time=args.contact_frame_time,
        include_contact_frame=args.include_contact_frame,
        metric=args.metric,
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
    baseline_end: float,
    dissipation_start: float,
    contact_frame_time: float,
    footprint_percentile: float,
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
    thermal_stack = np.stack(thermal_arrays).astype(float)
    times = rows["t_seconds"].astype(float).to_numpy()
    global_tmin = min(float(np.nanmin(thermal)) for thermal in thermal_arrays)
    global_tmax = max(float(np.nanmax(thermal)) for thermal in thermal_arrays)

    if np.isclose(global_tmax, global_tmin):
        raise ValueError(
            "No se puede normalizar la secuencia porque todas las temperaturas tienen el mismo valor."
        )

    baseline_frame_mask = times <= baseline_end
    if not np.any(baseline_frame_mask):
        raise ValueError(
            f"No se encontraron capturas baseline con t_seconds <= {baseline_end:g}."
        )

    baseline_image = np.nanmedian(thermal_stack[baseline_frame_mask], axis=0)
    residual_stack = thermal_stack - baseline_image

    footprint_frame_index = int(np.argmin(np.abs(times - dissipation_start)))
    footprint_source = residual_stack[footprint_frame_index]
    footprint_threshold = float(np.nanpercentile(footprint_source, footprint_percentile))
    footprint_mask = footprint_source >= footprint_threshold

    if not np.any(footprint_mask):
        raise ValueError("No se pudo definir una mascara de huella con los parametros actuales.")

    peak_positive_energy = float(
        np.nansum(np.clip(footprint_source[footprint_mask], 0, None))
    )
    if np.isclose(peak_positive_energy, 0.0):
        peak_positive_energy = 1.0

    curve_rows = []
    for residual, (_, row), thermal in zip(residual_stack, rows.iterrows(), thermal_arrays):
        normalized_thermal = (thermal - global_tmin) / (global_tmax - global_tmin)
        footprint_residual = residual[footprint_mask]
        positive_footprint_residual = np.clip(footprint_residual, 0, None)
        curve_row = {
            "file": Path(str(row["source_image_path"])).name,
            "sample_id": row["sample_id"],
            "sequence_id": row["sequence_id"],
            "snapshot_number": int(row["snapshot_number"]),
            "capture_datetime": row["capture_datetime"],
            "t_seconds": float(row["t_seconds"]),
            "phase": classify_phase(
                t_seconds=float(row["t_seconds"]),
                baseline_end=baseline_end,
                contact_frame_time=contact_frame_time,
                dissipation_start=dissipation_start,
            ),
            "img_tmean_C": float(np.nanmean(thermal)),
            "delta_tmean_C": float(row["delta_tmean_C"]),
            "img_tmin_sequence_C": global_tmin,
            "img_tmax_sequence_C": global_tmax,
            "hot_tmean_C_p95": float(row["hot_tmean_C_p95"]),
            "hot_delta_tmean_C_p95": float(row["hot_delta_tmean_C_p95"]),
            "normalized_tmean_0_1": float(np.nanmean(normalized_thermal)),
            "footprint_pixel_count": int(np.count_nonzero(footprint_mask)),
            "footprint_threshold_C": footprint_threshold,
            "footprint_mean_delta_C": float(np.nanmean(footprint_residual)),
            "footprint_positive_mean_delta_C": float(np.nanmean(positive_footprint_residual)),
            "footprint_energy_C_px": float(np.nansum(positive_footprint_residual)),
            "footprint_energy_normalized_0_1": float(
                np.nansum(positive_footprint_residual) / peak_positive_energy
            ),
        }
        curve_rows.append(curve_row)

    curve = pd.DataFrame(curve_rows)
    baseline = curve[curve["t_seconds"] <= baseline_end]

    curve["mean_delta_baseline_C"] = (
        curve["img_tmean_C"] - baseline["img_tmean_C"].mean()
    )
    curve["mean_deltaT_baseline_C"] = (
        curve["delta_tmean_C"] - baseline["delta_tmean_C"].mean()
    )
    curve["hot_p95_delta_baseline_C"] = (
        curve["hot_tmean_C_p95"] - baseline["hot_tmean_C_p95"].mean()
    )
    curve["hot_p95_deltaT_baseline_C"] = (
        curve["hot_delta_tmean_C_p95"] - baseline["hot_delta_tmean_C_p95"].mean()
    )

    return curve


def plot_curve(
    curve: pd.DataFrame,
    source_sequence_prefix: str,
    baseline_end: float,
    dissipation_start: float,
    contact_frame_time: float,
    include_contact_frame: bool,
    metric: str,
    output_path: Path,
    show: bool,
) -> None:
    metric_labels = {
        "footprint_mean_delta_C": "Fixed footprint temperature change vs baseline (C)",
        "footprint_positive_mean_delta_C": "Fixed footprint positive heat vs baseline (C)",
        "footprint_energy_normalized_0_1": "Fixed footprint heat energy (normalized)",
        "mean_delta_baseline_C": "ROI mean temperature change vs baseline (C)",
        "mean_deltaT_baseline_C": "ROI mean deltaT change vs baseline (C)",
        "hot_p95_delta_baseline_C": "Warmest 5% temperature change vs baseline (C)",
        "hot_p95_deltaT_baseline_C": "Warmest 5% deltaT change vs baseline (C)",
        "normalized_mean_0_1": "Normalized mean temperature (0-1)",
    }
    metric_column = (
        "normalized_tmean_0_1"
        if metric == "normalized_mean_0_1"
        else metric
    )

    plot_curve_df = curve.copy()
    if not include_contact_frame:
        plot_curve_df = plot_curve_df[plot_curve_df["t_seconds"] != contact_frame_time]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        plot_curve_df["t_seconds"],
        plot_curve_df[metric_column],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        label=metric_labels[metric],
    )
    contact_frame = curve[curve["t_seconds"] == contact_frame_time]
    if not contact_frame.empty and not include_contact_frame:
        contact_y = contact_frame[metric_column].copy()
        if metric in {"normalized_mean_0_1", "footprint_energy_normalized_0_1"}:
            contact_y = contact_y.clip(upper=1.0)

        ax.scatter(
            contact_frame["t_seconds"],
            contact_y,
            marker="x",
            s=70,
            color="#d62728",
            label="Hand still present",
            zorder=4,
        )

    ax.axhline(0, color="#222222", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvspan(
        curve["t_seconds"].min(),
        baseline_end,
        color="#4c78a8",
        alpha=0.12,
        label="No hand",
    )
    ax.axvspan(
        baseline_end,
        contact_frame_time,
        color="#f58518",
        alpha=0.16,
        label="Hand contact",
    )
    ax.axvspan(
        contact_frame_time,
        curve["t_seconds"].max(),
        color="#54a24b",
        alpha=0.10,
        label="Dissipation",
    )
    ax.axvline(baseline_end, color="#f58518", linestyle="--", linewidth=1)
    ax.axvline(contact_frame_time, color="#54a24b", linestyle="--", linewidth=1)
    ax.axvline(dissipation_start, color="#54a24b", linestyle=":", linewidth=1)

    ax.set_title(f"Dissipation curve - {translate_compound_name(source_sequence_prefix)}")
    ax.set_xlabel("Time relative to hand removal (s)")
    ax.set_ylabel(metric_labels[metric])
    if metric in {"normalized_mean_0_1", "footprint_energy_normalized_0_1"}:
        ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()

    plt.close(fig)


def classify_phase(
    t_seconds: float,
    baseline_end: float,
    contact_frame_time: float,
    dissipation_start: float,
) -> str:
    if t_seconds <= baseline_end:
        return "no_hand"
    if t_seconds <= contact_frame_time:
        return "hand_contact"
    if t_seconds >= dissipation_start:
        return "dissipation"
    return "transition"


if __name__ == "__main__":
    main()
