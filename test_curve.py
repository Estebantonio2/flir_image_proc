import argparse
from pathlib import Path, PureWindowsPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.thermal_dataset.naming import translate_compound_name


PROCESSED_ROOT = Path("processed_data")
DEFAULT_PERSON = "jd"
DEFAULT_SURFACE = "wood"
DEFAULT_SEQUENCE_PATTERN = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grafica baseline y disipacion termica desde processed_data, incluyendo t=0.",
    )
    parser.add_argument("--person", default=DEFAULT_PERSON)
    parser.add_argument("--surface", default=DEFAULT_SURFACE)
    parser.add_argument("--pattern", default=DEFAULT_SEQUENCE_PATTERN, help="Opcional: patron en sequence_id")
    parser.add_argument("--baseline-end", type=float, default=-10.0)
    parser.add_argument("--dissipation-start", type=float, default=0.0)
    parser.add_argument("--contact-frame-time", type=float, default=0.0)
    parser.add_argument("--footprint-percentile", type=float, default=85.0)
    parser.add_argument(
        "--metric",
        choices=("mean_delta_C", "positive_mean_delta_C", "energy_normalized"),
        default="positive_mean_delta_C",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(PROCESSED_ROOT / "metadata_train.csv")
    
    # Filtrar por persona y superficie
    mask = (metadata["name"].astype(str).str.lower() == args.person.lower()) & \
           (metadata["surface"].astype(str).str.lower() == args.surface.lower())
    if args.pattern:
        mask &= metadata["sequence_id"].astype(str).str.contains(args.pattern, case=False)
    
    filtered_meta = metadata[mask].copy()
    if filtered_meta.empty:
        print(f"No se encontraron datos para {args.person} {args.surface}")
        return

    sequences = filtered_meta["sequence_id"].unique()
    processed_curves = []
    
    output_base = args.output_dir or PROCESSED_ROOT / f"{args.person}_{args.surface}"
    output_base.mkdir(parents=True, exist_ok=True)

    for seq_id in sequences:
        print(f"Procesando secuencia: {seq_id}...")
        seq_rows = filtered_meta[filtered_meta["sequence_id"] == seq_id].sort_values("t_seconds")
        
        try:
            curve = build_curve_for_sequence(seq_rows, args)
            processed_curves.append(curve)
            
            # Guardar individual
            output_path = output_base / f"{seq_id}_dissipation.png"
            plot_curve(curve, args, output_path, title=f"Dissipation: {seq_id}")
            curve.to_csv(output_base / f"{seq_id}_dissipation.csv", index=False)
            
        except ValueError as e:
            print(f"  Omitiendo {seq_id}: {e}")
            continue

    if len(processed_curves) >= 3:
        print(f"Generando promedio de {len(processed_curves)} secuencias...")
        avg_curve = build_average_curve(processed_curves, args)
        output_path = output_base / f"{args.person}_{args.surface}_average_dissipation.png"
        plot_curve(avg_curve, args, output_path, title=f"Average Dissipation: {args.person} {args.surface} (n={len(processed_curves)})")
        avg_curve.to_csv(output_base / f"{args.person}_{args.surface}_average_dissipation.csv", index=False)


def build_curve_for_sequence(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    # Cargar stacks termicos, saltando archivos que no existen
    valid_rows = []
    thermal_arrays = []
    
    for _, row in rows.iterrows():
        p = resolve_path(row["thermal_path"])
        if p.exists():
            try:
                thermal_arrays.append(np.load(p))
                valid_rows.append(row)
            except Exception as e:
                print(f"  Error cargando {p.name}: {e}")
        else:
            # Silenciosamente saltar archivos faltantes como pidió el usuario
            pass
            
    if not valid_rows:
        raise ValueError("No se encontraron archivos validos para esta secuencia.")
        
    thermal_stack = np.stack(thermal_arrays).astype(float)
    rows = pd.DataFrame(valid_rows).reset_index(drop=True)
    times = rows["t_seconds"].astype(float).to_numpy()

    # Identificar baseline (t < baseline_end)
    baseline_mask = times <= args.baseline_end
    if np.count_nonzero(baseline_mask) < 3:
        raise ValueError(f"Menos de 3 capturas baseline (t <= {args.baseline_end})")

    baseline_image = np.nanmedian(thermal_stack[baseline_mask], axis=0)
    residual_stack = thermal_stack - baseline_image

    # Crear mascara de huella en dissipation_start (usualmente t=0)
    footprint_idx = np.argmin(np.abs(times - args.dissipation_start))
    footprint_source = residual_stack[footprint_idx]
    
    threshold = float(np.nanpercentile(footprint_source, args.footprint_percentile))
    footprint_mask = footprint_source >= threshold
    if not np.any(footprint_mask):
        raise ValueError("No se pudo construir la mascara de huella.")

    rows_out = []
    for i, (residual, (_, row)) in enumerate(zip(residual_stack, rows.iterrows())):
        t_seconds = float(row["t_seconds"])
        is_baseline = t_seconds <= args.baseline_end
        is_dissipation = t_seconds >= args.dissipation_start
        
        if not (is_baseline or is_dissipation):
            continue

        footprint = residual[footprint_mask]
        positive = np.clip(footprint, 0, None)
        rows_out.append(
            {
                "sequence_id": row["sequence_id"],
                "t_seconds": t_seconds,
                "time_after_removal_min": (t_seconds - args.contact_frame_time) / 60.0,
                "phase": "baseline" if is_baseline else "dissipation",
                "mean_delta_C": float(np.nanmean(footprint)),
                "positive_mean_delta_C": float(np.nanmean(positive)),
                "energy_C_px": float(np.nansum(positive)),
                "footprint_pixel_count": int(np.count_nonzero(footprint_mask)),
            }
        )

    curve = pd.DataFrame(rows_out)
    peak_energy = max(float(curve["energy_C_px"].max()), 1e-6)
    curve["energy_normalized"] = curve["energy_C_px"] / peak_energy
    
    # Suavizado por secuencia
    curve["rolling_median"] = (
        curve.groupby("phase")[args.metric]
        .transform(lambda values: values.rolling(3, center=True, min_periods=1).median())
    )
    return curve


def build_average_curve(curves: list[pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    # Combinar todas las curvas
    df = pd.concat(curves, ignore_index=True)
    
    # Agrupar por tiempo aproximado (redondear t_seconds para alinear si hay pequenas fallas)
    # Dado que el intervalo es 5s o mas, podemos redondear a 1s
    df["t_rounded"] = df["t_seconds"].round(0)
    
    # Promediar la métrica seleccionada
    avg_data = df.groupby("t_rounded").agg({
        "time_after_removal_min": "first",
        "phase": "first",
        args.metric: "mean",
        "energy_normalized": "mean"
    }).reset_index()
    
    avg_data = avg_data.sort_values("t_rounded")
    
    # Recalcular rolling median sobre el promedio
    avg_data["rolling_median"] = (
        avg_data.groupby("phase")[args.metric]
        .transform(lambda values: values.rolling(3, center=True, min_periods=1).median())
    )
    return avg_data


def plot_curve(curve: pd.DataFrame, args: argparse.Namespace, output_path: Path, title: str) -> None:
    labels = {
        "mean_delta_C": "Mean temperature change in footprint vs baseline (C)",
        "positive_mean_delta_C": "Positive heat in footprint vs baseline (C)",
        "energy_normalized": "Positive footprint heat energy (normalized)",
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    x = curve["time_after_removal_min"]
    y = curve[args.metric]
    baseline = curve[curve["phase"] == "baseline"]
    dissipation = curve[curve["phase"] == "dissipation"]

    # Zonas
    ax.axvspan(float(x.min()), args.baseline_end / 60.0, color="#4c78a8", alpha=0.12, label="Baseline (No hand)")
    ax.axvspan(
        args.baseline_end / 60.0,
        args.dissipation_start / 60.0,
        color="#f58518",
        alpha=0.14,
        label="Hand contact",
    )
    ax.axvspan(args.dissipation_start / 60.0, float(x.max()), color="#54a24b", alpha=0.10, label="Dissipation")

    ax.scatter(
        baseline["time_after_removal_min"],
        baseline[args.metric],
        s=24, alpha=0.65, color="#4c78a8", label="Baseline samples",
    )
    ax.scatter(
        dissipation["time_after_removal_min"],
        dissipation[args.metric],
        s=28, alpha=0.60, color="#59a14f", label="Dissipation samples",
    )
    ax.plot(
        dissipation["time_after_removal_min"],
        dissipation["rolling_median"],
        color="#d62728", linewidth=2.4, label="Trend (rolling median)",
    )
    
    ax.axhline(0, color="#222222", linestyle=":", linewidth=1)
    ax.axvline(0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlim(float(x.min()), max(float(x.max()), 0.1))
    
    set_y_limits(ax, y, args.metric)

    ax.set_title(title)
    ax.set_xlabel("Time relative to hand removal (min)")
    ax.set_ylabel(labels[args.metric])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(output_path, dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)


def set_y_limits(ax, values: pd.Series, metric: str) -> None:
    if metric == "energy_normalized":
        ax.set_ylim(-0.03, 1.03)
        return

    finite = values[np.isfinite(values)]
    if finite.empty:
        return

    low = min(0.0, float(finite.quantile(0.02)))
    high = float(finite.quantile(0.98))
    pad = max((high - low) * 0.15, 0.05)
    ax.set_ylim(low - pad, high + pad)


def resolve_path(path: str | Path) -> Path:
    normalized = Path(str(path).replace("\\", "/"))
    if normalized.is_absolute():
        return normalized
    return PROCESSED_ROOT / normalized


def filename_from_any_path(path: str | Path) -> str:
    text = str(path)
    if "\\" in text or ":" in text:
        return PureWindowsPath(text).name
    return Path(text).name


if __name__ == "__main__":
    main()
