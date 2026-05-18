import argparse
from pathlib import Path, PureWindowsPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.thermal_dataset.naming import translate_compound_name


PROCESSED_ROOT = Path("processed_data")
DEFAULT_PERSON = "fabricio"
DEFAULT_SURFACE = "glass"
DEFAULT_SEQUENCE_PREFIX = "test_pre_mano"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grafica baseline y disipacion termica desde processed_data, omitiendo t=0.",
    )
    parser.add_argument("--person", default=DEFAULT_PERSON)
    parser.add_argument("--surface", default=DEFAULT_SURFACE)
    parser.add_argument("--source-sequence-prefix", default=DEFAULT_SEQUENCE_PREFIX)
    parser.add_argument("--baseline-end", type=float, default=-10.0)
    parser.add_argument("--dissipation-start", type=float, default=5.0)
    parser.add_argument("--contact-frame-time", type=float, default=0.0)
    parser.add_argument("--footprint-percentile", type=float, default=85.0)
    parser.add_argument(
        "--metric",
        choices=("mean_delta_C", "positive_mean_delta_C", "energy_normalized"),
        default="positive_mean_delta_C",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curve = build_curve(args)

    output_dir = PROCESSED_ROOT / f"{args.person}_{args.surface}"
    output_stem = f"{translate_compound_name(args.source_sequence_prefix)}_dissipation_curve"
    output_path = args.output or output_dir / f"{output_stem}.png"
    csv_path = args.csv_output or output_dir / f"{output_stem}.csv"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(csv_path, index=False)
    plot_curve(curve, args, output_path)

    print(f"Muestras graficadas: {len(curve)}")
    print(f"CSV: {csv_path}")
    print(f"Grafica: {output_path}")


def build_curve(args: argparse.Namespace) -> pd.DataFrame:
    metadata = pd.read_csv(PROCESSED_ROOT / "metadata.csv")
    rows = select_rows(metadata, args).sort_values("snapshot_number").reset_index(drop=True)
    if rows.empty:
        raise ValueError("No se encontraron muestras para los filtros indicados.")

    thermal_stack = np.stack(
        [np.load(resolve_path(path)) for path in rows["thermal_path"]]
    ).astype(float)
    times = rows["t_seconds"].astype(float).to_numpy()

    baseline_mask = times <= args.baseline_end
    if not np.any(baseline_mask):
        raise ValueError(f"No hay capturas baseline con t_seconds <= {args.baseline_end:g}.")

    baseline_image = np.nanmedian(thermal_stack[baseline_mask], axis=0)
    residual_stack = thermal_stack - baseline_image

    footprint_index = int(np.argmin(np.abs(times - args.dissipation_start)))
    footprint_source = residual_stack[footprint_index]
    threshold = float(np.nanpercentile(footprint_source, args.footprint_percentile))
    footprint_mask = footprint_source >= threshold
    if not np.any(footprint_mask):
        raise ValueError("No se pudo construir la mascara de huella.")

    rows_out = []
    for residual, (_, row) in zip(residual_stack, rows.iterrows()):
        t_seconds = float(row["t_seconds"])
        is_baseline = t_seconds <= args.baseline_end
        is_dissipation = t_seconds >= args.dissipation_start
        if not (is_baseline or is_dissipation):
            continue

        footprint = residual[footprint_mask]
        positive = np.clip(footprint, 0, None)
        rows_out.append(
            {
                "file": filename_from_any_path(row["source_image_path"]),
                "sample_id": row["sample_id"],
                "sequence_id": row["sequence_id"],
                "t_seconds": t_seconds,
                "time_after_removal_s": t_seconds - args.contact_frame_time,
                "time_after_removal_min": (t_seconds - args.contact_frame_time) / 60.0,
                "phase": "baseline" if is_baseline else "dissipation",
                "mean_delta_C": float(np.nanmean(footprint)),
                "positive_mean_delta_C": float(np.nanmean(positive)),
                "energy_C_px": float(np.nansum(positive)),
                "footprint_pixel_count": int(np.count_nonzero(footprint_mask)),
                "footprint_threshold_C": threshold,
            }
        )

    curve = pd.DataFrame(rows_out)
    peak_energy = max(float(curve["energy_C_px"].max()), 1e-6)
    curve["energy_normalized"] = curve["energy_C_px"] / peak_energy
    curve["rolling_median"] = (
        curve.groupby("phase")[args.metric]
        .transform(lambda values: values.rolling(5, center=True, min_periods=1).median())
    )
    return curve


def select_rows(metadata: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    source_paths = metadata["source_image_path"].astype(str).str.replace("\\", "/", regex=False)
    return metadata[
        (metadata["name"].astype(str).str.lower() == args.person.lower())
        & (metadata["surface"].astype(str).str.lower() == args.surface.lower())
        & source_paths.str.contains(f"/{args.source_sequence_prefix}", case=False, regex=False)
    ].copy()


def plot_curve(curve: pd.DataFrame, args: argparse.Namespace, output_path: Path) -> None:
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

    ax.axvspan(float(x.min()), args.baseline_end / 60.0, color="#4c78a8", alpha=0.12, label="No hand")
    ax.axvspan(
        args.baseline_end / 60.0,
        args.dissipation_start / 60.0,
        color="#f58518",
        alpha=0.14,
        label="Hand contact / t=0 omitted",
    )
    ax.axvspan(args.dissipation_start / 60.0, float(x.max()), color="#54a24b", alpha=0.10, label="Dissipation")

    ax.scatter(
        baseline["time_after_removal_min"],
        baseline[args.metric],
        s=24,
        alpha=0.65,
        color="#4c78a8",
        label="Baseline samples",
    )
    ax.scatter(
        dissipation["time_after_removal_min"],
        dissipation[args.metric],
        s=28,
        alpha=0.60,
        color="#59a14f",
        label="Dissipation samples",
    )
    ax.plot(
        dissipation["time_after_removal_min"],
        dissipation["rolling_median"],
        color="#d62728",
        linewidth=2.4,
        label="Dissipation rolling median",
    )
    ax.axhline(0, color="#222222", linestyle=":", linewidth=1)
    ax.axvline(0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlim(float(x.min()), max(float(x.max()), 0.1))
    set_y_limits(ax, y, args.metric)

    title = translate_compound_name(args.source_sequence_prefix)
    ax.set_title(f"Thermal footprint baseline and dissipation - {title}")
    ax.set_xlabel("Time relative to hand removal (min)")
    ax.set_ylabel(labels[args.metric])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
