#!/usr/bin/env python3
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from tqdm import tqdm


DEFAULT_BASE_DIR = Path("/work/lvm/redux/1.2.0")
FALLBACK_BASE_DIR = Path("/data/users/gqzeng/lvm/spectro/redux/1.2.0")
DEFAULT_POINTINGS_CSV = Path("/home/dr20_pointings_final_v1.2.0.csv")
WINDOWS = {
    "b": (5400.0, 5500.0),
    "r": (6000.0, 6100.0),
    "z": (8800.0, 8900.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure continuum SNR from LVM SFrame data using median(flux) / "
            "median(noise) in fixed wavelength windows, then save one summary "
            "row per exposure."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base redux directory to scan recursively for lvmSFrame FITS files.",
    )
    parser.add_argument(
        "--pattern",
        default="**/lvmSFrame-*.fits",
        help="Glob pattern under --base-dir used to find SFrame files.",
    )
    parser.add_argument(
        "--pointings-csv",
        type=Path,
        default=DEFAULT_POINTINGS_CSV,
        help="CSV used to define the default list of pointings to process.",
    )
    parser.add_argument(
        "--targets",
        default="pointings",
        help="Use 'pointings' (default) to read targets from --pointings-csv, or 'all' to scan the full redux tree.",
    )
    parser.add_argument(
        "--input-files",
        nargs="*",
        default=None,
        help="Optional explicit list of SFrame FITS files. Overrides --targets and --pointings-csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/continuum_snr_tile_summary.parquet"),
        help="Output parquet file containing one summary row per SFrame.",
    )
    parser.add_argument(
        "--fiber-output",
        type=Path,
        default=None,
        help=(
            "Optional parquet file for per-fiber continuum measurements. "
            "Default: alongside --output with suffix _fibers.parquet."
        ),
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=32,
        help="Number of worker processes to use across files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of SFrame files to process.",
    )
    parser.add_argument(
        "--method",
        choices=("median", "all"),
        default="median",
        help=(
            "Window metric method. 'median' (default) uses the sample nearest the "
            "median per-pixel SNR; 'all' uses summed flux and summed noise across "
            "the full window."
        ),
    )
    return parser.parse_args()


def find_sframe_files(args: argparse.Namespace) -> list[Path]:
    if args.input_files:
        files = [Path(p) for p in args.input_files]
    elif args.targets == "all":
        files = sorted(args.base_dir.glob(args.pattern))
    else:
        pointings = pd.read_csv(args.pointings_csv, usecols=["tilegrp", "tileid", "mjd", "expnum"])
        files = []
        for row in pointings.itertuples(index=False):
            relative_path = (
                Path(str(row.tilegrp))
                / str(row.tileid)
                / str(row.mjd)
                / f"lvmSFrame-{int(row.expnum):08d}.fits"
            )
            path = resolve_pointing_path(args.base_dir, relative_path)
            if path.exists():
                files.append(path)
        files = sorted(set(files))
    if args.limit is not None:
        files = files[: args.limit]
    return files


def resolve_pointing_path(base_dir: Path, relative_path: Path) -> Path:
    primary_path = base_dir / relative_path
    if primary_path.exists():
        return primary_path

    fallback_path = FALLBACK_BASE_DIR / relative_path
    return fallback_path


def header_value(header, *keys, default=None):
    for key in keys:
        if key in header:
            return header[key]
    return default


def slitmap_value(slitmap, *keys, default=None):
    available = {name.lower(): name for name in slitmap.names}
    for key in keys:
        column = available.get(key.lower())
        if column is not None:
            return np.asarray(slitmap[column])
    return default


def format_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(Path(base_dir.name) / path.relative_to(base_dir))
    except ValueError:
        parts = path.parts
        if base_dir.name in parts:
            idx = parts.index(base_dir.name)
            return str(Path(*parts[idx:]))
        return str(path)


def summarize_array(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_n_valid": 0,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_med": np.nan,
            f"{prefix}_16th": np.nan,
            f"{prefix}_84th": np.nan,
        }
    return {
        f"{prefix}_n_valid": int(finite.size),
        f"{prefix}_min": float(np.nanmin(finite)),
        f"{prefix}_max": float(np.nanmax(finite)),
        f"{prefix}_med": float(np.nanmedian(finite)),
        f"{prefix}_16th": float(np.nanpercentile(finite, 16)),
        f"{prefix}_84th": float(np.nanpercentile(finite, 84)),
    }


def compute_window_metrics(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    region = (wave >= bounds[0]) & (wave <= bounds[1])
    if not np.any(region):
        empty = np.full(flux.shape[0], np.nan, dtype=float)
        return empty.copy(), empty.copy(), empty.copy()

    use = (
        region[None, :]
        & np.isfinite(flux)
        & np.isfinite(ivar)
        & np.isfinite(mask)
        & (ivar > 0.0)
        & (mask == 0)
    )

    signal = np.full(flux.shape[0], np.nan, dtype=float)
    noise = np.full(flux.shape[0], np.nan, dtype=float)
    snr = np.full(flux.shape[0], np.nan, dtype=float)
    for i in range(flux.shape[0]):
        good = use[i]
        if np.count_nonzero(good) == 0:
            continue

        flux_good = flux[i, good]
        noise_good = np.sqrt(1.0 / ivar[i, good])
        snr_good = np.full(flux_good.shape, np.nan, dtype=float)
        valid_noise = np.isfinite(noise_good) & (noise_good > 0.0)
        snr_good[valid_noise] = flux_good[valid_noise] / noise_good[valid_noise]

        finite = np.where(np.isfinite(snr_good))[0]
        if finite.size == 0:
            continue

        snr_med = float(np.nanmedian(snr_good[finite]))
        med_idx = finite[np.argmin(np.abs(snr_good[finite] - snr_med))]
        signal[i] = float(flux_good[med_idx]) if np.isfinite(flux_good[med_idx]) else np.nan
        noise[i] = float(noise_good[med_idx]) if np.isfinite(noise_good[med_idx]) else np.nan
        snr[i] = float(snr_good[med_idx])
    return signal, noise, snr


def compute_window_metrics_all(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    region = (wave >= bounds[0]) & (wave <= bounds[1])
    if not np.any(region):
        empty = np.full(flux.shape[0], np.nan, dtype=float)
        return empty.copy(), empty.copy(), empty.copy()

    use = (
        region[None, :]
        & np.isfinite(flux)
        & np.isfinite(ivar)
        & np.isfinite(mask)
        & (ivar > 0.0)
        & (mask == 0)
    )

    signal = np.full(flux.shape[0], np.nan, dtype=float)
    noise = np.full(flux.shape[0], np.nan, dtype=float)
    snr = np.full(flux.shape[0], np.nan, dtype=float)
    for i in range(flux.shape[0]):
        good = use[i]
        if np.count_nonzero(good) == 0:
            continue

        flux_good = flux[i, good]
        noise_good = 1.0 / ivar[i, good]
        sum_flux = float(np.nansum(flux_good))
        sum_noise = float(np.sqrt(np.nansum(noise_good)))
        signal[i] = sum_flux
        noise[i] = sum_noise
        if np.isfinite(sum_noise) and sum_noise > 0.0:
            snr[i] = sum_flux / sum_noise
    return signal, noise, snr


def summarize_window(
    signal: np.ndarray,
    noise: np.ndarray,
    snr: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    stats = summarize_array(snr, f"{prefix}_snr")
    finite = np.where(np.isfinite(snr))[0]
    if finite.size == 0:
        stats[f"{prefix}_med_signal"] = np.nan
        stats[f"{prefix}_med_noise"] = np.nan
        return stats
    snr_med = stats[f"{prefix}_snr_med"]
    med_idx = finite[np.argmin(np.abs(snr[finite] - snr_med))]
    stats[f"{prefix}_med_signal"] = float(signal[med_idx]) if np.isfinite(signal[med_idx]) else np.nan
    stats[f"{prefix}_med_noise"] = float(noise[med_idx]) if np.isfinite(noise[med_idx]) else np.nan
    return stats


def process_file(task: tuple[str, str, str]) -> tuple[pd.DataFrame, dict[str, object]]:
    path_str, base_dir_str, method = task
    path = Path(path_str)
    base_dir = Path(base_dir_str)

    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        wave = np.asarray(hdul["WAVE"].data, dtype=float)
        flux = np.asarray(hdul["FLUX"].data, dtype=float)
        ivar = np.asarray(hdul["IVAR"].data, dtype=float)
        mask = np.asarray(hdul["MASK"].data)
        slitmap = hdul["SLITMAP"].data

    fiber_df = pd.DataFrame(
        {
            "tileid": header_value(header, "TILE_ID", "TILEID"),
            "expnum": header_value(header, "EXPOSURE", "EXPNUM"),
            "mjd": header_value(header, "MJD"),
            "filename": path.name,
            "file_path": format_relative_path(path, base_dir),
            "fiberid": np.asarray(slitmap["fiberid"]),
            "spectrographid": slitmap_value(slitmap, "spectrographID", "spectrographid"),
            "ra": np.asarray(slitmap["ra"], dtype=float),
            "dec": np.asarray(slitmap["dec"], dtype=float),
            "targettype": np.asarray(slitmap["targettype"]).astype(str),
            "telescope": np.asarray(slitmap["telescope"]).astype(str),
            "fibstatus": np.asarray(slitmap["fibstatus"]),
        }
    )

    row: dict[str, object] = {
        "tileid": header_value(header, "TILE_ID", "TILEID"),
        "expnum": header_value(header, "EXPOSURE", "EXPNUM"),
        "mjd": header_value(header, "MJD"),
        "filename": path.name,
        "file_path": format_relative_path(path, base_dir),
    }

    total_valid = 0
    compute_metrics = compute_window_metrics_all if method == "all" else compute_window_metrics
    for name, bounds in WINDOWS.items():
        signal, noise, snr = compute_metrics(
            wave=wave,
            flux=flux,
            ivar=ivar,
            mask=mask,
            bounds=bounds,
        )
        fiber_df[f"{name}_signal"] = signal
        fiber_df[f"{name}_noise"] = noise
        fiber_df[f"{name}_snr"] = snr
        stats = summarize_window(signal, noise, snr, name)
        total_valid = max(total_valid, stats[f"{name}_snr_n_valid"])
        row.update(stats)

    row["n_valid_spectra"] = total_valid
    return fiber_df, row


def main() -> int:
    args = parse_args()
    files = find_sframe_files(args)
    if not files:
        raise FileNotFoundError("No SFrame files found. Check --base-dir, --pattern, or --input-files.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.fiber_output is None:
        args.fiber_output = args.output.with_name(f"{args.output.stem}_fibers.parquet")
    args.fiber_output.parent.mkdir(parents=True, exist_ok=True)
    tasks = [(str(path), str(args.base_dir), args.method) for path in files]
    rows: list[dict[str, object]] = []
    fiber_writer: pq.ParquetWriter | None = None

    try:
        if args.nproc == 1:
            iterator = (process_file(task) for task in tasks)
            for fiber_df, row in tqdm(iterator, total=len(tasks), desc="Processing SFrames"):
                fiber_table = pa.Table.from_pandas(fiber_df, preserve_index=False)
                if fiber_writer is None:
                    fiber_writer = pq.ParquetWriter(args.fiber_output, fiber_table.schema)
                fiber_writer.write_table(fiber_table)
                rows.append(row)
        else:
            with Pool(processes=args.nproc) as pool:
                iterator = pool.imap(process_file, tasks)
                for fiber_df, row in tqdm(iterator, total=len(tasks), desc="Processing SFrames"):
                    fiber_table = pa.Table.from_pandas(fiber_df, preserve_index=False)
                    if fiber_writer is None:
                        fiber_writer = pq.ParquetWriter(args.fiber_output, fiber_table.schema)
                    fiber_writer.write_table(fiber_table)
                    rows.append(row)
    finally:
        if fiber_writer is not None:
            fiber_writer.close()

    df = pd.DataFrame(rows).sort_values(["tileid", "expnum", "mjd"], kind="stable")
    df.to_parquet(args.output, index=False)

    print(f"Processed {len(files)} SFrame files")
    print(f"Wrote summary parquet: {args.output}")
    print(f"Wrote per-fiber parquet: {args.fiber_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
