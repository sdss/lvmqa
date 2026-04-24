#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from tqdm import tqdm


HA_REST = 6562.85
DEFAULT_BASE_DIR = Path("/work/lvm/redux/1.2.0")
FALLBACK_BASE_DIR = Path("/data/users/gqzeng/lvm/spectro/redux/1.2.0")
DEFAULT_POINTINGS_CSV = Path("/home/dr20_pointings_final_v1.2.0.csv")


@dataclass(frozen=True)
class singlet:
    w: float
    centre: tuple[float, float]
    left: tuple[float, float]
    right: tuple[float, float]
    name: str
    type: str


@dataclass(frozen=True)
class MeasureConfig:
    ha_line: singlet = singlet(
        w=6562.8,
        centre=(6560.0, 6566.0),
        left=(6550.0, 6560.0),
        right=(6566.0, 6576.0),
        name="H_alpha",
        type="strong",
    )
    peak_search_half_width: float = 5.0
    absorption_snr_threshold: float = 3.0
    centroid_no_shift_half_width: float = 1.0
    
# class MeasureConfigV2:
#     ha_line: singlet = singlet(
#         w=6562.8,
#         centre=(6553.0, 6573.0),
#         left=(6534.0, 6544.0),
#         right=(6590.0, 6600.0),
#         name="H_alpha",
#         type="strong",
#     )
#     peak_search_half_width: float = 5.0
#     peak_snr_threshold: float = 5.0
## left right windows outside [NII]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure fiber-level H-alpha SNR in LVM SFrame files and save one "
            "summary row per tile exposure."
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
        default=Path("/home/ha_snr_tile_summary.parquet"),
        help="Output parquet file containing one summary row per SFrame.",
    )
    parser.add_argument(
        "--fiber-output",
        type=Path,
        default=None,
        help=(
            "Optional parquet file for per-fiber H-alpha measurements. "
            "Default: alongside --output with suffix _fibers.parquet."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of SFrame files to process.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=32,
        help="Number of worker processes to use across files.",
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


def format_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(Path(base_dir.name) / path.relative_to(base_dir))
    except ValueError:
        parts = path.parts
        if base_dir.name in parts:
            idx = parts.index(base_dir.name)
            return str(Path(*parts[idx:]))
        return str(path)


def header_value(header, *keys, default=None):
    for key in keys:
        if key in header:
            return header[key]
    return default


def safe_numeric(series) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    return arr


def safe_text(series) -> np.ndarray:
    values = np.asarray(series)
    out: list[str] = []
    for value in values:
        if isinstance(value, (bytes, np.bytes_)):
            out.append(value.decode("utf-8", errors="ignore").strip())
        else:
            out.append(str(value).strip())
    return np.asarray(out, dtype=object)


def region_mask(wave: np.ndarray, bounds: tuple[float, float], inclusive_right: bool = True) -> np.ndarray:
    if inclusive_right:
        return (wave >= bounds[0]) & (wave <= bounds[1])
    return (wave >= bounds[0]) & (wave < bounds[1])


def fit_linear_continuum_single(
    wave: np.ndarray,
    flux_1d: np.ndarray,
    valid_1d: np.ndarray,
    left_bounds: tuple[float, float],
    right_bounds: tuple[float, float],
) -> tuple[float, float, float, float, float]:
    left_mask = region_mask(wave, left_bounds, inclusive_right=False)
    right_mask = region_mask(wave, right_bounds, inclusive_right=True)
    cont_mask = valid_1d & (left_mask | right_mask)
    if np.count_nonzero(cont_mask) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    x = wave[cont_mask]
    y = flux_1d[cont_mask]
    coeff = np.polyfit(x, y, 1)
    model = coeff[0] * x + coeff[1]
    resid = y - model
    cont_rms = float(np.sqrt(np.mean(resid**2)))

    left_use = valid_1d & left_mask
    right_use = valid_1d & right_mask
    left_med = float(np.nanmedian(flux_1d[left_use])) if np.any(left_use) else np.nan
    right_med = float(np.nanmedian(flux_1d[right_use])) if np.any(right_use) else np.nan
    return float(coeff[0]), float(coeff[1]), left_med, right_med, cont_rms


def find_centroid_shift(
    wave: np.ndarray,
    flux_1d: np.ndarray,
    ivar_1d: np.ndarray,
    valid_1d: np.ndarray,
    line_center: float,
    init_slope: float,
    init_intercept: float,
    search_half_width: float,
    absorption_snr_threshold: float,
    no_shift_half_width: float,
) -> float:
    search_mask = np.abs(wave - line_center) <= search_half_width
    search_use = valid_1d & search_mask
    if not np.any(search_use):
        return 0.0

    model = init_slope * wave + init_intercept
    wave_search = wave[search_use]
    resid_search = flux_1d[search_use] - model[search_use]
    noise_search = np.sqrt(1.0 / ivar_1d[search_use])

    good = np.isfinite(resid_search) & np.isfinite(noise_search) & (noise_search > 0.0)
    if not np.any(good):
        return 0.0

    wave_search = wave_search[good]
    resid_search = resid_search[good]
    noise_search = noise_search[good]
    snr_search = resid_search / noise_search

    keep = snr_search > -absorption_snr_threshold
    if not np.any(keep):
        return 0.0

    wave_search = wave_search[keep]
    resid_search = resid_search[keep]

    weights = np.clip(resid_search, 0.0, None)
    if not np.any(weights > 0.0):
        return 0.0

    centroid = float(np.sum(wave_search * weights) / np.sum(weights))
    if not np.isfinite(centroid):
        return 0.0
    if abs(centroid - line_center) <= no_shift_half_width:
        return 0.0
    return centroid - line_center


def compute_ha_snr(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    lsf: np.ndarray,
    meta_df: pd.DataFrame,
    cfg: MeasureConfig,
) -> pd.DataFrame:
    line = cfg.ha_line
    valid = (
        np.isfinite(flux)
        & np.isfinite(ivar)
        & np.isfinite(mask)
        & np.isfinite(lsf)
        & (ivar > 0.0)
        & (mask == 0)
    )
    line_flux = np.full(flux.shape[0], np.nan, dtype=float)
    line_err = np.full(flux.shape[0], np.nan, dtype=float)
    ha_snr = np.full(flux.shape[0], np.nan, dtype=float)
    n_line_pix = np.zeros(flux.shape[0], dtype=int)
    centre_width = line.centre[1] - line.centre[0]
    line_idx = int(np.argmin(np.abs(wave - line.w)))
    sigma_lambda = np.asarray(lsf[:, line_idx], dtype=float)
    half_window = np.full(flux.shape[0], 0.5 * centre_width, dtype=float)
    slope = np.full(flux.shape[0], np.nan, dtype=float)
    intercept = np.full(flux.shape[0], np.nan, dtype=float)
    cont_left = np.full(flux.shape[0], np.nan, dtype=float)
    cont_right = np.full(flux.shape[0], np.nan, dtype=float)
    cont_rms = np.full(flux.shape[0], np.nan, dtype=float)
    peak_shift = np.zeros(flux.shape[0], dtype=float)

    for i in range(flux.shape[0]):
        valid_i = valid[i]
        init_slope, init_intercept, _, _, _ = fit_linear_continuum_single(
            wave=wave,
            flux_1d=flux[i],
            valid_1d=valid_i,
            left_bounds=line.left,
            right_bounds=line.right,
        )
        shift = 0.0
        if np.isfinite(init_slope) and np.isfinite(init_intercept):
            shift = find_centroid_shift(
                wave=wave,
                flux_1d=flux[i],
                ivar_1d=ivar[i],
                valid_1d=valid_i,
                line_center=line.w,
                init_slope=init_slope,
                init_intercept=init_intercept,
                search_half_width=cfg.peak_search_half_width,
                absorption_snr_threshold=cfg.absorption_snr_threshold,
                no_shift_half_width=cfg.centroid_no_shift_half_width,
            )
        ## use the centroid in the search window after masking >3 sigma absorption;
        ## if centroid stays within +/-1 A of the original center, do not shift.

        peak_shift[i] = shift
        shifted_left = (line.left[0] + shift, line.left[1] + shift)
        shifted_right = (line.right[0] + shift, line.right[1] + shift)
        shifted_centre = (line.centre[0] + shift, line.centre[1] + shift)

        slope_i, intercept_i, left_i, right_i, rms_i = fit_linear_continuum_single(
            wave=wave,
            flux_1d=flux[i],
            valid_1d=valid_i,
            left_bounds=shifted_left,
            right_bounds=shifted_right,
        )
        slope[i] = slope_i
        intercept[i] = intercept_i
        cont_left[i] = left_i
        cont_right[i] = right_i
        cont_rms[i] = rms_i

        if not (np.isfinite(slope_i) and np.isfinite(intercept_i)):
            continue

        centre_mask_i = region_mask(wave, shifted_centre, inclusive_right=True)
        use_i = valid_i & centre_mask_i
        n_line_pix[i] = int(np.count_nonzero(use_i))
        if n_line_pix[i] == 0:
            continue

        continuum_model_i = slope_i * wave + intercept_i
        resid = flux[i, use_i] - continuum_model_i[use_i]
        var = 1.0 / ivar[i, use_i]
        # if np.isfinite(cont_rms[i]):
        #     var = var + cont_rms[i] ** 2
        # no using the continuum rms
        line_flux[i] = np.sum(resid)
        line_err[i] = np.sqrt(np.sum(var))
        if np.isfinite(line_err[i]) and line_err[i] > 0.0:
            ha_snr[i] = line_flux[i] / line_err[i]

    measure_df = pd.DataFrame(
        {
            "ha_wave": line.w,
            "ha_half_window_A": half_window,
            "ha_sigma_lsf_A": sigma_lambda,
            "ha_continuum_left": cont_left,
            "ha_continuum_right": cont_right,
            "ha_continuum_slope": slope,
            "ha_continuum_intercept": intercept,
            "ha_continuum_rms": cont_rms,
            "ha_peak_shift_A": peak_shift,
            "ha_npix": n_line_pix,
            "ha_flux": line_flux,
            "ha_flux_err": line_err,
            "ha_snr": ha_snr,
        }
    )
    
    
    return pd.concat([meta_df.reset_index(drop=True), measure_df], axis=1)




def process_file(task: tuple[str, str]) -> tuple[pd.DataFrame, dict[str, object]]:
    path_str, base_dir_str = task
    path = Path(path_str)
    base_dir = Path(base_dir_str)
    cfg = MeasureConfig()
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        wave = safe_numeric(hdul["WAVE"].data)
        flux = safe_numeric(hdul["FLUX"].data)
        ivar = safe_numeric(hdul["IVAR"].data)
        mask = np.asarray(hdul["MASK"].data)
        lsf = safe_numeric(hdul["LSF"].data)
        slitmap = hdul["SLITMAP"].data

        meta_df = pd.DataFrame(
            {
                "tileid": header_value(header, "TILE_ID", "TILEID"),
                "expnum": header_value(header, "EXPOSURE", "EXPNUM"),
                "mjd": header_value(header, "MJD"),
                "file_name": path.name,
                "file_path": format_relative_path(path, base_dir),
                "fiberid": np.asarray(slitmap["fiberid"]),
                "ra": safe_numeric(slitmap["ra"]),
                "dec": safe_numeric(slitmap["dec"]),
                "targettype": safe_text(slitmap["targettype"]),
                "telescope": safe_text(slitmap["telescope"]),
                "fibstatus": np.asarray(slitmap["fibstatus"]),
            }
        )

        science_mask = (meta_df["targettype"] == "science").to_numpy()
        if not np.any(science_mask):
            return pd.DataFrame(), {}

        flux = flux[science_mask]
        ivar = ivar[science_mask]
        mask = mask[science_mask]
        lsf = lsf[science_mask]
        meta_df = meta_df.loc[science_mask].reset_index(drop=True)

        fiber_df = compute_ha_snr(
            wave=wave,
            flux=flux,
            ivar=ivar,
            mask=mask,
            lsf=lsf,
            meta_df=meta_df,
            cfg=cfg,
        )
    valid_snr = fiber_df["ha_snr"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    if valid_snr.size == 0:
        snr_min = np.nan
        snr_max = np.nan
        snr_med = np.nan
        snr_p16 = np.nan
        snr_p84 = np.nan
        ha_flux_at_snr_med = np.nan
        ha_err_at_snr_med = np.nan
    else:
        snr_min = float(np.nanmin(valid_snr))
        snr_max = float(np.nanmax(valid_snr))
        snr_med = float(np.nanmedian(valid_snr))
        snr_p16 = float(np.nanpercentile(valid_snr, 16))
        snr_p84 = float(np.nanpercentile(valid_snr, 84))
        valid_rows = fiber_df[np.isfinite(fiber_df["ha_snr"])].copy()
        med_idx = (valid_rows["ha_snr"] - snr_med).abs().idxmin()
        ha_flux_at_snr_med = float(fiber_df.loc[med_idx, "ha_flux"])
        ha_err_at_snr_med = float(fiber_df.loc[med_idx, "ha_flux_err"])

    file_row = fiber_df.iloc[0]
    fiber_export = fiber_df[
        [
            "tileid",
            "expnum",
            "mjd",
            "file_name",
            "file_path",
            "fiberid",
            "ra",
            "dec",
            "targettype",
            "telescope",
            "fibstatus",
            "ha_peak_shift_A",
            "ha_flux",
            "ha_flux_err",
            "ha_snr",
        ]
    ]

    summary = {
        "tileid": file_row["tileid"],
        "expnum": file_row["expnum"],
        "mjd": file_row["mjd"],
        "file_name": file_row["file_name"],
        "file_path": file_row["file_path"],
        "n_valid_spectra": int(valid_snr.size),
        "ha_snr_min": snr_min,
        "ha_snr_max": snr_max,
        "ha_snr_med": snr_med,
        "ha_flux_at_snr_med": ha_flux_at_snr_med,
        "ha_err_at_snr_med": ha_err_at_snr_med,
        "ha_snr_16th": snr_p16,
        "ha_snr_84th": snr_p84,
    }
    return fiber_export, summary


def save_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    if not out_path.exists():
        raise OSError(f"Expected output table was not created: {out_path}")


def main() -> int:
    args = parse_args()
    files = find_sframe_files(args)
    if not files:
        raise FileNotFoundError("No SFrame files found. Check --base-dir, --pattern, or --input-files.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.fiber_output is None:
        args.fiber_output = args.output.with_name(f"{args.output.stem}_fibers.parquet")
    args.fiber_output.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    tasks = [(str(path), str(args.base_dir)) for path in files]
    fiber_writer: pq.ParquetWriter | None = None

    try:
        if args.nproc == 1:
            iterator = (process_file(task) for task in tasks)
            for fiber_df, summary in tqdm(iterator, total=len(tasks), desc="Processing SFrames"):
                if fiber_df.empty:
                    continue
                fiber_table = pa.Table.from_pandas(fiber_df, preserve_index=False)
                if fiber_writer is None:
                    fiber_writer = pq.ParquetWriter(args.fiber_output, fiber_table.schema)
                fiber_writer.write_table(fiber_table)
                summary_rows.append(summary)
        else:
            with Pool(processes=args.nproc) as pool:
                iterator = pool.imap(process_file, tasks)
                for fiber_df, summary in tqdm(iterator, total=len(tasks), desc="Processing SFrames"):
                    if fiber_df.empty:
                        continue
                    fiber_table = pa.Table.from_pandas(fiber_df, preserve_index=False)
                    if fiber_writer is None:
                        fiber_writer = pq.ParquetWriter(args.fiber_output, fiber_table.schema)
                    fiber_writer.write_table(fiber_table)
                    summary_rows.append(summary)
    finally:
        if fiber_writer is not None:
            fiber_writer.close()

    summary_df = pd.DataFrame(summary_rows)
    save_table(summary_df, args.output)

    print(f"Processed {len(files)} SFrame files")
    print(f"Wrote summary parquet: {args.output}")
    print(f"Wrote per-fiber parquet: {args.fiber_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
