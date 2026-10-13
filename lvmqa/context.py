"""
Per-exposure context: lazy, cached access to all inputs a metric may need.

Inputs (and nothing else): drpall row, SFrame (header / SLITMAP / data arrays) and the
DAP output file. Everything is loaded on first use and shared between metrics.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .config import PipelineConfig
from .paths import dap_output_path, dap_path, sframe_path, tile_group
from .registry import MissingInputError

log = logging.getLogger("lvmqa")

# Non-overlapping "core" wavelength windows of the three spectrograph channels [A]
CHANNEL_WINDOWS: Dict[str, tuple] = {
    "b": (3650.0, 5750.0),
    "r": (5850.0, 7450.0),
    "z": (7650.0, 9750.0),
}


def _is_valid(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (float, np.floating)):
        return math.isfinite(float(v))
    if isinstance(v, str):
        return v.strip() != ""
    return True


class ExposureContext:
    def __init__(self, cfg: PipelineConfig, drpall_row: Dict[str, Any]):
        self.cfg = cfg
        self.row = drpall_row
        self.expnum = int(drpall_row["expnum"])
        self.mjd = int(drpall_row["mjd"])
        self.tileid = int(drpall_row["tileid"])
        tg = str(drpall_row.get("tilegrp", "") or "").strip()
        self.tilegrp = tg or tile_group(self.tileid)
        self.sframe_file: Path = sframe_path(cfg, self.tileid, self.mjd, self.expnum, self.tilegrp)
        self.dap_file: Path = dap_path(cfg, self.tileid, self.mjd, self.expnum, self.tilegrp)
        self.dap_output_file: Path = dap_output_path(self.dap_file)
        self._output_requests: Dict[str, tuple] = {}
        self._sframe_hdul: Optional[fits.HDUList] = None
        self._dap_hdul: Optional[fits.HDUList] = None
        self._dap_tables: Dict[str, Table] = {}
        # accumulated input/output time [s]: 'io_sframe', 'io_dap'
        self.timings: Dict[str, float] = defaultdict(float)

    @contextmanager
    def timed(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] += time.perf_counter() - t0

    @property
    def io_time(self) -> float:
        """Time spent outside metric code: input I/O plus shared computations."""
        return float(sum(self.timings.values()))

    def shared(self, key: str, func):
        """
        Result of ``func(ctx)`` computed once per exposure and shared between metrics
        (e.g. the sky-line Gaussian fits used by qa_wavecal_* and qa_lsf). Its compute time
        (without the I/O it triggers) is accumulated as timings['shared_<key>'], so it is not
        attributed to whichever metric happens to call it first. Exceptions are cached too.
        """
        cache = self.__dict__.setdefault("_shared", {})
        if key not in cache:
            t0, io0 = time.perf_counter(), self.io_time
            try:
                cache[key] = (True, func(self))
            except Exception as exc:  # noqa: BLE001
                cache[key] = (False, exc)
            self.timings[f"shared_{key}"] += (time.perf_counter() - t0) - (self.io_time - io0)
        ok, val = cache[key]
        if not ok:
            raise val
        return val

    # ------------------------------------------------------------------ availability
    @cached_property
    def has_sframe(self) -> bool:
        return self.sframe_file.is_file()

    @cached_property
    def has_dap(self) -> bool:
        return self.dap_file.is_file()

    @cached_property
    def has_dap_output(self) -> bool:
        return self.dap_output_file.is_file()

    # ------------------------------------------------------------------ drpall / header
    def drp(self, key: str, default: Any = None) -> Any:
        v = self.row.get(key, default)
        return v if _is_valid(v) else default

    def value(self, drpall_key: Optional[str], header_keys: Iterable[str] = (),
              default: Any = None) -> Any:
        """drpall value if valid, else the first valid SFrame header keyword."""
        if drpall_key:
            v = self.drp(drpall_key)
            if v is not None:
                return v
        if header_keys and self.has_sframe:
            for k in header_keys:
                v = self.header.get(k)
                if _is_valid(v):
                    return v
        return default

    # ------------------------------------------------------------------ SFrame
    def _sframe(self) -> fits.HDUList:
        if not self.has_sframe:
            raise MissingInputError(f"SFrame not found: {self.sframe_file.name}")
        if self._sframe_hdul is None:
            with self.timed("io_sframe"):
                self._sframe_hdul = fits.open(self.sframe_file, memmap=True, lazy_load_hdus=True)
        return self._sframe_hdul

    @cached_property
    def header(self) -> fits.Header:
        hdul = self._sframe()
        with self.timed("io_sframe"):
            return hdul[0].header

    @cached_property
    def slitmap(self) -> Table:
        hdul = self._sframe()
        with self.timed("io_sframe"):
            return Table(hdul["SLITMAP"].data)

    @cached_property
    def wave(self) -> np.ndarray:
        hdul = self._sframe()
        with self.timed("io_sframe"):
            return np.asarray(hdul["WAVE"].data, dtype=float)

    def sframe_data(self, extname: str) -> np.ndarray:
        """Memory-mapped data array of an SFrame image extension (FLUX, IVAR, MASK, ...)."""
        hdul = self._sframe()
        with self.timed("io_sframe"):
            return hdul[extname].data

    def sframe_window(self, extname: str, wmin: float, wmax: float,
                      rows: Optional[np.ndarray] = None):
        """
        (wave, data) of an SFrame image extension restricted to wmin <= wave <= wmax
        (and to the given fiber rows). Read into memory and cached per (ext, window).
        """
        key = (extname, float(wmin), float(wmax), None if rows is None else rows.tobytes())
        cache = self.__dict__.setdefault("_window_cache", {})
        if key not in cache:
            wave = self.wave
            i0, i1 = np.searchsorted(wave, [wmin, wmax], side="left")
            i1 = min(i1 + 1, wave.size)
            arr = self.sframe_data(extname)
            with self.timed("io_sframe"):
                data = np.array(arr[:, i0:i1] if rows is None else arr[rows, i0:i1], dtype=float)
            cache[key] = (wave[i0:i1], data)
        return cache[key]

    @cached_property
    def sci_fiber_index(self) -> np.ndarray:
        """Row indices of science-IFU fibers in the SFrame arrays."""
        sm = self.slitmap
        return np.where(np.char.strip(np.asarray(sm["telescope"]).astype(str)) == "Sci")[0]

    @cached_property
    def good_sci_index(self) -> np.ndarray:
        """Science fibers with SLITMAP fibstatus == 0."""
        sm = self.slitmap
        idx = self.sci_fiber_index
        return idx[np.asarray(sm["fibstatus"])[idx] == 0]

    @cached_property
    def fiber_good_fraction(self) -> Dict[str, np.ndarray]:
        """
        Fraction of usable pixels per fiber (all fibers) in each channel window:
        finite flux, finite ivar > 0 and MASK == 0.
        """
        wave = self.wave
        flux = self.sframe_data("FLUX")
        ivar = self.sframe_data("IVAR")
        mask = self.sframe_data("MASK")
        out: Dict[str, np.ndarray] = {}
        for ch, (w0, w1) in CHANNEL_WINDOWS.items():
            i0, i1 = np.searchsorted(wave, [w0, w1])
            with self.timed("io_sframe"):  # memmap: the actual disk read happens here
                f = np.array(flux[:, i0:i1])   # np.array forces the read (copy)
                iv = np.array(ivar[:, i0:i1])
                m = np.array(mask[:, i0:i1])
            good = np.isfinite(f) & np.isfinite(iv) & (iv > 0) & (m == 0)
            out[ch] = good.mean(axis=1) if good.shape[1] else np.zeros(good.shape[0])
        return out

    # ------------------------------------------------------------------ DAP
    def _dap(self) -> fits.HDUList:
        if not self.has_dap:
            raise MissingInputError(f"DAP file not found for expnum {self.expnum}")
        if self._dap_hdul is None:
            with self.timed("io_dap"):
                self._dap_hdul = fits.open(self.dap_file)
        return self._dap_hdul

    def dap_table(self, extname: str) -> Table:
        if extname not in self._dap_tables:
            hdul = self._dap()
            with self.timed("io_dap"):
                if extname not in hdul:
                    raise MissingInputError(f"DAP extension {extname} missing")
                self._dap_tables[extname] = Table(hdul[extname].data)
        return self._dap_tables[extname]

    def dap_np_columns(self, colnames: Iterable[str]) -> Table:
        """
        Collect columns from the non-parametric emission-line extensions
        (NP_ELINES_B/R/I), aligned on ``id``. Missing columns are filled with NaN.
        """
        colnames = list(colnames)
        base = self.dap_table("PT")
        out = Table({"id": np.asarray(base["id"]).astype(str)})
        index = {k: i for i, k in enumerate(out["id"])}
        for c in colnames:
            out[c] = np.full(len(out), np.nan)
        for ext in ("NP_ELINES_B", "NP_ELINES_R", "NP_ELINES_I"):
            try:
                t = self.dap_table(ext)
            except MissingInputError:
                continue
            present = [c for c in colnames if c in t.colnames]
            if not present:
                continue
            rows = np.array([index.get(k, -1) for k in np.asarray(t["id"]).astype(str)])
            ok = rows >= 0
            for c in present:
                out[c][rows[ok]] = np.asarray(t[c], dtype=float)[ok]
        return out

    # ------------------------------------------------------------------ DAP output (model RSS)
    # planes of the output cube (NAMEk keywords of lvmdap)
    OUT_PLANES = {"org_spec": 0, "model_spec": 1, "mod_joint_spec": 2, "gas_spec": 3,
                  "res_joint_spec": 4, "no_gas_spec": 5, "gas_model_NP": 6,
                  "gas_model_PEK": 7, "mod_joint_spec_PEK": 8}

    def request_output(self, windows: Dict[str, tuple]) -> None:
        """Register DAP-output windows {key: (wmin, wmax, planes)} before the first read."""
        for k, (w0, w1, planes) in windows.items():
            if "_output_cache" in self.__dict__:
                raise RuntimeError("DAP output already read; request windows before computing")
            if k in self._output_requests:
                o0, o1, op = self._output_requests[k]
                self._output_requests[k] = (min(o0, w0), max(o1, w1), tuple(sorted(set(op) | set(planes))))
            else:
                self._output_requests[k] = (float(w0), float(w1), tuple(sorted(set(planes))))

    @property
    def _output(self) -> Dict[str, Any]:
        cache = self.__dict__.setdefault("_output_cache", {})
        if "v" not in cache:
            try:
                cache["v"] = (True, self._read_output())
            except Exception as exc:  # noqa: BLE001  (cache failures: the read is expensive)
                cache["v"] = (False, exc)
        ok, val = cache["v"]
        if not ok:
            raise val
        return val

    def _read_output(self) -> Dict[str, Any]:
        """
        One sequential pass through the (gzipped) DAP output cube (planes x fibers x pixels,
        ~0.9 GB float32 uncompressed): only the requested windows / planes are kept (float32).
        Random access (astropy section) on a gzip stream is prohibitively slow.
        """
        if not self.has_dap_output:
            raise MissingInputError(f"DAP output file not found: {self.dap_output_file.name}")
        import gzip
        with self.timed("io_dap_output"):
            path = self.dap_output_file
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rb") as fh:
                hdr = b""
                while True:
                    blk = fh.read(2880)
                    if len(blk) < 2880:
                        raise MissingInputError(f"truncated DAP output header: {path.name}")
                    hdr += blk
                    if any(blk[i:i + 8] == b"END     " for i in range(0, 2880, 80)):
                        break
                h = fits.Header.fromstring(hdr.decode("ascii", errors="replace"))
                nx, ny = int(h["NAXIS1"]), int(h["NAXIS2"])
                nz = int(h.get("NAXIS3", 1))
                if int(h["BITPIX"]) != -32:
                    raise MissingInputError(f"unexpected BITPIX {h['BITPIX']} in {path.name}")
                wave = h["CRVAL1"] + h["CDELT1"] * (np.arange(nx) + 1 - h["CRPIX1"])
                names = {int(k[4:]): str(h[k]).strip() for k in h if k.startswith("NAME") and k[4:].isdigit()}
                req = self._output_requests
                need = set(p for (_, _, planes) in req.values() for p in planes)
                data: Dict[str, Dict[int, np.ndarray]] = {k: {} for k in req}
                sl = {k: np.searchsorted(wave, [w0, w1 + 1e-6]) for k, (w0, w1, _) in req.items()}
                for z in range(nz):
                    if z > max(need, default=-1):
                        break
                    buf = fh.read(nx * ny * 4)
                    if len(buf) < nx * ny * 4:
                        raise MissingInputError(f"truncated DAP output data: {path.name}")
                    if z not in need:
                        continue
                    a = np.frombuffer(buf, dtype=">f4").reshape(ny, nx)
                    for k, (w0, w1, planes) in req.items():
                        if z in planes:
                            i0, i1 = sl[k]
                            data[k][z] = a[:, i0:i1].astype(np.float32)
        return {"wave": wave, "header": h, "names": names, "data": data, "sl": sl}

    def dap_output(self, key: str):
        """(wave, {plane: array (n_PT_fibers, n_pix)}) of a requested DAP-output window."""
        out = self._output
        if key not in out["data"]:
            raise KeyError(f"DAP output window {key!r} was not requested")
        i0, i1 = out["sl"][key]
        return out["wave"][i0:i1], out["data"][key]

    @cached_property
    def dap_sframe_rows(self) -> np.ndarray:
        """SFrame row index of every DAP (PT) fiber, in PT order (-1 if not found)."""
        pt = self.dap_table("PT")
        fid = np.asarray(self.slitmap["fiberid"]).astype(int)
        order = np.argsort(fid)
        want = np.asarray(pt["fiberid"]).astype(int)
        pos = np.searchsorted(fid[order], want)
        pos = np.clip(pos, 0, len(fid) - 1)
        rows = order[pos]
        rows[fid[rows] != want] = -1
        return rows

    def dap_sframe_window(self, extname: str, wmin: float, wmax: float):
        """SFrame data (wave, array) for the DAP (PT) fibers in PT order (NaN for unmatched)."""
        rows = self.dap_sframe_rows
        ok = rows >= 0
        w, d = self.sframe_window(extname, wmin, wmax, rows=rows[ok])
        out = np.full((rows.size, d.shape[1]), np.nan)
        out[ok] = d
        return w, out

    # ------------------------------------------------------------------ cleanup
    def close(self) -> None:
        for h in (self._sframe_hdul, self._dap_hdul):
            if h is not None:
                try:
                    h.close()
                except Exception:  # pragma: no cover
                    pass
        self._sframe_hdul = self._dap_hdul = None
        self._dap_tables.clear()
        self.__dict__.pop("_window_cache", None)
        self.__dict__.pop("_shared", None)
        self.__dict__.pop("_output_cache", None)

    def __enter__(self) -> "ExposureContext":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
