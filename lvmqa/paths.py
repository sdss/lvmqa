"""Path conventions for LVM DRP / DAP products."""

from __future__ import annotations

import glob
import logging
import re
from pathlib import Path
from typing import Dict, Optional

from .config import PipelineConfig


def tile_group(tileid: int) -> str:
    """Short tile id: 1056679 -> '1056XX'; 11111 (early science) -> '0011XX'."""
    return f"{int(tileid) // 1000:04d}XX"


def drpall_path(cfg: PipelineConfig) -> Path:
    return Path(cfg.drpall_file)


log = logging.getLogger("lvmqa")
_EXP = re.compile(r"\{expnum(?::[^}]*)?\}")
_SF_INDEX: Dict[str, Dict[int, Path]] = {}


def default_sframe_pattern(cfg: PipelineConfig) -> str:
    """Standard DRP tree as a pattern (used to list exposures without drpall)."""
    return str(Path(cfg.redux_dir) / cfg.drp_version / "*" / "*" / "*" / "lvmSFrame-{expnum}.fits")


def sframe_index(pattern: str) -> Dict[int, Path]:
    """{expnum: path} of all files matching ``pattern`` (one glob, cached per process). The
    {expnum} placeholder matches any digits (zero-padded or not); '*' / '?' are glob wildcards.
    Several files for one expnum: the first one (sorted) is used, with a warning."""
    if pattern in _SF_INDEX:
        return _SF_INDEX[pattern]
    if not _EXP.search(pattern):
        raise ValueError(f"SFrame pattern without an {{expnum}} placeholder: {pattern}")
    parts = _EXP.split(pattern)
    rx = "(\\d+)".join(re.escape(x).replace("\\*", "[^/]*").replace("\\?", "[^/]") for x in parts)
    rx = re.compile("^" + rx + "$")
    out: Dict[int, Path] = {}
    dup = 0
    for f in sorted(glob.glob(_EXP.sub("*", pattern))):
        m = rx.match(f)
        if not m:
            continue
        e = int(m.group(1))
        if e in out:
            dup += 1
            continue
        out[e] = Path(f)
    if dup:
        log.warning("SFrame pattern: %d expnum(s) matched by several files; the first (sorted) is used", dup)
    _SF_INDEX[pattern] = out
    return out


def sframe_path(cfg: PipelineConfig, tileid: int, mjd: int, expnum: int,
                tilegrp: Optional[str] = None) -> Path:
    if getattr(cfg, "sframe_pattern", None):
        p = sframe_index(cfg.sframe_pattern).get(int(expnum))
        return p if p is not None else Path(_EXP.sub(str(int(expnum)), cfg.sframe_pattern))
    tilegrp = tilegrp or tile_group(tileid)
    return (Path(cfg.redux_dir) / cfg.drp_version / tilegrp / str(int(tileid))
            / str(int(mjd)) / f"lvmSFrame-{int(expnum):08d}.fits")


def dap_path(cfg: PipelineConfig, tileid: int, mjd: int, expnum: int,
             tilegrp: Optional[str] = None) -> Path:
    """
    Expected DAP file; falls back to any dap-*-{expnum}.dap.fits[.gz] in the folder.

    Layout: analysis/{drpver}/{dapver}/{tilegrp}/{tileid}/{mjd}/{expnum:08d}/
            or, if cfg.dap_version is None (DR20 and earlier),
            analysis/{drpver}/{tilegrp}/{tileid}/{mjd}/{expnum:08d}/
    """
    if getattr(cfg, "no_dap", False):     # no DAP products: a path that does not exist
        return Path(cfg.analysis_dir) / "__no_dap__" / f"{int(expnum):08d}.dap.fits.gz"
    tilegrp = tilegrp or tile_group(tileid)
    root = Path(cfg.analysis_dir) / cfg.drp_version
    if cfg.dap_version:
        root = root / cfg.dap_version
    folder = root / tilegrp / str(int(tileid)) / str(int(mjd)) / f"{int(expnum):08d}"
    expected = folder / (f"dap-rsp{cfg.dap_n_models}-sn{cfg.dap_target_snr}-"
                         f"{int(expnum):08d}.dap.fits.gz")
    if expected.exists():
        return expected
    if folder.is_dir():
        cands = sorted(folder.glob(f"dap-*-{int(expnum):08d}.dap.fits*"))
        if cands:
            return cands[0]
    return expected


def dap_output_path(dap_file: Path) -> Path:
    """DAP model RSS ('output' file) next to the DAP table file:
    dap-rsp108-sn20-XXXXXXXX.dap.fits.gz -> dap-rsp108-sn20-XXXXXXXX.output.fits.gz
    (falls back to an uncompressed .output.fits)."""
    dap_file = Path(dap_file)
    name = dap_file.name
    base = name.split(".dap.fits")[0] if ".dap.fits" in name else name
    cand = dap_file.with_name(f"{base}.output.fits.gz")
    if not cand.exists() and dap_file.with_name(f"{base}.output.fits").exists():
        return dap_file.with_name(f"{base}.output.fits")
    return cand
