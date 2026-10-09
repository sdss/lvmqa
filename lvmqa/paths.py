"""Path conventions for LVM DRP / DAP products."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import PipelineConfig


def tile_group(tileid: int) -> str:
    """Short tile id: 1056679 -> '1056XX'; 11111 (early science) -> '0011XX'."""
    return f"{int(tileid) // 1000:04d}XX"


def drpall_path(cfg: PipelineConfig) -> Path:
    return Path(cfg.drpall_file)


def sframe_path(cfg: PipelineConfig, tileid: int, mjd: int, expnum: int,
                tilegrp: Optional[str] = None) -> Path:
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
