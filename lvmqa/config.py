"""
Configuration: default locations, versions and the run configuration object.

Path resolution order (for each of redux / analysis / output):
    1. explicit argument (CLI or Python API)
    2. environment variable (production, SDSS tree convention)
    3. development default (test data set on the development machine)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# --- development defaults (test set) -------------------------------------------------
DEV_REDUX_DIR = "/Volumes/iData/Science/LVM/sdsswork/lvm/spectro/redux"
DEV_ANALYSIS_DIR = "/Volumes/iData/Science/LVM/sdsswork/lvm/spectro/analysis"
DEV_OUTPUT_DIR = "/Users/mors/Science/LVM/QA/DR21_dev/Claude"

# --- environment variables used in production ----------------------------------------
ENV_REDUX = "LVM_SPECTRO_REDUX"        # same variable as used by lvmdrp
ENV_ANALYSIS = "LVM_SPECTRO_ANALYSIS"  # SDSS tree name for the DAP output root
ENV_OUTPUT = "LVM_QA_OUTPUT_DIR"

# --- versions -------------------------------------------------------------------------
DEFAULT_DRP_VERSION = "1.3.2"
DEFAULT_DAP_VERSION = "1.2.0.251218"
DEFAULT_DAP_N_MODELS = 108
DEFAULT_DAP_TARGET_SNR = 20


_NO_DAP_VERSION = ("", "none", "null", "-")


def normalize_dap_version(v: Optional[str]) -> Optional[str]:
    """None / '' / 'none' -> None (legacy DR20 layout without a DAP-version folder)."""
    if v is None:
        return None
    v = str(v).strip()
    return None if v.lower() in _NO_DAP_VERSION else v


def _resolve(explicit: Optional[os.PathLike | str], env: str, dev: str) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    val = os.getenv(env)
    if val:
        return Path(val).expanduser()
    return Path(dev)


@dataclass
class PipelineConfig:
    """Everything a worker needs to process one exposure (must stay picklable)."""

    drp_version: str = DEFAULT_DRP_VERSION
    # None -> legacy (DR20 and earlier) layout: analysis/{drpver}/{tilegrp}/... (no DAP-version folder)
    dap_version: Optional[str] = DEFAULT_DAP_VERSION
    redux_dir: Optional[Path] = None
    analysis_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    drpall_file: Optional[Path] = None
    cache_dir: Optional[Path] = None
    dap_n_models: int = DEFAULT_DAP_N_MODELS
    dap_target_snr: int = DEFAULT_DAP_TARGET_SNR
    # per-metric parameter overrides: {"qa_fit_consistency": {"min_snr": 3}, ...}
    metric_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # non-standard input layout (v1.15): SFrame file pattern with an {expnum} placeholder (glob
    # wildcards allowed, zero-padding of the expnum irrelevant), e.g. "/data/test/SFrame-*-{expnum}.fits"
    sframe_pattern: Optional[str] = None
    no_dap: bool = False       # no DAP products: DAP-based metrics are UNDEFINED
    no_drpall: bool = False    # no drpall: exposure table from the SFrame primary headers

    def __post_init__(self) -> None:
        self.dap_version = normalize_dap_version(self.dap_version)
        self.redux_dir = _resolve(self.redux_dir, ENV_REDUX, DEV_REDUX_DIR)
        self.analysis_dir = _resolve(self.analysis_dir, ENV_ANALYSIS, DEV_ANALYSIS_DIR)
        self.output_dir = _resolve(self.output_dir, ENV_OUTPUT, DEV_OUTPUT_DIR)
        if self.drpall_file is None:
            self.drpall_file = (
                self.redux_dir / self.drp_version / f"drpall-{self.drp_version}.fits"
            )
        else:
            self.drpall_file = Path(self.drpall_file).expanduser()
        if self.cache_dir is None:
            self.cache_dir = self.output_dir / "cache"
        else:
            self.cache_dir = Path(self.cache_dir).expanduser()

    @property
    def dap_label(self) -> str:
        return self.dap_version if self.dap_version else "none (legacy layout)"

    def options_for(self, metric_name: str) -> Dict[str, Any]:
        return dict(self.metric_options.get(metric_name, {}))
