"""
Shared sky-line measurements for qa_wavecal_* and qa_lsf (computed once per exposure via
ExposureContext.shared).

For every good science fiber the bright, isolated sky lines are fitted with a Gaussian +
constant in the *un-subtracted* spectrum FLUX + SKY (the sky lines are not present in FLUX
itself). The wavelengths of the SFrame are topocentric (the heliocentric correction is only
recorded in the header, WAVE HELIORV_*), so the measured centroids are compared directly with
the air rest wavelengths of the lines. The measured FWHM is compared with the per-fiber LSF
(FWHM, SFrame extension LSF) at the line wavelength.

Line list and channel assignment as in DR20 (scripts/dr20_bitmasks_from_measurements.py);
lines marked (*) are also used by the DRP to correct the wavelength zero point
(lvmdrp REF_SKYLINES, functions/rssMethod.shift_wave_skylines), so for them the test verifies
that the correction was applied, and their reference wavelengths are taken from the DRP
(differences to the DR20 values are <= 0.06 A); the other lines test the shape of the
wavelength solution. Line-to-line systematic offsets of up to ~0.05 A (unresolved OH
Lambda-doublets) are well below the LSF/5 ~ 0.26-0.36 A tolerance.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..utils.linefit import FWHM_PER_SIGMA, fit_gaussians

# name: (air rest wavelength [A], channel)
SKY_LINES: Dict[str, Tuple[float, str]] = {
    "sky5577": (5577.347, "b"),  # (*) [OI]
    "sky6533": (6533.01, "r"),
    "sky6923": (6923.15, "r"),
    "sky6939": (6939.55, "r"),
    "sky7392": (7392.210, "r"),  # (*)
    "sky7914": (7913.66, "z"),
    "sky8344": (8344.57, "z"),
    "sky8399": (8399.176, "z"),  # (*)
    "sky8827": (8827.07, "z"),
    "sky8988": (8988.384, "z"),  # (*)
    "sky9552": (9552.547, "z"),  # (*)
    "sky9719": (9719.839, "z"),  # (*)
}
CHANNELS = ("b", "r", "z")

HALF_WINDOW = 3.0     # [A] fit window around the rest wavelength
MIN_SNR = 10.0        # amplitude S/N of an accepted fit


def lines_of(channel: str):
    return [k for k, (_, ch) in SKY_LINES.items() if ch == channel]


def measure_skylines(ctx, half_window: float = HALF_WINDOW, min_snr: float = MIN_SNR) -> Dict:
    """
    Returns dict with
      rows   : SFrame row indices of the measured fibers (good science fibers)
      specid : spectrograph id per measured fiber
      lines  : {name: dict(rest, channel, mu, e_mu, fwhm, e_fwhm, lsf, amp, snr, ok)}
    All per-line arrays have one entry per measured fiber; ``ok`` marks accepted fits
    (converged, amplitude S/N >= min_snr, centroid within the window).
    """
    rows = ctx.good_sci_index
    specid = np.asarray(ctx.slitmap["spectrographid"])[rows]
    wave = ctx.wave
    out = {"rows": rows, "specid": specid, "lines": {}}
    for name, (rest, ch) in SKY_LINES.items():
        w0, w1 = rest - half_window, rest + half_window
        if w0 < wave[0] or w1 > wave[-1]:
            continue
        x, f = ctx.sframe_window("FLUX", w0, w1, rows=rows)
        _, sk = ctx.sframe_window("SKY", w0, w1, rows=rows)
        _, iv = ctx.sframe_window("IVAR", w0, w1, rows=rows)
        _, lsf = ctx.sframe_window("LSF", w0, w1, rows=rows)
        y = f + sk
        j = int(np.argmin(np.abs(x - rest)))
        lsf_line = lsf[:, j]
        sig0 = np.where(np.isfinite(lsf_line) & (lsf_line > 0), lsf_line, 1.5) / FWHM_PER_SIGMA
        # initial centroid: maximum within +-1 A of the rest wavelength
        core = np.abs(x - rest) <= 1.0
        with np.errstate(invalid="ignore"):
            yc = np.where(np.isfinite(y[:, core]), y[:, core], -np.inf)
        mu0 = x[core][np.argmax(yc, axis=1)]
        r = fit_gaussians(x, y, iv, mu0=mu0, sig0=sig0)
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = r["amp"] / r["e_amp"]
        ok = r["ok"] & (snr >= min_snr) & (np.abs(r["mu"] - rest) < half_window - 1.0)
        out["lines"][name] = dict(rest=rest, channel=ch, mu=r["mu"], e_mu=r["e_mu"],
                                  fwhm=r["fwhm"], e_fwhm=r["e_sig"] * FWHM_PER_SIGMA,
                                  lsf=lsf_line, amp=r["amp"], snr=snr, ok=ok)
    return out


def skylines(ctx) -> Dict:
    """Cached per exposure."""
    return ctx.shared("skylines", measure_skylines)
