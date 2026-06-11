"""
DRPCubeQA.py

Quality analysis utilities for LVM SFrame FITS datacubes.
"""

from __future__ import print_function
import os
import re
from pathlib import Path
import json
import warnings

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator, LogLocator
from astropy.stats import sigma_clip

class DRPCubeQA(object):
    """
    Quality analysis utilities for LVM "SFrame" FITS datacubes.

    Usage:
        qa = DRPCubeQA(path)
        report = qa.run_full_report(save_json='qa.json')
        qa.close()
    """

    def __init__(self, path, expnum=None, memmap=True, verbose=True,
                 zero_frac_warn_threshold=0.05,
                 arm_jump_threshold_warn = 1.0,
                 arm_jump_threshold_bad = 2.0,
                 warn_frac_threshold = 0.05,
                 bad_frac_threshold = 0.05,
                 log_path='./logs'):
        """
        Open the FITS file and load all relevant HDUs into memory.

        Parameters
        ----------
        path : str
            Path to FITS file
        expnum: int
            Exposure number as unique ID
        memmap : bool
            Use memory mapping when opening FITS
        verbose : bool
            Print messages to stdout
        zero_frac_warn_threshold : float
            Threshold for zero/NaN fraction warnings
        arm_jump_threshold_warn : float
            Threshold for continuum jump between arms (abs(jump)/mad_std) for flagging as warning
        arm_jump_threshold_bad : float
            Threshold for continuum jump between arms (abs(jump)/mad_std) for flagging as bad
        warn_frac_threshold : float
            Threshold for fraction of fibers with warning to trigger exposure-level warning flag
        bad_frac_threshold : float
            Threshold for fraction of bad fibers to trigger exposure-level bad flag
        log_path : str
            Directory where log files are saved
        """
        self.path = path
        self.expnum = expnum
        self.memmap = memmap
        self.verbose = verbose
        self.zero_frac_warn_threshold = zero_frac_warn_threshold
        self.arm_jump_threshold_warn = arm_jump_threshold_warn
        self.arm_jump_threshold_bad = arm_jump_threshold_bad
        self.warn_frac_threshold = warn_frac_threshold
        self.bad_frac_threshold = bad_frac_threshold
        
        self.log_dir = Path(log_path)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with fits.open(path, memmap=memmap) as data_file:
            self.hdr = data_file[0].header

            try:
                self._flux = data_file[1].data.copy() if data_file[1].data is not None else None
            except Exception:
                self._flux = None
            try:
                self._ivar = data_file[2].data.copy() if data_file[2].data is not None else None
            except Exception:
                self._ivar = None
            try:
                self._mask = data_file[3].data.copy() if len(data_file) > 3 and data_file[3].data is not None else None
            except Exception:
                self._mask = None
            try:
                self._wave = data_file[4].data.copy() if len(data_file) > 4 and data_file[4].data is not None else None
            except Exception:
                self._wave = None
            try:
                self._lsf = data_file[5].data.copy() if len(data_file) > 5 and data_file[5].data is not None else None
            except Exception:
                self._lsf = None
            try:
                self._sky = data_file[6].data.copy() if len(data_file) > 6 and data_file[6].data is not None else None
            except Exception:
                self._sky = None
            try:
                self._slitmap = data_file[-1].data.copy() if data_file[-1].data is not None else None
            except Exception:
                self._slitmap = None

        if self._flux is None:
            raise RuntimeError("FLUX HDU (index 1) missing or empty.")
        if self._wave is None:
            raise RuntimeError("WAVE HDU (index 4) missing.")
        if self._ivar is None:
            raise RuntimeError("IVAR HDU (index 2) missing.")

        inferred_nwave = np.asarray(self._wave).size

        fshape = self._flux.shape
        # if flux is 2D
        if getattr(self._flux, "ndim", None) == 2:
            if inferred_nwave is not None:
                if fshape[0] == inferred_nwave and fshape[1] != inferred_nwave:
                    if self.verbose:
                        print("Detected flux shaped (n_wave, n_fibers). Transposing to (n_fibers, n_wave).")
                    self._flux = self._flux.T
                    if self._ivar is not None and getattr(self._ivar, "ndim", None) == 2 and self._ivar.shape == fshape:
                        self._ivar = self._ivar.T
                    if self._sky is not None and getattr(self._sky, "ndim", None) == 2 and self._sky.shape == fshape:
                        self._sky = self._sky.T
                    if self._lsf is not None and getattr(self._lsf, "ndim", None) >= 2 and self._lsf.shape[0] == fshape[0]:
                        try:
                            self._lsf = np.swapaxes(self._lsf, 0, 1)
                        except Exception:
                            pass
            else:
                if fshape[0] > fshape[1]:
                    # suspect shape is (n_wave, n_fibers) -> transpose
                    if self.verbose:
                        print("No WAVE length available. Heuristically transposing FLUX to (n_fibers, n_wave).")
                    self._flux = self._flux.T
                    if self._ivar is not None and getattr(self._ivar, "ndim", None) == 2 and self._ivar.shape == fshape:
                        self._ivar = self._ivar.T
                    if self._sky is not None and getattr(self._sky, "ndim", None) == 2 and self._sky.shape == fshape:
                        self._sky = self._sky.T
                    if self._lsf is not None and getattr(self._lsf, "ndim", None) >= 2 and self._lsf.shape[0] == fshape[0]:
                        try:
                            self._lsf = np.swapaxes(self._lsf, 0, 1)
                        except Exception:
                            pass
        else:
            raise RuntimeError("FLUX HDU is not 2D.")

        # After orientation fix, identify sizes
        self.n_fibers = self._flux.shape[0]
        self.n_wave = self._flux.shape[1]
        if self.verbose:
            print("Data read. fibers = %d, wavelengths = %d" % (self.n_fibers, self.n_wave))

    @staticmethod
    def _normalize_parts(input_parts):
        """
        Normalize a path string/Path or list into cleaned list of parts.
        """
        if isinstance(input_parts, (str, Path)):
            p = Path(input_parts)
            parts = list(p.parts)
        else:
            parts = list(input_parts)
        cleaned = [str(x) for x in parts if str(x) not in ("", "/", "\\")]
        return cleaned

    def get_flux(self):
        """
        Return flux array shaped (n_fibers, n_wave).
        """
        return self._flux

    def get_ivar(self):
        """
        Return ivar array (None if missing). Expected shape (n_fibers, n_wave).
        """
        return self._ivar

    def get_wave(self):
        """
        Return wave vector (1D) if present.
        """
        return self._wave

    def get_sky(self):
        """
        Return sky array if present (n_fibers, n_wave) or None.
        """
        return self._sky

    def get_lsf(self):
        """
        Return LSF array if present.
        """
        return self._lsf

    def get_slitmap(self):
        """
        Return slitmap (BinTable) if present.
        """
        return self._slitmap

    # -------------------------
    # HDU zero/NaN checks
    # -------------------------
    def check_hdu_zero_nan(self, arr):
        """
        Given a numpy array, compute fraction of zeros, NaNs, and finite values.
        Returns dict with zero_frac, nan_frac, finite_frac, total_sampled.
        """
        if arr is None:
            return {"present": False, "zero_frac": None, "nan_frac": None, "finite_frac": None, "total_sampled": 0}
        a = np.asarray(arr)
        if a.size == 0:
            return {"present": True, "zero_frac": None, "nan_frac": None, "finite_frac": None, "total_sampled": 0}
        total = a.size
        n_zero = int(np.count_nonzero(a == 0))
        n_nan = int(np.count_nonzero(~np.isfinite(a)))
        n_finite = int(np.count_nonzero(np.isfinite(a)))
        return {"present": True, "zero_frac": float(n_zero) / total,
                "nan_frac": float(n_nan) / total,
                "finite_frac": float(n_finite) / total,
                "total_sampled": int(total)}

    def check_all_hdus_zero_nan(self):
        """
        Run check_hdu_zero_nan for main arrays and print warnings when thresholds exceeded.
        """
        results = {}
        mapping = {
            "FLUX": self._flux,
            "IVAR": self._ivar,
            "MASK": self._mask,
            "SKY": self._sky,
            "LSF": self._lsf,
            "WAVE": self._wave,
        }
        
        for name, arr in mapping.items():
            res = self.check_hdu_zero_nan(arr)
            results[name] = res
            if not res["present"]:
                if self.verbose:
                    print("HDU %s: not present." % name)
                continue
            if res["total_sampled"] == 0:
                if self.verbose:
                    print("HDU %s: present but empty." % name)
                continue

            if name == "MASK":
                bad_frac = 1.0 - res["zero_frac"]
                if bad_frac >= self.zero_frac_warn_threshold:
                    warnings.warn("MASK HDU: large fraction of flagged pixels: %.3f (threshold %.3f)"
                                  % (bad_frac, self.zero_frac_warn_threshold) )
            else:
                if res["zero_frac"] is not None and res["zero_frac"] >= self.zero_frac_warn_threshold:
                    warnings.warn("HDU %s: large fraction of zeros: %.3f (threshold %.3f)"
                                  % (name, res["zero_frac"], self.zero_frac_warn_threshold) )
            if res["nan_frac"] is not None and res["nan_frac"] >= self.zero_frac_warn_threshold:
                warnings.warn("HDU %s: large fraction of NaN/non-finite: %.3f (threshold %.3f)" %
                             (name, res["nan_frac"], self.zero_frac_warn_threshold))
            if self.verbose:
                print("HDU %s: zero_frac=%.4f, nan_frac=%.4f, finite_frac=%.4f" %
                     (name, res.get("zero_frac", float('nan')),
                      res.get("nan_frac", float('nan')), res.get("finite_frac", float('nan'))))
        return results

    def header_summary(self):
        """
        Return a compact summary extracted from the primary header as a dict.
        """
        h = self.hdr
        keys = ["FILENAME", "MJD", "EXPTIME", "QAQUAL", "QAFLAG", "SURVEY", "OBJECT", "SPEC"]
        out = {}
        for k in keys:
            out[k] = h.get(k) if k in h else None
        return out

    def check_hdu_consistency(self):
        """
        Check that IVAR/MASK shapes match FLUX shape and WAVE length matches FLUX second axis.
        """
        notes = {}
        notes["flux_shape"] = getattr(self._flux, "shape", None)
        if self._ivar is not None:
            notes["ivar_shape"] = getattr(self._ivar, "shape", None)
            notes["ivar_matches_flux"] = (notes["ivar_shape"] == notes["flux_shape"])
        else:
            notes["ivar_shape"] = None
            notes["ivar_matches_flux"] = False

        if self._mask is not None:
            notes["mask_shape"] = getattr(self._mask, "shape", None)
            notes["mask_matches_flux"] = (notes["mask_shape"] == notes["flux_shape"])
        else:
            notes["mask_shape"] = None
            notes["mask_matches_flux"] = False

        if self._wave is not None:
            notes["wave_length"] = np.asarray(self._wave).size
            notes["wave_matches_flux"] = (notes["wave_length"] == self.n_wave)
        else:
            notes["wave_length"] = None
            notes["wave_matches_flux"] = False

        if self._lsf is not None:
            notes["lsf_shape"] = getattr(self._lsf, "shape", None)
        else:
            notes["lsf_shape"] = None

        return notes

    def compute_mask_fraction(self):
        """
        Compute fraction of pixels flagged in MASK (MASK > 0 considered flagged).
        """
        if self._mask is None:
            warnings.warn("MASK HDU missing; returning 0.0")
            return 0.0
        mask = np.asarray(self._mask)
        total = mask.size
        nflag = int(np.count_nonzero(mask > 0))
        frac = float(nflag) / float(total) if total > 0 else 0.0
        if self.verbose:
            print("Mask fraction (full): %.4f (%d/%d)" % (frac, nflag, total))
        return frac

    def compute_snr_stats(self):
        """
        Compute S/N statistics using FLUX and IVAR on the full arrays.

        sigma = 1/sqrt(ivar) where ivar>0
        snr = flux * sqrt(ivar)

        Returns a dict with median, p16, p84, fraction above thresholds, and n_valid.
        """
        if self._ivar is None:
            raise RuntimeError("IVAR missing. Can't compute S/N.")

        flux = np.asarray(self._flux).astype(np.float32)
        ivar = np.asarray(self._ivar).astype(np.float32)

        # Ensure shapes match
        if flux.shape != ivar.shape:
            raise RuntimeError("FLUX and IVAR shapes don't match: %s vs %s" % (flux.shape, ivar.shape))

        ivar_pos = np.where(np.isfinite(ivar) & (ivar > 0), ivar, 0.0)
        sqrt_ivar = np.sqrt(ivar_pos, where=(ivar_pos > 0), out=np.zeros_like(ivar_pos))
        snr = flux * sqrt_ivar
        valid_mask = ivar_pos > 0
        snr_valid = snr[valid_mask]

        if snr_valid.size == 0:
            if self.verbose:
                print("No valid IVAR>0 pixels found in full arrays.")
            return {"median": float("nan"), "p16": float("nan"), "p84": float("nan"),
                    "frac_snr_gt_1": 0.0, "frac_snr_gt_3": 0.0, "frac_snr_gt_5": 0.0,
                    "n_valid": 0}

        median = float(np.median(snr_valid))
        p16 = float(np.percentile(snr_valid, 16))
        p84 = float(np.percentile(snr_valid, 84))

        frac1 = float(np.count_nonzero(snr_valid > 1) / snr_valid.size)
        frac3 = float(np.count_nonzero(snr_valid > 3) / snr_valid.size)
        frac5 = float(np.count_nonzero(snr_valid > 5) / snr_valid.size)

        if self.verbose:
            print("S/N median (full): %.3f, p16=%.3f, p84=%.3f" % (median, p16, p84))
            print("Frac S/N>1: %.3f, >3: %.3f, >5: %.3f" % (frac1, frac3, frac5))

        return {"median": median, "p16": p16, "p84": p84,
                "frac_snr_gt_1": frac1, "frac_snr_gt_3": frac3, "frac_snr_gt_5": frac5,
                "n_valid": int(snr_valid.size)}

    def lsf_summary(self):
        """
        Summarize LSF array (full). Returns median/mean/p16/p84 if available.
        """
        if self._lsf is None:
            return {"present": False}
        data = np.asarray(self._lsf)
        flat = data.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return {"present": True, "median": float("nan"), "mean": float("nan")}
        return {"present": True, "median": float(np.nanmedian(flat)), "mean": float(np.nanmean(flat)),
                "p16": float(np.percentile(flat, 16)), "p84": float(np.percentile(flat, 84))}

    def plot_snr_hist(self, bins=200, show=True):
        """
        Plot S/N histogram for the full arrays.
        """
        if self._ivar is None:
            raise RuntimeError("IVAR missing. Can't plot S/N histogram.")

        flux = np.asarray(self._flux).astype(np.float32)
        ivar = np.asarray(self._ivar).astype(np.float32)
        if flux.shape != ivar.shape:
            raise RuntimeError("FLUX and IVAR shapes don't match. Cannot plot S/N.")

        ivar_pos = np.where(np.isfinite(ivar) & (ivar > 0), ivar, 0.0)
        sqrt_ivar = np.sqrt(ivar_pos, where=(ivar_pos > 0), out=np.zeros_like(ivar_pos))
        snr = flux * sqrt_ivar
        snr_valid = snr[ivar_pos > 0]

        plt.figure(figsize=(8, 4))
        plt.hist(snr_valid.ravel(), bins=bins, histtype="stepfilled", alpha=0.8)
        plt.xlabel("S/N per pixel")
        plt.ylabel("Number of pixels (full)")
        plt.title("S/N histogram (n_pixels = %d)" % (snr_valid.size,))
        plt.grid(True, linestyle=":", alpha=0.3)
        if show:
            plt.show()

    def plot_example_spectrum(self, fiber=None, show=True):
        """
        Plot flux +/- sigma and sky for a chosen fiber (full wavelength range).
        If fiber is None, choose the middle fiber.
        """
        wave = np.asarray(self._wave) if self._wave is not None else np.arange(self.n_wave)
        if wave.ndim > 1:
            wave = wave.ravel()[: self.n_wave]

        if fiber is None:
            fiber = self.n_fibers // 2

        flux = np.asarray(self._flux)[fiber, :].astype(np.float32)
        sky = np.asarray(self._sky)[fiber, :].astype(np.float32) if self._sky is not None and \
                                            np.asarray(self._sky).shape == (self.n_fibers, self.n_wave) else None
        ivar = np.asarray(self._ivar)[fiber, :] if self._ivar is not None and \
                                            np.asarray(self._ivar).shape == (self.n_fibers, self.n_wave) else None

        if ivar is not None:
            ivar_pos = np.where(np.isfinite(ivar) & (ivar > 0), ivar, 0.0)
            sigma = 1.0 / np.sqrt(ivar_pos, where=(ivar_pos > 0), out=np.full_like(ivar_pos, np.nan))
        else:
            sigma = None

        plt.figure(figsize=(10, 4))
        x = wave if wave is not None else np.arange(len(flux))
        plt.plot(x, flux, label="flux", linewidth=0.8)
        if sigma is not None:
            plt.fill_between(x, flux - sigma, flux + sigma, alpha=0.25, label=r"$\pm 1\sigma$")
        if sky is not None:
            plt.plot(x, sky, label="sky", linewidth=0.7, linestyle="--")
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.title("Example spectrum — fiber %d" % (fiber))
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.3)
        if show:
            plt.show()

    def _plot_flagged_spectra(self,
                             flagged_global_indices,
                             out_png="flagged_zoom.png",
                             wmin=3600.,
                             wmax=9000.):
        """
        Plot flagged spectra around the arm boundary.

        Parameters
        ----------
        flagged_global_indices : list
            List of global fiber indices to plot.
        out_png : str
            Output PNG file path.

        Returns
        -------
        Path or None
            Path to saved PNG, or None if nothing to plot.
        """

        # nothing to plot
        if flagged_global_indices is None or len(flagged_global_indices) == 0:
            return None

        if self._flux is None or self._wave is None:
            raise RuntimeError("FLUX and WAVE must be present.")

        flux = np.asarray(self._flux)+ np.asarray(self._sky)
        wave = np.asarray(self._wave).ravel()

        i_lo = int(np.searchsorted(wave, wmin))
        i_hi = int(np.searchsorted(wave, wmax))

        if i_hi <= i_lo:
            return None

        x = wave[i_lo:i_hi]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

        for i, fib in enumerate(flagged_global_indices):

            if fib < 0 or fib >= flux.shape[0]:
                continue

            y = flux[fib, i_lo:i_hi]

            # normalization
            finite = y[np.isfinite(y)]
            y_plot = y - np.nanmedian(finite)

            ax.plot(x, y_plot, lw=1.2, alpha=0.8, label=f"idx = {fib}")

        #ax.axvline(center, color="k", ls="--", lw=1)
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Flux (rescaled)")
        ax.set_title(f"Flagged spectra (N={len(flagged_global_indices)})")
        ax.legend()

        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close(fig)

        return out_png

    # helper to compute per-fiber median and mad over a window
    def _integrated_median(self, flux2d, wave_arr, wlo, whi):
        li = int(np.argmin(np.abs(wave_arr - wlo)))
        ui = int(np.argmin(np.abs(wave_arr - whi)))
        if ui < li:
            li, ui = ui, li
        if ui == li:
            ui = min(li + 1, flux2d.shape[1])
        sub = flux2d[:, li:ui]
        med = np.nanmedian(sub, axis=1)
        mad = mad_std(sub, axis=1, ignore_nan=True)
        return med, mad

    # map helper
    def _make_map(self, ax, fig, values, ra, dec, title, cmap_local="RdBu_r", label="Median flux",
                 ra_dec_marker_size=10):
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            vmax = float(np.nanpercentile(np.abs(finite), 98))
        else:
            vmax = 1.0
        if vmax == 0:
            vmax = 1e-12
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        sc = ax.scatter(ra, dec, c=values, s=ra_dec_marker_size, marker="H", cmap=cmap_local, norm=norm, edgecolors="none")
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        ax.set_title(title)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(label)
        vmin = float(np.nanmin(values)) if np.isfinite(np.nanmin(values)) else 0.0
        vmax2 = float(np.nanmax(values)) if np.isfinite(np.nanmax(values)) else 0.0
        ax.text(0.02, 0.98, "min = %.2e\nmax = %.2e" % (vmin, vmax2), transform=ax.transAxes,
                ha="left", va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        return sc

    def _get_flux_residual(self, flux, wave, wmin, wmax,
                           poly_order=2, show_plot=False,
                           plot_spec_index=0, plot=False):
        """
        Fit and subtract continuum from flux using a polynomial.

        Parameters
        ----------
        flux : array
            Flux array (nfiber x nwave)
        wave : array
            Wavelength array (nwave,)
        wmin, wmax : float
            Wavelength range for continuum fitting
        poly_order : int
            Polynomial order
        show_plot : bool
            If True, show diagnostic plots for spectrum index `plot_spec_index`
        plot_spec_index : int
            Which spectrum to show diagnostics for (0..nfiber-1).

        Returns
        -------
        wfit : array
            Wavelength array used for fitting
        flux_residual : array
            Continuum-subtracted flux (nfiber x nwfit)
        """
        flux = np.asarray(flux)
        wave = np.asarray(wave).ravel()

        nfiber, nwave = flux.shape
        if wave.shape[0] != nwave:
            raise ValueError(f"Wave length ({wave.shape[0]}) != flux second dim ({nwave})")

        # find indices for fitting window
        i_lo = np.argmin(np.abs(wave - wmin))
        i_hi = np.argmin(np.abs(wave - wmax))
        if i_hi < i_lo:
            i_lo, i_hi = i_hi, i_lo

        slc = slice(i_lo, i_hi + 1)
        wfit = wave[slc]
        nwfit = wfit.size

        flux_residual = np.full((nfiber, nwfit), np.nan, dtype=float)
        continuum_arr = np.full((nfiber, nwfit), np.nan, dtype=float)

        for i in range(nfiber):
            y = flux[i, slc].astype(float)  # copy slice
            valid0 = np.isfinite(y) & np.isfinite(wfit)

            if np.sum(valid0) < (poly_order + 1):
                flux_residual[i, :] = y
                continue

            # mask invalid points before sigma clipping
            y_masked = np.ma.masked_array(y, mask=~valid0)

            try:
                clipped = sigma_clip(y_masked, sigma=3.0, maxiters=10, masked=True)
            except Exception:
                clipped = y_masked

            good = ~clipped.mask
            if np.sum(good) < (poly_order + 1):
                flux_residual[i, :] = y
                continue

            continuum = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    p = np.polynomial.polynomial.Polynomial.fit(wfit[good], y[good], deg=poly_order)
                    continuum = p(wfit)
            except Exception:
                    print('Cannot fit continuum. Assume zero continuum.')
                    flux_residual[i, :] = y
                    continue

            flux_residual[i, :] = y - continuum

            if plot:
                plt.figure(figsize=(10, 5))

                plt.plot(wfit, y, label="Flux", lw=1)

                plt.plot(wfit, continuum, label="Continuum fit", lw=2)

                plt.scatter(wfit[~good], y[~good], color="red", s=10, label="Clipped")

                plt.xlabel("Wavelength")
                plt.ylabel("Flux")
                plt.title(f"Fiber {i}")
                plt.legend()
                plt.tight_layout()

                outfile_ = os.path.join(self.log_dir, f"fiber_{i:04d}.png")
                plt.savefig(outfile_, dpi=150)
                plt.close()

        return wfit, flux_residual
    
    def continuum_mismatch_report(self,
                                  out_png,
                                  blue_edge_wmin=3600.0,
                                  blue_edge_wmax=4000.0,
                                  blue_wmin=5600.0,
                                  blue_wmax=5750.0,
                                  red_wmin=5850.0,
                                  red_wmax=6000.0,
                                  figsize=(17, 10),
                                  dpi=150,
                                  hist_bins=100,
                                  hist_upper=None,
                                  mad_ylim=(1e-3, 1e3),
                                  ra_dec_marker_size=40,
                                  cmap="RdBu_r"):
        """
        Produce QA figure checking continuum and blue/red arm mismatch.
        """
        out_png = str(out_png)

        # sanity checks
        if self._flux is None:
            raise RuntimeError("FLUX array not present.")
        if self._sky is None:
            raise RuntimeError("SKY array not present.")
        if self._slitmap is None:
            raise RuntimeError("SLITMAP not present; cannot select science fibers.")
        if self._wave is None:
            raise RuntimeError("WAVE array not present; windows need wavelength vector.")

        flux = np.asarray(self._flux)
        sky  = np.asarray(self._sky)
        wave = np.asarray(self._wave).ravel()
        slitmap = self._slitmap

        raw_tt = slitmap.field("targettype")
        tt = np.array([str(x).strip() for x in raw_tt])

        science_inds = np.where(np.char.lower(tt) == "science")[0]
        n_science = int(science_inds.size)
        if n_science == 0:
            raise RuntimeError("No science fibers found in slitmap.")

        all_ra = slitmap.field("ra")
        all_dec = slitmap.field("dec")

        ra = np.asarray(all_ra)[science_inds]
        dec = np.asarray(all_dec)[science_inds]

        # fit FLUX+SKY to get continuum
        flux_fullw = flux[science_inds, :] + sky[science_inds, :]

        # compute metrics
        integrated_flux, integrated_flux_mad_std = self._integrated_median(flux_fullw, wave, blue_edge_wmin, blue_edge_wmax)
        tiny = 1e-30
        ratio = np.full_like(integrated_flux_mad_std, np.nan, dtype=float)
        safe_mask = np.abs(integrated_flux) > tiny
        ratio[safe_mask] = integrated_flux_mad_std[safe_mask] / np.abs(integrated_flux[safe_mask])

        wave_resid, flux_resid = self._get_flux_residual(flux_fullw, wave, blue_wmin-300., red_wmax+300., poly_order=2)
        blue_resid, blue_resid_mad = self._integrated_median(flux_resid, wave_resid, blue_wmin, blue_wmax)
        red_resid,  red_resid_mad  = self._integrated_median(flux_resid, wave_resid, red_wmin, red_wmax)

        # per-fiber robust scale (max of blue_mad, red_mad)
        denom = np.nanmax(np.vstack([blue_resid_mad, red_resid_mad]), axis=0)
        denom_safe = denom.copy()
        denom_safe[~np.isfinite(denom_safe)] = np.nan
        denom_safe[denom_safe == 0] = np.nan
        diff_flux = (blue_resid - red_resid) / denom_safe

        # compute flag mask: |diff| > self.arm_jump_threshold
        flagged_mask_warn = np.isfinite(diff_flux) & (np.abs(diff_flux) > float(self.arm_jump_threshold_warn))
        flagged_science_indices_warn = list(map(int, np.where(flagged_mask_warn)[0]))   # indices into science_inds
        flagged_global_indices_warn = [int(science_inds[i]) for i in flagged_science_indices_warn]  # actual fiber indices
        n_flagged_warn = len(flagged_science_indices_warn)
        flagged_frac_warn = float(n_flagged_warn) / float(max(1, n_science))
        
        flagged_mask_bad = np.isfinite(diff_flux) & (np.abs(diff_flux) > float(self.arm_jump_threshold_bad))
        flagged_science_indices_bad = list(map(int, np.where(flagged_mask_bad)[0]))   # indices into science_inds
        flagged_global_indices_bad = [int(science_inds[i]) for i in flagged_science_indices_bad]  # actual fiber indices
        n_flagged_bad = len(flagged_science_indices_bad)
        flagged_frac_bad = float(n_flagged_bad) / float(max(1, n_science))

        warn_flag = None
        if (flagged_frac_warn >= self.warn_frac_threshold) and (flagged_frac_bad < self.bad_frac_threshold):
            warn_flag = 1
        else:
            warn_flag = 0

        bad_flag = None
        if (flagged_frac_bad >= self.bad_frac_threshold):
            bad_flag = 1
        else:
            bad_flag = 0
        
        findiff = diff_flux[np.isfinite(diff_flux)]
        median_diff = float(np.nanmedian(findiff)) if findiff.size > 0 else float("nan")
        max_diff = float(np.nanmax(findiff)) if findiff.size > 0 else float("nan")
        min_diff = float(np.nanmin(findiff)) if findiff.size > 0 else float("nan")

        # Histogram bounds
        if hist_upper is None:
            try:
                hist_upper = float(np.nanpercentile(integrated_flux, 99.5))
            except Exception:
                finite_iflux = integrated_flux[np.isfinite(integrated_flux)]
                hist_upper = float(np.nanmax(finite_iflux)) if finite_iflux.size > 0 else 1.0
        finite_iflux = integrated_flux[np.isfinite(integrated_flux)]
        if finite_iflux.size > 0:
            hist_lower = float(np.nanmin(finite_iflux))
        else:
            hist_lower = 0.0
        hist_lower -= (hist_upper - hist_lower) * 0.1

        fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi, constrained_layout=True)
        ax_hist, ax_mad, ax_map = axes[0]
        ax_blue_map, ax_red_map, ax_diff_map = axes[1]

        for ax in axes.ravel():
            ax.minorticks_on()
            ax.tick_params(which="major", length=6, width=1.0)
            ax.tick_params(which="minor", length=3, width=0.7)

        # TOP LEFT: histogram
        ax_hist.hist(integrated_flux[np.isfinite(integrated_flux)], bins=hist_bins, range=(hist_lower, hist_upper))
        ax_hist.set_xlabel("Median flux")
        ax_hist.set_ylabel("N")
        ax_hist.set_title("%d Å — %d Å" % (int(blue_edge_wmin), int(blue_edge_wmax)))

        # TOP MIDDLE: MAD/flux scatter (log y)
        ax_mad.plot(integrated_flux, ratio, ".k", markersize=4)
        ax_mad.set_xlabel("Median flux")
        ax_mad.set_ylabel("MAD std / |flux|")
        ax_mad.set_yscale("log")
        ax_mad.set_ylim(mad_ylim)
        ax_mad.set_xlim(hist_lower, hist_upper)
        ax_mad.set_title("Relative noise")
        ax_mad.yaxis.set_minor_locator(LogLocator(subs="auto"))
        ax_mad.xaxis.set_minor_locator(AutoMinorLocator())

        self._make_map(ax_map, fig, integrated_flux, ra, dec,
                       "Median flux (%.0f—%.0f Å)" % (blue_edge_wmin, blue_edge_wmax), ra_dec_marker_size=ra_dec_marker_size)
        self._make_map(ax_blue_map, fig, blue_resid, ra, dec,
                       "Blue arm resid (%.0f—%.0f Å)" % (blue_wmin, blue_wmax), ra_dec_marker_size=ra_dec_marker_size)
        self._make_map(ax_red_map,fig, red_resid, ra, dec,
                       "Red arm resid (%.0f—%.0f Å)" % (red_wmin, red_wmax), ra_dec_marker_size=ra_dec_marker_size)

        # diff map
        finite_diff = diff_flux[np.isfinite(diff_flux)]
        if finite_diff.size > 0:
            vmax_diff = float(np.nanpercentile(np.abs(finite_diff), 99))
        else:
            vmax_diff = 1.0
        if vmax_diff == 0:
            vmax_diff = 1e-12
        norm_diff = TwoSlopeNorm(vmin=-vmax_diff, vcenter=0.0, vmax=vmax_diff)
        scd = ax_diff_map.scatter(ra, dec, c=diff_flux, s=ra_dec_marker_size, marker="H",
                                  cmap=cmap, norm=norm_diff, edgecolors="none")
        ax_diff_map.set_xlabel("RA")
        ax_diff_map.set_ylabel("Dec")
        ax_diff_map.set_title("Blue - Red (normalized by MAD)")
        cbar = fig.colorbar(scd, ax=ax_diff_map, shrink=0.85)
        cbar.set_label("Flux (blue - red)/MAD")
        vmin_d = float(np.nanmin(diff_flux)) if np.isfinite(np.nanmin(diff_flux)) else 0.0
        vmax_d = float(np.nanmax(diff_flux)) if np.isfinite(np.nanmax(diff_flux)) else 0.0
        ax_diff_map.text(0.02, 0.98, "min = %.2f\nmax = %.2f" % (vmin_d, vmax_d),
                         transform=ax_diff_map.transAxes, ha="left", va="top",
                         fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        fig.suptitle(Path(self.path).name, fontsize=12)
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        #print('out_png = ', out_png)
        spec_plot_name = out_png.replace('.png','_spec.png')
        #print('spec_plot_name = ', spec_plot_name)
        continuum_jump_plot_path = self._plot_flagged_spectra(flagged_global_indices=flagged_global_indices_warn,
                                                            wmin=blue_wmin-100.,
                                                            wmax=red_wmax+100.,
                                                            out_png=spec_plot_name)
        
        summary = {
            "continuum_plot_path": out_png,
            "continuum_jump_plot_path": continuum_jump_plot_path,
            "continuum_jump_threshold_warn": float(self.arm_jump_threshold_warn),
            "continuum_jump_threshold_bad": float(self.arm_jump_threshold_bad),
            "n_science": n_science,
            "n_flagged_warn": n_flagged_warn,
            "n_flagged_bad": n_flagged_bad,
            "flagged_frac_warn": flagged_frac_warn,
            "flagged_frac_bad": flagged_frac_bad,
            "warn_frac_threshold": self.warn_frac_threshold,
            "bad_frac_threshold": self.bad_frac_threshold,
            "flagged_science_indices": flagged_science_indices_warn,
            "flagged_global_fiber_indices": flagged_global_indices_warn,
            "median_arm_diff": median_diff,
            "min_arm_diff": min_diff,
            "max_arm_diff": max_diff,
            "warning_flag": warn_flag,
            "bad_flag": bad_flag,
            }

        arrays = {
            "integrated_flux": integrated_flux,
            "integrated_flux_mad_std": integrated_flux_mad_std,
            "blue_resid": blue_resid,
            "blue_resid_mad": blue_resid_mad,
            "red_resid": red_resid,
            "red_resid_mad": red_resid_mad,
            "diff_flux": diff_flux,
            }

        if self.verbose:
            print("Continuum mismatch: %d/%d science fibers flagged (%.2f%%) with |blue-red|/MAD > %.2f" %
                  (n_flagged_warn, n_science, 100.*flagged_frac_warn, self.arm_jump_threshold_warn))
            if n_flagged_warn > 0:
                print("Flagged global fiber indices (first 20):", flagged_global_indices_warn[:20])

        return (summary, arrays)

    
    # -------------------------
    # Full report builder
    # -------------------------
    def run_full_report(self, save_json=None, save_jump_list=False):
        """
        Run QA computations on the full arrays and return a report dict.

        The report includes header summary, hdu consistency, mask fraction,
        S/N stats, sky residuals, LSF summary, and zero/NaN checks for main HDUs.

        If save_json is provided, the report is written to that file.
        """
        hdrsum = self.header_summary()
        notes = self.check_hdu_consistency()

        mask_frac = self.compute_mask_fraction()
        try:
            ivar_stats = self.compute_snr_stats() if self._ivar is not None else None
            median_snr = ivar_stats["median"] if ivar_stats is not None else float("nan")
            p16 = ivar_stats["p16"] if ivar_stats is not None else float("nan")
            p84 = ivar_stats["p84"] if ivar_stats is not None else float("nan")
        except Exception as exc:
            warnings.warn("Couldn't compute S/N: %s" % (exc,))
            ivar_stats = None
            median_snr = float("nan")
            p16 = float("nan")
            p84 = float("nan")

        lsf_sum = self.lsf_summary()

        ivar_pos_frac = 0.0
        if self._ivar is not None:
            iv = np.asarray(self._ivar)
            ivar_pos_frac = float(np.count_nonzero((np.isfinite(iv) & (iv > 0))) / iv.size)

        hdu_zero_nan = self.check_all_hdus_zero_nan()

        report = {
            "filename": self.path,
            "expnum": self.expnum,
            "mjd": self.hdr.get("MJD"),
            "n_wave": self.n_wave,
            "n_fibers": self.n_fibers,
            "exptime": self.hdr.get("EXPTIME"),
            "qaqual": self.hdr.get("QAQUAL"),
            "qaflag": self.hdr.get("QAFLAG"),
            "mask_fraction": mask_frac,
            "median_snr": median_snr,
            "extended_report": {
                "header_checks": hdrsum,
                "hdu_consistency": notes,
                "lsf": lsf_sum,
                "ivar_stats": ivar_stats,
                "hdu_zero_nan": hdu_zero_nan
            },
        }

        def to_jsonable(x):
            if isinstance(x, np.ndarray):
                x = x.tolist()
            return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in x]
        
        # call continuum_mismatch_report
        plot_name = f"{save_json.replace('.json', '')}_continuum_check_blue.png"        
        cont_summary_blue, cont_arrays_blue = self.continuum_mismatch_report(os.path.join(self.log_dir,plot_name))
        report["extended_report"]["continuum_check_blue_red"] = cont_summary_blue
        if save_jump_list == True:
            report["extended_report"]["continuum_check_blue_red"]["jump_list"] = to_jsonable(list(cont_arrays_blue["diff_flux"]))

        plot_name = f"{save_json.replace('.json', '')}_continuum_check_ir.png"        
        cont_summary_ir, cont_arrays_ir = self.continuum_mismatch_report(os.path.join(self.log_dir,plot_name),
                                                                   blue_edge_wmin=9000.0,
                                                                   blue_edge_wmax=9900.0,
                                                                   blue_wmin=7300.0,
                                                                   blue_wmax=7500.0,
                                                                   red_wmin=7650.0,
                                                                   red_wmax=7900.0,)
        report["extended_report"]["continuum_check_red_ir"] = cont_summary_ir

        report["continuum_jump_warning_flag"] = max(report["extended_report"]["continuum_check_blue_red"]["warning_flag"],
                                               report["extended_report"]["continuum_check_red_ir"]["warning_flag"])
        report["continuum_jump_bad_flag"] = max(report["extended_report"]["continuum_check_blue_red"]["bad_flag"],
                                               report["extended_report"]["continuum_check_red_ir"]["bad_flag"])
        
        if save_jump_list == True:
            report["extended_report"]["continuum_check_red_ir"]["jump_list"] = to_jsonable(list(cont_arrays_ir["diff_flux"]))

        if save_json:
            with open(os.path.join(self.log_dir,save_json), "w") as fh:
                json.dump(report, fh, indent=2, sort_keys=True)
            if self.verbose:
                print("Wrote QA JSON to", os.path.join(self.log_dir,save_json))

        return report

    def close(self):
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
