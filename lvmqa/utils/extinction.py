"""
Extinction curves and Balmer-decrement helpers.

``ccm89`` : Cardelli, Clayton & Mathis (1989) A(lambda)/A(V) for 0.3 <= x <= 3.3 um^-1
            (IR + optical/NIR parts, i.e. 3030 - 33000 A), parameterised by R_V.
"""

from __future__ import annotations

import numpy as np


def ccm89(wave_A, rv: float = 3.1) -> np.ndarray:
    """A(lambda)/A(V) for wavelengths in Angstrom (CCM89)."""
    x = 1.0e4 / np.atleast_1d(np.asarray(wave_A, dtype=float))  # 1/um
    if np.any((x < 0.3) | (x > 3.3)):
        raise ValueError("ccm89 implemented for 3030-33000 A only")
    a = np.empty_like(x)
    b = np.empty_like(x)
    ir = x < 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61
    op = ~ir
    y = x[op] - 1.82
    a[op] = (1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 + 0.72085 * y**4
             + 0.01979 * y**5 - 0.77530 * y**6 + 0.32999 * y**7)
    b[op] = (1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 - 5.38434 * y**4
             - 0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7)
    return a + b / rv


def k_lambda(wave_A, rv: float = 3.1, law: str = "ccm89") -> np.ndarray:
    """k(lambda) = A(lambda) / E(B-V)."""
    if law.lower() != "ccm89":
        raise ValueError(f"unknown extinction law {law!r} (available: ccm89)")
    return rv * ccm89(wave_A, rv)


def ebv_from_balmer(ha_hb_obs, ha_hb_int: float = 2.86, rv: float = 3.1,
                    law: str = "ccm89", wave_ha: float = 6562.85,
                    wave_hb: float = 4861.36) -> np.ndarray:
    """E(B-V) from the observed Halpha/Hbeta ratio; ratios below the intrinsic value give 0."""
    k_hb, k_ha = k_lambda([wave_hb, wave_ha], rv, law)
    r = np.asarray(ha_hb_obs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ebv = 2.5 / (k_hb - k_ha) * np.log10(r / ha_hb_int)
    return np.where(np.isfinite(ebv) & (ebv > 0), ebv, 0.0)
