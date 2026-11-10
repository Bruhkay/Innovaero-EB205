"""Level arithmetic and single-number ratings (SIL, dB(A))."""
from __future__ import annotations

import numpy as np

REFERENCE_PRESSURE_PA = 20e-6      # [LIT] ISO 1683: p0 = 20 uPa in air
REFERENCE_POWER_W = 1e-12          # [LIT] ISO 1683: W0 = 1 pW
SIL_OCTAVES_HZ = (500, 1000, 2000, 4000)


def pressure_level_db(p_rms_pa: np.ndarray | float) -> np.ndarray:
    """Lp = 20 log10(p_rms / p0)   [DOC] EB205 p.18,  p0 = 20 uPa."""
    return 20 * np.log10(np.asarray(p_rms_pa, float) / REFERENCE_PRESSURE_PA)


def mean_square_from_level(level_db: np.ndarray | float) -> np.ndarray:
    """Inverse of `pressure_level_db`: p^2 = p0^2 10^(L/10)   [Pa^2]."""
    return REFERENCE_PRESSURE_PA**2 * 10 ** (np.asarray(level_db, float) / 10)


def power_level_db(power_w: np.ndarray | float) -> np.ndarray:
    """Lw = 10 log10(W / 1 pW)   [DOC] EB205 p.18."""
    return 10 * np.log10(np.asarray(power_w, float) / REFERENCE_POWER_W)


def db_sum(levels_db: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Energetic sum L_tot = 10 log10( sum_i 10^(L_i/10) )   [DOC] EB205 p.18."""
    return 10 * np.log10(np.sum(10 ** (np.asarray(levels_db, float) / 10), axis=axis))


def band_edges_hz(centres_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1/3-octave band edges f_c / 2^(1/6) and f_c * 2^(1/6)   [LIT] IEC 61260-1 (base-2)."""
    centres = np.asarray(centres_hz, float)
    return centres / 2 ** (1 / 6), centres * 2 ** (1 / 6)


def a_weighting_db(frequency_hz: np.ndarray) -> np.ndarray:
    """A-weighting A(f) in dB   [LIT] IEC 61672-1:2013, eq. (E.6).

    R_A(f) = 12194^2 f^4 / ((f^2 + 20.6^2) sqrt((f^2 + 107.7^2)(f^2 + 737.9^2)) (f^2 + 12194^2))
    A(f)   = 20 log10(R_A) + 2.00
    """
    f2 = np.asarray(frequency_hz, float) ** 2
    ra = (12194.0**2 * f2**2) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2))
    return 20 * np.log10(ra) + 2.00


def third_octaves_to_octaves(centres_hz: np.ndarray, levels_db: np.ndarray,
                             octave_centres_hz: tuple[int, ...] = (63, 125, 250, 500, 1000, 2000, 4000)) -> dict[int, float]:
    """Combine the three 1/3-octave bands inside each octave energetically.

    Returns {octave centre: level in dB}; an octave with no bands gets -inf.
    """
    centres = np.asarray(centres_hz, float)
    levels = np.asarray(levels_db, float)
    result = {}
    for octave in octave_centres_hz:
        inside = (centres > octave / 2**0.5) & (centres < octave * 2**0.5)
        result[octave] = float(db_sum(levels[inside])) if inside.any() else -np.inf
    return result


def speech_interference_level(centres_hz: np.ndarray, levels_db: np.ndarray) -> float:
    """SIL = (L500 + L1000 + L2000 + L4000) / 4, octave levels   [DOC] EB205 p.18, [LIT] ANSI S3.14."""
    octaves = third_octaves_to_octaves(centres_hz, levels_db)
    return float(np.mean([octaves[f] for f in SIL_OCTAVES_HZ]))


def a_weighted_level(centres_hz: np.ndarray, levels_db: np.ndarray) -> float:
    """L_A = 10 log10( sum 10^((L_i + A(f_i))/10) )  [dB(A)]."""
    return float(db_sum(np.asarray(levels_db) + a_weighting_db(centres_hz)))


def overall_level(levels_db: np.ndarray) -> float:
    """Unweighted overall level (OASPL) of a band spectrum [dB]."""
    return float(db_sum(levels_db))
