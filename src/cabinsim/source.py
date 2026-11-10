"""Engine noise source: sound power per 1/3-octave band, split into broadband and tones."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import band_edges_hz, power_level_db


@dataclass(frozen=True)
class EngineSpectrum:
    """Sound power of ONE engine per band. Powers in W, levels in dB re 1 pW."""
    bands_hz: np.ndarray
    broadband_power_w: np.ndarray
    tonal_power_w: np.ndarray
    shaft_frequency_hz: float
    tone_frequencies_hz: np.ndarray     # frequency of each tone
    tone_power_w: np.ndarray            # power of each tone

    @property
    def total_power_w(self) -> np.ndarray:
        return self.broadband_power_w + self.tonal_power_w

    @property
    def total_level_db(self) -> float:
        return float(power_level_db(self.total_power_w.sum()))


def sound_power_level_db(acoustic_power_w: float) -> float:
    """Lw = 10 log10(W / 10^-12)   [DOC] EB205 p.18;  138 W -> 141.4 dB."""
    return float(power_level_db(acoustic_power_w))


def broadband_shape(bands_hz: np.ndarray, peak_hz: float, curvature_db_per_oct2: float) -> np.ndarray:
    """Relative band powers of a log-parabola spectrum, normalised to sum 1.   [ASM]

    L_rel(f) = -curvature * (log2(f / f_peak))^2
    """
    octaves_from_peak = np.log2(np.asarray(bands_hz, float) / peak_hz)
    relative_db = -curvature_db_per_oct2 * octaves_from_peak**2
    weights = 10 ** (relative_db / 10)
    return weights / weights.sum()


def band_index_of(frequency_hz: float, bands_hz: np.ndarray) -> int | None:
    """Index of the 1/3-octave band that contains `frequency_hz` (None if outside all bands)."""
    lower, upper = band_edges_hz(bands_hz)
    inside = np.where((frequency_hz >= lower) & (frequency_hz < upper))[0]
    return int(inside[0]) if len(inside) else None


def engine_spectrum(bands_hz: np.ndarray, engine_cfg: dict) -> EngineSpectrum:
    """Split the engine acoustic power W into a broadband part and shaft-order tones.   [ASM]

    shaft frequency f_s = N1 / 60;  tone k at f_k = order_k f_s with power W * fraction * w_k / sum(w)
    Tones that fall outside the band range are dropped (their power is not counted).
    """
    bands = np.asarray(bands_hz, float)
    total_w = engine_cfg["acoustic_power"]
    tone_fraction = engine_cfg["tone_energy_fraction"]
    shaft_hz = engine_cfg["n1_speed"] / 60.0

    broadband = (1 - tone_fraction) * total_w * broadband_shape(
        bands, engine_cfg["broadband_peak_frequency"], engine_cfg["broadband_curvature"])

    orders = np.array([order for order, _ in engine_cfg["tones"]], float)
    weights = np.array([weight for _, weight in engine_cfg["tones"]], float)
    tone_freqs = orders * shaft_hz
    tone_powers = tone_fraction * total_w * weights / weights.sum()

    tonal = np.zeros_like(bands)
    for freq, power in zip(tone_freqs, tone_powers):
        index = band_index_of(freq, bands)
        if index is not None:
            tonal[index] += power
    return EngineSpectrum(bands_hz=bands, broadband_power_w=broadband, tonal_power_w=tonal,
                          shaft_frequency_hz=shaft_hz, tone_frequencies_hz=tone_freqs, tone_power_w=tone_powers)


def engine_positions(engine_cfg: dict) -> list[np.ndarray]:
    """Fan-face centre of the right engine, plus its mirror image if `mirror_left_engine`."""
    right = np.asarray(engine_cfg["position"], float)
    positions = [right]
    if engine_cfg["mirror_left_engine"]:
        positions.append(right * np.array([1.0, -1.0, 1.0]))
    return positions
