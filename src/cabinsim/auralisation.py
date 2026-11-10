"""Auralisation: turn the cabin spectrum at the passenger's head into a sound file.

Broadband part (engine broadband + boundary layer): random noise shaped in the frequency domain so
that every 1/3-octave band has the model's mean-square pressure. Tonal part: the engine shaft-order
tones as sinusoids; each tone is split between the two engines whose N1 differ slightly, which gives
the slow beating heard in real cabins.
All files share ONE gain, so the level differences between solutions are kept. The absolute level
on headphones cannot be guaranteed. Signals are exactly periodic (FFT synthesis, tones with a whole
number of cycles), so they loop without a click.
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import band_edges_hz, mean_square_from_level, pressure_level_db, REFERENCE_PRESSURE_PA

CORRECTION_PASSES = 2


@dataclass
class Auralisation:
    key: str
    label: str
    signal_pa: np.ndarray
    sil_db: float
    dba: float
    spectrum_db: list[float]


def _bins_in_band(frequencies: np.ndarray, bands_hz: np.ndarray) -> list[np.ndarray]:
    lower, upper = band_edges_hz(bands_hz)
    return [(frequencies >= lo) & (frequencies < hi) for lo, hi in zip(lower, upper)]


def _smooth_psd(frequencies: np.ndarray, bands_hz: np.ndarray, psd_band: np.ndarray, rolloff_db_per_octave: float) -> np.ndarray:
    """PSD at every FFT bin: log-log interpolation between band centres, roll-off outside them."""
    log_psd = np.log10(np.maximum(psd_band, 1e-30))
    safe = np.maximum(frequencies, 1e-3)
    result = 10 ** np.interp(np.log(safe), np.log(bands_hz), log_psd)
    below, above = frequencies < bands_hz[0], frequencies > bands_hz[-1]
    result[below] *= 10 ** (-rolloff_db_per_octave / 10 * np.log2(bands_hz[0] / safe[below]))
    result[above] *= 10 ** (-rolloff_db_per_octave / 10 * np.log2(safe[above] / bands_hz[-1]))
    result[frequencies == 0] = 0.0
    return result


def broadband_noise(bands_hz: np.ndarray, band_mean_square_pa2: np.ndarray, sample_rate: int, n_samples: int,
                    rolloff_db_per_octave: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise [Pa] whose 1/3-octave band mean-squares equal `band_mean_square_pa2` (on average).

    Bin amplitude for a one-sided PSD S(f):  E|X_k|^2 = S(f_k) fs n / 2   (numpy rfft/irfft scaling).
    The smooth interpolation moves a little energy between bands, so the shape is corrected per band
    a couple of times before the random phases are applied.
    """
    frequencies = np.fft.rfftfreq(n_samples, 1 / sample_rate)
    lower, upper = band_edges_hz(bands_hz)
    psd_target = np.asarray(band_mean_square_pa2) / (upper - lower)
    masks = _bins_in_band(frequencies, bands_hz)
    psd_shape = psd_target.copy()
    bin_width = sample_rate / n_samples
    for _ in range(CORRECTION_PASSES):
        psd = _smooth_psd(frequencies, bands_hz, psd_shape, rolloff_db_per_octave)
        realised = np.array([psd[m].sum() * bin_width for m in masks])
        psd_shape *= np.where(realised > 0, band_mean_square_pa2 / np.maximum(realised, 1e-30), 1.0)
    psd = _smooth_psd(frequencies, bands_hz, psd_shape, rolloff_db_per_octave)
    amplitude = np.sqrt(psd * sample_rate * n_samples / 2)
    spectrum = amplitude * (rng.standard_normal(len(frequencies)) + 1j * rng.standard_normal(len(frequencies))) / np.sqrt(2)
    return np.fft.irfft(spectrum, n_samples)


def engine_tones(bands_hz: np.ndarray, band_tonal_mean_square_pa2: np.ndarray, tones: list[dict], sample_rate: int,
                 n_samples: int, n1_mismatch: float, rng: np.random.Generator) -> np.ndarray:
    """Sum of tones [Pa]. Each band's tonal mean-square is shared among the source tones in that band
    (by source power), then split equally between engine 1 (f) and engine 2 (f (1 + mismatch)).
    Frequencies are rounded to whole cycles over the signal so it loops cleanly."""
    duration = n_samples / sample_rate
    time = np.arange(n_samples) / sample_rate
    lower, upper = band_edges_hz(bands_hz)
    signal = np.zeros(n_samples)
    for b, band_ms in enumerate(band_tonal_mean_square_pa2):
        in_band = [t for t in tones if lower[b] <= t["frequency_hz"] < upper[b] and t["power_w"] > 0]
        total_power = sum(t["power_w"] for t in in_band)
        for tone in in_band:
            tone_ms = band_ms * tone["power_w"] / total_power
            for factor in (1.0, 1.0 + n1_mismatch):
                frequency = round(tone["frequency_hz"] * factor * duration) / duration
                amplitude = np.sqrt(2 * tone_ms / 2)            # half the mean-square per engine
                signal += amplitude * np.sin(2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi))
    return signal


def listening_cases(results: dict) -> list[tuple[str, str, dict]]:
    """(key, label, head level summary) for every solution, plus the tonal-ANC variant where present."""
    cases = []
    for name, solution in results["solutions"].items():
        cases.append((name, solution["label"], solution["head"]))
        if "active_trim_result" in solution:
            cases.append((f"{name}_anc", f"{solution['label']}, ANC on", solution["active_trim_result"]["head_with_anc"]))
    return cases


def synthesise(results: dict, visuals_cfg: dict, seed: int = 0) -> list[Auralisation]:
    """Pressure signals at the head for every listening case. The same noise seed is used for every
    case, so the files differ only in their spectra (a fair A/B comparison)."""
    bands = np.asarray(results["bands_hz"], float)
    fs = int(visuals_cfg["audio_sample_rate"])
    n_samples = int(round(visuals_cfg["audio_duration"] * fs))
    output = []
    for key, label, head in listening_cases(results):
        paths = {path: mean_square_from_level(levels) for path, levels in head["paths_db"].items()}
        broadband = broadband_noise(bands, paths["engine_broadband"] + paths["tbl"], fs, n_samples,
                                    visuals_cfg["audio_rolloff_db_per_octave"], np.random.default_rng(seed))
        tonal = engine_tones(bands, paths["engine_tonal"], results["source"]["tones"], fs, n_samples,
                             visuals_cfg["audio_n1_mismatch"], np.random.default_rng(seed + 1))
        output.append(Auralisation(key=key, label=label, signal_pa=broadband + tonal, sil_db=head["sil_db"],
                                   dba=head["dba"], spectrum_db=head["total_db"]))
    return output


def band_levels_of_signal(signal_pa: np.ndarray, sample_rate: int, bands_hz: np.ndarray) -> np.ndarray:
    """1/3-octave band levels [dB re 20 uPa] of a periodic signal, from its FFT."""
    spectrum = np.fft.rfft(signal_pa)
    power = 2 * np.abs(spectrum) ** 2 / len(signal_pa) ** 2                   # mean-square per bin
    frequencies = np.fft.rfftfreq(len(signal_pa), 1 / sample_rate)
    return np.array([pressure_level_db(np.sqrt(power[m].sum())) for m in _bins_in_band(frequencies, bands_hz)])


def common_gain(signals: list[np.ndarray], headroom_db: float) -> float:
    """One gain [1/Pa] for all files: the loudest peak lands `headroom_db` below full scale."""
    peak = max(np.max(np.abs(s)) for s in signals)
    return 10 ** (-headroom_db / 20) / peak


def wav_bytes(signal_pa: np.ndarray, gain: float, sample_rate: int) -> bytes:
    """16-bit mono PCM WAV of signal * gain."""
    samples = np.clip(np.round(signal_pa * gain * 32767), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def write_audio(results: dict, visuals_cfg: dict, folder: Path, seed: int = 0) -> dict:
    """Write one .wav per listening case; returns the list for the demo page."""
    cases = synthesise(results, visuals_cfg, seed)
    gain = common_gain([c.signal_pa for c in cases], visuals_cfg["audio_headroom_db"])
    fs = int(visuals_cfg["audio_sample_rate"])
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for case in cases:
        filename = f"{case.key}.wav"
        (folder / filename).write_bytes(wav_bytes(case.signal_pa, gain, fs))
        entries.append(dict(key=case.key, label=case.label, file=f"audio/{filename}", sil_db=case.sil_db,
                            dba=case.dba, spectrum_db=case.spectrum_db))
    # a full-scale sine (rms 1/sqrt 2) corresponds to this sound pressure level
    full_scale_spl = float(pressure_level_db(1 / gain / np.sqrt(2)))
    return dict(cases=entries, bands_hz=results["bands_hz"], full_scale_sine_spl_db=full_scale_spl,
                duration_s=visuals_cfg["audio_duration"], reference_pressure_pa=REFERENCE_PRESSURE_PA)
