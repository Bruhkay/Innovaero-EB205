"""Active trim panel simulation: build the plant and the sound field at the head, run FxLMS,
and measure how much quieter the head zone becomes.

The sound at the head has two parts, both scaled from the cabin model:
  * the controlled trim panel's own radiation (deterministic, from the plant model), and
  * everything else (reverberant field + other panels), modelled as a diffuse field.
Tones: the tacho reference is coherent with BOTH parts, so both can be reduced, but only in a small zone.
Broadband: the reference sensor sees only the controlled panel (with coherence gamma^2), so the
diffuse part is uncontrollable. This is why broadband ANC gains are small in a real cabin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt, sosfreqz, welch

from ..atmosphere import Air
from ..materials import PiezoConstants, Plate
from ..metrics import band_edges_hz
from .fxlms import run_broadband_fxlms, run_tonal_fxlms
from .plant import TrimPanelPlant, impulse_responses
from .primary_field import diffuse_noise, diffuse_tone, random_directions

N_FFT_PLANT = 8192
BANDPASS_ORDER = 4
HISTORY_STEP_S = 0.1


@dataclass(frozen=True)
class AncGeometry:
    plant: TrimPanelPlant
    head_m: np.ndarray        # (3,) panel coordinates
    mics_m: np.ndarray        # (n_mics, 3)
    zone_m: np.ndarray        # (n_zone, 3)

    @property
    def all_points(self) -> np.ndarray:
        """Mics first, then head centre, then zone points."""
        return np.vstack([self.mics_m, self.head_m[None, :], self.zone_m])


def build_geometry(active_cfg: dict, panel_size_m: tuple[float, float], head_distance_m: float, trim: Plate,
                   piezo: PiezoConstants, cabin_air: Air) -> AncGeometry:
    """Plant, error microphones and the quiet-zone evaluation grid from active.yaml."""
    a, b = panel_size_m
    plant = TrimPanelPlant(size_m=(a, b), trim=trim, loss_factor=active_cfg["panel_loss_factor"], piezo=piezo,
                           actuator_positions=[tuple(p) for p in active_cfg["actuator_positions"]],
                           actuator_size_m=active_cfg["actuator_size"],
                           actuator_thickness_m=active_cfg["actuator_thickness"],
                           primary_incidence_rad=np.deg2rad(active_cfg["primary_incidence_angle"]),
                           cabin_air=cabin_air, max_frequency_hz=0.75 * active_cfg["sample_rate"])
    cx, cy = active_cfg["head_centre_on_panel"]
    head = np.array([cx * a, cy * b, head_distance_m])
    mics = head + np.asarray(active_cfg["microphone_offsets"], float)
    half = np.asarray(active_cfg["zone_half_size"], float)
    nx, ny, nz = active_cfg["zone_points"]
    zone = np.array([head + [dx, dy, dz]
                     for dx in np.linspace(-half[0], half[0], nx)
                     for dy in np.linspace(-half[1], half[1], ny)
                     for dz in np.linspace(-half[2], half[2], nz)])
    return AncGeometry(plant=plant, head_m=head, mics_m=mics, zone_m=zone)


def _lock_in_power(signals: np.ndarray, frequency_hz: float, sample_rate_hz: float, first_sample: int) -> np.ndarray:
    """Mean-square of the component at `frequency_hz` in each column, over samples first_sample..end."""
    n = np.arange(first_sample, signals.shape[0])
    phasor = np.exp(-2j * np.pi * frequency_hz * n / sample_rate_hz)
    amplitude = 2 * (signals[first_sample:] * phasor[:, None]).mean(axis=0)
    return np.abs(amplitude) ** 2 / 2


def _error_at_points(disturbance: np.ndarray, control: np.ndarray, secondary_ir: np.ndarray) -> np.ndarray:
    """e = d + sum_k s_k * u_k at every point (full convolution, used for the evaluation points)."""
    error = disturbance.copy()
    n_samples = disturbance.shape[0]
    for k in range(control.shape[0]):
        for i in range(disturbance.shape[1]):
            error[:, i] += fftconvolve(control[k], secondary_ir[:, i, k])[:n_samples]
    return error


def _converged(error_mics: np.ndarray, window: int, tolerance_db: float) -> bool:
    """Error power in the last window vs the window before it."""
    last = np.mean(error_mics[-window:] ** 2)
    previous = np.mean(error_mics[-2 * window:-window] ** 2)
    return bool(abs(10 * np.log10(last / previous)) < tolerance_db)


def _history(disturbance_mics: np.ndarray, error_mics: np.ndarray, sample_rate_hz: float) -> dict:
    """Error power at the mics relative to the uncontrolled mean, every 0.1 s."""
    step = int(HISTORY_STEP_S * sample_rate_hz)
    reference = np.mean(disturbance_mics**2)
    starts = np.arange(0, error_mics.shape[0] - step + 1, step)
    level = [float(10 * np.log10(np.mean(error_mics[s:s + step] ** 2) / reference)) for s in starts]
    return dict(time_s=(starts / sample_rate_hz).tolist(), relative_level_db=level)


def simulate_tonal(geometry: AncGeometry, active_cfg: dict, tone_frequencies_hz: list[float],
                   controlled_ms_pa2: list[float], other_ms_pa2: list[float], duration_s: float,
                   rng: np.random.Generator) -> dict:
    """Run the tonal FxLMS. Mean-squares are the cabin-model values at the head for each tone."""
    fs = active_cfg["sample_rate"]
    n_samples = int(duration_s * fs)
    time = np.arange(n_samples) / fs
    points = geometry.all_points
    n_mics = len(geometry.mics_m)
    head_index = n_mics
    primary, _ = geometry.plant.frequency_responses(np.asarray(tone_frequencies_hz), points)
    directions = random_directions(active_cfg["diffuse_plane_waves"], rng)

    disturbance = np.zeros((n_samples, len(points)))
    for h, frequency in enumerate(tone_frequencies_hz):
        panel_part = primary[h] * np.sqrt(2 * controlled_ms_pa2[h]) / abs(primary[h][head_index])
        phases = rng.uniform(0, 2 * np.pi, len(directions))
        other_part = diffuse_tone(points, frequency, geometry.plant.air.sound_speed_m_s, directions, phases)
        other_part *= np.sqrt(2 * other_ms_pa2[h])
        amplitude = (panel_part + other_part) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        disturbance += np.real(amplitude[None, :] * np.exp(2j * np.pi * frequency * time)[:, None])

    frequencies = np.fft.rfftfreq(N_FFT_PLANT, 1 / fs)
    _, secondary = geometry.plant.frequency_responses(frequencies, points)
    secondary_ir = impulse_responses(secondary, N_FFT_PLANT, active_cfg["impulse_response_taps"])
    start = int(active_cfg["control_start"] * fs)
    result = run_tonal_fxlms(disturbance[:, :n_mics], secondary_ir[:, :n_mics, :], tone_frequencies_hz, fs,
                             active_cfg["step_size_tonal"], active_cfg["regularisation_tonal"],
                             active_cfg["leakage_tonal"], start)
    zone_slice = slice(n_mics + 1, len(points))
    error_zone = _error_at_points(disturbance[:, zone_slice], result.control, secondary_ir[:, zone_slice, :])

    window = int(active_cfg["evaluation_window"] * fs)
    first = n_samples - window
    att_mic, att_zone = [], []
    for frequency in tone_frequencies_hz:
        before_mic = _lock_in_power(disturbance[:, :n_mics], frequency, fs, first).mean()
        after_mic = _lock_in_power(result.error, frequency, fs, first).mean()
        before_zone = _lock_in_power(disturbance[:, zone_slice], frequency, fs, first).mean()
        after_zone = _lock_in_power(error_zone, frequency, fs, first).mean()
        att_mic.append(float(10 * np.log10(before_mic / after_mic)))
        att_zone.append(float(10 * np.log10(before_zone / after_zone)))
    voltage = float(np.max(np.sqrt(np.mean(result.control[:, first:] ** 2, axis=1))))
    return dict(attenuation_mic_db=att_mic, attenuation_zone_db=att_zone,
                converged=_converged(result.error, window, active_cfg["convergence_tolerance"]),
                voltage_rms=voltage, history=_history(disturbance[:, :n_mics], result.error, fs))


def _band_powers(signals: np.ndarray, sample_rate_hz: float, bands_hz: np.ndarray) -> np.ndarray:
    """Band power (summed PSD bins) per 1/3-octave band, averaged over the signal columns."""
    frequencies, psd = welch(signals, sample_rate_hz, nperseg=min(4096, signals.shape[0]), axis=0)
    psd_mean = psd.mean(axis=1)
    lower, upper = band_edges_hz(bands_hz)
    return np.array([psd_mean[(frequencies >= lo) & (frequencies < hi)].sum() for lo, hi in zip(lower, upper)])


def controllable_bands(bands_hz: np.ndarray, active_cfg: dict) -> np.ndarray:
    """Bands fully inside the broadband controller range and safely below Nyquist (0.45 fs)."""
    lower, upper = band_edges_hz(bands_hz)
    f_lo, f_hi = active_cfg["broadband_band"]
    return (lower >= f_lo) & (upper <= f_hi) & (upper <= 0.45 * active_cfg["sample_rate"])


def simulate_broadband(geometry: AncGeometry, active_cfg: dict, bands_hz: np.ndarray,
                       controlled_ms_pa2: np.ndarray, other_ms_pa2: np.ndarray, duration_s: float,
                       rng: np.random.Generator) -> dict:
    """Run the broadband FxLMS. Band mean-squares are cabin-model values at the head."""
    fs = active_cfg["sample_rate"]
    n_samples = int(duration_s * fs)
    points = geometry.all_points
    n_mics = len(geometry.mics_m)
    head_index = n_mics
    frequencies = np.fft.rfftfreq(N_FFT_PLANT, 1 / fs)
    primary, secondary = geometry.plant.frequency_responses(frequencies, points)
    primary_ir = impulse_responses(primary, N_FFT_PLANT, active_cfg["impulse_response_taps"])
    secondary_ir = impulse_responses(secondary, N_FFT_PLANT, active_cfg["impulse_response_taps"])

    bandpass = butter(BANDPASS_ORDER, active_cfg["broadband_band"], btype="bandpass", fs=fs, output="sos")
    source = sosfilt(bandpass, rng.standard_normal(n_samples))
    coherent = np.stack([fftconvolve(source, primary_ir[:, i])[:n_samples] for i in range(len(points))], axis=1)

    # Diffuse remainder: per-band ratio other/controlled at the head, same spectral shape as the panel part.
    controlled_band = controllable_bands(bands_hz, active_cfg)
    ratio_band = np.where(controlled_band, other_ms_pa2 / np.maximum(controlled_ms_pa2, 1e-30), 0.0)
    signal_frequencies = np.fft.rfftfreq(n_samples, 1 / fs)
    lower, upper = band_edges_hz(bands_hz)
    ratio = np.zeros_like(signal_frequencies)
    for b in range(len(bands_hz)):
        ratio[(signal_frequencies >= lower[b]) & (signal_frequencies < upper[b])] = ratio_band[b]
    _, bandpass_response = sosfreqz(bandpass, worN=signal_frequencies, fs=fs)
    head_response = np.interp(signal_frequencies, frequencies, np.abs(primary[:, head_index]))
    amplitude = np.sqrt(ratio * n_samples) * np.abs(bandpass_response) * head_response
    directions = random_directions(active_cfg["diffuse_plane_waves"], rng)
    other = diffuse_noise(points, n_samples, fs, geometry.plant.air.sound_speed_m_s, directions, amplitude, rng)

    # Physical scale: the panel part's mean-square at the head equals the cabin-model value in controlled bands.
    target_ms = float(np.sum(np.where(controlled_band, controlled_ms_pa2, 0.0)))
    scale = np.sqrt(target_ms / np.mean(coherent[:, head_index] ** 2))
    disturbance = (coherent + other) * scale

    coherence = active_cfg["reference_coherence"]
    sensor_noise = sosfilt(bandpass, rng.standard_normal(n_samples)) * np.sqrt((1 - coherence) / coherence)
    reference = (source + sensor_noise) * scale

    start = int(active_cfg["control_start"] * fs)
    result = run_broadband_fxlms(disturbance[:, :n_mics], secondary_ir[:, :n_mics, :], reference,
                                 active_cfg["step_size_broadband"], active_cfg["leakage_broadband"],
                                 active_cfg["filter_taps_broadband"], start)
    zone_slice = slice(n_mics + 1, len(points))
    error_zone = _error_at_points(disturbance[:, zone_slice], result.control, secondary_ir[:, zone_slice, :])

    window = int(active_cfg["evaluation_window"] * fs)
    first = n_samples - window

    def attenuation(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        power_before = _band_powers(before, fs, bands_hz)
        power_after = _band_powers(after, fs, bands_hz)
        result_db = np.zeros(len(bands_hz))
        result_db[controlled_band] = 10 * np.log10(power_before[controlled_band] / power_after[controlled_band])
        return result_db

    voltage = float(np.max(np.sqrt(np.mean(result.control[:, first:] ** 2, axis=1))))
    return dict(attenuation_zone_db=attenuation(disturbance[first:, zone_slice], error_zone[first:]),
                attenuation_mic_db=attenuation(disturbance[first:, :n_mics], result.error[first:]),
                converged=_converged(result.error, window, active_cfg["convergence_tolerance"]),
                voltage_rms=voltage, history=_history(disturbance[:, :n_mics], result.error, fs))
