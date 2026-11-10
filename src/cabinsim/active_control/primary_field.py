"""The part of the cabin sound at the head that does NOT come from the controlled trim panel:
reverberant sound plus sound from all other panels. We model it as a diffuse field, i.e. many
plane waves from random directions with random phases.   [ASM] [LIT] Elliott, Signal Processing
for Active Control (2001), ch. 1: the zone of quiet around a cancellation point in a diffuse
field is only about a tenth of a wavelength across.
"""
from __future__ import annotations

import numpy as np


def random_directions(count: int, rng: np.random.Generator) -> np.ndarray:
    """Unit vectors spread uniformly over the sphere, shape (count, 3)."""
    vectors = rng.standard_normal((count, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def diffuse_tone(points_m: np.ndarray, frequency_hz: float, sound_speed: float, directions: np.ndarray,
                 phases_rad: np.ndarray) -> np.ndarray:
    """Complex amplitude of a diffuse tone at each point, normalised so that E|p|^2 = 1.

    p(x) = (1 / sqrt(N)) sum_n exp(j phi_n) exp(-j k d_n . x)
    """
    k = 2 * np.pi * frequency_hz / sound_speed
    phase = phases_rad[None, :] - k * (points_m @ directions.T)
    return np.exp(1j * phase).sum(axis=1) / np.sqrt(len(directions))


def diffuse_noise(points_m: np.ndarray, n_samples: int, sample_rate_hz: float, sound_speed: float,
                  directions: np.ndarray, amplitude_spectrum: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random diffuse broadband field at the points, shape (n_samples, n_points).

    Each plane wave carries its own independent Gaussian noise (so the field is NOT coherent with
    any single reference). `amplitude_spectrum` (length n_samples//2 + 1) sets the PSD shape:
    E|X(f)|^2 is proportional to amplitude_spectrum^2.
    """
    frequencies = np.fft.rfftfreq(n_samples, 1 / sample_rate_hz)
    n_waves = len(directions)
    spectra = (rng.standard_normal((n_waves, len(frequencies))) + 1j * rng.standard_normal((n_waves, len(frequencies))))
    spectra *= amplitude_spectrum[None, :] / np.sqrt(2 * n_waves)
    delays = points_m @ directions.T / sound_speed                                   # (n_points, n_waves)
    signals = np.zeros((n_samples, len(points_m)))
    for i, point_delays in enumerate(delays):
        shifted = spectra * np.exp(-2j * np.pi * frequencies[None, :] * point_delays[:, None])
        signals[:, i] = np.fft.irfft(shifted.sum(axis=0), n_samples)
    return signals
