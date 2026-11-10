"""Multichannel filtered-x LMS (FxLMS) controllers, simulated sample by sample.

Signal model at the error microphones (m = mic, k = actuator):
    e_m(n) = d_m(n) + sum_k sum_l s_mk(l) u_k(n - l)
d: disturbance (primary noise), s: secondary path impulse response (actuator voltage -> mic pressure),
u: control voltages. The controller only sees e and a reference signal.
[LIT] Elliott, Signal Processing for Active Control (2001); Kuo & Morgan, Active Noise Control Systems (1996).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class FxlmsResult:
    error: np.ndarray        # (n_samples, n_mics)
    control: np.ndarray      # (n_actuators, n_samples)


def tonal_secondary_estimates(secondary_ir: np.ndarray, tone_frequencies_hz: list[float], sample_rate_hz: float) -> list[np.ndarray]:
    """S_hat(w_h) = sum_l s(l) exp(-j w_h l / fs) for each tone, shape (n_mics, n_actuators).

    In practice this comes from off-line system identification; here the model is assumed exact.
    """
    lags = np.arange(secondary_ir.shape[0])
    return [np.einsum("l,lmk->mk", np.exp(-2j * np.pi * f * lags / sample_rate_hz), secondary_ir)
            for f in tone_frequencies_hz]


def tonal_preconditioners(estimates: list[np.ndarray], regularisation: float) -> list[np.ndarray]:
    """G = (S^H S + beta I)^-1 S^H with beta = regularisation * largest eigenvalue of S^H S.

    Pre-multiplying the gradient by G makes all control directions converge at a similar rate
    (a Newton-like step); beta keeps weakly controllable directions from demanding huge voltages.
    """
    result = []
    for s_hat in estimates:
        s_h = s_hat.conj().T
        normal = s_h @ s_hat
        beta = regularisation * np.linalg.norm(normal, 2)
        result.append(np.linalg.solve(normal + beta * np.eye(normal.shape[0]), s_h))
    return result


class _SecondaryPath:
    """Adds each new control sample's future effect (u(n) * s) into a buffer, so the error at time
    n is just d(n) + buffer(n). Mathematically the same as the convolution sum."""

    def __init__(self, secondary_ir: np.ndarray, n_samples: int):
        self.ir = secondary_ir                                     # (L, n_mics, n_act)
        self.length = secondary_ir.shape[0]
        self.buffer = np.zeros((secondary_ir.shape[1], n_samples + self.length))

    def add(self, n: int, control: np.ndarray) -> None:
        self.buffer[:, n:n + self.length] += np.einsum("lmk,k->ml", self.ir, control)

    def output(self, n: int) -> np.ndarray:
        return self.buffer[:, n]


def run_tonal_fxlms(disturbance: np.ndarray, secondary_ir: np.ndarray, tone_frequencies_hz: list[float],
                    sample_rate_hz: float, step_size: float, regularisation: float, leakage: float,
                    start_sample: int) -> FxlmsResult:
    """Narrowband FxLMS with a tacho reference: one complex weight per tone and actuator.

    u_k(n) = sum_h Re( W_hk exp(j w_h n / fs) )
    W_h <- (1 - leakage) W_h - mu G_h E_h,   E_h = 2 e(n) exp(-j w_h n / fs)  (demodulated error)
    """
    n_samples, n_mics = disturbance.shape
    n_act = secondary_ir.shape[2]
    preconditioners = tonal_preconditioners(tonal_secondary_estimates(secondary_ir, tone_frequencies_hz, sample_rate_hz),
                                            regularisation)
    omegas = 2 * np.pi * np.asarray(tone_frequencies_hz) / sample_rate_hz
    weights = np.zeros((len(omegas), n_act), complex)
    path = _SecondaryPath(secondary_ir, n_samples)
    error = np.zeros((n_samples, n_mics))
    control = np.zeros((n_act, n_samples))
    for n in range(n_samples):
        if n >= start_sample:
            phasors = np.exp(1j * omegas * n)
            control[:, n] = np.real(weights.T @ phasors)
            path.add(n, control[:, n])
        error[n] = disturbance[n] + path.output(n)
        if n >= start_sample:
            for h in range(len(omegas)):
                demodulated = 2 * error[n] * np.conj(phasors[h])
                weights[h] = (1 - leakage) * weights[h] - step_size * (preconditioners[h] @ demodulated)
    return FxlmsResult(error=error, control=control)


def run_broadband_fxlms(disturbance: np.ndarray, secondary_ir: np.ndarray, reference: np.ndarray,
                        step_size: float, leakage: float, n_taps: int, start_sample: int) -> FxlmsResult:
    """Broadband FxLMS with an FIR filter of `n_taps` per actuator.

    u_k(n) = sum_l w_k(l) x(n - l)
    r_mk(n) = (s_mk * x)(n)   filtered reference (model assumed exact)
    w_k(l) <- (1 - leakage) w_k(l) - mu_n sum_m e_m(n) r_mk(n - l),   mu_n = step / (<r^2> L n_mics)
    """
    n_samples, n_mics = disturbance.shape
    n_act = secondary_ir.shape[2]
    filtered = np.zeros((n_mics, n_act, n_samples))
    for m in range(n_mics):
        for k in range(n_act):
            filtered[m, k] = fftconvolve(reference, secondary_ir[:, m, k])[:n_samples]
    normalised_step = step_size / (np.mean(filtered**2) * n_taps * n_mics + 1e-30)
    reference_padded = np.concatenate([np.zeros(n_taps - 1), reference])
    filtered_padded = np.concatenate([np.zeros((n_mics, n_act, n_taps - 1)), filtered], axis=2)
    weights = np.zeros((n_act, n_taps))
    path = _SecondaryPath(secondary_ir, n_samples)
    error = np.zeros((n_samples, n_mics))
    control = np.zeros((n_act, n_samples))
    for n in range(n_samples):
        # window[l] = x(n - l), l = 0..L-1
        window = reference_padded[n:n + n_taps][::-1]
        if n >= start_sample:
            control[:, n] = weights @ window
            path.add(n, control[:, n])
        error[n] = disturbance[n] + path.output(n)
        if n >= start_sample:
            filtered_window = filtered_padded[:, :, n:n + n_taps][:, :, ::-1]          # (m, k, l)
            gradient = np.einsum("m,mkl->kl", error[n], filtered_window)
            weights = (1 - leakage) * weights - normalised_step * gradient
    return FxlmsResult(error=error, control=control)
