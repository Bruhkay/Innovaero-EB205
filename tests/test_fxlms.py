"""FxLMS stability on small synthetic plants: the error power must drop and stay finite."""
import numpy as np

from cabinsim.active_control.fxlms import run_broadband_fxlms, run_tonal_fxlms

FS = 2000


def _plant(n_mics: int, n_act: int, seed: int = 1) -> np.ndarray:
    """Short random secondary paths with a 3-sample delay, shape (L, n_mics, n_act)."""
    rng = np.random.default_rng(seed)
    ir = np.zeros((16, n_mics, n_act))
    ir[3:8] = rng.standard_normal((5, n_mics, n_act)) * np.exp(-np.arange(5))[:, None, None]
    return ir


def _power_db(error: np.ndarray, start: int, stop: int) -> float:
    return 10 * np.log10(np.mean(error[start:stop] ** 2))


def test_tonal_fxlms_converges_and_stays_bounded():
    n = 4 * FS
    t = np.arange(n) / FS
    disturbance = np.stack([np.sin(2 * np.pi * 115 * t + 0.3 * i) + 0.5 * np.sin(2 * np.pi * 230 * t + i)
                            for i in range(2)], axis=1)
    result = run_tonal_fxlms(disturbance, _plant(2, 2), [115.0, 230.0], FS, step_size=0.005,
                             regularisation=0.01, leakage=0.0, start_sample=FS // 2)
    assert np.all(np.isfinite(result.error))
    before = _power_db(disturbance, 0, FS // 2)
    after = _power_db(result.error, n - FS // 2, n)
    assert before - after > 20.0


def test_broadband_fxlms_reduces_coherent_noise():
    rng = np.random.default_rng(0)
    n = 6 * FS
    reference = rng.standard_normal(n)
    primary = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, -0.2])                 # longer delay than the plant
    disturbance = np.convolve(reference, primary)[:n][:, None]
    result = run_broadband_fxlms(disturbance, _plant(1, 1), reference, step_size=0.05, leakage=0.0,
                                 n_taps=16, start_sample=FS // 2)
    assert np.all(np.isfinite(result.error))
    before = _power_db(disturbance, 0, FS // 2)
    after = _power_db(result.error, n - FS, n)
    assert before - after > 10.0
