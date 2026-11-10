"""Auralisation: the sound files carry the model's band levels and keep the differences between solutions."""
import io
import wave

import numpy as np
import pytest

from cabinsim.auralisation import (band_levels_of_signal, broadband_noise, common_gain, engine_tones, synthesise,
                                   wav_bytes)
from cabinsim.metrics import mean_square_from_level, pressure_level_db

BANDS = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                  1600, 2000, 2500, 3150, 4000, 5000], float)
FS, N = 44100, 8 * 44100


def test_shaped_noise_has_the_target_band_levels():
    target_db = 80 - 1.2 * np.arange(len(BANDS))                     # falling cabin-like spectrum
    signal = broadband_noise(BANDS, mean_square_from_level(target_db), FS, N, 12.0, np.random.default_rng(3))
    measured = band_levels_of_signal(signal, FS, BANDS)
    np.testing.assert_allclose(measured[1:-1], target_db[1:-1], atol=1.0)   # random scatter stays below 1 dB


def test_tone_level_and_periodicity():
    tones = [dict(frequency_hz=690.0, power_w=1.0)]
    band = np.argmin(np.abs(BANDS - 630))
    tonal = np.zeros(len(BANDS))
    tonal[band] = mean_square_from_level(60.0)
    signal = engine_tones(BANDS, tonal, tones, FS, N, 0.002, np.random.default_rng(0))
    assert pressure_level_db(np.sqrt(np.mean(signal**2))) == pytest.approx(60.0, abs=0.1)
    spectrum = np.abs(np.fft.rfft(signal))
    peaks = np.sort(np.argsort(spectrum)[-2:]) * FS / N
    assert peaks[0] == pytest.approx(690.0, abs=0.13) and peaks[1] == pytest.approx(691.38, abs=0.13)   # beating pair


def test_wav_header_and_common_gain():
    signal = np.sin(2 * np.pi * 440 * np.arange(FS) / FS) * 0.5
    gain = common_gain([signal, 0.5 * signal], 1.0)
    data = wav_bytes(signal, gain, FS)
    with wave.open(io.BytesIO(data)) as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()) == (1, 2, FS, FS)
        samples = np.frombuffer(handle.readframes(FS), "<i2")
    assert np.max(np.abs(samples)) == pytest.approx(32767 * 10 ** (-1 / 20), rel=1e-3)


def test_level_differences_between_solutions_are_kept(params):
    """Two fake solutions 6 dB apart in every band must be 6 dB apart in the synthesised sound."""
    loud = dict(paths_db=dict(engine_broadband=list(70 - np.arange(21.0)), engine_tonal=[-200.0] * 21,
                              tbl=list(65 - np.arange(21.0))), total_db=[0.0] * 21, sil_db=0.0, dba=0.0)
    quiet = dict(loud, paths_db={k: list(np.array(v) - 6.0) for k, v in loud["paths_db"].items()})
    results = dict(bands_hz=list(BANDS), source=dict(tones=[]),
                   solutions=dict(A=dict(label="A", head=loud), B=dict(label="B", head=quiet)))
    a, b = synthesise(results, params.visuals)
    difference = 20 * np.log10(np.std(a.signal_pa) / np.std(b.signal_pa))
    assert difference == pytest.approx(6.0, abs=0.05)
