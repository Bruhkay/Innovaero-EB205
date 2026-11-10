"""Physical plant of the active trim panel: a simply supported trim panel with PZT actuators,
radiating into the cabin (baffled Rayleigh integral, free field).

Point coordinates are panel coordinates: x along the panel's axial side (0..a), y along its
circumferential side (0..b), z = distance into the cabin from the panel surface.
"""
from __future__ import annotations

import numpy as np

from ..atmosphere import Air
from ..materials import PiezoConstants, Plate


class TrimPanelPlant:
    """Frequency responses from (a) the cavity pressure and (b) actuator voltages to cabin points."""

    def __init__(self, size_m: tuple[float, float], trim: Plate, loss_factor: float, piezo: PiezoConstants,
                 actuator_positions: list[tuple[float, float]], actuator_size_m: float, actuator_thickness_m: float,
                 primary_incidence_rad: float, cabin_air: Air, max_frequency_hz: float,
                 grid: tuple[int, int] = (24, 18), max_mode_index: tuple[int, int] = (13, 10)):
        self.a, self.b = size_m
        self.air = cabin_air
        self.loss_factor = loss_factor
        self.primary_incidence_rad = primary_incidence_rad
        modes = [(i, j) for i in range(1, max_mode_index[0] + 1) for j in range(1, max_mode_index[1] + 1)]
        omega_n = np.array([np.pi**2 * np.sqrt(trim.bending_stiffness_nm / trim.surface_mass_kg_m2)
                            * ((i / self.a) ** 2 + (j / self.b) ** 2) for i, j in modes])
        keep = omega_n / (2 * np.pi) < max_frequency_hz
        self.modes = [mode for mode, k in zip(modes, keep) if k]
        self.natural_omega = omega_n[keep]
        self.modal_mass = trim.surface_mass_kg_m2 * self.a * self.b / 4

        nx, ny = grid
        xs = (np.arange(nx) + 0.5) * self.a / nx
        ys = (np.arange(ny) + 0.5) * self.b / ny
        x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
        self.x, self.y = x_grid.ravel(), y_grid.ravel()
        self.element_area = self.a * self.b / (nx * ny)
        self.mode_shapes = np.array([np.sin(i * np.pi * self.x / self.a) * np.sin(j * np.pi * self.y / self.b)
                                     for i, j in self.modes]).T                        # (n_elements, n_modes)
        lever_arm = trim.thickness_m / 2 + actuator_thickness_m / 2
        self.actuator_coupling = np.array([self._actuator_vector(cx, cy, actuator_size_m, lever_arm, piezo.e31_effective_c_m2)
                                           for cx, cy in actuator_positions]).T          # (n_modes, n_act)

    def _actuator_vector(self, cx: float, cy: float, size_m: float, lever_arm_m: float, e31: float) -> np.ndarray:
        """Modal force per volt of a square patch: theta_m = -e31 z (kx^2 + ky^2) int_patch phi dA   [N/V]."""
        x0, x1 = cx * self.a - size_m / 2, cx * self.a + size_m / 2
        y0, y1 = cy * self.b - size_m / 2, cy * self.b + size_m / 2
        theta = []
        for i, j in self.modes:
            kx, ky = i * np.pi / self.a, j * np.pi / self.b
            integral_x = (np.cos(kx * x0) - np.cos(kx * x1)) / kx
            integral_y = (np.cos(ky * y0) - np.cos(ky * y1)) / ky
            theta.append(-e31 * lever_arm_m * (kx**2 + ky**2) * integral_x * integral_y)
        return np.array(theta)

    def radiation_operator(self, omega: float, points_m: np.ndarray) -> np.ndarray:
        """Rayleigh integral: p(r) = j w rho / (2 pi) sum_e v_e exp(-j k R) / R dA   [LIT] Fahy & Gardonio (2007)."""
        k = omega / self.air.sound_speed_m_s
        offset = points_m[:, None, :] - np.stack([self.x, self.y, np.zeros_like(self.x)], axis=-1)[None]
        distance = np.linalg.norm(offset, axis=-1)
        return 1j * omega * self.air.density_kg_m3 / (2 * np.pi) * np.exp(-1j * k * distance) / distance * self.element_area

    def frequency_responses(self, frequencies_hz: np.ndarray, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Primary P[f, point] (Pa per Pa of cavity pressure) and secondary S[f, point, actuator] (Pa/V).

        Modal receptance H = 1 / (M (w_n^2 (1 + j eta) - w^2)); cavity pressure is an oblique plane wave
        exp(-j kx x) with kx = k sin(incidence)   [ASM].
        """
        primary = np.zeros((len(frequencies_hz), len(points_m)), complex)
        secondary = np.zeros((len(frequencies_hz), len(points_m), self.actuator_coupling.shape[1]), complex)
        for n, frequency in enumerate(frequencies_hz):
            omega = 2 * np.pi * max(frequency, 1e-3)
            receptance = 1 / (self.modal_mass * (self.natural_omega**2 * (1 + 1j * self.loss_factor) - omega**2))
            kx = omega / self.air.sound_speed_m_s * np.sin(self.primary_incidence_rad)
            modal_force = self.mode_shapes.T @ np.exp(-1j * kx * self.x) * self.element_area
            pressure_per_modal_velocity = self.radiation_operator(omega, points_m) @ self.mode_shapes
            velocity_gain = pressure_per_modal_velocity * (1j * omega * receptance)
            primary[n] = velocity_gain @ modal_force
            secondary[n] = velocity_gain @ self.actuator_coupling
        return primary, secondary


def impulse_responses(frequency_response: np.ndarray, n_fft: int, n_taps: int, taper: int = 256) -> np.ndarray:
    """Real impulse responses from one-sided FRFs (axis 0 = frequency), truncated to n_taps with a
    half-Hann fade-out over the last `taper` samples."""
    impulse = np.fft.irfft(frequency_response, n_fft, axis=0)[:n_taps]
    window = np.ones(n_taps)
    window[-taper:] = np.hanning(2 * taper)[taper:]
    return impulse * window.reshape((-1,) + (1,) * (impulse.ndim - 1))
