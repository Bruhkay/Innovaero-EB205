"""One skin bay (between two frames and two stringers) with shunted piezo patches.

Modal model of a simply supported, curved, pressurised plate:
    [K (1 + j eta) - w^2 M + j w R_rad + sum_p theta_p theta_p^T / Cp * g_p / (1 + g_p)] q = F
    g_p = j w Cp Z_p   (Z_p: shunt impedance of patch p)
Each patch adds a rank-one electromechanical stiffness: short circuit (Z = 0) adds nothing,
open circuit (Z = inf) adds theta theta^T / Cp; a tuned RL shunt turns it into a damper.
[LIT] Hagood & von Flotow (1991), J. Sound Vib. 146(2).
Radiation into both half-spaces uses the Rayleigh resistance matrix of a baffled plate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import Air, FlightCondition
from .materials import PiezoConstants, Plate


@dataclass(frozen=True)
class Patch:
    centre_fraction: tuple[float, float]
    size_m: tuple[float, float]
    target_mode: tuple[int, int]


def patches_from_config(piezo_cfg: dict) -> list[Patch]:
    return [Patch(tuple(p["centre"]), tuple(p["size"]), tuple(p["target_mode"])) for p in piezo_cfg["patches"]]


def patch_mass_per_area(patches: list[Patch], patch_thickness_m: float, piezo: PiezoConstants,
                        bay_area_m2: float) -> float:
    """PZT mass smeared over the bay:  rho_p t_p sum(A_patch) / A_bay  [kg/m^2]."""
    patch_area = sum(p.size_m[0] * p.size_m[1] for p in patches)
    return piezo.density_kg_m3 * patch_thickness_m * patch_area / bay_area_m2


def modal_stiffness_per_area(plate: Plate, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """D k^4 + N_x kx^2 + N_y ky^2 + (E t / R^2)(kx / k)^4   [N/m^3]   (see closed_form.bay_mode_frequency)."""
    k2 = kx**2 + ky**2
    return (plate.bending_stiffness_nm * k2**2 + plate.tension_axial_n_m * kx**2 + plate.tension_circ_n_m * ky**2
            + plate.membrane_stiffness_n_m / plate.curvature_radius_m**2 * (kx**2 / k2) ** 2)


def rayleigh_resistance_matrix(omega: float, air: Air, distances_m: np.ndarray, element_area_m2: float) -> np.ndarray:
    """R_ij = w^2 rho dA^2 / (2 pi c) * sin(k r_ij) / (k r_ij)   [LIT] Fahy & Gardonio (2007)

    Radiated power of a baffled surface with element velocities v:  W = 1/2 Re(v^H R v).
    """
    k = omega / air.sound_speed_m_s
    kr = k * distances_m
    sinc = np.where(kr > 0, np.sin(kr) / np.maximum(kr, 1e-12), 1.0)
    return omega**2 * air.density_kg_m3 * element_area_m2**2 / (2 * np.pi * air.sound_speed_m_s) * sinc


class ShuntedBay:
    """Skin bay of size a (axial, frame pitch) x b (circumferential, stringer pitch)."""

    def __init__(self, length_axial_m: float, length_circ_m: float, skin: Plate, patches: list[Patch],
                 patch_thickness_m: float, piezo: PiezoConstants, flight: FlightCondition,
                 max_mode_axial: int, max_mode_circ: int, grid: tuple[int, int],
                 tuning_plate: Plate | None = None):
        """`skin` is the in-flight plate (with tension). `tuning_plate` (optional) is the plate the
        shunts are tuned for, e.g. the unpressurised skin for a ground-tuned shunt."""
        self.a, self.b = length_axial_m, length_circ_m
        self.skin = skin
        self.flight = flight
        self.patches = patches
        self.patch_thickness_m = patch_thickness_m
        self.added_mass_per_area = patch_mass_per_area(patches, patch_thickness_m, piezo, self.a * self.b)

        modes = [(i, j) for i in range(1, max_mode_axial + 1) for j in range(1, max_mode_circ + 1)]
        self.modes = sorted(modes, key=lambda mode: self._stiffness(skin, mode))
        self.modal_mass = (skin.surface_mass_kg_m2 + self.added_mass_per_area) * self.a * self.b / 4
        self.modal_stiffness = np.array([self._stiffness(skin, mode) for mode in self.modes]) * self.a * self.b / 4
        self.short_circuit_omega = np.sqrt(self.modal_stiffness / self.modal_mass)

        nx, ny = grid
        xs = (np.arange(nx) + 0.5) * self.a / nx
        ys = (np.arange(ny) + 0.5) * self.b / ny
        x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
        self.x, self.y = x_grid.ravel(), y_grid.ravel()
        self.element_area = self.a * self.b / (nx * ny)
        self.mode_shapes = np.array([np.sin(i * np.pi * self.x / self.a) * np.sin(j * np.pi * self.y / self.b)
                                     for i, j in self.modes]).T                     # (n_elements, n_modes)
        self.distances = np.hypot(self.x[:, None] - self.x[None, :], self.y[:, None] - self.y[None, :])

        self.lever_arm = (skin.thickness_m + patch_thickness_m) / 2
        self.capacitance = np.array([piezo.permittivity_effective_f_m * p.size_m[0] * p.size_m[1] / patch_thickness_m
                                     for p in patches])
        self.coupling = np.array([self._coupling_vector(p, piezo.e31_effective_c_m2) for p in patches]
                                 ).reshape(len(patches), len(self.modes))         # (n_patch, n_modes); a bare bay has 0 rows
        self.coupling_factor2 = self.coupling**2 / (self.capacitance[:, None] * self.modal_stiffness[None, :])
        self.target_index = [self.modes.index(tuple(p.target_mode)) for p in patches]
        self.tuning_omega = self._tuning_frequencies(tuning_plate or skin)

    # -------------------------------------------------------------- set-up helpers
    def _stiffness(self, plate: Plate, mode: tuple[int, int]) -> float:
        kx, ky = mode[0] * np.pi / self.a, mode[1] * np.pi / self.b
        return float(modal_stiffness_per_area(plate, np.array(kx), np.array(ky)))

    def _coupling_vector(self, patch: Patch, e31: float) -> np.ndarray:
        """theta_m = e31 z int_patch (phi_xx + phi_yy) dA = -e31 z (kx^2 + ky^2) int_patch phi dA   [N/V]."""
        cx, cy = patch.centre_fraction
        lx, ly = patch.size_m
        x0, x1 = cx * self.a - lx / 2, cx * self.a + lx / 2
        y0, y1 = cy * self.b - ly / 2, cy * self.b + ly / 2
        theta = []
        for i, j in self.modes:
            kx, ky = i * np.pi / self.a, j * np.pi / self.b
            integral_x = (np.cos(kx * x0) - np.cos(kx * x1)) / kx
            integral_y = (np.cos(ky * y0) - np.cos(ky * y1)) / ky
            theta.append(-e31 * self.lever_arm * (kx**2 + ky**2) * integral_x * integral_y)
        return np.array(theta)

    def _tuning_frequencies(self, plate: Plate) -> np.ndarray:
        """Open-circuit target-mode frequency the shunt is tuned to: w_oc = w_sc sqrt(1 + k^2)."""
        omegas = []
        for p, mode_index in enumerate(self.target_index):
            stiffness = self._stiffness(plate, self.modes[mode_index]) * self.a * self.b / 4
            k2 = self.coupling[p, mode_index] ** 2 / (self.capacitance[p] * stiffness)
            omegas.append(np.sqrt(stiffness / self.modal_mass * (1 + k2)))
        return np.array(omegas)

    # -------------------------------------------------------------- shunt circuits
    def shunt_impedances(self, omega: float, kind: str, negative_capacitance_ratio: float = -0.9) -> np.ndarray:
        """Shunt impedance Z_p(w) of every patch [Ohm].   [LIT] Hagood & von Flotow (1991)

        RL:   L = 1 / (Cp w_t^2),  R = sqrt(2) k / (Cp w_t)    (w_t: tuned open-circuit frequency)
        R:    R = 1 / (Cp w_t)
        negC: R = 0.5 / (Cp w_t) in series with C_neg = ratio * Cp
        open / short: very large / very small impedance
        """
        impedances = []
        for p, mode_index in enumerate(self.target_index):
            cp, w_t = self.capacitance[p], self.tuning_omega[p]
            k = np.sqrt(self.coupling_factor2[p, mode_index])
            if kind == "RL":
                impedances.append(np.sqrt(2) * k / (cp * w_t) + 1j * omega / (cp * w_t**2))
            elif kind == "R":
                impedances.append(1 / (cp * w_t) + 0j)
            elif kind == "negC":
                impedances.append(0.5 / (cp * w_t) + 1 / (1j * omega * negative_capacitance_ratio * cp))
            elif kind == "open":
                impedances.append(1e12 + 0j)
            elif kind == "short":
                impedances.append(1e-9 + 0j)
            else:
                raise ValueError(f"unknown shunt type {kind!r}")
        return np.array(impedances)

    def electrical_stiffness(self, omega: float, kind: str, negative_capacitance_ratio: float = -0.9) -> np.ndarray:
        """sum_p theta_p theta_p^T / Cp * g_p / (1 + g_p),   g_p = j w Cp Z_p."""
        added = np.zeros((len(self.modes), len(self.modes)), complex)
        for p, impedance in enumerate(self.shunt_impedances(omega, kind, negative_capacitance_ratio)):
            g = 1j * omega * self.capacitance[p] * impedance
            added += np.outer(self.coupling[p], self.coupling[p]) / self.capacitance[p] * g / (1 + g)
        return added

    # -------------------------------------------------------------- responses
    def natural_frequencies_hz(self, kind: str) -> np.ndarray:
        """Undamped natural frequencies with 'short' or 'open' patches (generalised eigenproblem)."""
        stiffness = np.diag(self.modal_stiffness).astype(float)
        if kind == "open":
            stiffness = stiffness + np.real(self.electrical_stiffness(1.0, "open"))
        eigenvalues = np.linalg.eigvalsh(stiffness / self.modal_mass)
        return np.sqrt(np.sort(eigenvalues)) / (2 * np.pi)

    def _radiation_modal(self, omega: float, air: Air) -> np.ndarray:
        resistance = rayleigh_resistance_matrix(omega, air, self.distances, self.element_area)
        return self.mode_shapes.T @ resistance @ self.mode_shapes

    def modal_radiation_efficiency(self, frequency_hz: float, mode_index: int, air: Air) -> float:
        """sigma_mn = W_rad / (rho c A <v^2>), with <v^2> = |v|^2 / 8 for a sine mode (time and space)."""
        omega = 2 * np.pi * frequency_hz
        r_modal = self._radiation_modal(omega, air)
        power = 0.5 * np.real(r_modal[mode_index, mode_index])          # for unit modal velocity amplitude
        return float(power / (air.impedance_rayl * self.a * self.b / 8))

    def modal_response(self, frequency_hz: float, incidence_rad: float, azimuth_rad: float, shunt: str,
                       negative_capacitance_ratio: float = -0.9) -> tuple[np.ndarray, np.ndarray]:
        """Modal velocities (n_modes, complex) for a unit-amplitude incident plane wave, and the cabin-side
        modal radiation resistance matrix. Blocked pressure p_b = 2 p_i exp(-j(kx x + ky y)), kx along the axis.
        The plate velocity is v(x, y) = sum_m velocity_m phi_m(x, y)."""
        omega = 2 * np.pi * frequency_hz
        outside, cabin = self.flight.outside, self.flight.cabin
        k1 = omega / outside.sound_speed_m_s
        kx = k1 * np.sin(incidence_rad) * np.cos(azimuth_rad)
        ky = k1 * np.sin(incidence_rad) * np.sin(azimuth_rad)
        blocked = 2.0 * np.exp(-1j * (kx * self.x + ky * self.y))
        force = self.mode_shapes.T @ blocked * self.element_area
        dynamic = np.diag(self.modal_stiffness * (1 + 1j * self.skin.loss_factor) - omega**2 * self.modal_mass).astype(complex)
        dynamic += self.electrical_stiffness(omega, shunt, negative_capacitance_ratio)
        r_cabin = self._radiation_modal(omega, cabin)
        dynamic += 1j * omega * (r_cabin + self._radiation_modal(omega, outside))
        return 1j * omega * np.linalg.solve(dynamic, force), r_cabin

    def transmission_coefficient(self, frequency_hz: float, incidence_rad: float, azimuth_rad: float,
                                 shunt: str, negative_capacitance_ratio: float = -0.9) -> float:
        """tau = W_radiated into the cabin / W_incident on the bay, for an incident plane wave."""
        velocity, r_cabin = self.modal_response(frequency_hz, incidence_rad, azimuth_rad, shunt, negative_capacitance_ratio)
        radiated = 0.5 * np.real(np.conj(velocity) @ r_cabin @ velocity)
        incident = np.cos(incidence_rad) * self.a * self.b / (2 * self.flight.outside.impedance_rayl)
        return float(max(radiated / incident, 1e-15))

    def diffuse_transmission(self, frequency_hz: float, shunt: str, n_incidence: int, n_azimuth: int,
                             negative_capacitance_ratio: float = -0.9, max_incidence_deg: float = 78.0) -> float:
        """Diffuse-field tau: Paris average over theta (sin cos weight) and uniform over psi in [0, 90 deg]."""
        incidence = np.linspace(0.02, np.deg2rad(max_incidence_deg), n_incidence)
        azimuth = np.linspace(0.0, np.pi / 2, n_azimuth)
        tau = np.array([[self.transmission_coefficient(frequency_hz, t, a, shunt, negative_capacitance_ratio)
                         for a in azimuth] for t in incidence]).mean(axis=1)
        weight = np.sin(incidence) * np.cos(incidence)
        return float(np.trapezoid(tau * weight, incidence) / np.trapezoid(weight, incidence))

    def summary(self) -> list[dict]:
        """Per patch: target mode, in-flight short-circuit frequency, coupling k^2, Cp, inductance."""
        rows = []
        for p, mode_index in enumerate(self.target_index):
            w_t = self.tuning_omega[p]
            rows.append(dict(patch=p, target_mode=list(self.modes[mode_index]),
                             short_circuit_hz=float(self.short_circuit_omega[mode_index] / (2 * np.pi)),
                             tuned_hz=float(w_t / (2 * np.pi)),
                             coupling_k2=float(self.coupling_factor2[p, mode_index]),
                             capacitance_nf=float(self.capacitance[p] * 1e9),
                             inductance_h=float(1 / (self.capacitance[p] * w_t**2))))
        return rows


def shunt_delta_tl_db(bay: ShuntedBay, bands_hz: np.ndarray, shunt: str, max_frequency_hz: float,
                      n_subbands: int, n_incidence: int, n_azimuth: int, negative_capacitance_ratio: float) -> np.ndarray:
    """Band TL gain of the shunt relative to short-circuited patches (positive = better).   [ASM]

    dTL = 10 log10( <tau_short> / <tau_shunt> ), energy-averaged inside each band.
    The patches' MASS is already in the TMM; this adds only their electrical (damping) effect.
    Bands above `max_frequency_hz` get 0 (bay grid too coarse there).
    """
    delta = np.zeros(len(bands_hz))
    for b, centre in enumerate(bands_hz):
        if centre > max_frequency_hz:
            continue
        frequencies = centre * 2 ** np.linspace(-1 / 6, 1 / 6, n_subbands + 2)[1:-1]
        tau_shunt = np.mean([bay.diffuse_transmission(f, shunt, n_incidence, n_azimuth, negative_capacitance_ratio)
                             for f in frequencies])
        tau_short = np.mean([bay.diffuse_transmission(f, "short", n_incidence, n_azimuth) for f in frequencies])
        delta[b] = 10 * np.log10(tau_short / tau_shunt)
    return delta
