"""Sound transmission through one sidewall build-up.

Engine path:  a plane wave arrives at incidence theta and in-plane direction psi. We tabulate the
              power transmission coefficient tau(band, theta, psi) so every fuselage panel can look
              up its own value.
TBL path:     the blocked wall pressure is a random field spread over wavenumbers. We integrate the
              stack's wavenumber response against the TBL wavenumber spectrum.
Bridge path:  added in parallel to both, driven by the skin velocity (see bridge.py).
The cabin-side radiation uses a finite trim panel ("spatial windowing") unless `panel` is None.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..atmosphere import FlightCondition
from ..tbl import BoundaryLayer
from .bridge import bridge_velocity_transfer, point_bridge_tl_db
from .closed_form import coincidence_frequency
from .radiation import windowed_radiation_efficiency
from .stack import Sidewall
from .tmm import PlateLayer, incident_transmission, stack_response

MIN_TAU = 1e-15        # floor before taking logs (150 dB), numerical guard


@dataclass(frozen=True)
class FinitePanel:
    """Size of the radiating trim panel used for spatial windowing."""
    length_axial_m: float
    length_circ_m: float
    minimum_quadrature: tuple[int, int] = (40, 80)


def subband_frequencies(centre_hz: float, count: int) -> np.ndarray:
    """`count` frequencies spread evenly (log scale) inside one 1/3-octave band."""
    return centre_hz * 2 ** np.linspace(-1 / 6, 1 / 6, count + 2)[1:-1]


def transmission_loss_db(tau: np.ndarray) -> np.ndarray:
    """TL = -10 log10(tau)  [dB]."""
    return -10 * np.log10(np.maximum(tau, MIN_TAU))


def diffuse_average(values: np.ndarray, incidence_rad: np.ndarray) -> np.ndarray:
    """Diffuse-field average over the last two axes (theta, psi).   [LIT] Paris formula

    <x> = int x(theta) sin(theta) cos(theta) dtheta / int sin(theta) cos(theta) dtheta, psi averaged uniformly.
    """
    weight = np.sin(incidence_rad) * np.cos(incidence_rad)
    over_psi = np.mean(values, axis=-1)
    return np.trapezoid(over_psi * weight, incidence_rad, axis=-1) / np.trapezoid(weight, incidence_rad)


# ------------------------------------------------------------------ cabin-side radiation (cached)
@lru_cache(maxsize=4096)
def _windowed_sigma_cached(frequency_hz: float, sound_speed: float, kt_key: tuple, psi_key: tuple,
                           length_axial: float, length_circ: float, minimum_quadrature: tuple) -> np.ndarray:
    kt = np.array(kt_key)[:, None]
    psi = np.array(psi_key)[None, :]
    wavenumber = 2 * np.pi * frequency_hz / sound_speed
    return windowed_radiation_efficiency(wavenumber, kt * np.cos(psi), kt * np.sin(psi),
                                         length_axial, length_circ, minimum_quadrature)


def windowed_sigma_grid(frequency_hz: float, sound_speed: float, kt: np.ndarray, psi: np.ndarray,
                        panel: FinitePanel) -> np.ndarray:
    """sigma(kt, psi) on a grid, memoised because it is the same for every solution."""
    return _windowed_sigma_cached(float(frequency_hz), float(sound_speed), tuple(np.round(kt, 9)),
                                  tuple(np.round(psi, 9)), panel.length_axial_m, panel.length_circ_m,
                                  tuple(panel.minimum_quadrature))


# ------------------------------------------------------------------ engine path: incidence tables
@dataclass(frozen=True)
class TransmissionTables:
    """Band-averaged results of one sidewall for incident plane waves.

    tau_air[b, i, j]          airborne transmission at band b, theta_i, psi_j
    skin_velocity[b, i, j]    <v_skin^2> per incident power per area [(m/s)^2 / (W/m^2)]
    bridge_transfer[b]        T_b, bridge power per area per <v_skin^2>
    """
    bands_hz: np.ndarray
    incidence_rad: np.ndarray
    azimuth_rad: np.ndarray
    tau_air: np.ndarray
    skin_velocity: np.ndarray
    bridge_transfer: np.ndarray

    @property
    def tau_bridge(self) -> np.ndarray:
        return self.bridge_transfer[:, None, None] * self.skin_velocity

    @property
    def tau_total(self) -> np.ndarray:
        return self.tau_air + self.tau_bridge

    def diffuse_tl_db(self, path: str = "total") -> np.ndarray:
        """Diffuse-field TL per band for 'air', 'bridge' or 'total'."""
        table = {"air": self.tau_air, "bridge": self.tau_bridge, "total": self.tau_total}[path]
        return transmission_loss_db(diffuse_average(table, self.incidence_rad))


def _band_tables(bands_hz: np.ndarray, sidewall: Sidewall, flight: FlightCondition, panel: FinitePanel | None,
                 incidence: np.ndarray, azimuth: np.ndarray, n_subbands: int) -> tuple[np.ndarray, np.ndarray]:
    """tau_air and skin velocity per incident power, shape (n_bands, n_incidence, n_azimuth)."""
    outside, cabin = flight.outside, flight.cabin
    theta_grid, psi_grid = np.meshgrid(incidence, azimuth, indexing="ij")
    tau_air = np.zeros((len(bands_hz), len(incidence), len(azimuth)))
    skin_velocity = np.zeros_like(tau_air)
    for b, centre in enumerate(bands_hz):
        for frequency in subband_frequencies(centre, n_subbands):
            omega = 2 * np.pi * frequency
            kt = omega / outside.sound_speed_m_s * np.sin(theta_grid)
            z_cab = None
            if panel is not None:
                sigma = windowed_sigma_grid(frequency, cabin.sound_speed_m_s, kt[:, 0], azimuth, panel)
                z_cab = cabin.impedance_rayl * sigma
            tau_air[b] += incident_transmission(sidewall.layers, frequency, theta_grid, psi_grid, outside, cabin, z_cab)
            response = stack_response(sidewall.layers, omega, kt, psi_grid, outside, cabin, z_cab)
            # <v^2> / (I cos(theta)) = 4 rho1 c1 |v_skin/p_b|^2 / cos(theta)   (p_b = 2 p_i)
            skin_velocity[b] += (4 * outside.impedance_rayl * np.abs(response.skin_velocity_ratio) ** 2
                                 / np.maximum(np.cos(theta_grid), 1e-6))
    return tau_air / n_subbands, skin_velocity / n_subbands


def incidence_tables(bands_hz: np.ndarray, sidewall: Sidewall, flight: FlightCondition,
                     panel: FinitePanel | None, bridge_cfg: dict, n_incidence: int, n_azimuth: int,
                     n_subbands: int, n_bridge_incidence: int = 181) -> TransmissionTables:
    """Tabulate tau(theta, psi) and skin velocity for every band (energy average over sub-bands).

    The bridge calibration needs the diffuse-field skin velocity, which has a narrow coincidence
    peak at high frequency; it is averaged on a separate, dense grid of `n_bridge_incidence` angles.
    """
    incidence = np.deg2rad(np.linspace(0.0, 89.0, n_incidence))
    azimuth = np.deg2rad(np.linspace(0.0, 90.0, n_azimuth))
    tau_air, skin_velocity = _band_tables(bands_hz, sidewall, flight, panel, incidence, azimuth, n_subbands)

    bridge_transfer = np.zeros(len(bands_hz))
    if bridge_cfg["enabled"]:
        outside = flight.outside
        skin_coincidence = coincidence_frequency(outside.sound_speed_m_s, sidewall.skin.surface_mass_kg_m2,
                                                 sidewall.skin.bending_stiffness_nm)
        tl_bridge = point_bridge_tl_db(bands_hz, sidewall.skin.surface_mass_kg_m2, sidewall.trim.surface_mass_kg_m2,
                                       skin_coincidence, bridge_cfg["point_spacing"], bridge_cfg["isolator_gain"])
        dense = np.deg2rad(np.linspace(0.0, 89.0, n_bridge_incidence))
        _, dense_velocity = _band_tables(bands_hz, sidewall, flight, panel, dense, azimuth, n_subbands)
        bridge_transfer = bridge_velocity_transfer(10 ** (-tl_bridge / 10), diffuse_average(dense_velocity, dense))
    return TransmissionTables(bands_hz=np.asarray(bands_hz, float), incidence_rad=incidence, azimuth_rad=azimuth,
                              tau_air=tau_air, skin_velocity=skin_velocity, bridge_transfer=bridge_transfer)


# ------------------------------------------------------------------ TBL path: wavenumber integral
def free_wavenumbers(omega: float, sidewall: Sidewall) -> list[float]:
    """Free bending/membrane wavenumbers of the stiff plates: D k^4 + N k^2 = w^2 m''.

    Used only to place extra integration points where the response peaks sharply.
    """
    result = []
    for layer in sidewall.layers:
        if not isinstance(layer, PlateLayer) or layer.plate.bending_stiffness_nm < 1.0:
            continue
        plate = layer.plate
        for tension in {plate.tension_axial_n_m, plate.tension_circ_n_m}:
            d, m = plate.bending_stiffness_nm, plate.surface_mass_kg_m2
            k_squared = (-tension + np.sqrt(tension**2 + 4 * d * omega**2 * m)) / (2 * d)
            result.append(float(np.sqrt(k_squared)))
    return result


def wavenumber_grid(omega: float, flight: FlightCondition, convective_wavenumber: float,
                    sidewall: Sidewall, n_points: int) -> np.ndarray:
    """Radial wavenumber points: log-spaced overall, dense around acoustic and structural peaks."""
    k_cabin = omega / flight.cabin.sound_speed_m_s
    k_outside = omega / flight.outside.sound_speed_m_s
    structural = free_wavenumbers(omega, sidewall)
    k_max = 4 * max([convective_wavenumber, k_outside, *structural])
    pieces = [np.geomspace(1e-3 * k_cabin, k_max, n_points)]
    for k_peak in [k_cabin, k_outside, *structural]:
        pieces.append(np.linspace(0.6 * k_peak, 1.4 * k_peak, n_points // 2))
    return np.unique(np.concatenate(pieces))


def _sigma_on_fine_grid(frequency_hz: float, flight: FlightCondition, kt_fine: np.ndarray, psi: np.ndarray,
                        panel: FinitePanel) -> np.ndarray:
    """Windowed sigma on a coarse kt grid, interpolated (log-log) onto the fine grid."""
    k_cabin = 2 * np.pi * frequency_hz / flight.cabin.sound_speed_m_s
    lobe = 2 * np.pi / min(panel.length_axial_m, panel.length_circ_m)
    coarse = np.unique(np.concatenate([
        np.geomspace(kt_fine[0], kt_fine[-1], 100),
        np.linspace(max(k_cabin - 4 * lobe, kt_fine[0]), k_cabin + 4 * lobe, 40)]))
    sigma_coarse = windowed_sigma_grid(frequency_hz, flight.cabin.sound_speed_m_s, coarse, psi, panel)
    log_sigma = np.log(np.maximum(sigma_coarse, 1e-300))
    fine = np.empty((len(kt_fine), len(psi)))
    for j in range(len(psi)):
        fine[:, j] = np.exp(np.interp(np.log(kt_fine), np.log(coarse), log_sigma[:, j]))
    return fine


@dataclass(frozen=True)
class WavenumberResponse:
    """Integrals of a normalised pressure spectrum against the stack response at one frequency.

    transmitted_per_ms   W''_cabin / <p_b^2>      [(W/m^2)/Pa^2]
    skin_velocity_per_ms <v_skin^2> / <p_b^2>     [(m/s)^2/Pa^2]
    """
    transmitted_per_ms: float
    skin_velocity_per_ms: float


def wavenumber_response(spectrum, frequency_hz: float, sidewall: Sidewall, flight: FlightCondition,
                        panel: FinitePanel | None, convective_wavenumber: float, n_wavenumbers: int,
                        n_directions: int) -> WavenumberResponse:
    """Integrate a normalised blocked-pressure spectrum phi(kx, ky) against the stack response.

    [LIT] Maxit et al. (2019) "wavenumber sensitivity functions"; Villot et al. (2001)
    W''  = int int phi(kx, ky) Re(Z2) |v_trim / p_b|^2 dkx dky
    <v^2> = int int phi(kx, ky) |v_skin / p_b|^2 dkx dky
    Polar coordinates kx = kt cos(psi), ky = kt sin(psi); the response is symmetric in the signs of
    kx and ky, so we integrate psi over [0, pi/2] and add phi at the four sign combinations.
    `spectrum(kx, ky)` must integrate to 1 over the whole plane.
    """
    omega = 2 * np.pi * frequency_hz
    psi = (np.arange(n_directions) + 0.5) * (np.pi / 2) / n_directions
    kt = wavenumber_grid(omega, flight, convective_wavenumber, sidewall, n_wavenumbers)
    kt_grid, psi_grid = np.meshgrid(kt, psi, indexing="ij")
    kx, ky = kt_grid * np.cos(psi_grid), kt_grid * np.sin(psi_grid)
    phi = spectrum(kx, ky) + spectrum(-kx, ky) + spectrum(kx, -ky) + spectrum(-kx, -ky)

    z_cab = None
    if panel is not None:
        z_cab = flight.cabin.impedance_rayl * _sigma_on_fine_grid(frequency_hz, flight, kt, psi, panel)
    response = stack_response(sidewall.layers, omega, kt_grid, psi_grid, flight.outside, flight.cabin, z_cab)
    transmitted = np.real(response.cabin_impedance) * np.abs(response.cabin_velocity_ratio) ** 2
    skin = np.abs(response.skin_velocity_ratio) ** 2

    d_psi = (np.pi / 2) / n_directions

    def integrate(values: np.ndarray) -> float:
        over_psi = np.sum(phi * values, axis=1) * d_psi
        return float(np.trapezoid(over_psi * kt, kt))

    return WavenumberResponse(transmitted_per_ms=integrate(transmitted), skin_velocity_per_ms=integrate(skin))


@dataclass(frozen=True)
class TblTransmission:
    """TBL path of one sidewall, per band (per unit skin area).

    coupling_penalty_db: how much less the TBL transmits than a diffuse sound field with the same
    wall pressure. This is the archive's "20-30 dB" number, here computed instead of assumed.
    """
    bands_hz: np.ndarray
    blocked_mean_square_pa2: np.ndarray
    transmitted_air_w_m2: np.ndarray
    transmitted_bridge_w_m2: np.ndarray
    coupling_penalty_db: np.ndarray

    @property
    def transmitted_total_w_m2(self) -> np.ndarray:
        return self.transmitted_air_w_m2 + self.transmitted_bridge_w_m2


def tbl_transmission(bands_hz: np.ndarray, sidewall: Sidewall, flight: FlightCondition, layer: BoundaryLayer,
                     band_mean_square_pa2: np.ndarray, spectrum_model, alpha_x: float, alpha_y: float,
                     panel: FinitePanel | None, tables: TransmissionTables, n_wavenumbers: int,
                     n_directions: int, n_subbands: int) -> TblTransmission:
    """Transmitted TBL power per band: <p_b^2>_band x (wavenumber response averaged over the band)."""
    air = np.zeros(len(bands_hz))
    velocity = np.zeros(len(bands_hz))
    for b, centre in enumerate(bands_hz):
        for frequency in subband_frequencies(centre, n_subbands):
            k_conv = 2 * np.pi * frequency / layer.convection_velocity_m_s

            def spectrum(kx, ky, kc=k_conv):
                return spectrum_model(kx, ky, kc, alpha_x, alpha_y)

            result = wavenumber_response(spectrum, frequency, sidewall, flight, panel, k_conv,
                                         n_wavenumbers, n_directions)
            air[b] += result.transmitted_per_ms / n_subbands
            velocity[b] += result.skin_velocity_per_ms / n_subbands
    mean_square = np.asarray(band_mean_square_pa2, float)
    transmitted_air = mean_square * air
    transmitted_bridge = tables.bridge_transfer * mean_square * velocity
    # Diffuse sound field with the same blocked mean-square: incident p^2 = p_b^2 / 2,
    # incident power per area = p^2 / (4 rho1 c1)  ->  W''_diffuse = tau_d p_b^2 / (8 rho1 c1)
    tau_diffuse = diffuse_average(tables.tau_total, tables.incidence_rad)
    w_diffuse = tau_diffuse * mean_square / (8 * flight.outside.impedance_rayl)
    penalty = 10 * np.log10(w_diffuse / np.maximum(transmitted_air + transmitted_bridge, 1e-300))
    return TblTransmission(bands_hz=np.asarray(bands_hz, float), blocked_mean_square_pa2=mean_square,
                           transmitted_air_w_m2=transmitted_air, transmitted_bridge_w_m2=transmitted_bridge,
                           coupling_penalty_db=penalty)
