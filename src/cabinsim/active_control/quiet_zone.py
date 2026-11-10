"""Quiet zone of the active trim panel: pressure maps around the head with ANC off and on.

Instead of a long time-domain run per map, we use the point the tonal FxLMS converges to. With the
update W <- (1 - nu) W - mu G (d + S W) (see fxlms.run_tonal_fxlms), the fixed point is
    W = -(mu G S + nu I)^-1 mu G d,     G = (S^H S + beta I)^-1 S^H
The disturbance d is the controlled panel's field plus a diffuse field (other panels + reverberation),
with the share of each taken from the cabin model. Maps are averaged over random diffuse realisations.
"""
from __future__ import annotations

import numpy as np

from .fxlms import tonal_preconditioners
from .primary_field import diffuse_tone, random_directions
from .simulation import AncGeometry

QUIET_THRESHOLD_DB = 10.0


def steady_state_weights(secondary_mic: np.ndarray, disturbance_mic: np.ndarray, step_size: float,
                         regularisation: float, leakage: float) -> np.ndarray:
    """Complex actuator amplitudes at convergence of the preconditioned tonal FxLMS (one tone)."""
    preconditioner = tonal_preconditioners([secondary_mic], regularisation)[0]
    n_act = secondary_mic.shape[1]
    system = step_size * preconditioner @ secondary_mic + leakage * np.eye(n_act)
    return -np.linalg.solve(system, step_size * preconditioner @ disturbance_mic)


def slice_points(geometry: AncGeometry, width_m: float, depth_m: float, n_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal slice through the head centre: x along the panel, z = distance into the cabin."""
    head = geometry.head_m
    xs = np.linspace(head[0] - width_m / 2, head[0] + width_m / 2, n_points)
    zs = np.linspace(0.02, depth_m, n_points)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
    points = np.stack([grid_x.ravel(), np.full(grid_x.size, head[1]), grid_z.ravel()], axis=-1)
    return xs, zs, points


def quiet_zone_maps(geometry: AncGeometry, active_cfg: dict, frequencies_hz: list[float], controlled_share: list[float],
                    width_m: float, depth_m: float, n_points: int, n_realisations: int, rng: np.random.Generator) -> dict:
    """Mean-square maps (dB re. the uncontrolled mean-square at the head) with ANC off and on.

    zone_diameter_m: diameter of a circle with the same area as the part of the slice where the
    reduction is at least 10 dB.
    """
    xs, zs, slice_pts = slice_points(geometry, width_m, depth_m, n_points)
    n_mics = len(geometry.mics_m)
    points = np.vstack([geometry.mics_m, geometry.head_m[None, :], slice_pts])
    head_index = n_mics
    cell_area = (xs[1] - xs[0]) * (zs[1] - zs[0])
    sound_speed = geometry.plant.air.sound_speed_m_s
    result = dict(x_m=xs.tolist(), z_m=zs.tolist(), frequencies_hz=list(map(float, frequencies_hz)),
                  head_m=geometry.head_m.tolist(), mics_m=geometry.mics_m.tolist(),
                  off_db=[], on_db=[], zone_diameter_m=[], wavelength_over_10_m=[], controlled_share=list(controlled_share))
    for frequency, share in zip(frequencies_hz, controlled_share):
        primary, secondary = geometry.plant.frequency_responses(np.array([frequency]), points)
        primary, secondary = primary[0], secondary[0]
        panel = primary * np.sqrt(2 * share) / abs(primary[head_index])
        off = np.zeros(len(points))
        on = np.zeros(len(points))
        for _ in range(n_realisations):
            directions = random_directions(active_cfg["diffuse_plane_waves"], rng)
            phases = rng.uniform(0, 2 * np.pi, len(directions))
            disturbance = panel + diffuse_tone(points, frequency, sound_speed, directions, phases) * np.sqrt(2 * (1 - share))
            weights = steady_state_weights(secondary[:n_mics], disturbance[:n_mics], active_cfg["step_size_tonal"],
                                           active_cfg["regularisation_tonal"], active_cfg["leakage_tonal"])
            controlled = disturbance + secondary @ weights
            off += np.abs(disturbance) ** 2 / 2 / n_realisations
            on += np.abs(controlled) ** 2 / 2 / n_realisations
        off_map = 10 * np.log10(off[n_mics + 1:]).reshape(n_points, n_points)
        on_map = 10 * np.log10(on[n_mics + 1:]).reshape(n_points, n_points)
        quiet_area = np.count_nonzero(off_map - on_map >= QUIET_THRESHOLD_DB) * cell_area
        result["off_db"].append(np.round(off_map, 2).tolist())
        result["on_db"].append(np.round(on_map, 2).tolist())
        result["zone_diameter_m"].append(float(2 * np.sqrt(quiet_area / np.pi)))
        result["wavelength_over_10_m"].append(float(sound_speed / frequency / 10))
    return result
