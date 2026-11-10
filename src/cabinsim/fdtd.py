"""2D acoustic FDTD of the fuselage cross-section (QUALITATIVE illustration).

Equations (linear acoustics with a Rayleigh drag term for porous blankets)   [LIT] Botteldooren (1995)
    rho dv/dt + sigma v = -grad p
    dp/dt = -kappa div v
Staggered (Yee) grid: p at cell centres (nx, nz), vx on vertical faces (nx+1, nz), vz on
horizontal faces (nx, nz+1). Drag and edge damping are applied semi-implicitly, which is stable
for any sigma. The browser version (demo/js/fdtd_core.js) uses exactly the same update.

Simplifications, shown on the demo page:
  * 2D cross-section: the engine acts as a line source (cylindrical spreading), no flow;
  * plates (skin, septum, trim) are thin dense fluid rings = limp masses, no bending/bay modes;
  * the sidewall is drawn `exaggeration` times thicker. Each fluid layer is stretched exactly
    (rho/e, kappa*e, sigma*1/e), which leaves normal-incidence transmission unchanged;
  * shunted piezo patches and active control are not in this view (see bay and quiet-zone pages).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import FlightCondition

CFL_SAFETY = 0.5


@dataclass(frozen=True)
class FluidMaterial:
    name: str
    density_kg_m3: float
    bulk_modulus_pa: float
    flow_resistivity: float = 0.0
    rigid: bool = False

    @property
    def sound_speed_m_s(self) -> float:
        return float(np.sqrt(self.bulk_modulus_pa / self.density_kg_m3)) if not self.rigid else 0.0


@dataclass
class FdtdModel:
    """Material map on a regular grid. Cell (i, j) centre is at (y0 + i dx, z0 + j dx)."""
    cell_size_m: float
    y0_m: float
    z0_m: float
    material_index: np.ndarray            # (nx, nz) uint8
    materials: list[FluidMaterial]
    sponge_cells: int
    source_cell: tuple[int, int] = (0, 0)
    probe_cell: tuple[int, int] = (0, 0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.material_index.shape

    def cell_of(self, y_m: float, z_m: float) -> tuple[int, int]:
        return int(round((y_m - self.y0_m) / self.cell_size_m)), int(round((z_m - self.z0_m) / self.cell_size_m))

    @property
    def max_sound_speed(self) -> float:
        return max(m.sound_speed_m_s for m in self.materials)

    @property
    def time_step_s(self) -> float:
        """CFL limit in 2D: dt = safety * dx / (c_max sqrt(2))."""
        return CFL_SAFETY * self.cell_size_m / (self.max_sound_speed * np.sqrt(2))


def homogeneous_model(material: FluidMaterial, shape: tuple[int, int], cell_size_m: float, sponge_cells: int) -> FdtdModel:
    return FdtdModel(cell_size_m=cell_size_m, y0_m=0.0, z0_m=0.0, material_index=np.zeros(shape, np.uint8),
                     materials=[material], sponge_cells=sponge_cells)


# ------------------------------------------------------------------ coefficients and stepping
def _face_average(values: np.ndarray, axis: int) -> np.ndarray:
    """Mean of the two cells on either side of every interior face along `axis`."""
    if axis == 0:
        return 0.5 * (values[1:, :] + values[:-1, :])
    return 0.5 * (values[:, 1:] + values[:, :-1])


def sponge_profile(shape: tuple[int, int], cells: int, sound_speed: float, cell_size: float,
                   along_z: bool = True) -> np.ndarray:
    """Damping rate gamma [1/s] growing quadratically into a `cells`-wide edge layer.

    gamma_max = 10 c / (cells dx): roughly 60 dB round-trip attenuation, gentle enough not to reflect.
    `along_z=False` puts the layer only at the left/right ends (1D plane-wave runs).
    """
    nx, nz = shape
    if cells <= 0:
        return np.zeros(shape)
    depth_x = np.maximum(cells - np.minimum(np.arange(nx), np.arange(nx)[::-1]), 0) / cells
    depth_z = np.maximum(cells - np.minimum(np.arange(nz), np.arange(nz)[::-1]), 0) / cells
    depth = np.maximum(depth_x[:, None], depth_z[None, :] if along_z else 0.0)
    return 10 * sound_speed / (cells * cell_size) * depth**2


@dataclass
class Coefficients:
    """Per-cell and per-face update factors (same names as in fdtd_core.js)."""
    vx_keep: np.ndarray     # (nx-1, nz) interior vertical faces
    vx_push: np.ndarray
    vz_keep: np.ndarray     # (nx, nz-1)
    vz_push: np.ndarray
    p_keep: np.ndarray      # (nx, nz)
    p_push: np.ndarray


def coefficients(model: FdtdModel, dt: float) -> Coefficients:
    """Semi-implicit factors:  v_new = keep v - push grad(p),  p_new = keep p - push div(v).

    keep = (1 - a) / (1 + a), push = (dt / (rho dx)) / (1 + a), a = (sigma/rho + gamma) dt / 2.
    Faces touching a rigid cell get push = keep = 0 (velocity stays zero).
    """
    density = np.array([m.density_kg_m3 for m in model.materials])[model.material_index]
    bulk = np.array([m.bulk_modulus_pa for m in model.materials])[model.material_index]
    resistivity = np.array([m.flow_resistivity for m in model.materials])[model.material_index]
    rigid = np.array([m.rigid for m in model.materials])[model.material_index]
    outside_speed = model.materials[0].sound_speed_m_s or model.max_sound_speed
    gamma = sponge_profile(model.shape, model.sponge_cells, outside_speed, model.cell_size_m,
                           along_z=model.shape[1] > 2 * model.sponge_cells)
    dx = model.cell_size_m

    def face(axis: int) -> tuple[np.ndarray, np.ndarray]:
        rho = _face_average(np.where(rigid, 1.0, density), axis)
        sigma = _face_average(resistivity, axis)
        damping = _face_average(gamma, axis)
        blocked = _face_average(rigid.astype(float), axis) > 0
        a = (sigma / rho + damping) * dt / 2
        keep = np.where(blocked, 0.0, (1 - a) / (1 + a))
        push = np.where(blocked, 0.0, dt / (rho * dx) / (1 + a))
        return keep, push

    vx_keep, vx_push = face(0)
    vz_keep, vz_push = face(1)
    b = gamma * dt / 2
    p_keep = np.where(rigid, 0.0, (1 - b) / (1 + b))
    p_push = np.where(rigid, 0.0, bulk * dt / dx / (1 + b))
    return Coefficients(vx_keep, vx_push, vz_keep, vz_push, p_keep, p_push)


class FdtdSolver:
    """Reference stepper. Outer box faces stay at zero velocity (rigid), behind the sponge."""

    def __init__(self, model: FdtdModel):
        self.model = model
        self.dt = model.time_step_s
        self.c = coefficients(model, self.dt)
        nx, nz = model.shape
        self.p = np.zeros((nx, nz))
        self.vx = np.zeros((nx + 1, nz))
        self.vz = np.zeros((nx, nz + 1))
        self.time_s = 0.0

    def step(self, source_rate: float = 0.0) -> None:
        """Advance one time step. `source_rate` [Pa/s] is added as dp/dt at the source cell (soft source);
        using a rate, not a pressure per step, keeps the source strength independent of dt."""
        c = self.c
        self.vx[1:-1, :] = c.vx_keep * self.vx[1:-1, :] - c.vx_push * (self.p[1:, :] - self.p[:-1, :])
        self.vz[:, 1:-1] = c.vz_keep * self.vz[:, 1:-1] - c.vz_push * (self.p[:, 1:] - self.p[:, :-1])
        divergence = (self.vx[1:, :] - self.vx[:-1, :]) + (self.vz[:, 1:] - self.vz[:, :-1])
        self.p = c.p_keep * self.p - c.p_push * divergence
        i, j = self.model.source_cell
        self.p[i, j] += source_rate * self.dt
        self.time_s += self.dt


def tone_source(time_s: float, frequency_hz: float, ramp_periods: float = 3.0) -> float:
    """sin(2 pi f t) with a smooth raised-cosine start over `ramp_periods` periods."""
    ramp_time = ramp_periods / frequency_hz
    envelope = 0.5 * (1 - np.cos(np.pi * min(time_s / ramp_time, 1.0)))
    return float(envelope * np.sin(2 * np.pi * frequency_hz * time_s))


# ------------------------------------------------------------------ cross-section geometry
def limp_plate(name: str, surface_mass: float, thickness_m: float, max_speed: float) -> tuple[FluidMaterial, float]:
    """Plate as a dense fluid ring: rho t = m''. The bulk modulus rho c_max^2 keeps the ring practically
    incompressible (a limp mass) without lowering the time step."""
    density = surface_mass / thickness_m
    return FluidMaterial(name, density, density * max_speed**2), thickness_m


def stretched_air(name: str, density: float, bulk_modulus: float, thickness_m: float, exaggeration: float,
                  flow_resistivity: float = 0.0) -> tuple[FluidMaterial, float]:
    """Fluid layer drawn `e` times thicker with rho/e, kappa*e, sigma/e: same 1D wave equation in the
    stretched coordinate, so the layer's mass, stiffness and flow resistance are unchanged."""
    return (FluidMaterial(name, density / exaggeration, bulk_modulus * exaggeration, flow_resistivity / exaggeration),
            thickness_m * exaggeration)


def sidewall_layers(solution: dict, materials_cfg: dict, sidewall_depth_m: float, flight: FlightCondition,
                    exaggeration: float, plate_thickness_m: float, max_speed: float) -> list[tuple[FluidMaterial, float]]:
    """(material, displayed thickness) of each sidewall layer, outside -> inside (same order as the TMM)."""
    cabin = flight.cabin
    kappa = cabin.density_kg_m3 * cabin.sound_speed_m_s**2
    skin_cfg = solution["skin"]
    solid = materials_cfg["solids"][skin_cfg["material"]]
    skin_mass = solid["density"] * skin_cfg["thickness"] + skin_cfg["added_mass"]
    blanket = solution["blanket"]
    porous = materials_cfg["porous"][blanket["material"]]
    layers = [limp_plate("skin", skin_mass, plate_thickness_m, max_speed),
              stretched_air("blanket", cabin.density_kg_m3, kappa, blanket["thickness"], exaggeration,
                            porous["flow_resistivity"])]
    if solution["septum"]["enabled"]:
        layers.append(limp_plate("septum", solution["septum"]["surface_mass"], plate_thickness_m, max_speed))
    layers.append(stretched_air("air gap", cabin.density_kg_m3, kappa, sidewall_depth_m - blanket["thickness"], exaggeration))
    trim_mass = materials_cfg["panels"][solution["trim"]["panel"]]["surface_mass"] + solution["trim"]["added_mass"]
    layers.append(limp_plate("trim", trim_mass, plate_thickness_m, max_speed))
    return layers


def cross_section_model(solution: dict, materials_cfg: dict, fuselage_cfg: dict,
                        engine_position_m: np.ndarray, head_m: np.ndarray, visuals_cfg: dict,
                        flight: FlightCondition) -> FdtdModel:
    """Material map of the cross-section for one solution.

    Materials: 0 outside air, 1 cabin air, then the sidewall layers, then a rigid floor.
    Radial positions use the distance of each cell centre from the fuselage axis.
    """
    dx = visuals_cfg["fdtd_cell_size"]
    exaggeration = visuals_cfg["fdtd_sidewall_exaggeration"]
    plate_thickness = visuals_cfg["fdtd_plate_cells"] * dx
    y_min, y_max, z_min, z_max = visuals_cfg["fdtd_domain"]
    outside, cabin = flight.outside, flight.cabin
    max_speed = cabin.sound_speed_m_s * exaggeration          # stretched layers are the fastest material
    outside_air = FluidMaterial("outside air", outside.density_kg_m3, outside.density_kg_m3 * outside.sound_speed_m_s**2)
    cabin_air = FluidMaterial("cabin air", cabin.density_kg_m3, cabin.density_kg_m3 * cabin.sound_speed_m_s**2)

    layers = sidewall_layers(solution, materials_cfg, fuselage_cfg["sidewall_depth"], flight, exaggeration,
                             plate_thickness, max_speed)
    materials =[outside_air, cabin_air] + [m for m, _ in layers] + [FluidMaterial("floor", 1.0, 0.0, rigid=True)]

    ys = np.arange(y_min, y_max + dx / 2, dx)
    zs = np.arange(z_min, z_max + dx / 2, dx)
    y_grid, z_grid = np.meshgrid(ys, zs, indexing="ij")
    radius = np.hypot(y_grid, z_grid)
    outer = fuselage_cfg["radius"]
    index = np.where(radius <= outer, 1, 0).astype(np.uint8)
    edge = outer
    for k, (_, thickness) in enumerate(layers):
        inside_edge = edge - thickness
        index[(radius <= edge) & (radius > inside_edge)] = 2 + k
        edge = inside_edge
    floor_z = fuselage_cfg["floor_height"]
    floor = (np.abs(z_grid - floor_z) <= 0.75 * dx) & (radius < edge)
    index[floor] = len(materials) - 1

    model = FdtdModel(cell_size_m=dx, y0_m=float(ys[0]), z0_m=float(zs[0]), material_index=index, materials=materials,
                      sponge_cells=visuals_cfg["fdtd_sponge_cells"])
    model.source_cell = model.cell_of(engine_position_m[1], engine_position_m[2])
    # keep the head's real distance from the trim although the sidewall is drawn thicker
    head_radius = np.hypot(head_m[1], head_m[2])
    distance_from_trim = (outer - fuselage_cfg["sidewall_depth"]) - head_radius
    scale = (edge - distance_from_trim) / head_radius
    model.probe_cell = model.cell_of(head_m[1] * scale, head_m[2] * scale)
    return model


def material_table(model: FdtdModel) -> list[dict]:
    """JSON-ready list of the materials of a model."""
    return [dict(name=m.name, density=m.density_kg_m3, bulk_modulus=m.bulk_modulus_pa,
                 flow_resistivity=m.flow_resistivity, rigid=m.rigid) for m in model.materials]
