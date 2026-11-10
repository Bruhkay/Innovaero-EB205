"""Turn one solution from solutions.yaml into a list of TMM layers."""
from __future__ import annotations

from dataclasses import dataclass

from ..atmosphere import FlightCondition
from ..materials import Plate, given_plate, pressurised_cylinder, solid_plate
from .tmm import AirLayer, Layer, PlateLayer, PorousLayer


@dataclass(frozen=True)
class Sidewall:
    """Sidewall of one solution: TMM layers plus the two plates other models need."""
    name: str
    layers: list[Layer]
    skin: Plate
    trim: Plate
    added_piezo_mass_kg_m2: float

    @property
    def surface_mass_kg_m2(self) -> float:
        """Total m'' of the build-up (plates + blanket)."""
        total = 0.0
        for layer in self.layers:
            if isinstance(layer, PlateLayer):
                total += layer.plate.surface_mass_kg_m2
            elif isinstance(layer, PorousLayer):
                total += layer.bulk_density_kg_m3 * layer.thickness_m
        return total


def build_sidewall(name: str, solution: dict, materials: dict, fuselage: dict,
                   flight: FlightCondition, piezo_mass_kg_m2: float = 0.0) -> Sidewall:
    """Skin (+damping, +piezo mass, curved, pressurised) | blanket | (septum) | air | trim.

    `piezo_mass_kg_m2` is the PZT patch mass smeared over the bay; only used when the
    solution has piezo_shunt: true. Its electrical effect is handled by the bay model.
    """
    skin_cfg = solution["skin"]
    skin = solid_plate(materials["solids"][skin_cfg["material"]], skin_cfg["thickness"])
    piezo_mass = piezo_mass_kg_m2 if solution["piezo_shunt"] else 0.0
    skin = skin.with_changes(surface_mass_kg_m2=skin.surface_mass_kg_m2 + skin_cfg["added_mass"] + piezo_mass,
                             loss_factor=skin.loss_factor + skin_cfg["added_loss_factor"])
    skin = pressurised_cylinder(skin, fuselage["radius"], flight.pressure_difference_pa,
                                fuselage["pressurised_skin_tension"])

    blanket_cfg = solution["blanket"]
    porous = materials["porous"][blanket_cfg["material"]]
    air_gap = fuselage["sidewall_depth"] - blanket_cfg["thickness"]
    if air_gap < 0:
        raise ValueError(f"{name}: blanket ({blanket_cfg['thickness']} m) is thicker than the sidewall depth")

    trim_cfg = solution["trim"]
    trim = given_plate(materials["panels"][trim_cfg["panel"]])
    trim = trim.with_changes(surface_mass_kg_m2=trim.surface_mass_kg_m2 + trim_cfg["added_mass"])

    layers: list[Layer] = [PlateLayer(skin, "skin"),
                           PorousLayer(blanket_cfg["thickness"], porous["flow_resistivity"], porous["density"])]
    if solution["septum"]["enabled"]:
        septum = given_plate(materials["panels"]["limp_septum"])
        septum = septum.with_changes(surface_mass_kg_m2=solution["septum"]["surface_mass"])
        layers.append(PlateLayer(septum, "septum"))
    layers.append(AirLayer(air_gap))
    layers.append(PlateLayer(trim, "trim"))
    return Sidewall(name=name, layers=layers, skin=skin, trim=trim, added_piezo_mass_kg_m2=piezo_mass)
