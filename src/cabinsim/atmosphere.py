"""Air properties from the International Standard Atmosphere (ISA).

Outside air at cruise altitude and cabin air at cabin pressure altitude are both computed
here, so density and sound speed are always consistent with each other.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ISA definition constants [LIT] ICAO Doc 7488/3 (1993)
SEA_LEVEL_PRESSURE_PA = 101325.0
SEA_LEVEL_TEMPERATURE_K = 288.15
LAPSE_RATE_K_PER_M = 0.0065
TROPOPAUSE_ALTITUDE_M = 11000.0
GAS_CONSTANT_AIR = 287.053          # J/(kg K)
GRAVITY = 9.80665                   # m/s^2
HEAT_CAPACITY_RATIO = 1.4
# Sutherland's law for air [LIT] (White, Viscous Fluid Flow, 2006)
SUTHERLAND_MU0 = 1.716e-5           # Pa s at 273.15 K
SUTHERLAND_T0 = 273.15              # K
SUTHERLAND_S = 110.4                # K


@dataclass(frozen=True)
class Air:
    """State of a body of air. All SI units."""
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    sound_speed_m_s: float
    dynamic_viscosity_pa_s: float

    @property
    def impedance_rayl(self) -> float:
        """Characteristic impedance Z = rho c  [Pa s/m]."""
        return self.density_kg_m3 * self.sound_speed_m_s

    @property
    def kinematic_viscosity_m2_s(self) -> float:
        return self.dynamic_viscosity_pa_s / self.density_kg_m3


def isa_temperature_pressure(altitude_m: float, temperature_offset_k: float = 0.0) -> tuple[float, float]:
    """ISA temperature [K] and pressure [Pa] up to 20 km.  [LIT] ICAO Doc 7488

    Troposphere:  T = T0 - L h,  p = p0 (T/T0)^(g / (R L))
    Above 11 km:  T = 216.65 K,  p = p11 exp(-g (h - 11000) / (R T11))
    `temperature_offset_k` shifts T only (ISA+dT day), pressure stays standard.
    """
    exponent = GRAVITY / (GAS_CONSTANT_AIR * LAPSE_RATE_K_PER_M)
    if altitude_m <= TROPOPAUSE_ALTITUDE_M:
        temperature = SEA_LEVEL_TEMPERATURE_K - LAPSE_RATE_K_PER_M * altitude_m
        pressure = SEA_LEVEL_PRESSURE_PA * (temperature / SEA_LEVEL_TEMPERATURE_K) ** exponent
    else:
        temperature = SEA_LEVEL_TEMPERATURE_K - LAPSE_RATE_K_PER_M * TROPOPAUSE_ALTITUDE_M
        pressure_11km = SEA_LEVEL_PRESSURE_PA * (temperature / SEA_LEVEL_TEMPERATURE_K) ** exponent
        pressure = pressure_11km * np.exp(-GRAVITY * (altitude_m - TROPOPAUSE_ALTITUDE_M)
                                          / (GAS_CONSTANT_AIR * temperature))
    return temperature + temperature_offset_k, float(pressure)


def air_state(temperature_k: float, pressure_pa: float) -> Air:
    """Ideal gas air:  rho = p / (R T),  c = sqrt(gamma R T),  mu from Sutherland's law.  [LIT]"""
    density = pressure_pa / (GAS_CONSTANT_AIR * temperature_k)
    sound_speed = np.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT_AIR * temperature_k)
    viscosity = (SUTHERLAND_MU0 * (temperature_k / SUTHERLAND_T0) ** 1.5
                 * (SUTHERLAND_T0 + SUTHERLAND_S) / (temperature_k + SUTHERLAND_S))
    return Air(temperature_k=temperature_k, pressure_pa=pressure_pa, density_kg_m3=float(density),
               sound_speed_m_s=float(sound_speed), dynamic_viscosity_pa_s=float(viscosity))


@dataclass(frozen=True)
class FlightCondition:
    """Outside air, cabin air, flight speed and the pressure difference across the skin."""
    outside: Air
    cabin: Air
    flight_speed_m_s: float
    pressure_difference_pa: float


def flight_condition(flight_cfg: dict) -> FlightCondition:
    """Build the flight condition from flight.yaml values.

    U_inf = M c_outside;  dp = p_cabin - p_outside (positive: skin in tension).
    """
    t_out, p_out = isa_temperature_pressure(flight_cfg["cruise_altitude"], flight_cfg["ground_temperature_offset"])
    outside = air_state(t_out, p_out)
    _, p_cab = isa_temperature_pressure(flight_cfg["cabin_altitude"])
    cabin = air_state(flight_cfg["cabin_temperature"], p_cab)
    return FlightCondition(outside=outside, cabin=cabin,
                           flight_speed_m_s=flight_cfg["cruise_mach"] * outside.sound_speed_m_s,
                           pressure_difference_pa=p_cab - p_out)
