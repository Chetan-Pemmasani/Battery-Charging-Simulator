"""Battery model and analytical references for the charging simulations.

The sign convention follows the report: a positive current charges the cell,
so the terminal-voltage contribution of both resistances is positive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

Array = np.ndarray
CurrentProfile = Callable[[float, Array], float]


@dataclass(frozen=True)
class BatteryParameters:
    """Illustrative Thevenin-cell parameters used in the research report."""

    capacity_ah: float = 2.0
    initial_soc: float = 0.20
    initial_polarization_voltage: float = 0.0
    r0_ohm: float = 0.05
    rp_ohm: float = 0.02
    cp_f: float = 2000.0

    def __post_init__(self) -> None:
        if self.capacity_ah <= 0 or self.cp_f <= 0:
            raise ValueError("capacity_ah and cp_f must be positive.")
        if self.r0_ohm < 0 or self.rp_ohm <= 0:
            raise ValueError("r0_ohm must be non-negative and rp_ohm positive.")
        if not 0 <= self.initial_soc <= 1:
            raise ValueError("initial_soc must lie in [0, 1].")

    @property
    def time_constant_s(self) -> float:
        return self.rp_ohm * self.cp_f

    @property
    def initial_state(self) -> Array:
        return np.array([self.initial_soc, self.initial_polarization_voltage], dtype=float)


def open_circuit_voltage(soc: Array | float) -> Array | float:
    """Affine OCV relation specified in the report, in volts."""
    return 3.0 + 1.2 * np.asarray(soc)


def constant_current(current_a: float) -> CurrentProfile:
    """Return a constant-current profile."""
    if not np.isfinite(current_a):
        raise ValueError("current_a must be finite.")
    return lambda _t, _x: float(current_a)


def pulse_current(amplitude_a: float = 1.0, period_s: float = 1200.0) -> CurrentProfile:
    """A 50%-duty charging profile, useful for an optional profile comparison."""
    if amplitude_a < 0 or period_s <= 0:
        raise ValueError("amplitude_a must be non-negative and period_s positive.")
    return lambda t, _x: amplitude_a if (t % period_s) < period_s / 2 else 0.0


def charging_current(profile: CurrentProfile, t: float, state: Array, physical_cutoff: bool) -> float:
    """Evaluate a profile, optionally stopping charge when the cell is full."""
    value = float(profile(t, state))
    if physical_cutoff and state[0] >= 1.0 and value > 0:
        return 0.0
    return value


def rhs(t: float, state: Array, params: BatteryParameters, profile: CurrentProfile,
        physical_cutoff: bool = False) -> Array:
    """Right-hand side of the state equations ``[SOC, v_p]``."""
    current = charging_current(profile, t, state, physical_cutoff)
    _ = t
    return np.array([
        current / (3600.0 * params.capacity_ah),
        -state[1] / params.time_constant_s + current / params.cp_f,
    ])


def terminal_voltage(state: Array, current_a: Array | float, params: BatteryParameters) -> Array:
    """Terminal voltage from SOC, polarization voltage and charging current."""
    state = np.asarray(state)
    return open_circuit_voltage(state[..., 0]) + np.asarray(current_a) * params.r0_ohm + state[..., 1]


def analytical_constant_current(time_s: Array | float, params: BatteryParameters,
                                current_a: float = 1.0) -> Array:
    """Exact state solution for *unconstrained* constant-current charging."""
    t = np.asarray(time_s, dtype=float)
    soc = params.initial_soc + current_a * t / (3600.0 * params.capacity_ah)
    vp = current_a * params.rp_ohm + (
        params.initial_polarization_voltage - current_a * params.rp_ohm
    ) * np.exp(-t / params.time_constant_s)
    return np.column_stack((soc, vp))


def analytical_step_polarization(time_s: Array | float, params: BatteryParameters,
                                 current_a: float = 1.0) -> Array:
    """Exact RC step response of the polarization subsystem."""
    t = np.asarray(time_s, dtype=float)
    return current_a * params.rp_ohm + (
        params.initial_polarization_voltage - current_a * params.rp_ohm
    ) * np.exp(-t / params.time_constant_s)
