"""Fixed-step Euler, Heun and classical fourth-order Runge--Kutta solvers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np

from battery_model import BatteryParameters, CurrentProfile, charging_current, rhs

Method = Literal["euler", "heun", "rk4"]


@dataclass
class SimulationResult:
    time_s: np.ndarray
    states: np.ndarray
    current_a: np.ndarray
    method: str
    physical_cutoff: bool

    @property
    def soc(self) -> np.ndarray:
        return self.states[:, 0]

    @property
    def polarization_voltage(self) -> np.ndarray:
        return self.states[:, 1]


def _step(fun: Callable[[float, np.ndarray], np.ndarray], t: float, x: np.ndarray,
          h: float, method: Method) -> np.ndarray:
    if method == "euler":
        return x + h * fun(t, x)
    if method == "heun":
        k1 = fun(t, x)
        k2 = fun(t + h, x + h * k1)
        return x + h * (k1 + k2) / 2.0
    if method == "rk4":
        k1 = fun(t, x)
        k2 = fun(t + h / 2.0, x + h * k1 / 2.0)
        k3 = fun(t + h / 2.0, x + h * k2 / 2.0)
        k4 = fun(t + h, x + h * k3)
        return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    raise ValueError("method must be 'euler', 'heun', or 'rk4'.")


def integrate(params: BatteryParameters, profile: CurrentProfile, duration_s: float = 7200.0,
              step_s: float = 10.0, method: Method = "rk4",
              physical_cutoff: bool = False) -> SimulationResult:
    """Simulate the model on a grid including a final partial step if needed.

    With ``physical_cutoff=False`` this reproduces the report's unconstrained
    two-hour experiment (SOC reaches 1.20).  The cutoff mode clips SOC only at
    1 and makes a positive current zero once full.
    """
    if duration_s < 0 or step_s <= 0:
        raise ValueError("duration_s must be non-negative and step_s positive.")
    if method not in {"euler", "heun", "rk4"}:
        raise ValueError("Unknown integration method.")
    full_steps = int(np.floor(duration_s / step_s + 1e-12))
    time = np.arange(full_steps + 1, dtype=float) * step_s
    if duration_s - time[-1] > 1e-10:
        time = np.append(time, duration_s)
    states = np.empty((len(time), 2), dtype=float)
    states[0] = params.initial_state
    fun = lambda t, x: rhs(t, x, params, profile, physical_cutoff)
    for index in range(len(time) - 1):
        next_state = _step(fun, time[index], states[index], time[index + 1] - time[index], method)
        if physical_cutoff:
            next_state[0] = min(next_state[0], 1.0)
        states[index + 1] = next_state
    current = np.array([charging_current(profile, t, x, physical_cutoff) for t, x in zip(time, states)])
    return SimulationResult(time, states, current, method, physical_cutoff)


def forward_euler(*args, **kwargs) -> SimulationResult:
    kwargs["method"] = "euler"
    return integrate(*args, **kwargs)


def heun(*args, **kwargs) -> SimulationResult:
    kwargs["method"] = "heun"
    return integrate(*args, **kwargs)


def rk4(*args, **kwargs) -> SimulationResult:
    kwargs["method"] = "rk4"
    return integrate(*args, **kwargs)
