"""State-space and transfer-function analysis of the polarization subsystem."""
from __future__ import annotations

import numpy as np
from battery_model import BatteryParameters


def state_space_matrices(params: BatteryParameters) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([[0.0, 0.0], [0.0, -1.0 / params.time_constant_s]])
    b = np.array([[1.0 / (3600.0 * params.capacity_ah)], [1.0 / params.cp_f]])
    return a, b


def eigenvalues(params: BatteryParameters) -> np.ndarray:
    return np.linalg.eigvals(state_space_matrices(params)[0])


def polarization_poles(params: BatteryParameters) -> np.ndarray:
    return np.array([-1.0 / params.time_constant_s])


def transfer_function_coefficients(params: BatteryParameters) -> tuple[np.ndarray, np.ndarray]:
    """Numerator and denominator of Vp(s)/I(s), descending powers of s."""
    return np.array([params.rp_ohm]), np.array([params.time_constant_s, 1.0])


def is_asymptotically_stable(params: BatteryParameters) -> bool:
    """The complete model has an SOC integrator; only its RC subsystem is asymptotically stable."""
    return bool(np.all(np.real(polarization_poles(params)) < 0))


def stability_summary(params: BatteryParameters) -> dict[str, object]:
    eig = eigenvalues(params)
    return {
        "state_eigenvalues": eig,
        "polarization_poles": polarization_poles(params),
        "polarization_asymptotically_stable": is_asymptotically_stable(params),
        "full_model_note": "The SOC eigenvalue is zero (charge accumulation), so the full two-state charging model is marginal, not asymptotically stable.",
    }


def polarization_step_response(time_s: np.ndarray, params: BatteryParameters,
                               current_step_a: float = 1.0) -> np.ndarray:
    return current_step_a * params.rp_ohm * (1.0 - np.exp(-np.asarray(time_s) / params.time_constant_s))
