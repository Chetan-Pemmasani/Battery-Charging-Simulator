"""State-space and transfer-function analysis of the polarization subsystem."""
from __future__ import annotations

import numpy as np
from battery_model import BatteryParameters


def state_space_matrices(params: BatteryParameters) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([[0.0, 0.0], [0.0, -1.0 / params.time_constant_s]])
    b = np.array([[1.0 / (3600.0 * params.capacity_ah)], [1.0 / params.cp_f]])
    return a, b


def output_matrices(params: BatteryParameters) -> tuple[np.ndarray, np.ndarray]:
    """Output matrices for the voltage perturbation ``Vt - 3 = Cx + DI``."""
    return np.array([[1.2, 1.0]]), np.array([[params.r0_ohm]])


def controllability_matrix(params: BatteryParameters) -> np.ndarray:
    a, b = state_space_matrices(params)
    return np.column_stack((b, a @ b))


def observability_matrix(params: BatteryParameters) -> np.ndarray:
    a, _ = state_space_matrices(params)
    c, _ = output_matrices(params)
    return np.vstack((c, c @ a))


def controllability_rank(params: BatteryParameters) -> int:
    return int(np.linalg.matrix_rank(controllability_matrix(params)))


def observability_rank(params: BatteryParameters) -> int:
    return int(np.linalg.matrix_rank(observability_matrix(params)))


def eigenvalues(params: BatteryParameters) -> np.ndarray:
    return np.linalg.eigvals(state_space_matrices(params)[0])


def polarization_poles(params: BatteryParameters) -> np.ndarray:
    return np.array([-1.0 / params.time_constant_s])


def transfer_function_coefficients(params: BatteryParameters) -> tuple[np.ndarray, np.ndarray]:
    """Numerator and denominator of Vp(s)/I(s), descending powers of s."""
    return np.array([params.rp_ohm]), np.array([params.time_constant_s, 1.0])


def terminal_transfer_function_coefficients(params: BatteryParameters) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients for ``G(s)=Vt(s)/I(s)`` including SOC, RC and R0 terms."""
    tau = params.time_constant_s
    soc_gain = 1.2 / (3600.0 * params.capacity_ah)
    numerator = np.array([
        params.r0_ohm * tau,
        soc_gain * tau + params.rp_ohm + params.r0_ohm,
        soc_gain,
    ])
    denominator = np.array([tau, 1.0, 0.0])
    return numerator, denominator


def terminal_zeros(params: BatteryParameters) -> np.ndarray:
    numerator, _ = terminal_transfer_function_coefficients(params)
    return np.roots(numerator)


def terminal_frequency_response(omega_rad_s: np.ndarray, params: BatteryParameters) -> np.ndarray:
    """Frequency response of full terminal-voltage transfer function."""
    omega = np.asarray(omega_rad_s, dtype=float)
    s = 1j * omega
    return (
        1.2 / (3600.0 * params.capacity_ah * s)
        + params.rp_ohm / (params.time_constant_s * s + 1.0)
        + params.r0_ohm
    )


def is_asymptotically_stable(params: BatteryParameters) -> bool:
    """The complete model has an SOC integrator; only its RC subsystem is asymptotically stable."""
    return bool(np.all(np.real(polarization_poles(params)) < 0))


def stability_summary(params: BatteryParameters) -> dict[str, object]:
    eig = eigenvalues(params)
    return {
        "state_eigenvalues": eig,
        "polarization_poles": polarization_poles(params),
        "polarization_asymptotically_stable": is_asymptotically_stable(params),
        "controllability_rank": controllability_rank(params),
        "observability_rank": observability_rank(params),
        "full_model_note": "The SOC eigenvalue is zero (charge accumulation), so the full two-state charging model is marginal, not asymptotically stable.",
    }


def polarization_step_response(time_s: np.ndarray, params: BatteryParameters,
                               current_step_a: float = 1.0) -> np.ndarray:
    return current_step_a * params.rp_ohm * (1.0 - np.exp(-np.asarray(time_s) / params.time_constant_s))
