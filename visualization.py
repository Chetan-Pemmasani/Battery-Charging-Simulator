"""Professional, reproducible Matplotlib figures for the experiments."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from battery_model import BatteryParameters, analytical_constant_current, open_circuit_voltage, terminal_voltage
from numerical_methods import SimulationResult

plt.style.use("seaborn-v0_8-whitegrid")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def constant_charging_plot(result: SimulationResult, params: BatteryParameters, path: Path) -> Path:
    hours = result.time_s / 3600.0
    ocv = open_circuit_voltage(result.soc)
    vt = terminal_voltage(result.states, result.current_a, params)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes[0, 0].plot(hours, result.soc * 100, color="#1677b8"); axes[0, 0].set(ylabel="SOC (%)", title="State of charge")
    axes[0, 1].plot(hours, result.polarization_voltage, color="#d95f02"); axes[0, 1].set(ylabel="Voltage (V)", title="Polarization voltage")
    axes[1, 0].plot(hours, ocv, color="#2a9d8f"); axes[1, 0].set(xlabel="Time (h)", ylabel="Voltage (V)", title="Open-circuit voltage")
    axes[1, 1].plot(hours, vt, color="#7b2cbf"); axes[1, 1].set(xlabel="Time (h)", ylabel="Voltage (V)", title="Terminal voltage")
    fig.suptitle("Experiment 1: Unconstrained 1 A constant-current charging", y=1.02)
    return _save(fig, path)


def method_comparison_plot(results: dict[str, SimulationResult], params: BatteryParameters, path: Path) -> Path:
    ref = analytical_constant_current(next(iter(results.values())).time_s, params)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, result in results.items():
        axes[0].plot(result.time_s / 60, result.soc * 100, label=method.upper())
        axes[1].plot(result.time_s / 60, result.polarization_voltage, label=method.upper())
    axes[0].plot(next(iter(results.values())).time_s / 60, ref[:, 0] * 100, "k--", label="Analytical")
    axes[1].plot(next(iter(results.values())).time_s / 60, ref[:, 1], "k--", label="Analytical")
    axes[0].set(xlabel="Time (min)", ylabel="SOC (%)", title="SOC method comparison")
    axes[1].set(xlabel="Time (min)", ylabel="Polarization voltage v_p (V)", title="RC-voltage comparison")
    for axis in axes: axis.legend()
    return _save(fig, path)


def error_plot(rows: list[dict[str, float | str]], path: Path) -> Path:
    methods = [str(row["method"]).upper() for row in rows]
    maxima = [float(row["max_abs_vp_error_v"]) for row in rows]
    rmse = [float(row["rmse_vp_v"]) for row in rows]
    x = np.arange(len(methods)); width = .36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, maxima, width, label="Maximum absolute error")
    ax.bar(x + width / 2, rmse, width, label="RMSE")
    ax.set(xticks=x, xticklabels=methods, ylabel="Polarization-voltage error (V)", title="Experiment 2: Error against analytical solution")
    ax.legend()
    return _save(fig, path)


def convergence_plot(rows: list[dict[str, float | str]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("euler", "heun", "rk4"):
        subset = [r for r in rows if r["method"] == method]
        h = [float(r["step_s"]) for r in subset]
        error = [float(r["max_abs_vp_error_v"]) for r in subset]
        ax.loglog(h, error, "o-", label=method.upper())
    ax.set(xlabel="Step size h (s)", ylabel="Maximum |v_p error| (V)", title="Experiment 3: Time-step convergence")
    ax.legend()
    return _save(fig, path)


def current_variation_plot(results: dict[float, SimulationResult], params: BatteryParameters, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for current, result in results.items():
        label = f"{current:g} A"
        time = result.time_s / 3600
        axes[0].plot(time, result.soc * 100, label=label)
        axes[1].plot(time, result.polarization_voltage, label=label)
        axes[2].plot(time, terminal_voltage(result.states, result.current_a, params), label=label)
    axes[0].set(xlabel="Time (h)", ylabel="SOC (%)", title="SOC")
    axes[1].set(xlabel="Time (h)", ylabel="v_p (V)", title="Polarization voltage")
    axes[2].set(xlabel="Time (h)", ylabel="V_t (V)", title="Terminal voltage")
    for ax in axes: ax.legend()
    fig.suptitle("Experiment 4: Current variation", y=1.03)
    return _save(fig, path)


def polarization_plot(result: SimulationResult, params: BatteryParameters, path: Path) -> Path:
    ref = analytical_constant_current(result.time_s, params)[:, 1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(result.time_s, result.polarization_voltage, label="RK4 numerical")
    ax.plot(result.time_s, ref, "--", label="Analytical exponential")
    ax.axhline(params.rp_ohm, color="black", lw=1, ls=":", label="Steady state: I R_p")
    ax.axvline(params.time_constant_s, color="gray", lw=1, ls=":", label="Time constant τ")
    ax.set(xlabel="Time (s)", ylabel="Polarization voltage v_p (V)", title="Experiment 5: RC polarization transient")
    ax.legend()
    return _save(fig, path)


def pole_zero_plot(pole: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0, color="black", lw=.8); ax.axvline(0, color="black", lw=.8)
    ax.plot([pole], [0], "x", ms=11, mew=3, label="Polarization pole")
    ax.set(xlabel="Real axis (1/s)", ylabel="Imaginary axis (1/s)", title="Experiment 6: Pole-zero diagram")
    ax.legend(); ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, path)


def step_response_plot(time_s: np.ndarray, response: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time_s, response)
    ax.set(xlabel="Time (s)", ylabel="v_p (V)", title="Transfer-function unit-step response")
    return _save(fig, path)


def cutoff_plot(unconstrained: SimulationResult, constrained: SimulationResult,
                params: BatteryParameters, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for result, label, style in ((unconstrained, "Unconstrained report case", "-"), (constrained, "Physical cutoff", "--")):
        axes[0].plot(result.time_s / 3600, result.soc * 100, style, label=label)
        axes[1].plot(result.time_s / 3600, result.current_a, style, label=label)
    axes[0].axhline(100, color="gray", lw=1, ls=":")
    axes[0].set(xlabel="Time (h)", ylabel="SOC (%)", title="SOC: report vs physical mode")
    axes[1].set(xlabel="Time (h)", ylabel="Charging current (A)", title="Current cutoff at full SOC")
    for ax in axes: ax.legend()
    return _save(fig, path)
