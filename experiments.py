"""Run the seven reproducible project experiments and export their data."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from battery_model import BatteryParameters, analytical_constant_current, constant_current, open_circuit_voltage, terminal_voltage
from linear_systems import polarization_poles, polarization_step_response, stability_summary
from numerical_methods import integrate, SimulationResult
import visualization as viz


def _error_row(method: str, step_s: float, result: SimulationResult, params: BatteryParameters) -> dict[str, float | str]:
    reference = analytical_constant_current(result.time_s, params)
    errors = result.states - reference
    vp_error = errors[:, 1]
    return {
        "method": method,
        "step_s": step_s,
        "max_abs_soc_error": float(np.max(np.abs(errors[:, 0]))),
        "rmse_soc": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "max_abs_vp_error_v": float(np.max(np.abs(vp_error))),
        "rmse_vp_v": float(np.sqrt(np.mean(vp_error ** 2))),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _observed_orders(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for method in ("euler", "heun", "rk4"):
        subset = sorted((r for r in rows if r["method"] == method), key=lambda r: float(r["step_s"]), reverse=True)
        for coarse, fine in zip(subset, subset[1:]):
            e1, e2 = float(coarse["max_abs_vp_error_v"]), float(fine["max_abs_vp_error_v"])
            order = np.log(e1 / e2) / np.log(float(coarse["step_s"]) / float(fine["step_s"]))
            output.append({"method": method, "coarse_step_s": coarse["step_s"], "fine_step_s": fine["step_s"], "observed_order": float(order)})
    return output


def run_all_experiments(output_dir: str | Path = "results", params: BatteryParameters | None = None,
                        duration_s: float = 7200.0, main_step_s: float = 10.0) -> dict[str, object]:
    """Execute all prescribed experiments and return actual computed summaries."""
    params = params or BatteryParameters()
    root = Path(output_dir); figures = root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    profile = constant_current(1.0)

    # 1. Baseline unconstrained report experiment.
    baseline = integrate(params, profile, duration_s, main_step_s, "rk4")
    viz.constant_charging_plot(baseline, params, figures / "01_constant_charging.png")

    # 2. Equal-condition method comparison.
    comparison = {m: integrate(params, profile, duration_s, main_step_s, m) for m in ("euler", "heun", "rk4")}
    comparison_errors = [_error_row(m, main_step_s, r, params) for m, r in comparison.items()]
    viz.method_comparison_plot(comparison, params, figures / "02_method_comparison.png")
    viz.error_plot(comparison_errors, figures / "03_method_errors.png")

    # 3. Convergence.
    convergence_rows: list[dict[str, float | str]] = []
    for method in ("euler", "heun", "rk4"):
        for step in (10.0, 5.0, 2.0, 1.0):
            convergence_rows.append(_error_row(method, step, integrate(params, profile, duration_s, step, method), params))
    orders = _observed_orders(convergence_rows)
    viz.convergence_plot(convergence_rows, figures / "04_convergence.png")

    # 4. Current variation.
    current_results = {current: integrate(params, constant_current(current), duration_s, main_step_s, "rk4") for current in (0.5, 1.0, 2.0)}
    viz.current_variation_plot(current_results, params, figures / "05_current_variation.png")

    # 5 and 6. Polarization subsystem and transfer-function analysis.
    transient = integrate(params, profile, 240.0, 1.0, "rk4")
    viz.polarization_plot(transient, params, figures / "06_polarization_transient.png")
    pole = float(polarization_poles(params)[0])
    viz.pole_zero_plot(pole, figures / "07_pole_zero.png")
    step_time = np.linspace(0, 240, 500)
    viz.step_response_plot(step_time, polarization_step_response(step_time, params), figures / "08_step_response.png")

    # 7. Explicit physical cutoff comparison.
    constrained = integrate(params, profile, duration_s, main_step_s, "rk4", physical_cutoff=True)
    viz.cutoff_plot(baseline, constrained, params, figures / "09_physical_cutoff.png")

    simulation_rows = [{
        "time_s": float(t), "soc": float(x[0]), "polarization_voltage_v": float(x[1]),
        "current_a": float(i), "ocv_v": float(open_circuit_voltage(x[0])),
        "terminal_voltage_v": float(terminal_voltage(x, i, params)),
    } for t, x, i in zip(baseline.time_s, baseline.states, baseline.current_a)]
    _write_csv(root / "simulation_data.csv", simulation_rows)
    all_error_rows: list[dict[str, object]] = comparison_errors + convergence_rows
    _write_csv(root / "error_analysis.csv", all_error_rows)
    _write_csv(root / "observed_orders.csv", orders)

    return {
        "params": params, "baseline": baseline, "comparison_errors": comparison_errors,
        "convergence": convergence_rows, "observed_orders": orders,
        "stability": stability_summary(params), "constrained": constrained,
        "figures": sorted(str(p) for p in figures.glob("*.png")),
    }
