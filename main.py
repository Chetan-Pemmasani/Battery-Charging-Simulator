"""Command-line entry point for the Battery Charging Simulation project."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments import run_all_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Battery charging numerical-method experiments")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV files and figures")
    parser.add_argument("--duration", type=float, default=7200.0, help="Simulation duration in seconds")
    parser.add_argument("--step", type=float, default=10.0, help="Main comparison step in seconds")
    args = parser.parse_args()
    summary = run_all_experiments(Path(args.output_dir), duration_s=args.duration, main_step_s=args.step)
    baseline = summary["baseline"]
    print("Battery Charging Simulation completed successfully.")
    print(f"Unconstrained final SOC: {baseline.soc[-1] * 100:.2f}% (report case)")
    print(f"Physical-cutoff final SOC: {summary['constrained'].soc[-1] * 100:.2f}%")
    print("Method errors against the analytical reference:")
    for row in summary["comparison_errors"]:
        print(f"  {row['method'].upper():5s} max |v_p error| = {row['max_abs_vp_error_v']:.6e} V; RMSE = {row['rmse_vp_v']:.6e} V")
    print("Observed convergence orders (using max |v_p error|):")
    for row in summary["observed_orders"]:
        print(f"  {row['method'].upper():5s} h={row['coarse_step_s']:g}→{row['fine_step_s']:g}: {row['observed_order']:.3f}")
    print(f"Created {len(summary['figures'])} figures in {args.output_dir}/figures")


if __name__ == "__main__":
    main()
