# Battery Charging Simulation Using Numerical Methods and Linear Systems

An executable undergraduate Mathematics and Computing project based on a first-order Thevenin equivalent circuit. It compares fixed-step Euler, Heun, and RK4 integration against an analytical reference, performs linear-system analysis, and generates the graphs and tables needed for an academic demonstration.

> **Scope:** The parameter values are illustrative values prescribed by the accompanying report; they are not a manufacturer-validated battery data sheet. Results are model predictions, not experimental measurements.

## Quick start

```bash
python -m pip install -r requirements.txt
python main.py
# Optional interactive demonstration
streamlit run app.py
```

The standard command writes CSV outputs to `results/` and nine PNG figures to `results/figures/`. It prints the actual errors and observed orders calculated during the run.

## Mathematical model

The states are SOC $z$ and polarization voltage $v_p$, with a positive current convention for charging:

$$
\frac{dz}{dt} = \frac{I(t)}{3600Q},
\qquad
\frac{dv_p}{dt} = -\frac{v_p}{R_p C_p} + \frac{I(t)}{C_p}.
$$

$$
V_{OC}(z) = 3.0 + 1.2z,
\qquad
V_t = V_{OC}(z) + I R_0 + v_p.
$$

For constant unconstrained current, the analytical reference is:

$$
z(t) = z_0 + \frac{It}{3600Q},
\qquad
v_p(t) = I R_p + \left(v_p(0) - I R_p\right)e^{-t/(R_pC_p)}.
$$

The state-space form is $x' = Ax + Bu$, where

$$
A =
\begin{bmatrix}
0 & 0 \\
0 & -\frac{1}{R_p C_p}
\end{bmatrix},
\qquad
B =
\begin{bmatrix}
\frac{1}{3600Q} \\
\frac{1}{C_p}
\end{bmatrix}.
$$

The transfer function of the stable RC polarization branch is:

$$
\frac{V_p(s)}{I(s)} = \frac{R_p}{R_p C_p s + 1}.
$$

## Report parameters

| Quantity | Value |
|---|---:|
| Capacity `Q` | 2 Ah |
| Initial SOC | 0.20 |
| Constant charging current | 1 A |
| Initial `v_p` | 0 V |
| `R_0`, `R_p`, `C_p` | 0.05 Ω, 0.02 Ω, 2000 F |
| RC time constant `R_pC_p` | 40 s |
| Duration / main step | 7200 s / 10 s |

The default simulation deliberately preserves the report's *unconstrained* two-hour 1 A experiment: SOC rises from 20% to **120%**. This is mathematically consistent with the supplied equation but not physically admissible. Experiment 7 separately activates a full-SOC cutoff: positive charging current is zero after SOC reaches 100%.

## Project structure

```text
Battery-Charging-Simulation/
├── main.py                 # Command-line runner
├── app.py                  # Optional Streamlit interface
├── battery_model.py        # Parameters, ODEs, terminal voltage, analytical solution
├── numerical_methods.py    # Euler, Heun, RK4, generic fixed-step integration
├── linear_systems.py       # Matrices, poles, stability and step response
├── visualization.py        # Matplotlib plots
├── experiments.py          # Seven reproducible experiments and CSV export
├── tests/test_simulation.py
├── results/
│   ├── figures/
│   ├── simulation_data.csv
│   ├── error_analysis.csv
│   └── observed_orders.csv
└── requirements.txt
```

## Simulation flow

```text
Parameters + current profile
            │
            ▼
  Select Euler / Heun / RK4 and h
            │
            ▼
Integrate [SOC, v_p] state ODEs ──► optional SOC=100% cutoff
            │
            ▼
Compute OCV and terminal voltage
            │
            ├──► analytical reference → errors and convergence orders
            └──► CSV tables and publication-ready figures
```

## Experiments and interpretation

1. **Constant-current charging:** plots SOC, `v_p`, OCV, and terminal voltage for the original 1 A report case.
2. **Method comparison:** uses the same initial state and 10 s grid for Euler, Heun, and RK4. Errors are calculated from the exact constant-current solution.
3. **Convergence:** repeats the calculation at `h = 10, 5, 2, 1` s and estimates `p = log(E_h/E_{h/2}) / log(h/(h/2))` only for adjacent available grids. Euler, Heun, and RK4 have theoretical global orders 1, 2, and 4. RK4 may reach floating-point precision at very small steps, so its observed order can become less informative.
4. **Current variation:** compares 0.5, 1, and 2 A. Higher current raises SOC faster and produces a higher steady `v_p = IR_p` and IR0 voltage contribution.
5. **Polarization transient:** validates the exponential approach of `v_p` to `IR_p`; after one time constant (40 s), the transient has completed about 63.2% of its change.
6. **Linear system:** plots the negative RC pole and its step response. The **polarization subsystem** is asymptotically stable because its pole is `-1/40 s⁻¹`. The complete state model has a zero SOC eigenvalue because charge accumulates; it is not asymptotically stable under continuing input. This distinction is important in a viva.
7. **Physical cutoff:** compares the report case with a SOC-clamped model in which charging stops at full SOC.

## Key files generated

- `simulation_data.csv`: baseline RK4, unconstrained, 1 A simulation (`time_s`, states, current, OCV, terminal voltage).
- `error_analysis.csv`: method-comparison and convergence errors. SOC error is essentially rounding-level for this constant-current equation; `v_p` is the meaningful integrator accuracy measure.
- `observed_orders.csv`: order estimates using actual computed errors.
- `results/figures/01_...png` through `09_...png`: all requested plots.

## Assumptions and limitations

- Parameters are temperature-invariant; there is no thermal, ageing, hysteresis, or self-discharge model.
- The OCV--SOC relationship is affine, while real cells have nonlinear chemistry-dependent curves.
- The Thevenin circuit has only one RC branch.
- The default is a constant-current mathematical experiment. The physical cutoff is a simple event rule, not CC-CV charging.
- A real BMS also applies voltage, temperature, and current safety limits. This code is educational and must not control actual charging hardware.

## Advanced extension

`app.py` offers parameter controls, solver selection, a physical-cutoff toggle, plots, state-space summary, and a downloadable CSV. A defensible next model extension is CC-CV charging: use constant current until a voltage limit, then solve for a decreasing current that holds terminal voltage. It should be documented as an extension rather than substituted silently for the report's constant-current model.

## Testing and troubleshooting

Run tests with:

```bash
python -m unittest discover -s tests -v
```

| Symptom | Resolution |
|---|---|
| `ModuleNotFoundError` | Run `python -m pip install -r requirements.txt` in the same environment. |
| No figures appear | Run `python main.py`; files are intentionally saved, not shown interactively. |
| Streamlit unavailable | Install dependencies, then run `streamlit run app.py`. |
| SOC is 120% | This is expected in the report-reproduction mode. Select physical cutoff in the app or pass `physical_cutoff=True` when using `integrate`. |
| Different error values | Confirm the same parameters, duration, method, and time step; errors are generated fresh on every run. |

## Viva checklist

- Explain that Euler evaluates one slope, Heun averages two slopes, and RK4 takes a weighted average of four slopes.
- Explain why the exponential RC state is sensitive to the step size while SOC is exactly linear for constant current.
- Distinguish stability of the RC branch from the charge-accumulating SOC integrator.
- State that the analytical result validates the numerical algorithm only for constant current without a cutoff.
# Battery-Charging-Simulator
