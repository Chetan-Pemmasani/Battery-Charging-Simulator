"""Interactive battery-charging simulation for an academic demonstration."""
from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from battery_model import (BatteryParameters, analytical_constant_current, constant_current,
    open_circuit_voltage, pulse_current, ramp_current, step_current, terminal_voltage)
from linear_systems import (controllability_rank, eigenvalues, observability_rank,
    output_matrices, polarization_poles, state_space_matrices, terminal_frequency_response,
    terminal_transfer_function_coefficients, terminal_zeros)
from numerical_methods import integrate

st.set_page_config(page_title="Battery Charging Simulation", layout="wide")

PRESETS = {
    "Report default": {"capacity": 2.0, "initial_soc": .20, "current": 1., "r0": .05, "rp": .02, "cp": 2000., "duration": 7200., "step": 10., "profile": "Constant current", "cutoff": True},
    "Fast charge (2 A)": {"capacity": 2.0, "initial_soc": .20, "current": 2., "r0": .05, "rp": .02, "cp": 2000., "duration": 3600., "step": 5., "profile": "Constant current", "cutoff": True},
    "Stiff case (tau = 1 s)": {"capacity": 2.0, "initial_soc": .20, "current": 1., "r0": .05, "rp": .02, "cp": 50., "duration": 60., "step": 3., "profile": "Constant current", "cutoff": False},
    "Pulse charging": {"capacity": 2.0, "initial_soc": .20, "current": 1., "r0": .05, "rp": .02, "cp": 2000., "duration": 3600., "step": 5., "profile": "Pulse train", "cutoff": True},
}


def apply_preset() -> None:
    for key, value in PRESETS[st.session_state.preset].items():
        st.session_state[key] = value


def profile_for(name: str, current_a: float):
    if name == "Constant current": return constant_current(current_a)
    if name == "Step current": return step_current(0., current_a, 600.)
    if name == "Pulse train": return pulse_current(current_a, 1200.)
    return ramp_current(0., current_a, 1800.)


def make_frame(result, params: BatteryParameters) -> pd.DataFrame:
    frame = pd.DataFrame({"time_s": result.time_s, "soc": result.soc,
        "polarization_voltage_v": result.polarization_voltage, "current_a": result.current_a})
    frame["ocv_v"] = open_circuit_voltage(frame["soc"].to_numpy())
    frame["terminal_voltage_v"] = terminal_voltage(result.states, result.current_a, params)
    return frame


def voltage_error(result, params: BatteryParameters, current_a: float) -> np.ndarray:
    exact = analytical_constant_current(result.time_s, params, current_a)
    return terminal_voltage(result.states, result.current_a, params) - terminal_voltage(exact, current_a, params)


@st.cache_data(show_spinner=False)
def convergence_sweep(capacity: float, initial_soc: float, r0: float, rp: float, cp: float,
                      current: float, duration: float) -> pd.DataFrame:
    params = BatteryParameters(capacity, initial_soc, 0., r0, rp, cp)
    rows = []
    for method in ("euler", "heun", "rk4"):
        for h in (20., 10., 5., 2., 1., .5):
            result = integrate(params, constant_current(current), duration, h, method, False)
            exact = analytical_constant_current(result.time_s, params, current)
            error = max(float(np.max(np.abs(result.polarization_voltage - exact[:, 1]))), 1e-16)
            rows.append({"Method": method.upper(), "Step size (s)": h, "Maximum error (V)": error})
    return pd.DataFrame(rows)


st.title("Battery Charging Simulation")
st.caption("First-order Thevenin model. Positive current means charging; the default applies a physical full-SOC cutoff.")

with st.sidebar:
    st.header("Simulation controls")
    st.selectbox("Preset", list(PRESETS), key="preset", on_change=apply_preset, help="Loads a teaching configuration.")
    with st.expander("Battery parameters", expanded=True):
        capacity = st.number_input("Capacity (Ah)", min_value=.1, value=2., step=.1, key="capacity", help="Nominal capacity Q.")
        initial_soc = st.slider("Initial SOC", 0., 1., .20, .01, key="initial_soc", help="Initial state of charge.")
        r0 = st.number_input("R0 (ohm)", min_value=0., value=.05, step=.01, key="r0", help="Ohmic resistance.")
        rp = st.number_input("Rp (ohm)", min_value=.001, value=.02, step=.005, key="rp", help="Polarization resistance.")
        cp = st.number_input("Cp (F)", min_value=1., value=2000., step=100., key="cp", help="Polarization capacitance.")
    with st.expander("Current profile", expanded=True):
        profile_name = st.selectbox("Profile", ["Constant current", "Step current", "Pulse train", "Ramp current"], key="profile", help="Time-varying profiles show why numerical solvers are useful.")
        current = st.number_input("Peak charging current (A)", min_value=0., value=1., step=.1, key="current", help="Constant value or peak value.")
        cutoff = st.checkbox("Apply physical full-SOC cutoff", value=True, key="cutoff", help="Stops positive current once SOC reaches 100%.")
    with st.expander("Solver settings", expanded=True):
        duration = st.number_input("Duration (s)", min_value=1., value=7200., step=60., key="duration", help="Total simulated time.")
        step = st.selectbox("Step size (s)", [20., 10., 5., 3., 2., 1., .5], key="step", help="Fixed integration step h.")
        method = st.selectbox("Method", ["rk4", "heun", "euler"], key="method", help="Heun is the explicit-trapezoidal RK2 method.")
    run = st.button("Run simulation", type="primary", width="stretch")

if not np.isclose(duration % step, 0., atol=1e-9):
    st.warning("Duration is not divisible by the step size; one shorter final integration step will be used.")

live_params = BatteryParameters(capacity, initial_soc, 0., r0, rp, cp)
tau, euler_limit = live_params.time_constant_s, 2 * live_params.time_constant_s
one, two, three, four = st.columns(4)
one.metric("RC time constant", f"{tau:.3g} s", border=True)
two.metric("Euler stability limit", f"h < {euler_limit:.3g} s", border=True)
three.metric("Selected h", f"{step:g} s", border=True)
four.metric("Euler RC stability", "Stable" if step < euler_limit else "Unstable", border=True)

if run:
    with st.spinner("Integrating the battery model..."):
        result = integrate(live_params, profile_for(profile_name, current), duration, step, method, cutoff)
        st.session_state.simulation = {"params": live_params, "result": result,
            "frame": make_frame(result, live_params), "profile_name": profile_name,
            "current": current, "cutoff": cutoff}
    st.toast("Simulation complete")

simulation = st.session_state.get("simulation")
if simulation is None:
    st.info("Choose a configuration and select Run simulation.")
    st.stop()

params, result, frame = simulation["params"], simulation["result"], simulation["frame"]
profile_name, current, cutoff = simulation["profile_name"], simulation["current"], simulation["cutoff"]
final_soc = float(result.soc[-1] * 100)
if np.any(result.soc > 1 + 1e-10):
    st.warning("SOC exceeds 100%. This is the optional unconstrained overcharge demonstration, not a physical charging mode.")
elif cutoff:
    st.success("Physical cutoff is active: positive charge current stops at full SOC.")

tabs = st.tabs(["SOC", "Voltage", "Error vs exact", "Compare methods", "Convergence", "Stability", "Linear systems", "Data preview"])
soc_tab, voltage_tab, error_tab, compare_tab, converge_tab, stability_tab, linear_tab, data_tab = tabs

with soc_tab:
    st.subheader("State-of-charge trajectory")
    data = pd.DataFrame({"Time (s)": frame.time_s, "SOC (%)": frame.soc * 100})
    top = max(110., final_soc * 1.05)
    x, y = alt.X("Time (s):Q", title="Time (s)"), alt.Y("SOC (%):Q", title="State of charge (%)", scale=alt.Scale(domain=[0, top]))
    line = alt.Chart(data).mark_line().encode(x=x, y=y, tooltip=[alt.Tooltip("Time (s):Q", format=",.0f"), alt.Tooltip("SOC (%):Q", format=".2f")])
    full = alt.Chart(pd.DataFrame({"SOC (%)": [100.]})).mark_rule(color="#ff4b4b", strokeDash=[6, 4]).encode(y=y)
    final = alt.Chart(pd.DataFrame({"Time (s)": [float(result.time_s[-1])], "SOC (%)": [final_soc]})).mark_point(size=100, filled=True).encode(x=x, y=y)
    st.altair_chart((line + full + final).properties(height=380), alt="State of charge over time with final point and full-SOC reference.")
    current_data = pd.DataFrame({"Time (s)": frame.time_s, "Current (A)": frame.current_a})
    st.altair_chart(alt.Chart(current_data).mark_line().encode(x=alt.X("Time (s):Q", title="Time (s)"), y=alt.Y("Current (A):Q", title="Charging current (A)"), tooltip=["Time (s):Q", "Current (A):Q"]).properties(height=180, title="Applied current profile"), alt="Applied charging current over time.")

with voltage_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Terminal and open-circuit voltage")
        voltages = frame.rename(columns={"time_s": "Time (s)", "terminal_voltage_v": "Terminal voltage (V)", "ocv_v": "Open-circuit voltage (V)"})
        chart = alt.Chart(voltages).transform_fold(["Terminal voltage (V)", "Open-circuit voltage (V)"], as_=["Quantity", "Voltage (V)"]).mark_line().encode(x=alt.X("Time (s):Q", title="Time (s)"), y=alt.Y("Voltage (V):Q", title="Voltage (V)"), color="Quantity:N", tooltip=["Time (s):Q", "Quantity:N", alt.Tooltip("Voltage (V):Q", format=".4f")]).properties(height=360)
        st.altair_chart(chart, alt="Terminal and open-circuit voltage over time.")
    with right:
        st.subheader("Polarization-voltage transient")
        vp = frame.rename(columns={"time_s": "Time (s)", "polarization_voltage_v": "Polarization voltage (V)"})
        chart = alt.Chart(vp).mark_line().encode(x=alt.X("Time (s):Q", title="Time (s)"), y=alt.Y("Polarization voltage (V):Q", title="Polarization voltage (V)"), tooltip=["Time (s):Q", alt.Tooltip("Polarization voltage (V):Q", format=".6f")]).properties(height=360)
        st.altair_chart(chart, alt="Polarization voltage on its own readable scale.")

exact_available = profile_name == "Constant current" and not cutoff
with error_tab:
    st.subheader("Numerical terminal-voltage error")
    if not exact_available:
        st.info("The analytical reference is available for unconstrained constant current. Select Constant current and turn off the physical cutoff to compare against Eq. (5).")
    else:
        error = voltage_error(result, params, current)
        data = pd.DataFrame({"Time (s)": result.time_s, "Absolute voltage error (V)": np.maximum(np.abs(error), 1e-16)})
        chart = alt.Chart(data).mark_line().encode(x=alt.X("Time (s):Q", title="Time (s)"), y=alt.Y("Absolute voltage error (V):Q", scale=alt.Scale(type="log")), tooltip=["Time (s):Q", alt.Tooltip("Absolute voltage error (V):Q", format=".3e")]).properties(height=360)
        st.altair_chart(chart, alt="Log-scale terminal-voltage error against the exact solution.")
        st.caption("SOC is integrated exactly for constant current, so this voltage error arises entirely from the RC polarization state.")

with compare_tab:
    st.subheader("Euler, Heun, and RK4 under identical conditions")
    if not exact_available:
        st.info("Method comparison against the analytical solution is available for the unconstrained constant-current case.")
    else:
        rows, vp_lines, err_lines = [], [], []
        for candidate in ("euler", "heun", "rk4"):
            start = perf_counter(); candidate_result = integrate(params, constant_current(current), duration, step, candidate, False); elapsed = perf_counter() - start
            error = voltage_error(candidate_result, params, current)
            rows.append({"Method": candidate.upper(), "Max |Vt error| (V)": float(np.max(np.abs(error))), "RMSE (V)": float(np.sqrt(np.mean(error**2))), "Runtime (ms)": elapsed * 1000, "RHS evaluations / step": {"euler": 1, "heun": 2, "rk4": 4}[candidate]})
            vp_lines.append(pd.DataFrame({"Time (s)": candidate_result.time_s, "Polarization voltage (V)": candidate_result.polarization_voltage, "Method": candidate.upper()}))
            err_lines.append(pd.DataFrame({"Time (s)": candidate_result.time_s, "Absolute error (V)": np.maximum(np.abs(error), 1e-16), "Method": candidate.upper()}))
        a, b = st.columns(2)
        with a: st.altair_chart(alt.Chart(pd.concat(vp_lines)).mark_line().encode(x="Time (s):Q", y="Polarization voltage (V):Q", color="Method:N", tooltip=["Time (s):Q", "Method:N", "Polarization voltage (V):Q"]).properties(height=320), alt="Polarization voltage for Euler, Heun, and RK4.")
        with b: st.altair_chart(alt.Chart(pd.concat(err_lines)).mark_line().encode(x="Time (s):Q", y=alt.Y("Absolute error (V):Q", scale=alt.Scale(type="log")), color="Method:N", tooltip=["Time (s):Q", "Method:N", "Absolute error (V):Q"]).properties(height=320), alt="Log-scale error comparison for the three numerical methods.")
        st.dataframe(pd.DataFrame(rows).round(9), hide_index=True, alt="Error, runtime, and evaluation counts for numerical methods.")

with converge_tab:
    st.subheader("Observed convergence order")
    if not exact_available:
        st.info("Convergence is calculated against the constant-current analytical solution. Use the unconstrained constant-current mode to enable it.")
    else:
        with st.spinner("Computing the six-step convergence sweep..."):
            sweep = convergence_sweep(params.capacity_ah, params.initial_soc, params.r0_ohm, params.rp_ohm, params.cp_f, current, duration)
        slopes = {name: np.polyfit(np.log(group["Step size (s)"]), np.log(group["Maximum error (V)"]), 1)[0]
                  for name, group in sweep.groupby("Method")}
        st.caption("Fitted log-log slopes — " + ", ".join(f"{name}: {value:.3f}" for name, value in slopes.items()) + ". Expected: Euler 1, Heun 2, RK4 4.")
        chart = alt.Chart(sweep).mark_line(point=True).encode(x=alt.X("Step size (s):Q", scale=alt.Scale(type="log")), y=alt.Y("Maximum error (V):Q", scale=alt.Scale(type="log")), color="Method:N", tooltip=["Method:N", "Step size (s):Q", "Maximum error (V):Q"]).properties(height=380)
        st.altair_chart(chart, alt="Log-log convergence of numerical error against step size.")

with stability_tab:
    st.subheader("Forward-Euler stability of the RC mode")
    lam, factor = -1 / params.time_constant_s, 1 - step / params.time_constant_s
    st.latex(r"\text{Euler is stable for the RC mode when } |1+h\lambda|<1,\quad \lambda=-1/\tau,\quad h<2\tau.")
    st.write(f"For this configuration: $h\\lambda = {step * lam:.4f}$ and Euler amplification factor is ${factor:.4f}$.")
    if abs(factor) < 1: st.success("Euler is stable for the RC polarization mode at the selected step size.")
    else: st.error("Euler is unstable for the RC polarization mode. Try the stiff preset to demonstrate divergence.")
    length = max(10 * params.time_constant_s, 10 * step)
    euler_case, rk4_case = integrate(params, constant_current(current or 1), length, step, "euler", False), integrate(params, constant_current(current or 1), length, step, "rk4", False)
    data = pd.concat([pd.DataFrame({"Time (s)": euler_case.time_s, "Polarization voltage (V)": euler_case.polarization_voltage, "Method": "Euler"}), pd.DataFrame({"Time (s)": rk4_case.time_s, "Polarization voltage (V)": rk4_case.polarization_voltage, "Method": "RK4"})])
    st.altair_chart(alt.Chart(data).mark_line().encode(x="Time (s):Q", y="Polarization voltage (V):Q", color="Method:N", tooltip=["Time (s):Q", "Method:N", "Polarization voltage (V):Q"]).properties(height=350), alt="Euler and RK4 polarization responses for the selected stability case.")

with linear_tab:
    st.subheader("State-space and transfer-function analysis")
    a, b = state_space_matrices(params); c, d = output_matrices(params); eig = eigenvalues(params)
    num, den = terminal_transfer_function_coefficients(params)
    st.latex(rf"A=\begin{{bmatrix}}0&0\\0&{a[1,1]:.6f}\end{{bmatrix}},\quad B=\begin{{bmatrix}}{b[0,0]:.8f}\\{b[1,0]:.8f}\end{{bmatrix}},\quad C=\begin{{bmatrix}}1.2&1\end{{bmatrix}},\quad D={d[0,0]:.3f}")
    st.latex(r"G(s)=\frac{V_t(s)}{I(s)}=\frac{1.2}{3600Qs}+\frac{R_p}{R_pC_ps+1}+R_0")
    p, q, r, s = st.columns(4)
    p.metric("Eigenvalue 1", f"{eig[0]:.6f}", border=True); q.metric("Eigenvalue 2", f"{eig[1]:.6f}", border=True)
    r.metric("Controllability rank", f"{controllability_rank(params)} / 2", border=True); s.metric("Observability rank", f"{observability_rank(params)} / 2", border=True)
    left, right = st.columns(2)
    with left:
        omega = np.logspace(-4, 1, 300); response = terminal_frequency_response(omega, params)
        bode = pd.DataFrame({"Frequency (rad/s)": omega, "Magnitude (dB)": 20 * np.log10(np.maximum(np.abs(response), 1e-16))})
        st.altair_chart(alt.Chart(bode).mark_line().encode(x=alt.X("Frequency (rad/s):Q", scale=alt.Scale(type="log")), y="Magnitude (dB):Q", tooltip=["Frequency (rad/s):Q", "Magnitude (dB):Q"]).properties(height=300, title="Bode magnitude"), alt="Bode magnitude of full terminal-voltage transfer function.")
    with right:
        poles, zeros = np.array([0., polarization_poles(params)[0]]), terminal_zeros(params)
        pz = pd.DataFrame({"Real part (1/s)": np.r_[poles, zeros.real], "Imaginary part (1/s)": np.r_[np.zeros_like(poles), zeros.imag], "Type": ["Pole", "Pole"] + ["Zero"] * len(zeros)})
        st.altair_chart(alt.Chart(pz).mark_point(filled=True, size=120).encode(x="Real part (1/s):Q", y="Imaginary part (1/s):Q", color="Type:N", shape="Type:N", tooltip=["Type:N", "Real part (1/s):Q", "Imaginary part (1/s):Q"]).properties(height=300, title="Pole-zero map"), alt="Pole-zero map of full terminal-voltage transfer function.")
    st.caption(f"Numerator coefficients: {np.array2string(num, precision=6)}; denominator: {np.array2string(den, precision=6)}. The RC subsystem is BIBO stable; the full model has an SOC integrator and is not BIBO stable under persistent current.")

with data_tab:
    st.subheader("Simulation data preview")
    preview = frame.rename(columns={"time_s": "Time (s)", "soc": "SOC", "polarization_voltage_v": "Polarization voltage (V)", "current_a": "Current (A)", "ocv_v": "OCV (V)", "terminal_voltage_v": "Terminal voltage (V)"}).round(6)
    st.dataframe(preview, hide_index=True, height=420, alt="Numerical simulation results including time, state of charge, current, and voltages.")
    with st.expander("Model limitations"):
        st.markdown("This educational first-order model omits temperature effects, coulombic efficiency, ageing, hysteresis, and SOC-dependent circuit parameters. The affine OCV curve and values are illustrative, not manufacturer specifications.")
