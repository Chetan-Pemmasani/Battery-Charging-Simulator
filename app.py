"""Optional Streamlit demonstration interface; calculations remain in core modules."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from battery_model import BatteryParameters, constant_current, open_circuit_voltage, terminal_voltage
from linear_systems import eigenvalues, polarization_poles, state_space_matrices
from numerical_methods import integrate

st.set_page_config(page_title="Battery Charging Simulation", layout="wide")
st.title("Battery Charging Simulation")
st.caption("Thevenin model — positive current means charging. Default mode reproduces the report's unconstrained 120% SOC experiment.")

with st.sidebar:
    st.header("Simulation controls")
    capacity = st.number_input("Capacity (Ah)", min_value=0.1, value=2.0, step=0.1)
    initial_soc = st.slider("Initial SOC", 0.0, 1.0, 0.20, 0.01)
    current = st.number_input("Charging current (A)", min_value=0.0, value=1.0, step=0.1)
    r0 = st.number_input("R₀ (Ω)", min_value=0.0, value=0.05, step=0.01)
    rp = st.number_input("Rₚ (Ω)", min_value=0.001, value=0.02, step=0.005)
    cp = st.number_input("Cₚ (F)", min_value=1.0, value=2000.0, step=100.0)
    duration = st.number_input("Duration (s)", min_value=1.0, value=7200.0, step=60.0)
    step = st.selectbox("Step size (s)", [10.0, 5.0, 2.0, 1.0])
    method = st.selectbox("Method", ["rk4", "heun", "euler"])
    cutoff = st.checkbox("Apply physical full-SOC cutoff", value=False)
    run = st.button("Run simulation", type="primary")

if run:
    params = BatteryParameters(capacity, initial_soc, 0.0, r0, rp, cp)
    result = integrate(params, constant_current(current), duration, step, method, cutoff)
    frame = pd.DataFrame({
        "time_s": result.time_s, "soc": result.soc, "polarization_voltage_v": result.polarization_voltage,
        "current_a": result.current_a,
    })
    frame["ocv_v"] = open_circuit_voltage(frame["soc"].to_numpy())
    frame["terminal_voltage_v"] = terminal_voltage(result.states, result.current_a, params)
    one, two, three = st.columns(3)
    one.metric("Final SOC", f"{result.soc[-1] * 100:.2f}%")
    two.metric("Final terminal voltage", f"{frame['terminal_voltage_v'].iloc[-1]:.4f} V")
    three.metric("RC time constant", f"{params.time_constant_s:.1f} s")
    response_tab, analysis_tab = st.tabs(["Simulation response", "State-space analysis"])
    with response_tab:
        left, right = st.columns(2)
        with left:
            st.markdown("#### State of charge")
            st.line_chart(
                pd.DataFrame({"SOC (%)": frame["soc"] * 100}, index=frame["time_s"]),
                alt="State of charge over time during charging.",
            )
        with right:
            st.markdown("#### Voltage response")
            st.line_chart(
                pd.DataFrame({
                    "Terminal voltage (V)": frame["terminal_voltage_v"],
                    "OCV (V)": frame["ocv_v"],
                    "Polarization voltage (V)": frame["polarization_voltage_v"],
                }, index=frame["time_s"]),
                alt="Terminal voltage, open-circuit voltage, and polarization voltage over time.",
            )
        with st.container(border=True):
            st.markdown("#### SOC summary")
            soc_summary = pd.DataFrame({
                "Stage": ["Initial", "Final", "Full-SOC reference"],
                "SOC (%)": [initial_soc * 100, result.soc[-1] * 100, 100.0],
            })
            st.bar_chart(
                soc_summary,
                x="Stage",
                y="SOC (%)",
                y_label="SOC (%)",
                alt="Initial and final state of charge, compared with the 100 percent full-SOC reference.",
            )

    with analysis_tab:
        st.subheader("State-space analysis")
        st.caption("State vector: x = [z, v_p]^T; input u = charging current I(t).")
        a_matrix, b_matrix = state_space_matrices(params)
        eig = eigenvalues(params)
        pole = polarization_poles(params)[0]

        matrix_col, input_col = st.columns(2)
        with matrix_col:
            st.markdown("**System matrix A**")
            st.latex(
                rf"A = \begin{{bmatrix}} 0 & 0 \\ 0 & -\frac{{1}}{{R_p C_p}} \end{{bmatrix}}"
                rf"= \begin{{bmatrix}} 0 & 0 \\ 0 & {a_matrix[1, 1]:.6f} \end{{bmatrix}}\;s^{{-1}}"
            )
        with input_col:
            st.markdown("**Input matrix B**")
            st.latex(
                rf"B = \begin{{bmatrix}} \frac{{1}}{{3600Q}} \\ \frac{{1}}{{C_p}} \end{{bmatrix}}"
                rf"= \begin{{bmatrix}} {b_matrix[0, 0]:.8f} \\ {b_matrix[1, 0]:.8f} \end{{bmatrix}}"
            )

        first, second, third = st.columns(3)
        first.metric("SOC eigenvalue", f"{eig[0]:.6f}")
        second.metric("RC eigenvalue / pole", f"{pole:.6f} s⁻¹")
        third.metric("RC time constant", f"{params.time_constant_s:.2f} s")

        st.success("RC polarization subsystem: asymptotically stable (its pole is negative).")
        st.warning(
            "Full charging model: marginal, not asymptotically stable. "
            "The zero SOC eigenvalue represents charge accumulation under continued input."
        )
    st.download_button("Download CSV", frame.to_csv(index=False), "battery_simulation.csv", "text/csv")
else:
    st.info("Choose parameters and select Run simulation.")
