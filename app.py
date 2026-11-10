"""Optional Streamlit demonstration interface; calculations remain in core modules."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import altair as alt

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
            st.markdown("#### SOC trajectory and final state")
            soc_chart = pd.DataFrame({
                "Time (s)": frame["time_s"],
                "SOC (%)": frame["soc"] * 100,
            })
            final_soc = float(result.soc[-1] * 100)
            axis_top = max(110.0, final_soc * 1.05)
            x_axis = alt.X("Time (s):Q", title="Time (s)")
            y_axis = alt.Y("SOC (%):Q", title="State of charge (%)", scale=alt.Scale(domain=[0, axis_top]))
            trajectory = alt.Chart(soc_chart).mark_line().encode(
                x=x_axis,
                y=y_axis,
                tooltip=[alt.Tooltip("Time (s):Q", format=",.0f"), alt.Tooltip("SOC (%):Q", format=".2f")],
            )
            full_soc_line = alt.Chart(pd.DataFrame({"SOC (%)": [100.0]})).mark_rule(
                color="#ff4b4b", strokeDash=[6, 4]
            ).encode(y=y_axis)
            final_marker = alt.Chart(pd.DataFrame({
                "Time (s)": [float(result.time_s[-1])], "SOC (%)": [final_soc]
            })).mark_point(size=100, filled=True).encode(
                x=x_axis, y=y_axis,
                tooltip=[alt.Tooltip("Time (s):Q", title="Final time", format=",.0f"), alt.Tooltip("SOC (%):Q", title="Final SOC", format=".2f")],
            )
            st.altair_chart(
                (trajectory + full_soc_line + final_marker).properties(height=330),
                alt="State of charge over time with the final state marked and a dashed 100 percent full-SOC reference.",
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
    st.subheader("Simulation data preview")
    preview = frame.rename(columns={
        "time_s": "Time (s)", "soc": "SOC", "polarization_voltage_v": "Polarization voltage (V)",
        "current_a": "Current (A)", "ocv_v": "OCV (V)", "terminal_voltage_v": "Terminal voltage (V)",
    }).round(6)
    st.dataframe(
        preview,
        hide_index=True,
        height=350,
        alt="Numerical simulation results including time, state of charge, current, and voltages.",
    )
else:
    st.info("Choose parameters and select Run simulation.")
