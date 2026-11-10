import unittest
import numpy as np

from battery_model import BatteryParameters, analytical_constant_current, constant_current, rhs, terminal_voltage
from numerical_methods import integrate


class BatterySimulationTests(unittest.TestCase):
    def setUp(self):
        self.params = BatteryParameters()

    def test_rhs_at_initial_condition(self):
        derivative = rhs(0.0, self.params.initial_state, self.params, constant_current(1.0))
        self.assertAlmostEqual(derivative[0], 1 / 7200)
        self.assertAlmostEqual(derivative[1], 1 / 2000)

    def test_analytical_initial_condition(self):
        exact = analytical_constant_current([0.0], self.params)[0]
        np.testing.assert_allclose(exact, self.params.initial_state)

    def test_rk4_matches_analytical_solution(self):
        result = integrate(self.params, constant_current(1.0), 240.0, 1.0, "rk4")
        exact = analytical_constant_current(result.time_s, self.params)
        # RK4 truncation error at h=1 s is below 3e-11 V for this 40 s RC mode.
        np.testing.assert_allclose(result.states, exact, rtol=0, atol=3e-11)

    def test_physical_cutoff(self):
        result = integrate(self.params, constant_current(1.0), 7200.0, 10.0, "rk4", physical_cutoff=True)
        self.assertAlmostEqual(result.soc[-1], 1.0)
        self.assertEqual(result.current_a[-1], 0.0)

    def test_unconstrained_report_case(self):
        result = integrate(self.params, constant_current(1.0), 7200.0, 10.0, "rk4")
        self.assertAlmostEqual(result.soc[-1], 1.2)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BatteryParameters(capacity_ah=0)
        with self.assertRaises(ValueError):
            integrate(self.params, constant_current(1.0), step_s=0)

    def test_terminal_voltage(self):
        voltage = terminal_voltage(np.array([0.2, 0.0]), 1.0, self.params)
        self.assertAlmostEqual(voltage, 3.29)


if __name__ == "__main__":
    unittest.main()
