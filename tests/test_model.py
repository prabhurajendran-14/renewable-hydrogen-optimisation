import unittest

import numpy as np
import pandas as pd

from src.baseline import ramp_limited_reactive_controller
from src.electrolyser_model import hydrogen_production_kg
from src.evaluation import evaluate_with_cost
from src.forecasting import noisy_renewable_forecast
from src.optimisation_engine import optimise_schedule_objective2
from src.pricing import generate_price_signal


class ModelTests(unittest.TestCase):
    def test_hydrogen_conversion_uses_timestep_and_physical_units(self):
        production = hydrogen_production_kg(
            np.array([100.0]),
            efficiency=0.65,
            step_hours=0.25,
            hydrogen_lhv_kwh_per_kg=33.33,
        )
        self.assertAlmostEqual(production[0], 100 * 0.25 * 0.65 / 33.33)

    def test_cost_calculation_uses_kwh_not_instantaneous_kw(self):
        metrics = evaluate_with_cost(
            np.array([100.0, 100.0]),
            efficiency=0.65,
            prices_gbp_per_kwh=np.array([0.20, 0.20]),
            step_hours=0.25,
            hydrogen_lhv_kwh_per_kg=33.33,
        )
        self.assertAlmostEqual(metrics["total_energy_kwh"], 50.0)
        self.assertAlmostEqual(metrics["total_cost_gbp"], 10.0)

    def test_noisy_inputs_are_reproducible(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=8, freq="15min"),
                "renewable_kw": np.linspace(0, 70, 8),
            }
        )
        first_forecast = noisy_renewable_forecast(frame, 5.0, seed=12)
        second_forecast = noisy_renewable_forecast(frame, 5.0, seed=12)
        np.testing.assert_allclose(first_forecast, second_forecast)

        first_prices = generate_price_signal(frame["timestamp"], seed=22)
        second_prices = generate_price_signal(frame["timestamp"], seed=22)
        np.testing.assert_allclose(first_prices, second_prices)

    def test_ramp_limited_baseline_respects_upward_limit(self):
        forecast = np.array([0.0, 40.0, 80.0, 5.0, 60.0])
        schedule = ramp_limited_reactive_controller(
            forecast,
            p_max=80.0,
            ramp_limit=15.0,
        )
        self.assertTrue(np.all(schedule <= forecast + 1e-9))
        self.assertTrue(np.all(np.diff(schedule) <= 15.0 + 1e-9))

    def test_smoothness_optimiser_respects_bounds_and_upward_ramp(self):
        forecast = np.array([0.0, 20.0, 50.0, 10.0, 45.0, 60.0])
        result = optimise_schedule_objective2(
            renewable_forecast=forecast,
            p_max=50.0,
            ramp_up_limit=12.0,
            efficiency=0.65,
            step_hours=0.25,
            hydrogen_lhv_kwh_per_kg=33.33,
            lambda_ramp=0.005,
        )
        self.assertTrue(result["success"], result["message"])
        schedule = result["schedule_kw"]
        self.assertTrue(np.all(schedule >= -1e-8))
        self.assertTrue(np.all(schedule <= np.minimum(50.0, forecast) + 1e-7))
        self.assertTrue(np.all(np.diff(schedule) <= 12.0 + 1e-7))


if __name__ == "__main__":
    unittest.main()
