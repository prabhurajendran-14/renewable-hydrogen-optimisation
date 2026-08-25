import numpy as np
import pandas as pd
from .electrolyser_model import hydrogen_production_kg

def evaluate(
    schedule_kw: np.ndarray,
    efficiency: float,
    step_hours: float,
    hydrogen_lhv_kwh_per_kg: float,
) -> dict:
    schedule_kw = np.asarray(schedule_kw, dtype=float)
    h = hydrogen_production_kg(
        schedule_kw, efficiency, step_hours, hydrogen_lhv_kwh_per_kg
    )
    ramps = np.abs(np.diff(schedule_kw, prepend=schedule_kw[0]))
    return {
        "total_hydrogen_kg": float(np.sum(h)),
        "total_energy_kwh": float(np.sum(schedule_kw) * step_hours),
        "total_ramp_kw": float(np.sum(ramps)),
        "max_ramp_kw": float(np.max(ramps)),
        "mean_power_kw": float(np.mean(schedule_kw)),
    }

def to_results_df(
    timestamps,
    renewable,
    forecast,
    prices,
    baseline_naive,
    baseline_ramp,
    optimised_smooth,
    optimised_cost,
) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": timestamps,
        "renewable_kw": renewable,
        "forecast_kw": forecast,
        "price_gbp_per_kwh": prices,
        "baseline_naive_kw": baseline_naive,
        "baseline_ramp_kw": baseline_ramp,
        "optimised_smooth_kw": optimised_smooth,
        "optimised_cost_kw": optimised_cost,
    })

def evaluate_with_cost(
    schedule_kw: np.ndarray,
    efficiency: float,
    prices_gbp_per_kwh: np.ndarray,
    step_hours: float,
    hydrogen_lhv_kwh_per_kg: float,
) -> dict:
    schedule_kw = np.asarray(schedule_kw, dtype=float)
    prices = np.asarray(prices_gbp_per_kwh, dtype=float)
    if schedule_kw.shape != prices.shape:
        raise ValueError("schedule and price arrays must have the same shape")

    metrics = evaluate(
        schedule_kw, efficiency, step_hours, hydrogen_lhv_kwh_per_kg
    )
    total_cost = float(np.sum(prices * schedule_kw * step_hours))
    total_h2 = metrics["total_hydrogen_kg"]
    metrics.update({
        "total_cost_gbp": total_cost,
        "cost_per_kg_h2_gbp": total_cost / total_h2 if total_h2 > 0 else float("inf"),
    })
    return metrics
