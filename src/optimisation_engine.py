import numpy as np
from scipy.optimize import linprog


def _validate_inputs(
    renewable_forecast: np.ndarray,
    p_max: float,
    ramp_up_limit: float,
) -> np.ndarray:
    forecast = np.asarray(renewable_forecast, dtype=float)
    if forecast.ndim != 1 or len(forecast) == 0:
        raise ValueError("renewable_forecast must be a non-empty 1D array")
    if not np.all(np.isfinite(forecast)):
        raise ValueError("renewable_forecast must contain only finite values")
    if p_max <= 0 or ramp_up_limit <= 0:
        raise ValueError("p_max and ramp_up_limit must be positive")
    return np.clip(forecast, 0, None)


def _solve_linear_schedule(
    renewable_forecast: np.ndarray,
    p_max: float,
    ramp_up_limit: float,
    power_coefficients: np.ndarray,
    ramp_coefficient: float,
) -> dict:
    """
    Solve a linear schedule with auxiliary variables for absolute ramping.

    Variables are [power_0 ... power_T-1, absolute_ramp_1 ... absolute_ramp_T-1].
    Downward curtailment is permitted when renewable supply drops, while upward
    changes remain limited.
    """
    forecast = _validate_inputs(renewable_forecast, p_max, ramp_up_limit)
    power_coefficients = np.asarray(power_coefficients, dtype=float)
    if power_coefficients.shape != forecast.shape:
        raise ValueError("power_coefficients must match renewable_forecast")
    if ramp_coefficient < 0:
        raise ValueError("ramp_coefficient cannot be negative")

    time_steps = len(forecast)
    ramp_steps = max(0, time_steps - 1)
    objective = np.concatenate(
        [power_coefficients, np.full(ramp_steps, ramp_coefficient)]
    )

    rows = []
    limits = []
    for t in range(1, time_steps):
        ramp_index = time_steps + t - 1

        # Upward ramp: power_t - power_t-1 <= ramp_up_limit.
        upward = np.zeros(time_steps + ramp_steps)
        upward[t] = 1.0
        upward[t - 1] = -1.0
        rows.append(upward)
        limits.append(ramp_up_limit)

        # Absolute ramp variable >= power_t - power_t-1.
        positive = upward.copy()
        positive[ramp_index] = -1.0
        rows.append(positive)
        limits.append(0.0)

        # Absolute ramp variable >= power_t-1 - power_t.
        negative = -upward
        negative[ramp_index] = -1.0
        rows.append(negative)
        limits.append(0.0)

    bounds = [(0.0, min(p_max, value)) for value in forecast]
    bounds.extend([(0.0, None)] * ramp_steps)
    result = linprog(
        objective,
        A_ub=np.asarray(rows) if rows else None,
        b_ub=np.asarray(limits) if limits else None,
        bounds=bounds,
        method="highs",
    )

    schedule = result.x[:time_steps] if result.success else np.full(time_steps, np.nan)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "schedule_kw": schedule,
        "objective_value": float(result.fun) if result.success else float("nan"),
        "nit": int(result.nit),
    }


def optimise_schedule_objective2(
    renewable_forecast: np.ndarray,
    p_max: float,
    ramp_up_limit: float,
    efficiency: float,
    step_hours: float,
    hydrogen_lhv_kwh_per_kg: float,
    lambda_ramp: float,
) -> dict:
    """Maximise hydrogen production while penalising absolute power ramping."""
    hydrogen_kg_per_kw_step = (
        efficiency * step_hours / hydrogen_lhv_kwh_per_kg
    )
    power_coefficients = np.full(
        len(renewable_forecast), -hydrogen_kg_per_kw_step
    )
    return _solve_linear_schedule(
        renewable_forecast,
        p_max,
        ramp_up_limit,
        power_coefficients,
        lambda_ramp,
    )


def optimise_schedule_objective3(
    renewable_forecast: np.ndarray,
    prices_gbp_per_kwh: np.ndarray,
    p_max: float,
    ramp_up_limit: float,
    efficiency: float,
    step_hours: float,
    hydrogen_lhv_kwh_per_kg: float,
    ramp_penalty_gbp_per_kw: float,
    hydrogen_value_gbp_per_kg: float,
) -> dict:
    """Minimise energy and ramping costs minus the synthetic value of hydrogen."""
    prices = np.asarray(prices_gbp_per_kwh, dtype=float)
    if np.any(prices < 0) or not np.all(np.isfinite(prices)):
        raise ValueError("prices must be finite and non-negative")
    hydrogen_value_per_kw_step = (
        hydrogen_value_gbp_per_kg
        * efficiency
        * step_hours
        / hydrogen_lhv_kwh_per_kg
    )
    power_coefficients = prices * step_hours - hydrogen_value_per_kw_step
    return _solve_linear_schedule(
        renewable_forecast,
        p_max,
        ramp_up_limit,
        power_coefficients,
        ramp_penalty_gbp_per_kw,
    )
