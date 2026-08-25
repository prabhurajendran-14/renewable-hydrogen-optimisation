import numpy as np

def naive_controller(renewable_forecast: np.ndarray, p_max: float) -> np.ndarray:
    """
    Baseline 1 (naive): use as much renewable as possible, clipped to [0, p_max].
    NOTE: No ramp constraints.
    """
    renewable_forecast = np.asarray(renewable_forecast, dtype=float)
    return np.clip(renewable_forecast, 0, p_max)


def ramp_limited_reactive_controller(
    renewable_forecast: np.ndarray,
    p_max: float,
    ramp_limit: float,
) -> np.ndarray:
    """
    Baseline 2 (reactive + realistic):
    Follow available renewable power, but enforce:
      - bounds [0, p_max]
      - ramp-up limit per timestep

    Downward curtailment is allowed when renewable availability falls faster
    than the plant's normal ramp limit.

    This gives a fair baseline vs the optimiser (which also respects ramp limits).
    """
    renewable_forecast = np.asarray(renewable_forecast, dtype=float)
    T = len(renewable_forecast)
    schedule = np.zeros(T)

    for t in range(T):
        # Desired power is "use what's available", clipped to max
        desired = min(renewable_forecast[t], p_max)

        if t == 0:
            schedule[t] = desired
            continue

        # Enforce ramp limit from previous step
        prev = schedule[t - 1]
        lower = max(0.0, prev - ramp_limit)
        upper = min(p_max, prev + ramp_limit)

        # Also cannot exceed renewable available at time t
        upper = min(upper, renewable_forecast[t])

        # Clip desired into [lower, upper]
        p_t = min(max(desired, lower), upper)

        schedule[t] = p_t

    return schedule
