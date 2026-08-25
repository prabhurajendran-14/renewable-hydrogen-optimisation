import numpy as np
import pandas as pd

def noisy_renewable_forecast(
    df: pd.DataFrame,
    noise_std: float,
    seed: int,
) -> pd.Series:
    """
    Create a reproducible imperfect forecast from the simulated renewable profile.

    This deliberately models forecast error without claiming a trained forecasting
    model. It can later be replaced by a separately validated forecasting pipeline.
    """
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")
    rng = np.random.default_rng(seed)
    values = df["renewable_kw"].to_numpy(dtype=float)
    forecast = np.clip(values + rng.normal(0, noise_std, size=len(values)), 0, None)
    return pd.Series(forecast, index=df.index, name="forecast_kw")
