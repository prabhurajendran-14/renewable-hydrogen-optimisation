import numpy as np
import pandas as pd

def generate_price_signal(
    timestamps: pd.Series,
    base: float = 0.18,
    noise_std: float = 0.005,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthetic electricity price pattern (£/kWh):
    - cheaper overnight
    - more expensive during evening peak
    """
    hours = pd.to_datetime(timestamps).dt.hour.to_numpy()
    price = np.full(len(hours), base, dtype=float)

    # Overnight discount
    price[(hours >= 0) & (hours < 6)] -= 0.05

    # Morning bump
    price[(hours >= 7) & (hours < 10)] += 0.10

    # Evening peak
    price[(hours >= 16) & (hours < 20)] += 0.30

    rng = np.random.default_rng(seed)
    price += rng.normal(0, noise_std, size=len(price))
    price = np.clip(price, 0.05, None)

    return price
