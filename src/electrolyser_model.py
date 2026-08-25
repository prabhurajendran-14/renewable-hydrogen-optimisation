import numpy as np

def hydrogen_production_kg(
    power_kw: np.ndarray,
    efficiency: float,
    step_hours: float,
    hydrogen_lhv_kwh_per_kg: float,
) -> np.ndarray:
    """
    Convert electrical power into hydrogen mass for each time step.

    This simplified model assumes constant electrolyser efficiency and uses
    hydrogen's lower heating value. It is suitable for a transparent portfolio
    prototype, not detailed plant design.
    """
    power_kw = np.asarray(power_kw, dtype=float)
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be between 0 and 1")
    if hydrogen_lhv_kwh_per_kg <= 0:
        raise ValueError("hydrogen_lhv_kwh_per_kg must be positive")
    energy_kwh = power_kw * step_hours
    return energy_kwh * efficiency / hydrogen_lhv_kwh_per_kg
