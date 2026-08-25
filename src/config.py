from dataclasses import dataclass

@dataclass
class Config:
    """Model assumptions for the synthetic 24-hour case study."""

    freq_minutes: int = 15
    horizon_hours: int = 24

    p_max: float = 80.0
    ramp_up_limit_kw: float = 15.0

    efficiency: float = 0.65
    hydrogen_lhv_kwh_per_kg: float = 33.33

    hydrogen_value_gbp_per_kg: float = 12.0
    ramp_penalty_gbp_per_kw: float = 0.05

    renewable_peak: float = 100.0
    noise_std: float = 8.0
    forecast_noise_std: float = 5.0
    price_noise_std: float = 0.005
    random_seed: int = 42

    @property
    def step_hours(self) -> float:
        return self.freq_minutes / 60.0
