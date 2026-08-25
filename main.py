import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.baseline import naive_controller, ramp_limited_reactive_controller
from src.config import Config
from src.data_simulation import simulate_solar_like_power
from src.evaluation import evaluate, evaluate_with_cost, to_results_df
from src.forecasting import noisy_renewable_forecast
from src.optimisation_engine import (
    optimise_schedule_objective2,
    optimise_schedule_objective3,
)
from src.plotting import plot_cost_comparison, plot_schedules, plot_tradeoff
from src.pricing import generate_price_signal


def _base_metrics(schedule, cfg):
    return evaluate(
        schedule,
        cfg.efficiency,
        cfg.step_hours,
        cfg.hydrogen_lhv_kwh_per_kg,
    )


def _cost_metrics(schedule, prices, cfg):
    return evaluate_with_cost(
        schedule,
        cfg.efficiency,
        prices,
        cfg.step_hours,
        cfg.hydrogen_lhv_kwh_per_kg,
    )


def run_analysis(output_dir: Path, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    periods = int((cfg.horizon_hours * 60) / cfg.freq_minutes)
    actual = simulate_solar_like_power(
        start="2026-02-24 00:00:00",
        periods=periods,
        freq=f"{cfg.freq_minutes}min",
        peak_kw=cfg.renewable_peak,
        noise_std=cfg.noise_std,
        seed=cfg.random_seed,
    )
    forecast = noisy_renewable_forecast(
        actual,
        noise_std=cfg.forecast_noise_std,
        seed=cfg.random_seed + 1,
    ).to_numpy()
    prices = generate_price_signal(
        actual["timestamp"],
        noise_std=cfg.price_noise_std,
        seed=cfg.random_seed + 2,
    )

    baseline_naive = naive_controller(forecast, cfg.p_max)
    baseline_ramp = ramp_limited_reactive_controller(
        forecast,
        cfg.p_max,
        ramp_limit=cfg.ramp_up_limit_kw,
    )

    lambda_values = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    tradeoff_rows = []
    schedules_by_lambda = {}
    for lambda_ramp in lambda_values:
        result = optimise_schedule_objective2(
            renewable_forecast=forecast,
            p_max=cfg.p_max,
            ramp_up_limit=cfg.ramp_up_limit_kw,
            efficiency=cfg.efficiency,
            step_hours=cfg.step_hours,
            hydrogen_lhv_kwh_per_kg=cfg.hydrogen_lhv_kwh_per_kg,
            lambda_ramp=lambda_ramp,
        )
        if not result["success"]:
            raise RuntimeError(
                f"Smoothness optimisation failed for lambda={lambda_ramp}: "
                f"{result['message']}"
            )
        schedule = result["schedule_kw"]
        metrics = _base_metrics(schedule, cfg)
        tradeoff_rows.append(
            {
                "lambda": lambda_ramp,
                **metrics,
                "iterations": result["nit"],
            }
        )
        schedules_by_lambda[lambda_ramp] = schedule

    chosen_lambda = 0.005
    optimised_smooth = schedules_by_lambda[chosen_lambda]

    cost_result = optimise_schedule_objective3(
        renewable_forecast=forecast,
        prices_gbp_per_kwh=prices,
        p_max=cfg.p_max,
        ramp_up_limit=cfg.ramp_up_limit_kw,
        efficiency=cfg.efficiency,
        step_hours=cfg.step_hours,
        hydrogen_lhv_kwh_per_kg=cfg.hydrogen_lhv_kwh_per_kg,
        ramp_penalty_gbp_per_kw=cfg.ramp_penalty_gbp_per_kw,
        hydrogen_value_gbp_per_kg=cfg.hydrogen_value_gbp_per_kg,
    )
    if not cost_result["success"]:
        raise RuntimeError(f"Cost optimisation failed: {cost_result['message']}")
    optimised_cost = cost_result["schedule_kw"]

    strategy_schedules = {
        "Naive baseline": baseline_naive,
        "Ramp-limited baseline": baseline_ramp,
        "Smoothness optimised": optimised_smooth,
        "Cost-aware optimised": optimised_cost,
    }
    metrics_rows = []
    for strategy, schedule in strategy_schedules.items():
        metrics_rows.append(
            {"strategy": strategy, **_cost_metrics(schedule, prices, cfg)}
        )
    metrics_df = pd.DataFrame(metrics_rows)
    tradeoff_df = pd.DataFrame(tradeoff_rows)
    results_df = to_results_df(
        actual["timestamp"],
        actual["renewable_kw"],
        forecast,
        prices,
        baseline_naive,
        baseline_ramp,
        optimised_smooth,
        optimised_cost,
    )

    results_df.to_csv(output_dir / "schedule_results.csv", index=False)
    tradeoff_df.to_csv(output_dir / "lambda_sensitivity.csv", index=False)
    metrics_df.to_csv(output_dir / "strategy_metrics.csv", index=False)
    plot_schedules(results_df, figures_dir / "schedule_comparison.png")
    plot_tradeoff(tradeoff_df, figures_dir / "tradeoff_curve.png")
    plot_cost_comparison(metrics_df, figures_dir / "cost_comparison.png")

    assumptions = asdict(cfg)
    assumptions["step_hours"] = cfg.step_hours
    (output_dir / "assumptions.json").write_text(
        json.dumps(assumptions, indent=2), encoding="utf-8"
    )
    summary = {
        "chosen_lambda": chosen_lambda,
        "cost_optimiser_iterations": cost_result["nit"],
        "strategies": metrics_df.set_index("strategy").to_dict(orient="index"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run the renewable hydrogen scheduling case study."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV, JSON, and figure outputs.",
    )
    args = parser.parse_args()
    summary = run_analysis(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
