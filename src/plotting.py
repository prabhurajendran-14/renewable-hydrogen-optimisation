import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_schedules(df, outpath=None):
    plt.figure(figsize=(11, 6))
    plt.plot(df["timestamp"], df["renewable_kw"], label="Renewable (actual)")
    plt.plot(
        df["timestamp"],
        df["forecast_kw"],
        label="Renewable forecast",
        alpha=0.65,
    )
    plt.plot(df["timestamp"], df["baseline_naive_kw"], label="Baseline (naive)")
    plt.plot(
        df["timestamp"],
        df["baseline_ramp_kw"],
        label="Baseline (ramp-limited)",
    )
    plt.plot(
        df["timestamp"],
        df["optimised_smooth_kw"],
        label="Optimised (smoothness)",
    )
    plt.plot(
        df["timestamp"],
        df["optimised_cost_kw"],
        label="Optimised (cost-aware)",
    )
    plt.legend(ncol=2)
    plt.title("Renewable Availability and Electrolyser Schedules")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.close()


def plot_tradeoff(tradeoff_df, outpath=None):
    plt.figure(figsize=(8, 5))
    plt.plot(
        tradeoff_df["total_ramp_kw"],
        tradeoff_df["total_hydrogen_kg"],
        marker="o",
    )
    grouped = (
        tradeoff_df.groupby(["total_ramp_kw", "total_hydrogen_kg"], sort=False)[
            "lambda"
        ]
        .apply(list)
        .reset_index()
    )
    for _, row in grouped.iterrows():
        lambda_values = ", ".join(f"{value:g}" for value in row["lambda"])
        plt.annotate(
            f"lambda={lambda_values}",
            (row["total_ramp_kw"], row["total_hydrogen_kg"]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    plt.title("Hydrogen Production vs Ramping")
    plt.xlabel("Total absolute ramp (kW)")
    plt.ylabel("Hydrogen production (kg)")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.close()


def plot_cost_comparison(metrics_df, outpath=None):
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["strategy"], metrics_df["cost_per_kg_h2_gbp"])
    plt.ylabel("Electricity cost (£/kg H₂)")
    plt.title("Cost Comparison by Control Strategy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.close()
