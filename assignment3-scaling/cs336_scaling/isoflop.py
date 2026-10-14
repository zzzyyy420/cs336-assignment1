"""Fit the compute-optimal parameter-count curve from isoflop results."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def power_law(log_compute_budget: np.ndarray, log_scale: float, exponent: float) -> np.ndarray:
    return log_scale + exponent * log_compute_budget


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "isoflops_curves.json"
    records = json.loads(input_path.read_text(encoding="utf-8"))

    grouped_by_budget = defaultdict(list)
    for record in records:
        grouped_by_budget[record["compute_budget"]].append(record)

    best_runs = [min(group, key=lambda record: record["final_loss"]) for group in grouped_by_budget.values()]
    print([(run["compute_budget"], run["parameters"], run["final_loss"]) for run in best_runs])

    log_compute_budgets = np.log10([run["compute_budget"] for run in best_runs])
    log_parameter_counts = np.log10([run["parameters"] for run in best_runs])
    log_token_counts = np.log10([run["compute_budget"] / (6 * run["parameters"]) for run in best_runs])
    parameter_fit, _ = curve_fit(power_law, log_compute_budgets, log_parameter_counts)
    token_fit, _ = curve_fit(power_law, log_compute_budgets, log_token_counts)

    plot_budgets = np.linspace(log_compute_budgets.min(), 24, 1000)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, observed, fit, label in (
        (axes[0], log_parameter_counts, parameter_fit, "parameter count"),
        (axes[1], log_token_counts, token_fit, "token count"),
    ):
        axis.scatter(log_compute_budgets, observed, label="IsoFLOPs optima")
        axis.plot(plot_budgets, power_law(plot_budgets, *fit), label="Power-law fit")
        axis.set_xlabel("log10(compute budget in FLOPs)")
        axis.set_ylabel(f"log10({label})")
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend()
    figure.tight_layout()
    figure.savefig(PROJECT_ROOT / "power_law_fit.png", dpi=300)
    plt.close()

    for target_compute_budget in (1e23, 1e24):
        optimal_parameters = 10 ** power_law(np.log10(target_compute_budget), *parameter_fit)
        optimal_tokens = 10 ** power_law(np.log10(target_compute_budget), *token_fit)
        print(f"C={target_compute_budget:.0e}, N={optimal_parameters:.2f}, D={optimal_tokens:.2f}")


if __name__ == "__main__":
    main()
