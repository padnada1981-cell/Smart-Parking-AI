"""
comparison.py
-------------
Benchmarks BFS, A*, and Hill Climbing side by side on the same parking scenarios.
Produces a console table, a CSV file, and matplotlib comparison charts.
Also exports helper functions imported by the Streamlit UI.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Safe non-GUI backend for macOS Terminal and Streamlit
import matplotlib.pyplot as plt
import pandas as pd

from parking_model import ParkingLot, create_vehicles
from bfs_algorithm import bfs_allocate
from astar_algorithm import astar_allocate
from hill_climbing import hill_climbing_allocate


# ---------- Core comparison helpers (also used by Stage 7 UI) ----------
def run_comparison(lot, vehicles, hc_seed=42, max_states=500_000):
    """Run all three algorithms on the same lot and vehicles."""
    return [
        bfs_allocate(lot, vehicles, max_states=max_states),
        astar_allocate(lot, vehicles, max_states=max_states),
        hill_climbing_allocate(lot, vehicles, seed=hc_seed),
    ]


def results_to_dataframe(results, scenario_name=None):
    """Convert a list of algorithm result dicts into a clean pandas DataFrame."""
    rows = []
    for res in results:
        row = {
            "Algorithm": res["algorithm"],
            "Success": res["success"],
            "Total Cost": res["total_cost"],
            "Explored States": res["explored_states"],
            "Time (ms)": round(res["time_ms"], 4),
            "Allocation": str(res["allocation"]) if res["allocation"] is not None else "None",
            "Message": res["message"],
        }
        if scenario_name is not None:
            row = {"Scenario": scenario_name, **row}
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- Benchmark suite ----------
def get_benchmark_scenarios():
    """Five scenarios designed to highlight cost, scalability, and edge cases."""
    return [
        {
            "name": "S1: Light Load",
            "occupied": ["P2", "P7", "P12"],
            "cars": 2,
            "bikes": 1,
        },
        {
            "name": "S2: Front Row Blocked",
            "occupied": ["P1", "P2", "P3", "P4"],
            "cars": 2,
            "bikes": 0,
        },
        {
            "name": "S3: Medium Load",
            "occupied": ["P1", "P2", "P6", "P7", "P16"],
            "cars": 3,
            "bikes": 1,
        },
        {
            "name": "S4: Heavy Load",
            "occupied": ["P2", "P4", "P8", "P12", "P17"],
            "cars": 4,
            "bikes": 2,
        },
        {
            "name": "S5: No Bike Slots",
            "occupied": ["P16", "P17", "P18", "P19", "P20"],
            "cars": 1,
            "bikes": 1,
        },
    ]


def run_benchmark_suite(hc_seed=42):
    """Run all benchmark scenarios and return one combined DataFrame."""
    frames = []
    for sc in get_benchmark_scenarios():
        lot = ParkingLot()
        lot.set_occupied_slots(sc["occupied"])
        vehicles = create_vehicles(sc["cars"], sc["bikes"])
        results = run_comparison(lot, vehicles, hc_seed=hc_seed)
        frames.append(results_to_dataframe(results, scenario_name=sc["name"]))
    return pd.concat(frames, ignore_index=True)


# ---------- Chart generation ----------
def _plot_grouped_metric(df, metric_col, ylabel, title, log_scale=False):
    """Helper to build one grouped bar chart across scenarios and algorithms."""
    valid = df[df["Success"]].copy()
    pivot = valid.pivot(index="Scenario", columns="Algorithm", values=metric_col)
    # Keep scenario and algorithm order consistent
    scenario_order = list(dict.fromkeys(valid["Scenario"]))
    algo_order = [a for a in ["BFS", "A*", "Hill Climbing"] if a in pivot.columns]
    pivot = pivot.reindex(index=scenario_order, columns=algo_order)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C78A8", "#54A24B", "#F58518"]
    pivot.plot(kind="bar", ax=ax, color=colors[:len(algo_order)], width=0.72)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Algorithm")

    if log_scale:
        ax.set_yscale("log")

    # Label each bar with its exact value
    for container in ax.containers:
        labels = []
        for bar in container:
            h = bar.get_height()
            if pd.isna(h) or h == 0:
                labels.append("")
            elif metric_col == "Time (ms)":
                labels.append(f"{h:.2f}")
            else:
                labels.append(f"{int(round(h))}")
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)

    fig.tight_layout()
    return fig


def create_comparison_charts(df):
    """Return a dict of matplotlib figures for cost, explored states, and time."""
    return {
        "cost_comparison.png": _plot_grouped_metric(
            df, "Total Cost", "Total Distance Cost (lower is better)",
            "Total Allocation Cost by Scenario", log_scale=False
        ),
        "states_comparison.png": _plot_grouped_metric(
            df, "Explored States", "Explored States (log scale)",
            "Explored States by Scenario", log_scale=True
        ),
        "time_comparison.png": _plot_grouped_metric(
            df, "Time (ms)", "Execution Time in ms (log scale)",
            "Execution Time by Scenario", log_scale=True
        ),
    }


def save_charts(charts, output_dir="charts"):
    """Save generated figures to disk as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for filename, fig in charts.items():
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)
    return saved_paths


# ---------- Self-test (run directly from Terminal) ----------
if __name__ == "__main__":
    df = run_benchmark_suite(hc_seed=42)

    print("=== Smart Parking Algorithm Comparison ===\n")
    display_cols = ["Scenario", "Algorithm", "Success", "Total Cost", "Explored States", "Time (ms)"]
    print(df[display_cols].to_string(index=False))
    print()

    csv_path = "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved table to: {csv_path}")

    charts = create_comparison_charts(df)
    paths = save_charts(charts, output_dir="charts")
    print("Saved charts:")
    for p in paths:
        print(f"  - {p}")
