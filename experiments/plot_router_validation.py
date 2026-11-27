"""Scientific figure for the frozen-model development validation comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.output, args.output.with_suffix(".svg")]
    if any(p.exists() for p in paths):
        raise ValueError("figure exists; choose a new path")
    data = json.loads(args.results.read_text())
    summary = data["summary"]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for kind, label, color, style in [
        ("logistic", "Frozen train500 logistic", "#2563eb", "o-"),
        ("random", "Seeded random", "#64748b", "o--"),
        ("agreement", "Agreement-ranked", "#16a34a", "o--"),
    ]:
        rows = [
            next(p for p in summary["policies"] if p["name"] == f"{kind}_budget_{b:g}")
            for b in data["config"]["query_route_budgets"]
        ]
        ax.plot(
            [p["modeled_mean_with_router_ms"] for p in rows],
            [p["ndcg_at_10"] for p in rows],
            style,
            color=color,
            label=label,
        )
    primary = summary["primary"]
    ax.scatter(
        primary["modeled_mean_with_router_ms"],
        primary["ndcg_at_10"],
        s=180,
        marker="*",
        color="#dc2626",
        zorder=10,
        label="Predeclared 10% primary",
    )
    ax.set(
        title=f"Frozen router: {summary['query_count']} development validation queries",
        xlabel="Modeled mean stage cost (ms/query), not live latency",
        ylabel="nDCG@10",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.text(
        0.5,
        -0.015,
        "Cohort query caps. Validation includes prior model-selection queries; test untouched.",
        ha="center",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved PNG/SVG:", paths)


if __name__ == "__main__":
    main()
