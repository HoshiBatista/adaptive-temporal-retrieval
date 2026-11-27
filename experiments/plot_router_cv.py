"""Static scientific quality-cost plot of the grouped out-of-fold training controls."""

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
    dest = [args.output, args.output.with_suffix(".svg")]
    if any(p.exists() for p in dest):
        raise ValueError("figure exists; choose a new path")
    data = json.loads(args.results.read_text())
    summary = data["summary"]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for kind, label, color in [
        ("full", "Logistic: full features", "#2563eb"),
        ("no_temporal", "Logistic: no explicit temporal features", "#7c3aed"),
    ]:
        variant = next(
            v for v in summary["variants"] if v["penalty"] == 0 and v["feature_set"] == kind
        )
        rows = [p for p in variant["policies"] if "query_budget_cap" in p]
        ax.plot(
            [p["modeled_mean_with_router_ms"] for p in rows],
            [p["ndcg_at_10"] for p in rows],
            "o-",
            color=color,
            label=label,
        )
    for kind, label, color in [
        ("random", "Seeded random query budgets", "#64748b"),
        ("agreement", "Agreement-ranked query budgets", "#16a34a"),
    ]:
        rows = [p for p in summary["controls"] if p["name"].startswith(kind + "_budget_")]
        ax.plot(
            [p["modeled_mean_stage_ms"] for p in rows],
            [p["ndcg_at_10"] for p in rows],
            "o--",
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
        label="Predeclared primary point",
    )
    ax.set(
        title=(
            f"Grouped OOF training: {summary['query_count']} queries / "
            f"{summary['group_count']} groups"
        ),
        xlabel="Modeled mean stage cost (ms/query), not live latency",
        ylabel="nDCG@10",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.text(
        0.5,
        -0.015,
        "Fixed per-fold query caps. No validation/test or measured routed-service latency.",
        ha="center",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for p in dest:
        fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved PNG/SVG:", dest)


if __name__ == "__main__":
    main()
