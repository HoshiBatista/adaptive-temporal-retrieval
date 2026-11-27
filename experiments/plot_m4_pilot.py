"""Plot the aggregate's descriptive training quality/cost, including the oracle bound."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new PNG path; also saves SVG")
    args = parser.parse_args()
    destinations = [args.output, args.output.with_suffix(".svg")]
    if any(path.exists() for path in destinations):
        raise ValueError("figure exists; choose a new path")
    summary = json.loads(args.results.read_text())["summary"]
    policies = summary["policies"]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for prefix, label, color, style in [
        ("random_query_budget_", "Seeded random routing", "#64748b", "o-"),
        ("oracle_query_budget_", "Outcome-aware oracle bound", "#d97706", "o--"),
    ]:
        rows = [r for r in policies if r["name"].startswith(prefix)]
        ax.plot(
            [r["modeled_mean_additive_stage_ms"] for r in rows],
            [r["ndcg_at_10"] for r in rows],
            style,
            label=label,
            color=color,
            markersize=5,
        )
    by_name = {r["name"]: r for r in policies}
    for name, label, color, marker in [
        ("always_fast", "Always hybrid", "#2563eb", "s"),
        ("always_deep", "Always rerank", "#dc2626", "s"),
        ("low_agreement", "Low-agreement rule", "#16a34a", "D"),
    ]:
        row = by_name[name]
        if name == "low_agreement":
            label = f"Low-agreement rule ({row['routed_fraction']:.0%} routed)"
        ax.scatter(
            row["modeled_mean_additive_stage_ms"],
            row["ndcg_at_10"],
            label=label,
            color=color,
            marker=marker,
            s=75,
            zorder=5,
        )
    ax.set(
        title=f"Train-only feasibility pilot: {summary['query_count']} queries, no fitted router",
        xlabel="Mean modeled additive stage cost (ms/query), not live latency",
        ylabel="nDCG@10",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.text(
        0.5,
        -0.015,
        "Oracle uses outcomes. All points are in-sample diagnostics; no held-out or service claim.",
        ha="center",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in destinations:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figures: {destinations}")


if __name__ == "__main__":
    main()
