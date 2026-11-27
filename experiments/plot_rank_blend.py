"""Plot train-only selection and the frozen development validation result."""

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
    train, val = data["summary"]["train_cv"], data["summary"]["validation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    grid = train["train_grid_in_sample"]
    axes[0].plot(
        [float(w) for w in grid], [grid[w] for w in grid], "o-", label="In-sample train500"
    )
    axes[0].axhline(
        train["primary"]["ndcg_at_10"], color="#dc2626", linestyle="--", label="Fold-selected OOF"
    )
    axes[0].axhline(
        train["controls"][0]["ndcg_at_10"], color="#16a34a", linestyle=":", label="Hybrid"
    )
    axes[0].set(
        xlabel="Deep rank weight (train grid only)",
        ylabel="nDCG@10",
        title="Weight selection: training data only",
    )
    axes[0].legend(fontsize=9)
    policies = {p["name"]: p for p in val["policies"]}
    names = [
        "fast",
        "raw_deep",
        "frozen_blend",
        "frozen_logistic_transfer",
        "frozen_agreement_transfer",
    ]
    labels = [
        "Hybrid",
        "Raw deep",
        "Frozen\nblend",
        "Old logistic\n+ blend",
        "Old agreement\n+ blend",
    ]
    axes[1].bar(
        labels,
        [policies[n]["ndcg_at_10"] for n in names],
        color=["#16a34a", "#64748b", "#2563eb", "#7c3aed", "#0d9488"],
    )
    axes[1].set(ylabel="nDCG@10", title=f"Validation495: frozen weight {val['selected_weight']:g}")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        -0.025,
        "Transfer uses unchanged old routes, not a new learned policy. "
        "Development validation; test untouched.",
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
