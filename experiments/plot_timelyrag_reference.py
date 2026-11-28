"""Static scientific comparison of matched native baseline and automatic temporal code."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.output, args.output.with_suffix(".svg")]
    if any(p.exists() for p in paths):
        raise ValueError("figure exists; choose a new path")
    public = json.loads(args.results.read_text())
    data = public["summary"]["results"]
    methods = ["bm25", "dense", "hybrid_rrf"]
    labels = ["BM25", "E5 dense", "E5 hybrid"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    axes[0].bar(
        x - 0.18,
        [data[m]["base_metrics"]["ndcg_at_10"] for m in methods],
        width=0.36,
        label="Native baseline",
        color="#16a34a",
    )
    axes[0].bar(
        x + 0.18,
        [data[m]["auto_metrics"]["ndcg_at_10"] for m in methods],
        width=0.36,
        label="+ Official auto-alpha",
        color="#2563eb",
    )
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Full-qrel nDCG@10",
        title="Matched validation495: retrieval quality",
    )
    axes[0].legend(fontsize=9)
    means = [data[m]["incremental_temporal_mean_ms"] for m in methods]
    axes[1].bar(labels, means, color="#64748b")
    axes[1].set(
        ylabel="Mean incremental temporal stage (ms/query)",
        title="CPU temporal module; cached fast scores",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        -0.025,
        "Code reproduction on local validation, not whole-paper scores. "
        "E5 on CUDA; temporal code on CPU. Test untouched.",
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
