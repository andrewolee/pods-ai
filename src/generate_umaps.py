#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Generate a UMAP visualization from an embeddings CSV.

The CSV must contain:
    - embedding_0 ... embedding_N
    - ground_truth_label
    - global_prediction_label
    - predicted_label

Example:
    python generate_umaps.py \
        --embeddings_csv validation_embeddings.csv \
        --label_type predicted_label \
        --output_file validation_predicted.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import umap

from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a UMAP plot from embedding vectors."
    )

    parser.add_argument(
        "--embeddings_csv",
        required=True,
        help="CSV file containing embeddings."
    )

    parser.add_argument(
        "--label_type",
        required=True,
        choices=[
            "ground_truth_label",
            "global_prediction_label",
            "predicted_label",
        ],
        help="Column used to color the UMAP."
    )

    parser.add_argument(
        "--output_file",
        required=True,
        help="Output PNG filename."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print(f"Loading {args.embeddings_csv}")

    df = pd.read_csv(args.embeddings_csv)

    # Automatically find embedding columns
    embedding_cols = sorted(
        [c for c in df.columns if c.startswith("embedding_")],
        key=lambda x: int(x.split("_")[1])
    )

    if len(embedding_cols) == 0:
        raise ValueError("No embedding columns found.")

    print(f"Found {len(embedding_cols)} embedding dimensions.")

    X = df[embedding_cols].values
    labels = df[args.label_type].astype(str)

    print("\nClass counts:")
    print(labels.value_counts())

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print("\nRunning UMAP...")

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )

    embedding_2d = reducer.fit_transform(X)

    print("Creating figure...")

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=y,
        cmap="tab20",
        s=8,
        alpha=0.7,
    )

    # Colors used by scatter
    colors = scatter.cmap(
        scatter.norm(range(len(le.classes_)))
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=colors[i],
            markersize=8,
            label=label,
        )
        for i, label in enumerate(le.classes_)
    ]

    ax.legend(
        handles=legend_elements,
        title=args.label_type.replace("_", " ").title(),
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    ax.set_title(
        f"UMAP Colored by {args.label_type.replace('_', ' ').title()}"
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()

    print(f"Saving {args.output_file}")

    plt.savefig(
        args.output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
