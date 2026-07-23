"""
Visualize SAM Audio Latents

Loads a *_latents.pt file produced by generate_latents.py:
    {
        "latents": torch.Tensor [E, C, T]  (fixed-length T across all samples)
        "labels":  torch.Tensor [E, N_C]   (multi-hot)
    }

Also tolerates the older list-of-variable-length-tensors format
([E] list of [C, T_i]) if you ever run this against an older file.

Produces:
    1. A 2D scatter plot (PCA and/or t-SNE) of pooled latent embeddings,
       colored by the dominant (or single, if one-hot) class per sample.
    2. A channel-by-time heatmap for a handful of example latents.
    3. A per-class count bar chart, so you can sanity check class balance
       in whatever slice you're visualizing.

Usage:
    python visualize_latents.py --latents_path /path/to/train_latents.pt --output_dir /path/to/plots
    python visualize_latents.py --latents_path train_latents.pt --output_dir plots --method tsne --max_samples 2000
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe; we're saving PNGs, not showing a window
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Class names for BirdSet HSN — edit/replace if you run this on a different split.
DEFAULT_CLASS_NAMES = [
    "gcrfin", "whcspa", "amepip", "sposan", "rocwre", "brebla", "daejun",
    "foxspa", "clanut", "moublu", "casfin", "mallar3", "herthr", "amerob",
    "yerwar", "yelwar", "dusfly", "mouchi", "orcwar", "warvir", "norfli",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--latents_path", required=True, help="Path to a *_latents.pt file")
parser.add_argument("--output_dir", required=True, help="Directory to save plots")
parser.add_argument(
    "--method",
    choices=["pca", "tsne", "both"],
    default="both",
    help="Dimensionality reduction method for the scatter plot",
)
parser.add_argument(
    "--pool",
    choices=["mean", "max", "flatten"],
    default="mean",
    help="How to collapse each [C, T] latent into a single feature vector. "
         "'flatten' keeps the full [C*T] latent (only sensible now that T is fixed-length).",
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=3000,
    help="Subsample this many latents before running t-SNE (t-SNE gets slow/memory-heavy above a few thousand points)",
)
parser.add_argument(
    "--num_heatmap_examples",
    type=int,
    default=4,
    help="How many individual latent heatmaps to render",
)
parser.add_argument(
    "--class_names",
    nargs="*",
    default=None,
    help="Optional list of class names matching the label columns. Defaults to the BirdSet HSN class list.",
)
parser.add_argument("--seed", type=int, default=42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pool_latent(latent: torch.Tensor, mode: str) -> np.ndarray:
    """Collapse a [C, T] latent into a single feature vector."""
    if mode == "mean":
        pooled = latent.mean(dim=-1)
    elif mode == "max":
        pooled = latent.amax(dim=-1)
    elif mode == "flatten":
        pooled = latent.reshape(-1)  # [C*T] — only meaningful when T is fixed across samples
    else:
        raise ValueError(f"Unknown pool mode: {mode}")
    return pooled.numpy()


def dominant_class(label_row: np.ndarray, class_names: list[str]) -> str:
    """Pick a single label to color a (possibly multi-label) sample by.
    Uses the highest-valued entry; falls back to 'none' if all zero."""
    if label_row.sum() == 0:
        return "none"
    idx = int(np.argmax(label_row))
    if idx < len(class_names):
        return class_names[idx]
    return f"class_{idx}"


def reduce_dims(features: np.ndarray, method: str, seed: int) -> dict[str, np.ndarray]:
    """Run the requested dimensionality reduction method(s), returning {method_name: 2D coords}."""
    results = {}
    if method in ("pca", "both"):
        pca = PCA(n_components=2, random_state=seed)
        results["pca"] = pca.fit_transform(features)
        print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")
    if method in ("tsne", "both"):
        from sklearn.manifold import TSNE  # imported lazily; slower dependency chain
        n = features.shape[0]
        perplexity = min(30, max(5, n // 100))
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
        results["tsne"] = tsne.fit_transform(features)
    return results


def plot_scatter(coords: np.ndarray, colors: list[str], title: str, out_path: str):
    unique_labels = sorted(set(colors))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for lab in unique_labels:
        mask = [c == lab for c in colors]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.7, color=label_to_color[lab], label=lab)

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    # Keep the legend readable even with ~20 classes: small font, outside the plot.
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, markerscale=1.5, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_heatmaps(get_latent, indices: list[int], out_path: str):
    n = len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), squeeze=False)
    for row, idx in enumerate(indices):
        latent = get_latent(idx).numpy()  # [C, T]
        ax = axes[row][0]
        im = ax.imshow(latent, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"sample {idx}  (shape {latent.shape})")
        ax.set_xlabel("time step")
        ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_class_counts(labels: np.ndarray, class_names: list[str], out_path: str):
    counts = labels.sum(axis=0)
    n_classes = len(counts)
    names = class_names[:n_classes] if len(class_names) >= n_classes else [
        class_names[i] if i < len(class_names) else f"class_{i}" for i in range(n_classes)
    ]

    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.4), 5))
    ax.bar(range(n_classes), counts[order])
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([names[i] for i in order], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("count")
    ax.set_title("Class frequency in this latent file")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    class_names = args.class_names if args.class_names else DEFAULT_CLASS_NAMES

    print(f">> Loading {args.latents_path}")
    data = torch.load(args.latents_path, map_location="cpu")
    latents = data["latents"]          # torch.Tensor [E, C, T], or a list of [C, T_i] (older format)
    labels = data["labels"].numpy()    # [E, N_C]

    if isinstance(latents, torch.Tensor):
        n = latents.shape[0]
        print(f"   {n} samples, fixed shape per sample = {tuple(latents.shape[1:])}, label dim = {labels.shape[1]}")
    else:
        n = len(latents)
        print(f"   {n} samples (variable-length latents), label dim = {labels.shape[1]}")

    get_latent = lambda i: latents[i]  # works for both a tensor (indexing) and a list  # noqa: E731

    # --- Subsample for the scatter plot if the file is large (mainly matters for t-SNE cost) ---
    rng = np.random.RandomState(args.seed)
    if n > args.max_samples:
        sample_idx = rng.choice(n, size=args.max_samples, replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(n)
    print(f">> Using {len(sample_idx)} samples for the scatter plot")

    # --- Pool each latent to a single vector, build color labels ---
    print(f">> Pooling latents (mode={args.pool})")
    features = np.stack([pool_latent(get_latent(i), args.pool) for i in sample_idx])
    colors = [dominant_class(labels[i], class_names) for i in sample_idx]

    # --- Dimensionality reduction + scatter plots ---
    print(f">> Reducing dimensions (method={args.method})")
    reduced = reduce_dims(features, args.method, args.seed)
    for method_name, coords in reduced.items():
        out_path = os.path.join(args.output_dir, f"scatter_{method_name}.png")
        plot_scatter(coords, colors, f"Latent space ({method_name.upper()}, pooled={args.pool})", out_path)

    # --- Per-sample heatmaps (raw, unpooled latents) ---
    print(">> Rendering example heatmaps")
    example_idx = list(sample_idx[: args.num_heatmap_examples])
    plot_heatmaps(get_latent, example_idx, os.path.join(args.output_dir, "example_heatmaps.png"))

    # --- Class balance bar chart (over the full dataset, not just the subsample) ---
    print(">> Rendering class frequency chart")
    plot_class_counts(labels, class_names, os.path.join(args.output_dir, "class_counts.png"))

    print("\n>> Done.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)