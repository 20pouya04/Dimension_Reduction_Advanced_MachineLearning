"""
Dimensionality Reduction: A Comparative Study
================================================

Compares linear and nonlinear dimensionality-reduction methods —
**PCA**, **MDS**, **Isomap**, and **t-SNE** — on both synthetic data (the
Swiss Roll manifold) and real-world data (the handwritten digits dataset),
benchmarking runtime and downstream classification accuracy.

Sections
--------
1. **Swiss Roll (synthetic, nonlinear manifold)** — projects a 3D
   Swiss Roll down to 2D with each method and compares the resulting
   embeddings, plus a runtime benchmark for PCA.
2. **Handwritten Digits (real-world, high-dimensional)** — projects the
   64-dimensional digits dataset down to 2D with each method for
   visualization, then measures downstream K-Nearest-Neighbors
   classification accuracy using PCA and Isomap as preprocessing steps
   across a range of target dimensionalities.

Usage
-----
    python dimensionality_reduction_comparison.py

This generates and saves comparison plots (PNG files) in the current
directory and prints runtime/accuracy results to the console.

Or import the functions to reuse in your own analysis:

    from dimensionality_reduction_comparison import (
        embed_swiss_roll, benchmark_pca_runtime,
        embed_digits, classification_accuracy_vs_components,
    )
"""

import timeit

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Section 1: Swiss Roll (synthetic data)
# ---------------------------------------------------------------------------

def make_swiss_roll_data(n_samples=500, flatten_factor=0.6, random_state=None):
    """
    Generate a Swiss Roll dataset: a classic nonlinear manifold used to
    test dimensionality-reduction methods. The manifold is flattened
    slightly along its third axis to make it a bit less trivial.
    """
    X, y = make_swiss_roll(n_samples=n_samples, random_state=random_state)
    X[:, 2] = flatten_factor * X[:, 2]
    return X, y


def embed_swiss_roll(X):
    """
    Project a 3D Swiss Roll dataset down to 2D using PCA (linear), MDS,
    Isomap, and t-SNE (all nonlinear). Returns a dict mapping method name
    to its 2D embedding.
    """
    embeddings = {}
    embeddings["PCA"] = PCA(n_components=2).fit_transform(X)
    embeddings["MDS"] = MDS(n_components=2, normalized_stress="auto").fit_transform(X)
    embeddings["Isomap"] = Isomap(n_components=2).fit_transform(X)
    embeddings["t-SNE"] = TSNE(n_components=2).fit_transform(X)
    return embeddings


def plot_swiss_roll_embeddings(embeddings, y, output_path="swiss_roll_comparison.png"):
    """Plot the four 2D embeddings side by side, colored by the manifold's intrinsic coordinate."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (name, X_2d) in zip(axes.ravel(), embeddings.items()):
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def benchmark_pca_runtime(X, n_repeats=1000):
    """Benchmark the runtime of fitting PCA (2 components) `n_repeats` times."""
    start = timeit.default_timer()
    for _ in range(n_repeats):
        PCA(n_components=2).fit_transform(X)
    stop = timeit.default_timer()
    return stop - start


# ---------------------------------------------------------------------------
# Section 2: Handwritten Digits (real-world data)
# ---------------------------------------------------------------------------

def embed_digits(X, isomap_neighbors=20):
    """
    Project the digits dataset down to 2D using PCA, Isomap, MDS, and
    t-SNE, for visualization purposes. Returns a dict mapping method name
    to its 2D embedding.
    """
    embeddings = {}
    embeddings["PCA"] = PCA(n_components=2).fit_transform(X)
    embeddings["Isomap"] = Isomap(n_components=2, n_neighbors=isomap_neighbors).fit_transform(X)
    embeddings["MDS"] = MDS(n_components=2, normalized_stress="auto").fit_transform(X)
    embeddings["t-SNE"] = TSNE(n_components=2).fit_transform(X)
    return embeddings


def plot_digit_embedding(X_2d, y, title, output_path):
    """Plot one 2D embedding of the digits dataset, colored by digit class (0-9)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for digit in range(10):
        mask = y == digit
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], alpha=0.4, label=str(digit))
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def knn_pipeline_accuracy(X_train, X_test, y_train, y_test, reducer=None, n_neighbors=5):
    """
    Train a KNN classifier (optionally preceded by a dimensionality-reduction
    step) and return its accuracy on the test set.
    """
    if reducer is None:
        clf = KNN(n_neighbors=n_neighbors)
    else:
        clf = Pipeline(steps=[("Dimensionality Reduction", reducer), ("Classifier", KNN(n_neighbors=n_neighbors))])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_true=y_test, y_pred=y_pred)


def classification_accuracy_vs_components(X_train, X_test, y_train, y_test, max_components=60, isomap_neighbors=9):
    """
    Sweep the number of target components from 1 to `max_components` for
    both PCA and Isomap, and record downstream KNN classification accuracy
    at each dimensionality. Returns three lists: n_components, PCA
    accuracies, and Isomap accuracies.
    """
    n_components_list, acc_pca, acc_iso = [], [], []

    for n in range(1, max_components + 1):
        acc_pca.append(knn_pipeline_accuracy(X_train, X_test, y_train, y_test, reducer=PCA(n_components=n)))
        acc_iso.append(
            knn_pipeline_accuracy(
                X_train, X_test, y_train, y_test, reducer=Isomap(n_components=n, n_neighbors=isomap_neighbors)
            )
        )
        n_components_list.append(n)

    return n_components_list, acc_pca, acc_iso


def plot_accuracy_vs_components(n_components_list, acc_pca, acc_iso, output_path="accuracy_vs_components.png"):
    """Plot downstream classification accuracy against number of retained components."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_components_list, acc_iso, label="Isomap")
    ax.plot(n_components_list, acc_pca, label="PCA")
    ax.axhline(0.98, color="green", linestyle="--", label="98% reference")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("KNN accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=== Section 1: Swiss Roll (synthetic data) ===")
    X_roll, y_roll = make_swiss_roll_data(n_samples=500, random_state=0)
    embeddings = embed_swiss_roll(X_roll)
    path = plot_swiss_roll_embeddings(embeddings, y_roll)
    print(f"Saved: {path}")

    runtime = benchmark_pca_runtime(X_roll, n_repeats=1000)
    print(f"PCA runtime for 1000 fits: {runtime:.4f} seconds")

    print("\n=== Section 2: Handwritten Digits (real-world data) ===")
    X_digits, y_digits = load_digits(return_X_y=True)

    digit_embeddings = embed_digits(X_digits)
    for name, X_2d in digit_embeddings.items():
        path = plot_digit_embedding(X_2d, y_digits, title=name, output_path=f"digits_{name.lower().replace('-', '')}.png")
        print(f"Saved: {path}")

    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.3, random_state=1111)

    print("\nBaseline classification accuracy (no dimensionality reduction, 2D PCA, 2D Isomap):")
    acc_baseline = knn_pipeline_accuracy(X_train, X_test, y_train, y_test, reducer=None)
    acc_pca_2d = knn_pipeline_accuracy(X_train, X_test, y_train, y_test, reducer=PCA(n_components=2))
    acc_iso_2d = knn_pipeline_accuracy(X_train, X_test, y_train, y_test, reducer=Isomap(n_components=2, n_neighbors=9))
    print(f"  KNN (no reduction): {acc_baseline:.4f}")
    print(f"  KNN + PCA (2D):     {acc_pca_2d:.4f}")
    print(f"  KNN + Isomap (2D):  {acc_iso_2d:.4f}")

    print("\nSweeping number of components (1-60) for PCA and Isomap...")
    n_components_list, acc_pca, acc_iso = classification_accuracy_vs_components(X_train, X_test, y_train, y_test, max_components=60)
    path = plot_accuracy_vs_components(n_components_list, acc_pca, acc_iso)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
