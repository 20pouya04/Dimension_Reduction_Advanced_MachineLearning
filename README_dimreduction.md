# Dimensionality Reduction: A Comparative Study (PCA, MDS, Isomap, t-SNE)

Compared linear and nonlinear dimensionality-reduction methods on
synthetic and real-world data, benchmarking runtime and downstream
classification accuracy.

**Status:** Course Project, Advanced Machine Learning (2024)

## Overview

This project compares four dimensionality-reduction methods:

- **PCA** (Principal Component Analysis) — linear
- **MDS** (Multidimensional Scaling) — nonlinear
- **Isomap** — nonlinear, manifold-based
- **t-SNE** (t-distributed Stochastic Neighbor Embedding) — nonlinear

across two datasets:

**1. Swiss Roll (synthetic data)** — A classic 3D nonlinear manifold. Each
method projects it down to 2D; PCA and MDS (which rely on straight-line/
global distances) fail to "unroll" the manifold, while Isomap and t-SNE
(which use local neighborhood structure) recover its intrinsic 2D shape.
This section also includes a **runtime benchmark** for PCA.

**2. Handwritten Digits (real-world data)** — The 64-dimensional
scikit-learn digits dataset (8×8 pixel images of digits 0-9). Each method
projects it down to 2D for visualization, and PCA/Isomap are also used as
preprocessing steps in a K-Nearest-Neighbors classification pipeline,
comparing **downstream classification accuracy** across a range of target
dimensionalities (1 to 60 components).

## Files

| File | Description |
|---|---|
| `dimensionality_reduction_comparison.py` | Clean, documented Python script — runs both sections and saves comparison plots. |
| `Dimention_Reduction_Advanced_MachineLearning.ipynb` | Original Jupyter notebook version of the project. |
| `LICENSE` | MIT License. |

## Requirements

- Python 3.8+
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/) (provides PCA, MDS, Isomap, t-SNE, KNN, and the datasets)

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the script directly to reproduce both sections of the study:

```bash
python dimensionality_reduction_comparison.py
```

This saves the following plots to the current directory and prints
runtime/accuracy results to the console:

- `swiss_roll_comparison.png` — PCA vs. MDS vs. Isomap vs. t-SNE on the Swiss Roll
- `digits_pca.png`, `digits_isomap.png`, `digits_mds.png`, `digits_tsne.png` — each method's 2D embedding of the digits dataset, colored by digit class
- `accuracy_vs_components.png` — downstream KNN classification accuracy vs. number of retained components, for PCA and Isomap

### Using individual functions

```python
from dimensionality_reduction_comparison import (
    make_swiss_roll_data, embed_swiss_roll, benchmark_pca_runtime,
    classification_accuracy_vs_components,
)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Swiss Roll comparison
X, y = make_swiss_roll_data(n_samples=500, random_state=0)
embeddings = embed_swiss_roll(X)  # dict: {"PCA": ..., "MDS": ..., "Isomap": ..., "t-SNE": ...}

# PCA runtime benchmark
runtime = benchmark_pca_runtime(X, n_repeats=1000)

# Downstream classification accuracy vs. number of components (digits dataset)
X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.3, random_state=1111)
n_components, acc_pca, acc_iso = classification_accuracy_vs_components(X_train, X_test, y_train, y_test)
```

Alternatively, open and run `Dimention_Reduction_Advanced_MachineLearning.ipynb`
in Jupyter for the original notebook walkthrough (which also includes
interactive 3D/2D Plotly visualizations of the Swiss Roll).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE)
for details.
