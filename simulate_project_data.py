"""
simulate_project_data.py

Generates every dataset used in the sigma sweep and writes them as CSV to
data/.  Run once before run_all_sweeps.py:

    python simulate_project_data.py            # write the CSVs
    python simulate_project_data.py --plot     # also save an overview figure

Datasets produced (Crabs is not generated -- it is Ripley's real dataset and
ships with the repository as data/crabs.csv):

  - moons              -> non-convex, non-Gaussian cluster SHAPE
  - gmm_overlap        -> cluster SEPARATION (can DQC resolve overlapping
                           density modes with a single global sigma?)
  - ellipsoid_clusters -> cluster ANISOTROPY (DQC's radial Gaussian kernel
                           implicitly assumes near-isotropic clusters;
                           elongated ellipsoids stress that)
  - filament_clusters  -> mixed DENSITY SCALES (sparse thin filaments vs.
                           a compact dense blob sharing one bandwidth)

Every parameter below -- especially the `_DIM` values -- is meant to be
freely edited to explore how each dataset's structure changes with
dimensionality, separation, elongation and noise.  The values as committed
are the ones that produced the results in results/.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# ─────────────────────────────────────────
# GMM OVERLAP PARAMETERS — edit these
# ─────────────────────────────────────────
GMM_N_PER_CLUSTER = 150   # points per Gaussian cluster
GMM_DIM           = 10    # ambient dimensionality (free parameter)
GMM_SEPARATION    = 3.5    # distance between consecutive cluster means; lower = more overlap
GMM_N_CLUSTERS    = 4     # number of Gaussian clusters
GMM_SEED          = 0

# ─────────────────────────────────────────
# ELLIPSOID CLUSTERS PARAMETERS — edit these
# ─────────────────────────────────────────
ELLIPSOID_N_PER_CLUSTER = 150   # points per ellipsoid cluster
ELLIPSOID_DIM            = 10   # ambient dimensionality (free parameter)
ELLIPSOID_ELONGATION     = 30   # scale factor of the elongated axis; higher = more anisotropic
ELLIPSOID_N_CLUSTERS     = 4    # number of ellipsoid clusters
ELLIPSOID_CLUSTER_GAP    = 30   # spacing between cluster means (kept large so only shape varies)
ELLIPSOID_SEED           = 0

# ─────────────────────────────────────────
# FILAMENT CLUSTERS PARAMETERS — edit these
# ─────────────────────────────────────────
FILAMENT_N_PER_FILAMENT = 300    # points per filament curve
FILAMENT_N_BLOB          = 100   # points in the optional compact blob
FILAMENT_DIM             = 10    # ambient dimensionality (free parameter)
FILAMENT_NOISE           = 1     # std of noise around each filament curve; higher = thicker/sparser filament
FILAMENT_INCLUDE_BLOB    = True  # whether to add a compact spherical blob (label 2) alongside the filaments
FILAMENT_SEED            = 2

# ─────────────────────────────────────────
# MOONS PARAMETERS — edit these
# ─────────────────────────────────────────
MOONS_N_SAMPLES = 300   # total points across both half-moons
MOONS_NOISE     = 0.1   # std of Gaussian noise added to each point
MOONS_SEED      = 50


# ─────────────────────────────────────────
# Dataset generators
# ─────────────────────────────────────────

def make_gmm_overlap(
    n_per_cluster: int,
    dim: int,
    separation: float,
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian clusters strung along one direction with tunable overlap.

    Key difficulty knob: `separation`. It sets the distance between
    consecutive cluster means (mean_i = direction * separation * i). Lower
    separation means more density overlap between neighboring clusters,
    which directly tests whether DQC's potential landscape still forms
    distinct wells (one per cluster) rather than merging into a single
    broad well when a global sigma can't resolve the overlap.
    """
    rng = np.random.default_rng(random_state)

    # Fixed random unit direction along which cluster means are spaced.
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)

    X_parts = []
    y_parts = []
    for i in range(n_clusters):
        mean = direction * separation * i

        # Random positive-semi-definite covariance: A @ A.T / dim.
        A = rng.normal(size=(dim, dim))
        cov = A @ A.T / dim

        X_i = rng.multivariate_normal(mean=mean, cov=cov, size=n_per_cluster)
        X_parts.append(X_i)
        y_parts.append(np.full(n_per_cluster, i, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def make_ellipsoid_clusters(
    n_per_cluster: int,
    dim: int,
    elongation: float,
    n_clusters: int,
    cluster_gap: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Well-separated, non-spherical hyper-ellipsoid clusters.

    Key difficulty knob: `elongation`. Each cluster's covariance is built
    from a random orthonormal rotation (QR of a random matrix) with one axis
    scaled by `elongation` and the rest scaled by 1. Because cluster means
    are spaced far apart (`cluster_gap`), overlap is not a confound here --
    this dataset isolates whether DQC's isotropic Gaussian kernel (single
    scalar sigma) can still correctly separate clusters whose *shape* is
    highly anisotropic, independent of how well-separated they are.
    """
    rng = np.random.default_rng(random_state)

    X_parts = []
    y_parts = []
    for i in range(n_clusters):
        mean = np.zeros(dim)
        mean[0] = cluster_gap * i

        # Random orthonormal rotation via QR decomposition.
        M = rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(M)

        # Scale axes: first axis by `elongation`, others by 1.
        scales = np.ones(dim)
        scales[0] = elongation
        cov = Q @ np.diag(scales ** 2) @ Q.T

        X_i = rng.multivariate_normal(mean=mean, cov=cov, size=n_per_cluster)
        X_parts.append(X_i)
        y_parts.append(np.full(n_per_cluster, i, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def make_filament_clusters(
    n_per_filament: int,
    n_blob: int,
    dim: int,
    filament_noise: float,
    include_blob: bool,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two curved filaments (spiral + arc), optionally plus a compact blob.

    Key difficulty knob: `filament_noise`. It controls how tightly points
    hug the underlying curve (sparse, thin, elongated density) -- the
    opposite density regime from the compact spherical blob added when
    `include_blob=True`. Testing both together in the same dataset checks
    whether a single global bandwidth/density-scale parameter (as DQC uses)
    can simultaneously represent a sparse thin filament and a dense compact
    region without over- or under-smoothing one of them.
    """
    rng = np.random.default_rng(random_state)

    # Filament 1: a spiral.
    t1 = np.linspace(0.5, 2.5 * np.pi, n_per_filament)
    fil1_2d = np.stack([t1 * np.cos(t1), t1 * np.sin(t1)], axis=1)
    fil1_2d += rng.normal(scale=filament_noise, size=fil1_2d.shape)

    # Filament 2: an arc, offset from the spiral so the two are apart.
    offset = np.array([15.0, 15.0])
    t2 = np.linspace(0.0, np.pi, n_per_filament)
    radius = 6.0
    fil2_2d = np.stack([radius * np.cos(t2), radius * np.sin(t2)], axis=1) + offset
    fil2_2d += rng.normal(scale=filament_noise, size=fil2_2d.shape)

    # Same random projection matrix for both filaments (and the blob),
    # so everything shares a consistent high-D space.
    projection = rng.normal(size=(2, dim))
    X_fil1 = fil1_2d @ projection
    X_fil2 = fil2_2d @ projection

    X_parts = [X_fil1, X_fil2]
    y_parts = [np.zeros(n_per_filament, dtype=int), np.ones(n_per_filament, dtype=int)]

    if include_blob:
        # Compact spherical blob, well-separated from both filaments,
        # placed in the same projected high-D space.
        blob_center_2d = np.array([-20.0, 20.0])
        blob_2d = blob_center_2d + rng.normal(scale=0.5, size=(n_blob, 2))
        X_blob = blob_2d @ projection
        X_parts.append(X_blob)
        y_parts.append(np.full(n_blob, 2, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y



# ─────────────────────────────────────────
# Section: Save each dataset as CSV
# ─────────────────────────────────────────

def _save_dataset_csv(name: str, X: np.ndarray, y: np.ndarray, fname: str) -> None:
    feat_cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df["label"] = y.astype(int)

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, fname)
    df.to_csv(path, index=False)

    counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"[{name}] Saved: {path}")
    print(f"        Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {len(counts)}")
    print(f"        Class counts: {counts}")
    print()


# ─────────────────────────────────────────
# Section: Plot all datasets
# ─────────────────────────────────────────

def plot_datasets(datasets, save_path=None, show=True) -> None:
    """Plot each (X, y) dataset in its own subplot, colored by ground truth.

    - 1D data is plotted against a constant y-axis.
    - 2D data is plotted directly.
    - Higher-D data is reduced to 2D via PCA (fit separately per dataset)
      purely as an approximate visual sanity check on structure -- this is
      NOT the space DQC actually clusters on, since DQC operates on the
      full-dimensional (or its own internally preprocessed) feature space.
    """
    n = len(datasets)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    cmap = plt.get_cmap("tab10")

    for ax, (name, (X, y)) in zip(axes, datasets.items()):
        classes = sorted(np.unique(y))
        n_features = X.shape[1]

        if n_features == 1:
            X2d = np.stack([X[:, 0], np.ones(X.shape[0])], axis=1)
            xlabel, ylabel = "Feature 1", ""
            title = name
        elif n_features == 2:
            X2d = X
            xlabel, ylabel = "Feature 1", "Feature 2"
            title = name
        else:
            # Approximate visual check only -- DQC clusters on the full
            # (or its own preprocessed) high-D space, not this PCA plane.
            pca = PCA(n_components=2)
            X2d = pca.fit_transform(X)
            xlabel, ylabel = "PC1", "PC2"
            title = f"{name} ({n_features}D -> PCA 2D)"

        for k in classes:
            mask = y == k
            ax.scatter(
                X2d[mask, 0], X2d[mask, 1],
                color=cmap(k % 10), s=15, alpha=0.8, label=f"class {k}",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    # Hide any unused subplot axes.
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

# ─────────────────────────────────────────
# Section: Generate and save every dataset
# ─────────────────────────────────────────

def main(plot: bool = False) -> None:
    X_moons, y_moons = make_moons(
        n_samples=MOONS_N_SAMPLES,
        noise=MOONS_NOISE,
        random_state=MOONS_SEED,
    )

    X_gmm, y_gmm = make_gmm_overlap(
        n_per_cluster=GMM_N_PER_CLUSTER,
        dim=GMM_DIM,
        separation=GMM_SEPARATION,
        n_clusters=GMM_N_CLUSTERS,
        random_state=GMM_SEED,
    )

    X_ellipsoid, y_ellipsoid = make_ellipsoid_clusters(
        n_per_cluster=ELLIPSOID_N_PER_CLUSTER,
        dim=ELLIPSOID_DIM,
        elongation=ELLIPSOID_ELONGATION,
        n_clusters=ELLIPSOID_N_CLUSTERS,
        cluster_gap=ELLIPSOID_CLUSTER_GAP,
        random_state=ELLIPSOID_SEED,
    )

    X_filament, y_filament = make_filament_clusters(
        n_per_filament=FILAMENT_N_PER_FILAMENT,
        n_blob=FILAMENT_N_BLOB,
        dim=FILAMENT_DIM,
        filament_noise=FILAMENT_NOISE,
        include_blob=FILAMENT_INCLUDE_BLOB,
        random_state=FILAMENT_SEED,
    )

    moons_fname = f"moons_n{MOONS_N_SAMPLES}_noise{MOONS_NOISE}_seed{MOONS_SEED}.csv"
    _save_dataset_csv("MOONS", X_moons, y_moons, moons_fname)

    gmm_fname = (f"gmm_overlap_n{X_gmm.shape[0]}_k{GMM_N_CLUSTERS}_d{GMM_DIM}"
                 f"_sep{GMM_SEPARATION}_seed{GMM_SEED}.csv")
    _save_dataset_csv("GMM OVERLAP", X_gmm, y_gmm, gmm_fname)

    ellipsoid_fname = (f"ellipsoid_clusters_n{X_ellipsoid.shape[0]}_k{ELLIPSOID_N_CLUSTERS}"
                       f"_d{ELLIPSOID_DIM}_elong{ELLIPSOID_ELONGATION}_seed{ELLIPSOID_SEED}.csv")
    _save_dataset_csv("ELLIPSOID CLUSTERS", X_ellipsoid, y_ellipsoid, ellipsoid_fname)

    filament_fname = (f"filament_clusters_n{X_filament.shape[0]}_d{FILAMENT_DIM}"
                      f"_noise{FILAMENT_NOISE}_seed{FILAMENT_SEED}.csv")
    _save_dataset_csv("FILAMENT CLUSTERS", X_filament, y_filament, filament_fname)

    if plot:
        datasets = {
            "Moons": (X_moons, y_moons),
            "GMM Overlap": (X_gmm, y_gmm),
            "Ellipsoid Clusters": (X_ellipsoid, y_ellipsoid),
            "Filament Clusters": (X_filament, y_filament),
        }
        overview_path = os.path.join(SCRIPT_DIR, "simulated_datasets_overview.png")
        plot_datasets(datasets, save_path=overview_path, show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true",
                        help="also save and show a PCA overview figure of the datasets")
    main(plot=parser.parse_args().plot)
