# -*- coding: utf-8 -*-
"""
run_all_sweeps.py

Runs the full DQC sigma/stage sweep (dqc_analysis.run_sweep) for every dataset
in the project, sphere-normalisation ON and OFF, back to back -- ten sweeps in
total. Each call uses the exact (sigma_min, sigma_max, sigma_step,
eig_threshold) combination used to produce the published results.

The eig_threshold overrides on GMM Overlap, Ellipsoid and Filament are not
optional tuning -- they are required to keep the sweep tractable (see
run_sweep's own docstring): the default 1e-5 keeps r ~ n on these datasets and
makes a single fit() prohibitively slow.

Run simulate_project_data.py first to create data/ (crabs.csv ships with the
repository). Results, report and plots for each run are written by run_sweep
itself to results/<csv-stem>_<sphere tag>_<sigma range>/.

    python simulate_project_data.py
    python run_all_sweeps.py
"""
import os

from dqc_analysis import run_sweep

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, "data")


def main() -> None:
    # ── Ripley's Crabs -- sigma 0.01-0.85, eig_threshold default (already fast) ─
    for sphere in (True, False):
        run_sweep(
            csv_path=os.path.join(DATA_FOLDER, "crabs.csv"),
            sigma_min=0.01, sigma_max=0.85, sigma_step=0.01,
            sphere_normalise=sphere, n_components=3,
        )

    # ── Moons (noise=0.1, seed=50) -- sigma 0.01-0.85, eig_threshold default ───
    for sphere in (True, False):
        run_sweep(
            csv_path=os.path.join(DATA_FOLDER, "moons_n300_noise0.1_seed50.csv"),
            sigma_min=0.01, sigma_max=0.85, sigma_step=0.01,
            sphere_normalise=sphere, n_components=3,
        )

    # ── GMM Overlap (sep=3.5) -- sigma 0.5-2, eig_threshold=1.0 required ───────
    for sphere in (True, False):
        run_sweep(
            csv_path=os.path.join(DATA_FOLDER, "gmm_overlap_n600_k4_d10_sep3.5_seed0.csv"),
            sigma_min=0.5, sigma_max=2, sigma_step=0.01,
            sphere_normalise=sphere, n_components=3, eig_threshold=1.0,
        )

    # ── Ellipsoid Clusters (elong=30) -- sigma 0.01-2, eig_threshold=1.0 required
    for sphere in (True, False):
        run_sweep(
            csv_path=os.path.join(DATA_FOLDER,
                                  "ellipsoid_clusters_n600_k4_d10_elong30_seed0.csv"),
            sigma_min=0.01, sigma_max=2, sigma_step=0.01,
            sphere_normalise=sphere, n_components=3, eig_threshold=1.0,
        )

    # ── Filament Clusters (seed=2, noise=1) -- sigma 0.01-0.75, eig_threshold=0.99
    # (0.99 handles the slow sigma=0.01 end; since higher sigma only needs LESS
    # aggressive truncation, one threshold value is safe across the whole range)
    for sphere in (True, False):
        run_sweep(
            csv_path=os.path.join(DATA_FOLDER,
                                  "filament_clusters_n700_d10_noise1_seed2.csv"),
            sigma_min=0.01, sigma_max=0.75, sigma_step=0.01,
            sphere_normalise=sphere, n_components=3, eig_threshold=0.99,
        )


if __name__ == "__main__":
    main()
