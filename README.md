# Dynamic Quantum Clustering — Gaussian width (σ) sweep

Code for the research project on how the Gaussian width parameter σ behaves in
Dynamic Quantum Clustering (DQC), and whether density-optimal bandwidths
(LSCV, Silverman/Scott) coincide with cluster-optimal σ.

## Files

| File | What it is |
|---|---|
| `dqc_core.py` | The DQC algorithm: overlap matrix, quantum potential, Hamiltonian, basis truncation, time evolution, k-means extraction |
| `dqc_analysis.py` | `run_sweep()` — the σ × stage sweep for one dataset: builds the working coordinates, computes the LSCV and Silverman/Scott bandwidths and the k-means baseline, runs four DQC stages at every σ, scores each stage, and writes the results CSV, report and plots |
| `run_all_sweeps.py` | Calls `run_sweep()` for all five datasets, sphere-normalisation on and off — ten sweeps in total |
| `simulate_project_data.py` | Generates the four simulated datasets into `data/` |
| `dqc_interface.py` | Streamlit app for stepping through a single σ stage by stage and viewing the cluster formation |
| `data/crabs.csv` | Ripley's Crabs — the one real dataset, not generated |

## Running it

```bash
pip install -r requirements.txt
python simulate_project_data.py     # writes the four simulated CSVs to data/
python run_all_sweeps.py            # runs all ten sweeps
```

Everything is seeded, so the generated datasets and all reported numbers are
reproducible exactly.

`run_all_sweeps.py` writes each run's results CSV, `report.txt`, near-optimal
regions and figures to `results/<dataset>_<sphere tag>_<sigma range>/`. That
directory is not tracked here — the reported results live in the paper.

To explore a single σ interactively:

```bash
streamlit run dqc_interface.py
```

## Datasets and sweep settings

| Dataset | n | d | k | σ range (step 0.01) | `eig_threshold` |
|---|---|---|---|---|---|
| Crabs (real) | 200 | 5 | 4 | 0.01 – 0.85 | 1e-5 (default) |
| Moons | 300 | 2 | 2 | 0.01 – 0.85 | 1e-5 (default) |
| GMM Overlap | 600 | 10 | 4 | 0.50 – 2.00 | 1.0 |
| Ellipsoid Clusters | 600 | 10 | 4 | 0.01 – 2.00 | 1.0 |
| Filament Clusters | 700 | 10 | 3 | 0.01 – 0.75 | 0.99 |

Every run uses `n_components=3`, mass `m = 0.20`, four DQC stages, and
t ∈ [0.025, 5.0] in 200 steps.

`eig_threshold` is raised on the three largest datasets for tractability: at
the default 1e-5 almost every eigenvalue of the overlap matrix survives
truncation, so `r ≈ n` and a single stage becomes prohibitively slow. Raising
it is what makes those sweeps finish, and also what produces the missing
(`NaN`) entries at later stages — once the evolved cloud contracts enough that
fewer than three eigenvalues clear the threshold, that stage and every stage
after it fails at that σ.
