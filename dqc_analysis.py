"""
dqc_analysis.py

Importable sigma/stage sweep over the existing DQC implementation. This
file has no CLI/main() and does nothing on its own -- import it and call
run_sweep() on whatever CSV you want to analyse:

    from dqc_analysis import run_sweep
    results_df, report = run_sweep(
        csv_path="C:/path/to/your_dataset.csv",
        sigma_min=0.05, sigma_max=1.25, sigma_step=0.05,
        n_components=3, sphere_normalise=False,
    )

The CSV just needs feature columns plus a label column (auto-detected --
see _find_label_column). A crabs.csv (Ripley's Crab dataset, the classic
Weinstein & Horn DQC benchmark) is bundled alongside this file for that
purpose.

This script does NOT reimplement DQC. It reuses, unmodified:
  - The DQC class (evolution, reinitialise, k-means extraction) from
    dqc_core.py.
  - The exact stage-workflow logic (fit -> select t* from the entropy
    peak -> extract clusters -> reinitialise -> repeat) and the exact
    metric functions (_pairwise_jaccard, _select_t_star,
    _compute_entropy_curve, _compute_pca) as coded in dqc_interface.py.
    Those helper functions are copied verbatim below (not reimplemented)
    because dqc_interface.py itself cannot be imported outside
    `streamlit run` (it calls st.set_page_config() at import time).

What run_sweep adds on top of the reused pieces above, per dataset:
  - An evenly-spaced sigma grid (sigma_min .. sigma_max in steps of
    sigma_step) x stage sweep (fixed at 4 stages, matching the interface's
    4-stage limit).
  - Two label-blind density-optimal reference bandwidths, computed only
    from the working coordinates' own spread (no reference to the true
    labels), for comparison against the label-aware cluster-optimal sigma
    the sweep finds:
      - Silverman/Scott normal-reference bandwidth (_silverman_scott_bandwidth)
      - Least-Squares Cross-Validation bandwidth (_lscv_bandwidth, via
        statsmodels' KDEMultivariate(bw='cv_ls'))
  - A plain k-means baseline (_kmeans_baseline): k-means run directly on
    the raw, un-evolved PCA coordinates (t=0, no DQC evolution at all),
    to show whether DQC's evolution step adds anything over doing nothing.
  - A results table (one row per sigma x stage), a text report (per-metric
    max + near-optimal region), and plots -- all saved to
    results/<csv-stem>/, and the results table also returned as a DataFrame.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'   # match dqc_core.py / dqc_interface.py convention

import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score)
from sklearn.preprocessing import LabelEncoder
from statsmodels.nonparametric.kernel_density import KDEMultivariate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from dqc_core import DQC   # noqa: E402  (existing DQC evolution + k-means extraction — reused as-is)


# ─────────────────────────────────────────
# Fixed DQC parameters — match dqc_interface.py's defaults exactly, and are
# NOT run_sweep() parameters: the interface itself doesn't let you tune
# these per sigma/dataset either — they're constants of "how a stage runs",
# not sweep inputs. Edit here only if the interface's own defaults change.
# ─────────────────────────────────────────
EIG_THRESH = 1e-5   # interface's default eigenvalue threshold for uploaded CSVs
T_MAX      = 5.0    # interface default "Max time T"
N_STEPS    = 200    # interface default "Time steps"
DELTA_T    = T_MAX / N_STEPS

KMEANS_SEED = 42   # fixed seed used by DQC.extract_clusters_kmeans (dqc_core.py) — not overridden here
N_STAGES    = 4    # DQC reinitialise stages swept per sigma — matches the interface's 4-stage limit

# Where per-dataset results/report/plots are written by default (one
# subfolder per CSV stem). Override via run_sweep(..., output_root=...).
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")

# Tolerance (absolute) for the "near-optimal region" reported alongside each metric's max.
NEAR_OPTIMAL_TOL = 0.02


# ─────────────────────────────────────────
# Reused verbatim from dqc_interface.py — DO NOT modify.
# These are the exact functions the interactive interface uses to run a
# stage, select t*, and score it. Copied rather than imported because
# dqc_interface.py calls st.set_page_config() at module import time and
# cannot be imported outside `streamlit run`.
# ─────────────────────────────────────────

def _pairwise_jaccard(true_labels, pred_labels):
    """Pairwise Jaccard index (Weinstein definition): TP / (TP + FP + FN)
    over all pairs of points, where TP = same true & same predicted cluster,
    FP = different true but same predicted, FN = same true but different
    predicted. O(n^2) — fine for n up to a few hundred."""
    tp = fp = fn = 0
    n = len(true_labels)
    for i, j in combinations(range(n), 2):
        same_true = true_labels[i] == true_labels[j]
        same_pred = pred_labels[i] == pred_labels[j]
        if same_true and same_pred:
            tp += 1
        elif not same_true and same_pred:
            fp += 1
        elif same_true and not same_pred:
            fn += 1
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def _select_t_star(t_vals, S_vals, prominence_frac=0.10):
    """Return (t_star, peak_indices). t_star = the HIGHEST entropy peak among
    all prominent peaks detected (not the first / earliest one). Falls back
    to the global argmax if no prominent peak is found."""
    rng = float(S_vals.max() - S_vals.min())
    if rng > 1e-4:
        peaks, props = find_peaks(S_vals, prominence=prominence_frac * rng)
    else:
        peaks = np.array([], dtype=int)
    if len(peaks) > 0:
        peaks = peaks[np.argsort(t_vals[peaks])]   # chronological, for plot markers
        best  = peaks[np.argmax(S_vals[peaks])]
        t_star = float(t_vals[best])
    else:
        t_star = float(t_vals[int(np.argmax(S_vals))])
    return t_star, peaks


def _compute_entropy_curve(model, t_vals):
    """Matrix-based reverse von Neumann entropy curve (entropy_values), computed
    in a single call — never loop over von_neumann_entropy(t)."""
    _, S_vals = model.get_trajectories_with_entropy(t_vals)
    return S_vals


def _compute_metrics(pos_end, cl, stage_num, k_cl, y_enc):
    """Compute silhouette + label-based metrics for one stage; return dict."""
    sil = (silhouette_score(pos_end, cl)
           if len(np.unique(cl)) > 1 else float('nan'))
    row = {'stage': stage_num, 'k': k_cl, 'silhouette': sil,
           'ARI': None, 'NMI': None, 'Jaccard': None}
    if y_enc is not None:
        row['ARI']     = adjusted_rand_score(y_enc, cl)
        row['NMI']     = normalized_mutual_info_score(y_enc, cl)
        row['Jaccard'] = _pairwise_jaccard(y_enc, cl)
    return row


def _compute_pca(X_raw, n_components):
    """Mean-centre -> SVD. Returns X_unit (basis for sphere-normalisation),
    X_scaled (real-unit PCA scores, used when sphere-normalise is off), and
    the explained variance ratio of each kept component (computed against
    the FULL singular-value spectrum, not just the kept components)."""
    Xc = X_raw - X_raw.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    n_components = min(n_components, U.shape[1])
    evr_full = (S ** 2) / np.sum(S ** 2)
    X_unit   = U[:, :n_components]
    X_scaled = U[:, :n_components] * S[:n_components]
    return X_unit, X_scaled, evr_full[:n_components]


# ─────────────────────────────────────────
# New code: dataset loading, bandwidth references, baseline, sweep,
# reporting, plotting.
# ─────────────────────────────────────────

def _find_label_column(df):
    """Auto-detect the ground-truth label column by name."""
    for candidate in ("true_label", "label", "y", "class", "target"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"No label column found (looked for true_label/label/y/class/target). "
        f"Columns present: {list(df.columns)}"
    )


def _load_csv_dataset(csv_path):
    """Load a CSV into (X_raw, y_enc, k_true, feature_cols). The label
    column is detected and excluded from X_raw BEFORE anything else (in
    particular, before PCA is ever fit) -- it is only used to build y_enc
    for scoring."""
    df = pd.read_csv(csv_path)
    label_col = _find_label_column(df)
    feature_cols = [c for c in df.columns if c != label_col]

    X_raw = df[feature_cols].values.astype(float)
    y_enc = LabelEncoder().fit_transform(df[label_col].values)
    k_true = int(len(np.unique(y_enc)))
    return X_raw, y_enc, k_true, feature_cols


def _build_working_coordinates(X_raw, n_components, sphere_normalise):
    """Reproduce dqc_interface.py's _apply_pca_and_work() exactly: PCA to
    min(n_components, n_features) components, then sphere-normalise or not
    per sphere_normalise."""
    n_components = min(n_components, X_raw.shape[1])
    X_unit, X_scaled, evr = _compute_pca(X_raw, n_components)
    if sphere_normalise:
        norms = np.linalg.norm(X_unit, axis=1, keepdims=True)
        X_work = X_unit / np.maximum(norms, 1e-12)
    else:
        X_work = X_scaled
    return X_work, evr


def _silverman_scott_bandwidth(X_work):
    """Isotropic Silverman/Scott normal-reference bandwidth, computed only
    from the spread of X_work -- label-blind, like any real unsupervised
    bandwidth choice would have to be.

    h = (4/(n*(d+2)))**(1/(d+4)) * s, s = sqrt(mean of per-column variances).
    """
    n, d = X_work.shape
    col_var = np.var(X_work, axis=0)
    s = float(np.sqrt(col_var.mean()))
    return (4.0 / (n * (d + 2))) ** (1.0 / (d + 4)) * s


def _lscv_bandwidth(X_work):
    """Least-Squares Cross-Validation bandwidth (statsmodels' own internal
    optimiser, not tied to labels either), collapsed to a single scalar
    (mean of the per-dimension bandwidths) so it's comparable to DQC's
    single isotropic sigma. Takes a few seconds per dataset -- LSCV runs
    its own optimisation over X_work, independent of everything else here."""
    d = X_work.shape[1]
    kde = KDEMultivariate(data=X_work, var_type='c' * d, bw='cv_ls')
    return float(np.mean(kde.bw))


def _kmeans_baseline(X_work, k_true, y_enc):
    """Plain k-means on the raw, un-evolved PCA coordinates (t=0, no DQC
    evolution at all) -- same k and same fixed seed as everywhere else in
    the pipeline. Tells us whether DQC's evolution step is adding anything
    over doing nothing."""
    km = KMeans(n_clusters=k_true, random_state=KMEANS_SEED, n_init=50, max_iter=500)
    cl = km.fit_predict(X_work)
    sil = silhouette_score(X_work, cl) if len(np.unique(cl)) > 1 else float('nan')
    return {
        'ari':        adjusted_rand_score(y_enc, cl),
        'nmi':        normalized_mutual_info_score(y_enc, cl),
        'jaccard':    _pairwise_jaccard(y_enc, cl),
        'silhouette': sil,
    }


def _build_sigma_grid(sigma_min, sigma_max, sigma_step):
    """Evenly-spaced sigma grid: sigma_min .. sigma_max inclusive in steps
    of sigma_step, with a half-step epsilon so floating-point step
    accumulation doesn't drop sigma_max itself."""
    return np.arange(sigma_min, sigma_max + sigma_step / 2.0, sigma_step)


def _run_one_sigma(X_work, sigma, k_true, y_enc, eig_threshold, m):
    """Run the exact stage workflow (dqc_interface.py's _run_stage /
    _reinitialise pattern) for N_STAGES stages at one fixed sigma.
    Returns a list of per-stage metric dicts."""
    # m: fixed value if given (matches the interface's Manual mass mode,
    # and what every prior manual/interface run in this project has used);
    # falls back to auto 1/sigma^2 (interface's Auto mode) only if m=None.
    m_val = m if m is not None else 1.0 / sigma ** 2
    t_vals = np.linspace(DELTA_T, T_MAX, N_STEPS)

    rows = []
    model = DQC(sigma=sigma, m=m_val, k=k_true, T=N_STEPS, delta_t=DELTA_T,
                eig_threshold=eig_threshold, preprocess='none')

    pos_end = None
    for stage in range(1, N_STAGES + 1):
        try:
            if stage == 1:
                model.fit(X_work)   # exact DQC evolution/fit, as coded
            else:
                model.reinitialise(pos_end)   # exact reinitialise-between-stages logic, as coded

            S_vals = _compute_entropy_curve(model, t_vals)
            t_star, _ = _select_t_star(t_vals, S_vals)
            pos_end = model.get_positions_at_t(t_star)
            cl = model.extract_clusters_kmeans(pos_end, k_true)   # exact k-means extraction, as coded (seed=42 inside dqc_core.py)

            metrics = _compute_metrics(pos_end, cl, stage, k_true, y_enc)
            metrics['sigma']  = sigma
            metrics['t_star'] = t_star
            rows.append(metrics)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # dqc_core.py raises ValueError when a stage's positions have
            # collapsed so tightly that fewer than 3 eigenvectors survive
            # eig_threshold (a real, expected failure mode of repeated
            # reinitialisation, not a bug). Record NaN for this and every
            # remaining stage at this sigma -- since reinitialise() needs
            # the previous stage's converged positions, we can't continue
            # past a broken stage -- and move on to the next sigma rather
            # than losing every other sigma's already-computed results.
            print(f"  [WARNING] sigma={sigma:.4f} stage={stage} failed ({exc}) "
                  f"-- recording NaN for stage {stage}-{N_STAGES} at this sigma and continuing.")
            for remaining_stage in range(stage, N_STAGES + 1):
                rows.append({
                    'stage': remaining_stage, 'k': k_true, 'silhouette': float('nan'),
                    'ARI': float('nan'), 'NMI': float('nan'), 'Jaccard': float('nan'),
                    'sigma': sigma, 't_star': float('nan'),
                })
            break

    return rows


def run_sweep(csv_path, sigma_min, sigma_max, sigma_step,
              n_components=3, sphere_normalise=False, eig_threshold=EIG_THRESH,
              m=0.20, output_root=None):
    """Run the full (sigma x stage) DQC sweep on ANY single CSV and save
    results/report/plots to output_root/<csv-stem>/.

    Parameters
    ----------
    csv_path         : path to a CSV with feature columns + a label column
                       (label auto-detected — see _find_label_column).
    sigma_min        : smallest sigma to sweep.
    sigma_max        : largest sigma to sweep (inclusive).
    sigma_step       : evenly-spaced step between sigma values.
    n_components     : number of PCA components to keep (default 3, matching
                       the interface's "Upload CSV" default).
    sphere_normalise : project PCA coordinates onto the unit sphere after
                       PCA (default False, matching the interface's
                       "Upload CSV" default of raw PCA scores).
    eig_threshold    : eigenvalue cutoff for truncating N's eigenvectors
                       (default 1e-5, the interface's default). Raise this
                       (e.g. 1e-2, 1e-1) on larger/higher-spread datasets
                       where the eigenvalue spectrum decays slowly and the
                       default keeps nearly all n eigenvectors -- that's
                       what makes a sweep slow. See dqc_interface.py's own
                       comment on the S&P 500 dataset for precedent: default
                       1e-5 kept r~n and took 35+ min unfinished per fit();
                       1e-1 got r down to ~1/3 of n and finished in ~14 min.
    m                : mass, held FIXED at this value for every sigma and
                       stage (default 0.20 -- matches the interface's Manual
                       mass mode, which every interface run validated in
                       this project has used). Pass m=None to instead
                       auto-track sigma via 1/sigma^2 (the interface's Auto
                       mode) -- NOT the default, since Manual/0.20 is what's
                       actually been used and cross-checked against the
                       interface so far.
    output_root      : where to write results/<csv-stem>/ (default: a
                       "results" folder next to this script).

    Returns (results_df, report_dict).
    """
    if output_root is None:
        output_root = RESULTS_ROOT

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    # run_tag identifies this specific run (dataset + sphere_normalise +
    # sigma range) -- used as both the output subfolder name and a filename
    # prefix on every saved file, so a second run on the same CSV with a
    # different sphere_normalise or sigma range doesn't silently overwrite
    # an earlier one. {:g} keeps the float formatting compact (0.01, 0.85,
    # 2 -- no trailing zeros).
    sphere_tag = "sphere_normalise=yes" if sphere_normalise else "sphere_normalise=no"
    range_tag = f"sigma{sigma_min:g}-{sigma_max:g}"
    run_tag = f"{stem}_{sphere_tag}_{range_tag}"
    out_dir = os.path.join(output_root, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}\n[{stem}] Loading {csv_path}")
    X_raw, y_enc, k_true, feature_cols = _load_csv_dataset(csv_path)
    X_work, evr = _build_working_coordinates(X_raw, n_components, sphere_normalise)
    n, d = X_work.shape
    m_desc = f"fixed at {m}" if m is not None else "auto (1/sigma^2, recomputed per sigma)"
    print(f"[{stem}] n={n}  raw_features={len(feature_cols)}  "
          f"working_dim={d} (PCA, {evr.sum() * 100:.1f}% variance, "
          f"sphere_normalise={sphere_normalise})  k_true={k_true}  "
          f"eig_threshold={eig_threshold:.0e}  m: {m_desc}")

    sigma_grid = _build_sigma_grid(sigma_min, sigma_max, sigma_step)
    h = _silverman_scott_bandwidth(X_work)
    h_in_range = sigma_min <= h <= sigma_max
    print(f"[{stem}] sigma grid: {len(sigma_grid)} values from "
          f"{sigma_grid[0]:.4f} to {sigma_grid[-1]:.4f} (step {sigma_step})")
    print(f"[{stem}] Silverman/Scott reference bandwidth h={h:.4f}  (inside range: {h_in_range})")

    h_lscv = _lscv_bandwidth(X_work)
    h_lscv_in_range = sigma_min <= h_lscv <= sigma_max
    print(f"[{stem}] LSCV reference bandwidth h_lscv={h_lscv:.4f}  (inside range: {h_lscv_in_range})")

    baseline = _kmeans_baseline(X_work, k_true, y_enc)
    print(f"[{stem}] k-means baseline (t=0, no DQC evolution): "
          f"ARI={baseline['ari']:.4f}  NMI={baseline['nmi']:.4f}  "
          f"Jaccard={baseline['jaccard']:.4f}  silhouette={baseline['silhouette']:.4f}")

    # Guarantee h and h_lscv are always actually swept -- not just drawn as
    # guideline positions -- even (especially) when they fall outside
    # [sigma_min, sigma_max]. Tag every sigma with where it came from so
    # the results table shows which rows are the regular grid vs the two
    # reference bandwidths.
    sigma_sources = [(float(s), 'grid') for s in sigma_grid]
    if not any(abs(s - h) < 1e-12 for s, _ in sigma_sources):
        sigma_sources.append((h, 'silverman'))
        print(f"[{stem}] adding sigma={h:.4f} to the sweep (Silverman/Scott reference"
              f"{' -- outside the chosen sigma range' if not h_in_range else ''})")
    if not any(abs(s - h_lscv) < 1e-12 for s, _ in sigma_sources):
        sigma_sources.append((h_lscv, 'lscv'))
        print(f"[{stem}] adding sigma={h_lscv:.4f} to the sweep (LSCV reference"
              f"{' -- outside the chosen sigma range' if not h_lscv_in_range else ''})")
    sigma_sources.sort(key=lambda t: t[0])

    all_rows = []
    t0 = time.time()
    for i, (sigma, source) in enumerate(sigma_sources, 1):
        stage_rows = _run_one_sigma(X_work, sigma, k_true, y_enc, eig_threshold, m)
        for row in stage_rows:
            row['sigma_source'] = source
        all_rows.extend(stage_rows)
        elapsed = time.time() - t0
        last = stage_rows[-1]
        print(f"[{stem}] sigma {i}/{len(sigma_sources)}={sigma:.4f} ({source})  "
              f"(final-stage ARI={last['ARI']:.3f} NMI={last['NMI']:.3f} "
              f"Jaccard={last['Jaccard']:.3f})  elapsed={elapsed:.1f}s")

    results_df = pd.DataFrame(all_rows)[
        ['sigma', 'sigma_source', 'stage', 'ARI', 'NMI', 'Jaccard', 'k', 'silhouette', 't_star']
    ].rename(columns={'ARI': 'ari', 'NMI': 'nmi', 'Jaccard': 'jaccard', 'k': 'k_used'})
    results_df.insert(0, '#', range(1, len(results_df) + 1))

    results_csv_path = os.path.join(out_dir, f"{run_tag}_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"[{stem}] Saved: {results_csv_path}")

    _print_and_save_highlighted_table(results_df, baseline, stem, run_tag, out_dir)

    report = _build_report(results_df, stem, run_tag, sigma_min, sigma_max, h, h_in_range,
                            h_lscv, baseline, out_dir)
    _make_plots(results_df, stem, run_tag, h, h_lscv, baseline, sigma_min, sigma_max, out_dir)

    return results_df, report


def _print_and_save_highlighted_table(results_df, baseline, stem, run_tag, out_dir):
    """Print the full numbered results table to console (with plain-text
    flags, since console output can't reliably render colour), and save a
    colour-highlighted version as an HTML file for a browser.

    Two independent things get flagged, on different visual channels so
    they can overlap on the same row without conflicting:
      - "beats k-means baseline" (row's ari/nmi/jaccard exceeds the
        corresponding baseline value in at least one metric): the '#' and
        'sigma' and 'stage' cells get a green background.
      - "this sigma IS the Silverman/Scott or LSCV reference bandwidth"
        (sigma_source column, already computed by run_sweep): the 'sigma'
        cell gets coloured bold text -- blue for Silverman, orange for
        LSCV, purple if a sigma coincides with both.
    """
    beats_baseline = (
        (results_df['ari'] > baseline['ari']) |
        (results_df['nmi'] > baseline['nmi']) |
        (results_df['jaccard'] > baseline['jaccard'])
    )

    # ── Plain-text console table (guaranteed to render regardless of
    # console colour support) ──────────────────────────────────────────
    flags = []
    for beats, source in zip(beats_baseline, results_df['sigma_source']):
        tags = []
        if beats:
            tags.append(">KMEANS")
        if source == 'silverman':
            tags.append("SILVERMAN")
        elif source == 'lscv':
            tags.append("LSCV")
        flags.append(",".join(tags))

    text_df = results_df.copy()
    text_df['flags'] = flags
    print(f"\nFull results table ({len(results_df)} rows) "
          f"-- '>KMEANS' = beats the k-means baseline on >=1 metric, "
          f"'SILVERMAN'/'LSCV' = this sigma is that reference bandwidth:")
    print(text_df.to_string(index=False))

    # ── Colour-highlighted HTML version ─────────────────────────────────
    def _row_style(row):
        styles = [''] * len(row)
        cols = list(row.index)
        if beats_baseline.loc[row.name]:
            for col in ('#', 'sigma', 'stage'):
                if col in cols:
                    styles[cols.index(col)] += 'background-color: #d4f7d4;'
        source = row.get('sigma_source', 'grid')
        if 'sigma' in cols:
            i = cols.index('sigma')
            if source == 'silverman':
                styles[i] += 'color: #0044cc; font-weight: bold;'
            elif source == 'lscv':
                styles[i] += 'color: #cc6600; font-weight: bold;'
            elif source not in ('grid',):
                styles[i] += 'color: #8800cc; font-weight: bold;'  # both / other
        return styles

    styler = (results_df.style
              .apply(_row_style, axis=1)
              .set_caption(
                  f"{run_tag} — full sweep results. Green background ('#'/sigma/stage) = "
                  f"beats k-means baseline on >=1 metric. Blue sigma text = Silverman/Scott "
                  f"reference bandwidth. Orange sigma text = LSCV reference bandwidth."))
    html_path = os.path.join(out_dir, f"{run_tag}_results_highlighted.html")
    styler.to_html(html_path)
    print(f"[{stem}] Saved colour-highlighted table: {html_path}")

    # Also render the coloured table inline (Spyder/Jupyter's IPython console
    # supports this rich HTML display; falls back to nothing extra if not).
    try:
        from IPython.display import display
        display(styler)
    except ImportError:
        pass


def _build_report(results_df, stem, run_tag, sigma_min, sigma_max, h, h_in_range,
                   h_lscv, baseline, out_dir):
    """Per-metric max value + near-optimal (sigma, stage) region; printed
    and saved as report.txt / near_optimal_region.csv."""
    lines = []
    lines.append(f"DQC sigma/stage sweep report — {run_tag}")
    lines.append(f"sigma range swept: [{sigma_min:.4f}, {sigma_max:.4f}]  (evenly spaced)")
    lines.append(f"Silverman/Scott reference bandwidth h = {h:.4f}")
    lines.append(f"h inside [sigma_min, sigma_max]: {h_in_range}")
    lines.append(f"LSCV reference bandwidth h_lscv = {h_lscv:.4f}")
    lines.append(f"k-means baseline (t=0, no DQC evolution): "
                 f"ARI={baseline['ari']:.4f}  NMI={baseline['nmi']:.4f}  "
                 f"Jaccard={baseline['jaccard']:.4f}  silhouette={baseline['silhouette']:.4f}")
    lines.append(f"near-optimal tolerance: {NEAR_OPTIMAL_TOL} (absolute)")
    lines.append("")

    near_optimal_rows = []
    metric_summary = {}
    for metric in ("ari", "nmi", "jaccard"):
        max_val = float(results_df[metric].max())
        near = results_df[np.abs(results_df[metric] - max_val) <= NEAR_OPTIMAL_TOL]
        near = near.sort_values(["sigma", "stage"])
        metric_summary[metric] = {
            'max_value': max_val,
            'near_optimal': near[['sigma', 'stage', metric]].to_dict('records'),
        }

        lines.append(f"[{metric.upper()}] max = {max_val:.4f}")
        lines.append(f"  near-optimal (within {NEAR_OPTIMAL_TOL}) region — "
                     f"{len(near)} (sigma, stage) combinations:")
        for _, r in near.iterrows():
            lines.append(f"    sigma={r['sigma']:.4f}  stage={int(r['stage'])}  {metric}={r[metric]:.4f}")
            near_optimal_rows.append({'metric': metric, 'max_value': max_val,
                                       'sigma': r['sigma'], 'stage': int(r['stage']),
                                       'value': r[metric]})
        lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    with open(os.path.join(out_dir, f"{run_tag}_report.txt"), "w") as f:
        f.write(report_text)
    pd.DataFrame(near_optimal_rows).to_csv(
        os.path.join(out_dir, f"{run_tag}_near_optimal_region.csv"), index=False)

    return {'h': h, 'sigma_min': sigma_min, 'sigma_max': sigma_max,
            'h_in_range': h_in_range, 'h_lscv': h_lscv, 'baseline': baseline,
            'metrics': metric_summary}


def _make_plots(results_df, stem, run_tag, h, h_lscv, baseline, sigma_min, sigma_max, out_dir):
    """Saves (dpi=300) 6 plots per dataset: ARI/NMI/Jaccard vs sigma (one
    line per stage, plus the Silverman/Scott line, the LSCV line, and the
    k-means-on-raw-data baseline line), and an ARI/NMI/Jaccard heatmap over
    the full (sigma, stage) grid."""
    stages = sorted(results_df['stage'].unique())
    sigmas = sorted(results_df['sigma'].unique())

    xlim = (min(sigma_min, h, h_lscv) - sigma_min * 0.2,
            max(sigma_max, h, h_lscv) + sigma_min * 0.2)

    for metric in ("ari", "nmi", "jaccard"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for stage in stages:
            sub = results_df[results_df['stage'] == stage].sort_values('sigma')
            ax.plot(sub['sigma'], sub[metric], marker='o', markersize=2.5,
                    linewidth=1.1, alpha=0.75, label=f"stage {stage}")
        ax.axvline(h, color='k', linestyle='--', linewidth=1.5,
                   label=f"Silverman/Scott h={h:.3f}")
        ax.axvline(h_lscv, color='tab:red', linestyle=':', linewidth=1.5,
                   label=f"LSCV h={h_lscv:.3f}")
        ax.axhline(baseline[metric], color='tab:green', linestyle='-.', linewidth=1.5,
                   label="k-means on raw data (no DQC)")
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('sigma')
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{run_tag} — {metric.upper()} vs sigma")
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{run_tag}_{metric}_vs_sigma.png"),
                    dpi=300, bbox_inches="tight")

    for metric in ("ari", "nmi", "jaccard"):
        Z = (results_df.pivot(index='stage', columns='sigma', values=metric)
             .reindex(index=stages, columns=sigmas).values)
        fig, ax = plt.subplots(figsize=(8, 4))
        mesh = ax.pcolormesh(sigmas, stages, Z, shading='nearest', cmap='viridis')
        ax.set_xlabel('sigma')
        ax.set_ylabel('stage')
        ax.set_yticks(stages)
        ax.set_title(f"{run_tag} — {metric.upper()} heatmap (sigma x stage)")
        fig.colorbar(mesh, ax=ax, label=metric.upper())
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{run_tag}_{metric}_heatmap.png"),
                    dpi=300, bbox_inches="tight")

    print(f"[{stem}] Saved 6 plots to {out_dir}")
    plt.show()  # display all 6 figures inline (Spyder Plots pane / Jupyter), in addition to the saved files
