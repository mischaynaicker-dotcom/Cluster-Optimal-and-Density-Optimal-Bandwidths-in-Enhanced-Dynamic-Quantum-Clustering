"""
DQC Streamlit Interface
Run with:  streamlit run dqc_interface.py
Generalised from the original Weinstein & Horn (arXiv:0908.2644) crab
replication to also support arbitrary uploaded CSVs and an S&P 500 preset.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.signal import find_peaks
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                              silhouette_score)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dqc_core import DQC
from dqc_analysis import _find_label_column  # reused as-is: same label-column detection used by run_sweep

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DQC Interface",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

N_ROWS_WARNING_THRESHOLD = 1000
N_ROWS_HARD_CAP = 5000
_SP500_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "sp500_2000_2011.csv")

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    'df': None, 'X_raw': None, 'X_sphere': None,
    'y_labels': None, 'feature_cols': None,
    'cluster_labels': None,
    't_star': 1.0,
    'entropy_t': None, 'entropy_s': None,  # cached entropy curve (current stage)
    'k_cl': 4,

    # Persistent parameter values (not owned by any widget key)
    'sigma': 0.10,
    'm_val': 100.0,

    # Stage iteration state
    'current_stage': 1,
    'stage_done': False,
    'current_model': None,        # DQC model for the *current* stage
    'stage_positions': [],        # list of (n,d) arrays — converged positions per stage
    'stage_trajectories': [],     # list of dicts: traj, pos0, pos_end, t_star
    'stage_t_star': [],           # list of t* used per stage
    'stage_silhouette': [],       # list of silhouette scores per stage
    'stage_metrics': [],          # list of dicts — full per-stage metric results

    # UI state
    'show_reset_confirm': False,

    # Dataset / geometry generalisation
    'dataset_choice': "Crabs (default)",
    'loaded_dataset_key': None,
    'sphere_normalise': True,
    'n_components': 3,
    'X_pcs_unit': None,           # U[:, :n_components]  (basis for sphere norm)
    'X_pcs_scaled': None,         # U[:, :n_components] * S[:n_components] (raw PCA scores)
    'evr': None,                  # explained variance ratio per kept component
    'pc_x_idx': 0, 'pc_y_idx': 1, 'pc_z_idx': 2,
    'eig_idx': 2,
    'subsampled_to': 0,
    'sp500_dates': None, 'sp500_ref': None, 'sp500_ref_is_approx': False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = (_v.copy() if isinstance(_v, list) else _v)
ss = st.session_state


# ── Data helpers ───────────────────────────────────────────────────────────────
@st.cache_data
def _load_crabs():
    """Load Ripley's Crab dataset; fallback to CSV download."""
    try:
        from statsmodels.datasets import get_rdataset
        df = get_rdataset("crabs", "MASS").data
    except Exception:
        try:
            import urllib.request
            url = ("https://raw.githubusercontent.com/vincentarelbundock/"
                   "Rdatasets/master/csv/MASS/crabs.csv")
            urllib.request.urlretrieve(url, "crabs.csv")
            df = pd.read_csv("crabs.csv")
        except Exception as exc:
            return None, str(exc)
    return df, None


def _load_crabs_dataset():
    df, err = _load_crabs()
    if df is None:
        st.error(f"Could not load crabs dataset: {err}")
        return False
    feat_cols = ["FL", "RW", "CL", "CW", "BD"]
    ss.df           = df
    ss.X_raw        = df[feat_cols].values.astype(float)
    ss.y_labels     = (df["sp"] + df["sex"]).values
    ss.feature_cols = feat_cols
    ss.sp500_dates  = None
    ss.sp500_ref    = None
    return True


def _load_sp500_dataset():
    if not os.path.exists(_SP500_CSV_PATH):
        st.error(
            "S&P 500 preset selected, but no bundled CSV was found at "
            f"`{_SP500_CSV_PATH}`.\n\n"
            "Expected format: first column named `Date` (trading days "
            "between roughly 2000-01-01 and 2011-02-24), followed by one "
            "column per stock — closing prices, each normalised so day-0 "
            "= 1.0. The paper's original file used 2803 rows × 440 stocks, "
            "but that exact file is not publicly available (only the "
            "paper's corresponding author would have it); a row/column "
            "count in the same order of magnitude, built via point-in-time "
            "survivor-filtered membership, is expected instead — see the "
            "Data & PCA tab for how the bundled reconstruction was built. "
            "An optional extra column named `SP500_Index` (the literal "
            "index level, not normalised) enables the date-aligned "
            "reference chart; without it, the mean of the normalised stock "
            "prices is used as an approximate stand-in."
        )
        return False
    df = pd.read_csv(_SP500_CSV_PATH)
    date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col])
    index_col = next((c for c in ("SP500_Index", "SP500", "Index")
                       if c in df.columns), None)
    stock_cols = [c for c in df.columns if c not in (date_col, index_col)]
    st.info(
        "**This is a faithful reconstruction**, not the original dataset "
        "from Weinstein & Horn's DQC paper. It was built from point-in-time "
        "S&P 500 membership history (fja05680/sp500 on GitHub) intersected "
        "across every trading day from 2000-01-01 to 2011-02-24 to get a "
        "survivor-filtered ticker set (analogous to the paper's 440 "
        "stocks), with daily adjusted-close prices pulled from a public "
        "market-data API and normalised to day-0 = 1.0. The original "
        "proprietary file is not publicly available — only the paper's "
        "corresponding author would have it — so **exact stock count and "
        "coverage differ from the paper** (this build has "
        f"{len(stock_cols)} stocks vs. the paper's 440, limited by which "
        "tickers a free historical-data source can still resolve for "
        "delisted/renamed companies)."
    )
    ss.df           = df
    ss.X_raw        = df[stock_cols].values.astype(float)
    ss.y_labels     = None
    ss.feature_cols = stock_cols
    ss.sp500_dates  = dates.values
    ss.sp500_ref    = (df[index_col].values if index_col is not None
                        else ss.X_raw.mean(axis=1))
    ss.sp500_ref_is_approx = (index_col is None)
    return True


def _detect_epochs(cluster_labels):
    """Contiguous same-cluster blocks of trading days. Returns a list of
    (start_idx, end_idx, cluster_id) tuples, inclusive indices."""
    epochs = []
    start = 0
    n = len(cluster_labels)
    for i in range(1, n + 1):
        if i == n or cluster_labels[i] != cluster_labels[start]:
            epochs.append((start, i - 1, cluster_labels[start]))
            start = i
    return epochs


def _render_sp500_epochs(dates, ref, cluster_labels, is_approx):
    """Fig-18-style epoch bar chart: index-level line with contiguous
    same-cluster ('market epoch') blocks drawn as coloured bars beneath it,
    plus a table of epoch date ranges/lengths."""
    dates = np.asarray(dates)
    ref   = np.asarray(ref, dtype=float)
    epochs = _detect_epochs(cluster_labels)

    _pal   = px.colors.qualitative.Safe
    uniq_c = sorted(np.unique(cluster_labels))
    _lc    = {c: _pal[i % len(_pal)] for i, c in enumerate(uniq_c)}

    st.markdown(f"**{len(epochs)} epochs** detected (contiguous same-cluster "
                f"blocks of trading days).")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=ref, mode='lines', line=dict(color='black', width=1),
        name='Index proxy' + (' (approx., mean of constituents)' if is_approx else ''),
    ))
    y_bar = float(ref.min()) - (float(ref.max()) - float(ref.min())) * 0.08
    for s, e, c in epochs:
        fig.add_trace(go.Scatter(
            x=[dates[s], dates[e]], y=[y_bar, y_bar],
            mode='lines', line=dict(color=_lc[c], width=10),
            showlegend=False, hoverinfo='skip',
        ))
    for c in uniq_c:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                  line=dict(color=_lc[c], width=10),
                                  name=f"Cluster {c} (epoch)"))
    fig.update_layout(
        title="Market epochs (contiguous DQC cluster blocks) vs S&P 500 index proxy",
        xaxis_title="Date", yaxis_title="Index level (proxy)", height=480,
    )
    st.plotly_chart(fig, use_container_width=True)
    if is_approx:
        st.caption("Index proxy = mean of the normalised constituent prices "
                   "(no literal `SP500_Index` column was supplied) — labelled "
                   "as a proxy, not the official S&P 500 index level.")

    epoch_df = pd.DataFrame({
        'Epoch':   list(range(1, len(epochs) + 1)),
        'Cluster': [c for _, _, c in epochs],
        'Start':   [pd.Timestamp(dates[s]).date() for s, _, _ in epochs],
        'End':     [pd.Timestamp(dates[e]).date() for _, e, _ in epochs],
        'Days':    [e - s + 1 for s, e, _ in epochs],
    })
    st.dataframe(epoch_df, use_container_width=True, height=300)
    return epochs


def _subsample(n_target):
    """Random subsample with a fixed seed, applied to every loaded array."""
    n = ss.X_raw.shape[0]
    if n_target <= 0 or n_target >= n:
        return
    rng = np.random.RandomState(42)
    idx = np.sort(rng.choice(n, size=n_target, replace=False))
    ss.df    = ss.df.iloc[idx].reset_index(drop=True)
    ss.X_raw = ss.X_raw[idx]
    if ss.y_labels is not None:
        ss.y_labels = np.asarray(ss.y_labels)[idx]
    if ss.sp500_dates is not None:
        ss.sp500_dates = np.asarray(ss.sp500_dates)[idx]
    if ss.sp500_ref is not None:
        ss.sp500_ref = np.asarray(ss.sp500_ref)[idx]


def _check_row_guardrail(n_points):
    """DQC's matrix ops scale poorly with n — warn, or hard-stop and ask
    the user to subsample, before any evolution is attempted."""
    if n_points > N_ROWS_HARD_CAP:
        st.error(f"Dataset has {n_points} rows, exceeding the hard cap of "
                 f"{N_ROWS_HARD_CAP}. DQC's matrix operations scale poorly "
                 f"beyond this size in this interface. Please subsample "
                 f"your data first.")
        st.stop()
    elif n_points > N_ROWS_WARNING_THRESHOLD:
        st.warning(f"Dataset has {n_points} rows. DQC computation may be "
                   f"slow (matrix operations scale roughly O(n²) to O(n³)). "
                   f"Consider subsampling if this is too slow.")


def _compute_pca(X_raw, n_components):
    """Mean-centre → SVD. Returns X_unit (basis for sphere-normalisation),
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


def _apply_pca_and_work():
    """(Re)compute ss.X_pcs_unit/scaled/evr from ss.X_raw using
    ss.n_components, then build ss.X_sphere — the working DQC coordinate
    matrix — per the ss.sphere_normalise toggle. ss.X_sphere is sphere-
    normalised when the toggle is on, or the raw PCA scores otherwise."""
    X_unit, X_scaled, evr = _compute_pca(ss.X_raw, ss.n_components)
    ss.X_pcs_unit   = X_unit
    ss.X_pcs_scaled = X_scaled
    ss.evr          = evr
    ss.n_components = X_unit.shape[1]
    if ss.sphere_normalise:
        norms = np.linalg.norm(X_unit, axis=1, keepdims=True)
        ss.X_sphere = X_unit / np.maximum(norms, 1e-12)
    else:
        ss.X_sphere = X_scaled


def _k_default():
    """Number of true classes if labels exist, else a neutral fallback."""
    if ss.y_labels is not None:
        return int(len(np.unique(ss.y_labels)))
    return 3


def _reset_stage_state():
    ss.cluster_labels      = None
    ss.t_star              = 1.0
    ss.entropy_t           = None
    ss.entropy_s           = None
    ss.current_stage       = 1
    ss.stage_done          = False
    ss.current_model       = None
    ss.stage_positions     = []
    ss.stage_trajectories  = []
    ss.stage_t_star        = []
    ss.stage_silhouette    = []
    ss.stage_metrics       = []
    ss.show_reset_confirm  = False


# ── Entropy / t* helper ─────────────────────────────────────────────────────────
def _compute_entropy_curve(model, t_vals):
    """Matrix-based reverse von Neumann entropy curve (entropy_values), computed
    in a single call — never loop over von_neumann_entropy(t)."""
    _, S_vals = model.get_trajectories_with_entropy(t_vals)
    return S_vals


def _pairwise_jaccard(true_labels, pred_labels):
    """Pairwise Jaccard index (Weinstein definition): TP / (TP + FP + FN)
    over all pairs of points, where TP = same true & same predicted cluster,
    FP = different true but same predicted, FN = same true but different
    predicted. O(n^2) — fine for n up to a few hundred."""
    from itertools import combinations
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


def _axis_limits(pos0, pad_frac=0.10):
    """Fixed axis ranges derived from the Stage-0 (t=0) positions only, with
    10% padding. Must be reused across every stage panel so that a collapsing
    cluster shrinks visibly toward the centre instead of being hidden by
    autoscaling."""
    lims = []
    for kk in range(pos0.shape[1]):
        lo, hi = float(pos0[:, kk].min()), float(pos0[:, kk].max())
        pad = (hi - lo) * pad_frac
        lims.append((lo - pad, hi + pad))
    return lims  # one (lo,hi) tuple per column of pos0


def _get_stage_view(s_idx, n_done, n_steps):
    """Return (traj, pos0, pos_end, t_star) for stage s_idx (0-based).

    The most recently completed stage (s_idx == n_done - 1) has NOT been
    reinitialised yet, so its model is still ss.current_model — recompute its
    trajectory/positions live from the current t* slider value (Bug A fix).
    Earlier stages were already reinitialised onto a new Hamiltonian, so their
    converged positions are frozen and must be read from the cached record.
    """
    if s_idx == n_done - 1 and ss.current_model is not None:
        info   = ss.stage_trajectories[s_idx]
        pos0   = info['pos0']
        t_star = ss.t_star
        n_traj = min(40, n_steps)
        t_traj = np.linspace(0.0, t_star, n_traj)
        traj    = ss.current_model.get_trajectories(t_traj)
        pos_end = ss.current_model.get_positions_at_t(t_star)
        return traj, pos0, pos_end, t_star
    info = ss.stage_trajectories[s_idx]
    return info['traj'], info['pos0'], info['pos_end'], info['t_star']


# ── Metrics helper ───────────────────────────────────────────────────────────────
def _compute_metrics(pos_end, cl, stage_num, k_cl):
    """Compute silhouette + label-based metrics for one stage; return dict."""
    sil = (silhouette_score(pos_end, cl)
           if len(np.unique(cl)) > 1 else float('nan'))
    row = {'stage': stage_num, 'k': k_cl, 'silhouette': sil,
           'ARI': None, 'NMI': None, 'Jaccard': None}
    if ss.y_labels is not None:
        y_enc = LabelEncoder().fit_transform(ss.y_labels)
        row['ARI']     = adjusted_rand_score(y_enc, cl)
        row['NMI']     = normalized_mutual_info_score(y_enc, cl)
        row['Jaccard'] = _pairwise_jaccard(y_enc, cl)
    return row


# ── Stage helpers ────────────────────────────────────────────────────────────────
def _run_stage(sigma, m_val, n_steps, delta_t, eig_thresh, T_max, k_cl):
    """Run the current stage: fit (stage 1) or use the reinitialised model
    (stage > 1), compute t*, trajectories, positions, and silhouette score."""
    stage = ss.current_stage

    if stage == 1:
        with st.spinner("Running Stage 1 …"):
            model = DQC(sigma=sigma, m=m_val, k=k_cl,
                        T=n_steps, delta_t=delta_t,
                        eig_threshold=eig_thresh,
                        preprocess='none')
            model.fit(ss.X_sphere)
            ss.current_model = model
            t_vals = np.linspace(delta_t, T_max, n_steps)
            S_vals = _compute_entropy_curve(model, t_vals)
            ss.entropy_t = t_vals
            ss.entropy_s = S_vals
            t_star, _ = _select_t_star(t_vals, S_vals)
            ss.t_star = t_star
            ss.t_star_default = t_star
            # Push the auto-selected t* into the slider/input widget state too.
            # Without this, the keyed widgets keep their previous value on the
            # next rerun and overwrite ss.t_star, so every stage silently reuses
            # stage 1's t* instead of its own (diverges from dqc_analysis sweep).
            st.session_state["t_star_slider"] = t_star
            st.session_state["t_star_input"]  = t_star
        pos0 = ss.X_sphere
    else:
        model = ss.current_model
        pos0  = ss.stage_positions[stage - 2]

    t_star = ss.t_star
    with st.spinner(f"Computing Stage {stage} trajectories …"):
        n_traj = min(40, n_steps)
        t_traj = np.linspace(0.0, t_star, n_traj)
        traj   = model.get_trajectories(t_traj)
        pos_end = model.get_positions_at_t(t_star)

    cl = model.extract_clusters_kmeans(pos_end, k_cl)
    metrics = _compute_metrics(pos_end, cl, stage, k_cl)

    idx = stage - 1
    _set_or_append(ss.stage_positions, idx, pos_end)
    _set_or_append(ss.stage_trajectories, idx,
                    {'traj': traj, 'pos0': pos0, 'pos_end': pos_end, 't_star': t_star})
    _set_or_append(ss.stage_t_star, idx, t_star)
    _set_or_append(ss.stage_silhouette, idx, metrics['silhouette'])
    _set_or_append(ss.stage_metrics, idx, metrics)

    ss.cluster_labels = cl
    ss.stage_done = True


def _set_or_append(lst, idx, val):
    if idx < len(lst):
        lst[idx] = val
    else:
        lst.append(val)


def _reinitialise(sigma, m_val, n_steps, delta_t, eig_thresh, T_max):
    """Rebuild the Hamiltonian from the current stage's converged positions,
    recompute the entropy curve and t* for the next stage."""
    new_positions = ss.stage_positions[ss.current_stage - 1]
    model = ss.current_model
    with st.spinner("Reinitialising …"):
        model.reinitialise(new_positions)
        t_vals = np.linspace(delta_t, T_max, n_steps)
        S_vals = _compute_entropy_curve(model, t_vals)
        ss.entropy_t = t_vals
        ss.entropy_s = S_vals
        t_star, _ = _select_t_star(t_vals, S_vals)
        ss.t_star = t_star
        ss.t_star_default = t_star
        # Same fix as in _run_stage: keep the keyed t* widgets in sync with the
        # newly auto-selected value, otherwise the stale widget state wins on
        # the next rerun and the new stage inherits the previous stage's t*.
        st.session_state["t_star_slider"] = t_star
        st.session_state["t_star_input"]  = t_star

    ss.current_stage += 1
    ss.stage_done = False


# ── Shared flat-grid potential renderer (used when sphere-normalise is OFF,
# and as the fixed illustrative comparison when it's ON) ───────────────────
def _flat_render(Xsrc, sig, fixed_idx, labels, pal, uniq):
    """2D contour + 3D landscape over two chosen dims of Xsrc, holding any
    remaining dims at their mean. Colour AND z-axis range on the 3D surface
    are clipped to the 5th-95th percentile of V so a sparse-corner spike
    doesn't flatten the data-relevant region — V itself is left untouched."""
    ix, iy = fixed_idx

    @st.cache_data
    def _compute(Xsrc_bytes, n_cols, sig, ix, iy):
        Xs_ = np.frombuffer(Xsrc_bytes, dtype=np.float64).reshape(-1, n_cols)
        d = Xs_.shape[1]
        sig2 = sig ** 2
        two_s = 2.0 * sig2

        def _V(pts):
            V = np.empty(len(pts))
            bsz = 256
            for s in range(0, len(pts), bsz):
                e = min(s + bsz, len(pts))
                gp = pts[s:e]
                db = np.sum((gp[:, None, :] - Xs_[None, :, :]) ** 2, axis=-1)
                eb = np.exp(-db / two_s)
                psb = np.maximum(eb.sum(axis=1), 1e-300)
                V[s:e] = ((db / sig2 - d) * eb).sum(axis=1) / (2.0 * psb)
            return V

        means = Xs_.mean(axis=0)
        x_min, x_max = Xs_[:, ix].min(), Xs_[:, ix].max()
        y_min, y_max = Xs_[:, iy].min(), Xs_[:, iy].max()
        padx = (x_max - x_min) * 0.1
        pady = (y_max - y_min) * 0.1
        gx = np.linspace(x_min - padx, x_max + padx, 100)
        gy = np.linspace(y_min - pady, y_max + pady, 100)
        G0, G1 = np.meshgrid(gx, gy)
        pts_grid = np.tile(means, (G0.size, 1))
        pts_grid[:, ix] = G0.ravel()
        pts_grid[:, iy] = G1.ravel()
        V_grid = _V(pts_grid).reshape(G0.shape)
        V_pts = _V(Xs_)
        v_min = min(V_grid.min(), V_pts.min())
        V_grid -= v_min
        V_pts  -= v_min
        return gx, gy, V_grid, V_pts

    n_cols = Xsrc.shape[1]
    gx, gy, V_grid, V_pts = _compute(Xsrc.tobytes(), n_cols, sig, ix, iy)

    xlab = f"PC{ix + 1}"
    ylab = f"PC{iy + 1}"

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(data=go.Contour(
            x=gx, y=gy, z=V_grid,
            colorscale='Viridis', colorbar=dict(title='V(x)'),
            contours=dict(showlabels=False),
        ))
        for lbl in uniq:
            idx = [i for i, l in enumerate(labels) if l == lbl]
            fig.add_trace(go.Scatter(
                x=Xsrc[idx, ix], y=Xsrc[idx, iy], mode='markers', name=lbl,
                marker=dict(size=6, color=pal[lbl],
                            line=dict(width=0.5, color='black')),
            ))
        fig.update_layout(title=f"V({xlab}, {ylab}) contour",
                           xaxis_title=xlab, yaxis_title=ylab, height=460)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        _cmin = float(np.percentile(V_grid, 5))
        _cmax = float(np.percentile(V_grid, 95))
        fig2 = go.Figure(data=go.Surface(
            x=gx, y=gy, z=V_grid,
            colorscale=[[0, '#e8eaf0'], [0.5, '#9aa5c4'], [1, '#3d4a7a']],
            opacity=0.85, showscale=False,
            cmin=_cmin, cmax=_cmax,
            contours=dict(z=dict(show=True, usecolormap=False,
                                  color='white', width=1)),
        ))
        for lbl in uniq:
            idx = [i for i, l in enumerate(labels) if l == lbl]
            fig2.add_trace(go.Scatter3d(
                x=Xsrc[idx, ix], y=Xsrc[idx, iy], z=V_pts[idx],
                mode='markers', name=lbl,
                marker=dict(size=3, color=pal[lbl],
                            line=dict(width=0.3, color='black')),
            ))
        fig2.update_layout(
            title=f"V({xlab}, {ylab}) landscape",
            scene=dict(xaxis_title=xlab, yaxis_title=ylab,
                       zaxis=dict(title="V(x)",
                                  range=[float(V_grid.min()), _cmax])),
            height=460,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚛️ DQC Interface")

    # ── Dataset selector ──────────────────────────────────────────────────────
    st.subheader("Dataset")
    dataset_choice = st.radio(
        "Dataset",
        ["Crabs (default)", "Upload CSV", "S&P 500 (2000-2011 epochs)"],
        key="dataset_choice",
    )
    if dataset_choice.startswith("S&P 500"):
        st.caption(
            "⚠️ Faithful reconstruction, not the original paper dataset — "
            "built from public point-in-time membership + price data. "
            "Exact stock count/dates differ from Weinstein & Horn's "
            "original 440-stock file (see Data & PCA tab for details)."
        )

    uploaded, df_up, feat_sel, lbl_sel = None, None, None, "(none)"
    if dataset_choice == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            st.caption("Preview (first 10 rows):")
            st.dataframe(df_up.head(10), use_container_width=True)
            num_cols = df_up.select_dtypes(include=np.number).columns.tolist()
            feat_sel = st.multiselect("Feature columns", num_cols,
                                       default=num_cols[:min(5, len(num_cols))])
            lbl_sel = st.selectbox("Label column (optional)",
                                    ["(none)"] + df_up.columns.tolist())

    if dataset_choice == "Upload CSV" and uploaded is not None and feat_sel:
        _data_key = ("csv", uploaded.name, tuple(feat_sel), lbl_sel)
    elif dataset_choice == "Upload CSV":
        _data_key = ("csv", None)
    else:
        _data_key = (dataset_choice,)

    if ss.loaded_dataset_key != _data_key:
        _reset_stage_state()
        ok = True
        if dataset_choice == "Crabs (default)":
            ok = _load_crabs_dataset()
            _default_sphere, _default_ncomp, _default_eig_idx = True, 3, 2
        elif dataset_choice == "Upload CSV":
            if uploaded is not None and feat_sel:
                # Auto-detect a ground-truth label/target column (same
                # detection used by dqc_analysis.py's run_sweep) and make sure
                # it never ends up inside DQC's input features, even if it's
                # ticked in "Feature columns" -- it must only ever be used
                # for scoring (ARI/NMI/Jaccard), never as an input coordinate
                # fed into PCA/DQC. If no label column is detected (e.g. a
                # generic CSV with no obvious label/target name), nothing
                # here changes anything.
                try:
                    detected_label_col = _find_label_column(df_up)
                except ValueError:
                    detected_label_col = None

                feat_sel_effective = feat_sel
                if detected_label_col is not None and detected_label_col in feat_sel:
                    feat_sel_effective = [c for c in feat_sel if c != detected_label_col]
                    st.warning(
                        f"Detected label column \"{detected_label_col}\" was ticked in "
                        f"\"Feature columns\" — it has been excluded from the PCA/DQC "
                        f"input features and is only used for scoring metrics "
                        f"(ARI/NMI/Jaccard), not for clustering itself."
                    )

                X_up = df_up[feat_sel_effective].dropna().values.astype(float)
                y_up = (df_up[lbl_sel].values if lbl_sel != "(none)" else None)
                ss.df, ss.X_raw, ss.y_labels = df_up, X_up, y_up
                ss.feature_cols = feat_sel_effective
                ss.sp500_dates, ss.sp500_ref = None, None
            else:
                ss.df, ss.X_raw, ss.X_sphere = None, None, None
                ok = False
            _default_sphere, _default_ncomp, _default_eig_idx = False, 3, 2
        else:  # S&P 500
            ok = _load_sp500_dataset()
            # n=2804 needs aggressive eigenvector truncation to be usable —
            # eig_threshold=1e-5 (the crabs/small-data default) keeps ~100%
            # of eigenvectors here (r=n), making the evolution loop O(n^3);
            # 1e-1 (r~937/2804) was empirically the first threshold that
            # brought fit() under ~15 min on this dataset (see project notes
            # / chat history — measured 814s vs 35+ min unfinished at 1e-5).
            _default_sphere, _default_ncomp, _default_eig_idx = True, 10, 6

        if ok:
            ss.loaded_dataset_key = _data_key
            ss.sphere_normalise   = _default_sphere
            ss.n_components       = min(_default_ncomp, ss.X_raw.shape[1])
            ss.k_cl                = _k_default()
            ss.eig_idx             = _default_eig_idx
            ss.pc_x_idx, ss.pc_y_idx, ss.pc_z_idx = (
                0, min(1, ss.n_components - 1), min(2, ss.n_components - 1))

    # ── Sampling / performance guardrail ────────────────────────────────────
    if ss.X_raw is not None:
        st.divider()
        st.subheader("Sampling")
        n_total = ss.X_raw.shape[0]
        subsample_n = int(st.number_input(
            "Subsample to N rows (0 = no subsampling)",
            min_value=0, max_value=n_total, value=0, step=100,
            help="Randomly samples N rows from the loaded data before "
                 "running anything. Set to 0 to use all rows. Use this "
                 "when computation is slow — for DQC, matrix operations "
                 "scale roughly O(n²) to O(n³) with row count, so halving "
                 "rows cuts time by 4–8x.",
        ))
        if subsample_n > 0 and ss.subsampled_to != subsample_n:
            _subsample(subsample_n)
            ss.subsampled_to = subsample_n
            ss.k_cl = _k_default()
            _reset_stage_state()

        n_points = ss.X_raw.shape[0]
        st.caption(f"Rows currently loaded: {n_points}")
        _check_row_guardrail(n_points)

    # ── Geometry: sphere toggle, component count, PC selectors ─────────────
    if ss.X_raw is not None:
        st.divider()
        st.subheader("Geometry")
        st.checkbox(
            "Sphere-normalise after PCA (recommended for crabs / "
            "Weinstein replication)",
            key="sphere_normalise",
            help="When ON: after PCA, each point is projected onto the unit "
                 "sphere (divided by its own norm). This is Weinstein's "
                 "preprocessing for the crab and S&P 500 cases. When OFF: "
                 "raw PCA scores (U×S) are used directly as working "
                 "coordinates. DQC runs identically either way — only the "
                 "coordinate space changes.",
        )
        max_ncomp = min(10, ss.X_raw.shape[1])
        ss.n_components = int(st.number_input(
            "PCA components to keep", min_value=2, max_value=max_ncomp,
            value=min(ss.n_components, max_ncomp), step=1,
        ))
        _apply_pca_and_work()

        _pc_labels = [f"PC{i + 1} ({ss.evr[i] * 100:.1f}% variance)"
                      for i in range(ss.n_components)]
        ss.pc_x_idx = st.selectbox(
            "X-axis PC", list(range(ss.n_components)),
            index=min(ss.pc_x_idx, ss.n_components - 1),
            format_func=lambda i: _pc_labels[i], key="pc_x_sel",
        )
        ss.pc_y_idx = st.selectbox(
            "Y-axis PC", list(range(ss.n_components)),
            index=min(ss.pc_y_idx, ss.n_components - 1),
            format_func=lambda i: _pc_labels[i], key="pc_y_sel",
        )
        if ss.n_components >= 3:
            ss.pc_z_idx = st.selectbox(
                "Z-axis PC", list(range(ss.n_components)),
                index=min(ss.pc_z_idx, ss.n_components - 1),
                format_func=lambda i: _pc_labels[i], key="pc_z_sel",
            )
        else:
            ss.pc_z_idx = ss.pc_y_idx
        _cum_var = (ss.evr[ss.pc_x_idx] + ss.evr[ss.pc_y_idx]
                    + ss.evr[ss.pc_z_idx]) * 100
        st.caption(f"Cumulative variance of selected PCs: {_cum_var:.1f}%")
    else:
        st.info("Load data to configure geometry.")

    # ── Parameters ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Parameters")

    sig_mode = st.radio("σ", ["Auto (0.10)", "Manual"], horizontal=True)
    if sig_mode == "Manual":
        _sig_max = float(st.number_input(
            "σ max (slider upper bound)", min_value=0.01, value=2.0,
            step=0.5, format="%.2f",
            help="Sets the right end of the σ slider. Increase if you need "
                 "values above the current maximum.",
        ))
        _sig_min = 0.0
        ss.sigma = float(np.clip(ss.sigma, _sig_min, _sig_max))
        sigma = st.slider("σ value", _sig_min, _sig_max,
                           value=ss.sigma, step=0.01,
                           key="sigma_slider")
        ss.sigma = sigma
    else:
        sigma = 0.10
        ss.sigma = sigma
        st.caption(f"Auto σ = {sigma:.4f}")

    m_mode = st.radio("m (mass)", ["Auto (1/σ²)", "Manual"], horizontal=True)
    if m_mode == "Manual":
        _m_min, _m_max = 0.01, 2.00
        ss.m_val = float(np.clip(ss.m_val, _m_min, _m_max))
        m_val = st.slider("m value", _m_min, _m_max,
                           value=ss.m_val, step=0.01,
                           key="m_slider")
        ss.m_val = m_val
    else:
        m_val = 1.0 / sigma ** 2
        ss.m_val = m_val
        st.caption(f"Auto m = {m_val:.4f}")

    # 0.99 is included because the Filament sweep in `results main project.py`
    # uses eig_threshold=0.99; without it that run cannot be reproduced here.
    _eig_opts  = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 0.99, 1.0]
    _eig_idx   = st.select_slider(
        "Eigenvalue threshold",
        options=list(range(len(_eig_opts))),
        # ".0e" would round both 0.99 and 1.0 to "1e+00" — show the large
        # values as plain decimals so they stay distinguishable on the slider.
        format_func=lambda i: (f"{_eig_opts[i]:g}" if _eig_opts[i] >= 0.1
                               else f"{_eig_opts[i]:.0e}"),
        help="Higher values discard more eigenvectors, making computation "
             "faster at some cost to accuracy. DQC does not require high "
             "accuracy — for large datasets (e.g. S&P 500) try 1e-3 or "
             "higher to reduce computation time significantly.",
        key="eig_idx",
    )
    eig_thresh = _eig_opts[_eig_idx]

    T_max   = st.number_input(
        "Max time T", value=5.0, min_value=0.1, step=0.5,
        help="Upper boundary of the DQC time window. Evolution is "
             "pre-computed from t=0 to this value in one run. Increase if "
             "the entropy curve has no peaks before the current limit. For "
             "large datasets start small (e.g. 3) for speed.",
    )
    n_steps = int(st.number_input(
        "Time steps", value=200, min_value=10, step=10,
        help="Number of evaluation points between t=0 and Max T. The full "
             "trajectory is pre-computed once — the slider then scrubs "
             "through this cached result instantly without re-running. More "
             "steps give a smoother entropy curve but increase computation "
             "time linearly.",
    ))
    delta_t = T_max / n_steps
    st.caption("These control time resolution only — t* is always chosen "
               "automatically from the entropy peaks.")

    # ── Stage control ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Iteration")

    stage = ss.current_stage
    st.markdown(f"**Current stage: {stage} / 4**")

    if ss.X_sphere is None:
        st.warning("Load data first.")
    elif stage > 4:
        st.info("Maximum stages reached.")
    elif not ss.stage_done:
        if st.button(f"▶ Run Stage {stage}", use_container_width=True, type="primary"):
            try:
                _run_stage(sigma, m_val, n_steps, delta_t, eig_thresh, T_max, ss.k_cl)
                st.success(f"Stage {stage} done — t* = {ss.t_star:.3f}")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    else:
        if stage < 4:
            if st.button("⏹ Stop & Reinitialise", use_container_width=True):
                # Must be guarded exactly like _run_stage above: dqc_core raises
                # ValueError when the contracted cloud leaves <3 eigenvectors
                # above eig_threshold. Uncaught, it halts the whole script run,
                # so every widget below here (including "Reset algorithm") never
                # renders and the app becomes unusable.
                try:
                    _reinitialise(sigma, m_val, n_steps, delta_t, eig_thresh, T_max)
                    st.success(f"Reinitialised — new t* = {ss.t_star:.3f}")
                    st.rerun()
                except (ValueError, np.linalg.LinAlgError) as exc:
                    st.error(
                        f"Stage {stage + 1} cannot be computed at this "
                        f"σ — {exc}"
                    )
                    st.info(
                        f"This matches the sweep, which records NaN for "
                        f"stages {stage + 1}–4 at this σ. Keep the "
                        f"{stage} stage(s) already computed; they are valid."
                    )
            st.button(f"▶ Run Stage {stage + 1}", use_container_width=True,
                      type="primary", disabled=True,
                      help="Click 'Stop & Reinitialise' first.")
        else:
            st.info("Maximum stages reached.")

    # ── Reset algorithm ───────────────────────────────────────────────────────
    if ss.stage_trajectories:
        st.divider()
        if not ss.show_reset_confirm:
            if st.button("🔄 Reset algorithm", use_container_width=True):
                ss.show_reset_confirm = True
                st.rerun()
        else:
            st.warning("This will clear all stage results. "
                       "Parameters and loaded data are kept.")
            _rc1, _rc2 = st.columns(2)
            with _rc1:
                if st.button("Confirm reset", use_container_width=True,
                              type="primary"):
                    _reset_stage_state()
                    st.rerun()
            with _rc2:
                if st.button("Cancel", use_container_width=True):
                    ss.show_reset_confirm = False
                    st.rerun()

    # ── Cluster extraction ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Cluster Extraction")
    k_cl = int(st.number_input("k (clusters)", value=ss.k_cl, min_value=2, step=1,
                                key="k_cl"))
    if ss.y_labels is None and ss.X_raw is not None:
        if ss.dataset_choice.startswith("S&P 500"):
            st.caption("No ground-truth labels exist for this dataset — k "
                       "is a starting guess. The expected result is "
                       "roughly a dozen or more temporal epochs; increase "
                       "k and re-run extraction to explore finer epoch "
                       "structure.")
        else:
            st.caption("No label column was provided — k defaults to 3 "
                       "and only silhouette-based metrics will be "
                       "available.")

    if st.button("Extract clusters (k-means)", use_container_width=True):
        if ss.current_model is None:
            st.warning("Run Stage 1 first.")
        else:
            pos = ss.current_model.get_positions_at_t(ss.t_star)
            cl_new = ss.current_model.extract_clusters_kmeans(pos, k_cl)
            ss.cluster_labels = cl_new
            # Update the current stage's metrics row in place
            n_done = len(ss.stage_trajectories)
            if n_done > 0:
                updated = _compute_metrics(pos, cl_new, n_done, k_cl)
                _set_or_append(ss.stage_metrics, n_done - 1, updated)
                _set_or_append(ss.stage_silhouette, n_done - 1, updated['silhouette'])
                # Refresh the stored positions/trajectory for this stage too.
                # Without this, a manually chosen t* updates the metrics and the
                # plot but NOT ss.stage_positions, so the next stage's
                # reinitialise() would branch from the old t*'s positions and
                # the chain would silently diverge.
                _set_or_append(ss.stage_positions, n_done - 1, pos)
                _set_or_append(ss.stage_t_star, n_done - 1, ss.t_star)
                _rec = ss.stage_trajectories[n_done - 1]
                _n_traj = min(40, len(ss.entropy_t)) if ss.entropy_t is not None else 40
                _rec['traj']    = ss.current_model.get_trajectories(
                    np.linspace(0.0, ss.t_star, _n_traj))
                _rec['pos_end'] = pos
                _rec['t_star']  = ss.t_star
            st.success(f"Extracted {k_cl} clusters — metrics updated.")

    if ss.cluster_labels is not None and ss.X_raw is not None:
        df_out = pd.DataFrame(ss.X_raw, columns=ss.feature_cols)
        pc_cols = [f"PC{i + 1}" for i in range(ss.X_sphere.shape[1])]
        df_out[pc_cols] = ss.X_sphere
        df_out['dqc_cluster'] = ss.cluster_labels
        if ss.y_labels is not None:
            df_out['true_label'] = ss.y_labels
        st.download_button(
            "⬇ Download CSV",
            df_out.to_csv(index=False).encode(),
            "dqc_results.csv", "text/csv",
            use_container_width=True,
        )

    # ── Guidelines ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Guidelines")
    st.markdown("""
**σ — potential sharpness**
- Too small → over-clusters / noisy
- Too large → under-clusters / merges
- Start: Auto (0.10) or **0.07** for crabs

**m — dynamics**
- = 1/σ² → ground state, minimal evolution
- < 1/σ² → tunnelling, joins nearby minima
- Start: **0.2** for crabs

**Eigenvalue threshold**
- Raise if computation is slow (fewer states)
- Lower if error "too few eigenvectors"
- Default 1e-5 works for most datasets

**t* — optimal stop time**
- Always the HIGHEST prominent entropy peak
- Earlier → less convergence
- Later → particles scatter past minima
- Auto-selection was tuned on crabs — sanity-check the entropy curve
  shape for other datasets (see Entropy tab)

**Multi-stage iteration**
- Run a stage → inspect → Stop & Reinitialise
- Up to 4 stages; silhouette guides the decision
- Weinstein reports Jaccard = 0.762 on crabs
    """)


# ── Main panel — 5 tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data & PCA",
    "🌊 Potential",
    "📈 Entropy & t*",
    "🔄 Trajectories",
    "🎯 Clustering",
])

# ── TAB 1 — Data & PCA ─────────────────────────────────────────────────────────
with tab1:
    if ss.df is None:
        st.info("Load data first (see sidebar).")
    else:
        if ss.dataset_choice.startswith("S&P 500"):
            st.warning(
                "**This S&P 500 dataset is a faithful reconstruction, not "
                "the original file from Weinstein & Horn's DQC paper.** "
                "The paper's original proprietary dataset is not publicly "
                "available — only the paper's corresponding author would "
                "have it. This version was built by: (1) taking point-in-"
                "time S&P 500 membership history and intersecting it across "
                "every trading day in the window to get a survivor-filtered "
                "ticker set, analogous to the paper's methodology for "
                "arriving at 440 stocks, and (2) pulling daily adjusted-"
                "close prices for that ticker set from a public market-data "
                "source, normalised to day-0 = 1.0. **Exact stock count and "
                f"date coverage differ from the paper** — this build has "
                f"{ss.df.shape[1] - 1} stocks over {ss.df.shape[0]} trading "
                "days, versus the paper's 440 stocks over 2803 days, "
                "limited by which delisted/renamed tickers a free "
                "historical-data source can still resolve."
            )
        st.subheader("Raw Data Preview")
        st.dataframe(ss.df.head(10), use_container_width=True)

        c_info, c_bar = st.columns([1, 2])
        with c_info:
            st.caption(f"Shape: {ss.df.shape[0]} × {ss.df.shape[1]}")
            if ss.y_labels is not None:
                uniq, cnts = np.unique(ss.y_labels, return_counts=True)
                st.caption("Class distribution:")
                st.bar_chart(pd.Series(cnts, index=uniq))

        ix, iy, iz = ss.pc_x_idx, ss.pc_y_idx, ss.pc_z_idx
        _geom = "sphere-normalised" if ss.sphere_normalise else "raw PCA scores"
        st.subheader(f"PCA — {_geom}")
        st.caption(f"Cumulative variance of PC{ix+1}/PC{iy+1}/PC{iz+1}: "
                   f"{(ss.evr[ix] + ss.evr[iy] + ss.evr[iz]) * 100:.1f}%")
        Xs = ss.X_sphere
        lstr = ([str(l) for l in ss.y_labels]
                if ss.y_labels is not None
                else ["?"] * len(Xs))

        c2d, c3d = st.columns(2)
        with c2d:
            fig2 = px.scatter(x=Xs[:, ix], y=Xs[:, iy], color=lstr,
                              labels={"x": f"PC{ix+1}", "y": f"PC{iy+1}",
                                      "color": "Class"},
                              title=f"PC{ix+1} vs PC{iy+1} ({_geom})")
            fig2.update_traces(marker=dict(size=5))
            st.plotly_chart(fig2, use_container_width=True)

        with c3d:
            fig3 = px.scatter_3d(x=Xs[:, ix], y=Xs[:, iy], z=Xs[:, iz],
                                  color=lstr,
                                  labels={"x": f"PC{ix+1}", "y": f"PC{iy+1}",
                                          "z": f"PC{iz+1}", "color": "Class"},
                                  title=f"3D PCA ({_geom})")
            fig3.update_traces(marker=dict(size=3))
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("PCA Variance Breakdown")
        _evr = ss.evr
        _cumulative = np.cumsum(_evr)
        _var_df = pd.DataFrame({
            'Component': [f"PC{i+1}" for i in range(len(_evr))],
            'Individual (%)': [f"{v*100:.2f}" for v in _evr],
            'Cumulative (%)': [f"{c*100:.2f}" for c in _cumulative],
        })
        _var_df_display = pd.concat([
            _var_df,
            pd.DataFrame({'Component': ['Total (kept)'],
                          'Individual (%)': [f"{_evr.sum()*100:.2f}"],
                          'Cumulative (%)': [f"{_cumulative[-1]*100:.2f}"]}),
        ], ignore_index=True)
        _vcol1, _vcol2 = st.columns([1, 2])
        with _vcol1:
            st.dataframe(_var_df_display.set_index('Component'), use_container_width=True)
        with _vcol2:
            _fig_var = go.Figure(go.Bar(
                x=[f"PC{i+1} ({v*100:.1f}%)" for i, v in enumerate(_evr)],
                y=[v*100 for v in _evr],
                marker_color=px.colors.qualitative.Safe[:len(_evr)],
            ))
            _fig_var.add_trace(go.Scatter(
                x=[f"PC{i+1} ({v*100:.1f}%)" for i, v in enumerate(_evr)],
                y=[c*100 for c in _cumulative],
                mode='lines+markers', name='Cumulative',
                line=dict(color='black', width=1.5), yaxis='y2',
            ))
            _fig_var.update_layout(
                title="Explained variance per kept PC",
                xaxis_title="Component", yaxis_title="Individual (%)",
                yaxis2=dict(title="Cumulative (%)", overlaying='y', side='right',
                            range=[0, 100]),
                height=320, showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )
            st.plotly_chart(_fig_var, use_container_width=True)


# ── TAB 2 — Potential ──────────────────────────────────────────────────────────
with tab2:
    if ss.X_sphere is None:
        st.info("Load data first.")
    else:
        Xw = ss.X_sphere
        ix, iy = ss.pc_x_idx, ss.pc_y_idx
        _geom = "sphere-normalised" if ss.sphere_normalise else "raw PCA scores"
        st.subheader(f"Quantum Potential V(x)   σ = {sigma:.4f}   ({_geom})")

        if ss.y_labels is not None:
            _col_lbl = [str(l) for l in ss.y_labels]
        elif ss.cluster_labels is not None:
            _col_lbl = [str(l) for l in ss.cluster_labels]
        else:
            _col_lbl = ["?"] * len(Xw)
        _pal_v  = px.colors.qualitative.Safe
        _uniq_v = list(dict.fromkeys(_col_lbl))
        _lc_v   = {l: _pal_v[i % len(_pal_v)] for i, l in enumerate(_uniq_v)}

        _flat_render(Xw, sigma, fixed_idx=(ix, iy), labels=_col_lbl,
                     pal=_lc_v, uniq=_uniq_v)


# ── TAB 3 — Entropy & t* ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Reverse von Neumann Entropy  &  t* Selection")
    st.caption("Auto-selected t* uses the highest entropy peak. This "
               "default was tuned against the crabs dataset's entropy "
               "curve shape — for other datasets, visually check that the "
               "curve has a clear, distinguishable peak before trusting "
               "the auto-selection. A flat, noisy, or multi-modal curve "
               "may need a manually chosen t*.")
    mdl = ss.current_model

    if mdl is None or ss.entropy_t is None:
        st.info("Run Stage 1 to compute the entropy curve.")
    else:
        t_vals = ss.entropy_t
        S_vals = ss.entropy_s

        _default_t, _peaks = _select_t_star(t_vals, S_vals)

        _t_min  = float(t_vals[0])
        _t_max  = float(t_vals[-1])
        _t_step = float(t_vals[1] - t_vals[0])
        ss.t_star = float(np.clip(ss.t_star, _t_min, _t_max))

        _tc_sl, _tc_in = st.columns([3, 1])
        with _tc_sl:
            _t_slider = st.slider(
                "t* — selected time",
                min_value=_t_min, max_value=_t_max,
                value=ss.t_star, step=_t_step,
                key="t_star_slider",
            )
        with _tc_in:
            _t_input = st.number_input(
                "", min_value=_t_min, max_value=_t_max,
                value=ss.t_star, step=_t_step,
                key="t_star_input",
                label_visibility="collapsed",
                format="%.3f",
            )
        if _t_slider != ss.t_star:
            ss.t_star = _t_slider
        elif _t_input != ss.t_star:
            ss.t_star = _t_input
        t_sel = ss.t_star
        st.caption("Default: highest entropy peak — corresponds to the "
                   "moment of tightest clustering.")

        fig_ent = go.Figure()
        fig_ent.add_trace(go.Scatter(
            x=t_vals, y=S_vals, mode='lines',
            line=dict(color='steelblue', width=2), name='S_rev(t)',
        ))
        if len(_peaks) > 0:
            _best_pi = _peaks[np.argmax(S_vals[_peaks])]
            fig_ent.add_vline(
                x=float(t_vals[_best_pi]), line_dash='dash',
                line_color='red', opacity=0.8,
                annotation_text=f"Highest peak  t={t_vals[_best_pi]:.2f}",
                annotation_position="top right",
            )
        fig_ent.add_vline(
            x=t_sel, line_color='black', line_width=2,
            annotation_text=f"t* = {t_sel:.2f}", annotation_position="top left",
        )
        fig_ent.update_layout(
            title="Reverse von Neumann Entropy  S(t)  (matrix-based, "
                  "non-negative by construction)",
            xaxis_title="Time t", yaxis_title="Reverse Entropy",
            height=400,
        )
        st.plotly_chart(fig_ent, use_container_width=True)
        if S_vals.min() < 0:
            st.error(f"Entropy went negative (min={S_vals.min():.4f}) — this "
                     "should not happen with the matrix-based formula. "
                     "Check dqc_core.get_trajectories_with_entropy.")

        _s_at_tstar = float(np.interp(t_sel, t_vals, S_vals))
        st.info(f"Selected  **t* = {t_sel:.3f}**   "
                f"(S = {_s_at_tstar:.4f}, default = highest peak at "
                f"t = {_default_t:.3f})")
        if len(_peaks) == 0:
            st.warning("No clear entropy peak detected. "
                       "Try reducing σ or increasing m.")


# ── TAB 4 — Trajectories ──────────────────────────────────────────────────────
with tab4:
    st.subheader("Evolution & Trajectories")

    if not ss.stage_trajectories:
        st.info("Run Stage 1 to see trajectories.")
    else:
        n_done = len(ss.stage_trajectories)
        n_p    = ss.X_sphere.shape[0]
        ix, iy = ss.pc_x_idx, ss.pc_y_idx
        iz = ss.pc_z_idx if ss.X_sphere.shape[1] >= 3 else iy

        if ss.y_labels is not None:
            lstr = [str(l) for l in ss.y_labels]
        elif ss.cluster_labels is not None:
            lstr = [str(l) for l in ss.cluster_labels]
        else:
            lstr = ["?"] * n_p
        _pal     = px.colors.qualitative.Safe
        _uniq    = list(dict.fromkeys(lstr))
        _lc      = {l: _pal[i % len(_pal)] for i, l in enumerate(_uniq)}
        _col_pts = [_lc[l] for l in lstr]

        pos0_global = ss.stage_trajectories[0]['pos0']
        _xlim, _ylim, _zlim = _axis_limits(pos0_global[:, [ix, iy, iz]])

        stage_views = [_get_stage_view(i, n_done, n_steps) for i in range(n_done)]

        st.subheader("Particle Trajectories — one subplot per stage")
        fig_grid = make_subplots(rows=1, cols=n_done,
                                  subplot_titles=[
                                      f"Stage {i+1} trajectories (t*={tv[3]:.2f})"
                                      for i, tv in enumerate(stage_views)
                                  ])
        for s_idx, (traj, pos0, pos_end, t_star) in enumerate(stage_views):
            col = s_idx + 1
            for _lbl in _uniq:
                _idx = [i for i, l in enumerate(lstr) if l == _lbl]
                for i in _idx:
                    fig_grid.add_trace(go.Scatter(
                        x=traj[:, i, ix], y=traj[:, i, iy],
                        mode='lines',
                        line=dict(color=_lc[_lbl], width=0.8),
                        opacity=0.4,
                        legendgroup=_lbl,
                        showlegend=(s_idx == 0 and i == _idx[0]),
                        name=_lbl,
                    ), row=1, col=col)
            fig_grid.add_trace(go.Scatter(
                x=pos0[:, ix], y=pos0[:, iy], mode='markers', name='start',
                marker=dict(symbol='x', size=5, color='black'),
                showlegend=(s_idx == 0), legendgroup='start',
            ), row=1, col=col)
            fig_grid.add_trace(go.Scatter(
                x=pos_end[:, ix], y=pos_end[:, iy], mode='markers', name='end',
                marker=dict(size=7, color=_col_pts,
                            line=dict(width=0.5, color='black')),
                showlegend=(s_idx == 0), legendgroup='end',
            ), row=1, col=col)
            fig_grid.update_xaxes(range=list(_xlim), title_text=f"PC{ix+1}", row=1, col=col)
            fig_grid.update_yaxes(range=list(_ylim), title_text=f"PC{iy+1}", row=1, col=col)
        fig_grid.update_layout(height=420)
        st.plotly_chart(fig_grid, use_container_width=True)

        st.subheader("Before → After Comparison")
        view_mode = st.radio("View", ["2D", "3D"], horizontal=True, key="ba_view_mode")

        n_panels = 1 + n_done
        cols_ba  = st.columns(n_panels)

        def _scatter2d(x, y, cols, ttl):
            f = go.Figure(go.Scatter(
                x=x, y=y, mode='markers',
                marker=dict(size=6, color=cols, line=dict(width=0.3, color='black')),
            ))
            f.update_layout(title=ttl, xaxis_title=f"PC{ix+1}", yaxis_title=f"PC{iy+1}",
                             xaxis_range=list(_xlim), yaxis_range=list(_ylim),
                             height=340, showlegend=False)
            return f

        def _scatter3d(x, y, z, cols, ttl):
            f = go.Figure(go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                marker=dict(size=4, color=cols, line=dict(width=0.3, color='black')),
            ))
            f.update_layout(
                title=ttl, height=380, showlegend=False,
                scene=dict(
                    xaxis=dict(title=f"PC{ix+1}", range=list(_xlim)),
                    yaxis=dict(title=f"PC{iy+1}", range=list(_ylim)),
                    zaxis=dict(title=f"PC{iz+1}", range=list(_zlim)),
                ),
            )
            return f

        if view_mode == "2D":
            with cols_ba[0]:
                st.plotly_chart(_scatter2d(pos0_global[:, ix], pos0_global[:, iy],
                                            _col_pts, "Before (t=0)"),
                                 use_container_width=True)
            for s_idx, (traj, pos0, pos_end, t_star) in enumerate(stage_views):
                with cols_ba[s_idx + 1]:
                    st.plotly_chart(_scatter2d(
                        pos_end[:, ix], pos_end[:, iy], _col_pts,
                        f"After Stage {s_idx + 1} (t*={t_star:.2f})"),
                        use_container_width=True)
        else:
            with cols_ba[0]:
                st.plotly_chart(_scatter3d(
                    pos0_global[:, ix], pos0_global[:, iy], pos0_global[:, iz],
                    _col_pts, "Before (t=0)"),
                    use_container_width=True)
            for s_idx, (traj, pos0, pos_end, t_star) in enumerate(stage_views):
                with cols_ba[s_idx + 1]:
                    st.plotly_chart(_scatter3d(
                        pos_end[:, ix], pos_end[:, iy], pos_end[:, iz], _col_pts,
                        f"After Stage {s_idx + 1} (t*={t_star:.2f})"),
                        use_container_width=True)

        for s_idx in range(n_done, 4):
            st.info(f"Run Stage {s_idx + 1} to see results.")


# ── TAB 5 — Clustering & Diagnostics ─────────────────────────────────────────
with tab5:
    st.subheader("Clustering & Diagnostics")

    if not ss.stage_trajectories:
        st.info("Run Stage 1 to see diagnostics.")
    else:
        n_done = len(ss.stage_trajectories)
        n_p    = ss.X_sphere.shape[0]
        y      = ss.y_labels
        ix, iy = ss.pc_x_idx, ss.pc_y_idx
        lstr   = ([str(l) for l in y] if y is not None else ["?"] * n_p)
        sort_idx = np.argsort(lstr)

        st.subheader("A — Euclidean Distances from Reference Point")
        ref_i = int(st.number_input("Reference point", value=0,
                                     min_value=0, max_value=n_p - 1))
        _pt_cols = [lstr[i] for i in sort_idx]
        _pal2    = px.colors.qualitative.Safe
        _ulbls2  = list(dict.fromkeys(_pt_cols))
        _lcm     = {l: _pal2[i % len(_pal2)] for i, l in enumerate(_ulbls2)}

        mdl = ss.current_model
        pos0_global = ss.stage_trajectories[0]['pos0']
        d0 = mdl.compute_euclidean_distances(pos0_global, ref_i)[sort_idx]
        _d_lo, _d_hi = 0.0, float(d0.max()) * 1.10

        def _dist_fig(dvals, title):
            fig = go.Figure()
            for lbl in _ulbls2:
                mask = [i for i, c in enumerate(_pt_cols) if c == lbl]
                fig.add_trace(go.Scatter(
                    x=mask, y=dvals[mask], mode='markers',
                    name=lbl, marker=dict(size=4, color=_lcm[lbl]),
                ))
            fig.update_layout(title=title,
                               xaxis_title="Point index (sorted by class)",
                               yaxis_title="Distance",
                               yaxis_range=[_d_lo, _d_hi],
                               height=300)
            return fig

        d_panels = [("d(i, ref)  at  t=0", d0)]
        for s_idx in range(n_done):
            traj, pos0, pos_end, t_star = _get_stage_view(s_idx, n_done, n_steps)
            d_s = mdl.compute_euclidean_distances(pos_end, ref_i)[sort_idx]
            d_panels.append((f"d(i, ref)  at  t*={t_star:.2f} [Stage {s_idx + 1}]", d_s))

        d_cols = st.columns(len(d_panels))
        for col, (title, dvals) in zip(d_cols, d_panels):
            with col:
                st.plotly_chart(_dist_fig(dvals, title), use_container_width=True)

        st.divider()
        st.subheader("B — Cluster Extraction")
        st.caption(f"k = {ss.k_cl}  (set in sidebar)")

        if ss.cluster_labels is not None:
            _, _, pos_latest, _ = _get_stage_view(n_done - 1, n_done, n_steps)
            fig_cl = px.scatter(
                x=pos_latest[:, ix], y=pos_latest[:, iy],
                color=ss.cluster_labels.astype(str),
                labels={"x": f"PC{ix+1}", "y": f"PC{iy+1}", "color": "Cluster"},
                title=f"K-means assignments  (k={ss.k_cl})",
            )
            fig_cl.update_traces(marker=dict(size=6))
            st.plotly_chart(fig_cl, use_container_width=True)
        else:
            st.info("Use 'Extract clusters (k-means)' in the sidebar to run k-means "
                    "on the latest stage's positions.")

        if (ss.dataset_choice.startswith("S&P 500") and ss.sp500_dates is not None
                and ss.cluster_labels is not None):
            st.divider()
            st.subheader("S&P 500 market epochs (Fig. 18-style)")
            _render_sp500_epochs(ss.sp500_dates, ss.sp500_ref,
                                  ss.cluster_labels, ss.sp500_ref_is_approx)

        st.divider()
        st.subheader("C — Metrics (all stages)")
        st.caption("Computed automatically after each stage using k-means on "
                   "the stage's converged positions. Use 'Extract clusters' in "
                   "the sidebar to re-run with a different k — this updates the "
                   "current stage's row only.")

        if not ss.stage_metrics:
            st.info("Run Stage 1 to see metrics.")
        else:
            _mrows = []
            for _m in ss.stage_metrics:
                _mrows.append({
                    'Stage':       f"Stage {_m['stage']}",
                    'k':           str(_m['k']),
                    'Silhouette':  f"{_m['silhouette']:.3f}",
                    'ARI':         f"{_m['ARI']:.3f}"     if _m['ARI']     is not None else "—",
                    'NMI':         f"{_m['NMI']:.3f}"     if _m['NMI']     is not None else "—",
                    'Jaccard':     f"{_m['Jaccard']:.3f}" if _m['Jaccard'] is not None else "—",
                })
            _metrics_df = pd.DataFrame(_mrows).set_index('Stage')
            st.table(_metrics_df)
            if y is None:
                st.caption("ARI, NMI, Jaccard require true labels — "
                           "load a dataset with a label column to enable them.")

            if y is not None and ss.cluster_labels is not None:
                from sklearn.metrics import confusion_matrix
                y_enc  = LabelEncoder().fit_transform(y)
                cl     = ss.cluster_labels
                cm     = confusion_matrix(y_enc, cl)
                fig_cm = px.imshow(
                    cm, text_auto=True,
                    labels=dict(x="DQC Cluster", y="True Label", color="Count"),
                    title="Confusion: True Labels vs DQC Clusters (latest extraction)",
                )
                st.plotly_chart(fig_cm, use_container_width=True)
