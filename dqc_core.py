import os
os.environ['OMP_NUM_THREADS'] = '1'   # FIX 5: eliminate KMeans non-determinism on Windows/MKL

import numpy as np
import scipy.linalg
from sklearn.cluster import KMeans   # FIX 5: DBSCAN removed entirely
import matplotlib.pyplot as plt


# ── Private helpers ───────────────────────────────────────────────────────────

def _preprocess(X, preprocess):
    """SVD+whiten+unit-sphere or unit-variance PCA (preprocess=True),
    z-score standardisation (preprocess=False),
    or pass-through with no transformation (preprocess='none').
    """
    n, p = X.shape
    if preprocess == 'none':
        # Data is already preprocessed externally (e.g. sphere-normalised).
        # Use as-is; d = number of columns supplied.
        return X.copy(), p
    if preprocess:
        Xc = X - X.mean(axis=0)          # centre so SVD == PCA
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        cumvar  = np.cumsum(S ** 2) / np.sum(S ** 2)

        # FIX 1 — d selection: variance threshold with n_features-based floor
        d_var = int(np.searchsorted(cumvar, 0.90)) + 1
        if p <= 4:
            d_min = 2
        elif p <= 10:
            d_min = 3
        else:
            d_min = 5          # high-dimensional data (e.g. 30 features) needs >=5 dims
        d = min(max(d_min, d_var), min(10, p))   # FIX 1

        if d <= 4:
            # Low-d: whiten then project to unit sphere
            U_w   = U[:, :d] / np.sqrt(S[:d])[np.newaxis, :]
            norms = np.linalg.norm(U_w, axis=1, keepdims=True)
            Xp    = U_w / np.maximum(norms, 1e-12)
        else:
            # High-d: unit-variance PCA whitening, no sphere projection.
            # U[:,k] has column norm 1, so U[:,k]*sqrt(n) has variance ~1.
            # Skipping unit sphere preserves radial cluster separation.
            Xp = U[:, :d] * np.sqrt(n)
    else:
        mu  = X.mean(axis=0)
        std = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
        Xp  = (X - mu) / std
        d   = p
        print(f"  [standardisation check] mean={Xp.mean(axis=0).round(4)}  "
              f"std={Xp.std(axis=0).round(4)}")
    return Xp, d


def _build_and_evolve(Xp, d, sigma, m, T, delta_t, lambda_thresh):
    """
    Build DQC matrices from preprocessed data and run time evolution.

    Returns
    -------
    trajectories   : (T, d, n) position expectation values
    mpd_values     : (T,) mean pairwise distance between particle positions per frame
    entropy_values : (T,) reverse von Neumann entropy per frame (GitHub formula)

    Entropy notes
    -------------
    N_step = real(psi_ev.T @ psi_ev)  — (n,n) symmetric, time-VARYING because
    phase² = exp(-2iωt) does NOT cancel (unlike the conjugate product which is
    time-invariant). Computed implicitly: N_step @ v = real(psi_ev.T @ (psi_ev @ v))
    in O(rn) without forming the O(n²) matrix, safe for large n.
    Reverse entropy: S = -Σ_d (1/p_i) log(1/p_i),  p_i = e_i^T N² e_i
    where e_i = position row i. Maximised at best clustering → use argmax.
    """
    n        = Xp.shape[0]
    sig2     = sigma ** 2
    two_sig2 = 2.0 * sig2

    # Squared distances and Gaussian overlap
    diff = Xp[:, np.newaxis, :] - Xp[np.newaxis, :, :]   # (n, n, d)
    D    = np.sum(diff ** 2, axis=-1)                      # (n, n)
    G    = np.exp(-D / (4.0 * sigma ** 2))                 # (n, n)

    midpoints = (Xp[:, np.newaxis, :] + Xp[np.newaxis, :, :]) / 2.0

    # Quantum potential at the n^2 pairwise midpoints, via closed form.
    # |midpoint(x_i,x_j) - x_k|^2 = (D[i,k]+D[j,k])/2 - D[i,j]/4 (law of
    # cosines), so the per-midpoint sum over k reduces to two dense n x n
    # matmuls (BLAS GEMM) on the D/G already computed above, instead of a
    # Python loop recomputing a fresh O(n) distance for each of n^2
    # midpoints (O(n^3 d) via slow broadcast+exp, not BLAS). The exp(D_ij/
    # 8sigma^2) prefactor common to numerator and denominator cancels
    # analytically, so it never needs to be evaluated (also avoids overflow
    # for distant pairs). Verified numerically identical to the original
    # loop (max abs diff ~1e-14) before replacing it.
    GG = G @ G                                  # psi(midpoint_ij), up to the cancelling factor
    P  = (D * G) @ G                            # weighted cross term
    GG_safe = np.maximum(GG, 1e-300)
    V_mid = (((P + P.T) / two_sig2 - (D / two_sig2 / 2.0 + d) * GG)
             / (2.0 * GG_safe))

    # Matrices N, H, X_op
    N    = G.copy()
    H    = (D / (4.0 * m * sigma ** 4) + V_mid) * G
    H    = (H + H.T) / 2.0
    X_op = midpoints.transpose(2, 0, 1) * G[np.newaxis, :, :]  # (d, n, n)

    # Orthonormalise via eigendecomposition of N
    Lam, Q = scipy.linalg.eigh(N)
    Lam    = np.clip(Lam, 0.0, None)
    keep   = Lam >= lambda_thresh    # threshold controls eigenvector truncation
    Q_r    = Q[:, keep]
    Lam_r  = Lam[keep]
    r      = len(Lam_r)
    print(f"  r={r}  (kept {r}/{n} eigenvectors)")

    sqrtL = np.sqrt(Lam_r)
    isqL  = 1.0 / sqrtL

    # H_tr = diag(isqL) (Q_r^T H Q_r) diag(isqL)
    H_tr = isqL[:, None] * (Q_r.T @ H @ Q_r) * isqL[None, :]
    H_tr = (H_tr + H_tr.T) / 2.0

    # X_tr[k] = diag(isqL) (Q_r^T X_op[k] Q_r) diag(isqL)
    X_tr = np.empty((d, r, r))
    for kk in range(d):
        X_tr[kk] = isqL[:, None] * (Q_r.T @ X_op[kk] @ Q_r) * isqL[None, :]

    # Initial states: psi_tr = sqrtL * Q_r^T   shape (r, n)
    psi_tr   = sqrtL[:, None] * Q_r.T
    norms_sq = np.sum(psi_tr ** 2, axis=0)   # (n,)

    # Eigendecompose H_tr once; phase recomputed fresh each step
    Omega, P_eig = np.linalg.eigh(H_tr)
    Pdag_psi     = P_eig.conj().T @ psi_tr   # (r, n)

    trajectories   = np.zeros((T, d, n), dtype=np.float64)
    mpd_values     = np.zeros(T, dtype=np.float64)
    entropy_values = np.zeros(T, dtype=np.float64)

    # Upper-triangle indices for MPD, reused every frame
    idx_u = np.triu_indices(n, k=1)

    for t in range(1, T + 1):
        phase  = np.exp(-1j * t * delta_t * Omega)
        psi_ev = P_eig @ (phase[:, None] * Pdag_psi)   # (r, n) complex

        for kk in range(d):
            Xpsi = X_tr[kk] @ psi_ev
            trajectories[t - 1, kk] = np.real(
                np.sum(psi_ev.conj() * Xpsi, axis=0)) / norms_sq

        # ── MPD ──────────────────────────────────────────────────────────────
        pos   = trajectories[t - 1].T           # (n, d) real
        diffs = pos[idx_u[0]] - pos[idx_u[1]]   # (n_pairs, d)
        mpd_values[t - 1] = np.mean(np.sqrt(np.sum(diffs ** 2, axis=-1)))

        # ── Reverse von Neumann entropy (GitHub formula) ──────────────────────
        # N_step = real(psi_ev.T @ psi_ev), (n,n) symmetric, time-varying.
        # Avoid forming the O(n²) matrix: apply N_step twice via matrix-vector
        # products in O(rn) each.
        #   N_step @ v = real(psi_ev.T @ (psi_ev @ v))       [O(rn)]
        #   v^T N² v   = v · (N_step @ (N_step @ v))         [O(rn)]
        # => p_i[kk] = e_kk^T N_step² e_kk  for each position dimension kk
        psi_pos = trajectories[t - 1]           # (d, n) real
        p_i     = np.empty(d)
        for kk in range(d):
            v        = psi_pos[kk]                          # (n,) real
            Nv       = np.real(psi_ev.T @ (psi_ev @ v))    # N_step @ v, (n,)
            NNv      = np.real(psi_ev.T @ (psi_ev @ Nv))   # N_step² @ v, (n,)
            p_i[kk]  = float(np.dot(v, NNv))               # scalar

        # -(1/p) log(1/p) = log(p)/p; skip non-positive (N_step not guaranteed PD)
        entropy = 0.0
        for p in p_i:
            if p > 1e-300:
                entropy += -(1.0 / p) * np.log(1.0 / p)
        entropy_values[t - 1] = entropy

    return trajectories, mpd_values, entropy_values


def _plot(Xp, trajectories, labels, k, best_frame, X_vis=None):
    """Two- or three-panel cluster plot."""
    n    = Xp.shape[0]
    cmap = 'tab10'

    ncols = 3 if X_vis is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5))
    ax1, ax2  = axes[0], axes[1]

    # Vectorised: pass (T, n) arrays so matplotlib draws all n lines in one call.
    # Looping over n individual plot() calls creates n Line2D objects and is
    # too slow for large n (e.g. n=569), causing Spyder to drop the figure.
    ax1.plot(trajectories[:, 0, :], trajectories[:, 1, :],
             color='gray', lw=0.4, alpha=0.25)
    ax1.scatter(Xp[:, 0], Xp[:, 1],
                c='k', s=14, zorder=4, marker='x', label='t=0')
    ax1.scatter(trajectories[best_frame, 0], trajectories[best_frame, 1],
                c=labels, cmap=cmap, s=55, zorder=5,
                edgecolors='k', linewidths=0.4, label=f'best frame {best_frame + 1}')
    ax1.set_title('DQC Trajectories')
    ax1.set_xlabel('Dim 1 (preprocessed)')
    ax1.set_ylabel('Dim 2 (preprocessed)')
    ax1.legend(fontsize=8)

    sc = ax2.scatter(Xp[:, 0], Xp[:, 1], c=labels, cmap=cmap, s=40,
                     edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax2, label='Cluster')
    ax2.set_title(f'Final Cluster Assignments (k={k})')
    ax2.set_xlabel('Dim 1 (preprocessed)')
    ax2.set_ylabel('Dim 2 (preprocessed)')

    if X_vis is not None:
        ax3 = axes[2]
        sc3 = ax3.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap=cmap, s=40,
                          edgecolors='k', linewidths=0.4)
        plt.colorbar(sc3, ax=ax3, label='Cluster')
        ax3.set_title(f'Clusters in PCA Space (k={k})')
        ax3.set_xlabel('PC 1')
        ax3.set_ylabel('PC 2')

    plt.tight_layout()
    plt.show()


def _plot_potential(Xp, sigma, resolution, interactive=False):
    """2D contourf + static 3D surface; optionally also opens Plotly in browser."""
    from mpl_toolkits.mplot3d import Axes3D  # registers '3d' projection

    sig2     = sigma ** 2
    two_sig2 = 2.0 * sig2
    Xp2      = Xp[:, :2]

    for axis, col in enumerate([Xp2[:, 0], Xp2[:, 1]]):
        rng = col.max() - col.min()
        pad = 0.10 * rng
        lo, hi = col.min() - pad, col.max() + pad
        if axis == 0:
            gx = np.linspace(lo, hi, resolution)
        else:
            gy = np.linspace(lo, hi, resolution)

    G0, G1   = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([G0.ravel(), G1.ravel()])

    V_flat = np.empty(resolution * resolution)
    bsz    = 512
    d_grid = 2
    for s in range(0, len(grid_pts), bsz):
        e    = min(s + bsz, len(grid_pts))
        gp   = grid_pts[s:e]
        db   = np.sum((gp[:, None, :] - Xp2[None, :, :]) ** 2, axis=-1)
        eb   = np.exp(-db / two_sig2)
        psib = np.maximum(eb.sum(axis=1), 1e-300)
        V_flat[s:e] = ((db / sig2 - d_grid) * eb).sum(axis=1) / (2.0 * psib)

    # FIX 7 — V negativity guard: shift by ground-state energy then clip
    V_flat -= V_flat.min()
    V_grid  = V_flat.reshape(resolution, resolution)
    V_grid  = np.clip(V_grid, 0, None)   # FIX 7
    assert V_grid.min() >= 0             # FIX 7

    # ── Matplotlib: 2D contourf + static 3D surface ──────────────────────────
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    cf  = ax1.contourf(G0, G1, V_grid, levels=40, cmap='viridis')
    plt.colorbar(cf, ax=ax1, label='V(x)')
    ax1.scatter(Xp2[:, 0], Xp2[:, 1],
                c='white', s=20, edgecolors='k', linewidths=0.5, zorder=5)
    ax1.set_title(f'Quantum Potential V(x)  (sigma={sigma:.3f})')
    ax1.set_xlabel('Dim 1 (preprocessed)')
    ax1.set_ylabel('Dim 2 (preprocessed)')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(G0, G1, V_grid, cmap='plasma',
                     linewidth=0, antialiased=True, alpha=0.9)
    ax2.set_title('Quantum Potential -- 3D Surface')
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    ax2.set_zlabel('V(x)')

    plt.tight_layout()
    plt.show()

    # ── Plotly interactive surface (additional, opens in browser) ─────────────
    if interactive:
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("  [plot_potential] plotly not found -- "
                  "install with `pip install plotly` for the interactive browser plot.")
            return

        fig_px = go.Figure(data=[
            go.Surface(x=G0, y=G1, z=V_grid, colorscale='Plasma',
                       colorbar=dict(title='V(x)')),
        ])
        fig_px.update_layout(
            title=f'Quantum Potential -- Interactive 3D  (sigma={sigma:.3f})',
            scene=dict(xaxis_title='Dim 1 (preprocessed)',
                       yaxis_title='Dim 2 (preprocessed)',
                       zaxis_title='V(x)'),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig_px.show(renderer='browser')


def _plot_trajectories_only(Xp, trajectories, best_frame, T, delta_t, sigma):
    """Trajectories + start/end positions, no cluster labels."""
    n       = Xp.shape[0]
    colours = np.arange(n)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Vectorised: same reason as _plot — avoids n individual Line2D objects
    ax.plot(trajectories[:, 0, :], trajectories[:, 1, :],
            color='steelblue', lw=0.5, alpha=0.3)
    ax.scatter(Xp[:, 0], Xp[:, 1],
               c='k', s=20, zorder=5, marker='x', label='t=0 (Xp)')
    ax.scatter(trajectories[best_frame, 0], trajectories[best_frame, 1],
               c=colours, cmap='tab10', s=50, zorder=6,
               edgecolors='k', linewidths=0.3, label=f'best frame {best_frame + 1}')
    ax.set_title(f'DQC Trajectories  (sigma={sigma:.3f}, T={T}, dt={delta_t})')
    ax.set_xlabel('Dim 1 (preprocessed)')
    ax.set_ylabel('Dim 2 (preprocessed)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ── DQC class ─────────────────────────────────────────────────────────────────

class DQC:
    """
    Dynamic Quantum Clustering (DQC).

    Reference: Weinstein & Horn, Physical Review E, 2009.
    Conventions: hbar=1, a=1/(2*sigma^2), D_ij = squared Euclidean distance.

    Parameters
    ----------
    sigma        : Gaussian width; auto-estimated from preprocessed data if None
    m            : particle mass; auto-estimated as 1/sigma^2 if None
    T            : number of time steps
    delta_t      : time step size
    k            : number of clusters; estimated from potential wells if None
    lambda_thresh: eigenvalue cutoff for N truncation (default 1e-5)
    preprocess   : True -> SVD+whiten+unit-sphere/PCA; False -> z-score

    Attributes set after fit()
    --------------------------
    labels_        : (n,) cluster labels
    trajectories_  : (T, d, n) position expectation values
    entropy_values_     : (T,) reverse von Neumann entropy per frame (GitHub formula)
    mpd_values_         : (T,) mean pairwise distance between trajectory positions per frame
    best_frame_         : frame of peak entropy (argmax, 20% burn-in); falls back to MPD argmin if entropy flat
    _stopping_criterion_: 'entropy' or 'mpd' — which criterion selected best_frame_
    Xp_            : (n, d) preprocessed data
    sigma_         : sigma value used
    m_             : mass value used
    d_             : preprocessed dimension
    k_             : number of clusters used
    """

    def __init__(self, sigma=None, m=None, T=200, delta_t=0.1, k=None,
                 lambda_thresh=1e-5, eig_threshold=None, preprocess=True):
        # eig_threshold is the interface-facing alias for lambda_thresh
        if eig_threshold is not None:
            lambda_thresh = eig_threshold
        self.sigma         = sigma
        self.m             = m
        self.T             = T
        self.delta_t       = delta_t
        self.k             = k
        self.lambda_thresh = lambda_thresh
        self.preprocess    = preprocess

    # ── Core API ──────────────────────────────────────────────────────────────

    def fit(self, X, y_true=None):
        """
        Preprocess X, build matrices, run time evolution, assign clusters.

        Parameters
        ----------
        X      : array (n_samples, n_features)
        y_true : optional ground-truth labels; prints ARI and NMI if provided
        """
        X       = np.array(X, dtype=np.float64)
        self.n_ = X.shape[0]

        self.Xp_, self.d_ = _preprocess(X, self.preprocess)

        # sigma, m and delta_t are provided by the caller (interface) — no
        # auto-estimation here.
        if self.sigma is None:
            raise ValueError("sigma must be provided (no auto-estimation in dqc_core).")
        self.sigma_   = self.sigma
        self.m_       = self.m if self.m is not None else 1.0 / self.sigma_ ** 2
        self.delta_t_ = self.delta_t

        print(f"  n={self.n_}  d={self.d_}  "
              f"sigma={self.sigma_:.4f}  m={self.m_:.4f}")

        # Store evolution machinery for get_positions_at_t / reinitialise API
        self._build_evolution_state(self.Xp_)

        # Evolution: returns trajectories, MPD per frame, reverse entropy per frame
        self.trajectories_, self.mpd_values_, self.entropy_values_ = _build_and_evolve(
            self.Xp_, self.d_, self.sigma_, self.m_,
            self.T, self.delta_t_, self.lambda_thresh,
        )

        # Stopping criterion: prefer reverse entropy (argmax after 20% burn-in).
        # Entropy is time-varying because N_step = real(psi_ev.T @ psi_ev) uses
        # phase² = exp(-2iωt) which does NOT cancel. Falls back to MPD argmin
        # if the entropy curve is flat (range < 0.01 — no useful signal).
        burn_in    = int(0.20 * self.T)
        ent_range  = float(self.entropy_values_.max() - self.entropy_values_.min())

        if ent_range > 0.01:
            self.best_frame_          = burn_in + int(np.argmax(self.entropy_values_[burn_in:]))
            self._stopping_criterion_ = 'entropy'
            print(f"  Best frame: {self.best_frame_ + 1}/{self.T}  "
                  f"(entropy={self.entropy_values_[self.best_frame_]:.4f}, criterion=entropy)")
        else:
            self.best_frame_          = burn_in + int(np.argmin(self.mpd_values_[burn_in:]))
            self._stopping_criterion_ = 'mpd'
            print(f"  Best frame: {self.best_frame_ + 1}/{self.T}  "
                  f"(MPD={self.mpd_values_[self.best_frame_]:.4f}, "
                  f"criterion=mpd [entropy flat, range={ent_range:.4f}])")

        # k is provided by the caller (interface) — no auto-estimation here.
        if self.k is None:
            raise ValueError("k must be provided (no auto-estimation in dqc_core).")
        self.k_ = self.k

        positions    = self.trajectories_[self.best_frame_].T   # (n, d)
        # FIX 5: KMeans only, n_init=50, max_iter=500
        self.labels_ = KMeans(n_clusters=self.k_, random_state=42,
                              n_init=50, max_iter=500).fit_predict(positions)

        if y_true is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            ari = adjusted_rand_score(y_true, self.labels_)
            nmi = normalized_mutual_info_score(y_true, self.labels_)
            print(f"  ARI = {ari:.4f}   NMI = {nmi:.4f}")

        return self

    def get_labels(self):
        """Return cluster labels assigned after fit()."""
        return self.labels_

    def plot_trajectories(self, X_vis=None):
        """Two-panel (or three-panel) trajectory + cluster plot."""
        _plot(self.Xp_, self.trajectories_, self.labels_, self.k_,
              self.best_frame_, X_vis=X_vis)

    def plot_potential(self, resolution=100, interactive=False):
        """Evaluate V(x) over a 2D grid and show contourf + 3D surface."""
        _plot_potential(self.Xp_, self.sigma_, resolution, interactive=interactive)

    def plot_trajectories_only(self):
        """Plot raw trajectories with no cluster labels."""
        _plot_trajectories_only(
            self.Xp_, self.trajectories_, self.best_frame_,
            self.T, self.delta_t_, self.sigma_,
        )

    # ── Interface API ─────────────────────────────────────────────────────────

    def _build_evolution_state(self, Xp):
        """
        Build and cache the evolution machinery (Omega, P_eig, Pdag_psi, X_tr,
        norms_sq) from preprocessed positions Xp.  Called automatically by
        fit() and reinitialise() so that get_positions_at_t() always works.
        Raises ValueError if fewer than 3 eigenvectors survive lambda_thresh.
        """
        n, d   = Xp.shape
        sigma  = self.sigma_
        m      = self.m_
        sig2   = sigma ** 2
        two_s2 = 2.0 * sig2

        diff = Xp[:, np.newaxis, :] - Xp[np.newaxis, :, :]
        D    = np.sum(diff ** 2, axis=-1)
        G    = np.exp(-D / (4.0 * sig2))

        midpoints = (Xp[:, np.newaxis, :] + Xp[np.newaxis, :, :]) / 2.0

        # Closed-form pairwise-midpoint potential — see _build_and_evolve
        # for the derivation; numerically identical to the original
        # per-midpoint loop, just via BLAS matmuls instead of a Python loop.
        GG = G @ G
        P  = (D * G) @ G
        GG_safe = np.maximum(GG, 1e-300)
        V_mid = (((P + P.T) / two_s2 - (D / two_s2 / 2.0 + d) * GG)
                 / (2.0 * GG_safe))

        H  = (D / (4.0 * m * sig2 ** 2) + V_mid) * G
        H  = (H + H.T) / 2.0
        Xop = midpoints.transpose(2, 0, 1) * G[np.newaxis, :, :]

        Lam, Q = scipy.linalg.eigh(G)
        Lam    = np.clip(Lam, 0.0, None)
        keep   = Lam >= self.lambda_thresh
        Q_r    = Q[:, keep]
        Lam_r  = Lam[keep]
        r      = len(Lam_r)
        if r < 3:
            raise ValueError(
                f"Only {r} eigenvectors retained (eig_threshold={self.lambda_thresh:.1e}). "
                "Increase sigma or lower eig_threshold."
            )

        sqrtL = np.sqrt(Lam_r)
        isqL  = 1.0 / sqrtL
        H_tr  = isqL[:, None] * (Q_r.T @ H @ Q_r) * isqL[None, :]
        H_tr  = (H_tr + H_tr.T) / 2.0

        X_tr = np.empty((d, r, r))
        for kk in range(d):
            X_tr[kk] = isqL[:, None] * (Q_r.T @ Xop[kk] @ Q_r) * isqL[None, :]

        psi_tr   = sqrtL[:, None] * Q_r.T          # (r, n)
        norms_sq = np.sum(psi_tr ** 2, axis=0)     # (n,)
        Omega, P_eig = np.linalg.eigh(H_tr)
        Pdag_psi     = P_eig.conj().T @ psi_tr     # (r, n)

        self._evo_d_        = d
        self._evo_Omega_    = Omega
        self._evo_P_eig_    = P_eig
        self._evo_Pdag_psi_ = Pdag_psi
        self._evo_norms_sq_ = norms_sq
        self._evo_X_tr_     = X_tr

    def reinitialise(self, new_positions):
        """
        Reset starting positions for a new DQC stage.

        new_positions : (n, d) array — converged positions from the previous stage.
        Recomputes N, H, X matrices from new_positions.
        Does not change sigma or m.  After this call, get_positions_at_t(t)
        evolves from the new starting configuration.
        """
        self.Xp_ = np.array(new_positions, dtype=np.float64)
        self._build_evolution_state(self.Xp_)

    def get_positions_at_t(self, t):
        """
        Return positions of all n points at time t (float, actual time units).
        Shape (n, d).
        """
        phase  = np.exp(-1j * t * self._evo_Omega_)
        psi_ev = self._evo_P_eig_ @ (phase[:, None] * self._evo_Pdag_psi_)
        d      = self._evo_d_
        n      = len(self._evo_norms_sq_)
        pos    = np.zeros((n, d))
        for kk in range(d):
            Xpsi       = self._evo_X_tr_[kk] @ psi_ev
            pos[:, kk] = np.real(
                np.sum(psi_ev.conj() * Xpsi, axis=0)) / self._evo_norms_sq_
        return pos

    def get_trajectories(self, t_values):
        """
        Return array of shape (len(t_values), n, d).
        Each slice [i] is the positions of all n points at time t_values[i].
        """
        return np.array([self.get_positions_at_t(t) for t in t_values])

    def get_trajectories_with_entropy(self, t_values):
        """
        Return (trajectories, entropy_values) for the given t_values, evolved
        from the current starting positions (post fit() or reinitialise()).

        trajectories  : (len(t_values), n, d) position expectation values
        entropy_values: (len(t_values),) matrix-based reverse von Neumann
                        entropy, computed from the N_step = real(psi_ev.T @ psi_ev)
                        density matrix — the same formula used in
                        _build_and_evolve / fit().
        """
        d = self._evo_d_
        n = len(self._evo_norms_sq_)
        trajectories   = np.zeros((len(t_values), n, d), dtype=np.float64)
        entropy_values = np.zeros(len(t_values), dtype=np.float64)

        for ti, t in enumerate(t_values):
            phase  = np.exp(-1j * t * self._evo_Omega_)
            psi_ev = self._evo_P_eig_ @ (phase[:, None] * self._evo_Pdag_psi_)

            pos = np.zeros((n, d))
            for kk in range(d):
                Xpsi      = self._evo_X_tr_[kk] @ psi_ev
                pos[:, kk] = np.real(
                    np.sum(psi_ev.conj() * Xpsi, axis=0)) / self._evo_norms_sq_
            trajectories[ti] = pos

            # Matrix-based reverse von Neumann entropy (N_step density matrix)
            psi_pos = pos.T   # (d, n)
            entropy = 0.0
            for kk in range(d):
                v   = psi_pos[kk]
                Nv  = np.real(psi_ev.T @ (psi_ev @ v))
                NNv = np.real(psi_ev.T @ (psi_ev @ Nv))
                p   = float(np.dot(v, NNv))
                if p > 1e-300:
                    entropy += -(1.0 / p) * np.log(1.0 / p)
            entropy_values[ti] = entropy

        return trajectories, entropy_values

    def extract_clusters_kmeans(self, positions, k):
        """
        Run k-means on positions (n, d) and return integer label array (n,).
        """
        return KMeans(n_clusters=k, random_state=42,
                      n_init=50, max_iter=500).fit_predict(positions)

    def compute_euclidean_distances(self, positions, ref_idx=0):
        """
        Return Euclidean distances from positions[ref_idx] to all points. Shape (n,).
        """
        ref = positions[ref_idx]
        return np.sqrt(np.sum((positions - ref) ** 2, axis=-1))

    def compute_dynamic_distance_matrix(self, positions):
        """
        Return full n×n pairwise Euclidean distance matrix. Shape (n, n).
        """
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def _mpd_entropy_deprecated(self, t):
        """
        Deprecated — use entropy_values returned by trajectory computation
        instead (see get_trajectories_with_entropy / _build_and_evolve).

        Computed as 1 - MPD(t)/MPD(0), where MPD is the mean pairwise distance
        between particle positions.  This is NOT the matrix-based von Neumann
        entropy and produces a noisy/flat signal — do not use for t* selection.
        """
        pos  = self.get_positions_at_t(t)
        pos0 = self.get_positions_at_t(0.0)
        idx  = np.triu_indices(len(pos), k=1)

        def _mpd(p):
            d = p[idx[0]] - p[idx[1]]
            return float(np.mean(np.sqrt(np.sum(d ** 2, axis=-1))))

        mpd0 = _mpd(pos0)
        if mpd0 < 1e-12:
            return 0.0
        return float(1.0 - _mpd(pos) / mpd0)

    def plot_entropy(self):
        """
        Plot the stopping criterion curve vs frame, marking the best frame.

        Shows reverse von Neumann entropy S(t) if it varies (range > 0.01),
        otherwise falls back to mean pairwise distance (MPD).  Best frame is
        marked at the PEAK for entropy (argmax) or the TROUGH for MPD (argmin).
        Burn-in region (first 20% of frames) is shaded grey.
        """
        fig, ax  = plt.subplots(figsize=(10, 4))
        burn_in  = int(0.20 * self.T)
        frames   = np.arange(self.T)
        use_ent  = getattr(self, '_stopping_criterion_', 'mpd') == 'entropy'

        if use_ent:
            ax.plot(frames, self.entropy_values_,
                    color='steelblue', linewidth=1.5, label='Reverse entropy S(t)')
            ax.axvspan(0, burn_in, alpha=0.15, color='gray', label='burn-in (20%)')
            ax.axvline(x=self.best_frame_, color='red', linestyle='--', linewidth=2,
                       label=f'Best frame (peak): {self.best_frame_ + 1}')
            ax.set_ylabel('Reverse Entropy  -(1/p) log(1/p)')
            ax.set_title(f'DQC Reverse Entropy  (sigma={self.sigma_:.4f},'
                         f' criterion=entropy)')
        else:
            ax.plot(frames, self.mpd_values_,
                    color='darkorange', linewidth=1.5, label='MPD(t)')
            ax.axvspan(0, burn_in, alpha=0.15, color='gray', label='burn-in (20%)')
            ax.axvline(x=self.best_frame_, color='red', linestyle='--', linewidth=2,
                       label=f'Best frame (trough): {self.best_frame_ + 1}')
            ax.set_ylabel('Mean Pairwise Distance')
            ax.set_title(f'DQC MPD Convergence  (sigma={self.sigma_:.4f},'
                         f' criterion=mpd [entropy flat])')

        ax.set_xlabel('Frame')
        ax.legend()
        plt.tight_layout()
        plt.show()
