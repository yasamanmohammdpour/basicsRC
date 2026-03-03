# =============================================================================
# Task 6: Hyperparameter Optimization for Reservoir Computing
# =============================================================================
# System   : Lorenz (from Task 4)
# Methods  : Random Search  +  Bayesian Optimization
# Reference: Chapter 4 (Prof. Lai)  +  Zheng-Meng GitHub
#
# Reused exactly from Tasks 1-4:
#   - Lorenz ODE  (sigma=10, rho=28, beta=2.67)
#   - Fixed-step RK4 integrator
#   - Z-score normalization
#   - Reservoir state update equation
#   - Ridge regression for W_out
#
# New in Task 6:
#   - Validation segment (teacher-forced one-step prediction error)
#   - Automated search loop over hyperparameters
#   - Even-element squaring  (Zheng-Meng MATLAB + Chapter 4 Sec. 4.2.2)
#   - Multiple random-seed repetitions per evaluation  (Zheng-Meng)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import networkx as nx
from bayes_opt import BayesianOptimization
import warnings
import time
import pickle
import os

warnings.filterwarnings("ignore")

# Output directory
OUT_DIR = "task6_results"
os.makedirs(OUT_DIR, exist_ok=True)

def out(fname):
    return os.path.join(OUT_DIR, fname)

# =============================================================================
# 1.  LORENZ SYSTEM  — identical to Task 1 / Task 4
# =============================================================================

def lorenz_ode(state, sigma=10.0, rho=28.0, beta=2.67):
    """Lorenz equations. Parameters exactly as used in Task 4."""
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])


def rk4_step(f, state, dt):
    """Fixed-step RK4. Identical to Task 4 implementation."""
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def generate_lorenz(n_steps, dt=0.01, warmup=5000):
    """
    Generate n_steps of Lorenz time series.
    Discards 'warmup' steps to eliminate initial transient (same as Task 4).
    """
    np.random.seed(42)
    x0 = np.array([
        np.random.uniform(0.5, 0.8),
        np.random.uniform(0.0, 0.3),
        np.random.uniform(8.0, 10.0)
    ])
    total = warmup + n_steps
    ts = np.zeros((total, 3))
    ts[0] = x0
    for i in range(total - 1):
        ts[i + 1] = rk4_step(lorenz_ode, ts[i], dt)
    return ts[warmup:]   # discard warmup, return exactly n_steps rows


# =============================================================================
# 2.  RESERVOIR CONSTRUCTION  — extends Task 3 / Task 4
# =============================================================================

def build_reservoir(N, d, eig_rho, gamma, dim, seed=None):
    """
    Build W_in and A for one reservoir instance.

    W_in : N x dim,  entries ~ Uniform(-gamma, +gamma)
    A    : N x N,    Erdos-Renyi undirected (symmetric), Gaussian weights,
                     rescaled so spectral_radius(A) = eig_rho

    Using undirected graph → symmetric A → real-only eigenvalues.
    Same approach as Task 3 and Zheng-Meng Python code.
    """
    rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(0, 1_000_000)
    )

    # Input matrix
    Win = rng.uniform(-gamma, gamma, (N, dim))

    # Reservoir adjacency matrix
    G = nx.erdos_renyi_graph(N, d, seed=int(rng.randint(0, 1_000_000)),
                             directed=False)
    A = np.zeros((N, N))
    for (u, v) in G.edges():
        w = rng.normal(0.0, 1.0)
        A[u, v] = w
        A[v, u] = w          # symmetric

    # Rescale spectral radius
    rho_actual = np.max(np.abs(np.linalg.eigvals(A)))
    if rho_actual > 1e-12:
        A *= (eig_rho / rho_actual)

    return Win, A


# =============================================================================
# 3.  TRAINING  — ridge regression for W_out
# =============================================================================

def train_rc(ts_norm, N, d, eig_rho, gamma, alpha, beta, noise_a,
             washout, train_len, seed=None):
    """
    Train reservoir computer for one hyperparameter set.

    Steps:
      1. Build fresh W_in, A
      2. Add optional noise to training input  (Zheng-Meng stochastic resonance)
      3. Run leaky-integrator state update for (washout + train_len) steps
      4. Discard washout states
      5. Even-element squaring  — breaks mirror-attractor symmetry
         (Zheng-Meng MATLAB:  r_out(2:2:end,:) = r_out(2:2:end,:).^2
          Chapter 4 Sec. 4.2.2)
      6. Ridge regression → W_out  (Eq. 4-17 of Chapter 4)

    Returns: Wout, r_end, Win, A
      r_end is the final reservoir state — passed directly to validation
      so the state is continuous across train / validate / test.
    """
    dim = ts_norm.shape[1]
    Win, A = build_reservoir(N, d, eig_rho, gamma, dim, seed=seed)
    rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(0, 1_000_000)
    )

    total = washout + train_len

    # Noisy input; clean targets
    noise   = noise_a * rng.randn(total, dim)
    train_x = (ts_norm[:total] + noise).T       # dim x total
    train_y = ts_norm[1:total + 1].T            # dim x total  (one step ahead)

    # Reservoir state update
    # r(t+1) = (1-alpha)*r(t) + alpha*tanh(A @ r(t) + Win @ u(t))
    r_all = np.zeros((N, total + 1))
    for t in range(total):
        r_all[:, t + 1] = (
            (1.0 - alpha) * r_all[:, t]
            + alpha * np.tanh(A @ r_all[:, t] + Win @ train_x[:, t])
        )

    r_out = r_all[:, washout + 1:]       # N x train_len  (washout discarded)
    r_end = r_all[:, -1:].copy()         # N x 1

    # Even-element squaring (mirror-attractor fix)
    r_aug = r_out.copy()
    r_aug[1::2, :] = r_aug[1::2, :] ** 2   # square neurons at indices 1,3,5,...

    y_train = train_y[:, washout:]       # dim x train_len

    # W_out = Y R^T (R R^T + beta I)^{-1}
    Wout = (y_train @ r_aug.T
            @ np.linalg.inv(r_aug @ r_aug.T + beta * np.eye(N)))

    return Wout, r_end, Win, A


# =============================================================================
# 4.  TEACHER-FORCED VALIDATION  — core of Task 6
# =============================================================================

def validate_tf(ts_norm, Wout, r_end, Win, A, alpha,
                washout, train_len, val_len):
    """
    Teacher-forced one-step prediction error on the validation segment.

    This is exactly what Task 6 describes:
      "the reservoir continues to receive the input signal [teacher-forced],
       but the output matrix W_out remains fixed to generate one-step
       predictions. The average one-step prediction error is calculated."

    At every step:
      1. Feed TRUE input u(t) into reservoir  (not its own prediction)
      2. Compute ŷ(t+1) = W_out @ r_aug(t+1)
      3. Compare with true u(t+1)

    Returns normalised RMSE  (Eq. 4-20, Chapter 4):
        RMSE = sqrt(mean ||ŷ - u||^2) / std(u)
    Dividing by std(u) makes RMSE dimensionless and comparable across systems.
    """
    N         = A.shape[0]
    dim       = ts_norm.shape[1]
    r         = r_end.copy()
    val_start = washout + train_len

    preds = np.zeros((val_len, dim))
    trues = np.zeros((val_len, dim))

    for t in range(val_len):
        idx = val_start + t
        u   = ts_norm[idx].reshape(-1, 1)   # TRUE input (teacher force)

        r = ((1.0 - alpha) * r
             + alpha * np.tanh(A @ r + Win @ u))

        r_aug = r.copy()
        r_aug[1::2] = r_aug[1::2] ** 2      # same squaring as training

        preds[t] = (Wout @ r_aug).ravel()
        trues[t] = ts_norm[min(idx + 1, len(ts_norm) - 1)]

    sigma_y = np.std(trues) + 1e-12
    rmse    = np.sqrt(np.mean((preds - trues) ** 2)) / sigma_y
    return float(rmse)


# =============================================================================
# 5.  GLOBAL CONSTANTS
# =============================================================================

N_RES    = 200    # reservoir size  (same as Task 4 Lorenz)
DIM      = 3      # Lorenz dimension

WASHOUT  = 1000   # washout steps  (Task 4 used washout=1000)
TRAIN    = 8000   # training steps
VAL      = 2000   # validation steps  (NEW in Task 6)
TEST     = 3000   # test steps  (final closed-loop evaluation)

TOTAL    = WASHOUT + TRAIN + VAL + TEST + 500   # +500 buffer

# Robustness: run ITER_TIME independent reservoirs per hyperparameter set,
# keep best KEEP_FRAC of them, average.  (Zheng-Meng: 16 runs, best 10)
ITER_TIME = 5
KEEP_FRAC = 0.8

# Hyperparameter search bounds  (same as Zheng-Meng opt_lorenz.py)
PBOUNDS = {
    "d":         (0.01, 0.30),   # was 0.40 — high d → dense reservoir, overfits
    "eig_rho":   (2.00, 5.00),   # was 0.10 — very low rho → dead reservoir
    "gamma":     (0.01, 5.00),
    "alpha":     (0.01, 0.95),   # was 1.00 — alpha=1 means zero memory
    "beta_log":  (-7.0, -1.0),
    "noise_log": (-7.0, -1.0),
}

# Task 4 hand-tuned Lorenz hyperparameters  (our baseline)
HAND = {
    "d":         0.027,
    "eig_rho":   4.78,
    "gamma":     0.27,
    "alpha":     0.41,
    "beta_log":  -3.36,
    "noise_log": -7.0,
}


# =============================================================================
# 6.  GENERATE DATA  (once; reused by all evaluations)
# =============================================================================

print("=" * 65)
print("Task 6 — Hyperparameter Optimization (Lorenz)")
print("=" * 65)
print(f"\nGenerating Lorenz data  ({TOTAL} steps, dt=0.01) ...")

ts_raw = generate_lorenz(TOTAL, dt=0.01, warmup=5000)

# Z-score normalization computed on training segment only (no data leakage)
scaler   = StandardScaler()
scaler.fit(ts_raw[:WASHOUT + TRAIN])
TS_NORM  = scaler.transform(ts_raw)

print(f"  Shape: {TS_NORM.shape}")
print(f"  Split: washout={WASHOUT}  train={TRAIN}  val={VAL}  test={TEST}\n")


# =============================================================================
# 7.  OBJECTIVE FUNCTION  (shared by both methods)
# =============================================================================

_count = [0]   # evaluation counter

def objective(d, eig_rho, gamma, alpha, beta_log, noise_log):
    """
    Evaluate one hyperparameter configuration.

    Runs ITER_TIME independent reservoir instances with different random seeds.
    Keeps the best KEEP_FRAC of resulting RMSEs and averages them.
    Returns  1 / mean_rmse  because bayesian-optimization MAXIMISES.

    Why multiple runs?
    Because W_in and A are random — a single run can get lucky/unlucky.
    Averaging several stabilises the GP surrogate's training signal.
    """
    _count[0] += 1
    beta    = 10.0 ** beta_log
    noise_a = 10.0 ** noise_log

    rmse_list = []
    for seed_i in range(ITER_TIME):
        try:
            Wout, r_end, Win, A = train_rc(
                TS_NORM, N_RES, d, eig_rho, gamma, alpha, beta, noise_a,
                WASHOUT, TRAIN, seed=seed_i * 137 + 7
            )
            rmse = validate_tf(
                TS_NORM, Wout, r_end, Win, A, alpha,
                WASHOUT, TRAIN, VAL
            )
            if np.isfinite(rmse) and rmse > 0:
                rmse_list.append(rmse)
        except Exception:
            pass   # bad hyperparams → numerical failure → skip

    if not rmse_list:
        return 1e-12

    rmse_list.sort()
    keep      = max(1, int(np.ceil(KEEP_FRAC * len(rmse_list))))
    mean_rmse = float(np.mean(rmse_list[:keep]))

    print(f"  [{_count[0]:4d}]  RMSE={mean_rmse:.5f}  "
          f"d={d:.3f}  rho={eig_rho:.3f}  gamma={gamma:.3f}  "
          f"alpha={alpha:.3f}  beta=10^{beta_log:.2f}  "
          f"noise=10^{noise_log:.2f}")

    return 1.0 / (mean_rmse + 1e-12)


# =============================================================================
# 8.  RANDOM SEARCH
# =============================================================================

def random_search(n_iter=60):
    """
    Random Search.

    At each of n_iter iterations:
      1. Draw each hyperparameter uniformly from its range
      2. Evaluate the objective
      3. Keep track of the best configuration found so far

    No memory between iterations — purely random sampling.
    Simple but competitive when good regions are not too small.
    """
    print("\n" + "=" * 65)
    print(f"RANDOM SEARCH  ({n_iter} iterations)")
    print("=" * 65 + "\n")

    best_score  = -np.inf
    best_params = None
    history     = []
    t0 = time.time()

    for it in range(n_iter):
        print(f"--- Random Search iteration {it+1}/{n_iter} ---")

        # Sample uniformly from each range
        params = {k: float(np.random.uniform(lo, hi))
                  for k, (lo, hi) in PBOUNDS.items()}

        score = objective(**params)
        rmse  = 1.0 / score if score > 1e-11 else np.inf

        history.append({"params": params.copy(), "rmse": rmse})

        if score > best_score:
            best_score  = score
            best_params = params.copy()
            print(f"  *** New best!  RMSE = {rmse:.5f} ***")

    elapsed   = time.time() - t0
    best_rmse = 1.0 / best_score if best_score > 1e-11 else np.inf

    print(f"\nRandom Search done in {elapsed:.1f}s")
    print(f"Best RMSE : {best_rmse:.6f}")
    _print_params(best_params)
    return best_params, best_rmse, history


# =============================================================================
# 9.  BAYESIAN OPTIMIZATION
# =============================================================================

def bayesian_opt(n_init=10, n_iter=50):
    """
    Bayesian Optimization with Gaussian Process surrogate.

    Algorithm (Chapter 4 Sec. 4.5):
      Phase 1 — Initialization (n_init random evaluations)
        Evaluate n_init random hyperparameter sets.
        Fit the initial GP to these observations.

      Phase 2 — Guided search (n_iter iterations)
        Repeat:
          a. Fit GP to all observed (theta, score) pairs
          b. Maximize Expected Improvement (EI) acquisition → next theta*
             EI(theta) = (Q* - mu_GP) * Phi(Z) + sigma_GP * phi(Z)
             where Z = (Q* - mu_GP) / sigma_GP
             First term = exploitation (predict better than best so far)
             Second term = exploration (high uncertainty regions)
          c. Evaluate objective at theta*
          d. Update observed dataset

    Why it outperforms random search:
      Each evaluation informs the GP.  EI concentrates future evaluations
      in the promising region of the search space.  Typically reaches
      the same RMSE as random search in 3-5x fewer evaluations.
      (Chapter 4 Figs. 4-5, 4-6)

    Uses bayesian-optimization package (same as Zheng-Meng opt_lorenz.py).
    NOTE: this package MAXIMISES, so we return 1/rmse from objective().
    """
    print("\n" + "=" * 65)
    print(f"BAYESIAN OPTIMIZATION  ({n_init} init + {n_iter} iterations)")
    print("=" * 65 + "\n")

    t0   = time.time()
    opt  = BayesianOptimization(f=objective, pbounds=PBOUNDS,
                                random_state=42, verbose=0)

    print("--- Phase 1: random initialization ---")
    opt.maximize(init_points=n_init, n_iter=0)

    print("\n--- Phase 2: GP-guided Bayesian search ---")
    opt.maximize(init_points=0, n_iter=n_iter)

    elapsed     = time.time() - t0
    best_params = opt.max["params"]
    best_rmse   = 1.0 / opt.max["target"]

    print(f"\nBayesian Optimization done in {elapsed:.1f}s")
    print(f"Best RMSE : {best_rmse:.6f}")
    _print_params(best_params)
    return best_params, best_rmse, opt


def _print_params(p):
    for k, v in p.items():
        print(f"  {k:<12} = {v:.5f}")


# =============================================================================
# 10.  FINAL EVALUATION  — closed-loop prediction on test segment
# =============================================================================

def final_eval(params, label=""):
    """
    Evaluate best hyperparameters on the held-out test segment.

    Steps:
      1. Train W_out  (seed=0 for reproducibility)
      2. Teacher-force through validation window to advance reservoir state
         (ensures state is continuous — same trick as Zheng-Meng)
      3. Closed-loop (autonomous) prediction:
         feed own output back as next input  (Chapter 4 Sec. 4.2.4)

    Returns: val_rmse, test_pred, test_true
    """
    beta    = 10.0 ** params["beta_log"]
    noise_a = 10.0 ** params["noise_log"]

    print(f"\n--- Final evaluation: {label} ---")

    Wout, r_end, Win, A = train_rc(
        TS_NORM, N_RES,
        params["d"], params["eig_rho"], params["gamma"],
        params["alpha"], beta, noise_a,
        WASHOUT, TRAIN, seed=0
    )

    val_rmse = validate_tf(
        TS_NORM, Wout, r_end, Win, A, params["alpha"],
        WASHOUT, TRAIN, VAL
    )
    print(f"  Validation RMSE = {val_rmse:.6f}")

    # Advance reservoir state through validation window (teacher-forced)
    r = r_end.copy()
    for t in range(VAL):
        idx = WASHOUT + TRAIN + t
        u   = TS_NORM[idx].reshape(-1, 1)
        r   = ((1.0 - params["alpha"]) * r
               + params["alpha"] * np.tanh(A @ r + Win @ u))
        ra  = r.copy(); ra[1::2] = ra[1::2] ** 2   # keep state advancing

    # Closed-loop prediction on test segment
    test_start = WASHOUT + TRAIN + VAL
    test_true  = TS_NORM[test_start: test_start + TEST]
    test_pred  = np.zeros((TEST, DIM))

    u = TS_NORM[test_start].reshape(-1, 1)   # seed with first true state
    for t in range(TEST):
        r  = ((1.0 - params["alpha"]) * r
              + params["alpha"] * np.tanh(A @ r + Win @ u))
        ra = r.copy(); ra[1::2] = ra[1::2] ** 2
        p  = Wout @ ra
        test_pred[t] = p.ravel()
        u  = p                               # feed own prediction back

    return val_rmse, test_pred, test_true


# =============================================================================
# 11.  VISUALIZATION
# =============================================================================

def plot_short_term(pred, true, label, val_rmse, n=500):
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(f"Task 6 — Lorenz Short-Term Prediction  [{label}]\n"
                 f"Validation RMSE = {val_rmse:.5f}",
                 fontsize=13, fontweight="bold")
    for i, var in enumerate(["x", "y", "z"]):
        axes[i].plot(true[:n, i],  "r-",  lw=1.8, label="True")
        axes[i].plot(pred[:n, i],  "b--", lw=1.4, label="Predicted")
        axes[i].set_ylabel(var, fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].grid(alpha=0.25)
    axes[2].set_xlabel("Time step", fontsize=11)
    fname = out(f"task6_short_{label.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved {fname}")
    plt.show()


def plot_attractor(pred, true, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(f"Task 6 — Lorenz Attractor Reconstruction  [{label}]",
                 fontsize=13, fontweight="bold")
    ax1.plot(true[:, 0], true[:, 2],  "r-", lw=0.35, alpha=0.7)
    ax1.set_title("True  (x–z)"); ax1.set_xlabel("x"); ax1.set_ylabel("z")
    ax2.plot(pred[:, 0], pred[:, 2],  "b-", lw=0.35, alpha=0.7)
    ax2.set_title("Predicted  (x–z)"); ax2.set_xlabel("x"); ax2.set_ylabel("z")
    fname = out(f"task6_attractor_{label.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved {fname}")
    plt.show()


def plot_convergence(rs_history, bo_opt):
    # Random search: best-so-far curve
    rs_curve, best = [], np.inf
    for h in rs_history:
        best = min(best, h["rmse"])
        rs_curve.append(best)

    # Bayesian: best-so-far curve
    bo_curve, best = [], np.inf
    for res in bo_opt.res:
        r = 1.0 / res["target"] if res["target"] > 1e-11 else np.inf
        best = min(best, r)
        bo_curve.append(best)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.semilogy(rs_curve, "r-o", ms=4, lw=1.5,
                label=f"Random Search  (final={rs_curve[-1]:.5f})")
    ax.semilogy(bo_curve, "b-s", ms=4, lw=1.5,
                label=f"Bayesian Opt.  (final={bo_curve[-1]:.5f})")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best Validation RMSE (log)")
    ax.set_title("Task 6 — Convergence Comparison",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3, which="both")
    plt.savefig(out("task6_convergence.png"), dpi=150, bbox_inches="tight")
    print("  Saved task6_convergence.png")
    plt.show()


def plot_bar(hand_rmse, rs_rmse, bo_rmse):
    methods = ["By Hand\n(Task 4)", "Random\nSearch", "Bayesian\nOpt."]
    values  = [hand_rmse, rs_rmse, bo_rmse]
    colors  = ["#d62728", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    bars = ax.bar(methods, values, color=colors, edgecolor="k", width=0.45)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values)*0.015,
                f"{v:.5f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation RMSE"); ax.set_ylim(0, max(values)*1.35)
    ax.set_title("Task 6 — Method Comparison (Lorenz)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.savefig(out("task6_bar.png"), dpi=150, bbox_inches="tight")
    print("  Saved task6_bar.png")
    plt.show()


def print_table(hand_p, rs_p, bo_p, hand_r, rs_r, bo_r):
    print("\n" + "="*72)
    print("TASK 6 — HYPERPARAMETER COMPARISON TABLE")
    print("="*72)
    print(f"{'Hyperparameter':<22} {'By Hand (Task4)':>16} "
          f"{'Random Search':>14} {'Bayesian Opt':>13}")
    print("-"*72)
    rows = [
        ("rho (spectral radius)", "eig_rho"),
        ("gamma (input scaling)", "gamma"),
        ("alpha (leakage rate)",  "alpha"),
        ("beta  (log10)",         "beta_log"),
        ("noise (log10)",         "noise_log"),
        ("d     (conn. prob.)",   "d"),
    ]
    for name, key in rows:
        print(f"  {name:<20} {hand_p[key]:>16.4f} "
              f"{rs_p[key]:>14.4f} {bo_p[key]:>13.4f}")
    print("-"*72)
    print(f"  {'Validation RMSE':<20} {hand_r:>16.6f} "
          f"{rs_r:>14.6f} {bo_r:>13.6f}")
    print("="*72)


# =============================================================================
# 12.  MAIN
# =============================================================================

if __name__ == "__main__":

    # Budget — reduce for a quick test run
    N_RS_ITER   = 60    # random search iterations
    N_BO_INIT   = 10    # Bayesian: random warm-up points
    N_BO_ITER   = 50    # Bayesian: GP-guided iterations

    np.random.seed(42)

    # ── Step 0: Task 4 hand-tuned baseline ───────────────────────────────────
    print("\n" + "="*65)
    print("STEP 0 — Task 4 hand-tuned baseline")
    print("="*65)
    hand_rmse, hand_pred, base_true = final_eval(HAND, "Task4 Hand-Tuned")

    # ── Step 1: Random Search ─────────────────────────────────────────────────
    rs_params, rs_rmse, rs_hist = random_search(n_iter=N_RS_ITER)

    # ── Step 2: Bayesian Optimization ────────────────────────────────────────
    bo_params, bo_rmse, bo_opt = bayesian_opt(n_init=N_BO_INIT, n_iter=N_BO_ITER)

    # ── Step 3: Final closed-loop test ───────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 3 — Final closed-loop test evaluation")
    print("="*65)
    rs_val_rmse, rs_pred,  test_true = final_eval(rs_params, "Random Search")
    bo_val_rmse, bo_pred,  test_true = final_eval(bo_params, "Bayesian Opt")

    # ── Step 4: Summary table ─────────────────────────────────────────────────
    print_table(HAND, rs_params, bo_params,
                hand_rmse, rs_val_rmse, bo_val_rmse)

    # ── Step 5: Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots ...")

    plot_short_term(hand_pred, base_true, "Task4 Hand-Tuned", hand_rmse)
    plot_short_term(rs_pred,   test_true,  "Random Search",   rs_val_rmse)
    plot_short_term(bo_pred,   test_true,  "Bayesian Opt",    bo_val_rmse)

    plot_attractor(hand_pred, base_true, "Task4 Hand-Tuned")
    plot_attractor(rs_pred,   test_true,  "Random Search")
    plot_attractor(bo_pred,   test_true,  "Bayesian Opt")

    plot_convergence(rs_hist, bo_opt)
    plot_bar(hand_rmse, rs_val_rmse, bo_val_rmse)

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    with open(out("task6_results.pkl"), "wb") as f:
        pickle.dump({
            "hand":   {"params": HAND,      "rmse": hand_rmse},
            "rs":     {"params": rs_params, "rmse": rs_val_rmse, "history": rs_hist},
            "bayes":  {"params": bo_params, "rmse": bo_val_rmse},
            "preds":  {"hand": hand_pred, "rs": rs_pred,
                       "bayes": bo_pred,  "true": test_true},
        }, f)
    print("\nResults saved to  task6_results.pkl")
    print("Task 6 complete.")