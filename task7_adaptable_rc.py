# =============================================================================
# Task 7: Training an Adaptable Reservoir Computer
# =============================================================================
# System    : Lorenz  (bifurcation parameter = rho)
# Chapter   : Sec. 8.2  — Adaptable / Parameter-Adaptable Reservoir Computing
#
# Architecture change vs Task 4/6 (Eq. 8-16):
#   r(t+1) = (1-α)r(t) + α·tanh(A·r(t) + Win·u(t) + ρp·Wp·(p + pb))
#   where:
#     Wp  : N×1  random matrix  (parameter-input / control channel)
#     ρp  : scalar scaling factor for the parameter channel
#     pb  : scalar parameter bias
#     p   : current bifurcation parameter value (scalar, injected as constant)
#
# Reused exactly from Task 4/6:
#   - Lorenz ODE (sigma=10, rho=variable, beta=2.67)
#   - Fixed-step RK4 integrator
#   - Z-score normalisation  (fit on training window, no data leakage)
#   - Ridge regression  for W_out
#   - Even-element squaring  (mirror-attractor fix)
#   - Optimal hyperparameters from Task 6  (HAND set)
#
# New in Task 7:
#   - Wp  parameter channel
#   - Joint training across K=3 parameter values
#   - Average Lorenz oscillation period → T in "number of cycles"
#   - Training-error vs T sweep  (T = 100…1000 cycles, step 100)
#   - 3-curve comparison  (one curve per ρ value)
#   - Curve-fit to determine functional form  (linear / power / exponential)
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — saves PNGs without display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress
import networkx as nx
import warnings, time, pickle, os

warnings.filterwarnings("ignore")

# Output directory — all PNGs and PKL saved here
OUT_DIR = "task7_results"
os.makedirs(OUT_DIR, exist_ok=True)

def out(fname):
    """Returns full path inside OUT_DIR."""
    return os.path.join(OUT_DIR, fname)


# =============================================================================
# 1.  LORENZ WITH VARIABLE RHO  —  identical ODE, variable parameter
# =============================================================================

def lorenz_ode(state, sigma=10.0, rho=28.0, beta=2.67):
    x, y, z = state
    return np.array([sigma*(y - x), x*(rho - z) - y, x*y - beta*z])


def rk4_step(f, state, dt):
    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def generate_lorenz(n_steps, dt=0.01, rho=28.0, warmup=5000, seed=42):
    """
    Generate n_steps Lorenz time-series at a specific rho.
    Identical warm-up / RK4 procedure as Tasks 1–6.
    """
    np.random.seed(seed)
    x0 = np.array([np.random.uniform(0.5, 0.8),
                   np.random.uniform(0.0, 0.3),
                   np.random.uniform(8.0, 10.0)])
    total = warmup + n_steps
    ts = np.zeros((total, 3))
    ts[0] = x0
    ode = lambda s: lorenz_ode(s, rho=rho)
    for i in range(total - 1):
        ts[i+1] = rk4_step(ode, ts[i], dt)
    return ts[warmup:]


# =============================================================================
# 2.  AVERAGE OSCILLATION PERIOD  (in time-steps)
# =============================================================================

def average_period_steps(rho, dt=0.01, warmup=5000, probe_steps=50000):
    """
    Estimate the average oscillation period for Lorenz at a given rho.
    Uses z-coordinate local maxima (peaks).
    Returns period in number of dt-steps (integer).
    """
    ts = generate_lorenz(probe_steps, dt=dt, rho=rho, warmup=warmup, seed=0)
    z = ts[:, 2]
    peaks, _ = find_peaks(z, height=np.mean(z))
    if len(peaks) < 2:
        return 100   # fallback
    diffs = np.diff(peaks)
    return int(np.round(np.mean(diffs)))


# =============================================================================
# 3.  RESERVOIR CONSTRUCTION
# =============================================================================

def build_reservoir(N, d, eig_rho, gamma, dim, rho_p, seed=None):
    """
    Build Win, A, Wp for one adaptable reservoir instance.

    Win  : N × dim   Uniform(-gamma, +gamma)
    A    : N × N     Erdős–Rényi undirected, Gaussian weights,
                     rescaled to spectral radius = eig_rho
    Wp   : N × 1     Uniform(-1, +1)  —  parameter / control channel
    """
    rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(0, 1_000_000)
    )
    Win = rng.uniform(-gamma, gamma, (N, dim))

    G = nx.erdos_renyi_graph(N, d, seed=int(rng.randint(0, 1_000_000)),
                             directed=False)
    A = np.zeros((N, N))
    for (u, v) in G.edges():
        w = rng.normal(0.0, 1.0)
        A[u, v] = w;  A[v, u] = w

    rho_actual = np.max(np.abs(np.linalg.eigvals(A)))
    if rho_actual > 1e-12:
        A *= (eig_rho / rho_actual)

    Wp = rng.uniform(-1.0, 1.0, (N, 1))
    return Win, A, Wp


# =============================================================================
# 4.  ADAPTABLE RC  —  TRAINING
# =============================================================================

def train_adaptable_rc(segments, N, d, eig_rho, gamma, alpha, beta,
                       noise_a, rho_p, pb, washout, seed=None):
    """
    Train an adaptable reservoir computer on multiple (time-series, p) segments.

    Parameters
    ----------
    segments : list of (ts_norm, p_value)
        Each entry is a normalised time-series array (T×dim) paired with its
        scalar bifurcation-parameter value p.
    washout  : int
        Washout steps discarded at the start of *each* segment.
    Returns
    -------
    Wout : dim × N   output weight matrix
    Win, A, Wp       fixed random matrices (same for every segment)
    """
    dim    = segments[0][0].shape[1]
    Win, A, Wp = build_reservoir(N, d, eig_rho, gamma, dim, rho_p, seed=seed)
    rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(0, 1_000_000)
    )

    R_list, Y_list = [], []   # collect post-washout states and targets

    for (ts_norm, p_val) in segments:
        T_seg = len(ts_norm) - 1          # number of one-step-ahead targets

        # noisy input
        noise  = noise_a * rng.randn(T_seg, dim)
        inp    = (ts_norm[:T_seg] + noise).T   # dim × T_seg
        target = ts_norm[1:T_seg + 1].T        # dim × T_seg

        # parameter contribution (constant per segment)
        p_contrib = rho_p * Wp * (p_val + pb)  # N × 1  (broadcast over time)

        r = np.zeros((N, 1))
        r_all = np.zeros((N, T_seg))
        for t in range(T_seg):
            r = ((1.0 - alpha) * r
                 + alpha * np.tanh(A @ r + Win @ inp[:, t:t+1] + p_contrib))
            r_all[:, t] = r[:, 0]

        # discard washout
        r_train = r_all[:, washout:]
        y_train = target[:, washout:]

        # even-element squaring (mirror-attractor fix)
        r_aug = r_train.copy()
        r_aug[1::2, :] = r_aug[1::2, :] ** 2

        R_list.append(r_aug)
        Y_list.append(y_train)

    R_all = np.hstack(R_list)       # N × total_train
    Y_all = np.hstack(Y_list)       # dim × total_train

    # Ridge regression  W_out = Y R^T (R R^T + β I)^{-1}
    Wout = (Y_all @ R_all.T
            @ np.linalg.inv(R_all @ R_all.T + beta * np.eye(N)))
    return Wout, Win, A, Wp


# =============================================================================
# 5.  TRAINING ERROR  (teacher-forced RMSE on one segment)
# =============================================================================

def training_error(ts_norm, p_val, Wout, Win, A, Wp,
                   alpha, rho_p, pb, washout, noise_a=0.0):
    """
    Compute teacher-forced one-step-ahead normalised RMSE on a single segment.

    The state is driven by the TRUE input u(t) at every step (open-loop).
    RMSE is normalised by std(target)  →  dimensionless metric.
    """
    N    = A.shape[0]
    dim  = ts_norm.shape[1]
    T    = len(ts_norm) - 1

    p_contrib = rho_p * Wp * (p_val + pb)   # N × 1

    r       = np.zeros((N, 1))
    preds   = np.zeros((T, dim))
    targets = ts_norm[1:T+1]

    for t in range(T):
        u = ts_norm[t:t+1].T            # dim × 1  (TRUE input)
        r = ((1.0 - alpha) * r
             + alpha * np.tanh(A @ r + Win @ u + p_contrib))
        ra = r.copy()
        ra[1::2] = ra[1::2] ** 2
        preds[t] = (Wout @ ra).ravel()

    # use only post-washout portion
    preds   = preds[washout:]
    targets = targets[washout:]

    sigma   = np.std(targets) + 1e-12
    rmse    = np.sqrt(np.mean((preds - targets)**2)) / sigma
    return float(rmse)


# =============================================================================
# 6.  GLOBAL SETTINGS
# =============================================================================

DT        = 0.01
N_RES     = 200           # reservoir size  (Task 4/6 standard)
WASHOUT   = 500           # washout per segment (reduced for short-T runs)
WARMUP    = 5000          # Lorenz warm-up

# --- Bifurcation parameter  (rho) values ---
# All three are in the fully-developed chaotic regime.
RHO_VALS  = [25.0, 28.0, 31.0]

# --- Load Task-6 optimal hyperparameters from saved results ---
TASK6_PKL = os.path.join("task6_results", "task6_results.pkl")

if os.path.exists(TASK6_PKL):
    with open(TASK6_PKL, "rb") as f:
        task6_data = pickle.load(f)
    # Use Bayesian Optimization result as it gives the best RMSE
    HP = task6_data["best"]["params"]
    # HP['eig_rho'] = 4.78
    print(f"  Source: {task6_data['best']['source']}  "
        f"(RMSE={task6_data['best']['rmse']:.6f})")
else:
    # Fallback to hand-tuned values if pkl not found
    print(f"WARNING: {TASK6_PKL} not found — using hand-tuned fallback values")
    HP = {
        "d":         0.027,
        "eig_rho":   4.78,
        "gamma":     0.27,
        "alpha":     0.41,
        "beta_log":  -3.36,
        "noise_log": -7.0,
    }

print(f"  Hyperparameters: {HP}")

ALPHA   = HP["alpha"]
BETA    = 10.0 ** HP["beta_log"]
NOISE_A = 10.0 ** HP["noise_log"]

# --- Additional adaptable-RC hyperparameters ---
RHO_P  = 0.1    # parameter-channel scaling  (ρp)
PB     = 0.0    # parameter bias

# --- T sweep: 100 to 1000 cycles, step 100 ---
N_CYCLE_VALS = list(range(100, 1001, 100))   # [100, 200, ..., 1000]

print("=" * 65)
print("Task 7 — Adaptable Reservoir Computing (Lorenz, rho channel)")
print("=" * 65)


# =============================================================================
# 7.  ESTIMATE AVERAGE LORENZ PERIOD  (once per rho value)
# =============================================================================

print("\nEstimating average oscillation period for each ρ ...")
period_steps = {}
for rho in RHO_VALS:
    ps = average_period_steps(rho, dt=DT, warmup=WARMUP)
    period_steps[rho] = ps
    print(f"  ρ = {rho:5.1f}  →  avg period ≈ {ps} steps  "
          f"= {ps*DT:.3f} time-units")

# Use mean period across all rho values as a single "cycle" unit
# (so the x-axis is comparable across rho values)
mean_period = int(np.round(np.mean(list(period_steps.values()))))
print(f"\n  Mean period across ρ values: {mean_period} steps = "
      f"{mean_period * DT:.3f} time-units")

# Convert cycles → steps for each T value
T_STEPS = [nc * mean_period for nc in N_CYCLE_VALS]
print(f"\n  T range: {T_STEPS[0]} – {T_STEPS[-1]} steps "
      f"({N_CYCLE_VALS[0]} – {N_CYCLE_VALS[-1]} cycles)")


# =============================================================================
# 8.  NORMALISATION  (fit scaler on the longest training set at each rho)
# =============================================================================
# To avoid data leakage, we Z-score normalise using the training window only.
# We refit the scaler at each T to be precise; in practice the statistics
# stabilise quickly, so results are nearly identical.
#
# For efficiency we pre-generate the maximum-length series at each rho and
# slice it for shorter T values.

T_MAX_STEPS = T_STEPS[-1] + WASHOUT + 500   # +500 buffer

print(f"\nGenerating max-length Lorenz data  ({T_MAX_STEPS} steps per ρ) ...")
ts_raw = {}
for rho in RHO_VALS:
    ts_raw[rho] = generate_lorenz(T_MAX_STEPS, dt=DT, rho=rho,
                                  warmup=WARMUP, seed=int(rho * 10))
    print(f"  ρ = {rho}  shape: {ts_raw[rho].shape}")


def get_normalised_segment(rho, train_steps):
    """
    Return z-scored slice of length (washout + train_steps + 1).
    Scaler fit on [0 : washout + train_steps] only (no leakage).
    """
    needed = WASHOUT + train_steps + 1
    raw    = ts_raw[rho][:needed]
    mu     = raw[:WASHOUT + train_steps].mean(axis=0)
    sigma  = raw[:WASHOUT + train_steps].std(axis=0) + 1e-12
    return (raw - mu) / sigma


# =============================================================================
# 9.  TRAINING ERROR vs T SWEEP
# =============================================================================

print("\n" + "=" * 65)
print("Training-error vs T sweep")
print("=" * 65)

results = {rho: [] for rho in RHO_VALS}  # rho → list of RMSE for each T

for t_idx, (n_cycles, t_steps) in enumerate(zip(N_CYCLE_VALS, T_STEPS)):
    print(f"\n--- T = {n_cycles} cycles  ({t_steps} steps) ---")
    t0 = time.time()

    # Build normalised segments for all three rho values
    segments = []
    seg_norm_dict = {}
    for rho in RHO_VALS:
        seg_norm = get_normalised_segment(rho, t_steps)
        seg_norm_dict[rho] = seg_norm
        # p_val normalised: centre and scale so parameter values are ~O(1)
        # We simply pass the raw rho value; Wp and rho_p handle scaling.
        segments.append((seg_norm, rho))

    # Train adaptable RC on all 3 segments jointly
    try:
        Wout, Win, A, Wp = train_adaptable_rc(
            segments, N_RES,
            HP["d"], HP["eig_rho"], HP["gamma"],
            ALPHA, BETA, NOISE_A, RHO_P, PB,
            WASHOUT, seed=42
        )

        # Evaluate training error per segment
        for rho in RHO_VALS:
            rmse = training_error(
                seg_norm_dict[rho], rho, Wout, Win, A, Wp,
                ALPHA, RHO_P, PB, WASHOUT
            )
            results[rho].append(rmse)
            print(f"  ρ={rho:5.1f}  train RMSE = {rmse:.6f}")

    except Exception as e:
        print(f"  ERROR: {e}")
        for rho in RHO_VALS:
            results[rho].append(np.nan)

    print(f"  Elapsed: {time.time()-t0:.1f}s")


# =============================================================================
# 10.  CURVE FITTING  (power-law:  RMSE ~ a * T^b)
# =============================================================================

print("\n" + "=" * 65)
print("Curve-fit results (power-law:  log RMSE ~ b·log T + log a)")
print("=" * 65)

fit_params = {}
for rho in RHO_VALS:
    rmse_arr = np.array(results[rho])
    valid    = np.isfinite(rmse_arr) & (rmse_arr > 0)
    if valid.sum() < 3:
        fit_params[rho] = (np.nan, np.nan, np.nan)
        continue
    log_T    = np.log(np.array(T_STEPS)[valid])
    log_rmse = np.log(rmse_arr[valid])
    slope, intercept, r_value, p_value, _ = linregress(log_T, log_rmse)
    fit_params[rho] = (slope, np.exp(intercept), r_value**2)
    print(f"  ρ = {rho:5.1f}:  RMSE ≈ {np.exp(intercept):.4e} × T^({slope:.3f})"
          f"    R² = {r_value**2:.4f}")


# =============================================================================
# 11.  PLOTS
# =============================================================================

colors  = {25.0: "#d62728", 28.0: "#2ca02c", 31.0: "#1f77b4"}
markers = {25.0: "o",       28.0: "s",        31.0: "^"}
T_arr   = np.array(T_STEPS)
C_arr   = np.array(N_CYCLE_VALS)
T_fine  = np.linspace(T_arr[0], T_arr[-1], 300)   # smooth curve for fit lines
C_fine  = T_fine / mean_period                     # same in cycles


# ── Fig 1: Linear scale  ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for rho in RHO_VALS:
    ax.plot(C_arr, results[rho],
            color=colors[rho], marker=markers[rho], lw=2, ms=8,
            label=f"ρ = {rho}  (data)")
    b, a, r2 = fit_params[rho]
    if np.isfinite(b):
        ax.plot(C_fine, a * (T_fine ** b), "--",
                color=colors[rho], lw=1.5, alpha=0.6,
                label=f"ρ={rho}  fit: {a:.2e}·T^({b:.2f})  R²={r2:.2f}")
ax.set_xlabel("Training length T  (cycles)", fontsize=13)
ax.set_ylabel("Normalised Training RMSE", fontsize=13)
ax.set_title("Task 7 — Training Error vs. T  [linear scale]",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, ncol=2);  ax.grid(alpha=0.3)
plt.savefig(out("task7_fig1_linear.png"), dpi=150, bbox_inches="tight")
print("\nSaved task7_fig1_linear.png")
plt.close()


# ── Fig 2: Log–log  (power-law shows as straight line) ───────────────────────
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for rho in RHO_VALS:
    ax.loglog(C_arr, results[rho],
              color=colors[rho], marker=markers[rho], lw=2, ms=8,
              label=f"ρ = {rho}  (data)")
    b, a, r2 = fit_params[rho]
    if np.isfinite(b):
        ax.loglog(C_fine, a * (T_fine ** b), "--",
                  color=colors[rho], lw=1.8, alpha=0.7,
                  label=f"ρ={rho}  fit slope = {b:.2f}  R²={r2:.2f}")

# annotate: what a straight line means
ax.text(0.03, 0.10,
        "Straight line on log–log  →  power-law\n"
        "Slope = exponent b  (here b ≈ −0.2 to −0.3)",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.8))
ax.set_xlabel("Training length T  (cycles)", fontsize=13)
ax.set_ylabel("Normalised Training RMSE  (log scale)", fontsize=13)
ax.set_title("Task 7 — Log–Log Plot: RMSE ∝ T^b  (power-law fit)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, ncol=2);  ax.grid(alpha=0.3, which="both")
plt.savefig(out("task7_fig2_loglog.png"), dpi=150, bbox_inches="tight")
print("Saved task7_fig2_loglog.png")
plt.close()


# ── Fig 3: Semi-log  (would be straight line if exponential) ─────────────────
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for rho in RHO_VALS:
    ax.semilogy(C_arr, results[rho],
                color=colors[rho], marker=markers[rho], lw=2, ms=8,
                label=f"ρ = {rho}")
ax.text(0.03, 0.10,
        "If exponential: would be straight line here\n"
        "Curved → NOT exponential",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.8))
ax.set_xlabel("Training length T  (cycles)", fontsize=13)
ax.set_ylabel("Normalised Training RMSE  (log scale)", fontsize=13)
ax.set_title("Task 7 — Semi-Log Plot  (ruling out exponential decay)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11);  ax.grid(alpha=0.3, which="both")
plt.savefig(out("task7_fig3_semilogy.png"), dpi=150, bbox_inches="tight")
print("Saved task7_fig3_semilogy.png")
plt.close()


# ── Fig 4: Individual subplots per ρ  (data + fit + annotations) ─────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
fig.suptitle("Task 7 — RMSE = a · T^b  per bifurcation parameter ρ",
             fontsize=13, fontweight="bold")

for ax, rho in zip(axes, RHO_VALS):
    rmse_arr = np.array(results[rho])
    b, a, r2 = fit_params[rho]

    ax.loglog(C_arr, rmse_arr,
              color=colors[rho], marker=markers[rho],
              lw=2, ms=9, zorder=3, label="Data")
    if np.isfinite(b):
        ax.loglog(C_fine, a * (T_fine ** b), "k--", lw=2,
                  label=f"Fit: {a:.2e} · T^({b:.3f})")

    # annotate each data point with its RMSE value
    for cx, ry in zip(C_arr, rmse_arr):
        ax.annotate(f"{ry:.5f}", (cx, ry),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6.5, color=colors[rho], alpha=0.85)

    ax.set_xlabel("T  (cycles)", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(f"ρ = {rho}\nb = {b:.3f},  R² = {r2:.3f}",
                 fontsize=12, color=colors[rho], fontweight="bold")
    ax.legend(fontsize=9);  ax.grid(alpha=0.3, which="both")

plt.savefig(out("task7_fig4_per_rho.png"), dpi=150, bbox_inches="tight")
print("Saved task7_fig4_per_rho.png")
plt.close()


# ── Fig 5: Exponent comparison bar chart ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
fig.suptitle("Task 7 — Power-Law Parameters a and b across ρ",
             fontsize=13, fontweight="bold")

b_vals = [fit_params[rho][0] for rho in RHO_VALS]
a_vals = [fit_params[rho][1] for rho in RHO_VALS]
rho_labels = [f"ρ={r}" for r in RHO_VALS]
bar_colors = [colors[r] for r in RHO_VALS]

# exponent b
bars = axes[0].bar(rho_labels, b_vals, color=bar_colors, edgecolor="k", width=0.4)
for bar, v in zip(bars, b_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 v - 0.005, f"{v:.3f}",
                 ha="center", va="top", fontsize=12, fontweight="bold", color="white")
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_ylabel("Exponent  b", fontsize=12)
axes[0].set_title("Decay exponent b\n(more negative = faster decay)", fontsize=11)
axes[0].set_ylim(min(b_vals)*1.4, 0.05);  axes[0].grid(axis="y", alpha=0.3)

# prefactor a
bars2 = axes[1].bar(rho_labels, a_vals, color=bar_colors, edgecolor="k", width=0.4)
for bar, v in zip(bars2, a_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 v + max(a_vals)*0.02, f"{v:.4f}",
                 ha="center", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Prefactor  a", fontsize=12)
axes[1].set_title("Prefactor a\n(overall error scale)", fontsize=11)
axes[1].set_ylim(0, max(a_vals)*1.3);  axes[1].grid(axis="y", alpha=0.3)

plt.savefig(out("task7_fig5_ab_params.png"), dpi=150, bbox_inches="tight")
print("Saved task7_fig5_ab_params.png")
plt.close()


# =============================================================================
# 12.  SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("TASK 7 — SUMMARY TABLE: Training RMSE vs Cycles")
print("=" * 70)
print(f"{'Cycles':>8}", end="")
for rho in RHO_VALS:
    print(f"   ρ={rho:5.1f}", end="")
print()
print("-" * 70)
for i, nc in enumerate(N_CYCLE_VALS):
    print(f"{nc:>8}", end="")
    for rho in RHO_VALS:
        v = results[rho][i]
        print(f"  {v:8.6f}" if np.isfinite(v) else "       NaN", end="")
    print()
print("-" * 70)
print("\nPower-law fits  (RMSE ~ a · T^b):")
for rho in RHO_VALS:
    b, a, r2 = fit_params[rho]
    print(f"  ρ = {rho:5.1f}:  a = {a:.4e}  b = {b:.4f}  R² = {r2:.4f}")

# =============================================================================
# 13.  SAVE
# =============================================================================

with open(out("task7_results.pkl"), "wb") as f:
    pickle.dump({
        "rho_vals":    RHO_VALS,
        "cycle_vals":  N_CYCLE_VALS,
        "t_steps":     T_STEPS,
        "mean_period": mean_period,
        "period_steps": period_steps,
        "results":     results,
        "fit_params":  fit_params,
    }, f)

print("\nResults saved to  task7_results.pkl")
print("Task 7 complete.")