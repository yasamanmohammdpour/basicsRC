# =============================================================================
# Task 6 — GPU-Accelerated Reservoir Computing  (PyTorch version)
# =============================================================================
# Speed-ups vs original numpy version:
#   1. All reservoir state updates run on GPU (CUDA / MPS / CPU fallback)
#   2. BATCHED seeds: all ITER_TIME (or FINAL_SEEDS) reservoirs run in ONE
#      parallel forward pass — no Python for-loop over seeds at runtime
#   3. Ridge regression uses batched torch.linalg.solve (no matrix inverse)
#   4. float32 throughout (2× faster than float64 on GPU, negligible accuracy loss)
#   5. Pre-upload entire ts_norm to GPU once; never copy per-call
#
# All science is identical to the original:
#   - Lorenz ODE / RK4 / z-score normalisation
#   - Leaky integrator reservoir, even-element squaring
#   - Per-dimension NRMSE (Improvement 1)
#   - Multi-seed final_eval with median seed (Improvement 2)
#   - Closed-loop test NRMSE printed (Improvement 3)
#   - 70% trimmed mean in objective (Zheng-Meng)
# =============================================================================

import os
import time
import pickle
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization

warnings.filterwarnings("ignore")

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("GPU : Apple MPS")
else:
    device = torch.device("cpu")
    print("GPU : not found — running on CPU (still batched)")

DTYPE = torch.float32   # float32 is 2× faster than float64 on GPU

OUT_DIR = "task6_results"
os.makedirs(OUT_DIR, exist_ok=True)
def out(fname): return os.path.join(OUT_DIR, fname)

# =============================================================================
# 1.  LORENZ  (numpy — run once at startup)
# =============================================================================

def lorenz_ode(state, sigma=10.0, rho=28.0, beta=2.67):
    x, y, z = state
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

def rk4_step(f, state, dt):
    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def generate_lorenz(n_steps, dt=0.01, warmup=5000):
    np.random.seed(42)
    x0 = np.array([np.random.uniform(0.5, 0.8),
                   np.random.uniform(0.0, 0.3),
                   np.random.uniform(8.0, 10.0)])
    total = warmup + n_steps
    ts = np.zeros((total, 3))
    ts[0] = x0
    for i in range(total-1):
        ts[i+1] = rk4_step(lorenz_ode, ts[i], dt)
    return ts[warmup:]

# =============================================================================
# 2.  BUILD BATCH OF RESERVOIRS  (numpy eigvals on CPU → GPU tensors)
# =============================================================================

def build_batch_gpu(B: int, N: int, d: float, eig_rho: float,
                    gamma: float, dim: int, seeds: list):
    """
    Build B reservoirs on CPU (networkx + eigvals), transfer to GPU.
    Returns:
        Win_batch : (B, N, dim)  float32  GPU tensor
        A_batch   : (B, N, N)   float32  GPU tensor
    """
    Wins, As = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        Win = rng.uniform(-gamma, gamma, (N, dim)).astype(np.float32)

        G = nx.erdos_renyi_graph(N, d,
                                 seed=int(rng.randint(0, 1_000_000)),
                                 directed=False)
        A = np.zeros((N, N), dtype=np.float32)
        for (u, v) in G.edges():
            w = rng.normal(0.0, 1.0)
            A[u, v] = w
            A[v, u] = w

        rho_a = np.max(np.abs(np.linalg.eigvals(A)))
        if rho_a > 1e-12:
            A *= eig_rho / rho_a

        Wins.append(Win)
        As.append(A)

    Win_batch = torch.from_numpy(np.stack(Wins)).to(device)   # (B, N, dim)
    A_batch   = torch.from_numpy(np.stack(As)).to(device)     # (B, N, N)
    return Win_batch, A_batch

# =============================================================================
# 3.  BATCHED TRAIN + VALIDATE  (all on GPU in one pass)
# =============================================================================

def train_validate_batch(ts_gpu:  torch.Tensor,    # (T, dim) on GPU
                         Win_b:   torch.Tensor,    # (B, N, dim)
                         A_b:     torch.Tensor,    # (B, N, N)
                         alpha:   float,
                         beta:    float,
                         noise_a: float,
                         washout: int,
                         train_len: int,
                         val_len:   int,
                         seeds:   list):
    """
    GPU-batched forward pass for B reservoirs.

    Returns
    -------
    Wout     : (B, dim, N)  — trained output weights
    r_end_tr : (B, N)       — reservoir state at end of training
    r_end_val: (B, N)       — reservoir state after validation window
    val_nrmse: (B,)         — per-seed validation NRMSE (per-dim formula)
    """
    B   = Win_b.shape[0]
    N   = Win_b.shape[1]
    dim = Win_b.shape[2]
    total = washout + train_len

    # ── Reproducible noise per seed (small CPU cost, done once) ───────────────
    noise_np = np.zeros((B, total, dim), dtype=np.float32)
    for i, s in enumerate(seeds):
        rng = np.random.RandomState(s + 9999)
        noise_np[i] = rng.randn(total, dim).astype(np.float32) * noise_a
    noise_gpu = torch.from_numpy(noise_np).to(device)  # (B, total, dim)

    # Noisy input: (B, total, dim)
    train_in = ts_gpu[:total].unsqueeze(0) + noise_gpu  # broadcast → (B, total, dim)

    # ── Reservoir forward pass (washout + training) ────────────────────────────
    r = torch.zeros(B, N, device=device, dtype=DTYPE)
    R_store = torch.empty(B, N, train_len, device=device, dtype=DTYPE)

    for t in range(total):
        u  = train_in[:, t, :]                                          # (B, dim)
        Ar = torch.bmm(A_b,   r.unsqueeze(-1)).squeeze(-1)             # (B, N)
        Wu = torch.bmm(Win_b, u.unsqueeze(-1)).squeeze(-1)             # (B, N)
        r  = (1.0 - alpha) * r + alpha * torch.tanh(Ar + Wu)
        if t >= washout:
            R_store[:, :, t - washout] = r

    r_end_tr = r.clone()   # (B, N)

    # Even-element squaring
    R_aug = R_store.clone()
    R_aug[:, 1::2, :] = R_aug[:, 1::2, :] ** 2                        # (B, N, T)

    # Target matrix Y: (dim, train_len) → (B, dim, train_len)
    Y     = ts_gpu[washout+1 : washout+train_len+1].T                  # (dim, T)
    Y_exp = Y.unsqueeze(0).expand(B, -1, -1)                           # (B, dim, T)

    # Ridge regression via linalg.solve  (numerically stabler than explicit inv)
    # We want: Wout = (Y @ R^T) @ inv(R @ R^T + β I)
    # ↔  (R @ R^T + β I) @ Wout^T = (Y @ R^T)^T
    YRt = torch.bmm(Y_exp, R_aug.permute(0, 2, 1))                    # (B, dim, N)
    RRT = torch.bmm(R_aug, R_aug.permute(0, 2, 1))                    # (B, N, N)
    reg = beta * torch.eye(N, device=device, dtype=DTYPE).unsqueeze(0) # (1, N, N)

    # solve A X = B  →  X = Wout^T  shape (B, N, dim)
    Wout = torch.linalg.solve(RRT + reg,
                               YRt.permute(0, 2, 1)).permute(0, 2, 1) # (B, dim, N)

    # ── Teacher-forced validation pass ────────────────────────────────────────
    r = r_end_tr.clone()
    val_preds = torch.empty(B, val_len, dim, device=device, dtype=DTYPE)
    val_trues = torch.empty(val_len, dim, device=device, dtype=DTYPE)
    T_max     = ts_gpu.shape[0] - 1

    for t in range(val_len):
        idx = washout + train_len + t
        u   = ts_gpu[idx].unsqueeze(0).expand(B, -1)                  # (B, dim)
        Ar  = torch.bmm(A_b,   r.unsqueeze(-1)).squeeze(-1)
        Wu  = torch.bmm(Win_b, u.unsqueeze(-1)).squeeze(-1)
        r   = (1.0 - alpha) * r + alpha * torch.tanh(Ar + Wu)

        r_aug = r.clone()
        r_aug[:, 1::2] = r_aug[:, 1::2] ** 2
        pred = torch.bmm(Wout, r_aug.unsqueeze(-1)).squeeze(-1)        # (B, dim)

        val_preds[:, t, :] = pred
        val_trues[t]       = ts_gpu[min(idx+1, T_max)]

    r_end_val = r.clone()   # (B, N)

    # Per-dimension NRMSE  (Improvement 1 from original)
    err        = val_preds - val_trues.unsqueeze(0)                    # (B, T, dim)
    rmse_d     = torch.sqrt(torch.mean(err**2, dim=1))                 # (B, dim)
    sigma_d    = torch.std(val_trues, dim=0).clamp(min=1e-12)          # (dim,)
    val_nrmse  = torch.mean(rmse_d / sigma_d.unsqueeze(0), dim=1)      # (B,)

    return Wout, r_end_tr, r_end_val, val_nrmse

# =============================================================================
# 4.  CLOSED-LOOP (AUTONOMOUS) PREDICTION  —  batched GPU
# =============================================================================

def closed_loop_batch(ts_gpu:    torch.Tensor,  # (T, dim)
                      Wout_b:   torch.Tensor,  # (B, dim, N)
                      r_start:  torch.Tensor,  # (B, N)
                      Win_b:    torch.Tensor,  # (B, N, dim)
                      A_b:      torch.Tensor,  # (B, N, N)
                      alpha:    float,
                      test_start: int,
                      test_len:   int):
    """Autonomous prediction for B reservoirs in parallel.  Returns (B, T, dim)."""
    B, N = r_start.shape
    dim  = Win_b.shape[2]

    r = r_start.clone()
    u = ts_gpu[test_start].unsqueeze(0).expand(B, -1)   # seed with first true step
    preds = torch.empty(B, test_len, dim, device=device, dtype=DTYPE)

    for t in range(test_len):
        Ar = torch.bmm(A_b,   r.unsqueeze(-1)).squeeze(-1)
        Wu = torch.bmm(Win_b, u.unsqueeze(-1)).squeeze(-1)
        r  = (1.0 - alpha)*r + alpha*torch.tanh(Ar + Wu)

        r_aug = r.clone()
        r_aug[:, 1::2] = r_aug[:, 1::2] ** 2
        p = torch.bmm(Wout_b, r_aug.unsqueeze(-1)).squeeze(-1)   # (B, dim)
        preds[:, t, :] = p
        u = p

    return preds    # (B, test_len, dim)

# =============================================================================
# 5.  CONSTANTS  (identical to original Task 6)
# =============================================================================

N_RES     = 200
DIM       = 3
WASHOUT   = 1000
TRAIN     = 8000
VAL       = 2000
TEST      = 3000
TOTAL     = WASHOUT + TRAIN + VAL + TEST + 500

ITER_TIME = 10      # seeds per objective call
KEEP_FRAC = 0.8     # trimmed mean fraction

PBOUNDS = {
    "d":         (0.01, 0.30),
    "eig_rho":   (0.10, 5.00),
    "gamma":     (0.01, 5.00),
    "alpha":     (0.01, 0.95),
    "beta_log":  (-7.0, -1.0),
    "noise_log": (-7.0, -1.0),
}

HAND = {
    'd': 0.027,
    'eig_rho': 4.78,
    'gamma': 0.15,
    'alpha': 0.41,
    'beta_log': -1.36,
    'noise_log': -6.0
}

FINAL_SEEDS = 10

# =============================================================================
# 6.  GENERATE + NORMALISE DATA  (once at startup, upload to GPU)
# =============================================================================

print("="*65)
print("Task 6 — Hyperparameter Optimisation  [GPU version]")
print("="*65)
print(f"\nGenerating Lorenz  ({TOTAL} steps) ...")

ts_raw  = generate_lorenz(TOTAL, dt=0.01, warmup=5000)
scaler  = StandardScaler()
scaler.fit(ts_raw[:WASHOUT + TRAIN])
ts_norm = scaler.transform(ts_raw).astype(np.float32)

# ── Upload once to GPU ────────────────────────────────────────────────────────
TS_GPU  = torch.from_numpy(ts_norm).to(device)   # stays there forever

print(f"  Shape: {ts_norm.shape}  |  Split: W={WASHOUT} T={TRAIN} V={VAL} Te={TEST}")
print(f"  GPU tensor: {TS_GPU.shape}  dtype={TS_GPU.dtype}  device={TS_GPU.device}\n")

# =============================================================================
# 7.  OBJECTIVE  (GPU batched — all ITER_TIME seeds in one call)
# =============================================================================

_count = [0]

def objective(d, eig_rho, gamma, alpha, beta_log, noise_log):
    _count[0] += 1
    beta    = 10.0 ** beta_log
    noise_a = 10.0 ** noise_log
    seeds   = [s * 137 + 7 for s in range(ITER_TIME)]

    try:
        Win_b, A_b = build_batch_gpu(ITER_TIME, N_RES, d, eig_rho, gamma, DIM, seeds)
        _, _, _, val_nrmses = train_validate_batch(
            TS_GPU, Win_b, A_b, alpha, beta, noise_a,
            WASHOUT, TRAIN, VAL, seeds
        )
        nrmses = val_nrmses.cpu().numpy()                  # (ITER_TIME,)
        nrmses = nrmses[np.isfinite(nrmses) & (nrmses > 0)]
        if len(nrmses) == 0:
            return 1e-12
        nrmses.sort()
        keep      = max(1, int(np.ceil(KEEP_FRAC * len(nrmses))))
        mean_nrmse = float(np.mean(nrmses[:keep]))
    except Exception as e:
        print(f"  [obj error] {e}")
        return 1e-12

    print(f"  [{_count[0]:4d}]  NRMSE={mean_nrmse:.5f}  "
          f"d={d:.3f}  rho={eig_rho:.3f}  gamma={gamma:.3f}  "
          f"alpha={alpha:.3f}  beta=10^{beta_log:.2f}  noise=10^{noise_log:.2f}")
    return 1.0 / (mean_nrmse + 1e-12)

# =============================================================================
# 8.  RANDOM SEARCH
# =============================================================================

def random_search(n_iter=60):
    print(f"\n{'='*65}\nRANDOM SEARCH  ({n_iter} iterations)\n{'='*65}\n")
    best_score, best_params, history = -np.inf, None, []
    rng_rs = np.random.RandomState(123)
    t0 = time.time()

    for it in range(n_iter):
        print(f"--- RS {it+1}/{n_iter} ---")
        params = {k: float(rng_rs.uniform(lo, hi)) for k, (lo, hi) in PBOUNDS.items()}
        score  = objective(**params)
        rmse   = 1.0/score if score > 1e-11 else np.inf
        history.append({"params": params.copy(), "rmse": rmse})
        if score > best_score:
            best_score  = score
            best_params = params.copy()
            print(f"  *** New best!  NRMSE={rmse:.5f} ***")

    best_rmse = 1.0/best_score if best_score > 1e-11 else np.inf
    print(f"\nRandom Search: {time.time()-t0:.1f}s  |  Best NRMSE={best_rmse:.6f}")
    _print_params(best_params)
    return best_params, best_rmse, history

# =============================================================================
# 9.  BAYESIAN OPTIMISATION
# =============================================================================

def bayesian_opt(n_init=10, n_iter=50):
    print(f"\n{'='*65}\nBAYESIAN OPT  ({n_init} init + {n_iter} iter)\n{'='*65}\n")
    t0  = time.time()
    opt = BayesianOptimization(f=objective, pbounds=PBOUNDS,
                               random_state=42, verbose=0)
    print("--- Phase 1: random init ---")
    opt.maximize(init_points=n_init, n_iter=0)
    print("\n--- Phase 2: GP-guided search ---")
    opt.maximize(init_points=0, n_iter=n_iter)

    best_params = opt.max["params"]
    best_rmse   = 1.0 / opt.max["target"]
    print(f"\nBayes Opt: {time.time()-t0:.1f}s  |  Best NRMSE={best_rmse:.6f}")
    _print_params(best_params)
    return best_params, best_rmse, opt

def _print_params(p):
    for k, v in p.items():
        print(f"  {k:<12} = {v:.5f}")

# =============================================================================
# 10.  FINAL EVALUATION  (multi-seed batched GPU — Improvements 2 & 3)
# =============================================================================

def final_eval(params, label=""):
    beta    = 10.0 ** params["beta_log"]
    noise_a = 10.0 ** params["noise_log"]
    seeds   = [s * 17 + 3 for s in range(FINAL_SEEDS)]

    print(f"\n--- Final eval: {label}  ({FINAL_SEEDS} seeds, GPU batched) ---")
    t0 = time.time()

    Win_b, A_b = build_batch_gpu(FINAL_SEEDS, N_RES,
                                  params["d"], params["eig_rho"], params["gamma"],
                                  DIM, seeds)

    Wout_b, r_end_tr, r_end_val, val_nrmses = train_validate_batch(
        TS_GPU, Win_b, A_b,
        params["alpha"], beta, noise_a,
        WASHOUT, TRAIN, VAL, seeds
    )

    val_arr  = val_nrmses.cpu().numpy()
    mean_val = float(np.mean(val_arr))
    std_val  = float(np.std(val_arr))
    print(f"  Val NRMSE: {mean_val:.6f} ± {std_val:.6f}  ({FINAL_SEEDS} seeds)")

    # Closed-loop test predictions for all seeds in one pass
    test_start = WASHOUT + TRAIN + VAL
    test_true  = ts_norm[test_start : test_start + TEST]   # numpy, for plotting
    test_true_gpu = TS_GPU[test_start : test_start + TEST]

    test_preds = closed_loop_batch(
        TS_GPU, Wout_b, r_end_val, Win_b, A_b,
        params["alpha"], test_start, TEST
    )   # (B, TEST, dim)

    # Closed-loop NRMSE — per-dimension, representative (median) seed
    median_val    = float(np.median(val_arr))
    rep_idx       = int(np.argmin(np.abs(val_arr - median_val)))
    pred_rep      = test_preds[rep_idx].cpu().numpy()       # (TEST, dim)

    rmse_d  = np.sqrt(np.mean((pred_rep - test_true)**2, axis=0))
    sigma_d = np.std(test_true, axis=0) + 1e-12
    cl_nrmse = float(np.mean(rmse_d / sigma_d))
    print(f"  CL-NRMSE : {cl_nrmse:.6f}  (rep seed #{rep_idx})"
          f"  [{time.time()-t0:.1f}s]")

    return mean_val, std_val, cl_nrmse, pred_rep, test_true

# =============================================================================
# 11.  PLOTS  (identical to original)
# =============================================================================

def plot_short_term(pred, true, label, val_nrmse, val_std, n=500):
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(f"Task 6 — Short-Term  [{label}]\n"
                 f"Val NRMSE = {val_nrmse:.5f} ± {val_std:.5f}  ({FINAL_SEEDS} seeds)",
                 fontsize=13, fontweight="bold")
    for i, var in enumerate(["x","y","z"]):
        axes[i].plot(true[:n,i], "r-",  lw=1.8, label="True")
        axes[i].plot(pred[:n,i], "b--", lw=1.4, label="Predicted")
        axes[i].set_ylabel(var, fontsize=12); axes[i].legend(fontsize=10)
        axes[i].grid(alpha=0.25)
    axes[2].set_xlabel("Time step", fontsize=11)
    fname = out(f"task6_short_{label.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {fname}")

def plot_attractor(pred, true, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(f"Task 6 — Attractor  [{label}]", fontsize=13, fontweight="bold")
    ax1.plot(true[:,0], true[:,2], "r-", lw=0.35, alpha=0.7)
    ax1.set_title("True (x–z)"); ax1.set_xlabel("x"); ax1.set_ylabel("z")
    ax2.plot(pred[:,0], pred[:,2], "b-", lw=0.35, alpha=0.7)
    ax2.set_title("Pred (x–z)"); ax2.set_xlabel("x"); ax2.set_ylabel("z")
    fname = out(f"task6_attractor_{label.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {fname}")

def plot_convergence(rs_history, bo_opt):
    rs_curve, best = [], np.inf
    for h in rs_history:
        best = min(best, h["rmse"]); rs_curve.append(best)
    bo_curve, best = [], np.inf
    for res in bo_opt.res:
        r = 1.0/res["target"] if res["target"] > 1e-11 else np.inf
        best = min(best, r); bo_curve.append(best)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.semilogy(rs_curve, "r-o", ms=4, lw=1.5,
                label=f"Random Search (final={rs_curve[-1]:.5f})")
    ax.semilogy(bo_curve, "b-s", ms=4, lw=1.5,
                label=f"Bayesian Opt  (final={bo_curve[-1]:.5f})")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Val NRMSE (log)", fontsize=12)
    ax.set_title("Task 6 — Convergence Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3, which="both")
    plt.savefig(out("task6_convergence.png"), dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved task6_convergence.png")

def plot_bar(hand_r, rs_r, bo_r, hand_std, rs_std, bo_std):
    methods = ["By Hand\n(Task4)", "Random\nSearch", "Bayesian\nOpt."]
    values  = [hand_r, rs_r, bo_r]
    stds    = [hand_std, rs_std, bo_std]
    colors  = ["#d62728", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    bars = ax.bar(methods, values, color=colors, edgecolor="k", width=0.45,
                  yerr=stds, capsize=6, error_kw={"elinewidth":2, "capthick":2})
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(values)*0.04,
                f"{v:.5f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Val NRMSE (mean±std)", fontsize=12)
    ax.set_ylim(0, max(values)*1.5)
    ax.set_title(f"Task 6 — Method Comparison  ({FINAL_SEEDS} seeds)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.savefig(out("task6_bar.png"), dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved task6_bar.png")

def print_table(hand_p, rs_p, bo_p,
                hand_r, rs_r, bo_r,
                hand_std, rs_std, bo_std,
                hand_cl, rs_cl, bo_cl):
    print("\n" + "="*80)
    print("TASK 6 — HYPERPARAMETER AND PERFORMANCE COMPARISON TABLE")
    print("="*80)
    print(f"{'Hyperparameter':<22} {'By Hand (Task4)':>16} {'Random Search':>14} {'Bayesian Opt':>13}")
    print("-"*80)
    rows = [("rho (spectral r.)", "eig_rho"), ("gamma (input sc.)", "gamma"),
            ("alpha (leakage)",   "alpha"),   ("beta  (log10)",    "beta_log"),
            ("noise (log10)",     "noise_log"),("d (conn. prob.)",  "d")]
    for name, key in rows:
        print(f"  {name:<20} {hand_p[key]:>16.4f} {rs_p[key]:>14.4f} {bo_p[key]:>13.4f}")
    print("-"*80)
    print(f"  {'Val NRMSE (mean)':<20} {hand_r:>16.6f} {rs_r:>14.6f} {bo_r:>13.6f}")
    print(f"  {'Val NRMSE (std)':<20}  {hand_std:>15.6f} {rs_std:>14.6f} {bo_std:>13.6f}")
    print(f"  {'Closed-loop NRMSE':<20} {hand_cl:>16.6f} {rs_cl:>14.6f} {bo_cl:>13.6f}")
    print("="*80)

# =============================================================================
# 12.  MAIN
# =============================================================================

if __name__ == "__main__":

    N_RS_ITER = 200
    N_BO_INIT = 20
    N_BO_ITER = 180

    np.random.seed(42)

    # Step 0: hand-tuned baseline
    print(f"\n{'='*65}\nSTEP 0 — Task4 hand-tuned baseline\n{'='*65}")
    hand_r, hand_std, hand_cl, hand_pred, base_true = final_eval(HAND, "Task4 Hand-Tuned")

    # Step 1: Random Search
    rs_params, _, rs_hist = random_search(n_iter=N_RS_ITER)

    # Step 2: Bayesian Optimisation
    bo_params, _, bo_opt  = bayesian_opt(n_init=N_BO_INIT, n_iter=N_BO_ITER)

    # Step 3: Final evaluation
    print(f"\n{'='*65}\nSTEP 3 — Final multi-seed evaluation\n{'='*65}")
    rs_r, rs_std, rs_cl, rs_pred, test_true = final_eval(rs_params, "Random Search")
    bo_r, bo_std, bo_cl, bo_pred, test_true = final_eval(bo_params, "Bayesian Opt")

    # Step 4: Summary table
    print_table(HAND, rs_params, bo_params,
                hand_r, rs_r, bo_r,
                hand_std, rs_std, bo_std,
                hand_cl, rs_cl, bo_cl)

    # Step 5: Plots
    print("\nGenerating plots ...")
    plot_short_term(hand_pred, base_true, "Task4 Hand-Tuned", hand_r, hand_std)
    plot_short_term(rs_pred,   test_true, "Random Search",    rs_r,   rs_std)
    plot_short_term(bo_pred,   test_true, "Bayesian Opt",     bo_r,   bo_std)
    plot_attractor(hand_pred, base_true, "Task4 Hand-Tuned")
    plot_attractor(rs_pred,   test_true, "Random Search")
    plot_attractor(bo_pred,   test_true, "Bayesian Opt")
    plot_convergence(rs_hist, bo_opt)
    plot_bar(hand_r, rs_r, bo_r, hand_std, rs_std, bo_std)

    # Step 6: Save results
    best_src = "bayesian" if bo_r <= rs_r else "random"
    best_p   = bo_params  if bo_r <= rs_r else rs_params
    with open(out("task6_results.pkl"), "wb") as f:
        pickle.dump({
            "hand":  {"params": HAND,      "rmse": hand_r, "std": hand_std, "cl_nrmse": hand_cl},
            "rs":    {"params": rs_params, "rmse": rs_r,   "std": rs_std,   "cl_nrmse": rs_cl, "history": rs_hist},
            "bayes": {"params": bo_params, "rmse": bo_r,   "std": bo_std,   "cl_nrmse": bo_cl},
            "best":  {"params": best_p, "rmse": min(rs_r, bo_r), "source": best_src},
            "preds": {"hand": hand_pred, "rs": rs_pred, "bayes": bo_pred, "true": test_true}
        }, f)

    print("\nResults saved to task6_results.pkl")
    print("Task 6 GPU version complete.")