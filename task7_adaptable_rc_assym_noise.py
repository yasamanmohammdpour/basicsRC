# =============================================================================
# Task 7 — GPU Reservoir Computing  (UNIFIED — all capabilities)
#
# ── MODE (set at top) ─────────────────────────────────────────────────────────
#   "standard"   → standard RC, ρ=28 only, symmetric reservoir, no squaring,
#                  no noise, no parameter injection
#                  → matches Zheng-Meng / reference code exactly
#
#   "adaptable"  → adaptable RC, ρ=25/28/31, directed reservoir,
#                  even-element squaring, input noise, parameter injection
#                  → original Task 7 adaptable RC (pre-clean-comparison version) experiment
#
# ── NRMSE_TYPE (set at top) ───────────────────────────────────────────────────
#   "training"   → ridge regression residual on training data
#                  (normal-equations form, numerically stable float64)
#                  → Zheng-Meng's definition
#
#   "prediction" → one-step-ahead prediction on held-out test segment
#                  → reveals full "two sides" curve (NRMSE≈1 at tiny T)
#
# ── PLOTS (all 6, always generated) ──────────────────────────────────────────
#   fig1: log-log primary
#   fig2: linear scale
#   fig3: semi-log diagnostic
#   fig4: per-rho subplots with point annotations
#   fig5: bar charts of power-law parameters a and b
#   fig6: reference-style panel (Zheng-Meng format)
# =============================================================================

import os, time, pickle, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import find_peaks
from scipy.stats import linregress

warnings.filterwarnings("ignore")

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("GPU : Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("GPU : not found — running on CPU")

DTYPE = torch.float32

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#   CONFIGURATION — edit here
# ══════════════════════════════════════════════════════════════════════════════

# RC mode: "standard" or "adaptable"
MODE = "standard"

# NRMSE type: "training" or "prediction"
NRMSE_TYPE = "training"

# Reservoir size
N_RES = 500

# Seeds
N_SEEDS = 5

# Power-law fit: exclude short-T transition regime
FIT_MIN_CYCLES = 10

# ══════════════════════════════════════════════════════════════════════════════

DT         = 0.01
DIM        = 3
CHUNK_SIZE = 10_000
TEST_STEPS = 5000   # steps used for held-out prediction NRMSE evaluation

# Mode-dependent settings
if MODE == "standard":
    RHO_VALS = [28.0]
    RHO_P    = 0.0
    PB       = 0.0
    WASHOUT  = 1000
    # Standard RC: symmetric reservoir, no squaring, no noise
    # → matches reference code (Zheng-Meng)
else:
    RHO_VALS = [25.0, 28.0, 31.0]
    RHO_P    = 0.1
    PB       = 0.0
    WASHOUT  = 500
    # Adaptable RC: directed reservoir, even-element squaring, noise, p-injection

WARMUP = 5000

OUT_DIR = f"task7_results_{MODE}_{NRMSE_TYPE}"
os.makedirs(OUT_DIR, exist_ok=True)
def out(f): return os.path.join(OUT_DIR, f)

print(f"\nMODE      : {MODE}")
print(f"NRMSE_TYPE: {NRMSE_TYPE}")
print(f"RHO_VALS  : {RHO_VALS}")

# =============================================================================
# 1.  LORENZ / RK4
# =============================================================================

def lorenz_ode(state, rho=28.0, sigma=10.0, beta=8/3):
    x, y, z = state
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

def rk4_step(f, s, dt):
    k1 = f(s)
    k2 = f(s + 0.5*dt*k1)
    k3 = f(s + 0.5*dt*k2)
    k4 = f(s + dt*k3)
    return s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def lorenz_chunk_from_state(state, n_steps, dt, rho):
    buf = np.empty((n_steps + 1, 3), dtype=np.float32)
    buf[0] = state
    f = lambda s: lorenz_ode(s, rho=rho)
    for i in range(n_steps):
        buf[i+1] = rk4_step(f, buf[i], dt)
    return buf

def lorenz_initial_state(rho, dt, warmup, seed):
    np.random.seed(seed)
    x0 = np.array([np.random.uniform(0.5, 0.8),
                   np.random.uniform(0.0, 0.3),
                   np.random.uniform(8.0, 10.0)])
    f = lambda s: lorenz_ode(s, rho=rho)
    s = x0.copy()
    for _ in range(warmup):
        s = rk4_step(f, s, dt)
    return s.astype(np.float32)

# =============================================================================
# 2.  NORMALIZATION
# =============================================================================

STAT_STEPS = 200_000

def compute_norm_stats(rho, dt, warmup, seed):
    s0    = lorenz_initial_state(rho, dt, warmup, seed)
    chunk = lorenz_chunk_from_state(s0, STAT_STEPS, dt, rho)
    mu    = chunk.mean(axis=0)
    sigma = chunk.std(axis=0) + 1e-7
    return mu.astype(np.float32), sigma.astype(np.float32)

def normalise(raw, mu, sigma):
    return ((raw - mu) / sigma).astype(np.float32)

# =============================================================================
# 3.  PERIOD
# =============================================================================

def average_period_steps(rho, dt, warmup, probe_steps=50_000):
    s0    = lorenz_initial_state(rho, dt, warmup, seed=0)
    chunk = lorenz_chunk_from_state(s0, probe_steps, dt, rho)
    peaks, _ = find_peaks(chunk[:, 2], height=np.mean(chunk[:, 2]))
    return int(np.round(np.mean(np.diff(peaks)))) if len(peaks) >= 2 else 100

# =============================================================================
# 4.  RESERVOIR CONSTRUCTION
#
#   STANDARD:   symmetric reservoir  (S + S.T)/2, no Wp
#               → matches reference code exactly
#
#   ADAPTABLE:  directed Erdős–Rényi, with Wp for parameter injection
#               → original Task 7 adaptable RC
# =============================================================================

def build_batch_gpu(B, N, d, eig_rho, gamma, dim, seeds, mode="standard"):
    Wins, As, Wps = [], [], []
    for s in seeds:
        if mode == "standard":
            rng_np  = np.random.default_rng(s)
            # Symmetric sparse random (matches reference)
            S_dense = np.zeros((N, N), dtype=np.float32)
            mask    = rng_np.random((N, N)) < d
            weights = rng_np.standard_normal((N, N)).astype(np.float32)
            S_dense[mask] = weights[mask]
            A     = (S_dense + S_dense.T) * 0.5          # SYMMETRIC
            rho_a = np.max(np.abs(np.linalg.eigvals(A)))
            if rho_a > 1e-12:
                A *= eig_rho / rho_a
            # Uniform [-gamma, gamma] input weights (matches reference)
            Win = (2 * rng_np.random((N, dim)).astype(np.float32) - 1) * gamma
            Wp  = np.zeros((N, 1), dtype=np.float32)     # unused in standard
        else:
            rng = np.random.RandomState(s)
            Win = rng.uniform(-gamma, gamma, (N, dim)).astype(np.float32)
            # Directed Erdős–Rényi reservoir
            G = nx.erdos_renyi_graph(N, d,
                                     seed=int(rng.randint(0, 1_000_000)),
                                     directed=True)
            A = np.zeros((N, N), dtype=np.float32)
            for (u, v) in G.edges():
                A[u, v] = rng.normal(0.0, 1.0)
            rho_a = np.max(np.abs(np.linalg.eigvals(A)))
            if rho_a > 1e-12:
                A *= eig_rho / rho_a
            Wp = rng.uniform(-1.0, 1.0, (N, 1)).astype(np.float32)

        Wins.append(Win); As.append(A); Wps.append(Wp)

    Win_b = torch.from_numpy(np.stack(Wins).astype(np.float32)).to(DEVICE)
    A_b   = torch.from_numpy(np.stack(As).astype(np.float32)).to(DEVICE)
    Wp_b  = torch.from_numpy(np.stack(Wps).astype(np.float32)).to(DEVICE)
    return Win_b, A_b, Wp_b

# =============================================================================
# 5.  JIT RECURRENT CHUNKS
#
#   Two separate JIT functions:
#     rc_forward_standard  → plain r, no squaring (standard RC)
#     rc_forward_adaptable → even-element squaring, p_contrib (adaptable RC)
# =============================================================================

@torch.jit.script
def rc_forward_standard(r: torch.Tensor,
                        A_b: torch.Tensor,
                        WinU: torch.Tensor,
                        alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard RC: plain r, no squaring, no parameter injection."""
    B, T, N = WinU.shape
    r_buf = torch.empty(B, T, N, dtype=r.dtype, device=r.device)
    for t in range(T):
        Ar  = torch.bmm(A_b, r.unsqueeze(-1)).squeeze(-1)
        r   = (1.0 - alpha) * r + alpha * torch.tanh(Ar + WinU[:, t, :])
        r_buf[:, t, :] = r
    return r, r_buf


@torch.jit.script
def rc_forward_adaptable(r: torch.Tensor,
                         A_b: torch.Tensor,
                         WinU: torch.Tensor,
                         p_contrib: torch.Tensor,
                         alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Adaptable RC: plain r (no squaring) + parameter injection only.
    Same architecture as standard RC — only difference is p_contrib is active.
    """
    B, T, N = WinU.shape
    r_buf = torch.empty(B, T, N, dtype=r.dtype, device=r.device)
    for t in range(T):
        Ar  = torch.bmm(A_b, r.unsqueeze(-1)).squeeze(-1)
        r   = (1.0 - alpha) * r + alpha * torch.tanh(
            Ar + WinU[:, t, :] + p_contrib)
        r_a = r.clone()
        r_a[:, 1::2] = r_a[:, 1::2] ** 2    # even-element squaring
        r_buf[:, t, :] = r_a
    return r, r_buf


def rc_forward(r, A_b, WinU, p_contrib, alpha, mode):
    """Unified dispatcher."""
    if mode == "standard":
        return rc_forward_standard(r, A_b, WinU, alpha)
    else:
        return rc_forward_adaptable(r, A_b, WinU, p_contrib, alpha)

# =============================================================================
# 6.  HELD-OUT PREDICTION NRMSE  (used when NRMSE_TYPE="prediction")
# =============================================================================

def evaluate_on_test(Wout_b, Win_b, A_b, Wp_b,
                     mu, sigma, rho, dt, warmup, alpha,
                     rho_p, pb, mode, test_steps=5000):
    """
    Run a held-out Lorenz segment, compute one-step prediction NRMSE.
    Uses seed offset +99999 — never overlaps training trajectory.
    Returns (B,) numpy array.
    """
    B = Win_b.shape[0]
    N = Win_b.shape[1]

    # Parameter contribution (zero for standard RC)
    p_val     = rho if mode == "adaptable" else 0.0
    p_contrib = rho_p * Wp_b[:, :, 0] * (p_val + pb)   # (B, N)

    test_seed  = int(rho * 10) + 99999
    test_state = lorenz_initial_state(rho, dt, warmup, seed=test_seed)
    raw        = lorenz_chunk_from_state(test_state, test_steps, dt, rho)

    inp_norm = normalise(raw[:-1], mu, sigma).astype(np.float32)
    tgt_norm = normalise(raw[1:],  mu, sigma).astype(np.float32)

    inp_gpu = torch.from_numpy(np.ascontiguousarray(inp_norm, dtype=np.float32)).to(DEVICE)
    tgt_gpu = torch.from_numpy(np.ascontiguousarray(tgt_norm, dtype=np.float32)).to(DEVICE)

    # Washout on test segment
    TEST_WASHOUT = 200
    r = torch.zeros(B, N, device=DEVICE, dtype=DTYPE)
    if TEST_WASHOUT > 0 and TEST_WASHOUT < test_steps:
        WinU_ws = torch.einsum('bni,ti->btn', Win_b, inp_gpu[:TEST_WASHOUT])
        r, _    = rc_forward(r, A_b, WinU_ws, p_contrib, alpha, mode)
        inp_eval = inp_gpu[TEST_WASHOUT:]
        tgt_eval = tgt_gpu[TEST_WASHOUT:]
    else:
        inp_eval = inp_gpu
        tgt_eval = tgt_gpu

    WinU = torch.einsum('bni,ti->btn', Win_b, inp_eval)
    r, r_buf = rc_forward(r, A_b, WinU, p_contrib, alpha, mode)

    y_pred  = torch.bmm(r_buf, Wout_b.permute(0, 2, 1))  # (B, T, dim)
    tgt_exp = tgt_eval.unsqueeze(0).expand(B, -1, -1)
    mse_b   = ((y_pred - tgt_exp) ** 2).mean(dim=(1, 2))
    std_y   = tgt_eval.std().item()
    return (mse_b.sqrt() / (std_y + 1e-12)).cpu().float().numpy()

# =============================================================================
# 7.  STREAMING TRAINING WITH CHECKPOINTS
#
#   At each checkpoint:
#     1. Solve Wout via float64 ridge regression
#     2a. (NRMSE_TYPE="training")   compute ridge residual NRMSE
#     2b. (NRMSE_TYPE="prediction") evaluate on held-out test segment
# =============================================================================

def streaming_train_checkpoints(rho, p_val, Win_b, A_b, Wp_b,
                                 mu, sigma, alpha, beta, noise_a,
                                 rho_p, pb, washout, checkpoint_steps,
                                 dt, warmup, seed, mode, nrmse_type,
                                 chunk_size=CHUNK_SIZE, test_steps=TEST_STEPS):
    B = Win_b.shape[0]
    N = Win_b.shape[1]
    d = Win_b.shape[2]

    # Parameter contribution (zero for standard RC)
    p_contrib = rho_p * Wp_b[:, :, 0] * (p_val + pb)   # (B, N)

    # ── Washout ───────────────────────────────────────────────────────────────
    lorenz_seed = int(rho * 10)
    state = lorenz_initial_state(rho, dt, warmup, lorenz_seed)
    rng_noise = np.random.RandomState(seed + 9000)
    r = torch.zeros(B, N, device=DEVICE, dtype=DTYPE)
    ws_done = 0
    while ws_done < washout:
        cs  = min(chunk_size, washout - ws_done)
        raw = lorenz_chunk_from_state(state, cs, dt, rho)
        state = raw[-1].copy()
        u_norm = normalise(raw[:-1], mu, sigma)
        if mode == "adaptable":
            noise  = rng_noise.randn(cs, d).astype(np.float32) * float(noise_a)
            u_norm = u_norm + noise
        u_gpu = torch.from_numpy(np.ascontiguousarray(u_norm, dtype=np.float32)).to(DEVICE)
        WinU  = torch.einsum('bni,ti->btn', Win_b, u_gpu)
        r, _  = rc_forward(r, A_b, WinU, p_contrib, alpha, mode)
        ws_done += cs

    # ── Float64 accumulators ──────────────────────────────────────────────────
    F64     = torch.float64
    RRT     = torch.zeros(B, N, N, device=DEVICE, dtype=F64)
    YRT     = torch.zeros(B, d, N, device=DEVICE, dtype=F64)
    Y_sq_s  = torch.zeros(d,     device=DEVICE, dtype=F64)
    Y_sum_s = torch.zeros(d,     device=DEVICE, dtype=F64)
    Y_cnt   = 0

    nrmse_at_cp = {}
    T_total   = checkpoint_steps[-1]
    T_done    = 0
    cp_sorted = sorted(checkpoint_steps)
    cp_idx    = 0

    while T_done < T_total:
        next_cp = cp_sorted[cp_idx] if cp_idx < len(cp_sorted) else T_total
        cs      = min(chunk_size, next_cp - T_done)

        raw      = lorenz_chunk_from_state(state, cs, dt, rho)
        state    = raw[-1].copy()
        inp_norm = normalise(raw[:-1], mu, sigma)
        tgt_norm = normalise(raw[1:],  mu, sigma)
        if mode == "adaptable":
            noise    = rng_noise.randn(cs, d).astype(np.float32) * float(noise_a)
            inp_norm = inp_norm + noise
        inp_gpu = torch.from_numpy(np.ascontiguousarray(inp_norm, dtype=np.float32)).to(DEVICE)
        tgt_gpu = torch.from_numpy(np.ascontiguousarray(tgt_norm, dtype=np.float32)).to(DEVICE)
        WinU    = torch.einsum('bni,ti->btn', Win_b, inp_gpu)
        r, r_buf = rc_forward(r, A_b, WinU, p_contrib, alpha, mode)

        r64   = r_buf.to(F64)
        tgt64 = tgt_gpu.to(F64)
        tgt64_exp = tgt64.unsqueeze(0).expand(B, -1, -1)

        RRT     += torch.bmm(r64.permute(0, 2, 1), r64)
        YRT     += torch.bmm(tgt64_exp.permute(0, 2, 1), r64)
        Y_sum_s += tgt64.sum(dim=0)
        Y_sq_s  += (tgt64 ** 2).sum(dim=0)
        Y_cnt   += cs
        T_done  += cs

        # ── Checkpoints ───────────────────────────────────────────────────────
        while cp_idx < len(cp_sorted) and T_done >= cp_sorted[cp_idx]:
            cp = cp_sorted[cp_idx]

            # Solve ridge regression (float64)
            beta_I = beta * torch.eye(N, device=DEVICE, dtype=F64).unsqueeze(0)
            Wout_b = torch.linalg.solve(
                RRT + beta_I, YRT.permute(0, 2, 1)
            ).permute(0, 2, 1)   # (B, dim, N)

            if nrmse_type == "training":
                # ── Ridge residual NRMSE on training data ─────────────────────
                T_eff    = float(Y_cnt)
                sum_sq_Y = Y_sq_s.sum().item()
                sum_Y    = Y_sum_s.sum().item()
                YW_trace = torch.bmm(YRT, Wout_b.permute(0, 2, 1))
                tr_YW    = YW_trace.diagonal(dim1=1, dim2=2).sum(dim=1)
                W_sq     = (Wout_b ** 2).sum(dim=(1, 2))
                sq_err_b = (sum_sq_Y - tr_YW - beta * W_sq) / (d * T_eff)
                sq_err_b = sq_err_b.clamp(min=0).cpu().numpy()
                mean_Y   = sum_Y / (d * T_eff)
                var_Y    = sum_sq_Y / (d * T_eff) - mean_Y ** 2
                std_Y    = float(max(var_Y, 1e-24) ** 0.5)
                nrmse_arr = np.sqrt(sq_err_b) / (std_Y + 1e-12)
            else:
                # ── Prediction NRMSE on held-out test segment ─────────────────
                Wout_f32  = Wout_b.to(DTYPE)
                nrmse_arr = evaluate_on_test(
                    Wout_f32, Win_b, A_b, Wp_b,
                    mu, sigma, rho, dt, warmup, alpha,
                    rho_p, pb, mode, test_steps=test_steps
                )

            nrmse_at_cp[cp] = nrmse_arr.copy()
            valid = nrmse_arr[np.isfinite(nrmse_arr) & (nrmse_arr > 0)]
            label = "train-NRMSE" if nrmse_type == "training" else "pred-NRMSE"
            print(f"      T={T_done:>10,}  {label}={valid.mean():.6f}"
                  f" ± {valid.std():.6f}  (n={len(valid)})", flush=True)
            cp_idx += 1

    return nrmse_at_cp

# =============================================================================
# 8.  HYPERPARAMETERS
# =============================================================================

TASK6_PKL = os.path.join("task6_results", "task6_results.pkl")
if os.path.exists(TASK6_PKL):
    with open(TASK6_PKL, "rb") as f:
        t6 = pickle.load(f)
    HP = t6["bayes"]["params"].copy()
    print(f"\nLoaded Task-6 Bayesian params  (Val-NRMSE={t6['bayes']['rmse']:.6f})")
else:
    print("\nWARNING: task6_results.pkl not found — using reference Lorenz params")
    HP = {"d": 0.027, "eig_rho": 4.78, "gamma": 0.27,
          "alpha": 0.41, "beta_log": -3.36, "noise_log": -7.0}
print(f"HP: {HP}\n")

ALPHA   = HP["alpha"]
BETA    = 10.0 ** HP["beta_log"]
NOISE_A = 10.0 ** HP["noise_log"]   # only used in adaptable mode
print(f"α={ALPHA:.4f}  β={BETA:.4e}  noise_a={NOISE_A:.4e}")

# =============================================================================
# 9.  PERIOD + CHECKPOINTS
# =============================================================================

print("\nEstimating oscillation period ...")
period_steps = {}
for rho in RHO_VALS:
    ps = average_period_steps(rho, DT, WARMUP)
    period_steps[rho] = ps
    print(f"  ρ={rho:5.1f}  →  {ps} steps  =  {ps*DT:.3f} time-units")
mean_period = int(np.round(np.mean(list(period_steps.values()))))
print(f"  Mean period = {mean_period} steps")

# ── Step-resolution checkpoints: LEFT SIDE (very small T, "large error" regime)
SMALL_T_STEPS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

# ── Cycle-resolution checkpoints: RIGHT SIDE (converging regime)
CYCLE_VALS = [
    1, 2, 3, 5,
    10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 1500, 2000, 3000, 4000, 5000, 10000, 100000
]
CYCLE_T_STEPS = [nc * mean_period for nc in CYCLE_VALS]

# Merge, deduplicate, sort — unified checkpoint list
T_STEPS   = sorted(set(SMALL_T_STEPS + CYCLE_T_STEPS))
cycle_set = set(CYCLE_T_STEPS)
small_set = set(SMALL_T_STEPS)

print("=" * 65)
print(f"Task 7  |  MODE={MODE}  |  NRMSE={NRMSE_TYPE}  |  "
      f"N_RES={N_RES}  |  {N_SEEDS} seeds")
print("=" * 65)
print(f"\n  Small-T steps (left side) : {SMALL_T_STEPS}")
print(f"  Cycle vals  (right side)  : {CYCLE_VALS}")
print(f"  Total checkpoints: {len(T_STEPS)}")
print(f"  T_max \u2248 {T_STEPS[-1]:,} steps  \u2248 {T_STEPS[-1]*DT/3600:.2f} h\n")
# =============================================================================
# 10.  NORM STATS + RESERVOIRS
# =============================================================================

print("Computing normalisation stats ...")
mu_fix, sigma_fix = {}, {}
for rho in RHO_VALS:
    mu_fix[rho], sigma_fix[rho] = compute_norm_stats(
        rho, DT, WARMUP, seed=int(rho*10))
    print(f"  ρ={rho}  mu={np.round(mu_fix[rho],3)}  sigma={np.round(sigma_fix[rho],3)}")

print(f"\nBuilding {N_SEEDS} reservoir batches (N={N_RES}, mode={MODE}) ...")
SEEDS    = [si * 137 + 42 for si in range(N_SEEDS)]
t0_build = time.time()
Win_b, A_b, Wp_b = build_batch_gpu(
    N_SEEDS, N_RES, HP["d"], HP["eig_rho"], HP["gamma"], DIM, SEEDS, mode=MODE)
print(f"  Done in {time.time()-t0_build:.1f}s")
print(f"  Win_b: {tuple(Win_b.shape)}  A_b: {tuple(A_b.shape)}")

# =============================================================================
# 11.  MAIN SWEEP
# =============================================================================

print("\n" + "="*65)
print(f"Streaming training sweep  "
      f"({N_SEEDS} seeds × {len(T_STEPS)} checkpoints × {len(RHO_VALS)} ρ)")
print(f"  NRMSE = {'ridge residual on training data' if NRMSE_TYPE=='training' else 'one-step-ahead prediction on held-out test'}")
print("="*65)

all_nrmses = {rho: {} for rho in RHO_VALS}
total_t0   = time.time()

for rho in RHO_VALS:
    print(f"\n── ρ = {rho} ─────────────────────────────────────")
    t0_rho = time.time()
    p_val  = rho if MODE == "adaptable" else 0.0

    nrmse_map = streaming_train_checkpoints(
        rho, p_val,
        Win_b, A_b, Wp_b,
        mu_fix[rho], sigma_fix[rho],
        ALPHA, BETA, NOISE_A,
        RHO_P, PB, WASHOUT, T_STEPS,
        DT, WARMUP, seed=42,
        mode=MODE, nrmse_type=NRMSE_TYPE
    )
    for T_s, arr in nrmse_map.items():
        all_nrmses[rho][T_s] = arr
    print(f"  ρ={rho} done in {time.time()-t0_rho:.1f}s")

print(f"\nTotal sweep: {time.time()-total_t0:.1f}s")

# =============================================================================
# 12.  AGGREGATE RESULTS
# =============================================================================

T_arr = np.array(T_STEPS)
C_arr = T_arr.astype(float)   # x-axis in steps (unified — includes sub-cycle points)

mean_res = {rho: np.zeros(len(T_STEPS)) for rho in RHO_VALS}
std_res  = {rho: np.zeros(len(T_STEPS)) for rho in RHO_VALS}

print("\n" + "="*65 + f"\nAggregated results  ({NRMSE_TYPE} NRMSE)")
print("="*65)
for i, ts in enumerate(T_STEPS):
    lbl = f"{ts}s" if (ts in small_set and ts not in cycle_set) else f"{ts/mean_period:.1f}c"
    print(f"\n  T = {ts:,} steps ({lbl}):")
    for rho in RHO_VALS:
        arr   = all_nrmses[rho].get(ts, np.array([np.nan]))
        arr_v = arr[np.isfinite(arr) & (arr > 0)]
        mean_res[rho][i] = arr_v.mean() if len(arr_v) else np.nan
        std_res[rho][i]  = arr_v.std()  if len(arr_v) else np.nan
        print(f"    ρ={rho:5.1f}  {mean_res[rho][i]:.6f} ± {std_res[rho][i]:.6f}"
              f"  (n={len(arr_v)})")

# =============================================================================
# 13.  POWER-LAW FIT
# =============================================================================

fit_mask   = T_arr >= (FIT_MIN_CYCLES * mean_period)  # exclude T < FIT_MIN_CYCLES cycles
fit_params = {}
print(f"\n{'='*65}")
print(f"Power-law fit: NRMSE(T) ≈ a·T^b  (fit region: T ≥ {FIT_MIN_CYCLES} cycles)")
print("="*65)
for rho in RHO_VALS:
    mn    = mean_res[rho]
    valid = fit_mask & np.isfinite(mn) & (mn > 0)
    if valid.sum() < 3:
        fit_params[rho] = (np.nan, np.nan, np.nan); continue
    slope, intercept, r_val, _, _ = linregress(
        np.log(T_arr[valid]), np.log(mn[valid]))
    fit_params[rho] = (slope, np.exp(intercept), r_val**2)
    tag = ("b<0: data helps" if slope < 0
           else "b≈0: saturated" if abs(slope) < 0.01 else "b>0: investigate")
    print(f"  ρ={rho:5.1f}:  a={np.exp(intercept):.4e}  b={slope:.4f}"
          f"  R²={r_val**2:.4f}  → {tag}")

# =============================================================================
# 14.  PLOTS  (all 6)
# =============================================================================

# Color/marker scheme
colors  = {25.0: "#d62728", 28.0: "steelblue", 31.0: "#2ca02c"}
markers = {25.0: "o",       28.0: "o",          31.0: "^"}

nrmse_label = ("Training NRMSE" if NRMSE_TYPE == "training"
               else "Prediction NRMSE")
mode_label  = ("Standard RC"
               if MODE == "standard"
               else "Adaptable RC")

print("\nGenerating plots ...")

# ── Plot mask: only show T >= 10 steps (exclude sub-10 points from plots)
PLOT_MIN_STEPS = 2
plot_mask = T_arr >= PLOT_MIN_STEPS

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Log-Log  (PRIMARY)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
for rho in RHO_VALS:
    mn, sd = mean_res[rho], std_res[rho]
    v  = np.isfinite(mn) & (mn > 0) & plot_mask
    vf = fit_mask & v
    lo = np.maximum(mn - sd, mn * 0.01)
    ax.errorbar(C_arr[v], mn[v], yerr=[mn[v]-lo[v], sd[v]],
                color=colors[rho], marker=markers[rho], lw=2, ms=9,
                capsize=6, capthick=2, elinewidth=1.8,
                label=f"ρ={rho}  (mean±std, n={N_SEEDS})", zorder=3)
    b, a, r2 = fit_params[rho]
    if np.isfinite(b) and vf.sum() > 0:
        C_fit = np.exp(np.linspace(np.log(C_arr[vf][0]),
                                    np.log(C_arr[vf][-1]), 300))
        T_fit = C_fit * mean_period
        ax.plot(C_fit, a*(T_fit**b), "--", color=colors[rho], lw=2, alpha=0.7,
                label=f"ρ={rho}  α={b:.3f}, R²={r2:.3f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Training length T  (steps)", fontsize=14)
ax.set_ylabel(f"{nrmse_label}  (log scale)", fontsize=14)
ax.set_title(
    f"Task 7 — Log-Log: NRMSE(T) ≈ a·T^b   [N={N_RES}, {N_SEEDS} seeds]\n"
    f"{mode_label}  |  {NRMSE_TYPE} NRMSE  |  Cycles: {CYCLE_VALS[0]}–{CYCLE_VALS[-1]}",
    fontsize=10, fontweight="bold")
ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.3, which="both")
plt.savefig(out("task7_fig1_loglog.png"), dpi=150, bbox_inches="tight")
print("  Saved task7_fig1_loglog.png"); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Linear scale
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
for rho in RHO_VALS:
    mn, sd = mean_res[rho], std_res[rho]
    v  = np.isfinite(mn) & (mn > 0) & plot_mask
    vf = fit_mask & v
    ax.errorbar(C_arr[v], mn[v], yerr=sd[v],
                color=colors[rho], marker=markers[rho],
                lw=2, ms=9, capsize=6, capthick=2, elinewidth=1.8,
                label=f"ρ={rho}")
    b, a, r2 = fit_params[rho]
    if np.isfinite(b) and vf.sum() > 0:
        C_fit = np.exp(np.linspace(np.log(C_arr[vf][0]),
                                    np.log(C_arr[vf][-1]), 300))
        T_fit = C_fit * mean_period
        ax.plot(C_fit, a*(T_fit**b), "--", color=colors[rho],
                lw=1.8, alpha=0.65, label=f"ρ={rho}  b={b:.3f}")
ax.set_xlabel("T (steps)", fontsize=14)
ax.set_ylabel(nrmse_label, fontsize=14)
ax.set_title(f"Task 7 — Linear scale  ({nrmse_label})",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.3)
plt.savefig(out("task7_fig2_linear.png"), dpi=150, bbox_inches="tight")
print("  Saved task7_fig2_linear.png"); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Semi-log diagnostic
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
for rho in RHO_VALS:
    mn, sd = mean_res[rho], std_res[rho]
    v = np.isfinite(mn) & (mn > 0) & plot_mask
    ax.errorbar(C_arr[v], mn[v], yerr=sd[v],
                color=colors[rho], marker=markers[rho],
                lw=2, ms=9, capsize=6, capthick=2, elinewidth=1.8,
                label=f"ρ={rho}")
ax.set_yscale("log")
ax.text(0.04, 0.08,
        "Power-law → curved on semi-log (straight on log-log)\n"
        "Exponential → straight on semi-log",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.85))
ax.set_xlabel("T (steps)", fontsize=14)
ax.set_ylabel(f"{nrmse_label} (log)", fontsize=14)
ax.set_title("Task 7 — Semi-log (power-law vs exponential diagnostic)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=12); ax.grid(alpha=0.3, which="both")
plt.savefig(out("task7_fig3_semilogy.png"), dpi=150, bbox_inches="tight")
print("  Saved task7_fig3_semilogy.png"); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Per-rho subplots with point annotations
# ─────────────────────────────────────────────────────────────────────────────
ncols = len(RHO_VALS)
fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 6), constrained_layout=True)
if ncols == 1:
    axes = [axes]
fig.suptitle(
    f"Task 7 — {nrmse_label} per ρ   [N={N_RES}, {N_SEEDS} seeds, "
    f"Cycles {CYCLE_VALS[0]}–{CYCLE_VALS[-1]}]",
    fontsize=12, fontweight="bold")
for ax, rho in zip(axes, RHO_VALS):
    mn, sd   = mean_res[rho], std_res[rho]
    b, a, r2 = fit_params[rho]
    v  = np.isfinite(mn) & (mn > 0) & plot_mask
    vf = fit_mask & v
    lo = np.maximum(mn - sd, mn * 0.01)
    ax.errorbar(C_arr[v], mn[v], yerr=[mn[v]-lo[v], sd[v]],
                color=colors[rho], marker=markers[rho],
                lw=2, ms=10, capsize=6, capthick=2, elinewidth=1.8,
                zorder=3, label="Mean ± std")
    if np.isfinite(b) and vf.sum() > 0:
        C_fit = np.exp(np.linspace(np.log(C_arr[vf][0]),
                                    np.log(C_arr[vf][-1]), 300))
        T_fit = C_fit * mean_period
        ax.plot(C_fit, a*(T_fit**b), "k--", lw=2.5,
                label=f"{a:.2e}·T^({b:.3f})\nR²={r2:.3f}")
    for cx, my in zip(C_arr[v], mn[v]):
        ax.annotate(f"{my:.4f}", (cx, my),
                    textcoords="offset points", xytext=(3, 6),
                    fontsize=6.5, color=colors[rho], alpha=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("T (steps)", fontsize=12)
    ax.set_ylabel(nrmse_label, fontsize=12)
    ax.set_title(f"ρ={rho}  b={b:.3f}  R²={r2:.3f}",
                 fontsize=13, color=colors[rho], fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
plt.savefig(out("task7_fig4_per_rho.png"), dpi=150, bbox_inches="tight")
print("  Saved task7_fig4_per_rho.png"); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Power-law parameters a and b (bar charts)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
fig.suptitle(f"Task 7 — Power-Law Parameters a and b  ({nrmse_label})",
             fontsize=13, fontweight="bold")
b_vals = [fit_params[r][0] for r in RHO_VALS]
a_vals = [fit_params[r][1] for r in RHO_VALS]
lbls   = [f"ρ={r}" for r in RHO_VALS]
cols   = [colors[r] for r in RHO_VALS]

bars = axes[0].bar(lbls, b_vals, color=cols, edgecolor="k", width=0.4)
for bar, v in zip(bars, b_vals):
    if np.isfinite(v):
        axes[0].text(bar.get_x()+bar.get_width()/2, v - abs(v)*0.05,
                     f"{v:.3f}", ha="center", va="top",
                     fontsize=13, fontweight="bold", color="white")
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_ylabel("Exponent b", fontsize=12)
axes[0].set_title("b<0: data helps | b≈0: saturated | b>0: investigate",
                  fontsize=10)
axes[0].grid(axis="y", alpha=0.3)

bars2 = axes[1].bar(lbls, a_vals, color=cols, edgecolor="k", width=0.4)
for bar, v in zip(bars2, a_vals):
    if np.isfinite(v):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     v + max(a_vals)*0.02, f"{v:.4f}",
                     ha="center", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Prefactor a", fontsize=12)
axes[1].set_title("Prefactor a (base error at T=1)", fontsize=11)
axes[1].set_ylim(0, max(a_vals)*1.3)
axes[1].grid(axis="y", alpha=0.3)
plt.savefig(out("task7_fig5_ab_params.png"), dpi=150, bbox_inches="tight")
print("  Saved task7_fig5_ab_params.png"); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Reference-style (Zheng-Meng format, x-axis = steps)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 6), constrained_layout=True)
if ncols == 1:
    axes = [axes]
fig.suptitle(
    f"LORENZ: {nrmse_label} vs Training Length  (N={N_RES}, {N_SEEDS} seeds)",
    fontsize=13, fontweight="bold")
for ax, rho in zip(axes, RHO_VALS):
    mn, sd   = mean_res[rho], std_res[rho]
    b, a, r2 = fit_params[rho]
    v  = np.isfinite(mn) & (mn > 0) & plot_mask
    vf = fit_mask & v
    lo = np.maximum(mn - sd, mn * 0.02)

    ax.errorbar(T_arr[v], mn[v], yerr=[mn[v]-lo[v], sd[v]],
                fmt='-o', color='steelblue', lw=2, ms=6,
                capsize=5, capthick=1.8, elinewidth=1.5,
                ecolor='steelblue', zorder=3, label='_nolegend_')

    # Gray out short-T excluded points
    excl = ~fit_mask & v
    if excl.sum() > 0:
        ax.scatter(T_arr[excl], mn[excl], color='lightgray', s=60,
                   zorder=4, label=f'T < {FIT_MIN_CYCLES} cycles (excluded)')

    if np.isfinite(b) and vf.sum() > 0:
        x_fit = np.exp(np.linspace(np.log(T_arr[vf][0]),
                                    np.log(T_arr[vf][-1]), 300))
        ax.plot(x_fit, a*(x_fit**b), 'r--', lw=2.5,
                label=f'Power law fit: α = {b:.3f},  R²={r2:.3f}')
        ax.plot(T_arr[vf], a*(T_arr[vf]**b),
                'o', color='red', ms=10, mfc='none', mew=1.8,
                label='Fitted points')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Training Length', fontsize=13)
    ax.set_ylabel(nrmse_label, fontsize=13)
    ax.set_title(f'ρ = {rho}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3, color='grey', linestyle='-')
    ax.set_facecolor('#f8f8f8')
plt.savefig(out("task7_fig6_reference_style.png"), dpi=150, bbox_inches="tight")
print("  Saved task7_fig6_reference_style.png"); plt.close()

# =============================================================================
# 15.  SUMMARY TABLE + SAVE
# =============================================================================

print("\n" + "="*80)
print(f"TASK 7 — FINAL SUMMARY  |  MODE={MODE}  |  NRMSE={NRMSE_TYPE}")
print(f"  N_RES={N_RES}  |  {N_SEEDS} seeds  |  β={BETA:.4e}  |  α={ALPHA:.4f}")
print("="*80)
header = f"{'Cycles':>8}"
for rho in RHO_VALS:
    header += f"   ρ={rho:5.1f}  mean       ±std"
print(header); print("-"*80)
for i, ts in enumerate(T_STEPS):
    lbl = f"{ts}s" if (ts in small_set and ts not in cycle_set) else f"{ts/mean_period:.0f}c"
    row = f"{lbl:>9}"
    for rho in RHO_VALS:
        mn = mean_res[rho][i]; sd = std_res[rho][i]
        row += (f"   {mn:.6f}  ±{sd:.6f}" if np.isfinite(mn) else "        NaN")
    print(row)
print("-"*80)
print(f"\nPower-law fits  (fit region: T ≥ {FIT_MIN_CYCLES} cycles):")
for rho in RHO_VALS:
    b, a, r2 = fit_params[rho]
    if np.isfinite(b):
        tag = ("b<0 ✓" if b < 0 else "b≈0" if abs(b) < 0.01 else "b>0 ⚠")
        print(f"  ρ={rho:5.1f}:  a={a:.4e}  b={b:.4f}  R²={r2:.4f}   {tag}")

with open(out("task7_results.pkl"), "wb") as f:
    pickle.dump({
        "mode": MODE, "nrmse_type": NRMSE_TYPE,
        "rho_vals": RHO_VALS, "cycle_vals": CYCLE_VALS, "t_steps": T_STEPS,
        "mean_period": mean_period, "period_steps": period_steps,
        "n_seeds": N_SEEDS, "n_res": N_RES,
        "beta": BETA, "alpha": ALPHA, "noise_a": NOISE_A,
        "washout": WASHOUT, "fit_min_cycles": FIT_MIN_CYCLES,
        "mean_results": mean_res, "std_results": std_res,
        "fit_params": fit_params,
        "HP": HP, "mu_fix": mu_fix, "sigma_fix": sigma_fix,
        "all_nrmses": all_nrmses,
    }, f)
print(f"\nResults saved to {OUT_DIR}/task7_results.pkl")
print("Task 7 complete.")