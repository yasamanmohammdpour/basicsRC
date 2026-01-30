import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs

# ============================================================
# SETTINGS
# ============================================================

DT = 0.01
T_TOTAL = 5000
SHORT_TERM = 800
ATTRACTOR_CUT_FRAC = 0.3

rng = np.random.default_rng(42)

PARAMS_BY_SYSTEM = {

    "lorenz": dict(
        N=200,
        p=0.027,
        rho=4.78,
        gamma=0.27,
        alpha=0.41,
        beta=10**(-3.36),
        washout=1000,
        train_frac=0.6
    ),

    "aizawa": dict(
        N=300,
        p=0.03,
        rho=2.5,
        gamma=0.4,
        alpha=0.5,
        beta=1e-5,
        washout=1500,
        train_frac=0.6
    ),

    "food_chain": dict(
        N=300,
        p=0.04,
        rho=1.2,
        gamma=0.2,
        alpha=0.25,
        beta=1e-6,
        washout=2000,
        train_frac=0.7
    ),

    "hastings_powell": dict(
        N=400,
        p=0.05,
        rho=1.5,
        gamma=0.3,
        alpha=0.35,
        beta=1e-6,
        washout=3000,
        train_frac=0.7
    )
}


# ============================================================
# DYNAMICAL SYSTEMS
# ============================================================

def lorenz(state):
    x, y, z = state
    sigma, rho, beta = 10.0, 28.0, 8/3
    return np.array([
        sigma*(y - x),
        x*(rho - z) - y,
        x*y - beta*z
    ])

def aizawa(state):
    x, y, z = state
    a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
    return np.array([
        (z - b)*x - d*y,
        d*x + (z - b)*y,
        c + a*z - z**3/3 - (x**2 + y**2)*(1 + e*z) + f*z*x**3
    ])

def food_chain(state):
    R, C, P = state
    K, yc, yp, xc, xp = 0.98, 2.009, 2.876, 0.4, 0.08
    R0, C0 = 0.16129, 0.5
    return np.array([
        R*(1-R/K) - xc*yc*C*R/(R+R0),
        xc*C*(yc*R/(R+R0)-1) - xp*yp*P*C/(C+C0),
        xp*P*(yp*C/(C+C0)-1)
    ])

def hastings_powell(state):
    V, H, P = state
    a1,a2,b1,b2,d1,d2 = 5.0,0.1,3.0,2.0,0.4,0.01
    return np.array([
        V*(1-V) - (a1*V*H)/(b1*V+1),
        (a1*V*H)/(b1*V+1) - (a2*H*P)/(b2*H+1) - d1*H,
        (a2*H*P)/(b2*H+1) - d2*P
    ])

# ============================================================
# RK4 INTEGRATOR
# ============================================================

def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def generate_data(system, T, dt, x0):
    N = int(T/dt)
    X = np.zeros((N, 3))
    X[0] = x0
    for i in range(N-1):
        X[i+1] = rk4_step(system, X[i], dt)
    return X

# ============================================================
# RESERVOIR
# ============================================================

def spectral_radius(A):
    try:
        vals, _ = eigs(A, k=1, which='LM')
        return np.abs(vals[0])
    except:
        return np.max(np.abs(np.linalg.eigvals(A)))

def generate_reservoir(N, p, rho):
    S = sparse_random(
        N, N, density=p,
        data_rvs=lambda s: rng.normal(0, 1, size=s),
        format='coo'
    )
    S = (S + S.T) * 0.5
    A = S.toarray()
    sr = spectral_radius(A)
    A *= rho / sr
    return A

# ============================================================
# TRAINING
# ============================================================

def reservoir_train(data, params):
    N,p,rho = params["N"],params["p"],params["rho"]
    gamma,alpha,beta = params["gamma"],params["alpha"],params["beta"]
    washout,train_frac = params["washout"],params["train_frac"]

    T = len(data)
    train_len = int(train_frac*T)

    mean = data[:train_len].mean(axis=0)
    std  = data[:train_len].std(axis=0)
    data_n = (data-mean)/std

    A = generate_reservoir(N,p,rho)
    W_in = gamma*(2*rng.random((N,3))-1)

    r = np.zeros(N)
    R,Y = [],[]

    for t in tqdm(range(train_len-1)):
        r = (1-alpha)*r + alpha*np.tanh(A@r + W_in@data_n[t])
        if t>=washout:
            R.append(r.copy())
            Y.append(data_n[t+1])

    R = np.array(R).T
    Y = np.array(Y).T
    W_out = Y@R.T@np.linalg.inv(R@R.T + beta*np.eye(N))

    return A,W_in,W_out,r,mean,std,train_len,alpha

# ============================================================
# PREDICTION
# ============================================================

def predict_short_term(data, model):
    A,W_in,W_out,r,mean,std,train_len,alpha = model
    data_n = (data-mean)/std
    pred = np.zeros_like(data_n[train_len:])
    u = data_n[train_len]
    for t in tqdm(range(len(pred))):
        r = (1-alpha)*r + alpha*np.tanh(A@r + W_in@u)
        u = W_out@r
        pred[t]=u
    return data[train_len:], pred*std+mean

def predict_attractor(data, model):
    A,W_in,W_out,r,mean,std,train_len,alpha = model
    data_n = (data-mean)/std
    pred = np.zeros_like(data_n[train_len:])
    for t in tqdm(range(len(pred))):
        u = data_n[train_len+t]
        r = (1-alpha)*r + alpha*np.tanh(A@r + W_in@u)
        pred[t]=W_out@r
    return data[train_len:], pred*std+mean

# ============================================================
# PLOTTING
# ============================================================

def plot_results(true,pred,labels,outdir):
    os.makedirs(outdir,exist_ok=True)
    cut = int(len(pred)*ATTRACTOR_CUT_FRAC)

    t = np.arange(SHORT_TERM)*DT
    fig,ax = plt.subplots(3,1,figsize=(8,6))
    for i in range(3):
        ax[i].plot(t,true[:SHORT_TERM,i],'b')
        ax[i].plot(t,pred[:SHORT_TERM,i],'orange')
        ax[i].set_ylabel(labels[i])
    ax[2].set_xlabel("t")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"panel_a.png"))
    plt.close()

    pairs=[(0,1),(0,2),(1,2)]
    for i,j in pairs:
        plt.figure()
        plt.plot(true[cut:,i],true[cut:,j],'b',alpha=0.4)
        plt.plot(pred[cut:,i],pred[cut:,j],'orange',alpha=0.4)
        plt.xlabel(labels[i]); plt.ylabel(labels[j])
        plt.savefig(os.path.join(outdir,f"phase_{labels[i]}_{labels[j]}.png"))
        plt.close()

# ============================================================
# MAIN LOOP (ALL SYSTEMS)
# ============================================================

systems = [
    ("lorenz", lorenz, [1,1,1], ["x","y","z"]),
    ("aizawa", aizawa, [0.1,0,0], ["x","y","z"]),
    ("food_chain", food_chain, [0.3,0.2,0.1], ["R","C","P"]),
    ("hastings_powell", hastings_powell, [0.5,0.3,0.1], ["V","H","P"]),
]

params = dict(
    N=200,
    p=0.027,
    rho=4.78,
    gamma=0.27,
    alpha=0.41,
    beta=10**(-3.36),
    washout=1000,
    train_frac=0.6
)

for name, system, x0, labels in systems:

    print(f"Running {name}...")

    params = PARAMS_BY_SYSTEM[name]
    data = generate_data(system, T_TOTAL, DT, np.array(x0))
    model = reservoir_train(data, params)
    true_s, pred_s = predict_short_term(data, model)
    true_a, pred_a = predict_attractor(data, model)

    plot_results(
        true_s, pred_s, labels,
        outdir=f"results/{name}/short_term"
    )

    plot_results(
        true_a, pred_a, labels,
        outdir=f"results/{name}/attractor"
    )
