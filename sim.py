"""3-axis drone point-mass simulator (PyTorch, batched, differentiable) following the mission path.
Models: actuator lag, quadratic drag (nonlinear), mass mismatch, wind gust, sensor noise."""
import math
import numpy as np
import torch
from path import T_TOTAL, default_path_params, sample_path_params, refs

G, DT, T = 9.81, 0.05, T_TOTAL
N = int(round(T / DT))
U_MAX = 8.0                 # max commanded acceleration (m/s^2), same for all controllers
from lqg_design import plant_taus
# per-axis velocity-loop lag (s): defaults to 0.1; identify_lag.py measures the real values in AirSim -> plant_lag.json
TAU_A = torch.tensor(plant_taus(), dtype=torch.float32)
GFF = torch.tensor([0.0, 0.0, G])        # hover feed-forward (nominal mass)


def make_scenario(B, kind, seed, scale=1.0):
    """kind: nominal | gust | mismatch | noise | combined (all on the canonical path),
             train (random path + random disturbances), unseen (wider paths, 1.5x harder)"""
    g = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
    U = lambda lo, hi, shape=(B, 1): lo + (hi - lo) * torch.rand(shape, generator=g)
    flag = lambda p: (torch.rand((B, 1), generator=g) < p).float()
    ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)
    if kind == "nominal":    fg, fm, fn = zeros, zeros, zeros
    elif kind == "gust":     fg, fm, fn = ones, zeros, zeros
    elif kind == "mismatch": fg, fm, fn = zeros, ones, zeros
    elif kind == "noise":    fg, fm, fn = zeros, zeros, ones
    elif kind in ("combined", "unseen", "extreme"): fg, fm, fn = ones, ones, ones
    elif kind == "train":    fg, fm, fn = flag(.7), flag(.7), flag(.7)
    else: raise ValueError(kind)
    if kind == "unseen": scale = 1.5
    if kind == "extreme": scale = 2.0
    path = sample_path_params(rng, B, wide=(kind in ("unseen", "extreme"))) if kind in ("train", "unseen", "extreme") else default_path_params(B)
    return dict(
        m=1 + fm * scale * U(-0.2, 0.2),
        cd=(0.15 + fm * scale * U(-0.1, 0.15)).clamp(min=0.0),
        amp=fg * scale * 3.0 * (2 * torch.rand((B, 3), generator=g) - 1),
        t0=U(6.0, 30.0), dur=U(1.0, 3.0),                     # gust hits somewhere along the path
        sp=0.01 + fn * scale * 0.04, sv=0.03 + fn * scale * 0.10, path=path,
    )


def gust_acc(t, sc):
    t0, dur = sc["t0"], sc["dur"]
    on = ((t >= t0) & (t < t0 + dur)).float()
    shape = 0.5 * (1 - torch.cos(2 * math.pi * (t - t0) / dur))
    return sc["amp"] * on * shape


def simulate(ctrl, sc, seed=0):
    """Returns P (N,B,3), U (N,B,3), R (N,B,3) reference positions."""
    B = sc["m"].shape[0]
    Rn, Vn = refs(sc["path"], DT, N)
    Rt, Vt = torch.from_numpy(Rn), torch.from_numpy(Vn)
    gn = torch.Generator().manual_seed(seed + 1)
    p, v, ua = torch.zeros(B, 3), torch.zeros(B, 3), torch.zeros(B, 3)
    ctrl.reset(B)
    Ps, Us = [], []
    for k in range(N):
        t = k * DT
        pm = p + sc["sp"] * torch.randn(B, 3, generator=gn)
        vm = v + sc["sv"] * torch.randn(B, 3, generator=gn)
        u = ctrl(Rt[k], Vt[k], pm, vm, DT).clamp(-U_MAX, U_MAX)
        ua = ua + (u - ua) * (DT / TAU_A)
        a = (ua + GFF) / sc["m"] - GFF - sc["cd"] * v.abs() * v / sc["m"] + gust_acc(t, sc) / sc["m"]
        v = v + a * DT
        p = p + v * DT
        Ps.append(p); Us.append(u)
    return torch.stack(Ps), torch.stack(Us), Rt


def loss_fn(P, U, R):
    """Same cost used to TRAIN SPNN and to TUNE PID / LQG (fair comparison)."""
    e = P - R
    dU = U[1:] - U[:-1]
    return (e ** 2).mean() + 0.005 * (U ** 2).mean() + 0.02 * (dU ** 2).mean()
