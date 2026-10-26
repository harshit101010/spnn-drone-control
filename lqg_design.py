"""LQG design that includes the measured velocity-loop lag (per axis). Shared by the Python sim and AirSim code."""
import json, os
import numpy as np
from scipy.linalg import solve_discrete_are, expm


def plant_taus():
    """Effective lag (lag + dead time) for x, y, z from identify_lag.py; defaults to 0.1 s."""
    if os.path.exists("plant_lag.json"):
        d = json.load(open("plant_lag.json"))["details"]
        return [sum(d["x (forward)"]), sum(d["y (sideways)"]), sum(d["z (climb)"])]
    return [0.1, 0.1, 0.1]


def _zoh(Ac, Bc, dt):
    n = Ac.shape[0]; M = np.zeros((n + 1, n + 1)); M[:n, :n] = Ac * dt; M[:n, n:] = Bc * dt
    E = expm(M); return E[:n, :n], E[:n, n:]


def design(taus, dt, qe, qi, qv, qa, r, sp, sv):
    """Per axis: LQR on z=[int e, e, v_err, a_act] and a Kalman filter on [p, v, a_act]."""
    K, Ad_e, Be, L = [], [], [], []
    C = np.array([[1.0, 0, 0], [0, 1.0, 0]]); Rn = np.diag([sp ** 2, sv ** 2]); Qn = np.diag([1e-4, 1e-3, 0.05])
    for tau in taus:
        Ac = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1 / tau]], float)
        A, B = _zoh(Ac, np.array([[0], [0], [0], [1 / tau]], float), dt)
        P = solve_discrete_are(A, B, np.diag([qi, qe, qv, qa]), np.array([[r]]))
        K.append(np.linalg.solve(r + B.T @ P @ B, B.T @ P @ A)[0])
        Ae = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1 / tau]], float)
        Ad, Bd = _zoh(Ae, np.array([[0], [0], [1 / tau]], float), dt)
        S = solve_discrete_are(Ad.T, C.T, Qn, Rn)
        L.append(S @ C.T @ np.linalg.inv(C @ S @ C.T + Rn)); Ad_e.append(Ad); Be.append(Bd[:, 0])
    return np.array(K), np.array(Ad_e), np.array(Be), np.array(L)
