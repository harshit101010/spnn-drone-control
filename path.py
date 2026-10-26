"""Mission path (numpy, shared by the Python sim and AirSim so both use the SAME path).
0-1 s hold | 1-5 CLIMB | 5-11 STRAIGHT line (+X) | 11-19 CURVE (semicircle) |
19-25 diagonal line + climb | 25-33 S-curve WEAVE | 33-42 RETURN home + descend | 42-45 hold
Axes: x, y, z-up (m), relative to the start point."""
import numpy as np

T_TOTAL = 45.0

def _s(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)

def default_path_params(B):
    f = lambda v: np.full(B, v, float)
    return dict(L=f(10.0), R=f(4.0), H1=f(3.0), H2=f(5.0), A=f(2.0))

def sample_path_params(rng, B, wide=False):
    U = lambda lo, hi: rng.uniform(lo, hi, B)
    if wide:   # beyond the training ranges (tests generalisation)
        return dict(L=U(6, 15), R=U(2.5, 6), H1=U(1.5, 5), H2=U(2.5, 7), A=U(1.0, 3.5))
    return dict(L=U(8, 12), R=U(3, 5), H1=U(2, 4), H2=U(3, 6), A=U(1.5, 3.0))

def path_pos(t, p):
    L, R, H1, H2, A = p["L"], p["R"], p["H1"], p["H2"], p["A"]
    z = np.zeros_like(L); V = lambda a, b, c: np.stack([a + z, b + z, c + z], 1)
    P1, P2, P3, P4, P5 = V(z, z, H1), V(L, z, H1), V(L, 2 * R, H1), V(z, 2 * R, H2), V(z, z, H2)
    if t < 1:    return V(z, z, z)
    if t < 5:    return V(z, z, H1 * _s((t - 1) / 4))
    if t < 11:   return P1 + (P2 - P1) * _s((t - 5) / 6)
    if t < 19:
        th = np.pi * _s((t - 11) / 8); return V(L + R * np.sin(th), R * (1 - np.cos(th)), H1)
    if t < 25:   return P3 + (P4 - P3) * _s((t - 19) / 6)
    if t < 33:
        s = _s((t - 25) / 8); return V(A * np.sin(2 * np.pi * s), 2 * R * (1 - s), H2)
    if t < 42:   return P5 + (V(z, z, z) - P5) * _s((t - 33) / 9)
    return V(z, z, z)

def refs(p, dt, n, h=0.01):
    """Reference position and velocity tables, shape (n, B, 3) each."""
    pos = np.stack([path_pos(k * dt, p) for k in range(n)])
    vel = np.stack([(path_pos(k * dt + h, p) - path_pos(k * dt - h, p)) / (2 * h) for k in range(n)])
    return pos.astype(np.float32), vel.astype(np.float32)
