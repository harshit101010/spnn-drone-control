"""Shared AirSim code: numpy controllers, wind scenarios, robust episode runner, metrics.
All three controllers use the SAME interface: acceleration command -> integrated to a
velocity setpoint -> moveByVelocityAsync (limited to V_MAX)."""
import os, sys, time, json
import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_lyapunov
from path import T_TOTAL, default_path_params, sample_path_params, refs

DT, T = 0.05, T_TOTAL
U_MAX, V_MAX, I_MAX = 8.0, 5.0, 5.0
HOVER_Z = -3.0                           # NED
Z_GUARD, FENCE = 1.5, 25.0               # abort if sinks 1.5 m below start / leaves 25 m box
KINDS = ["PID", "LQG", "PID+SPNN", "LQG+SPNN"]
BASE_KINDS = ["PID", "LQG"]
KEYS = ["lin.weight", "lin.bias", "prod.weight", "prod.bias",
        "hid.0.weight", "hid.0.bias", "hid.2.weight", "hid.2.bias"]


# ------------------------------------------------------------------ controllers (numpy)
class PID:
    def __init__(s, kp, ki, kd): s.kp, s.ki, s.kd = kp, ki, kd
    def reset(s): s.ie = np.zeros(3)
    def __call__(s, r, vr, p, v, dt):
        e = r - p; ve = v - vr
        u0 = s.kp * e + s.ki * s.ie - s.kd * ve
        s.ie = np.clip(s.ie + e * dt * (np.abs(u0) < U_MAX), -I_MAX, I_MAX)
        return np.clip(s.kp * e + s.ki * s.ie - s.kd * ve, -U_MAX, U_MAX)


class LQG:
    def __init__(s, qe, r, qi=1.0, qv=1.0, sp=0.03, sv=0.08):
        A = np.array([[1, DT, 0], [0, 1, -DT], [0, 0, 1.0]]); B = np.array([[0], [-DT ** 2 / 2], [DT]])
        P = solve_discrete_are(A, B, np.diag([qi, qe, qv]), np.array([[r]]))
        s.K = np.linalg.solve(r + B.T @ P @ B, B.T @ P @ A)[0]
        Ad = np.array([[1, DT], [0, 1.0]]); Rn = np.diag([sp ** 2, sv ** 2])
        S = solve_discrete_are(Ad.T, np.eye(2), np.diag([1e-4, 0.05]), Rn)
        s.L = S @ np.linalg.inv(S + Rn)
    def reset(s): s.ph = np.zeros(3); s.vh = np.zeros(3); s.ie = np.zeros(3); s.up = np.zeros(3)
    def __call__(s, r, vr, p, v, dt):
        pp = s.ph + dt * s.vh + 0.5 * dt ** 2 * s.up; vp = s.vh + dt * s.up
        ip, iv = p - pp, v - vp
        s.ph = pp + s.L[0, 0] * ip + s.L[0, 1] * iv; s.vh = vp + s.L[1, 0] * ip + s.L[1, 1] * iv
        e = r - s.ph; s.ie = np.clip(s.ie + e * dt, -I_MAX, I_MAX)
        s.up = np.clip(-(s.K[0] * s.ie + s.K[1] * e + s.K[2] * (s.vh - vr)), -U_MAX, U_MAX)
        return s.up


class SPNN:
    def __init__(s, W): s.W = {k: np.asarray(W[k], float) for k in KEYS}
    def reset(s): s.ie = np.zeros(3)
    def _f(s, e, v, ie):
        W = s.W; x = np.stack([e, v, ie], 1)
        P = np.stack([e * v, e * np.abs(e), v * np.abs(v), e * ie], 1)
        z = x @ W["lin.weight"].T + W["lin.bias"] + P @ W["prod.weight"].T + W["prod.bias"]
        h = np.tanh(x @ W["hid.0.weight"].T + W["hid.0.bias"]) @ W["hid.2.weight"].T + W["hid.2.bias"]
        return U_MAX * np.tanh((z + h)[:, 0] / U_MAX)
    def __call__(s, r, vr, p, v, dt):
        e = r - p; u = s._f(e, v - vr, s.ie)
        s.ie = np.clip(s.ie + e * dt * (np.abs(u) < 0.95 * U_MAX), -I_MAX, I_MAX)
        return u


class Comp:
    """Paper design: base controller + online-adaptive sigma-pi compensator (same law as controllers.Compensated)."""
    def __init__(s, base, gamma, leak=0.02, es=2.0, vs=5.0):
        s.base, s.gamma, s.leak, s.es, s.vs = base, float(gamma), leak, es, vs
        kp, kd = (base.kp, base.kd) if isinstance(base, PID) else (-base.K[1], base.K[2])
        P = solve_continuous_lyapunov(np.array([[0, 1], [-kp, -kd]]).T, -np.eye(2)); s.p12, s.p22 = P[0, 1], P[1, 1]
    def reset(s): s.base.reset(); s.W = np.zeros((3, 27))
    def __call__(s, r, vr, p, v, dt):
        ub = s.base(r, vr, p, v, dt); e, ed = r - p, vr - v
        x = np.clip(np.stack([e / s.es, v / s.vs, ub / U_MAX], 1), -1, 1)
        B = np.stack([np.ones_like(x), x, x ** 2], 2)
        beta = np.einsum('ai,aj,ak->aijk', B[:, 0], B[:, 1], B[:, 2]).reshape(3, 27)
        u_nn = (s.W * beta).sum(1); sg = s.p12 * e + s.p22 * ed
        s.W = np.clip(s.W + dt * (-s.gamma * sg[:, None] * beta - s.leak * s.W), -U_MAX, U_MAX)
        return np.clip(ub - u_nn, -U_MAX, U_MAX)


# ------------------------------------------------------------------ controller construction
def _params(tuned):
    P = json.load(open("params.json"))
    for k in BASE_KINDS:
        f = f"tuned_{k}.npz"
        if tuned and os.path.exists(f): P[k] = [float(x) for x in np.exp(np.load(f)["theta"])]
    gf = "gamma_airsim.json" if (tuned and os.path.exists("gamma_airsim.json")) else "gamma.json"
    G = json.load(open(gf)) if os.path.exists(gf) else {"PID": 30.0, "LQG": 30.0}
    return P, G

def init_theta(kind, tuned=False):
    return np.log(_params(tuned)[0][kind])

def make_controller(kind, theta=None, tuned=False):
    if theta is not None and kind in BASE_KINDS:               # used by the ES tuner
        return PID(*np.exp(theta)) if kind == "PID" else LQG(*np.exp(theta))
    P, G = _params(tuned); base = kind.split("+")[0]
    mk = {"PID": lambda: PID(*P["PID"]), "LQG": lambda: LQG(*P["LQG"])}[base]
    return Comp(mk(), G[base]) if "+" in kind else mk()


# ------------------------------------------------------------------ wind scenarios
def sample_scenario(seed, scale=None, random_path=False):
    rng = np.random.default_rng(seed)
    path = sample_path_params(rng, 1) if random_path else default_path_params(1)
    if scale is None: scale = 1.0 if rng.random() < 0.7 else 1.6      # 30% extreme episodes
    ang = rng.uniform(0, 2 * np.pi)
    base = np.array([np.cos(ang), np.sin(ang), 0.0]) * rng.uniform(0, 4) * scale     # steady wind (m/s)
    gd = rng.normal(size=3); gd[2] *= 0.3; gd /= np.linalg.norm(gd)
    return dict(seed=seed, scale=scale, base=base, base_speed=float(np.linalg.norm(base)), gust_dir=gd,
                gust_amp=float(rng.uniform(0, 7) * scale), t0=float(rng.uniform(6, 30)),
                dur=float(rng.uniform(1, 3)), turb=float(rng.uniform(0, 0.8) * scale), path=path)

def wind_profile(sc):
    """(n,3) NED wind vector per step: steady + 1-cos gust + OU turbulence. Same for every controller."""
    n = int(round(T / DT)); t = np.arange(n) * DT; rng = np.random.default_rng(sc["seed"] + 777)
    on = (t >= sc["t0"]) & (t < sc["t0"] + sc["dur"])
    shape = 0.5 * (1 - np.cos(2 * np.pi * (t - sc["t0"]) / sc["dur"])) * on
    turb = np.zeros((n, 3))
    for k in range(1, n): turb[k] = 0.9 * turb[k - 1] + np.sqrt(1 - 0.81) * sc["turb"] * rng.normal(size=3)
    return sc["base"] + sc["gust_amp"] * shape[:, None] * sc["gust_dir"] + turb


# ------------------------------------------------------------------ AirSim episode
def full_reset(client, airsim, do_reset=True):
    if do_reset:
        client.reset(); time.sleep(8.0)
    for _ in range(40):                                   # after a reset PX4 can take a while to answer again
        try:
            client.enableApiControl(True)
            if client.isApiControlEnabled(): break
        except Exception: pass
        time.sleep(3.0)
    else:
        raise RuntimeError("vehicle not responding after reset - restart PX4, press Play in Unreal again, rerun")
    for _ in range(60):
        try:
            if client.getGpsData().gnss.fix_type >= 3: break
        except Exception: pass
        time.sleep(1.0)
    time.sleep(3.0)
    for _ in range(15):
        if client.armDisarm(True): break
        time.sleep(2.0)
    else:
        raise RuntimeError("PX4 would not arm")
    client.takeoffAsync(timeout_sec=20).join()
    client.moveToPositionAsync(0.0, 0.0, HOVER_Z, 1.0, timeout_sec=20).join()
    client.hoverAsync().join(); time.sleep(3.0)

def return_home(client, airsim):
    client.simSetWind(airsim.Vector3r(0, 0, 0))
    client.moveToPositionAsync(0.0, 0.0, HOVER_Z, 2.0, timeout_sec=15).join()
    client.hoverAsync().join(); time.sleep(2.5)

def run_episode(client, airsim, ctrl, sc):
    n = int(round(T / DT)); W = wind_profile(sc)
    Rt, Vt = refs(sc["path"], DT, n); Rt, Vt = Rt[:, 0], Vt[:, 0]
    s0 = client.getMultirotorState().kinematics_estimated.position
    x0, y0, z0 = s0.x_val, s0.y_val, s0.z_val
    ctrl.reset(); vsp = np.zeros(3); log = []; aborted, reason = False, ""
    c0 = client.simGetCollisionInfo().time_stamp          # ignore old/ground-contact collisions from before this episode
    t_start = time.time()
    for k in range(n):
        wait = t_start + k * DT - time.time()
        if wait > 0: time.sleep(wait)
        client.simSetWind(airsim.Vector3r(*[float(w) for w in W[k]]))
        s = client.getMultirotorState().kinematics_estimated
        p = np.array([s.position.x_val - x0, s.position.y_val - y0, -(s.position.z_val - z0)])
        v = np.array([s.linear_velocity.x_val, s.linear_velocity.y_val, -s.linear_velocity.z_val])
        if np.abs(p).max() > FENCE or p[2] < -Z_GUARD: aborted, reason = True, "geofence/altitude"; break
        if k % 5 == 0:
            ci = client.simGetCollisionInfo()
            if ci.has_collided and ci.time_stamp != c0 and k > 20: aborted, reason = True, "collision with " + str(ci.object_name); break
        r, vr = Rt[k], Vt[k]; u = ctrl(r, vr, p, v, DT)
        vsp = np.clip(vsp + u * DT, -V_MAX, V_MAX)
        client.moveByVelocityAsync(float(vsp[0]), float(vsp[1]), float(-vsp[2]), 0.2)
        log.append([time.time() - t_start, *p, *v, *r, *u, *W[k]])
    client.simSetWind(airsim.Vector3r(0, 0, 0))
    if aborted: print(f"  abort at step {k}: {reason}", flush=True)
    return np.array(log).reshape(-1, 16), aborted, reason

def metrics(arr, sc, aborted):
    if len(arr) < 20:                                       # episode aborted almost immediately
        out = {k: float('nan') for k in ["RMSE (m)", "Max error (m)", "Max dev in gust (m)", "Recovery time (s)", "Final error (m)",
                                         "Steady jitter std (m)", "Control rate RMS", "Control effort"]}
        out["Total cost"] = 100.0; out["failed"] = True; return out
    t, P, R, U = arr[:, 0], arr[:, 1:4], arr[:, 7:10], arr[:, 10:13]
    e = np.linalg.norm(P - R, axis=1); ge = sc["t0"] + sc["dur"]
    gw = (t >= sc["t0"]) & (t < ge + 3); m = (t >= ge) & (t < ge + 8) & (e > 0.5)
    fin = t >= T - 2.5; dU = np.diff(U, axis=0)
    cost = np.mean((P - R) ** 2) + 0.005 * np.mean(U ** 2) + 0.02 * np.mean(dU ** 2)
    out = {"RMSE (m)": float(np.sqrt(np.mean(e ** 2))), "Max error (m)": float(e.max()),
           "Max dev in gust (m)": float((e * gw).max()),
           "Recovery time (s)": float(max(t[m].max() - ge, 0)) if m.any() else 0.0,
           "Final error (m)": float(e[-1]),
           "Steady jitter std (m)": float(P[fin].std(0).mean()) if fin.sum() > 3 else np.nan,
           "Control rate RMS": float(np.sqrt(np.mean(dU ** 2))), "Control effort": float(np.mean(U ** 2)),
           "Total cost": float(cost)}
    if aborted: out["Total cost"] = 100.0
    out["failed"] = bool(aborted or out["RMSE (m)"] > 1.0 or out["Recovery time (s)"] > 3.0)
    return out


class RigDead(Exception): pass

class Rig:
    """Owns the AirSim connection; resets every K episodes; retries on failures."""
    def __init__(s, reset_every=0):                      # 0 = never reset (only after an abort/crash)
        import airsim; s.airsim = airsim; s.K = reset_every; s.n = 0; s.need_reset = True; s.first = True; s.fails = 0; s.connect()
    def connect(s):
        s.client = s.airsim.MultirotorClient(); s.client.confirmConnection()
    def run(s, ctrl, sc):
        for attempt in range(3):
            try:
                if s.need_reset or (s.K and s.n >= s.K):
                    full_reset(s.client, s.airsim, do_reset=not s.first); s.first = False; s.n = 0; s.need_reset = False   # first start: NO reset
                else: return_home(s.client, s.airsim)
                out = run_episode(s.client, s.airsim, ctrl, sc); s.n += 1
                if out[1]: s.need_reset = True
                s.fails = 0; return out
            except Exception as ex:
                s.fails += 1; s.need_reset = True
                print(f"  !! episode error ({ex}); retry {attempt+1}/3", flush=True)
                if s.fails >= 8: raise RigDead("8 consecutive failures - restart PX4 + Unreal and rerun (it resumes).")
                time.sleep(min(60, 10 * s.fails))
                try: s.connect()
                except Exception: pass
        return None
