"""Measures how fast AirSim's flight controller follows a velocity command (step response),
so the Python simulator can match it.  Run with Unreal Play (SimpleFlight):   python identify_lag.py
Writes plant_lag.json  (used automatically by sim.py)."""
import json, time
import numpy as np
from scipy.optimize import curve_fit
import airsim

c = airsim.MultirotorClient(); c.confirmConnection(); c.enableApiControl(True); c.armDisarm(True)
c.takeoffAsync(timeout_sec=20).join(); c.moveToPositionAsync(0, 0, -3, 1.5, timeout_sec=20).join()
c.hoverAsync().join(); time.sleep(3)

def run_step(axis, V, hold):
    ts, vs = [], []; t0 = time.time(); t_step = 1.0
    while time.time() - t0 < t_step + hold:
        t = time.time() - t0
        cmd = [0.0, 0.0, 0.0]
        if t >= t_step: cmd[axis] = V
        c.moveByVelocityAsync(cmd[0], cmd[1], -cmd[2] if axis == 2 else cmd[2], 0.2)   # axis 2 = climb (up positive)
        v = c.getMultirotorState().kinematics_estimated.linear_velocity
        vs.append([v.x_val, v.y_val, -v.z_val][axis]); ts.append(t); time.sleep(0.05)
    c.hoverAsync().join()
    c.moveToPositionAsync(0, 0, -3, 2.0, timeout_sec=20).join(); time.sleep(2.0)
    t = np.array(ts) - t_step; v = np.array(vs); m = t >= 0
    return t[m], v[m]

def fit(t, v, V):
    f = lambda x, tau, td, G: np.where(x > td, G * (1 - np.exp(-(x - td) / tau)), 0.0)
    (tau, td, G), _ = curve_fit(f, t, v, p0=(0.5, 0.1, V), bounds=([0.02, 0.0, 0.2 * V], [6.0, 1.5, 2 * V]))
    vf = v[t > t[-1] - 1.0].mean()
    rt = lambda frac: float(t[np.argmax(v >= frac * vf)]) if (v >= frac * vf).any() else float("nan")
    return tau, td, G, vf, rt(0.1), rt(0.63), rt(0.9)

res = {}
for name, axis, V, hold in [("x (forward)", 0, 2.0, 4.0), ("y (sideways)", 1, 2.0, 4.0), ("z (climb)", 2, 1.5, 2.5)]:
    t, v = run_step(axis, V, hold)
    tau, td, G, vf, t10, t63, t90 = fit(t, v, V)
    res[name] = (tau, td)
    print(f"{name:14s} lag tau = {tau:.2f} s | dead time = {td:.2f} s | final speed {vf:.2f} of {V} m/s | reaches 10%/63%/90% at {t10:.2f}/{t63:.2f}/{t90:.2f} s", flush=True)

tau_xy = float(np.mean([res["x (forward)"][0] + res["x (forward)"][1], res["y (sideways)"][0] + res["y (sideways)"][1]]))
json.dump({"tau": tau_xy, "details": {k: list(v) for k, v in res.items()}}, open("plant_lag.json", "w"))
print(f"\nSaved plant_lag.json  (effective lag used by the simulator: {tau_xy:.2f} s)")
print("Now copy everything printed above and send it to me.")
