"""Tune PID + LQG fairly, load trained SPNN, run Monte Carlo on all scenarios (mission path),
print tables, save CSV + XYZ / XY / 3D / box plots.   Run:  python evaluate.py"""
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sim import *
from controllers import PID, LQG, SPNN, SPNNController

RUNS = 100
SCEN = ["nominal", "gust", "mismatch", "noise", "combined", "unseen"]
tvec = torch.arange(N) * DT


def tune_pid():
    sc = make_scenario(48, "train", seed=12345)
    def f(x):
        kp, ki, kd = np.exp(x)
        with torch.no_grad():
            l = loss_fn(*simulate(PID(kp, ki, kd), sc, seed=1)).item()
        return l if np.isfinite(l) else 1e6
    res = minimize(f, np.log([2.0, 0.4, 2.5]), method="Nelder-Mead", options=dict(maxiter=120))
    kp, ki, kd = np.exp(res.x); print(f"PID tuned: kp={kp:.3f} ki={ki:.3f} kd={kd:.3f}  cost={res.fun:.4f}")
    return PID(kp, ki, kd)


def tune_lqg():
    sc = make_scenario(48, "train", seed=12345); best = (1e9, None)
    for r in [0.02, 0.1, 0.5, 2.0]:
        for qe in [3.0, 10.0, 30.0]:
            c = LQG(qe=qe, r=r)
            with torch.no_grad():
                l = loss_fn(*simulate(c, sc, seed=1)).item()
            if l < best[0]: best = (l, (qe, r))
    print(f"LQG tuned: qe={best[1][0]} r={best[1][1]}  cost={best[0]:.4f}")
    return LQG(qe=best[1][0], r=best[1][1])


def metrics(P, U, R, sc):
    e = (P - R).norm(dim=2)                                   # (N,B) tracking error
    t = tvec[:, None]
    ge = sc["t0"].T + sc["dur"].T                             # gust end (1,B)
    gw = (t >= sc["t0"].T) & (t < ge + 3)
    maxdev = (e * gw).max(0).values
    rec = ((((t >= ge) & (t < ge + 8) & (e > 0.5)).float() * t).max(0).values - ge[0]).clamp(min=0)
    fin = tvec >= T - 2.5
    dU = U[1:] - U[:-1]
    return {
        "RMSE (m)": (e ** 2).mean(0).sqrt(),
        "Max error (m)": e.max(0).values,
        "Max dev in gust (m)": maxdev,
        "Recovery time (s)": rec,
        "Final error (m)": e[-1],
        "Steady jitter std (m)": P[fin].std(0).mean(1),
        "Control rate RMS": dU.pow(2).mean((0, 2)).sqrt(),
        "Control effort": U.pow(2).mean((0, 2)),
        "Total cost": (P - R).pow(2).mean((0, 2)) + 0.005 * U.pow(2).mean((0, 2)) + 0.02 * dU.pow(2).mean((0, 2)),
    }


if __name__ == "__main__":
    from build import build
    ctrls = build()
    rows, raw = [], {}
    with torch.no_grad():
        for s in SCEN:
            sc = make_scenario(RUNS, s, seed=10_000)           # unseen seeds; same scenarios for all controllers
            for name, c in ctrls.items():
                P, U, R = simulate(c, sc, seed=5)
                m = metrics(P, U, R, sc); raw[(s, name)] = m
                row = {"scenario": s, "controller": name}
                for k, v in m.items(): row[k] = f"{v.mean():.3f} ± {v.std():.3f}"
                rows.append(row)
    df = pd.DataFrame(rows); df.to_csv("results_table.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    for s in SCEN:
        print(f"\n=== {s} ({RUNS} runs, mean ± std) ==="); print(df[df.scenario == s].drop(columns="scenario").to_string(index=False))

    # ---- XYZ stability plot: canonical mission path, one gust + mismatch + noise run ----
    sc1 = make_scenario(1, "combined", seed=7)
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True); traj = {}
    with torch.no_grad():
        for name, c in ctrls.items():
            P, U, R = simulate(c, sc1, seed=5); traj[name] = P[:, 0].numpy()
            for i in range(3): ax[i].plot(tvec, traj[name][:, i], label=name)
    ref = R[:, 0].numpy()
    for i, a in enumerate("XYZ"):
        ax[i].plot(tvec, ref[:, i], "k--", label="Reference"); ax[i].set_ylabel(f"{a} (m)"); ax[i].grid(alpha=.3)
        ax[i].axvspan(sc1["t0"].item(), (sc1["t0"] + sc1["dur"]).item(), color="orange", alpha=.2)
    ax[0].legend(); ax[2].set_xlabel("Time (s)  (orange = wind gust)"); plt.tight_layout(); plt.savefig("xyz_stability.png", dpi=200)

    plt.figure(figsize=(7, 7))                                  # top-down view of the path
    for name, p in traj.items(): plt.plot(p[:, 0], p[:, 1], label=name)
    plt.plot(ref[:, 0], ref[:, 1], "k--", label="Reference"); plt.axis("equal"); plt.grid(alpha=.3); plt.legend()
    plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.title("Top-down path"); plt.savefig("path_xy.png", dpi=200)

    fig = plt.figure(figsize=(7, 6)); a3 = fig.add_subplot(projection="3d")
    for name, p in traj.items(): a3.plot(p[:, 0], p[:, 1], p[:, 2], label=name)
    a3.plot(ref[:, 0], ref[:, 1], ref[:, 2], "k--", label="Reference"); a3.legend()
    a3.set_xlabel("X"); a3.set_ylabel("Y"); a3.set_zlabel("Z"); plt.savefig("traj3d.png", dpi=200)

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for j, s in enumerate(["gust", "combined", "unseen"]):
        for i, k in enumerate(["Max dev in gust (m)", "Steady jitter std (m)"]):
            ax[i, j].boxplot([raw[(s, n)][k].numpy() for n in ctrls], labels=list(ctrls)); ax[i, j].set_title(f"{s}: {k}")
    plt.tight_layout(); plt.savefig("boxplots.png", dpi=200)
    print("\nSaved: results_table.csv, xyz_stability.png, path_xy.png, traj3d.png, boxplots.png")
