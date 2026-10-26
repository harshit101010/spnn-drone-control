"""Shows what the SPNN actually does: its output (what it subtracts) against the real disturbance acting on the drone.
Run:  python spnn_vs_disturbance.py            (wind gust + mass change + drag + noise)
      python spnn_vs_disturbance.py gust       (a clean wind gust only: the easiest case to read)
->  spnn_vs_disturbance_<scenario>.png  (+ printed 'how much is cancelled' table)"""
import sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sim import make_scenario, gust_acc, GFF, TAU_A, DT, N, U_MAX, simulate
from path import refs
from build import build

def run(ctrl, sc, seed=5):
    """Same plant as sim.simulate, but also logs the true disturbance d (acceleration the nominal model does not know about)
    and the SPNN output u_nn.  Disturbance d = actual acceleration - commanded acceleration."""
    B = sc["m"].shape[0]; Rn, Vn = refs(sc["path"], DT, N); Rt, Vt = torch.from_numpy(Rn), torch.from_numpy(Vn)
    gn = torch.Generator().manual_seed(seed + 1)
    p, v, ua = torch.zeros(B, 3), torch.zeros(B, 3), torch.zeros(B, 3); ctrl.reset(B); P, D, UN = [], [], []
    for k in range(N):
        t = k * DT
        pm = p + sc["sp"] * torch.randn(B, 3, generator=gn); vm = v + sc["sv"] * torch.randn(B, 3, generator=gn)
        u = ctrl(Rt[k], Vt[k], pm, vm, DT).clamp(-U_MAX, U_MAX)
        ua = ua + (u - ua) * (DT / TAU_A)
        a = (ua + GFF) / sc["m"] - GFF - sc["cd"] * v.abs() * v / sc["m"] + gust_acc(t, sc) / sc["m"]
        D.append(a - ua); UN.append(getattr(ctrl, "u_nn", torch.zeros(B, 3)))
        v = v + a * DT; p = p + v * DT; P.append(p)
    return torch.stack(P), Rt, torch.stack(D), torch.stack(UN)

KIND = sys.argv[1] if len(sys.argv) > 1 else "combined"          # combined | gust | mismatch | noise
ctrls = build(); t = np.arange(N) * DT
with torch.no_grad():
    # ---- one example flight (wind gust + mass change + drag + noise) ----
    sc1 = make_scenario(1, KIND, seed=7)
    P0, R, _, _ = run(ctrls["PID"], sc1); P1, _, D1, UN1 = run(ctrls["PID+SPNN"], sc1)
    e0 = (P0 - R)[:, 0].norm(dim=1).numpy(); e1 = (P1 - R)[:, 0].norm(dim=1).numpy()
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for i, n in enumerate("XYZ"):
        ax[i].plot(t, D1[:, 0, i], "k", lw=1.2, label="real disturbance on the drone")
        ax[i].plot(t, UN1[:, 0, i], lw=1.2, label="SPNN output (what it cancels)")
        ax[i].axvspan(sc1["t0"].item(), (sc1["t0"] + sc1["dur"]).item(), color="orange", alpha=.2)
        ax[i].set_ylabel(f"{n} accel (m/s²)"); ax[i].grid(alpha=.3)
    ax[0].legend(loc="upper right")
    ax[3].plot(t, e0, label=f"PID (RMSE {np.sqrt((e0**2).mean()):.3f} m)"); ax[3].plot(t, e1, label=f"PID+SPNN (RMSE {np.sqrt((e1**2).mean()):.3f} m)")
    ax[3].axvspan(sc1["t0"].item(), (sc1["t0"] + sc1["dur"]).item(), color="orange", alpha=.2, label="gust")
    ax[3].set_ylabel("tracking error (m)"); ax[3].set_xlabel("time (s)"); ax[3].legend(); ax[3].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"spnn_vs_disturbance_{KIND}.png", dpi=200)

    # ---- statistics over 100 random flights: how much of the disturbance does the SPNN cancel? ----
    sc = make_scenario(100, "train", seed=4321); _, _, D, UN = run(ctrls["PID+SPNN"], sc)
    rms = lambda x: x.pow(2).mean((0, 1)).sqrt()
    d_rms, res_rms = rms(D), rms(D - UN)
    print("\nHow much of the real disturbance does the SPNN cancel? (100 random flights, PID+SPNN)")
    for i, n in enumerate("XYZ"):
        print(f"  axis {n}: disturbance RMS {d_rms[i]:.3f} m/s²  ->  remaining after SPNN {res_rms[i]:.3f} m/s²   ({100*(1-res_rms[i]/d_rms[i]):.0f}% cancelled)")
    cc = [np.corrcoef(D[:, :, i].flatten().numpy(), UN[:, :, i].flatten().numpy())[0, 1] for i in range(3)]
    print("  correlation between disturbance and SPNN output (x, y, z):", np.round(cc, 2))
print(f"\nSaved spnn_vs_disturbance_{KIND}.png")
