"""Shows the network LEARNING online: tracking error with/without the compensator and the weight norm over time.
Run:  python adaptation_demo.py   ->  adaptation_demo.png"""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sim import make_scenario, simulate, DT, N
from build import build

ctrls = build(); sc = make_scenario(1, "combined", seed=7)
class Log:
    def __init__(s, c): s.c, s.w = c, []
    def reset(s, B): s.c.reset(B); s.w = []
    def __call__(s, *a):
        u = s.c(*a)
        if hasattr(s.c, "W"): s.w.append(s.c.W.norm().item())
        return u
t = np.arange(N) * DT; fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
with torch.no_grad():
    for name, c in ctrls.items():
        L = Log(c); P, U, R = simulate(L, sc, seed=5); e = (P - R)[:, 0].norm(dim=1).numpy()
        ax[0].plot(t, e, label=f"{name} (RMSE {np.sqrt((e**2).mean()):.3f} m)")
        if L.w: ax[1].plot(t, L.w, label=f"{name}: |W|")
ax[0].axvspan(sc["t0"].item(), (sc["t0"] + sc["dur"]).item(), color="orange", alpha=.2, label="gust")
ax[0].set_ylabel("tracking error (m)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_ylabel("weight norm |W| (learning)"); ax[1].set_xlabel("time (s)"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("adaptation_demo.png", dpi=200); print("Saved adaptation_demo.png")
