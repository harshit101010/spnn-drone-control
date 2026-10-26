"""Find the adaptation rate Gamma for each base controller (the paper's key hyper-parameter, its Algorithm 2).
Too small = no benefit, too large = unstable/chattering. Uses tuning scenarios that evaluation never sees.
Run:  python tune_gamma.py   ->  gamma.json"""
import json, numpy as np, torch
from sim import make_scenario, simulate, loss_fn
from controllers import PID, LQG, Compensated
from build import params

P = params(); sc = make_scenario(48, "train", seed=24680)
GRID = [3, 10, 20, 40, 80, 150, 300, 600]
bases = {"PID": lambda: PID(*P["PID"]), "LQG": lambda: LQG(qe=P["LQG"][0], r=P["LQG"][1])}
best = {}
for name, mk in bases.items():
    with torch.no_grad():
        c0 = loss_fn(*simulate(mk(), sc, seed=1)).item()
        print(f"\n{name} alone: cost {c0:.5f}")
        res = {}
        for g in GRID:
            c = loss_fn(*simulate(Compensated(mk(), g), sc, seed=1)).item()
            res[g] = c if np.isfinite(c) else np.inf
            print(f"  {name}+SPNN  Gamma={g:<4}  cost {res[g]:.5f}  ({100*(res[g]/c0-1):+.1f}% vs base)")
    best[name] = float(min(res, key=res.get)); print(f"  -> chosen Gamma for {name}: {best[name]}")
json.dump(best, open("gamma.json", "w")); print("\nSaved gamma.json", best)
