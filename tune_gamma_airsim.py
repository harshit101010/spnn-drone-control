"""Paper's Algorithm 2 idea, done in AirSim: find the adaptation rate Gamma that flies best (too large = unstable).
Tests each Gamma on the same 4 wind episodes. Resumable.   python tune_gamma_airsim.py   -> gamma_airsim.json"""
import argparse, csv, json, os
import numpy as np
from airsim_lib import *

ap = argparse.ArgumentParser()
ap.add_argument("--gammas", type=float, nargs="+", default=[10, 20, 40, 80, 150, 300])
ap.add_argument("--scenarios", type=int, default=4); a = ap.parse_args()
GAMMAS = a.gammas
SC = [sample_scenario(3_000_000 + i, scale=1.0 if i < 2 else 1.6, random_path=(i % 2 == 1)) for i in range(a.scenarios)]
LOG = "gamma_airsim_log.csv"; done = {}
if os.path.exists(LOG):
    for row in csv.DictReader(open(LOG)): done[(row["base"], float(row["gamma"]), int(row["sc"]))] = float(row["cost"])
rig = Rig(); best = {}
for base in BASE_KINDS:
    mean = {}
    for g in [0] + GAMMAS:                                   # 0 = base controller alone (reference)
        costs = []
        for i, sc in enumerate(SC):
            if (base, g, i) in done: costs.append(done[(base, g, i)]); continue
            c = make_controller(base if g == 0 else base + "+SPNN", tuned=True)
            if g: c.gamma = float(g)
            out = rig.run(c, sc); cost = 100.0 if out is None else metrics(out[0], sc, out[1])["Total cost"]
            costs.append(cost); done[(base, g, i)] = cost
            new = not os.path.exists(LOG)
            with open(LOG, "a", newline="") as f:
                w = csv.writer(f)
                if new: w.writerow(["base", "gamma", "sc", "cost"])
                w.writerow([base, g, i, cost])
        mean[g] = float(np.mean(costs)); print(f"{base} Gamma={g:<4} mean cost {mean[g]:.4f}", flush=True)
    best[base] = float(min([g for g in GAMMAS], key=lambda g: mean[g])); print(f"-> best Gamma for {base}: {best[base]}")
json.dump(best, open("gamma_airsim.json", "w")); print("Saved gamma_airsim.json", best)
