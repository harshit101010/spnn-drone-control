"""Label-free training INSIDE AirSim with Evolution Strategies (no gradients needed).
Same budget/algorithm for every controller so tuning effort is equal:
  python finetune_es.py --controller SPNN --generations 40
  python finetune_es.py --controller PID  --generations 40
  python finetune_es.py --controller LQG  --generations 40
Resumable. Writes tuned_<CTRL>.npz (best on validation episodes) and es_log_<CTRL>.csv."""
import argparse, csv, os, time
import numpy as np
from airsim_lib import *

ap = argparse.ArgumentParser()
ap.add_argument("--controller", required=True, choices=BASE_KINDS); ap.add_argument("--generations", type=int, default=20)
ap.add_argument("--pairs", type=int, default=4); ap.add_argument("--val-every", type=int, default=5)
a = ap.parse_args(); kind = a.controller
sigma, lr = 0.10, 0.05          # PID/LQG parameters are searched in log-space
state_f = f"es_state_{kind}.npz"; rng = np.random.default_rng(42)
theta, g0, best = init_theta(kind), 0, np.inf
if os.path.exists(state_f):
    s = np.load(state_f); theta, g0, best = s["theta"], int(s["gen"]), float(s["best"]); print(f"Resuming at generation {g0}")
rig = Rig(); VAL = [sample_scenario(2_000_000 + i, scale=1.0 if i < 2 else 1.6, random_path=(i % 2 == 1)) for i in range(4)]   # fixed validation winds

def episode_cost(th, sc):
    out = rig.run(make_controller(kind, th), sc)
    return 100.0 if out is None else metrics(out[0], sc, out[1])["Total cost"]

for g in range(g0, a.generations):
    t0 = time.time(); sc = sample_scenario(1_000_000 + g, random_path=True)        # tuning seeds never used in evaluation
    eps = rng.normal(size=(a.pairs, theta.size)); cand = np.vstack([eps, -eps])
    costs = np.array([episode_cost(theta + sigma * c, sc) for c in cand])   # same wind for all candidates
    rank = np.argsort(np.argsort(costs)); util = -(rank / (len(costs) - 1) - 0.5)   # low cost -> high utility
    theta = theta + lr * (util[:, None] * cand).sum(0) / (len(costs) * sigma)
    msg = f"gen {g+1}/{a.generations}  cand cost mean {costs.mean():.3f} min {costs.min():.3f}"
    val = np.nan
    if (g + 1) % a.val_every == 0 or g + 1 == a.generations:
        val = float(np.mean([episode_cost(theta, v) for v in VAL])); msg += f"  VAL {val:.3f}"
        if val < best: best = val; np.savez(f"tuned_{kind}.npz", theta=theta); msg += "  <- new best"
    np.savez(state_f, theta=theta, gen=g + 1, best=best)
    new = not os.path.exists(f"es_log_{kind}.csv")
    with open(f"es_log_{kind}.csv", "a", newline="") as f:
        w = csv.writer(f); 
        if new: w.writerow(["gen", "cand_mean", "cand_min", "val"])
        w.writerow([g + 1, costs.mean(), costs.min(), val])
    print(msg + f"  ({time.time()-t0:.0f}s)", flush=True)
print("Finished. Best validation cost:", best)
