"""Autonomous overnight AirSim benchmark: random wind episodes, PID vs LQG vs SPNN (paired, same wind).
Resumable: rerun the same command and it continues.
  python run_overnight.py --hours 8 --episodes 1000 [--use-tuned]"""
import argparse, csv, os, time
import numpy as np
from airsim_lib import *

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=1000); ap.add_argument("--hours", type=float, default=8.0)
ap.add_argument("--reset-every", type=int, default=0); ap.add_argument("--use-tuned", action="store_true"); ap.add_argument("--random-paths", action="store_true")
a = ap.parse_args()

ctrls = {k: make_controller(k, tuned=a.use_tuned) for k in KINDS}
os.makedirs("airsim_runs", exist_ok=True)
CSV = "airsim_results.csv"; done = set()
if os.path.exists(CSV):
    import pandas as pd; d = pd.read_csv(CSV); done = set(zip(d.episode, d.controller))
rig = Rig(a.reset_every); deadline = time.time() + a.hours * 3600; t0 = time.time()

for ep in range(a.episodes):
    sc = sample_scenario(ep, random_path=a.random_paths)
    order = KINDS[ep % len(KINDS):] + KINDS[:ep % len(KINDS)]            # rotate order so drift doesn't favour anyone
    for name in order:
        if (ep, name) in done: continue
        if time.time() > deadline: print("Time budget reached. Done."); raise SystemExit
        out = rig.run(ctrls[name], sc)
        if out is None: print(f"ep {ep} {name}: skipped"); continue
        arr, aborted, reason = out; m = metrics(arr, sc, aborted)
        np.savez_compressed(f"airsim_runs/ep{ep:05d}_{name}.npz", log=arr, t0=sc["t0"], dur=sc["dur"])
        row = {"episode": ep, "controller": name, "scale": sc["scale"], "base_wind": sc["base_speed"],
               "gust_amp": sc["gust_amp"], "gust_dur": sc["dur"], "turbulence": sc["turb"],
               "aborted": aborted, "reason": reason, **m}
        new = not os.path.exists(CSV)
        with open(CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); 
            if new: w.writeheader()
            w.writerow(row)
        print(f"[{(time.time()-t0)/3600:.2f} h] ep {ep:4d} {name:4s} wind {sc['base_speed']:.1f}+gust {sc['gust_amp']:.1f} "
              f"-> RMSE {m['RMSE (m)']:.3f} cost {m['Total cost']:.3f} {'ABORT '+reason if aborted else ''}", flush=True)
