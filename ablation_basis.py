"""Ablation: is it the sigma-pi (product) structure that helps, or just any adaptive compensator?
Same base controller, same scenarios, each variant gets its own best Gamma (fair).   Run:  python ablation_basis.py
Optional:  python ablation_basis.py --base LQG"""
import argparse, numpy as np, pandas as pd, torch
from sim import make_scenario, simulate, loss_fn
from controllers import PID, LQG, Compensated
from build import params
from evaluate import metrics

ap = argparse.ArgumentParser(); ap.add_argument("--base", default="PID", choices=["PID", "LQG"]); a = ap.parse_args()
P = params()
mk = (lambda: PID(*P["PID"])) if a.base == "PID" else (lambda: LQG(qe=P["LQG"][0], r=P["LQG"][1]))
VARIANTS = [("no compensator (base only)", None, None, 0.0),
            ("bias only (1 term: adaptive offset)", "bias", None, 0.02),
            ("linear basis (4 terms)", "linear", None, 0.02),
            ("additive polynomial (7 terms, no products)", "additive", None, 0.02),
            ("sigma-pi, 27 terms (paper)", "sigma_pi", None, 0.02),
            ("sigma-pi, no leakage", "sigma_pi", None, 0.0)]
GRID = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
tune_sc = make_scenario(48, "train", seed=24680)                       # tuning scenarios (never used for the results below)
tests = {k: make_scenario(200, k, seed=888_000) for k in ["train", "unseen", "extreme"]}
rows = []
with torch.no_grad():
    for name, basis, _, leak in VARIANTS:
        if basis is None: g_best, make = 0.0, mk
        else:
            costs = {}
            for g in GRID:
                c = loss_fn(*simulate(Compensated(mk(), g, leak=leak, basis=basis), tune_sc, seed=1)).item()
                costs[g] = c if np.isfinite(c) else np.inf
            g_best = min(costs, key=costs.get); make = (lambda b=basis, g=g_best, l=leak: Compensated(mk(), g, leak=l, basis=b))
        for k, sc in tests.items():
            Pp, U, R = simulate(make(), sc, seed=3); m = metrics(Pp, U, R, sc)
            rows.append({"variant": name, "Gamma": g_best, "scenario": k, "cost": m["Total cost"].mean().item(), "RMSE (m)": m["RMSE (m)"].mean().item(),
                         "max dev in gust (m)": m["Max dev in gust (m)"].mean().item(), "control rate": m["Control rate RMS"].mean().item(),
                         "failures (%)": 100 * ((m["RMSE (m)"] > 1) | (m["Recovery time (s)"] > 3)).float().mean().item()})
        print("done:", name, "| best Gamma", g_best, flush=True)
R_ = pd.DataFrame(rows).round(4); R_.to_csv(f"ablation_basis_{a.base}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
for k in tests: print(f"\n=== {a.base} base, scenario: {k} (200 fresh runs) ===\n", R_[R_.scenario == k].drop(columns="scenario").to_string(index=False))
print("\nHow to read it: if the 27-term sigma-pi row is not clearly better than the bias/linear rows, the benefit comes from adaptation in general, not from the product terms.")
