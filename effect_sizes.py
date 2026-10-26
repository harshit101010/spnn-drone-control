"""Effect sizes, spreads and failure-rate confidence intervals from stress_runs.csv (no p-values needed).
Run:  python effect_sizes.py     ->  effect_sizes.csv, failure_rates.csv, effect_sizes.png"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = pd.read_csv("stress_runs.csv")
PAIRS = [("PID+SPNN", "PID"), ("PID+SPNN", "LQG"), ("LQG+SPNN", "LQG")]
METRICS = ["RMSE (m)", "Max dev in gust (m)", "Recovery time (s)", "Steady jitter std (m)", "Control rate RMS", "Total cost"]
rng = np.random.default_rng(0)

def boot_ci(d, n=1000):
    idx = rng.integers(0, len(d), (n, len(d))); return np.percentile(np.median(d[idx], axis=1), [2.5, 97.5])

rows = []
for sc in D.scenario.unique():
    for subj, rival in PAIRS:
        A = D[(D.scenario == sc) & (D.controller == subj)].sort_values("run"); B = D[(D.scenario == sc) & (D.controller == rival)].sort_values("run")
        for m in METRICS:
            a, b = A[m].values, B[m].values; d = a - b                      # negative = subject is better (lower)
            lo, hi = boot_ci(d)
            rows.append({"scenario": sc, "subject": subj, "vs": rival, "metric": m, "median diff": np.median(d),
                         "diff 25%": np.percentile(d, 25), "diff 75%": np.percentile(d, 75), "median 95% CI low": lo, "median 95% CI high": hi,
                         "subject better in (%)": 100 * np.mean(d < 0), "mean change (%)": 100 * (a.mean() - b.mean()) / b.mean()})
E = pd.DataFrame(rows).round(4); E.to_csv("effect_sizes.csv", index=False, encoding="utf-8-sig")

def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return 100 * (c - h), 100 * (c + h)
fr = []
for (sc, ctl), g in D.groupby(["scenario", "controller"]):
    k, n = int(g.failed.sum()), len(g); lo, hi = wilson(k, n)
    fr.append({"scenario": sc, "controller": ctl, "runs": n, "failures": k, "failure rate (%)": 100 * k / n, "95% CI low": lo, "95% CI high": hi})
F = pd.DataFrame(fr).round(2); F.to_csv("failure_rates.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
print("\n=== Tracking error (RMSE): paired difference subject - rival, in metres (negative = subject better) ===")
print(E[E.metric == "RMSE (m)"][["scenario", "subject", "vs", "median diff", "diff 25%", "diff 75%", "median 95% CI low", "median 95% CI high", "subject better in (%)", "mean change (%)"]].to_string(index=False))
print("\n=== Failure rates with 95% confidence intervals ===\n", F.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.5)); sc_list = list(D.scenario.unique()); w = 0.25
for i, (subj, rival) in enumerate(PAIRS):
    s = E[(E.metric == "RMSE (m)") & (E.subject == subj) & (E.vs == rival)].set_index("scenario").loc[sc_list]
    ax.bar(np.arange(len(sc_list)) + i * w, s["median diff"], w, yerr=[s["median diff"] - s["median 95% CI low"], s["median 95% CI high"] - s["median diff"]], capsize=3, label=f"{subj} vs {rival}")
ax.axhline(0, color="k", lw=.8); ax.set_xticks(np.arange(len(sc_list)) + w); ax.set_xticklabels(sc_list)
ax.set_ylabel("median RMSE difference (m)\n(negative = compensator better)"); ax.legend(); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("effect_sizes.png", dpi=200); print("\nSaved effect_sizes.csv, failure_rates.csv, effect_sizes.png")
