"""Statistics + unsupervised analysis of airsim_results.csv.   python analyze_airsim.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

D = pd.read_csv("airsim_results.csv")
D["wind_total"] = D.base_wind + D.gust_amp
FE = ["RMSE (m)", "Max error (m)", "Max dev in gust (m)", "Recovery time (s)", "Steady jitter std (m)", "Control rate RMS", "Control effort"]
# only episodes where all three controllers finished (paired)
ok = D.groupby("episode").controller.nunique(); D = D[D.episode.isin(ok[ok == len(KINDS)].index)]
print(f"{D.episode.nunique()} complete paired episodes")

S = D.groupby("controller").agg(runs=("episode", "count"), fail_pct=("failed", lambda x: 100 * x.mean()),
        aborts=("aborted", "sum"), **{f"{k}_mean": (k, "mean") for k in ["RMSE (m)", "Total cost", "Max dev in gust (m)", "Steady jitter std (m)", "Control rate RMS"]},
        rmse_p95=("RMSE (m)", lambda x: x.quantile(.95))).round(3)
S.to_csv("airsim_summary.csv"); print("\n", S.to_string())

rows = []
for subj, rival in [("PID+SPNN", "PID"), ("PID+SPNN", "LQG"), ("LQG+SPNN", "LQG")]:
    for k in ["RMSE (m)", "Total cost", "Max dev in gust (m)", "Steady jitter std (m)", "Control rate RMS"]:
        a = D[D.controller == subj].sort_values("episode")[k].values; b = D[D.controller == rival].sort_values("episode")[k].values
        rows.append({"subject": subj, "vs": rival, "metric": k, "subject better (%)": 100 * np.mean(a < b), "median diff": np.median(a - b), "Wilcoxon p": wilcoxon(a, b).pvalue})
W = pd.DataFrame(rows).round(4); W.to_csv("airsim_winrates.csv", index=False); print("\n", W.to_string(index=False))

D["wind_bin"] = pd.cut(D.wind_total, [0, 3, 6, 9, 12, 20])
B = D.groupby(["wind_bin", "controller"], observed=True).agg(RMSE=("RMSE (m)", "mean"), fail=("failed", "mean")).round(3).unstack()
print("\nPerformance vs total wind speed (m/s):\n", B.to_string()); B.to_csv("airsim_by_wind.csv")

# ---- unsupervised: clusters + anomalies ----
X = StandardScaler().fit_transform(D[FE].fillna(D[FE].median())); Z = PCA(2).fit_transform(X)
bk, bs = 2, -1
for k in range(2, 7):
    s = silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X))
    if s > bs: bk, bs = k, s
D["cluster"] = KMeans(bk, n_init=10, random_state=0).fit_predict(X)
with open("airsim_cluster_report.txt", "w") as f:
    f.write(f"KMeans k={bk} silhouette={bs:.3f}\n\nController share per cluster (%):\n")
    f.write((pd.crosstab(D.controller, D.cluster, normalize="index") * 100).round(1).to_string())
    f.write("\n\nCluster means:\n" + D.groupby("cluster")[FE + ["base_wind", "gust_amp", "turbulence", "failed"]].mean().round(3).to_string())
    sp = D[D.controller == "PID+SPNN"].reset_index(drop=True)
    an = IsolationForest(contamination=0.03, random_state=0).fit_predict(StandardScaler().fit_transform(sp[FE].fillna(0))) == -1
    sp["anomaly"] = an
    f.write("\n\nSPNN anomalous runs vs normal (mean conditions):\n" + sp.groupby("anomaly")[["base_wind", "gust_amp", "turbulence", "RMSE (m)"]].mean().round(3).to_string())
print("\n" + open("airsim_cluster_report.txt").read())

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].scatter(Z[:, 0], Z[:, 1], c=D.cluster, s=6, cmap="tab10"); ax[0].set_title("Unsupervised clusters (PCA)")
for c, col in zip(KINDS, ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    d = D[D.controller == c]; ax[1].scatter(d.wind_total, d["RMSE (m)"], s=8, c=col, label=c, alpha=.5)
ax[1].set_xlabel("steady wind + gust amplitude (m/s)"); ax[1].set_ylabel("RMSE (m)"); ax[1].legend(); ax[1].set_title("Robustness vs wind")
ax[2].boxplot([D[D.controller == c]["Steady jitter std (m)"] for c in KINDS], labels=KINDS); ax[2].set_title("Steady jitter (lower = more stable)")
plt.tight_layout(); plt.savefig("airsim_overview.png", dpi=200)

# XYZ plot for the windiest paired episode
ep = int(D.loc[D.wind_total.idxmax(), "episode"]); fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for c in KINDS:
    z = np.load(f"airsim_runs/ep{ep:05d}_{c}.npz"); L = z["log"]
    for i in range(3): ax[i].plot(L[:, 0], L[:, 1 + i], label=c)
    for i in range(3): ax[i].axvspan(float(z["t0"]), float(z["t0"] + z["dur"]), color="orange", alpha=.05)
for i, n in enumerate("XYZ"): ax[i].plot(L[:, 0], L[:, 7 + i], "k--", label="Ref"); ax[i].set_ylabel(f"{n} (m)"); ax[i].grid(alpha=.3)
ax[0].legend(); ax[2].set_xlabel(f"Time (s) - episode {ep} (orange = gust)"); plt.tight_layout(); plt.savefig("airsim_xyz.png", dpi=200)
plt.figure(figsize=(7, 7))
for c in KINDS:
    L = np.load(f"airsim_runs/ep{ep:05d}_{c}.npz")["log"]; plt.plot(L[:, 1], L[:, 2], label=c)
plt.plot(L[:, 7], L[:, 8], "k--", label="Reference"); plt.axis("equal"); plt.grid(alpha=.3); plt.legend()
plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.title(f"Top-down path, episode {ep}"); plt.savefig("airsim_path_xy.png", dpi=200)
print("Saved airsim_summary.csv, airsim_winrates.csv, airsim_by_wind.csv, airsim_cluster_report.txt, airsim_overview.png, airsim_xyz.png")
