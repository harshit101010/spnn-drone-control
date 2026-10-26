"""Large-scale Monte Carlo stress test + UNSUPERVISED analysis of SPNN behaviour.
Run:  python stress_test.py          (needs trained_spnn.pt from train_spnn.py)
Outputs: stress_runs.csv, stress_summary.csv, stress_winrates.csv, cluster_report.txt,
         clusters_pca.png, outlier_report.txt"""
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sim import make_scenario, simulate
from evaluate import tune_pid, tune_lqg, metrics
from controllers import SPNN, SPNNController

TOTAL, BATCH = 3000, 500                 # runs per scenario kind (increase TOTAL to 10000 if you like)
KINDS = ["train", "unseen", "extreme"]              # random mix, 1.5x harder with wider paths, and 2x harder (extreme)
FAIL_RMSE, FAIL_REC = 1.0, 3.0           # a run "fails" if RMSE > 1 m or recovery time > 3 s
FEATS = ["RMSE (m)", "Max error (m)", "Max dev in gust (m)", "Recovery time (s)",
         "Steady jitter std (m)", "Control rate RMS", "Control effort"]

from build import build
ctrls = build()

# ---------------- 1. thousands of runs ----------------
frames = []
with torch.no_grad():
    for kind in KINDS:
        for b in range(TOTAL // BATCH):
            sc = make_scenario(BATCH, kind, seed=50_000 + 1000 * KINDS.index(kind) + b)   # seeds never used in training
            par = {"mass": sc["m"][:, 0], "drag": sc["cd"][:, 0], "gust_mag": sc["amp"].norm(dim=1),
                   "noise_pos": sc["sp"][:, 0]}
            for name, c in ctrls.items():
                P, U, R = simulate(c, sc, seed=b)
                m = metrics(P, U, R, sc)
                m["Total cost"] = (P - R).pow(2).mean((0, 2)) + 0.005 * U.pow(2).mean((0, 2)) \
                                  + 0.02 * (U[1:] - U[:-1]).pow(2).mean((0, 2))
                df = pd.DataFrame({k: v.numpy() for k, v in m.items()})
                for k, v in par.items(): df[k] = v.numpy()
                df["controller"], df["scenario"], df["run"] = name, kind, b * BATCH + np.arange(BATCH)
                frames.append(df)
            print(f"{kind}: {(b+1)*BATCH}/{TOTAL} runs done")
D = pd.concat(frames, ignore_index=True)
D["failed"] = (D["RMSE (m)"] > FAIL_RMSE) | (D["Recovery time (s)"] > FAIL_REC)
D.to_csv("stress_runs.csv", index=False, encoding="utf-8-sig")

# ---------------- 2. statistics ----------------
cols = ["RMSE (m)", "Total cost", "Max dev in gust (m)", "Recovery time (s)", "Steady jitter std (m)", "Control rate RMS"]
rows = []
for (s, c), g in D.groupby(["scenario", "controller"]):
    r = {"scenario": s, "controller": c, "runs": len(g), "failure rate (%)": 100 * g["failed"].mean()}
    for k in cols:
        r[k + " mean"], r[k + " median"], r[k + " p95"] = g[k].mean(), g[k].median(), g[k].quantile(.95)
    rows.append(r)
pd.DataFrame(rows).round(4).to_csv("stress_summary.csv", index=False, encoding="utf-8-sig")
print("\n", pd.DataFrame(rows)[["scenario", "controller", "runs", "failure rate (%)",
      "RMSE (m) mean", "RMSE (m) p95", "Total cost mean"]].round(3).to_string(index=False))

wr = []   # paired comparison: same scenario, same noise, SPNN vs each rival
for s in KINDS:
    for subj, rival in [("PID+SPNN", "PID"), ("PID+SPNN", "LQG"), ("LQG+SPNN", "LQG")]:
        for k in ["RMSE (m)", "Total cost", "Max dev in gust (m)", "Steady jitter std (m)", "Control rate RMS"]:
            a = D[(D.scenario == s) & (D.controller == subj)].sort_values("run")[k].values
            b = D[(D.scenario == s) & (D.controller == rival)].sort_values("run")[k].values
            wr.append({"scenario": s, "subject": subj, "vs": rival, "metric": k, "SPNN better in (%)": 100 * np.mean(a < b),
                       "median diff (SPNN-rival)": np.median(a - b), "Wilcoxon p": wilcoxon(a, b).pvalue})
W = pd.DataFrame(wr).round(4); W.to_csv("stress_winrates.csv", index=False, encoding="utf-8-sig")
print("\n", W.to_string(index=False))

# ---------------- 3. UNSUPERVISED: cluster the run behaviours (no labels used) ----------------
X = StandardScaler().fit_transform(D[FEATS].clip(upper=D[FEATS].quantile(.999), axis=1))
Z = PCA(n_components=2).fit_transform(X)
rng = np.random.default_rng(0); sub = rng.choice(len(X), min(4000, len(X)), replace=False)
best_k, best_s = 2, -1
for k in range(2, 7):
    sc_ = silhouette_score(X[sub], KMeans(k, n_init=10, random_state=0).fit_predict(X[sub]))
    if sc_ > best_s: best_k, best_s = k, sc_
D["cluster"] = KMeans(best_k, n_init=10, random_state=0).fit_predict(X)
with open("cluster_report.txt", "w") as f:
    f.write(f"KMeans clusters on behaviour metrics (k={best_k}, silhouette={best_s:.3f})\n\n")
    f.write("Share of each controller's runs in each cluster (%):\n")
    f.write((pd.crosstab(D.controller, D.cluster, normalize="index") * 100).round(1).to_string() + "\n\n")
    f.write("Mean metrics + disturbance conditions per cluster:\n")
    f.write(D.groupby("cluster")[FEATS + ["mass", "drag", "gust_mag", "noise_pos", "failed"]].mean().round(3).to_string() + "\n")
print("\n" + open("cluster_report.txt").read())
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(Z[:, 0], Z[:, 1], c=D.cluster, s=4, cmap="tab10"); ax[0].set_title("Unsupervised clusters (PCA of metrics)")
for c, col in zip(ctrls, ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    m_ = (D.controller == c).values; ax[1].scatter(Z[m_, 0], Z[m_, 1], s=4, c=col, label=c, alpha=.4)
ax[1].legend(); ax[1].set_title("Same plot coloured by controller")
for a in ax: a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); plt.savefig("clusters_pca.png", dpi=200)

# ---------------- 4. UNSUPERVISED: anomaly detection on SPNN runs ----------------
S = D[D.controller == "PID+SPNN"].reset_index(drop=True)
Xs = StandardScaler().fit_transform(S[FEATS])
S["anomaly"] = IsolationForest(contamination=0.03, random_state=0).fit_predict(Xs) == -1
with open("outlier_report.txt", "w") as f:
    f.write("SPNN runs flagged as anomalous by Isolation Forest (3%) vs normal runs: mean conditions\n\n")
    f.write(S.groupby("anomaly")[["mass", "drag", "gust_mag", "noise_pos", "RMSE (m)", "failed"]].mean().round(3).to_string() + "\n\n")
    f.write("Worst 10 SPNN runs by total cost:\n")
    f.write(S.nlargest(10, "Total cost")[["scenario", "run", "mass", "drag", "gust_mag", "noise_pos", "RMSE (m)", "Total cost"]].round(3).to_string() + "\n")
print("\n" + open("outlier_report.txt").read())
print("Saved: stress_runs.csv, stress_summary.csv, stress_winrates.csv, cluster_report.txt, outlier_report.txt, clusters_pca.png")
