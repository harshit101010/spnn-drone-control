# SPNN compensator vs PID and LQG for drone path tracking

Final-year project. A sigma-pi neural network (SPNN) is used as an **online-adaptive compensator** on top of a PID or LQG
controller, following Jiang, Pourpanah and Hao (2020). The controllers are compared on a 3-axis drone model flying a
mission path (climb, straight line, semicircle, diagonal climb, S-curve, return) under wind gusts, mass and drag changes
and sensor noise. AirSim scripts for a validation flight test are included.

> **Reference:** F. Jiang, F. Pourpanah, Q. Hao, "Design, Implementation, and Evaluation of a Neural-Network-Based
> Quadcopter UAV System," *IEEE Transactions on Industrial Electronics*, vol. 67, no. 3, 2020. DOI 10.1109/TIE.2019.2905808

## Method in short
- Controllers compared: **PID**, **LQG** (LQR with integral action + Kalman filter), **PID+SPNN**, **LQG+SPNN**.
- SPNN compensator: `u = u_base - W^T beta`, `beta` = products of normalised error, velocity and command terms (27 basis functions per axis).
  Weights start at zero and adapt online with the Lyapunov-based law `dW/dt = -Gamma (e^T P b) beta - leak W`. There is no offline training.
- Fairness: PID and LQG are tuned on the same cost as the compensated versions; every controller uses the same limits, noise, wind and metrics.
- Evaluation: 3,000 randomised runs per scenario type (train / unseen / extreme), paired comparisons, effect sizes with bootstrap confidence intervals,
  an ablation of the basis function, and unsupervised analysis (k-means, isolation forest).

## Repository layout
| File | Purpose |
|---|---|
| `path.py` | Mission path (shared by the simulator and AirSim code) |
| `sim.py` | Batched 3-axis simulator (PyTorch): lag, drag, mass change, gusts, noise |
| `controllers.py` | PID, LQG and the SPNN compensator |
| `lqg_design.py` | LQG design including the velocity-loop lag |
| `build.py` | Builds the four controllers from the tuned parameters |
| `tune_gamma.py` | Chooses the adaptation rate Gamma |
| `evaluate.py` | 100-run evaluation and X/Y/Z, path and 3D plots |
| `stress_test.py` | Thousands of runs, win rates, clustering, outliers |
| `effect_sizes.py` | Median differences, bootstrap CIs, failure rates with CIs |
| `ablation_basis.py` | Is it the sigma-pi structure or any adaptive term that helps? |
| `adaptation_demo.py`, `spnn_vs_disturbance.py` | Figures showing the online learning |
| `run_python_pipeline.py` | Runs the Python part end to end |
| `airsim_lib.py`, `run_overnight.py`, `tune_gamma_airsim.py`, `finetune_es.py`, `analyze_airsim.py`, `run_airsim_5h.py`, `run_airsim_pipeline.py` | AirSim validation |
| `identify_lag.py`, `diagnose_flight.py`, `diagnose_state.py` | AirSim helpers (measure the velocity-loop lag, check commands move the drone) |
| `airsim_settings/settings.json` | AirSim settings used (SimpleFlight) |
| `results/` | Result tables and figures used in the report |

## Project history
The first attempt used a standalone neural network trained offline as the controller. On the same plant it matched PID and was worse than LQG,
so the project was changed to the design in the reference paper (SPNN as an online-adaptive compensator on top of PID/LQG). The code of that
first attempt is not part of this repository; it is described in the report.

## How to run (Python part, no AirSim needed)
```
pip install -r requirements.txt
python run_python_pipeline.py      # tune Gamma, evaluate, stress test, adaptation demo
python effect_sizes.py             # effect sizes and failure-rate confidence intervals
python ablation_basis.py           # basis ablation (add --base LQG for the LQG version)
python spnn_vs_disturbance.py gust # SPNN output vs the real disturbance
```
The simulator uses a velocity-loop lag of 0.1 s by default. If `plant_lag.json` exists (written by `identify_lag.py` from an AirSim flight)
the simulator uses the measured per-axis lag instead. The results below use the default plant. `plant_lag_saved.json` is the AirSim measurement
(rename it to `plant_lag.json` to use it).

## AirSim (optional validation)
Needs Unreal Engine 4.27 + AirSim with the default SimpleFlight controller (`airsim_settings/settings.json`), then
`identify_lag.py`, a smoke test with `run_overnight.py --episodes 1 --hours 0.1`, and `run_airsim_5h.py`.
**Status:** scripts are provided; AirSim results are not part of the results below.
Note: AirSim's horizontal velocity loop is much slower than the default simulator plant (lag about 0.76 s plus about 0.2 s dead time);
in an offline test on that plant the compensator gave no benefit and became unstable at large Gamma.

## Results (default simulator plant, 3,000 runs per scenario type)
Mean tracking RMSE in metres, and failure rate (RMSE > 1 m or gust recovery > 3 s):

| Scenario | PID | PID+SPNN | LQG | LQG+SPNN |
|---|---|---|---|---|
| train | 0.150 / 3.13% | **0.114 / 0%** | 0.140 / 0.33% | **0.101 / 0%** |
| unseen (1.5x harder) | 0.194 / 7.50% | **0.138 / 0%** | 0.189 / 5.53% | **0.121 / 0%** |
| extreme (2x harder) | 0.233 / 10.53% | **0.159 / 0.03%** | 0.229 / 11.20% | **0.139 / 0.10%** |

- Median paired RMSE change from adding the SPNN: PID -3.5 / -5.4 / -7.0 cm (-24 / -29 / -32 %), LQG -3.6 / -6.4 / -8.5 cm (-28 / -36 / -39 %) for train / unseen / extreme; lower in about 100% of paired runs.
- **Trade-off:** control commands are more active with the compensator (higher control-rate RMS in all paired runs), and LQG remains best on the combined cost against PID+SPNN.
- **Ablation:** a one-term adaptive bias reaches similar accuracy to the 27-term sigma-pi basis with smoother control, so in this plant the benefit comes from online adaptation more than from the product terms.
- **Disturbance cancellation (100 flights, PID+SPNN):** SPNN output correlates with the real disturbance (r = 0.43 / 0.53 / 0.89 on X / Y / Z) but cancels only part of it (5 / 13 / 36 % of the RMS).

## Limitations
- Results come from a simplified point-mass simulator, not a full quadrotor with an attitude loop as in the paper.
- The adaptation rate Gamma must be set carefully; on a slow, delayed plant the compensator was not beneficial in an offline check.
- Absolute gains are centimetres. Statistical tests are not reported because with thousands of runs every difference is "significant"; effect sizes are reported instead.

## License
Add a license of your choice (check your university's policy on project code first).
