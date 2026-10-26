"""One command, ~5 hours, fully unattended (needs Unreal + PX4 running with the drone):
   1) quick Gamma check in AirSim (~25 min)   2) paired wind benchmark of PID / LQG / PID+SPNN / LQG+SPNN (rest of the time)
   3) analysis + plots.   Resumable: if it stops, run the SAME command again.
   python run_airsim_5h.py --hours 5"""
import argparse, subprocess, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=5.0); ap.add_argument("--max-retries", type=int, default=10); a = ap.parse_args()
T0 = time.time(); END = T0 + a.hours * 3600
py = lambda cmd: subprocess.run([sys.executable, "-u"] + cmd).returncode
left = lambda: (END - time.time()) / 3600

print(f"##### Stage 1: Gamma check   {time.strftime('%H:%M:%S')}", flush=True)
for attempt in range(a.max_retries):
    if py(["tune_gamma_airsim.py", "--gammas", "5", "10", "20", "--scenarios", "3"]) == 0: break
    print("Gamma stage failed; waiting 3 min, then resuming (restart Unreal/PX4 if possible)..."); time.sleep(180)
else: sys.exit("Gamma stage kept failing - check Unreal/PX4, then rerun this command.")

print(f"\n##### Stage 2: benchmark ({left():.2f} h left)   {time.strftime('%H:%M:%S')}", flush=True)
for attempt in range(a.max_retries):
    if left() < 0.15: break
    rc = py(["run_overnight.py", "--hours", f"{max(left() - 0.1, 0.05):.3f}", "--episodes", "100000", "--use-tuned"])
    if rc == 0: break
    print("Benchmark stopped with an error; waiting 3 min, then resuming..."); time.sleep(180)

print(f"\n##### Stage 3: analysis   {time.strftime('%H:%M:%S')}", flush=True)
py(["analyze_airsim.py"])
print(f"\nALL DONE after {(time.time()-T0)/3600:.1f} h. Open airsim_summary.csv, airsim_winrates.csv, airsim_overview.png, airsim_xyz.png, airsim_path_xy.png")
