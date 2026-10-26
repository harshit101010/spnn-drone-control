"""PART 2 - needs Unreal + PX4 running. Unattended: tune base controllers (ES), tune Gamma, wind benchmark, analysis.
Every stage resumes after a crash; failed stages are retried.
  python run_airsim_pipeline.py --es-generations 20 --bench-hours 8     (add --skip-es to skip base tuning)"""
import argparse, subprocess, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--es-generations", type=int, default=20); ap.add_argument("--bench-hours", type=float, default=8.0)
ap.add_argument("--skip-es", action="store_true"); ap.add_argument("--max-retries", type=int, default=20); a = ap.parse_args()

def run(cmd, label):
    for attempt in range(a.max_retries):
        print(f"\n##### {label} (attempt {attempt+1}) {time.strftime('%H:%M:%S')}", flush=True)
        if subprocess.run([sys.executable, "-u"] + cmd).returncode == 0: return True
        print("Stage failed; waiting 5 min then resuming (restart Unreal/PX4 if you can)..."); time.sleep(300)
    return False

if not a.skip_es:
    for c in ["PID", "LQG"]:
        if not run(["finetune_es.py", "--controller", c, "--generations", str(a.es_generations)], f"ES tune {c}"): sys.exit("gave up on " + c)
if not run(["tune_gamma_airsim.py"], "tune Gamma"): sys.exit("gave up on Gamma")
run(["run_overnight.py", "--hours", str(a.bench_hours), "--episodes", "100000", "--use-tuned"], "benchmark")
run(["analyze_airsim.py"], "analysis")
print("ALL DONE. Open airsim_overview.png, airsim_xyz.png, airsim_summary.csv")
