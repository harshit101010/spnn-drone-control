"""PART 1 - Python only (no AirSim). Runs everything unattended:
tune Gamma -> evaluate -> stress test -> adaptation demo.   Run:  python run_python_pipeline.py"""
import subprocess, sys, time
STAGES = ["tune_gamma.py", "evaluate.py", "stress_test.py", "adaptation_demo.py"]
t0 = time.time()
with open("pipeline_python_log.txt", "a", encoding="utf-8") as log:
    for s in STAGES:
        print(f"\n##### {s}  (elapsed {(time.time()-t0)/60:.0f} min)", flush=True)
        r = subprocess.run([sys.executable, "-u", s], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        log.write(f"\n##### {s}\n{r.stdout}\n"); log.flush(); print(r.stdout[-1500:])
        if r.returncode != 0: print(f"STAGE FAILED: {s} - see pipeline_python_log.txt"); sys.exit(1)
print(f"\nALL DONE in {(time.time()-t0)/60:.0f} min. Read check_report.txt first.")
