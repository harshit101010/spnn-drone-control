"""Builds the four controllers: PID, LQG, PID+SPNN, LQG+SPNN (paper design: SPNN = online compensator)."""
import json, os
from controllers import PID, LQG, Compensated

def params():
    if os.path.exists("params.json"): return json.load(open("params.json"))
    from evaluate import tune_pid, tune_lqg
    pid, lqg = tune_pid(), tune_lqg()
    P = {"PID": [float(pid.kp), float(pid.ki), float(pid.kd)], "LQG": [float(lqg.qe), float(lqg.r)]}
    json.dump(P, open("params.json", "w")); return P

def gammas():
    return json.load(open("gamma.json")) if os.path.exists("gamma.json") else {"PID": 30.0, "LQG": 30.0}

def build():
    P, G = params(), gammas()
    mk_pid = lambda: PID(*P["PID"]); mk_lqg = lambda: LQG(qe=P["LQG"][0], r=P["LQG"][1])
    return {"PID": mk_pid(), "LQG": mk_lqg(),
            "PID+SPNN": Compensated(mk_pid(), G["PID"]), "LQG+SPNN": Compensated(mk_lqg(), G["LQG"])}
