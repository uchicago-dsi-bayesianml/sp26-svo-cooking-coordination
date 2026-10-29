#!/usr/bin/env python3
"""Course-staff check: what the delegator believes about its partner with --infer-svo on.

Copies gym_cooking/ to a temporary folder (so nothing is written into the
repository), runs the authors' main.py twice with the same arguments and the
same --seed, and reads the metrics pickles the runs save. For every time step
it prints

  * the partner estimate each agent's delegator plans with
    (partner_svo_estimates, seeded in utils/agent.py L139-142 and blended in
    delegation_planner/bayesian_delegator.py L664 and L690-691), and
  * the particle filter's posterior mean (svo_posterior),

then says whether the two runs agree. Defaults: the README's selfish-partner
case (ego 90 deg, partner 0 deg) on open-divider_salad, cut to 10 time steps
with small planner caps so that a run takes seconds; the README's own runs used
--cap 25 --main-cap 15 and up to 100 steps.

Needs the packages in requirements.txt:
    python staff/trace_partner_estimates.py [--svo1 90] [--svo2 0] [--steps 10]
"""
import argparse
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]


def run_once(tmp, args):
    pkg = tmp / "gym_cooking"
    (pkg / "misc" / "metrics" / "pickles").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", "--num-agents", "2", "--level", args.level,
           "--model1", "bd", "--model2", "bd", "--svo1", str(args.svo1),
           "--svo2", str(args.svo2), "--max-num-timesteps", str(args.steps),
           "--cap", str(args.cap), "--main-cap", str(args.main_cap),
           "--seed", str(args.seed), "--infer-svo", "--n-particles", str(args.particles)]
    env = dict(os.environ, PYTHONPATH=str(tmp), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(cmd, cwd=pkg, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (pickle_path,) = (pkg / "misc" / "metrics" / "pickles").glob("*.pkl")
    sys.path.insert(0, str(pkg))
    import dill
    with open(pickle_path, "rb") as f:
        data = dill.load(f)
    pickle_path.unlink()
    return data


def deg(x):
    return f"{math.degrees(x):5.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", default="open-divider_salad")
    ap.add_argument("--svo1", type=float, default=90.0)
    ap.add_argument("--svo2", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--main-cap", type=int, default=5)
    ap.add_argument("--particles", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="svo-staff-"))
    try:
        shutil.copytree(REPO / "gym_cooking", tmp / "gym_cooking", symlinks=True)
        runs = [run_once(tmp, args) for _ in range(2)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    true = runs[0]["true_svo"]
    print(f"level {args.level}, --seed {args.seed}, true SVO: agent-1 {deg(true['agent-1'])} deg,"
          f" agent-2 {deg(true['agent-2'])} deg, {args.particles} particles")
    for i, data in enumerate(runs, 1):
        est = data["partner_svo_estimates"]
        post = data["svo_posterior"]
        pf12 = {d["t"]: d["mean"] for d in post["agent-1"].get("agent-2", [])}
        pf21 = {d["t"]: d["mean"] for d in post["agent-2"].get("agent-1", [])}
        print(f"\nrun {i}   step   agent-1 plans with   agent-1 filter mean"
              f"   agent-2 plans with   agent-2 filter mean")
        for t, (e1, e2) in enumerate(zip(est["agent-1"], est["agent-2"]), 1):
            f12 = deg(pf12[t]) if t in pf12 else "    -"
            f21 = deg(pf21[t]) if t in pf21 else "    -"
            print(f"        {t:4d}   {deg(e1['agent-2']):>18s}   {f12:>19s}"
                  f"   {deg(e2['agent-1']):>18s}   {f21:>19s}")
        print(f"        filter snapshots: agent-1 at {len(pf12)} steps, agent-2 at {len(pf21)}")
    same = runs[0]["partner_svo_estimates"] == runs[1]["partner_svo_estimates"]
    print(f"\nThe two runs with --seed {args.seed} give {'the same' if same else 'different'}"
          " partner estimates.")


if __name__ == "__main__":
    main()
