#!/usr/bin/env python3
"""Course-staff check: run the authors' SVOParticleFilter on synthetic partners.

Imports gym_cooking/delegation_planner/svo_particle_filter.py unchanged and runs
it without the simulator: a stand-in planner returns the same illustrative Q
values and distances as staff/check_pf_likelihood.py (a partner three steps
from its target, five valid actions), and the module's get_single_actions is
replaced by a function that returns those five actions. Everything else
(likelihood, log-weight update, ESS test, resampling with jitter) is the
authors' code.

It prints
  1. the authors' _likelihood next to the NumPy re-derivation in
     check_pf_likelihood.py, as a check that the two agree;
  2. the final posterior mean after 40 observed actions with N = 64 particles
     (the default, svo_particle_filter.py L47) for partners that stand still a
     given share of the time, over 20 filter seeds;
  3. the value of use_svo_cost on every planner copy the filter made.

Needs NumPy, SciPy, networkx, matplotlib and termcolor (the module's own import
chain; no gym):  python staff/run_authors_pf.py [--seeds 20] [--steps 40]
"""
import argparse
import copy
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gym_cooking"))
sys.path.insert(0, str(REPO / "staff"))

import delegation_planner.svo_particle_filter as pfmod  # noqa: E402
import check_pf_likelihood as ref  # noqa: E402

MOVES = {(0, 1): "step toward", (1, 0): "step sideways", (-1, 0): "step sideways",
         (0, -1): "step away", (0, 0): "stay"}
VALID = list(MOVES)
START = 3


class State:
    def __init__(self, d):
        self.d = d

    def get_repr(self):
        return self.d


class StandInPlanner:
    """Implements only the planner methods SVOParticleFilter._likelihood calls."""
    log = []

    def __init__(self):
        self.v_l = {}
        self.use_svo_cost = None

    def set_svo(self, theta):
        self.svo = theta

    def set_settings(self, env, subtask, subtask_agent_names):
        StandInPlanner.log.append(self.use_svo_cost)
        self.start = State(START)

    def get_actions(self, state_repr):
        return list(VALID)

    def Q(self, state, action, value_f):
        return ref.ACTIONS[MOVES[action]][0]

    def T(self, state_repr, action):
        return State(state_repr - ref.ACTIONS[MOVES[action]][1])

    def _subtask_distance(self, state):
        return state.d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    pfmod.get_single_actions = lambda env, agent: list(VALID)
    obs = SimpleNamespace(sim_agents=[SimpleNamespace(name="agent-2")])
    planner = StandInPlanner()

    print("1. Authors' _likelihood against the NumPy re-derivation")
    pf = pfmod.SVOParticleFilter("agent-2", planner, n_particles=4,
                                 rng=np.random.default_rng(0))
    p_coop = ref.p_coop_table()
    worst = 0.0
    for deg in np.linspace(0, 90, 91):
        th = float(np.radians(deg))
        for move, name in MOVES.items():
            theirs = pf._likelihood(obs, move, "subtask", ("agent-2",), th)
            worst = max(worst, abs(theirs - float(ref.likelihood(name, th, p_coop))))
    print(f"   largest difference over 91 angles and 5 actions: {worst:.1e}")

    T = args.steps
    print(f"\n2. Posterior mean after {T} observed actions, N = 64, {args.seeds} filter seeds")
    print("   (moves are steps toward the target; the order of stays and moves is random)")
    print("   stay share   mean of posterior means   range            mean posterior sd   resamplings")
    for f in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        means, sds, resamples = [], [], []
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            n_stay = int(round(T * f))
            seq = [(0, 0)] * n_stay + [(0, 1)] * (T - n_stay)
            rng.shuffle(seq)
            pf = pfmod.SVOParticleFilter("agent-2", planner, n_particles=64,
                                         rng=np.random.default_rng(seed))
            count = 0
            for a in seq:
                pf.update(copy.copy(obs), a, "subtask", ("agent-2",))
                if np.allclose(pf.weights, 1.0 / pf.n_particles):
                    count += 1
            means.append(np.degrees(pf.posterior_mean()))
            sds.append(np.degrees(pf.posterior_std()))
            resamples.append(count)
        print(f"   {f:9.1f}   {np.mean(means):22.1f}   {min(means):5.1f} to {max(means):5.1f}"
              f"   {np.mean(sds):17.1f}   {np.mean(resamples):11.1f}")

    seen = sorted(set(StandInPlanner.log), key=str)
    print(f"\n3. use_svo_cost on the {len(StandInPlanner.log)} planner copies the filter made: {seen}")


if __name__ == "__main__":
    main()
