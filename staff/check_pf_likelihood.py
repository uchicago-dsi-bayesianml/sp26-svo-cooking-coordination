#!/usr/bin/env python3
"""Course-staff check: what can the particle filter's likelihood tell apart?

Re-derives, in NumPy only, the per-particle likelihood P(a | theta) that
SVOParticleFilter._likelihood computes, and the exact posterior over theta that
the filter approximates. Every formula cites the file and line it comes from
(paths relative to gym_cooking/, line numbers as of the staff-polish branch):

  delegation_planner/svo_particle_filter.py
    L45       stay probability on the None policy: max(0.5, |cos theta| ** 0.5)
    L48, 78   prior: theta ~ Uniform(0, pi/2); beta = 1.3
    L157-161  mixing weights w_none = |cos| / (|cos| + |sin|), w_coop = |sin| / (...)
    L170-176  P(a | None policy): stay prob for (0, 0), the rest split over the
              other valid actions
    L187      planner.use_svo_cost = False, so the Q values below are the same
              for every particle
    L205-209  Boltzmann share softmax(-beta * Q)[a]
    L213-215  p_coop = 0.5 * Boltzmann share + 0.5 * distance score
    L234-237  distance score = 1 / (1 + exp(-2 * progress))
    L217      P(a | theta) = w_none * p_none + w_coop * p_coop
    L98-112   w_k <- w_k * P(a | theta_k), normalized

The distance score is not a distribution over actions, so P(a | theta) summed
over the valid actions is not 1; part 2 prints the total and part 3 shows what
the posterior would be if it were divided out.

The Q values need the authors' BRTDP planner and a kitchen state. They do not
depend on theta (L187), so this script uses an illustrative set for a partner
three steps from its target with five valid actions, built from the planner's
costs (navigation_planner/planners/e2e_brtdp.py L94-95: 1.0 per step plus 0.1
per move) and an exact cost-to-go of 1.1 per remaining step. Part 4 varies
p_coop to show how little the result depends on these values.

Needs only NumPy:  python staff/check_pf_likelihood.py [--steps T]
"""
import argparse

import numpy as np

BETA = 1.3                                   # svo_particle_filter.py L48
# action: (Q value under the original cost, change in distance to the target)
ACTIONS = {
    "step toward": (1.1 + 1.1 * 2, +1),
    "step sideways": (1.1 + 1.1 * 3, 0),     # two of the five actions
    "step away": (1.1 + 1.1 * 4, -1),
    "stay": (1.0 + 1.1 * 3, 0),
}
VALID = ["step toward", "step sideways", "step sideways", "step away", "stay"]


def stay_prob(theta):
    """svo_particle_filter.py L45 (the same formula as utils/agent.py L64)."""
    return np.maximum(0.5, np.abs(np.cos(theta)) ** 0.5)


def mixing_weights(theta):
    """svo_particle_filter.py L157-161."""
    c, s = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    total = c + s + 1e-10
    return c / total, s / total


def p_coop_table():
    """svo_particle_filter.py L205-215 and L234-237, for the illustrative Q values."""
    q = np.array([ACTIONS[a][0] for a in VALID])
    boltzmann = np.exp(-BETA * q - np.max(-BETA * q))
    boltzmann /= boltzmann.sum()
    out = {}
    for name, (_, progress) in ACTIONS.items():
        share = boltzmann[VALID.index(name)]
        dist = 1.0 / (1.0 + np.exp(-2.0 * progress))
        out[name] = 0.5 * share + 0.5 * dist
    return out


def likelihood(action, theta, p_coop):
    """svo_particle_filter.py L170-176 and L217."""
    w_none, w_coop = mixing_weights(theta)
    if action == "stay":
        p_none = stay_prob(theta)
    else:
        p_none = (1.0 - stay_prob(theta)) / (len(VALID) - 1)
    return w_none * p_none + w_coop * p_coop[action]


def total_mass(theta, p_coop):
    """Sum of P(a | theta) over the five valid actions. A likelihood that is a
    probability distribution over the partner's actions sums to 1."""
    return sum(likelihood(a, theta, p_coop) for a in VALID)


def grid_posterior(counts, p_coop, grid, normalize=False):
    """Exact posterior on a grid under the uniform prior (L48, L78) after
    observing counts[action] actions of each kind (L98-112 without particles).
    With normalize=True each P(a | theta) is divided by its total over the
    valid actions first, which the authors' code does not do."""
    logp = np.zeros_like(grid)
    z = total_mass(grid, p_coop) if normalize else 1.0
    for action, n in counts.items():
        if n:
            logp += n * np.log(likelihood(action, grid, p_coop) / z + 1e-300)
    w = np.exp(logp - logp.max())
    return w / w.sum()


def summarize(w, grid):
    mean = float((w * grid).sum())
    sd = float(np.sqrt((w * (grid - mean) ** 2).sum()))
    return np.degrees(mean), np.degrees(grid[np.argmax(w)]), np.degrees(sd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=40,
                    help="number of observed partner actions (default 40)")
    args = ap.parse_args()

    grid = np.radians(np.linspace(0.0, 90.0, 181))
    p_coop = p_coop_table()
    degs = [0, 15, 30, 45, 60, 75, 90]

    print("1. Likelihood P(a | theta) for each kind of observed action")
    print("   p_coop (the same for every theta):",
          ", ".join(f"{a} {p:.3f}" for a, p in p_coop.items()))
    print("   theta   " + "".join(f"{a:>15s}" for a in ACTIONS))
    for d in degs:
        th = np.radians(d)
        row = "".join(f"{likelihood(a, th, p_coop):15.3f}" for a in ACTIONS)
        print(f"   {d:4d} deg{row}")

    print("\n2. Total of P(a | theta) over the five valid actions, and the share of it on standing still")
    for d in degs:
        th = np.radians(d)
        tot = total_mass(th, p_coop)
        print(f"   theta {d:2d} deg: total {tot:.3f}, stay share {likelihood('stay', th, p_coop) / tot:.2f}")

    T = args.steps
    print(f"\n3. Exact posterior after {T} observed actions, by the share of them spent standing still")
    print("   (the other actions are steps toward the target; uniform prior on 0 to 90 deg)")
    print("   stay share   posterior mean   mode    sd   mean if P(a | theta) were normalized")
    for f in np.linspace(0.0, 1.0, 11):
        n_stay = int(round(T * f))
        counts = {"stay": n_stay, "step toward": T - n_stay}
        mean, mode, sd = summarize(grid_posterior(counts, p_coop, grid), grid)
        norm_mean = summarize(grid_posterior(counts, p_coop, grid, normalize=True), grid)[0]
        print(f"   {f:9.1f}   {mean:13.1f}   {mode:5.1f}   {sd:4.1f}   {norm_mean:9.1f}")

    print("\n   The same with the moves split evenly between toward, sideways and away:")
    for f in [0.0, 0.2, 0.4, 0.6]:
        n_stay = int(round(T * f))
        n_move = T - n_stay
        k = n_move // 3
        counts = {"stay": n_stay, "step toward": n_move - 2 * k,
                  "step sideways": k, "step away": k}
        w = grid_posterior(counts, p_coop, grid)
        mean, mode, sd = summarize(w, grid)
        print(f"   stay share {f:.1f}: posterior mean {mean:5.1f}, mode {mode:5.1f}, sd {sd:4.1f}")

    print(f"\n4. Posterior mode for a partner that never stands still in {T} steps,")
    print("   for other values of p_coop on its moves:")
    modes = []
    for pc in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]:
        w = grid_posterior({"step toward": T}, dict(p_coop, **{"step toward": pc}), grid)
        modes.append(f"p_coop {pc:.2f} -> {summarize(w, grid)[1]:.1f} deg")
    print("   " + "; ".join(modes))


if __name__ == "__main__":
    main()
