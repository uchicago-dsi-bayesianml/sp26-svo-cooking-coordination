<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Inferring Social Value Orientation in Multi-Agent Cooking Coordination

> **Example project from Spring 2026.** This is a copy of [jouisseuse/gym-cooking](https://github.com/jouisseuse/gym-cooking), shown with the authors' work and commit history intact. That repository is itself a fork of [rosewang2008/gym-cooking](https://github.com/rosewang2008/gym-cooking), the MIT-licensed code of Wu, Wang et al. (2021), "Too many cooks: Bayesian inference for coordinating multi-agent collaboration", which the team extended. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project). Every result comes from simulated episodes, so there is no dataset; two images in the write-up are borrowed and keep their owners' rights, as [data/README.md](data/README.md) explains.

Some teammates do their share and some leave it to others, and you can only watch what they do. Can an agent infer how much a partner cares about the team from the partner's actions, and plan around it? The team started from Bayesian Delegation, the model behind the "Too many cooks" paper, in which two cooks in a grid-world kitchen each keep a posterior over who is doing which part of a recipe and update it by inverse planning on the other's moves. They gave every cook a continuous Social Value Orientation (SVO), an angle θ from 0° (selfish) to 90° (altruistic), let it change how tasks get assigned and how an idle cook behaves, and wrote a particle filter with which each cook infers its partner's θ from the partner's actions. Three recorded episodes show a selfish partner standing still while the altruistic cook makes the salad alone, and posterior plots show the filter moving toward the true θ for selfish and altruistic partners and overshooting for a prosocial one.

## Summary

**Environment.** The upstream gym-cooking simulator: a 7 by 7 kitchen (`open-divider_salad`) with one tomato, one lettuce, two plates, two cutting boards and a delivery square, and a salad recipe of nine subtasks (chop, merge onto a plate, deliver). The two cooks are the upstream Bayesian Delegation agents: each plans its moves with a bounded real-time dynamic programming planner (BRTDP) and keeps a posterior over subtask allocations, updated with a Boltzmann likelihood of the observed moves.

**What the team added.** An SVO angle per cook (`--svo1`, `--svo2`) that enters the code in two places: it multiplies the prior over subtask allocations by |sin θ| for every cook given work and |cos θ| for every cook left idle, and it sets how often an idle cook stands still, from 0.5 at 90° to 1.0 at 0°. The utility cos θ · (own effort) + sin θ · (team progress) that the write-up describes is in the planner's code but never switched on.

**Inference.** With `--infer-svo`, each cook runs a particle filter over its partner's θ: 64 particles by default, a uniform prior on 0° to 90°, a hand-built likelihood mixing "idle" and "working" explanations of each move, resampling when the effective sample size halves, and the posterior mean blended into the estimate the cook plans with. The team also ported the code to gym 0.26 and wrote the plotting and GIF scripts.

**Experiments.** The ego cook at 90° and the partner at 0°, 45° or 90°, one episode each with `--seed 1`. All three deliver the salad, in 45, 52 and 41 steps. The posterior settles near 7° for the selfish partner and near 85° for the altruistic one, and climbs to about 87° for the prosocial partner, whose true value is 45°.

The staff section below explains why the cos/sin utility has no effect, why the likelihood cannot tell a 45° partner from a 90° one that keeps moving, why the inferred estimate starts from the true answer, and why none of the runs can be repeated exactly.

## Team and contributions

Team "Hidden Layer Heroes", Spring 2026.

- **Bufan Gao** ([@jouisseuse](https://github.com/jouisseuse)): all of the code, in twelve commits on May 23: the SVO helpers, the SVO factor in the allocation prior, the idle-cook rule, the particle filter and its fallback subtask, the plate-deadlock rule, the planner's SVO cost and its switch-off, posterior logging, the plotting, GIF and sweep scripts, and the gym 0.26 fixes. Also the three behaviour GIFs, the first inference figure, `PRESENTATION_OUTLINE.md` and the first README, whose layout, run instructions, caveats and citation survive in the write-up. The last code commit (`9cf9de7`) replaced the particle filter with a stub and a pointer to the full version in `89017fb`, for a teammate to fill in.
- **Tianrun Wu** (commits as 吴天润, from a local git identity not linked to a GitHub account): one commit on May 23, "Implement SVO particle filter inference" (`2b3f8f3`). The file it produces is identical to the full version in `89017fb`.
- **Jiabin Zou** ([@jiabinz-uchi](https://github.com/jiabinz-uchi)): 22 commits on May 26 and 27 through GitHub's web editor: most of the write-up's text, including the introduction, core idea, method, behaviour results and first inference demo; the headline image, the figure from Wu et al. and the update-loop diagram; and two figures of short 8-particle runs that the write-up does not use.
- **Yufei Mao** ([@YufeiM28](https://github.com/YufeiM28)): 17 commits from May 26 to 29 through the web editor: the interpretations of the second and third inference demos, the takeaways and the caveat on the middle SVO, and the three inference figures in the write-up, from runs with 64 particles. The Demo 1 figure was replaced on May 28, after the course deadline, and the last edits to the write-up are from May 29.

Every member has at least one commit. 9 of the 52 commits carry a `Co-Authored-By: Claude Opus 4.7` trailer. The team's commits run from May 23 to May 29, 2026. Much of it was uploaded through the web editor, so the account that committed a file is not necessarily its only author; the attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/sp26-svo-cooking-coordination/commits/main).

## Getting started

`setup.py` is the upstream file and pins 2020 versions, gym 0.17.2 among them. The team ran the code on Python 3.11 with gym 0.26, and `requirements.txt`, added by staff, lists the versions course staff used on Python 3.11.17. Install those, then the package without its pins:

```sh
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
cd gym_cooking
mkdir -p misc/metrics/pickles
```

**The pipeline**, run from `gym_cooking/`:

1. A behaviour run, recorded: `python main.py --num-agents 2 --level open-divider_salad --model1 bd --model2 bd --svo1 90 --svo2 0 --max-num-timesteps 100 --cap 25 --main-cap 15 --seed 1 --record`. Change `--svo2` to 45 or 90 for the other partners. Frames go to `misc/game/record/<run name>/`, the pickle to `misc/metrics/pickles/`.
2. The GIF: `python -m misc.metrics.make_gif --frames misc/game/record/<run name> --out <file>.gif --duration 250`.
3. An inference run: the same command with `--infer-svo --n-particles 64`. Commit `abdbd93` puts a run with 12 particles at about 25 minutes.
4. The posterior plot: `python -m misc.metrics.plot_svo_inference --pickle misc/metrics/pickles/<run>.pkl --out <file>.png`. The write-up's commands write into `../images_svo/`, over the authors' figures; choose another path.
5. `python -m experiments.run_svo` sweeps the partner over five SVO values with five seeds each (see below).

**Staff checks.** Four scripts in `staff/`, each documented at the top:

```sh
python staff/check_pf_likelihood.py      # NumPy only, under a second
python staff/run_authors_pf.py           # the authors' filter without gym, a few seconds
python staff/trace_partner_estimates.py  # two short simulator runs, about 15 seconds
python staff/check_gif_steps.py          # Pillow, episode lengths of the three GIFs
```

**What does not run as committed.** Without `misc/metrics/pickles/`, which is not in the repository, every run ends in a `FileNotFoundError` when it saves; the upstream code has the same problem. `pip install -e .` alone tries to install the 2020 pins. The renderer asks for `Plate.png`, so the team added `gym_cooking/misc/game/graphics/Plate.png` as a symbolic link to `plate.png` for Linux; on macOS and Windows, whose file systems ignore case, the two names collide: a fresh clone on macOS reports `Plate.png` as modified, and unpacking or copying the tree can leave a link that points at itself instead of the image. The write-up's "How to run" still says "Once Part 2 lands"; the filter is implemented on this branch. `--shaping-weight` changes nothing. `experiments/run_svo.py` defaults to another kitchen, an ego cook at 45° and the upstream planner caps of 75 and 100; it was never run, and nothing reads its output.

## Data

There is no dataset: the simulator generates every episode from a kitchen layout in `gym_cooking/utils/levels/`. Each run writes a `dill` pickle that, with the team's additions, holds the true SVOs, the partner estimates and the filter's state at every step. None of the pickles or frames behind the committed figures are in the repository. [data/README.md](data/README.md) describes the layouts, the pickle, how to make the records again and the terms: the simulator is MIT-licensed, the write-up's headline image is a screenshot of the Overcooked video game (Ghost Town Games and Team17), and `images_svo/Wu.jpg` is a figure from Wu et al. (2021).

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up. |
| `gym_cooking/main.py` | Entry point, with the team's SVO and filter flags and the gym 0.26 fixes. |
| `gym_cooking/utils/svo.py` | SVO presets and parsing. |
| `gym_cooking/utils/agent.py` | The cook, with the team's SVO, idle-cook rule and partner estimates. |
| `gym_cooking/delegation_planner/bayesian_delegator.py` | Bayesian Delegation (upstream), with the team's SVO prior factor, plate rule and filter call. |
| `gym_cooking/delegation_planner/svo_particle_filter.py` | The team's particle filter over a partner's SVO. |
| `gym_cooking/navigation_planner/planners/e2e_brtdp.py` | The BRTDP planner (upstream), with the team's inactive SVO cost and a distance helper. |
| `gym_cooking/misc/metrics/metrics_bag.py` | The per-run pickle (upstream), with the team's SVO entries. |
| `gym_cooking/misc/metrics/plot_svo_inference.py`, `make_gif.py` | The team's posterior plots and GIF maker. |
| `gym_cooking/experiments/run_svo.py` | The team's sweep over partner SVO and seeds. Never run. |
| `gym_cooking/utils/levels/open-divider_salad_1plate.txt` | A one-plate kitchen the team added; no reported run uses it. |
| `images_svo/` | The write-up's GIFs, figures and diagram (`updateloop.png` and `updateloop2.png` are the same bytes), plus three earlier inference figures it does not show. |
| `images/`, `docs/`, `oldpj_README.md` | The upstream figures (three GIFs of 1.9 to 2.8 MB among them), design notes and README. |
| `PRESENTATION_OUTLINE.md` | The team's outline for the presentation, with an early results table. |
| `requirements.txt`, `data/README.md`, `staff/` | Course staff's additions, described above. |

## Method

**Bayesian Delegation, as upstream.** Each cook keeps a distribution over subtask allocations, such as "I chop the tomato, my partner chops the lettuce", with a prior that favours allocations whose objects are close. After a step it multiplies each allocation's probability by the likelihood of the moves it saw, a softmax with β = 1.3 over the planner's Q-values for the assigned subtasks, and plans its own moves for the most probable allocation. The planner itself picks the move with the lowest Q-value; the softmax is only the observer's model.

**SVO in the allocation prior.** Each allocation's prior is multiplied by the product over cooks j of |sin θⱼ| + 0.001 if j has a subtask and |cos θⱼ| + 0.001 if j is idle (`bayesian_delegator.py` lines 242 to 273), with θⱼ the cook's own SVO for itself and its current estimate for the partner. At 45° for both cooks the factor is the same for every allocation, which gives the upstream prior; with a partner believed to be at 0°, every allocation that gives the partner work is multiplied by about 0.001.

**SVO in the idle cook's behaviour.** A cook with no subtask stands still with probability max(0.5, |cos θ|^0.5), otherwise it steps at random (`utils/agent.py` line 64). Upstream it is 0.5 for everyone; here it is 1.0 at 0°, so a selfish cook, whose own prior leaves it idle, never moves, and 0.84 at 45°.

**The SVO cost, inactive.** `e2e_brtdp.py` lines 519 to 529 compute cos θ · (own cost) − sin θ · λ · (team progress) only when the planner's `use_svo_cost` is true, and the one assignment in the repository, in the filter (`svo_particle_filter.py` line 187), sets it to false. The planner uses the upstream cost for every θ, and `--shaping-weight`, the λ, has no effect. The commit messages record why: the cost was added in `bb830c4`, taken out of the cooks' planning in `abdbd93` so that the planner would converge as upstream, and out of the filter in `b3a7496` because at θ = 90° it made all Q-values equal. The write-up's Core Idea and Part 1 describe the first version.

**Partner estimates.** Each cook keeps a number for its partner's θ. It is set at the start of every run from the partner's true SVO on the command line (`utils/agent.py` lines 139 to 142), with or without `--infer-svo`. With inference on, each filter update replaces it with 0.85 times the old value plus 0.15 times the filter's posterior mean (`bayesian_delegator.py` lines 664 and 690 to 691).

**The particle filter** (`svo_particle_filter.py`). Particles start uniform on 0° to 90° (lines 48, 78 to 79); negative angles were dropped because |cos θ| and |sin θ| cannot tell +θ from −θ. For an observed partner move a and a particle θ, the likelihood is

> P(a | θ) = w_idle(θ) · P_idle(a | θ) + w_work(θ) · P_work(a),

with w_idle = |cos θ| / (|cos θ| + |sin θ|) and w_work = |sin θ| / (|cos θ| + |sin θ|) (lines 157 to 161, 217). P_idle is the idle cook's rule above, with the remainder split over the other valid moves (lines 169 to 176). P_work averages the move's softmax share under the planner's Q-values for the partner's hypothesized subtask (lines 205 to 209) and a distance score 1 / (1 + e^(−2·progress)), progress being how much the move shortens the distance to the subtask's target (lines 213 to 215, 234 to 237). Because the planner runs with the upstream cost, P_work is the same for every particle, and θ enters only through the two weights and the stay probability. Weights are updated in log space and normalized (lines 98 to 112); when the effective sample size falls below half the particles, the filter resamples multinomially, adds Gaussian jitter of 0.05 rad (about 3°) and clips to the prior's range (lines 115 to 116, 239 to 249).

**When the filter runs.** A cook runs its delegator's update, and with it the filter, only on steps where it has a subtask itself and has not just reset its beliefs (`utils/agent.py` lines 182 to 201). The partner's hypothesized subtask comes from the cook's most probable allocation, or, if that leaves the partner idle, from the prior without the SVO factor (`bayesian_delegator.py` lines 277 to 319, 675 to 679). A selfish cook is idle and never runs its filter, which is why the write-up's selfish-partner figure has one panel.

**Other changes.** Once a plate holds an ingredient, allocations that would start a second plate are removed. The gym 0.26 fixes unwrap the environment and accept the new `reset` and `step` return values.

### What is Bayesian here

Two layers are Bayesian. The upstream Bayesian Delegation, unchanged by the team, is inverse planning: a posterior over subtask allocations, updated by Bayes' rule with a Boltzmann-rational likelihood. The team's layer is a particle filter for a fixed unknown parameter, the partner's θ. Its update is a correct sequential importance-sampling step (multiply by the likelihood, normalize, resample when the weights degenerate), and the write-up's figures show the posterior mean, a one-standard-deviation band and the effective sample size, which is the right way to look at such a filter.

What stops short. The likelihood is built by hand, not derived from how the simulated partner chooses moves. The weights |cos θ| / (|cos θ| + |sin θ|) stand in for the chance that the partner is idle, but in the simulator that chance comes from the partner's own allocation prior, in which the SVO factor is neutral at 45° and the upstream weighting favours allocations where both cooks work; the 45° partner in the write-up's GIF works throughout. The planner part carries no information about θ, and the whole is not a probability distribution over moves (see [what to learn](#what-to-learn-from-this-project)). The jitter adds movement to a parameter the model says is fixed, so the filter's spread is not exactly the posterior's. Downstream only the posterior mean is used, blended into an estimate that starts from the true value, so the filter's uncertainty never reaches a decision, and with one episode per condition nothing is known about how the posterior varies across episodes. For one bounded parameter a grid computes the same posterior exactly; `staff/check_pf_likelihood.py` does so, and the authors' filter agrees with it.

## Results

**Behaviour.** The write-up's three GIFs, ego cook at 90°, `--seed 1`. The step counts are the write-up's; `staff/check_gif_steps.py` reads the same lengths from the GIFs (46, 52 and 42 frames; the 45° GIF has one doubled frame).

| Partner | Steps in the write-up | Steps in the GIF | What the GIF shows |
|---|---:|---:|---|
| Selfish, 0° | 45 | 45 | The partner never moves; the ego cook makes the salad alone. |
| Prosocial, 45° | 52 | 52 | Both cooks work; the partner delivers the salad. |
| Altruistic, 90° | 41 | 41 | Both cooks work, in parallel. |

The write-up reads 41 against 45 steps as a 9 percent speedup from cooperation and 52 as friction at a middle SVO. Each is one episode with one seed, so the differences of 4 to 11 steps are not yet evidence of either. Commit `89017fb` and `PRESENTATION_OUTLINE.md` suggest the episodes were recorded with the filter on, which matters little because the ego cook's estimate starts at the true value.

**Inference.** The write-up's figures, read from their axes, next to the earlier results in `PRESENTATION_OUTLINE.md`.

| Pair (ego, partner) | Figure in the write-up | Particles (top of the ESS axis) | Steps plotted | Final posterior mean in the figure | Earlier result, 16 particles |
|---|---|---:|---:|---|---|
| 90°, 90° | `inf_traj_90-90.jpg` | 64 | 33 | about 83° (ego of partner), about 80° (partner of ego) | 85° both ways, in 41 steps |
| 90°, 45° | `inf_traj_90-45.png` | 64 | 41 | about 85° (ego of partner), about 84° (partner of ego) | 87°, in 52 steps |
| 90°, 0° | `inf_traj_90-0.png` | 64 | 45 | about 6° | 7°, in 45 steps |

The behaviour table and the earlier results come from Bufan Gao's 16-particle runs; the write-up's figures come from later 64-particle runs uploaded by Yufei Mao, two of them episodes of other lengths. At the deadline the Demo 1 figure was the 16-particle one still kept as `images_svo/inference_traj_svo90.png`. Both sets agree in direction: near 0° for the selfish partner and in the 80s for both the altruistic and the prosocial one.

**What the staff scripts add.** `staff/check_pf_likelihood.py` re-derives the likelihood with illustrative planner values for a partner three steps from its target. A step toward the target has likelihood 0.000 at 0°, 0.378 at 45° and 0.717 at 90°; standing still has 1.000, 0.583 and 0.325. After 40 observed actions, a partner that stands still on 0, 20, 40, 50 or 60 percent of them gets posterior means of 88.4°, 87.5°, 83.3°, 65.9° and 42.3°. `staff/run_authors_pf.py` runs the authors' filter on the same partners, 20 times each with 64 particles: its likelihood matches the re-derivation exactly, and the average posterior means are 88.4°, 87.5°, 81.1°, 65.0° and 42.8°. `staff/trace_partner_estimates.py` runs the selfish-partner case twice for 10 steps with `--seed 1`: the ego cook plans with the true 0.0° at step 1 and with 18.1° and 7.6° at step 10 in the two runs; the selfish cook keeps the true 90° for its partner and never runs its filter. A rerun gives different numbers after step 1, which is the point.

## What to learn from this project

Course staff added this section. The project extends a research codebase in a direction the paper left open and works end to end; the points below are about what its results can and cannot support.

- **Check that the mechanism you describe is the one that runs.** The write-up's central equation, U = cos θ · r_self + sin θ · r_team, is never used: the planner applies it only if `use_svo_cost` is true, and the one assignment in the repository sets it to false, as `staff/run_authors_pf.py` confirms on all 358,855 planner copies it makes. SVO acts through two hand-set multipliers instead, |sin θ| and |cos θ| on the allocation prior and the stay probability of an idle cook, so the selfish cook's stillness in the GIF is a rule, not the outcome of a utility. Both are reasonable modelling choices, but they are the model, and the write-up should say so.
- **Make sure your likelihood can tell your hypotheses apart.** The planner part of the likelihood is the same for every θ, so the filter learns θ almost only from how often the partner stands still. In the staff check a partner that never stands still is placed at 88.4°, and one must stand still on about 60 percent of its steps for the posterior mean to come down to about 45°. The simulated 45° cook works most of the time, as its GIF shows, so the overshoot to 85° to 87° in the write-up is built into the model, not noise or too few particles. The write-up's reading, that the evidence "may not be diagnostic enough", is right, and the cause is the likelihood.
- **Write the likelihood as a probability distribution over what you observe.** P(a | θ) should sum to 1 over the partner's possible moves. Here half of the working part is a distance score between 0 and 1 for each move, not a share of 1, so in the staff example the total grows from 1.00 at 0° to 1.75 at 90°, and every observation favours high θ a little. With the total divided out, about 40 percent standing still instead of 60 reaches 45°: better, though an active partner still reads as altruistic.
- **Do not seed the unknown with the answer.** The estimate a cook plans with starts at the partner's true SVO in every run, and each filter update moves it 15 percent of the way toward the filter's mean, which starts near the prior's 45°, so early on inference pulls the plan away from the truth: in the staff trace, from the true 0° to 18.1° and 7.6° after 10 steps in two runs. So the runs never show inference improving what the cooks do; here it makes the plan less accurate. `PRESENTATION_OUTLINE.md` planned the right experiment, the same episode with and without inference, and it was not made. Starting the estimate at 45° under `--infer-svo` and comparing with the run that knows the true value would answer the project's question.
- **Seed every random number generator, and run more than one seed.** The filter makes its own unseeded generator (`svo_particle_filter.py` line 75) and its estimate feeds back into the plan, so two runs with `--seed 1` differ, as the staff trace shows. Every result in the write-up is one episode. `experiments/run_svo.py` sets up the sweep the results need, five partner values times five seeds, and was never run.
- **Keep figures and tables from the same runs.** The behaviour table and the outline's numbers come from 16-particle episodes of 45, 52 and 41 steps; the inference figures in the write-up come from 64-particle episodes of 45, 41 and 33 steps. The pattern is the same, but a reader comparing a figure with the table compares two episodes, and one figure was swapped after the deadline.
- **Find out which agents actually run the filter.** The delegator, and with it the filter, runs only on steps where the cook has a subtask of its own. A selfish cook is idle, so it never infers anything about its partner; the write-up's Demo 3 says only that it "may not produce or save a meaningful posterior trace". The ego cook also skips its update whenever it is between subtasks.
- **Keep the repository's story in one place.** The filter's docstring still describes a prior on −90° to 90° and a pure softmax likelihood, the planner's says the filter switches the SVO cost on, the write-up's "How to run" waits for Part 2 and points to a deleted `SVO_PROJECT.md`, and until staff fast-forwarded it the default branch held the stub. Each was true at some point on May 23. The SVO angle and its online inference also have a literature the write-up could cite next to Wu et al., from Murphy, Ackermann and Handgraaf (2011, *Judgment and Decision Making*) on measuring SVO as an angle to Schwarting et al. (2019, *PNAS*) on inferring drivers' SVO from their motion.

What the project does well: a question from social psychology turned into a working computational model; a real extension of a substantial research codebase, ported to current gym; a particle filter with log-space weights, resampling on effective sample size and plots of the mean, spread, particles and ESS; commit messages that explain each design change and the bug behind it, such as the sign symmetry that led to the [0°, 90°] prior; GIFs that make the behaviour easy to see; and a write-up that says plainly that the middle SVO is the hard case, the caps were low and every result is one seed.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Spring 2026. It builds on gym-cooking by Rose E. Wang, Sarah A. Wu and colleagues, the code of Wu et al. (2021). It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' code is unchanged.

- Fast-forwarded `main` to the authors' `write-up` branch (`2c21791`), the branch the team submitted. The branches `write-up` and `implement-svo-particle-filter` stay as the authors left them.
- Removed `images_svo/.gitkeep`, an empty placeholder in a folder that holds the figures.
- Added `requirements.txt` with the versions course staff ran the code with on Python 3.11.
- Added `data/README.md`, on what the simulator reads and writes, how to make the records again and the terms of the borrowed images, and the four scripts in `staff/` described above.
- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.

## License

The repository's `LICENSE` is the upstream MIT License (copyright 2020 Rose E. Wang), which covers the gym-cooking code. The team did not add a license or a copyright line for their own changes, so ask them before reusing those. The Overcooked screenshot in `images_svo/image.png` and the figure from Wu et al. in `images_svo/Wu.jpg` are covered by neither; their rights stay with their owners.
