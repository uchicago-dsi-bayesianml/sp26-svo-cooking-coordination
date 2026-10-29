# Data

This project uses no external dataset. Every result comes from episodes that the gym-cooking simulator generates on the fly: two cooks in a small grid-world kitchen, each run by the authors' agents, with the Social Value Orientation of each cook set on the command line. The run records the team analyzed were never committed, so this note says what the simulator reads, what a run writes, and how to make the records again.

## What the simulator reads

- **Kitchen layouts:** `gym_cooking/utils/levels/*.txt`, from the upstream project. Each file is a character grid (`-` counter, `/` cutting board, `*` delivery, `t` tomato, `l` lettuce, `o` onion, `p` plate, space for floor), then the recipe name, then one starting position per agent. Every result in the write-up uses `open-divider_salad`: a 7 by 7 kitchen with one tomato, one lettuce, two plates, two cutting boards and one delivery square, and the tomato-and-lettuce salad recipe, which the planner breaks into nine subtasks.
- **`open-divider_salad_1plate.txt`:** a variant with one plate, added by the team in commit `b3a7496` to stop two cooks from plating half a salad each. The final runs went back to the two-plate level and handle that case with a pruning rule in the delegator instead, so this file is not used by any reported run.
- **Sprites:** `gym_cooking/misc/game/graphics/`, from the upstream project, used only when a run is recorded.

## What a run writes, and where

`main.py` saves one record per run to `gym_cooking/misc/metrics/pickles/<run name>.pkl`, a `dill` pickle of a dictionary. The team added three entries to it in `misc/metrics/metrics_bag.py`: `true_svo` (each agent's SVO from the command line, in radians), `partner_svo_estimates` (at every time step, the partner SVO each agent's delegator plans with) and `svo_posterior` (at every time step, each particle filter's mean, standard deviation, effective sample size, particles and weights). The folder `pickles/` is not in the repository, and `main.py` stops with a `FileNotFoundError` at the end of a run unless it exists; this is inherited from the upstream code.

With `--record`, the renderer writes one PNG per time step to `gym_cooking/misc/game/record/<run name>/`. `misc/metrics/make_gif.py` turns that folder into a GIF and `misc/metrics/plot_svo_inference.py` turns a pickle into the posterior-trajectory figures in `images_svo/`.

The upstream `.gitignore` already excludes `*.pkl` and `misc/game/record/`, so none of these outputs can be committed by accident. None of the pickles or frame folders behind the committed figures are in the repository.

## How to make the records again

From `gym_cooking/`, after the setup in the main README:

```sh
mkdir -p misc/metrics/pickles
# One behaviour run, as in the write-up (altruistic ego, selfish partner), recorded:
python main.py --num-agents 2 --level open-divider_salad --model1 bd --model2 bd \
    --svo1 90 --svo2 0 --max-num-timesteps 100 --cap 25 --main-cap 15 --seed 1 --record
# The same with the particle filter on:
python main.py --num-agents 2 --level open-divider_salad --model1 bd --model2 bd \
    --svo1 90 --svo2 0 --max-num-timesteps 100 --cap 25 --main-cap 15 --seed 1 \
    --infer-svo --n-particles 64
```

Change `--svo2` to 45 or 90 for the other two partners. The authors' commit message `abdbd93` puts a salad run with planner caps of 20 and 12 and 12 particles at about 25 minutes. The filter's random number generator is not seeded (`delegation_planner/svo_particle_filter.py` line 75), and its estimate feeds back into the plan, so a run with `--infer-svo` does not repeat exactly even with the same `--seed`. `staff/trace_partner_estimates.py` shows this on two short runs.

## Terms and citation

The simulator, the layouts and the sprites are from [rosewang2008/gym-cooking](https://github.com/rosewang2008/gym-cooking), under the MIT License in `LICENSE` (copyright 2020 Rose E. Wang). If you use them, cite the paper:

> Wu, S. A., Wang, R. E., Evans, J. A., Tenenbaum, J. B., Parkes, D. C., and Kleiman-Weiner, M. (2021). Too many cooks: Bayesian inference for coordinating multi-agent collaboration. *Topics in Cognitive Science*, 13(2), 414–432. https://doi.org/10.1111/tops.12525

The citation block at the end of the write-up gives this paper the title of its earlier conference version, "Coordinating multi-agent collaboration through inverse planning"; the DOI is the same.

Two images in `images_svo/` are not covered by that license. `image.png`, the headline picture of the write-up, is a screenshot of the Overcooked video game by Ghost Town Games and Team17. `Wu.jpg` is a figure reproduced from Wu et al. (2021). The rights to both stay with their owners.
