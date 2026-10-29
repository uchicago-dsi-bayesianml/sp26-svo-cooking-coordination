#!/usr/bin/env python3
"""Course-staff check: episode lengths in the README's three behaviour GIFs.

The authors' misc/metrics/make_gif.py writes one frame per recorded time step,
plus the starting frame, at a fixed duration, and Pillow merges consecutive
identical frames into one longer frame when it saves. So the number of steps is
the total duration divided by the per-frame duration, minus one. The write-up
reports 45, 52 and 41 steps for the selfish, prosocial and altruistic partner.

Needs Pillow:  python staff/check_gif_steps.py [--ms 250]
"""
import argparse
import pathlib

from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
GIFS = ["svo0_partner.gif", "svo45_partner.gif", "svo90_partner.gif"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ms", type=int, default=250,
                    help="per-frame duration used by make_gif (the README's command uses 250)")
    args = ap.parse_args()
    for name in GIFS:
        im = Image.open(REPO / "images_svo" / name)
        durations = []
        try:
            while True:
                durations.append(im.info.get("duration", 0))
                im.seek(im.tell() + 1)
        except EOFError:
            pass
        total = sum(durations)
        print(f"{name}: {len(durations)} frames, {total} ms in all, "
              f"{total // args.ms - 1} time steps")


if __name__ == "__main__":
    main()
