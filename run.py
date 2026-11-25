#!/usr/bin/env python
"""Ball possession pipeline — single entry point.

Examples
  python run.py                                   # whole match, all stages, stride 1
  python run.py --start 11185 --end 26184         # only this frame range
  python run.py --stage detect                    # only step 1 (resumable)
  python run.py --stage teams,ball,possession,report,eval
  python run.py --demo 11185:14935                # render visualisation for a frame range
"""
import argparse
import functools
import sys
import time

from src import config as C

STAGES = ["detect", "teams", "ball", "possession", "report", "eval"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--start", type=int, default=0, help="first frame (0-based, inclusive)")
    ap.add_argument("--end", type=int, default=C.N_FRAMES - 1, help="last frame (inclusive)")
    ap.add_argument("--stride", type=int, default=1, help="detect every N-th frame (1..5)")
    ap.add_argument("--stage", default="all",
                    help="comma list of " + ",".join(STAGES) + " or 'all'")
    ap.add_argument("--demo", default=None,
                    help="render demo video for START:END frames ('default' = first 2.5 min)")
    ap.add_argument("--method", default="hmm", choices=["hmm", "baseline"],
                    help="method written to out/possession.csv")
    a = ap.parse_args()
    if not 1 <= a.stride <= 5:
        ap.error("--stride must be in 1..5")
    log = functools.partial(print, flush=True)
    C.OUT.mkdir(exist_ok=True)
    stages = STAGES if a.stage == "all" else [s.strip() for s in a.stage.split(",") if s.strip()]
    if a.demo and a.stage == "all" and "--stage" not in " ".join(sys.argv):
        stages = []   # --demo alone: just render
    t0 = time.time()
    for s in stages:
        log(f"=== stage {s} ===")
        if s == "detect":
            from src import detect
            detect.run(a.start, a.end, a.stride, log)
        elif s == "teams":
            from src import teams
            teams.run(a.start, a.end, log)
        elif s == "ball":
            from src import ball
            ball.run(a.start, a.end, log)
        elif s == "possession":
            from src import possession
            possession.run(a.start, a.end, log)
        elif s == "report":
            from src import report
            report.run(a.start, a.end, a.method, log)
        elif s == "eval":
            from src import evaluate
            evaluate.run(a.start, a.end, log)
        else:
            ap.error(f"unknown stage {s}")
    if a.demo:
        from src import demo
        if a.demo == "default":
            s, e = C.DEMO_RANGE
        else:
            s, e = map(int, a.demo.split(":"))
        demo.render(s, e, log=log)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
