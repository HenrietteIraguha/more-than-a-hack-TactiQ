"""Check our answer file without using any answer key.

The public five minutes may only be used to check that the program works, not to choose between
versions (RULES.md). So this script looks at our output for the WHOLE match in two other ways:

  1. Does it look like real football?
     How much of the match is dead ball, how long stoppages and possessions last, how many very
     short possessions there are (a sign of flicker).

  2. Does it agree with the Smart Tagging?
     Smart Tagging is produced by Metrica's algorithm, not by a person, so it is an allowed input
     and an independent second opinion. Every set piece (throw-in, corner, goal kick, free kick,
     kick-off) is a restart, so there should be a stoppage just before it, and the team taking it
     should be the team our file gives the ball to when play restarts.
     Smart Tagging has its own mistakes, so disagreements are things to look into, not proof.

    python starter/check_output.py                                   # checks our_team.csv
    python starter/check_output.py --file evaluation/results/x.csv   # any other file
    python starter/check_output.py --list                            # also list each problem

The summary only contains totals and percentages. The --list output has times in the match:
keep it on your own screen, do not paste it into an online AI (terms of use).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, half_video_blocks  # noqa: E402
import load_smart_tagging  # noqa: E402

FPS = 25
DEFAULT_FILE = os.path.join(ROOT, "evaluation", "results", "our_team.csv")

# Kickoff frame of each half (published in DATA.md and FAQ.md). Only used to print match times.
KICKOFF = {1: 11185, 2: 113742}

# A set piece counts as "covered" if our file has dead ball somewhere from 5 s before the tag
# starts until 1 s after it ends. Tags are windows around the restart, opened a few seconds early.
BEFORE_TAG_FRAMES = 125
AFTER_TAG_FRAMES = 25
# A stoppage counts as "explained" if a set-piece tag starts within 10 s of it.
EXPLAIN_FRAMES = 250


def runs(values, wanted):
    """[(start, stop)] of consecutive frames whose label is in `wanted`, stop exclusive."""
    mask = np.isin(values, list(wanted)).astype(np.int8)
    edges = np.diff(np.concatenate(([0], mask, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def clock(frame, half):
    """Match clock as m:ss, second half counted from 45:00."""
    seconds = (frame - KICKOFF[half]) / float(FPS) + (45 * 60 if half == 2 else 0)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    return "{}{}:{:04.1f}".format(sign, int(seconds // 60), seconds % 60)


def describe(lengths_frames):
    """min / median / max in seconds, as a short string."""
    if not len(lengths_frames):
        return "none"
    seconds = np.array(lengths_frames) / float(FPS)
    return "min {:.1f} s, median {:.1f} s, max {:.1f} s".format(
        seconds.min(), np.median(seconds), seconds.max())


def side_of(team_tag):
    """Smart Tagging team name -> H / A / None."""
    name = (team_tag or "").strip().lower()
    if "barcelona" in name:
        return "H"
    if "atletic" in name or "athletic" in name:
        return "A"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default=DEFAULT_FILE, help="answer file to check")
    parser.add_argument("--list", action="store_true", help="also list each problem with its time")
    args = parser.parse_args()

    labels = pd.read_csv(args.file, dtype={"label": str})["label"].str.strip().str.upper().values
    print("checking {}\n".format(os.path.relpath(args.file, ROOT)))

    halves = half_video_blocks(GAME)
    set_pieces = [event for event in load_smart_tagging.load()
                  if (event["code"] or "").strip().lower() == "set pieces"]

    totals = {"H": 0, "A": 0, "D": 0, "X": 0}
    stoppages, possessions = [], []
    covered_by_type, missed, unexplained = {}, [], []
    team_agree, team_disagree, team_unknown = 0, 0, 0
    team_problems = []

    for half, first, last in halves:
        part = labels[first:last + 1]
        for key in totals:
            totals[key] += int(np.sum(part == key))

        dead = [(first + s, first + e) for s, e in runs(part, {"D"})]
        stoppages += [(s, e, half) for s, e in dead]
        for team in ("H", "A"):
            possessions += [e - s for s, e in runs(part, {team})]

        tags = [event for event in set_pieces if first <= event["start_frame"] <= last]

        # 1. Is there a stoppage just before each set piece?
        for event in tags:
            kind = (event["tags"].get("Type") or "unknown").strip()
            low = max(first, event["start_frame"] - BEFORE_TAG_FRAMES)
            high = min(last + 1, event["end_frame"] + AFTER_TAG_FRAMES)
            hit = bool(np.any(labels[low:high] == "D"))
            counts = covered_by_type.setdefault(kind, [0, 0])
            counts[0] += 1
            counts[1] += int(hit)
            if not hit:
                missed.append((event["start_frame"], half, kind))
                continue

            # 2. Does the restart go to the team taking the set piece?
            expected = side_of(event["tags"].get("Team"))
            if expected is None:
                team_unknown += 1
                continue
            dead_frames = np.flatnonzero(labels[low:high] == "D")
            restart = low + int(dead_frames[-1]) + 1
            after = labels[restart:min(last + 1, restart + 5 * FPS)]
            alive = after[np.isin(after, ["H", "A"])]
            if not len(alive):
                team_unknown += 1
            elif alive[0] == expected:
                team_agree += 1
            else:
                team_disagree += 1
                team_problems.append((restart, half, kind, expected, alive[0]))

        # 3. Stoppages with no set piece anywhere near them.
        starts = np.array([event["start_frame"] for event in tags]) if tags else np.array([])
        for s, e in dead:
            near = len(starts) and np.any((starts >= s - EXPLAIN_FRAMES) & (starts <= e + EXPLAIN_FRAMES))
            if not near:
                unexplained.append((s, e, half))

    graded = totals["H"] + totals["A"] + totals["D"] + totals["X"]
    alive = totals["H"] + totals["A"]
    possessions = np.array(possessions)

    print("1. Does it look like real football?  (whole match, both halves)")
    print("   ball in play            {:5.1f}% of scored frames   (a top-level match is roughly 50-60%)"
          .format(100.0 * alive / max(graded, 1)))
    print("   dead ball               {:5.1f}%".format(100.0 * totals["D"] / max(graded, 1)))
    if totals["X"]:
        print("   no answer (X)           {:5.1f}%".format(100.0 * totals["X"] / max(graded, 1)))
    print("   Barcelona share of live {:5.1f}%".format(100.0 * totals["H"] / max(alive, 1)))
    print("   stoppages               {:>5}   ({})".format(
        len(stoppages), describe([e - s for s, e, _ in stoppages])))
    print("   possessions             {:>5}   ({})".format(len(possessions), describe(possessions)))
    if len(possessions):
        print("   possessions under 1 s   {:>5}   ({:.0f}%)   under 3 s {:>5}   ({:.0f}%)".format(
            int(np.sum(possessions < FPS)), 100.0 * np.mean(possessions < FPS),
            int(np.sum(possessions < 3 * FPS)), 100.0 * np.mean(possessions < 3 * FPS)))

    print("\n2. Does it agree with the Smart Tagging?")
    n_tags = sum(counts[0] for counts in covered_by_type.values())
    n_hit = sum(counts[1] for counts in covered_by_type.values())
    print("   set pieces with a stoppage just before them: {} of {}".format(n_hit, n_tags))
    for kind in sorted(covered_by_type):
        total, hit = covered_by_type[kind]
        print("      {:<24} {:>3} of {:>3}".format(kind, hit, total))
    print("   stoppages with no set piece within 10 s:     {} of {}".format(
        len(unexplained), len(stoppages)))
    print("      (some will be fouls, offsides or injuries the tagging missed; many = false stoppages)")
    print("   restart given to the team taking the set piece: {} agree, {} disagree, "
          "{} can't tell (no team tag)".format(team_agree, team_disagree, team_unknown))

    if args.list:
        print("\n--- details: keep these on your screen only ---")
        print("set pieces with no stoppage before them:")
        for frame, half, kind in missed:
            print("   {}  (frame {})  {}".format(clock(frame, half), frame, kind))
        print("stoppages with no set piece near them:")
        for s, e, half in unexplained:
            print("   {} to {}  ({:.1f} s, frames {}-{})".format(
                clock(s, half), clock(e, half), (e - s) / float(FPS), s, e))
        print("restarts given to the other team:")
        for frame, half, kind, expected, got in team_problems:
            print("   {}  (frame {})  {}: tagging says {}, our file says {}".format(
                clock(frame, half), frame, kind, expected, got))


if __name__ == "__main__":
    main()