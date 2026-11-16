"""Metrica open sample games -> the same tracking dict the pipeline uses, plus possession labels.

Labels are derived mechanically from Metrica's human event data for those (other) games, using
the same three-state definition as the challenge:

  * dead ball   from a stoppage trigger until the next restart:
                  BALL OUT (end frame: the ball crosses the line), FAULT RECEIVED (start frame),
                  a SHOT ending in GOAL or OUT (end frame), BALL LOST by OFFSIDE / HAND BALL /
                  END HALF (end frame);
                restarts are SET PIECE events and goal kicks (PASS / BALL LOST "GOAL KICK...").
  * possession  the team of the latest on-ball event (PASS, RECOVERY, SET PIECE, SHOT, BALL LOST);
                CHALLENGE, CARD and FAULT RECEIVED do not move it -- a loose ball stays with the
                last team that had it, as CHALLENGE.md specifies.

The sample tracking has no speed channel, so speeds are computed from positions. Its ball is
missing on ~40% of frames, close to BAR-ATH's 41%, so no extra dropout is simulated.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "external_data", "metrica-sample-data")
LENGTH, WIDTH, FPS = 105.0, 68.0, 25
GAMES = (1, 2)

ON_BALL = {"PASS", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}


def _read_tracking(path):
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    players = [c for c in cols if c.startswith("Player")]
    xy = {}
    for p in players:
        i = cols.index(p)
        xy[p] = (raw.iloc[:, i].to_numpy(float), raw.iloc[:, i + 1].to_numpy(float))
    bi = cols.index("Ball")
    ball = (raw.iloc[:, bi].to_numpy(float), raw.iloc[:, bi + 1].to_numpy(float))
    return raw["Period"].to_numpy(), raw["Frame"].to_numpy(), xy, ball


def _speed(x, y):
    """m/s from positions, smoothed over 5 frames (the ATD speed channel is similarly smooth)."""
    step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) * FPS
    return pd.Series(step).rolling(5, center=True, min_periods=1).mean().to_numpy()


def load_game(game):
    folder = os.path.join(DATA, "Sample_Game_{}".format(game))
    stem = os.path.join(folder, "Sample_Game_{}_RawTrackingData_".format(game))
    period, frame, home, ball = _read_tracking(stem + "Home_Team.csv")
    _, frame_a, away, _ = _read_tracking(stem + "Away_Team.csv")
    if not np.array_equal(frame, frame_a):
        raise ValueError("home/away tracking frames differ")

    slots = [(p, 0) for p in home] + [(p, 1) for p in away]
    n = len(frame)
    px = np.full((n, len(slots)), np.nan, np.float32)
    py = np.full_like(px, np.nan)
    ps = np.full_like(px, np.nan)
    for j, (p, team) in enumerate(slots):
        x, y = (home if team == 0 else away)[p]
        px[:, j], py[:, j] = x * LENGTH, y * WIDTH
        ps[:, j] = _speed(px[:, j], py[:, j])
    trk = {"px": px, "py": py, "ps": ps,
           "bx": (ball[0] * LENGTH).astype(np.float32), "by": (ball[1] * WIDTH).astype(np.float32),
           "team": np.array([t for _, t in slots], np.int8), "n_frames": n}
    blocks = []
    for half in (1, 2):
        idx = np.where(period == half)[0]
        blocks.append((half, int(idx[0]), int(idx[-1])))
    events = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(game)))
    # Frames in the CSVs count from 1; arrays here count from 0.
    events["Start Frame"] -= 1
    events["End Frame"] -= 1
    return trk, blocks, events


def labels_from_events(events, blocks, n):
    """H/A/D per frame from the human event data of a sample game; X outside the halves."""
    lab = np.full(n, "X", dtype="<U1")
    ev = events.sort_values(["Start Frame"], kind="stable")
    for half, first, last in blocks:
        e = ev[ev.Period == half]
        state = "D"  # before the kickoff
        cursor = first
        marks = []   # (frame, new_state)
        for r in e.itertuples(index=False):
            typ = str(r.Type).upper()
            sub = str(r.Subtype).upper() if isinstance(r.Subtype, str) else ""
            team = "H" if r.Team == "Home" else "A"
            start, end = int(r[4]), int(r[6])
            restart = typ == "SET PIECE" or "GOAL KICK" in sub
            if restart or typ in ON_BALL:
                marks.append((start, team, restart))
            if typ == "BALL OUT":
                marks.append((end, "D", False))
            elif typ == "FAULT RECEIVED":
                marks.append((start, "D", False))
            elif typ == "SHOT" and ("GOAL" in sub or sub.endswith("OUT")):
                marks.append((max(end, start), "D", False))
            elif typ == "BALL LOST" and any(k in sub for k in ("OFFSIDE", "HAND BALL", "END HALF")):
                marks.append((max(end, start), "D", False))
        marks.sort(key=lambda m: m[0])
        for frame, new, restart in marks:
            frame = min(max(frame, first), last + 1)
            lab[cursor:frame] = state
            cursor = frame
            if new == "D":
                state = "D"
            elif state != "D" or restart:
                state = new          # possession changes; only a restart revives a dead ball
        lab[cursor:last + 1] = state
    return lab
