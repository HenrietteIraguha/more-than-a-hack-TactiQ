"""Our solution, version 2: who has the ball, frame by frame.

How we got here
---------------
We started from the organisers' baseline (starter/baseline.py): give the ball to whoever is
nearest to it, then smooth with a majority vote. Before changing anything we studied the first
five minutes of the match three ways at once: the tactical video, the tracking data in
notebook 02 (with our own helper tools for distances, ball speed and player speed), and the
public answer key. We wrote down what happened at every change of possession and every
stoppage, and compared it with what the data showed. Each rule below comes from something we
saw there. The notes say which moment taught us what.

What we learned, in short
-------------------------
* "Nearest player has the ball" is often wrong. At the 0:07 interception the Barcelona player
  who won the ball was never the nearest dot; an Athletic player stayed closer the whole time.
  During passes the ball rolls past opponents who never control it, and in tight duels a
  defender pressing the carrier is just as close as the carrier.
* The ball tells you when someone touches it. A pass travels fast and straight; a touch makes
  it suddenly slow down or change direction. That showed the 0:07 interception clearly when
  the player distances did not.
* Stoppages show up in the players, not the ball. Offsides and fouls stop play with the ball
  still on the pitch, and the ball often disappears from the tracking during a stoppage. What
  was there every time was the players: within about a second of the whistle almost everyone
  slows to a walk, and that is when the answer key starts its dead ball.
* The answer key changes possession at the first clear touch by the other team: an
  interception, a header won, a block. A player receiving a ball that is already rolling to him
  only counts once he plays it on.
* The tracking has fake dots: at least one track stands perfectly still beside Barcelona's goal,
  in the same spot before kickoff and during play. It must not count as a player.

What this version does
----------------------
It keeps everything in the baseline that already works (reading the tracking file, knowing
each track's team, writing the answer file) and replaces the "nearest player + vote" logic:

  1. Drop tracks that do not move.               (the fake dots by the goal)
  2. Dead ball = most players walking.           (the stoppages at 1:07, 3:08, 4:08, 4:34)
  3. Restart = the team of the player nearest the ball when it is played again (opponents must
     stand back at set pieces, so that player is the taker).
                                                 (the offside at 1:07, the ball given back at 3:28)
  4. Sticky possession: keep the current team until the other team clearly takes over.
                                                 (0:07, 0:15, 0:18, 3:03, 4:04)

Version 2 (13 November) brings in the best part of our teammate's baseline_plus.py:
  * possession inside each stretch of live play is decided by a Viterbi decoder, where every
    change of team has a cost, instead of version 1's frame-by-frame streak rule. Our whole-match
    check (check_output.py) showed version 1 still made about a third of possessions shorter
    than 3 seconds.
  * the ball being clearly off the pitch is an extra dead-ball signal.
We did not take two of its ideas: forcing a fixed block of frames to D (rules tied to specific
frames of this match are not allowed) and merging every possession shorter than 2 seconds (the
real match has possessions of about a second, like Barcelona's header at 3:03).

    pip install -r requirements.txt          # numpy and pandas, nothing else
    python starter/our_solution.py           # writes evaluation/results/our_team.csv
    python evaluation/run_evaluation.py      # score it

Every number we might want to tune is in the PARAMETERS block below, and the most important
ones can also be changed from the command line without editing this file, for example:

    python starter/our_solution.py --dead-fraction 0.7 --name our_team_df70

The public five minutes are used only to check that the program works and to understand what
the labels mean (as CHALLENGE.md suggests). Settings are chosen from how football works and
from what we saw, not from scores on those five minutes, which RULES.md does not allow. No hand
labels on this match are used anywhere in the program.
"""

import argparse
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import HOME, AWAY, team_names, team_of_each_track, write_submission  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
FPS = 25
NOBODY = -1

# ----------------------------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------------------------

# Rule 1: fake dots.
# At kickoff we counted about 10 dots per team, and one "Athletic" dot was standing inside
# Barcelona's six-yard box. It was in exactly the same place minutes before kickoff and did not
# move during the first seconds of play either. Real players never stand still that long, so a
# track that stays inside a 1 m box for a whole 25-second block is treated as an object, not a
# player, for that block. Otherwise it could be picked as "nearest to the ball" whenever play
# reaches that goal.
STATIC_BLOCK_FRAMES = 625          # 25 seconds
STATIC_MAX_SPAN_M = 1.0

# The ball is missing from the tracking on about 4 frames in 10 over the match (far more than in
# the first five minutes). Gaps shorter than this are filled with a straight line between the two
# known positions; longer gaps stay empty and the current team simply keeps the ball.
BALL_GAP_FILL_FRAMES = 12          # about half a second

# Speeds are measured over +/- this many frames.
SPEED_LAG_FRAMES = 2

# Rule 2: dead ball.
# In every stoppage we watched, players eased off to a walk within about a second of the whistle,
# and the answer key's dead ball started right then. In calm passing play a quarter to a third of
# the players are walking too, so the bar has to be high to avoid calling a slow build-up a
# stoppage.
WALK_SPEED_MS = 2.0                # slower than this counts as walking or standing
DEAD_FRACTION = 0.75               # this share of players walking means play has stopped
DEAD_SMOOTH_FRAMES = 25            # average the walking share over 1 second
MIN_PLAYERS_FOR_DEAD = 8           # need at least this many tracked players to decide
# A real stoppage takes time: the ball has to be fetched, placed or thrown back in before play
# restarts, so throw-ins, free kicks, goal kicks and corners take several seconds at least.
# A short spell where most players happen to be walking is far more likely to be calm passing,
# and calling it a stoppage would cut one possession into two. So a "stoppage" shorter than
# 5 seconds is treated as play.
MIN_DEAD_FRAMES = 125              # 5 seconds
# From 3:09 to 3:28 the ball moved around while play was still dead: Barcelona's goalkeeper threw
# it out and a Barcelona player handed it back to Athletic. A short burst of movement inside a
# stoppage is therefore not a restart.
MIN_ALIVE_GAP_FRAMES = 25

# Rule 3: restart.
# Who restarts can't be guessed from "who touched it last". After the offside at 1:07 the
# Athletic player was still standing by the ball but Barcelona took the free kick, and at 3:28
# Athletic restarted only because Barcelona gave the ball back.
#
# Version 1 gave the restart to the team most often near the ball in the first seconds of play.
# Our check against the Smart Tagging showed that was right only about half the time: after a
# goal kick or a corner the ball lands in a crowd, and the other team is often closer to it.
#
# The laws of the game give a better rule. At a free kick, corner or goal kick, opponents must
# stand at least 9.15 m from the ball, and at a throw-in at least 2 m. So at the moment the ball
# is played, the player nearest to it is the taker. We find the first moment the ball starts
# moving around the restart, and give the ball to the team of the nearest player just before.
RESTART_LOOKBACK_FRAMES = 50       # the stoppage often ends a little late, so start 2 s earlier
RESTART_SEARCH_FRAMES = 75         # and look up to 3 seconds into play
KICK_SPEED_MS = 3.0                # the ball moving faster than this = it has been played
STILL_SPEED_MS = 2.0               # ...and it was slower than this just before (a set ball)
KICK_RADIUS_M = 4.0                # the taker must be at least this close to the ball

# Rule 4: possession.
# "Near the ball" means within NEAR_RADIUS_M and clearly closer than the other team. The other
# team only takes over if it stays near the ball for about half a second, which removes most of
# the flickering that made the raw baseline invent many times more possessions than the match had.
NEAR_RADIUS_M = 2.5
NEAR_MARGIN_M = 0.5
SWITCH_FRAMES = 12                 # about half a second

# Touches. Before the 0:07 interception the ball travelled fast and almost perfectly straight;
# at the interception it slowed down sharply and turned. That is how a touch looks in the data,
# and it is also when the answer key switches: Barcelona's header at 3:03 and the pass cut out
# between two Barcelona players at 4:04 both counted as Barcelona possession straight away.
TOUCH_TURN_DEG = 40.0              # the ball turning more than this means someone touched it
TOUCH_SPEED_CHANGE_MS = 4.0        # ...or its speed jumping or dropping by more than this
TOUCH_WINDOW_FRAMES = 3            # compare the ball's movement 3 frames before and 3 after
TOUCH_RADIUS_M = 3.0               # the touch goes to a team whose player is this close
# At 0:08 the Barcelona player who intercepted had two Athletic players pressed right against
# him, moving the same way. A pressing defender is as close as the carrier, so a touch only
# switches possession if the touching team then stays near the ball for a few frames.
TOUCH_CONFIRM_WINDOW = 12
TOUCH_CONFIRM_FRAMES = 3
MIN_MOVE_M = 0.15                  # ignore direction changes when the ball barely moves

# Version 2: possession decoder (adapted from our teammate's baseline_plus.py).
# Version 1's frame-by-frame "streak" rule still left about a third of all possessions shorter
# than 3 seconds in our whole-match check: it switched teams too easily in duels and passes.
# A Viterbi decoder looks at the whole stretch of play at once and picks the most likely
# sequence of H and A, where every change of team has a cost. A handover then needs evidence
# that holds up over time, and frames with no evidence (ball missing, nobody close) simply keep
# the current team, which is the challenge's own rule for loose balls.
DECODER = "viterbi"                # "viterbi" (version 2) or "streak" (version 1)
CONTROL_RADIUS_M = 4.0             # only count the nearest player as evidence within 4 m
EVIDENCE_SCALE_M = 2.0             # evidence grows with how much closer that team is, 2 m = 1 unit
EVIDENCE_CAP = 3.0                 # and is capped, so one odd frame can't decide on its own
# A team that is clearly closer (about 6 m) earns 3 units a frame, so a change of possession needs
# about half a second of clear evidence, or around a second of weaker evidence. That is long
# enough to ignore a defender pressing the carrier, and short enough to keep the real one-second
# possessions we saw (Barcelona's header at 3:03).
SWITCH_COST = 40.0

# Ball off the pitch. By the laws of the game the ball is dead once it has wholly crossed a line,
# so this is an extra dead-ball signal next to the players walking. 1 m of margin allows for the
# tracking being slightly off near the lines.
OUT_MARGIN_M = 1.0


# ----------------------------------------------------------------------------------------------
# Step 1: read the tracking file into arrays (positions in metres, NaN where missing)
# ----------------------------------------------------------------------------------------------

def load_tracking(raw_path, n_frames, n_tracks, pitch):
    """Return ball (n, 2) and players (n, tracks, 2), in metres, NaN where missing.

    Same file format the baseline reads: "<frame_id>:<player>;<player>;...:<ball>", each group
    "x,y,speed". frame_id counts from 1, our rows from 0, so row = frame_id - 1.
    """
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_tracks, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue

            ball_fields = parts[2].split(",")
            if len(ball_fields) >= 2 and not ball_fields[0].startswith("N"):
                try:
                    ball[row, 0] = float(ball_fields[0]) * length
                    ball[row, 1] = float(ball_fields[1]) * width
                except ValueError:
                    pass

            groups = parts[1].split(";")
            values = parts[1].replace(";", ",").split(",")
            if len(values) == 3 * len(groups):
                # Fast path: every group has exactly x,y,speed.
                block = np.array(values, dtype=np.float64).reshape(-1, 3)[:n_tracks, :2]
                players[row, :len(block), 0] = block[:, 0] * length
                players[row, :len(block), 1] = block[:, 1] * width
            else:
                for slot, group in enumerate(groups[:n_tracks]):
                    fields = group.split(",")
                    if len(fields) < 2 or fields[0].startswith("N"):
                        continue
                    try:
                        players[row, slot, 0] = float(fields[0]) * length
                        players[row, slot, 1] = float(fields[1]) * width
                    except ValueError:
                        pass
    return ball, players


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def runs(mask):
    """[(start, stop)] of consecutive True values, stop exclusive."""
    mask = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], mask, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def rolling_mean(values, window):
    """Centred moving average that ignores NaN. Output NaN where the window has no values."""
    ok = ~np.isnan(values)
    filled = np.where(ok, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccount = np.concatenate(([0], np.cumsum(ok)))
    half = window // 2
    index = np.arange(len(values))
    low = np.maximum(0, index - half)
    high = np.minimum(len(values), index + half + 1)
    count = ccount[high] - ccount[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[high] - csum[low]) / np.maximum(count, 1), np.nan)


def velocity(positions, lag):
    """Centred velocity in m/s: (p[t+lag] - p[t-lag]) / time. Works for (n, 2) or (n, k, 2)."""
    out = np.full(positions.shape, np.nan, dtype=np.float32)
    if len(positions) > 2 * lag:
        out[lag:-lag] = (positions[2 * lag:] - positions[:-2 * lag]) / (2.0 * lag / FPS)
    return out


def fill_short_gaps(series, max_gap):
    """Fill gaps of at most max_gap frames with a straight line. Longer gaps stay NaN."""
    out = series.copy()
    missing = np.isnan(series[:, 0])
    known = np.flatnonzero(~missing)
    if len(known) < 2:
        return out
    index = np.arange(len(series))
    for axis in range(series.shape[1]):
        out[:, axis] = np.interp(index, known, series[known, axis])
    for start, stop in runs(missing):
        at_edge = start == 0 or stop == len(series)
        if at_edge or stop - start > max_gap:
            out[start:stop] = np.nan
    return out


# ----------------------------------------------------------------------------------------------
# Rule 1: remove fake dots that never move
# ----------------------------------------------------------------------------------------------

def remove_static_tracks(players):
    """Copy of players with tracks blanked out on 25-second blocks where they did not move.

    These are the fake dots we found beside Barcelona's goal: tracked like players, but standing
    in exactly the same spot before kickoff and during play.
    """
    n, n_tracks, _ = players.shape
    block = STATIC_BLOCK_FRAMES
    n_blocks = int(math.ceil(n / float(block)))
    padded = np.full((n_blocks * block, n_tracks, 2), np.nan, dtype=np.float32)
    padded[:n] = players
    shaped = padded.reshape(n_blocks, block, n_tracks, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        span = np.nanmax(shaped, axis=1) - np.nanmin(shaped, axis=1)   # (blocks, tracks, 2)
        span = np.nanmax(span, axis=2)                                  # (blocks, tracks)
    seen = np.sum(~np.isnan(shaped[..., 0]), axis=1)                    # (blocks, tracks)
    still = (span < STATIC_MAX_SPAN_M) & (seen >= block // 2)

    mask = np.repeat(still, block, axis=0)[:n]                          # (n, tracks)
    cleaned = players.copy()
    cleaned[mask] = np.nan
    return cleaned, int(still.sum())


# ----------------------------------------------------------------------------------------------
# Rule 2: dead ball from the share of players walking
# ----------------------------------------------------------------------------------------------

def dead_ball_mask(players, off_pitch=None):
    """True on frames where play looks stopped: most players walking, smoothed over 1 s.

    We tried the ball first, but it was not reliable for stoppages: at the offside at 1:07 it
    stayed on the pitch and then went missing, and while it was being handed back at 3:09 it
    was moving even though play was dead. The players were the clear signal every time.
    """
    player_velocity = velocity(players, SPEED_LAG_FRAMES)
    speed = np.hypot(player_velocity[..., 0], player_velocity[..., 1])  # (n, tracks)
    measured = ~np.isnan(speed)
    walking = np.sum(measured & (speed < WALK_SPEED_MS), axis=1)
    counted = np.sum(measured, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(counted >= MIN_PLAYERS_FOR_DEAD, walking / np.maximum(counted, 1), np.nan)
    share = rolling_mean(share, DEAD_SMOOTH_FRAMES)

    dead = share >= DEAD_FRACTION
    if off_pitch is not None:
        # Version 2: the ball being clearly off the pitch also means the ball is dead.
        dead = dead | off_pitch
    # Short bursts of movement inside a stoppage are not a restart.
    for start, stop in runs(~dead):
        if start > 0 and stop < len(dead) and stop - start < MIN_ALIVE_GAP_FRAMES:
            dead[start:stop] = True
    # Very short "stoppages" are just a quiet moment in play.
    for start, stop in runs(dead):
        if stop - start < MIN_DEAD_FRAMES:
            dead[start:stop] = False
    return dead, share


# ----------------------------------------------------------------------------------------------
# Rule 4 (part 1): which team is near the ball, frame by frame
# ----------------------------------------------------------------------------------------------

def nearest_by_team(ball, players, team_of_track):
    """Distance from the ball to the nearest home player and the nearest away player."""
    diff = players - ball[:, None, :]
    distance = np.hypot(diff[..., 0], diff[..., 1])                     # (n, tracks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        home = np.nanmin(np.where(team_of_track[None, :] == HOME, distance, np.nan), axis=1)
        away = np.nanmin(np.where(team_of_track[None, :] == AWAY, distance, np.nan), axis=1)
    return home, away


def team_near(home_dist, away_dist, radius, margin):
    """HOME / AWAY where one team is within `radius` and `margin` closer than the other."""
    home = np.where(np.isnan(home_dist), np.inf, home_dist)
    away = np.where(np.isnan(away_dist), np.inf, away_dist)
    out = np.full(len(home), NOBODY, dtype=np.int8)
    out[(home <= radius) & (home + margin <= away)] = HOME
    out[(away <= radius) & (away + margin <= home)] = AWAY
    return out


# ----------------------------------------------------------------------------------------------
# Rule 4 (part 2): touches, from the ball suddenly turning or changing speed
# ----------------------------------------------------------------------------------------------

def touch_frames(ball):
    """Frames where the ball suddenly turns or changes speed: someone probably touched it.

    A pass in flight is fast and straight. When someone stops it, heads it or redirects it,
    the ball slows down or turns within a few frames. This is what showed us the 0:07
    interception when the distances to the players did not.
    """
    k = TOUCH_WINDOW_FRAMES
    n = len(ball)
    candidate = np.zeros(n, dtype=bool)
    if n <= 2 * k:
        return []

    before = ball[k:-k] - ball[:-2 * k]           # movement over the k frames before t
    after = ball[2 * k:] - ball[k:-k]             # movement over the k frames after t
    len_before = np.hypot(before[:, 0], before[:, 1])
    len_after = np.hypot(after[:, 0], after[:, 1])

    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (before[:, 0] * after[:, 0] + before[:, 1] * after[:, 1]) / (len_before * len_after)
        turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        speed_change = np.abs(len_after - len_before) / (k / float(FPS))

    moving = (len_before > MIN_MOVE_M) & (len_after > MIN_MOVE_M)
    hit = (moving & (turn > TOUCH_TURN_DEG)) | (speed_change > TOUCH_SPEED_CHANGE_MS)
    candidate[k:-k] = np.nan_to_num(hit, nan=0).astype(bool)

    # One touch per burst of candidate frames: keep the middle frame.
    return [(start + stop - 1) // 2 for start, stop in runs(candidate)]


# ----------------------------------------------------------------------------------------------
# Putting it together for one half
# ----------------------------------------------------------------------------------------------

def ball_off_pitch(ball, pitch):
    """True where the tracked ball is clearly outside the pitch lines."""
    length, width = pitch
    x, y, m = ball[:, 0], ball[:, 1], OUT_MARGIN_M
    with np.errstate(invalid="ignore"):
        return (x < -m) | (x > length + m) | (y < -m) | (y > width + m)


def possession_evidence(ball, home_dist, away_dist):
    """Per frame: positive favours home, negative favours away, 0 = no evidence.

    Only counts when the ball is seen and someone is within CONTROL_RADIUS_M, and grows with how
    much closer one team is than the other. A loose or contested ball gives no evidence.
    """
    known = ~np.isnan(ball[:, 0])
    nearest = np.fmin(home_dist, away_dist)
    diff = away_dist - home_dist
    with np.errstate(invalid="ignore"):
        usable = known & ~np.isnan(diff) & (nearest <= CONTROL_RADIUS_M)
    return np.clip(np.where(usable, diff / EVIDENCE_SCALE_M, 0.0), -EVIDENCE_CAP, EVIDENCE_CAP)


def viterbi_two_teams(evidence, start_team):
    """Most likely H/A sequence for one stretch of live play.

    Each frame scores +e/2 for home and -e/2 for away; every change of team costs SWITCH_COST.
    Starting with the team that did not take the restart costs the same as one change.
    """
    n = len(evidence)
    gain = np.stack([evidence / 2.0, -evidence / 2.0], axis=1)        # (n, 2): home, away
    score = gain[0].copy()
    if start_team in (HOME, AWAY):
        score[1 - start_team] -= SWITCH_COST
    back = np.zeros((n, 2), dtype=np.int8)
    for t in range(1, n):
        stay, swap = score, score[::-1] - SWITCH_COST
        choose_swap = swap > stay
        back[t] = np.where(choose_swap, [1, 0], [0, 1])
        score = np.where(choose_swap, swap, stay) + gain[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path                                                         # 0 = home, 1 = away


def label_half(ball, players, team_of_track, pitch=None):
    """Labels H / A / D for every frame of one half (arrays already cut to that half).

    Each half is labelled on its own so nothing runs across halftime. The second half's scored
    frames start almost two minutes before its kickoff. Those frames are all dead ball, and the
    walking rule should catch them because everyone is standing in position waiting.
    """
    n = len(ball)
    labels = np.full(n, "D", dtype="<U1")

    players, n_static = remove_static_tracks(players)
    ball = fill_short_gaps(ball, BALL_GAP_FILL_FRAMES)
    dead, _ = dead_ball_mask(players, ball_off_pitch(ball, pitch) if pitch else None)

    home_dist, away_dist = nearest_by_team(ball, players, team_of_track)
    near = team_near(home_dist, away_dist, NEAR_RADIUS_M, NEAR_MARGIN_M)
    touch_team = team_near(home_dist, away_dist, TOUCH_RADIUS_M, NEAR_MARGIN_M)
    touches = set(touch_frames(ball))
    ball_v = velocity(ball, SPEED_LAG_FRAMES)
    ball_speed = np.hypot(ball_v[:, 0], ball_v[:, 1])
    evidence = possession_evidence(ball, home_dist, away_dist)

    current = NOBODY
    n_switches = 0
    n_restarts, n_by_kicker = 0, 0
    for start, stop in runs(~dead):
        n_restarts += 1
        # Rule 3: the restarting team is the team of the player nearest the ball when it is
        # played. Opponents have to stand back at set pieces, so that player is the taker.
        taker = restart_taker(start, stop, ball, ball_v, ball_speed, players, team_of_track)
        if taker != NOBODY:
            current = taker
            n_by_kicker += 1
            window, known = None, []
        else:
            # Could not see the kick (ball missing): fall back to the team most often near
            # the ball in the first seconds of play.
            window = near[start:min(stop, start + RESTART_SEARCH_FRAMES)]
            known = window[window != NOBODY]
        if len(known):
            current = HOME if np.sum(known == HOME) >= np.sum(known == AWAY) else AWAY
        elif current == NOBODY:
            # Nobody clearly near: fall back to whichever team is nearest at all.
            h, a = home_dist[start:stop], away_dist[start:stop]
            both = ~np.isnan(h) & ~np.isnan(a)
            if both.any():
                current = HOME if np.nanmean(h[both] - a[both]) <= 0 else AWAY
            else:
                current = HOME

        if DECODER == "viterbi":
            path = viterbi_two_teams(evidence[start:stop], current)
            labels[start:stop] = np.where(path == HOME, "H", "A")
            n_switches += int(np.sum(np.diff(path) != 0))
            current = int(path[-1])
            continue

        streak_team, streak_start, streak_len = NOBODY, None, 0
        frame = start
        while frame < stop:
            # A touch by the other team switches possession, as long as that team then stays
            # near the ball for a few frames. Without that check, a defender pressing the
            # carrier (like the two Athletic players around the Barcelona player at 0:08)
            # would steal possession on every dribble touch.
            if frame in touches and touch_team[frame] not in (NOBODY, current):
                toucher = int(touch_team[frame])
                after = near[frame:min(stop, frame + TOUCH_CONFIRM_WINDOW)]
                if np.sum(after == toucher) >= TOUCH_CONFIRM_FRAMES:
                    current = toucher
                    n_switches += 1
                    streak_team, streak_len = NOBODY, 0

            # Otherwise, the other team has to stay near the ball for SWITCH_FRAMES frames. A
            # near miss does not count: at about 0:16 a Barcelona player almost cut out Athletic's
            # long pass but fell, and the answer key stayed with Athletic.
            who = near[frame]
            if who != NOBODY and who != current:
                if who != streak_team:
                    streak_team, streak_start, streak_len = who, frame, 0
                streak_len += 1
                if streak_len >= SWITCH_FRAMES:
                    current = int(who)
                    n_switches += 1
                    # Back-date the switch to when the streak began.
                    labels[streak_start:frame] = "H" if current == HOME else "A"
                    streak_team, streak_len = NOBODY, 0
            elif who == current:
                streak_team, streak_len = NOBODY, 0
            # who == NOBODY (ball missing or nobody close): keep the current team and pause the
            # streak. When Barcelona's goalkeeper held the ball from 0:18 to 0:24, nothing was
            # moving, but play was live and it stayed Barcelona's ball.

            labels[frame] = "H" if current == HOME else "A"
            frame += 1

    return labels, {"static_blocks": n_static, "touches": len(touches), "switches": n_switches,
                    "dead_frames": int(dead.sum()), "restarts": n_restarts,
                    "restarts_by_kicker": n_by_kicker}


def restart_taker(start, stop, ball, ball_v, ball_speed, players, team_of_track):
    """Team of the player who restarts play, or NOBODY if the kick can't be found.

    Finds the moments around the restart where a ball that was still (or just placed) starts
    moving. During a stoppage the ball is often thrown or passed to the taker first (like the
    ball handed back at 3:09), and that also looks like "still, then moving". So we take the
    last such moment up to one second into play, which is the actual restart, and only look
    further into play if there was none.

    The ball is often missing from the tracking during a stoppage and only reappears once it
    is already moving. Then we work backwards: we take where it reappeared and its direction
    and speed, and step back along that line by the time it was out of sight (at most one
    second) to estimate where it was kicked from. The taker is the player nearest that spot.
    """
    low = max(1, start - RESTART_LOOKBACK_FRAMES)
    high = min(stop, start + RESTART_SEARCH_FRAMES)
    kicks = []
    for frame in range(low, high):
        if not ball_speed[frame] > KICK_SPEED_MS:
            continue
        before = ball_speed[max(0, frame - 5):frame]
        before = before[~np.isnan(before)]
        if len(before) and np.min(before) > STILL_SPEED_MS:
            continue  # already moving: this is play, not the moment of the kick
        kicks.append(frame)
    if not kicks:
        return NOBODY

    early = [frame for frame in kicks if frame <= start + FPS]
    kick = early[-1] if early else kicks[0]
    moment = max(0, kick - 3)
    radius = KICK_RADIUS_M

    if not np.isnan(ball[moment, 0]):
        # We can see the ball just before it was played.
        point, when = ball[moment], moment
    else:
        # The ball reappeared already moving: step back along its path.
        seen = np.flatnonzero(~np.isnan(ball[moment:kick + 1, 0]))
        if not len(seen):
            return NOBODY
        reappear = moment + int(seen[0])
        earlier = np.flatnonzero(~np.isnan(ball[max(0, reappear - 10 * FPS):reappear, 0]))
        hidden = reappear - (max(0, reappear - 10 * FPS) + int(earlier[-1])) if len(earlier) else FPS
        back = min(hidden, FPS)
        point = ball[reappear] - ball_v[kick] * (back / float(FPS))
        when = max(0, reappear - back)
        radius = KICK_RADIUS_M + 2.0  # the estimate is rougher

    at = players[max(0, when - 2)]
    distance = np.hypot(at[:, 0] - point[0], at[:, 1] - point[1])
    if np.all(np.isnan(distance)) or np.nanmin(distance) > radius:
        return NOBODY
    return HOME if team_of_track[int(np.nanargmin(distance))] == HOME else AWAY


def count_possessions(labels):
    """How many separate H or A runs the labels contain."""
    count = 0
    previous = None
    for label in labels:
        if label in ("H", "A") and label != previous:
            count += 1
        previous = label
    return count


def build(game=GAME, data_dir=None):
    """Read the data and return one label per video frame for the whole video."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team_of_track = team_of_each_track(meta_path)

    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))
    ball, players = load_tracking(raw_path, n_frames, len(team_of_track), GAMES[game]["pitch"])
    print("  ball located on {:.1f}% of frames\n".format(100.0 * np.mean(~np.isnan(ball[:, 0]))))

    # Frames outside the two halves are never scored: X there.
    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in half_video_blocks(game):
        stop = last + 1
        half_labels, stats = label_half(ball[first:stop], players[first:stop], team_of_track,
                                        GAMES[game]["pitch"])
        labels[first:stop] = half_labels
        print("half {}: H {:>6}  A {:>6}  D {:>6}   possessions {:>4}   touches {:>5}   "
              "fake-dot blocks removed {}   restarts {} (taker seen in {})".format(
                  half, int(np.sum(half_labels == "H")), int(np.sum(half_labels == "A")),
                  int(np.sum(half_labels == "D")), count_possessions(half_labels),
                  stats["touches"], stats["static_blocks"], stats["restarts"],
                  stats["restarts_by_kicker"]))
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="our_team", help="output file name, without .csv")
    parser.add_argument("--out", default=RESULTS_DIR, help="folder to write into")
    parser.add_argument("--dead-fraction", type=float, default=DEAD_FRACTION,
                        help="share of players walking that means a dead ball")
    parser.add_argument("--switch-frames", type=int, default=SWITCH_FRAMES,
                        help="frames the other team must stay near the ball to take over")
    parser.add_argument("--near-radius", type=float, default=NEAR_RADIUS_M,
                        help="metres from the ball that count as 'near'")
    parser.add_argument("--no-touches", action="store_true",
                        help="switch only on the 'stays near' rule, not on ball touches")
    parser.add_argument("--min-dead", type=int, default=MIN_DEAD_FRAMES,
                        help="shortest stoppage we believe, in frames (25 = 1 second)")
    parser.add_argument("--decoder", choices=["viterbi", "streak"], default=DECODER,
                        help="viterbi = version 2 (default), streak = version 1")
    parser.add_argument("--switch-cost", type=float, default=SWITCH_COST,
                        help="cost of a change of possession in the viterbi decoder")
    parser.add_argument("--min-alive-gap", type=int, default=MIN_ALIVE_GAP_FRAMES,
                        help="shortest burst of play inside a stoppage that counts as a restart")
    args = parser.parse_args()

    MIN_DEAD_FRAMES = args.min_dead
    DECODER = args.decoder
    SWITCH_COST = args.switch_cost
    MIN_ALIVE_GAP_FRAMES = args.min_alive_gap
    DEAD_FRACTION = args.dead_fraction
    SWITCH_FRAMES = args.switch_frames
    NEAR_RADIUS_M = args.near_radius
    if args.no_touches:
        TOUCH_TURN_DEG = 181.0
        TOUCH_SPEED_CHANGE_MS = float("inf")

    final = build()
    path = write_submission(os.path.join(args.out, args.name + ".csv"), final)
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
    print("check it:  python evaluation/validate_submission.py {}".format(os.path.relpath(path, ROOT)))
    print("score it:  python evaluation/run_evaluation.py")