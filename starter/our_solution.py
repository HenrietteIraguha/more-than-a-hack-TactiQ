"""TactiQ -- who has the ball, frame by frame.

    pip install -r requirements.txt
    python starter/our_solution.py     # writes evaluation/results/TactiQ.csv
    python starter/check_output.py     # our own check, no ground truth involved

One label per video frame for all 199,319 frames: H (Barcelona), A (Atletic Club), D (dead ball),
X (outside the two graded blocks). Input is the automatic tracking data and the Smart Tagging.

  1. read the tracking: ball and 46 player tracks, in metres
  2. drop tracks that never move -- objects beside the goal, not players
  3. fill gaps in the ball shorter than half a second
  4. dead ball from the players: three in four walking, too few tracked, or ball over a line
  5. who restarts: the Smart Tagging phase tags, else the player nearest the ball when it moves
  6. possession inside each stretch of play: a Viterbi decoder with a cost per change of hands
  7. rub out stoppages the ball was clearly played through -- a final pass that can only turn a
     D into a team, never the other way round, so nothing decided in step 6 can move

Why each rule is there, which assets we used and dropped, and where it fails are in the technical
document. The SET PIECES tags are deliberately never read here: check_output.py marks our work
with them, so the two must not share a source.
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
import load_smart_tagging  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
FPS = 25
NOBODY = -1

# ----------------------------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------------------------

STATIC_BLOCK_FRAMES = 625          # 25 seconds
STATIC_MAX_SPAN_M = 1.0

BALL_GAP_FILL_FRAMES = 12          # about half a second

SPEED_LAG_FRAMES = 2

WALK_SPEED_MS = 2.0                # slower than this counts as walking or standing
DEAD_FRACTION = 0.75               # this share of players walking means play has stopped
DEAD_SMOOTH_FRAMES = 25            # average the walking share over 1 second
MIN_PLAYERS_FOR_DEAD = 8           # need at least this many tracked players to decide
MIN_DEAD_FRAMES = 125              # 5 seconds
MIN_ALIVE_GAP_FRAMES = 25

RESTART_LOOKBACK_FRAMES = 50       # the stoppage often ends a little late, so start 2 s earlier
RESTART_SEARCH_FRAMES = 75         # and look up to 3 seconds into play
KICK_SPEED_MS = 3.0                # the ball moving faster than this = it has been played
STILL_SPEED_MS = 2.0               # ...and it was slower than this just before (a set ball)
KICK_RADIUS_M = 4.0                # the taker must be at least this close to the ball

NEAR_RADIUS_M = 2.5
NEAR_MARGIN_M = 0.5
SWITCH_FRAMES = 12                 # about half a second

TOUCH_TURN_DEG = 40.0              # the ball turning more than this means someone touched it
TOUCH_SPEED_CHANGE_MS = 4.0        # ...or its speed jumping or dropping by more than this
TOUCH_WINDOW_FRAMES = 3            # compare the ball's movement 3 frames before and 3 after
TOUCH_RADIUS_M = 3.0               # the touch goes to a team whose player is this close
TOUCH_CONFIRM_WINDOW = 12
TOUCH_CONFIRM_FRAMES = 3
MIN_MOVE_M = 0.15                  # ignore direction changes when the ball barely moves

DECODER = "viterbi"                # "viterbi" (version 2) or "streak" (version 1)
CONTROL_RADIUS_M = 4.0             # only count the nearest player as evidence within 4 m
EVIDENCE_SCALE_M = 2.0             # evidence grows with how much closer that team is, 2 m = 1 unit
EVIDENCE_CAP = 3.0                 # and is capped, so one odd frame can't decide on its own
SWITCH_COST = 40.0

OUT_MARGIN_M = 1.0

FAST_BALL_RULE = True
FLY_SPEED_MS = 10.0
FEW_PLAYERS_DEAD = True
OUT_DEAD_RULE = True
OUT_MIN_FRAMES = 25                # 1 second

MERGE_STOPPAGES = True
BALL_BUSY_FRACTION = 0.5           # share of a stoppage the ball must be in play through

TAG_IN_POSSESSION = {
    "build up", "progression", "creating chances", "attacking transition", "long ball",
    "ball in final third", "ball in the box", "players in the box", "goals",
}
TAG_OUT_OF_POSSESSION = {
    "defending in attacking third", "defending in middle third", "defending in defensive third",
    "defensive transition",
}
TAG_RESERVED_FOR_CHECKING = {"set pieces"}

TAG_NOT_A_TEAM = {"", "na", "n", "none", "null", "unknown", "both", "neither"}
TAG_MIN_PARTIAL_CHARS = 4          # shorter than this must match a club name exactly

TAG_RESTART = True
TAG_FILL_ONLY = False              # True = only use the tags where version 3 found no taker
TAG_WINDOW_FRAMES = 50             # 2 seconds
TAG_MAJORITY = 0.6                 # the winning team needs this share of the votes, else skip


# ----------------------------------------------------------------------------------------------
# Version 6: who the Smart Tagging phases say has the ball, as per-frame votes
# ----------------------------------------------------------------------------------------------

def side_from_team_name(value, home_name, away_name):
    """Smart Tagging Team value -> HOME / AWAY / None, matched against the ATD metadata names."""
    def plain(text):
        return "".join(ch for ch in (text or "").lower() if ch.isalnum())

    tag, home, away = plain(value), plain(home_name), plain(away_name)
    if not tag or tag in TAG_NOT_A_TEAM:
        return None
    if tag == home:
        return HOME
    if tag == away:
        return AWAY
    if len(tag) < TAG_MIN_PARTIAL_CHARS:
        return None
    home_like = tag in home or home in tag
    away_like = tag in away or away in tag
    if home_like and not away_like:
        return HOME
    if away_like and not home_like:
        return AWAY
    return None                                     # ambiguous: do not guess


def phase_possession_votes(n_frames, home_name, away_name, game=GAME, data_dir=None):
    """(n, 2) int16 of votes per frame: column HOME and column AWAY."""
    votes = np.zeros((n_frames, 2), dtype=np.int16)
    used, skipped_code, skipped_team, reserved = 0, {}, 0, 0

    for event in load_smart_tagging.load(game, data_dir):
        code = (event["code"] or "").strip().lower()
        if code in TAG_RESERVED_FOR_CHECKING:
            reserved += 1
            continue
        if code in TAG_IN_POSSESSION:
            flip = False
        elif code in TAG_OUT_OF_POSSESSION:
            flip = True
        else:
            skipped_code[code] = skipped_code.get(code, 0) + 1
            continue

        side = side_from_team_name(event["tags"].get("Team"), home_name, away_name)
        if side is None:
            skipped_team += 1
            continue
        if flip:
            side = AWAY if side == HOME else HOME

        start = max(0, int(event["start_frame"]))
        stop = min(n_frames, int(event["end_frame"]))
        if stop > start:
            votes[start:stop, side] += 1
            used += 1

    return votes, {"used": used, "reserved": reserved, "no_team": skipped_team,
                   "unknown_codes": skipped_code}


def restart_team_from_tags(start, stop, votes):
    """Team the phases say has the ball in the first seconds after a restart, or NOBODY."""
    high = min(stop, start + TAG_WINDOW_FRAMES)
    if high <= start:
        return NOBODY
    home = int(votes[start:high, HOME].sum())
    away = int(votes[start:high, AWAY].sum())
    total = home + away
    if total == 0 or max(home, away) < TAG_MAJORITY * total:
        return NOBODY                               # no tags, or the phases disagree
    return HOME if home > away else AWAY


# ----------------------------------------------------------------------------------------------
# Step 1: read the tracking file into arrays (positions in metres, NaN where missing)
# ----------------------------------------------------------------------------------------------

def load_tracking(raw_path, n_frames, n_tracks, pitch):
    """Return ball (n, 2) and players (n, tracks, 2), in metres, NaN where missing."""
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
    """Copy of players with tracks blanked out on 25-second blocks where they did not move."""
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
    """True on frames where play looks stopped: most players walking, smoothed over 1 s."""
    player_velocity = velocity(players, SPEED_LAG_FRAMES)
    speed = np.hypot(player_velocity[..., 0], player_velocity[..., 1])  # (n, tracks)
    measured = ~np.isnan(speed)
    walking = np.sum(measured & (speed < WALK_SPEED_MS), axis=1)
    counted = np.sum(measured, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(counted >= MIN_PLAYERS_FOR_DEAD, walking / np.maximum(counted, 1), np.nan)
    share = rolling_mean(share, DEAD_SMOOTH_FRAMES)

    dead = share >= DEAD_FRACTION

    if FEW_PLAYERS_DEAD:
        few = rolling_mean(counted.astype(np.float32), DEAD_SMOOTH_FRAMES) < MIN_PLAYERS_FOR_DEAD
        dead = dead | few

    out = np.zeros(len(dead), dtype=bool)
    if off_pitch is not None:
        if OUT_DEAD_RULE:
            for start, stop in runs(off_pitch):
                if stop - start >= OUT_MIN_FRAMES:
                    out[start:stop] = True
            dead = dead | out
        else:
            dead = dead | off_pitch
    for start, stop in runs(~dead):
        if start > 0 and stop < len(dead) and stop - start < MIN_ALIVE_GAP_FRAMES:
            dead[start:stop] = True
    for start, stop in runs(dead):
        if stop - start < MIN_DEAD_FRAMES and not out[start:stop].any():
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
    """Frames where the ball suddenly turns or changes speed: someone probably touched it."""
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


def possession_evidence(ball, home_dist, away_dist, ball_speed=None):
    """Per frame: positive favours home, negative favours away, 0 = no evidence."""
    known = ~np.isnan(ball[:, 0])
    nearest = np.fmin(home_dist, away_dist)
    diff = away_dist - home_dist
    with np.errstate(invalid="ignore"):
        usable = known & ~np.isnan(diff) & (nearest <= CONTROL_RADIUS_M)
        ev = np.clip(np.where(usable, diff / EVIDENCE_SCALE_M, 0.0), -EVIDENCE_CAP, EVIDENCE_CAP)
        if ball_speed is not None and FAST_BALL_RULE:
            ev = np.where(ball_speed > FLY_SPEED_MS, 0.0, ev)
    return ev


def viterbi_two_teams(evidence, start_team):
    """Most likely H/A sequence for one stretch of live play."""
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


def label_half(ball, players, team_of_track, pitch=None, votes=None):
    """Labels H / A / D for every frame of one half (arrays already cut to that half)."""
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
    evidence = possession_evidence(ball, home_dist, away_dist, ball_speed)

    current = NOBODY
    n_switches = 0
    n_restarts, n_by_kicker, n_by_tags = 0, 0, 0
    use_tags = TAG_RESTART and votes is not None
    for start, stop in runs(~dead):
        n_restarts += 1
        taker, from_tags = NOBODY, False

        if use_tags and not TAG_FILL_ONLY:
            taker = restart_team_from_tags(start, stop, votes)
            from_tags = taker != NOBODY

        if taker == NOBODY:
            taker = restart_taker(start, stop, ball, ball_v, ball_speed, players, team_of_track)

        if taker == NOBODY and use_tags and TAG_FILL_ONLY:
            taker = restart_team_from_tags(start, stop, votes)
            from_tags = taker != NOBODY

        if from_tags:
            n_by_tags += 1
        if taker != NOBODY:
            current = taker
            if not from_tags:
                n_by_kicker += 1
            window, known = None, []
        else:
            window = near[start:min(stop, start + RESTART_SEARCH_FRAMES)]
            known = window[window != NOBODY]
        if len(known):
            current = HOME if np.sum(known == HOME) >= np.sum(known == AWAY) else AWAY
        elif current == NOBODY:
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
            if frame in touches and touch_team[frame] not in (NOBODY, current):
                toucher = int(touch_team[frame])
                after = near[frame:min(stop, frame + TOUCH_CONFIRM_WINDOW)]
                if np.sum(after == toucher) >= TOUCH_CONFIRM_FRAMES:
                    current = toucher
                    n_switches += 1
                    streak_team, streak_len = NOBODY, 0

            who = near[frame]
            if who != NOBODY and who != current:
                if who != streak_team:
                    streak_team, streak_start, streak_len = who, frame, 0
                streak_len += 1
                if streak_len >= SWITCH_FRAMES:
                    current = int(who)
                    n_switches += 1
                    labels[streak_start:frame] = "H" if current == HOME else "A"
                    streak_team, streak_len = NOBODY, 0
            elif who == current:
                streak_team, streak_len = NOBODY, 0

            labels[frame] = "H" if current == HOME else "A"
            frame += 1

    merged = 0
    if MERGE_STOPPAGES and pitch is not None:
        labels, merged = merge_false_stoppages(labels, ball, ball_speed, pitch)

    return labels, {"static_blocks": n_static, "touches": len(touches), "switches": n_switches,
                    "dead_frames": int(dead.sum()), "restarts": n_restarts,
                    "restarts_by_kicker": n_by_kicker, "restarts_by_tags": n_by_tags,
                    "merged": merged}


def restart_taker(start, stop, ball, ball_v, ball_speed, players, team_of_track):
    """Team of the player who restarts play, or NOBODY if the kick can't be found."""
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
        point, when = ball[moment], moment
    else:
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


def merge_false_stoppages(labels, ball, ball_speed, pitch):
    """Turn a stoppage back into play where the ball was clearly being played through it.

    The dead-ball rule never looks at the ball, so a slow passage of play -- the back four
    passing sideways with nobody pressing -- looks exactly like a whistle: almost everyone is
    strolling. 87 of our 145 stoppages have no set piece anywhere near them, where a real match
    has perhaps 25, and a stoppage invented inside a real possession is the most expensive
    mistake available: it destroys that possession and manufactures two more.

    This runs last, on the finished labels, and can only ever turn a D into a team. Everything
    step 6 decided is frozen. Three separate guards have to agree before a stoppage is rubbed
    out:

      * the SAME team has the ball immediately before and after it. A real stoppage usually
        changes hands -- a throw-in, a foul won -- so one that does is left alone. One that does
        not is a possession we may have cut in half.
      * the ball never clearly crossed a line during it. That stoppage is not our inference, it
        is the laws of the game: the ball is dead the moment it wholly crosses. A hunch about
        ball speed has no business overruling it, and this is the guard that keeps the quick
        throw-ins we already get right.
      * the ball is visible, inside the lines and being played for more than half of it. A real
        stoppage does have the ball moving sometimes -- rolled back to the taker, bouncing away
        after going out -- but in bursts, not throughout.
    """
    off = ball_off_pitch(ball, pitch)
    with np.errstate(invalid="ignore"):
        moving = ball_speed > KICK_SPEED_MS
    busy = (~np.isnan(ball[:, 0])) & (~off) & moving

    merged = 0
    for start, stop in runs(labels == "D"):
        if start == 0 or stop >= len(labels):
            continue                            # at the edge of the half: no team on both sides
        before, after = labels[start - 1], labels[stop]
        if before not in ("H", "A") or before != after:
            continue                            # changes hands, or unknown: leave it alone
        if off[start:stop].any():
            continue                            # the ball crossed a line: dead by the laws
        if np.mean(busy[start:stop]) <= BALL_BUSY_FRACTION:
            continue                            # the ball was not being played through it
        labels[start:stop] = before
        merged += 1
    return labels, merged


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

    votes = None
    if TAG_RESTART:
        votes, tag_stats = phase_possession_votes(n_frames, home, away, game, data_dir)
        covered = 100.0 * np.mean(votes.sum(axis=1) > 0)
        print("Smart Tagging: {} phases used, {} set-piece tags left for check_output.py, "
              "{} with no usable Team".format(tag_stats["used"], tag_stats["reserved"],
                                              tag_stats["no_team"]))
        if tag_stats["unknown_codes"]:
            print("  codes we do not recognise, skipped: {}".format(
                ", ".join("{} ({})".format(code, n)
                          for code, n in sorted(tag_stats["unknown_codes"].items()))))
        print("  they say who has the ball on {:.1f}% of all frames\n".format(covered))

    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in half_video_blocks(game):
        stop = last + 1
        half_labels, stats = label_half(ball[first:stop], players[first:stop], team_of_track,
                                        GAMES[game]["pitch"],
                                        votes[first:stop] if votes is not None else None)
        labels[first:stop] = half_labels
        print("half {}: H {:>6}  A {:>6}  D {:>6}   possessions {:>4}   "
              "restarts {} (from the tags {}, from the dots {}, guessed {})".format(
                  half, int(np.sum(half_labels == "H")), int(np.sum(half_labels == "A")),
                  int(np.sum(half_labels == "D")), count_possessions(half_labels),
                  stats["restarts"], stats["restarts_by_tags"], stats["restarts_by_kicker"],
                  stats["restarts"] - stats["restarts_by_tags"] - stats["restarts_by_kicker"]))
        print("        stoppages rubbed out because the ball was being played through them: {}"
              .format(stats["merged"]))
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="TactiQ", help="output file name, without .csv")
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
    parser.add_argument("--no-fast-ball", action="store_true",
                        help="version 2 behaviour: a flying ball still gives evidence")
    parser.add_argument("--no-few-dead", action="store_true",
                        help="version 2 behaviour: few tracked players does not mean dead ball")
    parser.add_argument("--no-out-dead", action="store_true",
                        help="version 2 behaviour: ball out of pitch subject to the 5 s minimum")
    parser.add_argument("--no-merge-stoppages", action="store_true",
                        help="do not rub out stoppages the ball was clearly played through")
    parser.add_argument("--busy-fraction", type=float, default=BALL_BUSY_FRACTION,
                        help="share of a stoppage the ball must be in play through to rub it out")
    parser.add_argument("--no-tag-restart", action="store_true",
                        help="version 3 behaviour: work the restarting team out from the dots only")
    parser.add_argument("--tag-fill-only", action="store_true",
                        help="use the dots first and the tags only where the kick was not found")
    parser.add_argument("--tag-window", type=int, default=TAG_WINDOW_FRAMES,
                        help="frames of play after a restart the phases vote over (25 = 1 second)")
    args = parser.parse_args()

    MIN_DEAD_FRAMES = args.min_dead
    FAST_BALL_RULE = not args.no_fast_ball
    FEW_PLAYERS_DEAD = not args.no_few_dead
    OUT_DEAD_RULE = not args.no_out_dead
    MERGE_STOPPAGES = not args.no_merge_stoppages
    BALL_BUSY_FRACTION = args.busy_fraction
    TAG_RESTART = not args.no_tag_restart
    TAG_FILL_ONLY = args.tag_fill_only
    TAG_WINDOW_FRAMES = args.tag_window
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