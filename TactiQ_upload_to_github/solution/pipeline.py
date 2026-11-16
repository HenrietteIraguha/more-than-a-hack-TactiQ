"""Ball possession, frame by frame, from the automatic tracking + Smart Tagging.

    python solution/pipeline.py                      # writes evaluation/results/<team>.csv
    python solution/pipeline.py --name my_team

Two stages, both temporal models rather than per-frame classifiers, because the ranking metric
(spellF1) scores whole possessions and punishes flicker far more than per-frame error:

  Stage 1 -- ball alive vs dead.
      An unsupervised two-state Gaussian HMM, fitted by Baum-Welch on the full match (no labels),
      over three collective features: how fast the players move, how often the tracker sees the
      ball, and how fast the ball moves. Dead balls are walking-pace, ball-placed phases.
      Hard evidence on top: a ball seen outside the pitch is dead. Boundaries are then snapped to
      physical events -- the frame the ball crosses a line, the frame a restart is struck.

  Stage 2 -- which team, inside each live stretch.
      A two-state (H/A) Viterbi over per-frame "control" evidence: which team's nearest player is
      close to the ball, and by how much. A change of possession costs a fixed log-penalty, so the
      opponent needs sustained control (roughly a second of clear evidence) to take the ball --
      a deflection or a single uncontrolled touch cannot pay it. When the ball is invisible the
      evidence is zero and the current possession simply continues. Smart Tagging phases add a
      weak team prior that mostly matters while the ball is lost.

Every parameter below is a football / physical prior chosen up front, except SWITCH_COST and
MIN_DEAD_S, which were chosen on Metrica's open sample games (other matches) by
solution/external/tune_on_sample.py. Nothing is fitted to the public ground truth (RULES.md forbids
tuning on it), and no labels from this game are used anywhere.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from data_io import (FPS, PITCH_LEN, PITCH_WID, graded_mask, load_smart_tagging,  # noqa: E402
                     load_tracking)
from games import GAME, half_video_blocks  # noqa: E402
from hmm import fit_gaussian_hmm, gaussian_log_emissions, viterbi  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean over empty frames

# ---------------------------------------------------------------------------------------------
# Parameters (physical / football priors; see module docstring)
# ---------------------------------------------------------------------------------------------
BALL_INTERP_GAP = 12          # frames (~0.5 s): linear interpolation across short ball dropouts
OUT_MARGIN = 0.5              # m beyond a line before a seen ball counts as out of play
SMOOTH_HALF = 12              # frames: half-width of feature smoothing (~1 s window)
TOP_K = 3                     # "pace of the game" = mean speed of the 3 fastest tracked players
STATIC_BLOCK = 625            # frames (25 s): a track that stays inside STATIC_SPAN_M for a whole
STATIC_SPAN_M = 1.0           # block is an object, not a player (idea from TactiQ's our_solution.py)

# Stage 1 HMM
DEAD_EM_STRIDE = 5            # fit EM at 5 fps; decode at 25 fps
DEAD_MEAN_SEGMENT_S = 15.0    # prior mean length of an alive or dead stretch
DEAD_EMIT_SCALE = 0.2         # smoothed features are strongly autocorrelated; temper them
OUT_EVIDENCE = 6.0            # log-odds per frame that a ball seen out of play is dead
MIN_DEAD_S = 8.0              # a stoppage with no out-of-play sighting shorter than this is a slow
                              # live phase (tuned on Metrica's open sample games; was 4.0 a priori)
MIN_ALIVE_S = 1.0             # shorter live stretches between two dead ones are noise
SNAP_BEFORE, SNAP_AFTER = 50, 25    # search window (frames) for snapping alive->dead to a line
RESTART_WIN = 50              # +/- frames searched for the restart kick
KICK_SPEED, STILL_SPEED = 5.0, 2.0  # m/s: ball after / before a restart is struck

# Stage 2 Viterbi
CONTROL_R = 1.0               # m: within this the nearest player is treated as in control
FAR_R = 3.0                   # m: beyond this nobody is near the ball (in flight / loose)
BETA = 0.35                   # per-frame log-likelihood of full control evidence
SWITCH_COST = 3.0             # log-penalty for a change of possession (~0.35 s of clear control);
                              # tuned on Metrica's open sample games (was 8.0 a priori)
TAG_PRIOR = 0.0075            # per-frame log-likelihood from a Smart Tagging phase; kept at the
                              # same ratio to SWITCH_COST as the a-priori 0.02 / 8.0 (untuned:
                              # the sample games have no Smart Tagging)
RESTART_TAG_PRIOR = 3.0       # start-state log-odds for the team a set piece names

# Smart Tagging code -> who has the ball, relative to the tagged team.
TAG_POSSESSOR = {
    "BUILD UP": +1, "PROGRESSION": +1, "BALL IN FINAL THIRD": +1, "BALL IN THE BOX": +1,
    "CREATING CHANCES": +1, "LONG BALL": +1, "PLAYERS IN THE BOX": +1,
    "ATTACKING TRANSITION": +1,
    "DEFENDING IN DEFENSIVE THIRD": -1, "DEFENDING IN MIDDLE THIRD": -1,
    "DEFENDING IN ATTACKING THIRD": -1, "DEFENSIVE TRANSITION": -1,
}

H, A = 0, 1
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def runs(mask):
    """[(start, end_inclusive)] of True runs."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def rolling_mean(x, half):
    return pd.Series(x).rolling(2 * half + 1, center=True, min_periods=1).mean().to_numpy()


def interpolate_short_gaps(x, max_gap):
    s = pd.Series(x)
    filled = s.interpolate(limit=max_gap, limit_area="inside")
    # pandas' limit fills the first max_gap of a long gap too; undo that for long gaps
    for a, b in runs(s.isna().to_numpy()):
        if b - a + 1 > max_gap:
            filled.iloc[a:b + 1] = np.nan
    return filled.to_numpy()


def remove_static_tracks(px, py):
    """Blank tracks that stay inside a 1 m box for a whole 25-second block.

    The automatic tracking has "fake dots" -- e.g. a track standing perfectly still beside a goal,
    before kickoff and during play. Real players never stand still that long, and a fake dot near
    the ball would be read as a player in control of it. Found by team TactiQ; same rule as
    starter/our_solution.py in their repository.
    """
    n, k = px.shape
    nb = -(-n // STATIC_BLOCK)
    pad = nb * STATIC_BLOCK - n
    bx = np.concatenate([px, np.full((pad, k), np.nan, px.dtype)]).reshape(nb, STATIC_BLOCK, k)
    by = np.concatenate([py, np.full((pad, k), np.nan, py.dtype)]).reshape(nb, STATIC_BLOCK, k)
    span = np.fmax(np.nanmax(bx, 1) - np.nanmin(bx, 1), np.nanmax(by, 1) - np.nanmin(by, 1))
    seen = np.sum(~np.isnan(bx), axis=1)
    still = (span < STATIC_SPAN_M) & (seen >= STATIC_BLOCK // 2)
    mask = np.repeat(still, STATIC_BLOCK, axis=0)[:n]
    return np.where(mask, np.nan, px), np.where(mask, np.nan, py)


# ---------------------------------------------------------------------------------------------
# per-half feature extraction
# ---------------------------------------------------------------------------------------------
def half_features(trk, first, last):
    sl = slice(first, last + 1)
    bx_raw, by_raw = trk["bx"][sl].astype(float), trk["by"][sl].astype(float)
    seen = ~np.isnan(bx_raw)
    out = seen & ((bx_raw < -OUT_MARGIN) | (bx_raw > PITCH_LEN + OUT_MARGIN)
                  | (by_raw < -OUT_MARGIN) | (by_raw > PITCH_WID + OUT_MARGIN))

    bx = interpolate_short_gaps(bx_raw, BALL_INTERP_GAP)
    by = interpolate_short_gaps(by_raw, BALL_INTERP_GAP)
    have = ~np.isnan(bx)
    step = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan)) * FPS
    ball_speed = pd.Series(step).rolling(5, center=True, min_periods=1).median().to_numpy().copy()
    ball_speed[~have] = np.nan

    # Collective pace. Ball visibility is deliberately NOT a feature: the tracker loses the ball
    # in crowded live play as often as in stoppages, so it confuses "lost" with "dead".
    ps = trk["ps"][sl]
    player_speed = np.nanmean(ps, axis=1)
    top_speed = np.sort(np.nan_to_num(ps, nan=0.0), axis=1)[:, -TOP_K:].mean(axis=1)
    f_spd = np.log(rolling_mean(player_speed, SMOOTH_HALF) + 0.1)
    f_top = np.log(rolling_mean(top_speed, SMOOTH_HALF) + 0.1)
    f_bspd = np.log(pd.Series(ball_speed).rolling(2 * SMOOTH_HALF + 1, center=True,
                                                   min_periods=3).mean().to_numpy() + 0.3)

    # nearest player of each team to the (interpolated) ball, ignoring tracks that never move
    px, py = remove_static_tracks(trk["px"][sl], trk["py"][sl])
    team = trk["team"]
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    d_home = np.nanmin(np.where(team[None, :] == H, dist, np.nan), axis=1)
    d_away = np.nanmin(np.where(team[None, :] == A, dist, np.nan), axis=1)

    return {
        "X": np.column_stack([f_top, f_spd, f_bspd]),
        "out": out, "seen": seen, "have": have, "ball_speed": ball_speed,
        "in_bounds": have & ~out, "d_home": d_home, "d_away": d_away,
    }


# ---------------------------------------------------------------------------------------------
# stage 1: alive / dead
# ---------------------------------------------------------------------------------------------
def fit_dead_model(feats):
    X = np.concatenate([f["X"][::DEAD_EM_STRIDE] for f in feats])
    q = np.nanpercentile(X, [25, 75], axis=0)
    init_means = np.array([q[1], q[0]])          # state 0 = brisk (alive), 1 = slow (dead)
    init_vars = np.tile(np.nanvar(X, axis=0), (2, 1))
    p_stay = 1.0 - 1.0 / (DEAD_MEAN_SEGMENT_S * FPS / DEAD_EM_STRIDE)
    means, variances, _ = fit_gaussian_hmm(X, init_means, init_vars, p_stay=p_stay)
    alive = int(np.argmax(means[:, 0]))          # whichever state has the faster players
    return means, variances, alive


def decode_dead(f, means, variances, alive_state):
    order = [alive_state, 1 - alive_state]
    log_emit = DEAD_EMIT_SCALE * gaussian_log_emissions(f["X"], means[order], variances[order])
    log_emit[f["out"], 1] += OUT_EVIDENCE        # column 1 = dead
    p = 1.0 / (DEAD_MEAN_SEGMENT_S * FPS)
    log_trans = np.log(np.array([[1 - p, p], [p, 1 - p]]))
    dead = viterbi(log_emit, log_trans, np.log([0.5, 0.5])) == 1

    # The ball must be dead for as long as it is seen out of play.
    dead |= f["out"]
    dead = snap_boundaries(dead, f)

    # Clean-up: drop implausibly short stretches.
    for a, b in runs(dead):
        if b - a + 1 < MIN_DEAD_S * FPS and not f["out"][a:b + 1].any():
            dead[a:b + 1] = False
    for a, b in runs(~dead):
        if b - a + 1 < MIN_ALIVE_S * FPS and a > 0 and b < len(dead) - 1:
            dead[a:b + 1] = True
    return dead


def snap_boundaries(dead, f):
    """Move alive->dead to the line crossing and dead->alive to the restart kick."""
    dead = dead.copy()
    n = len(dead)
    bs, out, inb = f["ball_speed"], f["out"], f["in_bounds"]

    for a, b in runs(dead):
        # start of a dead stretch: the first out-of-play sighting near it, if any
        lo, hi = max(0, a - SNAP_BEFORE), min(n, a + SNAP_AFTER)
        hits = np.where(out[lo:hi])[0]
        if len(hits):
            dead[lo + hits[0]:a] = True
            dead[a:lo + hits[0]] = False

    for a, b in runs(dead):
        if b >= n - 1:
            continue
        restart = b + 1
        lo, hi = max(a + 1, restart - RESTART_WIN), min(n - 6, restart + RESTART_WIN)
        best = None
        for t in range(lo, hi):
            if not inb[t]:
                continue
            before = bs[max(0, t - 10):t]
            after = bs[t:t + 6]
            # a struck ball: still beforehand, fast afterwards; or a throw-in: back in from out
            kicked = np.nanmean(after) > KICK_SPEED and (np.all(np.isnan(before))
                                                         or np.nanmean(before) < STILL_SPEED)
            thrown = out[max(0, t - 10):t].any()
            if (kicked or thrown) and (best is None or abs(t - restart) < abs(best - restart)):
                best = t
        if best is not None:
            if best > restart:
                dead[restart:best] = True
            else:
                dead[best:restart] = False
    return dead


# ---------------------------------------------------------------------------------------------
# stage 2: which team
# ---------------------------------------------------------------------------------------------
def control_llr(f):
    """Per-frame log P(obs | H) - log P(obs | A) from ball-to-nearest-player geometry."""
    dh, da = f["d_home"], f["d_away"]
    dmin = np.fmin(dh, da)
    weight = np.clip((FAR_R - dmin) / (FAR_R - CONTROL_R), 0.0, 1.0)
    margin = np.clip(da - dh, -1.0, 1.0)
    llr = BETA * weight * margin
    llr[~f["in_bounds"] | np.isnan(llr)] = 0.0
    return llr


def tag_llr(tags, first, n):
    """Weak per-frame team prior from Smart Tagging phases, relative to H.

    Phases overlap heavily (one attack is often tagged PROGRESSION, BALL IN FINAL THIRD, PLAYERS IN
    THE BOX and DEFENDING IN ... at once), so the votes are reduced to their net sign: one vote per
    frame, not one per tag, or five correlated tags would outvote the ball itself.
    """
    votes = np.zeros(n)
    for row in tags.itertuples():
        sign = TAG_POSSESSOR.get(row.code)
        if sign is None or row.team is None:
            continue
        possessor = row.team if sign > 0 else ("A" if row.team == "H" else "H")
        a, b = max(0, row.start - first), min(n, row.end - first + 1)
        if a < b:
            votes[a:b] += 1.0 if possessor == "H" else -1.0
    return TAG_PRIOR * np.sign(votes)


def restart_prior(tags, frame):
    """Start-state log-odds (H minus A) from a set piece tagged near a restart."""
    near = tags[(tags.code == "SET PIECES") & (tags.start <= frame + RESTART_WIN)
                & (tags.end >= frame - RESTART_WIN)]
    for team in near.team:
        if team == "H":
            return RESTART_TAG_PRIOR
        if team == "A":
            return -RESTART_TAG_PRIOR
    return 0.0


def decode_team(llr, start_odds):
    log_emit = np.column_stack([llr / 2.0, -llr / 2.0])
    p = np.exp(-SWITCH_COST)
    log_trans = np.log(np.array([[1 - p, p], [p, 1 - p]]))
    log_start = np.array([start_odds / 2.0, -start_odds / 2.0])
    log_start -= np.logaddexp(*log_start)
    return viterbi(log_emit, log_trans, log_start)


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------
NO_TAGS = pd.DataFrame({"start": pd.Series([], dtype=int), "end": pd.Series([], dtype=int),
                        "code": pd.Series([], dtype=object), "team": pd.Series([], dtype=object),
                        "type": pd.Series([], dtype=object)})


def predict(game=GAME, verbose=True):
    """Labels for the competition match."""
    return predict_tracking(load_tracking(game), load_smart_tagging(game),
                            half_video_blocks(game), verbose=verbose)


def prepare(trk, blocks):
    """The expensive, parameter-free part: features per half and the unsupervised dead model."""
    feats = [half_features(trk, first, last) for _, first, last in blocks]
    return feats, fit_dead_model(feats)


def predict_tracking(trk, tags, blocks, verbose=False, prepared=None):
    """Labels for any game given a tracking dict, Smart Tagging (or NO_TAGS) and half blocks."""
    labels = np.full(trk["n_frames"], "X", dtype="<U1")
    feats, (means, variances, alive_state) = prepared or prepare(trk, blocks)
    if verbose:
        names = ["log top-3 speed", "log mean speed", "log ball speed"]
        for s, tag in ((alive_state, "alive"), (1 - alive_state, "dead")):
            print("  {:<5} state means: {}".format(tag, ", ".join(
                "{} {:.2f}".format(n, m) for n, m in zip(names, means[s]))))

    for (_, first, last), f in zip(blocks, feats):
        n = last - first + 1
        dead = decode_dead(f, means, variances, alive_state)
        # A half begins with its kickoff: nothing before the tagged "Kick Off (Start)" is live.
        ko = tags[(tags.code == "SET PIECES") & tags.type.str.lower().str.startswith("kick off (start")
                  & (tags.start >= first - RESTART_WIN) & (tags.start <= last)]
        if len(ko):
            k = int(ko.start.min()) - first
            if k > 0:
                dead[:k] = True
                dead = snap_boundaries(dead, f)
        ball_llr = control_llr(f)
        # The tag prior only speaks where the ball says nothing (lost, or in flight).
        team_llr = np.where(ball_llr == 0.0, tag_llr(tags, first, n), ball_llr)
        half = np.full(n, "D", dtype="<U1")
        for a, b in runs(~dead):
            path = decode_team(team_llr[a:b + 1], restart_prior(tags, first + a))
            half[a:b + 1] = np.where(path == H, "H", "A")
        labels[first:last + 1] = half
    return labels


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


def summarise(labels):
    g = graded_mask()
    lab = labels[g]
    counts = {k: int(np.sum(lab == k)) for k in "HAD"}
    n_poss = sum(1 for _ in runs(lab == "H")) + sum(1 for _ in runs(lab == "A"))
    alive = counts["H"] + counts["A"]
    print("  graded frames: H {H}  A {A}  D {D}".format(**counts))
    print("  ball alive {:.1%} of graded time; {} possessions, mean {:.1f} s".format(
        alive / len(lab), n_poss, alive / max(n_poss, 1) / FPS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="TactiQ", help="submission file stem (team name)")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--switch-cost", type=float, default=SWITCH_COST,
                        help="log-penalty for a change of possession (default %(default)s); lower "
                             "gives more, shorter possessions")
    args = parser.parse_args()
    SWITCH_COST = args.switch_cost

    print("fitting and decoding ...")
    labels = predict()
    summarise(labels)
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    print("wrote", os.path.relpath(path, ROOT))
