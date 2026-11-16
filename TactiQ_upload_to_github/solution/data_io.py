"""Fast, cached readers for the provided match assets (tactical recording only).

Everything returned here is indexed by VIDEO frame (0 .. 199318), i.e. by submission row.
The raw ATD counts from 1, so record `frame_id` lands at row `frame_id - 1`.

The first call parses the 195 MB ATD text file with pandas' C parser (~20 s) and caches the
arrays as a compressed .npz under solution/.cache/ (gitignored: it is derived data and must not be
published). Later calls load the cache in about a second.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402

CACHE_DIR = os.path.join(HERE, ".cache")
FPS = GAMES[GAME]["fps"]
PITCH_LEN, PITCH_WID = GAMES[GAME]["pitch"]


def team_of_slots(meta_path):
    """0 = home, 1 = away for every track slot, in slot order (first team in the file is home)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def _parse_block(lines, n_slots, out):
    px, py, ps, bx, by = out
    text = "".join(lines).replace(";", ",").replace(":", ",")
    arr = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32, engine="c").to_numpy()
    expected_cols = 1 + 3 * n_slots + 3
    if arr.shape[1] != expected_cols:
        raise ValueError("ATD has {} columns, expected {}".format(arr.shape[1], expected_cols))
    rows = arr[:, 0].astype(np.int64) - 1
    slots = arr[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    px[rows], py[rows], ps[rows] = slots[..., 0], slots[..., 1], slots[..., 2]
    bx[rows], by[rows] = arr[:, -3], arr[:, -2]


def _parse_atd(raw_path, n_slots, n_frames, block=10000):
    """Streamed in blocks so peak memory stays small."""
    px = np.full((n_frames, n_slots), np.nan, np.float32)
    py = np.full((n_frames, n_slots), np.nan, np.float32)
    ps = np.full((n_frames, n_slots), np.nan, np.float32)
    bx = np.full(n_frames, np.nan, np.float32)
    by = np.full(n_frames, np.nan, np.float32)
    out = (px, py, ps, bx, by)
    lines = []
    with open(raw_path) as handle:
        for line in handle:
            lines.append(line)
            if len(lines) == block:
                _parse_block(lines, n_slots, out)
                lines = []
    if lines:
        _parse_block(lines, n_slots, out)
    return out


def load_tracking(game=GAME, data_dir=None, use_cache=True):
    """dict with px, py (frames x slots, metres), ps (speed as given), bx, by (metres), team."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team = team_of_slots(meta_path)
    cache = os.path.join(CACHE_DIR, "{}_tactical_atd.npz".format(game))

    if use_cache and os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(raw_path):
        z = np.load(cache)
        px, py, ps, bx, by = z["px"], z["py"], z["ps"], z["bx"], z["by"]
    else:
        px, py, ps, bx, by = _parse_atd(raw_path, len(team), n_frames)
        if not os.path.isdir(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        np.savez_compressed(cache, px=px, py=py, ps=ps, bx=bx, by=by)

    # normalised 0-1 -> metres, so distances are isotropic
    return {
        "px": px * PITCH_LEN, "py": py * PITCH_WID, "ps": ps,
        "bx": bx * PITCH_LEN, "by": by * PITCH_WID,
        "team": team, "n_frames": n_frames,
    }


def load_smart_tagging(game=GAME, data_dir=None):
    """DataFrame: start, end (video frames), code, team ('H'/'A'/None), type, half."""
    path = require(["smart_tagging"], game, data_dir)[0]
    root = ET.parse(path).getroot()
    rows = []
    for inst in root.iter("instance"):
        labels = {}
        for lab in inst.findall("label"):
            labels[(lab.findtext("group") or "").strip().lower()] = (lab.findtext("text") or "").strip()
        team_txt = labels.get("team", "").lower()
        team = "H" if team_txt.startswith("barcelona") else ("A" if "atletic" in team_txt
                                                               or "athletic" in team_txt else None)
        rows.append({
            "start": int(round(float(inst.findtext("start")) * FPS)),
            "end": int(round(float(inst.findtext("end")) * FPS)),
            "code": (inst.findtext("code") or "").strip().upper(),
            "team": team,
            "type": labels.get("type", ""),
            "half": labels.get("half", ""),
        })
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def graded_mask(game=GAME):
    """Boolean over video frames: True inside the two halves (the only frames ever scored)."""
    n = source_frames("tactical", game)
    mask = np.zeros(n, bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


if __name__ == "__main__":
    import time
    t = time.time()
    trk = load_tracking(use_cache=False)
    print("parsed in {:.1f}s".format(time.time() - t))
    g = graded_mask()
    print("ball visible on graded frames: {:.1%}".format(np.mean(~np.isnan(trk["bx"][g]))))
    print("tracked objects per graded frame: {:.1f}".format(
        np.mean(np.sum(~np.isnan(trk["px"][g]), axis=1))))
    st = load_smart_tagging()
    print(st["code"].value_counts())
