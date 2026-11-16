"""Download Metrica Sports' open sample games (other matches, not BAR-ATH) for tuning.

    python solution/external/fetch_metrica_sample.py

Source: https://github.com/metrica-sports/sample-data (anonymised games, tracking + human event
data, 25 fps, 105 x 68 m). Used here only as labelled data from OTHER games, which RULES.md allows.
Files go to external_data/metrica-sample-data/ (gitignored). Already-present files are skipped.
"""

import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "external_data", "metrica-sample-data")
BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
# Pinned by size so a changed upstream file is noticed.
FILES = {
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv": 139446,
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv": 32782961,
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv": 32782073,
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv": 153897,
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv": 31890338,
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv": 29627711,
}


def fetch():
    for rel, size in FILES.items():
        dest = os.path.join(OUT, rel)
        if os.path.isfile(dest) and os.path.getsize(dest) == size:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print("downloading", rel)
        urllib.request.urlretrieve(BASE + rel, dest)
        if os.path.getsize(dest) != size:
            raise IOError("{}: expected {} bytes, got {}".format(rel, size, os.path.getsize(dest)))
    print("sample games in", OUT)


if __name__ == "__main__":
    fetch()
