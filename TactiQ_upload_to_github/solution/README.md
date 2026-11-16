# Possession HMM — More than a Hack 2027, Phase 1

Frame-by-frame ball possession (`H` / `A` / `D`) for BAR-ATH from the automatic tracking and Smart
Tagging, with two unsupervised temporal models and no labels from the provided game.
The method is described in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Reproduce the submission

Requires Python 3.12 (also works on 3.8+), numpy and pandas only.

```bash
pip install -r solution/requirements.txt
```

Unpack the data bundle into `data/` (or set `MTAH_DATA_DIR`), as described in [DATA.md](../DATA.md):

```
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

Then:

```bash
python solution/pipeline.py --name <team_name>
```

```bash
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

```bash
python evaluation/run_evaluation.py
```

The first run parses the 195 MB ATD file (~15 s) and caches it in `solution/.cache/`, which is
gitignored because it is derived match data. A full run takes about 15 s after that. The output is
deterministic: repeated runs produce a byte-identical CSV.

## Optional: repeat the tuning on other games

`SWITCH_COST` and `MIN_DEAD_S` were chosen on Metrica's open sample games, which are labelled
matches other than BAR-ATH. To repeat that (downloads about 160 MB, then runs for about 10 min):

```bash
python solution/external/fetch_metrica_sample.py
```

```bash
python solution/external/tune_on_sample.py
```

## Files

| file | what |
|---|---|
| `solution/data_io.py` | fast chunked ATD reader (metres, video-frame indexed), Smart Tagging reader, graded mask |
| `solution/hmm.py` | numpy Viterbi and Baum-Welch for a diagonal-Gaussian HMM with missing features |
| `solution/pipeline.py` | the two-stage model and the submission writer; every parameter is at the top |
| `solution/external/` | sample-game download, event→label conversion, the tuning script and its results table |
| `solution/TECHNICAL_REPORT.md` | the technical document |

## Results on the public 5-minute window (sanity check only — not a tuning set)

| submission | macroF1 | spellF1 |
|---|---|---|
| **possession_hmm** | **0.8169** | **0.4471** |
| baseline_smooth50 (published floor) | 0.6207 | 0.3256 |
| baseline_smooth125 | 0.5812 | 0.4242 |

The model predicts 17 possessions in the window, matching the truth's 17. With so few possessions,
spellF1 differences below ~0.37 are inside the bootstrap noise band, as
`run_evaluation.py --bootstrap 1000` reports. On the two open sample games used for tuning, the
mean spellF1 is 0.525.
