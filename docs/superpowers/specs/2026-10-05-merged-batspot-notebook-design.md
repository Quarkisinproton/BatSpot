# Merged BatSpot fine-tuning + inference notebook — design

**Date:** 2026-10-05
**Deliverable:** `batspot-train-merged.ipynb` (new file; neither parent notebook is modified)
**Status:** design approved, spec awaiting review

---

## 1. Purpose

Combine the best of two existing notebooks into one that (a) reports accuracy with
error bars, (b) runs on arbitrary uploaded recordings, and (c) tests the modelling
ideas that have never actually been tried.

### Parents

| Notebook | Cells | Provenance |
|---|---:|---|
| `batspot-train.ipynb` | 24 (22 code) | Windowing overhaul (`AGENTS.md` §8, §8.10), review fixes B1–B5, inference section (§10). Run end-to-end 3× locally and once on Kaggle. |
| `batspot_train(claude_bug_fixes _by__SPACE_BUNNY_MODEL).ipynb` | 22 (20 code) | Independent review (`AGENTS.md` §9, `Space_Bunny_implimentation.md`): cache hardening, 14 fail-loud fixes, prior-correction machinery, grouped + multi-seed cells. **Never run end to end.** |

Sources read for this design: `AGENTS.md` §1–§10.6, `Space_Bunny_implimentation.md`,
`Sonnet5.5_implimentation.md`, `Claude_improvement_plan.md`,
`Intention_in_creating_notebook.md`, and both notebooks' Cell 2 configs.

`Claude_improvement_plan.md` and `Intention_in_creating_notebook.md` are **stale** and
their proposals are explicitly out of scope (§8 below).

### Success criteria

1. Every reported accuracy carries an error bar, and no claim below the measured noise
   floor is presented as real.
2. A folder of new recordings (any length, any sample rate ≥ 192 kHz) can be processed
   end-to-end to a Raven-style selection table.
3. The operating threshold is chosen on real audio, not on a 19 %-noise test split.
4. Each ported fix is proven by a test that fails when the fix is reverted.

### Non-goals

Gu GUI/CLI compatibility is preserved, not extended. The training front end stays
bit-exact with the vendored official code. No new model capacity (§10.6 measured it as
useless). No change to the exported `.pk` format.

---

## 2. Notebook shape

One notebook, four phases, gated by config flags in Cell 2.

| Phase | Contents | Flag | Default | Runtime |
|---|---|---|---|---|
| **1 — Train** | config → imports → architecture → windowed dataset → split → model discovery → 3 detectors → classifier → cascade → export → summary | always on | — | ~15 min |
| **2 — Measure** | recording-grouped retrain, multi-seed mean ± sd | `RUN_MEASURE` | **True** | +~35 min |
| **3 — Experiments** | augmentation A/B, label smoothing, window length, 250 kHz detector | `RUN_EXPERIMENTS` | **False** | +~40 min |
| **4 — Inference** | scan → selections → Raven tables → truth scoring → calibration | needs uploaded data | — | ~10 min/file |

`Run All` therefore trains, evaluates, and reports error bars in **~50 min** on Kaggle
(2× T4), against a 12 h session limit. Flipping `RUN_EXPERIMENTS` adds the exploratory
sweep for ~90 min total.

Phase 2's ~35 min is dominated by the seed loop: 3 seeds × (classifier ≈ 5.7 min +
one detector ≈ 2.2 min) ≈ 24 min, plus the grouped retrain ≈ 10 min. It trains the
classifier and the **best mic only** — retraining all three mics would triple the cost
for no extra insight.

Phase 2 is on by default because it is what makes every other number believable; phase 3
is off because it is exploratory and its results are not needed to use the models.

Phase 2 re-splits data for its own runs. It **does not** disturb the phase-1 models: the
headline classifier and detector remain the ones trained on the original seed-42 split
and exported to `.pk`.

### 2.1 Flag placement

All three flags live in **Cell 2** (configuration), declared there and read where used:

| Flag | Default | Meaning |
|---|---|---|
| `RUN_MEASURE` | `True` | run phase 2 (grouped retrain + multi-seed) |
| `RUN_EXPERIMENTS` | `False` | run phase 3 (the untested-idea A/Bs) |
| `RUN_PHASE4` | `False` | run phase 4 (inference); also requires uploaded data |

Phase 4 is gated separately from the others because it needs `INFER_INPUT_DIR` to point
at a dataset that may not be attached, and a hard `assert` there would break `Run All`
for someone who only wants to train.

---

## 3. Merge map

Base is `batspot-train.ipynb` — the only parent ever run end to end, and the only one
holding the inference cells.

### 3.1 Ported from the Space Bunny notebook

Into the existing pipeline cells, not appended:

| Fix | Into | Change |
|---|---|---|
| Cache validity | dataset cell | source wav size+mtime in a sidecar, validated on load (was `os.path.exists`) |
| Cache filename | dataset cell | source directory hashed in, so two `DATA_DIR`s cannot collide |
| `_CACHE_VERSION` | dataset cell | derived from `clip_to_db_spectrogram`'s source via `inspect.getsource`, with fallback |
| Atomic save | dataset cell | write `.tmp` + `os.replace` |
| Per-file failure | dataset cell | `try/except` per wav, failures listed, clips dropped **visibly** |
| NaN guard | dataset cell | `np.nan_to_num` on the dB spectrogram |
| mmap cache | dataset cell | per-instance lazy cache, not a fresh `np.load` per `__getitem__` |
| `train_window_starts` | dataset cell | exact `ceil(top_frac·n)` by rank, stable sort (was a percentile threshold: ties kept every window; 3 clips were fully degenerate) |
| `predict_proba` zero-window | dataset cell | hard error listing offending files (was an all-zero row whose `argmax` = class 0 = a free correct prediction) |
| `_window` order (B2) | dataset cell | min-max **then** pad; count and print sub-window clips |
| `recording_id` (B4) | split cell | `(\d{8})[-_](\d{6})` + `.search()`, `n_groups == n_files` warning |
| Per-class recording counts | split cell | table making `heti`=1, `rhbe`=2 visible at split time |
| `save_path` (B1) | train cell | unconditional best-weight snapshot/restore; `best_score = -inf`; `.detach().clone()` |
| `SELECT_METRIC` (B3) | config + train | assert at config time, dict lookup at selection |
| `REPORT_TOPK_MEAN` | train cell | prints argmax vs top-3 mean vs overall mean, and the gap in clip units |
| Real `savefig` | train cell | `save_path` actually writes the curves (previously fiction) |
| Summary labels (B5) | summary cells | `kind` / `n_files` columns, `NO SIGNAL (AUC~0.5)` marker, guarded AUC, output-width confusion-matrix naming, `num_mels` fallback, closing balanced-accuracy gain line |
| `prior_correct_probs` | cascade cell | machinery only; see §5 |

### 3.2 Kept from Claude's notebook — do not regress

`SEED`, `RECORDING_SPLIT_SCOPE='per_species'`, `RUN_CLIP_EXTRACTION=False`, threshold
grid 0.05–0.95, CPU-copy export (`copy.deepcopy(model).cpu()`), unweighted
`CrossEntropyLoss` + `WeightedRandomSampler`, absolute LR (`base_lr` unscaled),
raw-epoch patience, `shortcut.*` key names, `min_max_norm=true` export note, cells 20–24.

`RECORDING_SPLIT_SCOPE='per_species'` is **not in any `.md` doc** — it postdates them and
is strictly better than Space Bunny's global `StratifiedGroupKFold`, which starves `heti`
into a single split. Preserved as found.

### 3.3 Explicitly out of scope

From `Claude_improvement_plan.md` / `Intention_in_creating_notebook.md`, all measured or
disproven: `n_fft` 2048, `sequence_len` 400 ms, 384 kHz single model, differential
encoder/head LRs, `betas=0.9`, weight decay, class-weighted loss, Mixup + the 7-transform
augmentation set as a default, tape-based 70/15/15 split.

Also out: hidden-layer heads and ResNet-34 (§10.6 — no gain, and a hidden-layer head
breaks `.pk` loading), LR 1e-2, detector ensembles, the detector→classifier hard gate as
a *replacement* for the classifier.

---

## 4. Phase 2 — measurement

The single most valuable addition, because every number in the project currently comes
from one 145-clip split with no error bar, and the measured run-to-run spread (m09
0.876–0.931, m11 0.876–0.924, classifier 0.848–0.876) is **larger than every difference
anyone has argued about**.

### 4.1 Multi-seed spread

Seeds 42/43/44, each **re-splitting and re-initialising**, so the reported sd is total
run-to-run spread. Trains the classifier and the best mic only (§2), and reports mean ±
sd for each.

The **noise floor** is defined once, in the phase-2 output, as
`2 × sd` of the per-seed test scores — roughly the smallest delta distinguishable from
run-to-run variation. Every later phase prints its delta beside that number, so a claim
can be read as real or not without leaving the notebook. Measured reference values:
m09 0.876–0.931, m11 0.876–0.924, classifier 0.848–0.876 over three identical-config runs.

### 4.2 Recording-grouped retrain

`StratifiedGroupKFold` with the `per_species` scope, retraining the classifier and the
best mic only. Asserts train/test recording overlap is 0, lists classes that vanish from
the fold, and prints in-split vs grouped side by side.

This cannot be derived arithmetically: the leakage check reports 145/145 test clips share
a recording with train, so there is no clean subset of the existing split to score. The
honest new-recording expectation is **detector balanced accuracy 0.64–0.71**, not 0.93.

---

## 5. Prior correction — the one merge conflict

**Claude:** with a *val-tuned* threshold, `p · (prior_eval/prior_train)` is monotone, so
the tuned gate makes the same decisions; and the test-time factor is computed from the
test split's class balance, which is not a deployment prior.

**Space Bunny:** the 50/50 sampler prior shifts the posterior by 4.18× in odds against an
81/19 eval split. Evidence: noise precision 0.70 against recall 0.93, and val-tuned
thresholds sinking to 0.30/0.40/0.10. Such a threshold encodes the split's noise fraction
and will not transfer to real audio, which is >95 % noise windows.

**Resolution.** Port the machinery; default to raw; drive the factor from a **supplied**
prior, never an inferred one.

```
INFER_EXPECTED_NOISE_FRACTION = 0.95   # deployment prior, set by the user
```

Declared in the phase-4 config cell (not Cell 2), because it is meaningless before
inference and keeping it out of Cell 2 avoids implying that phase 1 depends on it.

- Phase 1/2 report **raw** thresholds only, so they stay comparable with `AGENTS.md` §8–§10.
- Phase 4 prints **both** raw and prior-corrected thresholds for each setting.
- Nothing infers the factor from a split. That removes Claude's objection (it is no longer
  a test-split statistic) and keeps Space Bunny's insight (0.5 means 0.5 on real audio).
- The prior-corrected gate is **not** applied to the classifier: 10 of its 21 errors are
  false `acsh`, the *most common* class, so a prior shift would push the wrong way.

---

## 6. Phase 3 — experiments

All behind `RUN_EXPERIMENTS`, default off. Each is a self-contained A/B against the
phase-1 baseline on the same split, and each prints the delta with the phase-2 noise
floor beside it.

| Experiment | Status | Note |
|---|---|---|
| Augmentation A/B on the windowed dataset | never tested | random cropping is already the dominant augmenter (×43–56 distinct inputs), so these may be redundant |
| Label smoothing 0.1 | never tested | one-line change in the loss; train loss is 0.33–0.44, so not memorising |
| 40 ms / 60 ms input windows | **measured dead end** | 40 ms 0.852 vs 20 ms 0.853; 60 ms 0.821. Included as requested, labelled as re-testing a known negative |
| **250 kHz detector** | substituted | see below |

### 6.1 Why "raise detector fmax" became "250 kHz detector"

The request was to raise the detector `fmax` to recover rhle/rhro. **That is not
physically possible.** The detector runs at 192 kHz, so Nyquist is 96 kHz and
`fmax=95000` is already 1 kHz below it. rhle/rhro's 90–100 kHz CF calls are *aliased*
away, not cropped. Verified: `animal_spot/predict.py:214` reads `dataOpts["fmax"]`
directly and `animal_spot/data/audiodataset.py` uses `f_max` with no clamping.

The available lever is **sample rate**. The 250 kHz classifier (`fmax=125000` = its
Nyquist) sees those calls correctly. So the experiment fine-tunes a **250 kHz binary
detector** from the official 250 kHz classifier's encoder with a fresh 2-class head, and
compares it against the 192 kHz detectors specifically on rhle/rhro recall.

It stays in phase 3, not the main loop: it is a fourth architecture and a fourth training
run, and no measurement yet says it will help.

---

## 7. Phase 4 — inference

Cells 20–24 kept **verbatim**: 20 ms windows at 10 ms hop, per-window min-max,
detector→classifier combo, selection merge/split, frequency-band extraction, clock-time
extraction, Raven table emission, truth scoring. These are the only verified end-to-end
numbers in the project (0.981 of boxes found, 0.797 species correct, 3 of 36 noise boxes
hit, on full 5-min recordings) and re-deriving them would risk the one part that works.

Defaults unchanged: threshold 0.5 (official BatSpot), merge gap 0.1 s, ≥2 windows, max
selection 1.0 s.

### 7.1 New: threshold calibration on real recordings

Cell 24 can score detections against Raven tables. The new cell reuses that scorer to
sweep threshold × merge gap and print the precision/recall tradeoff, so the operating
point is chosen on the uploaded audio.

This is the largest unmeasured risk in the project: every threshold to date was tuned on a
split that is 81 % call / 19 % noise, while real recordings are >95 % noise windows.
Without this, `0.981 boxes found` says nothing about false alarms.

It degrades honestly — with no truth tables it prints a count-vs-threshold summary and
says precision is unmeasurable, because the training tables mark only a fraction of the
calls in each recording.

---

## 8. Verification

1. **Port the 7 suites** (122 assertions) and re-point them at cells extracted from the
   **delivered merged `.ipynb`**, never a copy.
2. **Extend**: unit tests for `prior_correct_probs`; a new suite for the inference cells
   (merge, split-at-silence, clock-time extraction, frequency-band extraction), which
   today exist only as one-off scripts in
   `/home/gb/batspot_gpu_experiments/inference_cells_2026-10-04/`.
3. **Red-green**: revert each ported fix in a scratch copy, assert the matching suite
   fails. One revert per fix. This is what makes the assertion count mean anything — the
   original work found four vacuous tests this way.
4. **Full local GPU run** on the RTX 4050, all phases, 0 errors, with a comparison table
   against both parents' recorded numbers.

Both parents' numbers are reported with their generation label. The corpus holds three
generations of the same quantity (classifier 0.7655 / 0.8552 / 0.8759 / 0.8483), all
inside the noise band.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Cache hardening drops a failed wav, desynchronising detector and classifier test lists so the cascade raises | Derive both lists from one shared manifest |
| Phase 2/3 cells have never run inside a 20+ cell notebook | Flags default so `Run All` stays short; full local GPU run before delivery |
| Space Bunny's notebook was never run as a whole | Phase 1 is byte-identical to Claude's verified pipeline; every change is additive or a proven bug fix |
| The 250 kHz detector may not help rhle/rhro | Phase 3, flagged, reported with the noise floor beside the result |
| Kaggle run was previously on CPU with no banner | Phase 1 keeps the device print; a loud banner is added if `DEVICE` is `cpu` while paths are Kaggle-style |
| Inference numbers are in-sample | Stated in the notebook's own markdown; phase 4's calibration cell is the honest test |

---

## 10. Documentation

The merged notebook's first markdown cell states: the windowing mechanism, the phase
gates, the noise floor, which numbers are in-split vs grouped, and that
`min_max_norm=true` is required at prediction. `AGENTS.md` gains a section recording the
merge and the cell map once the notebook is verified.
