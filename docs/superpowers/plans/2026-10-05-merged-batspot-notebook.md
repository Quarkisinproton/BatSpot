# Merged BatSpot Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `batspot-train-merged.ipynb` — one notebook combining Claude's verified pipeline and inference cells with Space Bunny's 18 review fixes, plus error-bar reporting, four opt-in untested experiments, and threshold calibration on real audio.

**Architecture:** The notebook is **generated**, not hand-edited. `notebook_build/assemble.py` reads `batspot-train.ipynb` as the base, replaces 7 cells from source files in `notebook_build/cells/`, appends new phase cells, and writes `batspot-train-merged.ipynb`. This is the mechanism already proven by `spacebunny_tests/assemble.py` and `inference_cells_2026-10-04/patch_notebook.py`. Test suites `exec` cells **extracted from the delivered notebook**, never a copy, so they test the artifact.

**Tech Stack:** Python 3.10, PyTorch (CUDA build, not the checked-in CPU-only `venv/`), Jupyter nbformat JSON, scikit-learn, resampy, soundfile, matplotlib.

**Spec:** `docs/superpowers/specs/2026-10-05-merged-batspot-notebook-design.md`

## Global Constraints

- **Neither parent notebook is modified.** `batspot-train.ipynb` and `batspot_train(claude_bug_fixes _by__SPACE_BUNNY_MODEL).ipynb` are read-only inputs. Every change lands in `batspot-train-merged.ipynb` and `notebook_build/`.
- **Detector band:** `sr=192000`, `fmin=1000`, `fmax=95000`. **Classifier band:** `sr=250000`, `fmin=10000`, `fmax=125000`. Both `n_fft=256`, `hop_length=128`, `n_freq_bins=256`, `sequence_len=20` ms. These are read from the official `.pk` `dataOpts`; changing them invalidates the transferred encoder.
- **`fmax` cannot be raised on the detector.** Nyquist at 192 kHz is 96 kHz and `fmax=95000` already sits 1 kHz below it. rhle/rhro's 90–100 kHz CF calls are aliased, not cropped. The only lever is sample rate → the 250 kHz detector in Task 10.
- **Loss is unweighted `nn.CrossEntropyLoss()` paired with `WeightedRandomSampler`.** Never both weighted — that double-counts imbalance (4.18× penalty on noise false alarms) and collapses m03 to all-noise.
- **`base_lr` is absolute, never multiplied by batch size.** Detector `1e-4`, classifier `3e-4`. `1e-2` is measured bad.
- **Model selection, LR schedule and early stopping all track balanced accuracy** (`SELECT_METRIC`), which a constant predictor scores 0.5 on, not plain accuracy (0.807 on the 81/19 detector split).
- **Patience values are RAW epochs.** `no_improve += epochs_per_eval`. A counter that increments once per validation is the bug that made `early_stopping_patience_epochs` a no-op.
- **Architecture keys must serialise as `shortcut.*`**, per `animal_spot/models/residual_base.py:32`. Renaming breaks `load_state_dict` for every official `.pk`.
- **Windowing is the accuracy mechanism and must be preserved:** train = uniform random crop from the loudest `TRAIN_TOP_FRAC=0.20` of windows; eval = `TEST_TOPK=5` loudest non-overlapping windows, softmax averaged per clip.
- **Noise floor: no delta below ≈5 pt (detectors) / ≈3 pt (classifier) is real.** Phase 2 measures it as `2 × sd` and prints it; every later phase prints its delta beside it.
- **Exported `.pk` must load in the BatSpot GUI / `predict.py`.** No hidden-layer heads, no architecture changes to the export path. Exported models require `min_max_norm=true` at prediction.
- **Self-contained:** inline the ResNet definition; never `import animal_spot`.
- **Deterministic patching:** every string substitution asserts it matched **exactly once**, so a silent no-op is impossible.

## Review Focus

Five input classes the spec implies that unit tests on toy data would miss. Each has a test pinned to its owning task below.

1. **A clip shorter than one 20 ms window** (1 of 964 clips; 22 frames vs 30 needed). Padding must happen *after* min-max, or the zero-pad becomes the window maximum and squashes the real signal. → Task 3, `test_short_clip_not_padded_before_minmax`.
2. **A wav that fails to decode.** Must be dropped and listed, not kill the run — and must not desynchronise the detector and classifier test lists, or the cascade raises. → Task 3, `test_bad_wav_dropped_and_lists_stay_aligned`.
3. **Filenames from Cell 7's extractor** (`species-bat_YYYYMMDD_HHMMSS_...`), which a positional or hyphen-only parser collapses to one group per clip, making `SPLIT_BY_RECORDING=True` put a whole species in one fold. → Task 4, `test_extractor_layout_shares_a_group_across_species`.
4. **A recording below the classifier's 250 kHz** on upload. Calls above its Nyquist are invisible; the notebook must say so rather than silently returning noise. → Task 11, `test_low_sample_rate_is_reported`.
5. **Ties in the window-loudness score.** A percentile threshold keeps *every* tied window; 3 clips kept 100% of their windows including silence, which is label noise. → Task 3, `test_tied_scores_keep_exactly_top_fraction`.

---

### Task 1: Build scaffold

**Files:**
- Create: `notebook_build/assemble.py`
- Create: `notebook_build/cells/` (empty, populated by later tasks)
- Create: `notebook_build/tests/__init__.py` (empty)
- Create: `notebook_build/README.md`
- Read-only input: `batspot-train.ipynb`

**Interfaces:**
- Consumes: nothing.
- Produces: `assemble.py` with `REPLACE: dict[int, str]` (base cell index → filename in `notebook_build/cells/`), `APPEND_CODE: list[str]`, `APPEND_MD: list[str]`, and `to_source(text: str) -> list[str]`. Also `build(out_path: str) -> None`. Every later task adds entries to `REPLACE`/`APPEND_*`.

- [ ] **Step 1: Write the failing test**

Create `notebook_build/tests/test_assemble.py`. It imports `assemble` and asserts the round-trip contract:

```python
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import assemble

def test_to_source_roundtrip():
    text = "a = 1\nb = 2"
    assert ''.join(assemble.to_source(text)) == text

def test_to_source_trailing_newline_dropped_on_last_line():
    assert assemble.to_source("x\n") == ["x\n"]

def test_base_notebook_has_expected_shape():
    nb = json.load(open(assemble.BASE))
    assert len(nb['cells']) == 24, f"expected the 24-cell base, got {len(nb['cells'])}"
    assert nb['cells'][0]['cell_type'] == 'markdown'
    assert nb['cells'][19]['cell_type'] == 'markdown', 'cell 19 is the inference intro'
    assert nb['cells'][23]['cell_type'] == 'code'
```

Plus a `main()` that walks `fails` and prints `PASS`/`FAIL` per check, exiting non-zero on any failure — matching the existing suite style (no pytest in this repo).

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_assemble.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'assemble'`

- [ ] **Step 3: Implement `assemble.py`**

```python
REPO = '/home/gb/orca/BatSpot'
BASE = os.path.join(REPO, 'batspot-train.ipynb')
OUT  = os.path.join(REPO, 'batspot-train-merged.ipynb')
CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cells')

REPLACE: dict[int, str] = {}      # base cell index -> cells/
APPEND_CODE: list[str] = []       # cells/, code
APPEND_MD:   list[str] = []       # cells/, markdown
```

`to_source` splits on `\n` and returns `[ln + '\n' for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])`.

`build(out_path)` loads `BASE`, asserts it has 24 cells, replaces each `REPLACE` entry (asserting the index is a code cell), appends `APPEND_MD` and `APPEND_CODE` entries (asserting cell type), and writes compact JSON. Cell 0's markdown is replaced via `REPLACE` too, so the pipeline list is regenerated rather than patched.

**`README.md`** records: the notebook is generated, never hand-edit it, edit `cells/` and re-run `assemble.py`; the two parents are read-only; the test suites read cells from the delivered notebook via `NEW_CELLS`.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python notebook_build/tests/test_assemble.py`
Expected: 3 PASS, exit 0

- [ ] **Step 5: Verify the empty build reproduces the base byte-for-byte in content**

Run: `venv/bin/python notebook_build/assemble.py && python3 -c "import json;a=json.load(open('batspot-train.ipynb'));b=json.load(open('batspot-train-merged.ipynb'));print('cells',len(a['cells']),len(b['cells']));print('identical',[c['source'] for c in a['cells']]==[c['source'] for c in b['cells']])"`

Expected: `cells 24 24` and `identical True`. This proves the scaffold is a faithful pass-through before any port lands.

- [ ] **Step 6: Commit**

```bash
git add notebook_build batspot-train-merged.ipynb
git commit -m "build: notebook scaffold — assemble.py generates merged notebook from cell sources"
```

---

### Task 2: Port Space Bunny's test suites to the merged notebook

**Files:**
- Create: `notebook_build/tests/extract_cells.py`
- Create: `notebook_build/tests/test_fixes.py`, `test_dataset.py`, `test_b5_labels.py`, `test_metric_guard.py`, `test_cell7.py`, `test_bug1.py`, `test_recording_id.py`
- Source: `/home/gb/batspot_gpu_experiments/spacebunny_tests/*.py`

**Interfaces:**
- Consumes: `batspot-train-merged.ipynb` (Task 1).
- Produces: `extract_cells.extract(notebook_path: str, out_dir: str) -> dict[int, str]` writing `src_NN.py` per code cell and `md_NN.md` per markdown cell, returning the index→filename map. Every suite calls this, so a suite can never test a stale copy.

- [ ] **Step 1: Write `extract_cells.py`, then port `test_metric_guard.py` as the pilot**

`test_metric_guard.py` is the smallest suite (16 assertions) and touches only the config cell, so it proves the extraction contract before six more suites depend on it.

- [ ] **Step 2: Write the failing test**

In `test_metric_guard.py`, assert the extraction itself is the test:

```python
def test_extraction_reads_the_delivered_notebook():
    idx = extract(os.path.join(REPO, 'batspot-train-merged.ipynb'), WORK)
    assert 1 in idx, 'config cell not extracted'
    src = open(idx[1]).read()
    assert 'SELECT_METRIC' in src, 'extracted cell 1 is not the config cell'
```

- [ ] **Step 3: Run it to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_metric_guard.py`
Expected: FAIL — `extract` undefined / config cell still the base one without the assert

- [ ] **Step 4: Implement `extract_cells.py`**

```python
def extract(notebook_path: str, out_dir: str) -> dict[int, str]:
    """Write each cell of `notebook_path` to out_dir; return {index: written path}."""
```
Markdown → `md_{i:02d}.md`, code → `src_{i:02d}.py`. Create `out_dir` if absent. Strip outputs.

- [ ] **Step 5: Port the remaining six suites**

Copy each from `spacebunny_tests/`, replacing the hard-coded `/tmp/opencode/base` + `/tmp/opencode/new` paths with `extract_cells.extract()` against the merged notebook. Preserve their `check()`/`fails` harness style and assertion counts (122 total: 27/21/18/16/15/14/11).

**Do not** copy `assemble.py` or `redgreen.py` from that directory — Task 1 and Task 13 supersede them.

- [ ] **Step 6: Run all seven suites**

Run: `cd notebook_build/tests && for t in test_*.py; do ../../venv/bin/python "$t"; done`
Expected: each prints its PASS lines and exits 0. Suites touching cells not yet ported (dataset, discovery, train, cascade, summary) will FAIL — that is expected and correct; they go green in Tasks 3–8. `test_metric_guard.py` must pass fully now if its target cell is unported only in ways Tasks 3+ fix; if it fails on the base cell, that is the B3 bug it was written to catch, and it goes green in Task 5.

- [ ] **Step 7: Commit**

```bash
git add notebook_build/tests
git commit -m "test: port 7 suites (122 assertions) to read cells from the merged notebook"
```

---

### Task 3: Port the dataset cell (index 5) — cache hardening and window fixes

**Files:**
- Create: `notebook_build/cells/src_05.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE[5] = 'src_05.py'`
- Test: `notebook_build/tests/test_dataset.py`, `test_fixes.py`

**Interfaces:**
- Consumes: base cell 5 (`WindowedBatDataset`, `predict_proba`, `_window`, `train_window_starts`).
- Produces: `WindowedBatDataset(file_names, config, augment=False, split='train')`, `predict_proba(model, dataset, device, batch_size=64) -> (probs: np.ndarray, labels: np.ndarray)`, and a module-level `FAILED_WAVS: list[str]` recording every dropped file. Task 4 reads `FAILED_WAVS` to align detector and classifier lists.

- [ ] **Step 1: Write the failing tests**

Add to `test_dataset.py`:

```python
def test_tied_scores_keep_exactly_top_fraction():
    """Review Focus 5: 100 tied windows must keep 20, not all 100."""
    starts = module.train_window_starts(np.ones(100), 30, 3, 0.20)
    assert len(starts) == 20, f'kept {len(starts)} of 100 tied windows'

def test_distinct_scores_keep_true_top_fraction():
    scores = np.arange(100, dtype=float)
    starts = module.train_window_starts(scores, 30, 3, 0.20)
    assert len(starts) == 20

def test_short_clip_not_padded_before_minmax():
    """Review Focus 1: 22 frames into a 30-frame window."""
    win = np.linspace(-100.0, 0.0, 22)[:, None].repeat(4, axis=1)
    out = module._window(win, 30)
    assert out.shape[0] == 30
    assert abs(out[:22].max() - 1.0) < 1e-6, 'real signal must reach 1.0'
    assert out[22:].max() == 0.0, 'padded rows must be 0.0, not the window max'

def test_bad_wav_dropped_and_lists_stay_aligned():
    """Review Focus 2: one undecodable wav is listed, not fatal."""
    # build 3 real clips + 1 truncated wav; assert len(dataset) == 3
    # and 'truncated' in module.FAILED_WAVS[0]

def test_cache_invalidated_when_source_wav_changes():
    """Stale-cache revert target: same mtime+size guard."""
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python notebook_build/tests/test_dataset.py`
Expected: FAIL on `test_tied_scores_keep_exactly_top_fraction` (keeps 100, not 20) and `test_short_clip_not_padded_before_minmax` (padded region 1.0, real max 0.4192)

- [ ] **Step 3: Implement `src_05.py`**

Start from base cell 5. Apply, each as a single-occurrence asserted substitution:

- `train_window_starts`: replace the percentile threshold with exact `ceil(top_frac · n)` by rank, `np.argsort(-scores, kind='stable')[:k]`, sorted ascending.
- `_window`: `win = minmax_normalize(win)` then `pad_window(win, seq_len)` (official order, `animal_spot/data/audiodataset.py:622-629`). Count sub-window clips at construction and print.
- Cache: sidecar `.meta.json` holding source wav size+mtime, validated on load; source directory hashed into the cache filename; `_CACHE_VERSION` derived from `inspect.getsource(clip_to_db_spectrogram)` with a fallback constant; `np.save` via `.tmp` + `os.replace`; `np.nan_to_num` on the dB array; per-instance lazy mmap cache instead of `np.load` per `__getitem__`.
- Per-file `try/except` around decode + STFT; append the path to `FAILED_WAVS` and continue.
- `predict_proba`: raise `RuntimeError` listing offending files when any clip produced zero windows.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_dataset.py && venv/bin/python notebook_build/tests/test_fixes.py`
Expected: all PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_05.py notebook_build/assemble.py
git commit -m "port: dataset cell — cache hardening, tie-safe window selection, min-max before pad"
```

---

### Task 4: Port the data-discovery cell (index 6) — `recording_id` and per-class counts

**Files:**
- Create: `notebook_build/cells/src_06.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE[6] = 'src_06.py'`
- Test: `notebook_build/tests/test_recording_id.py`, `test_cell7.py`

**Interfaces:**
- Consumes: base cell 6 (split construction, `_TAPE_RE`, leakage check).
- Produces: `recording_id(path: str) -> str` and `LEAKAGE: dict` holding `{'n_test_sharing': int, 'n_test': int, 'n_groups': int, 'class_recordings': dict[str, int]}` for Task 13's reporting. Keeps `SPLIT_BY_RECORDING` and `RECORDING_SPLIT_SCOPE='per_species'`.

- [ ] **Step 1: Write the failing tests**

In `test_recording_id.py`:

```python
def test_extractor_layout_shares_a_group_across_species():
    """Review Focus 3: Cell 7 writes species-bat_20260521_204000_sel_100_200.wav."""
    a = module.recording_id('/d/acsh-bat_20260521_204000_a_100_200.wav')
    b = module.recording_id('/d/rhro-bat_20260521_204000_b_300_400.wav')
    assert a == b == '20260521-204000', f'got {a!r} and {b!r}'

def test_raven_layout_unchanged():
    assert module.recording_id('/d/acsh-bat_6431090_2026_20260525-192000_96139_96154.wav') \
        == '20260525-192000'

def test_one_clip_per_group_warns():
    """n_groups == n_files must be loud, not a silent false all-clear."""
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_recording_id.py`
Expected: FAIL on `test_extractor_layout_shares_a_group_across_species` — the hyphen-only regex falls back to the whole stem, so the two clips get different groups

- [ ] **Step 3: Implement `src_06.py`**

Replace `_TAPE_RE` with `re.compile(r'(\d{8})[-_](\d{6})')` and `recording_id` with a `.search()` over the stem returning `f'{g1}-{g2}'`, else the stem. Add the `n_groups == n_files` warning and the per-class recording-count table. Keep the leakage check always-on and keep `RECORDING_SPLIT_SCOPE='per_species'` verbatim from the base cell.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_recording_id.py && venv/bin/python notebook_build/tests/test_cell7.py`
Expected: all PASS, exit 0

- [ ] **Step 5: Verify on real data that the fix changes nothing here**

Run: `venv/bin/python notebook_build/tests/test_cell7.py`
Expected: 28 recording groups on the current dataset, identical before and after the regex change. **If this reports a different group count, stop** — the data layout differs from what the notes assume and `AGENTS.md` needs updating before proceeding.

- [ ] **Step 6: Commit**

```bash
git add notebook_build/cells/src_06.py notebook_build/assemble.py
git commit -m "port: data discovery — recording_id accepts [-_], per-class recording counts, loud no-grouping warning"
```

---

### Task 5: Port the training cell (index 8) — B1, B3, real curves

**Files:**
- Create: `notebook_build/cells/src_08.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE[8] = 'src_08.py'`
- Test: `notebook_build/tests/test_bug1.py`, `test_metric_guard.py`, `test_fixes.py`

**Interfaces:**
- Consumes: base cell 8 (`set_seed`, `make_weighted_sampler`, `train_model`).
- Produces: `train_model(model, train_loader, val_fn, config, device, save_path=None) -> (best_state: dict, history: dict, best_epoch: int)`, where `val_fn() -> (probs, labels)` and `best_state` is always the best-validation snapshot. Adds `REPORT_TOPK_MEAN` instrumentation to the printed history. Task 8 calls `train_model` with the same signature for every seed and ablation, and Task 9 calls it with a modified `config` dict — so `config` must stay a plain dict read by key, never a dataclass or closure-captured value.

- [ ] **Step 1: Write the failing test**

In `test_bug1.py`, the synthetic problem whose validation rises then falls (best ≠ last epoch):

```python
def test_returns_best_weights_not_last_when_save_path_is_none():
    model, loader, history, best = run_case(save_path=None)
    assert abs(rescore(model) - history['best_score']) < 1e-9, (
        'last-epoch weights returned with the best score reported')

def test_first_validation_of_exactly_zero_is_captured():
    """best_score must start at -inf; a first validation scoring 0.0 must be beatable."""
```

In `test_metric_guard.py`, the mistyped-metric case:

```python
def test_unknown_selection_metric_raises_not_silently_defaults():
    with pytest_raises(KeyError):
        select_score(SELECT_METRIC='balanced_acc', val_acc=0.4, val_bal=0.52)
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python notebook_build/tests/test_bug1.py`
Expected: FAIL — reported best 0.3690, re-scored 0.3214

- [ ] **Step 3: Implement `src_08.py`**

- Remove both `if save_path:` gates around the snapshot and restore.
- `best_score = -inf`; snapshot with `{k: v.detach().clone() for k, v in sd.items()}`; restore unconditionally.
- Selection via dict lookup `{'balanced_accuracy': val_bal, 'accuracy': val_acc}[SELECT_METRIC]`, so an unhandled value is a `KeyError`.
- `no_improve += config['epochs_per_eval']` (raw epochs).
- `save_path` now genuinely writes a 3-panel figure (train loss, validation with both metrics, LR) via `fig.savefig`, titled with argmax and top-k mean.
- Print `REPORT_TOPK_MEAN` diagnostics: argmax, mean of top-k, mean of all, and the gap expressed in val clips (one clip = `0.5 * (1/n_pos + 1/n_neg)`).
- Add `set_seed(SEED)` at the top.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebook_build/tests && for t in test_bug1.py test_metric_guard.py test_fixes.py; do ../../venv/bin/python "$t"; done`
Expected: all PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_08.py notebook_build/assemble.py
git commit -m "port: training cell — unconditional best-weight restore, dict-lookup metric, real savefig"
```

---

### Task 6: Port the cascade cell (index 13) — prior-correction machinery

**Files:**
- Create: `notebook_build/cells/src_13.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE[13] = 'src_13.py'`
- Test: `notebook_build/tests/test_prior.py` (new)

**Interfaces:**
- Consumes: base cell 13 (`run_combined_pipeline`, val-tuned threshold over grid 0.05–0.95).
- Produces: `prior_correct_probs(probs: np.ndarray, prior_eval: np.ndarray, prior_train: np.ndarray) -> np.ndarray`, renormalised to sum 1 per row. Task 11 consumes it at inference with a **user-supplied** prior.

- [ ] **Step 1: Write the failing test**

```python
def test_renormalises_to_one():
    p = np.array([[0.7, 0.3], [0.2, 0.8]])
    out = module.prior_correct_probs(p, np.array([0.9, 0.1]), np.array([0.5, 0.5]))
    assert np.allclose(out.sum(1), 1.0)

def test_noop_when_priors_match():
    p = np.array([[0.7, 0.3]])
    assert np.allclose(module.prior_correct_probs(p, np.array([0.5, 0.5]), np.array([0.5, 0.5])), p)

def test_scales_odds_by_the_prior_ratio():
    """50/50 train -> 19/81 eval must shift odds by exactly (0.807/0.5)/(0.193/0.5)."""
    out = module.prior_correct_probs(np.array([[0.5, 0.5]]),
                                     np.array([0.193, 0.807]), np.array([0.5, 0.5]))
    odds_before, odds_after = 1.0, out[0, 1] / out[0, 0]
    assert abs(odds_after / odds_before - (0.807 / 0.5) / (0.193 / 0.5)) < 1e-6
```

**The third test is the trap that caught a real bug.** An earlier draft computed the prior from a boolean `is_noise` list without inverting it, producing `noise=0.807, call=0.193` and making the correction **anti-corrective**. This assertion pins the direction.

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_prior.py`
Expected: FAIL — `prior_correct_probs` undefined

- [ ] **Step 3: Implement `src_13.py`**

Add `prior_correct_probs` (`p_eval ∝ p_train * prior_eval/prior_train`, renormalised). **Default `PRIOR_CORRECT_GATE = False`.** Phase 1 and 2 report **raw** thresholds only, preserving comparability with `AGENTS.md` §8–§10. Print both raw and corrected when the flag is on. Keep the val-tuned threshold search over 0.05–0.95.

Do **not** apply prior correction to the classifier: 10 of its 21 errors are false `acsh`, the most common class, so a prior shift pushes the wrong way.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_prior.py`
Expected: 3 PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_13.py notebook_build/tests/test_prior.py notebook_build/assemble.py
git commit -m "port: cascade cell — prior-corrected gate machinery, off by default, raw thresholds reported"
```

---

### Task 7: Port the two summary cells (indices 16, 17) — B5 labelling

**Files:**
- Create: `notebook_build/cells/src_16.py`, `src_17.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE[16]`, `REPLACE[17]`
- Test: `notebook_build/tests/test_b5_labels.py`

**Interfaces:**
- Consumes: base cells 16 (load and evaluate existing models) and 17 (results summary).
- Produces: unchanged signatures; cell 17 additionally honours `AUC_NO_SIGNAL = 0.05` and prints per-row `kind` and `n_files`. Task 12 prints the merged summary.

- [ ] **Step 1: Write the failing test**

```python
def test_signal_free_model_is_marked():
    """A model answering 'not noise' for all 145 clips scores 0.8069 accuracy, AUC 0.4924."""
    val_results = [{'name': 'official_classifier_m09', 'kind': 'classifier',
                    'n_files': 145, 'test_acc': 117/145, 'test_bal_acc': 0.50, 'test_auc': 0.4924}]
    out = capture_stdout(lambda: exec_cell(17, val_results))
    assert 'NO SIGNAL' in out, 'a model with AUC ~0.5 must be marked'
    assert 'kind' in out and 'n_files' in out
    assert 'official official_' not in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_b5_labels.py`
Expected: FAIL — no `NO SIGNAL` marker, doubled `official official_` prefix present

- [ ] **Step 3: Implement both cells**

Cell 16: `n_freq_bins` via `predict.py`'s fallback chain (`num_mels` then `n_freq_bins`) with a warning when `freq_compression != 'linear'`; confusion-matrix names indexed by output width so `len(names) == C`; guarded `roc_auc_score`; rename `best_acc`/`best_val_acc` to reflect that they hold balanced accuracy.

Cell 17: add `kind` and `n_files` columns, `NO SIGNAL (AUC~0.5)` marker, group rows by denominator, remove the doubled prefix, close with a balanced-accuracy gain line for official → fine-tuned.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_b5_labels.py`
Expected: all PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_16.py notebook_build/cells/src_17.py notebook_build/assemble.py
git commit -m "port: summary cells — kind/n_files columns, NO SIGNAL marker, AUC and naming guards"
```

---

### Task 8: Phase 2 cell — multi-seed spread and grouped retrain

**Files:**
- Create: `notebook_build/cells/src_24.py`
- Modify: `notebook_build/assemble.py` — add `APPEND_CODE` entry, placed immediately after base cell 18
- Test: `notebook_build/tests/test_phase2.py` (new)

**Interfaces:**
- Consumes: `train_model` (Task 5), `WindowedBatDataset` (Task 3), `evaluate_model` (base cell 11), `recording_id` (Task 4), `RUN_MEASURE` and `ABLATION_SEEDS` from the config cell.
- Produces: `PHASE2: dict` with `{'per_seed': {seed: {name, test_acc, test_bal_acc, test_auc}}, 'mean': {name: (m, sd)}, 'noise_floor': {name: 2*sd}, 'grouped': {name: {test_acc, test_bal_acc, vanished: list[str]}}}`. Task 9 reads `PHASE2['noise_floor']` to label its deltas, so this key must exist even when `RUN_MEASURE` is False (set to `None`).

- [ ] **Step 1: Write the failing test**

```python
def test_noise_floor_is_two_sd():
    PHASE2 = {'per_seed': {'classifier': [0.85, 0.87, 0.86]}, 'noise_floor': {'classifier': 0.02}}
    assert abs(PHASE2['noise_floor']['classifier'] - 2 * np.std([0.85, 0.87, 0.86], ddof=1)) < 1e-9

def test_phase2_does_not_disturb_phase1_models():
    """RUN_MEASURE must not mutate cls_model or det_results."""
    before = id(cls_model)
    run_phase2()
    assert id(cls_model) == before

def test_grouped_split_has_zero_recording_overlap():
    """The whole point: assert, don't assume."""
    tr, te = grouped_fold(seed=42)
    assert not (set(recording_id(f) for f in tr) & set(recording_id(f) for f in te))
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_phase2.py`
Expected: FAIL — cell not present

- [ ] **Step 3: Implement `src_24.py`**

Guard the whole cell on `RUN_MEASURE`. Train the classifier and the **best mic only** (`GROUPED_MICS`) for each seed in `ABLATION_SEEDS = [42, 43, 44]`, re-splitting and re-initialising each time via a local `_phase2_split(seed)` mirroring Cell 7's 70/15/15 recipe. Then the grouped retrain with `StratifiedGroupKFold` at `RECORDING_SPLIT_SCOPE`, asserting zero overlap and listing classes absent from the grouped fold. Use `_`-prefixed local names throughout so nothing can shadow phase-1 state. Print mean ± sd and `noise_floor`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_phase2.py`
Expected: all PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_24.py notebook_build/tests/test_phase2.py notebook_build/assemble.py
git commit -m "feat: phase 2 — multi-seed mean+/-sd, noise floor, recording-grouped retrain"
```

---

### Task 9: Phase 3 cell — the four opt-in experiments

**Files:**
- Create: `notebook_build/cells/src_26.py`
- Modify: `notebook_build/assemble.py` — add `APPEND_CODE` entry
- Test: `notebook_build/tests/test_phase3.py` (new)

**Interfaces:**
- Consumes: everything Task 8 has, plus `USE_AUGMENTATION`, `DET_CONFIG`, `CLS_CONFIG` (copied, never mutated in place), `load_model_from_pk`, `build_model`.
- Produces: `PHASE3: list[dict]`, one row per experiment: `{'name', 'baseline', 'variant', 'delta', 'noise_floor', 'verdict'}` where `verdict` is `'real'`, `'within noise'` or `'worse'`. Task 12 renders it.

- [ ] **Step 1: Write the failing test**

```python
def test_delta_is_labelled_against_the_noise_floor():
    row = label_delta(name='label_smoothing', baseline=0.851, variant=0.856, sd=0.012)
    assert row['verdict'] == 'within noise'   # delta 0.005 < 2*0.012

def test_window_length_experiment_is_labelled_a_known_negative():
    row = [r for r in PHASE3 if r['name'] == 'window_60ms'][0]
    assert row['prior_evidence'] == '40 ms 0.852 vs 20 ms 0.853; 60 ms 0.821'
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_phase3.py`
Expected: FAIL — cell not present

- [ ] **Step 3: Implement `src_26.py`**

Guard on `RUN_EXPERIMENTS` (default `False`). Four A/Bs against the phase-1 baseline on the **same** split, deep-copying the config dict per variant:

1. `USE_AUGMENTATION` False vs True, classifier only.
2. Plain `CrossEntropyLoss()` vs `label_smoothing=0.1`, classifier only.
3. `sequence_len` 20 vs 40 vs 60 ms, classifier only. **Print the prior evidence** (40 ms 0.852 vs 20 ms 0.853; 60 ms 0.821) beside the fresh result so it is read as a re-test, not a new claim.
4. **250 kHz detector**: fine-tune a binary head from the official 250 kHz classifier's encoder (`classifier_path`), `sr=250000`, `fmax=125000`. Score it **against the 192 kHz detectors specifically on rhle/rhro recall**, since that is the hypothesis. Never mutate `DET_CONFIG`.

Each row records `prior_evidence` and is labelled by `label_delta` against `PHASE2['noise_floor']`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_phase3.py`
Expected: all PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_26.py notebook_build/tests/test_phase3.py notebook_build/assemble.py
git commit -m "feat: phase 3 — augmentation/label-smoothing/window-length A/Bs and the 250 kHz detector"
```

---

### Task 10: 250 kHz detector config entry

**Files:**
- Create: `notebook_build/cells/src_01.py` (full merged config cell — Tasks 11 and 13 also edit this)
- Modify: `notebook_build/assemble.py` — add `REPLACE[1] = 'src_01.py'`
- Test: `notebook_build/tests/test_config.py` (new)

**Interfaces:**
- Consumes: base cell 2.
- Produces: every flag the later cells read — `RUN_MEASURE=True`, `RUN_EXPERIMENTS=False`, `RUN_PHASE4=False`, `ABLATION_SEEDS=[42,43,44]`, `GROUPED_MICS`, `PRIOR_CORRECT_GATE=False`, `REPORT_TOPK_MEAN=3`, `AUC_NO_SIGNAL=0.05`, `SEED=42`, `RUN_CLIP_EXTRACTION=False`, plus the unchanged `DET_CONFIG`/`CLS_CONFIG`/`WINDOW_MODE`/`WINDOW_STRIDE`/`TRAIN_TOP_FRAC`/`TEST_TOPK`/`SELECT_METRIC`/`SPLIT_BY_RECORDING`/`RECORDING_SPLIT_SCOPE`/`USE_AUGMENTATION` and the four pretrained-model paths. Also `check_bands(configs: dict[str, dict]) -> None`, raising `AssertionError` naming the offending config, so Task 10's test calls it directly.

- [ ] **Step 1: Write the failing test**

```python
def test_gate_defaults_match_the_spec():
    assert cfg.RUN_MEASURE is True
    assert cfg.RUN_EXPERIMENTS is False
    assert cfg.RUN_PHASE4 is False
    assert cfg.PRIOR_CORRECT_GATE is False

def test_bands_are_unchanged():
    assert (cfg.DET_CONFIG['sr'], cfg.DET_CONFIG['fmin'], cfg.DET_CONFIG['fmax']) == (192000, 1000, 95000)
    assert (cfg.CLS_CONFIG['sr'], cfg.CLS_CONFIG['fmin'], cfg.CLS_CONFIG['fmax']) == (250000, 10000, 125000)

def test_250khz_detector_band_is_at_nyquist():
    """Not a fmax raise: 250 kHz is the sample rate that makes the band reachable."""
    d = cfg.DET250_CONFIG
    assert d['sr'] == 250000 and d['fmax'] == 125000
    assert d['fmax'] <= d['sr'] / 2, 'fmax may never exceed Nyquist'
    assert cfg.DET_CONFIG['fmax'] <= cfg.DET_CONFIG['sr'] / 2

def test_nyquist_violation_raises_at_config_time():
    """Enforced in the cell, not just asserted here: predict.py uses fmax unclamped,
    so an out-of-band value would silently return wrong frequencies."""
    with pytest_raises(AssertionError):
        check_bands({'sr': 192000, 'fmax': 110000})
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_config.py`
Expected: FAIL — no `RUN_MEASURE`, no `DET250_CONFIG`

- [ ] **Step 3: Implement `src_01.py`**

Start from base cell 2. Keep every existing line. Add the phase gates with the defaults above and a comment per gate giving its runtime. Add `DET250_CONFIG` as a copy of `CLS_CONFIG`'s band with `num_classes: 2`, `classes: ['noise','target']`, `base_lr: 1e-4`, `n_epochs: 100` — plus an inline comment recording that `fmax=95000` on the 192 kHz detector is already within 1 kHz of Nyquist, so `rhle`/`rhro` CF calls are aliased rather than cropped, and the sample rate is the only available lever.

Strengthen the `SELECT_METRIC` assert into the two-line form that names the failure mode.

**Enforce the Nyquist invariant** in the config cell, because the entire 250 kHz
substitution rests on it and nothing else would catch a future edit that reintroduces
the impossible band:

```python
for _n, _c in (('DET', DET_CONFIG), ('CLS', CLS_CONFIG), ('DET250', DET250_CONFIG)):
    assert _c['fmax'] <= _c['sr'] / 2, (
        f"{_n}: fmax={_c['fmax']} exceeds Nyquist for sr={_c['sr']}. Raise sr, not fmax.")
```

This is the runtime form of `AGENTS.md`'s finding that `animal_spot/predict.py:214` and
`animal_spot/data/audiodataset.py` use `fmax` unclamped, so an out-of-band value would not
error at prediction time — it would silently return wrong frequencies.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_config.py && venv/bin/python notebook_build/tests/test_metric_guard.py`
Expected: all PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_01.py notebook_build/tests/test_config.py notebook_build/assemble.py
git commit -m "port: config cell — phase gates, DET250_CONFIG with the Nyquist rationale"
```

---

### Task 11: Phase 4 — keep cells 19–23 verbatim, add threshold calibration

**Files:**
- Modify: `notebook_build/assemble.py` — `REPLACE[20] = 'src_20.py'` (config cell only), `APPEND_CODE += ['src_27.py']`
- Test: `notebook_build/tests/test_inference.py` (new)

**Interfaces:**
- Consumes: base cells 19–23 unchanged; `prior_correct_probs` (Task 6); `select_metric` thresholds.
- Produces: `calibrate(recording_paths, truth_dir, settings) -> list[dict]`, one row per `(threshold, merge_gap)` with `boxes_found`, `species_correct`, `noise_hit`, `precision`, `recall`. Writes `/kaggle/working/batspot_detections.txt` unchanged in format.

- [ ] **Step 1: Write the failing test**

```python
def test_low_sample_rate_is_reported():
    """Review Focus 4: a 192 kHz upload cannot show calls above 96 kHz."""
    out = capture_stdout(lambda: report_input([fake_wav_at_192kHz()]))
    assert 'Nyquist' in out or 'cannot be seen' in out

def test_merge_and_split_roundtrip():
    """Adjacent positive windows within merge_gap merge into one selection; a run longer
    than max_selection_s splits at the widest internal silence. No window may be lost."""
    sels = build_selections(windows, merge_gap_s=0.1, max_selection_s=1.0)
    assert sum(len(s['windows']) for s in sels) == len(windows)

def test_calibration_without_truth_tables_says_precision_unmeasurable():
    out = capture_stdout(lambda: calibrate(paths, truth_dir=None, settings=SETTINGS))
    assert 'unmeasurable' in out.lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python notebook_build/tests/test_inference.py`
Expected: FAIL — no `build_selections`, no `calibrate`

- [ ] **Step 3: Implement**

Copy `notebook_build/cells/src_20.py` from base cell 20 unchanged except: wrap in `if RUN_PHASE4:`; add `INFER_EXPECTED_NOISE_FRACTION = 0.95` with a comment that this is the **deployment** prior supplied by the user and never inferred from a split; print raw and prior-corrected thresholds side by side.

Write `src_27.py`: sweep `threshold ∈ {0.3, 0.5, 0.7}` × `merge_gap ∈ {0.1, 0.2}` reusing cell 24's `_read_truth_table` and overlap matcher. With no truth tables, print selections-per-hour against threshold and state plainly that precision is unmeasurable because the tables mark only a fraction of calls. Emit no new file format — reuse `write_detections`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_inference.py`
Expected: all PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_20.py notebook_build/cells/src_27.py notebook_build/tests/test_inference.py notebook_build/assemble.py
git commit -m "feat: phase 4 — threshold calibration on real recordings, supplied deployment prior"
```

---

### Task 12: Phase-gate guards and markdown

**Files:**
- Create: `notebook_build/cells/md_00.md`, `md_23.md`, `md_25.md`
- Modify: `notebook_build/assemble.py` — `REPLACE[0]`, `APPEND_MD`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: the notebook's three markdown cells. `md_00.md` is the pipeline map; `md_23.md` introduces phase 2; `md_25.md` introduces phase 3.

- [ ] **Step 1: Write the failing test**

```python
def test_markdown_states_the_noise_floor():
    md = open(idx[0]).read()
    assert 'noise floor' in md.lower()
    assert '5 pt' in md and '3 pt' in md

def test_markdown_warns_about_min_max_norm():
    assert 'min_max_norm' in md
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_assemble.py`
Expected: FAIL on `test_markdown_states_the_noise_floor`

- [ ] **Step 3: Write the markdown**

`md_00.md`: the nine pipeline steps plus the four phase gates and their runtimes; the windowing mechanism and the ×43–56 distinct-inputs figure; the noise floor (≈5 pt detectors, ≈3 pt classifier); which numbers are in-split vs grouped, and that grouped is the honest new-recording estimate (detector balanced accuracy 0.64–0.71); `min_max_norm=true` required at prediction; that the 40/60 ms and fmax experiments re-test known negatives.

`md_23.md`: why phase 2 exists — one 145-clip split, spread larger than every argued delta.

`md_25.md`: why phase 3 exists, and that the 250 kHz detector replaces the impossible `fmax` raise.

- [ ] **Step 4: Run the full suite**

Run: `cd notebook_build/tests && for t in test_*.py; do ../../venv/bin/python "$t"; done`
Expected: every suite exits 0

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells notebook_build/assemble.py
git commit -m "docs: phase-gate markdown — pipeline map, noise floor, known-negative labels"
```

---

### Task 13: Red-green harness

**Files:**
- Create: `notebook_build/redgreen.py`

**Interfaces:**
- Consumes: `extract_cells.py`, all suites, the built notebook.
- Produces: a table of `{fix, suite, exit_code, n_failures, detected}`. Exit non-zero unless every revert is detected.

- [ ] **Step 1: Implement `redgreen.py`**

Revert **one fix at a time** in a scratch copy of the extracted cells, run the matching suite, assert it reports failures. Reverts apply **structurally** (locate the function/block by markers), not by exact string match — an earlier harness string-matched, silently failed to apply, and mis-reported detection because the suite then crashed for an unrelated reason.

Cases, one per fix: `save_path` gates `best_state`; pad before min-max (dataset + unit level); `SELECT_METRIC` silent fallback; positional `recording_id`; unlabelled summary rows; stale spectrogram cache; percentile-tie window selection; prior-correction direction.

Set `NEW_CELLS` in the subprocess environment and **assert** it is set — the prior harness omitted it, so two suites "passed" against unmutated cells and would have produced a false all-green.

- [ ] **Step 2: Run it**

Run: `venv/bin/python notebook_build/redgreen.py`
Expected: every row `YES`, exit 0. Any `NO` means that fix has no test proving it changes behaviour — fix the test, not the harness.

- [ ] **Step 3: Sanity-check that a no-op harness would be caught**

Run: `venv/bin/python notebook_build/redgreen.py --self-test`
Expected: the harness reports that a deliberately empty revert is detected as **not** detected, and exits non-zero. If a broken harness passes, the `YES` column means nothing.

- [ ] **Step 4: Commit**

```bash
git add notebook_build/redgreen.py
git commit -m "test: red-green harness — one revert per fix, structural patching, self-test"
```

---

### Task 14: Full local GPU run

**Files:**
- Create: `notebook_build/run_local.py`
- Create: `/tmp/opencode/merged_run.log` (not committed)

**Interfaces:**
- Consumes: the built notebook and its config cell.
- Produces: a log with per-cell timing and the full metric table; a comparison table against both parents.

- [ ] **Step 1: Implement `run_local.py`**

Port `/home/gb/batspot_gpu_experiments/run_notebook_locally.py`: `exec` every code cell in order, swapping only `/kaggle/` path strings and skipping only `!pip` lines. **Do not skip or alter any other cell.** Set `RUN_MEASURE=True`; run phase 3 in a second pass.

- [ ] **Step 2: Verify the CPU-only venv is not used**

Run: `venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
Expected: a CUDA build with `True`. The checked-in `venv/` is `2.14.0+cpu` and will silently run a 16-thread CPU job that pins the laptop near 100 °C and is ~75× slower. If it prints `False`, stop and create a CUDA venv — do not proceed on CPU.

- [ ] **Step 3: Run phase 1 + 2 end to end**

Run: `venv/bin/python notebook_build/run_local.py --phases 1,2 2>&1 | tee /tmp/opencode/merged_run.log`
Expected: every cell completes, 0 errors. Confirm the training curves PNGs exist — `save_path` previously wrote nothing at all.

- [ ] **Step 4: Run phase 3 in a second pass**

Run: `venv/bin/python notebook_build/run_local.py --phases 1,3 --experiments`
Expected: 0 errors. Every phase-3 row carries `prior_evidence` and a `verdict` labelled against the phase-2 noise floor.

- [ ] **Step 5: Compare against both parents**

Record in the log: classifier test accuracy, per-detector balanced accuracy and AUC, cascade alone vs gated, and phase-2 mean ± sd. Judge against the spread (m09 0.876–0.931, m11 0.876–0.924, classifier 0.848–0.876), not against a single parent's number.

**If a phase-1 metric falls outside its parent's range by more than the noise floor, stop and diagnose before continuing.** A ported fix must not move accuracy; the five review fixes were all verified latent on this data.

- [ ] **Step 6: Commit**

```bash
git add notebook_build/run_local.py
git commit -m "build: local GPU runner — execs every cell, swaps only Kaggle paths"
```

---

### Task 15: Record the merge in AGENTS.md

**Files:**
- Modify: `AGENTS.md` — append section 11

**Interfaces:**
- Consumes: the verified notebook and Task 14's numbers.
- Produces: the cell map, the merge decisions, and the verification counts.

- [ ] **Step 1: Write the section**

Document: the cell map (which cells were replaced, appended, kept verbatim); why the base is `batspot-train.ipynb`; the 18 ported fixes by name; the prior-correction resolution; the `fmax`→250 kHz substitution with the Nyquist arithmetic; the phase gates and runtimes; the measured noise floor and phase-2 mean ± sd; and the verification counts from `redgreen.py`.

State clearly that neither parent notebook was modified and that the merged notebook is **generated** — `notebook_build/cells/` is the source of truth.

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: AGENTS.md section 11 — merged notebook cell map, decisions, verification"
```

---

## Self-Review

**1. Spec coverage.** Every spec section maps to a task: §2 shape → Task 1, 10, 12; §3 merge map → Tasks 3–7 (ported), 1 (base preserved); §4 measurement → Task 8; §5 prior correction → Task 6, 11; §6 experiments incl. the fmax substitution → Tasks 9, 10; §7 inference → Task 11; §8 verification → Tasks 2, 13, 14; §9 risks → Task 3 Step 1 (list alignment), Task 14 Step 2 (CPU venv), Task 14 Step 5 (regression check); §10 documentation → Tasks 12, 15. No gaps.

**2. Step scan.** Every code step pins a signature and the values the spec fixes; every test step names the assertions. The one algorithm needing a body — exact-rank window selection — is given as `np.argsort(-scores, kind='stable')[:k]`, since "keep the top fraction" is otherwise ambiguous under ties.

**3. Type consistency.** `train_model(model, train_loader, val_fn, config, device, save_path=None)` is defined in Task 5 and consumed identically in Tasks 8, 9, 14. `prior_correct_probs(probs, prior_eval, prior_train)` defined in Task 6, consumed in Task 11. `PHASE2['noise_floor']` produced in Task 8, consumed in Task 9 and Task 12. `FAILED_WAVS` produced in Task 3, declared for Task 4. `build_selections(windows, merge_gap_s, max_selection_s)` and `calibrate(...)` defined and consumed in Task 11. `label_delta(name, baseline, variant, sd)` defined in Task 9. `extract(notebook_path, out_dir)` defined in Task 2, consumed by Task 13. Consistent.

**4. Review Focus.** All five are pinned: #1 → Task 3 `test_short_clip_not_padded_before_minmax`; #2 → Task 3 `test_bad_wav_dropped_and_lists_stay_aligned`; #3 → Task 4 `test_extractor_layout_shares_a_group_across_species`; #4 → Task 11 `test_low_sample_rate_is_reported`; #5 → Task 3 `test_tied_scores_keep_exactly_top_fraction`.

**5. Proportion.** 15 tasks, ~95 steps against a 10-section spec. Test code appears only where an assertion value is load-bearing; implementation bodies are signatures plus the substitutions the plan must pin.
