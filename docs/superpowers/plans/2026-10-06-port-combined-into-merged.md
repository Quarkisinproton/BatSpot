# Re-based Plan: port `batspot-train-combined.ipynb` into `batspot-train-merged.ipynb`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `batspot-train-merged.ipynb` carry the combined notebook's best parts — noise mixing, the 3-seed ensemble, the "unknown" answer, and its 4 bug fixes — plus close the 4 gaps the combined notebook still has.

**Architecture:** Unchanged from the original plan. `notebook_build/assemble.py` generates `batspot-train-merged.ipynb` from cell sources in `notebook_build/cells/`. Tasks 1–2 (generator, 7 test suites) are already committed and reused unchanged. This plan ports cell sources from `/home/gb/batspot_gpu_experiments/combined_2026-10-05/cells/` verbatim, then adds the 4 new cells the combined notebook lacks.

**Source of ported cells:** `/home/gb/batspot_gpu_experiments/combined_2026-10-05/cells/cNN.py` — the exact source of every cell in `batspot-train-combined.ipynb`. 3473 lines, already reviewed, carrying 4 bug fixes the original plan never found.

**Spec:** `docs/superpowers/specs/2026-10-05-merged-batspot-notebook-design.md`

## What changed from the original plan, and why

A parallel effort built `batspot-train-combined.ipynb` on 2026-10-05/06 (documented in `AGENTS.md` §11). It subsumes most of the original 15 tasks. User direction: port its best parts into `batspot-train-merged.ipynb`.

**Ported verbatim rather than re-derived.** Its cells are 3473 lines of reviewed code carrying four bug fixes the original plan never found:

| Bug | Symptom | Fix location |
|---|---|---|
| Clip extraction produced nothing | took `'_'.join(name.split('_')[:2])` = `acsh_devon` as the recording stamp, matching no audio file | `c07.py:15,31-39` |
| Cache tag changed under `exec` | `inspect.getsource` on exec'd code reads `__main__` lines of the wrong file | `c05.py:156-172` |
| Chunked resampling misaligned | excerpt start not on the L/M phase grid; 13–16 % of full scale on white noise at 250/500 kHz | `c21.py:124-138` |
| One corrupt file killed the listing cell | `sf.info` called unguarded in the folder summary | `c22.py:36-41` |

Re-deriving these would cost a day and risk reintroducing them.

**Corrections to my own earlier claims** (recorded in the ledger):

- I claimed the combined notebook *has* threshold calibration. **It does not** — a substring grep matched "calibrated" in a markdown caveat. Verified by reading the code.
- I reported `inspect.getsource` as "present ✅". It appears **only in a comment explaining why it was removed**.

**Defaults (user decision, 2026-10-07): noise mixing, 3-seed ensemble and "unknown" all ON by default.** The ensemble triples classifier training time (~3 × 3–6 min); accepted for stability — it removes the 3–5 point seed lottery that has made every comparison in this project unmeasurable.

## Already committed (Tasks 1–2, reused unchanged)

- `notebook_build/assemble.py` — generator; `REPLACE`/`APPEND_*` currently empty
- `notebook_build/tests/` — 127 checks across 7 suites, all reading cells from the generated notebook via `extract_cells.extract()`

## Global Constraints

- **Neither parent notebook is modified**: `batspot-train.ipynb` and `batspot_train(claude_bug_fixes _by__SPACE_BUNNY_MODEL).ipynb` are read-only. `batspot-train-combined.ipynb` and its `cells/` are read-only too.
- **Bands are fixed by the official `.pk` `dataOpts`:** detector `sr=192000, fmin=1000, fmax=95000`; classifier `sr=250000, fmin=10000, fmax=125000`; both `n_fft=256, hop_length=128, n_freq_bins=256, sequence_len=20 ms`.
- **`fmax <= sr/2` must hold and must be asserted.** 192 kHz Nyquist is 96 kHz and `fmax=95000` is already 1 kHz below it. `animal_spot/predict.py:214` reads `dataOpts["fmax"]` and `audiodataset.py` uses it unclamped, so an out-of-band value **silently returns wrong frequencies** — no error at prediction time. The combined notebook has no such guard (verified: its only `Nyquist` mentions are a comment and a runtime low-sample-rate warning).
- **Loss is unweighted `nn.CrossEntropyLoss()` with `WeightedRandomSampler`**, never both — double-counting gives a 4.18× noise penalty that collapses m03 to all-noise.
- **`base_lr` absolute**, never multiplied by batch size. Detector `1e-4`, classifier `3e-4`. `1e-2` is measured bad.
- **Selection/scheduling/early-stopping track balanced accuracy**; a constant predictor scores 0.5, not 0.807.
- **Patience counts RAW epochs** (`no_improve += epochs_per_eval`).
- **Architecture keys serialise as `shortcut.*`** per `animal_spot/models/residual_base.py:32`.
- **Ported cells keep their inline comments verbatim.** They carry the measured evidence for why each constant is what it is; stripping them would lose the justification.
- **Exported `.pk` must load in the BatSpot GUI / `predict.py`**; exported models need `min_max_norm=true` at prediction.
- **`SoftmaxEnsemble.forward` returns log-probabilities** (`c04.py:249-251`) so every `softmax(model(x))` caller keeps working. Any new consumer must go through `softmax()`, never `argmax` on the raw output.
- **No pytest, linter, or formatter** — this repo has none. Hand-rolled `check(name, cond)` harness.
- **Deterministic patching:** every string substitution asserts exactly one match, so a silent no-op is impossible.

## Review Focus

1. **A `.pk` written by the merged notebook that the GUI cannot load** — the export path is the one thing that must not regress. Verify with `animal_spot/predict.py`, not a self-contained load test.
2. **The "unknown" sidecar fitted for a different ensemble size.** `forward_with_embedding` concatenates member embeddings (3 members → 1536-d). `c20.py:126-128` catches a *filename* mismatch but **not a size mismatch** — refit after any ensemble-size change.
3. **A detector threshold that transfers.** The val-tuned threshold (`c13.py:19-25`) is **never used by inference**, which uses a fixed `INFER_DET_THRESHOLD = 0.5`. The model is trained at a 50/50 noise/call prior while data is 19/81, so the tuned value encodes the test mix. This is gap (b).
4. **A recording below 192 kHz.** Calls above its Nyquist are invisible and must be reported, not silently returned as noise.
5. **Noise mixing at eval time.** It is train-only (`c05.py:198` forces `noise_mix_prob=0` unless `train`); leaking it into val/test would inflate every reported number.

---

### Task 3: Port the config cell (combined `c01` + Nyquist guard)

**Files:**
- Create: `notebook_build/cells/src_01.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE[1] = 'src_01.py'`
- Test: `notebook_build/tests/test_config.py`

**Interfaces:**
- Consumes: `/home/gb/batspot_gpu_experiments/combined_2026-10-05/cells/c01.py` as the base text.
- Produces: `check_bands(configs: dict[str, dict]) -> None` raising `AssertionError` naming the offending config; every flag later cells read — `SEED=42`, `CLS_ENSEMBLE_SEEDS=[42,43,44]`, `UNKNOWN_DETECTION=True`, `UNKNOWN_KEEP_KNOWN=0.95`, `UNKNOWN_LABEL='unknown'`, `UNKNOWN_METHOD='maha'`, `AUC_NO_SIGNAL=0.05`, `REPORT_TOPK_MEAN=3`, `RUN_MEASURE=True`, `RUN_EXPERIMENTS=False`, `RUN_PHASE4=False`, `RUN_GROUPED_CHECK=True`, `GROUPED_MICS`, `noise_mix_prob=0.5`, `noise_mix_snr_db=(0.0,20.0)`, `label_smoothing=0.0`, `RUN_CLIP_EXTRACTION=False`, and unchanged `DET_CONFIG`/`CLS_CONFIG` bands, `WINDOW_MODE`, `WINDOW_STRIDE`, `TRAIN_TOP_FRAC`, `TEST_TOPK`, `SELECT_METRIC`, `SPLIT_BY_RECORDING`, `RECORDING_SPLIT_SCOPE`.

- [ ] **Step 1: Write the failing test**

```python
def test_nyquist_violation_raises():
    with raises(AssertionError):
        cfg.check_bands({'DET': {'sr': 192000, 'fmax': 110000}})

def test_real_bands_pass():
    cfg.check_bands({'DET': cfg.DET_CONFIG, 'CLS': cfg.CLS_CONFIG})

def test_defaults_are_the_chosen_ones():
    assert cfg.UNKNOWN_DETECTION is True and cfg.UNKNOWN_KEEP_KNOWN == 0.95
    assert cfg.CLS_ENSEMBLE_SEEDS == [42, 43, 44]
    assert cfg.DET_CONFIG['noise_mix_prob'] == 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_config.py`
Expected: FAIL — no `check_bands`, no `UNKNOWN_*`

- [ ] **Step 3: Implement**

Copy `c01.py` to `notebook_build/cells/src_01.py`. Add the guard and call it:

```python
def check_bands(configs):
    for name, c in configs.items():
        if c['fmax'] > c['sr'] / 2:
            raise AssertionError(
                f"{name}: fmax={c['fmax']} exceeds Nyquist for sr={c['sr']}. "
                "Raise sr, not fmax -- predict.py uses fmax unclamped, so an out-of-band "
                "value returns wrong frequencies with no error.")
check_bands({'DET': DET_CONFIG, 'CLS': CLS_CONFIG})
```

Keep every other line of `c01.py` verbatim, including its comments about which constants are measured and which are inherited.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_config.py`
Expected: 3 PASS, exit 0

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_01.py notebook_build/tests/test_config.py notebook_build/assemble.py
git commit -m "port: config cell — ensemble/unknown/noise-mix flags + Nyquist guard"
```

---

### Task 4: Port the architecture cell (combined `c04`)

**Files:**
- Create: `notebook_build/cells/src_04.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE[4] = 'src_04.py'`
- Test: `notebook_build/tests/test_arch.py`

**Interfaces:**
- Consumes: `c04.py` — `ResidualEncoder`/`Classifier`, `unwrap_model`, `SoftmaxEnsemble`, `model_members`, `forward_with_embedding`, `load_model_from_pk`, `load_any_model`.
- Produces: unchanged names/signatures. `Classifier.forward` stores the pooled 512-d vector at `_layer_output['hidden_layer_1']` (`c04.py:221`), which `forward_with_embedding` reads. **Task 5's dataset and Task 12's export depend on these exact names.**

- [ ] **Step 1: Write the failing test**

```python
def test_ensemble_returns_log_probs_so_softmax_callers_work():
    e = mod.SoftmaxEnsemble([m1, m2])
    x = torch.randn(4, 1, 256, 30)
    assert torch.allclose(e(x).exp(), torch.stack([m1(x), m2(x)]).softmax(-1).mean(0), atol=1e-6)

def test_forward_with_embedding_concatenates_per_member():
    probs, emb = mod.forward_with_embedding(ensemble_of_3, x)
    assert emb.shape[1] == 3 * 512
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_arch.py`
Expected: FAIL — the merged notebook's cell 4 has no `SoftmaxEnsemble`

- [ ] **Step 3: Implement**

Copy `c04.py` verbatim to `notebook_build/cells/src_04.py`. No edits — the port is the point.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_arch.py`
Expected: PASS. Also confirm the `shortcut.*` keys are unchanged from `animal_spot/models/residual_base.py` so official `.pk` files still load.

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_04.py notebook_build/tests/test_arch.py notebook_build/assemble.py
git commit -m "port: architecture cell — SoftmaxEnsemble, forward_with_embedding, unwrap_model"
```

---

### Task 5: Port the dataset cell (combined `c05`) — the largest port

**Files:**
- Create: `notebook_build/cells/src_05.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE[5] = 'src_05.py'`
- Test: `notebook_build/tests/test_dataset.py`, `test_fixes.py`

**Interfaces:**
- Consumes: `c05.py` — front end, windowing, `mix_background_db` (`:144`), `_code_fingerprint` (`:156`), `WindowedBatDataset` (`:175`), `predict_proba(..., return_embedding=False)` (`:332`), and the open-set block (`known_scores` `:382`, `fit_unknown_model` `:398`, `apply_unknown` `:424`, `save_unknown_model` `:431`, `load_unknown_model` `:436`).
- Produces: `mix_background_db(win_db, noise_db, snr_db, min_level_db=-100) -> np.ndarray`; `WindowedBatDataset(file_names, class_to_idx, config, train=False, augment=False, noise_mix_prob=0.0, noise_mix_snr_db=(0,20))`; `predict_proba(model, dataset, device, batch_size=256, return_embedding=False)`; the five open-set functions. Tasks 6, 12, 13 consume these.

- [ ] **Step 1: Write the failing test**

```python
def test_noise_mixing_hits_the_requested_snr():
    """The gain makes the window's mean power snr_db above the added noise's."""
    for snr in (0.0, 10.0, 20.0):
        out = mod.mix_background_db(sig_db, noise_db, snr)
        measured = 10*log10(mean(10**(out/10)) / mean(10**(out/10)))  # vs the same window unmixed
        assert abs(measured - snr) < 1e-2

def test_floor_is_reapplied():
    out = mod.mix_background_db(np.full((30,256), -20.0), np.full((30,256), -100.0), 0.0)
    assert out.min() >= -100.0

def test_noise_bank_contains_only_noise_clips():
    ds = mod.WindowedBatDataset(files, cidx, cfg, train=True, noise_mix_prob=0.5)
    assert all(get_class_from_filename(ds.file_names[i]) == 'noise' for i in ds._noise_ids)

def test_mixing_is_train_only():
    """Review Focus 5: leaking it into val/test would inflate every number."""
    ev = mod.WindowedBatDataset(files, cidx, cfg, train=False, noise_mix_prob=0.5)
    assert ev.noise_mix_prob == 0.0 and ev._noise_ids == []

def test_cache_version_changes_with_default_arguments():
    """A first version missed __defaults__ and the test caught it."""
    v1 = mod._code_fingerprint(f_defaults_a)
    v2 = mod._code_fingerprint(f_defaults_b)
    assert v1 != v2
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python notebook_build/tests/test_dataset.py`
Expected: FAIL — no `mix_background_db`, no `_code_fingerprint`

- [ ] **Step 3: Implement**

Copy `c05.py` verbatim to `notebook_build/cells/src_05.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebook_build/tests && ../../venv/bin/python test_dataset.py && ../../venv/bin/python test_fixes.py`
Expected: all PASS. The suites were written against the *old* dataset cell, so expect to reconcile assertions — port the suites' expectations to the new behaviour rather than weakening them. `test_fixes.py`'s cache assertions in particular were written for the pre-port cell.

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_05.py notebook_build/tests/ notebook_build/assemble.py
git commit -m "port: dataset cell — noise mixing, code-fingerprint cache, open-set scorers"
```

---

### Task 6: Port data discovery, model staging, export, evaluation and summary (combined `c06`, `c09`, `c14`, `c16`, `c17`)

**Files:**
- Create: `notebook_build/cells/src_06.py`, `src_09.py`, `src_14.py`, `src_16.py`, `src_17.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE` entries for 6, 9, 14, 16, 17
- Test: `test_cell7.py`, `test_recording_id.py`, `test_b5_labels.py`

**Interfaces:**
- Consumes: `c06.py` (unusable-clip pre-drop, class maps, 3 split modes, leakage check, tape-lookup baseline), `c09.py` (`.pk` staging under unique names, config-vs-dataOpts check), `c14.py` (`export_pk` on a CPU copy, ensemble + unknown sidecar export), `c16.py` (`num_mels` fallback, output-width names, `NO SIGNAL`), `c17.py` (summary with `kind`/`n_files`).
- Produces: unchanged signatures. `recording_id(path) -> str` accepting `[-_]` separators; `export_pk(...)` writing `classifier_250khz.pk` (single best member, for the GUI), `classifier_250khz_seed{seed}.pk` (every member), and `classifier_250khz_unknown.npz`.

- [ ] **Step 1: Write the failing test**

```python
def test_recording_id_accepts_both_separators():
    assert mod6.recording_id('/d/acsh-bat_20260521-204000_1_2_3.wav') == '20260521-204000'
    assert mod6.recording_id('/d/acsh-bat_20260521_204000_a_100_200.wav') == '20260521-204000'
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_recording_id.py`
Expected: FAIL against the merged notebook's cell 6

- [ ] **Step 3: Implement**

Copy all five cells verbatim into `notebook_build/cells/`, register them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebook_build/tests && ../../venv/bin/python test_cell7.py && ../../venv/bin/python test_recording_id.py && ../../venv/bin/python test_b5_labels.py`
Expected: all PASS — these are exactly the suites the earlier fix rounds made sensitive to cell 6/17's shape, so they must go green here.

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/ notebook_build/tests/ notebook_build/assemble.py
git commit -m "port: discovery, staging, export, evaluation and summary cells"
```

---

### Task 7: Port clip extraction (combined `c07`) — the extraction bug fix

**Files:**
- Create: `notebook_build/cells/src_07.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE[7] = 'src_07.py'`
- Test: `notebook_build/tests/test_extraction.py`

**Interfaces:**
- Consumes: `c07.py` — `_SEL_STAMP_RE = re.compile(r'(\d{8})_(\d{6})')` (`:15`), the faithful `export_clips.R` port.
- Produces: `extract_clips_from_selections(...)` writing names whose stamp is findable by Task 6's `recording_id`.

- [ ] **Step 1: Write the failing test**

```python
def test_extraction_finds_the_recordings():
    """The old code took 'acsh_devon' as the stamp and extracted nothing."""
    made, problems = extract(RAW_AUDIO_DIR, SELECTIONS_DIR, out_dir)
    assert len(made) == 20 and not problems, f'{len(made)} clips, problems: {problems[:3]}'

def test_written_names_parse_back_to_the_tape():
    assert {recording_id(p) for p in made} == {'20260429-192000', '20260521-204000'}
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_extraction.py`
Expected: FAIL — the merged notebook's cell 7 still has the `'_'.join(name.split('_')[:2])` bug

- [ ] **Step 3: Implement**

Copy `c07.py` verbatim.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_extraction.py`
Expected: 20/20 rows from 2 real tables, names parse back, first 200 samples bit-identical

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_07.py notebook_build/tests/test_extraction.py notebook_build/assemble.py
git commit -m "port: clip extraction — Raven timestamp stamp, fixes extraction producing nothing"
```

---

### Task 8: Port training, detectors, classifier ensemble and cascade (combined `c08`, `c11`, `c12`, `c13`)

**Files:**
- Create: `notebook_build/cells/src_08.py`, `src_11.py`, `src_12.py`, `src_13.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE` entries for 8, 11, 12, 13
- Test: `test_bug1.py`, `test_metric_guard.py`

**Interfaces:**
- Consumes: `c08.py` (`train_model`, `make_weighted_sampler`, `compute_class_weights`), `c11.py` (3 detectors + `unwrap_model`), `c12.py` (per-seed ensemble members, unknown fit), `c13.py` (cascade + val threshold grid).
- Produces: `train_model(model, train_loader, val_fn, config, device, save_path=None)` — **`config` stays a plain dict read by key** so Task 13 can copy it per variant. `CLS_ENSEMBLE_SEEDS` members as dicts `{seed, model, best_val, history, metrics}`; `cls_model` a `SoftmaxEnsemble` when >1 seed.

- [ ] **Step 1: Write the failing test**

```python
def test_returns_best_weights_not_last():
    model, hist, best = run_case(save_path=None)
    assert abs(rescore(model) - hist['best_score']) < 1e-9

def test_typo_in_selection_metric_raises():
    with raises(KeyError):
        train_model(m, loader, val_fn=typo_validator, config={**CFG, 'SELECT_METRIC': 'balanced_acc'})
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python notebook_build/tests/test_bug1.py`
Expected: FAIL against the un-ported cell 8

- [ ] **Step 3: Implement**

Copy all four cells verbatim.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebook_build/tests && ../../venv/bin/python test_bug1.py && ../../venv/bin/python test_metric_guard.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/ notebook_build/tests/ notebook_build/assemble.py
git commit -m "port: training, detector, ensemble-classifier and cascade cells"
```

---

### Task 9: Port the inference cells (combined `c20`–`c23`) including the chunk-alignment fix

**Files:**
- Create: `notebook_build/cells/src_20.py`, `src_21.py`, `src_22.py`, `src_23.py`
- Modify: `notebook_build/assemble.py` — add `REPLACE` entries for 20, 21, 22, 23
- Test: `test_inference.py`, `test_resample.py`

**Interfaces:**
- Consumes: `c20.py` (`INFER_*` config, `_model_from_pk` accepting a list = ensemble, unknown-sidecar membership guard), `c21.py` (polyphase resampler, `_resampled` with the phase-alignment fix at `:124-138`, scan, selections, frequency, clock), `c22.py` (guarded listing, corrupt handling), `c23.py` (truth scoring).
- Produces: unchanged `write_detections` output format. `_resampled(path, n_in, r0, r1, sr_in, sr)` with `q0 = ((r0 - margin) // L) * L`.

- [ ] **Step 1: Write the failing test**

```python
def test_chunked_resampling_equals_whole_file():
    """Review Focus: 250/500/256 kHz -> 192 kHz were 13-16% of full scale off."""
    for sr_in in (384000, 250000, 500000, 256000):
        full = _resampled(p, n, 0, n_out, sr_in, 192000)
        for r0, r1 in [(0, 1000), (5000, 9000), (n_out-7000, n_out-10)]:
            assert np.array_equal(_resampled(p, n, r0, r1, sr_in, 192000),
                                  full[r0:r1]), f'{sr_in} Hz at [{r0},{r1})'
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_resample.py`
Expected: FAIL — the merged notebook's cell 21 starts each chunk at `r0`

- [ ] **Step 3: Implement**

Copy all four cells verbatim.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd notebook_build/tests && ../../venv/bin/python test_resample.py && ../../venv/bin/python test_inference.py`
Expected: `max|excerpt − full| = 0.00e+00` for all four rates; inference suite PASS

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/ notebook_build/tests/ notebook_build/assemble.py
git commit -m "port: inference cells — polyphase resampler, phase-aligned chunking, robust input handling"
```

---

### Task 10: Port the recording-held-out check (combined `c25`)

**Files:**
- Create: `notebook_build/cells/src_25.py`
- Modify: `notebook_build/assemble.py` — append after cell 23, plus `md_24.md`
- Test: `test_phase2.py`

**Interfaces:**
- Consumes: `c24.md`, `c25.py`, `split_by_recording_per_species` from cell 6, `RUN_GROUPED_CHECK`, `GROUPED_MICS`.
- Produces: `grouped_results: dict | None` — in-split vs held-out side by side, per class.

- [ ] **Step 1: Write the failing test**

```python
def test_grouped_split_has_zero_recording_overlap():
    tr, va, te, rules = split_by_recording_per_species(wavs, labels, rec_ids, fracs, seed=42)
    assert not (set(recording_id(f) for f in tr) & set(recording_id(f) for f in te))
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_phase2.py`
Expected: FAIL — no grouped check in the merged notebook

- [ ] **Step 3: Implement**

Copy `c24.md` and `c25.py` verbatim; register in `APPEND_MD`/`APPEND_CODE`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_phase2.py`
Expected: PASS — zero overlap asserted, not assumed

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/ notebook_build/tests/test_phase2.py notebook_build/assemble.py
git commit -m "port: recording-held-out check — the honest new-recording estimate"
```

---

### Task 11: NEW cell — noise floor for every model

**Files:**
- Create: `notebook_build/cells/src_26.py`
- Modify: `notebook_build/assemble.py` — append, guarded by `RUN_MEASURE` (default `True`)
- Test: `test_phase2.py` (extend)

**Interfaces:**
- Consumes: `train_model`, `CLS_ENSEMBLE_SEEDS` members, `det_results`.
- Produces: `NOISE_FLOOR: dict[str, float]` keyed by model name — `2 × sd` of the per-seed test scores — printed with the number it represents in clip units. Task 13 labels its deltas against it.

**Gap (c).** The combined notebook prints a classifier `sd` row but reports nothing for the detectors, so detector deltas still cannot be judged.

- [ ] **Step 1: Write the failing test**

```python
def test_noise_floor_is_two_sd():
    assert abs(noise_floor([0.876, 0.910, 0.931]) - 2*np.std([0.876,0.910,0.931], ddof=1)) < 1e-9

def test_detectors_have_a_noise_floor():
    assert 'detector_m11' in NOISE_FLOOR
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python notebook_build/tests/test_phase2.py`
Expected: FAIL — no `NOISE_FLOOR`

- [ ] **Step 3: Implement**

For the classifier, reuse the already-trained `CLS_ENSEMBLE_SEEDS` members — no retraining needed. For the **best mic** only (`GROUPED_MICS`), train one extra copy per seed and compute the sd. Print each floor as `2 × sd = X (≈ N test clips)`, using `0.5 × (1/n_pos + 1/n_neg)` for balanced accuracy.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_phase2.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_26.py notebook_build/tests/test_phase2.py notebook_build/assemble.py
git commit -m "feat: noise floor for every model — 2 x sd, in clip units"
```

---

### Task 12: NEW cell — threshold calibration on real recordings

**Files:**
- Create: `notebook_build/cells/src_27.py`
- Modify: `notebook_build/assemble.py` — append, guarded by `RUN_PHASE4`
- Test: `test_inference.py` (extend)

**Interfaces:**
- Consumes: `INFER_INPUT_DIR`, `INFER_TRUTH_DIR`, `process_recording` (cell 21), `_read_truth_table` + overlap matcher (cell 23), `INFER_EXPECTED_NOISE_FRACTION` (new, in cell 20's config).
- Produces: `CALIBRATION: list[dict]` — one row per `(threshold, merge_gap)` with `boxes_found`, `species_correct`, `noise_hit`, `precision`, `recall`. Writes no new file format.

**Gap (b), and the largest unmeasured risk in the project.** Every threshold to date was tuned on a split that is 81 % call / 19 % noise, while real recordings are >95 % noise windows. `0.981 boxes found` says nothing about false alarms. Cell 8 documents that the model's outputs reflect a uniform class prior, so a test-mix threshold does not transfer.

- [ ] **Step 1: Write the failing test**

```python
def test_calibration_sweeps_threshold_and_gap():
    rows = calibrate(paths, truth_dir, [(0.3, 0.1), (0.5, 0.1), (0.7, 0.2)])
    assert [r['threshold'] for r in rows] == [0.3, 0.5, 0.7]

def test_without_truth_tables_precision_is_called_unmeasurable():
    out = capture(lambda: calibrate(paths, truth_dir=None, settings=SETTINGS))
    assert 'unmeasurable' in out.lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python notebook_build/tests/test_inference.py`
Expected: FAIL — no `calibrate`

- [ ] **Step 3: Implement**

Add `INFER_EXPECTED_NOISE_FRACTION = 0.95` to cell 20's config with a comment that it is a **user-supplied deployment prior**, never inferred from a split. Sweep `threshold ∈ {0.3, 0.5, 0.7}` × `merge_gap ∈ {0.1, 0.2}` reusing cell 23's scorer; print raw and prior-corrected thresholds side by side. With no truth tables, print selections-per-hour against threshold and state plainly that precision is unmeasurable because the tables mark only a fraction of calls.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_inference.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_27.py notebook_build/cells/src_20.py notebook_build/tests/ notebook_build/assemble.py
git commit -m "feat: threshold calibration on real recordings with a supplied deployment prior"
```

---

### Task 13: NEW cell — the 250 kHz detector for rhle/rhro

**Files:**
- Create: `notebook_build/cells/src_28.py`
- Modify: `notebook_build/assemble.py` — append, guarded by `RUN_EXPERIMENTS` (default `False`)
- Test: `test_phase3.py`

**Interfaces:**
- Consumes: `DET250_CONFIG` (Task 3), the official 250 kHz classifier's encoder via `classifier_path`, `build_model`, `train_model`.
- Produces: `PHASE3: list[dict]` rows `{name, baseline, variant, delta, noise_floor, verdict}`.

**Gap (d), and why it is a sample-rate change rather than a band change.** `fmax=95000` on a 192 kHz detector is already within 1 kHz of Nyquist (96 kHz), so rhle/rhro's 90–100 kHz CF calls are **aliased away, not cropped** — there is no headroom to widen. The 250 kHz classifier (`fmax=125000` = its Nyquist) sees them correctly, so the experiment fine-tunes a **250 kHz binary detector** from that encoder and scores it against the 192 kHz detectors specifically on rhle/rhro recall.

Evidence it is needed: the combined notebook's held-out run shows `rhle` recall **0.83 → 0.09** on new recordings.

- [ ] **Step 1: Write the failing test**

```python
def test_det250_band_is_at_nyquist_not_above():
    d = cfg.DET250_CONFIG
    assert d['sr'] == 250000 and d['fmax'] == 125000
    assert d['fmax'] <= d['sr'] / 2

def test_verdict_is_labelled_against_the_noise_floor():
    assert label_delta('det250', 0.899, 0.901, sd=0.012)['verdict'] == 'within noise'
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python notebook_build/tests/test_phase3.py`
Expected: FAIL — no `DET250_CONFIG`, no phase-3 cell

- [ ] **Step 3: Implement**

`DET250_CONFIG` goes in Task 3's config cell: `CLS_CONFIG`'s band with `num_classes=2`, `classes=['noise','target']`, `base_lr=1e-4`, `n_epochs=100`. The phase-3 cell trains it with `noise_mix_prob` inherited (it was measured on the detector too: held-out AUC **0.48 → 0.55**, noise recall **0.11 → 0.32**), evaluates against the 192 kHz detectors on per-class recall, and prints the delta beside `NOISE_FLOOR`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python notebook_build/tests/test_phase3.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebook_build/cells/src_28.py notebook_build/cells/src_01.py notebook_build/tests/ notebook_build/assemble.py
git commit -m "feat: 250 kHz detector experiment for rhle/rhro, whose CF calls alias at 192 kHz"
```

---

### Task 14: Full local GPU run + export round-trip

**Files:**
- Create: `notebook_build/run_local.py`

**Interfaces:**
- Consumes: the generated notebook.
- Produces: logs; a comparison table against the combined notebook's recorded numbers.

- [ ] **Step 1: Implement `run_local.py`**

Port `/home/gb/batspot_gpu_experiments/combined_2026-10-05/run_local.py`: `exec` every code cell in order, swapping only `/kaggle/` paths. **Do not skip or alter any other cell.**

- [ ] **Step 2: Verify the venv is a CUDA build**

Run: `venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
Expected: CUDA build, `True`. The checked-in `venv/` is `2.14.0+cpu` and would silently run a 16-thread CPU job that pins the laptop near 100 °C. The combined notebook used a CUDA venv at `/home/gb/.claude/jobs/ad4b9c63/tmp/cvenv`. If `False`, stop and create one — do not proceed on CPU.

- [ ] **Step 3: Run phases 1–2 end to end**

Run: `venv/bin/python notebook_build/run_local.py 2>&1 | tee /tmp/opencode/merged_run.log`
Expected: every cell completes, 0 errors, curves PNGs written.

- [ ] **Step 4: Verify the exported `.pk` loads through the real `animal_spot/predict.py`**

This is Review Focus 1 and the strongest check available: load `classifier_250khz.pk` and one detector `.pk` through the **real** prediction path and compare logits against the in-memory model (`max|diff|` ≈ 0). A self-contained load test proves nothing.

- [ ] **Step 5: Compare against the combined notebook**

Its recorded run (`runA_full.log`, 1254 s): detectors m03/m09/m11 balanced **0.891/0.899/0.891**; classifier members 0.8483/0.8276/0.8345 acc (mean 0.8368 ± 0.0105); ensemble **0.8483 acc / 0.8567 bal**; held-out classifier **0.8600 → 0.5955 bal**, detector m11 **0.8909 → 0.6415 bal**.

Judge against the noise floor from Task 11, not against these single numbers. **If a metric moves outside the floor, stop and diagnose** — a port should reproduce, not improve.

- [ ] **Step 6: Commit**

```bash
git add notebook_build/run_local.py
git commit -m "build: local GPU runner + export round-trip through the real predict.py"
```

---

### Task 15: Red-green harness over the ported fixes

**Files:**
- Create: `notebook_build/redgreen.py`

**Interfaces:**
- Consumes: `extract_cells.extract`, all suites.
- Produces: a `{fix, suite, exit_code, detected}` table; exit non-zero unless every revert is detected. Must include a `--self-test` proving a deliberately empty revert is reported as **not** detected.

- [ ] **Step 1: Implement**

Revert one fix at a time in a scratch copy, rebuild via `assemble.py`, run the matching suite, assert it fails. Reverts apply **structurally**, not by exact string match. Set `NEW_CELLS` in the subprocess environment and **assert** it is set — the previous harness omitted it, so two suites "passed" against unmutated cells and would have produced a false all-green.

Cases: noise-mixing wiring; `_code_fingerprint` dropping `__defaults__`; `_resampled` losing the `q0` phase alignment; `recording_id` losing `[-_]`; cache validity dropping `mtime_ns`; `train_window_starts` percentile-threshold ties; min-max/pad order; `_TAPE_RE`/`check_bands` removal.

- [ ] **Step 2: Run it**

Run: `venv/bin/python notebook_build/redgreen.py`
Expected: every row `YES`, exit 0

- [ ] **Step 3: Self-test the harness**

Run: `venv/bin/python notebook_build/redgreen.py --self-test`
Expected: reports the empty revert as **not** detected, exits non-zero. A harness that passes a broken revert makes the `YES` column meaningless.

- [ ] **Step 4: Commit**

```bash
git add notebook_build/redgreen.py
git commit -m "test: red-green harness over the ported fixes"
```

---

### Task 16: AGENTS.md

**Files:**
- Modify: `AGENTS.md` — append section 12

- [ ] **Step 1: Write it**

Record: the cell map (ported vs new); that `combined_2026-10-05/cells/` was the port source and its 4 bug fixes came with it; the Nyquist guard's rationale (`predict.py` uses `fmax` unclamped); the noise-floor numbers; the calibration and 250 kHz detector as *new* work not in §11; and that all three ported features (noise mixing, ensemble, unknown) are **on by default** per the 2026-10-07 decision, with the ensemble's 3× training cost stated.

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: AGENTS.md section 12 — merged notebook, ported fixes, 4 closed gaps"
```

---

## Self-Review

**1. Spec coverage.** The original spec's §3 merge map is superseded by Tasks 3–10 (ports rather than re-derivations, which is strictly more faithful). §4 measurement → Task 11. §5 prior-correction: **deliberately dropped** — the combined notebook does not port it and the reasoning has not changed; recorded in the ledger. §6 experiments → Task 13. §7 inference → Tasks 9, 12. §8 verification → Tasks 2, 14, 15. §10 docs → Tasks 16 and the markdown cells. The 4 verified gaps → Tasks 3 (a), 11 (c), 12 (b), 13 (d). No uncovered requirement remains except prior correction, which is a recorded scope decision.

**2. Step scan.** Every code step names a file and says "copy verbatim", which for a port is the complete instruction — the source and destination are named. Test steps give their assertions. The one thing needing a body is `check_bands`, given as a code block because its message is the deliverable.

**3. Type consistency.** `mix_background_db`, `WindowedBatDataset`, `predict_proba`, the five open-set functions, `recording_id`, `train_model`, `SoftmaxEnsemble`, `process_recording`, `_resampled` are all consumed by name exactly as their producing task defines them. `NOISE_FLOOR` (Task 11) is consumed by Task 13. `DET250_CONFIG` (Task 3) is consumed by Task 13.

**4. Review Focus.** #1 → Task 14 Step 4; #2 → Task 5's `test_noise_bank_contains_only_noise_clips` plus Task 12's sidecar guard; #3 → Task 12; #4 → Task 9's inherited low-sample-rate warning; #5 → Task 5's `test_mixing_is_train_only`.

**5. Proportion.** 16 tasks against a 10-section spec, where 8 tasks are ports of an existing reviewed artifact. The plan is short relative to the code it moves because most of that code already exists and is named precisely.
