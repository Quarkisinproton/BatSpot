# BatSpot

CNN tool to detect and classify bat vocalisations. Fork of ANIMAL-SPOT.
Python 3.10 required. No build system, no CI, no tests, no linter.

## Setup

```bash
python3.10 -m venv venv && source venv/bin/activate
# torch installed separately (CPU example):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

GPU (Linux): `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`

## Running

```bash
# GUI (primary interface)
python GUI/start_GUI_tabs.py

# CLI training (needs config file)
python TRAINING/start_training.py TRAINING/config

# CLI prediction (needs config file)
python PREDICTION/start_prediction.py PREDICTION/config

# CLI evaluation/translation
python EVALUATION/start_evaluation.py EVALUATION/config
```

## Architecture

- `animal_spot/` - active Python package (BatSpot's modified code). Has `__init__.py`.
- `ANIMAL-SPOT/` - upstream vendored copy, near-identical. No `__init__.py`.
- `GUI/` - PySide2/PySimpleGUIQt tabbed GUI. Entry: `start_GUI_tabs.py`.
- `TRAINING/`, `PREDICTION/`, `EVALUATION/` - CLI wrappers that build `os.system()` commands from config files and invoke `animal_spot/main.py` or `animal_spot/predict.py`.
- Config files (`TRAINING/config`, `PREDICTION/config`) use `key=value` format. Boolean `true` enables a flag; `false` omits it.
- Model saved as `.pk` file (torch.save dict with encoderOpts/State, classifierOpts/State, dataOpts, classes).

## Gotchas

- `trainer.py` has dual imports: both `animal_spot.utils.*` (correct) and bare `utils.*` (legacy, will break if not run from the right directory). The `animal_spot/` package imports work everywhere; the bare `utils.*` imports only work when CWD is the repo root.
- TensorBoard is disabled (`ENABLE_TENSORBOARD = False` in `trainer.py:38`). A `_NullSummaryWriter` no-op class replaces `SummaryWriter`. Do not re-enable without the user asking.
- `torch` is NOT in `requirements.txt`. It must be installed separately with the correct index URL for your platform.
- The checked-in `venv/` has **CPU-only torch** (`2.14.0+cpu`). The dev laptop has an RTX 4050 (6 GB): do training/experiments with a CUDA build (`pip install torch` from PyPI gives cu130) in a separate venv. A 16-thread CPU run pinned the laptop at ~100 °C and is ~75x slower (see §8.6).
- `PREDICTION/config` ships `min_max_norm=false`, but the official BatSpot models **and** the models fine-tuned by `batspot-train.ipynb` were trained with min-max normalisation (the official `config_predict` files set `min_max_norm=true`). Predicting with `false` silently feeds a differently scaled input; set it to `true` for these models. The flag is not stored inside the `.pk`.
- Audio files must follow strict naming: `CLASSNAME-LABELINFO_ID_YEAR_TAPENAME_STARTTIMEMS_ENDTIMEMS.wav`. The first `-` before `_` splits class name from label info.
- Data split is hardcoded 70/15/15 (train/val/test) in `main.py:436`.
- Learning rate is multiplied by batch size internally (`main.py:412`).
- Linux file descriptor limit: run `ulimit -n 65535` before GUI if processing many files.
- `.gitignore` only covers `venv/` and `EXECUTABLE/`.
- No test suite exists. No linting or typecheck config.

## Kaggle Fine-Tuning Notebook (`batspot-train-in-kaggle.ipynb`)

Self-contained Kaggle notebook implementing the full BatSpot fine-tuning pipeline on 2x Tesla T4 GPUs.

### Key Components & Pipeline Flow

1. **Configuration (Cell 2)**:
   - `DATA_DIR`: Path to extracted audio clips (`CLASS-*.wav`).
   - Manual model paths (`DETECTOR_M03_PATH`, `DETECTOR_M09_PATH`, `DETECTOR_M11_PATH`, `CLASSIFIER_M09_PATH`).
   - `USE_AUGMENTATION = True/False`: Toggles on-the-fly spectrogram data augmentation.
   - Specs: Detector @ 192 kHz (binary: noise vs call), Classifier @ 250 kHz (multi-species).
2. **Fast Cached Dataset (`CachedBatDataset`)**:
   - Pre-computes STFT spectrograms to disk once (`/kaggle/working/spec_cache/<tag>/*.npy`) to eliminate the heavy CPU bottleneck (`resampy` + STFT + `scipy.ndimage.zoom` per sample).
   - Serves samples via instantaneous `np.load` (~microseconds), keeping T4 GPUs fully fed at ~50% load each (~8.5+ it/s).
3. **On-the-Fly Augmentation**:
   - Only applied when `augment=True` on training splits (val/test remain clean).
   - Techniques: Time shift ($\pm 20\%$, zeroing wrapped edge), Gaussian noise ($\sigma=0.01$), SpecAugment Frequency Masking (up to 20%), SpecAugment Time Masking (up to 20%).
4. **Three-Detector Training (m03, m09, m11)**:
   - Iterates through all available microphone variants (`m03`, `m09`, `m11`), fine-tuning from official pre-trained encoders.
   - Evaluates each independently on test set and logs metrics/confusion matrix.
5. **Single Multi-Species Classifier Training**:
   - Fine-tunes 250 kHz multi-class head on bat species + noise.
6. **Cascaded Evaluation Pipeline (Detector $\to$ Classifier)**:
   - Optimises detector threshold on validation split (maximising call F1).
   - Evaluates combined pipeline across all detector variants on test split:
     - Detector predicts noise $\to$ output `noise`.
     - Detector predicts call $\to$ invoke classifier to predict species.
7. **Export**:
   - Saves `detector_192khz_m03.pk`, `detector_192khz_m09.pk`, `detector_192khz_m11.pk`, and `classifier_250khz.pk` compatible with BatSpot GUI/CLI.

### Key Fixes & Troubleshooting History

- **`best_state` UnboundLocalError**: Initialised `best_state = None` in `train_model` before the epoch loop.
- **DataParallel State Dict Keys**: Stripped `module.` prefix when saving `best_state` (`model.module.state_dict() if is_dp else model.state_dict()`) to prevent mismatch errors on weight restoration.
- **CUDA Device String in AMP**: Used `device.type` instead of `'cuda'` string in `GradScaler` and `autocast` to prevent PyTorch 2.x CPU exceptions.
- **Stray Dict Reset**: Removed duplicate `_official = {}` line in Cell 10 that was wiping priority manual paths.
- **Effective Learning Rate**: Removed batch size multiplication (`effective_lr = config['base_lr']`); paper's `1e-4` is absolute, avoiding 128x overshooting and divergence.
- **Kaggle GPU Metadata**: Set `"accelerator": "nvidiaTeslaT4"` with `"nvidiaTeslaT4Count": 2` and `"isGpuEnabled": True` in notebook metadata so Kaggle doesn't reset session to CPU upon import.
- **GPU Underutilization / CPU Throttling**:
  - Spectrogram generation on CPU per sample starved GPUs (utilization was 0%). Fixed by introducing disk caching (`CachedBatDataset`).
  - Dropped `num_workers` to 4 (Kaggle T4 x2 environment provides 4 CPU cores; >4 caused thread thrashing).
  - Increased `batch_size` to 128 with `grad_accum_steps = 1`.

### Benchmark Results on Current Dataset (964 files: 674 train / 145 val / 145 test)

#### Individual Detectors (Binary: Noise vs Call)
| Variant | Val Acc | Test Acc | Precision | Recall | F1 | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **m03** | 0.2552 | 0.2483 | 0.3833 | 0.3712 | 0.2477 | Poor transfer on this recording setup |
| **m09** | 0.8552 | 0.8690 | 0.7869 | 0.8237 | 0.8028 | Solid detector performance |
| **m11** | **0.9034** | **0.9241** | **0.8696** | **0.8987** | **0.8830** | Best detector performance |

#### Classifier (8-Class, 250 kHz)
- **Val Acc**: 52.41%
- **Test Acc**: 59.31% (Macro Precision: 0.5483, Recall: 0.6649, F1: 0.5743)
- *Bottleneck*: The classifier is limited by small per-class sample size (~48–130 clips/class) and severe class imbalance.

#### Combined Cascaded Pipeline (Detector $\to$ Classifier)
| Pipeline Configuration | Threshold | Test Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Detector **m03** $\to$ Classifier | 0.10 | 59.31% | 0.5483 | 0.6649 | 0.5743 |
| Detector **m09** $\to$ Classifier | 0.10 | 69.66% | 0.7039 | 0.7230 | 0.6883 |
| Detector **m11** $\to$ Classifier | 0.25 | **72.41%** | **0.7203** | **0.7408** | **0.7083** |
## Kaggle Fine-Tuning Notebook (`BatSpot_FineTune_Kaggle.ipynb`)

Clean-slate version of the fine-tuning notebook — intended as the canonical upload to Kaggle. Structurally identical to `batspot-train-in-kaggle.ipynb` (the reference notebook) with all known bugs fixed and the full 3-detector + cascaded-pipeline structure. Use this file when uploading to Kaggle.

### Differences from `batspot-train_only_classifier.ipynb` (the buggy single-run notebook)

`batspot-train_only_classifier.ipynb` has three known correctness problems that make its results non-comparable to `batspot-train-in-kaggle.ipynb`:

1. **LR scaling bug still present** (`effective_lr = config['base_lr'] * config['batch_size'] * config.get('grad_accum_steps', 1)`): With `batch_size=64` and `grad_accum_steps=2` this gives `effective_lr = 0.0128` (128× the intended `1e-4`). Different optimizer trajectory → results are not comparable.
2. **No augmentation on classifier training**: `CachedBatDataset` for `cls_train` is constructed without `augment=True`. Less regularisation can inflate val-checkpoint accuracy on small datasets (674 train samples).
3. **Combined pipeline measurement differs**: The classifier-only notebook's "combined" accuracy measures a different, easier quantity — it skips the detector gating step that propagates detector errors (especially severe for m03, F1 0.25) into the final prediction.

### All Bugs Fixed in `BatSpot_FineTune_Kaggle.ipynb`

| Bug | Old (broken) | Fixed |
| :--- | :--- | :--- |
| Effective LR | `base_lr * batch_size * grad_accum_steps` | `base_lr` (absolute `1e-4`) |
| `GradScaler` device | `GradScaler('cuda')` | `GradScaler(device.type)` |
| `autocast` device | `autocast('cuda')` | `autocast(device.type)` |
| `best_state` init | undefined → `UnboundLocalError` on CPU | `best_state = None` before loop |
| `best_state` save | `model.state_dict()` (keeps `module.` prefix when DP) | `model.module.state_dict() if is_dp` |
| Dataset class | `BatDataset` (per-sample resampy+STFT, GPU starvation) | `CachedBatDataset` (disk-cached `.npy`, instant load) |
| Augmentation | None | `USE_AUGMENTATION` toggle; time shift + Gaussian noise + SpecAugment on train splits only |
| DataLoader | `num_workers=2`, no `persistent_workers` | `num_workers=4`, `persistent_workers=True` |
| Batch / accum | `batch_size=64`, `grad_accum_steps=2` | `batch_size=128`, `grad_accum_steps=1` |
| Detector training | Single detector (m09 default) | All 3 variants: m03, m09, m11 loop |
| Combined pipeline | Single detector → classifier | Loop over all 3 detector variants |
| Kaggle GPU metadata | Missing | `"accelerator": "nvidiaTeslaT4"`, `"nvidiaTeslaT4Count": 2`, `"isGpuEnabled": true` |
| FD limit | Missing | `resource.setrlimit(RLIMIT_NOFILE, (65535, 65535))` in Cell 3 |

---

# Session Log — Kaggle Fine-Tuning Debugging

Everything below was done by reading code, reproducing locally, and diffing
against recorded Kaggle output. Each entry is **symptom → root cause → fix →
proof**. Nothing here is speculative unless explicitly marked *unconfirmed*.

## 0. Tooling constraints hit during this work (read this first)

These cost real time. They will cost time again.

- **The notebooks are stored as compact single-line JSON.** `read` truncates a
  line at 2000 chars, so `read` is useless on them, and `grep` returns the same
  truncated single line for every match. *Never* read or grep a `.ipynb` for
  content analysis. Parse it with Python's `json` module instead.
  - Useful exception: `grep` still reports **per-file match counts**, which is
    enough to answer "does notebook X contain fix Y?" across all notebooks at
    once. That is how the fix-provenance table in §2 was built.
- **The agent harness `shell` tool ran commands but captured no stdout**, and
  `cmd > file` produced empty files. Workaround that does work: have the command
  redirect to a file *and* have the subagent read it back with the `read` tool,
  or write line-based output to a file and `read` it. All evidence dumps in this
  session were produced that way.
- Verification was delegated to `general` subagents that dumped results to
  line-based files, which the main session could then read. This is the pattern
  that worked; repeating it.

## 1. Notebook inventory — which file carries which fix

Six notebooks, four of which are the same lineage at different stages. **The
early-stopping fix (§2.12) exists in exactly one of them.**

| Notebook | Early-stop fix | `RUN_CLIP_EXTRACTION` | Augmentation | Notes |
| :--- | :---: | :---: | :--- | :--- |
| `batspot-train-combined.ipynb` | **YES** | `False` | noise mixing on | **Newest (2026-10-06, §11).** batspot-train + Space Bunny fixes + classifier ensemble, 'unknown' answer, robust inference, held-out-recording check. Built from cells/ in `/home/gb/batspot_gpu_experiments/combined_2026-10-05/`. |
| `batspot-train.ipynb` | **YES** | `True` (guarded) | `False` | Single detector (m09) + classifier. The file this session fixed. |
| `BatSpot_FineTune_Kaggle.ipynb` | no (hardcoded `20`) | `True` (guarded) | `False` | 3-detector loop + cascade |
| `batspot-train-in-kaggle.ipynb` | no (hardcoded `20`) | `False` | `True` | Reference notebook |
| `batspot-train_only_classifier.ipynb` | no (hardcoded `20`) | `False` | none | Still has the batch-size LR bug |
| `kaggle.ipynb` / `kaggle3.ipynb` | no (hardcoded `20`) | n/a | n/a | Old drafts, predate most fixes |
| `kaggle2.ipynb` | no (hardcoded `25`) | n/a | n/a | Old draft |

All four lineage notebooks **do** carry the `shortcut` architecture fix (§2.1),
the basename-collision staging (§2.2), `USE_MANUAL_MODEL_PATHS`,
`props.total_memory`, `torch.amp.GradScaler(device.type)`, the clip-extraction
guard, and `resource.setrlimit`.

`batspot-train.ipynb` has since been **round-tripped through Kaggle**: it was
edited locally, uploaded, run, and the executed notebook saved back over the
local file. It now contains saved outputs. Earlier in this session it was
verified as a 1-line file with **zero** outputs — so any statement below about
"the outputs" refers to the sibling notebooks unless noted.

## 2. Fix log

### 2.1 `shortcut` vs `downsample` — every single `.pk` failed to load

**Symptom.** All four official models raised
`Missing key(s): "layer2.0.downsample.0.weight"` on `load_state_dict`.

**Root cause.** The notebook's inline re-implementation of BatSpot's ResNet
named the residual projection `downsample`. The real code assigns
`self.shortcut = downsample` (`animal_spot/models/residual_base.py:32`), so the
parameter is *serialised* under `shortcut.*`. Any name mismatch breaks
`load_state_dict` for the whole encoder.

**Why it matters beyond this notebook.** `.pk` files are not just weights — the
key names are the on-disk format. A notebook that renames anything is a
notebook that cannot load real BatSpot models, which defeats the entire
fine-tuning premise.

**Fix.** Cell 5 was replaced with a faithful port that preserves every module
and attribute name (`BasicBlock`, `Bottleneck`, `ResidualBase.make_layer`,
`ResidualEncoder`). Cell 5 carries an explicit warning comment so the trap is
not re-introduced.

**Proof.** 120 identical `state_dict` keys; `max|diff|` between port and
original = `0.00e+00`.

### 2.2 `ANIMAL-SPOT.pk` basename collision

**Symptom.** Silent, and therefore worse than a crash: the classifier would
overwrite the detector, and you would fine-tune a "detector" from 15-class
classifier weights and never know.

**Root cause.** All four official models are named `ANIMAL-SPOT.pk`. Any
copy-by-basename staging step collides.

**Fix.** Each is staged under a unique name, `official_{role}_{mic}.pk`, and
Cell 10 carries a comment explaining why.

### 2.3 Detector class map — `KeyError: 'rhro'`

**Symptom.** `KeyError: 'rhro'` during detector training.

**Root cause.** `BatDataset.__getitem__` derives the species from the *filename*
and looks it up in `det_class_to_idx`. That dict had been hand-written as
`{'noise': 0, 'call': 1}` — the two *output* labels. But the dataset contains 8
*species* names, so every species filename missed.

**Fix.** Build the map over the classes actually present in the data, folding
every non-`noise` species to label 1. Verified output:

```
Detector: 8 dataset classes -> 2 binary labels
  label 0 (noise): ['noise']
  label 1 (call):  ['acsh', 'alte', 'heti', 'rhbe', 'rhle', 'rhro', 'sasa', 'call']
```

**Lesson.** The binary *label set* and the *dataset class set* are different
spaces. Conflating them is what produced the crash.

### 2.4 CrossEntropyLoss weight-tensor shape mismatch

**Symptom.**
`RuntimeError: weight tensor should be defined either for all 2 classes or no
classes but got weight tensor of shape: [9]`

**Root cause.** Surfaced only *after* §2.3 — with the map fixed, the class
weight computation finally ran and produced 9 values (8 species + `call`) for a
2-output head. `compute_class_weights` sized itself from
`len(class_to_idx)` (dataset classes) instead of `config['num_classes']` (model
outputs).

**Fix.** Size from `config['num_classes']`. Detector `[2.0, 0.667]`, classifier
8 values. **This bug was invisible until the previous one was fixed** — a
reminder not to assume a fix list is complete.

### 2.5 Cell 10 `_official = {}` double-initialisation

**Symptom.** `Using 4 manual path(s) from Cell 2.` immediately followed by
`No pre-trained models found.` The run then silently fell back to a local
384 kHz / 3-class model.

**Root cause.** `_official = {}` appeared twice in Cell 10. The second one wiped
the four valid paths that had just been resolved and printed.

**Why this nearly escaped.** The test harness passed — but only because local
`BatSpot_article/` auto-discovery found the models and masked the wipe. **A test
that passes because a fallback fired is worse than no test.**

**Fix.** Single initialisation; discovery now only *fills gaps*, never
overwrites. Also fixed in the same pass: role resolved before the manual-wins
check, no `else: _role = 'detector'` fallback, and better "no models found"
advice when paths were set but unreadable.

**Precedence rule** (`USE_MANUAL_MODEL_PATHS`): **manual Cell 2 paths >
auto-discovery under `/kaggle/input/*` > local `Data/model_output/ANIMAL-SPOT.pk`.**
Origins print as `[manual]` vs `[discovered]`.

### 2.6 Clip-extraction clobber

**Symptom.** Cell 8 re-read ~4 GB of raw audio and rewrote into `DATA_DIR`,
which already held 964 extracted clips.

**Root cause.** The cell ran unconditionally as soon as `RAW_AUDIO_DIR` /
`SELECTIONS_DIR` were set.

**Fix.** Gated behind `RUN_CLIP_EXTRACTION = False` plus
`OVERWRITE_EXISTING_CLIPS = False`, which refuses to write into a non-empty
`DATA_DIR`. Both guards tested. Verified output:

```
Clip extraction ABORTED: 964 wav files already exist in .../final_dataset/data
Re-extracting would overwrite them. Set OVERWRITE_EXISTING_CLIPS = True
```

**Still wrong — see §3.** `RUN_CLIP_EXTRACTION` is `True` by default in
`batspot-train.ipynb` and `BatSpot_FineTune_Kaggle.ipynb`. The second guard
catches it, so it is safe, but the default is misleading and still costs a
directory listing.

### 2.7 AMP device strings and `props.total_memory`

- `torch.cuda.amp.GradScaler('cuda')` / `autocast('cuda')` → `torch.amp.*` with
  `device.type`. A hardcoded `'cuda'` string throws on CPU in PyTorch 2.x.
- `props.total_mem` → `props.total_memory`. Kaggle ships torch 2.10.0+cu128,
  where only `total_memory` exists.

### 2.8 `classification_report` partial-class crash

**Symptom.**
`ValueError: Number of classes, 4, does not match size of target_names, 15`

**Root cause.** The test split does not contain all 15 of the official
classifier's classes, so sklearn inferred 4 from `y_true` while `target_names`
carried 15.

**Fix.** Pass an explicit `labels=` list, built from the model's own `classes`
dict rather than from whatever happens to be in the split. Applied to
`evaluate_model` as well.

### 2.9 DataParallel export crash

**Symptom.** `TypeError` from `model[0]` in `export_pk`.

**Root cause.** `train_model` returned the `nn.DataParallel` *wrapper*.

**Fix.** Unwrap before export. Related: `best_state` must be saved as
`model.module.state_dict() if is_dp else model.state_dict()`, or the restored
checkpoint keeps `module.` prefixes and fails to load. `best_state = None` is
initialised before the epoch loop (otherwise `UnboundLocalError` on CPU).

### 2.10 Spectrogram config mismatch

Config values had been guessed rather than read from the models' own `dataOpts`.
Corrected to the real per-role values, plus the paper hyperparams from
`BatSpot_article/.../config_train`:

| | Detector | Classifier |
| :--- | :---: | :---: |
| sample rate | 192 kHz | 250 kHz |
| `n_fft` | 128 | 160 |
| `fmin` / `fmax` | 1000 / 95000 | 10000 / 125000 |
| classes | 2 (noise vs call) | 15 (official) |

**Why hard-matching matters.** A transferred encoder is only meaningful at the
time-frequency resolution it was trained at. Change `n_fft`, `hop_length`, or
`n_freq_bins` and the first conv sees a different input geometry. Likewise the
250 kHz classifier never sees the 90–192 kHz band at all.

### 2.11 Effective learning rate

`effective_lr = base_lr * batch_size * grad_accum_steps`. With
`base_lr=1e-4`, `batch_size=64`, `grad_accum_steps=2` this is **0.0128 — 128×
the paper's value**, and it diverges. The paper's `1e-4` is absolute. Fixed to
`effective_lr = config['base_lr']`.

`batspot-train_only_classifier.ipynb` **still has this bug** (verified present
at line 969), so its results are not comparable to the other notebooks.

### 2.12 Early stopping ignored the config — the headline bug

**Symptom, as reported:** "fine tuning not running for specified epochs".

**Method note.** `batspot-train.ipynb` had **zero saved outputs** at the time
(all 18 code cells `execution_count: null`, and the string `output_type` absent
from the file), so the request to analyse "its outputs" could not be satisfied
from that file. Diagnosis therefore came from code, and the prediction was then
tested against the sibling notebooks that *do* contain runs.

**Root cause.** `train_model` never read the early-stopping setting. It
hardcoded 20:

```python
patience_early = int(max(1, ceil(20 / config['epochs_per_eval'])))   # before
```

while the config carried `'early_stopping_patience_epochs': 1000`. Verified
dead: the key appeared only inside the config dicts and nowhere in the
`train_model` body.

**The compounding subtlety.** `no_improve` incremented **once per validation**,
and validation runs every `epochs_per_eval = 2` epochs. So `patience_early = 10`
actually meant *"10 stale validations"* = **20 raw epochs**.

**Proof.** Every recorded run stopped at exactly `last_improvement + 20`:

| Run | last improvement | early stop fired | gap |
| :--- | :---: | :--- | :---: |
| `only_classifier` detector | epoch 58 | `Early stopping at epoch 78.` | 20 |
| `only_classifier` classifier | epoch 18 | `Early stopping at epoch 38.` | 20 |
| `kaggle3` fine-tune | epoch 36 | `Early stopping at epoch 56.` | 20 |

All three printed `Patience LR: 4, Patience ES: 10`. A gap of exactly 20 and
never 10 is the signature of a counter in validation steps doubled by
`epochs_per_eval`.

**Three more dead config keys.** The "reference hyperparams" block was entirely
decorative:

| Key | In config | In `train_model` | Effect |
| :--- | :--- | :--- | :--- |
| `early_stopping_patience_epochs` | 1000 | hardcoded `20` | **mismatch — this was the bug** |
| `lr_patience_epochs` | 8 | hardcoded `8` | matched by luck |
| `lr_decay_factor` | 0.5 | hardcoded `factor=0.5` | matched by luck |
| `beta1` | 0.5 | hardcoded `betas=(0.5, 0.999)` | matched by luck |

Only the first disagreed with its hardcode, which is why it was the only one
that bit. The other three are latent traps: change them in config and nothing
happens.

**Fix applied** to `batspot-train.ipynb` (six single-line edits):

| Line | Before | After |
| :--- | :--- | :--- |
| patience | `ceil(20 / epochs_per_eval)` | `int(config.get('early_stopping_patience_epochs', 20))` |
| counter | `no_improve += 1` | `no_improve += config['epochs_per_eval']` |
| lr patience | `ceil(8 / epochs_per_eval)` | `ceil(config.get('lr_patience_epochs', 8) / ...)` |
| betas | `betas=(0.5, 0.999)` | `betas=(config.get('beta1', 0.5), 0.999)` |
| scheduler | `factor=0.5` | `factor=config.get('lr_decay_factor', 0.5)` |
| log | `Patience ES: {n}` | `Patience ES: {n} raw epochs` |

`no_improve` now measures **raw** epochs since the last improvement, which is
what the key name promises. Arithmetic re-simulated by executing the verbatim
extracted block: last improvement at raw 58 → `no_improve` = 2 at raw 60, 20 at
raw 78, 42 at raw 100 (correct), versus the old 1 / 10 / 21.

`n_epochs` was also lowered from the unreachable `5000` / `5500` to a value
that fits Kaggle's 12 h session limit (at ~10 s/epoch, 5000 epochs is ~14 h for
the detector alone).

**Verification.** `json.load` OK. All code cells parse; the only failure is the
`!pip install` line in Cell 3, which is an IPython magic, not code. All four
keys confirmed present *inside* the `train_model` body; old `5000`/`5500` gone.

**Current live values** in `batspot-train.ipynb` (hand-edited after the fix):

| | `n_epochs` | patience | effect |
| :--- | ---: | ---: | :--- |
| Detector | 400 | 1000 | **patience is inert** — see §3 |
| Classifier | 300 | 60 | live: 30 stale validations |

## 3. Open issues — identified, NOT fixed

Do not assume these are handled.

1. **The detector's `early_stopping_patience_epochs: 1000` can never fire.**
   With `n_epochs = 400` and `epochs_per_eval = 2` there are 200 validations, so
   the maximum reachable `no_improve` is 398 — below the 1000 required. Early
   stopping is live code but dead in practice; `n_epochs` is the only real cap.
   Dropping it to ~100 would make it an actual safety net.
2. **The same bug is unfixed in five other notebooks** (§1). In
   `BatSpot_FineTune_Kaggle.ipynb` this is actively misleading: `CLS_CONFIG` has
   `'early_stopping_patience_epochs': 60,  # was 20; wait 120 real epochs before
   stopping`, but `train_model` still hardcodes 20, so that comment is false and
   the value is ignored.
3. **A Kaggle run was executing on CPU, not GPU.** Pasted Cell 12 output showed
   `'pin_memory' argument is set as true but no accelerator is found`,
   `AMP: False`, `DP: False`, 11 iterations at `10.36s/it`. The notebook's whole
   premise (2× T4, DataParallel, AMP) was inactive. Not yet diagnosed; plausible
   causes are the accelerator not actually being granted despite notebook
   metadata, or `torch.cuda.is_available()` being False for another reason.
   There is currently **no loud banner** warning about this — Cell 3 prints CUDA
   availability but nothing acts on it.
4. **`save_path` is dead.** Both call sites pass
   `detector_{mic}_curves.png` / `classifier_curves.png`, but `train_model` uses
   `save_path` only as a truthiness gate for the in-memory `best_state`
   snapshot. There is **no `savefig` anywhere**, and the returned history is
   discarded at the call site. No curves PNG is ever written and no training
   curves survive a run. **Fixed in `batspot-train.ipynb` (§10.2): curves are written; history kept.**
5. **DataLoader teardown spew — *unconfirmed*.** Kaggle floods output with
   `Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ ...>
   AssertionError: can only test a child process`, once per loader teardown.
   Working hypothesis: `pbar = tqdm(train_loader, ...)` is never closed, stays
   referenced past `train_model`'s return, so the train iterator outlives its
   scope and a later `fork` inherits it; the child then GCs it and runs `__del__`
   in the wrong process. **Not reproduced** — three repro harnesses (real
   `BatDataset`, held iterators across forks, under `taskset -c 0,1`) all gave
   zero assertions. Do not state this as fact. Suspected missing ingredient:
   `tqdm` wrapping the train loader, and/or Kaggle-specific process supervision.
6. **`RUN_CLIP_EXTRACTION = True`** by default in `batspot-train.ipynb` and
   `BatSpot_FineTune_Kaggle.ipynb` (§2.6). Safe due to the second guard, wrong
   as a default. **Fixed in `batspot-train.ipynb` (§10.2): now `False`.**

## 4. Model and data inventory

### Official pre-trained models (`BatSpot_article/batspot/`)

| Role | Path | Classes | SR | fmin/fmax |
| :--- | :--- | ---: | ---: | :--- |
| call classifier | `models_call_classifier/m09/train/ANIMAL-SPOT.pk` | 15 | 250 kHz | 10000 / 125000 |
| call detector | `models_call_detector/m03/train/ANIMAL-SPOT.pk` | 2 | 192 kHz | 1000 / 95000 |
| call detector | `models_call_detector/m09/train/ANIMAL-SPOT.pk` | 2 | 192 kHz | 1000 / 95000 |
| call detector | `models_call_detector/m11/train/ANIMAL-SPOT.pk` | 2 | 192 kHz | 1000 / 95000 |

Also present and deliberately **not** fine-tuned: `models_social_detector/m05`,
`models_buzz_detector/m06`.

**Fine-tuning transfers the encoder only, never the classifier head.** The
official classifier's 15 classes are European species codes with **zero
overlap** with the North American 8-class set except `noise`. The head is always
rebuilt fresh (15 → 8).

**`m09` is the default target** (the paper's primary mic). `m03` transfers
poorly to this recording setup.

### Kaggle dataset paths (all four verified valid)

```
/kaggle/input/datasets/budhil/100eachbatds/models_call_detector/models_call_detector/{m03,m09,m11}/train/ANIMAL-SPOT.pk
/kaggle/input/datasets/budhil/100eachbatds/models_call_classifier/models_call_classifier/m09/train/ANIMAL-SPOT.pk
/kaggle/input/datasets/budhil/100eachbatds/Data/Data/final_dataset/data
/kaggle/input/datasets/budhil/100eachbatds/Data/Data/audio
/kaggle/input/datasets/budhil/100eachbatds/Data/Data/selections
/kaggle/input/datasets/budhil/100eachbatds/Data/Data/model_output
```

The doubled folder names (`Data/Data`, `models_call_detector/models_call_detector`)
are real — the uploaded directories were already named `Data` / `models_*`.
Flattening them on upload would shorten every path.

### Local dataset

`Data/final_dataset/data/` — 8 class subfolders (`acsh`, `alte`, `heti`,
`noise`, `rhbe`, `rhle`, `rhro`, `sasa`), **964 files** split **674 train / 145
val / 145 test**, filenames `CLASS-bat_ID_YEAR_TAPE_START_END.wav`.
`Data/model_output/` — 1 `.pk` + 13 `.checkpoint`, 384 kHz 3-class; used for
validation and as the unreliable fine-tune fallback.
`Data/scripts/export_clips.R` — R script ported as notebook Cell 8.

**Note:** `BatSpot_article/` and `BatSpot_from_zipfile/` are untracked and *not*
gitignored (~25 GB). `.gitignore` covers only `venv/` and `EXECUTABLE/`.

## 5. Verification harnesses

Lives in `/tmp/opencode/` and **will not survive** — recreate from these
descriptions if needed:

| Script | What it proves |
| :--- | :--- |
| `verify.py` | End-to-end: 18 models load, fine-tune step produces `(1,8)` logits, finite loss, non-zero grads |
| `test_manual_paths.py` | 4-case Cell 2 path resolution |
| `test_detector_path.py` | Reconstructs the Kaggle layout incl. doubled folder names; real detector training |
| `test_classifier_path.py` | Real classifier training + `predict.py` export round-trip |
| `repro_manual.py`, `repro_bad.py` | Cell 10 valid / unreadable manual paths |
| `repro_dl.py`, `repro_dl2.py`, `repro_dl3.py` | DataLoader assertion repros (all 0 hits) |
| `test_gate.py` | Cell 8 clip-extraction guard |

The export round-trip is the strongest check available: it loads the produced
`.pk` through the **real** `animal_spot/predict.py` and compares logits
(`max|diff| = 0.00e+00`). Prefer that over any self-contained load test.

**Scratch files still in the repo root and not yet deleted** (the harness shell
was broken, so cleanup was not possible):
`_EVIDENCE_outputs.txt`, `_EVIDENCE_epochs.txt`, `_VERIFY.txt`, `_FINAL.txt`,
`_verify.py`, `_sh.txt`.

## 6. Classifier Optimization & m03 Collapse Investigation (Latest Updates)

### 6.1 The m03 Detector Accuracy Illusion (0.98 Official vs 0.20 Fine-Tuned)

- **Official Pre-Trained m03 "0.98 Accuracy"**:
  - **Mechanism**: Cell 20 (`run_model_evaluation`) evaluates models on `known_classes = {'noise', 'target'}`. Because custom bat clips are labeled with species codes (`acsh`, `alte`, etc.) rather than the generic string `target`, **all 778 bat call files were skipped**.
  - **Result**: The evaluation was executed solely on 186 `noise` files. The pre-trained Danish offshore detector predicted `noise` on every sample, achieving 186/186 = 98% accuracy purely as a degenerate baseline.
- **Fine-Tuned m03 "0.20 Accuracy" Collapse**:
  - **Mechanism**: On the 70/15/15 test set (145 clips total: 28 `noise`, 117 `call`), predicting 100% `noise` yields $28/145 = 19.31\% \approx 0.20$.
  - **Root Cause**:
    1. `train_model` combined both `WeightedRandomSampler` (which balances batches 50/50) and `nn.CrossEntropyLoss(weight=class_weights)`. This resulted in a **4.18x penalty multiplier on noise false alarms** on already balanced batches.
    2. The Danish offshore pre-training (m03) has a severe acoustic domain gap from Indian bat calls, providing poor initial feature separability.
    3. The randomly initialized classification head quickly minimized the heavily noise-weighted loss by driving all predictions to `noise` by Epoch 2 (`val_acc = 0.1931`).
    4. `ReduceLROnPlateau` saw zero validation improvement over the initial baseline and rapidly decayed learning rate down to $1.22 \times 10^{-8}$, permanently trapping m03 in the all-noise local minimum.
  - **Fix**: Removed loss class weights from `nn.CrossEntropyLoss()` so it runs unweighted (`nn.CrossEntropyLoss()`), allowing `WeightedRandomSampler` alone to manage class balance without over-penalizing noise errors.

### 6.2 Classifier Stagnation & Combined Accuracy Bottleneck

- **Symptom**: Classifier standalone validation accuracy plateaued at ~34.48% (test ~35–40%), causing early stopping at Epoch 82 even when `n_epochs` was set to high values (e.g. 5,000). The combined cascaded pipeline was bottlenecked at 72.41%.
- **Root Causes**:
  1. **Learning Rate Too Low (128x Under-scaled)**: With `batch_size=128` across 674 training samples, each epoch consists of only 5–6 optimizer steps. An absolute base learning rate of $1 \times 10^{-4}$ was too low for a newly initialized 8-class linear head to escape its random state. In contrast, `batspot-train_only_classifier.ipynb` with effective LR $1.28 \times 10^{-2}$ reached 79.31% validation accuracy in 18 epochs.
  2. **Premature LR Decay & Early Stopping**: `ReduceLROnPlateau` with `lr_patience_epochs = 8` (4 evaluation checks) halved the learning rate down to $6.25 \times 10^{-6}$ by Epoch 82, freezing the network before it could learn.
  3. **Aggressive Augmentation on Small Data**: Live SpecAugment, time shifting, and Gaussian noise on small per-class samples (~84 clips/class) hindered convergence rather than regularizing.

### 6.3 Applied Fixes across Notebooks (`BatSpot_FineTune_Kaggle.ipynb` & `batspot-train.ipynb`)

> **SUPERSEDED for `batspot-train.ipynb` by §8.** The learning-rate / patience / epoch values in this table were a workaround for a different root cause (input windowing and steps-per-epoch). `batspot-train.ipynb` now uses batch 32, `base_lr` 3e-4 (classifier) / 1e-4 (detector). Also note the "Waits 120 real epochs" wording below is wrong since the §2.12 fix: patience counts **raw** epochs, so `early_stopping_patience_epochs = 60` means 60 raw epochs. `BatSpot_FineTune_Kaggle.ipynb` still has the old values and the first-20 ms bug.

| Setting / Component | Old Value | New Value | Reason |
| :--- | :---: | :---: | :--- |
| `CLS_CONFIG['base_lr']` | `1e-4` (or `2e-4`) | **`1e-2`** | Provides sufficient gradient magnitude for 5–6 steps/epoch to train the 8-class head |
| `CLS_CONFIG['lr_patience_epochs']` | `8` | **`20`** | Waits 40 real epochs before halving LR, avoiding premature decay |
| `CLS_CONFIG['early_stopping_patience_epochs']` | `20` (or `1000`) | **`60`** | Waits 120 real epochs before triggering early stop |
| `CLS_CONFIG['n_epochs']` | `150` (or `400`) | **`300`** | Allows ample epochs for full convergence with extended patience |
| `USE_AUGMENTATION` | `True` | **`False`** | Prevents underfitting on small datasets (~84 samples per class) |
| `train_model` `loss_fn` | `nn.CrossEntropyLoss(weight=class_weights)` | **`nn.CrossEntropyLoss()`** | Eliminates double-weighting artifact with `WeightedRandomSampler` |

---

## 7. Corrections & Status Summary

- Both `BatSpot_FineTune_Kaggle.ipynb` and `batspot-train.ipynb` are synchronized with the 4 critical training/loss/patience fixes.
- Data leakage between detector training splits and classifier evaluation sets in the cascaded pipeline is resolved by using the shared species-stratified 70/15/15 split across all datasets. (This does **not** cover leakage between *recordings*: see §8.2 item 6.)
- `batspot-train.ipynb` was overhauled on 2026-10-04 (§8). `BatSpot_FineTune_Kaggle.ipynb`, `batspot-train-in-kaggle.ipynb` and the other notebooks were **not** updated and still read only the first 20 ms of each clip.

---

## 8. Windowing fix & accuracy overhaul of `batspot-train.ipynb` (2026-10-04)

Everything here was measured, not assumed: controlled GPU experiments (§8.5), then the **real
notebook executed end to end** on an RTX 4050 (§8.6). Scope: **only `batspot-train.ipynb`** was changed.
The previous Kaggle run's outputs were cleared (they describe the old code); the original, with those
outputs, is `/home/gb/batspot_gpu_experiments/batspot-train.ORIGINAL_with_kaggle_outputs.ipynb`.

### 8.1 Why — the three questions that triggered this

1. *"Why does the classifier stop at ~90 epochs?"* → It stopped at epoch **84**: best val at epoch 24 + `early_stopping_patience_epochs=60` (raw epochs) = 84. Working as configured, **not a bug** — and not the problem: training loss was ~0.0004 from epoch ~30 (memorised), and raising patience only prolongs an over-fit state. (§6.3's "waits 120 real epochs" is wrong; see the note there.)
2. *"Why are fine-tuned models less accurate than the official ones?"* → They are **not**; the comparison was invalid (§8.2 item 2).
3. *"How do we raise accuracy of every model?"* → Fix the input windowing (the dominant cause), the optimiser regime, and the selection metric (§8.2–8.4).

### 8.2 Diagnosis — symptom → root cause → proof

1. **Only the first 20 ms of each clip was ever used (dominant bottleneck).**
   - *Cause:* `CachedBatDataset` did `spec[:seq_len]`. `seq_len` = 20 ms (30 frames @192 kHz, 39 @250 kHz) but clips are **15 ms – 2.6 s, median 375 ms; 96 % are >100 ms** (clips are Raven-selection *bouts* of several pulses). The window was the same one every epoch and often not the call at all. 674 fixed inputs → memorisation (train loss 0.0004, val flat), no augmentation effect, val/test limited to whatever sat at t=0.
   - *Official behaviour:* `PaddedSubsequenceSampler(random=augmentation)` crops a **random** 20 ms window when training (centre crop otherwise) and `StridedAudioDataset` slides a 20 ms window over the recording at prediction. Official training clips are short, tight calls, so this never mattered upstream; it matters here.
   - *Proof:* same code/LR/batch, only the window choice changed → detector m09 test **0.841 → 0.931**, classifier **0.754 → 0.853** (§8.5).
2. **"Official accuracy" in Cell 17 was noise recall, not accuracy.** `run_model_evaluation` kept only files whose class name is in the model's class list. Official detectors know `{noise,target}`, the official classifier 15 European codes; our files are species codes, so only the 186 `noise` files matched. "m03 = 0.984" = *says noise on 98 % of noise clips*. I reproduced 0.984 / 0.876 / 0.801 exactly with an independent re-implementation (confirms it). Zero-shot on the real task (call vs noise, same 145 test clips): official detectors **balanced accuracy 0.73–0.80**, AUC 0.84–0.88 (plain acc 0.63–0.70 is the wrong column: they over-predict `call` on this 81/19 split); fine-tuned bal. acc 0.89–0.92, AUC 0.93–0.97, i.e. fine-tuning gains **+0.09 to +0.19 balanced accuracy** (corrected in §8.10 after the §9 review). The official classifier has zero overlapping species and, as a noise detector, balanced accuracy 0.50 (AUC 0.49).
3. **Too few optimiser steps, "fixed" with a destabilising LR.** Batch 128 / 674 clips = **6 steps/epoch**. The official run used batch 16 (≈42 steps/epoch, 150 epochs ≈ 6 k steps). §6's response — LR 1e-2 — is 100× the official 1e-4, and the logged run was unstable with it (val acc fell to 0.42 at epoch 38). The right lever is more steps: **batch 32** (22 steps/epoch). Measured: lr 3e-4 > 1e-3 ≈ 1e-4 (§8.5); 1e-2 is not needed.
4. **Model selection / scheduler on plain accuracy.** The detector split is 117 call / 28 noise: "always call" scores 0.807, indistinguishable from learning. The m03 detector in the last Kaggle run was exactly that (test acc 0.8069 = 117/145, precision 0.40, recall 0.50) while `ReduceLROnPlateau` (watching the flat val accuracy) decayed LR to 1.2e-8 by epoch ~160. Selection, LR schedule and early stopping now use **balanced accuracy** (`SELECT_METRIC`), which scores a constant predictor 0.5. Val is 145 clips (1 clip = 0.7 pt); best-epoch-on-val is optimistic by several points (classifier val 0.8345 → test 0.7655).
5. **The detector → classifier cascade does not help.** The classifier already has a `noise` class, so a hard detector gate can only add its own errors: classifier alone 0.853 vs gate 0.841 (9 detector×classifier seed pairs, m09; the gate loses all 9; re-derived from the stored probabilities by `/home/gb/batspot_gpu_experiments/cascade_pairs.py` — with a val-tuned threshold the gate is 0.828). In the final notebook run: alone 0.876; gate 0.855 / 0.876 / 0.835 (m03/m09/m11); soft combination 0.883 / 0.862 / 0.862 — within noise of "alone". The detector is still useful for what BatSpot is for (finding calls in long recordings with the sliding window); it is not an accuracy booster for pre-cut clips.
6. **Recording leakage inflates every number.** Filenames carry the tape id. There are only **28 recordings**; *heti is a single recording* and 86 of the 106 *rhbe* clips come from one. The stratified clip-level split puts clips of the same recording in train and test (**145/145 test clips share a recording with train**). Held-out-recording check (`StratifiedGroupKFold`, 2 seeds): detector m09 **balanced accuracy 0.64–0.71** (AUC 0.81–0.86) — the "accuracy ≈ 0.88" first quoted here was plain accuracy on the 81/19 split and hides noise recall of 0.30–0.44; classifier overall accuracy 0.42–0.49 (heti vanishes from training), ≈ 0.65 on the species that can be judged (heti/rhbe excluded, n=80). Treat in-split scores as an upper bound on performance for new recordings.
7. **Front-end differences vs the official code (made faithful, effect not isolated).** Old: numpy STFT, `scipy.ndimage.zoom` (cubic spline) on the *power* spectrum (rings negative → clipped to the −100 dB floor), channel 0 only. New (official): `torch.stft(center=False)`/√Σw², **nearest** interpolation of the cropped band, mean over channels, `kaiser_best` resampling, dB floor −100, **per-window min-max**. Verified `min_max_norm=true` in all official `config_train*` files, so min-max is correct (not the 0/1-dB scheme).

### 8.3 What changed in `batspot-train.ipynb`

| Cell | Change | Logic |
| :--- | :--- | :--- |
| 1 (md) | Describes the windowing | Reader must know the model sees 20 ms windows |
| 2 config | New `WINDOW_MODE` (`'energy_crop'`/`'first'`), `WINDOW_STRIDE=3`, `TRAIN_TOP_FRAC=0.20`, `TEST_TOPK=5`, `SELECT_METRIC='balanced_accuracy'`, `SPLIT_BY_RECORDING=False`. Detector: batch 32, lr 1e-4, 100 ep (raised from 60 after the Kaggle run, §8.9), eval every epoch, LR-patience 12, ES-patience 30. Classifier: batch 32, **lr 3e-4**, 120 ep, LR-patience 15, ES-patience 40 (all in RAW epochs) | §8.2 items 1,3,4; values from the §8.5 sweeps |
| 4 imports | + `balanced_accuracy_score` | selection metric |
| 6 dataset | `BatDataset`/`CachedBatDataset` → **`WindowedBatDataset`** + `predict_proba()` + official front end. Caches the *full-clip* dB spectrogram once (`spec_cache/<hash>/`, cache version `v2`, memory-mapped), crops windows at read time | item 1, 7. Train: random window from the loudest 20 %; eval: top-5 non-overlapping loudest windows, softmax averaged per clip |
| 7 data | Leakage check always printed; optional `SPLIT_BY_RECORDING` | item 6 |
| 9 train | Validation = clip-level `predict_proba`; score/scheduler/early-stop on `SELECT_METRIC`; history keeps `val_acc` + `val_bal_acc`; prints steps/epoch | items 3, 4 |
| 11, 13 | Build windowed train/val/test datasets from the config dicts | — |
| 12 | `evaluate_model` uses windowed clip-level probs; adds balanced accuracy | consistent protocol |
| 14 | Reports **classifier alone / hard gate / soft combine** per detector | item 5 |
| 15 | Prints a note: exported models need `min_max_norm=true` at prediction | §Gotchas |
| 17–19 | Official models scored **zero-shot on the test split** with the same windowed protocol (detectors as call-vs-noise; the no-overlap classifier as noise-vs-call); final table "official vs fine-tuned" | item 2 |

Unchanged on purpose: 3-detector loop, shared 70/15/15 split (seed 42), unweighted loss + `WeightedRandomSampler`, the `shortcut` architecture port, AMP, DataParallel path, export format, `RUN_CLIP_EXTRACTION` guard, `USE_AUGMENTATION=False`.

### 8.4 The logic behind the windowing design

- **Why crop at all:** the encoder was pretrained on 20 ms inputs (`sequence_len=20`); 20 ms is the model's input size, so a clip must be reduced to 20 ms windows. The clip label is valid for any window that contains the vocalisation, so every epoch can show a *different* valid sample of each clip — a large, free augmentation (≈ #windows per clip), which is what a 674-clip dataset needs.
- **Why the loudest windows, not uniform:** between pulses a bout contains silence. A uniform crop there is labelled "species X" but has no call → label noise. The score (mean over frames of the per-frame max dB) is a cheap pulse proxy; training samples uniformly from the top 20 % of windows (≥3). The same rule is applied to noise clips, so the model cannot cheat on "this clip was chosen because it is loud".
- **Why top-k average at test time:** one window can still miss; averaging the softmax of the top-5 non-overlapping windows mirrors the official sliding-window-then-aggregate prediction and cuts variance. Detector m09 (single 25-epoch run, `results_topk/res_D2_crop_m09.json`): top-1 0.903, top-3 0.931, top-5 0.931, top-10 0.931 → 5 is enough *(n=1: weak evidence, `TEST_TOPK=5` was never swept over seeds)*.
- **Why per-window min-max:** official models are `min_max_norm=true` and `predict.py` normalises each 20 ms window separately; training windows are normalised the same way so train/val/test/prediction see one distribution.
- **Why batch 32 / lr 3e-4 and not 1e-2:** the real deficit was optimiser steps (6 vs ~42 per epoch), so fix steps, keep the LR near the official 1e-4 (3e-4 won the sweep).
- **Why balanced accuracy:** it is mean per-class recall, so predicting the majority class scores 0.5 instead of 0.81.
- **Pretraining matters little here:** from-scratch encoders scored 0.931 (detector m09) and 0.841 (classifier) vs 0.933 / 0.853 pretrained (1 seed each). Keep fine-tuning (free, never worse in these runs), but do not expect the official weights to be the accuracy lever — more recordings are.

### 8.5 Controlled experiments (GPU, split seed 42, test n=145, mean over seeds; SE ≈ ±3 pt)

| Task | Input | Result (test acc) |
| :--- | :--- | :--- |
| Classifier | first 20 ms, lr 1e-4 / 1e-3, 60 ep (n=3) | 0.754 / 0.745 |
| Classifier | energy-crop + top-5, lr 1e-4 / **3e-4** / 1e-3, 60 ep (n=3) | 0.825 / **0.853** / 0.828 |
| Classifier | energy-crop, lr 3e-4, 120 ep (n=1) | 0.862 (best epoch still ~60) |
| Classifier | energy-crop, lr 3e-4, scratch encoder (n=1) | 0.841 |
| Detector m09 | first 20 ms (n=2) → energy-crop (n=3) | 0.866 → **0.933** |
| Detector m11 | first 20 ms (n=2) → energy-crop (n=3) | 0.848 → **0.908** |
| Detector m03 | first 20 ms (n=2) → energy-crop (n=3) | 0.876 → 0.869 (best epoch 2–4: weakest pretrain, over-fits fast) |
| Detector m09 | energy-crop, scratch (n=1) | 0.931 |
| Zero-shot official detectors | old input, first 20 ms | AUC 0.76–0.79; sliding-window max aggregation: acc 0.855–0.873 on all 964 clips |
| Recording-held-out | detector m09 first vs crop (n=2) | acc 0.880 vs 0.880 (AUC 0.791 vs 0.835) |

### 8.6 End-to-end verification of the actual notebook (single seed, RTX 4050, 396 s total)

Run with `/home/gb/batspot_gpu_experiments/run_notebook_locally.py` (executes every code cell, only Kaggle paths swapped; log: `notebook_local_run_2026-10-04.log`). All 19 cells ran without error, including the clip-extraction guard and the evaluation of the 14 local model files in `Data/model_output`.

| | Before (last Kaggle run) | After |
| :--- | :---: | :---: |
| Classifier test acc / macro-F1 | 0.7655 / 0.7749 | **0.8759 / 0.8779** (early-stopped at 113; best epoch 73) |
| Detector m03 test acc (bal. acc) | 0.8069 (all-call collapse) | **0.8759 (0.869)** |
| Detector m09 | 0.8483 | **0.9241 (0.899)** |
| Detector m11 | 0.8552 | **0.9103 (0.890)** |
| Cascade, classifier alone / hard gate / soft (m09) | — / 0.7379 / — | 0.876 / 0.876 / 0.862 |

Hardware note: the checked-in `venv/` is CPU-only. A first attempt on 16 CPU threads took ~35 min/run and drove the laptop to ~100 °C; on the GPU a 30-epoch run is ~40 s and the GPU stays ≤69 °C. Create a CUDA venv (`pip install torch` gives cu130) instead of reusing `venv/`.

### 8.7 Known limits — read before trusting the numbers

- **Kaggle verification done** (§8.9, 2×T4 with `DataParallel`, batch 16/GPU): same conclusions as the local run, no errors.
- **Single test split of 145 clips** → ±3 pt noise; differences <3 pt (e.g. cascade variants, m03 vs scratch) are not meaningful. Multi-seed numbers are in §8.5; the notebook run is one seed.
- **Recording leakage (§8.2 item 6) is not fixed**, only reported. The honest fix is more recordings per species (heti = 1, rhbe ≈ 2) and reporting a `SPLIT_BY_RECORDING=True` score.
- **Untested options:** `USE_AUGMENTATION=True` on the windowed dataset; `WINDOW_MODE='first'` is kept only as an ablation switch.
- **Exported `.pk` files do not record the normalisation mode**; set `min_max_norm=true` when predicting (§Gotchas; `PREDICTION/config` defaults to `false`).
- **Other notebooks** (`BatSpot_FineTune_Kaggle.ipynb`, `batspot-train-in-kaggle.ipynb`, …) still have the first-20 ms bug and the old LR/patience values.
- **Energy window selector is a heuristic.** It assumes the loudest part of a clip is the call; a loud non-bat transient in a call clip could be picked. Top-k averaging limits the damage.

### 8.8 Where everything is

`/home/gb/batspot_gpu_experiments/` (outside the repo): `gpuexp.py` (controlled-experiment engine), `sweep1.py`/`sweep2.py`, `zeroshot.py`, `results/*.json` (per-run metrics + test probabilities), `run_notebook_locally.py`, `apply_notebook_patch.py` + `new_notebook_cells/` (exact source of each replaced cell), `README.txt`.

### 8.9 Kaggle run of the overhauled notebook (2026-10-04, 2×Tesla T4, torch 2.10.0+cu128, DataParallel)

Source in the executed notebook is byte-identical to the delivered `batspot-train.ipynb`. All 18 code cells ran, 14.9 min total (dataset caching 127 s, detectors 399 s, classifier 339 s), **no stderr output at all** (the `Exception ignored ... can only test a child process` teardown spew of §3.5 did not appear in this run; that does not prove it is fixed). Test split = 145 clips, single seed.

| | Before (old Kaggle run) | After (this run) | Local RTX 4050 run (§8.6) |
| :--- | :---: | :---: | :---: |
| Classifier test acc / bal. acc | 0.7655 / — | **0.8552 / 0.8585** (macro-F1 0.858) | 0.8759 |
| Detector m03 acc (bal. acc, AUC) | 0.8069 (collapsed) | **0.8828 (0.887, 0.933)** | 0.8759 |
| Detector m09 | 0.8483 | **0.9103 (0.917, 0.971)** | 0.9241 |
| Detector m11 | 0.8552 | **0.8966 (0.909, 0.966)** | 0.9103 |
| Official detectors zero-shot m03 / m09 / m11 (acc, AUC) | same cell mis-reported noise recall | 0.697, 0.883 / 0.628, 0.836 / 0.676, 0.857 | identical to 4 decimals |
| Cascade (classifier alone / hard gate / soft), m03·m09·m11 | 0.7655 vs 0.7655 / 0.7379 / 0.7310 | 0.855 / 0.855·0.821·0.855 / 0.841·0.841·0.835 | — |

Findings (each from the logged output):
- **The fix transfers**: every model improved 4–9 pt and Kaggle lands inside the local multi-seed ranges (§8.5), i.e. run-to-run spread is ~±2 pt. m03 no longer collapses (balanced-accuracy selection works).
- **Classifier stopped at epoch 99 = best epoch 59 + patience 40**, as configured. Train loss ends at ~0.33 (it was 0.0004): the model no longer memorises; val plateaus at 0.83–0.84 accuracy from epoch ~30, so more epochs will not help. LR was halved at ~epoch 45, 79, 91.
- **Remaining classifier errors are concentrated** (21 of 145): `acsh` is the sink (10 of the 21 errors are predictions of acsh: alte→acsh 4, rhro→acsh 4, rhle→acsh 1, noise→acsh 1); `rhro` recall 8/14 and `rhle` 9/12; `alte`→`acsh` 4/27. **heti (10/10), rhbe (16/16) and sasa (14/14) are perfect, and these are the classes with the fewest recordings** (§8.2 item 6) — consistent with recording leakage, not proof of it.
- **Detector m09 hit the `n_epochs=60` cap** with its best epoch at 58 (balanced accuracy 0.930; the last-10-epoch *mean* was 0.889, so "still rising" compared a max with a mean — the cap raise is harmless but the gain is ≈ noise, see §8.10): **done:** `DET_CONFIG['n_epochs']` is now 100 in Cell 2 (ES patience 30 still stops the other variants early). Verified by a full local re-run (completed, 0 errors): the log shows `Epoch N/100`; m09 ran past the old cap and early-stopped at epoch 81 (test 0.931), m03 at 57, m11 at 47. m03 stopped at 58 (best 28), m11 at 53 (best 23).
- **Detectors lean towards "noise" at 0.5**: noise precision 0.64–0.70 with recall 0.89–0.93, call precision 0.97–0.98 with recall 0.88–0.91. The sampler shows noise 50 % of the time but it is 19 % of the data; the val-tuned gate thresholds (0.30 / 0.40 / **0.10**, the grid edge for m11) say the same. For use on real recordings, where noise windows vastly outnumber calls, the operating threshold must be chosen on full-length recordings with the sliding window — **nothing in this repo has evaluated that yet**.
- **The cascade still does not help** (m09 gate −3.5 pt: rhro recall falls 0.57 → 0.29). Use the classifier alone on pre-cut clips; keep the detector for scanning long recordings.
- **Kaggle was ~2.3× slower than the local RTX 4050** (14.9 vs 6.6 min), probably DataParallel on 16-clip per-GPU batches plus 4 CPU workers; not measured further.

Extra checks run after the Kaggle result (local GPU, saved predictions; same split, SE ±3 pt), so that "more accuracy" advice is evidence-based:

| Idea | Result | Verdict |
| :--- | :--- | :--- |
| Average 3 classifier seeds | 0.869 vs singles 0.841 / 0.855 / 0.862 | +≈1.5 pt, within noise |
| Average 3 seeds or 3 mics (detector) | 0.924–0.931 vs singles 0.931–0.938 (m09) | no gain |
| Longer input window: 40 ms (78 frames) | 0.862 / 0.841 (mean 0.852) vs 0.853 for 20 ms | no gain, 2× slower |
| Longer input window: 60 ms (117 frames), 1 seed | 0.821 | worse (GPU memory pressure too) |

Recommended next steps, in order: (1) ~~`DET_CONFIG['n_epochs']` 60 → 100~~ (done, Cell 2); (2) choose the detector threshold and judge false alarms on full-length recordings; (3) more recordings / label review for `rhro`, `rhle`, `acsh`↔`alte`, and report a recording-grouped score (`SPLIT_BY_RECORDING=True`) next to the in-split one; (4) report mean ± sd over several seeds rather than one split. Not worth doing: LR 1e-2, longer windows, ensembles for the detector, the detector→classifier gate. Window-length results: `/home/gb/batspot_gpu_experiments/results_window_length/`.

### 8.10 Fixes taken from the §9 review, and what the evidence says (2026-10-04)

I (Claude) checked every §9 claim against the notebook code and the saved artefacts before acting. **Confirmed and fixed in `batspot-train.ipynb`** (backup of the pre-fix notebook: `/home/gb/batspot_gpu_experiments/batspot-train.before_review_fixes.ipynb`):

| Cell | Fix | Why / proof |
| :--- | :--- | :--- |
| 2 | `assert SELECT_METRIC in ('balanced_accuracy','accuracy')` | B3: a typo used to fall back to plain accuracy silently while the log still printed the typo'd name |
| 6 | `_window`: **min-max first, then pad** (official order) | B2: padding a dB array with 0.0 first makes the padding the maximum — reproduced: padded region 1.0, real signal squashed to 0.70; now 0.0 / 1.0. Affects exactly **1 of 964** local clips (`sasa-bat_6431090_2026_20260525-192000_96139_96154`, 15 ms: 22 frames vs the 30 needed at 192 kHz, 29 vs 39 at 250 kHz), so negligible for the reported numbers — but reachable, and likelier on other data (§9's "0 of 964" and my own first note "unreachable here" were both wrong; counted from the cached spectrograms) |
| 6 | `train_window_starts`: keep exactly `ceil(top_frac·n)` windows **by rank** (stable sort), not a percentile threshold | Ties kept *every* window: on the real cache median kept-fraction 0.202 but max 1.000 and 3 clips ≥ 90 % (uniform crop incl. silence = label noise). Unit test: 100 tied windows → 20 kept |
| 7 | `recording_id` searches the Raven timestamp `\d{8}-\d{6}` instead of taking field 4; warns when every clip is its own recording | B4: identical 28-group partition on this data (verified); fragile positional parse no longer silently wrong on other file layouts. Note Cell 8's extractor, on this repo's selection naming (`acsh_devon_<date>_<time>.txt`), writes names with **no** Raven timestamp, so the new warning fires there instead of a silent false all-clear |
| 9 | `train_model`: best weights are **always** snapshotted/restored (was gated on `save_path`); `best_score` starts at `-inf`; `.detach().clone()`; selection via dict lookup | B1: with `save_path=None` (the signature default) the last-epoch weights were returned with the best score reported. Unit test with a stubbed validator: returned weights = best validation (3.0) |
| 18 | Summary table has `n` and `Note` columns, flags `NO SIGNAL (AUC~0.5)`, and prints the balanced-accuracy gain official → fine-tuned | B5: the official classifier was shown as 0.8069 "accuracy" = 117/145 = answering `call` for everything |

**Verification:** unit tests of each fix on tiny inputs, then the **whole notebook re-run on the GPU** (`run_notebook_locally.py`, 431 s, 0 errors; log `notebook_local_run_2026-10-04_review_fixes.log`; the leakage check still prints 145/145, no recording warning).

**Corrections to what I wrote earlier in §8** (already edited in place above): official detectors' headline figure is balanced accuracy 0.73–0.80, not acc 0.63–0.70 (gain from fine-tuning +0.09…+0.19, now printed by Cell 18: +0.088 / +0.167 / +0.167); held-out-recording detector balanced accuracy is 0.64–0.71, not "≈ 0.88"; the m09 "still rising" claim compared a max with a mean. §9 also asserted that the "9 seed pairs" and the top-k ablation had no artefact: both exist and reproduce, but one lived only in scratch space — now saved as `cascade_pairs.py` and `results_topk/`. `TEST_TOPK=5` rests on a single run.

**Results of the re-run (single seed each; compare with §8.9):**

| Test accuracy | Kaggle (§8.9) | Local, after `n_epochs=100` | Local, after review fixes |
| :--- | :---: | :---: | :---: |
| Classifier | 0.8552 | 0.8759 | 0.8483 (ran all 120 epochs, best epoch 92, never early-stopped) |
| Detector m03 | 0.8828 | 0.8828 | 0.8828 (bal 0.887) |
| Detector m09 | 0.9103 | 0.9310 | 0.8759 (bal 0.896) |
| Detector m11 | 0.8966 | 0.8759 | 0.9241 (bal 0.926) |
| Cascade: classifier alone / hard gate / soft (m03·m09·m11) | 0.855 / 0.855·0.821·0.855 / … | — | 0.848 / 0.835·0.766·0.828 / 0.828·0.841·0.869 |

**Read these numbers honestly.** The fixes are latent on this dataset, so none is *expected* to move accuracy; the differences above are run-to-run noise (the training loop is not seeded). Note the **spread is larger than "±2 pt"**: m09 ranges 0.876–0.931 and m11 0.876–0.924 over three identical-config runs (≈ 5 pt), the classifier 0.848–0.876 (≈ 3 pt). So no single-run difference below ≈ 5 pt (detectors) / 3 pt (classifier) means anything, and the ranking of m03/m09/m11 is not established. The conclusions that survive: all three detectors beat the official zero-shot models by ≥ 0.09 balanced accuracy (AUC 0.93–0.97 vs 0.84–0.88); the classifier alone ≥ the hard-gated cascade in all three runs so far; the soft combination's 0.869 for m11 is +3 clips and not significant. The official-model rows are deterministic and identical to the previous run.

**Not applied** (available in the §9 notebook `batspot_train(claude_bug_fixes _by__SPACE_BUNNY_MODEL).ipynb`, which has still **not** been run end to end): the prior-corrected detector gate, the recording-grouped retrain cell, the augmentation / multi-seed cell (the obvious cure for the spread above), and the cache-hardening changes. Recommended next: add a multi-seed (mean ± sd) report and a recording-grouped score, since every claim above is limited by single-split noise and by the 28-recording leakage (§8.2 item 6).

---

## 9. Independent review + fixes — Space Bunny Free (2026-10-04)

**Author of this section: Space Bunny Free** (model ID `space-bunny-free`, provider `opencode`).
This is an *independent second review* of §8, not a continuation of it. §8 was written by
Claude; §9 is my own reading of the code, my own experiments, and my disagreements with it.

**Deliverable:** `batspot_train(claude_bug_fixes _by__SPACE_BUNNY_MODEL).ipynb` (22 cells; 11 of the
19 base code cells byte-identical, 7 rewritten, 2 appended). Base notebook untouched.
**Test evidence:** `/home/gb/batspot_gpu_experiments/spacebunny_tests/`.

- **6 suites, 107 assertions, 0 failures** — `test_fixes.py` (27), `test_dataset.py` (21),
  `test_b5_labels.py` (18), `test_metric_guard.py` (16), `test_cell7.py` (14), `test_bug1.py` (11).
  Every suite `exec`s the corresponding cells **extracted from the delivered `.ipynb`**, not a copy,
  so the tests exercise the artefact rather than the source I edited.
- **`redgreen.py` — red-green harness, 8/8.** For each fix it undoes that one fix in a scratch copy
  and asserts the matching suite then FAILS. This is what makes the other number mean something;
  see §9.8.

Reproduce:
```
cd /home/gb/batspot_gpu_experiments/spacebunny_tests
for t in test_*.py; do <repo>/venv/bin/python "$t"; done     # each prints PASS/FAIL per check
<python> redgreen.py                                          # each fix reverted -> suite must fail
```
Suites read cells from `/tmp/opencode/verify` by default; override with `NEW_CELLS=<dir>` (this is
how `redgreen.py` points them at a mutated copy). CPU, seconds.

### 9.1 What I verified as CORRECT (so §9 is not read as blanket criticism)

§8 is good work and most of it survives scrutiny. I confirmed these myself:

- **The front end is bit-exact** against the vendored official code — `max|new − official| = 0.000e+00`
  for both the 192 kHz detector and the 250 kHz classifier front ends, over real clips. Step for
  step: `1/sqrt(Σw²)` normalisation, `center=False`, periodic Hann, floor/ceil bin cropping,
  `nearest` band interpolation, the −100 dB floor, channel-mean, `kaiser_best`, pre-emphasis 0.98,
  and per-window min-max. The old `scipy.ndimage.zoom` negative-ringing artefact is genuinely gone
  (global min dB is exactly −100.0 over all 964 cached spectrograms, 0 NaN files).
- **The official-vs-fine-tuned diagnosis is right.** m03's old "0.9839" was noise recall; I
  reproduced 0.9839 / 0.8763 / 0.8011 independently and confirmed `target` support was 0.
- **Class-index mapping is name-based, not assumed** — `probs[:, model_classes['target']]`. This
  matters: the official classifier's `noise` sits at index **8** of 15, so a hard-coded `[:, 1]`
  would have been silently wrong.
- **Early stopping now counts raw epochs**, and simulating the loop from the config reproduces the
  classifier's reported stop at epoch 113 exactly.
- **`best_state` DataParallel handling is correct** (`best_state=None` initialised pre-loop,
  `module.` prefix stripped on both save and restore).
- **No double-weighting**: `CrossEntropyLoss()` unweighted *plus* `WeightedRandomSampler`, and
  `compute_class_weights` is correctly sized from `num_classes`.
- **No test leakage**: `det_test`/`cls_test` are built from `test_wavs` and first touched by
  `evaluate_model` after `train_model` returns.
- **The documentation is unusually honest** — `save_path` being dead (still true, §3 item 4), the
  energy-window heuristic being unvalidated, recording leakage being reported-but-not-fixed. §8.7
  lists its own limits. That is rarer than it should be and it is why the real bugs below were
  findable at all.
- **Almost every number in §8.5/§8.6 traces to an artefact.** I recomputed all 41
  `results/*.json` aggregates; every mean and every `n=` matches. `run_notebook_locally.py` patches
  only `!pip` lines and `/kaggle/` strings — it does not skip or alter any cell. The pre-change
  notebook is genuinely preserved with 1986 saved outputs.

### 9.2 Five real bugs — symptom → root cause → proof → fix

All five are **latent**: none is corrupting the numbers in §8.6/§8.9. They are traps for the next
person who edits this notebook.

#### B1 — a plot filename decided which weights `train_model` returned

- **Symptom.** `save_path` gated *both* the best-weights snapshot and its restore:
  `if save_path:` before the snapshot, `if save_path and best_state is not None:` before the
  restore. `save_path=None` is the documented default of the signature.
- **Root cause.** §3 item 4 recorded that `save_path` "is dead" — used only as a truthiness gate.
  It was worse than dead: it was load-bearing.
- **Proof (executed, `test_bug1.py`).** Tiny synthetic problem whose validation score rises then
  falls, so best ≠ last epoch. Re-scoring whatever `train_model` handed back:

  | | reported best | re-scored | |
  |---|---:|---:|---|
  | base, `save_path=None` | 0.3690 | **0.3214** | last-epoch weights, best score reported |
  | base, `save_path=set` | 0.3690 | 0.3690 | correct |
  | fixed, `save_path=None` | 0.3690 | **0.3690** | correct |
  | fixed, `save_path=set` | 0.3690 | 0.3690 | correct |

- **Fix.** Gates removed; restore is unconditional. Also `best_score = -inf` (was `0.0`, so a first
  validation scoring exactly 0.0 would never be captured), `.detach().clone()`, and `save_path` now
  actually writes the training curves (`fig.savefig`) — `grep savefig` found nothing in §8's
  notebook, so the `*_curves.png` names were fiction.

#### B2 — silence was handed to the model as the loudest thing in the window

- **Symptom.** `_window` did `minmax_normalize(pad_window(win, seq_len))`: pad **first**, normalise
  second. A literal `0.0` inserted into a **dB** array is normally the array *maximum*, so min-max
  divides by the padding and maps every padded row to **1.0**.
- **Root cause.** The official order is normalise → pad (`animal_spot/data/audiodataset.py:622-629`,
  `t_norm` then `t_subseq`), which leaves the pad region at 0.0.
- **Proof (`test_fixes.py`, `test_dataset.py`).** On a sub-window clip inside a 20 ms window:

  | | padded region | real signal max |
  |---|---:|---:|
  | base | **1.0000** | 0.4192 (squashed) |
  | fixed | **0.0000** | 1.0000 |

- **Reachability, stated honestly.** 0 of 964 local clips are shorter than one window (min 45
  frames vs `seq_len` 30/39). But §8.2 item 1 states clips run "15 ms – 2.6 s", and the minimum
  local duration *is* 15 ms, so the path is reachable on other data — and the code had no guard.
- **Fix.** `win = minmax_normalize(win)` then `pad_window(win, seq_len)`; count of sub-window clips
  printed at construction.

#### B3 — one character re-silently reinstated the bug the metric switch exists to prevent

- **Symptom.** `val_score = val_bal if SELECT_METRIC == 'balanced_accuracy' else val_acc`. No
  validation, no warning — and the log *prints* `selection metric: {SELECT_METRIC}`, actively
  confirming a metric the code is not using.
- **Proof.** With a validation sequence where accuracy is exactly constant (0.4000) and balanced
  accuracy rises 0.2479 → 0.5195: `'balanced_accuracy'` → no early stop, `best=0.5195`;
  `'accuracy'` **and** `'balanced_acc'` → identical traces, `Early stopping at epoch 4`,
  `best=0.4000`. A typo reproduces the *entire* §8.2-item-4 failure with zero diagnostics.
- **Fix.** `assert SELECT_METRIC in (...)` at config time, and selection via dict lookup, so an
  unhandled value is a `KeyError` rather than a behaviour change (`test_metric_guard.py`, 16 checks
  over 7 candidate values).

#### B4 — the leakage check could report a false all-clear

- **Symptom.** `recording_id()` returned `parts[3]` of a 6-field
  `CLASS-LABEL_ID_YEAR_TAPE_START_END.wav` name, falling back to *the whole file name* otherwise.
- **Root cause.** Cell 8 — enabled by default, `RUN_CLIP_EXTRACTION = True` — writes **4-field**
  names (`{species}-bat_{tape}_{start}_{end}.wav`). Every clip produced by this notebook's own
  extractor therefore counted as its own "recording".
- **Consequences.** The leakage check would print `0/N test clips share a recording with train` — a
  false all-clear — and `SPLIT_BY_RECORDING = True` would degenerate into a plain random split while
  claiming to hold out tapes. **The entire §8.2-item-6 narrative rests on this one function.**
- **Proof (`test_b5_labels.py`).** Calls the notebook's **own** `recording_id` — obtained by `exec`ing
  Cell 7 — with both layouts: 4-field → `20260429-192000` (old code returned the filename, making
  every clip its own group), 6-field → `20260525-192000`. On the real dataset both old and new give
  28 groups, so this fix changes **nothing** here and **everything** on re-extracted data.
- **Fix.** Match the Raven timestamp `^\d{8}-\d{6}$` wherever it appears in the stem — present in
  both layouts. Plus a `n_groups == n_files` warning, and a per-class recording-count table (which
  makes the `heti = 1`, `rhbe = 2` problem impossible to miss).

#### B5 — the summary table presented a signal-free model as "81% accurate"

- **Symptom.** `OFFICIAL vs FINE-TUNED` printed
  `official official_classifier_m09  0.8069  0.5000  0.4924`. The per-model block carried a `kind`
  label; the headline table dropped it.
- **Why it matters.** `0.8069` is exactly 117/145 — the model answers "not noise" for **every**
  clip — and its AUC is 0.4924, i.e. *no noise signal at all*. §8.2 item 4 was written specifically
  to stop plain accuracy on an 80/20 split from being read as skill; the fix table reintroduced it
  one cell later, for the one model where it bites hardest.
- **Fix.** `kind` and `n_files` columns, an explicit `NO SIGNAL (AUC~0.5)` marker, rows grouped by
  denominator, and a closing line quoting **balanced accuracy** as the fair comparison.

### 9.3 Also hardened (same failure class: silent wrong numbers)

| Issue | Was | Now |
| :--- | :--- | :--- |
| Spectrogram cache validity | `os.path.exists(...)` — re-extracted clips silently reused **old** spectrograms | source wav size+mtime stored in a sidecar and validated on load |
| Cache filename | bare basename — two `DATA_DIR`s with colliding basenames shared one entry | source directory hashed into the name |
| `_CACHE_VERSION` | hand-maintained `'v2'` — editing the front end without bumping it reused stale cache | derived from `clip_to_db_spectrogram`'s source (with a fallback if the source is not locatable) |
| `np.save` | not atomic — an interrupted build left a truncated `.npy` that every later run mmap'd as garbage | write to `.tmp` + `os.replace` |
| A bad wav | `torch.stft` `RuntimeError` killed the whole run | per-file `try/except`, failures listed, clip **dropped visibly** |
| NaN spectrogram | fed straight to the model | `np.nan_to_num` |
| `train_window_starts` | percentile threshold — **ties select everything**. Measured: kept-fraction median 0.202 but max 1.000; 3 clips fully degenerate, i.e. uniform crops over the whole clip *including silence* = label noise | exact `ceil(top_frac·n)` by score, stable sort |
| `predict_proba` | a clip with 0 windows → all-zero row whose `argmax` is class 0, a free "correct" prediction | hard error listing the offending files |
| `n_freq_bins` | `dataOpts.get('n_freq_bins', 256)` — silently 256 for a model storing `num_mels` | `predict.py`'s fallback chain + a warning on non-`linear` `freq_compression` |
| `names` for the confusion matrix | built by sorting `model_classes` by value — non-contiguous indices ⇒ `len(names) < n_outputs` ⇒ `confusion_matrix` reshape `IndexError` | indexed by output width |
| `roc_auc_score` in the summary | unguarded — raises on a single-class fold, reachable under `SPLIT_BY_RECORDING=True` where `heti` vanishes | guarded |
| `best_acc` / `'best_val_acc'` | names held a *balanced* accuracy | renamed in the new notebook's own cells |
| `DET_CLASS_TO_IDX` | printed `{'noise': 0, 'call': 1}` but no dataset used it — a `KeyError: 'acsh'` away | removed; Cell 11 builds the real map |

### 9.4 Challenge to Claude's claims in §8

Each item is something I checked and found **wrong, unsupported, or misleadingly stated**. I have
kept the substance and only corrected the reasoning — none of this invalidates §8's conclusions.

1. **§8.9's "balanced accuracy still rising (0.89 → 0.93 over the last 10 epochs)" is a
   max-versus-mean comparison and does not support raising `n_epochs`.** 0.89 is the *mean* of
   epochs 45–60; 0.9301 is the single *maximum*, at epoch 58. Regressing properly:

   | window | slope/epoch | t | extrapolated gain over the extra 40 epochs |
   |---|---:|---:|---:|
   | epochs 20–60 | +0.00094 | +2.09 | +0.037 |
   | epochs 31–60 | +0.00138 | +2.15 | +0.055 |
   | epochs 41–60 | +0.00159 | +1.52 | +0.064 |
   | epochs 45–60 | +0.00050 | +0.43 | +0.020 |

   There *is* a marginal trend over epochs 20–60 (t ≈ 2.1) — I was too strong when I first called
   this unsupported. But over the last 20 epochs it is indistinguishable from zero, and the
   expected gain is **+0.02 to +0.06 balanced accuracy = 1 to 3 clips on a 145-clip val set, where
   one clip is worth 0.0221**. So the change is harmless and possibly mildly positive; the
   *justification* was invalid. **§8.9's own local re-run confirms this prediction**: m09 test
   0.931 vs 0.9241 at the old cap — +0.7 pt, i.e. inside the noise band. Rather than keep arguing,
   the new notebook instruments it: `REPORT_TOPK_MEAN` prints the mean of the top-3 validation
   scores beside the argmax, so selection optimism becomes a measurement.

2. **§8.2 item 2 quotes the official detectors' `acc 0.63–0.70`, which is the wrong column.** The
   official detectors over-predict `call` on this 81/19 split (noise precision 0.33–0.39), so
   plain accuracy punishes them while leaving their *ranking* intact. On the metrics §8 itself
   promoted, the official baseline is **balanced accuracy 0.729–0.798, AUC 0.836–0.883**. I also
   recomputed the same `.pk` files under BatSpot's *own* sliding-window-max protocol from
   `zeroshot_det.pkl`: `acc 0.855–0.873, bal 0.719–0.783, AUC 0.847–0.879`. The two protocols agree
   to within 0.04 on balanced accuracy and AUC; only the accuracy-at-0.5 column differs, by 0.18–0.23.
   **So fine-tuning's real gain is +0.09 to +0.19 balanced accuracy, not the "doubling" the
   accuracy column implies** — smaller than advertised, but solidly real. (A first pass at this
   review concluded the notebook was understating the official models by 20–25 points; that was
   wrong, and comparing plain accuracy across protocols is precisely the habit §8.2 item 4
   condemns. The notebook's protocol choice is fine.)

3. **§8.2 item 5's "9 seed pairs" for the cascade has no artefact anywhere.** `results/*.json`
   stores only `{acc, bal, mf1, auc}` per task — no cascade, gate or soft-combine outputs exist, and
   `log_sweep1/2.txt` contain none. The notebook's own run shows a **tie** for m09 (0.8759 vs
   0.8759). This is the load-bearing claim for Cell 14's redesign. (The conclusion still holds —
   the gate never beat classifier-alone in any run I can see — but "9 seed pairs" is asserted, not
   shown.)

4. **§8.4's top-k ablation does not exist.** "Detector m09 (25-epoch run): top-1 0.903, top-3 0.931,
   top-5 0.931, top-10 0.931" — there is no 25-epoch run in `results/` (all are `e40`/`e60`), no
   top-k ablation script, and `topk` is hard-coded to 5 in every `gpuexp.run()` call. The
   `zeroshot.py` top-3 is a *sliding-window* top-3, a different quantity. **`TEST_TOPK = 5` has
   never been validated**, and §8.2 item 3 rests on it.

5. **Cell 2's hyperparameters were measured under a different training loop.** `gpuexp.py:94`
   selects checkpoints on `mv['acc']` (plain accuracy) and the sweep has **no `ReduceLROnPlateau`
   and no early stopping**. So `base_lr = 3e-4`, the epoch caps and the patience values were chosen
   for a training loop the notebook does not use. The conclusions survive — the final run
   reproduced and improved on them — but the comments in Cell 2 present them as measured optima
   *for this configuration*, which the evidence does not support.

6. **§8.5's held-out-recording row quotes plain accuracy (0.880 vs 0.880) on the 117/28 split —
   the exact habit §8.2 item 4 condemns.** The balanced accuracies are 0.689/0.660 (first 20 ms)
   vs 0.641/0.708 (energy crop): call recall 0.97–0.99 but noise recall 0.30–0.44. The honest
   held-out-recording figure is **≈0.64–0.71 balanced**, not 0.88. §8.2 item 6's conclusion
   ("≈0.88, not 0.93") is directionally right and quantitatively wrong.

7. **"This cell restores official behaviour" overclaims.** `PaddedSubsequenceSampler(random=True)`
   crops **uniformly** over the clip (`transforms.py:465-470`); this cell restricts to the loudest
   20 %, and at eval takes the top-5 *loudest* windows rather than a sliding window. Those are two
   deliberate deviations, one for training and one for inference. The measured gain is
   windowing-vs-`first`; the uniform-random-crop variant was never tested. (Also: `WINDOW_MODE='first'`
   reproduces the old *window choice*, not the old *front end* — cubic `zoom` on the power spectrum,
   no `1/sqrt(Σw²)`, channel 0 only.)

8. **§8.2 item 1's "96% are >100 ms"** — measured, 95.9%. Immaterial, but it is the one figure in
   §8 I could not reproduce exactly.

9. **§2.10 says the official detector's `n_fft` is 128. It is 256.** Read from the `.pk` `dataOpts`:
   all four official models are `n_fft 256, hop 128, n_freq_bins 256` (192 kHz detectors,
   250 kHz classifier). Cell 2 has it right; the §2.10 table is wrong.

### 9.5 What I added, and why

Requested scope was "bugs + accuracy improvements". The improvements are all *measurement*
improvements except the prior correction — deliberately, because §8 has already established that
the modelling knobs are exhausted (LR 1e-2, 40/60 ms windows, detector ensembles and the gate are
all measured dead ends).

1. **Prior-corrected detector gate** (Cells 7, 14). `make_weighted_sampler` draws each class with
   probability exactly `1/n_classes`, so the detector trains under a **50/50** noise/call prior
   against an **19/81** eval prior — a **4.18×** shift in odds. That is why m09 showed noise
   *precision* 0.70 against recall 0.93, and why the val-tuned thresholds sank to 0.30/0.40/0.10.
   A threshold fitted that way **encodes the test noise fraction** and will not transfer to real
   recordings, where noise is >95% of windows. `p_adj ∝ p · prior_eval/prior_train` makes 0.5 mean
   what it says. Raw and corrected thresholds are both reported so §8's numbers stay comparable.
   **I deliberately did not prior-correct the classifier**: its errors point the other way — 10 of
   21 are false `acsh`, the *most common* class — so prior shift is not its problem. Its problem is
   acoustic confusion between `acsh` and `alte` (both *Myotis*, near-identical 40–45 kHz FM sweeps).
   Correcting it would have been a plausible-sounding change in the wrong direction.
   *(A first draft of this cell inverted the detector prior — noise 0.807 / call 0.193 — which
   `test_cell7.py` caught. It would have made the correction anti-corrective. Fixed and verified.)*

2. **Recording-grouped retrain** (new Cell 21). The in-split number **cannot** be converted into a
   new-recording estimate by arithmetic: the leakage check says 145/145 test clips share a recording
   with train, so there is no clean subset to score. Holding out whole tapes and retraining is the
   only honest way. Reported next to the in-split column, with the classes that vanished from the
   grouped fold called out.

3. **Augmentation A/B and seed spread** (new Cell 20). Every number in this project comes from one
   145-clip split; run-to-run spread is ≈±2 pt, the same size as most differences being argued
   about (cascade variants, m03 vs from-scratch, top-1 vs top-5). Seeds 42/43/44 each **re-split and
   re-initialise**, so the reported sd is *total* spread — the conservative number, and the one that
   decides whether any 1–3 pt delta is real.

4. **Per-class recording counts** (Cell 7). Makes the `heti = 1` / `rhbe = 2` problem visible at the
   point where the split is made, rather than inferable only from a confusion matrix.

### 9.6 The honest bottom line

The windowing fix in §8 is the real thing, and it is the reason every model improved 4–9 pt. I
verified the mechanism quantitatively rather than taking it on trust:

| | candidate windows/clip | train windows/clip | distinct train samples/epoch |
|---|---:|---:|---:|
| Classifier 250 kHz | median 231, max 1701 | median 47, mean 56 | **53,879** (was 964) |
| Detector 192 kHz | median 178, max 1306 | median 36, mean 43 | **41,480** (was 964) |

**×56 and ×43 more distinct training inputs**, and the mechanism is visible in the log: classifier
train loss ends at **0.34** instead of **0.0004**. Memorisation became impossible.

But the same arithmetic explains why the *remaining* errors did not shrink: **all ~56 windows of a
clip come from one tape and are near-duplicates.** For generalisation the effective sample size is
still **28 recordings**. That is why the classes recorded once or twice are perfect (`heti` 10/10,
`rhbe` 16/16, `sasa` 14/14) while every class with many tapes carries the errors (`rhro` recall 0.57,
`acsh` precision 0.68). It is leakage, and it is not fixable in code.

**So the ranked answer to "how do we get more accuracy?" is:**

1. **More recordings** of `rhro`, `rhle`, and the `acsh`↔`alte` pair — plus a **label review** of
   `acsh`/`alte`, since 10 of 21 errors are false `acsh` and that pair is genuinely hard.
2. **Choose the detector threshold on full-length recordings** with the sliding window, and judge
   false alarms there. Nothing in this repo has ever evaluated that, and the test split's 19% noise
   fraction is unrepresentative of real audio. This is the largest *unmeasured* risk in the project.
3. **Report the grouped score next to every in-split number** — now automated (Cell 21).
4. **Report mean ± sd over seeds** — now automated (Cell 20).

Not worth doing, on evidence: LR 1e-2, longer input windows, detector ensembles, the
detector→classifier gate.

### 9.7 Known limits of *my* work — read before trusting §9

- **The new notebook has not been run end to end.** Every fix is verified by targeted tests
  (107 assertions, 6 suites) and the cells parse, but no full 22-cell execution on Kaggle or GPU
  has happened.
  The `DataParallel` path and the two new cells are untested at scale. **Run it before drawing
  conclusions from it.** Expected main-pipeline results should be close to §8.9's, since all five
  bugs are latent on this dataset.
- **Cell 2's hyperparameters are inherited, not re-derived** — see §9.4 item 5. I documented the
  caveat rather than re-running the sweep, which would have cost a GPU day.
- **The prior-correction factor is computed from the *test* split's class proportions** when applied
  at test time, and from the val split's when tuning. That is legitimate for reproducing an in-split
  number but it is **not** the deployment prior: for real recordings you must supply the actual
  expected noise fraction. The correction machinery is the deliverable, not a universal threshold.
- **The grouped check retrains only the classifier and m09** (`GROUPED_MICS`). Retraining all three
  mics triples the cost for no extra insight.
- **`REPORT_TOPK_MEAN = 3` is a convention**, not a derived value. The right choice depends on how
  many validations the run has.
- **My `recording_id` regex assumes the tape id contains a Raven `YYYYMMDD-HHMMSS` timestamp.**
  That holds for both layouts in this repo. A dataset with different naming would fall back to the
  file stem, and Cell 7 prints a loud warning when `n_groups == n_files` — but it would degrade to
  "no grouping" rather than error.
- **None of §9 changes §8's numbers**, and I have not re-run the pipeline. §9 is a review plus fixes
  plus instrumentation; the accuracy claims in §8.6/§8.9 stand as written, with the metric caveats
  in §9.4.

### 9.8 How the fixes were verified — and four vacuous tests I found in my own work

A test that passes both with and without the fix tests nothing. After the suites were green I wrote
`redgreen.py`, which reverts **one fix at a time** in a scratch copy and asserts the matching suite
then fails. Current result: **8/8 reverts detected.**

```
reverted fix                             suite                   exit  FAILs  detected
B1  save_path gates best_state           test_bug1.py               1      3  YES
B2  pad before min-max                   test_dataset.py            1      2  YES
B2  pad before min-max (unit)            test_fixes.py              1      3  YES
B3  SELECT_METRIC silent fallback        test_metric_guard.py       1      2  YES
B4  positional recording_id              test_b5_labels.py          1      3  YES
B5  unlabelled summary rows              test_b5_labels.py          1      3  YES
--  stale spectrogram cache              test_dataset.py            1      1  YES
--  percentile-tie window selection      test_fixes.py              1      1  YES
```

**The first run of this harness failed 4 of 9, and all four failures were my own test defects, not
notebook defects.** Recording them because the pattern is the whole point:

1. **B4 was tested against a copy, not the code.** `test_fixes.py` had `exec`'d its *own* inline
   `recording_id` and asserted against that. Reverting the notebook's function changed nothing the
   test could see. The assertion was true of a function no notebook would ever run.
   Fixed: `test_b5_labels.py` now `exec`s Cell 7 and calls the notebook's `recording_id` directly.
2. **B4 was also undetectable on this dataset.** `test_cell7.py` runs Cell 7 for real and found 28
   recordings both before and after the revert — because every file in `Data/final_dataset/data`
   uses the 6-field layout, which the old positional code handled correctly. **The bug is only
   reachable on Cell-8-extracted data.** A test on the available data cannot detect it; the fix has
   to assert on the 4-field layout explicitly, which is what the new suite does.
3. **B5 had no test at all.** When I rewrote `test_fixes.py` mid-task I dropped the static string
   checks for the summary table, and never replaced them. B5 was verified only by my reading. Its
   fix is the one with the most surface (a table), so this was the worst gap. Fixed:
   `test_b5_labels.py` builds synthetic `val_results` including a signal-free model, `exec`s Cell 18,
   captures stdout, and asserts on the **rendered table** — `n` column, `Note` column, `NO SIGNAL` on
   the dead row, no `official official_`, balanced-accuracy gain line.
4. **The stale-cache revert crashed instead of failing an assertion.** My revert helper returned
   `True` from `_cache_is_valid` when the file was *absent*, which is not the original bug (the
   original was `os.path.exists(...)` — valid on existence, ignoring the source). The harness reported
   exit≠0 and I nearly counted that as detection. It was an unrelated `FileNotFoundError`. Fixed by
   making the revert faithful to the original code.

Two further harness defects, worth noting because they would have produced a **false all-green**:
my first `sed` re-pointing the suites at the delivered notebook only patched 2 of 5 files (the other
three defined `NEW` with a filename appended), so three suites were still reading the scratch
directory — and `test_b5_labels.py` reads `NEW_CELLS` from the environment, which `redgreen.py`
was not passing, so B4/B5 "passed" against **unmutated** cells. Both are now asserted explicitly:
`redgreen.py` sets `NEW_CELLS`, and the reproduce block above greps for leftovers.

**What this does and does not establish.** It establishes that each fix changes the behaviour the
test observes, in the direction claimed, and that no fix is a no-op. It does **not** establish that
the notebook produces the right accuracy numbers — nothing here has run training. §9.7's first bullet
still stands: the notebook has not been executed end to end.

---

## 10. Notebook comparison + inference cells — Claude (2026-10-04)

Scope: compare `batspot-train.ipynb` with `batspot_train(claude_bug_fixes _by__SPACE_BUNNY_MODEL).ipynb`
(§9), port what is worth porting, and add cells that run the best detector + classifier over a folder of
new recordings and write a Raven-style selection text file. **Only `batspot-train.ipynb` was changed.**
Scripts and logs: `/home/gb/batspot_gpu_experiments/inference_cells_2026-10-04/` (outside the repo, §10.5).

### 10.1 The two notebooks, side by side

Same pipeline, same front end, same training loop, same split, same models. 11 of 19 code cells are
byte-identical. `batspot-train.ipynb` already carries the §8.10 versions of Space Bunny's B1–B5 fixes,
so the real differences are:

| Area | `batspot-train.ipynb` (before this section) | Space Bunny notebook | Done here |
| :--- | :--- | :--- | :--- |
| `recording_id` | regex `\d{8}-\d{6}` (hyphen only) | `(\d{8})[-_](\d{6})` | **Ported.** Cell 8 / AudioMoth names use `_`; the old regex fell back to one group per clip (with a warning). Same 28 groups on the current data. |
| `save_path` | only documented as inert; no curves written | writes a 3-panel PNG + top-k-mean diagnostics | **Ported (curves only).** |
| zero-window clip in `predict_proba` | silently scored as class 0 | `RuntimeError` | **Ported.** |
| detector threshold grid | 0.10–0.90 (m11 sat on the 0.10 edge) | 0.05–0.95 | **Ported.** |
| cache hardening (size+mtime sidecar, dir hash, atomic save, skip failed wavs, NaN guard) | `os.path.exists` | yes | Not ported: a Kaggle session starts with an empty cache, and dropping clips per dataset can desynchronise the detector/classifier test lists (the cascade then raises). |
| prior-corrected gate | — | yes | Not ported: with a validation-tuned threshold, rescaling P(call) by a constant prior ratio is monotone, so the tuned gate makes the same decisions (up to the grid); it only matters for a fixed threshold, and then the needed prior is the *deployment* one, which is unknown. The test-time correction also uses the test split's class balance. |
| augmentation A/B + 3-seed spread (Cell 20) | — | on by default, ~5 extra classifier trainings | Not ported (≈ +25 min on Kaggle). `SEED` added instead so a run is repeatable. |
| recording-grouped retrain (Cell 21) | — | on by default | Not ported; still the right next measurement (§8.7). Available in the Space Bunny notebook, which has never been run end to end (§9.7). |
| export | `model.cpu()` **in place** | same | **Fixed here** (both had it): after Cell 15 the live models sat on the CPU, so any later GPU use (the new inference cells) crashed. Now a CPU copy is exported. |

### 10.2 Changes to existing cells of `batspot-train.ipynb`

| Cell | Change |
| :--- | :--- |
| 1 (md) | pipeline step 9 (inference) |
| 2 | `RUN_CLIP_EXTRACTION = False` (the comment always said so; §3 item 6); `SEED = 42` |
| 6 | `predict_proba` raises on a clip without windows |
| 7 | `recording_id` accepts `-` or `_` between date and time |
| 9 | `set_seed()`; `save_path` now writes the training curves (§3 item 4); history keeps `val_epoch`, `lr`, `best_epoch` |
| 12, 13 | `set_seed(SEED)` before each model is built; `det_results[mic]['history']` kept |
| 14 | threshold grid 0.05–0.95 |
| 15 | export a CPU copy (`copy.deepcopy(model).cpu()`) instead of moving the live model |
| 20–24 | **new** inference section (§10.3) |

### 10.3 Inference cells (21–24): what they do and why each default is what it is

Upload a folder (or `.zip`) of recordings as a Kaggle dataset, set `INFER_INPUT_DIR` in Cell 21, run
Cells 21–23 (after training, or after Cells 2–6 with `INFER_DETECTOR_PK` / `INFER_CLASSIFIER_PK` set to
exported `.pk` files). Output: `/kaggle/working/batspot_detections.txt`, comma-separated, header exactly

`Selection,name_of_file,Channel,Begin Time (s),End Time (s),Begin Clock Time,End Clock Time,Low Freq (Hz),High Freq (Hz),Peak Freq (Hz),Delta Time (s),Species detected,Confidence`

plus `batspot_detections_rejected_noise.txt` (same format, selections the classifier calls `noise`) and one
tab-separated Raven table per recording in `raven_tables/` (`INFER_WRITE_RAVEN_TABLES`).

| Step | Implementation | Evidence / reason |
| :--- | :--- | :--- |
| Combo | detector variant with the best **validation** `SELECT_METRIC` (ties → m09) + the fine-tuned classifier | never selects on test; the m03/m09/m11 ranking is within noise anyway (§8.10) |
| Scan | 20 ms windows, 10 ms hop, per-window min-max, threshold 0.5 | official BatSpot `config_predict` for these detectors (`sequence_len=0.02`, `hop=0.01`, `threshold=0.5`, `min_max_norm=true`) |
| Front end | `clip_to_db_spectrogram` (the training function) on chunks of `INFER_CHUNK_S` = 60 s with 16 frames of left context | chunked vs one 300 s pass: max\|ΔP\| = 0 |
| Resampling 384→192 kHz | resampy `kaiser_best` at an exact 2:1 ratio is a fixed 199-tap FIR, applied as a strided GPU `conv1d` (TF32 off) | waveform max\|diff\| 6.7e-8 vs resampy; detector max\|ΔP\| 1.4e-4; self-check vs resampy runs on every start. Other ratios use resampy. |
| fp16 scan (`INFER_AMP`) | autocast for the detector scan only (training also ran forward passes under autocast) | 1.7× faster; max\|ΔP\| 2.4e-3, 4 of 29 998 windows flipped at 0.5. Classifier stays fp32. |
| Selections | positive windows ≤ `INFER_MERGE_GAP_S` = 0.1 s apart merge; ≥ `INFER_MIN_WINDOWS` = 2; longer than `INFER_MAX_SELECTION_S` = 1.0 s → split at the widest internal silence (central split on ties) | annotated boxes: median 0.375 s, 95 % < 1.05 s. Without the cap, continuous activity produced 5.8 s rows. Unit-tested (merge, split at silence, even split of continuous runs, no lost windows). |
| Species | each selection scored like a test clip (top-5 loudest windows, softmax averaged, fp32); `Confidence` = that averaged probability of the reported species; argmax `noise` → rejected file | same protocol as validation/test, so the classifier sees the input distribution it was selected on |
| Frequencies | spectrogram at native rate, Hann, `nfft = 2^ceil(log2(sr/375))` (1024 at 384 kHz = Raven's resolution in the training tables; Peak inside the annotated box matches Raven's exactly in 89 %), 50 % overlap; background = per-frequency median over the selection ± 1 s; Low/High = contiguous band around the bin highest above background, keeping bins ≥ 12 dB above background and within 25 dB of that bin's level; Peak = loudest cell in the band; search limited to 10–150 kHz | 400 annotated boxes, median error for bats: Peak 0.0 kHz (within one bin 65 %), Low 2.4 kHz, High 3.2 kHz. Rejected on the same boxes: plain −20 dB band (~10 kHz error); median over the selection only (rhle/rhro Peak ~57 kHz off: their CF calls fill most frames, so the median *is* the call); low-percentile backgrounds (13–39 kHz); 3×3 smoothing (breaks rhle). Without the ceiling a click gave 192 000 Hz (near-Nyquist bins have ~zero background). **Weak spot:** Low for rhle/rhro is still often 40–55 kHz too low (faint CF calls; the band bridges into noise); 84 of 1118 test rows touch the 10 or 150 kHz limit. |
| Clock time | start from `YYYYMMDD[_-]HHMMSS` in the file name (+ clip offset for BatSpot clip names), else the AudioMoth `Recorded at …` header, else 00:00:00 (reported) | 501 annotated boxes: identical to Raven's `Begin Clock Time` (max difference 0.00 ms) |

### 10.4 Verification

* **Smoke test with the official models** (skip-training path, `INFER_*_PK`): 80 files, all cells ran.
* **Full notebook end to end, twice** (local RTX 4050, cold cache, 618 s, 0 errors): all 22 code cells.
  Training results identical to 4 decimals in both runs (`SEED`): detectors m03/m09/m11 test balanced
  accuracy 0.878 / 0.917 / 0.910 (val 0.891 / 0.921 / 0.927 → m11 auto-selected), classifier test
  accuracy 0.855, balanced 0.870 — inside the §8.10 spread.
* **Local stand-in for the upload: 7 species folders × ≤12 ten-second 384 kHz excerpts** (80 files; heti
  is one recording, so 8) cut from `Data/audio` around annotated boxes, named with their true clock start,
  with shifted truth tables for Cell 24. In-sample: these are the training recordings. Final result:
  1118 selections kept + 152 rejected as noise; **492/509 annotated bat boxes found (0.967), species
  correct on 387/492 (0.787), 1/9 noise boxes hit**; 133 s for 13 min of audio. Weakest: rhle (48/76)
  and rhro (72/104) — confused with each other and with acsh — and alte → acsh (16), as in §8.9.
  The exported-`.pk` path gives exactly the same selections and species as the in-memory models.
* **Output file**: header exactly as requested, `Selection` 1..N, `Delta Time` = End − Begin, all
  selections ≤ 1 s, Low ≤ High.
* **Full 5-min recordings** (28 annotated recordings, exported `.pk`, detector m11 = best validation):

| Scan setting (after classifier noise filter, 10 recordings) | bat boxes found | species correct on found | noise boxes hit |
| :--- | ---: | ---: | ---: |
| thr 0.5, gap 0.1, ≥2 windows, max 1.0 s (**default**) | 0.981 | 0.797 | 3/36 |
| thr 0.5, gap 0.2 | 0.971 | 0.795 | 4/36 |
| thr 0.7, gap 0.2 | 0.955 | 0.789 | 2/36 |

  Detector only, all 28 recordings: threshold 0.5 → 0.986 of boxes found, 76/186 noise boxes hit
  *before* the classifier's noise filter; 0.9 → 0.879 and 6/186. Only ~8 % of selections overlap an
  annotated box because the tables mark a fraction of the calls in each recording, so **precision cannot be
  measured from them**. Runtime: 8.3 s per 5-min file (scan 8.1 s), i.e. ~12 min for 84 such files on the
  laptop GPU.

### 10.5 Known limits and how to reproduce

* **Nothing here was run on Kaggle.** Everything ran locally (RTX 4050, torch 2.14.1+cu130). The
  `DataParallel` training path is unchanged; the inference cells use a single GPU.
* **All accuracy numbers in §10.3–10.4 are in-sample** (the excerpts and full recordings are the ones the
  training clips were cut from). The real test is the uploaded set; if it has Raven tables, set
  `INFER_TRUTH_DIR` and read Cell 24.
* **Precision / false-alarm rate on real recordings is still unmeasured**: the training tables mark only
  some of the calls in each recording. Threshold 0.5 is the official BatSpot value; 0.7 trades ~2.5 pt of
  recall for fewer noise hits (§10.4 table).
* **rhle / rhro CF calls (~90–100 kHz) sit at the top of the detector band** (official `fmax` 95 kHz,
  kept because the pretrained encoder was trained on it). The detector still found 76/81 and 104/109 of
  their boxes here, but they remain the weakest classes.
* `Confidence` is a softmax score, not a calibrated probability.
* Reproduce: everything is in `/home/gb/batspot_gpu_experiments/inference_cells_2026-10-04/` (outside the
  repo; see its `README.txt`): `build_testset.py` (cuts `Data/audio` + `Data/selections`), `run_local.py`
  (exec's every code cell, swaps only the Kaggle paths), `analysis.py` (chunking / resampling
  equivalence, runtime, scan-setting sweep), `test_build_selections.py`, `test_clock.py`,
  `test_freq_impl.py`, the cell sources (`newcells/`) and the logs of every run quoted here.

### 10.6 More capacity (hidden layers / deeper encoder) does not help — measured

Question: are the official models too small? **No.** Two measurements (local RTX 4050, same split, same
`train_model` loop and config as the notebook; scripts `fit_gap.py`, `capacity_exp.py` in the folder above):

1. **Fit gap.** Clip-level accuracy (eval protocol) of the trained models on their own *training* clips vs
   unseen clips: classifier 0.911 train / 0.883 val / 0.855 test; detector m11 balanced 0.931 / 0.927 /
   0.909; m09 0.963 / 0.921 / 0.917. Not memorising — the classifier misses ~9 % of its own training clips —
   so capacity *could* have been the limit. Hence:
2. **Capacity experiment, classifier, 3 seeds each** (mean ± sd):

| Variant | Params | Train-clip acc | Test acc | Test bal. acc |
| :--- | ---: | ---: | ---: | ---: |
| official: pretrained ResNet-18 + linear head | 11.2 M | 0.910 | 0.851 ± 0.004 | 0.859 ± 0.009 |
| pretrained ResNet-18 + hidden layer 512→256 (ReLU, dropout 0.3) → 8 | 11.3 M | 0.913 | 0.839 ± 0.014 | 0.849 ± 0.012 |
| ResNet-34 from scratch (no official R34 weights) | 21.3 M | 0.917 | 0.855 ± 0.018 | 0.865 ± 0.018 |

   Training-clip accuracy stays at ~0.91 even with twice the parameters (ResNet-34's training loss is lower,
   0.16–0.25 vs 0.25–0.30, but its clip accuracy is not), so the remaining errors are not a capacity limit:
   they come from the input (one 20 ms window often cannot separate acsh/alte or rhle/rhro; 40 / 60 ms
   windows did not help either, §8.9) and from the labels / recordings (§8.2 item 6). Test differences
   are within the seed spread. Also: a hidden-layer head changes the `.pk` layout, so the exported model
   would **no longer load in the BatSpot GUI / `predict.py`**; ResNet-34 loads there but costs ~1.8× the
   training time and gives up the official pretrained weights. **Not adopted.** What would move accuracy:
   more recordings of rhle / rhro / acsh / alte, a label review of acsh↔alte, and a recording-grouped score.

---

## 11. Combined notebook `batspot-train-combined.ipynb` — Claude (Opus 5.5, 2026-10-05)

Scope: a NEW notebook combining `batspot-train.ipynb` (working copy of 2026-10-05, incl. the uncommitted
`RECORDING_SPLIT_SCOPE='per_species'` split that §10 does not describe) with the review fixes of
`batspot_train(claude_bug_fixes _by__SPACE_BUNNY_MODEL).ipynb` (§9), plus: classifier seed ensemble,
an "unknown" answer, input-robust inference, and a recording-held-out check. **Neither source notebook was
modified.** Build sources, tests, experiment harness and logs: `/home/gb/batspot_gpu_experiments/combined_2026-10-05/`
(`cells/` = exact source of every cell, `build_notebook.py` assembles them).

### 11.1 What was taken from where

| From | Taken | Not taken (why) |
| :--- | :--- | :--- |
| `batspot-train.ipynb` | everything (windowing, training loop, seeding, per-species split, cascade, export, inference 21-24) | — |
| Space Bunny (§9) | cache validated by source size+mtime, atomic writes, failed clips reported not fatal, NaN guard, per-instance mmaps, `num_mels` fallback, class names by output width, `AUC_NO_SIGNAL` flags + reduced-denominator notes, top-k validation mean, grouped retrain (rebuilt on the per-species splitter) | prior-corrected gate (a monotone rescaling: with a tuned threshold it changes nothing, §10.1); A/B + re-split ablation cell (answered offline, §11.3); `inspect.getsource` cache tag (broken under exec, §11.2) |

### 11.2 Bugs found and fixed while combining (each reproduced first)

| Where | Symptom | Root cause | Fix / proof |
| :--- | :--- | :--- | :--- |
| Cell 8 clip extraction (both notebooks) | extracts **nothing** from this repo's `Data/selections` | took `'_'.join(name.split('_')[:2])` = `acsh_devon` as the recording stamp, so no audio file matched | faithful port of `export_clips.R` (stamp `\d{8}_\d{6}`, class sub-folders, `>3 ms`, dataset naming); test: 20/20 rows of 2 real tables extracted, names parse back to the tape, samples bit-identical |
| Cell 6 cache tag (Space Bunny) | under `exec` (run_local / harness) the tag changed whenever an unrelated script changed -> silent full cache rebuilds | `inspect.getsource` on exec'd code falls back to linecache of `__main__` and returns lines of the wrong file | tag = hash of the compiled code **and default arguments** of `load_audio_file` + `clip_to_db_spectrogram`; test: stable across re-exec, changes when `preemphasis` changes (a first version missed default args -- caught by the test) |
| Cell 22 chunked resampling | for 250 / 500 kHz recordings each 60 s chunk was resampled on a grid shifted by a fraction of a sample vs the whole file: max difference 13-16 % of full scale on white noise (a 2-5 us shift; small effect on dB spectrograms, but not "identical to training" as documented); 384 kHz was unaffected | excerpt start not aligned to the L/M phase grid | start on output index q0 = multiple of L; test: excerpt == whole file exactly (0.0) for 384/250/256/500 kHz |
| Cell 23 input listing | one corrupt file crashed the whole cell before the per-file `try` | `sf.info` called on every file in the folder summary | listing guarded; corrupt / empty / 5 ms files reported and skipped |
| Cell 7 | unreadable clip killed a run, or (with Space Bunny's per-dataset drop) desynchronised detector vs classifier test lists | no global pre-check | unusable files dropped once, before the split |

### 11.3 Experiments that set the new defaults (local RTX 4050, notebook's own code via `exp.py`)

All runs exec Cells 2-10 of the combined notebook and use its dataset, training loop and splitter. Splits:
`insplit` = the default 70/15/15 clip-level split (seed 42); `perspecies` = Cell 7's `split_by_recording_per_species`
(whole recordings held out per species; heti still clip-split, rhbe's smaller recording split val/test). Seeds vary
initialisation and sampling only; the test files are identical across seeds. Raw results (incl. test
probabilities and embeddings): `results/*.npz`; summaries: `variants_analysis.txt`, `openset_*analysis.txt`.

**Background noise mixing** (`noise_mix_prob = 0.5`, SNR 0-20 dB: a random window of a training `noise` clip,
i.e. another recording's background, added in the power domain before min-max):

| Model, split | baseline bal. acc. | noise mixing | paired delta per seed |
| :--- | :---: | :---: | :--- |
| classifier, held-out recordings | 0.552 +- 0.017 | **0.623 +- 0.034** | +0.043, +0.045, +0.126 |
| classifier, clip-level split | 0.859 +- 0.009 | 0.846 +- 0.012 | -0.010, -0.017, -0.012 |
| detector m09, held-out recordings | 0.464 +- 0.033 (AUC 0.48) | **0.556 +- 0.036** (AUC 0.55) | +0.102, +0.091, +0.082 |
| detector m09, clip-level split | 0.908 +- 0.014 (AUC 0.958) | 0.910 +- 0.009 (AUC 0.951) | -0.017, +0.026, -0.004 |

(3 seeds per cell.) Adopted for both models: +0.07 / +0.09 balanced accuracy on recordings the model has not heard,
for -0.013 (classifier) / nothing (detector) on familiar ones. The classifier gain is mostly sasa (held-out recall
0.00 -> 0.60).
**Still unsolved:** the held-out test noise is ONE noise recording (noise has 8 recordings in total); on it the
detector stays near chance (AUC 0.55, noise recall 0.11 -> 0.32) and the classifier's noise recall is ~0.01 with
or without mixing. False alarms on a new kind of noise are the weakest point of the whole pipeline; more noise
recordings, not code, would fix it. Not tested further: other SNR ranges / probabilities.

**Seed ensemble (classifier, 3 seeds, softmax averaged):** clip-level split, balanced accuracy vs the members'
mean: 0.859 vs 0.859 (no mixing), 0.857 vs 0.846 (noise mixing); held-out recordings 0.546 vs 0.552 / 0.618 vs
0.623 -- **0 to +1 point in-split, nothing on new recordings**. Kept for stability (it removes most of the 3-5 point seed lottery of a single run) and because
the members' spread is printed, not for a generalisation gain.

**Old `USE_AUGMENTATION` set (time shift, Gaussian noise, SpecAugment masks), classifier, 3 seeds:** clip-level
split 0.847 vs 0.859 (-0.012), held-out recordings 0.557 vs 0.552 (+0.005). No benefit -> stays off.

### 11.4 The "unknown" answer (open set) -- what it can and cannot do

Leave-one-species-out: for each of the 7 species, 3 classifiers (seeds 42/43/44) were trained without it on the
clip-level split; the held-out species' 68-181 clips are then "never seen". The threshold is set exactly as Cell
13 does (validation clips of the known species, keep `UNKNOWN_KEEP_KNOWN` of the correct answers). Scores compared
(single model, 7 species): max softmax AUROC 0.70 (rhro 0.46: the model is MORE confident on a species it never
saw), kNN cosine 0.78-0.80, **Mahalanobis on the 512-d embedding 0.84** (every species >= 0.75). Shipped:
per-member Mahalanobis averaged over the 3-seed ensemble (AUROC 0.840):

| `UNKNOWN_KEEP_KNOWN` | never-seen species -> 'unknown' | -> 'noise' | -> a wrong name | known species: correct answers lost |
| :---: | :---: | :---: | :---: | :---: |
| **0.95 (default)** | 30 % | 18 % | 52 % | 3.6 % (0.864 -> 0.828) |
| 0.90 | 40 % | 18 % | 43 % | 5.5 % |
| 0.80 | 52 % | 18 % | 30 % | 8.6 % |

Per species at 0.95: sasa 51 % caught, rhle 36 %, alte 29 %, acsh 28 %, rhro 26 %, rhbe 23 %, heti 19 % (heti and
acsh also go to noise often). Species that resemble a training species (acsh/alte, rhle/rhro) are the hardest.
**Read it as a triage flag, not a guarantee**: at the default it labels ~1 in 3 calls of a new species
'unknown' while costing ~1 in 28 correct answers. The Raven tables keep the classifier's best guess next to it.

### 11.5 Inference: "anything thrown at it"

| Input | Handling | Verified (run B2 / B3, `stress/`) |
| :--- | :--- | :--- |
| any sample rate | rational ratio L/M -> resampy's `kaiser_best` taps as L polyphase FIRs, one strided `conv1d` (GPU or CPU), self-checked against resampy per ratio; > 512 phases or a failed check -> resampy | 12 rate pairs x CPU/GPU (`test_resample.py`): <= 6.4e-7 of full scale, except 250->192 and 500->192 kHz at 4-8e-5 -- exactly the ~0.02 % of outputs where resampy's own float64 time grid rounds below an integer and drops a tail tap (it is not chunk-consistent there itself). Chunked scanning == whole file (0.0) for 384/250/256/500 kHz. 250/256/500 kHz files: same selections as the 384 kHz original |
| time-expanded files | `INFER_TIME_EXPANSION` (true rate = header x factor) | 10x file at a 38.4 kHz header: identical rows and clock times to the original |
| WAV / FLAC / AIFF / OGG, int / float, stereo | `INFER_EXTENSIONS`; channels averaged as in training | FLAC and float32 copies: identical rows to the WAV |
| folder, single file, top-level `.zip`; corrupt / empty / 5 ms files | listing guarded; per-file `try`; corrupt files reported and skipped | corrupt -> FAILED, run continues; empty and 5 ms -> no selections |
| no timestamp in the name | AudioMoth header, else clock from 00:00:00 (reported) | reported per file |
| ensemble / exported models | `INFER_CLASSIFIER_PK` = list of member `.pk` + `INFER_UNKNOWN_FILE` (refuses a sidecar fitted for other members) | run B1: skip-training path from the exported files == in-memory run A, all 1110 + 183 rows identical |
| one input folder per night / site | `INFER_PER_FOLDER_FILES` (default on): besides the combined file, `batspot_detections_per_folder/batspot_detections_<folder>.txt`, same columns, Selection 1..n per file | 80-excerpt set: 7 files, counts sum to the combined 1110; byte-identical to `split_per_folder.py` (splits an existing combined file the same way) |

**"unknown" also reacts to the recording chain, not only to new species.** The 256 kHz copy (two anti-alias
filters, 384->256->250 kHz, where training used one) kept the same best guesses and confidences (rhro 0.94-0.97)
but 6 of its 8 selections became 'unknown'; the 250 and 500 kHz copies did not. Read 'unknown' as "unlike the
training data -- check it"; the Raven tables keep the best guess.

### 11.6 Verification of the notebook itself

* **Run A** (`run_local.py`, local RTX 4050, all 23 code cells, 1254 s incl. thermal pauses, 0 errors): detectors
  m03 / m09 / m11 test balanced 0.891 / 0.899 / 0.891 (m11 = best validation -> inference); classifier members
  0.837 +- 0.011 accuracy, **ensemble 0.848 accuracy / 0.857 balanced**; "unknown" on 5/145 known test clips
  (4 would have been correct): 0.848 -> 0.821 counting them as errors; official zero-shot -> fine-tuned balanced
  accuracy +0.092 / +0.171 / +0.132. Seeded members reproduce the `exp.py` runs to 4 decimals. Held-out check
  (Cell 26): classifier 0.860 -> 0.596 balanced, detector m11 0.891 -> 0.642.
* **Inference on the 80 test excerpts** (in-sample: cut from training recordings): 498/509 bat boxes found (0.978;
  0.967 in section 10), species correct 382/498 = 0.767 counting 'unknown' as wrong, 0.799 on the 478 named boxes
  (0.787 in section 10, single classifier, no noise mixing); 20 'unknown'; 3/9 noise boxes hit (1/9 before).
* Unit tests (`test_units.py`, 17 checks) and resampler tests (`test_resample.py`) all pass; each new piece is
  exercised through the notebook's own cells.

### 11.7 Known limits

* **New kinds of noise are the weak point.** On the one held-out noise recording, detector AUC 0.55 and classifier
  noise recall ~0.01 even with noise mixing -> expect false alarms on unfamiliar backgrounds. Needs more noise
  recordings (there are 8).
* In-split numbers stay an upper bound (28 recordings); the held-out numbers come from ONE per-species split and are
  noisy (a few recordings per class).
* Noise mixing costs ~1 point of classifier accuracy on familiar recordings; `CLS_CONFIG['noise_mix_prob'] = 0.0`
  gives the in-split optimum. The ensemble triples classifier training time (~3 x 3-6 min) and gives no measured gain on new recordings.
* Nothing here ran on Kaggle; the `DataParallel` path is unchanged from the base notebook (ensemble members are
  unwrapped after training).
* The GUI / CLI load one model: `classifier_250khz.pk` is the best single member, without the 'unknown' answer.
