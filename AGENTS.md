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
   curves survive a run.
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
   as a default.

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
- Data leakage between detector training splits and classifier evaluation sets in the cascaded pipeline is resolved by using the shared species-stratified 70/15/15 split across all datasets.
