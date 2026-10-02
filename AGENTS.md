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
