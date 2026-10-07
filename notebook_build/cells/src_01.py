# Cell 2: Configuration
import os

# --- Paths (Kaggle defaults, override as needed) ---
DATA_DIR = '/kaggle/input/datasets/budhil/100eachbatds/Data/Data/final_dataset/data'  # Pre-extracted clips (class subfolders)
RAW_AUDIO_DIR = '/kaggle/input/datasets/budhil/100eachbatds/Data/Data/audio'                                         # Raw .WAV recordings (for clip extraction)
SELECTIONS_DIR = '/kaggle/input/datasets/budhil/100eachbatds/Data/Data/selections'                                        # Raven selection tables (for clip extraction)

# Clip extraction re-reads ALL raw audio and writes into DATA_DIR. You already
# have the extracted clips in final_dataset/data, so leave this False unless
# you specifically want to rebuild them from the raw recordings.
RUN_CLIP_EXTRACTION = False
OVERWRITE_EXISTING_CLIPS = False   # guard: refuse to write into a non-empty DATA_DIR
MODEL_OUTPUT_DIR = None                                      # Your own previously-trained models (validation)

# --- Auto-detect MODEL_OUTPUT_DIR relative to DATA_DIR ---
if MODEL_OUTPUT_DIR is None and DATA_DIR is not None:
    _data_parent = os.path.dirname(os.path.abspath(DATA_DIR))
    for _candidate in [
        os.path.join(_data_parent, 'model_output'),
        os.path.join(_data_parent, '..', 'model_output'),
        os.path.join(_data_parent, '..', '..', 'model_output'),
        'Data/model_output',
    ]:
        _candidate = os.path.normpath(_candidate)
        if os.path.isdir(_candidate):
            MODEL_OUTPUT_DIR = _candidate
            break

# ---------------------------------------------------------------------------
# OFFICIAL PRE-TRAINED MODELS -- set paths manually here.
#
# Fill in the 3 call detectors + 1 call classifier you uploaded as a Kaggle
# dataset. Leave any of them as None to skip it (Cell 10 falls back to
# auto-discovery under /kaggle/input/*, then to Data/model_output/).
#
# Click "Add Data" in Kaggle, then use "Copy Path" on the dataset folder to
# get the exact /kaggle/input/... prefix.
#
#   Detector   192 kHz, 2 classes (noise/target)  -> fine-tunes the detector
#   Classifier 250 kHz, 15 classes                -> fine-tunes the classifier
# ---------------------------------------------------------------------------
DETECTOR_M03_PATH = '/kaggle/input/datasets/budhil/100eachbatds/Batspot_Models/Batspot_Models/models_call_detector/m03/ANIMAL-SPOT.pk'   # e.g. '/kaggle/input/mymodels/batspot/models_call_detector/m03/train/ANIMAL-SPOT.pk'
DETECTOR_M09_PATH = '/kaggle/input/datasets/budhil/100eachbatds/Batspot_Models/Batspot_Models/models_call_detector/m09/ANIMAL-SPOT.pk'    # e.g. '/kaggle/input/mymodels/batspot/models_call_detector/m09/train/ANIMAL-SPOT.pk'
DETECTOR_M11_PATH = '/kaggle/input/datasets/budhil/100eachbatds/Batspot_Models/Batspot_Models/models_call_detector/m11/ANIMAL-SPOT.pk'    # e.g. '/kaggle/input/mymodels/batspot/models_call_detector/m11/train/ANIMAL-SPOT.pk'
CLASSIFIER_M09_PATH = '/kaggle/input/datasets/budhil/100eachbatds/Batspot_Models/Batspot_Models/models_call_classifier/ANIMAL-SPOT.pk'  # e.g. '/kaggle/input/mymodels/batspot/models_call_classifier/m09/train/ANIMAL-SPOT.pk'

# Manual paths take priority over auto-discovery. Set to False to ignore them.
USE_MANUAL_MODEL_PATHS = True

WORKING_DIR = '/kaggle/working'
PRETRAINED_DIR = os.path.join(WORKING_DIR, 'pretrained')
os.makedirs(PRETRAINED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Spectrogram params below are the REAL values read out of the official
# BatSpot .pk files (BatSpot_article/batspot/models_*/**/ANIMAL-SPOT.pk).
# They MUST match the model you fine-tune from, otherwise the transferred
# encoder weights were trained on a different time-frequency resolution and
# are effectively noise. Cell 10 re-verifies this against each .pk at runtime.
# ---------------------------------------------------------------------------

# --- Detector config (192 kHz, binary: noise vs target) ---
DET_CONFIG = {
    'sr': 192000,
    'num_classes': 2,
    'classes': ['noise', 'target'],
    'fmin': 1000,               # from official detector dataOpts
    'fmax': 95000,
    'n_fft': 256,
    'hop_length': 128,
    'n_freq_bins': 256,
    'sequence_len': 20,         # ms
    'max_pool': 2,
    'freq_compression': 'linear',
    # Batch 32 (official: 16). 674 clips / 32 = ~21 optimizer steps per epoch. Batch 128 gave
    # only 6 steps/epoch, which is why the old config needed absurd learning rates.
    'batch_size': 32,
    'base_lr': 1e-4,            # absolute LR (official value) — NOT scaled by batch size
    'n_epochs': 100,            # cap only; early_stopping_patience_epochs below usually stops sooner.
                                # (m09 hit the old cap of 60 at its best epoch 58; the expected gain of
                                # the higher cap is 1-3 val clips, i.e. at the noise level -- AGENTS.md 9.4)
    'epochs_per_eval': 1,       # validation is cheap (clip-level, top-k windows)
    'grad_accum_steps': 1,
    'use_multi_gpu': True,
    # reference hyperparams from models_call_detector/*/config_training_gui
    'beta1': 0.5,
    'lr_decay_factor': 0.5,
    'lr_patience_epochs': 12,              # RAW epochs without improvement before halving LR
    'early_stopping_patience_epochs': 30,  # RAW epochs without improvement before stopping
    # Background noise mixing (training windows only): probability that a training window gets a random
    # window of a 'noise' training clip (another recording) added in the power domain. Measured, 3 seeds
    # (AGENTS.md 11): on recordings held out of training, balanced accuracy 0.464 -> 0.556 (+0.09, all 3
    # seeds), noise recall 0.11 -> 0.32; on the usual clip-level split 0.908 -> 0.910 (3 seeds: no cost).
    'noise_mix_prob': 0.5,
    'noise_mix_snr_db': (0.0, 20.0),       # window-to-added-noise power ratio, drawn uniformly (dB)
    'label_smoothing': 0.0,
}

# --- Classifier config (250 kHz, multi-species) ---
CLS_CONFIG = {
    'sr': 250000,
    'num_classes': None,        # auto-detected from your data in Cell 7
    'fmin': 10000,              # from official classifier dataOpts
    'fmax': 125000,
    'n_fft': 256,
    'hop_length': 128,
    'n_freq_bins': 256,
    'sequence_len': 20,         # ms
    'max_pool': 2,
    'freq_compression': 'linear',
    'batch_size': 32,           # ~21 steps/epoch (batch 128 = 6 steps/epoch was the real problem)
    'base_lr': 3e-4,            # measured best of {1e-4, 3e-4, 1e-3} (3 seeds each); 1e-2 is NOT needed
                                # and destabilises training (val acc collapsed to 0.42 at epoch 38)
    'n_epochs': 120,            # val score was still rising at epoch 60 with lr 3e-4
    'epochs_per_eval': 1,
    'grad_accum_steps': 1,
    'use_multi_gpu': True,
    # reference hyperparams from models_call_classifier/m09/config_train
    'beta1': 0.5,
    'lr_decay_factor': 0.5,
    'lr_patience_epochs': 15,              # RAW epochs without improvement before halving LR
    'early_stopping_patience_epochs': 40,  # RAW epochs without improvement before stopping
    # Background noise mixing, as in DET_CONFIG. Measured, 3 seeds: recordings held out of training,
    # balanced accuracy 0.552 -> 0.623 (+0.07, all 3 seeds; mostly sasa 0.00 -> 0.60); usual clip-level
    # split 0.859 -> 0.846 (-0.013, 3 seeds). 0.0 = the in-split optimum for the classifier.
    'noise_mix_prob': 0.5,
    'noise_mix_snr_db': (0.0, 20.0),
    'label_smoothing': 0.0,
}
# NOTE on the values above: the LR / batch / epoch values come from a sweep (gpuexp.py) that selected
# checkpoints on plain accuracy and had no LR schedule or early stopping -- not this exact loop. The
# notebook's own runs reproduce the gains, but treat them as good defaults, not proven optima.

# --- Nyquist guard ------------------------------------------------------------------------
# animal_spot/predict.py:214 reads dataOpts["fmax"] and data/audiodataset.py:691 hands it to
# Interpolate unclamped. Nothing complains: transforms.py:524 clamps max_bin with min(n_fft - 1, ...),
# so an out-of-band fmax is silently cropped/misaligned and the model is trained and scored on the
# wrong frequencies with no message anywhere. The margin is thin -- 192 kHz has its Nyquist at 96000
# and the detector ships fmax=95000, 1 kHz below it -- so this is checked, not trusted.
def check_bands(configs):
    for name, c in configs.items():
        if c['fmax'] > c['sr'] / 2:
            raise AssertionError(
                f"{name}: fmax={c['fmax']} exceeds Nyquist for sr={c['sr']}. "
                "Raise sr, not fmax -- predict.py uses fmax unclamped, so an out-of-band "
                "value returns wrong frequencies with no error.")
check_bands({'DET': DET_CONFIG, 'CLS': CLS_CONFIG})

# --- Input windowing (THE main accuracy fix; see AGENTS.md "Windowing fix") ---------------
# Clips are 15 ms - 2.6 s (median ~375 ms) but every model consumes a 20 ms window.
#   'energy_crop' : TRAIN  = a fresh random crop each epoch from the loudest 20 % of the clip's windows
#                   TEST/VAL = the TEST_TOPK loudest windows (overlap < half a window), probabilities averaged
#   'first'       : old behaviour (only the first 20 ms of every clip) -- kept for ablation
WINDOW_MODE = 'energy_crop'
WINDOW_STRIDE = 3          # frames between candidate windows (3 frames ~ 2 ms)
TRAIN_TOP_FRAC = 0.20      # train crops come from this loudest fraction of windows
TEST_TOPK = 5              # windows averaged per clip at validation / test time

# Model selection, LR schedule and early stopping all track this validation score.
# 'balanced_accuracy' (mean per-class recall) cannot be gamed by predicting the majority class;
# the detector's 80/20 call/noise split made plain accuracy hide the all-call collapse of m03.
SELECT_METRIC = 'balanced_accuracy'   # or 'accuracy'
# Validated here AND read through a dict lookup in Cell 9: an unrecognised value used to fall back to
# plain accuracy silently, re-creating the all-call collapse while the log printed the typo'd name.
assert SELECT_METRIC in ('balanced_accuracy', 'accuracy'), \
    f"SELECT_METRIC must be 'balanced_accuracy' or 'accuracy', got {SELECT_METRIC!r}"
# Printed next to the best validation score: the mean of the top-k validation scores. The best of
# ~100 validations of a discrete metric on 145 clips sits a clip or two above the run's typical level;
# showing both makes that selection optimism visible.
REPORT_TOPK_MEAN = 3

# The filename carries the recording (tape) id and there are only ~28 recordings, so a random
# clip-level split leaks recordings between train and test (Cell 7 prints a leakage check).
#   False = clip-level split, stratified by species: every species gives ~15 % to val and to test, but
#           clips of one recording land in train AND test (optimistic about new recordings).
#   True  = hold out whole recordings; RECORDING_SPLIT_SCOPE below says how.
SPLIT_BY_RECORDING = False
# Only used when SPLIT_BY_RECORDING = True.
#   'per_species' : each species is split ON ITS OWN. Its recordings go (whole) to train / val / test so that
#                   each part holds ~70/15/15 % of THAT species' clips -> every class appears in val and test.
#                   Species with 2 recordings: the bigger one trains, the smaller one is split val/test by clip.
#                   Species with 1 recording (heti): clip-level split (the only option; reported in Cell 7).
#                   CAVEAT: a recording holds several species and each is split on its own, so a test clip whose
#                   recording is in train always meets train clips of ANOTHER species there. A model that leans on
#                   the recording is scored BELOW chance (Cell 7 prints a tape-lookup baseline). Pessimistic bound.
#   'global'      : one StratifiedGroupKFold over all clips (7 folds). Recordings are indivisible, so val/test get
#                   whatever species sit in their ~4 recordings; classes can be missing, sizes are not 70/15/15.
RECORDING_SPLIT_SCOPE = 'per_species'
assert RECORDING_SPLIT_SCOPE in ('per_species', 'global'), \
    f"RECORDING_SPLIT_SCOPE must be 'per_species' or 'global', got {RECORDING_SPLIT_SCOPE!r}"

# --- Reproducibility ---
# Seeds python / numpy / torch before each model is built and trained (Cells 12-13), so a re-run
# repeats the same initialisation and sampling order. Unseeded runs of the same config spread by
# ~3-5 points (AGENTS.md 8.10); seeding makes a run repeatable but does NOT shrink that spread
# (cuDNN kernels stay non-deterministic). None = unseeded.
SEED = 42

# --- Classifier seed ensemble ---
# The classifier is trained once per seed on the SAME split. Cell 13 prints every member, their
# mean +- sd (the run-to-run noise any comparison must beat) and the ENSEMBLE = softmax averaged over
# the members, which is what the cascade (Cell 14) and the inference cells (21-24) use. Each extra
# seed costs one more classifier training (~3 min on an RTX 4050, ~6 min on Kaggle 2xT4).
# [SEED] = a single model, as before.
CLS_ENSEMBLE_SEEDS = [42, 43, 44]

# --- "Unknown" answer (open set) ---
# The classifier only knows the species it was trained on, so a call of any OTHER species still gets
# one of those names. With UNKNOWN_DETECTION the inference cells answer UNKNOWN_LABEL instead when the
# selection looks unlike every training species. The threshold is set on the VALIDATION clips so that
# UNKNOWN_KEEP_KNOWN of the correctly classified known-species clips keep their name (Cell 13), and the
# trade-off was measured by hiding each species from training in turn (AGENTS.md section 11).
# Measured trade-off (3-seed ensemble, each of the 7 species hidden from training in turn, AGENTS.md 11):
#   UNKNOWN_KEEP_KNOWN    never-seen species -> 'unknown' / 'noise' / a wrong name    known answers lost
#        0.95                       30 %       /   18 %  /   52 %                       3.6 %
#        0.90                       40 %       /   18 %  /   43 %                       5.5 %
#        0.80                       52 %       /   18 %  /   30 %                       8.6 %
# A species that resembles a training species (acsh/alte, rhle/rhro) is the hardest to flag. Lower the
# value if your recordings may hold many species the model was not trained on.
UNKNOWN_DETECTION = True
UNKNOWN_KEEP_KNOWN = 0.95      # share of correct known-species answers that must survive the threshold
UNKNOWN_LABEL = 'unknown'
UNKNOWN_METHOD = 'maha'       # 'maha' (best measured) | 'knn5' | 'msp' -- see Cell 6

# --- Augmentation toggle ---
# True  = time shift + gaussian noise + freq/time masking applied to train windows
# False = no extra augmentation. Measured (classifier, 3 seeds, AGENTS.md 11): clip-level split -0.012,
#         held-out recordings +0.005 balanced accuracy -- no benefit, so it stays off.
# Background noise mixing is set per model in DET_CONFIG / CLS_CONFIG ('noise_mix_prob').
USE_AUGMENTATION = False

# --- Recording-held-out check (Cell 26) ---
# The default split shares recordings between train and test (Cell 7 prints the leakage), so its
# scores are an upper bound for NEW recordings. Cell 26 retrains one classifier and one detector with
# whole recordings held out per species (the 'per_species' scheme above) and prints both side by side.
# Costs one classifier + one detector training. Skipped automatically if SPLIT_BY_RECORDING is True.
RUN_GROUPED_CHECK = True
GROUPED_DETECTOR = 'auto'      # 'auto' = the detector variant with the best validation score, or 'm03'/'m09'/'m11'

# --- Phase gates --------------------------------------------------------------------------
# Phase 1 (train -> evaluate -> export) always runs. Everything past it is gated, so a stranger can
# train and use the models without paying for the extras or having the data they need. Kaggle allows
# a 12 h session; 'Run All' with the defaults below costs ~50 min on 2x T4.
RUN_MEASURE = True        # phase 2: per-model noise floor (2 x sd over seeds) + multi-seed mean +- sd
                          # (+~35 min). This is the error bar every later delta is judged against.
RUN_EXPERIMENTS = False   # phase 3: exploratory A/Bs, incl. the 250 kHz detector for rhle/rhro
                          # (+~40 min). Off because its answers are not needed to USE the models.
RUN_PHASE4 = False        # phase 4: threshold calibration on real recordings. Gated separately
                          # because it needs INFER_INPUT_DIR uploaded, and a hard assert there would
                          # break 'Run All' for someone who only wants to train.

# A model whose test AUC lies within this of 0.5 carries no usable signal; summary tables say so,
# so a high plain accuracy on the 81/19 call/noise split cannot be misread as skill.
AUC_NO_SIGNAL = 0.05

# --- GPU check ---
# If DEVICE prints 'cpu', the Kaggle GPU accelerator is OFF.
# To enable: Session options (right sidebar) -> Accelerator -> GPU T4 x2 -> Save.
# The notebook must then be restarted. No code change is needed.
DEVICE = __import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')

print(f'Detector:   {DET_CONFIG["sr"]//1000}kHz, {DET_CONFIG["num_classes"]} classes '
      f'(noise/target), fmin={DET_CONFIG["fmin"]}, fmax={DET_CONFIG["fmax"]}')
print(f'Classifier: {CLS_CONFIG["sr"]//1000}kHz, {CLS_CONFIG["num_classes"]} classes (auto), '
      f'fmin={CLS_CONFIG["fmin"]}, fmax={CLS_CONFIG["fmax"]}')
print(f'Device: {DEVICE}')
if DEVICE.type != 'cuda':
    print('!! WARNING: no GPU. Training will be ~75x slower. On Kaggle: Session options -> '
          'Accelerator -> GPU T4 x2, then restart the session.')
print(f'Selection metric: {SELECT_METRIC}   classifier ensemble seeds: {CLS_ENSEMBLE_SEEDS}   '
      f'unknown answer: {UNKNOWN_DETECTION} (keep {UNKNOWN_KEEP_KNOWN:.0%} of correct known answers)')
print(f'Augmentation: extra={USE_AUGMENTATION}  noise mixing det={DET_CONFIG["noise_mix_prob"]} '
      f'cls={CLS_CONFIG["noise_mix_prob"]}   recording-held-out check: {RUN_GROUPED_CHECK}')

# Report which manual paths resolved (a typo here is otherwise silent)
if USE_MANUAL_MODEL_PATHS:
    print('\nManual pre-trained model paths:')
    for _lbl, _p in [('detector m03', DETECTOR_M03_PATH),
                     ('detector m09', DETECTOR_M09_PATH),
                     ('detector m11', DETECTOR_M11_PATH),
                     ('classifier m09', CLASSIFIER_M09_PATH)]:
        if _p is None:
            print(f'  {_lbl:15s} (not set)')
        elif os.path.isfile(_p):
            print(f'  {_lbl:15s} OK  {_p}')
        else:
            print(f'  {_lbl:15s} !! NOT FOUND: {_p}')
else:
    print('\nManual model paths DISABLED (USE_MANUAL_MODEL_PATHS = False)')
