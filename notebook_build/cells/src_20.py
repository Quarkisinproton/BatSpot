# Cell 21: INFERENCE on new recordings -- configuration + pick the best detector/classifier combo
#
# Upload the test recordings as a Kaggle dataset (Add Data -> Upload -> the folder, or a .zip of
# it) and paste its path ("Copy path") into INFER_INPUT_DIR. Expected layout (any depth works):
#     <INFER_INPUT_DIR>/<folder 1>/*.wav  ...  <folder 7>/*.wav      e.g. 7 folders x 12 recordings
# Every .wav / .WAV below INFER_INPUT_DIR is processed; each may hold zero, one or many calls.
#
# What happens to each recording (the BatSpot workflow: the detector scans, the classifier labels):
#   1. DETECTOR   20 ms windows every INFER_HOP_S over the whole file -- official BatSpot scan
#                 settings (10 ms hop, threshold 0.5) -- with the training front end and the same
#                 per-window min-max normalisation.
#   2. SELECTIONS windows with P(call) >= INFER_DET_THRESHOLD are merged into one selection when
#                 they are at most INFER_MERGE_GAP_S apart (the pulses of one pass), mirroring the
#                 ~0.4 s bouts in the training selection tables.
#   3. CLASSIFIER each selection is scored exactly like a test clip: the TEST_TOPK loudest 20 ms
#                 windows, softmax averaged. Species detected = argmax, Confidence = that averaged
#                 probability. Selections the classifier labels 'noise' go to *_rejected_noise.txt.
#   4. MEASURE    Low / High / Peak frequency from a ~375 Hz-resolution spectrogram (Raven's
#                 resolution for these 384 kHz files); clock times from the recording start time
#                 in the file name (YYYYMMDD_HHMMSS) or the AudioMoth header.
#
#   5. UNKNOWN    a selection whose classifier output looks unlike every training species is labelled
#                 UNKNOWN_LABEL ('unknown') instead of a species (Cell 2: UNKNOWN_DETECTION).
#
# Models: by default the ones trained above -- the detector variant with the best VALIDATION score
# (never the test score) plus the classifier ensemble. To skip training: run Cells 2-6, then Cells
# 21-24 with INFER_DETECTOR_PK / INFER_CLASSIFIER_PK pointing at .pk files exported by Cell 15 (a LIST
# of .pk files = an ensemble) and INFER_UNKNOWN_FILE at classifier_250khz_unknown.npz.
#
# Input robustness: any sample rate (resampled exactly like training; a fast exact GPU path covers
# 384/256/250/500 kHz and other rational ratios), mono or multi-channel (averaged), WAV/FLAC/AIFF/OGG,
# any length (scanned in chunks), corrupt files are reported and skipped. Time-expanded recordings
# (e.g. 10x, stored at 44.1 kHz) need INFER_TIME_EXPANSION.

INFER_INPUT_DIR = '/kaggle/input/<your-test-dataset>'    # folder (or .zip, or a single file) with the recordings
INFER_OUTPUT_FILE = os.path.join(WORKING_DIR, 'batspot_detections.txt')
INFER_DETECTOR = 'auto'        # 'auto' = best validation score of the detectors trained above, 'm03'/'m09'/'m11',
                               # or 'ensemble' = average of all trained detector variants (slower scan)
INFER_DETECTOR_PK = None       # optional: exported detector .pk (or a list = ensemble), e.g. '/kaggle/input/<run>/detector_192khz_m09.pk'
INFER_CLASSIFIER_PK = None     # optional: exported classifier .pk or LIST of .pk (ensemble), e.g.
                               # ['/kaggle/input/<run>/classifier_250khz_seed42.pk', '.../seed43.pk', '.../seed44.pk']
INFER_UNKNOWN_FILE = None      # with INFER_CLASSIFIER_PK: '/kaggle/input/<run>/classifier_250khz_unknown.npz'
                               # (written by Cell 15 for exactly those classifier files); None = no 'unknown' answer

INFER_DET_THRESHOLD = 0.5      # window P(call) needed to open / extend a selection (official BatSpot value)
INFER_HOP_S = 0.010            # detector window hop in seconds (official BatSpot value)
INFER_MERGE_GAP_S = 0.10       # positive windows closer than this join one selection
INFER_MIN_WINDOWS = 2          # selections supported by fewer positive windows are dropped (isolated clicks)
INFER_MAX_SELECTION_S = 1.0    # longer selections (continuous activity) are split at their widest silence
INFER_DROP_NOISE = True        # True: classifier-'noise' selections go to *_rejected_noise.txt, not the main file
INFER_FREQ_FLOOR_HZ = 10000    # ignore energy below this when measuring Low/High/Peak (AudioMoth high-pass = 10 kHz)
INFER_FREQ_CEIL_HZ = 150000    # ...and above this (99.7 % of the annotated boxes end below 150 kHz)
INFER_FREQ_SNR_DB = 12         # Low/High = contiguous band around the peak at least this far above the background
                               # (Low is unreliable for faint rhle/rhro CF calls -- see Cell 22)
INFER_CHUNK_S = 60             # files are scanned in chunks of this many seconds (bounded memory for long files)
INFER_AMP = True               # fp16 for the detector scan on GPU (as in training): ~1.7x faster, P(call) moves
                               # by <= 0.003 (4 of 30 000 windows flipped at 0.5); the classifier stays fp32
INFER_TIME_EXPANSION = 1       # 10 for 10x time-expanded recordings (true rate = header rate x this factor)
INFER_EXTENSIONS = ('.wav', '.flac', '.aif', '.aiff', '.ogg', '.w64', '.rf64')   # audio files picked up (any case)
INFER_NAME_WITH_FOLDER = True  # name_of_file = '<folder>/<file>.wav' (AudioMoth file names repeat across sites)
INFER_WRITE_RAVEN_TABLES = True  # also write one tab-separated Raven selection table per recording
INFER_PER_FOLDER_FILES = True  # also write one detections file per input folder (e.g. one per night):
                               # batspot_detections_per_folder/batspot_detections_<folder>.txt, same columns,
                               # Selection numbered 1..n within each file
INFER_TRUTH_DIR = None         # optional: folder with YOUR Raven selection tables for these files -> Cell 24 scores the output


def _model_from_pk(path):
    """Load an exported (or official) .pk for inference, or a LIST of .pk files as a softmax ensemble
    (they must share classes and data options). Returns (model, names, cfg)."""
    paths = [path] if isinstance(path, str) else list(path)
    models, names, cfg = [], None, None
    for p in paths:
        model, classes, d = load_model_from_pk(p, DEVICE)
        inv = {v: k for k, v in classes.items()}
        n = [inv.get(i, f'class{i}') for i in range(model[1].linear.out_features)]
        c = {'sr': d['sr'], 'n_fft': d.get('n_fft', 256), 'hop_length': d.get('hop_length', 128),
             'n_freq_bins': d.get('n_freq_bins') or d.get('num_mels') or 256,
             'fmin': d.get('fmin'), 'fmax': d.get('fmax'), 'sequence_len': 20}
        if names is not None and (n != names or c != cfg):
            raise ValueError(f'{p} does not match {paths[0]} (classes or data options differ)')
        models.append(model); names, cfg = n, c
    return (models[0] if len(models) == 1 else SoftmaxEnsemble(models)), names, cfg


def _label_of(path):
    paths = [path] if isinstance(path, str) else list(path)
    return ', '.join(os.path.basename(p) for p in paths) + (f' (ensemble of {len(paths)})' if len(paths) > 1 else '')


# ---- detector -------------------------------------------------------------------------------
if INFER_DETECTOR_PK:
    _m, _names, _cfg = _model_from_pk(INFER_DETECTOR_PK)
    _call = [i for i, n in enumerate(_names) if n in ('call', 'target')]
    assert len(_names) == 2 and _call, f'{INFER_DETECTOR_PK} is not a call/noise detector: {_names}'
    INFER_DET = {'model': _m, 'cfg': _cfg, 'call_idx': _call[0],
                 'label': f'detector from {_label_of(INFER_DETECTOR_PK)}'}
else:
    assert globals().get('det_results'), ('No trained detector in memory: run Cells 7-13 first, '
                                          'or set INFER_DETECTOR_PK to an exported detector .pk')
    print(f'Detector variants (selection uses the VALIDATION {SELECT_METRIC}; test shown for reference):')
    for _mic, _r in det_results.items():
        print(f'  {_mic:<8} val {_r["best_val_acc"]:.4f}   test acc {_r["metrics"]["accuracy"]:.4f}  '
              f'test bal acc {_r["metrics"]["balanced_accuracy"]:.4f}')
    if INFER_DETECTOR == 'ensemble' and len(det_results) > 1:
        INFER_DET = {'model': SoftmaxEnsemble([r['model'] for r in det_results.values()]), 'cfg': DET_CONFIG,
                     'call_idx': 1, 'label': f'ensemble of the fine-tuned detectors {list(det_results)}'}
    else:
        if INFER_DETECTOR in ('auto', 'ensemble'):
            # ties -> m09, the paper's primary microphone
            _mic = max(det_results, key=lambda m: (round(det_results[m]['best_val_acc'], 6), m == 'm09'))
        else:
            assert INFER_DETECTOR in det_results, f'{INFER_DETECTOR!r} not in {list(det_results)}'
            _mic = INFER_DETECTOR
        INFER_DET = {'model': unwrap_model(det_results[_mic]['model']), 'cfg': DET_CONFIG, 'call_idx': 1,
                     'label': f'fine-tuned detector {_mic} (val {SELECT_METRIC} '
                              f'{det_results[_mic]["best_val_acc"]:.4f})'}

# ---- classifier -----------------------------------------------------------------------------
INFER_UNKNOWN = None
if INFER_CLASSIFIER_PK:
    _m, _names, _cfg = _model_from_pk(INFER_CLASSIFIER_PK)
    INFER_CLS = {'model': _m, 'cfg': _cfg, 'names': _names,
                 'label': f'classifier from {_label_of(INFER_CLASSIFIER_PK)}'}
    if INFER_UNKNOWN_FILE:
        INFER_UNKNOWN = load_unknown_model(INFER_UNKNOWN_FILE)
        _given = [os.path.basename(p) for p in ([INFER_CLASSIFIER_PK] if isinstance(INFER_CLASSIFIER_PK, str)
                                                else INFER_CLASSIFIER_PK)]
        if INFER_UNKNOWN['members'] and sorted(INFER_UNKNOWN['members']) != sorted(_given):
            raise ValueError(f'{os.path.basename(INFER_UNKNOWN_FILE)} was fitted for {INFER_UNKNOWN["members"]}, '
                             f'not for {_given}: its threshold would be meaningless for these models')
else:
    assert 'cls_model' in globals(), ('No trained classifier in memory: run Cell 13 first, '
                                      'or set INFER_CLASSIFIER_PK to an exported classifier .pk')
    INFER_CLS = {'model': unwrap_model(cls_model), 'cfg': CLS_CONFIG, 'names': list(CLS_CLASSES),
                 'label': ('fine-tuned classifier ' + (f'ensemble of {len(cls_members)} seeds '
                                                       if len(cls_members) > 1 else '')
                           + f'(val {SELECT_METRIC} {cls_best_acc:.4f})')}
    INFER_UNKNOWN = globals().get('UNKNOWN_MODEL')
INFER_CLS['noise_idx'] = INFER_CLS['names'].index('noise') if 'noise' in INFER_CLS['names'] else None
if INFER_UNKNOWN is not None and not UNKNOWN_DETECTION:
    INFER_UNKNOWN = None

for _r in (INFER_DET, INFER_CLS):
    _r['model'] = _r['model'].to(DEVICE).eval()   # Cell 15 used to leave the live models on the CPU

print(f'\nINFERENCE COMBO')
print(f'  detector  : {INFER_DET["label"]}  ({INFER_DET["cfg"]["sr"]//1000} kHz, '
      f'{INFER_DET["cfg"]["fmin"]}-{INFER_DET["cfg"]["fmax"]} Hz)')
print(f'  classifier: {INFER_CLS["label"]}  ({INFER_CLS["cfg"]["sr"]//1000} kHz, classes {INFER_CLS["names"]})')
print(f'  unknown   : ' + (f'"{UNKNOWN_LABEL}" when the {INFER_UNKNOWN["method"]} score < '
                           f'{INFER_UNKNOWN["threshold"]:.4f}' if INFER_UNKNOWN is not None else 'off'))
print(f'  scan: {INFER_HOP_S*1000:.0f} ms hop, threshold {INFER_DET_THRESHOLD}, merge gap '
      f'{INFER_MERGE_GAP_S*1000:.0f} ms, min {INFER_MIN_WINDOWS} windows, drop classifier-noise: {INFER_DROP_NOISE}'
      + (f', time expansion x{INFER_TIME_EXPANSION}' if INFER_TIME_EXPANSION != 1 else ''))
if INFER_CLS['noise_idx'] is None:
    print('  NOTE: the classifier has no "noise" class, so every detector selection gets a species.')
