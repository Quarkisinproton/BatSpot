# Cell 10: Load Pre-trained Models from the uploaded Kaggle dataset
#
# Expects the official BatSpot models (extracted from BatSpot_article):
#   <dataset>/batspot/models_call_detector/m03|m09|m11/train/ANIMAL-SPOT.pk
#   <dataset>/batspot/models_call_classifier/m09/train/ANIMAL-SPOT.pk
#
# NOTE: all four files share the basename ANIMAL-SPOT.pk, so each is staged
# here under a UNIQUE name. Copying by basename would let the classifier
# overwrite the detector and silently fine-tune the detector from the
# 15-class classifier weights.

os.makedirs(PRETRAINED_DIR, exist_ok=True)

# --- Priority 1: explicit paths from Cell 2 ---
_official = {}
_manual_set = set()
_missing_manual = []
_manual_set_attempted = False
if USE_MANUAL_MODEL_PATHS:
    for _lbl, _role, _mic, _p in [
            ('detector m03', 'detector', 'm03', DETECTOR_M03_PATH),
            ('detector m09', 'detector', 'm09', DETECTOR_M09_PATH),
            ('detector m11', 'detector', 'm11', DETECTOR_M11_PATH),
            ('classifier m09', 'classifier', 'm09', CLASSIFIER_M09_PATH)]:
        if _p is None:
            continue
        _manual_set_attempted = True
        if os.path.isfile(_p):
            _ap = os.path.abspath(_p)
            _manual_set.add(_ap)
            _official.setdefault(_role, {})[_mic] = _ap
        else:
            _missing_manual.append(f'{_lbl}: {_p}')
    if _official:
        print(f'Using {sum(len(v) for v in _official.values())} manual path(s) from Cell 2.')
    for _m in _missing_manual:
        print(f'  manual path NOT FOUND -> {_m}')
    if _missing_manual:
        print('  falling back to auto-discovery for the missing ones.')
else:
    print('USE_MANUAL_MODEL_PATHS = False -> ignoring Cell 2 paths.')

# --- Priority 2: discover the dataset directory (local repo or /kaggle/input/*) ---
BATSPOT_ARTICLE_DIR = None
_MODEL_GLOB = os.path.join('models_*', '*', 'train', 'ANIMAL-SPOT.pk')

for _cand in ['BatSpot_article/batspot', 'BatSpot_article']:
    if glob.glob(os.path.join(_cand, _MODEL_GLOB)):
        BATSPOT_ARTICLE_DIR = _cand
        break

if BATSPOT_ARTICLE_DIR is None:
    for _root in sorted(glob.glob('/kaggle/input/*')):
        for _cand in [os.path.join(_root, 'batspot'), _root]:
            if glob.glob(os.path.join(_cand, _MODEL_GLOB)):
                BATSPOT_ARTICLE_DIR = _cand
                break
        if BATSPOT_ARTICLE_DIR:
            break

# --- Collect sources, keyed role -> mic -> path ---
#
# _official ALREADY holds whatever the manual block above resolved. Do NOT
# re-initialise it here: doing so discards the Cell 2 paths and makes this
# cell report "No pre-trained models found" even though the manual lookup
# just succeeded. Discovery only fills roles/mics the manual block missed.
_manual_n = sum(len(v) for v in _official.values())
_skipped = []
if BATSPOT_ARTICLE_DIR:
    for _p in sorted(glob.glob(os.path.join(BATSPOT_ARTICLE_DIR, _MODEL_GLOB))):
        _parts = os.path.relpath(_p, BATSPOT_ARTICLE_DIR).split(os.sep)
        # e.g. ['models_call_detector', 'm09', 'train', 'ANIMAL-SPOT.pk']
        if len(_parts) < 4:
            continue
        _family, _mic = _parts[0], _parts[1]
        # Resolve the role FIRST, and only ever to 'classifier' or 'detector'.
        # buzz_detector / social_detector use different band limits and are
        # not fine-tuned here; classifying them as detectors would feed their
        # weights into the call-detector slot.
        if _family == 'models_call_classifier':
            _role = 'classifier'
        elif _family == 'models_call_detector':
            _role = 'detector'
        else:
            _skipped.append(f'{_family}/{_mic}')
            continue
        if _mic in _official.get(_role, {}):
            continue  # already supplied manually via Cell 2 (manual wins)
        _official.setdefault(_role, {})[_mic] = _p
    if _manual_n:
        print(f'Auto-discovery added nothing (Cell 2 supplied {_manual_n} model(s)); '
              f'manual paths stand.')

# --- Stage under unique names and read REAL metadata from each .pk ---
# PRETRAINED_MODELS[role][mic] = {path, sr, classes, num_classes, data, source}
PRETRAINED_MODELS = {}
detector_path = None
classifier_path = None

if _official:
    if BATSPOT_ARTICLE_DIR:
        print(f'BatSpot_article dataset: {BATSPOT_ARTICLE_DIR}\n')
    for _role, _mics in _official.items():
        for _mic, _src in sorted(_mics.items()):
            _dest = os.path.join(PRETRAINED_DIR, f'official_{_role}_{_mic}.pk')
            shutil.copy2(_src, _dest)
            try:
                _o = torch.load(_dest, map_location='cpu', weights_only=False)
                _d = _o.get('dataOpts', {})
                PRETRAINED_MODELS.setdefault(_role, {})[_mic] = {
                    'path': _dest,
                    'data': _d,
                    'sr': _d.get('sr'),
                    'classes': _o.get('classes', {}),
                    'num_classes': _o.get('classifierOpts', {}).get('num_classes'),
                    'source': _src,
                    'origin': 'manual' if os.path.abspath(_src) in _manual_set else 'discovered',
                }
            except Exception as e:
                print(f'  WARNING: could not read {_dest}: {e}')
    print('Official models staged:')
    for _role in sorted(PRETRAINED_MODELS):
        for _mic, _info in sorted(PRETRAINED_MODELS[_role].items()):
            _sr = _info['sr'] or 0
            print(f'  {_role:9s} {_mic}: {_info["num_classes"]:2d} classes, {_sr//1000:3d} kHz'
                  f'  [{_info.get("origin", "?")}] <- {_info["source"]}')
    # m09 is the paper's primary microphone (call detector test F1 0.985).
    # Prefer it; fall back to the lowest-numbered variant.
    for _role in ('detector', 'classifier'):
        _mics = PRETRAINED_MODELS.get(_role, {})
        if not _mics:
            continue
        _pick = 'm09' if 'm09' in _mics else sorted(_mics)[0]
        if _role == 'detector':
            detector_path = _mics[_pick]['path']
        else:
            classifier_path = _mics[_pick]['path']
        print(f'  -> fine-tuning {_role} from {_pick}')
    if _skipped:
        print(f'  (skipped, not fine-tuned here: {sorted(set(_skipped))})')
elif not _official:
    print('No pre-trained models found.')
    if USE_MANUAL_MODEL_PATHS and _manual_set_attempted:
        print('All Cell 2 paths were set but none resolved to a readable file.')
        print('Re-run Cell 2 and check the per-path OK / NOT FOUND lines above it.')
    else:
        print('Either set DETECTOR_M03_PATH / DETECTOR_M09_PATH / DETECTOR_M11_PATH /')
        print('CLASSIFIER_M09_PATH in Cell 2, or attach the dataset via Add Data.')

# --- Verify our spectrogram config matches each official model's dataOpts ---
print('\nConfig vs model dataOpts check:')
for _role, _cfg in (('detector', DET_CONFIG), ('classifier', CLS_CONFIG)):
    _mics = PRETRAINED_MODELS.get(_role)
    if not _mics:
        continue
    for _mic, _info in sorted(_mics.items()):
        _d = _info['data']
        _bad = []
        for _k in ('sr', 'n_fft', 'hop_length', 'fmin', 'fmax', 'n_freq_bins', 'freq_compression'):
            if _k in _d and _k in _cfg and _d[_k] != _cfg[_k]:
                _bad.append(f'{_k}: model={_d[_k]} config={_cfg[_k]}')
        _tag = f'official {_role} ({_mic})'
        if _bad:
            print(f'  MISMATCH {_tag}: ' + '; '.join(_bad))
        else:
            print(f'  OK        {_tag}')

# --- Class-overlap report (official models use European species codes) ---
if 'classes' in dir() or 'classes' in globals():
    _mine = list(globals().get('classes', []))
    for _role, _cfg in (('detector', DET_CONFIG), ('classifier', CLS_CONFIG)):
        for _mic, _info in sorted(PRETRAINED_MODELS.get(_role, {}).items()):
            _theirs = set(_info['classes'].keys())
            _ov = sorted(set(_mine) & _theirs)
            print(f'\n{_role} ({_mic}) knows {len(_theirs)} classes; '
                  f'overlap with your data {sorted(_mine)}: {_ov or "NONE"}')
            if not _ov and _role == 'classifier':
                print('  -> classifier head cannot transfer; only the encoder will be used.')
            elif _ov and _role == 'detector':
                print('  -> evaluated on the overlapping classes only (noise).')

# --- Fallback: local Data/model_output (384 kHz, 3-class) ---
if detector_path is None and classifier_path is None:
    _local = None
    for _cand in ['Data/model_output/ANIMAL-SPOT.pk',
                  '/kaggle/input/batspot-data/model_output/ANIMAL-SPOT.pk']:
        if os.path.isfile(_cand):
            _local = _cand
            break
    if _local is None and MODEL_OUTPUT_DIR and os.path.isdir(MODEL_OUTPUT_DIR):
        for _root, _, _files in os.walk(MODEL_OUTPUT_DIR):
            for _f in _files:
                if _f.endswith('.pk'):
                    _local = os.path.join(_root, _f)
                    break
            if _local:
                break
    if _local:
        _dest = os.path.join(PRETRAINED_DIR, 'fallback_local.pk')
        shutil.copy2(_local, _dest)
        detector_path = _dest
        classifier_path = _dest
        print(f'\nFallback local model: {_local}')
        print('  NOTE: 384 kHz / 3-class model. Spectrogram params will NOT match')
        print('        DET_CONFIG or CLS_CONFIG, so encoder transfer is unreliable.')
    else:
        print('\nWARNING: no pre-trained model found. Both models train from scratch.')

print(f'\nFine-tuning starting points:')
print(f'  Detector:   {detector_path}')
print(f'  Classifier: {classifier_path}')