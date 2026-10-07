# Cell 7: Data Discovery

# Find all wav files
all_wavs = sorted(glob.glob(os.path.join(DATA_DIR, '**', '*.wav'), recursive=True))
if not all_wavs:
    # Try flat structure
    all_wavs = sorted(glob.glob(os.path.join(DATA_DIR, '*.wav')))

print(f'Found {len(all_wavs)} wav files')

# Drop files that cannot be used BEFORE the split, so the detector and the classifier always see the
# same clip list (dropping a clip inside one dataset only would desynchronise the cascade in Cell 14).
_MIN_CLIP_S = max(DET_CONFIG['n_fft'] / DET_CONFIG['sr'], CLS_CONFIG['n_fft'] / CLS_CONFIG['sr'])


def _unusable_reason(path):
    try:
        info = sf.info(path)
    except Exception as e:
        return f'unreadable ({type(e).__name__})'
    if info.frames == 0:
        return 'empty'
    if info.frames / info.samplerate < _MIN_CLIP_S:
        return f'shorter than one STFT frame ({_MIN_CLIP_S * 1000:.2f} ms)'
    if '-' not in os.path.basename(path):
        return "no 'CLASS-' prefix in the file name"
    return None


_bad_files = [(w, r) for w, r in ((w, _unusable_reason(w)) for w in all_wavs) if r]
if _bad_files:
    print(f'Skipping {len(_bad_files)} unusable file(s):')
    for _w, _r in _bad_files[:10]:
        print(f'  {os.path.relpath(_w, DATA_DIR)}: {_r}')
    _bad_set = {w for w, _ in _bad_files}
    all_wavs = [w for w in all_wavs if w not in _bad_set]
assert all_wavs, f'no usable .wav files under {DATA_DIR}'
_mismatch = [w for w in all_wavs if os.path.dirname(w) != os.path.normpath(DATA_DIR)
             and os.path.basename(os.path.dirname(w)) != get_class_from_filename(w)]
if _mismatch:
    print(f'NOTE: {len(_mismatch)} file(s) sit in a folder whose name differs from their file-name class '
          f'(the FILE NAME decides the class), e.g. {os.path.relpath(_mismatch[0], DATA_DIR)}')

# Auto-detect classes from folder names
class_counts = Counter()
for w in all_wavs:
    cls = get_class_from_filename(w)
    class_counts[cls] += 1

all_classes = sorted(class_counts.keys())
print(f'Classes ({len(all_classes)}):')
for c in all_classes:
    print(f'  {c}: {class_counts[c]} files')

# Update classifier config with actual num_classes
CLS_CONFIG['num_classes'] = len(all_classes)
print(f'\nClassifier num_classes: {CLS_CONFIG["num_classes"]}')

# Detector OUTPUT labels: 0 = noise, 1 = call (Cell 11 maps every species file name onto them)
DET_CLASSES = ['noise', 'call']
assert 'noise' in all_classes, "the dataset needs a 'noise' class (noise-*.wav) for the detector"

# Classifier class mapping: sorted species classes (incl. noise)
CLS_CLASSES = all_classes
CLS_CLASS_TO_IDX = {c: i for i, c in enumerate(all_classes)}
CLS_IDX_TO_CLASS = {i: c for c, i in CLS_CLASS_TO_IDX.items()}

print(f'\nDetector output labels: {DET_CLASSES}')
print(f'Classifier mapping: {CLS_CLASS_TO_IDX}')

# ---------------------------------------------------------------------------
# DATA SPLIT: train / val / test = 70 / 15 / 15, ONE shared split for detector and classifier
# so the cascade has no leakage. Three modes (Cell 2):
#
#   SPLIT_BY_RECORDING = False
#       clip-level, stratified by species (incl. noise): every species gives ~15 % to val and to test.
#
#   SPLIT_BY_RECORDING = True, RECORDING_SPLIT_SCOPE = 'per_species'   (default for True)
#       each species is split ON ITS OWN, and whole recordings are the unit: a species' recordings go to
#       train / val / test so that each part holds ~70 / 15 / 15 % of THAT species' clips. Every class
#       therefore appears in val and test, and no recording contributes the same species to train and
#       to val/test (species with < 3 recordings cannot be split this way; see the rules below).
#
#   SPLIT_BY_RECORDING = True, RECORDING_SPLIT_SCOPE = 'global'
#       one StratifiedGroupKFold over all clips: recordings are indivisible and shared between species,
#       so val/test get whatever species happen to sit in their ~4 recordings (classes go missing).
#
# RECORDING LEAKAGE WARNING: the filename carries the tape (recording) id
# (CLASS-LABEL_ID_YEAR_TAPE_START_END.wav). This dataset has only ~28 recordings (heti is a
# single one), so a clip-level split puts clips of the SAME recording in train and test.
# The network can then score well by recognising the recording (background, mic distance)
# rather than the species -> optimistic numbers. The check at the end of this cell quantifies it.
# ---------------------------------------------------------------------------
import re
import itertools
# Recording timestamp YYYYMMDD + HHMMSS. The dataset clips write 20260529-202000; Cell 8's extractor
# (and AudioMoth file names) write 20260529_202000, so accept either separator.
_TAPE_RE = re.compile(r'(\d{8})[-_](\d{6})')

def recording_id(path):
    """Recording (tape) id = the YYYYMMDD-HHMMSS timestamp inside the file name. Searching for it
    (instead of taking field 4) works for CLASS-LABEL_ID_YEAR_TAPE_START_END.wav and for other
    layouts such as Cell 8's extractor output. Fallback: the file stem (= no grouping)."""
    stem = os.path.basename(path)[:-4]
    m = _TAPE_RE.search(stem)
    return f'{m.group(1)}-{m.group(2)}' if m else stem

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)          # train / val / test


def _assign_recordings(sizes, fractions, rng):
    """Give every recording (an indivisible group of clips) to one of len(fractions) parts so that each
    part's clip total is as close as possible to fraction * total, every part getting >= 1 recording.
    Up to 11 recordings: exhaustive search, i.e. the optimum (ties are broken by `rng`).
    More: largest-first greedy into the part furthest below its target."""
    sizes = np.asarray(sizes, dtype=float)
    k, p = len(sizes), len(fractions)
    target = np.asarray(fractions, dtype=float) * sizes.sum()
    if k <= 11:
        allp = np.array(list(itertools.product(range(p), repeat=k)))             # (p**k, k)
        tot = np.stack([(allp == j) @ sizes for j in range(p)], axis=1)          # clips per part
        cost = ((tot - target) ** 2).sum(axis=1)
        cost[~np.all([(allp == j).any(axis=1) for j in range(p)], axis=0)] = np.inf   # no empty part
        return allp[rng.choice(np.flatnonzero(cost <= cost.min() + 1e-9))]
    parts, tot = np.full(k, -1), np.zeros(p)
    for i in np.argsort(-sizes, kind='stable'):
        j = int(np.argmax(target - tot))
        parts[i] = j
        tot[j] += sizes[i]
    for j in range(p):                                                             # repair: no empty part
        if not (parts == j).any():
            cand = np.flatnonzero(parts == np.bincount(parts, minlength=p).argmax())
            parts[cand[np.argmin(sizes[cand])]] = j
    return parts


def split_by_recording_per_species(files, species, rec_ids, fractions=SPLIT_FRACTIONS, seed=42):
    """SPLIT_BY_RECORDING, one species at a time. Returns (train, val, test, rules); `rules[species]`
    = (number of recordings, rule applied). Every file lands in exactly one part.
      >= 3 recordings : whole recordings -> train / val / test, ~fractions of that species' clips
      == 2 recordings : the bigger recording -> train; the smaller one is split val/test by clip
                        (train is still recording-disjoint from val/test; val and test share a tape)
      == 1 recording  : clip-level split (the only option; train, val and test share the recording)"""
    rng = np.random.default_rng(seed)
    parts, rules = [[], [], []], {}
    for sp in sorted(set(species)):
        by_rec = {}
        for i, (s, r) in enumerate(zip(species, rec_ids)):
            if s == sp:
                by_rec.setdefault(r, []).append(i)
        recs = sorted(by_rec)
        n, k = sum(len(v) for v in by_rec.values()), len(recs)
        if k >= 3:
            rules[sp] = (k, 'whole recordings held out')
            for r, j in zip(recs, _assign_recordings([len(by_rec[r]) for r in recs], fractions, rng)):
                parts[j] += by_rec[r]
        elif k == 2:
            rules[sp] = (k, 'bigger recording -> train; smaller one split val/test by clip')
            big, small = sorted(recs, key=lambda r: (-len(by_rec[r]), r))
            parts[0] += by_rec[big]
            idx = list(by_rec[small])
            rng.shuffle(idx)
            parts[1] += idx[:(len(idx) + 1) // 2]
            parts[2] += idx[(len(idx) + 1) // 2:]
        else:
            idx = [i for r in recs for i in by_rec[r]]
            if n < 3:
                rules[sp] = (k, 'LESS THAN 3 CLIPS -> all in train (cannot be evaluated)')
                parts[0] += idx
                continue
            rules[sp] = (k, 'CLIP-LEVEL (single recording: shared by train/val/test)')
            rng.shuffle(idx)
            n_val, n_test = max(1, round(fractions[1] * n)), max(1, round(fractions[2] * n))
            parts[1] += idx[:n_val]
            parts[2] += idx[n_val:n_val + n_test]
            parts[0] += idx[n_val + n_test:]
    return tuple([files[i] for i in sorted(p)] for p in parts) + (rules,)


all_species_labels = [get_class_from_filename(w) for w in all_wavs]
all_rec_ids = [recording_id(w) for w in all_wavs]
_rec_of = dict(zip(all_wavs, all_rec_ids))
all_rec_groups = set(all_rec_ids)

# Recordings PER CLASS: predicts which classes will look perfect on a clip-level split (few
# recordings = test clips are near-copies of train clips) and which cannot be held out at all.
_rec_per_class = defaultdict(set)
for _w, _s in zip(all_wavs, all_species_labels):
    _rec_per_class[_s].add(_rec_of[_w])
print(f'\nRecordings: {len(all_rec_groups)} distinct tape ids for {len(all_wavs)} clips')
for _c in all_classes:
    _n = len(_rec_per_class.get(_c, ()))
    print(f'  {_c:<8} {_n:>3} recording(s)' + ('   <-- single recording: cannot be validated on a new tape'
                                               if _n == 1 else ('   <-- only 2 recordings' if _n == 2 else '')))

if SPLIT_BY_RECORDING and RECORDING_SPLIT_SCOPE == 'per_species':
    train_wavs, val_wavs, test_wavs, _split_rules = split_by_recording_per_species(
        all_wavs, all_species_labels, all_rec_ids, SPLIT_FRACTIONS, seed=42)
    train_labels = [get_class_from_filename(w) for w in train_wavs]
    val_labels = [get_class_from_filename(w) for w in val_wavs]
    test_labels = [get_class_from_filename(w) for w in test_wavs]
    print('SPLIT_BY_RECORDING = True, scope per_species: every species is split on its own, whole '
          f'recordings per part, target {SPLIT_FRACTIONS[0]:.0%}/{SPLIT_FRACTIONS[1]:.0%}/'
          f'{SPLIT_FRACTIONS[2]:.0%} of THAT species\' clips.')
    _tr_c, _va_c, _te_c = Counter(train_labels), Counter(val_labels), Counter(test_labels)
    print(f'\n  {"species":<8}{"clips":>6}{"recs":>5} | {"train":>5}{"val":>5}{"test":>5} | '
          f'{"train%":>7}{"val%":>6}{"test%":>6} | rule')
    for _sp in all_classes:
        _n = class_counts[_sp]
        _k, _rule = _split_rules[_sp]
        _c = (_tr_c[_sp], _va_c[_sp], _te_c[_sp])
        print(f'  {_sp:<8}{_n:>6}{_k:>5} | {_c[0]:>5}{_c[1]:>5}{_c[2]:>5} | '
              f'{100 * _c[0] / _n:>7.1f}{100 * _c[1] / _n:>6.1f}{100 * _c[2] / _n:>6.1f} | {_rule}')
elif SPLIT_BY_RECORDING:        # RECORDING_SPLIT_SCOPE == 'global'
    from sklearn.model_selection import StratifiedGroupKFold
    _folds = [te for _, te in StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=1)
              .split(all_wavs, all_species_labels, all_rec_ids)]
    test_wavs = [all_wavs[i] for i in _folds[0]]
    val_wavs = [all_wavs[i] for i in _folds[1]]
    _held = set(_folds[0]) | set(_folds[1])
    train_wavs = [w for i, w in enumerate(all_wavs) if i not in _held]
    train_labels = [get_class_from_filename(w) for w in train_wavs]
    val_labels = [get_class_from_filename(w) for w in val_wavs]
    test_labels = [get_class_from_filename(w) for w in test_wavs]
    print('SPLIT_BY_RECORDING = True, scope global: whole recordings are held out as ONE pool. This is NOT '
          '70/15/15 and classes with few recordings will be missing from val/test -- read per-class '
          'numbers with care (RECORDING_SPLIT_SCOPE = "per_species" splits each species on its own).')
else:
    train_wavs, test_wavs, train_labels, test_labels = train_test_split(
        all_wavs, all_species_labels, test_size=0.15, random_state=42,
        stratify=all_species_labels)
    train_wavs, val_wavs, train_labels, val_labels = train_test_split(
        train_wavs, train_labels, test_size=0.15/0.85, random_state=42,
        stratify=train_labels)

print(f'\nSplit: train={len(train_wavs)}, val={len(val_wavs)}, test={len(test_wavs)}')
print('Train class dist:', Counter(train_labels))
print('Val class dist:  ', Counter(val_labels))
print('Test class dist: ', Counter(test_labels))
_missing = {n: [c for c in all_classes if c not in d] for n, d in
            (('train', Counter(train_labels)), ('val', Counter(val_labels)), ('test', Counter(test_labels)))}
for _n, _m in _missing.items():
    if _m:
        print(f'  WARNING: {_n} has NO clips of {_m} -- those classes cannot be scored/selected on there.')

# --- recording leakage -----------------------------------------------------------------
_train_rec = {_rec_of[w] for w in train_wavs}
_train_rec_sp = {(_rec_of[w], get_class_from_filename(w)) for w in train_wavs}
_n_leak = sum(_rec_of[w] in _train_rec for w in test_wavs)
print(f'\nLEAKAGE CHECK: {_n_leak}/{len(test_wavs)} test clips come from a recording that also '
      f'appears in train ({len(all_rec_groups)} recordings in total).')
_same_va = sum((_rec_of[w], get_class_from_filename(w)) in _train_rec_sp for w in val_wavs)
_same_te = sum((_rec_of[w], get_class_from_filename(w)) in _train_rec_sp for w in test_wavs)
print(f'  same SPECIES and same recording as a train clip: val {_same_va}/{len(val_wavs)}, '
      f'test {_same_te}/{len(test_wavs)}  (the shortcut "recognise the tape -> name the species")')
_leaky = Counter(get_class_from_filename(w) for w in test_wavs
                 if (_rec_of[w], get_class_from_filename(w)) in _train_rec_sp)
if _leaky:
    print('  test clips per species that still share a recording with train:', dict(sorted(_leaky.items())))
if len(all_rec_groups) == len(all_wavs):
    print('  WARNING: every clip is its own recording -- recording_id() found no tape id in the '
          'file names, so this check and SPLIT_BY_RECORDING are meaningless for this dataset.')


def _tape_lookup_baseline(test_files, label_of):
    """No audio, no model: label every test clip like the MAJORITY of the train clips from the same recording.
    Returns (clips whose recording is in train, accuracy, balanced accuracy) or None."""
    by = {}
    for w in train_wavs:
        by.setdefault(_rec_of[w], Counter())[label_of(w)] += 1
    ys = [label_of(w) for w in test_files if _rec_of[w] in by]
    ps = [by[_rec_of[w]].most_common(1)[0][0] for w in test_files if _rec_of[w] in by]
    if not ys:
        return None
    bal = float(np.mean([np.mean([p == c for y, p in zip(ys, ps) if y == c]) for c in sorted(set(ys))]))
    return len(ys), float(np.mean([y == p for y, p in zip(ys, ps)])), bal


print('\nTAPE-LOOKUP BASELINE (predict a test clip like the majority of train clips from its own recording):')
for _task, _label_of, _chance in (('species      ', get_class_from_filename, f'1/{len(all_classes)}'),
                                  ('noise-vs-call', lambda w: 'noise' if get_class_from_filename(w) == 'noise'
                                   else 'call', '1/2')):
    _r = _tape_lookup_baseline(test_wavs, _label_of)
    if _r is None:
        print(f'  {_task}: no test clip shares a recording with train (the recording tells nothing)')
    else:
        print(f'  {_task}: {_r[0]}/{len(test_wavs)} test clips have their recording in train -> '
              f'accuracy {_r[1]:.3f}, balanced accuracy {_r[2]:.3f}   (chance {_chance})')
print('  Reading: ~chance = the recording says nothing (a fair estimate for NEW recordings); far above = the split '
      'leaks the recording; far below = the recording points at ANOTHER label (adversarial).')
if SPLIT_BY_RECORDING and RECORDING_SPLIT_SCOPE == 'per_species':
    print('  NOTE per_species: every species is split on its own, so a test clip whose recording is in train always '
          'meets train clips of ANOTHER species there. A model that leans on the recording (background, mic) is '
          'then scored BELOW chance. Read this split as a pessimistic bound; the "same SPECIES" line above shows '
          'the own-species leakage is 0 wherever a species had >= 2 recordings.')
elif _n_leak:
    print('  -> test scores are optimistic about NEW recordings; set SPLIT_BY_RECORDING = True '
          'for a stricter (but noisier) estimate.')
