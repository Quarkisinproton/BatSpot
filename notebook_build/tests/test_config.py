#!/usr/bin/env python3
"""Cell 1 (config): the Nyquist guard, and the defaults the ported pipeline reads.

The guard is the whole point of this suite. `animal_spot/predict.py:214` reads
`dataOpts["fmax"]` and `animal_spot/data/audiodataset.py` uses `f_max` unclamped,
so an fmax above `sr / 2` is not an error anywhere -- it returns wrong
frequencies with no message. The detector's 95 kHz fmax already sits 1 kHz under
its 96 kHz Nyquist, so the margin is thin enough to lose by typing.

`check_bands` is exec'd out of the delivered cell rather than pasted here, and so
is the cell's own call site: a pasted copy of the guard would keep passing no
matter what the notebook does.

The band values below are the official `.pk` `dataOpts` of
`BatSpot_article/batspot/models_call_{detector,classifier}/**/ANIMAL-SPOT.pk`:
all three detectors read `sr 192000, fmin 1000, fmax 95000`, the classifier
`sr 250000, fmin 10000, fmax 125000`, and both `n_fft 256, hop_length 128,
n_freq_bins 256`. A transferred encoder is only meaningful at the resolution it
was trained at, so these are pinned here rather than left to drift.

Run: venv/bin/python notebook_build/tests/test_config.py
"""
import ast
import contextlib
import io
import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


MISSING = '<MISSING>'


def check_eq(name, triples):
    """One check over (label, got, want) triples; every mismatch is named.

    A name the cell does not define at all is a mismatch like any other, not a
    traceback: a dropped flag must be reported, not abort the remaining checks.
    """
    wrong = [f'{lbl}: {got!r} != {want!r}' for lbl, got, want in triples if got != want]
    check(name, not wrong, '; '.join(wrong) if wrong else f'{len(triples)} value(s)')


def raised_by(fn, *a, **kw):
    """The exception fn(*a, **kw) raised, or None if it returned.

    Any exception type is returned as-is; the caller asserts on which one it got,
    so a wrong type reads as a failure rather than a crash.
    """
    try:
        fn(*a, **kw)
    except Exception as e:
        return e
    return None


# --- the cell under test, read out of the artifact that ships --------------------------

CELL_CONFIG = 1
idx = extract_cells.merged_cells()
UNDER_TEST = os.environ.get('NEW_CELLS') or extract_cells.MERGED

check('the extraction returns the config cell', CELL_CONFIG in idx, f'from {UNDER_TEST}')
CONFIG_PATH = idx[CELL_CONFIG]
with open(CONFIG_PATH, encoding='utf-8') as f:
    CONFIG_SRC = f.read()
check('extracted cell 1 really is the config cell',
      'SELECT_METRIC' in CONFIG_SRC, os.path.basename(CONFIG_PATH))

# Exec the whole cell, not a hand-picked subset: the cell's own module-level guard
# call is part of what is under test. Two things are stood down for the duration --
# os.makedirs (the cell creates /kaggle/working/pretrained, which this machine
# cannot have) and stdout (the cell prints a report; the suite's output is its own).
MADE = []
_ns = {}
with mock.patch.object(os, 'makedirs', side_effect=lambda p, *a, **k: MADE.append(p)), \
        contextlib.redirect_stdout(io.StringIO()):
    exec(compile(CONFIG_SRC, CONFIG_PATH, 'exec'), _ns)
cfg = types.SimpleNamespace(**_ns)

check('the config cell creates PRETRAINED_DIR and nothing else',
      MADE == [getattr(cfg, 'PRETRAINED_DIR', None)], f'{MADE}')

# --- 1. the Nyquist guard ----------------------------------------------------------------

check_bands = getattr(cfg, 'check_bands', None)
check('the config cell defines check_bands', check_bands is not None,
      '' if check_bands is not None else 'nothing in cell 1 defines check_bands')

if check_bands is not None:
    err = raised_by(check_bands, {'DET': {'sr': 192000, 'fmax': 110000}})
    check('fmax above Nyquist raises AssertionError', isinstance(err, AssertionError),
          f'got {type(err).__name__}: {err}' if err else 'no exception')
    if isinstance(err, AssertionError):
        msg = str(err)
        check('   ...naming the offending config', 'DET' in msg, msg)
        check('   ...naming both numbers', '110000' in msg and '192000' in msg, msg)
        check('   ...saying the consequence is a silently wrong answer',
              'wrong frequencies' in msg, msg)

    # The classifier's fmax is exactly its Nyquist (125000 == 250000/2), so `>`
    # rather than `>=` is the comparison that keeps the shipped config legal.
    at_nyq = raised_by(check_bands, {'CLS': {'sr': 250000, 'fmax': 125000}})
    check('fmax exactly AT Nyquist is accepted (the classifier ships there)',
          at_nyq is None, f'{at_nyq}')

    err = raised_by(check_bands, {'DET': cfg.DET_CONFIG, 'CLS': cfg.CLS_CONFIG})
    check('the shipped detector and classifier bands both pass',
          err is None, f'{err}' if err else 'sr 192000/fmax 95000, sr 250000/fmax 125000')

    # The cell must not merely define the guard: it must run it on its own configs,
    # so a bad band stops the notebook at the config cell rather than after training.
    call = None
    for node in ast.parse(CONFIG_SRC).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and getattr(node.value.func, 'id', None) == 'check_bands':
            call = ast.Module(body=[node], type_ignores=[])
    check('the config cell CALLS check_bands on its own configs', call is not None,
          '' if call is not None else 'no module-level check_bands(...) call')

    if call is not None:
        bad = dict(_ns)
        bad['DET_CONFIG'] = {**cfg.DET_CONFIG, 'fmax': cfg.DET_CONFIG['fmax'] + 20000}
        err = raised_by(exec, compile(call, CONFIG_PATH, 'exec'), dict(_ns))
        check("the cell's own call site accepts the shipped bands", err is None, f'{err}')
        err = raised_by(exec, compile(call, CONFIG_PATH, 'exec'), bad)
        check("the cell's own call site REJECTS an out-of-band DET_CONFIG",
              isinstance(err, AssertionError),
              f'{type(err).__name__}: {err}' if err else 'no exception -- silent wrong answer')

# --- 2. the defaults the later cells read ------------------------------------------------

check_eq('the ported feature flags are ON by default (2026-10-07 decision)', [
    ('UNKNOWN_DETECTION', getattr(cfg, 'UNKNOWN_DETECTION', MISSING), True),
    ('UNKNOWN_KEEP_KNOWN', getattr(cfg, 'UNKNOWN_KEEP_KNOWN', MISSING), 0.95),
    ('UNKNOWN_LABEL', getattr(cfg, 'UNKNOWN_LABEL', MISSING), 'unknown'),
    ('UNKNOWN_METHOD', getattr(cfg, 'UNKNOWN_METHOD', MISSING), 'maha'),
    ('CLS_ENSEMBLE_SEEDS', getattr(cfg, 'CLS_ENSEMBLE_SEEDS', MISSING), [42, 43, 44]),
    ('SEED', getattr(cfg, 'SEED', MISSING), 42),
])
check_eq('background noise mixing is on for both models', [
    ('DET_CONFIG[noise_mix_prob]', cfg.DET_CONFIG.get('noise_mix_prob', MISSING), 0.5),
    ('CLS_CONFIG[noise_mix_prob]', cfg.CLS_CONFIG.get('noise_mix_prob', MISSING), 0.5),
    ('DET_CONFIG[noise_mix_snr_db]', cfg.DET_CONFIG.get('noise_mix_snr_db', MISSING), (0.0, 20.0)),
    ('CLS_CONFIG[noise_mix_snr_db]', cfg.CLS_CONFIG.get('noise_mix_snr_db', MISSING), (0.0, 20.0)),
    ('DET_CONFIG[label_smoothing]', cfg.DET_CONFIG.get('label_smoothing', MISSING), 0.0),
    ('CLS_CONFIG[label_smoothing]', cfg.CLS_CONFIG.get('label_smoothing', MISSING), 0.0),
])
check_eq('the phase gates later cells guard on are present with the agreed defaults', [
    ('RUN_MEASURE', getattr(cfg, 'RUN_MEASURE', MISSING), True),
    ('RUN_EXPERIMENTS', getattr(cfg, 'RUN_EXPERIMENTS', MISSING), False),
    ('RUN_PHASE4', getattr(cfg, 'RUN_PHASE4', MISSING), False),
    ('RUN_GROUPED_CHECK', getattr(cfg, 'RUN_GROUPED_CHECK', MISSING), True),
    ('RUN_CLIP_EXTRACTION', getattr(cfg, 'RUN_CLIP_EXTRACTION', MISSING), False),
])
check_eq('windowing / selection / split flags are unchanged from the base cell', [
    ('WINDOW_MODE', getattr(cfg, 'WINDOW_MODE', MISSING), 'energy_crop'),
    ('WINDOW_STRIDE', getattr(cfg, 'WINDOW_STRIDE', MISSING), 3),
    ('TRAIN_TOP_FRAC', getattr(cfg, 'TRAIN_TOP_FRAC', MISSING), 0.20),
    ('TEST_TOPK', getattr(cfg, 'TEST_TOPK', MISSING), 5),
    ('SELECT_METRIC', getattr(cfg, 'SELECT_METRIC', MISSING), 'balanced_accuracy'),
    ('SPLIT_BY_RECORDING', getattr(cfg, 'SPLIT_BY_RECORDING', MISSING), False),
    ('RECORDING_SPLIT_SCOPE', getattr(cfg, 'RECORDING_SPLIT_SCOPE', MISSING), 'per_species'),
    ('REPORT_TOPK_MEAN', getattr(cfg, 'REPORT_TOPK_MEAN', MISSING), 3),
    ('AUC_NO_SIGNAL', getattr(cfg, 'AUC_NO_SIGNAL', MISSING), 0.05),
    ('USE_AUGMENTATION', getattr(cfg, 'USE_AUGMENTATION', MISSING), False),
])

# --- 3. the bands, pinned to the official .pk dataOpts -----------------------------------

OFFICIAL = {                       # read out of BatSpot_article/batspot/models_*/**/ANIMAL-SPOT.pk
    'DET': {'sr': 192000, 'fmin': 1000, 'fmax': 95000,
            'n_fft': 256, 'hop_length': 128, 'n_freq_bins': 256, 'sequence_len': 20},
    'CLS': {'sr': 250000, 'fmin': 10000, 'fmax': 125000,
            'n_fft': 256, 'hop_length': 128, 'n_freq_bins': 256, 'sequence_len': 20},
}
for role, conf in (('DET', cfg.DET_CONFIG), ('CLS', cfg.CLS_CONFIG)):
    check_eq(f'{role}_CONFIG matches the official .pk dataOpts', [
        (k, conf.get(k, MISSING), v) for k, v in OFFICIAL[role].items()
    ])
    fmax, sr = conf.get('fmax'), conf.get('sr')
    check(f'{role}_CONFIG fmax <= sr/2', fmax is not None and sr is not None
          and fmax <= sr / 2, f'fmax={fmax}, sr={sr}, Nyquist={sr // 2 if sr else MISSING}')

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)