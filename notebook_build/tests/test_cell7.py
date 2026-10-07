#!/usr/bin/env python3
"""The data-discovery cell on the REAL dataset: split, recording_id, leakage check, priors.

Runs the cell exactly as the notebook runs it (only its config globals are supplied),
so `train_wavs` / `test_wavs` / `recording_id` / the priors are the artifact's, not copies.

Checks 1-2 compare against the read-only base parent, purely as a regression guard: a port
that silently re-split the data would be invisible everywhere else. While the merged cell is
byte-identical to the base those two are trivially true, which the run says out loud.

Run: venv/bin/python notebook_build/tests/test_cell7.py
"""
import glob
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

CELL_DATA = 6   # Cell 7: Data Discovery

fails = []


def check(n, c, d=''):
    print(f'  [{"PASS" if c else "FAIL"}] {n}' + (f'  -- {d}' if d else ''))
    if not c:
        fails.append(n)


CLIPS = sorted(glob.glob(f'{extract_cells.REPO}/Data/final_dataset/data/*/*.wav'))
gcls = lambda f: os.path.basename(f).split('-', 1)[0]

# The cell's config globals are built by extract_cells.cell6_ns at the REAL values
# (192/250 kHz, n_fft 256), so `_MIN_CLIP_S` is the real unusable-clip threshold.
idx = extract_cells.merged_cells()
nn, out_MERGED = extract_cells.run_cell6(idx, CELL_DATA)

print(f'=== {os.path.basename(idx[CELL_DATA])} output (filtered) ===')
for ln in out_MERGED.split('\n'):
    if any(k in ln for k in ('LEAKAGE', 'Recordings', 'noise=', 'correction factor',
                             'single recording', '2 recordings', 'detector  train')):
        print('   ', ln)

base = extract_cells.base_cells()
IDENTICAL = open(base[CELL_DATA], encoding='utf-8').read() == \
    open(idx[CELL_DATA], encoding='utf-8').read()
if IDENTICAL:
    print('  (note: the merged cell is byte-identical to the base parent, so the two '
          'base-comparison checks\n   below are trivially true until the cell is ported; '
          'redgreen.py covers the port itself)')
nb, _ = extract_cells.run_cell6(base, CELL_DATA)

print('\n=== checks ===')
check('split sizes are the base notebook\'s 674/145/145',
      (len(nn['train_wavs']), len(nn['val_wavs']), len(nn['test_wavs']))
      == (len(nb['train_wavs']), len(nb['val_wavs']), len(nb['test_wavs'])),
      f"{len(nn['train_wavs'])}/{len(nn['val_wavs'])}/{len(nn['test_wavs'])}")
check('same test clips as the base notebook', nb['test_wavs'] == nn['test_wavs'])
check('recording_id groups the real dataset into 28 tapes',
      len({nn['recording_id'](w) for w in CLIPS}) == 28,
      f"{len({nn['recording_id'](w) for w in CLIPS})} groups")
check('train / val / test are disjoint',
      not (set(nn['train_wavs']) & set(nn['test_wavs']))
      and not (set(nn['train_wavs']) & set(nn['val_wavs']))
      and not (set(nn['val_wavs']) & set(nn['test_wavs'])))
tr = {nn['recording_id'](w) for w in nn['train_wavs']}
leak = sum(nn['recording_id'](w) in tr for w in nn['test_wavs'])
check('leakage check still reports 145/145 (not broken by the rewrite)', leak == 145, f'{leak}/145')

# --- class priors -------------------------------------------------------------------------
# SCOPE DECISION, and these five checks previously asserted the opposite of it.
# They used to require the cell to publish DET_EVAL_PRIOR / CLS_EVAL_PRIOR /
# DET_EFFECTIVE_TRAIN_PRIOR, i.e. Space Bunny's prior-corrected gate. The combined
# notebook deliberately does NOT port that gate: with a validation-tuned threshold,
# rescaling P(call) by a constant prior ratio is monotone, so the tuned gate makes the same
# decisions (plan 2026-10-06 Task 8 "prior-corrected gate: not taken"; AGENTS.md 11.1). The
# ported cell therefore cannot define those names, and a check demanding them can never pass.
#
# So the DIAGNOSIS is kept and re-anchored to the cell's own split: the 4.18x shift in odds is
# a property of the split the cell produces, and is computed here FROM that split rather than
# read from a global. Those checks are still falsifiable -- re-split the data and they fail.
# What is asserted about the design is the deliberate absence of the correction globals, so a
# later edit that silently reintroduces the gate (and with it a fixed test-mix prior baked
# into the threshold) is caught rather than passing unnoticed.
PRIOR_GLOBALS = ('DET_EVAL_PRIOR', 'CLS_EVAL_PRIOR', 'DET_EFFECTIVE_TRAIN_PRIOR',
                 'CLS_EFFECTIVE_TRAIN_PRIOR', 'prior_correct_probs')
_present = [g for g in PRIOR_GLOBALS if g in nn]
check('the cell publishes NO prior-correction global (deliberate scope decision)',
      not _present,
      f'unexpected: {_present}' if _present else 'none of ' + ', '.join(PRIOR_GLOBALS))

_test = nn['test_wavs']
_n_noise = sum(1 for w in _test if gcls(w) == 'noise')
_n_call = len(_test) - _n_noise
_dp = np.array([_n_noise / len(_test), _n_call / len(_test)])
check('the test split really is 28 noise / 117 call, the eval prior the diagnosis rests on',
      (_n_noise, _n_call) == (28, 117), f'{_n_noise}/{_n_call} of {len(_test)}')
# WeightedRandomSampler draws each class with probability 1/n_classes, so the TRAIN prior is
# uniform whatever the sampler is fed. That is what makes the odds ratio below the reported 4.18.
_dtp = np.array([0.5, 0.5])
_shift = (_dp[1] / _dtp[1]) / (_dp[0] / _dtp[0])
check('prior shift factor ~4.2 in odds (train 50/50 under the sampler vs the test 19/81)',
      abs(_shift - 4.2) < 0.1, f'{_shift:.2f}x')
# What the cell DOES report instead of the prior correction: a no-model tape-lookup baseline.
# This is the measurement that says how much of the split is explained by the recording alone,
# which is the question the prior correction was answering indirectly. Falsifiable: the numbers
# must match an independent recomputation from the split, not merely be printed.
_tl = nn['_tape_lookup_baseline']
_rc = nn['_rec_of']
_by = {}
for w in nn['train_wavs']:
    _by.setdefault(_rc[w], Counter())[gcls(w)] += 1
_ys = [gcls(w) for w in _test if _rc[w] in _by]
_ps = [_by[_rc[w]].most_common(1)[0][0] for w in _test if _rc[w] in _by]
_ref_bal = float(np.mean([np.mean([p == c for y, p in zip(_ys, _ps) if y == c])
                          for c in sorted(set(_ys))]))
check('the tape-lookup baseline reports the recording-only accuracy this split allows',
      'TAPE-LOOKUP BASELINE' in out_MERGED and len(_ys) == 145
      and f'balanced accuracy {_ref_bal:.3f}' in out_MERGED,
      f'{len(_ys)}/145 clips share a recording with train, reference balanced acc {_ref_bal:.3f}')
check('that baseline is far ABOVE chance, i.e. the clip-level split does leak the recording',
      _ref_bal > 0.5, f'{_ref_bal:.3f} vs 1/8 chance')

cmap = nn.get('CLS_CLASS_TO_IDX')
# The per-species test counts the classifier's own eval prior would have held. Asserted
# against the split, so a re-split that changes a class share fails here.
_want = {'acsh': 24, 'alte': 27, 'noise': 28, 'rhro': 14}
_got = {c: sum(1 for w in _test if gcls(w) == c) for c in _want}
check('the classifier test distribution matches the reported one (acsh 24 / alte 27 / '
      'noise 28 / rhro 14)',
      cmap is not None and all(c in cmap for c in _want) and _got == _want,
      f'got {_got}' if _got != _want else '4 classes checked')
check('CLS_CLASS_TO_IDX covers all 8 species',
      cmap is not None and len(cmap) == 8,
      'CLS_CLASS_TO_IDX missing from the cell' if cmap is None else f'{len(cmap)} entries')
check('CLS_CONFIG num_classes still auto-set to 8', nn['CLS_CONFIG']['num_classes'] == 8,
      str(nn['CLS_CONFIG']['num_classes']))
check('dead DET_CLASS_TO_IDX no longer defined', 'DET_CLASS_TO_IDX' not in nn,
      'Cell 11 builds the real map; this dict is never read')
check('per-class recording counts printed',
      'single recording' in out_MERGED and 'heti' in out_MERGED)

# The unusable-clip threshold is derived from the config, not typed in. These two checks are
# what stop the fixtures above from quietly testing a different threshold than the notebook
# uses: cell6_ns takes DET_CONFIG/CLS_CONFIG out of the delivered config cell, so a config
# change flows through. Asserting only "a number exists" would pass for any constant.
_MIN = nn.get('_MIN_CLIP_S')
_cfg_max = max(nn['DET_CONFIG']['n_fft'] / nn['DET_CONFIG']['sr'],
               nn['CLS_CONFIG']['n_fft'] / nn['CLS_CONFIG']['sr'])
check('_MIN_CLIP_S equals max(n_fft/sr) over BOTH configs',
      _MIN is not None and abs(_MIN - _cfg_max) < 1e-12,
      f'cell {_MIN}, config {_cfg_max}' if _MIN is not None else 'not defined')
# And a hard-coded constant would NOT satisfy that, which is what makes the check falsifiable.
check('a plausible hard-coded threshold would fail the check above (self-check on the oracle)',
      abs(1e-3 - _cfg_max) > 1e-6, f'1e-3 vs {_cfg_max}')

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)
