#!/usr/bin/env python3
"""The data-discovery cell on the REAL dataset: split, recording_id, leakage check, priors.

Runs the cell exactly as the notebook runs it (only its config globals are supplied),
so `train_wavs` / `test_wavs` / `recording_id` / the priors are the artifact's, not copies.

Checks 1-2 compare against the read-only base parent, purely as a regression guard: a port
that silently re-split the data would be invisible everywhere else. While the merged cell is
byte-identical to the base those two are trivially true, which the run says out loud.

Run: venv/bin/python notebook_build/tests/test_cell7.py
"""
import contextlib
import glob
import io
import os
import re
import sys
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

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


def run(cells, i):
    """Exec cell i from `cells` with the config globals it reads; return (namespace, output)."""
    ns = {'os': os, 'np': np, 'Counter': Counter, 'train_test_split': train_test_split,
          're': __import__('re'),
          'DATA_DIR': f'{extract_cells.REPO}/Data/final_dataset/data',
          'glob': glob, 'get_class_from_filename': gcls, 'SPLIT_BY_RECORDING': False,
          'DET_CONFIG': {}, 'CLS_CONFIG': {'num_classes': None}}
    path = cells[i]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    return ns, buf.getvalue()


idx = extract_cells.merged_cells()
nn, out_MERGED = run(idx, CELL_DATA)

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
nb, _ = run(base, CELL_DATA)

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

# --- class priors: the detector trains under a uniform sampler prior against an 19/81 eval
# prior, a 4.18x shift in odds, which is why its val-tuned thresholds sit off 0.5. -----------
dp = nn.get('DET_EVAL_PRIOR')
check('detector eval prior matches the observed 28/117 split',
      dp is not None and abs(dp[0] - 28 / 145) < 1e-9 and abs(dp[1] - 117 / 145) < 1e-9,
      'DET_EVAL_PRIOR missing from the cell' if dp is None
      else f'noise={dp[0]:.4f} call={dp[1]:.4f}')
dtp = nn.get('DET_EFFECTIVE_TRAIN_PRIOR')
check('detector TRAIN prior is uniform (the sampler overrides it)',
      dtp is not None and np.allclose(dtp, 0.5),
      'DET_EFFECTIVE_TRAIN_PRIOR missing' if dtp is None else str(dtp))
check('prior shift factor ~4.2 in odds (matches the reported diagnosis)',
      dp is not None and dtp is not None
      and abs((dp[1] / dtp[1]) / (dp[0] / dtp[0]) - 4.2) < 0.1,
      'needs both priors' if (dp is None or dtp is None)
      else f'{(dp[1] / dtp[1]) / (dp[0] / dtp[0]):.2f}x')
cp = nn.get('CLS_EVAL_PRIOR')
check('classifier eval prior sums to 1', cp is not None and abs(cp.sum() - 1.0) < 1e-9,
      'CLS_EVAL_PRIOR missing from the cell' if cp is None else f'sum={cp.sum():.9f}')
cmap = nn.get('CLS_CLASS_TO_IDX')
_want = [('acsh', 24 / 145), ('alte', 27 / 145), ('noise', 28 / 145), ('rhro', 14 / 145)]
_missing = [c for c, _ in _want if not cmap or c not in cmap]
check('classifier prior matches the reported test distribution',
      cp is not None and not _missing
      and all(abs(cp[cmap[c]] - v) < 1e-9 for c, v in _want),
      'CLS_EVAL_PRIOR missing from the cell' if cp is None
      else f'CLS_CLASS_TO_IDX lacks {_missing}' if _missing else 'mismatch')
check('CLS_CLASS_TO_IDX covers all 8 species',
      cmap is not None and len(cmap) == 8,
      'CLS_CLASS_TO_IDX missing from the cell' if cmap is None else f'{len(cmap)} entries')
check('CLS_CONFIG num_classes still auto-set to 8', nn['CLS_CONFIG']['num_classes'] == 8,
      str(nn['CLS_CONFIG']['num_classes']))
check('dead DET_CLASS_TO_IDX no longer defined', 'DET_CLASS_TO_IDX' not in nn,
      'Cell 11 builds the real map; this dict is never read')
check('per-class recording counts printed',
      'single recording' in out_MERGED and 'heti' in out_MERGED)

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)
