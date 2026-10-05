#!/usr/bin/env python3
"""B4 and B5, executed against the NOTEBOOK'S OWN CODE.

Both bugs were previously under-tested in two ways, both found by the red-green
check in redgreen.py:

  * B4 (recording_id) was tested against an inline COPY of the function written
    inside the test, not the function the notebook actually runs -- so reverting
    the notebook's fix changed nothing the test could see.
  * B5 (summary-table labelling) had NO assertion anywhere; it was only ever
    checked by reading.

Both are covered here by exec'ing cells extracted from the delivered notebook and
inspecting what they produce. The B5 check for reduced-denominator rows was
itself vacuous in the first version (`'clips' in tail or True`); it now asserts
on a row whose denominator really is reduced, which the merged cell must flag.

Run: venv/bin/python notebook_build/tests/test_b5_labels.py
"""
import glob
import io
import os
import sys
import contextlib
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

CELL_DATA, CELL_SUMMARY = 6, 17

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


idx = extract_cells.merged_cells()
REPO = extract_cells.REPO


def exec_cell(i, ns):
    path = idx[i]
    exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    return path


# =========================================================================================
# B4 -- run the notebook's Cell 7 and call ITS recording_id
# =========================================================================================
print('=== B4: recording_id, from the delivered data-discovery cell ===')
CLIPS = sorted(glob.glob(f'{REPO}/Data/final_dataset/data/*/*.wav'))
gcls = lambda f: os.path.basename(f).split('-', 1)[0]

ns7 = {'os': os, 'np': np, 'Counter': Counter, 'train_test_split': train_test_split,
       're': __import__('re'), 'DATA_DIR': f'{REPO}/Data/final_dataset/data',
       'glob': glob, 'get_class_from_filename': gcls, 'SPLIT_BY_RECORDING': False,
       'DET_CONFIG': {}, 'CLS_CONFIG': {'num_classes': None}}
_cell7 = exec_cell(CELL_DATA, ns7)
rid = ns7['recording_id']          # <-- the notebook's function, not a copy

six = 'acsh-bat_3379376_2026_20260525-192000_91577_92135.wav'
four = 'sasa-bat_20260429-192000_215755_215781.wav'          # what Cell 8 writes
check('6-field name -> tape id', rid(six) == '20260525-192000', rid(six))
check('4-field name -> tape id (the bug)', rid(four) == '20260429-192000', rid(four))
check('two clips from one tape group together',
      rid(four) == rid(four.replace('215755_215781', '999999_999999')))
check('different tapes stay apart', rid(four) != rid(six))
check('a name with no timestamp degrades to the stem (documented)',
      rid('weird-name.wav') == 'weird-name')
grp = {rid(w) for w in CLIPS}
check('real dataset still groups into 28 recordings', len(grp) == 28, f'{len(grp)} groups')
# Under the old positional parse a Cell 8 clip falls back to its own file name, i.e. one group per
# clip, so on a re-extracted dataset every clip is its own "recording" and the leakage check
# reports a false all-clear. A CONTRAST on that outcome, so the check depends on the artifact:
# what the old parse did to the group count, AND that the notebook does not do it. Measured on
# Cell 8-style output, because the positional parse works fine on the 6-field names.
_old = lambda p: (p[:-4].split('_')[3] if len(p[:-4].split('_')) >= 6 else p)
_c8a, _c8b = 'sasa-bat_20260429-192000_215755_215781.wav', 'sasa-bat_20260429-192000_999999_999999.wav'
check('on Cell 8 output the old parse gives one group per clip; the notebook does not',
      len({_old(_c8a), _old(_c8b)}) == 2 and len({rid(_c8a), rid(_c8b)}) == 1,
      f'old 2 groups / 2 clips vs notebook {len(grp)} groups on the real dataset')

# =========================================================================================
# B5 -- run the summary cell and inspect the table it prints
# =========================================================================================
print('\n=== B5: the results-summary table, from the delivered cell ===')

# Two full-split "official" rows plus one REDUCED-denominator row, and one fine-tuned row.
# The official classifier row is the signal-free one: it answers "call" for every clip, so its
# plain accuracy is exactly 117/145 = 0.8069 while its AUC is ~0.5.
NS, NS_REDUCED = 145, 90
y_bin = np.array([0] * 28 + [1] * 117)
p_call_sig = np.concatenate([np.linspace(0.05, 0.45, 28), np.linspace(0.55, 0.95, 117)])
p_call_dead = np.full(NS, 0.9)          # says "call" for everything -> acc 0.807, AUC ~0.5
y_red = np.array([0] * 17 + [1] * (NS_REDUCED - 17))
p_red = np.concatenate([np.linspace(0.05, 0.45, 17), np.linspace(0.55, 0.95, NS_REDUCED - 17)])

val_results = [
    {'name': 'official_detector_m03.pk', 'path': 'p3', 'kind': 'detector: call vs noise',
     'n_files': NS, 'target_names': ['noise', 'call'], 'labels': y_bin,
     'preds': (p_call_sig >= 0.5).astype(int), 'accuracy': 0.70, 'balanced_accuracy': 0.79,
     'auc': 0.88, 'confusion_matrix': np.zeros((2, 2), int)},
    {'name': 'official_classifier_m09.pk', 'path': 'pc', 'kind': 'classifier as noise-vs-call ONLY',
     'n_files': NS, 'target_names': ['noise', 'call'], 'labels': y_bin,
     'preds': np.ones(NS, int), 'accuracy': 0.8069, 'balanced_accuracy': 0.50, 'auc': 0.4924,
     'no_signal': True, 'confusion_matrix': np.zeros((2, 2), int)},
    {'name': 'official_detector_m11.pk', 'path': 'p11', 'kind': 'detector: call vs noise',
     'n_files': NS_REDUCED, 'target_names': ['noise', 'call'], 'labels': y_red,
     'preds': (p_red >= 0.5).astype(int), 'accuracy': 0.71, 'balanced_accuracy': 0.75,
     'auc': 0.86, 'confusion_matrix': np.zeros((2, 2), int)},
]
CLS_CLASSES = ['acsh', 'alte', 'noise']
det_results = {'m09': {'metrics': {'labels': y_bin, 'accuracy': 0.91, 'balanced_accuracy': 0.92,
                                   'probs': np.stack([1 - p_call_sig, p_call_sig], 1)}}}
ns18 = {'np': np, 'plt': plt, 'val_results': val_results, 'det_results': det_results,
        'test_wavs': CLIPS[:NS], 'CLS_CLASSES': CLS_CLASSES, 'AUC_NO_SIGNAL': 0.05,
        'roc_auc_score': roc_auc_score, 'model_tags': {'p3': 'official:detector/m03',
                                                       'pc': 'official:classifier/m09',
                                                       'p11': 'official:detector/m11'},
        'classification_report': classification_report,
        'confusion_matrix': confusion_matrix, 'accuracy_score': accuracy_score,
        'balanced_accuracy_score': balanced_accuracy_score}
buf = io.StringIO()
_cell17 = idx[CELL_SUMMARY]
with contextlib.redirect_stdout(buf):
    exec(compile(open(_cell17, encoding='utf-8').read(), _cell17, 'exec'), ns18)
out18 = buf.getvalue()
tail = out18[out18.index('OFFICIAL (zero-shot) vs FINE-TUNED'):]

print(f'  --- the table {os.path.basename(_cell17)} actually printed ---')
for ln in tail.split('\n'):
    if ln.strip():
        print('   |' + ln[:100])

summary_src = open(_cell17, encoding='utf-8').read()
check('table shows an n (denominator) column', '"n":>4' in summary_src
      and '|n:' not in tail and 'n  ' in tail)
check('table shows a Note/kind column', 'Note' in tail)
check('the signal-free row is marked NO SIGNAL', 'NO SIGNAL' in tail)
# Each official row must carry ITS OWN kind in the Note column. Three things this must not do:
#  * match the banner above the table, which mentions "call vs noise" regardless of the column;
#  * key on `name`, because Space Bunny's cell takes the label from `model_tags` instead;
#  * count the aggregate gain line, which also starts with "official".
# So a row is located by the identifier the cell actually prints -- the `model_tags` value, or
# the name minus ".pk" for the cell that predates that dict -- and the gain line matches neither.
_table_lines = tail.split('\n')
_ident = [(ns18['model_tags'].get(r['path'], ''), r['name'].replace('.pk', ''), r)
          for r in val_results]


def _row_for(tag, name):
    return next((ln for ln in _table_lines if (tag and tag in ln) or name in ln), None)


_matched = [(r, _row_for(tag, name)) for tag, name, r in _ident]
_no_row = [name for (_tag, name, _r), (_m, ln) in zip(_ident, _matched) if ln is None]
_no_kind = [name for (_tag, name, r), (_m, ln) in zip(_ident, _matched)
            if ln is not None and r['kind'] not in ln]
_kinds_ok = not _no_row and not _no_kind
_detail = []
if _no_row:
    _detail.append(f'no printed row for {_no_row}')
if _no_kind:
    _detail.append(f'kind missing from {_no_kind}')
check('each table row carries its own kind text in the Note column', _kinds_ok,
      '; '.join(_detail) if _detail
      else f'{len(_matched)}/{len(val_results)} rows, each with its kind')
check('no doubled "official official_"', 'official official_' not in tail)
check('balanced accuracy is quoted as the fair comparison',
      'BALANCED ACCURACY (the fair comparison)' in tail)
check('fine-tuned gain line printed', 'fine-tuned best' in tail)
# The row evaluated on 90 of the 145 test clips must not be printed as if it were a
# full-split row: its 90/145 denominator has to be visible in the table.
check('reduced-denominator rows are flagged',
      f'{NS_REDUCED}/{NS}' in tail, f'looking for {NS_REDUCED}/{NS}')

print('\n  --- full summary block, checking the per-model section too ---')
head = out18[:out18.index('OFFICIAL (zero-shot) vs FINE-TUNED')]
check('per-model section names the Mode', 'Mode:' in head)
check('per-model section prints Files evaluated', 'Files evaluated' in head)
check('the no-signal model gets an explicit warning in the per-model section',
      'NO USABLE SIGNAL' in head or 'no usable signal' in head.lower())

print('\n' + '=' * 66)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)
