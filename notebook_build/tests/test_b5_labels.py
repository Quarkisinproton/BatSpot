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


# =========================================================================================
# B4 -- run the notebook's Cell 7 and call ITS recording_id
# =========================================================================================
print('=== B4: recording_id, from the delivered data-discovery cell ===')
CLIPS = sorted(glob.glob(f'{REPO}/Data/final_dataset/data/*/*.wav'))
gcls = lambda f: os.path.basename(f).split('-', 1)[0]

ns7, _ = extract_cells.run_cell6(idx, CELL_DATA)
_cell7 = idx[CELL_DATA]
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
    # A fourth row for the mic that WAS fine-tuned. The cell pairs an official row with a
    # fine-tuned one by mic name to print the gain line, so without this the gain section is
    # empty and the two gain checks below would pass on nothing (or fail for the wrong reason).
    # Its official balanced accuracy 0.75 and fine-tuned 0.92 give a gain of +0.170, which the
    # checks verify against the fixture rather than merely looking for a line.
    {'name': 'official_detector_m09.pk', 'path': 'p9', 'kind': 'detector: call vs noise',
     'n_files': NS, 'target_names': ['noise', 'call'], 'labels': y_bin,
     'preds': (p_call_sig >= 0.5).astype(int), 'accuracy': 0.69, 'balanced_accuracy': 0.75,
     'auc': 0.85, 'confusion_matrix': np.zeros((2, 2), int)},
]
# The ported summary cell also prints a row for the fine-tuned CLASSIFIER scored as
# noise-vs-call, read off `cls_metrics`. Without it the cell raises before printing anything,
# so the fixture must carry it -- as the cell's own contract requires. Labels are class
# INDICES (the cell compares them against CLS_CLASS_TO_IDX['noise']), so they must be.
# 'noise' sits at index 2, NOT at index 1: the official 15-class .pk has it at 8 (AGENTS.md 9.1),
# and the point is that the cell must look it up by NAME. Every other column is a near-constant
# 0.01, so reading the wrong index (or hard-coding 1) gives P(call) ~ 0.99 for every clip --
# the "answers call for everything" row AGENTS.md 9.2 is about.
CLS_CLASSES = ['acsh', 'alte', 'noise']
CLS_CLASS_TO_IDX = {c: i for i, c in enumerate(CLS_CLASSES)}
det_results = {'m09': {'metrics': {'labels': y_bin, 'accuracy': 0.91, 'balanced_accuracy': 0.92,
                                   'probs': np.stack([1 - p_call_sig, p_call_sig], 1)}}}
# The fine-tuned classifier's OWN numbers, computed here by hand from `cls_metrics` so the
# checks below assert the fixture's arithmetic rather than a substring.
#
# The fixture is deliberately IMPERFECT. An earlier version separated the two classes perfectly,
# which made accuracy, balanced accuracy and AUC all exactly 1.0000 -- so the three printed
# columns were indistinguishable and the check read the ACCURACY column while claiming to test
# the AUC. Now:
#   * 2 of the 28 noise clips get a LOW P(noise) -> predicted "call"  (false alarms)
#   * 5 of the 117 call clips get a HIGH P(noise) -> predicted "noise" (misses)
# so the three numbers differ: accuracy is majority-weighted (call is 117/145 here, so it is the
# HIGHER of the two), balanced accuracy is the per-class mean, and the AUC ranks by confidence.
def _cls_row_truth():
    labels = cls_metrics['labels']
    y = (labels != CLS_CLASS_TO_IDX['noise']).astype(int)      # 0 = noise, 1 = call
    p = 1.0 - cls_metrics['probs'][:, CLS_CLASS_TO_IDX['noise']]
    return labels, y, p


_p_cls_noise = np.concatenate([np.linspace(0.55, 0.95, 28), np.linspace(0.05, 0.45, 117)])
# 2 noise clips look like calls, 5 call clips look like noise. Both errors are made with a
# probability close to 0.5 (an unsure model), which is what leaves the AUC high while dragging
# accuracy down -- the real shape of a model that is good but not perfect.
_p_cls_noise[[2, 11]] = 0.40                                  # noise clips, predicted "call"
_p_cls_noise[[30, 45, 60, 75, 100]] = 0.62                    # call clips, predicted "noise"
_noise_col = CLS_CLASS_TO_IDX['noise']
cls_metrics = {
    'labels': np.array([CLS_CLASS_TO_IDX['noise']] * 28 + [CLS_CLASS_TO_IDX['acsh']] * 117),
    'probs': np.stack([np.full(NS, 0.01), np.full(NS, 0.01), _p_cls_noise], 1),
}
assert _noise_col != 1, 'noise must not be at index 1, or the wrong-column read is invisible'
ns18 = {'np': np, 'plt': plt, 'val_results': val_results, 'det_results': det_results,
        'cls_metrics': cls_metrics, 'test_wavs': CLIPS[:NS], 'CLS_CLASSES': CLS_CLASSES,
        'CLS_CLASS_TO_IDX': CLS_CLASS_TO_IDX, 'AUC_NO_SIGNAL': 0.05,
        'roc_auc_score': roc_auc_score, 'model_tags': {'p3': 'official:detector/m03',
                                                       'pc': 'official:classifier/m09',
                                                       'p11': 'official:detector/m11',
                                                       'p9': 'official:detector/m09'},
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

# The header must name its columns, and each data row must carry a number under each. An
# earlier version asserted `'"n":>4' in summary_src` (grepping the SOURCE, not the output),
# `'|n:' not in tail` (no cell prints `n:`) and `'n  ' in tail` (a bare substring that a row's
# Note would satisfy). All three near-trivially true. What is actually asserted here is the
# rendered header and, below, a per-row column count.
_hdr = next((ln for ln in tail.split('\n')
             if ln.strip().startswith('Model')), '')
check('the header names n, Acc, Bal acc, AUC and Note',
      all(h in _hdr for h in ('n', 'Acc', 'Bal acc', 'AUC', 'Note')), _hdr.strip())

# --- locating a printed row -----------------------------------------------------------------
# Each official row must carry ITS OWN kind in the Note column. Three things this must not do:
#  * match the banner above the table, which mentions "call vs noise" regardless of the column;
#  * key on `name`, because Space Bunny's cell takes the label from `model_tags` instead;
#  * count the aggregate gain line, which also starts with "official".
# So a row is located by the identifier the cell actually prints, and the gain line matches
# neither. Two details of the ported cell's label have to be honoured, or the locator finds
# nothing and a check below fails for the wrong reason:
#  * it prints `official detector_m03` -- one underscore turned into a space -- which is the
#    B5 fix (the pre-fix cell prepended "official " to a name already starting "official_",
#    printing "official official_detector_m03"). Matching the raw name would never locate a row;
#  * a row flagged no_signal gets the NO SIGNAL note INSTEAD of its kind. That is the intended
#    reading (the point of the row is that its number means nothing), so such a row counts as
#    labelled if it carries NO SIGNAL.
_table_lines = tail.split('\n')


def _labels(tag, name):
    """Every spelling of this row's label the cell might print, for locating it."""
    return [s for s in (tag, name, name.replace('_', ' ', 1)) if s]


def _row_for(tag, name):
    return next((ln for ln in _table_lines
                 if any(s in ln for s in _labels(tag, name))), None)


def _row_of(r):
    return _row_for(ns18['model_tags'].get(r['path'], ''), r['name'].replace('.pk', ''))


# The NO SIGNAL marker must be ON the signal-free row, not anywhere in the section. The dead
# row is located the same way the kind check locates rows, and its marker asserted there; the
# three informative detectors are checked NOT to carry it. Asserting only "'NO SIGNAL' in tail"
# would pass if the marker were printed on any row at all, or in the legend.
_dead = next(r for r in val_results if r.get('no_signal'))
_dead_row = _row_of(_dead)
# Each row must carry FOUR numbers after its label (n, acc, bal, auc) and then a non-empty
# Note. This is the "Note/kind column" check done properly: the previous version asserted only
# that the word "Note" appeared somewhere in the section, which says nothing about any row.
def _note_for(r):
    """The Note text this row's fixture implies the cell should print for it: the NO SIGNAL
    marker for a signal-free row, else the row's own kind, plus the reduced-denominator
    suffix when the row was scored on fewer clips than the split has."""
    if r.get('no_signal'):
        return 'NO SIGNAL (AUC~0.5): plain accuracy just reflects the class mix'
    return r['kind'] + ('' if r['n_files'] == NS else f' (only {r["n_files"]} clips)')


def _label_for(r):
    """The label the cell prints for this row: the name minus '.pk', with the first underscore
    turned into a space (the B5 fix -- the pre-fix cell printed 'official official_...')."""
    return r['name'].replace('.pk', '').replace('_', ' ', 1)


def _numeric_cols(r, row):
    """The four numeric tokens between the row's label and its Note, or None if the row has
    the wrong shape.

    Sliced from the FRONT by the label's length: the row is `{label} {n} {acc} {bal} {auc} {note}`
    and both the label and the Note contain spaces, so the note's length cannot be used to find
    the numbers from the back (and a bare tail slice silently returns Note words as numbers)."""
    if row is None:
        return None
    toks = row.split()
    i = len(_label_for(r).split())
    return toks[i:i + 4] if len(toks) >= i + 4 else None


def _all_numeric(toks):
    """True iff every token parses as a float. A row's numbers must be numbers, not note words."""
    try:
        [float(t) for t in toks]
        return True
    except ValueError:
        return False


_bad_shape = []
for r in val_results:
    cols = _numeric_cols(r, _row_of(r))
    if cols is None or len(cols) != 4 or not _all_numeric(cols):
        _bad_shape.append((r['name'], cols))
check('every official row prints 4 NUMERIC columns between its label and its Note',
      not _bad_shape, str(_bad_shape) if _bad_shape
      else f'{len(val_results)} rows, each n / acc / bal / auc')
# And the n column must equal the row's own n_files -- a cell printing the total on every row
# would pass a "4 numbers" check while misrepresenting the reduced-denominator row.
_bad_n = []
for r in val_results:
    cols = _numeric_cols(r, _row_of(r))
    if cols is None:
        _bad_n.append((r['name'], 'row not found or malformed'))
    elif int(cols[0]) != r['n_files']:
        _bad_n.append((r['name'], f'printed {cols[0]}, fixture {r["n_files"]}'))
check("the n column on each row equals that row's own n_files",
      not _bad_n, str(_bad_n) if _bad_n
      else f'm03={NS}, classifier={NS}, m11={NS_REDUCED}, m09={NS} as the fixture holds')
check('the signal-free row itself is marked NO SIGNAL',
      _dead_row is not None and 'NO SIGNAL' in _dead_row,
      (_dead_row or 'row not found').strip())
_live_rows = [_row_of(r) for r in val_results if not r.get('no_signal')]
check('no informative row is marked NO SIGNAL',
      all(ln is not None and 'NO SIGNAL' not in ln for ln in _live_rows),
      str([ln.strip()[:60] for ln in _live_rows if ln and 'NO SIGNAL' in ln]))

# The fine-tuned CLASSIFIER row is new in the ported cell (the base cell had no such line),
# and the pre-fix report showed its noise-vs-call accuracy with no denominator or note. It
# must be present, on the full split, and marked as a classifier -- not as a detector.
_cls_row = next((ln for ln in _table_lines if 'fine-tuned classifier' in ln), None)
check('the fine-tuned classifier has its own call-vs-noise row',
      _cls_row is not None and f'{NS:>4}' in _cls_row,
      (_cls_row or 'row not found').strip())
check('that row is labelled as a classifier, not a detector',
      _cls_row is not None and 'classifier as noise-vs-call' in _cls_row
      and 'detector' not in _cls_row, (_cls_row or 'row not found').strip())
# The printed layout is `{label:<34} {n:>4} {acc:>7.4f} {bal:>8.4f} {auc:>7.4f}  {note}`, and both
# the label and the note contain spaces, so the numbers are located by the LABEL's length from
# the front -- not by counting back from the note, which silently returns note words instead.
_lbl, _y, _p = _cls_row_truth()
check("the classifier fixture's labels are its class INDICES, matching what the cell compares",
      np.array_equal(_lbl, np.array([CLS_CLASS_TO_IDX['noise']] * 28
                                    + [CLS_CLASS_TO_IDX['acsh']] * 117)),
      f'first={_lbl[0]}, 28th={_lbl[27]}, 29th={_lbl[28]}, last={_lbl[-1]} '
      f'(CLS_CLASS_TO_IDX={CLS_CLASS_TO_IDX})')
_want_acc = float(np.mean((_p >= 0.5) == _y))
_want_bal = float(balanced_accuracy_score(_y, (_p >= 0.5).astype(int)))
_want_auc = float(roc_auc_score(_y, _p))
check("the classifier row's three metrics differ, so the columns are distinguishable",
      len({_want_acc, _want_bal, _want_auc}) == 3,
      f'acc={_want_acc:.4f} bal={_want_bal:.4f} auc={_want_auc:.4f}')
# The ordering of the three numbers is itself a discriminator: with `call` the majority class,
# majority-weighted accuracy is the highest of the three, balanced accuracy lower, and the
# AUC (a ranking measure) in between. A cell printing the right values in the wrong COLUMNS
# would break this ordering, so it is asserted rather than left to the individual value checks.
check('accuracy > balanced accuracy > 0 and AUC > balanced accuracy (the fixture\'s ordering)',
      _want_acc > _want_bal and _want_auc > _want_bal,
      f'acc={_want_acc:.4f} bal={_want_bal:.4f} auc={_want_auc:.4f}')
_cls_cols = _numeric_cols({'name': 'fine-tuned classifier (noise?)', 'n_files': NS},
                          _cls_row)
_cls_floats = [float(c) for c in _cls_cols] if _cls_cols and len(_cls_cols) == 4 else None
check("the classifier row's ACC is the fixture's, so P(noise) was read by class NAME",
      _cls_floats is not None and abs(_cls_floats[1] - _want_acc) < 5e-5,
      f'printed {_cls_cols} vs expected acc={_want_acc:.4f}'
      + ('' if _cls_cols is None else f'; reading column 1 gives '
         f'{float(np.mean((cls_metrics["probs"][:, 1] >= 0.5) == _y)):.4f}'))
check("the classifier row's BALANCED ACC is the fixture's",
      _cls_floats is not None and abs(_cls_floats[2] - _want_bal) < 5e-5,
      f'printed {_cls_cols[2] if _cls_cols else None} vs expected {_want_bal:.4f}')
check("the classifier row's AUC is the fixture's (a separate quantity from accuracy)",
      _cls_floats is not None and abs(_cls_floats[3] - _want_auc) < 5e-5,
      f'printed auc={_cls_cols[3] if _cls_cols else None} vs expected {_want_auc:.4f} '
      f'(accuracy is {_want_acc:.4f})')

_ident = [(ns18['model_tags'].get(r['path'], ''), r['name'].replace('.pk', ''), r)
          for r in val_results]


_matched = [(r, _row_for(tag, name)) for tag, name, r in _ident]
_no_row = [name for (_tag, name, _r), (_m, ln) in zip(_ident, _matched) if ln is None]
_no_kind = [name for (_tag, name, r), (_m, ln) in zip(_ident, _matched)
            if ln is not None and r['kind'] not in ln and not r.get('no_signal')]
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
# The gain section must state the numbers the FIXTURE holds, not merely exist. The pre-fix
# cell printed only "official best X -> fine-tuned best Y"; the ported cell prints one line per
# mic, pairing the official row with the fine-tuned one by name. Both forms are accepted as
# long as the m09 gain equals the fixture's 0.750 -> 0.920 (+0.170), so a cell that printed the
# wrong pairing, or the same number twice, fails here.
_gain = [ln.strip() for ln in tail.split('\n') if 'm09:' in ln]
_want_gain = '0.750 -> 0.920  (+0.170)'
check('the fine-tuned gain line is printed with the fixture\'s own numbers',
      any(_want_gain in ln for ln in _gain),
      f'looking for {_want_gain!r} in {_gain}')
# The name says "above the gain lines", so the ORDER is asserted: the heading must be printed
# before the per-mic gain lines, not merely somewhere in the section. A cell that printed the
# heading last would still satisfy a bare "a line starts with balanced accuracy".
_lines = [ln.strip() for ln in tail.split('\n')]
_bal_i = next((i for i, ln in enumerate(_lines) if ln.lower().startswith('balanced accuracy')), -1)
_gain_i = next((i for i, ln in enumerate(_lines) if ln.startswith('m09:')), -1)
check('balanced accuracy is named as the fair comparison, ABOVE the gain lines',
      _bal_i >= 0 and _gain_i >= 0 and _bal_i < _gain_i,
      f'heading at line {_bal_i}, m09 gain at line {_gain_i}'
      if _bal_i >= 0 and _gain_i >= 0 else
      f'heading found={_bal_i >= 0}, gain line found={_gain_i >= 0}')
head = out18[:out18.index('OFFICIAL (zero-shot) vs FINE-TUNED')]
# The row evaluated on 90 of the 145 test clips must not be printed as if it were a
# full-split row. The ported cell flags it in the Note column as "(only 90 clips)" and repeats
# the count in the per-model section as "90 of 145 test clips"; both are asserted, because the
# Note column is what a reader scanning the table actually sees.
_red_row = _row_of(next(r for r in val_results if r['n_files'] == NS_REDUCED))
check('reduced-denominator rows are flagged in the Note column',
      _red_row is not None and f'only {NS_REDUCED} clips' in _red_row,
      (_red_row or 'row not found').strip())
check('the per-model section also shows the reduced denominator as n of total',
      f'Files evaluated: {NS_REDUCED} of {NS} test clips' in head, 'looking for '
      f'Files evaluated: {NS_REDUCED} of {NS} test clips')

print('\n  --- full summary block, checking the per-model section too ---')
check('per-model section names the Mode', 'Mode:' in head)
check('per-model section prints Files evaluated', 'Files evaluated' in head)
check('the no-signal model gets an explicit warning in the per-model section',
      'NO USABLE SIGNAL' in head or 'no usable signal' in head.lower())

print('\n' + '=' * 66)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)
