#!/usr/bin/env python3
"""B3: SELECT_METRIC can no longer silently fall back to plain accuracy.

The guard is *read out of* the delivered config cell rather than pasted here, and
so is the selection expression in the training cell. A pasted copy of the assert
would keep passing no matter what the notebook does, which is the mistake this
file exists to prevent.

`SELECT_METRIC` drives validation scoring, ReduceLROnPlateau and early stopping.
With the pre-fix if/else, a typo ('balanced_acc') made the cell fall back to
plain accuracy on an 81/19 split -- a constant "always predict call" predictor
scores 0.81 there -- while the log still printed the typo'd metric name.

Run: venv/bin/python notebook_build/tests/test_metric_guard.py
"""
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


def source(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


# The cell indices this suite tests, named so a reordering fails loudly here
# rather than silently testing some other cell.
CELL_CONFIG, CELL_TRAIN = 1, 8

idx = extract_cells.merged_cells()
UNDER_TEST = os.environ.get('NEW_CELLS') or extract_cells.MERGED

# --- the extraction is itself under test: these suites must read the artifact --------------
check('the extraction returns the config cell', CELL_CONFIG in idx,
      f'from {UNDER_TEST}')
check('extracted cell 1 really is the config cell',
      'SELECT_METRIC' in source(idx[CELL_CONFIG]), os.path.basename(idx[CELL_CONFIG]))
check('extracted cell 8 really is the training cell',
      'def train_model(' in source(idx[CELL_TRAIN]), os.path.basename(idx[CELL_TRAIN]))

# --- the guard, taken from the config cell itself -----------------------------------------
cfg_src = source(idx[CELL_CONFIG])
guard = None
for _node in ast.parse(cfg_src).body:
    if isinstance(_node, ast.Assert) and 'SELECT_METRIC' in ast.unparse(_node.test):
        guard = ast.unparse(_node)
GUARD = guard  # None when the config cell carries no such assert

for bad, should_raise in [('balanced_accuracy', False), ('accuracy', False),
                          ('balanced_acc', True), ('BALANCED_ACCURACY', True),
                          ('', True), ('f1', True), ('balanced-accuracy', True)]:
    if GUARD is None:
        check(f'SELECT_METRIC={bad!r} is {"REJECTED" if should_raise else "accepted"}',
              False, 'the config cell has no SELECT_METRIC assert to run')
        continue
    ns = {'SELECT_METRIC': bad}
    msg = ''
    try:
        exec(GUARD, ns)
        raised = False
    except AssertionError as e:
        raised, msg = True, str(e)
    check(f'SELECT_METRIC={bad!r} is {"REJECTED" if should_raise else "accepted"}',
          raised == should_raise, f'raised={raised}')
    if raised:
        check(f'   ...message names the bad value ({bad!r})', repr(bad) in msg)

# --- how train_model consumes it ----------------------------------------------------------
train_src = source(idx[CELL_TRAIN])
check('train_model selects the metric via a dict lookup',
      "{'balanced_accuracy': val_bal, 'accuracy': val_acc}[SELECT_METRIC]" in train_src)
_code_only = '\n'.join(ln.split('#')[0] for ln in train_src.split('\n'))
check('the old if/else fallback is gone from CODE (it survives only in the fix comment)',
      "SELECT_METRIC == 'balanced_accuracy'" not in _code_only)
# The dict lookup must be exercised, not merely recognised: run the cell's OWN selection
# expression with a typo'd metric and require a KeyError. A local dict literal would only
# prove CPython raises, and would pass even if the cell had reverted to an if/else.
SELECT = re.search(r"\{'balanced_accuracy':\s*\w+,\s*'accuracy':\s*\w+\}\[SELECT_METRIC\]",
                   train_src)
raised, detail = False, 'no {...}[SELECT_METRIC] expression found in the training cell'
if SELECT:
    expr = SELECT.group(0)
    try:
        eval(expr, {'SELECT_METRIC': 'balanced_acc'}, {'val_bal': 0.9, 'val_acc': 0.8})
    except KeyError:
        raised = True
        detail = f'KeyError from {expr!r}'
    else:
        detail = f'{expr!r} returned instead of raising'
check('the cell\'s own lookup expression raises KeyError on a typo, rather than degrading',
      raised, detail)

# The pre-fix suite asserted the base notebook still carried the fallback, to show the risk
# was real. There is no un-fixed notebook left to read, so the same intent is expressed as a
# sweep: the silent fallback must appear in NO cell of the delivered artifact, since a single
# reintroduced copy anywhere would reinstate the bug this guard exists to stop.
SILENT_FALLBACK = "if SELECT_METRIC == 'balanced_accuracy' else val_acc"
_offenders = sorted(i for i, p in idx.items()
                    if isinstance(i, int) and SILENT_FALLBACK in source(p))
check('the silent fallback appears in NO cell of the delivered notebook '
      '(the risk it addressed is still real)',
      not _offenders, f'found in cell(s) {_offenders}' if _offenders else '0 cells')

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)
