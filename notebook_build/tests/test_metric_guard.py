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
import _trainfix    # pulls in torch; only the execution block below needs it
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

# The old form was `val_score = val_bal if SELECT_METRIC == 'balanced_accuracy' else val_acc`:
# a ternary on SELECT_METRIC that DECIDES the validation score. Scanning for the bare substring
# "SELECT_METRIC == 'balanced_accuracy'" is broader than that, and broader than the defect: the
# training cell also compares SELECT_METRIC to pick the printout constant that turns a score gap
# into "~how many validation clips", which decides nothing. So the scan is for the ASSIGNMENT
# FORM -- a conditional on SELECT_METRIC under which a validation-score name is assigned.
#
# Both node types are handled, because the bug has two spellings and catching only the ternary
# would be a silent hole: `x = a if C else b` is an IfExp, while `if C: x = a else: x = b` is an
# If whose body assigns. A first version of this check handled only the IfExp.
#
# Narrower than a substring scan, but not weaker for this defect: reintroduce the original line
# and `val_score` matches on both counts, while the legitimate printout arithmetic does not.
_SCORE_TARGETS = {'val_score', 'val_acc', 'val_bal', 'best_score'}


def _test_mentions_metric(node):
    """True if this expression's test subtree compares SELECT_METRIC to anything."""
    return any(isinstance(c, ast.Compare)
               and any(isinstance(x, ast.Name) and x.id == 'SELECT_METRIC'
                       for x in ast.walk(c))
               for c in ast.walk(node))


def score_conditionals(src):
    """Validation-score names assigned inside a conditional on SELECT_METRIC."""
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.If) and _test_mentions_metric(node.test):
            scopes = [node.body, node.orelse]
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp) \
                and _test_mentions_metric(node.value.test):
            scopes = [[node]]
        else:
            continue
        for scope in scopes:
            for stmt in scope:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if isinstance(t, ast.Name) and t.id in _SCORE_TARGETS:
                                out.add(t.id)
    return sorted(out)


# Self-test of the oracle, so "no conditional found" cannot mean "the scan does not work".
_ORACLE_CASES = [
    ("val_score = val_bal if SELECT_METRIC == 'balanced_accuracy' else val_acc",
     ['val_score'], 'the original ternary'),
    ("if SELECT_METRIC == 'accuracy':\n    val_score = val_acc\nelse:\n    val_score = val_bal",
     ['val_score'], 'the same bug as an if/else statement'),
    ("_clip = (x if SELECT_METRIC == 'balanced_accuracy' else y) if len(v) > 1 else None",
     [], 'a printout constant, not a score'),
    ("val_acc = accuracy_score(l, p)", [], 'an unconditional score'),
]
for _src, _want, _why in _ORACLE_CASES:
    check(f'the conditional scan detects {_why}', score_conditionals(_src) == _want,
          f'{score_conditionals(_src)} != {_want}')

_cond = score_conditionals(train_src)
check('no validation score is chosen by an if/else on SELECT_METRIC, in either the ternary '
      '(IfExp) or the statement (If) spelling', not _cond,
      f'score assigned inside a SELECT_METRIC conditional: {_cond}' if _cond
      else '0 score assignments under a SELECT_METRIC conditional')

# The scan above is the general defence; this one pins the specific legitimate survivor, so a
# reader who meets a comparison in the cell is not left deciding whether it is the bug. The
# training cell keeps ONE: the printout constant that turns a score gap into "~how many
# validation clips". Fail if a comparison appears anywhere else.
_lines = train_src.split('\n')
_cmp_lines = [i for i, ln in enumerate(_lines) if 'SELECT_METRIC ==' in ln]
_offending = [i + 1 for i in _cmp_lines
              if not any('_clip' in ln for ln in _lines[max(0, i - 1):i + 2])]
check('every SELECT_METRIC == comparison in the training cell is the clip-count printout, '
      'not a score choice', not _offending,
      f'unaccounted comparison(s) on line(s) {_offending}' if _offending
      else f'{len(_cmp_lines)} comparison(s), all the _clip printout')
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

# --- and by EXECUTION: the cell's own train_model, with a typo'd metric -----------------------------
# The checks above prove the expression is present and that it raises. They cannot prove the
# function REACHES it: a cell could keep the lookup as a dead statement while selecting on
# something else spelled differently, which is why this runs the real training loop.
#
# Same fixture as test_bug1 (notebook_build/tests/_trainfix.py) -- two hand-written datasets are
# how a suite ends up exercising something other than what its name claims; AGENTS.md 9.8 records
# a check that asserted against its own inline copy of the function it meant to test.
print('\n=== the cell\'s own train_model, executed with a typo\'d SELECT_METRIC ===')

if not os.path.exists(idx[CELL_TRAIN]):
    check('the training cell is present to execute', False, idx[CELL_TRAIN])
else:
    # Control: the same run with a VALID metric must complete. Without it, "raises KeyError"
    # could be true for the trivial reason that every configuration fails.
    _ctrl = _trainfix.base_cfg(n_epochs=2, epochs_per_eval=1)
    _ctrl_ok, _ctrl_detail = True, ''
    try:
        _trainfix.run_trial(idx[CELL_TRAIN], _ctrl)
    except Exception as e:                              # noqa: BLE001 - the point is that it must not
        _ctrl_ok, _ctrl_detail = False, f'{type(e).__name__}: {e}'
    check('a VALID SELECT_METRIC trains to completion (the control run)', _ctrl_ok, _ctrl_detail)

    _typo = _trainfix.base_cfg(n_epochs=2, epochs_per_eval=1)
    _raised, _why = False, 'no exception raised'
    try:
        _trainfix.run_trial(idx[CELL_TRAIN], _typo, extra={'SELECT_METRIC': 'balanced_acc'})
    except KeyError as e:
        _raised = True
        _why = f'KeyError({e})'
    except Exception as e:                              # noqa: BLE001 - any other error is a failure
        _why = f'{type(e).__name__} instead of KeyError: {e}'
    check("train_model raises KeyError when SELECT_METRIC is 'balanced_acc', rather than "
          'silently training on plain accuracy', _raised, _why)
    # The message must name the rejected metric, not some unrelated missing key: a KeyError from
    # elsewhere in the function would otherwise satisfy the check above.
    check('   ...and the KeyError names the bad metric, not an unrelated missing key',
          _raised and "'balanced_acc'" in _why, _why)

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)
