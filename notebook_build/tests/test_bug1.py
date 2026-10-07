#!/usr/bin/env python3
"""B1: train_model returns the BEST checkpoint, decided by execution rather than by reading.

train_model() from the delivered training cell is run on a tiny synthetic problem whose
validation score rises for a few epochs and then FALLS, so "best weights" and "last-epoch
weights" are genuinely different. Whatever train_model hands back is then re-scored:

  * re-scored == reported best_score  -> it returned the best checkpoint
  * the synthetic problem is only a valid test if the last validation really was worse

The pre-fix version gated both the snapshot and the restore on `if save_path:`, so with the
signature's own `save_path=None` default it returned the last epoch while reporting the best
score. There is no un-fixed notebook left to compare against, so the suite now runs BOTH
save_path modes and requires them to agree -- which is the same defect seen from the other side.

Run: venv/bin/python notebook_build/tests/test_bug1.py
"""
import atexit
import os
import re
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _trainfix
import extract_cells

CELL_TRAIN = 8   # Cell 9: Training Infrastructure

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


CFG = _trainfix.base_cfg()

idx = extract_cells.merged_cells()
_TMP = tempfile.mkdtemp(prefix='batspot-bug1-')
atexit.register(shutil.rmtree, _TMP, True)
CURVE = os.path.join(_TMP, '_curve_test.png')


def run(save_path, cfg=None, extra=None):
    """Train one trial; return (flat weights, best, history, re-scored validation score, stdout)."""
    t = _trainfix.run_trial(idx[CELL_TRAIN], cfg or CFG, save_path=save_path,
                            extra=extra, verbose=True)
    return t.weights, t.best, t.history, t.score, t.out


if os.path.exists(CURVE):
    os.remove(CURVE)
print(f'=== under test: {os.environ.get("NEW_CELLS") or extract_cells.MERGED} cell '
      f'{CELL_TRAIN} ({os.path.basename(idx[CELL_TRAIN])}) ===')

none_w, none_best, hist_none, none_score, none_out = run(None)
path_w, path_best, hist_path, path_score, _path_out = run(CURVE)
print(f'  save_path=None : reported best={none_best:.4f}  re-scored={none_score:.4f}')
print(f'  save_path=set  : reported best={path_best:.4f}  re-scored={path_score:.4f}')

check('with save_path=None the returned model MATCHES the reported best',
      abs(none_score - none_best) < 1e-9, f'{none_score:.4f} vs {none_best:.4f}')
check('both save_path modes return IDENTICAL weights', torch.allclose(none_w, path_w))

# `val_score` is the series the pre-fix gate could have desynchronised from best_score. The
# merged cell records the validation series as `val_bal_acc` today; the port renames it to
# `val_score` so the selection metric is explicit rather than implied by SELECT_METRIC.
VAL_MISSING = ('history has no "val_score" (the merged cell still records it as "val_bal_acc")')
val = hist_none.get('val_score')
check('the returned model is the best checkpoint, not the last epoch',
      val is not None and none_score > val[-1] + 1e-9,
      VAL_MISSING if val is None else f'{none_score:.4f} vs last-epoch {val[-1]:.4f}')
check('the synthetic problem is non-trivial (the last validation is worse than the best)',
      val is not None and val[-1] < none_best - 1e-9,
      VAL_MISSING if val is None else f'{val[-1]:.4f} < {none_best:.4f}')
check('the reported best equals the maximum of the recorded validation history',
      val is not None and abs(max(val) - none_best) < 1e-9,
      VAL_MISSING if val is None else f'max {max(val):.4f} vs {none_best:.4f}')

topk = hist_none.get('topk_mean')
check('history carries the top-k mean', topk is not None,
      'no "topk_mean" in history' if topk is None else f'topk_mean={topk:.4f}')
check('the top-k mean is below the argmax (selection optimism is visible)',
      # A SCALAR, not a per-epoch list: the cell publishes the mean of the top-k validation
      # scores of the whole run, and draws it as one axhline. An earlier version of this check
      # subscripted it as topk[-1] and only ever passed because the key was ABSENT -- it never
      # got far enough to notice. `isinstance(topk, float)` keeps that from drifting back.
      isinstance(topk, float) and topk < none_best,
      'no float "topk_mean" in history' if not isinstance(topk, float)
      else f'gap={none_best - topk:+.4f}')
check('best_epoch recorded', isinstance(hist_none.get('best_epoch'), int),
      f'epoch {hist_none.get("best_epoch")}')
check('the training-curve PNG actually exists',
      os.path.exists(CURVE),
      f'{os.path.getsize(CURVE) if os.path.exists(CURVE) else 0} bytes')
check('val_score history recorded', val is not None and len(val) > 0,
      VAL_MISSING if val is None else f'{len(val)} validations')
check('lr history recorded', val is not None and len(hist_none['lr']) == len(val),
      VAL_MISSING if val is None
      else f'{len(hist_none["lr"])} lr vs {len(val)} validations')

# --- the interface this cell is supposed to publish -------------------------------------------------
# The plan names `hist['best_score']` as what a caller compares the re-scored model against, and
# the cell's own selection print reads it back out of the history. Without it a caller has to
# trust the returned third value and cannot cross-check it against the run.
check("history carries best_score (the plan's interface for re-checking the returned model)",
      isinstance(hist_none.get('best_score'), float)
      and abs(hist_none['best_score'] - none_best) < 1e-12,
      f'returned {none_best!r}, history {hist_none.get("best_score")!r}')
check('history best_score is the max of the recorded val_score series',
      val is not None and isinstance(hist_none.get('best_score'), float)
      and abs(max(val) - hist_none['best_score']) < 1e-12,
      f'max {max(val):.4f} vs {hist_none.get("best_score")}' if val else VAL_MISSING)

# --- base_lr is ABSOLUTE, never scaled by the batch size --------------------------------------------
# The notebook's LRs are the paper's 1e-4 (detector) / 3e-4 (classifier); an earlier version
# multiplied by the batch size, which at 64 x 2 gradient accumulation was 128x too large and
# diverged. The cell reports the LR it actually used, so this reads the real number rather than
# the source text. batch_size is 8 here, so a scaled LR would be 8x off -- not confusable.
_reported = re.search(r'Effective LR:\s*([0-9.eE+-]+)', none_out)
check('the cell reported an effective LR at all (this check reads that line)',
      _reported is not None,
      '' if _reported is not None else 'no "Effective LR:" line in the run output')
if _reported:
    _lr = float(_reported.group(1))
    check('the effective LR is config["base_lr"] itself, not scaled by the batch size',
          abs(_lr - CFG['base_lr']) / CFG['base_lr'] < 1e-9,
          f'reported {_lr:.3e} vs base_lr {CFG["base_lr"]:.3e} (batch {CFG["batch_size"]})')

# --- exactly one class-balancing mechanism, observed by execution -----------------------------------
# WeightedRandomSampler already draws each class with probability 1/n_classes; adding
# CrossEntropyLoss class weights on top double-counts it, and on the detector that is a 4.18x
# noise penalty which collapses the weakest-pretrained variant to all-noise. So both halves are
# asserted: "no loss weights" alone would also be satisfied by dropping the sampler and losing
# class balance altogether, which is the other half of the same defect. The shims delegate to the
# real implementations, so the run being inspected is a real one.
_spy = _trainfix.Spy()
run(None, extra={'nn': _trainfix.NNShim(_spy),
                 'DataLoader': _trainfix.make_loader_spy(_spy)})
check('the cell built its loss without class weights (observed, not read)',
      _spy.loss_kwargs is not None and _spy.loss_kwargs.get('weight') is None,
      f'CrossEntropyLoss kwargs {_spy.loss_kwargs}')
check('the cell balanced the classes with the WeightedRandomSampler',
      _spy.loader_kwargs is not None and _spy.loader_kwargs.get('sampler') is not None,
      f'DataLoader kwargs sampler={_spy.loader_kwargs.get("sampler") if _spy.loader_kwargs else None}')

# --- early-stopping patience counts RAW epochs, not validation steps --------------------------------
# The pre-fix counter incremented once per validation while validation runs every
# epochs_per_eval epochs, so `early_stopping_patience_epochs: 20` silently meant 40 raw epochs.
# Run with epochs_per_eval=2 and a 6-raw-epoch patience: a validation-step counter needs 6
# validations, i.e. a 12-epoch gap; a raw-epoch counter stops at exactly 6.
RAW_CFG = _trainfix.base_cfg(epochs_per_eval=2, early_stopping_patience_epochs=6, n_epochs=40)
_raw_w, _raw_best, hist_raw, _raw_score, _raw_out = run(None, cfg=RAW_CFG)
_cap = hist_raw['val_epoch'][-1] < RAW_CFG['n_epochs']
check('the patience trial early-stopped before the epoch cap '
      '(so the counter below is actually exercised)', _cap,
      f'last validation epoch {hist_raw["val_epoch"][-1]} of {RAW_CFG["n_epochs"]}')
_gap = hist_raw['val_epoch'][-1] - hist_raw['best_epoch']
_step_counter_gap = (RAW_CFG['early_stopping_patience_epochs'] * RAW_CFG['epochs_per_eval'])
check('early stopping fired exactly early_stopping_patience_epochs RAW epochs after the best '
      f'(a validation-step counter would give {_step_counter_gap})',
      _gap == RAW_CFG['early_stopping_patience_epochs'],
      f'{_gap} raw epochs after epoch {hist_raw["best_epoch"]}, wanted '
      f'{RAW_CFG["early_stopping_patience_epochs"]}')

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + (': ' + ', '.join(fails) if fails else ''))
sys.exit(1 if fails else 0)
