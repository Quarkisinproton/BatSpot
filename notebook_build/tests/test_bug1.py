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
import shutil
import sys
import tempfile
from collections import Counter
from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

CELL_TRAIN = 8   # Cell 9: Training Infrastructure

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


N_CLASS, DIM, NTR, NVA = 3, 16, 24, 18
CFG = {'batch_size': 8, 'base_lr': 5e-3, 'n_epochs': 30, 'epochs_per_eval': 1,
       'grad_accum_steps': 1, 'use_multi_gpu': False, 'num_classes': N_CLASS,
       'beta1': 0.5, 'lr_decay_factor': 0.5, 'lr_patience_epochs': 3,
       'early_stopping_patience_epochs': 4}

idx = extract_cells.merged_cells()
_TMP = tempfile.mkdtemp(prefix='batspot-bug1-')
atexit.register(shutil.rmtree, _TMP, True)
CURVE = os.path.join(_TMP, '_curve_test.png')


def get_class_from_filename(f):
    return os.path.basename(f).split('-', 1)[0]


class FakeDS(Dataset):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, DIM, generator=g)
        self.y = torch.randint(0, N_CLASS, (n,), generator=g)
        self.file_names = [f'c{int(c)}-x_{i}.wav' for i, c in enumerate(self.y)]
        self.class_to_idx = {f'c{i}': i for i in range(N_CLASS)}
        self.labels = self.y.numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def make_model():
    torch.manual_seed(1234)
    return nn.Sequential(nn.Linear(DIM, 32), nn.ReLU(), nn.Linear(32, N_CLASS))


def predict_proba(model, dataset, device, batch_size=256):
    """Stand-in for the real clip-level scorer, which needs the full spectrogram front end.

    The synthetic FakeDS holds tensors, not audio, so this is the minimal contract train_model
    needs: (probs, labels) over the dataset's items.
    """
    model.eval()
    with torch.no_grad():
        out = torch.softmax(model(dataset.x), dim=1).numpy()
    return out, dataset.y.numpy()


def load_train_model(path):
    """Exec the cell's train_model with the globals it expects.

    Only `train_model` is requested: cell_defs pulls in `make_weighted_sampler`,
    `compute_class_weights` and anything else it reads, by closure analysis. An earlier version
    whitelisted those three names by hand, which worked only because the cell happened to
    define exactly those; one added helper would have been a NameError deep inside a 30-epoch
    training run.
    """
    ns = dict(np=np, torch=torch, nn=nn, optim=optim, os=os, plt=plt, Counter=Counter, ceil=ceil,
              Dataset=Dataset, DataLoader=DataLoader,
              WeightedRandomSampler=WeightedRandomSampler,
              accuracy_score=accuracy_score,
              balanced_accuracy_score=balanced_accuracy_score,
              get_class_from_filename=get_class_from_filename,
              SELECT_METRIC='balanced_accuracy', REPORT_TOPK_MEAN=3,
              tqdm=lambda x, **k: x,
              predict_proba=predict_proba)
    return extract_cells.cell_defs(path, {'train_model'}, ns)['train_model']


def rescore(model, val):
    model.eval()
    with torch.no_grad():
        return balanced_accuracy_score(val.y.numpy(), model(val.x).argmax(1).numpy())


def run(save_path):
    train_model = load_train_model(idx[CELL_TRAIN])
    tr, va = FakeDS(NTR, 7), FakeDS(NVA, 99)
    torch.manual_seed(7)
    np.random.seed(7)
    m = make_model()
    m, hist, best = train_model(m, tr, va, CFG, torch.device('cpu'), save_path=save_path)
    flat = torch.cat([v.flatten() for v in m.state_dict().values()]).detach().clone()
    return flat, best, hist, rescore(m, va)


if os.path.exists(CURVE):
    os.remove(CURVE)
print(f'=== under test: {os.environ.get("NEW_CELLS") or extract_cells.MERGED} cell '
      f'{CELL_TRAIN} ({os.path.basename(idx[CELL_TRAIN])}) ===')

none_w, none_best, hist_none, none_score = run(None)
path_w, path_best, hist_path, path_score = run(CURVE)
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
      'no "topk_mean" in history' if topk is None else f'topk_mean={topk[-1]:.4f}')
check('the top-k mean is below the argmax (selection optimism is visible)',
      topk is not None and len(topk) > 0 and topk[-1] < none_best,
      'no "topk_mean" in history' if not topk else f'gap={none_best - topk[-1]:+.4f}')
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

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + (': ' + ', '.join(fails) if fails else ''))
sys.exit(1 if fails else 0)
