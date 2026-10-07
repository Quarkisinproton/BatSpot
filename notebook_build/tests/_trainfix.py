#!/usr/bin/env python3
"""The synthetic problem the training cell is EXECUTED against, shared by two suites.

`train_model` cannot be exec'd bare: it builds a DataLoader, a loss and an optimizer, and calls
`predict_proba` and a plotting backend. Two suites execute it, and they share this fixture on
purpose:

  * test_bug1.py needs a validation curve that RISES and then FALLS, so that "the best weights"
    and "the last-epoch weights" are genuinely different things and the re-score is meaningful.
  * test_metric_guard.py needs only a run that reaches its first validation, so that a typo'd
    SELECT_METRIC has something to be applied to.

Two hand-written fixtures would be how a suite quietly stops exercising what its name claims --
AGENTS.md 9.8 records a check that asserted against its own inline copy of `recording_id` instead
of the notebook's, and passed whatever the notebook did. One fixture, two consumers, one thing
under test.

The dataset holds tensors, not audio, so `predict_proba` below is a stand-in with the contract the
real clip-level scorer has: `(probs, labels)` over the dataset's items.

`Spy` records how the cell built its loss and its loader, which is how the suite can observe the
WeightedRandomSampler / CrossEntropyLoss pairing by execution rather than by reading the source.
Both shims delegate to the real implementation, so the run that is inspected is a genuine one.
"""
import contextlib
import io
import os
import sys
from collections import Counter, namedtuple
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')   # the training cell writes a PNG; these suites run headless
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

N_CLASS, DIM, NTR, NVA = 3, 16, 24, 18

Trial = namedtuple('Trial', 'model history best val score weights out')

# The training cell's top-level globals it expects to already exist. Everything `train_model`
# reads at module level is here; `cell_defs` pulls in the rest by closure analysis, so this list
# is the test's own contract rather than a guess at what the cell happens to use today.
BASE_NS = dict(
    np=np, torch=torch, nn=nn, optim=optim, os=os, plt=plt, Counter=Counter, ceil=ceil,
    Dataset=Dataset, DataLoader=DataLoader, WeightedRandomSampler=WeightedRandomSampler,
    accuracy_score=accuracy_score, balanced_accuracy_score=balanced_accuracy_score,
    tqdm=lambda x, **k: x,
    SELECT_METRIC='balanced_accuracy', REPORT_TOPK_MEAN=3,
)


def base_cfg(**over):
    """A config train_model can run to completion on the synthetic problem.

    Deliberately small batch and epochs: several suites here execute a real 30-epoch training
    loop, and the fixture's job is to be fast, not realistic.
    """
    cfg = {'batch_size': 8, 'base_lr': 5e-3, 'n_epochs': 30, 'epochs_per_eval': 1,
           'grad_accum_steps': 1, 'use_multi_gpu': False, 'num_classes': N_CLASS,
           'beta1': 0.5, 'lr_decay_factor': 0.5, 'lr_patience_epochs': 3,
           'early_stopping_patience_epochs': 4}
    cfg.update(over)
    return cfg


def get_class_from_filename(f):
    """The cell's own class-from-filename convention: the part before the first '-'."""
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

    FakeDS holds tensors, not audio, so this is the minimal contract train_model needs.
    """
    model.eval()
    with torch.no_grad():
        out = torch.softmax(model(dataset.x), dim=1).numpy()
    return out, dataset.y.numpy()


class Spy:
    """Records the kwargs the training cell passed to CrossEntropyLoss and DataLoader."""

    def __init__(self):
        self.loss_kwargs = None
        self.loader_kwargs = None


class NNShim:
    """`nn` with CrossEntropyLoss recorded; every other attribute is the real one."""

    def __init__(self, spy):
        self._spy = spy

    def __getattr__(self, name):
        return getattr(nn, name)

    def CrossEntropyLoss(self, *a, **k):
        self._spy.loss_kwargs = k
        return nn.CrossEntropyLoss(*a, **k)


_REAL_LOADER = DataLoader


def make_loader_spy(spy):
    """A DataLoader stand-in that records its kwargs, then builds the real loader.

    `_REAL_LOADER` rather than `DataLoader`: the shim is itself named DataLoader, so referring
    to the global by its own name inside would resolve to the shim and recurse forever.
    """
    def DataLoader(*a, **k):
        spy.loader_kwargs = k
        return _REAL_LOADER(*a, **k)
    return DataLoader


def load_train_model(path, extra=None):
    """Exec the training cell's train_model with the globals it expects.

    Only `train_model` is requested: cell_defs pulls in `make_weighted_sampler`,
    `compute_class_weights` and anything else it reads, by closure analysis. Whitelisting those
    names by hand would work only while the cell happened to define exactly those; one added
    helper would have been a NameError deep inside a 30-epoch run.

    `extra` overrides namespace entries -- how a suite substitutes a spy, or a typo'd
    SELECT_METRIC.
    """
    ns = dict(BASE_NS)
    ns.update(get_class_from_filename=get_class_from_filename, predict_proba=predict_proba)
    if extra:
        ns.update(extra)
    return extract_cells.cell_defs(path, {'train_model'}, ns)['train_model']


def rescore(model, val):
    model.eval()
    with torch.no_grad():
        return balanced_accuracy_score(val.y.numpy(), model(val.x).argmax(1).numpy())


def flat_weights(model):
    """Every parameter flattened into one vector, for an exact weights comparison."""
    return torch.cat([v.flatten() for v in model.state_dict().values()]).detach().clone()


def run_trial(path, cfg, save_path=None, extra=None, verbose=False):
    """Train one trial on the synthetic problem; return a Trial.

    `extra` entries are passed to `load_train_model`. Pass `{'nn': NNShim(spy)}` and
    `{'DataLoader': make_loader_spy(spy)}` to record how the cell built its loss and its loader.
    stdout is captured (it carries the effective LR the cell reports); `verbose` re-prints it so a
    suite's log keeps the shape an operator is used to reading.
    """
    train_model = load_train_model(path, extra)
    tr, va = FakeDS(NTR, 7), FakeDS(NVA, 99)
    torch.manual_seed(7)
    np.random.seed(7)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model, hist, best = train_model(make_model(), tr, va, cfg, torch.device('cpu'),
                                        save_path=save_path)
    if verbose:
        print(buf.getvalue())
    return Trial(model, hist, best, va, rescore(model, va), flat_weights(model), buf.getvalue())