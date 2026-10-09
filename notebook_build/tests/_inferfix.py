#!/usr/bin/env python3
"""Shared fixture for the two inference suites (test_resample.py, test_inference.py).

The inference cells (base cells 20-23) cannot be exec'd bare. Four different reasons, one per
cell, and both suites need the same answers:

  * cell 20 reads the trained models (`det_results` / `cls_model`) OR an exported `.pk`, moves the
    models onto `DEVICE` and prints the combo it chose;
  * cell 21 is pure definitions plus one print, but its `_resampled` reads real audio through
    `soundfile` and resamples with `resampy`;
  * cell 22 globs a folder, `sf.info`s every file in it and runs a detector over each;
  * cell 23 reads the variables cell 22 wrote, so it must share cell 22's namespace.

`load_model_from_pk` is stubbed here and that is a deliberate, bounded substitution. What is
under test in cell 20 is `_model_from_pk` -- whether it accepts a LIST as an ensemble, whether it
orders class names by OUTPUT INDEX rather than by name, and whether it rejects a second `.pk`
whose classes differ. All three are decided by the `classes` dict and the model width, which the
stub supplies exactly; building a real 11 M-parameter `.pk` per case would add ~45 MB and seconds
per check while testing `torch.load` instead. The real loader is exercised by the export
round-trip elsewhere.

Everything else here is the real thing: the real `SoftmaxEnsemble` and `unwrap_model` out of the
architecture cell, the real `load_unknown_model` out of the dataset cell, and real WAV files on
disk written with `soundfile`.
"""
import atexit
import contextlib
import io
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from glob import glob

import numpy as np
import resampy
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

# The cell indices this work covers, named so a reordering fails loudly in the suites rather than
# silently testing some other cell.
CELL_CONFIG, CELL_FUNCS, CELL_RUN, CELL_TRUTH = 20, 21, 22, 23
CELL_ARCH, CELL_DATASET = 4, 5

TARGET_SR = 192000            # the detectors' rate; the classifier's is 250 kHz
DET_DATA_OPTS = {'sr': TARGET_SR, 'n_fft': 256, 'hop_length': 128, 'n_freq_bins': 256,
                 'fmin': 1000, 'fmax': 95000}
CLS_DATA_OPTS = {'sr': 250000, 'n_fft': 256, 'hop_length': 128, 'n_freq_bins': 256,
                 'fmin': 10000, 'fmax': 125000}

# One WORKING_DIR per process, removed on exit. The inference cells write the detections file and
# the Raven tables under it, so it has to be a real directory -- and it must not be the repo.
_WORKING = None


def working_dir():
    global _WORKING
    if _WORKING is None:
        _WORKING = tempfile.mkdtemp(prefix='batspot-infer-')
        atexit.register(shutil.rmtree, _WORKING, True)
    return _WORKING


# --- stand-ins for a loaded .pk ---------------------------------------------------------------

class _Head(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.linear = nn.Linear(8, n_out)


class StubNet(nn.Module):
    """What `load_model_from_pk` hands back: `model[1].linear.out_features` gives the class count,
    and `.to(DEVICE).eval()` works because it is a real nn.Module."""

    def __init__(self, n_out):
        super().__init__()
        self[1] = _Head(n_out)

    def forward(self, x):
        return self[1].linear(x.flatten(1))


def stub_loader(table):
    """`load_model_from_pk(path, device)` reading `{basename: (n_out, classes, dataOpts)}`.

    Keyed by basename because the cell is handed whatever path the operator typed, and the
    lookup must not depend on the directory it lives in.
    """
    def load(path, device):
        try:
            n_out, classes, opts = table[os.path.basename(path)]
        except KeyError:
            raise FileNotFoundError(f'no stub .pk named {os.path.basename(path)!r}') from None
        return StubNet(n_out).to(device), dict(classes), dict(opts)
    return load


def write_unknown_sidecar(path, members, method='mahalanobis', threshold=0.42, noise_idx=0):
    """A real `.npz` for the real `load_unknown_model` to read."""
    np.savez(path, mu=np.zeros((3, 4)), prec=np.zeros((3, 4, 4)), method=method,
             threshold=threshold, keep=0.95, noise_idx=noise_idx,
             members=np.array([os.path.basename(m) for m in members]))
    return path


# --- real audio on disk ----------------------------------------------------------------------

def write_wav(path, sr, n, seed=0, amp=0.1):
    """White noise, float32, exact -- white noise because it is the worst case for a sub-sample
    misalignment: the resampling filters pass up to ~95 % of the lower Nyquist, so a shifted grid
    decorrelates the output instead of hiding under a smooth waveform."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = (np.random.default_rng(seed).standard_normal(int(n)) * amp).astype(np.float32)
    sf.write(path, x, int(sr), subtype='FLOAT')
    return path


def write_corrupt(path, nbytes=512):
    """Bytes with a .wav extension and no soundfile header: `sf.info` must raise on it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'RIFF\x00\x00\x00\x00WAVEthis-is-not-a-wav-file' * (nbytes // 34 + 1))
    return path


# --- namespaces ------------------------------------------------------------------------------

def funcs_ns(cells=None, **over):
    """Exec the inference-functions cell (21). Everything it reads at module level is here."""
    cells = cells if cells is not None else extract_cells.merged_cells()
    ns = {'np': np, 'torch': torch, 'resampy': resampy, 'sf': sf,
          'DEVICE': torch.device('cpu'), 'INFER_TIME_EXPANSION': 1,
          'UNKNOWN_LABEL': 'unknown'}
    ns.update(over)
    path = cells[CELL_FUNCS]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    return ns


def config_ns(cells=None, **over):
    """Exec the inference-config cell (20) and return its namespace.

    `INFER_PKS` is the `{basename: (n_out, classes, dataOpts)}` table the stub loader reads, so
    the cell resolves real `.pk` paths without any .pk existing.
    """
    cells = cells if cells is not None else extract_cells.merged_cells()
    arch = extract_cells.cell_defs(cells[CELL_ARCH], {'SoftmaxEnsemble', 'unwrap_model'},
                                   {'torch': torch, 'nn': nn})
    ns = {'np': np, 'torch': torch, 'nn': nn, 'os': os, 'DEVICE': torch.device('cpu'),
          'WORKING_DIR': working_dir(), 'SoftmaxEnsemble': arch['SoftmaxEnsemble'],
          'unwrap_model': arch['unwrap_model'], 'SELECT_METRIC': 'balanced_accuracy',
          'UNKNOWN_DETECTION': True, 'UNKNOWN_LABEL': 'unknown', 'UNKNOWN_MODEL': None,
          'DET_CONFIG': dict(DET_DATA_OPTS, sequence_len=20),
          'CLS_CONFIG': dict(CLS_DATA_OPTS, sequence_len=20),
          'INFER_PKS': {}}
    ns.update(over)
    path = cells[CELL_CONFIG]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    return ns


def run_ns(cells=None, **over):
    """Exec the inference-functions cell and then the run cell (21 then 22) in ONE namespace.

    Cell 22 reads `_time` (cell 21's `import time as _time`) and writes `_src`, `infer_files` and
    `infer_selections`, which cell 23 then reads -- so the three cells have to share a namespace
    or nothing downstream of them can be tested.
    """
    ns = funcs_ns(cells, **over)
    path = cells[CELL_RUN]
    exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    return ns


def truth_ns(cells, run_namespace, **over):
    """Exec the scoring cell (23) in the namespace the run cell left behind."""
    path = cells[CELL_TRUTH]
    ns = dict(run_namespace)
    ns.update(over)
    exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    return ns


# --- misc -------------------------------------------------------------------------------------

def selection(begin, end, species, confidence=0.9, low=40000.0, high=60000.0, peak=50000.0,
              begin_clock='00:00:00.0000', end_clock='00:00:00.1000', best_guess=None,
              is_noise=False):
    """One selection dict shaped like the ones `process_recording` returns."""
    return {'begin': begin, 'end': end, 'species': species, 'confidence': confidence,
            'low': low, 'high': high, 'peak': peak, 'n': 2, 'p_max': confidence,
            'begin_clock': begin_clock, 'end_clock': end_clock,
            'best_guess': species if best_guess is None else best_guess, 'is_noise': is_noise}


__all__ = ['CELL_CONFIG', 'CELL_FUNCS', 'CELL_RUN', 'CELL_TRUTH', 'CELL_ARCH', 'CELL_DATASET',
           'TARGET_SR', 'DET_DATA_OPTS', 'CLS_DATA_OPTS', 'StubNet', 'stub_loader',
           'write_unknown_sidecar', 'write_wav', 'write_corrupt', 'funcs_ns', 'config_ns',
           'run_ns', 'truth_ns', 'working_dir', 'selection', 'Counter', 'defaultdict', 'glob',
           'confusion_matrix', 'np', 'sf']