#!/usr/bin/env python3
"""Shared fixture for the two inference suites (test_resample.py, test_inference.py).

The four inference cells cannot be exec'd bare. Five different reasons, one per cell, and both
suites need the same answers:

  * cell 20 reads the trained models (`det_results` / `cls_model`) OR exported `.pk` files, moves
    them onto DEVICE and prints the combo it chose;
  * cell 21 is pure definitions plus one print, but its functions read audio through `soundfile`
    and resample it with `resampy`, and they call the FRONT END (`clip_to_db_spectrogram`,
    `minmax_normalize`, `eval_window_starts`, ...) and `forward_with_embedding`, which live in
    earlier cells;
  * cell 22 globs a folder, `sf.info`s every file in it and runs the detector over each;
  * cell 23 reads the variables cell 22 wrote, so it must share cell 22's namespace.

WHAT IS STUBBED, AND WHY
`load_model_from_pk` is the only substitution, and it is bounded. What is under test in cell 20
is `_model_from_pk` -- whether it accepts a LIST as an ensemble, whether it orders class names by
OUTPUT INDEX rather than by name, and whether it refuses a second `.pk` whose classes differ --
and all three are decided by the `classes` dict and the model width, which the stub supplies
exactly. Building a real 11 M-parameter `.pk` per case would add ~45 MB and seconds per check
while testing `torch.load` instead.

Everything else here is the real thing:
  * the models are built from the REAL architecture cell (`_assemble` + `Classifier` + the linear
    head), so `model[1].linear.out_features`, `.to(DEVICE).eval()`, `_layer_output` and
    `SoftmaxEnsemble` all behave as they do after training -- only the encoder and the head
    weights are replaced, by a deterministic function of the window;
  * the front end, the transforms and the open-set scorers are the delivered cells' own code,
    pulled in by name with `extract_cells.cell_defs`;
  * the INFER_* constants (cell 20) and the WINDOW_*/UNKNOWN_* settings (cell 1) are read out of
    the DELIVERED cells by AST, so a check can never pass against a copy of a default that has
    since changed;
  * the audio is real WAV/FLAC files written by `soundfile`.
"""
import ast
import atexit
import contextlib
import glob as _glob
import io
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

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
CELL_ARCH, CELL_DATASET, CELL_SETTINGS = 4, 5, 1

TARGET_SR = 192000            # the detectors' rate
CLS_SR = 250000               # the classifier's rate
DET_DATA_OPTS = {'sr': TARGET_SR, 'n_fft': 256, 'hop_length': 128, 'n_freq_bins': 256,
                 'fmin': 1000, 'fmax': 95000}
CLS_DATA_OPTS = {'sr': CLS_SR, 'n_fft': 256, 'hop_length': 128, 'n_freq_bins': 256,
                 'fmin': 10000, 'fmax': 125000}
DET_CONFIG = dict(DET_DATA_OPTS, sequence_len=20)
CLS_CONFIG = dict(CLS_DATA_OPTS, sequence_len=20)
CLS_CLASSES = ['noise', 'rhro', 'rhle']       # deliberately has 'noise' FIRST in name order but
                                                # index 0 -- see the class-order checks

# The settings the inference cells read out of the config cell (1). Read from the delivered cell,
# never retyped: `classify_selections` scores the TEST_TOPK loudest windows of a selection under
# WINDOW_MODE / WINDOW_STRIDE, so a suite that ran it against hand-copied numbers would test a
# different protocol if the config ever changed.
SETTINGS = ('WINDOW_MODE', 'WINDOW_STRIDE', 'TRAIN_TOP_FRAC', 'TEST_TOPK', 'SELECT_METRIC',
            'UNKNOWN_DETECTION', 'UNKNOWN_KEEP_KNOWN', 'UNKNOWN_LABEL', 'UNKNOWN_METHOD')

_DEVNULL = open(os.devnull, 'w')
atexit.register(_DEVNULL.close)

# One WORKING_DIR per process, removed on exit. The inference cells write the detections file and
# the Raven tables under it, so it has to be a real directory -- and it must not be the repo.
_WORKING = None


def working_dir():
    global _WORKING
    if _WORKING is None:
        _WORKING = tempfile.mkdtemp(prefix='batspot-infer-')
        atexit.register(shutil.rmtree, _WORKING, True)
    return _WORKING


def quiet_tqdm(it, **kw):
    """tqdm to /dev/null: a progress bar is noise in a test log, and cell 22 drives it itself."""
    from tqdm import tqdm
    return tqdm(it, **{**kw, 'file': _DEVNULL})


# --- the delivered cells' own code ------------------------------------------------------------

def source_of(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _assignments_before(path, stop_at, wanted=None, ns=None):
    """Exec the top-level `NAME = <literal>` assignments of a cell, up to `stop_at`."""
    ns = {} if ns is None else ns
    body = ast.parse(open(path, encoding='utf-8').read()).body
    for node in body[:stop_at]:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            names = [t.id for t in node.targets]
            if wanted is not None and not any(n in wanted for n in names):
                continue
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), ns)
    return ns


def settings_ns(cells=None):
    """WINDOW_*/UNKNOWN_*/SELECT_METRIC AS THE DELIVERED CONFIG CELL WRITES THEM."""
    cells = cells if cells is not None else extract_cells.merged_cells()
    ns = _assignments_before(cells[CELL_SETTINGS], len(ast.parse(source_of(cells[CELL_SETTINGS])).body),
                             wanted=set(SETTINGS))
    missing = [n for n in SETTINGS if n not in ns]
    if missing:
        raise KeyError(f'cell {CELL_SETTINGS}: settings not bound: {missing}')
    return ns


def arch_ns(cells=None):
    """The architecture cell's real `_assemble`, `Classifier`, `SoftmaxEnsemble`, `unwrap_model`,
    `model_members` and `forward_with_embedding`."""
    cells = cells if cells is not None else extract_cells.merged_cells()
    return extract_cells.cell_defs(
        cells[CELL_ARCH],
        {'_assemble', 'Classifier', 'SoftmaxEnsemble', 'unwrap_model', 'model_members',
         'forward_with_embedding'},
        {'torch': torch, 'nn': nn, 'OrderedDict': __import__('collections').OrderedDict,
         'np': np, 'plt': None})


FRONT_END = ('clip_to_db_spectrogram', 'minmax_normalize', 'pad_window', 'window_scores',
             'eval_window_starts', 'apply_unknown', 'known_scores', 'load_unknown_model')


def front_ns(cells=None):
    """The dataset cell's front end and open-set scorers -- the functions cell 21 calls."""
    cells = cells if cells is not None else extract_cells.merged_cells()
    return extract_cells.cell_defs(
        cells[CELL_DATASET], set(FRONT_END),
        {'np': np, 'torch': torch, 'sf': sf, 'resampy': resampy, 'os': os})


def infer_constants(cells=None):
    """The INFER_* constants AS THE DELIVERED CELL 20 WRITES THEM.

    Every top-level assignment before `def _model_from_pk` -- i.e. the configuration block, and
    nothing that loads a model or prints. Read out of the delivered cell by exec'ing its AST
    rather than retyped here, so a check can never pass against a stale copy of a default.
    """
    cells = cells if cells is not None else extract_cells.merged_cells()
    path = cells[CELL_CONFIG]
    body = ast.parse(open(path, encoding='utf-8').read()).body
    cut = next(i for i, n in enumerate(body)
               if isinstance(n, ast.FunctionDef) and n.name == '_model_from_pk')
    ns = {'os': os, 'WORKING_DIR': working_dir()}
    got = []
    for node in body[:cut]:
        if isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id.startswith('INFER_') for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'cell20', 'exec'), ns)
            got += [t.id for t in node.targets]
    missing = [n for n in got if n not in ns]
    if missing:
        raise KeyError(f'cell {CELL_CONFIG}: constants not bound: {missing}')
    return ns, sorted(got)


# --- stub models: the real architecture, a deterministic encoder --------------------------------

def bin_of_hz(opts, hz):
    """The first bin of `clip_to_db_spectrogram`'s output that is at or above `hz`.

    The front end keeps FFT bins [lo, hi) and nearest-upsamples them to `n_freq_bins`, so output
    bin b shows FFT bin `lo + (b*(hi-lo))//n_freq_bins` (verified against torch's interpolate for
    both detector and classifier configs). Deriving the index from that -- rather than from a
    fmin..fmax linear map, which is what a 256-bin model looks like but is NOT what it is -- is
    what makes a "the model only sees 60-95 kHz" fixture statement true. With lo=1, hi=127 at
    192 kHz a linear map would put 60 kHz at bin 160, i.e. in the zero padding.
    """
    n_fft, sr = int(opts['n_fft']), int(opts['sr'])
    lo = int(max(0, np.floor(n_fft * opts['fmin'] / sr)))
    hi = int(min(n_fft - 1, np.ceil(n_fft * opts['fmax'] / sr)))
    n_bins = int(opts['n_freq_bins'])
    for b in range(n_bins):
        if (lo + (b * (hi - lo)) // n_bins) * sr / n_fft >= hz:
            return b
    return n_bins


class TopBandEncoder(nn.Module):
    """(B, 1, seq, bins) min-max-normalised window -> (B, 512, 1) feature map.

    The single feature is the CONTRAST between the band at or above `band_lo_hz` and everything
    below it: `max` over frames and bins inside the band, minus the same maximum outside it. It
    stands in for a trained encoder but is a real one in every respect the cells touch -- shape,
    dtype, `.to(device)`, `nn.Module`, and the head's `_layer_output` -- and it is deterministic,
    which is what makes a scan test reproducible.

    Three properties matter for the tests and all three are deliberate:
      * `max` over frames and bins, not a mean. A bat call is NARROWBAND at any instant (a
        20 ms window of a 70-85 kHz sweep lights ~2 of 256 bins), so a mean over the band would
        dilute it 60-fold and nothing would ever be detected.
      * contrast, not level. A window's min-max floor maps to 0, and spectral leakage puts ~5 % of
        the window's dB range into the bins of a call that is NOT in the band -- enough to saturate
        any level-based probe. Subtracting the out-of-band maximum is what makes "only the
        70-85 kHz sweep is detected" a statement about the BAND rather than about loudness.
      * band-limited by construction: a call below `band_lo_hz` scores 0.
    """

    def __init__(self, opts, band_lo_hz, scale=1.0):
        super().__init__()
        self.k = bin_of_hz(opts, band_lo_hz)
        self.scale = float(scale)

    def forward(self, x):
        f = x[:, 0].max(dim=1).values                 # (B, bins) loudest level per frame
        band = f[:, self.k:].max(dim=1).values
        below = f[:, :self.k].max(dim=1).values if self.k else torch.zeros_like(band)
        feat = torch.zeros(f.shape[0], 512, device=f.device, dtype=f.dtype)
        feat[:, 0] = self.scale * (band - below)
        return feat[:, :, None]


def stub_model(arch, opts, n_out, gains, bias, band_lo_hz):
    """A real `_assemble`d model whose head is a fixed linear probe on TopBandEncoder's feature.

    `gains[i]` scales the feature into logit i. The Classifier pools with `avg` over a size-1 axis
    and the head weight is `gains[i] * 512` (the encoder's width), so the logit comes out as
    `gains[i] * 512 * contrast + bias[i]` -- independent of random weights, which is what makes a
    scan test reproducible. Returns a model in `.eval()` mode.
    """
    enc, head = TopBandEncoder(opts, band_lo_hz), arch['Classifier'](
        {'input_channels': 512, 'pooling': 'avg', 'num_classes': n_out})
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.copy_(torch.tensor(bias, dtype=torch.float32))
        for i, g in enumerate(gains):
            head.linear.weight[i, 0] = g * 512.0
    model = arch['_assemble'](enc, head)
    return model.eval()


def detector_model(arch):
    """A 2-output detector that fires only when 60 kHz-95 kHz is the loudest part of the window:
    logit_call = 512 * contrast - 2, so P(call) is 0.12 on silence, < 1e-4 on a 40-50 kHz sweep
    and > 0.99 on a 70-85 kHz one."""
    return stub_model(arch, DET_DATA_OPTS, 2, [0.0, 1.0], [0.0, -2.0], band_lo_hz=60000)


def classifier_model(arch, names=None, tilt=0.0, tilt_class=None):
    """A classifier over CLS_CLASSES, ignoring everything below 40 kHz: silence -> 'noise', a
    high-frequency call -> 'rhro'. The gains are chosen by class NAME, so the probe follows
    CLS_CLASSES if the fixture ever changes it.

    `tilt` shifts ONE logit's bias -- the 'noise' class by default, which is the DOMINANT logit
    on a silent window, so the shift is visible in a probability of order 1 rather than in one of
    order 1e-6. It exists so two ensemble members of the same shape can DISAGREE: an ensemble
    check that averages two identical models cannot tell averaging from returning member 0, so the
    fixture makes the members differ. One logit of +-1 cannot change any argmax here (the classes
    are 6-10 logits apart), which the suites rely on.
    """
    names = list(names or CLS_CLASSES)
    gains, bias = [], []
    for n in names:
        if n == 'noise':
            gains.append(-1.0); bias.append(4.0)
        elif n == 'rhro':
            gains.append(1.0); bias.append(-2.0)
        else:
            gains.append(0.0); bias.append(-10.0)
    if tilt:
        which = names.index('noise') if (tilt_class is None and 'noise' in names) \
            else (-1 if tilt_class is None else tilt_class)
        bias[which] += float(tilt)
    return stub_model(arch, CLS_DATA_OPTS, len(names), gains, bias, band_lo_hz=40000)


# The classes dicts the stub loader hands back. `DET_CLASSES` is deliberately NOT in name order --
# `{'call': 0, 'noise': 1}` is the read-only base notebook's official detector layout, and a cell
# that read `probs[:, 1]` instead of looking the name up would then score 'noise' as 'call'.
DET_CLASSES = {'call': 0, 'noise': 1}


def stub_loader(table, arch=None):
    """`load_model_from_pk(path, device)` reading `{basename: (n_out, classes, dataOpts)}`.

    Keyed by basename because the cell is handed whatever path the operator typed, so the lookup
    must not depend on the directory it lives in. Each key gets its own model -- two ensemble
    members therefore differ in one logit's bias, so "the ensemble averages them" is observable
    rather than trivially true.
    """
    arch = arch if arch is not None else arch_ns()
    cache = {}
    tilts = {k: 1.0 * (i % 3 - 1) for i, k in enumerate(sorted(table))}

    def load(path, device):
        key = os.path.basename(path)
        if key not in table:
            raise FileNotFoundError(f'no stub .pk named {key!r}')
        n_out, classes, opts = table[key]
        if key not in cache:
            names = [k for k, _ in sorted(classes.items(), key=lambda kv: kv[1])]
            cache[key] = (classifier_model(arch, names, tilt=tilts[key]) if len(names) != 2
                          else detector_model(arch))
        return cache[key].to(device), dict(classes), dict(opts)

    load.tilts = tilts
    return load


def write_unknown_sidecar(path, members, method='msp', threshold=0.42, noise_idx=0, keep=0.95):
    """A real `.npz` for the real `load_unknown_model` to read. `method='msp'` keeps it independent
    of the embedding, so the guard being tested is the MEMBERSHIP check, not the scorer."""
    np.savez(path, threshold=threshold, noise_idx=noise_idx, keep=keep, method=method,
             members=np.array([os.path.basename(m) for m in members]))
    return path


# --- real audio on disk ----------------------------------------------------------------------

def write_wav(path, sr, n, seed=0, amp=0.1):
    """White noise, float32, exact -- white noise because it is the worst case for a sub-sample
    misalignment: the resampling filters pass up to ~95 % of the lower Nyquist, so a shifted grid
    decorrelates the output instead of hiding under a smooth waveform."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    x = (np.random.default_rng(seed).standard_normal(int(n)) * amp).astype(np.float32)
    sf.write(path, x, int(sr), subtype='FLOAT')
    return path


def write_time_expanded(path, src_path, factor):
    """`src_path` re-headered at 1/factor of its rate (what a time-expanded recorder writes).

    The samples are unchanged, so a scan with INFER_TIME_EXPANSION=factor must return the SAME
    times in seconds -- which is the only way to tell a time-expanded recording from a real one.
    """
    x, sr = sf.read(src_path, dtype='float32')
    new_sr = int(round(sr / factor))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x, new_sr, subtype='FLOAT')
    return path


def write_corrupt(path, nbytes=512):
    """Bytes with a .wav extension and no soundfile header: `sf.info` must raise on it."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'RIFF\x00\x00\x00\x00WAVEthis-is-not-a-wav-file' * (nbytes // 34 + 1))
    return path


def touch(path, data=b''):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    return path


# --- namespaces ------------------------------------------------------------------------------

def _base_ns(cells, **over):
    cfg, cfg_names = infer_constants(cells)
    ns = {'np': np, 'torch': torch, 'sf': sf, 'os': os, 'resampy': resampy,
          'DEVICE': torch.device('cpu'), 'WORKING_DIR': working_dir(),
          'Counter': Counter, 'defaultdict': defaultdict,
          'glob': _glob,          # the MODULE: cell 22 calls glob.glob(...) / glob.glob(..., **kw)
          'confusion_matrix': confusion_matrix, 'tqdm': quiet_tqdm}
    ns.update(cfg)
    ns.update(settings_ns(cells))
    ns.update(front_ns(cells))
    ns.update(over)
    return ns


def funcs_ns(cells=None, **over):
    """Exec the inference-functions cell (21) with everything it reads."""
    cells = cells if cells is not None else extract_cells.merged_cells()
    # The architecture cell's names stay in the namespace: `forward_with_embedding` resolves
    # `model_members` out of it at call time, and an unused extra name costs nothing.
    ns = _base_ns(cells, **over)
    ns.update(arch_ns(cells))
    path = cells[CELL_FUNCS]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    return ns


def _compiled_with_config_overrides(src, path, overrides):
    """`src` compiled with the listed INFER_* assignments replaced by their override values.

    Cell 20 re-assigns its own configuration at module level, so injecting an override into the
    namespace before exec'ing it would simply be overwritten -- the cell would then load the
    default models and the check would silently test the wrong branch. The assignment is
    therefore rewritten in the AST, and a name the cell does not define at all is prepended.
    Comments are lost by `ast.parse`, which costs nothing for an exec and keeps this honest: the
    bytes the suite runs are the cell's own code apart from the value of the setting under test.
    """
    tree = ast.parse(src)
    seen = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in overrides:
            node.value = ast.parse(repr(overrides[node.targets[0].id]), mode='eval').body
            seen.add(node.targets[0].id)
    pre = [ast.Assign(targets=[ast.Name(id=k, ctx=ast.Store())],
                      value=ast.parse(repr(v), mode='eval').body)
           for k, v in overrides.items() if k not in seen]
    tree.body = pre + tree.body
    ast.fix_missing_locations(tree)
    return compile(tree, path, 'exec')


def config_ns(cells=None, **over):
    """Exec the inference-config cell (20) -- the whole cell, model loading included.

    Any keyword beginning with `INFER_` overrides a configuration setting of the cell itself (see
    `_compiled_with_config_overrides`); every other keyword is injected into the namespace, which
    is how a suite chooses the branch: `load_model_from_pk` + the .pk settings for the
    skip-training path, or `det_results` / `cls_model` for the in-memory one.

    Returns (namespace, stdout).
    """
    cells = cells if cells is not None else extract_cells.merged_cells()
    overrides = {k: v for k, v in over.items() if k.startswith('INFER_')}
    injected = {k: v for k, v in over.items() if not k.startswith('INFER_')}
    ns = _base_ns(cells, **injected)
    ns.update(arch_ns(cells))
    # The class list is computed from the data in the discovery cell (cell 6), which cannot be
    # exec'd here, so the fixture supplies it. Cell 20 only reads its LENGTH and its ORDER.
    ns['CLS_CLASSES'] = list(CLS_CLASSES)
    ns['DET_CONFIG'] = dict(DET_CONFIG)
    ns['CLS_CONFIG'] = dict(CLS_CONFIG)
    path = cells[CELL_CONFIG]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(_compiled_with_config_overrides(source_of(path), path, overrides), ns)
    return ns, buf.getvalue()


def det_result(model, best_val_acc, test_acc=0.0, mic=None):
    """A `det_results[mic]` entry with exactly the keys cell 20 reads."""
    return {'model': model, 'best_val_acc': best_val_acc,
            'metrics': {'accuracy': test_acc, 'balanced_accuracy': test_acc}}


def run_ns(cells=None, **over):
    """Exec cells 21 and 22 in ONE namespace and return (namespace, stdout).

    Cell 22 reads `_time` (cell 21's `import time as _time`) and writes `_src`, `infer_wavs`,
    `infer_selections` and `infer_files`, which cell 23 then reads -- so the cells have to share a
    namespace or nothing downstream of them can be tested.
    """
    cells = cells if cells is not None else extract_cells.merged_cells()
    ns = funcs_ns(cells, **over)
    buf = io.StringIO()
    path = cells[CELL_RUN]
    with contextlib.redirect_stdout(buf):
        exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    return ns, buf.getvalue()


def truth_ns(cells, run_namespace, **over):
    """Exec the scoring cell (23) in the namespace cell 22 left behind. Returns (ns, its stdout)."""
    ns = dict(run_namespace)
    ns.update(over)
    buf = io.StringIO()
    path = cells[CELL_TRUTH]
    with contextlib.redirect_stdout(buf):
        exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    return ns, buf.getvalue()


def selection(begin, end, species, confidence=0.9, low=40000.0, high=60000.0, peak=50000.0,
              begin_clock='00:00:00.0000', end_clock='00:00:00.1000', best_guess=None,
              is_noise=False):
    """One selection dict shaped like the ones `process_recording` returns."""
    return {'begin': begin, 'end': end, 'species': species, 'confidence': confidence,
            'low': low, 'high': high, 'peak': peak, 'n': 2, 'p_max': confidence,
            'begin_clock': begin_clock, 'end_clock': end_clock,
            'best_guess': species if best_guess is None else best_guess, 'is_noise': is_noise}


__all__ = ['CELL_CONFIG', 'CELL_FUNCS', 'CELL_RUN', 'CELL_TRUTH', 'CELL_ARCH', 'CELL_DATASET',
           'CELL_SETTINGS', 'TARGET_SR', 'CLS_SR', 'DET_CONFIG', 'CLS_CONFIG', 'DET_DATA_OPTS',
           'CLS_DATA_OPTS', 'CLS_CLASSES', 'DET_CLASSES', 'arch_ns', 'front_ns', 'infer_constants',
           'settings_ns', 'bin_of_hz', 'detector_model', 'classifier_model', 'stub_model',
           'stub_loader', 'write_unknown_sidecar', 'write_wav', 'write_time_expanded',
           'write_corrupt', 'touch', 'funcs_ns', 'config_ns', 'det_result', 'run_ns', 'truth_ns',
           'selection', 'working_dir', 'quiet_tqdm', 'Counter', 'defaultdict',
           'confusion_matrix', 'np', 'sf']
