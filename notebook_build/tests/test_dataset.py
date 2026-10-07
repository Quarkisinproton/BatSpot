#!/usr/bin/env python3
"""Integration test of the windowed dataset cell on REAL audio.

Covers the paths a unit test cannot: cache build, cache REUSE, cache invalidation when the
source wav changes, directory-tagging of cache keys, the short-clip pad path, dropping unusable
clips, windowing counts, and clip-level predict_proba.

The cell is exec'd from the delivered notebook, so this is the cell that ships. The pre-fix
min-max/pad ORDER is reproduced locally as a *reference* (used only for attribution) rather than
by reading a second, un-fixed notebook: there is no un-fixed notebook left in this repo.

Run: venv/bin/python notebook_build/tests/test_dataset.py   (~40 s: builds real spectrograms)
"""
import atexit
import contextlib
import glob
import hashlib
import io
import os
import random
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import soundfile as sf
import resampy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

CELL_DATASET = 5   # Cell 6: Transforms & Dataset

fails = []
# One guard slot per section, set from whatever that section could not build. Checks that
# need those objects read them through check(..., guarded='mix'), so a definition the cell
# does not have yet is a reportable FAIL naming the cause instead of a traceback part-way
# through the section.
GUARD_ERRORS = {'mix': None, 'fp': None, 'emb': None, 'pp': None}


def check(name, cond, detail='', guarded=None):
    """One assertion.

    `guarded='mix'` names the guard slot. When that slot holds a reason, the check reports
    FAIL with it and neither `cond` nor `detail` is evaluated -- which is what keeps a
    missing definition from turning into a traceback after earlier checks have printed.
    NOTE: `guarded` only reaches arguments that are NOT yet evaluated, i.e. callables. A
    `cond` written as an eager expression is computed before check() is called, so it must
    short-circuit on its own (or be a lambda).
    """
    if guarded is not None and GUARD_ERRORS.get(guarded):
        cond, detail = False, GUARD_ERRORS[guarded]
    # A lambda that RAISES is a failing check, not a reason to abort the suite: without this
    # a renamed _layer_output key killed the run with a KeyError after the earlier checks had
    # printed, and nothing was reported at all. The exception becomes the FAIL detail -- and
    # note it must NOT try to render `detail` here, since that callable raises too.
    try:
        if callable(cond):
            cond = cond()
        if callable(detail):
            detail = detail()
    except Exception as e:
        cond, detail = False, f'{type(e).__name__}: {e}'
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


REPO = extract_cells.REPO
_TMP = tempfile.mkdtemp(prefix='batspot-dataset-')
atexit.register(shutil.rmtree, _TMP, True)

CLIPS = sorted(glob.glob(f'{REPO}/Data/final_dataset/data/*/*.wav'))[:12]


# species -> 'call', exactly as the dataset-building cell builds it (NOT a dead {'noise','call'} dict)
def _cls(f):
    return os.path.basename(f).split('-', 1)[0]


def map_for(files):
    """species -> index, derived from the filenames (mirrors what the cell builds)."""
    cs = sorted({_cls(f) for f in files})
    return {c: i for i, c in enumerate(cs)}


DET_MAP = {'noise': 0, **{_cls(f): 1 for f in CLIPS if _cls(f) != 'noise'}}
CFG = {'sr': 192000, 'n_fft': 256, 'hop_length': 128, 'n_freq_bins': 256,
       'fmin': 1000, 'fmax': 95000, 'sequence_len': 20}
SEQ = int(0.02 * CFG['sr'] / CFG['hop_length'])     # 30 frames


_DEVNULL = open(os.devnull, 'w')
atexit.register(_DEVNULL.close)


def _quiet_tqdm(it, **kw):
    """The cell's cache loop drives tqdm. A progress bar is noise in a test log, and the
    cache-reuse assertion reads the cell's stdout lines, which are captured separately."""
    return tqdm(it, **{**kw, 'file': _DEVNULL})


def load_cell(path, workdir):
    ns = dict(np=np, os=os, torch=torch, nn=nn, plt=plt, sf=sf, resampy=resampy,
              Dataset=Dataset, DataLoader=DataLoader, tqdm=_quiet_tqdm,
              WORKING_DIR=workdir, WINDOW_MODE='energy_crop', WINDOW_STRIDE=3,
              TRAIN_TOP_FRAC=0.20, TEST_TOPK=5)
    exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    return ns


def make_short_copy(src, dst, ms=8):
    """A deliberately sub-window clip, to exercise the pad path."""
    info = sf.info(src)
    n = int(info.samplerate * ms / 1000)
    data, sr = sf.read(src, frames=n, dtype='float32')
    sf.write(dst, data, sr)
    return dst


idx = extract_cells.merged_cells()
CELL = idx[CELL_DATASET]
WD = os.path.join(_TMP, 'wd')
os.makedirs(WD, exist_ok=True)
print(f'=== under test: {os.environ.get("NEW_CELLS") or extract_cells.MERGED} cell '
      f'{CELL_DATASET} ({os.path.basename(CELL)}) ===')
ns = load_cell(CELL, WD)
DS = ns['WindowedBatDataset']

d_train = DS(CLIPS, DET_MAP, CFG, train=True)
n_cached = len(glob.glob(os.path.join(WD, 'spec_cache', '*', '*.npy')))
check('cache built for every clip', n_cached == len(CLIPS), f'{n_cached}/{len(CLIPS)}')
check('every clip has a sidecar meta record',
      len(glob.glob(os.path.join(WD, 'spec_cache', '*', '*.meta.json'))) == len(CLIPS))
# The tag is a hash of the clip's absolute PATH. Two DATA_DIRs with colliding basenames must
# not share a cache entry, and -- since the ported cell tags per FILE rather than per folder
# -- neither must two clips that happen to share a basename inside one folder. Asserted
# exactly, because an underscore in the basename would satisfy a looser "is there a tag in
# the name" test without any tagging at all.
# (Ported expectation: the pre-port cell hashed the containing DIRECTORY and took 6 hex
# chars; the ported cell hashes each file's absolute path and takes 8. That is the strictly
# finer discriminator, and the check below is correspondingly stricter than the old one.)
check('cache filename carries a path tag, 8 hex chars, prepended to the name',
      all(re.fullmatch(r'[0-9a-f]{8}_.+\.npy', os.path.basename(d_train._spec_path(f)))
          for f in CLIPS),
      os.path.basename(d_train._spec_path(CLIPS[0])))
check('the tag is the md5 of that clip\'s absolute path',
      all(os.path.basename(d_train._spec_path(f))
          == f'{hashlib.md5(os.path.abspath(f).encode()).hexdigest()[:8]}_'
            f'{os.path.basename(f)}.npy' for f in CLIPS),
      os.path.basename(d_train._spec_path(CLIPS[0])))
# A folder-level tag (what the pre-port cell used) would give all 12 clips ONE tag here, so
# this is what would catch a regression to tagging the directory instead of the file.
_tags = {os.path.basename(d_train._spec_path(f)).split('_', 1)[0] for f in CLIPS}
check('every clip gets its OWN tag, even 12 clips from one folder',
      len(_tags) == len(CLIPS), f'{len(_tags)} distinct tags for {len(CLIPS)} clips')
# (Ported expectation: 'v2-' was the base cell's hand-written literal; the ported cell
# derives the suffix from the front-end's compiled code, so the prefix moved to 'v3-'.)
# What is asserted has not changed: the tag is DERIVED, and a hand-written 'v2' would fail.
check('cache version tag is derived from the front-end source, not hand-written',
      bool(re.fullmatch(r'v3-[0-9a-f]{6}', DS._CACHE_VERSION)), DS._CACHE_VERSION)

# Cache REUSE. The .npy count is unchanged whether the cell re-converts into the same
# cache_dir or not, so counting files proves nothing. Two things do distinguish the two: the
# cell's own "caching N clips" line, and the mtimes of the entries (a rebuild rewrites them).
_mtimes_before = {f: os.stat(d_train._spec_path(f)).st_mtime_ns for f in CLIPS}
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    DS(CLIPS, DET_MAP, CFG, train=True)
_reuse_out = _buf.getvalue()
check('second construction reuses the cache (the cell prints no "caching" line)',
      'caching' not in _reuse_out,
      f'cell said: {_reuse_out.strip()[:80]!r}' if _reuse_out.strip() else 'silent')
check('reusing the cache does not rewrite the entries',
      all(os.stat(d_train._spec_path(f)).st_mtime_ns == _mtimes_before[f] for f in CLIPS),
      f'{sum(os.stat(d_train._spec_path(f)).st_mtime_ns != _mtimes_before[f] for f in CLIPS)} '
      f'of {len(CLIPS)} rewritten')

# --- the stale-cache bug: touch a wav and confirm the entry is rebuilt -----------------------
# Run on a private copy, so the suite never mutates the mtime of a file in Data/.
_probe = os.path.join(_TMP, 'probe_clips')
os.makedirs(_probe, exist_ok=True)
victim = os.path.join(_probe, os.path.basename(CLIPS[0]))
shutil.copy(CLIPS[0], victim)
DS([victim], map_for([victim]), CFG, train=True)
_entry = DS([victim], map_for([victim]), CFG, train=True)._spec_path(victim)
before = os.stat(_entry).st_mtime_ns
_wav = os.stat(victim).st_mtime_ns
os.utime(victim, ns=(_wav, _wav + 10_000_000))
DS([victim], map_for([victim]), CFG, train=True)
after = os.stat(_entry).st_mtime_ns
check('touching a wav INVALIDATES its cache entry (stale-cache bug fixed)', after != before,
      f'{before} -> {after}')

# --- windowing ------------------------------------------------------------------------------
tot_train = sum(len(st) for st in d_train.train_starts)
check('training crops per clip > 1 (this is the augmentation that fixed memorisation)',
      tot_train / len(CLIPS) > 3, f'{tot_train / len(CLIPS):.1f} windows/clip, '
      f'{tot_train} distinct samples from {len(CLIPS)} clips')

d_eval = DS(CLIPS, DET_MAP, CFG, train=False)
clips_with_windows = len({ci for ci, _ in d_eval.items})   # items are (clip_idx, start)
check('every clip contributes >=1 eval window', clips_with_windows == len(CLIPS),
      f'{clips_with_windows}/{len(CLIPS)}')
per_clip = {}
for ci, _st in d_eval.items:
    per_clip[ci] = per_clip.get(ci, 0) + 1
check('no clip exceeds TEST_TOPK', max(per_clip.values()) <= 5,
      f'max {max(per_clip.values())}, mean {np.mean(list(per_clip.values())):.2f}')
x, y, _ci = d_eval[0]
check('eval item shape is (1, seq_len, n_freq_bins)', tuple(x.shape) == (1, SEQ, CFG['n_freq_bins']),
      str(tuple(x.shape)))
check('window values are in [0, 1]', float(x.min()) >= 0.0 and float(x.max()) <= 1.0,
      f'[{float(x.min()):.3f}, {float(x.max()):.3f}]')

# --- short clip: must be padded to the right value, not dropped -------------------------------
SHORT = make_short_copy(CLIPS[1], os.path.join(_TMP, '_short.wav'))
try:
    d_short = DS([SHORT], map_for([SHORT]), CFG, train=False)
    sx, _, _ = d_short[0]
    check('sub-window clip is zero-padded to seq_len', tuple(sx.shape) == (1, SEQ, CFG['n_freq_bins']),
          str(tuple(sx.shape)))
    info = sf.info(SHORT)
    n_real = min(1 + (int(round(info.frames / info.samplerate * 192000)) - 256) // 128, SEQ)
    p = (SEQ - n_real) // 2
    check('padded region == 0.0 (bug 2 fixed end-to-end)', float(sx[0, :p].abs().max()) == 0.0,
          f'pad max={float(sx[0, :p].abs().max()):.4f}, n_real={n_real}, pad={p}')
    check('real region reaches 1.0 after min-max', float(sx[0, p:p + n_real].max()) > 0.99,
          f'max={float(sx[0, p:p + n_real].max()):.4f}')
    # ATTRIBUTION: the pre-fix order, built from the cell's own pad/min-max functions, shows
    # what the bug did -- padding a dB array with 0.0 first makes the padding the maximum.
    raw = np.array(np.load(d_short._spec_path(SHORT)))[:SEQ]
    ref = ns['minmax_normalize'](ns['pad_window'](raw, SEQ))
    check('a pad-first reference fills that padding with 1.0 (the bug it avoids)',
          abs(ref[:p].mean() - 1.0) < 1e-6, f'reference pad mean={ref[:p].mean():.4f}')
    check('the notebook disagrees with the pad-first reference',
          not np.allclose(np.array(sx[0]), ref))
except Exception as e:
    check('sub-window clip handled', False, repr(e))

# --- unreadable clip is dropped, not fatal ---------------------------------------------------
BAD = os.path.join(_TMP, '_bad.wav')
sf.write(BAD, np.zeros(50, dtype='float32'), 384000)   # 50 samples < n_fft=256
try:
    d_bad = DS([BAD, CLIPS[2]], map_for([BAD, CLIPS[2]]), CFG, train=False)
    check('unusable clip dropped with a warning instead of raising', len(d_bad.file_names) == 1,
          f'kept {len(d_bad.file_names)} of 2')
except Exception as e:
    check('unusable clip dropped with a warning instead of raising', False, repr(e))

# --- predict_proba ----------------------------------------------------------------------------
torch.manual_seed(0)
model = nn.Sequential(nn.Flatten(), nn.Linear(SEQ * CFG['n_freq_bins'], 2))
# Guarded: predict_proba is where a signature change in the cell would blow up, and an
# unguarded call here kills the whole suite AFTER 25 checks have printed -- the section below
# would then never report at all. A FAIL naming the exception is strictly more useful.
_pp_err = None
try:
    probs, labels = ns['predict_proba'](model, d_eval, torch.device('cpu'))
except Exception as e:
    probs = labels = None
    _pp_err = f'{type(e).__name__}: {e}'
    GUARD_ERRORS['pp'] = _pp_err
check('predict_proba returns one row per clip',
      lambda: probs is not None and probs.shape[0] == len(CLIPS),
      lambda: str(probs.shape) if probs is not None else '', guarded='pp')
check('probabilities are normalised',
      lambda: probs is not None and np.allclose(probs.sum(1), 1.0, atol=1e-5),
      lambda: f'max dev {np.abs(probs.sum(1) - 1).max():.2e}' if probs is not None else '',
      guarded='pp')
check('labels line up with the clips',
      lambda: labels is not None and len(labels) == len(CLIPS),
      lambda: str(None if labels is None else len(labels)), guarded='pp')

# cache-key collision: the same basename from a different directory must not share a cache entry
alt_clip = os.path.join(_TMP, 'altdir', os.path.basename(CLIPS[0]))
os.makedirs(os.path.dirname(alt_clip), exist_ok=True)
shutil.copy(CLIPS[0], alt_clip)
d_alt = DS([alt_clip], map_for([alt_clip]), CFG, train=False)
check('same basename in a different directory gets its OWN cache entry',
      d_alt._spec_path(alt_clip) != d_train._spec_path(CLIPS[0]),
      f'{os.path.basename(d_alt._spec_path(alt_clip))} vs '
      f'{os.path.basename(d_train._spec_path(CLIPS[0]))}')

# ============================================================================================
# Background noise mixing (the combined notebook's c05). Measured there: +0.07 classifier /
# +0.09 detector balanced accuracy on recordings held out of training, for -0.013 in-split
# (AGENTS.md 11.3). The mechanism is that the label stays the window's own label while the
# background comes from ANOTHER recording, which breaks the "recognise the tape" shortcut.
# ============================================================================================
print('\n=== background noise mixing: mix_background_db (cell 5) ===')

NOISE_CLIPS = sorted(glob.glob(f'{REPO}/Data/final_dataset/data/noise/*.wav'))[:2]
MIX = sorted(f for f in glob.glob(f'{REPO}/Data/final_dataset/data/*/*.wav')
             if _cls(f) != 'noise')[:6] + NOISE_CLIPS
MIX_MAP = map_for(MIX)

mix = ns.get('mix_background_db')
check('the cell defines mix_background_db', mix is not None,
      'nothing in cell 5 defines mix_background_db' if mix is None else '')
if mix is None:      # every check below then reports this, and none of them is evaluated
    GUARD_ERRORS['mix'] = 'nothing in cell 5 defines mix_background_db'

# A window's dB spectrogram: 30 frames x 4 bins. Two noise windows -- one CONSTANT, so the
# added power can be recovered exactly from the output, and one varying, which is the real case.
sig_db = np.linspace(-60, -5, 30 * 4).reshape(30, 4).astype(np.float32)
noi_const = np.full((30, 4), -40.0, dtype=np.float32)
noi_var = np.linspace(-50, -20, 30 * 4).reshape(30, 4).astype(np.float32)
ps = 10.0 ** (sig_db.astype(np.float64) / 10.0)

for snr in (0.0, 10.0, 20.0):
    # The cell's documented contract: the window's mean power ends up `snr` dB above the
    # ADDED NOISE's mean power. With a constant noise window the added power is exactly
    # (p_out - p_sig), so this is measured from the function's output, not from its formula.
    out = mix(sig_db, noi_const, snr) if mix else None
    measured = (10 * np.log10(ps.mean() / (10.0 ** (out.astype(np.float64) / 10.0) - ps).mean())
                if out is not None else None)
    check(f'the gain puts the added noise exactly {snr:g} dB below the window',
          measured is not None and abs(measured - snr) < 1e-2,
          lambda snr=snr, m=measured: f'measured {m:.4f} dB' if m is not None else 'no output',
          guarded='mix')
    # With a VARYING noise window the added power is not recoverable elementwise, but the
    # mean is: mean(p_out) = mean(p_sig) + g*mean(p_noise) = mean(p_sig) * (1 + 10^-snr/10).
    out2 = mix(sig_db, noi_var, snr) if mix else None
    ratio = ((10.0 ** (out2.astype(np.float64) / 10.0)).mean() / ps.mean()) if out2 is not None else None
    check(f'   ...same for a varying noise window (mean power x {1 + 10.0 ** (-snr / 10):.4f})',
          ratio is not None and abs(ratio / (1 + 10.0 ** (-snr / 10)) - 1) < 1e-4,
          lambda r=ratio, s=snr: f'ratio {r:.6f}' if r is not None else 'no output', guarded='mix')

check('mixing preserves shape and dtype',
      mix is not None and mix(sig_db, noi_var, 10.0).shape == sig_db.shape
      and mix(sig_db, noi_var, 10.0).dtype == np.float32, guarded='mix')
check('mixing returns dB, NOT a min-max normalised array (the caller normalises afterwards)',
      mix is not None and not (0.0 <= float(mix(sig_db, noi_var, 10.0).min())
                              and float(mix(sig_db, noi_var, 10.0).max()) <= 1.0),
      lambda: f'[{float(mix(sig_db, noi_var, 10.0).min()):.2f}, '
              f'{float(mix(sig_db, noi_var, 10.0).max()):.2f}]', guarded='mix')
# The brief's version of the floor check fed a -20 dB signal with a -100 dB noise window at 0 dB
# SNR. That can never reach the floor: mix() scales the noise UP to the signal's mean power,
# so `ps + g*pn >= ps >= 1e-10` for every input the front end can produce (it clamps at
# -100 dB itself). The result was -17 dB and the check was unfalsifiable. Reaching the floor
# needs a signal BELOW it, which is what these two use.
_floor_mix = mix(np.full((30, 256), -200.0), np.full((30, 256), -200.0), 0.0) if mix else None
check('the floor is re-applied to the SUM (a signal below it comes back AT it, not below)',
      _floor_mix is not None and float(_floor_mix.min()) == -100.0,
      lambda: f'min {float(_floor_mix.min()):.2f}' if _floor_mix is not None else 'no output',
      guarded='mix')
# Without the np.maximum(..., floor) in the cell the value above would be about -200 dB.
check('   ...and the value it would have had WITHOUT the floor is genuinely lower',
      _floor_mix is not None
      and 10.0 ** (float(_floor_mix.min()) / 10) > 10.0 ** (-200.0 / 10),
      lambda: f'{float(_floor_mix.min()):.2f} dB vs the unclamped -200 dB'
      if _floor_mix is not None else '', guarded='mix')
# A custom floor must be honoured, not silently replaced by the default.
_custom = mix(np.full((30, 4), -200.0), np.full((30, 4), -200.0), 0.0,
              min_level_db=-60.0) if mix else None
check('an explicit min_level_db is honoured',
      _custom is not None and float(_custom.min()) == -60.0,
      lambda: f'min {float(_custom.min()):.2f}' if _custom is not None else 'no output',
      guarded='mix')

print('\n=== mixing is a TRAIN-ONLY augmentation (Review Focus 5) ===')
d_mix_train = d_mix_eval = d_mix_non = d_mix_always = None
try:
    d_mix_train = DS(MIX, MIX_MAP, CFG, train=True, noise_mix_prob=0.5)
    d_mix_eval = DS(MIX, MIX_MAP, CFG, train=False, noise_mix_prob=0.5)
    d_mix_non = DS(MIX, MIX_MAP, CFG, train=True, noise_mix_prob=0.0)
    d_mix_always = DS(MIX, MIX_MAP, CFG, train=True, noise_mix_prob=1.0)
    # Control: the new argument is simply not passed, i.e. the pre-mixing behaviour.
    d_mix_default = DS(MIX, MIX_MAP, CFG, train=True)
    # The control for the eval comparison below: the SAME files, mixing never requested.
    d_eval_control = DS(MIX, MIX_MAP, CFG, train=False, noise_mix_prob=0.0)
except Exception as e:       # a construction failure is a reportable FAIL, not a traceback
    GUARD_ERRORS['mix'] = GUARD_ERRORS['mix'] or f'{type(e).__name__}: {e}'

check('the noise bank holds ONLY noise clips (a species clip as a noise source would relabel '
      'the background and teach the model to ignore it)',
      d_mix_train is not None and len(d_mix_train._noise_ids) == len(NOISE_CLIPS)
      and all(ns['get_class_from_filename'](d_mix_train.file_names[i]) == 'noise'
              for i in d_mix_train._noise_ids),
      lambda: f'{d_mix_train._noise_ids} of {len(MIX)} clips' if d_mix_train else 'not built',
      guarded='mix')
check('mixing is OFF for an evaluation dataset even when noise_mix_prob is passed',
      d_mix_eval is not None and d_mix_eval.noise_mix_prob == 0.0 and d_mix_eval._noise_ids == [],
      lambda: f'prob={d_mix_eval.noise_mix_prob}, bank={d_mix_eval._noise_ids}'
      if d_mix_eval else 'not built', guarded='mix')
check('a training dataset with noise_mix_prob=0 has an empty noise bank',
      d_mix_non is not None and d_mix_non.noise_mix_prob == 0.0 and d_mix_non._noise_ids == [],
      lambda: f'prob={d_mix_non.noise_mix_prob}' if d_mix_non else 'not built', guarded='mix')
# Same file list on both sides: `len(d_mix_eval.file_names) == len(d_eval.file_names)` would
# compare MIX (8 clips) against CLIPS (12) and pass/fail for the wrong reason.
check('the eval dataset\'s window count is unaffected by the ignored mixing request',
      d_mix_eval is not None and d_eval_control is not None
      and len(d_mix_eval.items) == len(d_eval_control.items)
      and d_mix_eval.items == d_eval_control.items,
      lambda: f'{len(d_mix_eval.items)} vs {len(d_eval_control.items)} windows over the same clips'
      if d_mix_eval is not None and d_eval_control is not None else 'not built',
      guarded='mix')

if d_mix_train is not None and d_mix_non is not None and d_mix_always is not None:
    mx, my = d_mix_train[0]
    check('a mixed training item keeps the (1, seq_len, n_freq_bins) shape',
          tuple(mx.shape) == (1, SEQ, CFG['n_freq_bins']), str(tuple(mx.shape)), guarded='mix')
    check('a mixed training item is still min-max normalised to [0, 1] (mixing happens BEFORE '
          'min-max, in the power domain)',
          float(mx.min()) >= 0.0 and float(mx.max()) <= 1.0,
          lambda: f'[{float(mx.min()):.3f}, {float(mx.max()):.3f}]', guarded='mix')
    check('mixing does not change the label (the window keeps its own class)',
          int(my) == int(d_mix_train.labels[0]),
          lambda: f'{int(my)} == {int(d_mix_train.labels[0])}', guarded='mix')
    # prob must be 1.0 here: with 0.5 the coin flip in __getitem__ resolves the same way in
    # both datasets under the same seed, and the "does mixing do anything" check is vacuous.
    # Same seed -> same crop start, so any difference is the mixing alone.
    random.seed(7)
    a = d_mix_always[0][0].numpy()
    random.seed(7)
    b = d_mix_non[0][0].numpy()
    check('mixing actually changes the training window (it is wired into __getitem__, not just '
          'stored on the dataset)',
          not np.allclose(a, b),
          lambda: f'max |diff| {np.abs(a - b).max():.4f}', guarded='mix')
    # Omitting noise_mix_prob must give bit-for-bit the pre-mixing behaviour. Both calls are
    # reseeded: __getitem__ draws its crop start from the global RNG, so without that the two
    # draws would be different windows and the comparison would prove nothing.
    random.seed(7)
    b2 = d_mix_non[0][0].numpy()
    random.seed(7)
    b3 = d_mix_default[0][0].numpy()
    check('omitting noise_mix_prob gives bit-for-bit the pre-mixing behaviour',
          np.array_equal(b2, b3),
          lambda: f'max |diff| {np.abs(b2 - b3).max():.2e}' if b2 is not None and b3 is not None
          else 'not built', guarded='mix')

# A dataset with no noise clips must not crash when mixing is asked for: the cell says so
# and turns it off. Real class clips only, no 'noise'.
try:
    NO_NOISE = [f for f in MIX if _cls(f) != 'noise']
    d_none = DS(NO_NOISE, map_for(NO_NOISE), CFG, train=True, noise_mix_prob=0.5)
    built, err = True, None
except Exception as e:
    d_none, built, err = None, False, f'{type(e).__name__}: {e}'
check('mixing requested with NO noise clips degrades to off instead of raising',
      built and d_none._noise_ids == [], err or f'{len(d_none.file_names)} clips kept', guarded='mix')

# ============================================================================================
# The cache tag (_code_fingerprint). This replaced inspect.getsource, which on exec'd code
# silently reads the lines of an unrelated file -- so under a harness the tag changed on
# every run and the whole 964-clip cache was rebuilt for nothing.
# ============================================================================================
print('\n=== _code_fingerprint: the cache tag (cell 5) ===')
fp = ns.get('_code_fingerprint')
check('the cell defines _code_fingerprint', fp is not None,
      'nothing in cell 5 defines _code_fingerprint' if fp is None else '')
if fp is None:
    GUARD_ERRORS['fp'] = 'nothing in cell 5 defines _code_fingerprint'


def _f_default_a(cfg, preemphasis=0.98):
    return preemphasis


def _f_default_b(cfg, preemphasis=0.5):
    return preemphasis


def _f_same_code_one_line(cfg, preemphasis=0.98):
    return preemphasis


def _f_same_code_two_lines(cfg, preemphasis=0.98):   # identical code, different source lines
    return preemphasis


def _f_other_body(cfg, preemphasis=0.98):
    return 0.0


check('the fingerprint changes when a DEFAULT ARGUMENT value changes',
      fp is not None and fp(_f_default_a) != fp(_f_default_b),
      lambda: f'{fp(_f_default_a)} vs {fp(_f_default_b)}' if fp else '', guarded='fp')
# This is the check that caught the first version, which hashed only the bytecode.
check('   ...i.e. the fingerprint reads __defaults__, not just the bytecode',
      fp is not None
      and hashlib.md5(_f_default_a.__code__.co_code).hexdigest() == hashlib.md5(
          _f_default_b.__code__.co_code).hexdigest(),
      'bytecode alone cannot see a default value', guarded='fp')
check('the fingerprint is STABLE for identical code written on different lines (a comment '
      'edit must not invalidate the whole cache)',
      fp is not None and fp(_f_same_code_one_line) == fp(_f_same_code_two_lines),
      lambda: f'{fp(_f_same_code_one_line)} vs {fp(_f_same_code_two_lines)}' if fp else '',
      guarded='fp')
check('the fingerprint changes when the function BODY changes',
      fp is not None and fp(_f_default_a) != fp(_f_other_body),
      lambda: f'{fp(_f_default_a)} vs {fp(_f_other_body)}' if fp else '', guarded='fp')
check('the fingerprint does NOT change when only the FUNCTION is renamed (a rename does not '
      'change the computation, so it must not cost a rebuild of the 964-clip cache)',
      fp is not None and fp(_f_default_a) == fp(_f_same_code_one_line),
      lambda: f'{fp(_f_default_a)} vs {fp(_f_same_code_one_line)}' if fp else '', guarded='fp')
check('_CACHE_VERSION is the fingerprint of the audio loader and the front end, not a literal',
      fp is not None
      and DS._CACHE_VERSION == 'v3-' + fp(ns['load_audio_file'], ns['clip_to_db_spectrogram']),
      lambda: f'{DS._CACHE_VERSION} vs v3-{fp(ns["load_audio_file"], ns["clip_to_db_spectrogram"])}'
      if fp else '', guarded='fp')
# The whole point of deriving it: editing the front end must invalidate the cache. A
# separate hash of the same two functions must therefore move.
check('a hand-written _CACHE_VERSION could not satisfy the two checks above (falsifiable)',
      DS._CACHE_VERSION != 'v3' and DS._CACHE_VERSION != 'v2', DS._CACHE_VERSION)

# ============================================================================================
# predict_proba(return_embedding=True): the 'unknown' answer's input. forward_with_embedding
# comes from CELL 4 (Task 4's port) -- it is genuinely a cross-cell dependency at run time,
# and this is where the two cells are checked to actually fit together.
# ============================================================================================
print('\n=== predict_proba return_embedding (cell 5 reads cell 4) ===')
CELL_ARCH = 4
try:
    arch = extract_cells.cell_defs(idx[CELL_ARCH], {'build_model', 'forward_with_embedding',
                                                     'SoftmaxEnsemble'},
                                   {'torch': torch, 'nn': nn, 'OrderedDict': OrderedDict})
    ns['forward_with_embedding'] = arch['forward_with_embedding']
    build_real = arch['build_model']
    real, _, _, _ = build_real({'input_channels': 1, 'conv_kernel_size': 7, 'max_pool': 1,
                                'resnet_size': 18}, 3, torch.device('cpu'))
    real.eval()
    probs_e, labels_e, emb_e = ns['predict_proba'](real, d_eval, torch.device('cpu'),
                                                   return_embedding=True)
    emb_err = None
except Exception as e:       # a construction failure is a reportable FAIL, not a traceback
    real = probs_e = labels_e = emb_e = None
    emb_err = f'{type(e).__name__}: {e}'
    GUARD_ERRORS['emb'] = emb_err
    print(f'  (embedding path could not be built: {emb_err})')

check("predict_proba(return_embedding=True) returns probabilities, labels AND an embedding",
      emb_err is None and len((probs_e, labels_e, emb_e)) == 3 and emb_e.shape[0] == len(CLIPS),
      emb_err or str(emb_e.shape), guarded='emb')
check('the embedding is the per-clip MEAN over that clip\'s windows, 512-d for one model',
      emb_err is None and tuple(emb_e.shape) == (len(CLIPS), 512),
      emb_err or str(emb_e.shape), guarded='emb')
check('the probabilities are unaffected by the embedding request',
      emb_err is None and np.allclose(probs_e.sum(1), 1.0, atol=1e-5),
      emb_err or f'max dev {np.abs(probs_e.sum(1) - 1).max():.2e}', guarded='emb')
check('the labels are the same ones the 2-tuple form returned',
      emb_err is None and np.array_equal(labels_e, labels), guarded='emb')
check('predict_proba WITHOUT return_embedding still returns exactly 2 arrays',
      lambda: len(ns['predict_proba'](real, d_eval, torch.device('cpu'))) == 2, guarded='emb')

try:
    ens2 = arch['SoftmaxEnsemble']([real, real])
    _, _, emb2 = ns['predict_proba'](ens2, d_eval, torch.device('cpu'), return_embedding=True)
    ens_err = None
except Exception as e:
    emb2, ens_err = None, f'{type(e).__name__}: {e}'
check('the embedding of a 2-member ensemble is 2x512 wide (cells 4 and 5 fit together)',
      ens_err is None and tuple(emb2.shape) == (len(CLIPS), 2 * 512),
      ens_err or str(None if emb2 is None else emb2.shape), guarded='emb')

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + (': ' + ', '.join(fails) if fails else ''))
sys.exit(1 if fails else 0)
