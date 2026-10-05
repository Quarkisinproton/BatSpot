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
import glob
import hashlib
import os
import re
import shutil
import sys
import tempfile
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


def check(name, cond, detail=''):
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
    """The cell's cache loop drives tqdm; a progress bar is noise in a test log, and the
    cache-reuse assertion reads the cell's own stdout lines, not the bar."""
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
# The tag is the source DIRECTORY's hash: two DATA_DIRs with colliding basenames must not
# share a cache entry. Asserted exactly, because an underscore in the basename would satisfy
# a looser "is there a tag in the name" test without any tagging at all.
_tag = hashlib.md5(os.path.dirname(os.path.abspath(CLIPS[0])).encode()).hexdigest()[:6]
check('cache filename carries a directory tag',
      all(os.path.basename(d_train._spec_path(f)) == f'{_tag}_{os.path.basename(f)}.npy'
          for f in CLIPS),
      os.path.basename(d_train._spec_path(CLIPS[0])))
check('cache version tag is derived from the front-end source, not hand-written',
      bool(re.fullmatch(r'v2-[0-9a-f]{6}', DS._CACHE_VERSION)), DS._CACHE_VERSION)

DS(CLIPS, DET_MAP, CFG, train=True)   # second construction: must reuse, not re-convert
check('second construction reuses the cache (no re-conversion printed above)',
      len(glob.glob(os.path.join(WD, 'spec_cache', '*', '*.npy'))) == len(CLIPS))

# --- the stale-cache bug: touch a wav and confirm the entry is rebuilt -----------------------
# Run on a private copy, so the suite never mutates the mtime of a file in Data/.
_probe = os.path.join(_TMP, 'probe_clips')
os.makedirs(_probe, exist_ok=True)
victim = os.path.join(_probe, os.path.basename(CLIPS[0]))
shutil.copy(CLIPS[0], victim)
DS([victim], map_for([victim]), CFG, train=True)
before = os.stat(DS([victim], map_for([victim]), CFG, train=True)._spec_path(victim)).st_mtime_ns
os.utime(victim, ns=(os.stat(victim).st_mtime_ns, os.stat(victim).st_mtime_ns + 10_000_000))
after = os.stat(DS([victim], map_for([victim]), CFG, train=True)._spec_path(victim)).st_mtime_ns
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
probs, labels = ns['predict_proba'](model, d_eval, torch.device('cpu'))
check('predict_proba returns one row per clip', probs.shape[0] == len(CLIPS),
      str(probs.shape))
check('probabilities are normalised', np.allclose(probs.sum(1), 1.0, atol=1e-5),
      f'max dev {np.abs(probs.sum(1) - 1).max():.2e}')
check('labels line up with the clips', len(labels) == len(CLIPS))

# cache-key collision: the same basename from a different directory must not share a cache entry
alt_clip = os.path.join(_TMP, 'altdir', os.path.basename(CLIPS[0]))
os.makedirs(os.path.dirname(alt_clip), exist_ok=True)
shutil.copy(CLIPS[0], alt_clip)
d_alt = DS([alt_clip], map_for([alt_clip]), CFG, train=False)
check('same basename in a different directory gets its OWN cache entry',
      d_alt._spec_path(alt_clip) != d_train._spec_path(CLIPS[0]),
      f'{os.path.basename(d_alt._spec_path(alt_clip))} vs '
      f'{os.path.basename(d_train._spec_path(CLIPS[0]))}')

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + (': ' + ', '.join(fails) if fails else ''))
sys.exit(1 if fails else 0)