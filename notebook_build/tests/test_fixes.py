#!/usr/bin/env python3
"""Window selection and the two functions that were silently wrong, executed from the artifact.

Every function under test is read out of the delivered notebook: cell 5 for the windowing,
cell 6 for `recording_id`, cell 13 for the prior correction. An earlier version of this file
defined `recording_id` and `prior_correct_probs` inline and asserted against its own copies,
so it passed whatever the notebook did -- including when the notebook was wrong.

The pre-fix behaviour is reproduced here as a local *reference* used only for attribution, the
way the old base-vs-fixed comparison is expressed now that there is no un-fixed notebook left
to read: a reference shows what the bug did, the notebook's own output is what is asserted.

Run: venv/bin/python notebook_build/tests/test_fixes.py
"""
import atexit
import glob
import os
import shutil
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

CELL_DATASET, CELL_DATA, CELL_PIPELINE = 5, 6, 13

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


def defs(path, names, **ns):
    """Exec the named top-level definitions out of a cell.

    A cell cannot simply be exec'd here: they print, glob Kaggle paths and build datasets.
    `extract_cells.cell_defs` pulls the definitions plus, by closure analysis, every
    module-level binding they read -- so this does not care whether a cell names its regex
    constant `_TAPE_RE`, `_TS_RE`, or anything else.
    """
    return extract_cells.cell_defs(path, names, ns)


idx = extract_cells.merged_cells()
print(f'=== under test: {os.environ.get("NEW_CELLS") or extract_cells.MERGED} ===')

FN = {'minmax_normalize', 'pad_window', 'window_scores', 'train_window_starts',
      'eval_window_starts'}
fns = defs(idx[CELL_DATASET], FN, np=np, os=os)

# --- reference for attribution only: the pre-fix min-max/pad ORDER -------------------------
def pad_then_minmax(win, seq_len):
    """Pad a dB array with 0.0 FIRST, then min-max. 0.0 is normally the array MAXIMUM, so
    min-max divides by the padding and maps every padded row to 1.0 -- silence handed to the
    model as the loudest thing in the window. The official order is normalise, then pad."""
    return fns['minmax_normalize'](fns['pad_window'](win, seq_len))


# Build a 2-D dB array: 12 frames of real signal inside a 30-frame (20 ms) window.
rs = np.random.RandomState(0)
db = np.sort(rs.uniform(-60, -3, (12, 4))).astype(np.float32)
SEQ = 30
PAD = (SEQ - db.shape[0]) // 2

_HOLD_DIR = tempfile.mkdtemp(prefix='batspot-hold-')
atexit.register(shutil.rmtree, _HOLD_DIR, True)
HOLD = os.path.join(_HOLD_DIR, '_hold.npy')


class Holder:
    """Just enough of a WindowedBatDataset for _window to run unmodified.

    `_window` reads through `self._spec_path(self.file_names[i])`; `_mmap` is here too because
    the mmapped-window form of the same method is equally plausible, and a Holder that only
    supported one of them would make this suite test a method that no longer exists.
    """
    seq_len = SEQ
    file_names = [HOLD]

    def _mmap(self, i):
        return db

    def _spec_path(self, f):
        return f


np.save(HOLD, db)   # _window np.load()s this; _mmap() bypasses it


print('\n=== FIX 2: min-max / pad ORDER (cell 5, _window) ===')
w = extract_cells.cell_method(idx[CELL_DATASET], 'WindowedBatDataset', '_window',
                              dict(fns))(Holder(), 0, 0)
w_ref = pad_then_minmax(db, SEQ)
check('shapes match', w.shape == (SEQ, 4) == w_ref.shape, str(w.shape))
check('a pad-first reference fills the padded region with 1.0 (silence as loudest feature)',
      abs(w_ref[:PAD].mean() - 1.0) < 1e-6, f'reference pad mean={w_ref[:PAD].mean():.4f}')
check('the notebook leaves the padded region at 0.0 (matches official)',
      abs(w[:PAD].mean()) < 1e-9, f'pad mean={w[:PAD].mean():.4f}')
check('the notebook still normalises the real signal to 1.0',
      abs(w[PAD:PAD + 12].max() - 1.0) < 1e-6, f'max={w[PAD:PAD + 12].max():.4f}')
check('the pad-first reference squashes the real signal below 1.0',
      w_ref[PAD:PAD + 12].max() < 1.0, f'reference max={w_ref[PAD:PAD + 12].max():.4f}')
check('the notebook output differs from the pad-first reference', not np.allclose(w, w_ref))
check('a window at least seq_len long is unaffected by the order',
      np.allclose(fns['minmax_normalize'](db), pad_then_minmax(db, db.shape[0])))


def percentile_top(starts, scores, top_frac):
    """The pre-fix selection: a percentile threshold, which keeps EVERY tied window."""
    cand = starts[scores >= np.percentile(scores, 100 * (1 - top_frac))]
    return cand if len(cand) >= 3 else starts[np.argsort(-scores)[:3]]


print('\n=== train_window_starts: percentile ties selected EVERYTHING (cell 5) ===')
starts = np.arange(0, 60, 3)
n = len(starts)
tied = np.ones(n)
k_ref = len(percentile_top(starts, tied, 0.20))
k_new = len(fns['train_window_starts'](starts, tied, 'energy_crop', 0.20))
# Asserting only `k_ref == n` would test the local reference and nothing else. The property that
# matters is the CONTRAST: the reference keeps every tied window and the cell does not.
check('a percentile-threshold reference keeps ALL tied windows; the notebook does not',
      k_ref == n and k_new != k_ref, f'reference kept {k_ref}/{n}, notebook kept {k_new}')
check('the notebook selects exactly ceil(20%)', k_new == int(np.ceil(n * 0.20)), f'{k_new}/{n}')
uniq = rs.uniform(0, 1, n)
kn = len(fns['train_window_starts'](starts, uniq, 'energy_crop', 0.20))
topk = set(starts[np.argsort(-uniq, kind='stable')[:kn]].tolist())
check('the notebook returns the loudest k, not a threshold set',
      set(fns['train_window_starts'](starts, uniq, 'energy_crop', 0.20).tolist()) == topk)
# Under ties every window is equally loud, so a rank-based stable selection must return the
# first k in position order -- the same windows every time. Asserting only the COUNT would be
# near-vacuous: any implementation returning n_keep values would pass.
_tie_a = fns['train_window_starts'](starts, tied, 'energy_crop', 0.20)
_tie_b = fns['train_window_starts'](starts, tied, 'energy_crop', 0.20)
check('the notebook is deterministic under ties (same windows on every call)',
      np.array_equal(_tie_a, _tie_b) and np.array_equal(_tie_a, starts[:k_new]),
      f'{_tie_a.tolist()} vs {starts[:k_new].tolist()}')
check('fewer than 3 windows returns all of them',
      len(fns['train_window_starts'](np.array([0, 3]), np.array([1.0, 2.0]),
                                     'energy_crop', 0.20)) == 2)
check("mode='first' still returns exactly one window",
      len(fns['train_window_starts'](starts, uniq, 'first', 0.20)) == 1)

print('\n=== eval_window_starts: greedy NMS over the loudest windows (cell 5) ===')
sel = np.sort(fns['eval_window_starts'](starts, uniq, 'energy_crop', 5, 30))
check('the first pick is the loudest window', sel[0] == starts[int(np.argmax(uniq))],
      f'first pick {sel[0]}, loudest {starts[int(np.argmax(uniq))]}')
g = np.diff(sel)
check('consecutive picks >= seq_len//2 apart', len(g) == 0 or g.min() >= 15,
      f'min gap {g.min() if len(g) else "-"}')
check("mode='first' unchanged", len(fns['eval_window_starts'](starts, uniq, 'first', 5, 30)) == 1)

# How bad the tie rule could get on the real clips, if every window scored the same.
allw = sorted(glob.glob(f'{extract_cells.REPO}/Data/final_dataset/data/*/*.wav'))
import soundfile as sf
n_fft, hop, stride, top_frac = 256, 128, 3, 0.20
degen, worst = 0, 0
for f in allw:
    info = sf.info(f)
    bad = False
    for sr in (250000, 192000):          # the two sample rates the pipeline trains at
        seq = int(0.02 * sr / hop)
        nf = 1 + (int(round(info.frames / info.samplerate * sr)) - n_fft) // hop
        if nf <= seq:
            continue
        w = len(np.arange(0, nf - seq + 1, stride))
        sc = np.ones(w)                      # worst case: all tied
        k = len(np.arange(w)[sc >= np.percentile(sc, 100 * (1 - top_frac))])
        worst = max(worst, w)
        bad = bad or k == w
    degen += bad
print(f'    (worst-case tied-score clips the percentile rule would keep ALL {worst} windows for: '
      f'{degen} of {len(allw)}, vs the notebook\'s ceil(20%) cap)')

print('\n=== recording_id (cell 6) ===')
# No constant name is requested: cell_defs follows what recording_id actually reads, so this
# works whether the cell binds _TAPE_RE, _TS_RE, or imports `re as _re`. If it cannot resolve
# them it raises here, before any check in this section has printed.
try:
    rid = defs(idx[CELL_DATA], {'recording_id'}, np=np, os=os)['recording_id']
except KeyError as e:
    rid = None
    print(f'  (cell 6 could not be isolated: {e})')
six = 'acsh-bat_3379376_2026_20260525-192000_91577_92135.wav'
four = 'sasa-bat_20260429-192000_215755_215781.wav'
_NO_RID = 'recording_id not extractable from cell 6'
if rid is None:
    check('6-field name still yields the tape id', False, _NO_RID)
    check('4-field name (what the clip-extraction cell writes) yields the tape id', False, _NO_RID)
    check('two clips from one tape group together', False, _NO_RID)
    check('different tapes stay apart', False, _NO_RID)
else:
    check('6-field name still yields the tape id', rid(six) == '20260525-192000', rid(six))
    check('4-field name (what the clip-extraction cell writes) yields the tape id',
          rid(four) == '20260429-192000', rid(four))
    check('two clips from one tape group together',
          rid(four) == rid(four.replace('215755_215781', '999999_999999')))
    check('different tapes stay apart', rid(four) != rid(six))
_old = lambda p: (os.path.basename(p)[:-4].split('_')[3]
                  if len(os.path.basename(p)[:-4].split('_')) >= 6 else os.path.basename(p))
check('ATTRIBUTION: the positional parse was broken on 4-field names',
      _old(four) == four, '-> every clip its own group')
grp = {rid(w) for w in allw} if rid is not None else set()
check('real dataset: 28 groups (not 964)',
      rid is not None and len(grp) == 28,
      _NO_RID if rid is None else f'{len(grp)} groups / {len(allw)} clips')

print('\n=== prior correction (cell 13) ===')
# WeightedRandomSampler draws every class with probability 1/n_classes, so the detector trains
# under a 50/50 prior against a 19/81 evaluation prior: a 4.18x shift in odds.
try:
    pc = defs(idx[CELL_PIPELINE], {'prior_correct_probs'}, np=np, os=os).get('prior_correct_probs')
except KeyError as e:
    pc = None
    MISSING = f'prior_correct_probs is not defined in the combined-pipeline cell ({e})'
else:
    MISSING = 'prior_correct_probs is not defined in the combined-pipeline cell'
PE, PT = np.array([0.193, 0.807]), np.array([0.5, 0.5])
raw = np.array([[0.5, 0.5]])


def corr(p, pe=PE, pt=PT):
    return pc(p, pe, pt) if pc is not None else None


c = corr(raw)
check('rows renormalise to 1', c is not None and abs(c.sum() - 1.0) < 1e-12,
      MISSING if c is None else f'{c.sum():.12f}')
check('a 50/50 posterior becomes call-leaning under an 81/19 prior',
      c is not None and c[0, 1] > 0.5, MISSING if c is None else f'-> {c[0, 1]:.3f}')
c95 = corr(raw, np.array([0.95, 0.05]), PT)
check('a 50/50 posterior becomes noise-leaning under a 95/5 prior',
      c95 is not None and c95[0, 0] > 0.5, MISSING if c95 is None else f'-> {c95[0, 0]:.3f}')
c_same = corr(raw, PT, PT)
check('matching priors are a no-op', c_same is not None and abs(c_same[0, 1] - 0.5) < 1e-12,
      MISSING if c_same is None else f'-> {c_same[0, 1]:.3f}')
c_odds = corr(np.array([[0.9, 0.1]]))
odds_raw = 0.1 / 0.9
check('corrected odds = raw odds x (prior_eval ratio / prior_train ratio)',
      c_odds is not None
      and abs(c_odds[0, 1] / c_odds[0, 0] - odds_raw * (PE[1] / PE[0]) / (PT[1] / PT[0])) < 1e-9,
      MISSING if c_odds is None
      else f'{c_odds[0, 1] / c_odds[0, 0]:.6f} vs '
           f'{odds_raw * (PE[1] / PE[0]) / (PT[1] / PT[0]):.6f}')

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + (': ' + ', '.join(fails) if fails else ''))
sys.exit(1 if fails else 0)
