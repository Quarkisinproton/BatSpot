#!/usr/bin/env python3
"""The polyphase resampler, and the phase-aligned chunking it exists for.

WHAT THIS IS FOR
`_resampled(path, n_in, sr_in, sr, r0, r1)` returns output samples [r0, r1) of a whole recording
resampled to `sr`, but computes them from a margin-padded EXCERPT -- which is what lets a
5-minute file be scanned in 60-second chunks without holding 115 M samples in memory.

That only works if the excerpt's output samples ARE the whole file's output samples. They are
only so when the excerpt starts on an input sample that falls exactly on the output grid: for
the ratio L/M = sr_out/sr_in, output index q is built from input index q*M/L, so an excerpt that
begins at input s reproduces whole-file output n = q + q0 only if s == q0*M/L, i.e. q0 is a
multiple of L. The base notebook's cell 21 started the excerpt at `floor((r0 - margin)*M/L)`,
which is generally NOT such a sample: for 250 / 500 / 256 kHz -> 192 kHz it shifts the chunk's
output grid by a fraction of a sample, and on white noise the excerpt then differs from the whole
file by 7-13 % of full scale (measured below, against the base cell). 384 kHz was unaffected
because its ratio is exactly 1/2.

So the headline check is `_resampled` at a range == `_resampled` over the whole file, sliced.
The signal is white noise because it is the worst case for a misaligned grid: the kaiser_best
filters pass ~95 % of the lower Nyquist, so a sub-sample shift decorrelates the output instead of
hiding under a smooth waveform.

HOW TIGHT IS "TIGHT"?  (measured, not assumed -- see the report)
For 384 kHz the result is BITWISE identical and `np.array_equal` says so. For 250 / 256 kHz the
residual is ~1.2e-07 of full scale and is NOT the cell's arithmetic: padding both conv1d inputs
to the same length makes it exactly 0.000e+00. torch's conv1d accumulates a length-K dot product
in a length-dependent order, so a shorter excerpt rounds a few ulps differently. The check below
therefore uses a budget of 1e-5 of full scale -- four orders of magnitude above the float32
rounding actually observed, and four orders of magnitude BELOW the pre-fix error -- and pins both
ends of that gap by measuring the base cell's own `_resampled` rather than asserting a remembered
number.

Run: venv/bin/python notebook_build/tests/test_resample.py
"""
import os
import sys
from fractions import Fraction

import numpy as np
import resampy
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _inferfix
import extract_cells

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


def source(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


idx = extract_cells.merged_cells()
UNDER_TEST = os.environ.get('NEW_CELLS') or extract_cells.MERGED
FUNCS = idx[_inferfix.CELL_FUNCS]

# --- the extraction is itself under test -----------------------------------------------------
check('the extraction returns the inference-functions cell',
      _inferfix.CELL_FUNCS in idx, f'from {UNDER_TEST}')
check(f'extracted cell {_inferfix.CELL_FUNCS} really is the inference-functions cell',
      'def _resampled(' in source(FUNCS), os.path.basename(FUNCS))
check('   ...and it carries the phase alignment, not the pre-fix grid',
      'q0 = ((r0 - margin) // L) * L' in source(FUNCS))

ns = _inferfix.funcs_ns(idx)
resampled = ns['_resampled']
read_mono = ns['_read_mono']

# --- fixtures ---------------------------------------------------------------------------------
# Long enough that every range below is far from the ends, and white so a misaligned grid shows.
RATES = (384000, 250000, 500000, 256000)
LENGTHS = {384000: 40000, 250000: 20000, 500000: 40000, 256000: 30000}
BUDGET_REL = 1e-5        # of full scale; the float32 rounding observed is ~3e-7, the pre-fix
                         # error ~1e-1, so this sits four orders above each end
work = _inferfix.working_dir()
wavs, sigs = {}, {}
for sr_in, n in LENGTHS.items():
    p = _inferfix.write_wav(os.path.join(work, f'white_{sr_in}.wav'), sr_in, n, seed=1)
    wavs[sr_in] = p
    sigs[sr_in] = sf.read(p, dtype='float32')[0]

print('\n=== _resampled at a range == the whole file, sliced (the deliverable) ===')
devs = {}
for sr_in in RATES:
    n = LENGTHS[sr_in]
    n_out = int(n * _inferfix.TARGET_SR / sr_in)
    full = resampled(wavs[sr_in], n, sr_in, _inferfix.TARGET_SR, 0, n_out)
    peak = float(np.abs(full).max())
    budget = BUDGET_REL * peak
    ranges = [(0, 1000), (5000, 9000), (n_out - 7000, n_out - 10)]
    devs[sr_in] = {}
    for r0, r1 in ranges:
        got = resampled(wavs[sr_in], n, sr_in, _inferfix.TARGET_SR, r0, r1)
        right_len = len(got) == r1 - r0
        d = float(np.abs(got - full[r0:r1]).max()) if right_len else float('inf')
        devs[sr_in][(r0, r1)] = d
        check(f'{sr_in} Hz: [{r0},{r1}) reproduces the whole file ({r1-r0} samples)',
              right_len and d <= budget,
              f'max|excerpt - full| = {d:.3e} of full scale {peak:.3f}'
              + ('' if right_len else f'  (returned {len(got)} samples)'))
    worst = max(devs[sr_in].values())
    check(f'{sr_in} Hz: every range bitwise identical (max|d| = 0.00e+00)',
          all(d == 0.0 for d in devs[sr_in].values()),
          f'worst max|d| = {worst:.3e}' + ('' if worst == 0.0
                                           else '  (float32 accumulation order, see below)'))

print('\n=== an offset sweep: 60 chunk starts per rate, not three hand-picked ones ===')
for sr_in in RATES:
    n = LENGTHS[sr_in]
    n_out = int(n * _inferfix.TARGET_SR / sr_in)
    full = resampled(wavs[sr_in], n, sr_in, _inferfix.TARGET_SR, 0, n_out)
    peak = float(np.abs(full).max())
    bad, worst = 0, 0.0
    for r0 in range(0, max(1, n_out - 900), 197):        # 197 is prime: no aliasing with the stride
        got = resampled(wavs[sr_in], n, sr_in, _inferfix.TARGET_SR, r0, r0 + 800)
        d = float(np.abs(got - full[r0:r0 + 800]).max())
        worst = max(worst, d)
        bad += d > BUDGET_REL * peak
    check(f'{sr_in} Hz: every swept chunk start agrees with the whole file',
          bad == 0, f'worst max|d| = {worst:.3e}, {bad} chunk(s) outside budget')

# --- the gap this fix closes, measured against the BASE cell's own code ------------------------
# `_resampled` is pulled out of the read-only base notebook by name. Re-typing the pre-fix
# arithmetic here would let the two implementations drift apart and make the comparison a
# statement about this file rather than about the notebook.
print('\n=== how far the pre-fix chunking was off (the number the budget must stay under) ===')
base = extract_cells.base_cells()
base_resampled = None
try:
    base_resampled = extract_cells.cell_defs(
        base[_inferfix.CELL_FUNCS], {'_resampled'},
        {'np': np, 'torch': torch, 'resampy': resampy, 'sf': sf,
         'DEVICE': torch.device('cpu'), 'INFER_TIME_EXPANSION': 1})['_resampled']
except Exception as _e:                                   # noqa: BLE001
    check('the base notebook\'s _resampled could be pulled out for comparison', False,
          f'{type(_e).__name__}: {_e}')

for sr_in in RATES:
    if base_resampled is None:
        break
    n = LENGTHS[sr_in]
    n_out = int(n * _inferfix.TARGET_SR / sr_in)
    full = resampled(wavs[sr_in], n, sr_in, _inferfix.TARGET_SR, 0, n_out)
    peak = float(np.abs(full).max())
    worst_old = 0.0
    for (r0, r1) in devs[sr_in]:
        old = base_resampled(wavs[sr_in], n, sr_in, _inferfix.TARGET_SR, r0, r1)
        worst_old = max(worst_old, float(np.abs(old - full[r0:r1]).max()))
    worst_new = max(devs[sr_in].values())
    # A rate pair was immune to the pre-fix bug exactly when L == 1: s = floor((r0-margin)*M/L)
    # is then already exact, because M/L is a whole number of input samples per output sample.
    # 384 -> 192 kHz is the case (L=1, M=2); 500 -> 192 is not (L=48).
    unit_fraction = Fraction(_inferfix.TARGET_SR, sr_in).numerator == 1
    note = ('  [ratio M/L = 1/2 is a unit fraction: this rate pair was never affected]'
            if unit_fraction else '')
    check(f'{sr_in} Hz: the pre-fix chunking is >= 1000x outside the budget the port must meet '
          f'(so the budget discriminates)',
          worst_old >= 1000 * BUDGET_REL * peak or unit_fraction,
          f'pre-fix max|d| = {worst_old:.3e} ({worst_old/peak:.1%} of full scale), '
          f'budget {BUDGET_REL*peak:.1e}' + note)
    if not unit_fraction:
        check(f'{sr_in} Hz: the port is >= 1000x closer to the whole file than the base cell',
              worst_new <= worst_old / 1000.0,
              f'ported {worst_new:.3e} vs pre-fix {worst_old:.3e}, '
              f'ratio {worst_old/max(worst_new, 1e-30):.1e}')

# --- the alignment itself, observed rather than read --------------------------------------------
print('\n=== the excerpt starts on an output-grid input sample (s % M == 0) ===')
_real_read_mono = ns['_read_mono']
seen = []


def recording_read_mono(path, n_in, start, stop):
    seen.append((start, stop))
    return _real_read_mono(path, n_in, start, stop)


ns['_read_mono'] = recording_read_mono
for sr_in in RATES:
    M = Fraction(_inferfix.TARGET_SR, sr_in).denominator
    n = LENGTHS[sr_in]
    n_out = int(n * _inferfix.TARGET_SR / sr_in)
    seen.clear()
    for r0 in (0, 1, 7, 5000, 12345, n_out - 3000):
        resampled(wavs[sr_in], n, sr_in, _inferfix.TARGET_SR, r0, min(r0 + 900, n_out))
    starts = [s for s, _ in seen]
    # s == q0*M/L with q0 a multiple of L  <=>  s is a multiple of M
    off_grid = [s for s in starts if s % M]
    check(f'{sr_in} Hz: all {len(starts)} excerpt starts land on the output grid (M={M})',
          not off_grid, f'first 5 starts {starts[:5]}' if off_grid else f'{starts[:3]} ...')
ns['_read_mono'] = _real_read_mono

# --- where the residual comes from, so the budget is not a mystery ----------------------------
print('\n=== attribution: the residual is conv1d accumulation order, not the cell\'s arithmetic ===')
print('    (both excerpts hold the SAME global samples; only the conv1d input length differs)')


def excerpt(sig, s, e):
    """`sig[s:e]` with zeros outside the file -- what `_read_mono` returns for the same range."""
    a, b = max(0, s), min(len(sig), e)
    x = sig[a:b]
    if s < a or e > b:
        x = np.concatenate([np.zeros(a - s, np.float32), x,
                            np.zeros(max(0, e - max(b, a)), np.float32)])
    return x


for sr_in in (250000, 256000):
    fr = Fraction(_inferfix.TARGET_SR, sr_in)
    L, M = fr.numerator, fr.denominator
    if '_kaiser_best_polyphase' not in ns:
        check(f'{sr_in} Hz: with both conv1d inputs the same length the chunk is bitwise '
              'identical', False,
              'the cell has no _kaiser_best_polyphase, so it has no fast polyphase path at all')
        continue
    poly = ns['_kaiser_best_polyphase'](sr_in, _inferfix.TARGET_SR)
    lo = poly[2]
    sig = sigs[sr_in]
    r0, r1 = 5000, 9000
    q0a, q0b = ((0 - 4096) // L) * L, ((r0 - 4096) // L) * L
    xa = excerpt(sig, q0a * M // L, int(np.ceil((r1 + 4096) * M / L)))
    xb = excerpt(sig, q0b * M // L, int(np.ceil((r1 + 4096) * M / L)))
    common = max(len(xa), len(xb)) + abs(lo) + 16

    def conv_padded(x, n_out_, pad_len, _poly=poly, _lo=lo, _L=L, _M=M):
        """The cell's own `_polyphase_resample`, with one variable changed: the zero-pad length.

        Nothing else. If the cell's tap placement `W[p, j]` were wrong, matching the input lengths
        would not make the discrepancy vanish, so this still tests the bank.
        """
        xp = np.zeros(pad_len, np.float32)
        xp[-_lo:-_lo + len(x)] = x
        xt = torch.from_numpy(xp)[None, None]
        wt = torch.from_numpy(_poly[3])[:, None, :]
        Q = -(-n_out_ // _L)
        y = torch.nn.functional.conv1d(xt, wt, stride=_M)[0, :, :Q]
        return y.T.reshape(-1)[:n_out_].numpy()

    a_out = conv_padded(xa, r1 - q0a, common)[r0 - q0a:r1 - q0a]
    b_out = conv_padded(xb, r1 - q0b, common)[r0 - q0b:r1 - q0b]
    d_padded = float(np.abs(a_out - b_out).max()) if len(a_out) == len(b_out) else float('inf')
    check(f'{sr_in} Hz: with both conv1d inputs the same length the chunk is bitwise identical',
          d_padded == 0.0, f'max|d| = {d_padded:.3e}  (as shipped, unequal lengths: '
                           f'{devs[sr_in][(5000, 9000)]:.3e})')

# --- the margin is load-bearing ---------------------------------------------------------------
print('\n=== the margin: it must clear the filter, or the equality is luck ===')
for sr_in in RATES:
    if '_kaiser_best_polyphase' not in ns:
        check(f'{sr_in} Hz: the default margin clears the filter', False,
              'the cell has no polyphase filter bank to measure the filter width against')
        continue
    poly = ns['_kaiser_best_polyphase'](sr_in, _inferfix.TARGET_SR)
    K = poly[3].shape[1]
    n = LENGTHS[sr_in]
    n_out = int(n * _inferfix.TARGET_SR / sr_in)
    full = resampled(wavs[sr_in], n, sr_in, _inferfix.TARGET_SR, 0, n_out)
    peak = float(np.abs(full).max())
    tight = ns['_resampled'].__defaults__[0] if ns['_resampled'].__defaults__ else None
    broke = None
    for m in (8, 32, 64, 128, K - 1, K):
        got = resampled(wavs[sr_in], n, sr_in, _inferfix.TARGET_SR, 5000, 9000, margin=m)
        d = (float(np.abs(got - full[5000:9000]).max()) if len(got) == 4000 else float('inf'))
        if d > BUDGET_REL * peak:
            broke = m
            break
    check(f'{sr_in} Hz: the default margin clears the {K}-tap filter (a margin of '
          f'{broke} already breaks the equality)',
          tight is not None and tight >= K and broke is not None and tight > broke,
          f'filter width K={K}, default margin={tight}, first failing margin={broke}')

# --- the polyphase bank itself is faithful -----------------------------------------------------
print('\n=== the polyphase bank vs resampy (the self-check the cell prints) ===')
for sr_in in RATES:
    if '_POLY_OK' not in ns:
        check(f'{sr_in} Hz: the fast polyphase path agrees with resampy (the cell accepts <= 2e-4)',
              False, 'the cell has no polyphase fast path at all')
        continue
    x = sigs[sr_in][:4096]
    ref = resampy.resample(x, sr_in, _inferfix.TARGET_SR, filter='kaiser_best')
    got = ns['resample_like_training'](x, sr_in, _inferfix.TARGET_SR)
    same_len = len(ref) == len(got)
    d = float(np.abs(ref - got).max() / max(float(np.abs(ref).max()), 1e-12)) if same_len else float('inf')
    check(f'{sr_in} Hz: the fast polyphase path agrees with resampy (the cell accepts <= 2e-4)',
          same_len and d <= 2e-4 and ns['_POLY_OK'].get((sr_in, _inferfix.TARGET_SR)) is True,
          f'{d:.2e} of full scale, _POLY_OK={ns["_POLY_OK"].get((sr_in, _inferfix.TARGET_SR))}')

# --- a ratio the polyphase path must still get right -------------------------------------------
print('\n=== a ratio with 147 phases is still resampled correctly ===')
ok_441, why_441 = True, ''
try:
    x = sigs[384000][:8192]
    a = ns['resample_like_training'](x, 441000, 192000)
    b = resampy.resample(x, 441000, _inferfix.TARGET_SR, filter='kaiser_best').astype(np.float32)
    d441 = (float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-12))
            if len(a) == len(b) else float('inf'))
    ok_441, why_441 = d441 <= 2e-4, f'{d441:.2e} of full scale'
except Exception as _e:                                        # noqa: BLE001
    ok_441, why_441 = False, f'raised {type(_e).__name__}: {_e}'
check('441 -> 192 kHz matches resampy', ok_441, why_441)

print('\n' + '=' * 70)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)