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
multiple of L. The base notebook's cell 21 started the excerpt at `floor((r0 - margin)*sr_in/sr)`,
which is generally NOT such a sample: for 250 / 500 / 256 kHz -> 192 kHz it shifts the chunk's
output grid by a fraction of a sample. Measured below against the base cell's OWN `_resampled` on
the same fixtures: 77 % of full scale at 250 kHz, 16 % at 500 kHz, 67 % at 256 kHz -- on white
noise, which decorrelates completely under a sub-sample shift. 384 kHz was unaffected because its
ratio is exactly 1/2 (M/L = 2 input samples per output sample, so the floor is exact), which is
why one rate pair alone could never catch this bug and the suite uses four.

So the headline check is `_resampled` at a range == `_resampled` over the whole file, sliced.

HOW TIGHT IS "TIGHT"?  (measured, not assumed -- see the report)
Two separate questions, so two separate checks:

  1. AS SHIPPED, the two calls are bitwise identical for 384 / 500 kHz and differ by <= 2.1e-07
     absolute (5.7e-07 of full scale, a few ulp of float32) for 250 / 256 kHz. That residual is
     NOT the cell's arithmetic: hand both calls a conv1d input of ONE length and it becomes
     exactly 0.00e+00 for all four rates (the `attribution` section below). torch's conv1d
     accumulates a length-K dot product in a length-dependent order, so a shorter excerpt rounds
     a few ulps differently from a whole-file one.
  2. The budget the port must meet is therefore 1e-5 of full scale: ~20x above the worst residual
     measured over a 120-start sweep, and >10 000x below the pre-fix error. BOTH ends of the gap
     are measured here -- the rounding by the equal-length check, the pre-fix error by running
     the base cell's own `_resampled` -- so no remembered number appears in an assertion.

Every check is exception-proof: a call that raises is a FAIL carrying the exception, never a
traceback that loses the rest of the suite.

Run: venv/bin/python notebook_build/tests/test_resample.py
"""
import contextlib
import inspect
import io
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


def check(name, thunk):
    """`thunk` returns (cond, detail). An exception in it is a FAIL, not a dead suite."""
    try:
        cond, detail = thunk()
    except Exception as e:                                        # noqa: BLE001
        cond, detail = False, f'{type(e).__name__}: {e}'
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)
    return cond


def source(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def rel(x, peak):
    return float(x) / max(float(peak), 1e-30)


def is_immune(sr_in):
    """True when M/L is a whole number: each output sample then reads a whole number of input
    samples, so the pre-fix floor() already landed on the output grid for that rate pair."""
    fr = Fraction(TARGET, sr_in)
    return fr.denominator % fr.numerator == 0


idx = extract_cells.merged_cells()
UNDER_TEST = os.environ.get('NEW_CELLS') or extract_cells.MERGED
FUNCS = idx[_inferfix.CELL_FUNCS]
TARGET = _inferfix.TARGET_SR

print(f'=== under test: {UNDER_TEST} (cell {_inferfix.CELL_FUNCS} = {os.path.basename(FUNCS)}) ===')

# --- the port itself is under test -------------------------------------------------------------
print('\n=== the cell carries the phase-aligned grid ===')
txt = source(FUNCS)
check('the excerpt start is aligned to the output grid (q0 a multiple of L)',
      lambda: ('q0 = ((r0 - margin) // L) * L' in txt,
               'the pre-fix cell has no q0 at all -- it computes '
               's = floor((r0 - margin) * sr_in / sr)'))
check('   ...and reaches the input sample that grid point sits on (s = q0 * M // L)',
      lambda: ('s = q0 * M // L' in txt, ''))
check('   ...and indexes the excerpt by whole output samples (i0 = r0 - q0, not a rounded ratio)',
      lambda: ('i0 = r0 - q0' in txt, 'the pre-fix cell rounds r0 - s*sr/sr_in instead'))

ns = _inferfix.funcs_ns(idx)
resampled = ns['_resampled']
read_mono = ns['_read_mono']

# --- fixtures ---------------------------------------------------------------------------------
# The four rates the cell's own comments name, and a length per rate that gives the sweep room.
RATES = (384000, 250000, 500000, 256000)
LENGTHS = {384000: 40000, 250000: 20000, 500000: 40000, 256000: 30000}
BUDGET_REL = 1e-5        # of full scale. Measured: shipped residual <= 5.7e-07 (120 starts/rate),
                         # pre-fix error >= 0.16. Two orders above each end, not one.
work = _inferfix.working_dir()
wavs, sigs = {}, {}
for sr_in, n in LENGTHS.items():
    p = _inferfix.write_wav(os.path.join(work, f'white_{sr_in}.wav'), sr_in, n, seed=1)
    wavs[sr_in] = p
    sigs[sr_in] = sf.read(p, dtype='float32')[0]


def n_out_of(sr_in):
    return int(LENGTHS[sr_in] * TARGET / sr_in)


def full_of(sr_in):
    return resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, 0, n_out_of(sr_in))


def ranges_of(sr_in):
    """r0 = 0 / r1 = n_out are the edge-of-file shapes; 5000 and n_out-7000 are interior; r0 = 1 is
    the one whose grid alignment differs most from the pre-fix floor."""
    n_out = n_out_of(sr_in)
    return [(0, 1000), (1, 1000), (5000, 9000), (n_out - 7000, n_out - 10), (n_out - 300, n_out)]


print('\n=== _resampled at a range == the whole file, sliced (the deliverable) ===')
devs, peaks = {}, {}
for sr_in in RATES:
    full = full_of(sr_in)
    peaks[sr_in] = float(np.abs(full).max())
    devs[sr_in] = {}
    for r0, r1 in ranges_of(sr_in):
        devs[sr_in][(r0, r1)] = None

        def one(sr_in=sr_in, r0=r0, r1=r1, full=full):
            got = resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, r0, r1)
            if len(got) != r1 - r0:
                return False, f'wrong length {len(got)} (want {r1 - r0})'
            d = float(np.abs(got - full[r0:r1]).max())
            devs[sr_in][(r0, r1)] = d
            return d <= BUDGET_REL * peaks[sr_in], \
                f'max|excerpt - full| = {d:.3e} = {rel(d, peaks[sr_in]):.1e} of full scale'
        check(f'{sr_in} Hz: [{r0},{r1}) reproduces the whole file ({r1 - r0} samples)', one)

print('\n=== an offset sweep: 120 chunk starts per rate, not five hand-picked ones ===')
for sr_in in RATES:
    def swept(sr_in=sr_in):
        full = full_of(sr_in)
        n_out = n_out_of(sr_in)
        starts = sorted({int(round(i * (n_out - 900) / 119.0)) for i in range(120)})
        worst, bad = 0.0, 0
        for r0 in starts:
            got = resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, r0, r0 + 800)
            if len(got) != 800:
                bad += 1
                continue
            d = float(np.abs(got - full[r0:r0 + 800]).max())
            worst = max(worst, d)
            bad += d > BUDGET_REL * peaks[sr_in]
        return (len(starts) == 120 and bad == 0), \
            (f'{len(starts)} distinct starts, worst max|d| = {worst:.3e} = '
             f'{rel(worst, peaks[sr_in]):.1e} of full scale, {bad} outside the '
             f'{BUDGET_REL:.0e} budget')
    check(f'{sr_in} Hz: every one of 120 swept chunk starts agrees with the whole file', swept)

# --- how far off the pre-fix chunking was, measured on the BASE cell's own code -----------------
print('\n=== the gap this fix closes: the read-only base notebook\'s own _resampled ===')
print('    (pulled out by name, not re-typed, so the comparison is a statement about that notebook)')
base_cells = extract_cells.base_cells()
base_ns = extract_cells.cell_defs(
    base_cells[_inferfix.CELL_FUNCS], {'_resampled'},
    {'np': np, 'torch': torch, 'resampy': resampy, 'sf': sf, 'os': os,
     'DEVICE': torch.device('cpu'), 'INFER_TIME_EXPANSION': 1})
base_resampled = base_ns['_resampled']

for sr_in in RATES:
    def compare(sr_in=sr_in):
        full = full_of(sr_in)
        worst_old = worst_new = 0.0
        for r0, r1 in ranges_of(sr_in):
            old = base_resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, r0, r1)
            new = resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, r0, r1)
            if len(old) != r1 - r0 or len(new) != r1 - r0:
                return False, 'a call returned the wrong length'
            worst_old = max(worst_old, float(np.abs(old - full[r0:r1]).max()))
            worst_new = max(worst_new, float(np.abs(new - full[r0:r1]).max()))
        detail = (f'pre-fix max|d| = {worst_old:.3e} = {rel(worst_old, peaks[sr_in]):.1%} of full '
                  f'scale; ported max|d| = {worst_new:.3e} = {rel(worst_new, peaks[sr_in]):.1e}')
        if is_immune(sr_in):
            # M/L is a whole number, so the pre-fix floor lands on the grid anyway. A check that
            # demanded the port beat the base cell HERE would pass with the bug in place, so the
            # check says the opposite: this rate pair is immune and cannot witness the fix.
            return worst_old <= BUDGET_REL * peaks[sr_in], \
                f'IMMUNE to the bug: M/L is a whole number, so the pre-fix floor was exact too ' \
                f'and this rate pair CANNOT see the fix. {detail}'
        return (worst_old >= 1000 * BUDGET_REL * peaks[sr_in]
                and worst_new * 1000 <= worst_old), \
            (f'the pre-fix grid is {rel(worst_old, peaks[sr_in]):.1%} of full scale = '
             f'{worst_old / max(worst_new, BUDGET_REL * peaks[sr_in]):.1e}x the budget (ported '
             f'max|d| = {worst_new:.3e}). {detail}')
    check(f'{sr_in} Hz: base cell vs port, on the same fixtures', compare)

# --- attribution: the residual is conv1d accumulation order, not the cell's arithmetic ---------
print('\n=== attribution: equal-length conv1d inputs -> bitwise identical (max|d| = 0.00e+00) ===')
print('    Appending zeros to the excerpt changes nothing the cell computes -- _polyphase_resample')
print('    anchors the excerpt at xp[|lo|:] and zero-pads the rest -- but it hands conv1d two')
print('    inputs of the SAME length, which is the only difference left between the two calls.')
real_poly = ns.get('_polyphase_resample')
TARGET_LEN = 70000        # above every excerpt here, so nothing is ever truncated
for sr_in in RATES:
    def equal_length(sr_in=sr_in):
        if real_poly is None:
            return False, 'the cell has no _polyphase_resample, so it has no fast polyphase path'
        pre_pad = []

        def wrapped(x, poly, n_out_):
            pre_pad.append(len(x))
            if len(x) < TARGET_LEN:
                x = np.concatenate([x, np.zeros(TARGET_LEN - len(x), np.float32)])
            return real_poly(x, poly, n_out_)

        ns['_polyphase_resample'] = wrapped
        try:
            pre_pad.clear()
            a = resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, 0, n_out_of(sr_in))
            b = resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, 5000, 9000)
        finally:
            ns['_polyphase_resample'] = real_poly
        if len(b) != 4000 or len(pre_pad) != 2:
            return False, f'the experiment did not run as designed: lens {pre_pad}'
        if min(pre_pad) >= TARGET_LEN:
            return False, f'an excerpt was already {min(pre_pad)} samples: padding changed nothing'
        if pre_pad[0] == pre_pad[1]:
            return False, (f'both calls already handed conv1d {pre_pad[0]} samples, so padding '
                           'changed nothing -- this check would prove nothing')
        d = float(np.abs(b - a[5000:9000]).max())
        return d == 0.0, \
            (f'max|d| = {d:.3e} once both calls were handed the same 70000-sample conv1d input '
             f'(as shipped they got {sorted(pre_pad)} samples, and max|d| = '
             f'{devs[sr_in][(5000, 9000)]:.3e})')
    check(f'{sr_in} Hz: with both conv1d inputs the same length the chunk is BITWISE identical',
          equal_length)

# --- the alignment, observed rather than read ---------------------------------------------------
print('\n=== the excerpt starts on an output-grid input sample (s % M == 0) ===')
for sr_in in RATES:
    def on_grid(sr_in=sr_in):
        seen = []
        base = read_mono

        def rec(path, n_in, start, stop):
            seen.append(start)
            return base(path, n_in, start, stop)

        ns['_read_mono'] = rec
        try:
            for r0 in (0, 1, 7, 5000, 12345, n_out_of(sr_in) - 3000):
                resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, r0,
                          min(r0 + 900, n_out_of(sr_in)))
        finally:
            ns['_read_mono'] = base
        M = Fraction(TARGET, sr_in).denominator
        off = [s for s in seen if s % M]
        # The pre-fix formula on the same r0 values, so the fixture is shown to DISCRIMINATE: if
        # the old start were also on the grid for every r0, the checks above would all pass with
        # the bug still in place.
        old = [int(np.floor((r0 - 4096) * sr_in / TARGET))
               for r0 in (0, 1, 7, 5000, 12345, n_out_of(sr_in) - 3000)]
        old_off = [s for s in old if s % M]
        if is_immune(sr_in):
            return (not off and len(seen) == 6 and not old_off), \
                (f'port starts {seen[:4]}... and pre-fix starts {old[:4]}... are both multiples '
                 f'of M={M}: this rate pair is integral, so both formulas agree. 0 off-grid')
        return (not off and len(seen) == 6 and len(old_off) >= 1), \
            (f'port starts {seen[:4]}... are all multiples of M={M}; the pre-fix formula gives '
             f'{old[:4]}..., of which {len(old_off)}/6 are OFF-grid -- the fixture can witness '
             f'the bug at this rate')
    check(f'{sr_in} Hz: all 6 excerpt starts land on the output grid', on_grid)

# --- the margin is load-bearing -----------------------------------------------------------------
print('\n=== the margin: it must clear the filter, or the equality is luck ===')
kaiser = ns.get('_kaiser_best_polyphase')
for sr_in in RATES:
    def margin(sr_in=sr_in):
        if kaiser is None:
            return False, 'the cell has no polyphase filter bank to measure the filter against'
        K = kaiser(sr_in, TARGET)[3].shape[1]
        if 'margin' not in inspect.signature(resampled).parameters:
            return False, 'this _resampled has no margin parameter to vary'
        full = full_of(sr_in)
        tight = inspect.signature(resampled).parameters['margin'].default
        smallest = None
        for m in (0, 1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            got = resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, 5000, 9000, margin=m)
            if len(got) != 4000:
                return False, f'margin={m} returned {len(got)} samples'
            if float(np.abs(got - full[5000:9000]).max()) <= BUDGET_REL * peaks[sr_in]:
                smallest = m
                break
        zero = resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, 5000, 9000, margin=0)
        d0 = float(np.abs(zero - full[5000:9000]).max())
        return (tight is not None and smallest is not None and tight >= smallest
                and d0 > 100 * BUDGET_REL * peaks[sr_in]), \
            (f'{K}-tap filter: the smallest margin that works is {smallest} (the default is '
             f'{tight}), and margin=0 is off by {rel(d0, peaks[sr_in]):.1%} of full scale')
    check(f'{sr_in} Hz: the default margin clears the filter, and no lead-in at all does not',
          margin)

# --- the tail of the file, and ranges that run past it -----------------------------------------
print('\n=== a range that runs past the end of the file still has the requested length ===')
for sr_in in RATES:
    def past_end(sr_in=sr_in):
        full = full_of(sr_in)
        n_out = n_out_of(sr_in)
        got = resampled(wavs[sr_in], LENGTHS[sr_in], sr_in, TARGET, n_out - 10, n_out + 5)
        if len(got) != 15:
            return False, f'asked for 15 samples, got {len(got)}'
        d = float(np.abs(got[:10] - full[n_out - 10:]).max())
        return d <= BUDGET_REL * peaks[sr_in], \
            f'[n_out-10, n_out+5) -> 15 samples; the 10 real ones match the whole file to {d:.3e}'
    check(f'{sr_in} Hz: a range ending 5 samples past the file end is padded, not short', past_end)

# --- the polyphase bank itself is faithful -----------------------------------------------------
print('\n=== the polyphase bank vs resampy (the self-check the cell prints once per ratio) ===')
resample_like = ns.get('resample_like_training')
for sr_in in RATES:
    def poly_vs_resampy(sr_in=sr_in):
        if resample_like is None or '_POLY_OK' not in ns:
            return False, 'the cell has no polyphase fast path at all'
        with contextlib.redirect_stdout(io.StringIO()):
            x = sigs[sr_in][:4096]
            ref = resampy.resample(x, sr_in, TARGET, filter='kaiser_best')
            got = resample_like(x, sr_in, TARGET)
        if len(ref) != len(got):
            return False, f'length {len(got)} vs resampy {len(ref)}'
        d = float(np.abs(ref - got).max() / max(float(np.abs(ref).max()), 1e-12))
        used = ns['_POLY_OK'].get((sr_in, TARGET)) is True
        return d <= 2e-4 and used, \
            (f'{d:.2e} of full scale against the cell\'s own 2e-4 acceptance, and the fast path '
             f'was the one used (_POLY_OK={used})')
    check(f'{sr_in} Hz: the fast polyphase path agrees with resampy', poly_vs_resampy)

if resample_like is None:
    check('441 -> 192 kHz matches resampy', lambda: (False, 'no polyphase fast path to use'))
else:
    def ratio_441():
        with contextlib.redirect_stdout(io.StringIO()):
            x = sigs[384000][:8192]
            a = resample_like(x, 441000, TARGET)
            b = resampy.resample(x, 441000, TARGET, filter='kaiser_best').astype(np.float32)
        if len(a) != len(b):
            return False, f'length {len(a)} vs resampy {len(b)}'
        d = float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-12))
        used = ns['_POLY_OK'].get((441000, TARGET)) is True
        return d <= 2e-4 and used, f'{d:.2e} of full scale, 147 phases, _POLY_OK={used}'
    check('441 -> 192 kHz (147 phases, not one of the four rates) matches resampy', ratio_441)

print('\n' + '=' * 70)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)
