#!/usr/bin/env python3
"""B4 regression, tested against filenames this repo ACTUALLY produces, using the
NOTEBOOK'S OWN recording_id (exec'd from the delivered cell, not a copy).

Two earlier mistakes this file exists to prevent:
  1. A first version asserted against a filename I invented
     (`sasa-bat_20260429-192000_215755_215781.wav`), which the shipped '^\\d{8}-\\d{6}$'
     pattern happened to match. It did NOT match what Cell 8 really writes.
  2. A second version defined its own `new_rid` copy instead of reading the cell, so it could
     never detect a change to the notebook. (redgreen.py caught this: 3 undetected reverts.)

Both mistakes are structurally impossible here: `recording_id` below is the one the merged
notebook's data-discovery cell defines, extracted from the artifact itself.

Run: venv/bin/python notebook_build/tests/test_recording_id.py
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

CELL_DATA = 6   # Cell 7: Data Discovery

fails = []


def check(n, c, d=''):
    print(f'  [{"PASS" if c else "FAIL"}] {n}' + (f'  -- {d}' if d else ''))
    if not c:
        fails.append(n)


# ---- reference implementations of the PRE-FIX code. Used only to say what the bug did; ----
# ---- every check below contrasts them against the notebook's own recording_id, so it is   ----
# ---- an assertion about the artifact rather than about this file.                      ----
def old_rid(path):
    """The pre-fix positional parse."""
    parts = os.path.basename(path)[:-4].split('_')
    return parts[3] if len(parts) >= 6 else os.path.basename(path)


_OLD_ME = re.compile(r'^\d{8}-\d{6}$')


def first_fix_rid(path):
    """My FIRST fix: hyphen-only, matched against whole underscore-delimited fields."""
    stem = os.path.basename(path)[:-4]
    for part in stem.split('_'):
        if _OLD_ME.match(part):
            return part
    return stem


# ---- the notebook's own function -------------------------------------------------------
REPO = extract_cells.REPO
CLIPS = sorted(glob.glob(f'{REPO}/Data/final_dataset/data/*/*.wav'))
gcls = lambda f: os.path.basename(f).split('-', 1)[0]
idx = extract_cells.merged_cells()
# The cell's config globals (192/250 kHz, n_fft 256) come from extract_cells.cell6_ns, so
# the unusable-clip pre-drop this port adds runs at the real threshold.
ns7, _ = extract_cells.run_cell6(idx, CELL_DATA)
rid = ns7['recording_id']          # <-- the notebook's function


def cell8_name(sel_file, species, start_ms=1234, end_ms=5678):
    """Replicate Cell 8's naming exactly, including its own date_time quirk."""
    sel_name = os.path.basename(sel_file).replace('.txt', '')
    date_time = '_'.join(sel_name.split('_')[:2])
    return f'{species}-bat_{date_time}_{sel_name}_{start_ms}_{end_ms}.wav'


sels = sorted(glob.glob(f'{REPO}/Data/selections/*/*.txt'))

print(f'=== under test: {os.environ.get("NEW_CELLS") or extract_cells.MERGED} cell {CELL_DATA} '
      f'({os.path.basename(idx[CELL_DATA])}) ===')
print('=== the two real filename families ===')
print('  Raven/manual (what Data/final_dataset/data holds):')
print('    acsh-bat_3379376_2026_20260525-192000_91577_92135.wav')
print(f'  Cell 8 output, from the {len(sels)} real selection files (e.g. '
      f'{os.path.basename(sels[0])}):')
print(f'    {cell8_name(sels[0], os.path.basename(os.path.dirname(sels[0])))}')

print('\n=== 1. Raven/manual layout ===')
six = 'acsh-bat_3379376_2026_20260525-192000_91577_92135.wav'
check('notebook extracts the session stamp', rid(six) == '20260525-192000', rid(six))

print('\n=== 2. Cell 8 layout, real filenames ===')
c8 = cell8_name('acsh_devon_20260521_204000', 'acsh')
check('notebook extracts the session stamp from Cell 8 output',
      rid(c8) == '20260521-204000', rid(c8))
# Each contrast below says what the old code produced AND that the notebook produces something
# NEITHER of the two old behaviours did -- not merely "different". Asserting only `old_rid(c8) ==
# 'acsh'` would pass whatever the notebook does, including returning 'acsh' itself.
check('the old positional parse returns the SPECIES name; the notebook returns neither that '
      'nor the stem',
      old_rid(c8) == 'acsh' and rid(c8) not in ('acsh', c8[:-4]),
      f'old -> {old_rid(c8)!r}, first fix -> {first_fix_rid(c8)!r}, notebook -> {rid(c8)!r}')
check('the first (hyphen-only) fix fell back to the stem; the notebook does not',
      first_fix_rid(c8) == c8[:-4] and rid(c8) != c8[:-4],
      f'first fix -> {first_fix_rid(c8)!r} vs notebook -> {rid(c8)!r}')
check('the notebook agrees with the separator-agnostic reference on every real name',
      all(rid(cell8_name(s, os.path.basename(os.path.dirname(s))))
          == re.sub(r'[-_]', '-', re.search(r'(\d{8})[-_](\d{6})',
               cell8_name(s, os.path.basename(os.path.dirname(s)))).group(0))
          for s in sels))

print('\n=== 3. all real selection filenames ===')
n_ok = sum(rid(cell8_name(s, os.path.basename(os.path.dirname(s))))
           != os.path.basename(cell8_name(s, os.path.basename(os.path.dirname(s))))[:-4]
           for s in sels)
check(f'all {len(sels)} Cell 8 outputs resolve to a tape id', n_ok == len(sels),
      f'{n_ok}/{len(sels)}')
old_keys, nb_keys = {}, {}
for s in sels:
    _n = cell8_name(s, os.path.basename(os.path.dirname(s)))
    old_keys.setdefault(old_rid(_n), set()).add(os.path.basename(s)[:-4])
    nb_keys.setdefault(rid(_n), set()).add(os.path.basename(s)[:-4])
check(f'the old code collapses {len(sels)} sessions into {len(old_keys)} groups; the '
      f'notebook keeps them apart',
      len(old_keys) == 8 and len(nb_keys) > len(old_keys),
      f'old {len(old_keys)} groups vs notebook {len(nb_keys)}')
# The pre-fix parse keys on the SPECIES, so every species becomes one group. That is a strictly
# stronger statement than "fewer groups": it names the collapse the bug actually caused.
check('the old parse keys on the species, merging every tape of a species; the notebook '
      'separates them',
      set(old_keys) == {os.path.basename(os.path.dirname(s)) for s in sels}
      and len(nb_keys) == 28,
      f'old keys={sorted(old_keys)} vs notebook {len(nb_keys)} groups')
_worst = max(len(v) for v in old_keys.values())
_worst_nb = max(len(v) for v in nb_keys.values())
check(f'the worst old group merges {_worst} tapes; the notebook merges at most {_worst_nb}',
      _worst == 10 and _worst_nb < _worst,
      f'old max={_worst}, notebook max={_worst_nb}')

print('\n=== 4. one tape, several species and both layouts -> ONE key ===')
a = 'acsh-bat_1653604_2026_20260521-204000_130273_130648.wav'
n = 'noise-bat_1653604_2026_20260521-204000_130273_130648.wav'
check('two species from one Raven session share a key', rid(a) == rid(n), f'{rid(a)}')
check('Cell 8 output of the same session gives the SAME key',
      rid(cell8_name('acsh_devon_20260521_204000', 'acsh')) == rid(a)
      == rid(cell8_name('noise_devon_20260521_204000', 'noise')))

print('\n=== 5. different tapes stay apart; graceful fallback ===')
check('consecutive sessions differ',
      rid(cell8_name('acsh_devon_20260521_204000', 'acsh'))
      != rid(cell8_name('acsh_devon_20260525_192000', 'acsh')))
check('a name with no timestamp degrades to the stem', rid('weird-name.wav') == 'weird-name')

print('\n=== 6. the real dataset, through the notebook ===')
g = {rid(w) for w in CLIPS}
check('964 clips group into 28 tapes', len(g) == 28, f'{len(g)} groups')
per_cls = {}
for w in CLIPS:
    per_cls.setdefault(gcls(w), set()).add(rid(w))
check('heti is a single recording', len(per_cls['heti']) == 1, f'{len(per_cls["heti"])}')
check('rhbe spans 2 recordings', len(per_cls['rhbe']) == 2, f'{len(per_cls["rhbe"])}')

print('\n' + '=' * 66)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)
