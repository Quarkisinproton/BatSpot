#!/usr/bin/env python3
"""Cell 8 (clip extraction) run on the REAL Raven tables of this repo.

The cell being ported is the fix for AGENTS.md 11.2: the previous extractor took
`'_'.join(name.split('_')[:2])` -- `acsh_devon` for this repo's tables -- as the recording stamp,
so `if date_time in audio_name` never held, every table was skipped, and the cell reported
"Extracted 0 clips" on the dataset it was supposed to rebuild.

Everything here is the cell's own code, exec'd from the delivered artifact (never a copy), and
the `recording_id` it has to interoperate with is Task 6's, taken from the delivered
data-discovery cell. Clips are written to a temp dir; `Data/` is only ever read.

WHAT THE FIXTURE IS FOR. Nine mutations of the ported cell were run against this suite (the wrong
stamp match, the stamp dropped from the written name, a flat output dir, the view filter removed,
the clip cut from t=0, one constant class folder, the 3 ms floor raised, clips re-quantised, and
`stamp = time-of-day only`). The first eight are caught by the checks below. The ninth -- M9 -- is
caught ONLY because of how `TABLES` is chosen, which is the reason the choice is spelled out:

  Matching on the time-of-day instead of the whole stamp (M9) is a substring test over Data/audio,
  and MANY recordings share a time-of-day: 6 files are `192000`, 5 are `194000`, 3 are `202000`,
  2 are `200000`. M9 then silently cuts a table's clips from the wrong recording and still names
  them with the right tape. It is caught only if some staged table's own recording is NOT the
  first of its time-of-day in `sorted()` order -- so `rhro_devon_20260528_200000` (whose recording
  sorts second behind `20260521_200000.WAV`) is in `TABLES`, and a fixture guard asserts that
  property still holds. An earlier fixture (acsh_devon_20260521_204000 + noise_devon_20260429_192000,
  each on a time-of-day where the correct file happens to sort first) hid the whole class.

Run: venv/bin/python notebook_build/tests/test_extraction.py
"""
import contextlib
import csv
import glob
import io
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

CELL_EXTRACT, CELL_DATA = 7, 6

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


REPO = extract_cells.REPO
AUDIO = os.path.join(REPO, 'Data', 'audio')

# Two real tables, 20 clips on two tapes: 12 bat-call rows (acsh_devon_20260521_204000) and 8 noise
# rows (rhro_devon_20260528_200000 -- see the module docstring for why THIS one and not a table on
# a unique time-of-day). Both are staged in full, both must yield their rows.
TABLES = [os.path.join(REPO, 'Data', 'selections', 'acsh', 'acsh_devon_20260521_204000.txt'),
          os.path.join(REPO, 'Data', 'selections', 'rhro', 'rhro_devon_20260528_200000.txt')]

# The table whose recording does NOT sort first within its time-of-day. Kept as a named constant
# because the M9-detection argument above depends on it, and the fixture guard checks the property
# rather than trusting this comment.
DISCRIMINATING = os.path.basename(TABLES[1])

# Every one of the 48 tables under Data/selections carries View 'Spectrogram 1', so the R
# script's "Spectrogram rows only" rule cannot be falsified by real data alone. A third table is
# staged: the acsh rows, re-viewed as a waveform, under another tape's name. It must contribute
# nothing. It is a COPY rather than a rewrite of the real table on purpose -- replacing a row of
# the acsh table would change the headline count, and a doctored copy with the same rows would
# simply be extracted twice.
WAVE_TABLE_NAME = 'acsh_devon_20260525_192000_waveform.txt'

# The shipped clip layout: <class>-bat_<ID>_<YEAR>_<YYYYMMDD-HHMMSS>_<start ms>_<end ms>.wav
SHIP_SHAPE = re.compile(r'^.+-bat_\d+_\d{4}_\d{8}-\d{6}_\d+_\d+\.wav$')

idx = extract_cells.merged_cells()

# `re`, `os`, `glob`, `np`, `sf` are the names the cell reads, supplied by the earlier cells of
# the notebook (cell 4 imports re). The cell's own imports (csv, zlib) come from the cell.
# RUN_CLIP_EXTRACTION is False so the cell's module-level driver takes its SKIPPED branch and
# never touches DATA_DIR; the function under test is then called directly, into a temp dir.
ns = {'os': os, 'glob': glob, 'sf': sf, 'np': np, 're': re,
      'RUN_CLIP_EXTRACTION': False, 'OVERWRITE_EXISTING_CLIPS': False,
      'DATA_DIR': os.path.join(REPO, 'Data', 'final_dataset', 'data'),
      'RAW_AUDIO_DIR': '', 'SELECTIONS_DIR': ''}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(open(idx[CELL_EXTRACT]).read(), idx[CELL_EXTRACT], 'exec'), ns)
extract = ns['extract_clips_from_selections']

print(f'=== {os.path.basename(idx[CELL_EXTRACT])} (merged cell {CELL_EXTRACT}) ===')
print(f'  module-level driver: {buf.getvalue().strip().splitlines()[0]}')

# Task 6's recording_id, from the delivered data-discovery cell: the extractor writes names and
# this function has to recover the tape from them. Not a copy -- a re-implementation here would
# pass regardless of what the notebook does. `cell_defs` pulls just this function and the regex it
# reads; running the whole data-discovery cell instead would glob and sf.info all 964 clips and
# print a split report, all of it discarded, to obtain one two-line function.
recording_id = extract_cells.cell_defs(idx[CELL_DATA], {'recording_id'},
                                       {'os': os, 're': re})['recording_id']


def read_table(path):
    """Raven rows, keys lower-cased and stripped -- the cell's own reading convention."""
    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        return [{(k or '').strip().lower(): (v or '').strip() for k, v in r.items()}
                for r in csv.DictReader(fh, delimiter='\t')]


def qualifying(rows):
    """The rows export_clips.R keeps: Spectrogram view, longer than 3 ms, parseable times."""
    keep = [r for r in rows
            if ('view' not in r or 'spectrogram' in r['view'].lower())
            and r.get('begin time (s)') and r.get('end time (s)')]
    return [r for r in keep
            if float(r['end time (s)']) - float(r['begin time (s)']) > 0.003]


def audio_for(stamp):
    """Every recording in Data/audio whose name carries this tape stamp, sorted as the cell sorts."""
    stem = stamp.replace('-', '_')
    return [p for p in sorted(glob.glob(os.path.join(AUDIO, '*')))
            if stem in os.path.basename(p).lower()]


# --- fixture guards, before anything is run -----------------------------------------------
# These say the fixture still has the properties the checks below rely on. They are cheap and they
# are what stops a future data change from quietly turning the suite into a no-op.
_disc_tape = '20260528_200000'
_disc_tod = '200000'
# What M9 would match: the time-of-day alone, as a substring over Data/audio in sorted() order.
# That is the FIRST hit, and for _disc_tape it is the wrong recording.
_m9_first = os.path.basename(audio_for(_disc_tod)[0]) if audio_for(_disc_tod) else None
print('\n=== fixture guards (properties the checks below rely on) ===')
check(f'{DISCRIMINATING}: its recording is NOT the first of its time-of-day, so matching on the '
      f'time-of-day alone (M9) cuts from the wrong recording',
      _m9_first is not None and _m9_first != _disc_tape + '.WAV',
      f'{len(audio_for(_disc_tod))} files share {_disc_tod}; M9 substring first = {_m9_first}, '
      f'correct = {_disc_tape}.WAV')

_n_matches = {}
for _t in TABLES:
    _stamp = re.search(r'(\d{8})_(\d{6})', os.path.basename(_t)).group(0)
    _n_matches[os.path.basename(_t)[:-4]] = len(audio_for(_stamp))
check("every staged table's full stamp matches exactly ONE recording in Data/audio",
      set(_n_matches.values()) == {1}, ', '.join(f'{k} -> {v}' for k, v in _n_matches.items()))
check('the base parent\'s stamp is a substring of no recording name (that is why it finds nothing)',
      not [p for p in glob.glob(os.path.join(AUDIO, '*'))
           for pre in ('acsh_devon', 'rhro_devon')
           if pre in os.path.basename(p).lower()],
      "premises 'acsh_devon' / 'rhro_devon' against all Data/audio names")

# --- the real work -------------------------------------------------------------------------
tmp = tempfile.mkdtemp(prefix='batspot-extract-')
try:
    sel, out = os.path.join(tmp, 'selections'), os.path.join(tmp, 'data')
    for t in TABLES:
        os.makedirs(os.path.join(sel, os.path.basename(os.path.dirname(t))), exist_ok=True)
        shutil.copy(t, os.path.join(sel, os.path.basename(os.path.dirname(t))))
    # The waveform table: real rows and a real tape name, none of which the cell may cut.
    rows = read_table(TABLES[0])
    for r in rows:
        if r.get('view'):
            r['view'] = 'Waveform 1'
    wave_path = os.path.join(sel, os.path.basename(os.path.dirname(TABLES[0])), WAVE_TABLE_NAME)
    with open(wave_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in rows[0]})

    run_out = io.StringIO()
    with contextlib.redirect_stdout(run_out):
        written = extract(AUDIO, sel, out)
    cell_log = run_out.getvalue()
    made = sorted(glob.glob(os.path.join(out, '*', '*.wav')))
    print(f'\n=== 2 real tables + 1 waveform table -> temp dir ({len(made)} files) ===')
    for ln in cell_log.splitlines():
        print('   ', ln)

    n_acsh_all = len(read_table(TABLES[0]))
    n_acsh = len(qualifying(read_table(TABLES[0])))
    n_wave_all = len(read_table(wave_path))
    n_rhro = len(qualifying(read_table(TABLES[1])))
    n_rows = n_acsh + n_rhro
    # Fixture guard, not a cell check: it says the count below really is testing the view filter,
    # because the waveform table does carry extractable-looking rows and none of them qualify.
    # Without it, "20 clips" could also be satisfied by a cell that keeps waveform rows.
    check('the waveform table has rows but none of them qualify (fixture guard)',
          n_wave_all == n_acsh_all and len(qualifying(read_table(wave_path))) == 0,
          f'{n_wave_all} rows, 0 qualifying')
    check('two real tables yield 20 clips (the waveform table contributes none)',
          written == 20 and len(made) == 20, f'{written} written, {len(made)} files')
    # A different claim from the one above: the cell agreeing with an independent count of the
    # table rows the rules keep. If it kept a wrong row and dropped a right one, the totals could
    # still both be 20 -- the per-species and id checks below are what pin that down.
    check('one clip per Spectrogram row longer than 3 ms', written == n_rows,
          f'{written} written vs {n_rows} qualifying rows ({n_acsh} + {n_rhro})')
    # The brief's `not problems` conjunct. On this fixture a skipped table would drop the count
    # below 20, so nothing is uncovered today -- but a future fixture with a deliberately
    # unmatched table would otherwise tolerate skips silently.
    check('no table was skipped with a reason', 'skipped' not in cell_log,
          [ln.strip() for ln in cell_log.splitlines() if 'skipped' in ln] or 'clean log')
    # count files per sub-folder without assuming the folders exist: a RED where nothing was
    # written must report, not crash half-way through the suite.
    def n_in(species):
        d = os.path.join(out, species)
        return len(os.listdir(d)) if os.path.isdir(d) else 0

    check('the 20 clips are spread over both species folders',
          (n_in('acsh'), n_in('rhro')) == (12, 8), f'{n_in("acsh")} acsh + {n_in("rhro")} rhro')

    ids = {recording_id(p) for p in made}
    check('written names parse back to the tape', ids == {'20260521-204000', '20260528-200000'},
          f'{sorted(ids)}')
    # `made` must be non-empty: an all() over an empty list is True, which would let this pass on
    # the unported cell -- the vacuous-check failure this repo keeps hitting.
    check('every written name carries a findable timestamp (no stem fallback)',
          bool(made) and all(recording_id(p) != os.path.basename(p)[:-4] for p in made),
          f'{len(made)} names')

    # Samples: the clip must BE the selection row, at the row's offset. The offset is taken from
    # the selection table, NOT from the written name -- reading it from the name would be circular:
    # a cell that cut the wrong slice and named it wrong would agree with itself. A mutation that
    # does exactly that (`start = 0`) passed the name-based version of this check undetected.
    # Key: the ms-rounded boundaries the cell writes in the file name. Value: the source slice
    # those boundaries stand for, taken from the table's own times -- not from the name.
    rows_by_ms, missing_tape = {}, []
    for t in TABLES + [wave_path]:
        hits = audio_for(re.search(r'(\d{8})_(\d{6})', os.path.basename(t)).group(0))
        if not hits:                       # a renamed/absent tape: report, don't raise inside sf.info
            missing_tape.append(os.path.basename(t))
            continue
        info = sf.info(hits[0])
        for r in qualifying(read_table(t)):
            a, b = float(r['begin time (s)']), float(r['end time (s)'])
            s0, s1 = int(round(a * info.samplerate)), min(int(round(b * info.samplerate)),
                                                          info.frames)
            rows_by_ms.setdefault((int(round(s0 / info.samplerate * 1000)),
                                   int(round(s1 / info.samplerate * 1000))),
                                  (hits[0], s0, s1))
    # Reported as a failing check rather than raised: an absent tape must surface as one FAIL with
    # the name in it, not as a TypeError from sf.info(None) that aborts the rest of the suite.
    check('every staged table has a recording in Data/audio (fixture guard)', not missing_tape,
          str(missing_tape) if missing_tape else 'all resolved')
    # 4 of the 12 acsh clips and the 2 last rhro clips, so BOTH species -- and therefore the
    # discriminating tape -- are represented in the probe.
    probe = made[:4] + made[-2:]
    bad = []
    for p in probe:
        fields = os.path.basename(p)[:-4].split('_')
        hits = audio_for(recording_id(p))
        src = rows_by_ms.get((int(fields[-2]), int(fields[-1])))
        clip, _ = sf.read(p, dtype='float32')
        if src is None or not hits or os.path.basename(src[0]) != os.path.basename(hits[0]):
            bad.append(os.path.basename(p))
            continue
        want, _ = sf.read(src[0], start=src[1], stop=src[2], dtype='float32')
        if clip.shape != want.shape or not np.array_equal(clip, want):
            bad.append(os.path.basename(p))
    # `not bad` over an empty probe is vacuously true, so the probe must be non-empty.
    check("clip samples are the selection row's samples, at the row's offset",
          bool(probe) and not bad, f'{len(probe) - len(bad)}/{len(probe)} clips exact'
                                    + (f', unmatched: {bad[:3]}' if bad else ''))
    check('the probe includes clips from the discriminating tape (M9 must be reachable)',
          bool(probe) and any(recording_id(p).startswith('20260528-200000') for p in probe),
          f'{sum(1 for p in probe if recording_id(p).startswith("20260528-200000"))} of '
          f'{len(probe)} probe clips from {DISCRIMINATING}')

    check('clips land in <output>/<species>/ with the shipped name shape',
          bool(made) and
          all(os.path.basename(os.path.dirname(p)) ==
              os.path.basename(p).split('-', 1)[0] and
              SHIP_SHAPE.match(os.path.basename(p)) for p in made),
          made[0].replace(tmp, '<tmp>') if made else 'none')

    # Attribution: the read-only base parent's extractor, on the same inputs. While the port is in
    # place this is 0; if it ever stops being 0 the port's headline check is no longer evidence
    # of anything, so the run says so rather than staying quiet. The mechanism is asserted above
    # ('acsh_devon' / 'rhro_devon' match no recording name), so 0 here means the bug, not some
    # other reason the base stopped matching.
    base = extract_cells.base_cells()
    bns = dict(ns)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(open(base[CELL_EXTRACT]).read(), base[CELL_EXTRACT], 'exec'), bns)
    bout = os.path.join(tmp, 'base_out')
    with contextlib.redirect_stdout(io.StringIO()):
        base_written = bns['extract_clips_from_selections'](AUDIO, sel, bout)
    print()
    check('the base parent\'s extractor finds 0 clips here (why this port exists)',
          base_written == 0 and not glob.glob(os.path.join(bout, '*', '*.wav')),
          f'base wrote {base_written}, merged wrote {written}')
finally:
    shutil.rmtree(tmp, True)

print('\n' + '=' * 66)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)
