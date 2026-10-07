#!/usr/bin/env python3
"""Cell 8 (clip extraction) run on the REAL Raven tables of this repo.

The cell being ported is the fix for AGENTS.md 11.2: the previous extractor took
`'_'.join(name.split('_')[:2])` -- `acsh_devon` for this repo's tables -- as the recording stamp,
so `if date_time in audio_name` never held, every table was skipped, and the cell reported
"Extracted 0 clips" on the dataset it was supposed to rebuild.

Everything here is the cell's own code, exec'd from the delivered artifact (never a copy), and
the `recording_id` it has to interoperate with is Task 6's, taken from the delivered
data-discovery cell. Clips are written to a temp dir; `Data/` is only ever read.

Eight mutations of the cell were run against this suite (wrong stamp match, stamp dropped from the
written name, flat output dir, view filter removed, clip cut from t=0, wrong species folder, the 3 ms
floor raised, clips re-quantised). All eight are caught; the last one is what the sample check is for.

What each check is for, and the input that would break it:

  1. fixture guard: the waveform table really has extractable-looking rows and no qualifying ones,
     so that "20 clips" below is testing the view filter rather than passing because nothing was
     there to cut
  2. 20 clips from the two real tables ....... a wrong stamp match, a lost Spectrogram row, a row
                                              under the 3 ms floor, or a name collision all change
                                              the count
  3. the cell agrees with an independent count of the qualifying table rows
  4. per-species split 12/8 ................. a cell writing flat, or mixing species up, fails it
  5. names parse back to the two stamps ...... a name without a findable timestamp falls back to
                                              the stem, so the id set would not be these two
  6. no name falls back to its stem .......... guards 5 against passing on an empty output
  7. clip samples ARE the table row's samples, at the row's offset -- the offset comes from the
     SELECTION TABLE, not from the written name. Reading it from the name would be circular: a cell
     that cut the wrong slice and named it wrong would agree with itself, and a mutation that does
     exactly that (`start = 0`) passed the name-based version of this check undetected.
  8. <output>/<species>/ with the shipped name shape
  9. the read-only base parent extracts 0 .... attribution: if the old code ever stops extracting
                                              nothing here, the checks above prove less

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

# Two real tables, one species with bat calls and one noise: acsh_devon_20260521_204000
# (12 Spectrogram rows) and noise_devon_20260429_192000 (8). 20 clips, two tapes.
TABLES = [os.path.join(REPO, 'Data', 'selections', 'acsh', 'acsh_devon_20260521_204000.txt'),
          os.path.join(REPO, 'Data', 'selections', 'noise', 'noise_devon_20260429_192000.txt')]

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
# pass regardless of what the notebook does.
ns7, _ = extract_cells.run_cell6(idx, CELL_DATA)
recording_id = ns7['recording_id']


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


def audio_for(stem):
    """The raw recording whose name carries this tape stamp, or None."""
    hits = [p for p in sorted(glob.glob(os.path.join(AUDIO, '*')))
            if stem.replace('-', '_') in os.path.basename(p).lower()]
    return hits[0] if hits else None


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
    wave_path = os.path.join(sel, os.path.basename(os.path.dirname(TABLES[0])),
                             WAVE_TABLE_NAME)
    with open(wave_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in rows[0]})

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        written = extract(AUDIO, sel, out)
    made = sorted(glob.glob(os.path.join(out, '*', '*.wav')))
    print(f'\n=== 2 real tables + 1 waveform table -> temp dir ({len(made)} files) ===')
    for ln in buf.getvalue().splitlines():
        print('   ', ln)

    n_acsh = len(qualifying(read_table(TABLES[0])))
    n_wave_all = len(read_table(wave_path))
    n_rows = n_acsh + len(qualifying(read_table(TABLES[1])))
    # Fixture guard, not a cell check: it says the count below really is testing the view filter,
    # because the waveform table does carry extractable-looking rows and none of them qualify.
    # Without it, "20 clips" could also be satisfied by a cell that keeps waveform rows.
    check('the waveform table has rows but none of them qualify (fixture guard)',
          n_wave_all == n_acsh and len(qualifying(read_table(wave_path))) == 0,
          f'{n_wave_all} rows, 0 qualifying')
    check('two real tables yield 20 clips (the waveform table contributes none)',
          written == 20 and len(made) == 20, f'{written} written, {len(made)} files')
    # A different claim from the one above: the cell agreeing with an independent count of the
    # table rows the rules keep. If it kept a wrong row and dropped a right one, the totals could
    # still both be 20 -- the per-species and id checks below are what pin that down.
    check('one clip per Spectrogram row longer than 3 ms', written == n_rows,
          f'{written} written vs {n_rows} qualifying rows')
    # count files per sub-folder without assuming the folders exist: a RED where nothing was
    # written must report, not crash half-way through the suite.
    def n_in(species):
        d = os.path.join(out, species)
        return len(os.listdir(d)) if os.path.isdir(d) else 0

    check('the 20 clips are spread over both species folders',
          (n_in('acsh'), n_in('noise')) == (12, 8), f'{n_in("acsh")} acsh + {n_in("noise")} noise')

    ids = {recording_id(p) for p in made}
    check('written names parse back to the tape', ids == {'20260429-192000', '20260521-204000'},
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
    rows_by_ms = {}
    for t in TABLES + [wave_path]:
        wav = audio_for(re.search(r'(\d{8})_(\d{6})', os.path.basename(t)).group(0))
        info = sf.info(wav)
        for r in qualifying(read_table(t)):
            a, b = float(r['begin time (s)']), float(r['end time (s)'])
            s0, s1 = int(round(a * info.samplerate)), min(int(round(b * info.samplerate)),
                                                          info.frames)
            rows_by_ms.setdefault((int(round(s0 / info.samplerate * 1000)),
                                   int(round(s1 / info.samplerate * 1000))),
                                  (wav, s0, s1))
    # 4 of the 12 acsh clips and the 2 last noise clips, so both species are represented.
    probe = made[:4] + made[-2:]
    bad = []
    for p in probe:
        fields = os.path.basename(p)[:-4].split('_')
        wav = audio_for(recording_id(p))
        src = rows_by_ms.get((int(fields[-2]), int(fields[-1])))
        clip, _ = sf.read(p, dtype='float32')
        if src is None or wav is None or os.path.basename(src[0]) != os.path.basename(wav):
            bad.append(os.path.basename(p))
            continue
        want, _ = sf.read(src[0], start=src[1], stop=src[2], dtype='float32')
        if clip.shape != want.shape or not np.array_equal(clip, want):
            bad.append(os.path.basename(p))
    # `not bad` over an empty probe is vacuously true, so the probe must be non-empty.
    check('clip samples are the selection row\'s samples, at the row\'s offset',
          bool(probe) and not bad, f'{len(probe) - len(bad)}/{len(probe)} clips exact'
                                    + (f', unmatched: {bad[:3]}' if bad else ''))

    check('clips land in <output>/<species>/ with the shipped name shape',
          bool(made) and
          all(os.path.basename(os.path.dirname(p)) ==
              os.path.basename(p).split('-', 1)[0] and
              SHIP_SHAPE.match(os.path.basename(p)) for p in made),
          made[0].replace(tmp, '<tmp>') if made else 'none')

    # Attribution: the read-only base parent's extractor, on the same inputs. While the port is in
    # place this is 0; if it ever stops being 0 the port's headline check is no longer evidence
    # of anything, so the run says so rather than staying quiet.
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
