#!/usr/bin/env python3
"""Generate batspot-train-merged.ipynb from batspot-train.ipynb.

The merged notebook is a BUILD ARTIFACT. Never hand-edit it. Edit the cell
sources in notebook_build/cells/, register them in REPLACE / APPEND_CODE /
APPEND_MD below, and re-run this script:

    venv/bin/python notebook_build/assemble.py

`batspot-train.ipynb` and `batspot_train(claude_bug_fixes _by__SPACE_BUNNY_MODEL).ipynb`
are read-only parents; neither is modified by this script.

With REPLACE and APPEND_* empty, build() is a lossless pass-through: every cell's
source is the base's, cell for cell. The one thing that never carries over is
execution state -- `outputs` and `execution_count` are cleared on every code
cell, so the artifact never claims to have been run. That property is what makes
every later port auditable: a diff against the base shows exactly what was
ported.

Every guard rail raises ValueError rather than using `assert`, because `assert`
is stripped under `python -O` and this script's whole premise is that a silent
no-op is impossible.
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, 'batspot-train.ipynb')
OUT = os.path.join(REPO, 'batspot-train-merged.ipynb')
CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cells')

# The merge map is written against a 24-cell base. If the base ever changes
# shape, these indices (and the plan's) stop meaning what they say.
N_BASE_CELLS = 24

# base cell index -> filename in cells/. A '.md' file supplies a markdown cell,
# any other filename a code cell; build() checks the pair agrees.
#
# 1 = config: the combined notebook's cell 2 (noise mixing, 3-seed ensemble, "unknown"),
# ported verbatim, plus a Nyquist guard the combined notebook does not have.
# 4 = architecture: the combined notebook's cell 5, ported verbatim. The ResNet's residual
# projection must stay spelled `shortcut` -- that is the on-disk key format of every
# official .pk -- so src_04.py is copied, never reformatted.
# 5 = transforms + dataset: the combined notebook's cell 6, ported verbatim. Carries the
# background-noise mixing, the _code_fingerprint cache tag and the open-set scorers, each
# with the measurements that chose its default. Not reformatted either: the comments record
# which constants are measured and why, which is the entire value of the port.
# 6 = data discovery: the combined notebook's cell 7. Unusable-clip pre-drop BEFORE the split
# (so detector and classifier always see the same clip list), the three split modes
# including per_species, the per-class recording counts and the tape-lookup baseline.
# 7 = clip extraction: the combined notebook's cell 8, the faithful port of export_clips.R.
# THIS IS THE EXTRACTION BUG FIX. The base cell took `'_'.join(name.split('_')[:2])` --
# `acsh_devon` for this repo's selection tables -- as the recording stamp, so the match
# against the audio file names never held, every table was skipped and it reported
# "Extracted 0 clips" on the dataset it was meant to rebuild. The port matches the Raven
# YYYYMMDD_HHMMSS stamp and writes the shipped clip layout, so cell 6's recording_id
# finds the tape. Copied verbatim: the comments recording the faithfulness to the R script
# and its three deliberate deviations are the point of the cell.
# 9 = model staging: the combined notebook's cell 10. The four official models are ALL
# basenamed ANIMAL-SPOT.pk, so staging by basename collides silently -- the classifier
# overwrites the detector and you fine-tune a "detector" from 15-class weights.
# 14 = export: the combined notebook's cell 15. export_pk deep-copies before .cpu(), so the
# LIVE models stay on the GPU; also writes every ensemble member and the unknown sidecar.
# KNOWN GAP (not a defect in the port): this cell reads `cls_members`, `cls_best_member` and
# `UNKNOWN_MODEL`, which no cell of the notebook defines yet -- the ensemble cell that will
# define them is Task 8's port of c12. The artifact therefore executes through cell 13 and
# stops there. test_assemble.py's `every loaded name is bound by an earlier cell` check FAILS
# on exactly this, and is meant to: it is the guard that will go green when Task 8 lands.
# 16 = evaluation: the combined notebook's cell 17. num_mels fallback, class names indexed by
# output width (not by sorted dict value), and the no_signal flag.
# 17 = summary: the combined notebook's cell 18. kind / n_files columns, NO SIGNAL markers and
# the reduced-denominator warning.
REPLACE: dict[int, str] = {
    1: 'src_01.py', 4: 'src_04.py', 5: 'src_05.py', 6: 'src_06.py', 7: 'src_07.py',
    9: 'src_09.py', 14: 'src_14.py', 16: 'src_16.py', 17: 'src_17.py',
}

# Cells appended after the base cells, markdown first then code.
APPEND_MD: list[str] = []    # cells/, markdown
APPEND_CODE: list[str] = []  # cells/, code


def to_source(text: str) -> list[str]:
    """Notebook source as a list of lines, newline-terminated except the last."""
    lines = text.split('\n')
    return [ln + '\n' for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def _cell_type_for(fname):
    """Cell type a cells/ filename supplies: '.md' is markdown, else code."""
    return 'markdown' if fname.endswith('.md') else 'code'


def _read_source(fname):
    """Read one cell source from cells/, naming the path if it is missing."""
    path = os.path.join(CELLS, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f'no such cell source: {path}')
    with open(path, encoding='utf-8') as f:
        return f.read()


def build(out_path: str) -> None:
    with open(BASE, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    if len(cells) != N_BASE_CELLS:
        raise ValueError(f'base has {len(cells)} cells, expected {N_BASE_CELLS}')

    # --- replacements, in ascending index order so the log is deterministic ---
    for idx in sorted(REPLACE):
        fname = REPLACE[idx]
        if not 0 <= idx < len(cells):
            raise ValueError(
                f'REPLACE index {idx} is outside the {len(cells)}-cell base')
        want = _cell_type_for(fname)
        got = cells[idx]['cell_type']
        if got != want:
            raise ValueError(
                f'REPLACE[{idx}] = {fname} supplies a {want} cell but base cell {idx} is {got}')
        new = _read_source(fname)
        old_len = len(''.join(cells[idx]['source']))
        cells[idx]['source'] = to_source(new)
        print(f'  cell {idx:>2} <- {fname:<14} ({old_len} -> {len(new)} chars)')

    # --- appended cells -------------------------------------------------------------------
    appended = []
    for fname in APPEND_MD:
        if _cell_type_for(fname) != 'markdown':
            raise ValueError(f'APPEND_MD entry {fname} is not a .md file')
        appended.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': to_source(_read_source(fname)),
        })
    for fname in APPEND_CODE:
        if _cell_type_for(fname) == 'markdown':
            raise ValueError(f'APPEND_CODE entry {fname} is a .md file')
        appended.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': to_source(_read_source(fname)),
        })
    base_n = len(cells)
    cells.extend(appended)

    # The output is a fresh artifact, so it must not claim to have been run:
    # saved outputs belong to a previous build's code, not to this one's.
    for c in cells:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None

    nb['cells'] = cells
    # Write atomically: an interrupted build must not leave a half-written
    # artifact where a complete one used to be.
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, separators=(',', ':'))
    os.replace(tmp, out_path)

    print(f'  {len(REPLACE)} replaced, {len(appended)} appended '
          f'({base_n} base cells -> {len(cells)} total)')
    print(f'wrote {out_path}  ({os.path.getsize(out_path):,} bytes)')


if __name__ == '__main__':
    build(OUT)
